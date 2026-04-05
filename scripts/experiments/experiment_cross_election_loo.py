"""Cross-election leave-one-pair-out (LOPO) experiment.

Investigates which presidential election cycles are hardest to predict and
which county types generate the most error in the WetherVane model.

Three analyses:
  Part 1 — Leave-one-pair-out CV across all 4 presidential pairs (2008+).
            Holds out one pair at a time, re-runs KMeans on the rest,
            reports type-mean LOO r on the held-out pair.

  Part 2 — Per-type error decomposition using the production J=100 assignments.
            For each type: MAE, signed bias, county count, and dominant
            demographic features on the 2020→2024 holdout.

  Part 3 — Geographic concentration of error.
            For the 10 worst-performing types, which states are they
            concentrated in?

Usage:
    uv run python scripts/experiment_cross_election_loo.py

Outputs:
    Prints formatted tables for each part.
    Saves summary to data/experiments/cross_election_loo_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types
from src.validation.holdout_accuracy import (
    holdout_accuracy_county_prior_loo,
)

# ── Constants matching the production pipeline ────────────────────────────────
PRESIDENTIAL_WEIGHT = 8.0  # post-scaling weight applied to pres_* columns
PCA_COMPONENTS = 15        # PCA before KMeans (production setting)
PCA_WHITEN = True          # whitening validated in experiment_pca_whitening_ridge.py
TEMPERATURE = 10.0         # soft-membership sharpening (production default)
J = 100                    # number of types
MIN_YEAR = 2008            # start year filter for shift pairs in discovery
RANDOM_STATE = 42

# Min year applied as a filter on the start year of each shift pair.
# e.g. pres_d_shift_08_12 has start year 2008, which passes.
# pres_d_shift_04_08 has start year 2004, which is excluded.

# The production holdout (pres_*_20_24) is excluded from type discovery.
HOLDOUT_SUFFIX = "20_24"

# Key demographic columns to characterize error-prone types.
# These give a compact profile without overwhelming the output.
DEMO_COLS = [
    "pct_white_nh",
    "pct_black",
    "pct_hispanic",
    "pct_bachelors_plus",
    "median_hh_income",
    "manufacturing_share",
    "evangelical_share",
    "pct_rural",  # may not be present; handled gracefully below
    "pop_total",
]


# ── Data structures ───────────────────────────────────────────────────────────


class PresidentialPair(NamedTuple):
    """All column names belonging to one presidential shift pair."""

    label: str          # e.g. "08→12"
    d_col: str          # pres_d_shift_08_12
    r_col: str          # pres_r_shift_08_12
    turnout_col: str    # pres_turnout_shift_08_12


# ── Helper functions ──────────────────────────────────────────────────────────


def parse_start_year(col: str) -> int:
    """Extract integer start year (4-digit) from a shift column name.

    Column names follow the pattern: {race}_{metric}_shift_{y1}_{y2}
    where y1 and y2 are 2-digit years.
    """
    parts = col.split("_")
    y2_str = parts[-2]
    y2 = int(y2_str)
    # Treat years >= 50 as 1900s (covers governor pairs like 94→98)
    return y2 + (1900 if y2 >= 50 else 2000)


def collect_presidential_pairs(df: pd.DataFrame) -> list[PresidentialPair]:
    """Identify all presidential shift pairs present in the shift DataFrame.

    Requires that all three columns (d, r, turnout) are present and have
    a start year >= MIN_YEAR. Excludes the production holdout (20_24).
    """
    pairs: list[PresidentialPair] = []
    for col in sorted(df.columns):
        if not col.startswith("pres_d_shift_"):
            continue
        # Extract the year suffix, e.g. "08_12"
        year_suffix = col.replace("pres_d_shift_", "")
        if year_suffix == HOLDOUT_SUFFIX:
            continue  # skip the production holdout

        d_col = f"pres_d_shift_{year_suffix}"
        r_col = f"pres_r_shift_{year_suffix}"
        t_col = f"pres_turnout_shift_{year_suffix}"
        if not all(c in df.columns for c in [d_col, r_col, t_col]):
            continue

        start_year = parse_start_year(d_col)
        if start_year < MIN_YEAR:
            continue

        y1_str, y2_str = year_suffix.split("_")
        label = f"{y1_str}→{y2_str}"
        pairs.append(PresidentialPair(label=label, d_col=d_col, r_col=r_col, turnout_col=t_col))

    return pairs


def build_training_matrix(
    df: pd.DataFrame,
    exclude_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build the scaled, presidential-weighted shift matrix used for KMeans.

    Mirrors the production pipeline in run_type_discovery.main():
      1. Exclude holdout columns and those explicitly requested.
      2. Filter to shift pairs with start year >= MIN_YEAR.
      3. StandardScaler normalize.
      4. Apply presidential weight post-scaling.

    Parameters
    ----------
    df : DataFrame with county_fips and shift columns.
    exclude_cols : shift column names to omit (the held-out pair's columns).

    Returns
    -------
    (scaled_matrix, col_names) tuple.
    """
    # All shift columns except county_fips and excluded cols
    shift_cols = [
        c for c in df.columns
        if c != "county_fips"
        and c not in exclude_cols
        # Exclude the production holdout regardless
        and HOLDOUT_SUFFIX not in c
    ]

    # Filter by MIN_YEAR
    filtered_cols = [
        c for c in shift_cols
        if parse_start_year(c) >= MIN_YEAR
    ]

    if not filtered_cols:
        raise ValueError("No training columns remain after year filter and exclusions.")

    raw_matrix = df[filtered_cols].values.astype(float)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw_matrix)

    # Apply presidential weight post-scaling (same as production)
    pres_indices = [i for i, c in enumerate(filtered_cols) if c.startswith("pres_")]
    if PRESIDENTIAL_WEIGHT != 1.0:
        scaled[:, pres_indices] *= PRESIDENTIAL_WEIGHT

    return scaled, filtered_cols


def compute_loo_r(
    scores: np.ndarray,
    full_shift_matrix: np.ndarray,
    training_col_indices: list[int],
    holdout_col_index: int,
) -> float:
    """Return the type-mean LOO Pearson r for a single holdout column."""
    result = holdout_accuracy_county_prior_loo(
        scores,
        full_shift_matrix,
        training_cols=training_col_indices,
        holdout_cols=[holdout_col_index],
    )
    return result["mean_r"]


# ── Part 1: Leave-one-pair-out across presidential pairs ─────────────────────


def run_part1_lopo(df: pd.DataFrame) -> list[dict]:
    """Hold out each presidential pair in turn and measure LOO r.

    For each pair P:
      - Build training matrix from all other pairs (+ non-presidential dims).
      - Run KMeans (J=100, PCA(15,whiten), T=10, pw=8).
      - Compute type-mean LOO r on each held-out shift dimension (d, r, turnout).
      - Report mean r across the three held-out dimensions and r for d-only.

    The d (Democratic share) dimension is the primary target because it's
    what the production forecast is calibrated to. We report it separately.
    """
    print("\n" + "=" * 70)
    print("Part 1: Leave-One-Pair-Out CV — Presidential Shift Pairs")
    print("=" * 70)

    pairs = collect_presidential_pairs(df)
    print(f"Found {len(pairs)} presidential pairs (min_year={MIN_YEAR}, excl. holdout {HOLDOUT_SUFFIX}):")
    for p in pairs:
        print(f"  {p.label}: {p.d_col}, {p.r_col}, {p.turnout_col}")

    results: list[dict] = []

    for held_pair in pairs:
        held_cols = [held_pair.d_col, held_pair.r_col, held_pair.turnout_col]
        print(f"\nHolding out {held_pair.label} ({held_pair.d_col})...")

        # Build the discovery matrix excluding the held-out pair
        training_scaled, training_col_names = build_training_matrix(df, exclude_cols=held_cols)
        print(
            f"  Discovery matrix: {training_scaled.shape[0]} counties x "
            f"{training_scaled.shape[1]} dims"
        )

        # Run type discovery on the reduced matrix
        type_result = discover_types(
            training_scaled,
            j=J,
            random_state=RANDOM_STATE,
            temperature=TEMPERATURE,
            pca_components=PCA_COMPONENTS,
            pca_whiten=PCA_WHITEN,
        )

        # Build the full matrix for validation: training cols (raw) + held-out cols (raw).
        # holdout_accuracy_county_prior_loo expects raw (unscaled) values because it computes
        # county training means in the original shift space.
        raw_training = df[training_col_names].values.astype(float)
        raw_held = df[held_cols].values.astype(float)
        # Stack: training first (indices 0..D-1), then held-out cols (indices D..D+2)
        full_raw = np.column_stack([raw_training, raw_held])

        training_idx = list(range(len(training_col_names)))
        held_idx = list(range(len(training_col_names), len(training_col_names) + len(held_cols)))

        # LOO r for each held-out dimension
        loo_result = holdout_accuracy_county_prior_loo(
            type_result.scores,
            full_raw,
            training_cols=training_idx,
            holdout_cols=held_idx,
        )

        # Per-dimension breakdown: d, r, turnout
        per_dim = loo_result["per_dim_r"]
        loo_r_d = per_dim[0] if per_dim else 0.0
        loo_r_r = per_dim[1] if len(per_dim) > 1 else 0.0
        loo_r_t = per_dim[2] if len(per_dim) > 2 else 0.0

        result = {
            "pair": held_pair.label,
            "loo_r_dem": round(loo_r_d, 4),
            "loo_r_rep": round(loo_r_r, 4),
            "loo_r_turnout": round(loo_r_t, 4),
            "loo_r_mean": round(loo_result["mean_r"], 4),
            "loo_rmse_mean": round(loo_result["mean_rmse"], 4),
        }
        results.append(result)
        print(
            f"  LOO r — dem: {loo_r_d:.4f}, rep: {loo_r_r:.4f}, "
            f"turnout: {loo_r_t:.4f} | mean: {loo_result['mean_r']:.4f}"
        )

    # Summary table
    print("\n── Part 1 Results ──────────────────────────────────────────────────")
    print(f"{'Pair':<8} {'LOO r (D)':<12} {'LOO r (R)':<12} {'LOO r (T)':<12} {'Mean':<10} {'RMSE'}")
    print("-" * 64)
    for r in results:
        print(
            f"{r['pair']:<8} {r['loo_r_dem']:<12.4f} {r['loo_r_rep']:<12.4f} "
            f"{r['loo_r_turnout']:<12.4f} {r['loo_r_mean']:<10.4f} {r['loo_rmse_mean']:.4f}"
        )
    print("-" * 64)
    dem_rs = [r["loo_r_dem"] for r in results]
    mean_rs = [r["loo_r_mean"] for r in results]
    print(
        f"{'Mean':<8} {np.mean(dem_rs):<12.4f} "
        f"{'':12} {'':12} {np.mean(mean_rs):<10.4f}"
    )
    print(
        f"{'Std':<8} {np.std(dem_rs):<12.4f} "
        f"{'':12} {'':12} {np.std(mean_rs):<10.4f}"
    )

    return results


# ── Part 2: Per-type error analysis on production holdout ─────────────────────


def run_part2_type_errors(
    df: pd.DataFrame,
    ta_df: pd.DataFrame,
    feat_df: pd.DataFrame,
) -> list[dict]:
    """Compute MAE and signed bias per type on the production 2020→2024 holdout.

    Uses the production type assignments (data/communities/type_assignments.parquet)
    rather than re-running discovery, so the results reflect the deployed model.

    For each type:
      - Leave-one-out MAE: for each county, predict using the type mean of all
        *other* counties in the same type. This is the honest per-county error —
        using the in-sample type mean (including the county itself) always gives
        zero mean signed error by construction.
      - Signed error (mean LOO predicted minus actual) — positive = over-predicts Dem.
      - Type size (number of counties assigned as dominant type).
      - Key demographic means from county_features_national.parquet.
    """
    print("\n" + "=" * 70)
    print("Part 2: Per-Type Error Analysis — Production 2020→2024 Holdout")
    print("=" * 70)

    holdout_col = "pres_d_shift_20_24"

    # Join type assignments and actual shift values on county_fips
    # ta_df has: county_fips, type_0_score, ..., type_99_score, dominant_type
    analysis = ta_df[["county_fips", "dominant_type"]].copy()
    analysis = analysis.merge(
        df[["county_fips", holdout_col]], on="county_fips", how="inner"
    )

    # LOO type-mean prediction: each county is predicted by the mean of all OTHER
    # counties in the same dominant type, not including itself.
    #
    # Pre-compute type sum and count, then subtract each county's contribution:
    #   loo_type_mean_i = (type_sum - actual_i) / (type_count - 1)
    # For singleton types (n=1) we fall back to the global mean (can't LOO).
    global_mean = analysis[holdout_col].mean()
    type_sum = analysis.groupby("dominant_type")[holdout_col].transform("sum")
    type_count = analysis.groupby("dominant_type")[holdout_col].transform("count")

    loo_mean = (type_sum - analysis[holdout_col]) / (type_count - 1).clip(lower=1)
    # For singletons, (type_count - 1) == 0 → replace with global mean
    is_singleton = type_count == 1
    loo_mean = loo_mean.where(~is_singleton, global_mean)

    analysis["loo_type_mean_pred"] = loo_mean
    analysis["error"] = analysis["loo_type_mean_pred"] - analysis[holdout_col]
    analysis["abs_error"] = analysis["error"].abs()

    # Aggregate per type
    per_type = (
        analysis.groupby("dominant_type")
        .agg(
            mae=("abs_error", "mean"),
            mean_signed_error=("error", "mean"),
            county_count=("county_fips", "count"),
            actual_mean=(holdout_col, "mean"),
            pred_mean=("loo_type_mean_pred", "mean"),
        )
        .reset_index()
    )
    per_type.rename(columns={"dominant_type": "type_id"}, inplace=True)
    per_type = per_type.sort_values("mae", ascending=False)

    # Join demographic data for the top-N worst types.
    # feat_df uses county_fips as a string; ta_df uses the same format.
    available_demo_cols = [c for c in DEMO_COLS if c in feat_df.columns]
    if not available_demo_cols:
        print("  Warning: no demographic columns found in features parquet.")
        demo_by_type: pd.DataFrame = pd.DataFrame()
    else:
        feat_slim = feat_df[["county_fips"] + available_demo_cols].copy()
        # Ensure FIPS is string-matched
        feat_slim["county_fips"] = feat_slim["county_fips"].astype(str).str.zfill(5)
        analysis["county_fips"] = analysis["county_fips"].astype(str).str.zfill(5)
        with_demo = analysis.merge(feat_slim, on="county_fips", how="left")
        demo_by_type = (
            with_demo.groupby("dominant_type")[available_demo_cols]
            .mean()
            .reset_index()
            .rename(columns={"dominant_type": "type_id"})
        )

    top_n = 10
    worst_types = per_type.head(top_n)

    # Flag small types: for types with n<=5, high LOO MAE mostly reflects within-type
    # variance rather than prediction bias — there are too few reference counties to form
    # a stable type mean. These are real model weaknesses (insufficient type cohesion)
    # but are distinct from systematic miscalibration in larger types.
    small_type_threshold = 10

    print(f"\nTop {top_n} worst-performing types (highest MAE on 2020→2024 D shift):")
    print(f"  Note: types with n<{small_type_threshold} marked with * — high MAE may reflect")
    print(f"  within-type variance (too few counties to form a stable reference mean)")
    print(
        f"\n{'Type':<7} {'MAE':<8} {'Bias':<10} {'N':<6} "
        f"{'Act. mean':<12} {'Pred. mean':<12}"
    )
    print("-" * 60)
    for _, row in worst_types.iterrows():
        flag = "*" if int(row.county_count) < small_type_threshold else " "
        print(
            f"{int(row.type_id):<5}{flag} {row.mae:<8.4f} {row.mean_signed_error:<10.4f} "
            f"{int(row.county_count):<6} {row.actual_mean:<12.4f} {row.pred_mean:<12.4f}"
        )

    if not demo_by_type.empty:
        print(f"\nDemographics for top {top_n} worst types:")
        worst_type_ids = worst_types["type_id"].tolist()
        demo_subset = demo_by_type[demo_by_type["type_id"].isin(worst_type_ids)]
        demo_subset = demo_subset.set_index("type_id").loc[worst_type_ids]
        # Print subset of demo cols that are available
        display_cols = [c for c in ["pct_white_nh", "pct_black", "pct_hispanic",
                                    "pct_bachelors_plus", "manufacturing_share",
                                    "evangelical_share"] if c in demo_subset.columns]
        print(f"\n{'Type':<6} " + " ".join(f"{c[:12]:<14}" for c in display_cols))
        print("-" * (6 + 14 * len(display_cols)))
        for tid in worst_type_ids:
            if tid not in demo_subset.index:
                continue
            row = demo_subset.loc[tid]
            vals = " ".join(f"{row[c]:<14.3f}" for c in display_cols)
            print(f"{int(tid):<6} {vals}")

    # Large-type worst performers: same as above but filtered to n >= small_type_threshold
    # These are the cases where the model genuinely fails to cluster a coherent group.
    large_types_worst = per_type[per_type["county_count"] >= small_type_threshold].head(top_n)
    print(f"\nTop {top_n} worst large types (n>={small_type_threshold}, cleaner signal):")
    print(
        f"\n{'Type':<7} {'MAE':<8} {'Bias':<10} {'N':<6} "
        f"{'Act. mean':<12} {'Pred. mean':<12}"
    )
    print("-" * 60)
    for _, row in large_types_worst.iterrows():
        print(
            f"{int(row.type_id):<7} {row.mae:<8.4f} {row.mean_signed_error:<10.4f} "
            f"{int(row.county_count):<6} {row.actual_mean:<12.4f} {row.pred_mean:<12.4f}"
        )

    # Also show best 5 for comparison
    best_types = per_type.tail(5).sort_values("mae")
    print(f"\nTop 5 best-performing types (lowest MAE):")
    print(
        f"\n{'Type':<7} {'MAE':<8} {'Bias':<10} {'N':<6} "
        f"{'Act. mean':<12} {'Pred. mean':<12}"
    )
    print("-" * 60)
    for _, row in best_types.iterrows():
        print(
            f"{int(row.type_id):<7} {row.mae:<8.4f} {row.mean_signed_error:<10.4f} "
            f"{int(row.county_count):<6} {row.actual_mean:<12.4f} {row.pred_mean:<12.4f}"
        )

    # Return as list of dicts for JSON serialisation
    per_type_records = per_type.to_dict(orient="records")
    if not demo_by_type.empty:
        demo_dict = demo_by_type.set_index("type_id").to_dict(orient="index")
        for rec in per_type_records:
            tid = rec["type_id"]
            rec["demographics"] = {
                k: round(float(v), 4) for k, v in demo_dict.get(tid, {}).items()
            }

    return per_type_records


# ── Part 3: Geographic concentration of worst-performing types ────────────────


def run_part3_geography(
    df: pd.DataFrame,
    ta_df: pd.DataFrame,
    per_type_records: list[dict],
    top_n: int = 10,
) -> dict:
    """Show which states the worst-performing types are concentrated in.

    Uses the FIPS crosswalk to extract state abbreviations from county_fips.
    The first 2 digits of a 5-digit FIPS code identify the state.

    Returns a dict mapping type_id -> {state_abbr: county_count}.
    """
    print("\n" + "=" * 70)
    print("Part 3: Geographic Distribution of Worst-Performing Types")
    print("=" * 70)

    # Load FIPS → state crosswalk (county_name contains ", ST" suffix)
    xwalk_path = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"
    if not xwalk_path.exists():
        print(f"  Warning: FIPS crosswalk not found at {xwalk_path}. Skipping Part 3.")
        return {}

    xwalk = pd.read_csv(xwalk_path)
    xwalk["county_fips"] = xwalk["county_fips"].astype(str).str.zfill(5)
    # state_abbr is the 2-letter code at the end of county_name, e.g. "Autauga County, AL"
    xwalk["state_abbr"] = xwalk["county_name"].str.extract(r", ([A-Z]{2})$")
    xwalk = xwalk[["county_fips", "state_abbr"]].dropna()

    # Join type assignments with state info
    ta_geo = ta_df[["county_fips", "dominant_type"]].copy()
    ta_geo["county_fips"] = ta_geo["county_fips"].astype(str).str.zfill(5)
    ta_geo = ta_geo.merge(xwalk, on="county_fips", how="left")

    # Sort per_type_records by MAE (descending) — already sorted from Part 2
    worst_type_ids = [rec["type_id"] for rec in per_type_records[:top_n]]

    geo_summary: dict = {}
    print(f"\nState distribution for the {top_n} highest-MAE types (all sizes):")
    print("  Types with missing state data are Alaska boroughs absent from the FIPS crosswalk.")

    for type_id in worst_type_ids:
        counties_in_type = ta_geo[ta_geo["dominant_type"] == type_id]
        state_counts = (
            counties_in_type["state_abbr"]
            .value_counts()
            .head(8)  # top 8 states per type for readability
        )
        state_str = ", ".join(f"{st}({n})" for st, n in state_counts.items())
        mae = next(
            (round(r["mae"], 4) for r in per_type_records if r["type_id"] == type_id), 0.0
        )
        print(f"  Type {int(type_id):>3} (MAE={mae:.4f}, n={len(counties_in_type)}): {state_str}")
        geo_summary[int(type_id)] = state_counts.to_dict()

    return geo_summary


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Cross-Election Leave-One-Pair-Out Experiment")
    print("=" * 70)
    print(f"Settings: J={J}, pw={PRESIDENTIAL_WEIGHT}, PCA({PCA_COMPONENTS}, whiten={PCA_WHITEN}), T={TEMPERATURE}")
    print(f"Min year filter: {MIN_YEAR}. Holdout excluded from discovery: pres_*_{HOLDOUT_SUFFIX}")

    # ── Load data ─────────────────────────────────────────────────────────────
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    feat_path = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"

    for p in [shifts_path, ta_path, feat_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required data file not found: {p}\n"
                "Run the data assembly pipeline before this script."
            )

    df = pd.read_parquet(shifts_path)
    ta_df = pd.read_parquet(ta_path)
    feat_df = pd.read_parquet(feat_path)

    print(f"\nLoaded shifts: {df.shape} | type assignments: {ta_df.shape} | features: {feat_df.shape}")

    # ── Part 1 ────────────────────────────────────────────────────────────────
    part1_results = run_part1_lopo(df)

    # ── Part 2 ────────────────────────────────────────────────────────────────
    per_type_records = run_part2_type_errors(df, ta_df, feat_df)

    # ── Part 3 ────────────────────────────────────────────────────────────────
    geo_summary = run_part3_geography(df, ta_df, per_type_records, top_n=10)

    # ── Save JSON results ─────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "data" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_election_loo_results.json"

    output = {
        "settings": {
            "j": J,
            "presidential_weight": PRESIDENTIAL_WEIGHT,
            "pca_components": PCA_COMPONENTS,
            "pca_whiten": PCA_WHITEN,
            "temperature": TEMPERATURE,
            "min_year": MIN_YEAR,
            "holdout_suffix": HOLDOUT_SUFFIX,
        },
        "part1_lopo": part1_results,
        "part2_per_type_errors": [
            {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in rec.items()}
            for rec in per_type_records
        ],
        "part3_geography": {str(k): v for k, v in geo_summary.items()},
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_path}")
    print("=" * 70)

    # ── Cross-election summary ────────────────────────────────────────────────
    if part1_results:
        dem_rs = [r["loo_r_dem"] for r in part1_results]
        print(
            f"\nCross-election LOO r (dem shift): "
            f"mean={np.mean(dem_rs):.3f}, std={np.std(dem_rs):.3f}, "
            f"range=[{min(dem_rs):.3f}, {max(dem_rs):.3f}]"
        )
        hardest = min(part1_results, key=lambda r: r["loo_r_dem"])
        easiest = max(part1_results, key=lambda r: r["loo_r_dem"])
        print(f"Hardest pair to predict: {hardest['pair']} (LOO r={hardest['loo_r_dem']:.4f})")
        print(f"Easiest pair to predict: {easiest['pair']} (LOO r={easiest['loo_r_dem']:.4f})")

    worst_type = per_type_records[0]
    print(
        f"\nWorst type (MAE={worst_type['mae']:.4f}): "
        f"type {int(worst_type['type_id'])}, "
        f"n={int(worst_type['county_count'])} counties, "
        f"bias={worst_type['mean_signed_error']:.4f}"
    )


if __name__ == "__main__":
    main()
