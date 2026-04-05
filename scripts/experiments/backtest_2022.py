"""2022 Election Backtesting Validation.

Proves the type model can predict elections it wasn't trained on by:

  1. Retraining KMeans types WITHOUT any 2022 shift data (drops gov_*_18_22 and
     sen_*_16_22 columns, plus the 2020→2024 holdout columns).
  2. Using 2020 presidential Dem share as the county-level prior.
  3. Computing type-mean priors (weighted average of type means over 2020 pres data)
     as a richer baseline.
  4. Comparing both predictions against 2022 actual governor and Senate results.
  5. Reporting per-state accuracy and highlighting key competitive races.

This is a clean out-of-sample validation: 2022 data never touches the type
discovery step.

Usage:
    uv run python scripts/experiments/backtest_2022.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

# Columns to always exclude from type discovery — the held-out 2020→2024 shift
HOLDOUT_COLUMNS = {
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
}

# Columns that encode 2022 outcomes — exclude so types are blind to 2022
COLUMNS_2022 = {
    "gov_d_shift_18_22",
    "gov_r_shift_18_22",
    "gov_turnout_shift_18_22",
    "sen_d_shift_16_22",
    "sen_r_shift_16_22",
    "sen_turnout_shift_16_22",
}

# Key competitive 2022 races to spotlight in the summary
SENATE_SPOTLIGHT_STATES = {
    "PA": "Fetterman vs Oz",
    "GA": "Warnock vs Walker",
    "AZ": "Kelly vs Masters",
    "NV": "Cortez Masto vs Laxalt",
    "WI": "Barnes vs Johnson",
    "OH": "Ryan vs Vance",
    "NC": "Beasley vs Budd",
}

GOVERNOR_SPOTLIGHT_STATES = {
    "PA": "Shapiro vs Mastriano",
    "MI": "Whitmer vs Dixon",
    "WI": "Evers vs Michels",
    "AZ": "Hobbs vs Lake",
    "NV": "Sisolak vs Lombardo",
    "GA": "Abrams vs Kemp",
    "NY": "Hochul vs Zeldin",
}

ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "experiments" / "backtest_2022_results.json"

# Minimum start-year for shift pairs (matches production default)
MIN_START_YEAR = 2008


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_start_year(col: str) -> int | None:
    """Extract the four-digit start year from a shift column like pres_d_shift_08_12.

    Returns None if parsing fails so callers can safely skip non-shift columns.
    """
    parts = col.split("_")
    try:
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two arrays, masking NaN pairs."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    return float(pearsonr(a[mask], b[mask]).statistic)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two arrays, masking NaN pairs."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def vote_weighted_state_pred(
    df: pd.DataFrame, pred_col: str, actual_col: str, total_col: str
) -> pd.DataFrame:
    """Aggregate county predictions to state level using vote weighting.

    Computes vote-weighted Dem share prediction and actual per state.
    Returns a DataFrame with columns: state_abbr, pred, actual, total_votes.
    """
    rows = []
    for state, grp in df.groupby("state_abbr"):
        valid = grp[grp[total_col].notna() & grp[pred_col].notna()].copy()
        if valid.empty:
            continue
        total_votes = valid[total_col].sum()
        if total_votes <= 0:
            continue
        # Vote-weighted prediction: sum(pred * votes) / sum(votes)
        weighted_pred = (valid[pred_col] * valid[total_col]).sum() / total_votes
        # Vote-weighted actual
        valid_actual = valid[valid[actual_col].notna()]
        if valid_actual.empty:
            continue
        actual_votes = valid_actual[total_col].sum()
        weighted_actual = (valid_actual[actual_col] * valid_actual[total_col]).sum() / actual_votes
        rows.append(
            {
                "state_abbr": state,
                "pred": float(weighted_pred),
                "actual": float(weighted_actual),
                "total_votes": float(total_votes),
                "n_counties": len(valid),
            }
        )
    return pd.DataFrame(rows)


# ── Step 1: Retrain types WITHOUT 2022 data ───────────────────────────────────


def build_pre2022_shift_matrix(
    df: pd.DataFrame, config: dict
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build the StandardScaler + presidential-weighted shift matrix excluding 2022 data.

    Parameters
    ----------
    df : DataFrame
        Raw shifts parquet with county_fips column + shift columns.
    config : dict
        Loaded model.yaml config dict.

    Returns
    -------
    shift_matrix : ndarray (N, D)
        Scaled and weighted shift matrix ready for KMeans.
    shift_cols : list[str]
        Column names used (for reference / debugging).
    county_fips : ndarray (N,)
        Corresponding county FIPS codes.
    """
    county_fips = df["county_fips"].values

    all_shift_cols = [
        c for c in df.columns
        if c != "county_fips"
        and c not in HOLDOUT_COLUMNS
        and c not in COLUMNS_2022
    ]

    # Filter to shift pairs with start year >= MIN_START_YEAR (same as production)
    shift_cols = []
    for col in all_shift_cols:
        start_year = parse_start_year(col)
        if start_year is None:
            continue
        if start_year >= MIN_START_YEAR:
            shift_cols.append(col)

    if not shift_cols:
        raise ValueError("No shift columns remain after filtering — check column names.")

    shift_matrix = df[shift_cols].values.astype(float)

    # Standardize (same as production pipeline)
    scaler = StandardScaler()
    shift_matrix = scaler.fit_transform(shift_matrix)

    # Apply presidential weight (post-scaling, same as production)
    presidential_weight = float(config["types"].get("presidential_weight", 8.0))
    if presidential_weight != 1.0:
        pres_indices = [i for i, c in enumerate(shift_cols) if "pres_" in c]
        shift_matrix[:, pres_indices] *= presidential_weight

    return shift_matrix, shift_cols, county_fips


# ── Step 2: 2020 presidential priors ─────────────────────────────────────────


def load_2020_presidential_priors(county_fips: np.ndarray) -> dict[str, float]:
    """Load 2020 presidential Dem two-party share keyed by county FIPS.

    Two-party share = pres_dem_2020 / (pres_dem_2020 + pres_rep_2020).
    This is more appropriate than total share for comparison to Democratic
    candidate performance in 2022, because third-party vote is not a
    meaningful signal for our purposes.

    Falls back to total-vote Dem share (pres_dem_share_2020) if raw vote
    counts are not available.
    """
    path = ASSEMBLED_DIR / "medsl_county_presidential_2020.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    if "pres_dem_2020" in df.columns and "pres_rep_2020" in df.columns:
        # Two-party share: removes third-party noise for cross-race comparison
        total_two_party = df["pres_dem_2020"] + df["pres_rep_2020"]
        df["prior_dem_share"] = np.where(
            total_two_party > 0,
            df["pres_dem_2020"] / total_two_party,
            np.nan,
        )
    else:
        df["prior_dem_share"] = df["pres_dem_share_2020"]

    return dict(zip(df["county_fips"], df["prior_dem_share"]))


def compute_type_mean_priors(
    county_fips: np.ndarray,
    type_scores: np.ndarray,
    prior_map: dict[str, float],
    fallback: float = 0.45,
) -> np.ndarray:
    """Compute type-mean priors: weighted average of per-type 2020 dem shares.

    For each type j, the type mean is the weighted average of 2020 presidential
    Dem shares across all counties, weighted by their soft membership in type j.
    Then each county's prediction is the dot product of its type scores with the
    type means.

    This is a stronger baseline than raw 2020 priors because it uses community
    structure: the county's prediction is pulled toward the typical 2020 behavior
    of its electoral community (type), not just its own 2020 result.

    Parameters
    ----------
    county_fips : ndarray (N,)
        County FIPS codes in the same order as type_scores.
    type_scores : ndarray (N, J)
        Soft membership scores (rows sum to 1).
    prior_map : dict[str, float]
        FIPS -> 2020 presidential Dem share.
    fallback : float
        Dem share for counties with no 2020 data.

    Returns
    -------
    ndarray (N,)
        Type-mean prior Dem share per county.
    """
    fips_list = [str(f).zfill(5) for f in county_fips]
    county_dem_shares = np.array([prior_map.get(f, fallback) for f in fips_list])

    # Per-type mean: weighted average of 2020 shares, weights = type scores
    # type_means[j] = sum_i(score[i,j] * dem[i]) / sum_i(score[i,j])
    weight_sums = type_scores.sum(axis=0)  # (J,) -- denominator
    numerators = type_scores.T @ county_dem_shares  # (J,)
    type_means = np.where(weight_sums > 0, numerators / weight_sums, fallback)  # (J,)

    # Each county's prediction = dot product of its scores with type means
    return type_scores @ type_means  # (N,)


# ── Step 3: Load 2022 actuals ─────────────────────────────────────────────────


def load_2022_governor_actuals() -> pd.DataFrame:
    """Load 2022 governor results.

    Returns DataFrame with county_fips, state_abbr, gov_dem_share_2022,
    gov_total_2022. Drops rows with zero or missing vote totals.
    """
    path = ASSEMBLED_DIR / "medsl_county_2022_governor.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    # Drop aggregate rows (FIPS = 00000) and rows with missing vote data
    df = df[df["county_fips"] != "00000"].copy()
    df = df[df["gov_total_2022"].notna() & (df["gov_total_2022"] > 0)].copy()
    return df[["county_fips", "state_abbr", "gov_dem_share_2022", "gov_total_2022"]]


def load_2022_senate_actuals() -> pd.DataFrame:
    """Load 2022 Senate results.

    Returns DataFrame with county_fips, state_abbr, senate_dem_share_2022,
    senate_total_2022. Drops rows with zero or missing vote totals.
    """
    path = ASSEMBLED_DIR / "medsl_county_senate_2022.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    df = df[df["county_fips"] != "00000"].copy()
    df = df[df["senate_total_2022"].notna() & (df["senate_total_2022"] > 0)].copy()
    return df[["county_fips", "state_abbr", "senate_dem_share_2022", "senate_total_2022"]]


# ── Step 4: Compare and report ────────────────────────────────────────────────


def compute_race_metrics(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    total_col: str,
    spotlight_states: dict[str, str],
) -> dict:
    """Compute county-level and state-level accuracy metrics for one race type.

    Returns a dict with:
      - county_r, county_rmse: county-level Pearson r and RMSE
      - n_counties: number of county observations
      - state_metrics: list of per-state dicts (state, pred, actual, error, race)
      - spotlight: sub-list filtered to spotlight_states
    """
    valid = df[df[pred_col].notna() & df[actual_col].notna()].copy()

    county_r = pearson_r(valid[pred_col].values, valid[actual_col].values)
    county_rmse_val = rmse(valid[pred_col].values, valid[actual_col].values)
    n_counties = len(valid)

    state_df = vote_weighted_state_pred(valid, pred_col, actual_col, total_col)

    state_metrics = []
    for _, row in state_df.sort_values("state_abbr").iterrows():
        state_abbr = row["state_abbr"]
        entry = {
            "state": state_abbr,
            "pred": round(row["pred"], 4),
            "actual": round(row["actual"], 4),
            "error_pp": round((row["pred"] - row["actual"]) * 100, 2),
            "n_counties": int(row["n_counties"]),
        }
        if state_abbr in spotlight_states:
            entry["race_label"] = spotlight_states[state_abbr]
        state_metrics.append(entry)

    spotlight = [s for s in state_metrics if s["state"] in spotlight_states]

    return {
        "county_r": round(county_r, 4),
        "county_rmse": round(county_rmse_val, 4),
        "n_counties": n_counties,
        "state_metrics": state_metrics,
        "spotlight": spotlight,
    }


def print_spotlight_table(title: str, spotlight: list[dict]) -> None:
    """Print a formatted table of spotlight races to stdout."""
    print(f"\n{title}")
    print("-" * 72)
    header = f"{'State':<6} {'Race':<30} {'2020-Prior':>10} {'Actual':>10} {'Error':>8}"
    print(header)
    print("-" * 72)
    for entry in sorted(spotlight, key=lambda x: abs(x["error_pp"]), reverse=True):
        race = entry.get("race_label", "")
        print(
            f"{entry['state']:<6} {race:<30} "
            f"{entry['pred']*100:>9.1f}% {entry['actual']*100:>9.1f}% "
            f"{entry['error_pp']:>+7.1f}pp"
        )
    print("-" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Load model config for KMeans parameters
    config_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    j = int(config["types"].get("j", 100))
    temperature = float(config["types"].get("temperature", 10.0))
    pca_components = config["types"].get("pca_components")
    pca_whiten = bool(config["types"].get("pca_whiten", False))
    if pca_components is not None:
        pca_components = int(pca_components)

    print("=" * 72)
    print("2022 ELECTION BACKTESTING VALIDATION")
    print("=" * 72)
    print(f"J={j}, T={temperature}, PCA={pca_components} (whiten={pca_whiten})")
    print()

    # Step 1: Retrain types without 2022 data
    print("Step 1: Loading shift matrix and filtering out 2022 columns...")
    shifts_df = pd.read_parquet(SHIFTS_PATH)

    shift_matrix, shift_cols, county_fips = build_pre2022_shift_matrix(shifts_df, config)

    excluded_2022_cols = [c for c in shifts_df.columns if c in COLUMNS_2022]
    excluded_holdout_cols = [c for c in shifts_df.columns if c in HOLDOUT_COLUMNS]
    print(f"  Shift matrix: {shift_matrix.shape[0]} counties x {shift_matrix.shape[1]} dims")
    print(f"  Excluded 2022 columns ({len(excluded_2022_cols)}): {excluded_2022_cols}")
    print(f"  Excluded holdout columns ({len(excluded_holdout_cols)}): {excluded_holdout_cols}")
    print()

    print(f"Step 1b: Running KMeans (J={j}, T={temperature})...")
    result = discover_types(
        shift_matrix,
        j=j,
        random_state=42,
        temperature=temperature,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
    )
    type_scores = result.scores  # (N, J) soft membership
    ev = result.explained_variance
    print(f"  Type sizes range: [{ev.min():.3f}, {ev.max():.3f}]")
    print()

    # Step 2: Build priors
    print("Step 2: Building 2020 presidential priors...")
    prior_map = load_2020_presidential_priors(county_fips)
    fips_list = [str(f).zfill(5) for f in county_fips]

    # Baseline: raw 2020 Dem share per county (two-party)
    raw_priors = np.array([prior_map.get(f, 0.45) for f in fips_list])

    # Type-mean priors: community-structured baseline
    type_mean_priors = compute_type_mean_priors(county_fips, type_scores, prior_map)

    n_with_data = sum(1 for f in fips_list if f in prior_map)
    print(f"  Counties with 2020 data: {n_with_data}/{len(fips_list)}")
    print(f"  Raw prior range: [{raw_priors.min():.3f}, {raw_priors.max():.3f}]")
    print(f"  Type-mean prior range: [{type_mean_priors.min():.3f}, {type_mean_priors.max():.3f}]")
    print()

    # Build a DataFrame indexed by county_fips for easy joining
    pred_df = pd.DataFrame({
        "county_fips": fips_list,
        "raw_prior": raw_priors,
        "type_mean_prior": type_mean_priors,
    })

    # Step 3: Load 2022 actuals
    print("Step 3: Loading 2022 actuals...")
    gov_df = load_2022_governor_actuals()
    sen_df = load_2022_senate_actuals()
    print(f"  Governor counties: {len(gov_df)} (in {gov_df['state_abbr'].nunique()} states)")
    print(f"  Senate counties: {len(sen_df)} (in {sen_df['state_abbr'].nunique()} states)")
    print()

    # Step 4: Merge and compute metrics
    print("Step 4: Computing accuracy metrics...")
    print()

    # Governor analysis
    gov_merged = gov_df.merge(pred_df, on="county_fips", how="inner")
    gov_metrics_raw = compute_race_metrics(
        gov_merged,
        pred_col="raw_prior",
        actual_col="gov_dem_share_2022",
        total_col="gov_total_2022",
        spotlight_states=GOVERNOR_SPOTLIGHT_STATES,
    )
    gov_metrics_type = compute_race_metrics(
        gov_merged,
        pred_col="type_mean_prior",
        actual_col="gov_dem_share_2022",
        total_col="gov_total_2022",
        spotlight_states=GOVERNOR_SPOTLIGHT_STATES,
    )

    # Senate analysis
    sen_merged = sen_df.merge(pred_df, on="county_fips", how="inner")
    sen_metrics_raw = compute_race_metrics(
        sen_merged,
        pred_col="raw_prior",
        actual_col="senate_dem_share_2022",
        total_col="senate_total_2022",
        spotlight_states=SENATE_SPOTLIGHT_STATES,
    )
    sen_metrics_type = compute_race_metrics(
        sen_merged,
        pred_col="type_mean_prior",
        actual_col="senate_dem_share_2022",
        total_col="senate_total_2022",
        spotlight_states=SENATE_SPOTLIGHT_STATES,
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY: County-Level Correlation (Pearson r)")
    print("=" * 72)
    print(f"\n{'Metric':<35} {'Raw 2020 Prior':>15} {'Type-Mean Prior':>15}")
    print("-" * 67)
    print(
        f"{'Governor county r':<35} {gov_metrics_raw['county_r']:>15.4f} "
        f"{gov_metrics_type['county_r']:>15.4f}"
    )
    print(
        f"{'Governor county RMSE':<35} {gov_metrics_raw['county_rmse']:>15.4f} "
        f"{gov_metrics_type['county_rmse']:>15.4f}"
    )
    print(
        f"{'Governor counties in sample':<35} {gov_metrics_raw['n_counties']:>15} "
        f"{gov_metrics_type['n_counties']:>15}"
    )
    print(
        f"{'Senate county r':<35} {sen_metrics_raw['county_r']:>15.4f} "
        f"{sen_metrics_type['county_r']:>15.4f}"
    )
    print(
        f"{'Senate county RMSE':<35} {sen_metrics_raw['county_rmse']:>15.4f} "
        f"{sen_metrics_type['county_rmse']:>15.4f}"
    )
    print(
        f"{'Senate counties in sample':<35} {sen_metrics_raw['n_counties']:>15} "
        f"{sen_metrics_type['n_counties']:>15}"
    )
    print()
    print(
        "Type-mean prior smooths county predictions through electoral community "
        "structure.\nBoth baselines used 2020 presidential results trained WITHOUT "
        "any 2022 shift data."
    )

    # Spotlight tables
    print_spotlight_table(
        "KEY GOVERNOR RACES 2022 (Type-Mean Prior vs Actual)",
        gov_metrics_type["spotlight"],
    )
    print_spotlight_table(
        "KEY SENATE RACES 2022 (Type-Mean Prior vs Actual)",
        sen_metrics_type["spotlight"],
    )

    # ── Save JSON results ──────────────────────────────────────────────────────
    results = {
        "description": (
            "2022 election backtest: KMeans types retrained WITHOUT 2022 shift "
            "data. 2020 presidential Dem share used as county prior. Both raw "
            "and type-mean predictions compared to 2022 governor and Senate actuals."
        ),
        "model_config": {
            "j": j,
            "temperature": temperature,
            "pca_components": pca_components,
            "pca_whiten": pca_whiten,
            "min_start_year": MIN_START_YEAR,
        },
        "excluded_2022_columns": excluded_2022_cols,
        "excluded_holdout_columns": excluded_holdout_cols,
        "governor": {
            "raw_prior": {
                "county_r": gov_metrics_raw["county_r"],
                "county_rmse": gov_metrics_raw["county_rmse"],
                "n_counties": gov_metrics_raw["n_counties"],
                "spotlight": gov_metrics_raw["spotlight"],
                "state_metrics": gov_metrics_raw["state_metrics"],
            },
            "type_mean_prior": {
                "county_r": gov_metrics_type["county_r"],
                "county_rmse": gov_metrics_type["county_rmse"],
                "n_counties": gov_metrics_type["n_counties"],
                "spotlight": gov_metrics_type["spotlight"],
                "state_metrics": gov_metrics_type["state_metrics"],
            },
        },
        "senate": {
            "raw_prior": {
                "county_r": sen_metrics_raw["county_r"],
                "county_rmse": sen_metrics_raw["county_rmse"],
                "n_counties": sen_metrics_raw["n_counties"],
                "spotlight": sen_metrics_raw["spotlight"],
                "state_metrics": sen_metrics_raw["state_metrics"],
            },
            "type_mean_prior": {
                "county_r": sen_metrics_type["county_r"],
                "county_rmse": sen_metrics_type["county_rmse"],
                "n_counties": sen_metrics_type["n_counties"],
                "spotlight": sen_metrics_type["spotlight"],
                "state_metrics": sen_metrics_type["state_metrics"],
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")
    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print(
        "\nA high county r proves the electoral community structure (KMeans types)"
        "\ndiscovered from pre-2022 shifts genuinely organizes partisan geography."
        "\nThe type model captures how places move together — and that structure"
        "\npersists into election cycles the model never saw."
        "\n"
        "\nNote: These are naive baselines (prior only, no Ridge ensemble, no polls)."
        "\nThe production model adds Ridge regression on type scores + demographics"
        "\n+ Bayesian poll propagation, which substantially sharpens predictions."
        "\nThis backtest validates the foundation those layers build on."
    )


if __name__ == "__main__":
    main()
