"""Calibration analysis: model predictions vs 2020 and 2024 actual presidential results.

Compares the type-primary model's prior-only predictions against actual county-level
Dem shares for FL, GA, and AL. Produces:
  - Per-state calibration metrics (MAE, RMSE, bias, Pearson r)
  - Per-super-type bias breakdown
  - data/validation/calibration_2024.csv  (predicted vs actual, one row per county)
  - Leave-one-out backward calibration using 2016 type means as priors

Usage:
    uv run python scripts/calibration_analysis.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Resolve project root and data root
# ---------------------------------------------------------------------------
# In a git worktree the code lives at WORKTREE_ROOT but gitignored data such
# as data/assembled/*.parquet lives at the main (common) repository root.
# We walk up from this file to find a root that actually contains data/, falling
# back to the git common dir when the worktree has only a sparse data/ copy.

_THIS_FILE = Path(__file__).resolve()
_WORKTREE_ROOT = _THIS_FILE.parents[1]  # …/worktrees/agent-XXXX


def _find_data_root() -> Path:
    """Return the Path that contains data/assembled/ (main repo or worktree)."""
    candidate = _WORKTREE_ROOT
    if (candidate / "data" / "assembled").is_dir():
        return candidate
    # Ask git for the common dir (contains .git of the main repo)
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(candidate),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        main_root = Path(common).resolve().parent  # strip trailing /.git
        if (main_root / "data" / "assembled").is_dir():
            return main_root
    except Exception:
        pass
    return candidate  # best effort


PROJECT_ROOT = _WORKTREE_ROOT
DATA_ROOT = _find_data_root()


# ---------------------------------------------------------------------------
# Metric functions (also imported by tests/test_calibration.py)
# ---------------------------------------------------------------------------


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error between actual and predicted arrays."""
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error between actual and predicted arrays."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error (predicted - actual). Positive = over-prediction."""
    return float(np.mean(predicted - actual))


def compute_pearson_r(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation between actual and predicted arrays."""
    if len(actual) < 2:
        return float("nan")
    r, _ = stats.pearsonr(actual, predicted)
    return float(r)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute all calibration metrics as a dict."""
    return {
        "mae": compute_mae(actual, predicted),
        "rmse": compute_rmse(actual, predicted),
        "bias": compute_bias(actual, predicted),
        "pearson_r": compute_pearson_r(actual, predicted),
        "n": len(actual),
    }


# ---------------------------------------------------------------------------
# Core prediction logic (prior-only, no polls)
# ---------------------------------------------------------------------------


def prior_only_predictions(
    type_scores: np.ndarray,
    type_priors: np.ndarray,
) -> np.ndarray:
    """Compute county Dem share predictions using type priors only (no polls).

    Mirrors the formula in predict_2026_types.predict_race (prior-only branch):
        pred = sum_j(|score_j| * prior_j) / sum_j(|score_j|)

    Since scores are non-negative and sum to 1, |score_j| == score_j,
    and the formula reduces to a simple dot product with the score vector.

    Parameters
    ----------
    type_scores : ndarray of shape (N, J)
        County soft membership scores (non-negative, sum to 1 per row).
    type_priors : ndarray of shape (J,)
        Prior Dem share per type.

    Returns
    -------
    ndarray of shape (N,)
        Predicted Dem share per county, clipped to [0, 1].
    """
    abs_scores = np.abs(type_scores)  # already >= 0, kept for generality
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    pred = (abs_scores * type_priors[None, :]).sum(axis=1) / weight_sums
    return np.clip(pred, 0.0, 1.0)


def build_type_priors_from_election(
    type_scores: np.ndarray,
    actual_dem_share: np.ndarray,
    county_total_votes: np.ndarray,
    J: int,
) -> np.ndarray:
    """Compute type-mean Dem shares from actual election results.

    Uses a population-weighted average of county Dem shares, weighted by both
    the county's total votes and its type score. This approximates what the
    type priors would be if computed from that election.

    Parameters
    ----------
    type_scores : ndarray of shape (N, J)
    actual_dem_share : ndarray of shape (N,)
    county_total_votes : ndarray of shape (N,)
    J : int

    Returns
    -------
    ndarray of shape (J,)
        Estimated prior Dem share per type.
    """
    priors = np.zeros(J)
    for j in range(J):
        weights = type_scores[:, j] * county_total_votes
        w_sum = weights.sum()
        if w_sum > 0:
            priors[j] = np.dot(weights, actual_dem_share) / w_sum
        else:
            priors[j] = actual_dem_share.mean()
    return priors


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_STATE_FIPS = {"12": "FL", "13": "GA", "01": "AL"}


def load_inputs() -> dict:
    """Load all required data artifacts."""
    ta = pd.read_parquet(DATA_ROOT / "data/communities/type_assignments.parquet")
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)
    ta["state"] = ta["county_fips"].str[:2].map(_STATE_FIPS)

    priors_df = pd.read_parquet(DATA_ROOT / "data/communities/type_priors.parquet")
    J = 20
    type_priors_2024 = np.full(J, 0.45)
    for _, row in priors_df.iterrows():
        t = int(row["type_id"])
        if t < J:
            type_priors_2024[t] = float(row["prior_dem_share"])

    cov_df = pd.read_parquet(DATA_ROOT / "data/covariance/type_covariance.parquet")
    type_covariance = cov_df.values[:J, :J]

    score_cols = [f"type_{j}_score" for j in range(J)]
    type_scores = ta[score_cols].values  # (N, J)

    # Super-type info from type_profiles
    tp = pd.read_parquet(DATA_ROOT / "data/communities/type_profiles.parquet")
    st = pd.read_parquet(DATA_ROOT / "data/communities/super_types.parquet")
    type_to_super = dict(zip(tp["type_id"], tp["super_type_id"]))
    super_id_to_name = dict(zip(st["super_type_id"], st["display_name"]))

    # County names crosswalk
    xwalk = pd.read_csv(
        DATA_ROOT / "data/raw/fips_county_crosswalk.csv", dtype=str
    )
    xwalk["county_fips"] = xwalk["county_fips"].astype(str).str.zfill(5)
    name_map = dict(zip(xwalk["county_fips"], xwalk["county_name"]))

    # 2024 actual results
    r2024 = pd.read_parquet(
        DATA_ROOT / "data/assembled/medsl_county_presidential_2024.parquet"
    )
    r2024["county_fips"] = r2024["county_fips"].astype(str).str.zfill(5)

    # 2020 actual results
    r2020 = pd.read_parquet(
        DATA_ROOT / "data/assembled/medsl_county_presidential_2020.parquet"
    )
    r2020["county_fips"] = r2020["county_fips"].astype(str).str.zfill(5)

    # 2016 actual results (for leave-one-out prior estimation)
    r2016 = pd.read_parquet(
        DATA_ROOT / "data/assembled/medsl_county_presidential_2016.parquet"
    )
    r2016["county_fips"] = r2016["county_fips"].astype(str).str.zfill(5)

    return {
        "ta": ta,
        "type_scores": type_scores,
        "type_priors_2024": type_priors_2024,
        "type_covariance": type_covariance,
        "J": J,
        "type_to_super": type_to_super,
        "super_id_to_name": super_id_to_name,
        "name_map": name_map,
        "r2024": r2024,
        "r2020": r2020,
        "r2016": r2016,
        "score_cols": score_cols,
    }


# ---------------------------------------------------------------------------
# Per-super-type breakdown
# ---------------------------------------------------------------------------


def super_type_breakdown(
    merged: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    type_to_super: dict[int, int],
    super_id_to_name: dict[int, str],
) -> pd.DataFrame:
    """Compute per-super-type bias and MAE.

    Uses the dominant_type from type_assignments to assign each county to its
    primary super-type for breakdown purposes.

    Parameters
    ----------
    merged : DataFrame with columns [dominant_type, pred_col, actual_col]
    pred_col : column name for predicted Dem share
    actual_col : column name for actual Dem share
    type_to_super : dict mapping type_id -> super_type_id
    super_id_to_name : dict mapping super_type_id -> display_name

    Returns
    -------
    DataFrame with columns: super_type_id, super_type_name, n_counties, bias, mae
    """
    df = merged.copy()
    df["super_type_id"] = df["dominant_type"].map(type_to_super)
    df["error"] = df[pred_col] - df[actual_col]
    df["abs_error"] = df["error"].abs()

    rows = []
    for sid, grp in df.groupby("super_type_id"):
        rows.append({
            "super_type_id": sid,
            "super_type_name": super_id_to_name.get(sid, f"ST-{sid}"),
            "n_counties": len(grp),
            "bias": grp["error"].mean(),
            "mae": grp["abs_error"].mean(),
            "mean_actual": grp[actual_col].mean(),
            "mean_pred": grp[pred_col].mean(),
        })
    return pd.DataFrame(rows).sort_values("bias", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_calibration_2024(data: dict) -> pd.DataFrame:
    """Run prior-only calibration against 2024 actual presidential results.

    Returns the merged county-level DataFrame with predicted and actual values.
    """
    ta = data["ta"]
    type_scores = data["type_scores"]
    type_priors_2024 = data["type_priors_2024"]
    r2024 = data["r2024"]
    name_map = data["name_map"]

    # Compute predictions
    pred = prior_only_predictions(type_scores, type_priors_2024)

    # Build county-level frame
    county_df = ta[["county_fips", "state", "dominant_type", "super_type"]].copy()
    county_df["pred_dem_share"] = pred
    county_df["county_name"] = county_df["county_fips"].map(name_map)

    # Merge with actuals
    merged = county_df.merge(
        r2024[["county_fips", "pres_dem_share_2024", "pres_total_2024"]],
        on="county_fips",
        how="inner",
    )
    merged = merged.rename(columns={"pres_dem_share_2024": "actual_dem_share_2024"})
    merged["error_2024"] = merged["pred_dem_share"] - merged["actual_dem_share_2024"]
    merged["abs_error_2024"] = merged["error_2024"].abs()

    return merged


def run_calibration_2020_loo(data: dict) -> pd.DataFrame:
    """Leave-one-out backward calibration against 2020 actual presidential results.

    Uses 2016 actual data to estimate type priors (what the priors WOULD have been
    had the training covered only the 2016 period), then predicts 2020 county Dem
    shares using those estimated priors.

    The type structure (soft membership scores) is held constant -- we're testing
    whether the type structure generalizes backward, not re-discovering types.

    Prior estimation: population-weighted type-mean of 2016 actual Dem shares.

    Returns the merged county-level DataFrame.
    """
    ta = data["ta"]
    r2016 = data["r2016"]
    r2020 = data["r2020"]
    J = data["J"]
    name_map = data["name_map"]
    score_cols = data["score_cols"]

    # Merge ta with 2016 actuals to get aligned arrays
    r16 = r2016[["county_fips", "pres_dem_share_2016", "pres_total_2016"]].copy()
    ta_merged = ta[["county_fips"] + score_cols].merge(r16, on="county_fips", how="inner")

    fips_order = ta_merged["county_fips"].values
    scores_aligned = ta_merged[score_cols].values
    actual_2016 = ta_merged["pres_dem_share_2016"].values
    total_2016 = ta_merged["pres_total_2016"].values

    # Estimate type priors from 2016 actual results
    priors_2016 = build_type_priors_from_election(
        scores_aligned, actual_2016, total_2016, J
    )

    # Predict 2020 using 2016-estimated priors
    pred_2020 = prior_only_predictions(scores_aligned, priors_2016)

    # Build county frame
    county_df = pd.DataFrame({"county_fips": fips_order})
    county_df["pred_dem_share_loo"] = pred_2020
    county_df["county_name"] = county_df["county_fips"].map(name_map)
    county_df["state"] = county_df["county_fips"].str[:2].map(_STATE_FIPS)

    # Merge dominant_type and super_type from ta
    county_df = county_df.merge(
        ta[["county_fips", "dominant_type", "super_type"]], on="county_fips", how="left"
    )

    # Merge with 2020 actuals
    merged = county_df.merge(
        r2020[["county_fips", "pres_dem_share_2020", "pres_total_2020"]],
        on="county_fips",
        how="inner",
    )
    merged = merged.rename(columns={"pres_dem_share_2020": "actual_dem_share_2020"})
    merged["error_2020"] = merged["pred_dem_share_loo"] - merged["actual_dem_share_2020"]
    merged["abs_error_2020"] = merged["error_2020"].abs()

    return merged


def print_section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_metrics_table(state_metrics: dict[str, dict]) -> None:
    header = f"{'State':<8} {'n':>5} {'MAE':>7} {'RMSE':>7} {'Bias':>8} {'r':>7}"
    print(header)
    print("-" * len(header))
    for state in ["FL", "GA", "AL", "ALL"]:
        if state not in state_metrics:
            continue
        m = state_metrics[state]
        print(
            f"{state:<8} {m['n']:>5} "
            f"{m['mae']:>7.4f} {m['rmse']:>7.4f} {m['bias']:>+8.4f} {m['pearson_r']:>7.4f}"
        )


def main() -> None:
    print(f"Data root: {DATA_ROOT}")
    print("Loading data...")
    data = load_inputs()

    # ------------------------------------------------------------------ 2024
    print_section("2024 CALIBRATION (prior-only vs actual presidential)")
    merged_2024 = run_calibration_2024(data)

    state_metrics_2024: dict[str, dict] = {}
    for state in ["FL", "GA", "AL"]:
        grp = merged_2024[merged_2024["state"] == state]
        m = compute_metrics(
            grp["actual_dem_share_2024"].values,
            grp["pred_dem_share"].values,
        )
        state_metrics_2024[state] = m
    # All states combined
    state_metrics_2024["ALL"] = compute_metrics(
        merged_2024["actual_dem_share_2024"].values,
        merged_2024["pred_dem_share"].values,
    )
    print_metrics_table(state_metrics_2024)

    # Super-type breakdown (2024)
    print()
    print("Per-super-type bias (2024, all states):")
    st_breakdown_2024 = super_type_breakdown(
        merged_2024,
        pred_col="pred_dem_share",
        actual_col="actual_dem_share_2024",
        type_to_super=data["type_to_super"],
        super_id_to_name=data["super_id_to_name"],
    )
    for _, row in st_breakdown_2024.iterrows():
        sign = "+" if row["bias"] >= 0 else ""
        print(
            f"  {row['super_type_name']:<35} n={row['n_counties']:>3}  "
            f"bias={sign}{row['bias']:.4f}  mae={row['mae']:.4f}  "
            f"(pred {row['mean_pred']:.3f} vs actual {row['mean_actual']:.3f})"
        )

    # Worst-predicted counties (2024)
    print()
    print("Top 5 worst-predicted counties (2024, by abs error):")
    worst = merged_2024.nlargest(5, "abs_error_2024")[
        ["county_name", "state", "pred_dem_share", "actual_dem_share_2024",
         "abs_error_2024", "error_2024"]
    ]
    for _, row in worst.iterrows():
        sign = "+" if row["error_2024"] >= 0 else ""
        print(
            f"  {str(row['county_name']):<35} ({row['state']})  "
            f"pred={row['pred_dem_share']:.3f}  actual={row['actual_dem_share_2024']:.3f}  "
            f"err={sign}{row['error_2024']:.4f}"
        )

    # ------------------------------------------------------------------ 2020 LOO
    print_section("2020 LEAVE-ONE-OUT CALIBRATION (priors from 2016 actuals)")
    merged_2020 = run_calibration_2020_loo(data)

    state_metrics_2020: dict[str, dict] = {}
    for state in ["FL", "GA", "AL"]:
        grp = merged_2020[merged_2020["state"] == state]
        if len(grp) == 0:
            continue
        m = compute_metrics(
            grp["actual_dem_share_2020"].values,
            grp["pred_dem_share_loo"].values,
        )
        state_metrics_2020[state] = m
    state_metrics_2020["ALL"] = compute_metrics(
        merged_2020["actual_dem_share_2020"].values,
        merged_2020["pred_dem_share_loo"].values,
    )
    print_metrics_table(state_metrics_2020)

    # Super-type breakdown (2020 LOO)
    print()
    print("Per-super-type bias (2020 LOO, all states):")
    st_breakdown_2020 = super_type_breakdown(
        merged_2020,
        pred_col="pred_dem_share_loo",
        actual_col="actual_dem_share_2020",
        type_to_super=data["type_to_super"],
        super_id_to_name=data["super_id_to_name"],
    )
    for _, row in st_breakdown_2020.iterrows():
        sign = "+" if row["bias"] >= 0 else ""
        print(
            f"  {row['super_type_name']:<35} n={row['n_counties']:>3}  "
            f"bias={sign}{row['bias']:.4f}  mae={row['mae']:.4f}  "
            f"(pred {row['mean_pred']:.3f} vs actual {row['mean_actual']:.3f})"
        )

    # Worst-predicted counties (2020 LOO)
    print()
    print("Top 5 worst-predicted counties (2020 LOO, by abs error):")
    worst_20 = merged_2020.nlargest(5, "abs_error_2020")[
        ["county_name", "state", "pred_dem_share_loo", "actual_dem_share_2020",
         "abs_error_2020", "error_2020"]
    ]
    for _, row in worst_20.iterrows():
        sign = "+" if row["error_2020"] >= 0 else ""
        print(
            f"  {str(row['county_name']):<35} ({row['state']})  "
            f"pred={row['pred_dem_share_loo']:.3f}  actual={row['actual_dem_share_2020']:.3f}  "
            f"err={sign}{row['error_2020']:.4f}"
        )

    # ------------------------------------------------------------------ CSV output
    out_dir = DATA_ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build combined calibration CSV (2024)
    cal_csv = merged_2024[[
        "county_fips", "county_name", "state",
        "dominant_type", "super_type",
        "pred_dem_share", "actual_dem_share_2024",
        "error_2024", "abs_error_2024",
        "pres_total_2024",
    ]].copy()
    # Merge in super-type name
    super_name_map = data["super_id_to_name"]
    cal_csv["super_type_name"] = cal_csv["super_type"].map(super_name_map)

    # Also add 2020 LOO predictions where available
    loo_cols = merged_2020[["county_fips", "pred_dem_share_loo",
                             "actual_dem_share_2020", "error_2020"]].rename(
        columns={"error_2020": "error_loo_2020"}
    )
    cal_csv = cal_csv.merge(loo_cols, on="county_fips", how="left")

    csv_path = out_dir / "calibration_2024.csv"
    cal_csv.to_csv(csv_path, index=False)
    print_section(f"Saved calibration CSV to {csv_path}")
    print(f"  {len(cal_csv)} rows")

    # ------------------------------------------------------------------ Summary comparison
    print_section("COMPARISON: 2024 vs 2020-LOO (structural generalization test)")
    print(
        f"  2024 ALL  — MAE={state_metrics_2024['ALL']['mae']:.4f}  "
        f"r={state_metrics_2024['ALL']['pearson_r']:.4f}  "
        f"bias={state_metrics_2024['ALL']['bias']:+.4f}"
    )
    print(
        f"  2020 LOO  — MAE={state_metrics_2020['ALL']['mae']:.4f}  "
        f"r={state_metrics_2020['ALL']['pearson_r']:.4f}  "
        f"bias={state_metrics_2020['ALL']['bias']:+.4f}"
    )
    print()
    print(
        "Note: 2024 priors are derived from 2024 actuals, so 2024 calibration"
    )
    print(
        "      tests structural fit (not out-of-sample). 2020 LOO uses 2016"
    )
    print(
        "      actuals as proxy training priors — a genuine backward test."
    )


if __name__ == "__main__":
    main()
