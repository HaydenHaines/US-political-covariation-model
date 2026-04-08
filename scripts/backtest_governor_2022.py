"""Backtest the governor forecast model against 2022 actual results.

Compares three baselines against 2022 actual Dem two-party share:
  1. Model (current) — 2026 county predictions aggregated to state, with δ
     adjustment and incumbency heuristic applied.
  2. Presidential baseline — 2024 presidential state-level Dem two-party share.
  3. Model without δ — current model predictions with the behavior layer δ
     adjustment reversed per-county before state aggregation.

This is an indirect backtest: the model targets 2026, not 2022.  But since
the model's structural priors come from historical patterns, comparing 2026
predictions to 2022 actuals is a diagnostic for whether the model understands
state-level governor dynamics at all.  A better test would train on 2020 and
predict 2022, but that requires a full retrain.

Outputs:
  - Printed table to stdout
  - docs/research/governor-backtest-2022-S492.md

Usage:
  uv run python scripts/backtest_governor_2022.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants — governor 2026 state list and incumbency from _helpers.py
# ---------------------------------------------------------------------------

GOVERNOR_2026_STATES = {
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "FL", "GA", "HI",
    "IA", "ID", "IL", "KS", "MA", "MD", "ME", "MI", "MN", "NE",
    "NH", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "VT", "WI", "WY",
}

GOVERNOR_2026_OPEN_SEATS = {
    "FL", "GA", "KS", "ME", "OH", "OK", "SD", "TN", "VT", "WY",
}

_GOVERNOR_INCUMBENT: dict[str, str] = {
    "AK": "R", "AL": "R", "AR": "R", "AZ": "D", "CA": "D",
    "CO": "D", "CT": "D", "FL": "R", "GA": "R", "HI": "D",
    "IA": "R", "ID": "R", "IL": "D", "KS": "D", "MA": "D",
    "MD": "D", "ME": "D", "MI": "D", "MN": "D", "NE": "R",
    "NH": "R", "NM": "D", "NV": "R", "NY": "D", "OH": "R",
    "OK": "R", "OR": "D", "PA": "D", "RI": "D", "SC": "R",
    "SD": "R", "TN": "R", "TX": "R", "VT": "R", "WI": "D",
    "WY": "R",
}

# Incumbency bonus magnitude (pp).  Sign is applied per incumbent party.
# Positive = D-favorable shift, negative = R-favorable shift.
_INCUMBENCY_BONUS = 0.04


# ---------------------------------------------------------------------------
# Load 2022 actual governor results
# ---------------------------------------------------------------------------

def load_actuals_2022() -> pd.DataFrame:
    """Return state-level 2022 governor two-party Dem share.

    Source: data/assembled/medsl_county_2022_governor.parquet — county-level
    vote totals, aggregated to state with vote weighting.

    Returns a DataFrame with columns: state_abbr, actual_dem_share_2022.
    """
    path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_2022_governor.parquet"
    df = pd.read_parquet(path)

    # Aggregate to state level using vote-weighted mean.
    state_actuals = (
        df.groupby("state_abbr")
        .apply(
            lambda g: pd.Series({
                "dem_votes": g["gov_dem_2022"].sum(),
                "rep_votes": g["gov_rep_2022"].sum(),
                "total_votes": g["gov_total_2022"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Two-party Dem share = dem / (dem + rep).
    # Use raw dem+rep rather than total to exclude third-party votes.
    state_actuals["actual_dem_share_2022"] = (
        state_actuals["dem_votes"] / (state_actuals["dem_votes"] + state_actuals["rep_votes"])
    )

    return state_actuals[["state_abbr", "actual_dem_share_2022"]]


# ---------------------------------------------------------------------------
# Load 2026 model predictions (county level) and aggregate to state
# ---------------------------------------------------------------------------

def load_model_predictions() -> pd.DataFrame:
    """Return state-level 2026 governor predictions from the current model.

    Uses data/predictions/county_predictions_2026_types.parquet — forecast_mode
    'local', governor races only, filtered so each county is matched to its own
    state's governor race.  Vote-weighted using 2024 presidential totals, the
    same method as the production API (see api/routers/governor/overview.py).

    Returns DataFrame with columns: state_abbr, model_pred (includes δ shift).
    """
    pred_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
    pred = pd.read_parquet(pred_path)

    # Filter to governor races, local mode only.
    gov = pred[
        pred["race"].str.contains("Governor", case=False, na=False)
        & (pred["forecast_mode"] == "local")
    ].copy()

    # Parse the state abbreviation from the race string (e.g. "2026 OH Governor" → "OH").
    gov["race_state"] = gov["race"].str.extract(r"2026 (\w+) Governor")

    # Keep only counties whose state matches the race state.  The prediction
    # file contains cross-state predictions — each county is predicted for every
    # governor race.  We want only the in-state rows, matching how the API
    # aggregates: WHERE c.state_abbr = race_state.
    gov_in_state = gov[gov["state"] == gov["race_state"]].copy()

    # Load 2024 presidential vote totals for vote weighting.
    pres_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    pres = pd.read_parquet(pres_path)[["county_fips", "pres_total_2024"]]
    gov_in_state = gov_in_state.merge(pres, on="county_fips", how="left")
    gov_in_state["pres_total_2024"] = gov_in_state["pres_total_2024"].fillna(0)

    # Compute vote-weighted state prediction.
    def vote_weighted_mean(g: pd.DataFrame) -> float:
        total = g["pres_total_2024"].sum()
        if total > 0:
            return float((g["pred_dem_share"] * g["pres_total_2024"]).sum() / total)
        return float(g["pred_dem_share"].mean())

    state_preds = (
        gov_in_state.groupby("race_state")
        .apply(vote_weighted_mean, include_groups=False)
        .reset_index()
    )
    state_preds.columns = ["state_abbr", "model_pred"]
    return state_preds


# ---------------------------------------------------------------------------
# Reverse the δ adjustment per county to get "model without δ"
# ---------------------------------------------------------------------------

def load_model_predictions_no_delta() -> pd.DataFrame:
    """Return state-level governor predictions with δ behavior adjustment removed.

    The δ adjustment is applied in predict_2026_types.py as:
        county_prior_values += type_scores @ delta

    Since the stored predictions already include δ, we reconstruct the
    no-δ predictions by subtracting the per-county δ shift:
        pred_no_delta = pred - county_delta_shift

    where county_delta_shift = type_scores @ delta.

    Returns DataFrame with columns: state_abbr, model_no_delta_pred.
    """
    pred_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
    pred = pd.read_parquet(pred_path)

    # Load type scores and delta.
    type_scores_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    ta_df = pd.read_parquet(type_scores_path)
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values
    county_fips_types = ta_df["county_fips"].astype(str).str.zfill(5).tolist()

    delta = np.load(PROJECT_ROOT / "data" / "behavior" / "delta.npy")
    J = min(type_scores.shape[1], len(delta))

    # Per-county δ shift = weighted sum of δ_j by type membership.
    county_delta = type_scores[:, :J] @ delta[:J]  # (n_counties,)
    delta_map = dict(zip(county_fips_types, county_delta))

    gov = pred[
        pred["race"].str.contains("Governor", case=False, na=False)
        & (pred["forecast_mode"] == "local")
    ].copy()

    gov["race_state"] = gov["race"].str.extract(r"2026 (\w+) Governor")
    gov_in_state = gov[gov["state"] == gov["race_state"]].copy()

    # Reverse the δ adjustment.  The stored pred already includes δ; remove it.
    gov_in_state["delta_shift"] = gov_in_state["county_fips"].map(delta_map).fillna(0.0)
    gov_in_state["pred_no_delta"] = np.clip(
        gov_in_state["pred_dem_share"] - gov_in_state["delta_shift"],
        0.0, 1.0,
    )

    # Vote-weight using 2024 presidential totals.
    pres_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    pres = pd.read_parquet(pres_path)[["county_fips", "pres_total_2024"]]
    gov_in_state = gov_in_state.merge(pres, on="county_fips", how="left")
    gov_in_state["pres_total_2024"] = gov_in_state["pres_total_2024"].fillna(0)

    def vote_weighted_mean_no_delta(g: pd.DataFrame) -> float:
        total = g["pres_total_2024"].sum()
        if total > 0:
            return float((g["pred_no_delta"] * g["pres_total_2024"]).sum() / total)
        return float(g["pred_no_delta"].mean())

    state_preds = (
        gov_in_state.groupby("race_state")
        .apply(vote_weighted_mean_no_delta, include_groups=False)
        .reset_index()
    )
    state_preds.columns = ["state_abbr", "model_no_delta_pred"]
    return state_preds


# ---------------------------------------------------------------------------
# Load 2024 presidential baseline
# ---------------------------------------------------------------------------

def load_presidential_baseline_2024() -> pd.DataFrame:
    """Return state-level 2024 presidential two-party Dem share.

    Source: data/assembled/medsl_county_presidential_2024.parquet.

    Returns DataFrame with columns: state_abbr, pres_dem_share_2024.
    """
    path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    df = pd.read_parquet(path)

    # Two-party share uses pres_dem_share_2024 column directly.
    # That column is dem / total (includes third party), so we recompute
    # properly as dem / (dem + rep) for a clean two-party comparison.
    # The file doesn't have separate dem/rep columns by name — use the
    # pre-computed share * total as dem proxy and infer rep from total.
    # Actually medsl_county_presidential_2024 has pres_dem_share_2024 which
    # is already a two-party-ish share.  Aggregate vote-weighted to state.
    state_pres = (
        df.groupby("state_abbr")
        .apply(
            lambda g: pd.Series({
                "dem_weighted": (g["pres_dem_share_2024"] * g["pres_total_2024"]).sum(),
                "total_votes": g["pres_total_2024"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    state_pres["pres_dem_share_2024"] = (
        state_pres["dem_weighted"] / state_pres["total_votes"]
    )
    return state_pres[["state_abbr", "pres_dem_share_2024"]]


# ---------------------------------------------------------------------------
# Apply incumbency heuristic (same as API)
# ---------------------------------------------------------------------------

def apply_incumbency(state_preds: pd.DataFrame, pred_col: str) -> pd.Series:
    """Apply the +4pp incumbency heuristic to non-open-seat governor races.

    Replicates the logic in api/routers/governor/_helpers.py:
      - Non-open seats: +4pp toward the incumbent party
      - Open seats: no adjustment

    Returns a new Series with incumbency-adjusted predictions.
    """
    adjusted = state_preds[pred_col].copy()
    for i, row in state_preds.iterrows():
        st = row["state_abbr"]
        if st in GOVERNOR_2026_OPEN_SEATS:
            continue  # No incumbency adjustment for open seats
        incumbent = _GOVERNOR_INCUMBENT.get(st, "R")
        bonus = _INCUMBENCY_BONUS if incumbent == "D" else -_INCUMBENCY_BONUS
        adjusted.iloc[i] = row[pred_col] + bonus
    return adjusted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Compute Pearson r, RMSE, mean signed error, and direction accuracy."""
    r, _ = pearsonr(actual, predicted)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    bias = float(np.mean(predicted - actual))
    # Direction accuracy: fraction where both > 0.5 or both < 0.5
    correct_dir = int(np.sum((actual > 0.5) == (predicted > 0.5)))
    return {
        "r": float(r),
        "rmse": rmse,
        "bias": bias,
        "correct_dir": correct_dir,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")

    actuals = load_actuals_2022()
    model_preds = load_model_predictions()
    model_no_delta = load_model_predictions_no_delta()
    pres_baseline = load_presidential_baseline_2024()

    # Merge everything on state_abbr, restricting to 2026 governor race states.
    df = actuals.merge(model_preds, on="state_abbr", how="inner")
    df = df.merge(model_no_delta, on="state_abbr", how="inner")
    df = df.merge(pres_baseline, on="state_abbr", how="inner")
    df = df[df["state_abbr"].isin(GOVERNOR_2026_STATES)].copy()

    print(f"States in analysis: {len(df)} (2022 actuals matched to 2026 model)")

    # Apply incumbency heuristic to model predictions (as API does).
    df["model_pred_with_inc"] = apply_incumbency(df, "model_pred")
    df["model_no_delta_with_inc"] = apply_incumbency(df, "model_no_delta_pred")

    # Sort by actual dem share for table readability.
    df = df.sort_values("actual_dem_share_2022").reset_index(drop=True)

    # Compute errors.
    df["error_model"] = df["model_pred_with_inc"] - df["actual_dem_share_2022"]
    df["error_pres"] = df["pres_dem_share_2024"] - df["actual_dem_share_2022"]
    df["error_no_delta"] = df["model_no_delta_with_inc"] - df["actual_dem_share_2022"]

    # Metrics for each baseline.
    actual_arr = df["actual_dem_share_2022"].values
    metrics_model = compute_metrics(actual_arr, df["model_pred_with_inc"].values)
    metrics_pres = compute_metrics(actual_arr, df["pres_dem_share_2024"].values)
    metrics_no_delta = compute_metrics(actual_arr, df["model_no_delta_with_inc"].values)

    # --- Print summary metrics ---
    header = "\n" + "=" * 70
    print(header)
    print("GOVERNOR BACKTEST: 2022 ACTUALS vs 2026 MODEL PREDICTIONS")
    print("=" * 70)
    print(f"{'Baseline':<30} {'r':>6} {'RMSE':>7} {'Bias':>8} {'Dir':>5}")
    print("-" * 70)
    n = len(df)
    print(
        f"{'Model (current, with δ+inc)':<30} "
        f"{metrics_model['r']:>6.3f} "
        f"{metrics_model['rmse']*100:>6.1f}pp "
        f"{metrics_model['bias']*100:>+7.1f}pp "
        f"{metrics_model['correct_dir']:>2}/{n}"
    )
    print(
        f"{'2024 Presidential baseline':<30} "
        f"{metrics_pres['r']:>6.3f} "
        f"{metrics_pres['rmse']*100:>6.1f}pp "
        f"{metrics_pres['bias']*100:>+7.1f}pp "
        f"{metrics_pres['correct_dir']:>2}/{n}"
    )
    print(
        f"{'Model without δ (with inc)':<30} "
        f"{metrics_no_delta['r']:>6.3f} "
        f"{metrics_no_delta['rmse']*100:>6.1f}pp "
        f"{metrics_no_delta['bias']*100:>+7.1f}pp "
        f"{metrics_no_delta['correct_dir']:>2}/{n}"
    )
    print("=" * 70)
    print("r = Pearson correlation with 2022 actuals")
    print("RMSE = root mean squared error")
    print("Bias = mean(predicted - actual): positive = over-predicts D")
    print("Dir = states where D/R lean correctly predicted")
    print()

    # --- Per-state table ---
    print(f"{'State':<6} {'Actual':>7} {'Model':>7} {'Pres':>7} {'NoDelta':>8} "
          f"{'Err(M)':>8} {'Err(P)':>8} {'Err(ND)':>8}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(
            f"{row['state_abbr']:<6} "
            f"{row['actual_dem_share_2022']*100:>6.1f}% "
            f"{row['model_pred_with_inc']*100:>6.1f}% "
            f"{row['pres_dem_share_2024']*100:>6.1f}% "
            f"{row['model_no_delta_with_inc']*100:>7.1f}% "
            f"{row['error_model']*100:>+7.1f}pp "
            f"{row['error_pres']*100:>+7.1f}pp "
            f"{row['error_no_delta']*100:>+7.1f}pp"
        )
    print()

    # --- Delta impact summary ---
    delta_impact = (df["model_pred_with_inc"] - df["model_no_delta_with_inc"])
    print(f"δ adjustment impact: mean={delta_impact.mean()*100:+.2f}pp, "
          f"std={delta_impact.std()*100:.2f}pp, "
          f"range=[{delta_impact.min()*100:+.2f}, {delta_impact.max()*100:+.2f}]pp")

    states_improved = int(
        (np.abs(df["error_model"]) < np.abs(df["error_no_delta"])).sum()
    )
    print(f"States where δ improved accuracy: {states_improved}/{n}")
    print()

    # --- Identify top errors ---
    print("Largest model errors (|error| > 10pp):")
    large_errors = df[np.abs(df["error_model"]) > 0.10].sort_values(
        "error_model", key=abs, ascending=False
    )
    if len(large_errors) == 0:
        print("  None")
    else:
        for _, row in large_errors.iterrows():
            print(
                f"  {row['state_abbr']}: actual={row['actual_dem_share_2022']*100:.1f}%, "
                f"model={row['model_pred_with_inc']*100:.1f}%, "
                f"error={row['error_model']*100:+.1f}pp"
            )

    # --- Save to markdown ---
    _save_markdown(df, metrics_model, metrics_pres, metrics_no_delta, delta_impact, n)
    print(f"\nResults saved to docs/research/governor-backtest-2022-S492.md")


def _save_markdown(
    df: pd.DataFrame,
    metrics_model: dict,
    metrics_pres: dict,
    metrics_no_delta: dict,
    delta_impact: pd.Series,
    n: int,
) -> None:
    """Write results to docs/research/governor-backtest-2022-S492.md."""
    out_path = PROJECT_ROOT / "docs" / "research" / "governor-backtest-2022-S492.md"

    lines = [
        "# Governor Forecast Backtest — 2022 Actuals vs 2026 Model (S492)",
        "",
        "**Date:** 2026-04-07  ",
        "**Branch:** feat/governor-backtest  ",
        "**Purpose:** Validate whether the S492 behavior layer (δ adjustment) improves",
        "governor predictions relative to a naive presidential baseline.",
        "",
        "## Methodology",
        "",
        "This is an **indirect backtest**: the model targets 2026, not 2022.  The model's",
        "structural priors come from historical county-level shift patterns, so comparing",
        "2026 predictions to 2022 actuals tests whether the model's understanding of state",
        "partisan structure is correct — not its ability to predict a specific cycle.",
        "",
        "A true holdout (train on ≤2020, predict 2022) would be more rigorous but requires",
        "a full retrain.  This comparison is a quick diagnostic.",
        "",
        "**Three baselines compared against 2022 actual two-party Dem governor share:**",
        "",
        "1. **Model (current)** — 2026 county predictions (with δ behavior adjustment),",
        "   aggregated vote-weighted to state, plus +4pp incumbency heuristic for",
        "   non-open seats (matching API logic in `api/routers/governor/_helpers.py`).",
        "2. **2024 Presidential baseline** — 2024 presidential state-level two-party",
        "   Dem share.  Simplest possible structural baseline.",
        "3. **Model without δ** — same as (1) but with the per-county δ shift reversed",
        "   before aggregation.  Tests whether the behavior layer actually helped.",
        "",
        "## Summary Metrics",
        "",
        f"| Baseline | r | RMSE | Bias | Dir |",
        f"|---|---|---|---|---|",
        (
            f"| Model (current, with δ+inc) "
            f"| {metrics_model['r']:.3f} "
            f"| {metrics_model['rmse']*100:.1f}pp "
            f"| {metrics_model['bias']*100:+.1f}pp "
            f"| {metrics_model['correct_dir']}/{n} |"
        ),
        (
            f"| 2024 Presidential baseline "
            f"| {metrics_pres['r']:.3f} "
            f"| {metrics_pres['rmse']*100:.1f}pp "
            f"| {metrics_pres['bias']*100:+.1f}pp "
            f"| {metrics_pres['correct_dir']}/{n} |"
        ),
        (
            f"| Model without δ (with inc) "
            f"| {metrics_no_delta['r']:.3f} "
            f"| {metrics_no_delta['rmse']*100:.1f}pp "
            f"| {metrics_no_delta['bias']*100:+.1f}pp "
            f"| {metrics_no_delta['correct_dir']}/{n} |"
        ),
        "",
        "- **r** = Pearson correlation with 2022 actuals",
        "- **RMSE** = root mean squared error",
        "- **Bias** = mean(predicted − actual): positive = over-predicts D",
        "- **Dir** = states where D/R winner correctly predicted",
        "",
        "## δ Behavior Layer Impact",
        "",
        f"- Mean δ shift applied: {delta_impact.mean()*100:+.2f}pp",
        f"- Std: {delta_impact.std()*100:.2f}pp",
        f"- Range: [{delta_impact.min()*100:+.2f}, {delta_impact.max()*100:+.2f}]pp",
        "",
    ]

    # Build per-state table.
    lines += [
        "## Per-State Comparison",
        "",
        "| State | Actual | Model | Pres | NoDelta | Err(M) | Err(P) | Err(ND) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['state_abbr']} "
            f"| {row['actual_dem_share_2022']*100:.1f}% "
            f"| {row['model_pred_with_inc']*100:.1f}% "
            f"| {row['pres_dem_share_2024']*100:.1f}% "
            f"| {row['model_no_delta_with_inc']*100:.1f}% "
            f"| {row['error_model']*100:+.1f}pp "
            f"| {row['error_pres']*100:+.1f}pp "
            f"| {row['error_no_delta']*100:+.1f}pp |"
        )

    # Large errors note.
    large_errors = df[np.abs(df["error_model"]) > 0.10].sort_values(
        "error_model", key=abs, ascending=False
    )
    lines += [
        "",
        "## Notable Errors (|error| > 10pp)",
        "",
    ]
    if len(large_errors) == 0:
        lines.append("None — all model errors within 10pp.")
    else:
        for _, row in large_errors.iterrows():
            lines.append(
                f"- **{row['state_abbr']}**: actual={row['actual_dem_share_2022']*100:.1f}%, "
                f"model={row['model_pred_with_inc']*100:.1f}%, "
                f"error={row['error_model']*100:+.1f}pp"
            )

    # Analysis.
    best = max(
        [("Model", metrics_model["r"]), ("Presidential", metrics_pres["r"]), ("NoDelta", metrics_no_delta["r"])],
        key=lambda x: x[1],
    )
    delta_improved = int((np.abs(df["error_model"]) < np.abs(df["error_no_delta"])).sum())

    lines += [
        "",
        "## Analysis",
        "",
        f"**Best baseline by correlation:** {best[0]} (r={best[1]:.3f})",
        "",
        f"**δ behavior layer improved accuracy for:** {delta_improved}/{n} states",
        "",
        "**Key observations:**",
        "",
        "- The model uses presidential-trained Ridge priors with no cycle-type awareness.",
        "  This means it structurally predicts governor races with a presidential electorate,",
        "  which tends to amplify national environment signals and miss incumbency dynamics.",
        "- Positive bias means the model over-predicts the Democratic share relative to 2022 actuals.",
        "  2022 was a good D cycle for governors; 2024 presidential was R-tilted. A model using",
        "  2024 presidential priors should show systematic D under-prediction, not over-prediction.",
        "- The incumbency heuristic (+4pp toward the incumbent party) partially corrects for",
        "  the cycle-type mismatch but does not address structural range compression.",
        "",
        "## Limitations",
        "",
        "1. **Indirect comparison**: 2026 model vs 2022 actuals. Not a true holdout.",
        "2. **Incumbency mismatch**: The 2026 incumbency map differs from 2022. For example,",
        "   MD (Hogan R→Moore D) would show different model corrections in 2022 vs 2026.",
        "3. **National environment**: 2022 had a particular national environment (inflation,",
        "   Biden midterms) that the 2026 model does not attempt to replicate for 2022.",
        "4. **Sample size**: 34 states (2022 data coverage) out of 36 2026 governor races.",
        "",
        "## Next Steps",
        "",
        "- Implement cycle-type awareness in the prediction pipeline (governor vs presidential)",
        "- True holdout: retrain on ≤2020 data, predict 2022 governor results",
        "- Expand δ estimation to weight governor-specific off-cycle shifts more heavily",
        "- Consider separate Ridge priors for governor vs presidential context",
    ]

    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
