"""Backtest CES-derived δ against 2022 governor actuals.

Compares baselines:
  1. Model (no δ) — current county predictions with blended governor priors + incumbency
  2. Model + CES governor δ — adds CES-derived per-type governor δ post-hoc
  3. Model + model δ — adds original model-computed δ for comparison
  4. 2024 Presidential baseline

The CES δ should outperform the model δ because the CES validation (S500)
showed model δ has r=-0.008 correlation with reality while CES δ is based
on 248K validated voters.

Usage:
  uv run python scripts/backtest_ces_delta.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Same constants as the existing backtest
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
_INCUMBENCY_BONUS = 0.04


def load_actuals_2022() -> pd.DataFrame:
    """State-level 2022 governor two-party Dem share."""
    path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_2022_governor.parquet"
    df = pd.read_parquet(path)
    state_actuals = (
        df.groupby("state_abbr")
        .apply(
            lambda g: pd.Series({
                "dem_votes": g["gov_dem_2022"].sum(),
                "rep_votes": g["gov_rep_2022"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    state_actuals["actual_dem_share_2022"] = (
        state_actuals["dem_votes"] / (state_actuals["dem_votes"] + state_actuals["rep_votes"])
    )
    return state_actuals[["state_abbr", "actual_dem_share_2022"]]


def load_county_predictions() -> pd.DataFrame:
    """Load county-level governor predictions with type scores for δ application."""
    pred_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
    pred = pd.read_parquet(pred_path)

    gov = pred[
        pred["race"].str.contains("Governor", case=False, na=False)
        & (pred["forecast_mode"] == "local")
    ].copy()
    gov["race_state"] = gov["race"].str.extract(r"2026 (\w+) Governor")
    return gov[gov["state"] == gov["race_state"]].copy()


def load_type_scores_and_deltas() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load county type scores and both δ arrays."""
    ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    ta_df = pd.read_parquet(ta_path)
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_map = dict(zip(county_fips, type_scores))

    # Load both δ sources
    ces_delta = np.load(PROJECT_ROOT / "data" / "behavior" / "delta_ces_governor.npy")
    model_delta = np.load(PROJECT_ROOT / "data" / "behavior" / "delta.npy")

    return score_map, ces_delta, model_delta


def apply_delta_to_predictions(
    gov_df: pd.DataFrame,
    score_map: dict[str, np.ndarray],
    delta: np.ndarray,
) -> pd.DataFrame:
    """Apply per-county δ shift based on type scores."""
    df = gov_df.copy()

    # Compute per-county δ shift = type_scores @ delta
    def county_delta_shift(fips: str) -> float:
        scores = score_map.get(fips)
        if scores is None:
            return 0.0
        J = min(len(scores), len(delta))
        return float(scores[:J] @ delta[:J])

    df["delta_shift"] = df["county_fips"].apply(county_delta_shift)
    df["pred_with_delta"] = np.clip(df["pred_dem_share"] + df["delta_shift"], 0.0, 1.0)
    return df


def aggregate_to_state(
    df: pd.DataFrame,
    pred_col: str,
) -> pd.DataFrame:
    """Vote-weighted state aggregation."""
    pres_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    pres = pd.read_parquet(pres_path)[["county_fips", "pres_total_2024"]]
    merged = df.merge(pres, on="county_fips", how="left")
    merged["pres_total_2024"] = merged["pres_total_2024"].fillna(0)

    def vote_weighted_mean(g: pd.DataFrame) -> float:
        total = g["pres_total_2024"].sum()
        if total > 0:
            return float((g[pred_col] * g["pres_total_2024"]).sum() / total)
        return float(g[pred_col].mean())

    state_preds = (
        merged.groupby("race_state")
        .apply(vote_weighted_mean, include_groups=False)
        .reset_index()
    )
    state_preds.columns = ["state_abbr", pred_col]
    return state_preds


def apply_incumbency(state_preds: pd.DataFrame, pred_col: str) -> pd.Series:
    """Apply +4pp incumbency heuristic."""
    adjusted = state_preds[pred_col].copy()
    for i, row in state_preds.iterrows():
        st = row["state_abbr"]
        if st in GOVERNOR_2026_OPEN_SEATS:
            continue
        incumbent = _GOVERNOR_INCUMBENT.get(st, "R")
        bonus = _INCUMBENCY_BONUS if incumbent == "D" else -_INCUMBENCY_BONUS
        adjusted.iloc[i] = row[pred_col] + bonus
    return adjusted


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Pearson r, RMSE, bias, direction accuracy."""
    r, _ = pearsonr(actual, predicted)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    bias = float(np.mean(predicted - actual))
    correct_dir = int(np.sum((actual > 0.5) == (predicted > 0.5)))
    return {"r": float(r), "rmse": rmse, "bias": bias, "correct_dir": correct_dir}


def load_presidential_baseline() -> pd.DataFrame:
    """2024 presidential state-level Dem share."""
    path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    df = pd.read_parquet(path)
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


def main() -> None:
    print("Loading data...")
    actuals = load_actuals_2022()
    gov_preds = load_county_predictions()
    score_map, ces_delta, model_delta = load_type_scores_and_deltas()
    pres_baseline = load_presidential_baseline()

    # Baseline 1: No δ (current production)
    state_no_delta = aggregate_to_state(gov_preds, "pred_dem_share")

    # Baseline 2: With CES governor δ
    gov_ces = apply_delta_to_predictions(gov_preds, score_map, ces_delta)
    state_ces = aggregate_to_state(gov_ces, "pred_with_delta")
    state_ces = state_ces.rename(columns={"pred_with_delta": "ces_delta_pred"})

    # Baseline 3: With model δ
    gov_model = apply_delta_to_predictions(gov_preds, score_map, model_delta)
    state_model = aggregate_to_state(gov_model, "pred_with_delta")
    state_model = state_model.rename(columns={"pred_with_delta": "model_delta_pred"})

    # Merge everything
    df = actuals.merge(state_no_delta, on="state_abbr", how="inner")
    df = df.merge(state_ces, on="state_abbr", how="inner")
    df = df.merge(state_model, on="state_abbr", how="inner")
    df = df.merge(pres_baseline, on="state_abbr", how="inner")
    df = df[df["state_abbr"].isin(GOVERNOR_2026_STATES)].copy()

    # Apply incumbency to all model variants
    df["no_delta_inc"] = apply_incumbency(df, "pred_dem_share")
    df["ces_delta_inc"] = apply_incumbency(df, "ces_delta_pred")
    df["model_delta_inc"] = apply_incumbency(df, "model_delta_pred")

    print(f"States in analysis: {len(df)}")
    n = len(df)
    actual_arr = df["actual_dem_share_2022"].values

    # Compute metrics
    m_none = compute_metrics(actual_arr, df["no_delta_inc"].values)
    m_ces = compute_metrics(actual_arr, df["ces_delta_inc"].values)
    m_model = compute_metrics(actual_arr, df["model_delta_inc"].values)
    m_pres = compute_metrics(actual_arr, df["pres_dem_share_2024"].values)

    # Print results
    print("\n" + "=" * 75)
    print("CES δ BACKTEST: 2022 Governor Actuals vs Model Variants")
    print("=" * 75)
    print(f"{'Baseline':<35} {'r':>6} {'RMSE':>7} {'Bias':>8} {'Dir':>5}")
    print("-" * 75)
    for label, m in [
        ("No δ (current production)", m_none),
        ("+ CES governor δ (248K voters)", m_ces),
        ("+ Model δ (tract-computed)", m_model),
        ("2024 Presidential baseline", m_pres),
    ]:
        print(
            f"{label:<35} "
            f"{m['r']:>6.3f} "
            f"{m['rmse']*100:>6.1f}pp "
            f"{m['bias']*100:>+7.1f}pp "
            f"{m['correct_dir']:>2}/{n}"
        )
    print("=" * 75)

    # CES δ impact analysis
    ces_impact = df["ces_delta_inc"] - df["no_delta_inc"]
    print(f"\nCES δ impact: mean={ces_impact.mean()*100:+.2f}pp, "
          f"std={ces_impact.std()*100:.2f}pp")

    model_impact = df["model_delta_inc"] - df["no_delta_inc"]
    print(f"Model δ impact: mean={model_impact.mean()*100:+.2f}pp, "
          f"std={model_impact.std()*100:.2f}pp")

    # Per-state comparison where CES δ helps vs hurts
    df["err_none"] = np.abs(df["no_delta_inc"] - actual_arr)
    df["err_ces"] = np.abs(df["ces_delta_inc"] - actual_arr)
    df["err_model"] = np.abs(df["model_delta_inc"] - actual_arr)

    ces_better = (df["err_ces"] < df["err_none"]).sum()
    model_better = (df["err_model"] < df["err_none"]).sum()
    print(f"\nStates where CES δ improves accuracy: {ces_better}/{n}")
    print(f"States where model δ improves accuracy: {model_better}/{n}")

    # Top improvements from CES δ
    df["ces_improvement"] = df["err_none"] - df["err_ces"]
    top_improved = df.nlargest(5, "ces_improvement")
    print("\nTop 5 states improved by CES δ:")
    for _, row in top_improved.iterrows():
        print(f"  {row['state_abbr']}: {row['ces_improvement']*100:+.1f}pp "
              f"(actual={row['actual_dem_share_2022']*100:.1f}%, "
              f"no_δ={row['no_delta_inc']*100:.1f}%, "
              f"CES_δ={row['ces_delta_inc']*100:.1f}%)")

    top_hurt = df.nsmallest(5, "ces_improvement")
    print("\nTop 5 states hurt by CES δ:")
    for _, row in top_hurt.iterrows():
        print(f"  {row['state_abbr']}: {row['ces_improvement']*100:+.1f}pp "
              f"(actual={row['actual_dem_share_2022']*100:.1f}%, "
              f"no_δ={row['no_delta_inc']*100:.1f}%, "
              f"CES_δ={row['ces_delta_inc']*100:.1f}%)")


if __name__ == "__main__":
    main()
