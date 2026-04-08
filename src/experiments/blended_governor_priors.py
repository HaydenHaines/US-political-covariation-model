"""Blended governor priors experiment.

Evaluates weighted combinations of governor-trained and presidential Ridge
county priors against 2022 governor election actuals using a proper temporal
holdout.

Key design decision: the production governor Ridge model targets 2022
(train_ridge_model_governor.py, target=gov_dem_share_2022).  Evaluating those
predictions against 2022 would be in-sample and misleadingly optimistic.
Instead this experiment mirrors the holdout_governor_2022.py methodology:

  GOVERNOR PRIOR SOURCE:  holdout Ridge trained on 2006–2014 → 2018 targets
                          applied to predict 2022 (true out-of-sample)
  PRESIDENTIAL PRIOR:     2020 presidential county Dem share
                          (the naive carry-forward used as a baseline in
                           holdout_governor_2022.py, r=0.800, bias=+4.6pp)

This makes blend metrics directly comparable to the per-metric baselines
already published in data/experiments/holdout_governor_2022.json.

Blend formula (gov_weight w in [0.0, 0.1, …, 1.0]):
  blended_prior = w * gov_holdout_pred + (1 − w) * pres_2020_share

County-level priors are aggregated to state level using 2020 presidential
total votes as population weights (identical to holdout_governor_2022.py).

Outputs:
    data/experiments/blended_governor_priors.json

Usage:
    uv run python -m src.experiments.blended_governor_priors
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV

# Re-use production feature construction to guarantee identical inputs.
from src.prediction.train_ridge_model import build_feature_matrix

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = PROJECT_ROOT / "data" / "assembled"
COMMUNITIES = PROJECT_ROOT / "data" / "communities"
OUTPUT_PATH = PROJECT_ROOT / "data" / "experiments" / "blended_governor_priors.json"

# Blend weights: governor fraction from 0 (pure presidential) to 1 (pure governor)
BLEND_WEIGHTS = [round(w * 0.1, 1) for w in range(11)]  # [0.0, 0.1, ..., 1.0]

# Holdout training parameters (mirror holdout_governor_2022.py exactly)
_HISTORY_YEARS = [2006, 2010, 2014]   # governor mean feature for training
_TRAINING_TARGET_YEAR = 2018           # Ridge fitted to predict 2018 results
_HOLDOUT_YEAR = 2022                   # evaluation year (never seen during training)

# Fallback Dem share for counties with no governor history (nationwide prior)
_GOV_FALLBACK = 0.45


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_type_assignments() -> tuple[list[str], np.ndarray]:
    """Load county FIPS and soft-membership type score matrix.

    Returns
    -------
    county_fips : list[str]  — zero-padded 5-digit FIPS codes (N counties)
    scores      : ndarray (N, J) — row-normalised type membership scores
    """
    path = COMMUNITIES / "type_assignments.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    score_cols = sorted(c for c in df.columns if c.endswith("_score"))
    return df["county_fips"].tolist(), df[score_cols].values.astype(float)


def _load_demographics() -> pd.DataFrame:
    """Load county demographics used to build the Ridge feature matrix."""
    path = ASSEMBLED / "county_features_national.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    return df


def _compute_gov_mean(
    county_fips: list[str],
    years: list[int],
) -> np.ndarray:
    """Compute per-county mean governor Dem share across the given history years.

    Governor data is sparse: only states with a race that year appear in the
    Algara parquets (~2,150 counties per year vs ~3,100 for presidential).
    Counties without any governor history fall back to _GOV_FALLBACK.

    Parameters
    ----------
    county_fips : list[str]  — zero-padded 5-digit FIPS codes, length N
    years       : list[int]  — Algara years to average

    Returns
    -------
    ndarray (N,)  — county governor Dem share mean; _GOV_FALLBACK where missing
    """
    accumulator: dict[str, list[float]] = {f: [] for f in county_fips}
    for year in years:
        path = ASSEMBLED / f"algara_county_governor_{year}.parquet"
        if not path.exists():
            log.warning("Missing Algara file for %d — skipping", year)
            continue
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"gov_dem_share_{year}"
        if share_col not in df.columns:
            continue
        share_map = dict(zip(df["county_fips"], df[share_col]))
        for fips in county_fips:
            val = share_map.get(fips)
            if val is not None and not np.isnan(float(val)):
                accumulator[fips].append(float(val))

    means = np.full(len(county_fips), _GOV_FALLBACK)
    for i, fips in enumerate(county_fips):
        vals = accumulator[fips]
        if vals:
            means[i] = float(np.mean(vals))
    return means


def _load_gov_target(county_fips: list[str], year: int) -> np.ndarray:
    """Load county-level governor Dem share for a given year.

    2018 loads from Algara; 2022 loads from MEDSL.
    Returns NaN for counties with no race that year.
    """
    if year == 2022:
        path = ASSEMBLED / "medsl_county_2022_governor.parquet"
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = "gov_dem_share_2022"
    else:
        path = ASSEMBLED / f"algara_county_governor_{year}.parquet"
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"gov_dem_share_{year}"

    share_map = dict(zip(df["county_fips"], df[share_col]))
    return np.array([share_map.get(f, float("nan")) for f in county_fips])


def _load_pres_2020(
    county_fips: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load 2020 presidential Dem share and total votes per county.

    Returns
    -------
    dem_share   : ndarray (N,) — NaN where county not in 2020 data
    total_votes : ndarray (N,) — NaN where county not in 2020 data
    """
    path = ASSEMBLED / "medsl_county_presidential_2020.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    # Drop the national aggregate row (FIPS "00000")
    df = df[df["county_fips"] != "00000"]
    share_map = dict(zip(df["county_fips"], df["pres_dem_share_2020"]))
    votes_map = dict(zip(df["county_fips"], df["pres_total_2020"]))
    shares = np.array([share_map.get(f, float("nan")) for f in county_fips])
    votes = np.array([votes_map.get(f, float("nan")) for f in county_fips])
    return shares, votes


def _load_state_abbr(county_fips: list[str]) -> list[str | None]:
    """Get state abbreviation for each county, using 2022 gov data + 2020 fallback."""
    df22 = pd.read_parquet(ASSEMBLED / "medsl_county_2022_governor.parquet")
    df22["county_fips"] = df22["county_fips"].astype(str).str.zfill(5)
    state_map: dict[str, str] = dict(zip(df22["county_fips"], df22["state_abbr"]))

    df20 = pd.read_parquet(ASSEMBLED / "medsl_county_presidential_2020.parquet")
    df20["county_fips"] = df20["county_fips"].astype(str).str.zfill(5)
    for _, row in df20.iterrows():
        fips = row["county_fips"]
        if fips not in state_map and row["state_abbr"]:
            state_map[fips] = row["state_abbr"]

    return [state_map.get(f) for f in county_fips]


# ---------------------------------------------------------------------------
# Holdout Ridge training
# ---------------------------------------------------------------------------


def _train_holdout_ridge(
    county_fips: list[str],
    scores: np.ndarray,
    demo_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Train Ridge on 2006–2014 history → 2018 targets, predict all matched counties.

    This is the temporal holdout governor Ridge: trained entirely on pre-2022
    data so that predictions against 2022 are genuinely out-of-sample.
    Feature construction mirrors train_ridge_model_governor.py exactly.

    Parameters
    ----------
    county_fips : all N county FIPS from type_assignments
    scores      : type score matrix (N, J)
    demo_df     : county demographics for inner-join feature construction

    Returns
    -------
    gov_pred   : ndarray (N_matched,) — clipped [0,1] holdout predictions
    matched_fips: ndarray (N_matched,) — FIPS codes after demo inner-join
    """
    # County mean governor share from pre-training years (used as a feature)
    county_mean = _compute_gov_mean(county_fips, _HISTORY_YEARS)

    # Build feature matrix; row_mask marks counties that survived the inner-join
    X_full, _, row_mask = build_feature_matrix(
        scores, np.array(county_fips), demo_df, county_mean
    )
    matched_fips = np.array(county_fips)[row_mask]

    # Load 2018 governor targets (what the Ridge is trained on)
    y_all = _load_gov_target(list(matched_fips), _TRAINING_TARGET_YEAR)

    # Train only on counties that had a 2018 governor race
    valid = ~np.isnan(y_all)
    X_fit, y_fit = X_full[valid], y_all[valid]
    log.info(
        "Holdout Ridge: %d training counties (2018 governor), %d features",
        len(y_fit), X_fit.shape[1],
    )

    # RidgeCV with broad alpha sweep — matches train_ridge_model_governor.py
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X_fit, y_fit)
    log.info("Holdout Ridge: R²_train=%.4f, alpha=%.4g", rcv.score(X_fit, y_fit), rcv.alpha_)

    # Predict all matched counties (including those without 2022 data)
    gov_pred = np.clip(rcv.predict(X_full), 0.0, 1.0)
    return gov_pred, matched_fips


# ---------------------------------------------------------------------------
# State-level aggregation
# ---------------------------------------------------------------------------


def _aggregate_to_states(
    county_fips: np.ndarray,
    state_abbr: list[str | None],
    pred_dem_share: np.ndarray,
    actual_dem_share: np.ndarray,
    vote_weights: np.ndarray,
) -> pd.DataFrame:
    """Vote-weight county predictions up to state-level aggregates.

    Only counties where prediction, actual, and weight are all finite are
    included.  Mirrors aggregate_to_states() in holdout_governor_2022.py.

    Returns
    -------
    DataFrame with columns: state, pred_dem_share, actual_dem_share, n_counties
    """
    rows = []
    for fips, state, pred, actual, w in zip(
        county_fips, state_abbr, pred_dem_share, actual_dem_share, vote_weights
    ):
        if state is None:
            continue
        if np.isnan(pred) or np.isnan(actual) or np.isnan(w) or w <= 0:
            continue
        rows.append({"state": state, "pred": pred, "actual": actual, "weight": w})

    if not rows:
        return pd.DataFrame(columns=["state", "pred_dem_share", "actual_dem_share", "n_counties"])

    df = pd.DataFrame(rows)
    return (
        df.groupby("state")[["pred", "actual", "weight"]]
        .apply(
            lambda g: pd.Series({
                "pred_dem_share": float(np.average(g["pred"], weights=g["weight"])),
                "actual_dem_share": float(np.average(g["actual"], weights=g["weight"])),
                "n_counties": len(g),
            }),
            include_groups=False,
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(state_df: pd.DataFrame) -> dict[str, float]:
    """Compute prediction quality metrics from a state-level DataFrame.

    Parameters
    ----------
    state_df : DataFrame with columns pred_dem_share, actual_dem_share

    Returns
    -------
    dict with r, rmse_pp, bias_pp, direction_accuracy, n_states
    """
    pred = state_df["pred_dem_share"].values
    actual = state_df["actual_dem_share"].values

    valid = ~(np.isnan(pred) | np.isnan(actual))
    pred, actual = pred[valid], actual[valid]

    if len(pred) < 2:
        return {
            "r": float("nan"), "rmse_pp": float("nan"),
            "bias_pp": float("nan"), "direction_accuracy": float("nan"),
            "n_states": int(valid.sum()),
        }

    r, _ = pearsonr(pred, actual)
    errors = pred - actual
    rmse_pp = float(np.sqrt(np.mean(errors**2)) * 100)
    bias_pp = float(np.mean(errors) * 100)
    # Direction accuracy: same side of 50% threshold
    direction_accuracy = float(np.mean((pred > 0.5) == (actual > 0.5)))

    return {
        "r": round(float(r), 4),
        "rmse_pp": round(rmse_pp, 2),
        "bias_pp": round(bias_pp, 2),
        "direction_accuracy": round(direction_accuracy, 3),
        "n_states": int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# Blend sweep
# ---------------------------------------------------------------------------


def run_blend_sweep(
    gov_pred: np.ndarray | None = None,
    pres_share: np.ndarray | None = None,
    actual_2022: np.ndarray | None = None,
    vote_weights: np.ndarray | None = None,
    state_abbr: list[str | None] | None = None,
    matched_fips: np.ndarray | None = None,
    blend_weights: list[float] | None = None,
) -> list[dict]:
    """Run the full blend weight sweep and return per-weight metrics.

    Optional parameters allow injection for testing.  Production usage loads
    all inputs from disk via the main() entry point.

    Parameters
    ----------
    gov_pred     : holdout Ridge governor predictions per matched county
    pres_share   : 2020 presidential Dem share per matched county
    actual_2022  : 2022 governor actual Dem share per matched county (NaN if missing)
    vote_weights : 2020 presidential total votes per matched county
    state_abbr   : state abbreviation per matched county (None if unknown)
    matched_fips : FIPS codes for the matched county array
    blend_weights: governor fractions to sweep (defaults to BLEND_WEIGHTS)

    Returns
    -------
    List of dicts, one per blend weight:
        gov_weight, pres_weight, r, rmse_pp, bias_pp, direction_accuracy, n_states
    """
    if blend_weights is None:
        blend_weights = BLEND_WEIGHTS

    results = []
    for w in blend_weights:
        pres_w = 1.0 - w

        # Blended county prior: if presidential share is NaN (county missing from
        # 2020 data), fall back to governor prediction alone.  Vice versa for
        # counties missing governor predictions (shouldn't happen but defensive).
        pres_ok = ~np.isnan(pres_share)
        gov_ok = ~np.isnan(gov_pred)
        blended = np.where(
            pres_ok & gov_ok,
            w * gov_pred + pres_w * pres_share,
            np.where(pres_ok, pres_share, gov_pred),
        )

        state_df = _aggregate_to_states(
            matched_fips, state_abbr, blended, actual_2022, vote_weights,
        )
        metrics = compute_metrics(state_df)
        results.append({
            "gov_weight": w,
            "pres_weight": round(pres_w, 1),
            **metrics,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def find_optimal_weight(results: list[dict]) -> dict:
    """Return the blend entry that maximises the correlation–bias tradeoff.

    Composite score = r − |bias_pp| / 10.0.
    A +0.01 r gain is worth a +0.1pp bias increase.  The divisor 10 reflects
    that bias is in percentage points while r is dimensionless.
    """
    return max(results, key=lambda row: row["r"] - abs(row["bias_pp"]) / 10.0)


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'gov_wt':>6}  {'pres_wt':>7}  {'r':>6}  {'rmse_pp':>8}  "
        f"{'bias_pp':>8}  {'dir_acc':>7}  {'composite':>9}"
    )
    print("\nBlended Governor Priors — Sweep Results vs 2022 Actuals (temporal holdout)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in results:
        composite = row["r"] - abs(row["bias_pp"]) / 10.0
        marker = " *" if row["gov_weight"] in (0.0, 1.0) else "  "
        print(
            f"{row['gov_weight']:>6.1f}  {row['pres_weight']:>7.1f}  "
            f"{row['r']:>6.4f}  {row['rmse_pp']:>8.2f}  "
            f"{row['bias_pp']:>+8.2f}  {row['direction_accuracy']:>7.3f}  "
            f"{composite:>9.4f}{marker}"
        )

    print("-" * len(header))
    print("  * = pure model baseline (no blending)")
    print("  composite = r - |bias_pp| / 10.0")

    optimal = find_optimal_weight(results)
    pure_pres = next(r for r in results if r["gov_weight"] == 0.0)
    pure_gov = next(r for r in results if r["gov_weight"] == 1.0)
    print(
        f"\nOptimal blend (highest composite score): "
        f"gov_weight={optimal['gov_weight']:.1f}  "
        f"r={optimal['r']:.4f}  bias={optimal['bias_pp']:+.2f}pp  "
        f"rmse={optimal['rmse_pp']:.2f}pp"
    )
    print(
        f"\nBaselines:"
        f"\n  Pure presidential (w=0.0): r={pure_pres['r']:.4f}  "
        f"bias={pure_pres['bias_pp']:+.2f}pp  rmse={pure_pres['rmse_pp']:.2f}pp"
        f"\n  Pure governor     (w=1.0): r={pure_gov['r']:.4f}  "
        f"bias={pure_gov['bias_pp']:+.2f}pp  rmse={pure_gov['rmse_pp']:.2f}pp"
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(results: list[dict]) -> None:
    """Save sweep results to data/experiments/blended_governor_priors.json."""
    optimal = find_optimal_weight(results)
    pure_pres = next(r for r in results if r["gov_weight"] == 0.0)
    pure_gov = next(r for r in results if r["gov_weight"] == 1.0)

    output = {
        "experiment": "blended_governor_priors",
        "description": (
            "Blend sweep: w * gov_holdout_pred + (1-w) * pres_2020_share vs "
            "2022 governor actuals.  Governor prior is a temporal holdout: "
            "Ridge trained on 2006-2014 history → 2018 target → predicting "
            "2022 (true out-of-sample).  State-level metrics use 2020 "
            "vote-weighted county aggregation."
        ),
        "holdout_year": _HOLDOUT_YEAR,
        "governor_training_history_years": _HISTORY_YEARS,
        "governor_training_target_year": _TRAINING_TARGET_YEAR,
        "presidential_baseline_year": 2020,
        "blend_weights_tested": BLEND_WEIGHTS,
        "optimal_gov_weight": optimal["gov_weight"],
        "optimal_metrics": {k: v for k, v in optimal.items() if k not in ("gov_weight", "pres_weight")},
        "baselines": {
            "pure_presidential": pure_pres,
            "pure_governor": pure_gov,
        },
        "sweep_results": results,
        "methodology": {
            "aggregation": "vote-weighted state mean (2020 presidential total votes)",
            "composite_score_formula": "r - abs(bias_pp) / 10.0",
            "note_holdout_integrity": (
                "Governor predictions are from a Ridge model trained on 2006-2014 "
                "history targeting 2018 outcomes.  2022 data is never seen during "
                "training, making this a genuine temporal holdout."
            ),
            "note_presidential_baseline": (
                "Presidential baseline is 2020 Dem share carried forward — the same "
                "naive baseline used in holdout_governor_2022.py (r=0.800, +4.6pp bias)."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the blended governor priors experiment end-to-end."""
    logging.basicConfig(level=logging.WARNING)

    print("Loading type assignments and demographics...")
    county_fips, scores = _load_type_assignments()
    demo_df = _load_demographics()
    print(f"  {len(county_fips):,} counties, J={scores.shape[1]} types")

    print("Training holdout Ridge (2006-2014 → 2018 → predict 2022)...")
    gov_pred, matched_fips = _train_holdout_ridge(county_fips, scores, demo_df)
    print(f"  Holdout Ridge predictions: {len(matched_fips):,} matched counties")

    print("Loading 2020 presidential shares (blend partner + vote weights)...")
    matched_fips_list = matched_fips.tolist()
    pres_share, pres_votes = _load_pres_2020(matched_fips_list)

    print("Loading 2022 governor actuals...")
    actual_2022 = _load_gov_target(matched_fips_list, _HOLDOUT_YEAR)
    n_with_actuals = int((~np.isnan(actual_2022)).sum())
    print(f"  {n_with_actuals:,} counties have 2022 governor data")

    print("Loading state abbreviations...")
    state_abbr = _load_state_abbr(matched_fips_list)

    print(f"\nRunning blend sweep over {len(BLEND_WEIGHTS)} weights...")
    results = run_blend_sweep(
        gov_pred=gov_pred,
        pres_share=pres_share,
        actual_2022=actual_2022,
        vote_weights=pres_votes,
        state_abbr=state_abbr,
        matched_fips=matched_fips,
    )

    print_comparison_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
