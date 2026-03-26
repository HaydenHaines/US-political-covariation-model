"""Train Ridge+HGB ensemble model for county-level Dem share priors.

Fits RidgeCV and HistGradientBoostingRegressor on type scores + county
historical mean + demographic features to predict 2024 presidential Dem
share. Blends predictions 50/50 and saves to the same output location as
train_ridge_model.py so the production API picks it up without changes.

This module improves LOO r from 0.650 (Ridge-only) to 0.690 (ensemble).

Inputs:
  data/communities/type_assignments.parquet    -- county type scores (N x J)
  data/assembled/county_features_national.parquet -- demographics (N x ~27)
  data/assembled/medsl_county_presidential_*.parquet -- historical results

Outputs:
  data/models/ridge_model/ridge_county_priors.parquet
      county_fips + ridge_pred_dem_share (3,106 matched counties)
      NOTE: column name kept as ridge_pred_dem_share for backward compatibility;
            values are ensemble (50% Ridge + 50% HGB) predictions.
  data/models/ridge_model/ridge_meta.json
      alpha, r2_ridge, r2_hgb, hgb_params, ensemble_method, n_counties, date_trained, ...

Feature matrix X = [type_scores (N x J) | county_mean_dem_share (N x 1) | demo_std (N x 20)]
Target y = 2024 presidential Dem share (absolute, not shift)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from src.prediction.train_ridge_model import (
    _HISTORY_YEARS,
    build_feature_matrix,
    compute_county_historical_mean,
    load_target,
)

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# HGB hyperparameters (updated 2026-03-26: max_iter 300→800, lr 0.05→0.1, max_depth 4→5, min_samples_leaf 20→30)
HGB_PARAMS = {
    "max_iter": 800,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 42,
}

ENSEMBLE_RIDGE_WEIGHT = 0.5
ENSEMBLE_HGB_WEIGHT = 0.5


def train_and_save(
    type_assignments_path: Path | None = None,
    demographics_path: Path | None = None,
    assembled_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Full ensemble training run: load data, fit Ridge+HGB, blend, save artifacts.

    Returns
    -------
    dict with keys: alpha, r2_ridge, r2_hgb, n_counties, output_parquet, output_json
    """
    if type_assignments_path is None:
        type_assignments_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    if demographics_path is None:
        demographics_path = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"
    if assembled_dir is None:
        assembled_dir = PROJECT_ROOT / "data" / "assembled"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "models" / "ridge_model"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load type assignments ────────────────────────────────────────────────
    log.info("Loading type assignments from %s", type_assignments_path)
    ta_df = pd.read_parquet(type_assignments_path)
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    scores = ta_df[score_cols].values.astype(float)
    J = scores.shape[1]
    N = len(county_fips)
    log.info("Type assignments: %d counties, J=%d types", N, J)

    # ── Load demographics ────────────────────────────────────────────────────
    log.info("Loading demographics from %s", demographics_path)
    demo_df = pd.read_parquet(demographics_path)
    demo_df["county_fips"] = demo_df["county_fips"].astype(str).str.zfill(5)
    n_demo = len([c for c in demo_df.columns if c != "county_fips"])
    log.info("Demographics: %d counties, %d features", len(demo_df), n_demo)

    # ── Compute county historical mean (2008-2020, excludes 2024 target) ────
    log.info("Computing county historical mean Dem share (years: %s)", _HISTORY_YEARS)
    county_mean = compute_county_historical_mean(county_fips, assembled_dir)

    # ── Load 2024 target ─────────────────────────────────────────────────────
    log.info("Loading 2024 presidential Dem share (target)")
    y_full = load_target(county_fips, assembled_dir)

    # ── Build feature matrix (inner join with demographics) ──────────────────
    log.info("Building feature matrix (inner join with demographics)...")
    X, feature_names, row_mask = build_feature_matrix(
        scores, np.array(county_fips), demo_df, county_mean
    )
    y = y_full[row_mask]

    # Drop rows where target is NaN
    valid_mask = ~np.isnan(y)
    X_fit = X[valid_mask]
    y_fit = y[valid_mask]

    n_fit = len(y_fit)
    log.info("Training ensemble on %d counties, %d features", n_fit, X_fit.shape[1])

    if n_fit < 10:
        raise ValueError(f"Too few valid training samples: {n_fit}. Check data.")

    # ── Fit RidgeCV ──────────────────────────────────────────────────────────
    log.info("Fitting RidgeCV...")
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X_fit, y_fit)

    alpha = float(rcv.alpha_)
    r2_ridge = float(rcv.score(X_fit, y_fit))
    log.info("RidgeCV: alpha=%.4g, train R²=%.4f", alpha, r2_ridge)

    # ── Fit HistGradientBoostingRegressor ────────────────────────────────────
    log.info("Fitting HGB with params: %s", HGB_PARAMS)
    hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
    hgb.fit(X_fit, y_fit)

    r2_hgb = float(hgb.score(X_fit, y_fit))
    log.info("HGB train R²=%.4f", r2_hgb)

    # ── Generate predictions for ALL matched counties ─────────────────────────
    # Use ALL row_mask (not just valid_mask) so counties without 2024 target
    # still get priors (model uses all history features).
    ridge_pred_matched = rcv.predict(X)    # shape: (n_matched,)
    hgb_pred_matched = hgb.predict(X)     # shape: (n_matched,)

    # 50/50 blend
    ensemble_pred = (
        ENSEMBLE_RIDGE_WEIGHT * ridge_pred_matched
        + ENSEMBLE_HGB_WEIGHT * hgb_pred_matched
    )
    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)

    matched_fips = np.array(county_fips)[row_mask]

    # ── Save outputs ─────────────────────────────────────────────────────────
    # Column name stays ridge_pred_dem_share for backward compat with API
    priors_df = pd.DataFrame({
        "county_fips": matched_fips,
        "ridge_pred_dem_share": ensemble_pred,
    })
    out_parquet = output_dir / "ridge_county_priors.parquet"
    priors_df.to_parquet(out_parquet, index=False)
    log.info("Saved %d county ensemble priors to %s", len(priors_df), out_parquet)

    meta = {
        "alpha": alpha,
        "r2_ridge": r2_ridge,
        "r2_train": r2_ridge,  # kept for backward compat with existing tests
        "r2_hgb": r2_hgb,
        "ensemble_method": "50pct_ridge_50pct_hgb",
        "ensemble_ridge_weight": ENSEMBLE_RIDGE_WEIGHT,
        "ensemble_hgb_weight": ENSEMBLE_HGB_WEIGHT,
        "hgb_params": HGB_PARAMS,
        "loo_r_ridge": 0.650,     # measured in S197
        "loo_r_hgb": 0.687,       # measured in exp1b_hgb_full_loo.py
        "loo_r_ensemble": 0.690,  # measured in exp1b_hgb_full_loo.py
        "feature_names": feature_names,
        "n_counties": int(len(priors_df)),
        "n_training_samples": int(n_fit),
        "date_trained": str(date.today()),
        "target": "pres_dem_share_2024",
        "history_years": _HISTORY_YEARS,
        "n_type_scores": J,
        "n_demo_features": n_demo,
    }
    out_json = output_dir / "ridge_meta.json"
    out_json.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata to %s", out_json)

    print(f"Ridge: alpha={alpha:.4g}, train R²={r2_ridge:.4f}")
    print(f"HGB:   train R²={r2_hgb:.4f}")
    print(f"Ensemble (50/50): {len(priors_df)} counties saved → {out_parquet}")
    print(f"Prediction range: [{ensemble_pred.min():.3f}, {ensemble_pred.max():.3f}]")

    return {
        "alpha": alpha,
        "r2_ridge": r2_ridge,
        "r2_hgb": r2_hgb,
        "r2": r2_ridge,  # backward compat
        "n_counties": len(priors_df),
        "output_parquet": out_parquet,
        "output_json": out_json,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train_and_save()
