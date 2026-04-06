"""Ridge regression LOO holdout accuracy for tract-level electoral prediction.

Provides holdout_accuracy_ridge_tract(), the tract-level analogue of
holdout_accuracy_ridge_augmented() in holdout_accuracy_ridge.py.

Method: RidgeCV with GCV alpha selection; exact LOO via hat matrix diagonal.
The hat-matrix LOO formula avoids n refits and is numerically identical to
leave-one-out cross-validation for linear models.

Baseline to beat: county type-mean LOO r = 0.448 (from CLAUDE.md).
Current county Ridge+Demo LOO r = 0.731 (Ridge+40 pruned features, PCA).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def holdout_accuracy_ridge_tract(
    type_assignments_path: Path | None = None,
    tract_features_path: Path | None = None,
    elections_path: Path | None = None,
) -> dict:
    """Compute tract Ridge LOO r for the 2024 presidential prediction task.

    Loads the three required data files, builds the feature matrix identical to
    train_ridge_model_tract.run(), fits RidgeCV, and computes exact LOO predictions
    via the hat matrix.

    This is an end-to-end validation function: it mirrors the full training pipeline
    and returns the same LOO r reported in ridge_tract_meta.json, giving an honest
    estimate of out-of-sample accuracy.

    Parameters
    ----------
    type_assignments_path : Path or None
        tract_type_assignments.parquet. Defaults to data/communities/.
    tract_features_path : Path or None
        tract_features.parquet. Defaults to data/assembled/.
    elections_path : Path or None
        tract_elections.parquet. Defaults to data/assembled/.

    Returns
    -------
    dict with keys:
        "loo_r"         -- float, LOO Pearson r for 2024 pres dem_share
        "loo_rmse"      -- float, LOO RMSE
        "r2_train"      -- float, training R²
        "alpha"         -- float, GCV-selected alpha
        "n_tracts"      -- int, number of tracts used for training
        "pred_range"    -- tuple[float, float], (min, max) of in-sample predictions
    Returns None if required data files are missing.
    """
    # Import training helpers to avoid duplicating the build/train logic
    from src.prediction.train_ridge_model_tract import (
        _load_tract_elections,
        _resolve_tract_paths,
        _HISTORY_YEARS,
        build_tract_feature_matrix,
        compute_tract_historical_mean,
        compute_tract_loo_predictions,
        load_tract_target,
        _TRACT_METADATA_COLS,
        _ALPHA_GRID,
    )

    try:
        import pandas as pd
        from sklearn.linear_model import RidgeCV
    except ImportError as exc:
        raise ImportError(
            "holdout_accuracy_ridge_tract requires pandas and scikit-learn"
        ) from exc

    # Resolve paths
    type_assignments_path, tract_features_path, elections_path, _ = _resolve_tract_paths(
        type_assignments_path, tract_features_path, elections_path, None
    )

    # Validate all inputs exist before doing any work
    for p in (type_assignments_path, tract_features_path, elections_path):
        if not p.exists():
            log.warning(
                "holdout_accuracy_ridge_tract: required file not found at %s; returning None",
                p,
            )
            return None

    # ---- Load type assignments ----
    ta_df = pd.read_parquet(type_assignments_path)
    ta_df["tract_geoid"] = ta_df["tract_geoid"].astype(str)
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    scores = ta_df[score_cols].values.astype(float)
    tract_geoids = ta_df["tract_geoid"].tolist()

    # ---- Load demographics ----
    demo_df = pd.read_parquet(tract_features_path)
    demo_df["tract_geoid"] = demo_df["tract_geoid"].astype(str)

    # ---- Load elections and compute derived arrays ----
    pres_elections = _load_tract_elections(elections_path)
    tract_mean = compute_tract_historical_mean(tract_geoids, pres_elections, years=_HISTORY_YEARS)
    y_full = load_tract_target(tract_geoids, pres_elections)

    # ---- Build feature matrix ----
    X_all, feature_names, row_mask = build_tract_feature_matrix(
        scores, np.array(tract_geoids), demo_df, tract_mean
    )
    y_matched = y_full[row_mask]

    # ---- Filter to tracts with 2024 data ----
    valid_mask = ~np.isnan(y_matched)
    X_fit = X_all[valid_mask]
    y_fit = y_matched[valid_mask]
    n_tracts = int(len(y_fit))

    if n_tracts < 100:
        log.warning(
            "holdout_accuracy_ridge_tract: only %d tracts with 2024 data; "
            "results may be unreliable",
            n_tracts,
        )

    # ---- Fit Ridge (GCV alpha selection) ----
    rcv = RidgeCV(alphas=_ALPHA_GRID, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X_fit, y_fit)
    alpha = float(rcv.alpha_)
    r2_train = float(rcv.score(X_fit, y_fit))

    # ---- Exact LOO via hat matrix ----
    y_loo, _ = compute_tract_loo_predictions(X_fit, y_fit, alpha)

    if np.std(y_fit) < 1e-10 or np.std(y_loo) < 1e-10:
        loo_r = 0.0
        log.warning("holdout_accuracy_ridge_tract: near-zero variance; LOO r = 0")
    else:
        loo_r_val, _ = pearsonr(y_fit, y_loo)
        loo_r = float(np.clip(loo_r_val, -1.0, 1.0))

    loo_rmse = float(np.sqrt(np.mean((y_fit - y_loo) ** 2)))

    y_pred_all = np.clip(rcv.predict(X_all), 0.0, 1.0)

    log.info(
        "holdout_accuracy_ridge_tract: n=%d, alpha=%.4g, R²=%.4f, LOO r=%.4f, LOO RMSE=%.4f",
        n_tracts,
        alpha,
        r2_train,
        loo_r,
        loo_rmse,
    )

    return {
        "loo_r": loo_r,
        "loo_rmse": loo_rmse,
        "r2_train": r2_train,
        "alpha": alpha,
        "n_tracts": n_tracts,
        "pred_range": (float(y_pred_all.min()), float(y_pred_all.max())),
    }
