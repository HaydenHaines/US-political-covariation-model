"""Train Ridge regression model for tract-level Dem share priors.

Fits RidgeCV on tract type scores + tract historical mean + demographic features
to predict 2024 presidential Dem share. Part of the T.5 tract-primary migration.
Analogous to train_ridge_model.py but operating at the census tract level.

Inputs:
  data/communities/tract_type_assignments.parquet
      tract_geoid + 100 type scores (80,507 tracts × 102 cols)
  data/assembled/tract_features.parquet
      tract_geoid + demographics (84,415 tracts × 15 cols)
      Metadata columns excluded from features: is_uninhabited, n_features_imputed
  data/assembled/tract_elections.parquet
      All tract-level election results: tract_geoid, year, race_type, dem_share
      2024 presidential: 69,640 tracts (target)
      Historical presidential (2008/2012/2016/2020): used for tract_mean_dem_share

Outputs:
  data/models/ridge_model/ridge_tract_priors.parquet
      tract_geoid + ridge_pred_dem_share (~66K matched tracts)
  data/models/ridge_model/ridge_tract_meta.json
      alpha, r2_train, loo_r, feature_names, n_tracts, n_training_samples, date_trained

Feature matrix X = [type_scores (N×100) | tract_mean_dem_share (N×1) | demographics (N×D)]
  tract_mean_dem_share = mean of 2008/2012/2016/2020 presidential dem_share per tract
  (2024 is excluded from the mean to prevent target leakage)
Target y = 2024 presidential Dem share

LOO via hat matrix: y_loo_i = y_i - (y_i - y_hat_i) / (1 - h_ii)
  h_ii = diag(X_aug @ inv(X_aug.T @ X_aug + α·I_pen) @ X_aug.T)
  where X_aug = [1 | X] with the intercept column unpenalized.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Historical presidential years for computing tract mean; excludes the 2024 target.
_HISTORY_YEARS = [2008, 2012, 2016, 2020]

# Race-type string used in tract_elections.parquet for presidential races.
_PRES_RACE_TYPE = "PRES"

# Columns in tract_features.parquet that are metadata, not predictive features.
_TRACT_METADATA_COLS = {"is_uninhabited", "n_features_imputed"}

# Fallback Dem share for tracts with no historical presidential data.
_FALLBACK_DEM_SHARE = 0.45

# Alpha grid for RidgeCV (matches county model for comparability).
_ALPHA_GRID = np.logspace(-3, 6, 100)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_tract_elections(elections_path: Path) -> pd.DataFrame:
    """Load tract_elections.parquet and return all presidential rows.

    Parameters
    ----------
    elections_path : Path
        Path to tract_elections.parquet.

    Returns
    -------
    DataFrame with columns: tract_geoid, year, dem_share (presidential only).
    """
    df = pd.read_parquet(elections_path, columns=["tract_geoid", "year", "race_type", "dem_share"])
    pres_df = df[df["race_type"] == _PRES_RACE_TYPE].copy()
    pres_df["tract_geoid"] = pres_df["tract_geoid"].astype(str)
    return pres_df


def compute_tract_historical_mean(
    tract_geoids: list[str],
    pres_elections: pd.DataFrame,
    years: list[int] | None = None,
) -> np.ndarray:
    """Compute each tract's mean Dem share across historical presidential elections.

    Only the years specified (default: 2008/2012/2016/2020) are averaged.
    2024 is explicitly excluded to prevent target leakage into the feature matrix.

    Parameters
    ----------
    tract_geoids : list[str]
        Tract GEOIDs to compute means for (N tracts).
    pres_elections : DataFrame
        Presidential election results with columns: tract_geoid, year, dem_share.
        Must already be filtered to presidential races only.
    years : list[int] or None
        Historical years to average. Defaults to _HISTORY_YEARS (excludes 2024).

    Returns
    -------
    ndarray of shape (N,)
        Mean historical Dem share per tract. Tracts with no data fall back to
        _FALLBACK_DEM_SHARE (0.45) — the approximate national presidential average.
    """
    if years is None:
        years = _HISTORY_YEARS

    # Guard: 2024 must never appear in the mean (would be target leakage)
    if 2024 in years:
        raise ValueError(
            "compute_tract_historical_mean: 2024 must not appear in 'years' "
            "(would leak the 2024 target into the feature matrix)"
        )

    # Filter to requested history years only
    hist_df = pres_elections[pres_elections["year"].isin(years)].copy()

    # Compute per-tract mean across all available historical years
    tract_means = (
        hist_df.groupby("tract_geoid")["dem_share"]
        .mean()
        .rename("hist_mean")
    )

    # Align to the requested tract list, filling missing tracts with fallback
    N = len(tract_geoids)
    result = np.full(N, _FALLBACK_DEM_SHARE)
    for i, geoid in enumerate(tract_geoids):
        if geoid in tract_means.index:
            val = tract_means[geoid]
            if pd.notna(val):
                result[i] = float(val)
    return result


def load_tract_target(
    tract_geoids: list[str],
    pres_elections: pd.DataFrame,
) -> np.ndarray:
    """Load 2024 presidential Dem share as the regression target.

    Parameters
    ----------
    tract_geoids : list[str]
        Tract GEOIDs aligned with the model's row order (N tracts).
    pres_elections : DataFrame
        Presidential election results (all years); must include 2024.

    Returns
    -------
    ndarray of shape (N,)
        2024 Dem share. NaN for tracts without 2024 data.
    """
    target_df = pres_elections[pres_elections["year"] == 2024][["tract_geoid", "dem_share"]]
    share_map = dict(zip(target_df["tract_geoid"], target_df["dem_share"]))
    return np.array([share_map.get(g, float("nan")) for g in tract_geoids])


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_tract_feature_matrix(
    scores: np.ndarray,
    tract_geoids: np.ndarray,
    demo_df: pd.DataFrame,
    tract_mean: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build the augmented feature matrix for tract-level Ridge regression.

    Feature layout: [type_scores (J cols) | tract_mean_dem_share (1 col) | demographics (D cols)]

    Performs an inner join between type_assignments and tract_features on tract_geoid,
    so only tracts present in BOTH tables are retained. Demographics are standardized
    (zero-mean, unit-variance) with NaN imputed to column means before standardization.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        Soft membership type scores (already row-normalized by KMeans pipeline).
    tract_geoids : ndarray of shape (N,)
        Tract GEOIDs aligned with rows of ``scores``.
    demo_df : DataFrame
        tract_features.parquet with 'tract_geoid' column. Metadata columns
        (is_uninhabited, n_features_imputed) must already be excluded from the
        numeric columns, or will be excluded here.
    tract_mean : ndarray of shape (N,)
        Per-tract mean Dem share across historical presidential elections
        (2024 excluded). Fallback 0.45 where no history.

    Returns
    -------
    X : ndarray of shape (n_matched, J + 1 + D)
        Feature matrix, with demographics standardized.
    feature_names : list[str]
        Column names for X, one per feature.
    row_mask : ndarray of int
        Indices into the original N rows that were retained after the inner join.
    """
    # Exclude metadata columns; everything else is a predictive demographic feature
    demo_numeric_cols = [
        c for c in demo_df.columns
        if c != "tract_geoid" and c not in _TRACT_METADATA_COLS
    ]

    if not demo_numeric_cols:
        raise ValueError(
            "build_tract_feature_matrix: no numeric demographic columns found in demo_df "
            f"after excluding metadata {_TRACT_METADATA_COLS!r}"
        )

    # Inner join: only keep tracts present in both type_assignments and tract_features
    idx_df = pd.DataFrame({
        "tract_geoid": tract_geoids,
        "_row_idx": np.arange(len(tract_geoids)),
    })
    merged = idx_df.merge(
        demo_df[["tract_geoid"] + demo_numeric_cols],
        on="tract_geoid",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "build_tract_feature_matrix: inner join on tract_geoid produced no rows. "
            "Check that tract_type_assignments and tract_features share GEOIDs."
        )

    row_mask = merged["_row_idx"].values
    demo_raw = merged[demo_numeric_cols].values.astype(float)

    # Impute NaN with column means before standardization
    col_means = np.nanmean(demo_raw, axis=0)
    nan_mask = np.isnan(demo_raw)
    if nan_mask.any():
        nan_row_idx, nan_col_idx = np.where(nan_mask)
        demo_raw[nan_row_idx, nan_col_idx] = col_means[nan_col_idx]

    # Standardize: zero-mean, unit-variance per demographic column
    # (Avoids magnitude differences across features from dominating Ridge penalty)
    scaler = StandardScaler()
    demo_std_feat = scaler.fit_transform(demo_raw)

    scores_sub = scores[row_mask]       # (n_matched, J)
    mean_sub = tract_mean[row_mask]     # (n_matched,)

    X = np.column_stack([scores_sub, mean_sub, demo_std_feat])

    J = scores.shape[1]
    score_names = [f"type_{j}_score" for j in range(J)]
    feature_names = score_names + ["tract_mean_dem_share"] + demo_numeric_cols

    log.info(
        "build_tract_feature_matrix: %d tracts matched (dropped %d unmatched), "
        "%d features (%d type scores + 1 mean + %d demo)",
        len(row_mask),
        len(tract_geoids) - len(row_mask),
        len(feature_names),
        J,
        len(demo_numeric_cols),
    )
    return X, feature_names, row_mask


# ---------------------------------------------------------------------------
# LOO validation via hat matrix
# ---------------------------------------------------------------------------


def compute_tract_loo_predictions(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact LOO predictions for tract Ridge via augmented hat matrix.

    Uses the identity: y_loo_i = y_i - (y_i - y_hat_i) / (1 - h_ii)
    where h_ii = [X_aug (X_aug.T X_aug + α·I_pen)^{-1} X_aug.T]_ii

    X_aug = [1 | X] with intercept column prepended. The intercept is unpenalized
    (pen[0,0] = 0), matching sklearn's fit_intercept=True behavior.

    This is an exact LOO — no refitting required. Valid because Ridge has a
    closed-form solution and the hat matrix can be computed directly.

    Parameters
    ----------
    X : ndarray of shape (N, F)
        Feature matrix (without intercept column).
    y : ndarray of shape (N,)
        Target vector (must be finite; filter NaN before calling).
    alpha : float
        Ridge regularization strength (best alpha from RidgeCV).

    Returns
    -------
    y_loo : ndarray of shape (N,)
        LOO predictions for each training sample.
    h_diag : ndarray of shape (N,)
        Hat matrix diagonal (leverage scores).
    """
    n, n_feat = X.shape
    # Augment with unpenalized intercept column
    X_aug = np.column_stack([np.ones(n), X])  # (N, F+1)

    # Penalty matrix: intercept (index 0) is NOT penalized
    pen = alpha * np.eye(n_feat + 1)
    pen[0, 0] = 0.0

    A = X_aug.T @ X_aug + pen
    A_inv = np.linalg.inv(A)

    # Hat matrix diagonal: h_ii = (X_aug A^{-1} X_aug.T)_ii
    # Computed efficiently without forming the full N×N hat matrix
    h_diag = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)

    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    residuals = y - y_hat

    # Denominator guard: 1 - h_ii can be near zero for high-leverage points
    denom = 1.0 - h_diag
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    y_loo = y - residuals / denom
    return y_loo, h_diag


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_tract_ridge(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[RidgeCV, float, float, float]:
    """Fit RidgeCV and compute LOO r via hat matrix.

    Parameters
    ----------
    X : ndarray of shape (N, F)
        Feature matrix (StandardScaler already applied to demographics).
    y : ndarray of shape (N,)
        Target (2024 presidential Dem share). Must not contain NaN.

    Returns
    -------
    rcv : fitted RidgeCV
    alpha : float — best regularization parameter from GCV
    r2_train : float — training R² at selected alpha
    loo_r : float — leave-one-out Pearson r via hat matrix
    """
    if np.isnan(y).any():
        raise ValueError("train_tract_ridge: y must not contain NaN. Filter before calling.")

    rcv = RidgeCV(alphas=_ALPHA_GRID, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    alpha = float(rcv.alpha_)
    r2_train = float(rcv.score(X, y))
    log.info("RidgeCV: alpha=%.4g, train R²=%.4f", alpha, r2_train)

    # LOO predictions via hat matrix (exact, no refit needed)
    y_loo, h_diag = compute_tract_loo_predictions(X, y, alpha)

    if np.std(y) < 1e-10 or np.std(y_loo) < 1e-10:
        loo_r = 0.0
        log.warning("train_tract_ridge: near-zero variance in y or y_loo; LOO r set to 0")
    else:
        loo_r_val, _ = pearsonr(y, y_loo)
        loo_r = float(np.clip(loo_r_val, -1.0, 1.0))

    log.info("LOO r (hat matrix): %.4f", loo_r)
    return rcv, alpha, r2_train, loo_r


def save_tract_priors(
    output_dir: Path,
    matched_geoids: np.ndarray,
    y_pred: np.ndarray,
    meta: dict,
) -> tuple[Path, Path]:
    """Write tract priors parquet and metadata JSON to output_dir.

    Parameters
    ----------
    output_dir : Path
        Directory to write outputs. Created if it doesn't exist.
    matched_geoids : ndarray of shape (n_matched,)
        Tract GEOIDs for which we have predictions (inner-join survivors).
    y_pred : ndarray of shape (n_matched,)
        Ridge predictions, clipped to [0, 1].
    meta : dict
        Metadata to serialize as JSON (alpha, r2, LOO r, feature names, etc.).

    Returns
    -------
    out_parquet, out_json : paths to saved files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    priors_df = pd.DataFrame({
        "tract_geoid": matched_geoids,
        "ridge_pred_dem_share": y_pred,
    })
    out_parquet = output_dir / "ridge_tract_priors.parquet"
    priors_df.to_parquet(out_parquet, index=False)
    log.info("Saved %d tract priors to %s", len(priors_df), out_parquet)

    out_json = output_dir / "ridge_tract_meta.json"
    out_json.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata to %s", out_json)

    return out_parquet, out_json


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_tract_paths(
    type_assignments_path: Path | None,
    tract_features_path: Path | None,
    elections_path: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    """Fill in default paths for the tract Ridge training pipeline."""
    data_dir = PROJECT_ROOT / "data"
    if type_assignments_path is None:
        type_assignments_path = data_dir / "communities" / "tract_type_assignments.parquet"
    if tract_features_path is None:
        tract_features_path = data_dir / "assembled" / "tract_features.parquet"
    if elections_path is None:
        elections_path = data_dir / "assembled" / "tract_elections.parquet"
    if output_dir is None:
        output_dir = data_dir / "models" / "ridge_model"
    return type_assignments_path, tract_features_path, elections_path, output_dir


def run(
    type_assignments_path: Path | None = None,
    tract_features_path: Path | None = None,
    elections_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Full training pipeline: load → feature matrix → Ridge → LOO → save.

    Parameters
    ----------
    type_assignments_path : Path or None
        tract_type_assignments.parquet. Defaults to data/communities/.
    tract_features_path : Path or None
        tract_features.parquet. Defaults to data/assembled/.
    elections_path : Path or None
        tract_elections.parquet. Defaults to data/assembled/.
    output_dir : Path or None
        Where to write outputs. Defaults to data/models/ridge_model/.

    Returns
    -------
    dict with keys: alpha, r2_train, loo_r, n_tracts, output_parquet, output_json
    """
    type_assignments_path, tract_features_path, elections_path, output_dir = (
        _resolve_tract_paths(type_assignments_path, tract_features_path, elections_path, output_dir)
    )

    # ---- Load type assignments ----
    log.info("Loading tract type assignments from %s", type_assignments_path)
    ta_df = pd.read_parquet(type_assignments_path)
    ta_df["tract_geoid"] = ta_df["tract_geoid"].astype(str)
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    if not score_cols:
        raise ValueError(f"No '*_score' columns found in {type_assignments_path}")
    scores = ta_df[score_cols].values.astype(float)
    tract_geoids = ta_df["tract_geoid"].tolist()
    J = scores.shape[1]
    log.info("Type assignments: %d tracts, J=%d types", len(tract_geoids), J)

    # ---- Load demographics ----
    log.info("Loading tract features from %s", tract_features_path)
    demo_df = pd.read_parquet(tract_features_path)
    demo_df["tract_geoid"] = demo_df["tract_geoid"].astype(str)
    n_demo_raw = len([c for c in demo_df.columns if c not in {"tract_geoid"} | _TRACT_METADATA_COLS])
    log.info("Tract features: %d tracts, %d demographic features", len(demo_df), n_demo_raw)

    # ---- Load elections ----
    log.info("Loading tract elections from %s", elections_path)
    pres_elections = _load_tract_elections(elections_path)
    log.info(
        "Presidential elections loaded: %d rows across years %s",
        len(pres_elections),
        sorted(pres_elections["year"].unique().tolist()),
    )

    # ---- Compute historical mean (2024 EXCLUDED — no target leakage) ----
    log.info("Computing tract historical mean Dem share (years: %s)", _HISTORY_YEARS)
    tract_mean = compute_tract_historical_mean(tract_geoids, pres_elections, years=_HISTORY_YEARS)

    # ---- 2024 presidential target ----
    log.info("Loading 2024 presidential Dem share (target)")
    y_full = load_tract_target(tract_geoids, pres_elections)
    n_with_target = int(np.sum(~np.isnan(y_full)))
    log.info("Tracts with 2024 presidential data: %d / %d", n_with_target, len(tract_geoids))

    # ---- Build feature matrix (inner join with demographics) ----
    log.info("Building feature matrix (inner join: type_assignments ∩ tract_features)...")
    X_all, feature_names, row_mask = build_tract_feature_matrix(
        scores, np.array(tract_geoids), demo_df, tract_mean
    )
    y_matched = y_full[row_mask]
    matched_geoids = np.array(tract_geoids)[row_mask]

    # ---- Filter to tracts with 2024 data for training ----
    valid_mask = ~np.isnan(y_matched)
    X_fit = X_all[valid_mask]
    y_fit = y_matched[valid_mask]
    n_fit = int(len(y_fit))
    log.info("Training Ridge on %d tracts, %d features", n_fit, X_fit.shape[1])

    if n_fit < 100:
        raise ValueError(
            f"Too few valid training samples: {n_fit}. "
            "Check that tract_elections has 2024 presidential data."
        )

    # ---- Fit Ridge + compute LOO r ----
    rcv, alpha, r2_train, loo_r = train_tract_ridge(X_fit, y_fit)

    # ---- Predict for ALL matched tracts (including those without 2024 target) ----
    # This gives Ridge priors for the full matched set, useful in forecasting.
    y_pred = np.clip(rcv.predict(X_all), 0.0, 1.0)

    # ---- Save artifacts ----
    meta = {
        "alpha": alpha,
        "r2_train": r2_train,
        "loo_r": loo_r,
        "feature_names": feature_names,
        "n_tracts": int(len(matched_geoids)),
        "n_training_samples": n_fit,
        "date_trained": str(date.today()),
        "target": "pres_dem_share_2024",
        "history_years": _HISTORY_YEARS,
        "n_type_scores": J,
        "n_demo_features": n_demo_raw,
        "baseline_loo_r_to_beat": 0.45,  # county type-mean LOO baseline from CLAUDE.md
    }
    out_parquet, out_json = save_tract_priors(output_dir, matched_geoids, y_pred, meta)

    # ---- Summary ----
    print(f"Tract Ridge model trained:")
    print(f"  n_tracts    = {len(matched_geoids):,}")
    print(f"  n_train     = {n_fit:,} (with 2024 pres data)")
    print(f"  alpha       = {alpha:.4g}")
    print(f"  train R²    = {r2_train:.4f}")
    print(f"  LOO r       = {loo_r:.4f}  (baseline: 0.45)")
    print(f"  pred range  = [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"  priors → {out_parquet}")

    return {
        "alpha": alpha,
        "r2_train": r2_train,
        "loo_r": loo_r,
        "n_tracts": len(matched_geoids),
        "n_training_samples": n_fit,
        "output_parquet": out_parquet,
        "output_json": out_json,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run()
