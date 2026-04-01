"""Train Ridge regression model for county-level Dem share priors.

Fits RidgeCV on type scores + county historical mean + demographic features
to predict 2024 presidential Dem share. Saves trained priors to disk for use
by the production prediction pipeline.

Inputs:
  data/communities/type_assignments.parquet    -- county type scores (N x J)
  data/assembled/county_features_national.parquet -- demographics (N x 20)
  data/assembled/medsl_county_presidential_*.parquet -- historical results

Outputs:
  data/models/ridge_model/ridge_county_priors.parquet
      county_fips + ridge_pred_dem_share (3,106 matched counties)
  data/models/ridge_model/ridge_meta.json
      alpha, r2, feature_names, n_counties, date_trained

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
from sklearn.linear_model import RidgeCV

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Historical years used to compute county mean (excludes 2024 target)
_HISTORY_YEARS = [2008, 2012, 2016, 2020]


def build_feature_matrix(
    scores: np.ndarray,
    county_fips: np.ndarray,
    demo_df: pd.DataFrame,
    county_mean: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build augmented feature matrix: scores | county_mean | demo_std.

    Performs inner join with demographics; returns matched rows only.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        County type scores.
    county_fips : ndarray of shape (N,)
        County FIPS codes (zero-padded 5-digit strings).
    demo_df : DataFrame
        Demographics with 'county_fips' column. All other columns are numeric
        demographic features.
    county_mean : ndarray of shape (N,)
        County mean Dem share across historical elections (excluding target).

    Returns
    -------
    X : ndarray of shape (n_matched, J + 1 + n_demo)
        Feature matrix.
    feature_names : list[str]
        Column names for X.
    row_mask : ndarray of int
        Indices into the original N rows that were matched.
    """
    demo_numeric_cols = [c for c in demo_df.columns if c != "county_fips"]

    idx_df = pd.DataFrame({
        "county_fips": county_fips,
        "_row_idx": np.arange(len(county_fips)),
    })
    merged = idx_df.merge(demo_df, on="county_fips", how="inner")

    if merged.empty:
        raise ValueError("Inner join between type_assignments and county_features_national produced no rows.")

    row_mask = merged["_row_idx"].values
    demo_raw = merged[demo_numeric_cols].values.astype(float)

    # Impute NaN with column means
    col_means = np.nanmean(demo_raw, axis=0)
    nan_mask = np.isnan(demo_raw)
    if nan_mask.any():
        col_idx = np.where(nan_mask)
        demo_raw[col_idx] = col_means[col_idx[1]]

    # Standardize demographics
    demo_mean = demo_raw.mean(axis=0)
    demo_std_val = demo_raw.std(axis=0)
    demo_std_val = np.where(demo_std_val < 1e-10, 1.0, demo_std_val)
    demo_std_feat = (demo_raw - demo_mean) / demo_std_val

    scores_sub = scores[row_mask]          # (n_matched, J)
    mean_sub = county_mean[row_mask]       # (n_matched,)

    X = np.column_stack([scores_sub, mean_sub, demo_std_feat])

    J = scores.shape[1]
    score_names = [f"type_{j}_score" for j in range(J)]
    feature_names = score_names + ["county_mean_dem_share"] + demo_numeric_cols

    return X, feature_names, row_mask


def compute_county_historical_mean(
    county_fips: list[str],
    assembled_dir: Path,
    years: list[int] = None,
) -> np.ndarray:
    """Compute each county's mean Dem share across historical elections.

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes (zero-padded to 5 digits).
    assembled_dir : Path
        Directory with medsl_county_presidential_{year}.parquet files.
    years : list[int] or None
        Years to average over. Defaults to _HISTORY_YEARS (excludes 2024).

    Returns
    -------
    ndarray of shape (N,)
        Mean Dem share per county (fallback 0.45 if no data).
    """
    if years is None:
        years = _HISTORY_YEARS

    N = len(county_fips)
    fips_set = set(county_fips)
    dem_shares: dict[str, list[float]] = {f: [] for f in county_fips}

    for year in years:
        path = assembled_dir / f"medsl_county_presidential_{year}.parquet"
        if not path.exists():
            log.debug("Missing historical file: %s", path)
            continue
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"pres_dem_share_{year}"
        if share_col not in df.columns:
            continue
        for _, row in df.iterrows():
            fips = row["county_fips"]
            if fips in fips_set and pd.notna(row[share_col]):
                dem_shares[fips].append(float(row[share_col]))

    means = np.full(N, 0.45)
    for i, fips in enumerate(county_fips):
        vals = dem_shares[fips]
        if vals:
            means[i] = float(np.mean(vals))
    return means


def load_target(county_fips: list[str], assembled_dir: Path) -> np.ndarray:
    """Load 2024 presidential Dem share as the regression target.

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes. Length N.
    assembled_dir : Path
        Directory with MEDSL parquet files.

    Returns
    -------
    ndarray of shape (N,)
        2024 Dem share (NaN for counties with no data).
    """
    path = assembled_dir / "medsl_county_presidential_2024.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    share_map = dict(zip(df["county_fips"], df["pres_dem_share_2024"]))
    return np.array([share_map.get(f, float("nan")) for f in county_fips])


def _resolve_ridge_paths(
    type_assignments_path: Path | None,
    demographics_path: Path | None,
    assembled_dir: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    """Fill in default paths for the Ridge training pipeline."""
    if type_assignments_path is None:
        type_assignments_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    if demographics_path is None:
        demographics_path = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"
    if assembled_dir is None:
        assembled_dir = PROJECT_ROOT / "data" / "assembled"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "models" / "ridge_model"
    return type_assignments_path, demographics_path, assembled_dir, output_dir


def _load_ridge_training_data(
    type_assignments_path: Path,
    demographics_path: Path,
    assembled_dir: Path,
) -> tuple[list[str], np.ndarray, pd.DataFrame, int, np.ndarray, np.ndarray]:
    """Load all inputs for Ridge training: type assignments, demographics, targets.

    Returns
    -------
    county_fips : list[str]
    scores : ndarray (N, J)
    demo_df : DataFrame
    n_demo : int
    county_mean : ndarray (N,)
    y_full : ndarray (N,)  — 2024 presidential Dem share, NaN where missing
    """
    log.info("Loading type assignments from %s", type_assignments_path)
    ta_df = pd.read_parquet(type_assignments_path)
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    scores = ta_df[score_cols].values.astype(float)
    log.info("Type assignments: %d counties, J=%d types", len(county_fips), scores.shape[1])

    log.info("Loading demographics from %s", demographics_path)
    demo_df = pd.read_parquet(demographics_path)
    demo_df["county_fips"] = demo_df["county_fips"].astype(str).str.zfill(5)
    n_demo = len([c for c in demo_df.columns if c != "county_fips"])
    log.info("Demographics: %d counties, %d features", len(demo_df), n_demo)

    log.info("Computing county historical mean Dem share (years: %s)", _HISTORY_YEARS)
    county_mean = compute_county_historical_mean(county_fips, assembled_dir)

    log.info("Loading 2024 presidential Dem share (target)")
    y_full = load_target(county_fips, assembled_dir)

    return county_fips, scores, demo_df, n_demo, county_mean, y_full


def _fit_ridge(X_fit: np.ndarray, y_fit: np.ndarray) -> tuple[RidgeCV, float, float]:
    """Fit RidgeCV over a log-spaced alpha grid and return model + metrics.

    Returns
    -------
    rcv : fitted RidgeCV
    alpha : best regularization parameter
    r2 : training R² at the selected alpha
    """
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X_fit, y_fit)
    alpha = float(rcv.alpha_)
    r2 = float(rcv.score(X_fit, y_fit))
    log.info("RidgeCV: alpha=%.4g, train R²=%.4f", alpha, r2)
    return rcv, alpha, r2


def _save_ridge_artifacts(
    output_dir: Path,
    matched_fips: np.ndarray,
    y_pred_matched: np.ndarray,
    meta: dict,
) -> tuple[Path, Path]:
    """Write county priors parquet and metadata JSON to output_dir.

    Returns
    -------
    out_parquet, out_json : paths to saved files
    """
    priors_df = pd.DataFrame({
        "county_fips": matched_fips,
        "ridge_pred_dem_share": y_pred_matched,
    })
    out_parquet = output_dir / "ridge_county_priors.parquet"
    priors_df.to_parquet(out_parquet, index=False)
    log.info("Saved %d county priors to %s", len(priors_df), out_parquet)

    out_json = output_dir / "ridge_meta.json"
    out_json.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata to %s", out_json)

    return out_parquet, out_json


def train_and_save(
    type_assignments_path: Path | None = None,
    demographics_path: Path | None = None,
    assembled_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Full training run: load data, fit Ridge, save artifacts.

    Returns
    -------
    dict with keys: alpha, r2, n_counties, output_parquet, output_json
    """
    type_assignments_path, demographics_path, assembled_dir, output_dir = (
        _resolve_ridge_paths(type_assignments_path, demographics_path, assembled_dir, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    county_fips, scores, demo_df, n_demo, county_mean, y_full = _load_ridge_training_data(
        type_assignments_path, demographics_path, assembled_dir
    )
    J = scores.shape[1]

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
    log.info("Training Ridge on %d counties, %d features", n_fit, X_fit.shape[1])
    if n_fit < 10:
        raise ValueError(f"Too few valid training samples: {n_fit}. Check data.")

    rcv, alpha, r2 = _fit_ridge(X_fit, y_fit)

    # Predict for ALL matched counties (including those without a 2024 target)
    # so that Ridge priors are available everywhere the model has history features.
    y_pred_matched = np.clip(rcv.predict(X), 0.0, 1.0)
    matched_fips = np.array(county_fips)[row_mask]

    meta = {
        "alpha": alpha,
        "r2_train": r2,
        "feature_names": feature_names,
        "n_counties": int(len(matched_fips)),
        "n_training_samples": int(n_fit),
        "date_trained": str(date.today()),
        "target": "pres_dem_share_2024",
        "history_years": _HISTORY_YEARS,
        "n_type_scores": J,
        "n_demo_features": n_demo,
    }
    out_parquet, out_json = _save_ridge_artifacts(output_dir, matched_fips, y_pred_matched, meta)

    print(f"Ridge model trained: alpha={alpha:.4g}, train R²={r2:.4f}")
    print(f"County priors saved: {len(matched_fips)} counties → {out_parquet}")
    print(f"Prediction range: [{y_pred_matched.min():.3f}, {y_pred_matched.max():.3f}]")

    return {
        "alpha": alpha,
        "r2": r2,
        "n_counties": len(matched_fips),
        "output_parquet": out_parquet,
        "output_json": out_json,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train_and_save()
