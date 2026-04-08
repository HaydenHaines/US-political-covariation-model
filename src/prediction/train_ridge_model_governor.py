"""Train Ridge regression model for county-level governor Dem share priors.

Fits RidgeCV on type scores + county historical governor mean + demographic
features to predict 2022 governor Dem share.  Saves trained priors to disk
for use by the production prediction pipeline when forecasting governor races.

The presidential Ridge model (train_ridge_model.py) targets 2024 presidential
Dem share and uses 2008-2020 presidential history.  That model is structurally
wrong for governor races — presidential and governor electorates differ in
composition, turnout, and swing patterns.  The governor Ridge model targets
governor outcomes directly, yielding structurally appropriate priors for the
governor forecaster.

Inputs:
  data/communities/type_assignments.parquet    -- county type scores (N x J)
  data/assembled/county_features_national.parquet -- demographics (N x 20)
  data/assembled/algara_county_governor_{year}.parquet -- historical governor
  data/assembled/medsl_county_2022_governor.parquet    -- 2022 governor (target)

Outputs:
  data/models/ridge_model_governor/ridge_county_priors_governor.parquet
      county_fips + ridge_pred_dem_share (matched counties)
  data/models/ridge_model_governor/ridge_meta.json
      alpha, r2, feature_names, n_counties, date_trained

Feature matrix X = [type_scores | county_mean_gov_share | demo_std]
Target y = 2022 governor Dem share (absolute, not shift)

Governor data is sparse — only states with a governor race that year have
data (~2,150 counties per year vs ~3,100 for presidential).  Counties with
NO governor history fall back to presidential priors in the consumer
(load_county_priors_with_ridge_governor in county_priors.py).
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

# Re-use build_feature_matrix and the exclusion loader from the presidential
# Ridge module — identical feature construction logic, just with governor mean.
from src.prediction.train_ridge_model import build_feature_matrix  # noqa: E402

# Historical governor years used to compute the county mean.
# These are off-cycle years that had enough state coverage to be useful.
# 2006/2010/2014/2018 avoids the sparse early-cycle years (1994-2002) that
# have ~10x variance vs presidential cycles due to uncontested/missing races.
_GOV_HISTORY_YEARS = [2006, 2010, 2014, 2018]

# Year of the governor data used as the regression target.
_TARGET_YEAR = 2022


def compute_county_historical_gov_mean(
    county_fips: list[str],
    assembled_dir: Path,
    years: list[int] = None,
) -> np.ndarray:
    """Compute each county's mean Dem share across historical governor elections.

    Governor data is sparse: only states with a governor election that year
    appear in the Algara parquets (~2,150 counties per year vs ~3,100 for
    presidential).  Counties with no governor history at all receive a
    fallback value of 0.45 (the same nationwide prior used elsewhere).

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes (zero-padded to 5 digits).
    assembled_dir : Path
        Directory containing algara_county_governor_{year}.parquet files.
    years : list[int] or None
        Years to average over.  Defaults to _GOV_HISTORY_YEARS.

    Returns
    -------
    ndarray of shape (N,)
        Mean governor Dem share per county (fallback 0.45 if no data).
    """
    if years is None:
        years = _GOV_HISTORY_YEARS

    N = len(county_fips)
    fips_set = set(county_fips)
    gov_shares: dict[str, list[float]] = {f: [] for f in county_fips}

    for year in years:
        path = assembled_dir / f"algara_county_governor_{year}.parquet"
        if not path.exists():
            log.debug("Missing Algara governor file: %s", path)
            continue
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"gov_dem_share_{year}"
        if share_col not in df.columns:
            log.warning("Column %s missing in %s; skipping year", share_col, path)
            continue
        for _, row in df.iterrows():
            fips = row["county_fips"]
            if fips in fips_set and pd.notna(row[share_col]):
                gov_shares[fips].append(float(row[share_col]))

    # Counties with governor data: use their mean.
    # Counties without governor data: fall back to 0.45 national prior.
    means = np.full(N, 0.45)
    n_with_history = 0
    for i, fips in enumerate(county_fips):
        vals = gov_shares[fips]
        if vals:
            means[i] = float(np.mean(vals))
            n_with_history += 1

    log.info(
        "Governor historical mean: %d/%d counties have history (years: %s)",
        n_with_history,
        N,
        years,
    )
    return means


def load_governor_target(county_fips: list[str], assembled_dir: Path) -> np.ndarray:
    """Load 2022 governor Dem share as the regression target.

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes.
    assembled_dir : Path
        Directory containing medsl_county_2022_governor.parquet.

    Returns
    -------
    ndarray of shape (N,)
        2022 governor Dem share (NaN for counties not in the file — i.e.,
        states that did not have a 2022 governor race or had no data).
    """
    path = assembled_dir / "medsl_county_2022_governor.parquet"
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    share_map = dict(zip(df["county_fips"], df["gov_dem_share_2022"]))
    return np.array([share_map.get(f, float("nan")) for f in county_fips])


def _resolve_gov_ridge_paths(
    type_assignments_path: Path | None,
    demographics_path: Path | None,
    assembled_dir: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    """Fill in default paths for the governor Ridge training pipeline."""
    if type_assignments_path is None:
        type_assignments_path = (
            PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
        )
    if demographics_path is None:
        demographics_path = (
            PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"
        )
    if assembled_dir is None:
        assembled_dir = PROJECT_ROOT / "data" / "assembled"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "models" / "ridge_model_governor"
    return type_assignments_path, demographics_path, assembled_dir, output_dir


def _load_gov_ridge_training_data(
    type_assignments_path: Path,
    demographics_path: Path,
    assembled_dir: Path,
) -> tuple[list[str], np.ndarray, pd.DataFrame, int, np.ndarray, np.ndarray]:
    """Load all inputs for governor Ridge training.

    Returns
    -------
    county_fips : list[str]
    scores : ndarray (N, J)
    demo_df : DataFrame
    n_demo : int
    county_mean : ndarray (N,)   governor historical mean per county
    y_full : ndarray (N,)        2022 governor Dem share, NaN where missing
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

    log.info("Computing county historical governor mean (years: %s)", _GOV_HISTORY_YEARS)
    county_mean = compute_county_historical_gov_mean(county_fips, assembled_dir)

    log.info("Loading %d governor Dem share (target)", _TARGET_YEAR)
    y_full = load_governor_target(county_fips, assembled_dir)

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


def _save_gov_ridge_artifacts(
    output_dir: Path,
    matched_fips: np.ndarray,
    y_pred_matched: np.ndarray,
    meta: dict,
) -> tuple[Path, Path]:
    """Write governor county priors parquet and metadata JSON to output_dir.

    Returns
    -------
    out_parquet, out_json : paths to saved files
    """
    priors_df = pd.DataFrame({
        "county_fips": matched_fips,
        "ridge_pred_dem_share": y_pred_matched,
    })
    out_parquet = output_dir / "ridge_county_priors_governor.parquet"
    priors_df.to_parquet(out_parquet, index=False)
    log.info("Saved %d governor county priors to %s", len(priors_df), out_parquet)

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
    """Full training run: load data, fit Ridge on governor targets, save artifacts.

    Matches the same interface as train_ridge_model.train_and_save() to allow
    the training pipeline to call either interchangeably.

    Returns
    -------
    dict with keys: alpha, r2, n_counties, output_parquet, output_json
    """
    type_assignments_path, demographics_path, assembled_dir, output_dir = (
        _resolve_gov_ridge_paths(
            type_assignments_path, demographics_path, assembled_dir, output_dir
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    county_fips, scores, demo_df, n_demo, county_mean, y_full = (
        _load_gov_ridge_training_data(type_assignments_path, demographics_path, assembled_dir)
    )
    J = scores.shape[1]

    log.info("Building feature matrix (inner join with demographics)...")
    X, feature_names, row_mask = build_feature_matrix(
        scores, np.array(county_fips), demo_df, county_mean
    )
    y = y_full[row_mask]

    # The governor target is sparse: only counties in 2022 governor states have
    # a real y value.  Drop rows where y is NaN (no 2022 governor race).
    valid_mask = ~np.isnan(y)
    X_fit = X[valid_mask]
    y_fit = y[valid_mask]

    n_fit = len(y_fit)
    n_total = len(y)
    log.info(
        "Governor training data: %d counties with 2022 data out of %d in feature matrix",
        n_fit,
        n_total,
    )
    if n_fit < 10:
        raise ValueError(
            f"Too few governor training samples: {n_fit}. "
            "Check that medsl_county_2022_governor.parquet exists and has data."
        )

    rcv, alpha, r2 = _fit_ridge(X_fit, y_fit)

    # Predict for ALL matched counties even if they lacked a 2022 governor race.
    # The model generalizes from governor-data counties to the rest.
    # Counties with NO governor history in their features (county_mean=0.45)
    # will receive a prior shaped primarily by their type scores + demographics.
    y_pred_matched = np.clip(rcv.predict(X), 0.0, 1.0)
    matched_fips = np.array(county_fips)[row_mask]

    meta = {
        "alpha": alpha,
        "r2_train": r2,
        "feature_names": feature_names,
        "n_counties": int(len(matched_fips)),
        "n_training_samples": int(n_fit),
        "date_trained": str(date.today()),
        "target": f"gov_dem_share_{_TARGET_YEAR}",
        "history_years": _GOV_HISTORY_YEARS,
        "n_type_scores": J,
        "n_demo_features": n_demo,
    }
    out_parquet, out_json = _save_gov_ridge_artifacts(
        output_dir, matched_fips, y_pred_matched, meta
    )

    print(f"Governor Ridge model trained: alpha={alpha:.4g}, train R²={r2:.4f}")
    print(f"Governor county priors saved: {len(matched_fips)} counties → {out_parquet}")
    print(f"Prediction range: [{y_pred_matched.min():.3f}, {y_pred_matched.max():.3f}]")
    print(f"Training on {n_fit} counties (2022 governor data), predicting all {len(matched_fips)}")

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
