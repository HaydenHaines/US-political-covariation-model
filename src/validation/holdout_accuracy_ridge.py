"""Ridge regression holdout accuracy for electoral type validation.

Contains the Ridge-based prediction functions that extend the basic county-prior
holdout accuracy with regularized regression:
- holdout_accuracy_ridge: Ridge on type scores, LOO via hat matrix
- holdout_accuracy_ridge_augmented: Ridge + demographic features

Both use GCV alpha selection and exact LOO predictions via the hat matrix
diagonal, as validated in experiment_ridge_prediction.py (S197).

Feature pruning (S306): _load_and_standardize_demographics() applies a config-
driven exclusion list from config/ridge_feature_exclusions.yaml. 74 of 114
demographic features hurt LOO r individually; pruning to 40 features improves
LOO r from 0.717 to 0.731. The full parquet is still used for type profiling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the feature exclusion config (relative to PROJECT_ROOT)
_EXCLUSIONS_CONFIG_PATH = PROJECT_ROOT / "config" / "ridge_feature_exclusions.yaml"


def _load_feature_exclusions() -> frozenset[str]:
    """Load the set of demographic feature names to exclude from Ridge.

    Reads config/ridge_feature_exclusions.yaml. Returns an empty frozenset
    if the file is missing (falls back to using all features, matching the
    pre-S306 behaviour).

    Returns
    -------
    frozenset[str]
        Feature names to exclude from the Ridge demographic feature matrix.
    """
    if not _EXCLUSIONS_CONFIG_PATH.exists():
        log.warning(
            "_load_feature_exclusions: exclusion config not found at %s; using all features",
            _EXCLUSIONS_CONFIG_PATH,
        )
        return frozenset()

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        try:
            # PyYAML ships as 'yaml'; some environments expose it as 'pyyaml'
            import pyyaml as yaml  # type: ignore[import, no-redef]
        except ImportError:
            log.warning(
                "_load_feature_exclusions: PyYAML not installed; using all features"
            )
            return frozenset()

    with _EXCLUSIONS_CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)

    excluded = cfg.get("excluded_features", []) or []
    return frozenset(excluded)


def _ridge_loo_for_dimension(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact LOO predictions for one Ridge dimension via the hat matrix.

    Uses the augmented hat matrix formula y_loo_i = y_i - e_i / (1 - H_ii),
    where H is the hat matrix for [1 | X] with the intercept unpenalized.
    This matches sklearn's fit_intercept=True behavior.

    Parameters
    ----------
    X : ndarray of shape (N, F)
        Feature matrix (without intercept column).
    y : ndarray of shape (N,)
        Target vector.
    alpha : float
        Ridge regularization strength.

    Returns
    -------
    y_loo : ndarray of shape (N,)
        LOO predictions.
    h : ndarray of shape (N,)
        Hat matrix diagonal (leverage scores).
    """
    n, n_feat = X.shape
    X_aug = np.column_stack([np.ones(n), X])  # (N, n_feat+1)
    pen = alpha * np.eye(n_feat + 1)
    pen[0, 0] = 0.0  # intercept is unpenalized
    A = X_aug.T @ X_aug + pen
    A_inv = np.linalg.inv(A)
    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)  # hat diag
    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    e = y - y_hat
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    y_loo = y - e / denom
    return y_loo, h


def _compute_loo_metrics(
    X: np.ndarray,
    shift_matrix: np.ndarray,
    holdout_cols: list[int],
    alphas: np.ndarray,
    RidgeCV: type,
) -> tuple[list[float], list[float], list[float]]:
    """Run LOO Ridge regression over all holdout dimensions.

    Selects alpha via GCV for each dimension, computes exact LOO predictions
    via the hat matrix, and accumulates per-dimension r and RMSE.

    Returns
    -------
    per_dim_r, per_dim_rmse, best_alphas : lists of floats, one per holdout dim
    """
    per_dim_r: list[float] = []
    per_dim_rmse: list[float] = []
    best_alphas: list[float] = []

    for col in holdout_cols:
        y = shift_matrix[:, col].astype(float)

        # GCV alpha selection
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        alpha = float(rcv.alpha_)
        best_alphas.append(alpha)

        y_loo, _ = _ridge_loo_for_dimension(X, y, alpha)

        if np.std(y) < 1e-10 or np.std(y_loo) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(y, y_loo)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((y - y_loo) ** 2)))
        per_dim_rmse.append(rmse)

    return per_dim_r, per_dim_rmse, best_alphas


def holdout_accuracy_ridge(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
    include_county_mean: bool = True,
) -> dict:
    """Holdout accuracy using Ridge regression on type membership scores.

    Fits RidgeCV (GCV alpha selection) per holdout dimension.
    Uses the exact LOO formula via augmented hat matrix: y_loo_i = y_i - e_i / (1 - H_ii),
    where H is the hat matrix for the augmented feature matrix [1 | X] with
    the intercept unpenalized (matching sklearn fit_intercept=True behavior).

    This method was validated in experiment_ridge_prediction.py (S197) and
    achieves LOO r=0.5335 at J=100, beating LOO type-mean (0.448) by +0.085.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        County type scores (soft membership, row-normalized).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix (training + holdout columns).
    training_cols : list[int]
        Column indices of training dimensions.
    holdout_cols : list[int]
        Column indices of holdout dimensions.
    include_county_mean : bool
        If True, augment features with county training mean (recommended).
        Method (d) in S197: LOO r=0.5335. Method (c) without mean: 0.5223.

    Returns
    -------
    dict with keys:
        "mean_r"         -- float, mean LOO Pearson r across holdout dims
        "per_dim_r"      -- list[float], one r per holdout dim
        "mean_rmse"      -- float, mean LOO RMSE across holdout dims
        "per_dim_rmse"   -- list[float], one RMSE per holdout dim
        "best_alphas"    -- list[float], GCV-selected alpha per holdout dim
    """
    try:
        from sklearn.linear_model import RidgeCV
    except ImportError as exc:
        raise ImportError("scikit-learn required for holdout_accuracy_ridge") from exc

    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)  # (N,)

    # Build feature matrix
    if include_county_mean:
        X = np.column_stack([scores, county_training_means])  # (N, J+1)
    else:
        X = scores  # (N, J)

    alphas = np.logspace(-3, 6, 100)
    per_dim_r, per_dim_rmse, best_alphas = _compute_loo_metrics(
        X, shift_matrix, holdout_cols, alphas, RidgeCV
    )

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    mean_rmse = float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0

    return {
        "mean_r": mean_r,
        "per_dim_r": per_dim_r,
        "mean_rmse": mean_rmse,
        "per_dim_rmse": per_dim_rmse,
        "best_alphas": best_alphas,
    }


def _load_and_standardize_demographics(
    demo_path: Path,
    county_fips: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load demographics parquet, inner-join on county_fips, and standardize.

    Parameters
    ----------
    demo_path : Path
        Resolved path to the demographics parquet file.
    county_fips : ndarray of shape (N,)
        FIPS codes aligned with the model's score/shift rows.

    Returns
    -------
    (row_mask, demo_std_feat, demo_numeric_cols) if successful, else None.
    row_mask : ndarray of int indices into the original N rows after inner join.
    demo_std_feat : ndarray of shape (n_matched, n_demo), standardized.
    demo_numeric_cols : list of demographic column names used.
    """
    import pandas as pd

    demo_df = pd.read_parquet(demo_path)
    demo_numeric_cols = [c for c in demo_df.columns if c != "county_fips"]

    # Apply feature pruning: drop columns that hurt Ridge LOO r (S306).
    # The exclusion list lives in config/ridge_feature_exclusions.yaml.
    # The full parquet is still used elsewhere for type profiling/description.
    excluded = _load_feature_exclusions()
    if excluded:
        before = len(demo_numeric_cols)
        demo_numeric_cols = [c for c in demo_numeric_cols if c not in excluded]
        n_pruned = before - len(demo_numeric_cols)
        if n_pruned:
            log.info(
                "_load_and_standardize_demographics: pruned %d / %d features "
                "(config/ridge_feature_exclusions.yaml); %d retained",
                n_pruned,
                before,
                len(demo_numeric_cols),
            )

    demo_df = demo_df[["county_fips"] + demo_numeric_cols].copy()

    idx_df = pd.DataFrame({"county_fips": county_fips, "_row_idx": np.arange(len(county_fips))})
    merged = idx_df.merge(demo_df, on="county_fips", how="inner")

    if merged.empty:
        log.warning("holdout_accuracy_ridge_augmented: inner join produced no rows; skipping")
        return None

    row_mask = merged["_row_idx"].values  # indices into scores/shift_matrix

    # Standardize demographic features (impute NaN with column means)
    demo_raw = merged[demo_numeric_cols].values.astype(float)
    col_means = np.nanmean(demo_raw, axis=0)
    nan_mask = np.isnan(demo_raw)
    if nan_mask.any():
        col_idx = np.where(nan_mask)
        demo_raw[col_idx] = col_means[col_idx[1]]
    demo_mean = demo_raw.mean(axis=0)
    demo_std = demo_raw.std(axis=0)
    demo_std = np.where(demo_std < 1e-10, 1.0, demo_std)
    demo_std_feat = (demo_raw - demo_mean) / demo_std

    return row_mask, demo_std_feat, demo_numeric_cols


def holdout_accuracy_ridge_augmented(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
    county_fips: np.ndarray | None = None,
    demographics_path: str = "data/assembled/county_features_national.parquet",
    include_county_mean: bool = True,
) -> dict | None:
    """Holdout accuracy using Ridge regression on type scores + demographics.

    Extends holdout_accuracy_ridge() by joining county-level demographic features
    from county_features_national.parquet and appending them to the Ridge feature
    matrix. This captures demographic signal beyond the type membership scores.

    Expected gain: +0.07–0.10 LOO r over the scores-only Ridge baseline.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        County type scores (soft membership, row-normalized).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix (training + holdout columns).
    training_cols : list[int]
        Column indices of training dimensions.
    holdout_cols : list[int]
        Column indices of holdout dimensions.
    county_fips : ndarray of shape (N,), optional
        County FIPS strings aligned with rows of scores/shift_matrix.
        Required for the demographics join; if None, returns None.
    demographics_path : str
        Path to the demographics parquet (absolute or relative to project root).
    include_county_mean : bool
        If True, also include the county training mean as a feature.

    Returns
    -------
    dict with keys:
        "mean_r"             -- float, mean LOO Pearson r across holdout dims
        "per_dim_r"          -- list[float]
        "mean_rmse"          -- float
        "per_dim_rmse"       -- list[float]
        "best_alphas"        -- list[float]
        "n_matched_counties" -- int, rows kept after inner join
        "n_demo_features"    -- int, number of demographic columns added
    Returns None if demographics file is missing or county_fips is None.
    """
    if county_fips is None:
        log.warning("holdout_accuracy_ridge_augmented: county_fips not provided; skipping")
        return None

    try:
        from sklearn.linear_model import RidgeCV
    except ImportError as exc:
        raise ImportError("scikit-learn required for holdout_accuracy_ridge_augmented") from exc

    try:
        import pandas as pd  # noqa: F401 — imported for type-checking; used via _load_and_standardize_demographics
    except ImportError as exc:
        raise ImportError("pandas required for holdout_accuracy_ridge_augmented") from exc

    # Resolve demographics path
    demo_path = Path(demographics_path)
    if not demo_path.is_absolute():
        demo_path = PROJECT_ROOT / demo_path

    if not demo_path.exists():
        log.warning(
            "holdout_accuracy_ridge_augmented: demographics file not found at %s; skipping",
            demo_path,
        )
        return None

    demo_result = _load_and_standardize_demographics(demo_path, county_fips)
    if demo_result is None:
        return None
    row_mask, demo_std_feat, demo_numeric_cols = demo_result

    n_matched = len(row_mask)
    n_demo = len(demo_numeric_cols)

    # Subset arrays to matched rows
    scores_sub = scores[row_mask]
    shift_sub = shift_matrix[row_mask]
    training_data = shift_sub[:, training_cols]
    county_training_means = training_data.mean(axis=1)

    # Build augmented feature matrix
    if include_county_mean:
        X = np.column_stack([scores_sub, county_training_means, demo_std_feat])
    else:
        X = np.column_stack([scores_sub, demo_std_feat])

    alphas = np.logspace(-3, 6, 100)
    per_dim_r, per_dim_rmse, best_alphas = _compute_loo_metrics(
        X, shift_sub, holdout_cols, alphas, RidgeCV
    )

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    mean_rmse = float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0

    return {
        "mean_r": mean_r,
        "per_dim_r": per_dim_r,
        "mean_rmse": mean_rmse,
        "per_dim_rmse": per_dim_rmse,
        "best_alphas": best_alphas,
        "n_matched_counties": n_matched,
        "n_demo_features": n_demo,
    }
