"""Type validation suite for the electoral type discovery pipeline.

Four validation functions test whether the discovered type structure is real
and stable, plus a report generator that runs all validations end-to-end.

Usage:
    python -m src.validation.validate_types
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Core validation functions ─────────────────────────────────────────────────


def type_coherence(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    holdout_cols: list[int],
) -> dict:
    """Within-type vs between-type variance on holdout shifts.

    For each holdout dimension:
    1. Assign each county to its dominant type (highest absolute score).
    2. Compute within-type variance (mean of per-type variances).
    3. Compute between-type variance (variance of type means).
    4. Compute ratio = between / (within + between).

    A higher ratio means types explain more holdout variance.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        Rotated county type scores (soft membership).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix (all dimensions including holdout).
    holdout_cols : list[int]
        Column indices of holdout dimensions in shift_matrix.

    Returns
    -------
    dict with keys:
        "mean_ratio"     -- float, mean coherence ratio across holdout dims
        "per_dim_ratios" -- list[float], one ratio per holdout dim
    """
    dominant_types = np.argmax(np.abs(scores), axis=1)
    unique_types = np.unique(dominant_types)

    per_dim_ratios: list[float] = []

    for col in holdout_cols:
        values = shift_matrix[:, col]

        # Per-type variances and means
        type_variances = []
        type_means = []
        for t in unique_types:
            mask = dominant_types == t
            if mask.sum() < 2:
                # Single-county type has zero variance; skip from within-var
                type_variances.append(0.0)
            else:
                type_variances.append(float(np.var(values[mask], ddof=0)))
            type_means.append(float(np.mean(values[mask])))

        type_means_arr = np.array(type_means)

        within_var = float(np.mean(type_variances))
        # Between-type variance: variance of type means (unweighted)
        between_var = float(np.var(type_means_arr, ddof=0))

        total = within_var + between_var
        if total < 1e-12:
            ratio = 0.0
        else:
            ratio = between_var / total

        # Clamp to [0, 1] for numerical safety
        ratio = float(np.clip(ratio, 0.0, 1.0))
        per_dim_ratios.append(ratio)

    mean_ratio = float(np.mean(per_dim_ratios)) if per_dim_ratios else 0.0

    return {"mean_ratio": mean_ratio, "per_dim_ratios": per_dim_ratios}


def type_stability(
    shift_matrix: np.ndarray,
    window_a_cols: list[int],
    window_b_cols: list[int],
    j: int,
) -> dict:
    """Subspace angle between types discovered from two time windows.

    1. Fit SVD+varimax on window A -> scores_a (N x J).
    2. Fit SVD+varimax on window B -> scores_b (N x J).
    3. Compute principal angles between column spaces of scores_a and scores_b.
    4. Report max angle (degrees).

    Parameters
    ----------
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix. Windows are column subsets.
    window_a_cols : list[int]
        Column indices for window A (e.g., 2000-2012 pairs).
    window_b_cols : list[int]
        Column indices for window B (e.g., 2012-2024 pairs).
    j : int
        Number of types to discover in each window.

    Returns
    -------
    dict with keys:
        "max_angle_degrees"  -- float
        "mean_angle_degrees" -- float
        "stable"             -- bool, True when max_angle < 30 degrees
    """
    from src.discovery.run_type_discovery import discover_types

    matrix_a = shift_matrix[:, window_a_cols]
    matrix_b = shift_matrix[:, window_b_cols]

    result_a = discover_types(matrix_a, j=j, random_state=42)
    result_b = discover_types(matrix_b, j=j, random_state=42)

    scores_a = result_a.scores  # N x J
    scores_b = result_b.scores  # N x J

    # Compute principal angles between column spaces via SVD of Q_a^T Q_b
    # QR-decompose each score matrix to get orthonormal bases
    Q_a, _ = np.linalg.qr(scores_a)
    Q_b, _ = np.linalg.qr(scores_b)

    # Only keep J columns (qr returns min(N,J) columns with full mode)
    Q_a = Q_a[:, :j]
    Q_b = Q_b[:, :j]

    # Singular values of Q_a^T @ Q_b are cosines of principal angles
    M = Q_a.T @ Q_b
    sigma = np.linalg.svd(M, compute_uv=False)

    # Clamp to [-1, 1] to guard against numerical drift
    sigma = np.clip(sigma, -1.0, 1.0)
    angles_rad = np.arccos(np.abs(sigma))
    angles_deg = np.degrees(angles_rad)

    max_angle = float(np.max(angles_deg))
    mean_angle = float(np.mean(angles_deg))

    return {
        "max_angle_degrees": max_angle,
        "mean_angle_degrees": mean_angle,
        "stable": max_angle < 30.0,
    }


def holdout_accuracy(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    holdout_cols: list[int],
    dominant_types: np.ndarray,
) -> dict:
    """Holdout Pearson r: predict holdout shifts from type means.

    For each holdout column:
    1. Compute type-level mean of that column (weighted by absolute scores).
    2. Predict each county's value = weighted sum of type means.
    3. Compute Pearson r between predicted and actual.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        Rotated county type scores (soft membership).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix.
    holdout_cols : list[int]
        Column indices of holdout dimensions in shift_matrix.
    dominant_types : ndarray of shape (N,)
        Dominant type index per county (argmax of abs scores).

    Returns
    -------
    dict with keys:
        "mean_r"    -- float, mean Pearson r across holdout dims
        "per_dim_r" -- list[float], one r per holdout dim
    """
    n, j = scores.shape

    # Normalize absolute scores to weights summing to 1 per county
    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums  # N x J

    # Type-level weighted means: type_means[t, d] = weighted mean over counties
    weight_sums = weights.sum(axis=0)  # J
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    per_dim_r: list[float] = []

    for col in holdout_cols:
        actual = shift_matrix[:, col]

        # Weighted type means for this column
        type_means = (weights.T @ actual) / weight_sums  # J

        # Predicted: weighted sum of type means per county
        predicted = weights @ type_means  # N

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0

    return {"mean_r": mean_r, "per_dim_r": per_dim_r}


def holdout_accuracy_county_prior(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
) -> dict:
    """Holdout accuracy using county-level priors + type covariance adjustment.

    This mirrors the production prediction pipeline where:
    - Each county's prior = its own historical mean shift (from training cols)
    - Types determine only the comovement adjustment

    For each holdout column:
    1. Compute each county's historical mean shift from training columns.
    2. Compute type-level mean shift for training columns.
    3. Compute type-level mean shift for holdout column.
    4. Type adjustment = holdout type mean - training type mean (per type).
    5. County prediction = county training mean + score-weighted type adjustment.
    6. Compute Pearson r and RMSE between predicted and actual.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        County type scores (soft membership).
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix (training + holdout columns).
    training_cols : list[int]
        Column indices of training dimensions.
    holdout_cols : list[int]
        Column indices of holdout dimensions.

    Returns
    -------
    dict with keys:
        "mean_r"    -- float, mean Pearson r across holdout dims
        "per_dim_r" -- list[float], one r per holdout dim
        "mean_rmse" -- float, mean RMSE across holdout dims
        "per_dim_rmse" -- list[float], one RMSE per holdout dim
    """
    n, j = scores.shape

    # Normalize absolute scores to weights summing to 1 per county
    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums  # N x J

    weight_sums_per_type = weights.sum(axis=0)  # J
    weight_sums_per_type = np.where(weight_sums_per_type == 0, 1.0, weight_sums_per_type)

    # County-level training mean (each county's own historical baseline)
    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)  # N

    # Type-level training mean
    type_training_means = (weights.T @ county_training_means) / weight_sums_per_type  # J

    per_dim_r: list[float] = []
    per_dim_rmse: list[float] = []

    for col in holdout_cols:
        actual = shift_matrix[:, col]

        # Type-level holdout mean
        type_holdout_means = (weights.T @ actual) / weight_sums_per_type  # J

        # Type adjustment: how much each type shifted from training to holdout
        type_adjustment = type_holdout_means - type_training_means  # J

        # County prediction = own baseline + score-weighted type adjustment
        county_adjustment = (weights * type_adjustment[None, :]).sum(axis=1)  # N
        predicted = county_training_means + county_adjustment

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        per_dim_rmse.append(rmse)

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    mean_rmse = float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0

    return {
        "mean_r": mean_r,
        "per_dim_r": per_dim_r,
        "mean_rmse": mean_rmse,
        "per_dim_rmse": per_dim_rmse,
    }


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

    n, j = scores.shape
    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)  # (N,)

    # Build feature matrix
    if include_county_mean:
        X = np.column_stack([scores, county_training_means])  # (N, J+1)
    else:
        X = scores  # (N, J)

    alphas = np.logspace(-3, 6, 100)
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

        # Exact LOO via augmented hat matrix
        N_feat = X.shape[1]
        X_aug = np.column_stack([np.ones(n), X])  # (N, N_feat+1)
        pen = alpha * np.eye(N_feat + 1)
        pen[0, 0] = 0.0  # unpenalized intercept
        A = X_aug.T @ X_aug + pen
        A_inv = np.linalg.inv(A)
        h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)  # (N,) hat diag
        beta = A_inv @ X_aug.T @ y
        y_hat = X_aug @ beta
        e = y - y_hat
        denom = 1.0 - h
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        y_loo = y - e / denom  # exact LOO predictions

        if np.std(y) < 1e-10 or np.std(y_loo) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(y, y_loo)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((y - y_loo) ** 2)))
        per_dim_rmse.append(rmse)

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    mean_rmse = float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0

    return {
        "mean_r": mean_r,
        "per_dim_r": per_dim_r,
        "mean_rmse": mean_rmse,
        "per_dim_rmse": per_dim_rmse,
        "best_alphas": best_alphas,
    }


def holdout_accuracy_county_prior_loo(
    scores: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
) -> dict:
    """Leave-one-out holdout accuracy using county-level priors.

    Same as holdout_accuracy_county_prior but removes each county from the
    type mean computation before predicting it. This eliminates inflation
    from small types where a county dominates its own type mean.

    Returns
    -------
    dict with keys:
        "mean_r"       -- float, mean LOO Pearson r across holdout dims
        "per_dim_r"    -- list[float], one r per holdout dim
        "mean_rmse"    -- float, mean LOO RMSE across holdout dims
        "per_dim_rmse" -- list[float], one RMSE per holdout dim
    """
    n, j = scores.shape

    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums

    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)

    # Precompute global weighted sums for efficient LOO
    global_weight_sums = weights.sum(axis=0)  # J
    global_weighted_train = weights.T @ county_training_means  # J

    per_dim_r: list[float] = []
    per_dim_rmse: list[float] = []

    for col in holdout_cols:
        actual = shift_matrix[:, col]
        global_weighted_hold = weights.T @ actual  # J

        predicted = np.zeros(n)
        for i in range(n):
            # LOO: subtract county i's contribution from type sums
            loo_ws = global_weight_sums - weights[i]
            loo_ws = np.where(loo_ws < 1e-12, 1e-12, loo_ws)
            loo_train = (global_weighted_train - weights[i] * county_training_means[i]) / loo_ws
            loo_hold = (global_weighted_hold - weights[i] * actual[i]) / loo_ws
            type_adj = loo_hold - loo_train
            predicted[i] = county_training_means[i] + (weights[i] * type_adj).sum()

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        per_dim_rmse.append(rmse)

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    mean_rmse = float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0

    return {
        "mean_r": mean_r,
        "per_dim_r": per_dim_r,
        "mean_rmse": mean_rmse,
        "per_dim_rmse": per_dim_rmse,
    }


# ── Report generator ──────────────────────────────────────────────────────────


def generate_type_validation_report(
    shift_parquet_path: str = "data/shifts/county_shifts_multiyear.parquet",
    type_assignments_path: str = "data/communities/type_assignments.parquet",
    type_covariance_path: str = "data/covariance/type_covariance.parquet",
    type_profiles_path: str = "data/communities/type_profiles.parquet",
    min_year: int = 2008,
) -> dict:
    """Run all type validations and save a JSON report.

    Loads shift matrix and type assignments, runs type_coherence,
    type_stability, and holdout_accuracy, then saves results to
    data/validation/type_validation_report.json.

    Returns
    -------
    dict with all validation results.
    """
    import pandas as pd

    from src.covariance.construct_type_covariance import (
        CovarianceResult,
        validate_covariance,
    )

    # Resolve paths relative to project root if not absolute
    def resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    shifts_path = resolve(shift_parquet_path)
    assignments_path = resolve(type_assignments_path)

    log.info("Loading shift matrix from %s", shifts_path)
    shifts_df = pd.read_parquet(shifts_path)

    # Separate FIPS and shift columns
    all_cols = [c for c in shifts_df.columns if c != "county_fips"]

    # Identify holdout columns (2020→2024 presidential)
    holdout_keywords = ["20_24"]
    holdout_col_names = [c for c in all_cols if any(kw in c for kw in holdout_keywords)]
    training_col_names_unfiltered = [c for c in all_cols if c not in holdout_col_names]

    # Filter training columns to min_year (match type discovery's filter)
    training_col_names = []
    for c in training_col_names_unfiltered:
        parts = c.split("_")
        try:
            y2 = int(parts[-2])
            y1 = y2 + (1900 if y2 >= 50 else 2000) if len(parts[-2]) == 2 else y2
            if y1 >= min_year:
                training_col_names.append(c)
        except (ValueError, IndexError):
            training_col_names.append(c)  # keep if can't parse

    if not training_col_names:
        training_col_names = training_col_names_unfiltered  # fallback

    # Use filtered training + holdout as "all" for coherence/holdout metrics
    used_cols = training_col_names + holdout_col_names
    full_matrix = shifts_df[used_cols].values
    holdout_indices = [used_cols.index(c) for c in holdout_col_names]

    # Training-only matrix for stability windows
    training_matrix = shifts_df[training_col_names].values
    n_train = len(training_col_names)
    mid = n_train // 2
    window_a_cols = list(range(0, mid))
    window_b_cols = list(range(mid, n_train))

    log.info("Loading type assignments from %s", assignments_path)
    assignments_df = pd.read_parquet(assignments_path)

    # Extract scores and dominant types
    score_cols = [c for c in assignments_df.columns if c.startswith("type_") and c.endswith("_score")]
    if not score_cols:
        score_cols = [c for c in assignments_df.columns if c != "county_fips" and c != "dominant_type"]
    scores = assignments_df[score_cols].values
    if "dominant_type" in assignments_df.columns:
        dominant_types = assignments_df["dominant_type"].values
    else:
        dominant_types = np.argmax(np.abs(scores), axis=1)

    j = scores.shape[1]

    # --- Run validations ---
    log.info("Running type_coherence...")
    coherence = type_coherence(scores, full_matrix, holdout_indices)

    log.info("Running type_stability...")
    stability = type_stability(training_matrix, window_a_cols, window_b_cols, j=j)

    log.info("Running holdout_accuracy (type-mean prior)...")
    accuracy = holdout_accuracy(scores, full_matrix, holdout_indices, dominant_types)

    log.info("Running holdout_accuracy (county-level prior)...")
    training_indices = [used_cols.index(c) for c in training_col_names]
    accuracy_county_prior = holdout_accuracy_county_prior(
        scores, full_matrix, training_indices, holdout_indices,
    )

    log.info("Running holdout_accuracy_county_prior_loo...")
    accuracy_county_prior_loo = holdout_accuracy_county_prior_loo(
        scores, full_matrix, training_indices, holdout_indices,
    )

    log.info("Running holdout_accuracy_ridge (LOO via hat matrix)...")
    accuracy_ridge = holdout_accuracy_ridge(
        scores, full_matrix, training_indices, holdout_indices, include_county_mean=True,
    )

    # --- Covariance validation (if data available) ---
    cov_path = resolve(type_covariance_path)
    cov_validation_r: float | None = None
    if cov_path.exists():
        try:
            cov_df = pd.read_parquet(cov_path)
            cov_matrix = cov_df.values

            # Build a minimal CovarianceResult for validate_covariance
            corr_std = np.sqrt(np.diag(cov_matrix))
            corr_std = np.where(corr_std == 0, 1.0, corr_std)
            corr_matrix = cov_matrix / np.outer(corr_std, corr_std)

            constructed = CovarianceResult(
                correlation_matrix=corr_matrix,
                covariance_matrix=cov_matrix,
                validation_r=float("nan"),
                used_hybrid=False,
                sigma_base=0.07,
            )

            # Group training shift columns into election pairs (every 3 dims)
            n_train_cols = training_matrix.shape[1]
            election_col_groups = [
                list(range(i, min(i + 3, n_train_cols)))
                for i in range(0, n_train_cols, 3)
            ]
            cov_validation_r = validate_covariance(
                constructed, scores, training_matrix, election_col_groups
            )
            log.info("Covariance validation r = %.3f", cov_validation_r)
        except Exception as e:
            log.warning("Covariance validation failed: %s", e)

    # --- Assemble report ---
    report: dict = {
        "coherence": coherence,
        "stability": stability,
        "holdout_accuracy": accuracy,
        "holdout_accuracy_county_prior": accuracy_county_prior,
        "holdout_accuracy_county_prior_loo": accuracy_county_prior_loo,
        "holdout_accuracy_ridge": accuracy_ridge,
        "covariance_validation_r": cov_validation_r,
        "j": j,
        "n_counties": int(full_matrix.shape[0]),
        "n_training_dims": n_train,
        "n_holdout_dims": len(holdout_indices),
        "holdout_columns": holdout_col_names,
        "min_year": min_year,
    }

    # Print summary
    print("\n" + "=" * 65)
    print("Type Validation Report")
    print("=" * 65)
    print(f"  J (types):             {j}")
    print(f"  N (counties):          {full_matrix.shape[0]}")
    print(f"  Training dims:         {n_train}")
    print(f"  Holdout dims:          {len(holdout_indices)}")
    print()
    print(f"  Coherence (mean ratio): {coherence['mean_ratio']:.3f}  (> 0.5 = strong)")
    print(f"  Stability (max angle):  {stability['max_angle_degrees']:.1f} deg  (< 30 = stable)")
    print(f"  Holdout accuracy (r):   {accuracy['mean_r']:.3f}  (> 0.5 = good, type-mean prior)")
    print(f"  Holdout accuracy (r):   {accuracy_county_prior['mean_r']:.3f}  (county-level prior)")
    if "mean_rmse" in accuracy_county_prior:
        print(f"  Holdout RMSE:           {accuracy_county_prior['mean_rmse']:.4f}  (county-level prior)")
    print(f"  Holdout LOO r:          {accuracy_county_prior_loo['mean_r']:.3f}  (county-level prior, LOO)")
    if "mean_rmse" in accuracy_county_prior_loo:
        print(f"  Holdout LOO RMSE:       {accuracy_county_prior_loo['mean_rmse']:.4f}  (county-level prior, LOO)")
    print(f"  Holdout Ridge LOO r:    {accuracy_ridge['mean_r']:.3f}  (Ridge scores+county_mean, LOO)")
    if "mean_rmse" in accuracy_ridge:
        print(f"  Holdout Ridge RMSE:     {accuracy_ridge['mean_rmse']:.4f}  (Ridge, LOO)")
    if cov_validation_r is not None:
        print(f"  Covariance val r:       {cov_validation_r:.3f}  (> 0.4 = acceptable)")
    print("=" * 65)

    # Save JSON report
    out_dir = PROJECT_ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "type_validation_report.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    log.info("Report saved to %s", out_path)
    print(f"\n  Full report: {out_path}")

    return report


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Type validation report")
    parser.add_argument("--shifts", default="data/shifts/county_shifts_multiyear.parquet")
    parser.add_argument("--assignments", default="data/communities/type_assignments.parquet")
    parser.add_argument("--covariance", default="data/covariance/type_covariance.parquet")
    parser.add_argument("--profiles", default="data/communities/type_profiles.parquet")
    parser.add_argument("--min-year", type=int, default=2008,
                        help="Min start year for training shifts (default: 2008, matching type discovery)")
    args = parser.parse_args()

    generate_type_validation_report(
        shift_parquet_path=args.shifts,
        type_assignments_path=args.assignments,
        type_covariance_path=args.covariance,
        type_profiles_path=args.profiles,
        min_year=args.min_year,
    )


if __name__ == "__main__":
    main()
