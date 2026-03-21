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


# ── Report generator ──────────────────────────────────────────────────────────


def generate_type_validation_report(
    shift_parquet_path: str = "data/shifts/county_shifts_multiyear.parquet",
    type_assignments_path: str = "data/communities/type_assignments.parquet",
    type_covariance_path: str = "data/covariance/type_covariance.parquet",
    type_profiles_path: str = "data/communities/type_profiles.parquet",
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
    training_col_names = [c for c in all_cols if c not in holdout_col_names]

    full_matrix = shifts_df[all_cols].values
    holdout_indices = [all_cols.index(c) for c in holdout_col_names]

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

    log.info("Running holdout_accuracy...")
    accuracy = holdout_accuracy(scores, full_matrix, holdout_indices, dominant_types)

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
        "covariance_validation_r": cov_validation_r,
        "j": j,
        "n_counties": int(full_matrix.shape[0]),
        "n_training_dims": n_train,
        "n_holdout_dims": len(holdout_indices),
        "holdout_columns": holdout_col_names,
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
    print(f"  Holdout accuracy (r):   {accuracy['mean_r']:.3f}  (> 0.5 = good)")
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
    args = parser.parse_args()

    generate_type_validation_report(
        shift_parquet_path=args.shifts,
        type_assignments_path=args.assignments,
        type_covariance_path=args.covariance,
        type_profiles_path=args.profiles,
    )


if __name__ == "__main__":
    main()
