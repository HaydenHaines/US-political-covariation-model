"""Experiment: observed-primary covariance vs demographic-primary baseline.

Tests whether using the LW-regularized observed electoral correlation as the
PRIMARY matrix (shrunk toward the demographic correlation) improves the
covariance validation r over the current demographic-primary approach.

Current baseline: covariance val r = 0.556

Sweep: C(alpha) = alpha * observed_LW + (1-alpha) * demographic_corr
       for alpha in {0.0, 0.1, ..., 1.0}

alpha=0.0  →  pure demographic correlation (current approach after LW vs 1s shrink)
alpha=1.0  →  pure LW-regularized observed correlation

IMPORTANT: The naive in-sample validation is also computed (for reference), but
the primary evaluation uses Leave-One-Election-Out (LOEO) cross-validation.

LOEO CV:
  For each held-out election e:
    - Build the blend matrix using only the remaining T-1 elections
    - Validate against the held-out election's J-vector of type shifts
  This gives an unbiased estimate of how well the blend predicts unseen elections.

Run from project root:
    uv run python scripts/experiment_observed_primary_cov.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure project src is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.covariance.construct_type_covariance import (
    CovarianceResult,
    _compute_type_shifts,
    _enforce_pd,
    _load_type_profiles,
    _load_type_scores_and_shifts,
    compute_observed_correlation,
    construct_type_covariance,
    validate_covariance,
)


def build_demographic_corr(type_profiles: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Compute the demographic Pearson correlation matrix (before shrinkage toward 1s).

    This is step 3 of construct_type_covariance() — the raw demographic signal
    before the lambda_shrinkage blend toward all-1s.  We need it as the
    shrinkage target for the observed-primary blend.
    """
    X = type_profiles[feature_cols].values.astype(float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_scaled = (X - X_min) / X_range

    C = np.corrcoef(X_scaled)
    C = np.where(np.isnan(C), 0.0, C)
    np.fill_diagonal(C, 1.0)
    # Floor negatives (same as production default)
    C = np.maximum(C, 0.0)
    return C


def make_cov_result_from_corr(corr: np.ndarray, sigma_base: float = 0.07) -> CovarianceResult:
    """Wrap a correlation matrix in a CovarianceResult for validation."""
    return CovarianceResult(
        correlation_matrix=corr,
        covariance_matrix=sigma_base ** 2 * corr,
        validation_r=float("nan"),
        used_hybrid=False,
        sigma_base=sigma_base,
    )


def loeo_validate_blend(
    alpha: float,
    demo_corr_pd: np.ndarray,
    type_scores: np.ndarray,
    shift_matrix: np.ndarray,
    election_col_groups: list[list[int]],
) -> float:
    """Leave-One-Election-Out cross-validation of the blended matrix.

    For each held-out election e:
      1. Compute obs_LW using only the T-1 training elections
      2. Build blend: C = alpha * obs_LW_train + (1-alpha) * demo_corr
      3. Validate: correlate blend's off-diagonal against the sample
         correlation from ALL T elections (the gold standard target)

    This is an honest out-of-sample test: the blend matrix is built without
    seeing the held-out election's contribution to the observed covariance.
    The validation target is the full-sample observed correlation, which
    represents the best unbiased estimate of the true structure.

    Returns the mean LOEO Pearson r across all held-out elections.
    """
    T = len(election_col_groups)
    J = type_scores.shape[1]

    # Precompute all type-shift vectors: (T, J)
    all_type_shifts = _compute_type_shifts(type_scores, shift_matrix, election_col_groups)

    # Full-sample observed correlation (the validation target / gold standard)
    obs_cov_full = np.cov(all_type_shifts.T)  # (J, J)
    obs_std_full = np.sqrt(np.diag(obs_cov_full))
    obs_std_full[obs_std_full == 0] = 1.0
    obs_corr_full = obs_cov_full / np.outer(obs_std_full, obs_std_full)
    np.fill_diagonal(obs_corr_full, 1.0)
    mask = ~np.eye(J, dtype=bool)
    obs_off_full = obs_corr_full[mask]

    loeo_rs = []

    for held_out_idx in range(T):
        # Training election groups (exclude held-out)
        train_groups = [g for i, g in enumerate(election_col_groups) if i != held_out_idx]

        if len(train_groups) < 2:
            continue  # Need at least 2 elections to estimate a covariance

        # Recompute observed LW using only training elections (no peeking at held-out)
        obs_lw_train = compute_observed_correlation(
            type_scores, shift_matrix, train_groups, shrinkage=None
        )

        # Build blend from training-only observed LW
        blend = alpha * obs_lw_train + (1 - alpha) * demo_corr_pd
        blend_pd = _enforce_pd(blend)
        np.fill_diagonal(blend_pd, 1.0)

        blend_off = blend_pd[mask]

        if np.std(blend_off) < 1e-10 or np.std(obs_off_full) < 1e-10:
            continue

        r = float(np.corrcoef(blend_off, obs_off_full)[0, 1])
        loeo_rs.append(np.clip(r, -1.0, 1.0))

    if not loeo_rs:
        return float("nan")
    return float(np.mean(loeo_rs))


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT: Observed-primary covariance vs demographic-primary")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading type profiles...")
    type_profiles, feature_cols = _load_type_profiles()
    J = len(type_profiles)
    print(f"  J={J} types, {len(feature_cols)} feature columns")

    print("Loading type scores and shift matrix...")
    type_scores, shift_matrix, election_col_groups = _load_type_scores_and_shifts()
    T = len(election_col_groups)
    print(f"  type_scores shape: {type_scores.shape}")
    print(f"  shift_matrix shape: {shift_matrix.shape}")
    print(f"  election groups (T): {T}")

    # ------------------------------------------------------------------
    # Baseline: current demographic-primary covariance (in-sample)
    # ------------------------------------------------------------------
    print("\nComputing BASELINE (demographic-primary, lambda_shrinkage=0.75)...")
    baseline_result = construct_type_covariance(
        type_profiles,
        feature_cols,
        lambda_shrinkage=0.75,
        sigma_base=0.07,
        floor_negatives=True,
    )
    baseline_val_r = validate_covariance(
        baseline_result, type_scores, shift_matrix, election_col_groups
    )
    print(f"  Baseline in-sample covariance val r = {baseline_val_r:.4f}")
    # Baseline LOEO (demographic-primary doesn't use observed data, so LOEO == in-sample)
    baseline_loeo_r = loeo_validate_blend(
        0.0,
        _enforce_pd(build_demographic_corr(type_profiles, feature_cols)),
        type_scores, shift_matrix, election_col_groups,
    )
    print(f"  Baseline LOEO val r (alpha=0.0):      {baseline_loeo_r:.4f}")

    # ------------------------------------------------------------------
    # Observed LW-regularized correlation (in-sample reference)
    # ------------------------------------------------------------------
    print("\nComputing observed LW-regularized correlation (full sample)...")
    obs_lw_corr = compute_observed_correlation(
        type_scores, shift_matrix, election_col_groups, shrinkage=None
    )
    obs_result = make_cov_result_from_corr(obs_lw_corr)
    obs_val_r_insample = validate_covariance(
        obs_result, type_scores, shift_matrix, election_col_groups
    )
    print(f"  Pure observed-LW in-sample val r (alpha=1.0): {obs_val_r_insample:.4f}")
    print("  (NOTE: in-sample r=1.0 is expected — this is tautological.)")

    # ------------------------------------------------------------------
    # Demographic correlation matrix (shrinkage target)
    # ------------------------------------------------------------------
    print("\nComputing demographic correlation matrix (shrinkage target)...")
    demo_corr = build_demographic_corr(type_profiles, feature_cols)
    demo_corr_pd = _enforce_pd(demo_corr)
    np.fill_diagonal(demo_corr_pd, 1.0)

    # ------------------------------------------------------------------
    # In-sample alpha sweep (for reference, expected to be trivially 1.0 at alpha=1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 1: IN-SAMPLE alpha sweep (REFERENCE ONLY — biased at alpha>0)")
    print("=" * 70)
    print(f"\n{'alpha':>6}  {'in-sample r':>12}")
    print("-" * 22)

    insample_results: list[tuple[float, float]] = []
    for alpha_i in range(11):
        alpha = round(alpha_i * 0.1, 1)
        blend = alpha * obs_lw_corr + (1 - alpha) * demo_corr_pd
        blend_pd = _enforce_pd(blend)
        np.fill_diagonal(blend_pd, 1.0)
        r = validate_covariance(
            make_cov_result_from_corr(blend_pd), type_scores, shift_matrix, election_col_groups
        )
        insample_results.append((alpha, r))
        marker = " <-- baseline" if alpha == 0.0 else ""
        print(f"  {alpha:>4.1f}  {r:>12.4f}{marker}")

    # ------------------------------------------------------------------
    # LOEO cross-validation alpha sweep (unbiased)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SECTION 2: LOEO CV alpha sweep (T={T}, leave-one-election-out)")
    print("=" * 70)
    print("  Computing LOEO for each alpha (this may take ~30s)...")
    print(f"\n{'alpha':>6}  {'LOEO val_r':>12}  {'vs baseline LOEO':>17}")
    print("-" * 40)

    loeo_results: list[tuple[float, float]] = []
    for alpha_i in range(11):
        alpha = round(alpha_i * 0.1, 1)
        loeo_r = loeo_validate_blend(
            alpha, demo_corr_pd,
            type_scores, shift_matrix, election_col_groups,
        )
        loeo_results.append((alpha, loeo_r))
        delta = loeo_r - baseline_loeo_r
        marker = " <-- baseline" if alpha == 0.0 else ""
        print(f"  {alpha:>4.1f}  {loeo_r:>12.4f}  {delta:>+17.4f}{marker}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    best_loeo_alpha, best_loeo_r = max(loeo_results, key=lambda x: x[1])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Production baseline in-sample val r:       {baseline_val_r:.4f}")
    print(f"  LOEO val r at alpha=0.0 (demographic):     {baseline_loeo_r:.4f}")
    print(f"  Best LOEO alpha: {best_loeo_alpha:.1f}")
    print(f"  Best LOEO val r: {best_loeo_r:.4f}")
    print(f"  LOEO improvement over alpha=0.0:           {best_loeo_r - baseline_loeo_r:+.4f}")
    print()

    if best_loeo_r > baseline_loeo_r and best_loeo_alpha > 0.0:
        print(f"  RESULT: Observed-primary blend IMPROVES LOEO val_r at alpha={best_loeo_alpha}.")
        print(f"          LOEO r {baseline_loeo_r:.4f} --> {best_loeo_r:.4f} "
              f"({best_loeo_r - baseline_loeo_r:+.4f})")
        print(f"          Recommend: adopt alpha={best_loeo_alpha} in production pipeline.")
        print()
        print("  CAVEAT: In-sample val_r is trivially 1.0 at alpha=1.0 (tautological).")
        print("          LOEO is the honest metric for unseen-election generalization.")
    else:
        print("  RESULT: No LOEO improvement from observed-primary blend.")
        print("          Demographic-primary remains the better approach for unseen elections.")

    print()


if __name__ == "__main__":
    main()
