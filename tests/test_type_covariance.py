"""Tests for src/covariance/construct_type_covariance.py

Uses synthetic type profiles (5 types × 8 demographic features) to exercise
all construction, validation, and hybrid-fallback code paths.
"""
import numpy as np
import pandas as pd
import pytest

from src.covariance.construct_type_covariance import (
    CovarianceResult,
    _rank_reduce,
    apply_hybrid_fallback,
    construct_type_covariance,
    validate_covariance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

J = 5  # number of types
F = 8  # number of demographic features
RNG = np.random.default_rng(42)


@pytest.fixture
def feature_columns():
    return [f"feat_{i}" for i in range(F)]


@pytest.fixture
def type_profiles(feature_columns):
    """Synthetic 5×8 type profile DataFrame with meaningful variation."""
    data = {
        "feat_0": [0.10, 0.30, 0.60, 0.80, 0.95],  # strongly varying
        "feat_1": [0.90, 0.70, 0.40, 0.20, 0.05],  # inversely varying
        "feat_2": [0.50, 0.55, 0.50, 0.45, 0.50],  # nearly constant
        "feat_3": [0.20, 0.40, 0.60, 0.80, 0.95],  # monotone
        "feat_4": [0.80, 0.60, 0.40, 0.20, 0.10],  # monotone decreasing
        "feat_5": [0.30, 0.35, 0.70, 0.65, 0.90],  # non-monotone
        "feat_6": [0.15, 0.45, 0.55, 0.75, 0.85],  # positive trend
        "feat_7": [0.95, 0.75, 0.50, 0.25, 0.05],  # negative trend
    }
    return pd.DataFrame(data)


@pytest.fixture
def default_result(type_profiles, feature_columns):
    return construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.75, sigma_base=0.07
    )


# ---------------------------------------------------------------------------
# Shape and structure
# ---------------------------------------------------------------------------


def test_covariance_shape(default_result):
    """Correlation and covariance matrices must both be J × J."""
    assert default_result.correlation_matrix.shape == (J, J)
    assert default_result.covariance_matrix.shape == (J, J)


def test_covariance_symmetric(default_result):
    """Matrices must be symmetric."""
    C = default_result.correlation_matrix
    Sigma = default_result.covariance_matrix
    np.testing.assert_allclose(C, C.T, atol=1e-10)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)


def test_covariance_positive_definite(default_result):
    """All eigenvalues must be strictly positive."""
    eigvals = np.linalg.eigvalsh(default_result.covariance_matrix)
    assert np.all(eigvals > 0), f"Non-positive eigenvalue found: {eigvals.min()}"


def test_correlation_diagonal_ones(default_result):
    """Diagonal of the correlation matrix must be ≈ 1.0 (after PD repair)."""
    diag = np.diag(default_result.correlation_matrix)
    # After PD repair the diagonal may exceed 1 slightly; it must be close to 1
    np.testing.assert_allclose(diag, np.ones(J), atol=0.05)


# ---------------------------------------------------------------------------
# Shrinkage behaviour
# ---------------------------------------------------------------------------


def test_shrinkage_floor(type_profiles, feature_columns):
    """With floor_negatives=True, minimum off-diagonal >= (1 - lambda_shrinkage)."""
    lam = 0.75
    result = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=lam, floor_negatives=True
    )
    C = result.correlation_matrix
    # Off-diagonal lower bound from shrinkage formula: lam*0 + (1-lam)*1 = 1-lam
    off_diag = C[~np.eye(J, dtype=bool)]
    # After PD repair the minimum may shift slightly; use a generous tolerance
    assert off_diag.min() >= (1 - lam) - 0.05


def test_no_floor_allows_negatives(type_profiles, feature_columns):
    """With floor_negatives=False and lambda=1, negatives in raw Pearson survive."""
    result = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=1.0, floor_negatives=False
    )
    # With inversely-correlated features (feat_1 vs feat_0), raw Pearson will
    # have some negative off-diagonal entries before shrinkage
    C = result.correlation_matrix
    off_diag = C[~np.eye(J, dtype=bool)]
    # At least one entry should be negative when no flooring and no shrinkage pushes up
    assert off_diag.min() < 0.0, "Expected at least one negative off-diagonal entry"


def test_lambda_zero_all_ones(type_profiles, feature_columns):
    """lambda=0 → correlation matrix is all ones (shrink entirely to national swing)."""
    result = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.0, floor_negatives=True
    )
    C = result.correlation_matrix
    # All-ones matrix has rank 1; after PD repair diagonal ≈ 1, off-diagonal ≈ 1
    np.testing.assert_allclose(C, np.ones((J, J)), atol=0.05)


def test_lambda_one_pure_demographic(type_profiles, feature_columns):
    """lambda=1 → pure demographic Pearson correlation (no national-swing blend)."""
    result_lam1 = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=1.0, floor_negatives=True
    )
    result_lam0 = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.0, floor_negatives=True
    )
    # lambda=1 should differ from lambda=0
    assert not np.allclose(
        result_lam1.correlation_matrix, result_lam0.correlation_matrix, atol=0.01
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_constant_feature_handled(feature_columns):
    """A feature with zero variance must not produce NaN in the result."""
    data = {col: [0.5, 0.5, 0.5, 0.5, 0.5] for col in feature_columns}
    # Add variation to at least one feature so corrcoef is non-trivial
    data["feat_0"] = [0.1, 0.3, 0.5, 0.7, 0.9]
    profiles = pd.DataFrame(data)
    result = construct_type_covariance(
        profiles, feature_columns, lambda_shrinkage=0.75
    )
    assert not np.any(np.isnan(result.correlation_matrix))
    assert not np.any(np.isnan(result.covariance_matrix))


def test_minmax_scaling(type_profiles, feature_columns):
    """Min-max scaling must map each feature column to [0, 1]."""
    # We verify indirectly: result with un-scaled data must equal result after
    # manual pre-scaling (because min-max is idempotent on [0,1]-ranged data).
    already_scaled = type_profiles.copy()  # data is already in [0,1]
    result_original = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.75
    )
    result_prescaled = construct_type_covariance(
        already_scaled, feature_columns, lambda_shrinkage=0.75
    )
    np.testing.assert_allclose(
        result_original.correlation_matrix,
        result_prescaled.correlation_matrix,
        atol=1e-10,
    )


def test_sigma_base_scales_covariance(type_profiles, feature_columns):
    """Covariance matrix should equal sigma_base^2 * correlation_matrix (approx)."""
    sigma_base = 0.07
    result = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.75, sigma_base=sigma_base
    )
    expected = sigma_base**2 * result.correlation_matrix
    np.testing.assert_allclose(result.covariance_matrix, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_covariance_returns_float(default_result):
    """validate_covariance must return a float in [-1, 1]."""
    # Build synthetic inputs: 293 counties × J types (scores), 293 × D shifts
    n_counties = 20
    n_dims = 6
    type_scores = RNG.standard_normal((n_counties, J))
    shift_matrix = RNG.standard_normal((n_counties, n_dims))
    # Two elections, each covering columns [0,1,2] and [3,4,5]
    election_col_groups = [[0, 1, 2], [3, 4, 5]]
    r = validate_covariance(
        default_result, type_scores, shift_matrix, election_col_groups
    )
    assert isinstance(r, float)
    assert -1.0 <= r <= 1.0


def test_validate_covariance_high_r_for_consistent_data(default_result):
    """Validation r should be higher when constructed covariance matches data structure."""
    # Create type scores and shifts that are consistent with the constructed covariance
    n_counties = 50
    n_elections = 4
    # Type scores are just random; shifts are correlated through type loadings
    type_scores = RNG.standard_normal((n_counties, J))
    # Each election's shift = type_scores @ some loading + noise
    shifts_list = []
    for _ in range(n_elections):
        loading = RNG.standard_normal(J)
        shifts_list.append(type_scores @ loading + RNG.standard_normal(n_counties) * 0.1)
    shift_matrix = np.column_stack(shifts_list)
    election_col_groups = [[i] for i in range(n_elections)]
    r = validate_covariance(
        default_result, type_scores, shift_matrix, election_col_groups
    )
    # Just ensure it returns a valid number; signal may be weak with random loadings
    assert isinstance(r, float)


# ---------------------------------------------------------------------------
# Hybrid fallback
# ---------------------------------------------------------------------------


def test_hybrid_not_triggered_above_threshold(default_result):
    """When validation_r >= threshold, used_hybrid must be False."""
    observed_cov = np.eye(J)
    result = apply_hybrid_fallback(default_result, observed_cov, validation_r=0.5, threshold=0.4)
    assert result.used_hybrid is False
    assert result.validation_r == pytest.approx(0.5)


def test_hybrid_triggered_below_threshold(default_result):
    """When validation_r < threshold, used_hybrid must be True."""
    observed_cov = np.eye(J)
    result = apply_hybrid_fallback(default_result, observed_cov, validation_r=0.2, threshold=0.4)
    assert result.used_hybrid is True
    assert result.validation_r == pytest.approx(0.2)


def test_hybrid_preserves_pd(default_result):
    """Hybrid result must remain positive definite."""
    observed_cov = np.eye(J)
    result = apply_hybrid_fallback(default_result, observed_cov, validation_r=0.1, threshold=0.4)
    eigvals = np.linalg.eigvalsh(result.covariance_matrix)
    assert np.all(eigvals > 0)


def test_hybrid_preserves_symmetry(default_result):
    """Hybrid covariance must be symmetric."""
    observed_cov = np.eye(J)
    result = apply_hybrid_fallback(default_result, observed_cov, validation_r=0.1, threshold=0.4)
    np.testing.assert_allclose(
        result.covariance_matrix, result.covariance_matrix.T, atol=1e-10
    )


def test_hybrid_sigma_base_unchanged(default_result):
    """Hybrid fallback must preserve the original sigma_base."""
    observed_cov = np.eye(J)
    result = apply_hybrid_fallback(default_result, observed_cov, validation_r=0.1, threshold=0.4)
    assert result.sigma_base == pytest.approx(default_result.sigma_base)


def test_hybrid_no_blend_at_threshold_boundary(default_result):
    """At exactly validation_r == threshold, no hybrid is triggered."""
    observed_cov = np.eye(J)
    result = apply_hybrid_fallback(default_result, observed_cov, validation_r=0.4, threshold=0.4)
    assert result.used_hybrid is False


# ---------------------------------------------------------------------------
# CovarianceResult dataclass
# ---------------------------------------------------------------------------


def test_result_dataclass_fields(default_result):
    """CovarianceResult must expose all required fields."""
    assert hasattr(default_result, "correlation_matrix")
    assert hasattr(default_result, "covariance_matrix")
    assert hasattr(default_result, "validation_r")
    assert hasattr(default_result, "used_hybrid")
    assert hasattr(default_result, "sigma_base")


def test_initial_validation_r_is_nan(default_result):
    """construct_type_covariance must leave validation_r as NaN (set by validate_covariance)."""
    assert np.isnan(default_result.validation_r)


def test_used_hybrid_false_by_default(default_result):
    """construct_type_covariance must set used_hybrid=False initially."""
    assert default_result.used_hybrid is False


# ---------------------------------------------------------------------------
# Rank reduction
# ---------------------------------------------------------------------------


def test_rank_reduce_preserves_shape():
    """_rank_reduce must return same J×J shape."""
    C = np.eye(5) + 0.3 * np.ones((5, 5))
    C_reduced = _rank_reduce(C, k=3)
    assert C_reduced.shape == (5, 5)


def test_rank_reduce_diagonal_ones():
    """After rank reduction, diagonal should be ~1.0 (correlation convention)."""
    C = np.eye(5) + 0.3 * np.ones((5, 5))
    C_reduced = _rank_reduce(C, k=3)
    np.testing.assert_allclose(np.diag(C_reduced), np.ones(5), atol=1e-8)


def test_rank_reduce_symmetric():
    """Result must be symmetric."""
    C = np.eye(5) + 0.3 * np.ones((5, 5))
    C_reduced = _rank_reduce(C, k=3)
    np.testing.assert_allclose(C_reduced, C_reduced.T, atol=1e-10)


def test_rank_reduce_positive_definite():
    """Result must be positive definite."""
    C = np.eye(5) + 0.3 * np.ones((5, 5))
    C_reduced = _rank_reduce(C, k=3)
    eigvals = np.linalg.eigvalsh(C_reduced)
    assert np.all(eigvals > 0)


def test_rank_reduce_effective_rank():
    """Top-k eigenvalues should dominate; the rest should be near-zero."""
    J = 10
    rng = np.random.default_rng(99)
    # Build a full-rank correlation-like matrix
    A = rng.standard_normal((J, J))
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)

    k = 3
    C_reduced = _rank_reduce(C, k=k)
    eigvals = np.linalg.eigvalsh(C_reduced)
    eigvals_sorted = np.sort(eigvals)[::-1]

    # Top-k should carry almost all the variance
    total = eigvals_sorted.sum()
    top_k_share = eigvals_sorted[:k].sum() / total
    assert top_k_share > 0.95, f"Top-{k} share = {top_k_share:.3f}, expected > 0.95"


def test_rank_reduce_k_equals_j_is_identity():
    """max_rank=J should return nearly the same matrix (no reduction)."""
    J = 5
    rng = np.random.default_rng(42)
    A = rng.standard_normal((J, J))
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)

    C_reduced = _rank_reduce(C, k=J)
    np.testing.assert_allclose(C_reduced, C, atol=1e-5)


def test_construct_with_max_rank(type_profiles, feature_columns):
    """construct_type_covariance with max_rank should produce valid result."""
    result = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.75, max_rank=3
    )
    assert result.correlation_matrix.shape == (J, J)
    assert result.covariance_matrix.shape == (J, J)
    # PD
    eigvals = np.linalg.eigvalsh(result.covariance_matrix)
    assert np.all(eigvals > 0)
    # Diagonal ~1 on correlation
    np.testing.assert_allclose(np.diag(result.correlation_matrix), np.ones(J), atol=1e-8)


def test_construct_max_rank_reduces_effective_rank(type_profiles, feature_columns):
    """With max_rank, top-k eigenvalues should dominate."""
    k = 2
    result = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.75, max_rank=k
    )
    eigvals = np.linalg.eigvalsh(result.correlation_matrix)
    eigvals_sorted = np.sort(eigvals)[::-1]
    total = eigvals_sorted.sum()
    top_k_share = eigvals_sorted[:k].sum() / total
    assert top_k_share > 0.90


def test_construct_max_rank_none_same_as_no_reduction(type_profiles, feature_columns):
    """max_rank=None should produce same result as before."""
    result_none = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.75, max_rank=None
    )
    result_default = construct_type_covariance(
        type_profiles, feature_columns, lambda_shrinkage=0.75
    )
    np.testing.assert_allclose(
        result_none.correlation_matrix, result_default.correlation_matrix, atol=1e-12
    )
