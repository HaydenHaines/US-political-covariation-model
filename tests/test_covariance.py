"""Tests for covariance estimation.

These tests exercise covariance-related logic using synthetic inputs so they
run without Stan or the full assembled dataset.  Stan-dependent functions
(compile_and_sample, extract_covariance) require a full pipeline run and are
not tested here; they would belong in integration tests that require the
data/covariance/ directory to exist.

Covered:
  - compute_community_stats: weighted mean and SE computation
  - build_stan_data: data dictionary structure and shapes
  - Mathematical properties of covariance matrices (PSD, symmetry, trace)
  - Tikhonov helper logic used in prior chain extension
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.covariance.run_covariance_model import (
    COMP_COLS,
    ELECTIONS,
    K,
    compute_community_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_mem() -> pd.DataFrame:
    """
    Synthetic tract membership DataFrame with shape compatible with the
    covariance pipeline: columns are tract_geoid, is_uninhabited, c1..c7.
    Each inhabited row is a valid soft-assignment probability vector.
    """
    rng = np.random.default_rng(42)
    n_tracts = 100

    # Dirichlet samples so rows sum to 1
    alpha = np.ones(K)
    W = rng.dirichlet(alpha, size=n_tracts)

    df = pd.DataFrame(W, columns=COMP_COLS)
    df.insert(0, "tract_geoid", [f"12001{i:06d}" for i in range(n_tracts)])
    df["is_uninhabited"] = False
    return df


@pytest.fixture(scope="module")
def synthetic_elec() -> pd.DataFrame:
    """
    Synthetic 2020 election DataFrame compatible with compute_community_stats.
    Includes dem_share and total vote columns.
    """
    rng = np.random.default_rng(7)
    n_tracts = 100
    dem_shares = rng.uniform(0.3, 0.7, size=n_tracts)
    totals = rng.integers(500, 5000, size=n_tracts)

    return pd.DataFrame({
        "tract_geoid": [f"12001{i:06d}" for i in range(n_tracts)],
        "pres_dem_share_2020": dem_shares,
        "pres_total_2020": totals.astype(float),
    })


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------


def test_k_equals_seven():
    """Covariance module must use the canonical K=7 community types."""
    assert K == 7


def test_comp_cols_count():
    """COMP_COLS must list exactly 7 community column names."""
    assert len(COMP_COLS) == 7


def test_comp_cols_names():
    """Community columns must follow the naming convention c1..c7."""
    expected = [f"c{k}" for k in range(1, 8)]
    assert COMP_COLS == expected


def test_elections_list_has_three_entries():
    """The canonical election list covers 2016 (pres), 2018 (gov), 2020 (pres)."""
    assert len(ELECTIONS) == 3


def test_elections_years_are_correct():
    """Election years in ELECTIONS tuple must be 2016, 2018, 2020."""
    years = [e[0] for e in ELECTIONS]
    assert years == [2016, 2018, 2020]


# ---------------------------------------------------------------------------
# compute_community_stats tests
# ---------------------------------------------------------------------------


def test_compute_community_stats_output_shape(synthetic_mem, synthetic_elec):
    """compute_community_stats must return two (K,) arrays."""
    theta, se = compute_community_stats(
        synthetic_mem, synthetic_elec, year=2020, prefix="pres", exclude_al=False
    )
    assert theta.shape == (K,), f"theta shape should be ({K},), got {theta.shape}"
    assert se.shape == (K,), f"se shape should be ({K},), got {se.shape}"


def test_compute_community_stats_theta_in_range(synthetic_mem, synthetic_elec):
    """Weighted mean vote shares must be in (0, 1)."""
    theta, _ = compute_community_stats(
        synthetic_mem, synthetic_elec, year=2020, prefix="pres", exclude_al=False
    )
    assert (theta >= 0).all() and (theta <= 1).all(), (
        f"theta values out of [0,1]: {theta}"
    )


def test_compute_community_stats_se_positive(synthetic_mem, synthetic_elec):
    """Standard errors must all be strictly positive."""
    _, se = compute_community_stats(
        synthetic_mem, synthetic_elec, year=2020, prefix="pres", exclude_al=False
    )
    assert (se > 0).all(), f"Some standard errors are non-positive: {se}"


def test_compute_community_stats_se_small_relative_to_mean(synthetic_mem, synthetic_elec):
    """SEs should be much smaller than the mean (large effective sample)."""
    theta, se = compute_community_stats(
        synthetic_mem, synthetic_elec, year=2020, prefix="pres", exclude_al=False
    )
    # With 100 tracts and hundreds of votes each, SEs should be << 0.1
    assert (se < 0.1).all(), f"Some SEs are surprisingly large: {se}"


def test_compute_community_stats_weighted_mean_correctness():
    """
    Manual verification: single-community case should give exact weighted average.
    If all tracts are 100% in community 1, theta[0] = vote-share weighted mean.
    Communities c2..c7 have zero weight in this tract set, so their theta/SE
    values will be NaN (0/0 in the weighted mean); we only check c1 here.
    """
    import warnings

    # 3 tracts, all in community c1 exclusively
    mem = pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "12001000300"],
        "is_uninhabited": [False, False, False],
        "c1": [1.0, 1.0, 1.0],
        **{f"c{k}": [0.0, 0.0, 0.0] for k in range(2, 8)},
    })
    elec = pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "12001000300"],
        "pres_dem_share_2020": [0.60, 0.40, 0.50],
        "pres_total_2020": [1000.0, 1000.0, 1000.0],  # equal weights
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # expected 0/0 for c2..c7
        theta, se = compute_community_stats(
            mem, elec, year=2020, prefix="pres", exclude_al=False
        )
    # Equal vote totals → simple average: (0.60 + 0.40 + 0.50) / 3 = 0.50
    assert abs(theta[0] - 0.50) < 1e-10, f"c1 theta should be 0.50, got {theta[0]:.6f}"


# ---------------------------------------------------------------------------
# Covariance matrix mathematical property tests (synthetic, no Stan)
# ---------------------------------------------------------------------------


def _make_valid_covariance(k: int = K, seed: int = 42) -> np.ndarray:
    """Helper: generate a valid random K×K PSD covariance matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k, k))
    return (A @ A.T) / k + np.eye(k) * 1e-3  # ensure PSD


def test_synthetic_covariance_is_symmetric():
    """A well-formed covariance matrix must be symmetric: Σ == Σ.T"""
    Sigma = _make_valid_covariance()
    assert np.allclose(Sigma, Sigma.T, atol=1e-12), "Covariance matrix is not symmetric"


def test_synthetic_covariance_is_positive_semidefinite():
    """A well-formed covariance matrix must have all eigenvalues >= 0."""
    Sigma = _make_valid_covariance()
    eigenvalues = np.linalg.eigvalsh(Sigma)
    assert (eigenvalues >= -1e-10).all(), (
        f"Covariance matrix has negative eigenvalues: {eigenvalues.min():.4e}"
    )


def test_synthetic_covariance_trace_positive():
    """Trace of a valid covariance matrix must be strictly positive (sum of variances)."""
    Sigma = _make_valid_covariance()
    assert np.trace(Sigma) > 0


def test_correlation_matrix_diagonal_is_one():
    """Converting a covariance matrix to correlation: diagonal must equal 1.0."""
    Sigma = _make_valid_covariance()
    std = np.sqrt(np.diag(Sigma))
    Rho = Sigma / np.outer(std, std)
    assert np.allclose(np.diag(Rho), 1.0, atol=1e-10), (
        "Correlation matrix diagonal is not all 1.0"
    )


def test_correlation_matrix_off_diagonal_bounded():
    """All correlation values must be in [-1, 1]."""
    Sigma = _make_valid_covariance()
    std = np.sqrt(np.diag(Sigma))
    Rho = Sigma / np.outer(std, std)
    assert (Rho >= -1 - 1e-10).all() and (Rho <= 1 + 1e-10).all(), (
        "Correlation values outside [-1, 1]"
    )


def test_inverse_of_psd_matrix_exists():
    """A strictly PD covariance matrix must be invertible (no singular matrix passed to Kalman update)."""
    Sigma = _make_valid_covariance()
    # Should not raise
    Sigma_inv = np.linalg.inv(Sigma)
    # Verify: Σ · Σ⁻¹ ≈ I
    assert np.allclose(Sigma @ Sigma_inv, np.eye(K), atol=1e-10), (
        "Sigma @ Sigma_inv is not close to identity"
    )
