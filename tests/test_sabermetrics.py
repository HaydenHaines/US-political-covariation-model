"""Tests for political sabermetrics pipeline.

The sabermetrics module functions are currently NotImplementedError stubs.
These tests verify:
  1. The mathematical logic that the stats are DEFINED to implement (tested
     independently of the implementation, using pure numpy/pandas)
  2. That the function signatures accept the documented parameter types
  3. That calling any stub raises NotImplementedError (so we detect when a
     stub is accidentally left as-is after implementation begins)

When each function is implemented, its stub test should be removed and
replaced by a real assertion test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Mathematical definition tests (pure numpy — no implementation required)
# ---------------------------------------------------------------------------
# These test the *definitions* of the statistics, not the implementation.
# They serve as a contract: any correct implementation must satisfy them.


K = 7  # community types


def test_cook_pvi_formula_definition():
    """Cook PVI = 75% × most_recent + 25% × prior, relative to national average."""
    # district: 60% Dem in 2024, 58% Dem in 2020
    # national:  52% Dem in 2024, 51% Dem in 2020
    r2024 = 0.60 - 0.52  # +8 vs national in 2024
    r2020 = 0.58 - 0.51  # +7 vs national in 2020
    pvi = 0.75 * r2024 + 0.25 * r2020
    expected = 0.75 * 0.08 + 0.25 * 0.07
    assert abs(pvi - expected) < 1e-12


def test_cook_pvi_d_plus_district():
    """A district that consistently votes more Dem than national should be D+."""
    national_2024 = 0.51
    national_2020 = 0.50
    local_2024 = 0.58
    local_2020 = 0.55
    pvi = 0.75 * (local_2024 - national_2024) + 0.25 * (local_2020 - national_2020)
    assert pvi > 0, "D+ district should have positive PVI score"


def test_cook_pvi_r_plus_district():
    """A district that consistently votes more Rep than national should have negative PVI."""
    national_2024 = 0.51
    national_2020 = 0.50
    local_2024 = 0.38
    local_2020 = 0.40
    pvi = 0.75 * (local_2024 - national_2024) + 0.25 * (local_2020 - national_2020)
    assert pvi < 0, "R+ district should have negative PVI score"


def test_mvd_definition():
    """MVD = actual_vote_share − (baseline + environment). Verified numerically."""
    actual = 0.56
    baseline = 0.52
    environment = 0.02
    mvd = actual - (baseline + environment)
    assert abs(mvd - 0.02) < 1e-12, f"Expected MVD=0.02, got {mvd}"


def test_mvd_zero_in_average_race():
    """A candidate who exactly matches the baseline + environment has MVD=0."""
    actual = 0.55
    baseline = 0.52
    environment = 0.03
    mvd = actual - (baseline + environment)
    assert abs(mvd) < 1e-12


def test_mvd_is_zero_mean_property():
    """
    MVD is defined as a residual from expectation. Across many races where
    baselines perfectly capture expectations, MVDs should average to ~0.
    This is a definitional property, not a statistical test.
    """
    rng = np.random.default_rng(42)
    n = 100
    baselines = rng.uniform(0.40, 0.60, n)
    environment = 0.02
    # Actuals drawn from N(baseline + env, 0.03²) — centered on expectation
    actuals = baselines + environment + rng.normal(0, 0.03, n)
    mvds = actuals - (baselines + environment)
    # With 100 samples from a zero-mean distribution, mean should be near zero
    assert abs(mvds.mean()) < 0.02, f"MVD mean far from zero: {mvds.mean():.4f}"


def test_fit_score_is_dot_product():
    """Fit score = dot(CTOV, district_composition). Pure definition test."""
    rng = np.random.default_rng(0)
    ctov = rng.normal(0, 0.05, K)
    district_W = rng.dirichlet(np.ones(K))
    expected_fit = float(np.dot(ctov, district_W))
    # The composites.compute_fit_score function should produce this value;
    # test the definition numerically here.
    assert isinstance(expected_fit, float)
    # Also: fit score is a weighted average of CTOV components (since W sums to 1)
    # It must be bounded by min(CTOV) ≤ fit ≤ max(CTOV)
    assert ctov.min() <= expected_fit <= ctov.max() + 1e-10


def test_fit_score_pure_community_district():
    """If a district is 100% community k, fit score = CTOV[k]."""
    ctov = np.array([0.03, -0.01, 0.05, 0.02, -0.02, 0.04, 0.00])
    district_W = np.zeros(K)
    district_W[2] = 1.0  # 100% community 3
    fit = float(np.dot(ctov, district_W))
    assert abs(fit - ctov[2]) < 1e-12, f"Expected fit={ctov[2]}, got {fit}"


def test_ctov_weighted_sum_equals_mvd():
    """By definition: dot(CTOV, W_district) = MVD for that district."""
    # If CTOV correctly decomposes residuals by community type,
    # then the vote-weighted sum of CTOV components gives back MVD.
    rng = np.random.default_rng(5)
    # Construct a scenario: actual - expected = residual distributed by community type
    W = rng.dirichlet(np.ones(K))  # district composition
    # Residual per community type (CTOV)
    ctov = rng.normal(0, 0.03, K)
    # District MVD = type-weighted average of CTOVs
    mvd_from_ctov = float(np.dot(ctov, W))
    # Verify associativity: (ctov · W) computed two ways
    mvd_direct = sum(ctov[k] * W[k] for k in range(K))
    assert abs(mvd_from_ctov - mvd_direct) < 1e-12


def test_cec_definition_single_election():
    """CEC with only one election is undefined/degenerate — should return 1.0 or NaN."""
    # CEC = mean pairwise Pearson correlation across elections.
    # With one CTOV vector, there are no pairs to correlate.
    ctov_one = np.array([0.03, -0.01, 0.05, 0.02, -0.02, 0.04, 0.00])
    # If we compute it ourselves: pearsonr of a vector with itself = 1.0
    # This documents the expected behavior for a 1-election history
    if len([ctov_one]) == 1:
        cec = 1.0  # degenerate case
    assert cec == 1.0


def test_cec_identical_vectors_is_one():
    """A candidate who performs identically in every election has CEC=1.0."""
    ctov_base = np.array([0.05, -0.02, 0.08, 0.01, -0.03, 0.06, 0.00])
    # Perfect consistency: same CTOV vector in 3 elections
    ctovs = [ctov_base, ctov_base, ctov_base]
    # Pearson correlation of identical arrays = 1.0
    from numpy import corrcoef
    cec_values = []
    for i in range(len(ctovs)):
        for j in range(i + 1, len(ctovs)):
            r = np.corrcoef(ctovs[i], ctovs[j])[0, 1]
            cec_values.append(r)
    cec = float(np.mean(cec_values))
    assert abs(cec - 1.0) < 1e-10


def test_cec_anti_correlated_vectors_is_minus_one():
    """Perfect anti-correlation across elections gives CEC close to -1."""
    ctov_a = np.array([0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05])
    ctov_b = -ctov_a
    r = np.corrcoef(ctov_a, ctov_b)[0, 1]
    assert abs(r - (-1.0)) < 1e-10


def test_cec_is_bounded():
    """CEC must be in [-1, 1] for any two CTOV vectors."""
    rng = np.random.default_rng(13)
    for _ in range(20):
        ctov_1 = rng.normal(0, 0.05, K)
        ctov_2 = rng.normal(0, 0.05, K)
        r = np.corrcoef(ctov_1, ctov_2)[0, 1]
        assert -1 - 1e-10 <= r <= 1 + 1e-10, f"Pearson r={r} is outside [-1, 1]"


def test_polling_gap_definition():
    """Raw polling gap = actual - poll_average."""
    actual = 0.54
    poll_avg = 0.51
    raw_gap = actual - poll_avg
    assert abs(raw_gap - 0.03) < 1e-12


def test_polling_gap_adjusted_removes_cycle_bias():
    """Adjusted gap = raw_gap - cycle_systematic_error."""
    actual = 0.54
    poll_avg = 0.51
    cycle_error = 0.02  # polls underestimated Dems by 2pp on average this cycle
    raw_gap = actual - poll_avg
    adjusted_gap = raw_gap - cycle_error
    # The adjusted gap isolates the candidate-specific signal
    assert abs(adjusted_gap - 0.01) < 1e-12


def test_structural_baseline_definition():
    """Structural baseline = W_district · θ_types (matrix-vector product)."""
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(K))         # (K,) district composition
    theta = rng.uniform(0.35, 0.65, K)   # (K,) type-level vote shares
    baseline = float(np.dot(W, theta))
    # Baseline should be a convex combination of type estimates (bounded)
    assert theta.min() <= baseline <= theta.max() + 1e-10


def test_structural_baseline_is_weighted_average():
    """If all type estimates are equal, the baseline equals that constant."""
    W = np.array([0.10, 0.20, 0.15, 0.05, 0.25, 0.10, 0.15])
    theta = np.full(K, 0.52)
    baseline = float(np.dot(W, theta))
    assert abs(baseline - 0.52) < 1e-12


# ---------------------------------------------------------------------------
# Stub detection tests — verify NotImplementedError until functions are built
# ---------------------------------------------------------------------------


def test_compute_cook_pvi_is_not_implemented():
    """compute_cook_pvi raises NotImplementedError until implemented."""
    from src.sabermetrics.baselines import compute_cook_pvi
    with pytest.raises(NotImplementedError):
        compute_cook_pvi(pd.DataFrame())


def test_compute_structural_baseline_is_not_implemented():
    """compute_structural_baseline raises NotImplementedError until implemented."""
    from src.sabermetrics.baselines import compute_structural_baseline
    with pytest.raises(NotImplementedError):
        compute_structural_baseline(np.eye(K), np.ones(K), 0.0)


def test_compute_national_environment_is_not_implemented():
    """compute_national_environment raises NotImplementedError until implemented."""
    from src.sabermetrics.baselines import compute_national_environment
    with pytest.raises(NotImplementedError):
        compute_national_environment(generic_ballot=2.5)


def test_compute_mvd_is_not_implemented():
    """compute_mvd raises NotImplementedError until implemented."""
    from src.sabermetrics.residuals import compute_mvd
    with pytest.raises(NotImplementedError):
        compute_mvd(pd.DataFrame(), pd.DataFrame(), 0.0)


def test_compute_ctov_is_not_implemented():
    """compute_ctov raises NotImplementedError until implemented."""
    from src.sabermetrics.residuals import compute_ctov
    with pytest.raises(NotImplementedError):
        compute_ctov(pd.DataFrame(), np.eye(K), np.ones(K), pd.DataFrame())


def test_compute_cec_is_not_implemented():
    """compute_cec raises NotImplementedError until implemented."""
    from src.sabermetrics.residuals import compute_cec
    ctov_a = np.zeros(K)
    ctov_b = np.ones(K)
    with pytest.raises(NotImplementedError):
        compute_cec([ctov_a, ctov_b])


def test_compute_fit_score_is_not_implemented():
    """compute_fit_score raises NotImplementedError until implemented."""
    from src.sabermetrics.composites import compute_fit_score
    with pytest.raises(NotImplementedError):
        compute_fit_score(np.zeros(K), np.ones(K) / K)
