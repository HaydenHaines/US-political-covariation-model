"""Tests for core metric and prediction functions in scripts/calibration_analysis.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root so scripts/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibration_analysis import (
    build_type_priors_from_election,
    compute_bias,
    compute_mae,
    compute_metrics,
    compute_pearson_r,
    compute_rmse,
    prior_only_predictions,
)


# ---------------------------------------------------------------------------
# compute_mae
# ---------------------------------------------------------------------------


def test_mae_perfect_prediction():
    a = np.array([0.4, 0.5, 0.6])
    assert compute_mae(a, a) == pytest.approx(0.0)


def test_mae_constant_error():
    actual = np.array([0.4, 0.5, 0.6])
    predicted = actual + 0.1
    assert compute_mae(actual, predicted) == pytest.approx(0.1)


def test_mae_symmetric():
    a = np.array([0.3, 0.5, 0.7])
    b = np.array([0.4, 0.6, 0.8])
    assert compute_mae(a, b) == pytest.approx(compute_mae(b, a))


def test_mae_single_value():
    assert compute_mae(np.array([0.5]), np.array([0.3])) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------


def test_rmse_perfect_prediction():
    a = np.array([0.4, 0.5, 0.6])
    assert compute_rmse(a, a) == pytest.approx(0.0)


def test_rmse_constant_error():
    actual = np.array([0.4, 0.5, 0.6])
    predicted = actual + 0.1
    assert compute_rmse(actual, predicted) == pytest.approx(0.1)


def test_rmse_larger_than_mae_for_heterogeneous_errors():
    actual = np.array([0.4, 0.5, 0.6])
    predicted = np.array([0.3, 0.5, 0.9])  # mixed errors including large outlier
    assert compute_rmse(actual, predicted) >= compute_mae(actual, predicted)


def test_rmse_single_value():
    assert compute_rmse(np.array([0.5]), np.array([0.2])) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# compute_bias
# ---------------------------------------------------------------------------


def test_bias_perfect_prediction():
    a = np.array([0.4, 0.5, 0.6])
    assert compute_bias(a, a) == pytest.approx(0.0)


def test_bias_positive_overprediction():
    actual = np.array([0.4, 0.5, 0.6])
    predicted = actual + 0.05
    assert compute_bias(actual, predicted) == pytest.approx(0.05)


def test_bias_negative_underprediction():
    actual = np.array([0.4, 0.5, 0.6])
    predicted = actual - 0.03
    assert compute_bias(actual, predicted) == pytest.approx(-0.03)


def test_bias_sign_convention():
    """bias = predicted - actual; positive means over-prediction."""
    actual = np.array([0.5])
    predicted = np.array([0.6])
    assert compute_bias(actual, predicted) == pytest.approx(0.1)


def test_bias_cancellation():
    """Equal over and under prediction should cancel out."""
    actual = np.array([0.4, 0.6])
    predicted = np.array([0.5, 0.5])
    assert compute_bias(actual, predicted) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_pearson_r
# ---------------------------------------------------------------------------


def test_pearson_r_perfect_correlation():
    a = np.linspace(0.2, 0.8, 10)
    assert compute_pearson_r(a, a) == pytest.approx(1.0)


def test_pearson_r_perfect_anticorrelation():
    a = np.linspace(0.2, 0.8, 10)
    assert compute_pearson_r(a, 1 - a) == pytest.approx(-1.0)


def test_pearson_r_bounded():
    rng = np.random.RandomState(0)
    a = rng.uniform(0.2, 0.8, 50)
    b = rng.uniform(0.2, 0.8, 50)
    r = compute_pearson_r(a, b)
    assert -1.0 <= r <= 1.0


def test_pearson_r_too_few_points():
    assert np.isnan(compute_pearson_r(np.array([0.5]), np.array([0.6])))


def test_pearson_r_linear_relationship():
    a = np.array([0.2, 0.4, 0.6, 0.8])
    b = 0.5 * a + 0.1  # linear but not identity
    assert compute_pearson_r(a, b) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# compute_metrics (integration)
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_all_keys():
    a = np.array([0.4, 0.5, 0.6])
    b = np.array([0.45, 0.48, 0.62])
    m = compute_metrics(a, b)
    assert "mae" in m
    assert "rmse" in m
    assert "bias" in m
    assert "pearson_r" in m
    assert "n" in m


def test_compute_metrics_n_matches_input():
    a = np.linspace(0.3, 0.7, 15)
    m = compute_metrics(a, a)
    assert m["n"] == 15


def test_compute_metrics_perfect_prediction():
    a = np.linspace(0.2, 0.9, 20)
    m = compute_metrics(a, a)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["bias"] == pytest.approx(0.0)
    assert m["pearson_r"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# prior_only_predictions
# ---------------------------------------------------------------------------


def test_prior_only_predictions_shape():
    N, J = 10, 5
    rng = np.random.RandomState(1)
    scores = rng.dirichlet(np.ones(J), size=N)  # non-negative, sum to 1
    priors = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    pred = prior_only_predictions(scores, priors)
    assert pred.shape == (N,)


def test_prior_only_predictions_bounded():
    N, J = 50, 20
    rng = np.random.RandomState(2)
    scores = rng.dirichlet(np.ones(J), size=N)
    priors = rng.uniform(0.1, 0.9, J)
    pred = prior_only_predictions(scores, priors)
    assert (pred >= 0.0).all()
    assert (pred <= 1.0).all()


def test_prior_only_predictions_uniform_scores_equal_mean_prior():
    """If all types have equal weight, prediction = mean of type priors."""
    J = 4
    scores = np.full((1, J), 1.0 / J)
    priors = np.array([0.2, 0.4, 0.6, 0.8])
    pred = prior_only_predictions(scores, priors)
    assert pred[0] == pytest.approx(np.mean(priors))


def test_prior_only_predictions_pure_type_equals_its_prior():
    """A county fully assigned to type j should predict type j's prior."""
    J = 4
    for j in range(J):
        scores = np.zeros((1, J))
        scores[0, j] = 1.0
        priors = np.array([0.2, 0.4, 0.6, 0.8])
        pred = prior_only_predictions(scores, priors)
        assert pred[0] == pytest.approx(priors[j])


def test_prior_only_predictions_zero_weight_row_handled():
    """Zero-weight rows should not cause division by zero."""
    J = 4
    scores = np.zeros((1, J))  # all zero row
    priors = np.array([0.4, 0.5, 0.4, 0.5])
    pred = prior_only_predictions(scores, priors)
    assert np.isfinite(pred[0])
    assert 0.0 <= pred[0] <= 1.0


# ---------------------------------------------------------------------------
# build_type_priors_from_election
# ---------------------------------------------------------------------------


def test_build_type_priors_shape():
    N, J = 20, 5
    rng = np.random.RandomState(3)
    scores = rng.dirichlet(np.ones(J), size=N)
    actual = rng.uniform(0.3, 0.7, N)
    total = rng.randint(5000, 50000, N).astype(float)
    priors = build_type_priors_from_election(scores, actual, total, J)
    assert priors.shape == (J,)


def test_build_type_priors_bounded():
    """All estimated priors should be within the range of actual values."""
    N, J = 30, 4
    rng = np.random.RandomState(4)
    scores = rng.dirichlet(np.ones(J), size=N)
    actual = rng.uniform(0.2, 0.8, N)
    total = rng.randint(5000, 50000, N).astype(float)
    priors = build_type_priors_from_election(scores, actual, total, J)
    # Priors should be a weighted average, so bounded by min/max of actual
    assert (priors >= actual.min() - 1e-10).all()
    assert (priors <= actual.max() + 1e-10).all()


def test_build_type_priors_homogeneous_returns_input():
    """When all counties have the same actual dem share, priors should match."""
    N, J = 10, 3
    scores = np.full((N, J), 1.0 / J)
    actual = np.full(N, 0.55)
    total = np.ones(N) * 10000.0
    priors = build_type_priors_from_election(scores, actual, total, J)
    np.testing.assert_allclose(priors, 0.55, atol=1e-10)


def test_build_type_priors_population_weighting():
    """Counties with higher total votes should dominate the type prior estimate."""
    J = 2
    # Two counties: one small (100 votes) mostly type 0, one large (10000 votes) mostly type 1
    scores = np.array([[0.9, 0.1], [0.1, 0.9]])
    actual = np.array([0.8, 0.4])  # county 0 high D, county 1 low D
    total = np.array([100.0, 10000.0])  # county 1 dominates
    priors = build_type_priors_from_election(scores, actual, total, J)
    # Type 1 prior should be strongly pulled toward 0.4 (the large county)
    assert priors[1] == pytest.approx(0.4, abs=0.05)
