"""Tests for scripts/experiment_soft_membership.py.

Tests cover the core mathematical functions:
  - temperature_soft_membership: exponent scaling and row-normalization
  - county_predictions: weighted average of type priors
  - compute_metrics: MAE, RMSE, Pearson r, bias

These tests use synthetic data only and do not touch the filesystem.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the experiment module (lives in scripts/, not a package)
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "experiment_soft_membership.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_soft_membership", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# temperature_soft_membership
# ---------------------------------------------------------------------------


class TestTemperatureSoftMembership:
    def test_rows_sum_to_one(self, mod):
        rng = np.random.RandomState(0)
        dists = np.abs(rng.randn(50, 20))
        for T in [1.0, 2.0, 5.0, 10.0, 999.0]:
            scores = mod.temperature_soft_membership(dists, T)
            np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-10,
                                       err_msg=f"Row sums not 1 at T={T}")

    def test_scores_non_negative(self, mod):
        rng = np.random.RandomState(1)
        dists = np.abs(rng.randn(30, 20))
        scores = mod.temperature_soft_membership(dists, T=3.0)
        assert (scores >= 0).all(), "Scores should be non-negative"

    def test_t1_matches_baseline(self, mod):
        """T=1.0 should reproduce the original inverse-distance formula exactly."""
        rng = np.random.RandomState(2)
        dists = np.abs(rng.randn(10, 5)) + 0.01  # ensure no zeros
        scores_t1 = mod.temperature_soft_membership(dists, T=1.0)

        # Compute baseline manually
        inv = 1.0 / (dists + 1e-10)
        expected = inv / inv.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(scores_t1, expected, atol=1e-12)

    def test_high_temperature_approaches_hard_assignment(self, mod):
        """At T=999, the nearest centroid should capture nearly all weight."""
        # County 0 is very close to centroid 0, far from others
        dists = np.array([[0.001, 2.0, 3.0, 4.0, 5.0]])
        scores = mod.temperature_soft_membership(dists, T=999.0)
        # Column 0 should be near 1.0
        assert scores[0, 0] > 0.999, f"Expected near-1 weight on nearest centroid, got {scores[0,0]:.6f}"

    def test_uniform_distances_give_uniform_scores(self, mod):
        """When all distances are equal, membership should be uniform."""
        dists = np.ones((5, 4))  # All distances = 1
        for T in [1.0, 3.0, 10.0]:
            scores = mod.temperature_soft_membership(dists, T)
            expected = np.full((5, 4), 0.25)
            np.testing.assert_allclose(scores, expected, atol=1e-10,
                                       err_msg=f"Uniform distances should give uniform scores at T={T}")

    def test_sharpening_increases_dominant_weight(self, mod):
        """Higher T should increase the weight on the nearest centroid."""
        dists = np.array([[0.5, 1.5, 2.5, 3.5]])  # nearest = col 0
        w1 = mod.temperature_soft_membership(dists, T=1.0)[0, 0]
        w5 = mod.temperature_soft_membership(dists, T=5.0)[0, 0]
        w10 = mod.temperature_soft_membership(dists, T=10.0)[0, 0]
        assert w1 < w5 < w10, f"Higher T should give more weight to nearest: {w1:.4f} < {w5:.4f} < {w10:.4f}"

    def test_zero_distance_handled(self, mod):
        """A county exactly on a centroid (dist=0) should not cause NaN/Inf.

        At T=1 and T=5, log-space softmax handles dist=0 via eps.
        At T=999, the hard-assignment shortcut (argmax) is used directly.
        In all cases the result should be finite and row-sum to 1.
        """
        dists = np.array([[0.0, 1.0, 2.0]])
        for T in [1.0, 5.0, 999.0]:
            scores = mod.temperature_soft_membership(dists, T)
            assert np.isfinite(scores).all(), f"NaN/Inf at T={T} with zero distance"
            np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-10)
        # At T=999, argmax shortcut: column 0 (dist=0) should have all weight
        scores_hard = mod.temperature_soft_membership(dists, T=999.0)
        assert scores_hard[0, 0] == pytest.approx(1.0), (
            "At T=999, county on centroid 0 should have weight=1 on type 0"
        )

    def test_shape_preserved(self, mod):
        """Output shape should match input shape."""
        dists = np.abs(np.random.randn(17, 20)) + 0.1
        scores = mod.temperature_soft_membership(dists, T=2.0)
        assert scores.shape == dists.shape

    def test_t999_is_effectively_hard(self, mod):
        """At T=999, entropy of weight distribution should be near zero."""
        rng = np.random.RandomState(7)
        dists = np.abs(rng.randn(100, 20)) + 0.01
        scores = mod.temperature_soft_membership(dists, T=999.0)
        # For each county, max score should dominate
        max_scores = scores.max(axis=1)
        assert (max_scores > 0.99).all(), "At T=999, max score should be > 0.99"


# ---------------------------------------------------------------------------
# county_predictions
# ---------------------------------------------------------------------------


class TestCountyPredictions:
    def test_output_clipped_to_unit_interval(self, mod):
        """All predictions should be in [0, 1]."""
        scores = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        priors = np.array([0.2, 0.8])
        pred = mod.county_predictions(scores, priors)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_pure_type_assignment(self, mod):
        """A county fully in type j should get exactly prior_j."""
        priors = np.array([0.3, 0.5, 0.7, 0.4])
        J = len(priors)
        for j in range(J):
            scores = np.zeros((1, J))
            scores[0, j] = 1.0
            pred = mod.county_predictions(scores, priors)
            np.testing.assert_allclose(pred[0], priors[j], atol=1e-12)

    def test_equal_weights_give_mean_prior(self, mod):
        """Equal weights across all types should give the mean prior."""
        J = 5
        priors = np.array([0.2, 0.4, 0.6, 0.3, 0.5])
        scores = np.full((3, J), 1.0 / J)
        pred = mod.county_predictions(scores, priors)
        expected = priors.mean()
        np.testing.assert_allclose(pred, expected, atol=1e-12)

    def test_output_shape(self, mod):
        """Output should have shape (N,)."""
        N, J = 50, 20
        scores = np.random.dirichlet(np.ones(J), size=N)
        priors = np.random.uniform(0.2, 0.8, J)
        pred = mod.county_predictions(scores, priors)
        assert pred.shape == (N,)

    def test_high_dem_prior_gives_high_prediction(self, mod):
        """Counties in high-Dem types should predict higher than low-Dem types."""
        priors = np.array([0.15, 0.75])
        low_dem_scores = np.array([[0.95, 0.05]])
        high_dem_scores = np.array([[0.05, 0.95]])
        pred_low = mod.county_predictions(low_dem_scores, priors)
        pred_high = mod.county_predictions(high_dem_scores, priors)
        assert pred_high[0] > pred_low[0]


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_prediction_zero_mae(self, mod):
        actual = np.array([0.3, 0.5, 0.7])
        metrics = mod.compute_metrics(actual, actual.copy())
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["bias"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["pearson_r"] == pytest.approx(1.0, abs=1e-10)

    def test_bias_sign_convention(self, mod):
        """bias = mean(predicted - actual): over-predicting Dem should be positive."""
        actual = np.array([0.3, 0.4, 0.5])
        predicted = actual + 0.1  # model too high
        metrics = mod.compute_metrics(actual, predicted)
        assert metrics["bias"] == pytest.approx(0.1, abs=1e-10)

    def test_mae_is_mean_abs_error(self, mod):
        actual = np.array([0.2, 0.4, 0.6])
        predicted = np.array([0.3, 0.3, 0.7])  # errors: +0.1, -0.1, +0.1
        metrics = mod.compute_metrics(actual, predicted)
        assert metrics["mae"] == pytest.approx(0.1, abs=1e-10)

    def test_rmse_larger_than_mae_with_outlier(self, mod):
        actual = np.array([0.3, 0.3, 0.3])
        predicted = np.array([0.3, 0.3, 0.7])  # one big error
        metrics = mod.compute_metrics(actual, predicted)
        assert metrics["rmse"] > metrics["mae"]

    def test_pred_range_reported(self, mod):
        actual = np.array([0.2, 0.5, 0.8])
        predicted = np.array([0.35, 0.45, 0.55])
        metrics = mod.compute_metrics(actual, predicted)
        assert metrics["pred_min"] == pytest.approx(0.35)
        assert metrics["pred_max"] == pytest.approx(0.55)
        assert metrics["pred_range"] == pytest.approx(0.20, abs=1e-12)

    def test_all_expected_keys_present(self, mod):
        actual = np.array([0.3, 0.5, 0.7])
        predicted = np.array([0.32, 0.48, 0.68])
        metrics = mod.compute_metrics(actual, predicted)
        expected_keys = {"mae", "rmse", "pearson_r", "bias", "pred_min", "pred_max", "pred_range"}
        assert expected_keys.issubset(set(metrics.keys()))


# ---------------------------------------------------------------------------
# Integration: sweep monotonicity
# ---------------------------------------------------------------------------


class TestSweepMonotonicity:
    """Sanity checks on the temperature sweep logic using synthetic data."""

    def test_higher_temperature_sharpens_predictions(self, mod):
        """As T increases, the spread of predictions should widen (or stay equal)."""
        rng = np.random.RandomState(42)
        N, J = 100, 20
        # Heterogeneous distances — counties spread across the type space
        dists = np.abs(rng.randn(N, J)) + 0.5
        priors = rng.uniform(0.15, 0.85, J)

        spreads = []
        for T in [1.0, 2.0, 5.0, 10.0, 999.0]:
            scores = mod.temperature_soft_membership(dists, T)
            pred = mod.county_predictions(scores, priors)
            spreads.append(pred.max() - pred.min())

        # Spread should be non-decreasing with T
        for i in range(len(spreads) - 1):
            assert spreads[i] <= spreads[i + 1] + 1e-10, (
                f"Spread should not decrease: T={[1.0,2.0,5.0,10.0,999.0][i]} spread={spreads[i]:.4f}, "
                f"T={[1.0,2.0,5.0,10.0,999.0][i+1]} spread={spreads[i+1]:.4f}"
            )

    def test_soft_membership_sweep_returns_dataframe(self, mod):
        """Smoke test: run_sweep returns a DataFrame with expected columns and rows."""
        # We can't call run_sweep() (it reads real data), so test the loop logic
        # directly using synthetic data.
        rng = np.random.RandomState(99)
        N, J_ = 50, 20
        dists = np.abs(rng.randn(N, J_)) + 0.3
        priors = rng.uniform(0.2, 0.8, J_)
        actual = rng.uniform(0.1, 0.9, N)

        import pandas as pd
        rows = []
        temperatures = [1.0, 2.0, 5.0, 999.0]
        for T in temperatures:
            scores = mod.temperature_soft_membership(dists, T)
            pred = mod.county_predictions(scores, priors)
            m = mod.compute_metrics(actual, pred)
            m["temperature"] = T
            rows.append(m)

        df = pd.DataFrame(rows)
        assert len(df) == len(temperatures)
        assert "mae" in df.columns
        assert "temperature" in df.columns
        assert "pearson_r" in df.columns
