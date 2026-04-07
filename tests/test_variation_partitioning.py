"""Tests for scripts/variation_partitioning.py

Tests cover:
  - R² computation correctness (perfect, zero, negative, degenerate)
  - Variance partition math (four components sum to 1.0)
  - Edge cases for individual models
  - Integration smoke test on real data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.variation_partitioning import (
    compute_r2,
    partition_variance,
    predict_combined,
    predict_demographics_only,
    predict_types_only,
    run_partitioning,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def simple_data(rng):
    """100 counties, 6 training dims, 1 holdout dim, J=5 types, P=4 demo features.

    The holdout is constructed as a linear combination of types + noise
    so the types-only model should achieve good R².
    """
    N, D_train, J, P = 100, 6, 5, 4

    # Type centroids in training space
    centroids = rng.standard_normal((J, D_train)) * 2.0

    # Hard type assignments (20 counties each)
    labels = np.repeat(np.arange(J), N // J)

    # Shift matrix (training + holdout)
    training = centroids[labels] + rng.standard_normal((N, D_train)) * 0.3

    # Holdout = a linear function of the true type index + small noise
    type_signal = np.array([0.5, 0.2, -0.3, -0.1, 0.4])
    holdout = type_signal[labels] + rng.standard_normal(N) * 0.1

    shift_matrix = np.hstack([training, holdout.reshape(-1, 1)])
    training_cols = list(range(D_train))
    holdout_cols = [D_train]

    # Soft membership: strong weight on true type
    scores = np.zeros((N, J))
    for i, lbl in enumerate(labels):
        scores[i, lbl] = 5.0
        # small weight on other types
        for j in range(J):
            if j != lbl:
                scores[i, j] = 0.1

    # Normalize rows to sum to 1
    scores = scores / scores.sum(axis=1, keepdims=True)

    # Demographics weakly correlated with types
    demo = rng.standard_normal((N, P))
    for i, lbl in enumerate(labels):
        demo[i, 0] += lbl * 0.5  # one demographic correlated with type

    return {
        "shift_matrix": shift_matrix,
        "training_cols": training_cols,
        "holdout_cols": holdout_cols,
        "scores": scores,
        "demo_features": demo,
    }


@pytest.fixture(scope="module")
def perfect_data():
    """Perfect types: holdout is EXACTLY the type mean (no noise).

    Types-only model should achieve R² very close to 1.0.
    """
    N, D_train, J = 60, 4, 3

    # 20 counties per type, perfectly separated
    labels = np.repeat(np.arange(J), N // J)
    type_holdout_values = np.array([1.0, -1.0, 0.0])

    training = np.zeros((N, D_train))
    holdout = type_holdout_values[labels].reshape(-1, 1)
    shift_matrix = np.hstack([training, holdout])

    # Hard scores
    scores = np.zeros((N, J))
    for i, lbl in enumerate(labels):
        scores[i, lbl] = 1.0

    demo = np.random.default_rng(1).standard_normal((N, 2))

    return {
        "shift_matrix": shift_matrix,
        "training_cols": list(range(D_train)),
        "holdout_cols": [D_train],
        "scores": scores,
        "demo_features": demo,
    }


@pytest.fixture(scope="module")
def zero_signal_data(rng):
    """Pure noise: no type structure, no demographic signal.

    R² for all models should be near zero.
    """
    N, D_train, J, P = 80, 4, 4, 3
    shift_matrix = rng.standard_normal((N, D_train + 1))
    scores = np.ones((N, J)) / J  # uniform soft membership
    demo = rng.standard_normal((N, P))

    return {
        "shift_matrix": shift_matrix,
        "training_cols": list(range(D_train)),
        "holdout_cols": [D_train],
        "scores": scores,
        "demo_features": demo,
    }


# ── Tests: compute_r2 ─────────────────────────────────────────────────────────


class TestComputeR2:
    def test_perfect_prediction(self):
        """R² = 1.0 when predicted equals actual."""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_r2(actual, actual.copy()) == pytest.approx(1.0, abs=1e-10)

    def test_mean_prediction(self):
        """R² = 0.0 when predicted is the constant mean."""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.full_like(actual, actual.mean())
        assert compute_r2(actual, predicted) == pytest.approx(0.0, abs=1e-10)

    def test_negative_r2(self):
        """R² < 0 when predictions are worse than the mean."""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([4.0, 3.0, 2.0, 1.0])  # reversed
        assert compute_r2(actual, predicted) < 0.0

    def test_known_r2(self):
        """R² matches manual calculation."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 2.0, 2.5])
        ss_res = (0.5**2 + 0.0**2 + 0.5**2)
        ss_tot = (1.0 + 0.0 + 1.0)
        expected = 1.0 - ss_res / ss_tot
        assert compute_r2(actual, predicted) == pytest.approx(expected, abs=1e-8)

    def test_degenerate_zero_variance(self):
        """R² = 0.0 when actual is constant (SS_tot = 0)."""
        actual = np.array([3.0, 3.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert compute_r2(actual, predicted) == 0.0

    def test_returns_float(self):
        """Return type is always float."""
        actual = np.array([1.0, 2.0])
        predicted = np.array([1.1, 1.9])
        result = compute_r2(actual, predicted)
        assert isinstance(result, float)


# ── Tests: partition_variance ─────────────────────────────────────────────────


class TestPartitionVariance:
    def test_partition_sums_to_one(self):
        """unique_types + unique_demo + shared + residual == 1.0."""
        result = partition_variance(r2_types=0.6, r2_demo=0.5, r2_combined=0.7)
        total = (result["unique_types"] + result["unique_demographics"]
                 + result["shared"] + result["residual"])
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_partition_sums_to_one_varies(self):
        """Partition sums to 1.0 across many random R² triplets."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            r2_t = float(rng.uniform(0, 0.5))
            r2_d = float(rng.uniform(0, 0.5))
            # combined must be >= max(r2_t, r2_d) for the partition to be sensible;
            # but mathematically the sum still holds even if it isn't
            r2_c = float(rng.uniform(max(r2_t, r2_d), min(r2_t + r2_d, 0.95)))
            result = partition_variance(r2_t, r2_d, r2_c)
            total = (result["unique_types"] + result["unique_demographics"]
                     + result["shared"] + result["residual"])
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_unique_types_formula(self):
        """unique_types = R2(combined) - R2(demographics)."""
        result = partition_variance(r2_types=0.6, r2_demo=0.4, r2_combined=0.75)
        assert result["unique_types"] == pytest.approx(0.75 - 0.4, abs=1e-10)

    def test_unique_demo_formula(self):
        """unique_demographics = R2(combined) - R2(types)."""
        result = partition_variance(r2_types=0.6, r2_demo=0.4, r2_combined=0.75)
        assert result["unique_demographics"] == pytest.approx(0.75 - 0.6, abs=1e-10)

    def test_shared_formula(self):
        """shared = R2(types) + R2(demographics) - R2(combined)."""
        result = partition_variance(r2_types=0.6, r2_demo=0.4, r2_combined=0.75)
        assert result["shared"] == pytest.approx(0.6 + 0.4 - 0.75, abs=1e-10)

    def test_residual_formula(self):
        """residual = 1.0 - R2(combined)."""
        result = partition_variance(r2_types=0.6, r2_demo=0.4, r2_combined=0.75)
        assert result["residual"] == pytest.approx(0.25, abs=1e-10)

    def test_zero_types_all_demo(self):
        """When types R² = 0, unique_types = R2_combined - R2_demo."""
        result = partition_variance(r2_types=0.0, r2_demo=0.5, r2_combined=0.5)
        assert result["unique_types"] == pytest.approx(0.0, abs=1e-10)
        assert result["shared"] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_combined(self):
        """With R2_combined=1.0, residual=0."""
        result = partition_variance(r2_types=0.8, r2_demo=0.7, r2_combined=1.0)
        assert result["residual"] == pytest.approx(0.0, abs=1e-10)
        total = (result["unique_types"] + result["unique_demographics"]
                 + result["shared"] + result["residual"])
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_output_keys(self):
        """Return dict has all required keys."""
        result = partition_variance(0.5, 0.4, 0.6)
        for key in ["r2_types", "r2_demographics", "r2_combined",
                    "unique_types", "unique_demographics", "shared", "residual", "total"]:
            assert key in result

    def test_total_field_correct(self):
        """The 'total' key in the result matches the sum of the four components."""
        result = partition_variance(0.55, 0.45, 0.65)
        expected_total = (result["unique_types"] + result["unique_demographics"]
                          + result["shared"] + result["residual"])
        assert result["total"] == pytest.approx(expected_total, abs=1e-10)


# ── Tests: predict_types_only ─────────────────────────────────────────────────


class TestPredictTypesOnly:
    def test_output_shape(self, simple_data):
        """Output shape is (N, len(holdout_cols))."""
        d = simple_data
        pred = predict_types_only(
            d["scores"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        N = d["shift_matrix"].shape[0]
        assert pred.shape == (N, len(d["holdout_cols"]))

    def test_perfect_types_high_r2(self, perfect_data):
        """R² approaches 1.0 for perfectly separated types."""
        d = perfect_data
        pred = predict_types_only(
            d["scores"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        actual = d["shift_matrix"][:, d["holdout_cols"][0]]
        r2 = compute_r2(actual, pred[:, 0])
        assert r2 > 0.9, f"Expected R² > 0.9 for perfect types, got {r2:.3f}"

    def test_structured_data_positive_r2(self, simple_data):
        """R² > 0.5 on data where types have clear signal."""
        d = simple_data
        pred = predict_types_only(
            d["scores"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        actual = d["shift_matrix"][:, d["holdout_cols"][0]]
        r2 = compute_r2(actual, pred[:, 0])
        assert r2 > 0.5, f"Expected R² > 0.5 on structured data, got {r2:.3f}"


# ── Tests: predict_demographics_only ─────────────────────────────────────────


class TestPredictDemographicsOnly:
    def test_output_shape(self, simple_data):
        """Output shape is (N, len(holdout_cols))."""
        d = simple_data
        pred = predict_demographics_only(
            d["demo_features"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        N = d["shift_matrix"].shape[0]
        assert pred.shape == (N, len(d["holdout_cols"]))

    def test_returns_finite_values(self, simple_data):
        """All predicted values are finite (no NaN or inf)."""
        d = simple_data
        pred = predict_demographics_only(
            d["demo_features"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        assert np.all(np.isfinite(pred))

    def test_zero_signal_low_r2(self, zero_signal_data):
        """On pure noise, demographics R² should be near zero or weakly positive."""
        d = zero_signal_data
        pred = predict_demographics_only(
            d["demo_features"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        actual = d["shift_matrix"][:, d["holdout_cols"][0]]
        r2 = compute_r2(actual, pred[:, 0])
        # Ridge will over-fit even noise, but should stay well below 0.5
        assert r2 < 0.5, f"Expected low R² on noise, got {r2:.3f}"


# ── Tests: predict_combined ───────────────────────────────────────────────────


class TestPredictCombined:
    def test_output_shape(self, simple_data):
        """Output shape is (N, len(holdout_cols))."""
        d = simple_data
        pred = predict_combined(
            d["scores"], d["demo_features"],
            d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        N = d["shift_matrix"].shape[0]
        assert pred.shape == (N, len(d["holdout_cols"]))

    def test_combined_r2_geq_types(self, simple_data):
        """Combined R² should be >= types-only R² (more features = at least as good)."""
        d = simple_data
        pred_t = predict_types_only(
            d["scores"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        pred_c = predict_combined(
            d["scores"], d["demo_features"],
            d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        actual = d["shift_matrix"][:, d["holdout_cols"][0]]
        r2_t = compute_r2(actual, pred_t[:, 0])
        r2_c = compute_r2(actual, pred_c[:, 0])
        # Ridge may fit slightly better with more features; allow small tolerance
        assert r2_c >= r2_t - 0.05, (
            f"Combined R² ({r2_c:.3f}) should be >= types R² ({r2_t:.3f})"
        )

    def test_combined_r2_geq_demo(self, simple_data):
        """Combined R² should be >= demographics-only R²."""
        d = simple_data
        pred_d = predict_demographics_only(
            d["demo_features"], d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        pred_c = predict_combined(
            d["scores"], d["demo_features"],
            d["shift_matrix"], d["training_cols"], d["holdout_cols"]
        )
        actual = d["shift_matrix"][:, d["holdout_cols"][0]]
        r2_d = compute_r2(actual, pred_d[:, 0])
        r2_c = compute_r2(actual, pred_c[:, 0])
        assert r2_c >= r2_d - 0.05, (
            f"Combined R² ({r2_c:.3f}) should be >= demographics R² ({r2_d:.3f})"
        )


# ── Integration test on real data ─────────────────────────────────────────────


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "data" / "shifts" / "county_shifts_multiyear.parquet").exists(),
    reason="county_shifts_multiyear.parquet not available (CI environment)",
)
class TestRunPartitioning:
    @pytest.fixture(scope="class")
    def results(self):
        """Run full partitioning on real data (loaded once for the class)."""
        from scripts.variation_partitioning import load_data
        data = load_data()
        return run_partitioning(data, verbose=False)

    def test_returns_expected_keys(self, results):
        """Results dict has all expected top-level keys."""
        for key in ["per_dimension", "aggregate", "n_counties",
                    "n_training_dims", "n_holdout_dims", "n_types", "n_demo_features"]:
            assert key in results

    def test_n_counties_correct(self, results):
        """Should have 3154 national counties after Phase D retrain."""
        assert results["n_counties"] == 3154

    def test_n_holdout_dims_correct(self, results):
        """3 holdout dimensions (pres D-shift, R-shift, turnout)."""
        assert results["n_holdout_dims"] == 3

    def test_per_dimension_length(self, results):
        """One result per holdout dimension."""
        assert len(results["per_dimension"]) == 3

    def test_types_r2_above_threshold(self, results):
        """Types-only R² should be positive on real data (national: ~0.2, 3-state: ~0.68)."""
        agg = results["aggregate"]
        assert agg["r2_types"] > 0.1, (
            f"Types R² = {agg['r2_types']:.3f} is below minimum threshold"
        )

    def test_combined_r2_geq_types(self, results):
        """Combined model R² >= types-only (adding features doesn't hurt)."""
        agg = results["aggregate"]
        assert agg["r2_combined"] >= agg["r2_types"] - 0.05

    def test_combined_r2_geq_demographics(self, results):
        """Combined model R² >= demographics-only."""
        agg = results["aggregate"]
        assert agg["r2_combined"] >= agg["r2_demographics"] - 0.05

    def test_partition_sums_to_one(self, results):
        """Aggregate partition sums to 1.0."""
        agg = results["aggregate"]
        total = (agg["unique_types"] + agg["unique_demographics"]
                 + agg["shared"] + agg["residual"])
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_types_have_positive_unique_contribution(self, results):
        """Types should have positive unique R² contribution (adds value beyond demographics)."""
        agg = results["aggregate"]
        assert agg["unique_types"] > 0, (
            f"unique_types = {agg['unique_types']:.4f}; expected positive"
        )

    def test_r2_values_in_range(self, results):
        """R² values for all three models are between -0.5 and 1.0."""
        agg = results["aggregate"]
        for key in ["r2_types", "r2_demographics", "r2_combined"]:
            val = agg[key]
            assert -0.5 <= val <= 1.0, f"{key} = {val:.4f} out of expected range"

    def test_per_dimension_partition_sums(self, results):
        """Each per-dimension partition also sums to 1.0."""
        for d in results["per_dimension"]:
            total = (d["unique_types"] + d["unique_demographics"]
                     + d["shared"] + d["residual"])
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Dimension {d['dimension']}: partition sum = {total:.6f}"
            )

    def test_shared_plus_unique_leq_combined(self, results):
        """Shared + unique_types = R2(types); shared + unique_demo = R2(demo)."""
        agg = results["aggregate"]
        # By definition: shared = R2_t + R2_d - R2_c
        # So: shared + unique_types = R2_t + R2_d - R2_c + R2_c - R2_d = R2_t
        assert (agg["shared"] + agg["unique_types"]) == pytest.approx(
            agg["r2_types"], abs=1e-6
        )
        assert (agg["shared"] + agg["unique_demographics"]) == pytest.approx(
            agg["r2_demographics"], abs=1e-6
        )
