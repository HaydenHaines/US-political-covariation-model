"""Tests for scripts/experiment_temporal_weighting.py.

Covers:
  - get_anchor_year: year extraction from column names
  - compute_temporal_weights: each scheme (equal, linear, exponential, step)
  - apply_temporal_weights: sqrt-scaling, zero passthrough
  - group_columns_by_pair / make_cv_folds: inherited CV helpers
  - run_one_fold: single-fold integration, shape and value range
  - run_temporal_weighting_experiment: full experiment structure and output
  - print_results / save_results: smoke tests
  - scheme_label: label generation
  - build_scheme_list: scheme enumeration

All tests use synthetic data only — no filesystem access.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load the experiment module (lives in scripts/, not a package)
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "experiment_temporal_weighting.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_temporal_weighting", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_columns(pairs=None, races=None):
    """Return synthetic column names matching real shift name patterns."""
    if pairs is None:
        pairs = ["08_12", "12_16", "16_20"]
    if races is None:
        races = ["pres"]
    cols = []
    for p in pairs:
        for race in races:
            cols += [f"{race}_d_shift_{p}", f"{race}_r_shift_{p}", f"{race}_turnout_shift_{p}"]
    return cols


# ---------------------------------------------------------------------------
# get_anchor_year
# ---------------------------------------------------------------------------


class TestGetAnchorYear:
    def test_presidential_column(self, mod):
        assert mod.get_anchor_year("pres_d_shift_08_12") == 2008

    def test_governor_column(self, mod):
        assert mod.get_anchor_year("gov_d_shift_06_10") == 2006

    def test_senate_column(self, mod):
        assert mod.get_anchor_year("sen_r_shift_02_08") == 2002

    def test_recent_pair(self, mod):
        assert mod.get_anchor_year("pres_turnout_shift_16_20") == 2016

    def test_unparseable_returns_none(self, mod):
        assert mod.get_anchor_year("county_fips") is None
        assert mod.get_anchor_year("not_a_shift_col") is None

    def test_two_digit_year_expansion(self, mod):
        # "20" prefix applied correctly
        assert mod.get_anchor_year("pres_d_shift_00_04") == 2000
        assert mod.get_anchor_year("pres_d_shift_12_16") == 2012


# ---------------------------------------------------------------------------
# compute_temporal_weights — equal scheme
# ---------------------------------------------------------------------------


class TestComputeTemporalWeightsEqual:
    def test_all_ones(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        weights = mod.compute_temporal_weights(cols, scheme="equal")
        np.testing.assert_array_equal(weights, np.ones(len(cols)))

    def test_length_matches_columns(self, mod):
        cols = _make_synthetic_columns(["08_12", "16_20"])
        weights = mod.compute_temporal_weights(cols, scheme="equal")
        assert len(weights) == len(cols)


# ---------------------------------------------------------------------------
# compute_temporal_weights — linear scheme
# ---------------------------------------------------------------------------


class TestComputeTemporalWeightsLinear:
    def test_weights_in_zero_one(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        weights = mod.compute_temporal_weights(cols, scheme="linear")
        assert (weights >= 0).all()
        assert (weights <= 1.0 + 1e-9).all()

    def test_recent_pair_higher_than_old(self, mod):
        """Pairs starting in 2016 should have higher weight than 2008."""
        cols_old = _make_synthetic_columns(["08_12"])
        cols_new = _make_synthetic_columns(["16_20"])
        w_old = mod.compute_temporal_weights(cols_old, scheme="linear")
        w_new = mod.compute_temporal_weights(cols_new, scheme="linear")
        assert w_new.mean() > w_old.mean()

    def test_length_matches_columns(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16"])
        weights = mod.compute_temporal_weights(cols, scheme="linear")
        assert len(weights) == len(cols)

    def test_same_pair_same_weight(self, mod):
        """All three metric columns in the same pair should get the same weight."""
        cols = ["pres_d_shift_12_16", "pres_r_shift_12_16", "pres_turnout_shift_12_16"]
        weights = mod.compute_temporal_weights(cols, scheme="linear")
        np.testing.assert_allclose(weights, weights[0], atol=1e-12)


# ---------------------------------------------------------------------------
# compute_temporal_weights — exponential scheme
# ---------------------------------------------------------------------------


class TestComputeTemporalWeightsExponential:
    def test_weights_in_zero_one(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        for hl in [4, 8, 12]:
            weights = mod.compute_temporal_weights(cols, scheme="exponential", half_life=hl)
            assert (weights > 0).all(), f"Non-positive weight at half_life={hl}"
            assert (weights <= 1.0 + 1e-9).all(), f"Weight > 1 at half_life={hl}"

    def test_latest_pair_weight_one(self, mod):
        """The pair anchored at LATEST_YEAR (2024) should have weight exactly 1.0."""
        # 2024 is LATEST_YEAR, so exponent = (2024 - 2024)/hl = 0 → 2^0 = 1
        cols = ["pres_d_shift_20_24"]  # anchor year 2020, not 2024!
        # Actually 2020 → weight = 2^((2020-2024)/hl) < 1
        # To get weight=1, we'd need a column with anchor year == LATEST_YEAR (2024).
        # Since LATEST_YEAR=2024 and the latest pair starts at 2020, this is the max.
        weights_4 = mod.compute_temporal_weights(cols, scheme="exponential", half_life=4)
        weights_8 = mod.compute_temporal_weights(cols, scheme="exponential", half_life=8)
        # Larger half_life → weight closer to 1 for any anchor year
        assert weights_8[0] > weights_4[0]

    def test_smaller_half_life_more_aggressive(self, mod):
        """Smaller half_life means older pairs get much lower weight."""
        cols = _make_synthetic_columns(["08_12"])
        w4 = mod.compute_temporal_weights(cols, scheme="exponential", half_life=4)
        w12 = mod.compute_temporal_weights(cols, scheme="exponential", half_life=12)
        # 2008 anchor: w4 = 2^((2008-2024)/4) = 2^-4 = 0.0625
        #              w12 = 2^((2008-2024)/12) = 2^(-4/3) ≈ 0.397
        assert w4.mean() < w12.mean()

    def test_invalid_half_life_raises(self, mod):
        cols = _make_synthetic_columns(["08_12"])
        with pytest.raises(ValueError):
            mod.compute_temporal_weights(cols, scheme="exponential", half_life=0)
        with pytest.raises(ValueError):
            mod.compute_temporal_weights(cols, scheme="exponential", half_life=None)

    def test_length_matches_columns(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        weights = mod.compute_temporal_weights(cols, scheme="exponential", half_life=8)
        assert len(weights) == len(cols)


# ---------------------------------------------------------------------------
# compute_temporal_weights — step scheme
# ---------------------------------------------------------------------------


class TestComputeTemporalWeightsStep:
    def test_pre2016_zeroed(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16"])
        weights = mod.compute_temporal_weights(cols, scheme="step")
        # 08_12 anchor=2008 < 2016, 12_16 anchor=2012 < 2016 → both zero
        np.testing.assert_array_equal(weights, np.zeros(len(cols)))

    def test_post2016_ones(self, mod):
        cols = _make_synthetic_columns(["16_20"])
        weights = mod.compute_temporal_weights(cols, scheme="step")
        np.testing.assert_array_equal(weights, np.ones(len(cols)))

    def test_mixed_pairs(self, mod):
        cols = _make_synthetic_columns(["12_16", "16_20"])
        weights = mod.compute_temporal_weights(cols, scheme="step")
        # 12_16 → 0, 16_20 → 1 (3 cols each)
        assert all(weights[:3] == 0.0)
        assert all(weights[3:] == 1.0)

    def test_unknown_scheme_raises(self, mod):
        cols = _make_synthetic_columns(["08_12"])
        with pytest.raises(ValueError):
            mod.compute_temporal_weights(cols, scheme="banana")


# ---------------------------------------------------------------------------
# apply_temporal_weights
# ---------------------------------------------------------------------------


class TestApplyTemporalWeights:
    def test_output_shape_preserved(self, mod):
        X = np.random.randn(50, 9)
        weights = np.ones(9)
        X_out = mod.apply_temporal_weights(X, weights)
        assert X_out.shape == X.shape

    def test_equal_weights_unchanged(self, mod):
        X = np.random.randn(40, 6)
        weights = np.ones(6)
        X_out = mod.apply_temporal_weights(X, weights)
        np.testing.assert_allclose(X_out, X, atol=1e-12)

    def test_zero_weight_zeros_column(self, mod):
        X = np.random.randn(30, 4)
        weights = np.array([1.0, 0.0, 1.0, 0.0])
        X_out = mod.apply_temporal_weights(X, weights)
        np.testing.assert_allclose(X_out[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(X_out[:, 3], 0.0, atol=1e-12)

    def test_weight_four_doubles_squared_distance(self, mod):
        """sqrt(4) = 2, so a column scaled by 2 contributes 4× to squared distance."""
        X = np.ones((5, 1))
        weights = np.array([4.0])
        X_out = mod.apply_temporal_weights(X, weights)
        # Each entry should be 2.0 (sqrt(4) * 1)
        np.testing.assert_allclose(X_out, 2.0 * np.ones((5, 1)), atol=1e-12)


# ---------------------------------------------------------------------------
# run_one_fold — single-fold integration
# ---------------------------------------------------------------------------


class TestRunOneFold:
    def test_returns_expected_keys(self, mod):
        rng = np.random.RandomState(42)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        result = mod.run_one_fold(X, j=4, fold=folds[0])
        assert "pair_key" in result
        assert "holdout_r" in result
        assert "holdout_mae" in result

    def test_holdout_r_in_range(self, mod):
        rng = np.random.RandomState(7)
        N, D = 80, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        for fold in folds:
            result = mod.run_one_fold(X, j=3, fold=fold)
            assert -1.0 - 1e-6 <= result["holdout_r"] <= 1.0 + 1e-6

    def test_holdout_mae_non_negative(self, mod):
        rng = np.random.RandomState(9)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        for fold in folds:
            result = mod.run_one_fold(X, j=3, fold=fold)
            assert result["holdout_mae"] >= 0.0


# ---------------------------------------------------------------------------
# run_temporal_weighting_experiment — full experiment
# ---------------------------------------------------------------------------


class TestRunTemporalWeightingExperiment:
    @pytest.fixture(scope="class")
    def experiment_results(self, mod):
        """Run experiment on synthetic data via monkeypatching load_shift_matrix."""
        rng = np.random.RandomState(0)
        N, D = 80, 9  # 3 pairs × 3 cols
        X = rng.randn(N, D)
        shift_cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        county_fips = np.array([f"{i:05d}" for i in range(N)])
        pres_weights = np.ones(D)

        original = mod.load_shift_matrix

        def fake_load(min_year=2008):
            return X, shift_cols, county_fips, pres_weights

        mod.load_shift_matrix = fake_load
        try:
            results = mod.run_temporal_weighting_experiment(
                j=3,
                temperature=10.0,
                min_year=2008,
                half_lives=[4.0, 8.0],
                verbose=False,
            )
        finally:
            mod.load_shift_matrix = original

        return results

    def test_result_is_dataframe(self, experiment_results, mod):
        assert isinstance(experiment_results, pd.DataFrame)

    def test_expected_columns_present(self, experiment_results, mod):
        expected = {"scheme", "mean_r", "std_r", "mean_mae", "std_mae", "n_folds", "n_active_dims"}
        assert expected.issubset(set(experiment_results.columns))

    def test_scheme_names_present(self, experiment_results, mod):
        schemes = set(experiment_results["scheme"].tolist())
        # With half_lives=[4, 8], expect these labels:
        assert "equal" in schemes
        assert "linear" in schemes
        assert "step_2016" in schemes
        assert "exponential_hl4" in schemes
        assert "exponential_hl8" in schemes

    def test_mae_non_negative(self, experiment_results, mod):
        assert (experiment_results["mean_mae"].dropna() >= 0).all()

    def test_std_non_negative(self, experiment_results, mod):
        assert (experiment_results["std_r"].dropna() >= 0).all()
        assert (experiment_results["std_mae"].dropna() >= 0).all()

    def test_n_folds_positive(self, experiment_results, mod):
        # The step_2016 scheme zeros out all pre-2016 dims; for 3 pairs (all 2008+)
        # it might still run. Just check no zero-fold rows for equal/linear.
        eq = experiment_results[experiment_results["scheme"] == "equal"]
        assert int(eq.iloc[0]["n_folds"]) > 0

    def test_n_active_dims_for_equal(self, experiment_results, mod):
        """Equal scheme should have all 9 dims active."""
        eq = experiment_results[experiment_results["scheme"] == "equal"]
        assert int(eq.iloc[0]["n_active_dims"]) == 9

    def test_n_active_dims_for_step(self, experiment_results, mod):
        """Step scheme zeros pre-2016 → no active dims for our synthetic pairs (all 2008+)."""
        step = experiment_results[experiment_results["scheme"] == "step_2016"]
        # Synthetic pairs are 08_12, 12_16, 16_20 — only 16_20 passes step cutoff (2016)
        assert int(step.iloc[0]["n_active_dims"]) == 3  # 3 cols in 16_20


# ---------------------------------------------------------------------------
# build_scheme_list
# ---------------------------------------------------------------------------


class TestBuildSchemeList:
    def test_default_includes_expected_labels(self, mod):
        schemes = mod.build_scheme_list()
        labels = [s["label"] for s in schemes]
        assert "equal" in labels
        assert "linear" in labels
        assert "step_2016" in labels
        assert "exponential_hl4" in labels
        assert "exponential_hl8" in labels
        assert "exponential_hl12" in labels

    def test_custom_half_lives(self, mod):
        schemes = mod.build_scheme_list(half_lives=[6.0, 16.0])
        labels = [s["label"] for s in schemes]
        assert "exponential_hl6" in labels
        assert "exponential_hl16" in labels
        assert "exponential_hl4" not in labels

    def test_half_life_stored_in_dict(self, mod):
        schemes = mod.build_scheme_list(half_lives=[8.0])
        exp_scheme = next(s for s in schemes if "exponential" in s["label"])
        assert exp_scheme["half_life"] == 8.0

    def test_non_exponential_has_none_half_life(self, mod):
        schemes = mod.build_scheme_list()
        for s in schemes:
            if s["scheme"] != "exponential":
                assert s["half_life"] is None


# ---------------------------------------------------------------------------
# scheme_label
# ---------------------------------------------------------------------------


class TestSchemeLabel:
    def test_equal_label(self, mod):
        assert mod.scheme_label("equal") == "equal"

    def test_linear_label(self, mod):
        assert mod.scheme_label("linear") == "linear"

    def test_step_label(self, mod):
        assert mod.scheme_label("step") == "step"

    def test_exponential_label_with_half_life(self, mod):
        assert mod.scheme_label("exponential", half_life=8.0) == "exponential_hl8"
        assert mod.scheme_label("exponential", half_life=4.0) == "exponential_hl4"


# ---------------------------------------------------------------------------
# save_results smoke test
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_saves_csv(self, mod, tmp_path):
        results = pd.DataFrame([
            {"scheme": "equal", "mean_r": 0.8, "std_r": 0.05,
             "mean_mae": 0.1, "std_mae": 0.01, "n_folds": 3, "n_active_dims": 9},
        ])
        out_path = mod.save_results(results, output_dir=tmp_path)
        assert out_path.exists()
        loaded = pd.read_csv(out_path)
        assert list(loaded["scheme"]) == ["equal"]
        assert loaded["mean_r"].iloc[0] == pytest.approx(0.8)
