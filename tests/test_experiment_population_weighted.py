"""Tests for scripts/experiment_population_weighted.py.

Covers:
  - load_population_weights: normalization, shape, missing-value handling
  - temperature_soft_membership: row normalization, hard-assignment limit
  - group_columns_by_pair: column grouping logic
  - make_cv_folds: fold count and partition correctness
  - compute_centroid_distances: shape and zero-distance case
  - predict_holdout_columns: weighted mean reconstruction
  - compute_holdout_r: Pearson r, edge cases
  - compute_holdout_mae: correctness
  - run_one_condition: returns valid metrics, weight vs unweighted runs
  - run_experiment: output shape, scheme coverage, CSV output path

All tests use synthetic data only — no filesystem access except where the
module is loaded.
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

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "experiment_population_weighted.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "experiment_population_weighted", _MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_columns(pairs=None):
    """Return synthetic column names matching real shift name patterns."""
    if pairs is None:
        pairs = ["08_12", "12_16", "16_20"]
    cols = []
    for p in pairs:
        cols += [
            f"pres_d_shift_{p}",
            f"pres_r_shift_{p}",
            f"pres_turnout_shift_{p}",
        ]
    return cols


def _make_fake_census(county_fips: list[str], populations: list[int]) -> pd.DataFrame:
    """Build a minimal census DataFrame with pop_total for testing."""
    return pd.DataFrame({
        "county_fips": county_fips,
        "pop_total": populations,
        "pop_white_nh": [0] * len(county_fips),
        "year": [2020] * len(county_fips),
    })


# ---------------------------------------------------------------------------
# load_population_weights
# ---------------------------------------------------------------------------


class TestLoadPopulationWeights:
    def test_returns_three_schemes(self, mod, tmp_path, monkeypatch):
        fips = ["01001", "01003", "01005", "01007", "01009"]
        pops = [50000, 200000, 30000, 80000, 150000]
        census_df = _make_fake_census(fips, pops)

        monkeypatch.setattr(mod, "CENSUS_PATH", tmp_path / "census.parquet")
        census_df.to_parquet(tmp_path / "census.parquet", index=False)

        fips_arr = np.array(fips)
        weights = mod.load_population_weights(fips_arr)

        assert set(weights.keys()) == {"unweighted", "population", "log_population"}

    def test_unweighted_is_none(self, mod, tmp_path, monkeypatch):
        fips = ["01001", "01003"]
        census_df = _make_fake_census(fips, [100000, 200000])
        monkeypatch.setattr(mod, "CENSUS_PATH", tmp_path / "census.parquet")
        census_df.to_parquet(tmp_path / "census.parquet", index=False)

        weights = mod.load_population_weights(np.array(fips))
        assert weights["unweighted"] is None

    def test_population_normalized_to_mean_one(self, mod, tmp_path, monkeypatch):
        fips = ["01001", "01003", "01005"]
        pops = [100000, 200000, 300000]
        census_df = _make_fake_census(fips, pops)
        monkeypatch.setattr(mod, "CENSUS_PATH", tmp_path / "census.parquet")
        census_df.to_parquet(tmp_path / "census.parquet", index=False)

        weights = mod.load_population_weights(np.array(fips))
        np.testing.assert_allclose(weights["population"].mean(), 1.0, atol=1e-9)

    def test_log_population_normalized_to_mean_one(self, mod, tmp_path, monkeypatch):
        fips = ["01001", "01003", "01005"]
        pops = [100000, 200000, 300000]
        census_df = _make_fake_census(fips, pops)
        monkeypatch.setattr(mod, "CENSUS_PATH", tmp_path / "census.parquet")
        census_df.to_parquet(tmp_path / "census.parquet", index=False)

        weights = mod.load_population_weights(np.array(fips))
        np.testing.assert_allclose(weights["log_population"].mean(), 1.0, atol=1e-9)

    def test_population_weights_positive(self, mod, tmp_path, monkeypatch):
        fips = ["01001", "01003", "01005"]
        pops = [50000, 200000, 10000]
        census_df = _make_fake_census(fips, pops)
        monkeypatch.setattr(mod, "CENSUS_PATH", tmp_path / "census.parquet")
        census_df.to_parquet(tmp_path / "census.parquet", index=False)

        weights = mod.load_population_weights(np.array(fips))
        assert (weights["population"] > 0).all()
        assert (weights["log_population"] > 0).all()

    def test_missing_fips_filled_with_median(self, mod, tmp_path, monkeypatch):
        """A FIPS in the shift matrix but absent from census should not crash."""
        census_fips = ["01001", "01003"]
        shift_fips = ["01001", "01003", "99999"]  # 99999 not in census

        census_df = _make_fake_census(census_fips, [100000, 200000])
        monkeypatch.setattr(mod, "CENSUS_PATH", tmp_path / "census.parquet")
        census_df.to_parquet(tmp_path / "census.parquet", index=False)

        weights = mod.load_population_weights(np.array(shift_fips))
        assert len(weights["population"]) == 3
        assert not np.isnan(weights["population"]).any()

    def test_larger_county_gets_larger_raw_weight(self, mod, tmp_path, monkeypatch):
        """Larger population -> larger population weight (raw, before normalization)."""
        fips = ["01001", "01003"]
        pops = [50000, 500000]
        census_df = _make_fake_census(fips, pops)
        monkeypatch.setattr(mod, "CENSUS_PATH", tmp_path / "census.parquet")
        census_df.to_parquet(tmp_path / "census.parquet", index=False)

        weights = mod.load_population_weights(np.array(fips))
        # Second county is 10x larger — should have larger weight
        assert weights["population"][1] > weights["population"][0]
        assert weights["log_population"][1] > weights["log_population"][0]


# ---------------------------------------------------------------------------
# temperature_soft_membership
# ---------------------------------------------------------------------------


class TestTemperatureSoftMembership:
    def test_rows_sum_to_one(self, mod):
        rng = np.random.RandomState(0)
        dists = np.abs(rng.randn(50, 10)) + 0.1
        for T in [1.0, 5.0, 10.0, 999.0]:
            scores = mod.temperature_soft_membership(dists, T)
            np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-9)

    def test_t999_gives_hard_assignment(self, mod):
        rng = np.random.RandomState(3)
        dists = np.abs(rng.randn(40, 8)) + 0.1
        scores = mod.temperature_soft_membership(dists, T=999.0)
        assert (scores.max(axis=1) > 0.99).all()

    def test_non_negative(self, mod):
        rng = np.random.RandomState(1)
        dists = np.abs(rng.randn(30, 5)) + 0.01
        scores = mod.temperature_soft_membership(dists, T=10.0)
        assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# group_columns_by_pair
# ---------------------------------------------------------------------------


class TestGroupColumnsByPair:
    def test_groups_by_suffix(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16"])
        groups = mod.group_columns_by_pair(cols)
        assert "08_12" in groups
        assert len(groups["08_12"]) == 3

    def test_column_indices_correct(self, mod):
        cols = _make_synthetic_columns(["08_12"])
        groups = mod.group_columns_by_pair(cols)
        assert sorted(groups["08_12"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# make_cv_folds
# ---------------------------------------------------------------------------


class TestMakeCvFolds:
    def test_fold_count(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        assert len(folds) == 3

    def test_train_holdout_partition_covers_all(self, mod):
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        n_total = len(cols)
        for _, train_cols, holdout_cols in folds:
            assert len(train_cols) + len(holdout_cols) == n_total
            assert set(train_cols) & set(holdout_cols) == set()

    def test_min_year_filters_old_pairs(self, mod):
        cols = _make_synthetic_columns(["04_08", "08_12", "12_16"])
        folds = mod.make_cv_folds(cols, min_year=2008)
        fold_keys = [f[0] for f in folds]
        assert "04_08" not in fold_keys
        assert "08_12" in fold_keys


# ---------------------------------------------------------------------------
# compute_centroid_distances
# ---------------------------------------------------------------------------


class TestComputeCentroidDistances:
    def test_shape(self, mod):
        N, D, J = 50, 8, 5
        X = np.random.randn(N, D)
        centroids = np.random.randn(J, D)
        dists = mod.compute_centroid_distances(X, centroids)
        assert dists.shape == (N, J)

    def test_zero_distance_on_centroid(self, mod):
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        X = centroids.copy()
        dists = mod.compute_centroid_distances(X, centroids)
        assert dists[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert dists[1, 1] == pytest.approx(0.0, abs=1e-12)

    def test_non_negative(self, mod):
        X = np.random.randn(20, 6)
        centroids = np.random.randn(4, 6)
        dists = mod.compute_centroid_distances(X, centroids)
        assert (dists >= 0).all()


# ---------------------------------------------------------------------------
# predict_holdout_columns
# ---------------------------------------------------------------------------


class TestPredictHoldoutColumns:
    def test_output_shape(self, mod):
        N, J, D_hold = 40, 5, 3
        scores = np.random.dirichlet(np.ones(J), size=N)
        X_holdout = np.random.randn(N, D_hold)
        pred = mod.predict_holdout_columns(scores, X_holdout)
        assert pred.shape == (N, D_hold)

    def test_equal_scores_gives_global_mean(self, mod):
        N, J, D = 20, 4, 3
        scores = np.full((N, J), 1.0 / J)
        X_holdout = np.random.randn(N, D)
        pred = mod.predict_holdout_columns(scores, X_holdout)
        grand_mean = X_holdout.mean(axis=0)
        np.testing.assert_allclose(pred, np.tile(grand_mean, (N, 1)), atol=1e-10)


# ---------------------------------------------------------------------------
# compute_holdout_r and compute_holdout_mae
# ---------------------------------------------------------------------------


class TestHoldoutMetrics:
    def test_r_perfect_prediction(self, mod):
        X = np.random.randn(50, 3)
        r = mod.compute_holdout_r(X, X.copy())
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_r_zero_variance_returns_zero(self, mod):
        actual = np.ones((30, 1))
        predicted = np.random.randn(30, 1)
        r = mod.compute_holdout_r(actual, predicted)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_mae_zero_error(self, mod):
        X = np.random.randn(40, 3)
        mae = mod.compute_holdout_mae(X, X.copy())
        assert mae == pytest.approx(0.0, abs=1e-12)

    def test_mae_known_value(self, mod):
        actual = np.zeros((4, 2))
        predicted = np.ones((4, 2))
        mae = mod.compute_holdout_mae(actual, predicted)
        assert mae == pytest.approx(1.0, abs=1e-12)

    def test_mae_non_negative(self, mod):
        rng = np.random.RandomState(99)
        X = rng.randn(30, 5)
        Y = rng.randn(30, 5)
        assert mod.compute_holdout_mae(X, Y) >= 0


# ---------------------------------------------------------------------------
# run_one_condition
# ---------------------------------------------------------------------------


class TestRunOneCondition:
    def test_returns_expected_keys(self, mod):
        rng = np.random.RandomState(42)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        result = mod.run_one_condition(X, cols, j=4, sample_weight=None)
        for key in ("mean_r", "std_r", "mean_mae", "std_mae", "n_folds"):
            assert key in result

    def test_r_in_valid_range(self, mod):
        rng = np.random.RandomState(7)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        result = mod.run_one_condition(X, cols, j=3, sample_weight=None)
        assert -1.0 - 1e-6 <= result["mean_r"] <= 1.0 + 1e-6

    def test_mae_non_negative(self, mod):
        rng = np.random.RandomState(8)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        result = mod.run_one_condition(X, cols, j=3, sample_weight=None)
        assert result["mean_mae"] >= 0

    def test_weighted_run_completes(self, mod):
        """sample_weight != None should not crash."""
        rng = np.random.RandomState(5)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        pop = rng.rand(N) + 0.1
        result = mod.run_one_condition(X, cols, j=3, sample_weight=pop)
        assert isinstance(result["mean_r"], float)
        assert isinstance(result["mean_mae"], float)

    def test_n_folds_equals_number_of_pairs(self, mod):
        rng = np.random.RandomState(6)
        N, D = 60, 9
        X = rng.randn(N, D)
        cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])  # 3 pairs
        result = mod.run_one_condition(X, cols, j=3, sample_weight=None)
        assert result["n_folds"] == 3


# ---------------------------------------------------------------------------
# run_experiment (monkeypatched)
# ---------------------------------------------------------------------------


class TestRunExperiment:
    @pytest.fixture(scope="class")
    def experiment_results(self, mod):
        rng = np.random.RandomState(0)
        N, D = 80, 9
        X = rng.randn(N, D)
        shift_cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        county_fips = np.array([f"{i:05d}" for i in range(N)])

        pop = rng.rand(N) * 1e5 + 1e4
        pop_norm = pop / pop.mean()
        log_pop = np.log(pop)
        log_pop_norm = log_pop / log_pop.mean()

        def fake_load_shift(min_year=2008):
            return X, shift_cols, county_fips

        def fake_load_pop(fips):
            return {
                "unweighted": None,
                "population": pop_norm,
                "log_population": log_pop_norm,
            }

        orig_shift = mod.load_shift_matrix
        orig_pop = mod.load_population_weights
        mod.load_shift_matrix = fake_load_shift
        mod.load_population_weights = fake_load_pop
        try:
            results = mod.run_experiment(j_values=[3, 5], verbose=False)
        finally:
            mod.load_shift_matrix = orig_shift
            mod.load_population_weights = orig_pop

        return results

    def test_returns_dataframe(self, experiment_results, mod):
        assert isinstance(experiment_results, pd.DataFrame)

    def test_row_count(self, experiment_results, mod):
        # 2 J values x 3 weight schemes = 6 rows
        assert len(experiment_results) == 6

    def test_expected_columns(self, experiment_results, mod):
        expected = {"j", "weight_scheme", "mean_r", "std_r", "mean_mae", "std_mae", "n_folds"}
        assert expected.issubset(set(experiment_results.columns))

    def test_all_schemes_present(self, experiment_results, mod):
        schemes = set(experiment_results["weight_scheme"].unique())
        assert schemes == {"unweighted", "population", "log_population"}

    def test_all_j_values_present(self, experiment_results, mod):
        assert set(experiment_results["j"].unique()) == {3, 5}

    def test_mean_r_in_valid_range(self, experiment_results, mod):
        assert (experiment_results["mean_r"].between(-1.0 - 1e-6, 1.0 + 1e-6)).all()

    def test_mean_mae_non_negative(self, experiment_results, mod):
        assert (experiment_results["mean_mae"] >= 0).all()

    def test_std_r_non_negative(self, experiment_results, mod):
        assert (experiment_results["std_r"] >= 0).all()

    def test_n_folds_positive(self, experiment_results, mod):
        assert (experiment_results["n_folds"] > 0).all()

    def test_save_csv(self, mod, tmp_path):
        """run_experiment result can be saved to CSV without error."""
        rng = np.random.RandomState(1)
        N, D = 40, 9
        X = rng.randn(N, D)
        shift_cols = _make_synthetic_columns(["08_12", "12_16", "16_20"])
        county_fips = np.array([f"{i:05d}" for i in range(N)])
        pop = rng.rand(N) * 1e5 + 1e4
        pop_norm = pop / pop.mean()
        log_pop_norm = np.log(pop) / np.log(pop).mean()

        def fake_load_shift(min_year=2008):
            return X, shift_cols, county_fips

        def fake_load_pop(fips):
            return {
                "unweighted": None,
                "population": pop_norm,
                "log_population": log_pop_norm,
            }

        orig_shift = mod.load_shift_matrix
        orig_pop = mod.load_population_weights
        orig_output = mod.OUTPUT_DIR
        mod.load_shift_matrix = fake_load_shift
        mod.load_population_weights = fake_load_pop
        mod.OUTPUT_DIR = tmp_path
        try:
            results = mod.run_experiment(j_values=[3], verbose=False)
            out_path = tmp_path / "population_weighted_results.csv"
            results.to_csv(out_path, index=False)
            assert out_path.exists()
            reloaded = pd.read_csv(out_path)
            assert len(reloaded) == 3  # 1 J x 3 schemes
        finally:
            mod.load_shift_matrix = orig_shift
            mod.load_population_weights = orig_pop
            mod.OUTPUT_DIR = orig_output
