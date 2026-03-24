"""Tests for the type stability sub-window experiment (P4.2).

Covers:
- ARI/NMI computation on known inputs
- Hungarian matching correctness
- County stability rate computation
- Sub-window column splitting logic
- Year parsing
- Holdout r computation
- Edge cases
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scripts.experiment_type_stability import (
    BLIND_HOLDOUT_COLUMNS,
    MIN_YEAR,
    SPLIT_YEAR,
    compute_holdout_r,
    compute_subwindow_holdout_r,
    county_stability_rate,
    get_pair_start_year,
    hungarian_match,
    parse_2digit_year,
    predict_from_types,
    run_kmeans,
    split_columns_by_window,
    temperature_soft_membership,
)


# ---------------------------------------------------------------------------
# Year parsing tests
# ---------------------------------------------------------------------------


class TestParse2DigitYear:
    def test_recent_years_map_to_2000s(self):
        assert parse_2digit_year("00") == 2000
        assert parse_2digit_year("08") == 2008
        assert parse_2digit_year("16") == 2016
        assert parse_2digit_year("24") == 2024
        assert parse_2digit_year("29") == 2029

    def test_old_years_map_to_1900s(self):
        assert parse_2digit_year("94") == 1994
        assert parse_2digit_year("98") == 1998
        assert parse_2digit_year("30") == 1930
        assert parse_2digit_year("99") == 1999

    def test_boundary_year_30(self):
        # 29 -> 2029, 30 -> 1930
        assert parse_2digit_year("29") == 2029
        assert parse_2digit_year("30") == 1930


class TestGetPairStartYear:
    def test_presidential_column(self):
        assert get_pair_start_year("pres_d_shift_08_12") == 2008
        assert get_pair_start_year("pres_r_shift_16_20") == 2016

    def test_governor_column(self):
        assert get_pair_start_year("gov_d_shift_10_14") == 2010
        assert get_pair_start_year("gov_r_shift_94_98") == 1994

    def test_senate_column(self):
        assert get_pair_start_year("sen_d_shift_08_14") == 2008
        assert get_pair_start_year("sen_r_shift_16_22") == 2016

    def test_non_shift_column_returns_none(self):
        assert get_pair_start_year("county_fips") is None
        assert get_pair_start_year("population_2020") is None

    def test_holdout_column_parses_correctly(self):
        assert get_pair_start_year("pres_d_shift_20_24") == 2020


# ---------------------------------------------------------------------------
# Sub-window column splitting tests
# ---------------------------------------------------------------------------


class TestSplitColumnsByWindow:
    def setup_method(self):
        self.cols = [
            "pres_d_shift_08_12",
            "pres_r_shift_08_12",
            "pres_d_shift_12_16",
            "gov_d_shift_10_14",
            "gov_d_shift_14_18",
            "sen_d_shift_08_14",
            "pres_d_shift_16_20",
            "gov_d_shift_18_22",
            "sen_d_shift_16_22",
        ]

    def test_early_indices_correct(self):
        early_idx, late_idx = split_columns_by_window(self.cols, split_year=2016, min_year=2008)
        early_names = [self.cols[i] for i in early_idx]
        assert "pres_d_shift_08_12" in early_names
        assert "pres_d_shift_12_16" in early_names
        assert "gov_d_shift_10_14" in early_names
        assert "gov_d_shift_14_18" in early_names
        assert "sen_d_shift_08_14" in early_names

    def test_late_indices_correct(self):
        early_idx, late_idx = split_columns_by_window(self.cols, split_year=2016, min_year=2008)
        late_names = [self.cols[i] for i in late_idx]
        assert "pres_d_shift_16_20" in late_names
        assert "gov_d_shift_18_22" in late_names
        assert "sen_d_shift_16_22" in late_names

    def test_split_year_boundary_goes_to_late(self):
        early_idx, late_idx = split_columns_by_window(self.cols, split_year=2016, min_year=2008)
        late_names = [self.cols[i] for i in late_idx]
        # pres_d_shift_16_20 starts at 2016 -> late
        assert "pres_d_shift_16_20" in late_names

    def test_split_year_minus_one_goes_to_early(self):
        early_idx, late_idx = split_columns_by_window(self.cols, split_year=2016, min_year=2008)
        early_names = [self.cols[i] for i in early_idx]
        # pres_d_shift_12_16 starts at 2012 -> early
        assert "pres_d_shift_12_16" in early_names

    def test_disjoint_partition(self):
        early_idx, late_idx = split_columns_by_window(self.cols, split_year=2016, min_year=2008)
        assert len(set(early_idx) & set(late_idx)) == 0

    def test_covers_all_eligible_cols(self):
        early_idx, late_idx = split_columns_by_window(self.cols, split_year=2016, min_year=2008)
        # All 9 cols have parseable years; all start >= 2008 and none is holdout
        assert len(early_idx) + len(late_idx) == len(self.cols)

    def test_custom_split_year(self):
        # Split at 2012 should push 12_16 to late
        early_idx, late_idx = split_columns_by_window(self.cols, split_year=2012, min_year=2008)
        early_names = [self.cols[i] for i in early_idx]
        late_names = [self.cols[i] for i in late_idx]
        assert "pres_d_shift_08_12" in early_names
        assert "pres_d_shift_12_16" in late_names


# ---------------------------------------------------------------------------
# Hungarian matching tests
# ---------------------------------------------------------------------------


class TestHungarianMatch:
    def test_identical_labels_returns_identity_perm(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        perm, remapped = hungarian_match(labels, labels.copy())
        np.testing.assert_array_equal(remapped, labels)

    def test_permuted_labels_recovers_original(self):
        # labels_b is labels_a with 0<->1 swapped
        labels_a = np.array([0, 0, 1, 1, 2, 2])
        labels_b = np.array([1, 1, 0, 0, 2, 2])  # 0 and 1 swapped
        _, remapped = hungarian_match(labels_a, labels_b)
        np.testing.assert_array_equal(remapped, labels_a)

    def test_stability_rate_is_one_for_permuted_labels(self):
        labels_a = np.array([0, 0, 1, 1, 2, 2])
        labels_b = np.array([1, 1, 0, 0, 2, 2])
        rate = county_stability_rate(labels_a, labels_b)
        assert rate == pytest.approx(1.0)

    def test_random_labels_have_low_stability(self):
        rng = np.random.default_rng(0)
        labels_a = rng.integers(0, 10, size=100)
        labels_b = rng.integers(0, 10, size=100)
        rate = county_stability_rate(labels_a, labels_b)
        # Random agreement for J=10 is ~1/10 = 10%. Allow generous bound.
        assert rate < 0.4

    def test_perm_is_a_valid_mapping(self):
        rng = np.random.default_rng(42)
        labels_a = rng.integers(0, 5, size=50)
        labels_b = rng.integers(0, 5, size=50)
        perm, remapped = hungarian_match(labels_a, labels_b)
        # remapped should only contain values in range 0..4
        assert set(remapped).issubset(set(range(5)))


# ---------------------------------------------------------------------------
# ARI / NMI on known inputs
# ---------------------------------------------------------------------------


class TestAriNmiKnownInputs:
    def test_ari_identical_labels_is_one(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0])
        ari = adjusted_rand_score(labels, labels)
        assert ari == pytest.approx(1.0)

    def test_ari_random_labels_near_zero(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 20, 300)
        b = rng.integers(0, 20, 300)
        ari = adjusted_rand_score(a, b)
        assert abs(ari) < 0.1

    def test_nmi_identical_labels_is_one(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        nmi = normalized_mutual_info_score(labels, labels)
        assert nmi == pytest.approx(1.0)

    def test_ari_is_symmetric(self):
        rng = np.random.default_rng(7)
        a = rng.integers(0, 5, 50)
        b = rng.integers(0, 5, 50)
        assert adjusted_rand_score(a, b) == pytest.approx(adjusted_rand_score(b, a))


# ---------------------------------------------------------------------------
# Holdout r computation
# ---------------------------------------------------------------------------


class TestComputeHoldoutR:
    def test_perfect_prediction_gives_r_one(self):
        actual = np.random.default_rng(0).normal(size=(100, 3))
        r = compute_holdout_r(actual, actual)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_negated_prediction_gives_r_minus_one(self):
        actual = np.random.default_rng(1).normal(size=(50, 2))
        r = compute_holdout_r(actual, -actual)
        assert r == pytest.approx(-1.0, abs=1e-6)

    def test_random_prediction_gives_low_abs_r(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(size=(500, 3))
        predicted = rng.normal(size=(500, 3))
        r = compute_holdout_r(actual, predicted)
        assert abs(r) < 0.15

    def test_constant_column_gives_zero_r(self):
        actual = np.ones((50, 2))  # constant columns
        predicted = np.random.default_rng(5).normal(size=(50, 2))
        r = compute_holdout_r(actual, predicted)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# predict_from_types
# ---------------------------------------------------------------------------


class TestPredictFromTypes:
    def test_single_type_predicts_column_mean(self):
        """With J=1, every county maps to the sole type, whose mean is the column mean."""
        rng = np.random.default_rng(3)
        N, D_train, D_holdout = 20, 5, 3
        scores = np.ones((N, 1))  # single type, all weight = 1
        X_train = rng.normal(size=(N, D_train))
        X_holdout = rng.normal(size=(N, D_holdout))

        predicted = predict_from_types(scores, X_train, X_holdout)
        # Should be the column mean repeated for each county
        expected = X_holdout.mean(axis=0)[None, :].repeat(N, axis=0)
        np.testing.assert_allclose(predicted, expected, atol=1e-10)

    def test_output_shape(self):
        rng = np.random.default_rng(4)
        N, J, D = 30, 5, 4
        scores = rng.dirichlet(np.ones(J), size=N)
        X_train = rng.normal(size=(N, D))
        X_holdout = rng.normal(size=(N, 3))
        out = predict_from_types(scores, X_train, X_holdout)
        assert out.shape == (N, 3)


# ---------------------------------------------------------------------------
# KMeans integration
# ---------------------------------------------------------------------------


class TestRunKMeans:
    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 10))
        j = 5
        labels, centroids, scores = run_kmeans(X, j=j, n_init=3)
        assert labels.shape == (50,)
        assert centroids.shape == (j, 10)
        assert scores.shape == (50, j)

    def test_scores_sum_to_one(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 8))
        _, _, scores = run_kmeans(X, j=4, n_init=3)
        np.testing.assert_allclose(scores.sum(axis=1), np.ones(40), atol=1e-6)

    def test_labels_are_valid_cluster_indices(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 6))
        j = 6
        labels, _, _ = run_kmeans(X, j=j, n_init=3)
        assert labels.min() >= 0
        assert labels.max() < j

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(80, 5))
        labels1, _, _ = run_kmeans(X, j=5, random_state=99, n_init=3)
        labels2, _, _ = run_kmeans(X, j=5, random_state=99, n_init=3)
        np.testing.assert_array_equal(labels1, labels2)
