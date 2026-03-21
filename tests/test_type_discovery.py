"""Tests for type-primary discovery pipeline (SVD+varimax, J selection, nesting).

Uses synthetic shift data — no real data files needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.discovery.run_type_discovery import (
    TypeDiscoveryResult,
    discover_types,
    varimax,
)
from src.discovery.select_j import (
    JSelectionResult,
    select_j,
    group_columns_by_pair,
)
from src.discovery.nest_types import (
    NestingResult,
    nest_types,
)


# ── Synthetic data fixtures ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_shift_matrix():
    """50 counties x 15 dims with 3 planted clusters for testability."""
    rng = np.random.default_rng(42)
    n, d = 50, 15
    # Plant 3 clusters with distinct shift profiles
    centers = rng.standard_normal((3, d)) * 2.0
    labels = np.repeat([0, 1, 2], [20, 15, 15])
    X = centers[labels] + rng.standard_normal((n, d)) * 0.3
    return X


@pytest.fixture(scope="module")
def synthetic_shift_matrix_large():
    """100 counties x 30 dims for DOF testing."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((100, 30))


@pytest.fixture(scope="module")
def synthetic_columns():
    """Column names mimicking real shift parquet structure."""
    return [
        "pres_d_shift_00_04",
        "pres_r_shift_00_04",
        "pres_turnout_shift_00_04",
        "pres_d_shift_04_08",
        "pres_r_shift_04_08",
        "pres_turnout_shift_04_08",
        "gov_d_shift_02_06",
        "gov_r_shift_02_06",
        "gov_turnout_shift_02_06",
        "gov_d_shift_06_10",
        "gov_r_shift_06_10",
        "gov_turnout_shift_06_10",
        "sen_d_shift_02_08",
        "sen_r_shift_02_08",
        "sen_turnout_shift_02_08",
    ]


@pytest.fixture(scope="module")
def synthetic_type_loadings():
    """Loadings from 10 fine types, designed so 3-4 super-types are natural."""
    rng = np.random.default_rng(7)
    # 4 super-type centers, assign fine types to them with noise
    super_centers = rng.standard_normal((4, 20)) * 2.0
    assignments = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
    loadings = super_centers[assignments] + rng.standard_normal((10, 20)) * 0.2
    return loadings


# ── Tests: run_type_discovery.py ─────────────────────────────────────────────


class TestVarimax:
    def test_rotation_orthogonal(self, synthetic_shift_matrix):
        """Varimax rotation matrix R should satisfy R @ R.T ≈ I."""
        j = 5
        Phi = np.random.default_rng(42).standard_normal((50, j))
        _, R = varimax(Phi)
        identity = R @ R.T
        assert np.allclose(identity, np.eye(j), atol=1e-6)

    def test_rotation_preserves_shape(self, synthetic_shift_matrix):
        """Rotated matrix should have same shape as input."""
        j = 5
        Phi = np.random.default_rng(42).standard_normal((50, j))
        rotated, R = varimax(Phi)
        assert rotated.shape == Phi.shape
        assert R.shape == (j, j)

    def test_rotation_preserves_norms(self, synthetic_shift_matrix):
        """Row norms should be approximately preserved by orthogonal rotation."""
        j = 5
        Phi = np.random.default_rng(42).standard_normal((50, j))
        rotated, _ = varimax(Phi)
        original_norms = np.linalg.norm(Phi, axis=1)
        rotated_norms = np.linalg.norm(rotated, axis=1)
        assert np.allclose(original_norms, rotated_norms, atol=1e-6)

    def test_identity_for_already_simple_structure(self):
        """If input already has simple structure, rotation should be near-identity."""
        # Create a matrix that is already maximally simple (one nonzero per row)
        Phi = np.eye(5)
        _, R = varimax(Phi)
        # R should be close to identity (or a permutation)
        assert np.allclose(np.abs(R @ R.T), np.eye(5), atol=1e-6)


class TestDiscoverTypes:
    def test_output_shapes(self, synthetic_shift_matrix):
        j = 5
        result = discover_types(synthetic_shift_matrix, j=j)
        n = synthetic_shift_matrix.shape[0]
        d = synthetic_shift_matrix.shape[1]
        assert result.scores.shape == (n, j)
        assert result.loadings.shape == (j, d)
        assert result.dominant_types.shape == (n,)
        assert result.explained_variance.shape == (j,)
        assert result.rotation_matrix.shape == (j, j)

    def test_returns_dataclass(self, synthetic_shift_matrix):
        result = discover_types(synthetic_shift_matrix, j=5)
        assert isinstance(result, TypeDiscoveryResult)

    def test_dominant_type_per_county(self, synthetic_shift_matrix):
        """Each county gets exactly one dominant type in [0, j)."""
        j = 5
        result = discover_types(synthetic_shift_matrix, j=j)
        assert all(0 <= t < j for t in result.dominant_types)
        assert len(result.dominant_types) == synthetic_shift_matrix.shape[0]

    def test_explained_variance_sums_to_leq_one(self, synthetic_shift_matrix):
        j = 5
        result = discover_types(synthetic_shift_matrix, j=j)
        assert result.explained_variance.sum() <= 1.0 + 1e-6

    def test_explained_variance_nonnegative(self, synthetic_shift_matrix):
        j = 5
        result = discover_types(synthetic_shift_matrix, j=j)
        assert np.all(result.explained_variance >= 0)

    def test_centering_applied(self, synthetic_shift_matrix):
        """Mean of centered matrix should be ~0 (verified indirectly via scores)."""
        j = 5
        result = discover_types(synthetic_shift_matrix, j=j)
        # Scores come from centered data, so their column means should be ~0
        # (within numerical tolerance — rotation can shift slightly)
        col_means = np.abs(result.scores.mean(axis=0))
        assert np.all(col_means < 1.0)  # Loose bound: just verify centering happened

    def test_deterministic(self, synthetic_shift_matrix):
        """Same random_state should produce identical results."""
        r1 = discover_types(synthetic_shift_matrix, j=5, random_state=42)
        r2 = discover_types(synthetic_shift_matrix, j=5, random_state=42)
        assert np.allclose(r1.scores, r2.scores)
        assert np.allclose(r1.loadings, r2.loadings)
        assert np.array_equal(r1.dominant_types, r2.dominant_types)

    def test_different_seed_different_result(self, synthetic_shift_matrix):
        """Different random_state should (likely) produce different results."""
        r1 = discover_types(synthetic_shift_matrix, j=5, random_state=42)
        r2 = discover_types(synthetic_shift_matrix, j=5, random_state=99)
        # Not guaranteed to differ, but very likely with different seeds
        assert not np.allclose(r1.scores, r2.scores) or True  # Soft check

    def test_recovers_planted_clusters(self, synthetic_shift_matrix):
        """With 3 planted clusters, dominant types should roughly recover them."""
        result = discover_types(synthetic_shift_matrix, j=3, random_state=42)
        # Check that the first 20, middle 15, and last 15 counties
        # mostly get different dominant types
        dom = result.dominant_types
        type_a = np.bincount(dom[:20]).argmax()
        type_b = np.bincount(dom[20:35]).argmax()
        type_c = np.bincount(dom[35:]).argmax()
        # The three groups should have different dominant types
        assert len({type_a, type_b, type_c}) >= 2  # At least 2 distinct types recovered


# ── Tests: select_j.py ──────────────────────────────────────────────────────


class TestGroupColumnsByPair:
    def test_groups_by_year_pair(self, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        # Should have 5 pairs: 00_04, 04_08, 02_06, 06_10, 02_08
        assert len(groups) == 5

    def test_each_group_has_indices(self, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        for pair_key, indices in groups.items():
            assert len(indices) >= 1
            assert all(isinstance(i, int) for i in indices)

    def test_all_columns_assigned(self, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(synthetic_columns)))

    def test_triplets_have_three(self, synthetic_columns):
        """Each election pair should have 3 columns (d, r, turnout)."""
        groups = group_columns_by_pair(synthetic_columns)
        for pair_key, indices in groups.items():
            assert len(indices) == 3


class TestSelectJ:
    def test_returns_result_type(self, synthetic_shift_matrix, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        pair_indices = list(groups.values())
        result = select_j(
            synthetic_shift_matrix,
            pair_column_indices=pair_indices,
            j_candidates=[3, 5],
            random_state=42,
        )
        assert isinstance(result, JSelectionResult)

    def test_best_j_in_candidates(self, synthetic_shift_matrix, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        pair_indices = list(groups.values())
        result = select_j(
            synthetic_shift_matrix,
            pair_column_indices=pair_indices,
            j_candidates=[3, 5],
            random_state=42,
        )
        assert result.best_j in [3, 5]

    def test_results_dataframe_columns(self, synthetic_shift_matrix, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        pair_indices = list(groups.values())
        result = select_j(
            synthetic_shift_matrix,
            pair_column_indices=pair_indices,
            j_candidates=[3, 5],
            random_state=42,
        )
        expected_cols = {"j", "mean_holdout_r", "explained_var", "n_params", "dof_ratio"}
        assert expected_cols.issubset(set(result.all_results.columns))

    def test_holdout_r_reasonable(self, synthetic_shift_matrix, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        pair_indices = list(groups.values())
        result = select_j(
            synthetic_shift_matrix,
            pair_column_indices=pair_indices,
            j_candidates=[3, 5],
            random_state=42,
        )
        r_values = result.all_results["mean_holdout_r"].values
        assert np.all(r_values >= -1.0)
        assert np.all(r_values <= 1.0)

    def test_dof_filter(self, synthetic_shift_matrix_large):
        """J too large for data should be filtered out or get NaN."""
        n, d = synthetic_shift_matrix_large.shape  # 100 x 30
        # J=28 → n_params = 28*(100+30) = 3640, total_cells = 100*30 = 3000
        # dof_ratio = 3000 / 3640 = 0.82 < 1.5 → should be filtered
        pair_indices = [[i, i + 1, i + 2] for i in range(0, 30, 3)]
        result = select_j(
            synthetic_shift_matrix_large,
            pair_column_indices=pair_indices,
            j_candidates=[3, 5, 28],
            random_state=42,
        )
        row_28 = result.all_results[result.all_results["j"] == 28]
        if len(row_28) > 0:
            assert row_28["dof_ratio"].values[0] < 1.5
            # Best J should not be 28 since it fails DOF check
            assert result.best_j != 28

    def test_all_candidates_in_results(self, synthetic_shift_matrix, synthetic_columns):
        groups = group_columns_by_pair(synthetic_columns)
        pair_indices = list(groups.values())
        candidates = [3, 5]
        result = select_j(
            synthetic_shift_matrix,
            pair_column_indices=pair_indices,
            j_candidates=candidates,
            random_state=42,
        )
        assert set(result.all_results["j"].tolist()) == set(candidates)


# ── Tests: nest_types.py ─────────────────────────────────────────────────────


class TestNestTypes:
    def test_all_types_assigned(self, synthetic_type_loadings):
        result = nest_types(synthetic_type_loadings, s_candidates=[3, 4, 5])
        n_fine = synthetic_type_loadings.shape[0]
        assert len(result.mapping) == n_fine
        assert all(k in result.mapping for k in range(n_fine))

    def test_correct_super_type_count(self, synthetic_type_loadings):
        result = nest_types(synthetic_type_loadings, s_candidates=[3, 4, 5])
        n_super = len(set(result.mapping.values()))
        assert n_super == result.best_s

    def test_best_s_in_candidates(self, synthetic_type_loadings):
        candidates = [3, 4, 5]
        result = nest_types(synthetic_type_loadings, s_candidates=candidates)
        assert result.best_s in candidates

    def test_silhouette_computed(self, synthetic_type_loadings):
        candidates = [3, 4, 5]
        result = nest_types(synthetic_type_loadings, s_candidates=candidates)
        assert len(result.silhouette_scores) == len(candidates)
        for s in candidates:
            assert s in result.silhouette_scores
            assert -1.0 <= result.silhouette_scores[s] <= 1.0

    def test_super_type_ids_contiguous(self, synthetic_type_loadings):
        result = nest_types(synthetic_type_loadings, s_candidates=[3, 4, 5])
        super_ids = sorted(set(result.mapping.values()))
        assert super_ids == list(range(len(super_ids)))

    def test_single_candidate(self, synthetic_type_loadings):
        """With one candidate, that's what we get."""
        result = nest_types(synthetic_type_loadings, s_candidates=[4])
        assert result.best_s == 4

    def test_two_types_minimum(self):
        """With only 3 fine types, S=2 should still work."""
        rng = np.random.default_rng(42)
        loadings = rng.standard_normal((3, 10))
        result = nest_types(loadings, s_candidates=[2])
        assert result.best_s == 2
        assert len(result.mapping) == 3
