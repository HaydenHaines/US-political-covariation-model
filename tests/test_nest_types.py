"""Tests for src/discovery/nest_types.py — hierarchical nesting of fine types into super-types.

Uses synthetic feature matrices to test the Ward HAC nesting logic without real data.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.discovery.nest_types import NestingResult, nest_types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clustered_features(n_clusters: int, n_per_cluster: int, n_dims: int, sep: float = 5.0) -> np.ndarray:
    """Create a feature matrix with clear cluster structure."""
    rng = np.random.RandomState(42)
    centers = rng.randn(n_clusters, n_dims) * sep
    features = []
    for center in centers:
        cluster_points = center + rng.randn(n_per_cluster, n_dims) * 0.5
        features.append(cluster_points)
    return np.vstack(features)


# ---------------------------------------------------------------------------
# Tests: NestingResult dataclass
# ---------------------------------------------------------------------------

class TestNestingResult:
    def test_fields(self):
        r = NestingResult(mapping={0: 0, 1: 1}, best_s=2, silhouette_scores={2: 0.8, 3: 0.6})
        assert r.best_s == 2
        assert r.mapping[0] == 0
        assert r.silhouette_scores[2] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Tests: nest_types core function
# ---------------------------------------------------------------------------

class TestNestTypes:
    def test_basic_two_clusters(self):
        """Two well-separated clusters should nest cleanly into S=2."""
        features = _make_clustered_features(2, 10, 5, sep=10.0)
        result = nest_types(features, s_candidates=[2, 3, 4])
        assert isinstance(result, NestingResult)
        # With two clear clusters, S=2 should win
        assert result.best_s == 2
        # All 20 types should be mapped to 0 or 1
        assert set(result.mapping.values()) == {0, 1}
        assert len(result.mapping) == 20

    def test_mapping_covers_all_types(self):
        features = _make_clustered_features(3, 5, 4)
        result = nest_types(features, s_candidates=[3, 4, 5])
        assert len(result.mapping) == 15  # 3 clusters * 5 per cluster
        for i in range(15):
            assert i in result.mapping

    def test_silhouette_scores_for_all_candidates(self):
        features = _make_clustered_features(3, 5, 4)
        candidates = [2, 3, 4, 5]
        result = nest_types(features, s_candidates=candidates)
        for s in candidates:
            assert s in result.silhouette_scores

    def test_default_s_candidates(self):
        """Default candidates are [5, 6, 7, 8]."""
        features = _make_clustered_features(5, 4, 3)
        result = nest_types(features)
        assert set(result.silhouette_scores.keys()) == {5, 6, 7, 8}

    def test_s_exceeding_j_gets_negative_score(self):
        """When S >= J, silhouette should be -1 (can't have more super-types than types)."""
        features = np.random.randn(5, 3)
        result = nest_types(features, s_candidates=[3, 5, 6])
        assert result.silhouette_scores[5] == -1.0
        assert result.silhouette_scores[6] == -1.0

    def test_mapping_values_are_zero_indexed(self):
        features = _make_clustered_features(4, 8, 5)
        result = nest_types(features, s_candidates=[3, 4])
        super_types = set(result.mapping.values())
        assert min(super_types) == 0

    def test_mapping_values_contiguous(self):
        features = _make_clustered_features(4, 8, 5)
        result = nest_types(features, s_candidates=[3, 4])
        super_types = sorted(set(result.mapping.values()))
        assert super_types == list(range(len(super_types)))

    def test_best_s_is_in_candidates(self):
        features = _make_clustered_features(3, 5, 4)
        candidates = [2, 3, 5]
        result = nest_types(features, s_candidates=candidates)
        assert result.best_s in candidates

    def test_single_point_per_cluster(self):
        """Edge case: each 'cluster' has only one type."""
        features = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        result = nest_types(features, s_candidates=[2, 3])
        assert len(result.mapping) == 4
        assert result.best_s in (2, 3)

    def test_identical_features_handled(self):
        """All types have identical features — S=2 should yield one cluster."""
        features = np.ones((10, 3))
        result = nest_types(features, s_candidates=[2, 3])
        # With identical features, silhouette is -1 for all S
        # (since we can't distinguish clusters)
        # The function should still return without error
        assert isinstance(result, NestingResult)
        assert len(result.mapping) == 10

    def test_high_dimensional_features(self):
        """Should work with many dimensions."""
        features = _make_clustered_features(3, 10, 50, sep=8.0)
        result = nest_types(features, s_candidates=[2, 3, 4])
        assert result.best_s in (2, 3, 4)

    def test_reproducibility(self):
        """Same input should give same output (Ward HAC is deterministic)."""
        features = _make_clustered_features(3, 5, 4)
        r1 = nest_types(features, s_candidates=[2, 3, 4])
        r2 = nest_types(features, s_candidates=[2, 3, 4])
        assert r1.mapping == r2.mapping
        assert r1.best_s == r2.best_s
