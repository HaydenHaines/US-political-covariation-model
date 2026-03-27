"""Tests for src/discovery/score_borders.py — border gradient scoring.

Uses synthetic adjacency matrices and shift vectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.discovery.score_borders import compute_border_gradients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain_adjacency(n: int) -> csr_matrix:
    """Build a chain adjacency: 0-1, 1-2, 2-3, ..., (n-2)-(n-1)."""
    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeBorderGradients:
    def test_no_cross_community_edges(self):
        """All same community -> empty result."""
        W = _make_chain_adjacency(5)
        labels = np.array([0, 0, 0, 0, 0])
        shifts = np.random.randn(5, 3)
        geoids = [f"t{i}" for i in range(5)]
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert len(result) == 0
        assert list(result.columns) == ["community_a", "community_b", "gradient", "n_boundary_pairs"]

    def test_single_border(self):
        """Chain split into two communities -> one border."""
        W = _make_chain_adjacency(4)
        labels = np.array([0, 0, 1, 1])
        # Communities differ at indices 1-2
        shifts = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
        ])
        geoids = ["a", "b", "c", "d"]
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert len(result) == 1
        assert result.iloc[0]["community_a"] == 0
        assert result.iloc[0]["community_b"] == 1
        assert result.iloc[0]["n_boundary_pairs"] == 1
        # Gradient should be distance between shifts[1] and shifts[2]
        expected_grad = np.linalg.norm(shifts[1] - shifts[2])
        assert result.iloc[0]["gradient"] == pytest.approx(expected_grad)

    def test_multiple_borders(self):
        """Three communities -> up to 3 border pairs."""
        W = _make_chain_adjacency(6)
        labels = np.array([0, 0, 1, 1, 2, 2])
        shifts = np.random.RandomState(42).randn(6, 3)
        geoids = [f"t{i}" for i in range(6)]
        result = compute_border_gradients(labels, shifts, W, geoids)
        # Borders: 0-1 (indices 1,2) and 1-2 (indices 3,4)
        assert len(result) == 2
        assert set(result["community_a"]) | set(result["community_b"]) == {0, 1, 2}

    def test_community_pairs_are_sorted(self):
        """community_a should always be < community_b."""
        W = _make_chain_adjacency(4)
        labels = np.array([1, 1, 0, 0])  # reversed labels
        shifts = np.random.randn(4, 2)
        geoids = [str(i) for i in range(4)]
        result = compute_border_gradients(labels, shifts, W, geoids)
        for _, row in result.iterrows():
            assert row["community_a"] < row["community_b"]

    def test_gradient_is_mean_of_multiple_edges(self):
        """When multiple edges cross the same border, gradient is the mean."""
        # Grid: 0-1, 0-2, 1-3, 2-3 (a 2x2 grid)
        rows = [0, 1, 0, 2, 1, 3, 2, 3]
        cols = [1, 0, 2, 0, 3, 1, 3, 2]
        data = np.ones(8, dtype=np.int8)
        W = csr_matrix((data, (rows, cols)), shape=(4, 4))

        labels = np.array([0, 1, 0, 1])
        # shifts: community 0 at origin, community 1 offset
        shifts = np.array([
            [0.0, 0.0],  # comm 0
            [3.0, 0.0],  # comm 1
            [0.0, 0.0],  # comm 0
            [3.0, 4.0],  # comm 1
        ])
        geoids = ["a", "b", "c", "d"]
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert len(result) == 1
        # Two cross-community edges: (0,1) and (2,3)
        # dist(0,1) = 3.0, dist(2,3) = 5.0
        expected_gradient = (3.0 + 5.0) / 2
        assert result.iloc[0]["gradient"] == pytest.approx(expected_gradient)
        assert result.iloc[0]["n_boundary_pairs"] == 2

    def test_n_boundary_pairs_is_int(self):
        W = _make_chain_adjacency(4)
        labels = np.array([0, 0, 1, 1])
        shifts = np.random.randn(4, 2)
        geoids = [str(i) for i in range(4)]
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert result["n_boundary_pairs"].dtype in (np.int64, np.int32, int)

    def test_zero_gradient_when_shifts_identical(self):
        """If both sides of a border have identical shifts, gradient = 0."""
        W = _make_chain_adjacency(4)
        labels = np.array([0, 0, 1, 1])
        shifts = np.zeros((4, 3))
        geoids = [str(i) for i in range(4)]
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert len(result) == 1
        assert result.iloc[0]["gradient"] == pytest.approx(0.0)

    def test_result_sorted_by_community_pair(self):
        """Result should be sorted by (community_a, community_b)."""
        # Ensure 3 borders exist in arbitrary order
        n = 8
        W = _make_chain_adjacency(n)
        labels = np.array([2, 2, 0, 0, 1, 1, 2, 2])
        shifts = np.random.RandomState(7).randn(n, 2)
        geoids = [str(i) for i in range(n)]
        result = compute_border_gradients(labels, shifts, W, geoids)
        pairs = list(zip(result["community_a"], result["community_b"]))
        assert pairs == sorted(pairs)
