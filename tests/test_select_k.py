"""Tests for src/discovery/select_k.py"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from src.discovery.select_k import (
    run_k_sweep,
    pick_optimal_k,
    KSweepResult,
)


@pytest.fixture
def tiny_shifts():
    """20 counties with 33 shift dimensions."""
    rng = np.random.default_rng(42)
    n, d = 20, 33
    fips = [f"12{str(i).zfill(3)}" for i in range(n)]
    data = rng.normal(0, 0.1, (n, d))
    df = pd.DataFrame(data, columns=[f"shift_{i}" for i in range(d)])
    df.insert(0, "county_fips", fips)
    return df, fips


@pytest.fixture
def tiny_adjacency(tiny_shifts):
    """Chain adjacency for 20 counties."""
    _, fips = tiny_shifts
    n = len(fips)
    rows, cols = [], []
    for i in range(n - 1):
        rows += [i, i+1]; cols += [i+1, i]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_k_sweep_returns_results(tiny_shifts, tiny_adjacency):
    df, fips = tiny_shifts
    results = run_k_sweep(df, fips, tiny_adjacency,
                          train_cols=list(df.columns[1:31]),
                          holdout_cols=list(df.columns[31:]),
                          k_values=[3, 5],
                          min_community_size=2)
    assert len(results) >= 1


def test_k_sweep_result_fields(tiny_shifts, tiny_adjacency):
    df, fips = tiny_shifts
    results = run_k_sweep(df, fips, tiny_adjacency,
                          train_cols=list(df.columns[1:31]),
                          holdout_cols=list(df.columns[31:]),
                          k_values=[3],
                          min_community_size=2)
    r = results[0]
    assert hasattr(r, 'k')
    assert hasattr(r, 'holdout_r')
    assert hasattr(r, 'min_community_size')
    assert r.k == 3


def test_k_sweep_skips_small_communities(tiny_shifts, tiny_adjacency):
    """K values where min community < min_community_size are excluded."""
    df, fips = tiny_shifts
    results = run_k_sweep(df, fips, tiny_adjacency,
                          train_cols=list(df.columns[1:31]),
                          holdout_cols=list(df.columns[31:]),
                          k_values=[3, 5],
                          min_community_size=15)
    for r in results:
        assert r.min_community_size >= 15


def test_pick_optimal_k_highest_r():
    results = [
        KSweepResult(k=5, holdout_r=0.85, min_community_size=50, community_sizes=[50,50,50,50,43]),
        KSweepResult(k=7, holdout_r=0.87, min_community_size=42, community_sizes=[42]*7),
        KSweepResult(k=10, holdout_r=0.82, min_community_size=29, community_sizes=[29]*10),
    ]
    assert pick_optimal_k(results) == 7


def test_pick_optimal_k_empty_raises():
    with pytest.raises(ValueError, match="No valid K"):
        pick_optimal_k([])
