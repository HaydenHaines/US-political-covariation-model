"""Tests for src/assembly/build_hac_community_weights.py"""
import numpy as np
import pandas as pd
import pytest
from src.assembly.build_hac_community_weights import (
    build_county_weights,
    build_state_weights,
)


@pytest.fixture
def sample_assignments():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "community_id": [0, 0, 1, 1, 2],
    })


@pytest.fixture
def sample_vote_totals():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "recent_total": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
        "state_fips": ["12", "12", "13", "13", "01"],
    })


def test_build_county_weights_shape(sample_assignments, sample_vote_totals):
    w = build_county_weights(sample_assignments, sample_vote_totals)
    assert "county_fips" in w.columns
    assert "community_id" in w.columns
    assert len(w) == len(sample_assignments)


def test_build_county_weights_hard_assignment(sample_assignments, sample_vote_totals):
    """With hard HAC assignments, each county's weight is 1.0 for its community."""
    w = build_county_weights(sample_assignments, sample_vote_totals)
    merged = w.merge(sample_assignments, on="county_fips")
    assert (merged["community_id_x"] == merged["community_id_y"]).all()


def test_build_state_weights_shape(sample_assignments, sample_vote_totals):
    k = sample_assignments["community_id"].nunique()
    w = build_state_weights(sample_assignments, sample_vote_totals)
    assert "state_fips" in w.columns
    assert len(w) == sample_vote_totals["state_fips"].nunique()
    weight_cols = [c for c in w.columns if c.startswith("community_")]
    assert len(weight_cols) == k


def test_build_state_weights_sum_to_one(sample_assignments, sample_vote_totals):
    w = build_state_weights(sample_assignments, sample_vote_totals)
    weight_cols = [c for c in w.columns if c.startswith("community_")]
    row_sums = w[weight_cols].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)
