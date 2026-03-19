"""Tests for src/models/nmf_types.py"""
import numpy as np
import pandas as pd
import pytest
from src.models.nmf_types import (
    compute_community_profiles,
    fit_nmf,
    NMFResult,
    sweep_j,
)


@pytest.fixture
def sample_shifts():
    rng = np.random.default_rng(99)
    n_counties = 30
    n_dims = 30
    fips = [f"12{str(i).zfill(3)}" for i in range(n_counties)]
    shift_cols = [f"pres_d_shift_{i:02d}_{i+4:02d}" for i in range(n_dims)]
    data = rng.normal(0, 0.1, (n_counties, n_dims))
    df = pd.DataFrame(data, columns=shift_cols)
    df.insert(0, "county_fips", fips)
    return df, shift_cols


@pytest.fixture
def sample_assignments(sample_shifts):
    df, _ = sample_shifts
    rng = np.random.default_rng(0)
    n = len(df)
    return pd.DataFrame({
        "county_fips": df["county_fips"],
        "community_id": rng.integers(0, 5, n),
    })


def test_compute_profiles_shape(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    k = sample_assignments["community_id"].nunique()
    assert profiles.shape == (k, len(shift_cols))


def test_compute_profiles_mean(sample_shifts, sample_assignments):
    """Profile for community 0 = mean of counties in community 0."""
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    merged = df.merge(sample_assignments, on="county_fips")
    expected_mean = merged[merged["community_id"] == 0][shift_cols[0]].mean()
    assert abs(profiles[0, 0] - expected_mean) < 1e-10


def test_fit_nmf_output_shape(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    result = fit_nmf(profiles, j=4, random_state=42)
    k = sample_assignments["community_id"].nunique()
    assert isinstance(result, NMFResult)
    assert result.W.shape == (k, 4)
    assert result.H.shape == (4, len(shift_cols))
    assert len(result.dominant_type) == k


def test_fit_nmf_weights_sum_to_one(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    result = fit_nmf(profiles, j=4, random_state=42)
    row_sums = result.W.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_dominant_type_is_argmax(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    result = fit_nmf(profiles, j=4, random_state=42)
    expected = np.argmax(result.W, axis=1)
    np.testing.assert_array_equal(result.dominant_type, expected)


def test_sweep_j_returns_ordered_results(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    sweep = sweep_j(profiles, j_values=[3, 4, 5], random_state=42)
    assert len(sweep) == 3
    assert [s.j for s in sweep] == [3, 4, 5]
    for s in sweep:
        assert s.reconstruction_error >= 0
