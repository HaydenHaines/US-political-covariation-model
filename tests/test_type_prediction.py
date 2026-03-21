"""Tests for src/prediction/predict_2026_types.py.

Tests the type-based prediction pipeline using synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.predict_2026_types import predict_race


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create synthetic type-based prediction inputs.

    293 counties, J=4 types, simple structure for verifiable predictions.
    """
    N = 293
    J = 4
    rng = np.random.RandomState(42)

    # County FIPS: 12xxx (FL), 13xxx (GA), 01xxx (AL)
    fl_fips = [f"12{i:03d}" for i in range(1, 168)]  # 167 FL counties
    ga_fips = [f"13{i:03d}" for i in range(1, 100)]   # 99 GA counties
    al_fips = [f"01{i:03d}" for i in range(1, 28)]    # 27 AL counties
    county_fips = fl_fips + ga_fips + al_fips
    assert len(county_fips) == N

    # Type scores: soft membership, can be negative
    type_scores = rng.randn(N, J) * 0.5
    # Make dominant types clear
    for i in range(N):
        dominant = i % J
        type_scores[i, dominant] += 2.0

    # Type covariance: positive definite J x J
    A = rng.randn(J, J) * 0.02
    type_covariance = A @ A.T + np.eye(J) * 0.001

    # Type priors: reasonable Dem shares
    type_priors = np.array([0.35, 0.55, 0.48, 0.42])

    # State abbreviations and county names
    states = (["FL"] * 167) + (["GA"] * 99) + (["AL"] * 27)
    county_names = [f"County_{f}" for f in county_fips]

    return {
        "county_fips": county_fips,
        "type_scores": type_scores,
        "type_covariance": type_covariance,
        "type_priors": type_priors,
        "states": states,
        "county_names": county_names,
        "N": N,
        "J": J,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_produces_county_rows(synthetic_data):
    """predict_race should produce one row per county (293 total)."""
    d = synthetic_data
    result = predict_race(
        race="FL Senate",
        poll_dem_share=None,
        poll_n=None,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    assert len(result) == d["N"]
    assert "county_fips" in result.columns


def test_predict_dem_share_bounded(synthetic_data):
    """All predicted Dem shares should be in [0, 1]."""
    d = synthetic_data
    result = predict_race(
        race="FL Senate",
        poll_dem_share=0.45,
        poll_n=800,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    assert (result["pred_dem_share"] >= 0).all()
    assert (result["pred_dem_share"] <= 1).all()


def test_predict_has_ci_columns(synthetic_data):
    """Output should include ci_lower and ci_upper columns."""
    d = synthetic_data
    result = predict_race(
        race="FL Senate",
        poll_dem_share=0.45,
        poll_n=800,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    assert "ci_lower" in result.columns
    assert "ci_upper" in result.columns


def test_predict_ci_ordered(synthetic_data):
    """ci_lower <= pred_dem_share <= ci_upper for all rows."""
    d = synthetic_data
    result = predict_race(
        race="FL Senate",
        poll_dem_share=0.45,
        poll_n=800,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    assert (result["ci_lower"] <= result["pred_dem_share"] + 1e-10).all()
    assert (result["pred_dem_share"] <= result["ci_upper"] + 1e-10).all()


def test_poll_shifts_predictions(synthetic_data):
    """Feeding a poll should change predictions compared to no poll."""
    d = synthetic_data
    no_poll = predict_race(
        race="FL Senate",
        poll_dem_share=None,
        poll_n=None,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    with_poll = predict_race(
        race="FL Senate",
        poll_dem_share=0.55,
        poll_n=800,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        state_filter="FL",
    )
    # Filter no_poll to FL for comparison
    no_poll_fl = no_poll[no_poll["state"] == "FL"]
    assert not np.allclose(
        no_poll_fl["pred_dem_share"].values,
        with_poll["pred_dem_share"].values,
    )


def test_state_filter(synthetic_data):
    """state_filter='FL' should return only FL counties."""
    d = synthetic_data
    result = predict_race(
        race="FL Senate",
        poll_dem_share=0.45,
        poll_n=800,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        state_filter="FL",
    )
    assert len(result) == 167  # FL counties
    assert (result["state"] == "FL").all()


def test_no_poll_uses_prior(synthetic_data):
    """When poll_dem_share is None, predictions should reflect priors."""
    d = synthetic_data
    result = predict_race(
        race="FL Senate",
        poll_dem_share=None,
        poll_n=None,
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    # Each county prediction = weighted average of type priors by scores
    for idx in range(min(5, len(result))):
        scores = d["type_scores"][idx]
        weights = np.abs(scores)
        expected = np.dot(weights, d["type_priors"]) / weights.sum()
        actual = result["pred_dem_share"].iloc[idx]
        np.testing.assert_allclose(actual, expected, atol=1e-10)
