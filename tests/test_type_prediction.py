"""Tests for src/prediction/predict_2026_types.py.

Tests the type-based prediction pipeline using synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.predict_2026_types import (
    compute_county_priors_from_data,
    predict_race,
)


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
        polls=[(0.45, 800, "FL")],
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
        polls=[(0.45, 800, "FL")],
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
        polls=[(0.45, 800, "FL")],
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
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    with_poll = predict_race(
        race="FL Senate",
        polls=[(0.55, 800, "FL")],
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
        polls=[(0.45, 800, "FL")],
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
    """When polls is None and no county_priors, predictions should reflect type priors."""
    d = synthetic_data
    result = predict_race(
        race="FL Senate",
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


# ---------------------------------------------------------------------------
# County-level prior tests
# ---------------------------------------------------------------------------


@pytest.fixture
def county_prior_data():
    """Create synthetic data with county-level priors for testing.

    Two counties in the same type but with very different baselines:
    - County A (Black Belt): 0.80 Dem share
    - County B (rural white): 0.25 Dem share
    Both assigned to the same type (type 0).
    """
    N = 4
    J = 2
    # All 4 counties, 2 types
    county_fips = ["12001", "12002", "13001", "13002"]
    states = ["FL", "FL", "GA", "GA"]
    county_names = ["County_A", "County_B", "County_C", "County_D"]

    # Type scores: counties 0,1 in type 0; counties 2,3 in type 1
    type_scores = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8],
    ])

    # Type covariance
    type_covariance = np.array([[0.01, 0.005], [0.005, 0.01]])

    # Type priors (average of their counties)
    type_priors = np.array([0.525, 0.40])

    # County-level priors: very different baselines within same type
    county_priors = np.array([0.80, 0.25, 0.45, 0.35])

    return {
        "county_fips": county_fips,
        "type_scores": type_scores,
        "type_covariance": type_covariance,
        "type_priors": type_priors,
        "county_priors": county_priors,
        "states": states,
        "county_names": county_names,
        "N": N,
        "J": J,
    }


def test_county_prior_no_poll_preserves_baseline(county_prior_data):
    """With county priors and no poll, prediction should equal county baseline.

    When there is no poll, the Bayesian update produces zero type shift,
    so county_pred = county_prior + 0 = county_prior.
    """
    d = county_prior_data
    result = predict_race(
        race="FL Senate",
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=d["county_priors"],
    )
    # With no poll, type_shift = 0, so pred = county_prior exactly
    np.testing.assert_allclose(
        result["pred_dem_share"].values,
        d["county_priors"],
        atol=1e-10,
    )


def test_county_prior_differentiates_same_type_counties(county_prior_data):
    """Counties in the same type should get different predictions when using county priors."""
    d = county_prior_data
    result = predict_race(
        race="FL Senate",
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=d["county_priors"],
    )
    # Counties 0 and 1 are both in type 0 but have priors of 0.80 vs 0.25
    assert abs(result["pred_dem_share"].iloc[0] - result["pred_dem_share"].iloc[1]) > 0.4


def test_county_prior_type_mean_gives_identical_predictions(county_prior_data):
    """Without county priors (legacy), same-type counties get nearly identical predictions."""
    d = county_prior_data
    result = predict_race(
        race="FL Senate",
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        # No county_priors -> legacy type-mean approach
    )
    # Counties 0 and 1 in same type should get similar predictions (within ~0.05)
    diff = abs(result["pred_dem_share"].iloc[0] - result["pred_dem_share"].iloc[1])
    assert diff < 0.1, f"Legacy type-mean predictions too different: {diff:.3f}"


def test_county_prior_poll_shifts_all_counties(county_prior_data):
    """A poll should shift all same-state counties from their baselines."""
    d = county_prior_data
    no_poll = predict_race(
        race="FL Senate",
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=d["county_priors"],
    )
    with_poll = predict_race(
        race="FL Senate",
        polls=[(0.60, 1000, "FL")],
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=d["county_priors"],
        state_filter="FL",
    )
    no_poll_fl = no_poll[no_poll["state"] == "FL"]
    # Poll should shift predictions
    assert not np.allclose(
        no_poll_fl["pred_dem_share"].values,
        with_poll["pred_dem_share"].values,
    )


def test_county_prior_bounded(county_prior_data):
    """County-prior predictions should be clipped to [0, 1]."""
    d = county_prior_data
    # Use extreme county priors to test clipping
    extreme_priors = np.array([0.99, 0.01, 0.50, 0.50])
    result = predict_race(
        race="FL Senate",
        polls=[(0.90, 10000, "FL")],
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=extreme_priors,
    )
    assert (result["pred_dem_share"] >= 0).all()
    assert (result["pred_dem_share"] <= 1).all()


def test_county_prior_ci_ordered(county_prior_data):
    """CIs should be ordered: ci_lower <= pred <= ci_upper."""
    d = county_prior_data
    result = predict_race(
        race="FL Senate",
        polls=[(0.45, 800, "FL")],
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=d["county_priors"],
    )
    assert (result["ci_lower"] <= result["pred_dem_share"] + 1e-10).all()
    assert (result["pred_dem_share"] <= result["ci_upper"] + 1e-10).all()


def test_compute_county_priors_from_data():
    """compute_county_priors_from_data should use mapping with fallback."""
    fips = ["12001", "12002", "99999"]
    dem_map = {"12001": 0.55, "12002": 0.40}
    result = compute_county_priors_from_data(fips, dem_map, fallback=0.45)
    np.testing.assert_allclose(result, [0.55, 0.40, 0.45])


def test_compute_county_priors_from_data_empty_map():
    """All fallback when map is empty."""
    fips = ["12001", "12002"]
    result = compute_county_priors_from_data(fips, {}, fallback=0.50)
    np.testing.assert_allclose(result, [0.50, 0.50])


def test_multi_poll_stacking(county_prior_data):
    """Two polls from different states should produce different result than one poll alone."""
    d = county_prior_data
    # Single FL poll
    single = predict_race(
        race="FL Senate",
        polls=[(0.60, 1000, "FL")],
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=d["county_priors"],
    )
    # Same FL poll plus a GA poll — should produce different county predictions
    # because the GA poll provides an additional, potentially contradictory signal
    # that shifts type means further before mapping back to counties.
    multi = predict_race(
        race="FL Senate",
        polls=[(0.60, 1000, "FL"), (0.40, 1000, "GA")],
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=d["county_priors"],
    )
    # Adding a contradictory GA poll should change predictions
    assert not np.allclose(
        single["pred_dem_share"].values,
        multi["pred_dem_share"].values,
    )


def test_county_prior_backward_compat(synthetic_data):
    """Without county_priors, predict_race behaves identically to legacy."""
    d = synthetic_data
    result_legacy = predict_race(
        race="FL Senate",
        polls=[(0.45, 800, "FL")],
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
        county_priors=None,
    )
    result_no_arg = predict_race(
        race="FL Senate",
        polls=[(0.45, 800, "FL")],
        type_scores=d["type_scores"],
        type_covariance=d["type_covariance"],
        type_priors=d["type_priors"],
        county_fips=d["county_fips"],
        states=d["states"],
        county_names=d["county_names"],
    )
    np.testing.assert_allclose(
        result_legacy["pred_dem_share"].values,
        result_no_arg["pred_dem_share"].values,
        atol=1e-10,
    )
