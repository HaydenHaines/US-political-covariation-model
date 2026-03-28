"""Tests for voter behavior layer: turnout ratio (τ) and choice shift (δ)."""
import numpy as np
import pandas as pd
import pytest

from src.behavior.voter_behavior import compute_turnout_ratios, compute_choice_shifts, apply_behavior_adjustment


@pytest.fixture
def mock_tract_data():
    """Minimal tract data: 4 tracts, 2 types, presidential + off-cycle."""
    tract_votes = pd.DataFrame({
        "tract_geoid": (["T1", "T2", "T3", "T4"] * 4),
        "year": ([2020] * 4 + [2024] * 4 + [2018] * 4 + [2022] * 4),
        "race": (["president"] * 4 + ["president"] * 4 +
                 ["governor"] * 4 + ["governor"] * 4),
        "votes_total": ([1000, 800, 600, 400] + [1000, 800, 600, 400] +
                        [700, 500, 500, 350] + [720, 520, 480, 340]),
        "votes_dem": ([500, 300, 400, 100] + [500, 300, 400, 100] +
                      [380, 200, 320, 80] + [400, 220, 300, 75]),
        "dem_share": None,
        "state": ["AL", "AL", "GA", "GA"] * 4,
    })
    tract_votes["dem_share"] = np.where(
        tract_votes["votes_total"] > 0,
        tract_votes["votes_dem"] / tract_votes["votes_total"],
        np.nan,
    )

    type_scores = pd.DataFrame({
        "GEOID": ["T1", "T2", "T3", "T4"],
        "type_0_score": [0.8, 0.7, 0.2, 0.1],
        "type_1_score": [0.2, 0.3, 0.8, 0.9],
    }).set_index("GEOID")

    return tract_votes, type_scores


def test_turnout_ratios_shape(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert tau.shape == (2,)


def test_turnout_ratios_less_than_one(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert (tau < 1.0).all(), f"Expected τ < 1, got {tau}"
    assert (tau > 0.0).all()


def test_turnout_ratios_vary_by_type(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert tau[0] != tau[1]


def test_choice_shift_shape(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    delta = compute_choice_shifts(votes, scores, tau, n_types=2)
    assert delta.shape == (2,)


def test_choice_shift_bounded(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    delta = compute_choice_shifts(votes, scores, tau, n_types=2)
    assert (np.abs(delta) < 0.2).all(), f"δ too large: {delta}"


def test_behavior_adjustment_noop_for_presidential():
    priors = np.array([0.45, 0.55, 0.50])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    tau = np.array([0.65, 0.85])
    delta = np.array([0.02, -0.01])
    result = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=False)
    np.testing.assert_array_equal(result, priors)


def test_behavior_adjustment_changes_for_offcycle():
    priors = np.array([0.45, 0.55, 0.50])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    tau = np.array([0.65, 0.85])
    delta = np.array([0.02, -0.01])
    result = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    assert not np.allclose(result, priors)


def test_behavior_adjustment_bounded():
    priors = np.array([0.02, 0.98, 0.50])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    tau = np.array([0.5, 0.9])
    delta = np.array([0.05, -0.05])
    result = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    assert (result >= 0.0).all() and (result <= 1.0).all()
