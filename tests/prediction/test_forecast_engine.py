"""Tests for the forecast engine: θ_prior → θ_national → δ_race → county predictions."""

import numpy as np
import pytest

from src.prediction.forecast_engine import (
    ForecastResult,
    build_W_state,
    compute_theta_prior,
    prepare_polls,
    run_forecast,
)


def test_theta_prior_shape():
    """θ_prior should have J elements."""
    J = 5
    n_counties = 10
    type_scores = np.random.rand(n_counties, J)
    type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
    county_priors = np.random.rand(n_counties) * 0.3 + 0.35  # ~[0.35, 0.65]
    theta = compute_theta_prior(type_scores, county_priors)
    assert theta.shape == (J,)


def test_theta_prior_weighted_average():
    """θ_prior[j] = weighted mean of county priors by type membership."""
    type_scores = np.array([
        [1.0, 0.0],  # county 0 is 100% type 0
        [0.0, 1.0],  # county 1 is 100% type 1
    ])
    county_priors = np.array([0.6, 0.4])
    theta = compute_theta_prior(type_scores, county_priors)
    np.testing.assert_allclose(theta, [0.6, 0.4])


def test_theta_prior_mixed_membership():
    """Mixed membership produces blended priors."""
    type_scores = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    county_priors = np.array([0.6, 0.4])
    theta = compute_theta_prior(type_scores, county_priors)
    np.testing.assert_allclose(theta, [0.5, 0.5])


def test_theta_prior_bounded():
    """θ_prior should be in [0, 1] (valid Dem share range)."""
    J = 20
    n_counties = 100
    type_scores = np.random.rand(n_counties, J)
    type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
    county_priors = np.random.rand(n_counties) * 0.6 + 0.2
    theta = compute_theta_prior(type_scores, county_priors)
    assert np.all(theta >= 0) and np.all(theta <= 1)


# --- Task 4: Orchestration tests ---


def test_build_W_state():
    """W for a state should be vote-weighted mean of county type memberships."""
    J = 3
    type_scores = np.array([
        [0.8, 0.1, 0.1],  # county 0 (state A)
        [0.2, 0.7, 0.1],  # county 1 (state A)
        [0.1, 0.1, 0.8],  # county 2 (state B)
    ])
    states = ["A", "A", "B"]
    votes = np.array([1000, 500, 800])
    W = build_W_state("A", type_scores, states, votes)
    # Vote-weighted: (1000*[0.8,0.1,0.1] + 500*[0.2,0.7,0.1]) / 1500
    expected = (1000 * np.array([0.8, 0.1, 0.1]) + 500 * np.array([0.2, 0.7, 0.1])) / 1500
    np.testing.assert_allclose(W, expected, atol=1e-6)


def test_run_forecast_no_polls():
    """With no polls, national and local modes should return prior-based predictions."""
    J = 3
    n_counties = 4
    type_scores = np.eye(J + 1, J)[:n_counties]  # Simple membership
    county_priors = np.array([0.6, 0.4, 0.5, 0.45])
    states = ["A", "A", "B", "B"]
    votes = np.array([100, 200, 150, 250])
    polls = {}  # No polls for any race
    races = ["2026 A Senate", "2026 B Governor"]

    result = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=votes,
        polls_by_race=polls,
        races=races,
        lam=1.0,
        mu=1.0,
    )
    assert "2026 A Senate" in result
    assert result["2026 A Senate"].theta_national is not None
    assert result["2026 A Senate"].delta_race is not None
    np.testing.assert_allclose(result["2026 A Senate"].delta_race, np.zeros(J))


def test_run_forecast_with_polls():
    """With polls, local mode predictions should differ from national mode."""
    J = 2
    type_scores = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])
    county_priors = np.array([0.55, 0.45])
    states = ["A", "A"]
    votes = np.array([100, 100])
    polls = {
        "2026 A Senate": [
            {"dem_share": 0.60, "n_sample": 800, "state": "A"},
        ],
    }
    races = ["2026 A Senate"]

    result = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=votes,
        polls_by_race=polls,
        races=races,
        lam=1.0,
        mu=1.0,
    )
    r = result["2026 A Senate"]
    # National and local should differ (δ != 0)
    national_preds = type_scores @ r.theta_national
    local_preds = type_scores @ (r.theta_national + r.delta_race)
    assert not np.allclose(national_preds, local_preds)


class TestForecastResult:
    def test_has_both_modes(self):
        """ForecastResult should expose national and local county predictions."""
        J = 2
        result = ForecastResult(
            theta_prior=np.array([0.5, 0.5]),
            theta_national=np.array([0.52, 0.48]),
            delta_race=np.array([0.01, -0.01]),
            county_preds_national=np.array([0.50, 0.49]),
            county_preds_local=np.array([0.51, 0.48]),
            n_polls=3,
        )
        assert result.n_polls == 3
        assert len(result.county_preds_national) == 2
        assert len(result.county_preds_local) == 2


class TestEnrichedForecast:
    def test_w_vector_mode_parameter(self):
        """run_forecast should accept w_vector_mode parameter."""
        J = 3
        n = 6
        type_scores = np.random.RandomState(42).rand(n, J)
        type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
        county_priors = np.full(n, 0.5)
        states = ["GA"] * 3 + ["FL"] * 3
        county_votes = np.ones(n)

        polls = {"2026 GA Senate": [
            {"dem_share": 0.53, "n_sample": 600, "state": "GA",
             "date": "2026-03-01", "pollster": "Test", "notes": "LV"},
        ]}

        result = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls,
            races=["2026 GA Senate"],
            w_vector_mode="core",
        )
        assert "2026 GA Senate" in result
        assert result["2026 GA Senate"].n_polls == 1

    def test_reference_date_parameter(self):
        """run_forecast should accept reference_date parameter and apply quality weighting."""
        J = 3
        n = 6
        type_scores = np.random.RandomState(7).rand(n, J)
        type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
        county_priors = np.full(n, 0.5)
        states = ["GA"] * 3 + ["FL"] * 3
        county_votes = np.ones(n)

        polls = {"2026 GA Senate": [
            {"dem_share": 0.53, "n_sample": 600, "state": "GA",
             "date": "2026-03-01", "pollster": "Test", "notes": "LV"},
        ]}

        # With reference_date, quality weighting is applied; should still return valid result
        result = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls,
            races=["2026 GA Senate"],
            reference_date="2026-03-29",
        )
        assert "2026 GA Senate" in result
        assert result["2026 GA Senate"].n_polls == 1

    def test_type_profiles_none_uses_state_w(self):
        """Without type_profiles, W vectors fall back to state-level build_W_state."""
        J = 3
        n = 4
        type_scores = np.random.RandomState(1).rand(n, J)
        type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
        county_priors = np.full(n, 0.5)
        states = ["TX"] * 2 + ["CA"] * 2
        county_votes = np.array([100.0, 200.0, 150.0, 300.0])

        polls = {"2026 TX Senate": [
            {"dem_share": 0.45, "n_sample": 500, "state": "TX",
             "date": "2026-03-15", "pollster": "Test", "notes": "LV"},
        ]}

        result = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls,
            races=["2026 TX Senate"],
            type_profiles=None,
        )
        assert "2026 TX Senate" in result
        assert result["2026 TX Senate"].n_polls == 1


class TestPreparePolls:
    def test_returns_adjusted_dicts(self):
        """prepare_polls should return dicts with adjusted dem_share and n_sample."""
        raw = {
            "2026 GA Senate": [
                {"dem_share": 0.53, "n_sample": 600, "state": "GA",
                 "date": "2026-03-01", "pollster": "Emerson College",
                 "notes": "LV"},
            ]
        }
        result = prepare_polls(raw, reference_date="2026-03-29")
        assert "2026 GA Senate" in result
        polls = result["2026 GA Senate"]
        assert len(polls) == 1
        p = polls[0]
        # Should still have required fields
        assert "dem_share" in p
        assert "n_sample" in p
        assert "state" in p
        # n_sample should be reduced by time decay (28 days with 30-day half-life)
        assert p["n_sample"] < 600

    def test_preserves_state_and_notes(self):
        """Metadata fields should survive the transformation."""
        raw = {
            "2026 GA Senate": [
                {"dem_share": 0.53, "n_sample": 600, "state": "GA",
                 "date": "2026-03-15", "pollster": "TestPollster",
                 "notes": "RV; src=test"},
            ]
        }
        result = prepare_polls(raw, reference_date="2026-03-29")
        p = result["2026 GA Senate"][0]
        assert p["state"] == "GA"
        assert "notes" in p

    def test_empty_input(self):
        result = prepare_polls({}, reference_date="2026-03-29")
        assert result == {}
