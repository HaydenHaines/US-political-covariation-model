"""Tests for section weight scaling in forecast pipeline."""
import numpy as np
import pytest

from src.prediction.predict_2026_types import predict_race


@pytest.fixture
def mock_model():
    """Minimal 3-county, 2-type model for testing weight effects."""
    J = 2
    type_scores = np.array([
        [0.8, 0.2],  # county 0: mostly type 0
        [0.3, 0.7],  # county 1: mostly type 1
        [0.5, 0.5],  # county 2: mixed
    ])
    type_covariance = np.array([
        [0.01, 0.002],
        [0.002, 0.01],
    ])
    type_priors = np.array([0.55, 0.40])
    county_fips = ["01001", "01003", "01005"]
    states = ["AL", "AL", "AL"]
    county_names = ["Autauga", "Baldwin", "Barbour"]
    county_priors = np.array([0.52, 0.38, 0.45])
    return {
        "type_scores": type_scores,
        "type_covariance": type_covariance,
        "type_priors": type_priors,
        "county_fips": county_fips,
        "states": states,
        "county_names": county_names,
        "county_priors": county_priors,
    }


def test_prior_weight_zero_ignores_prior(mock_model):
    """With prior_weight near 0, polls should dominate completely."""
    polls = [(0.60, 500, "AL")]
    result_full = predict_race(
        race="test", polls=polls, prior_weight=1.0, **mock_model,
    )
    result_zero = predict_race(
        race="test", polls=polls, prior_weight=0.01, **mock_model,
    )
    # With near-zero prior weight, predictions should move further toward poll
    shift_full = abs(result_full["pred_dem_share"].mean() - 0.45)
    shift_zero = abs(result_zero["pred_dem_share"].mean() - 0.45)
    assert shift_zero > shift_full


def test_prior_weight_one_is_default(mock_model):
    """prior_weight=1.0 should produce identical results to no weight arg."""
    polls = [(0.60, 500, "AL")]
    result_default = predict_race(race="test", polls=polls, **mock_model)
    result_explicit = predict_race(
        race="test", polls=polls, prior_weight=1.0, **mock_model,
    )
    np.testing.assert_array_almost_equal(
        result_default["pred_dem_share"].values,
        result_explicit["pred_dem_share"].values,
    )


def test_no_polls_prior_weight_irrelevant(mock_model):
    """Without polls, prior_weight doesn't change output (no update to scale)."""
    result_a = predict_race(race="test", prior_weight=0.5, **mock_model)
    result_b = predict_race(race="test", prior_weight=1.0, **mock_model)
    np.testing.assert_array_almost_equal(
        result_a["pred_dem_share"].values,
        result_b["pred_dem_share"].values,
    )


def test_poll_n_scaling(mock_model):
    """Section weight < 1 should reduce poll influence (smaller effective N)."""
    polls_full = [(0.60, 500, "AL")]
    polls_scaled = [(0.60, 250, "AL")]  # 500 * 0.5 = 250
    result_full = predict_race(race="test", polls=polls_full, **mock_model)
    result_scaled = predict_race(race="test", polls=polls_scaled, **mock_model)
    # Halved N should move predictions less toward 0.60
    shift_full = abs(result_full["pred_dem_share"].mean() - 0.45)
    shift_scaled = abs(result_scaled["pred_dem_share"].mean() - 0.45)
    assert shift_full > shift_scaled
