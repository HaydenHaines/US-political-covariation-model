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


def test_no_polls_prior_weight_blends_toward_type_baseline(mock_model):
    """Without polls, lower prior_weight blends county priors toward type baseline.

    The slider means 'how much to trust the ridge model refinement vs the raw
    type-mean baseline.'  At pw=1, predictions use ridge county priors.  At
    pw=0, predictions use type-weighted priors (the simpler baseline).
    """
    result_full = predict_race(race="test", prior_weight=1.0, **mock_model)
    result_half = predict_race(race="test", prior_weight=0.5, **mock_model)
    result_zero = predict_race(race="test", prior_weight=0.0, **mock_model)

    # pw=1.0 should match county_priors exactly (no polls → no shift)
    np.testing.assert_array_almost_equal(
        result_full["pred_dem_share"].values,
        mock_model["county_priors"],
    )
    # pw=0.0 should match type-weighted baseline (no ridge influence)
    scores = np.abs(mock_model["type_scores"])
    wsums = scores.sum(axis=1)
    type_baseline = (scores * mock_model["type_priors"][None, :]).sum(axis=1) / wsums
    np.testing.assert_array_almost_equal(
        result_zero["pred_dem_share"].values,
        type_baseline,
    )
    # pw=0.5 should be halfway between
    expected_half = 0.5 * mock_model["county_priors"] + 0.5 * type_baseline
    np.testing.assert_array_almost_equal(
        result_half["pred_dem_share"].values,
        expected_half,
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


def test_section_weights_model():
    """Verify SectionWeights model has correct defaults and validation."""
    from api.models import SectionWeights, MultiPollInput

    # Defaults
    sw = SectionWeights()
    assert sw.model_prior == 1.0
    assert sw.state_polls == 1.0
    assert sw.national_polls == 1.0

    # Custom values
    sw2 = SectionWeights(model_prior=0.5, state_polls=1.5)
    assert sw2.model_prior == 0.5
    assert sw2.state_polls == 1.5

    # Embedded in MultiPollInput with defaults
    mpi = MultiPollInput(cycle="2026", state="FL")
    assert mpi.section_weights.model_prior == 1.0

    # Embedded with custom weights
    mpi2 = MultiPollInput(
        cycle="2026", state="FL",
        section_weights=SectionWeights(model_prior=0.3),
    )
    assert mpi2.section_weights.model_prior == 0.3


def test_prior_weight_exact_zero_polls_dominate(mock_model):
    """With prior_weight=0.0 exactly, polls should completely dominate predictions.

    Regression test for bug where `0.0 > 0` was False, skipping covariance
    inflation and preventing polls from moving type means.
    """
    polls = [(0.60, 500, "AL")]
    result = predict_race(
        race="test", polls=polls, prior_weight=0.0, **mock_model,
    )
    mean_pred = result["pred_dem_share"].mean()
    # With pw=0 and a D+60 poll, the weighted state prediction should be
    # well above the type priors (~0.49) — closer to the poll than the prior.
    assert mean_pred > 0.55, (
        f"prior_weight=0 should let polls dominate, but mean pred={mean_pred:.3f}"
    )


def test_higher_prior_weight_anchors_closer(mock_model):
    """Higher model_prior weight should keep predictions closer to the prior."""
    polls = [(0.60, 500, "AL")]
    mean_prior = mock_model["county_priors"].mean()  # 0.45
    result_low = predict_race(race="test", polls=polls, prior_weight=0.3, **mock_model)
    result_high = predict_race(race="test", polls=polls, prior_weight=1.5, **mock_model)
    shift_low = abs(result_low["pred_dem_share"].mean() - mean_prior)
    shift_high = abs(result_high["pred_dem_share"].mean() - mean_prior)
    # Higher prior weight → less shift from baseline
    assert shift_high < shift_low


def test_offcycle_behavior_adjustment_shifts_prediction(mock_model):
    """Off-cycle behavior adjustment should shift predictions vs presidential baseline."""
    from src.behavior.voter_behavior import apply_behavior_adjustment

    priors = mock_model["county_priors"]
    scores = mock_model["type_scores"]
    tau = np.array([0.65, 0.85])
    delta = np.array([0.02, -0.01])

    adjusted = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    unadjusted = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=False)

    np.testing.assert_array_equal(unadjusted, priors)
    assert not np.allclose(adjusted, priors), "Behavior adjustment had no effect"


def test_offcycle_behavior_preserves_bounds(mock_model):
    """Behavior-adjusted predictions must stay in [0, 1]."""
    from src.behavior.voter_behavior import apply_behavior_adjustment

    priors = np.array([0.02, 0.98, 0.50])
    scores = mock_model["type_scores"]
    tau = np.array([0.5, 0.9])
    delta = np.array([0.05, -0.05])

    adjusted = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    assert (adjusted >= 0.0).all() and (adjusted <= 1.0).all()
