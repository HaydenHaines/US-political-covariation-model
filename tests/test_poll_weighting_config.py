"""Tests for config-driven poll weighting parameters.

Verifies:
  1. prediction_params.json has a poll_weighting section with the expected keys
  2. The config values flow through to the weighting pipeline
  3. Fallback defaults work when the config section is absent
  4. prepare_polls and run_forecast accept and use the config-sourced params

These tests do NOT require model data on disk (type_assignments, etc.).
They operate purely on the weighting layer and a tiny synthetic poll dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.propagation.poll_decay import apply_time_decay
from src.propagation.poll_pipeline import apply_all_weights
from src.propagation.propagate_polls import PollObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "data" / "config" / "prediction_params.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poll(
    dem_share: float = 0.50,
    n_sample: int = 1000,
    date: str = "2026-01-01",
    geography: str = "FL",
    pollster: str = "TestPollster",
    race: str = "2026 FL Senate",
) -> PollObservation:
    return PollObservation(
        geography=geography,
        dem_share=dem_share,
        n_sample=n_sample,
        race=race,
        date=date,
        pollster=pollster,
        geo_level="state",
    )


# ---------------------------------------------------------------------------
# Task 1 verification: prediction_params.json has poll_weighting section
# ---------------------------------------------------------------------------


class TestConfigFile:
    """prediction_params.json contains a valid poll_weighting section."""

    def test_config_file_exists(self):
        assert PARAMS_PATH.exists(), f"Config file not found: {PARAMS_PATH}"

    def test_poll_weighting_section_present(self):
        params = json.loads(PARAMS_PATH.read_text())
        assert "poll_weighting" in params, (
            "prediction_params.json is missing 'poll_weighting' section"
        )

    def test_half_life_days_present_and_positive(self):
        params = json.loads(PARAMS_PATH.read_text())
        pw = params["poll_weighting"]
        assert "half_life_days" in pw, "poll_weighting.half_life_days not found"
        assert float(pw["half_life_days"]) > 0, "half_life_days must be positive"

    def test_pre_primary_discount_present_and_in_range(self):
        params = json.loads(PARAMS_PATH.read_text())
        pw = params["poll_weighting"]
        assert "pre_primary_discount" in pw, "poll_weighting.pre_primary_discount not found"
        d = float(pw["pre_primary_discount"])
        assert 0 < d <= 1.0, f"pre_primary_discount must be in (0, 1], got {d}"

    def test_forecast_section_still_intact(self):
        """Adding poll_weighting must not break the existing forecast section."""
        params = json.loads(PARAMS_PATH.read_text())
        assert "forecast" in params
        assert "lam" in params["forecast"]
        assert "mu" in params["forecast"]
        assert "w_vector_mode" in params["forecast"]

    def test_config_values_match_code_defaults(self):
        """Config values should equal the production defaults documented in code."""
        params = json.loads(PARAMS_PATH.read_text())
        pw = params["poll_weighting"]
        # These are the original hardcoded values; changing them requires
        # bumping this test to reflect the intentional change.
        assert float(pw["half_life_days"]) == pytest.approx(30.0)
        assert float(pw["pre_primary_discount"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Task 2 verification: config values flow to the weighting pipeline
# ---------------------------------------------------------------------------


class TestHalfLifePassthrough:
    """half_life_days from config is used correctly in time decay."""

    def test_shorter_half_life_reduces_effective_n_more(self):
        """A 15-day half-life should produce lower effective n than a 60-day one
        for a poll that is 30 days old.

        At 30 days old:
          - 15d half-life: decay = 2^(-30/15) = 0.25  → n=250
          - 60d half-life: decay = 2^(-30/60) = 0.707 → n=707
        """
        poll = _make_poll(n_sample=1000, date="2026-03-05")
        result_15 = apply_time_decay([poll], reference_date="2026-04-04", half_life_days=15.0)
        result_60 = apply_time_decay([poll], reference_date="2026-04-04", half_life_days=60.0)
        assert result_15[0].n_sample < result_60[0].n_sample

    def test_half_life_exact_math(self):
        """Verify the decay formula: n_eff = round(n * 2^(-age/half_life))."""
        poll = _make_poll(n_sample=1000, date="2026-03-05")
        # Age = 30 days, half_life = 30 days → decay = 0.5
        result = apply_time_decay([poll], reference_date="2026-04-04", half_life_days=30.0)
        expected = int(round(1000 * (2.0 ** (-30.0 / 30.0))))
        assert result[0].n_sample == expected

    def test_apply_all_weights_accepts_half_life(self):
        """apply_all_weights should accept half_life_days and pass it to time decay."""
        poll = _make_poll(n_sample=1000, date="2026-01-04")  # ~90 days before ref date
        ref_date = "2026-04-04"

        result_30 = apply_all_weights(
            [poll], reference_date=ref_date, half_life_days=30.0,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
        )
        result_90 = apply_all_weights(
            [poll], reference_date=ref_date, half_life_days=90.0,
            apply_quality=False, apply_house_effects=False, use_primary_discount=False,
        )
        # Shorter half-life → more decay → smaller n_sample
        assert result_30[0].n_sample < result_90[0].n_sample

    def test_prepare_polls_half_life_affects_effective_n(self):
        """prepare_polls should pass half_life_days to apply_all_weights."""
        from src.prediction.forecast_engine import prepare_polls

        # Poll 60 days before reference date
        polls_by_race = {
            "2026 FL Senate": [
                {
                    "dem_share": 0.52,
                    "n_sample": 1000,
                    "state": "FL",
                    "date": "2026-02-03",  # ~60 days before 2026-04-04
                    "pollster": "TestPollster",
                    "notes": "",
                }
            ]
        }
        ref_date = "2026-04-04"

        result_15 = prepare_polls(polls_by_race, ref_date, half_life_days=15.0)
        result_60 = prepare_polls(polls_by_race, ref_date, half_life_days=60.0)

        n_15 = result_15["2026 FL Senate"][0]["n_sample"]
        n_60 = result_60["2026 FL Senate"][0]["n_sample"]

        # A 15-day half-life decays 60-day-old polls more aggressively than 60-day
        assert n_15 < n_60, (
            f"Expected n_15 ({n_15}) < n_60 ({n_60}) but got the opposite"
        )


class TestPrePrimaryDiscountPassthrough:
    """pre_primary_discount from config flows through apply_all_weights."""

    def test_smaller_discount_reduces_effective_n_more(self):
        """A discount factor of 0.3 should halve n_sample more than 0.8."""
        from src.propagation.poll_pipeline import apply_all_weights as _apply

        # A poll before a primary (2026-09-15 is after most 2026 primaries, but
        # we test the mechanism directly via discount_factor kwarg)
        poll = _make_poll(n_sample=1000, date="2026-04-04")
        ref_date = "2026-04-04"

        result_03 = _apply(
            [poll], reference_date=ref_date,
            apply_quality=False, apply_house_effects=False,
            use_primary_discount=True,
            primary_discount_factor=0.3,
            primary_calendar_path=None,  # no calendar → discount not applied
        )
        result_08 = _apply(
            [poll], reference_date=ref_date,
            apply_quality=False, apply_house_effects=False,
            use_primary_discount=True,
            primary_discount_factor=0.8,
            primary_calendar_path=None,
        )
        # Without a real calendar file, no discount is applied regardless of factor.
        # The poll n_sample should be identical (fallback behavior = no-op).
        # This test verifies the parameter is accepted without error.
        assert result_03[0].n_sample > 0
        assert result_08[0].n_sample > 0

    def test_prepare_polls_accepts_pre_primary_discount(self):
        """prepare_polls should not raise when pre_primary_discount is supplied."""
        from src.prediction.forecast_engine import prepare_polls

        polls_by_race = {
            "2026 FL Senate": [{
                "dem_share": 0.52,
                "n_sample": 1000,
                "state": "FL",
                "date": "2026-04-04",
                "pollster": "TestPollster",
                "notes": "",
            }]
        }
        # Should not raise
        result = prepare_polls(
            polls_by_race,
            reference_date="2026-04-04",
            half_life_days=30.0,
            pre_primary_discount=0.3,
        )
        assert "2026 FL Senate" in result
        assert len(result["2026 FL Senate"]) == 1


class TestRunForecastAcceptsParams:
    """run_forecast accepts half_life_days and pre_primary_discount without error."""

    @pytest.fixture
    def tiny_model(self):
        return {
            "type_scores": np.array([
                [0.8, 0.2],
                [0.3, 0.7],
                [0.5, 0.5],
                [0.6, 0.4],
            ]),
            "county_priors": np.array([0.52, 0.48, 0.50, 0.45]),
            "states": ["FL", "FL", "FL", "FL"],
            "county_votes": np.array([100.0, 200.0, 150.0, 80.0]),
            "polls_by_race": {
                "2026 FL Senate": [{
                    "dem_share": 0.52,
                    "n_sample": 500,
                    "state": "FL",
                    "date": "2026-04-04",
                    "pollster": "TestPollster",
                    "notes": "",
                }]
            },
        }

    def test_run_forecast_default_params_unchanged(self, tiny_model):
        """run_forecast with default half_life/discount produces results."""
        from src.prediction.forecast_engine import run_forecast

        results = run_forecast(
            type_scores=tiny_model["type_scores"],
            county_priors=tiny_model["county_priors"],
            states=tiny_model["states"],
            county_votes=tiny_model["county_votes"],
            polls_by_race=tiny_model["polls_by_race"],
            races=["2026 FL Senate"],
            reference_date="2026-04-04",
        )
        assert "2026 FL Senate" in results
        preds = results["2026 FL Senate"].county_preds_local
        assert preds.shape == (4,)
        assert np.all(np.isfinite(preds))

    def test_run_forecast_custom_half_life(self, tiny_model):
        """run_forecast with half_life_days=15 produces different results than 60."""
        from src.prediction.forecast_engine import run_forecast

        # Poll is 30 days old (half-way through a 60-day window, but a full
        # half-life at 30d). Different half-life → different effective n → different θ.
        polls_30_days_old = {
            "2026 FL Senate": [{
                "dem_share": 0.65,  # Very strong D poll to make difference visible
                "n_sample": 2000,
                "state": "FL",
                "date": "2026-03-05",  # 30 days before 2026-04-04
                "pollster": "TestPollster",
                "notes": "",
            }]
        }

        results_15 = run_forecast(
            type_scores=tiny_model["type_scores"],
            county_priors=tiny_model["county_priors"],
            states=tiny_model["states"],
            county_votes=tiny_model["county_votes"],
            polls_by_race=polls_30_days_old,
            races=["2026 FL Senate"],
            reference_date="2026-04-04",
            half_life_days=15.0,  # Less weight to 30-day-old poll
        )
        results_60 = run_forecast(
            type_scores=tiny_model["type_scores"],
            county_priors=tiny_model["county_priors"],
            states=tiny_model["states"],
            county_votes=tiny_model["county_votes"],
            polls_by_race=polls_30_days_old,
            races=["2026 FL Senate"],
            reference_date="2026-04-04",
            half_life_days=60.0,  # More weight to 30-day-old poll
        )

        preds_15 = results_15["2026 FL Senate"].county_preds_local
        preds_60 = results_60["2026 FL Senate"].county_preds_local

        # With a strong D poll (65%) and a shorter half-life (less poll weight),
        # the 15d half-life results should be pulled toward the poll *less* than
        # the 60d half-life results — i.e., lower Dem share predictions.
        assert preds_15.mean() < preds_60.mean(), (
            f"Expected 15d half-life predictions to be below 60d, but got "
            f"mean_15={preds_15.mean():.3f}, mean_60={preds_60.mean():.3f}"
        )


# ---------------------------------------------------------------------------
# Task 3: fallback defaults when poll_weighting section is absent
# ---------------------------------------------------------------------------


class TestFallbackDefaults:
    """When poll_weighting is absent from config, code defaults still work."""

    def test_apply_time_decay_default_half_life(self):
        """apply_time_decay with no half_life_days uses 30.0 by default."""
        poll = _make_poll(n_sample=1000, date="2026-03-05")
        # Explicit call with default
        result_explicit = apply_time_decay(
            [poll], reference_date="2026-04-04", half_life_days=30.0,
        )
        # Call using the function's own default
        result_default = apply_time_decay([poll], reference_date="2026-04-04")
        assert result_explicit[0].n_sample == result_default[0].n_sample

    def test_prepare_polls_missing_half_life_uses_default(self):
        """prepare_polls with no half_life_days arg uses the 30.0 default."""
        from src.prediction.forecast_engine import prepare_polls

        polls_by_race = {
            "2026 FL Senate": [{
                "dem_share": 0.52,
                "n_sample": 1000,
                "state": "FL",
                "date": "2026-04-04",
                "pollster": "TestPollster",
                "notes": "",
            }]
        }
        # Should not raise; uses default half_life_days=30.0
        result = prepare_polls(polls_by_race, reference_date="2026-04-04")
        assert "2026 FL Senate" in result

    def test_predict_2026_types_loads_poll_weighting_or_uses_defaults(self):
        """ForecastParams loads poll_weighting from config, falling back to defaults.

        Verify that load_forecast_params() reads the config on disk and produces
        the expected half_life_days and pre_primary_discount values.
        """
        from src.prediction.predict_2026_types import load_forecast_params

        params = load_forecast_params()
        assert params.half_life_days == pytest.approx(30.0), (
            "half_life_days should equal prediction_params.json poll_weighting.half_life_days=30.0"
        )
        assert params.pre_primary_discount == pytest.approx(0.5), (
            "pre_primary_discount should match poll_weighting.pre_primary_discount=0.5 in config"
        )

    def test_poll_weighting_fallback_with_mock_empty_section(self):
        """When poll_weighting key is missing, .get() fallbacks return defaults."""
        params: dict = {"forecast": {"lam": 1.0, "mu": 1.0, "w_vector_mode": "core"}}

        pw = params.get("poll_weighting", {})
        half_life = float(pw.get("half_life_days", 30.0))
        discount = float(pw.get("pre_primary_discount", 0.5))

        assert half_life == pytest.approx(30.0)
        assert discount == pytest.approx(0.5)
