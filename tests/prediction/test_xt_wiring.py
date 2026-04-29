"""Tests for xt_* demographic data wiring into the forecast prediction pipeline.

Covers three behaviours:
1. _extract_raw_demographics maps xt_* poll keys to type_profiles column names.
2. Polls that carry xt_* data produce Tier 1 W vectors that differ from the
   plain state-level W produced by polls without any xt_* data.
3. Polls without xt_* data still succeed (Tier 3 fallback).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.forecast_engine import _extract_raw_demographics, run_forecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_type_profiles(J: int = 4) -> pd.DataFrame:
    """Return a minimal type_profiles DataFrame with the columns touched by the
    xt_ mapping.  Values are chosen so that types 0/1 have high college share
    and types 2/3 have low college share — this creates a meaningful signal
    when a poll reports xt_education_college.
    """
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "pct_bachelors_plus": [0.55, 0.60, 0.20, 0.18],
        "pct_white_nh":       [0.70, 0.65, 0.75, 0.80],
        "pct_black":          [0.08, 0.12, 0.10, 0.06],
        "pct_hispanic":       [0.10, 0.08, 0.07, 0.09],
        "pct_asian":          [0.06, 0.09, 0.03, 0.02],
        "median_age":         [38.0, 36.0, 42.0, 44.0],
        "log_pop_density":    [6.5, 7.0, 4.0, 3.5],
        "evangelical_share":  [0.15, 0.10, 0.30, 0.35],
    })


def _make_type_scores(n_counties: int = 6, J: int = 4) -> np.ndarray:
    rng = np.random.RandomState(1)
    raw = rng.rand(n_counties, J)
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# _extract_raw_demographics
# ---------------------------------------------------------------------------


class TestExtractRawDemographics:
    def test_returns_none_for_poll_without_xt(self):
        """A standard poll dict with no xt_ keys should produce None."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA"}
        result = _extract_raw_demographics(poll)
        assert result is None

    def test_maps_education_college_to_pct_bachelors_plus(self):
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_education_college": 0.45}
        result = _extract_raw_demographics(poll)
        assert result is not None
        assert "pct_bachelors_plus" in result
        assert result["pct_bachelors_plus"] == pytest.approx(0.45)

    def test_maps_race_white_to_pct_white_nh(self):
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_race_white": 0.70}
        result = _extract_raw_demographics(poll)
        assert result is not None
        assert "pct_white_nh" in result
        assert result["pct_white_nh"] == pytest.approx(0.70)

    def test_maps_race_black(self):
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_race_black": 0.20}
        result = _extract_raw_demographics(poll)
        assert result is not None
        assert "pct_black" in result

    def test_maps_race_hispanic(self):
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_race_hispanic": 0.15}
        result = _extract_raw_demographics(poll)
        assert result is not None
        assert "pct_hispanic" in result

    def test_maps_age_senior_to_median_age(self):
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_age_senior": 0.18}
        result = _extract_raw_demographics(poll)
        assert result is not None
        assert "median_age" in result

    def test_noncollege_is_skipped_no_duplicate(self):
        """xt_education_noncollege maps to None in the dimension map and should
        be silently dropped rather than appearing as a key collision."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_education_college": 0.45,
                "xt_education_noncollege": 0.55}
        result = _extract_raw_demographics(poll)
        assert result is not None
        # noncollege should NOT appear as a key — it has no direct column
        assert "pct_bachelors_plus" in result
        # Only one key from the two education fields
        assert len([k for k in result if "bachelors" in k or "college" in k]) == 1

    def test_multiple_xt_fields_all_mapped(self):
        """All mappable xt_ fields should appear in the output dict."""
        poll = {
            "dem_share": 0.52, "n_sample": 600, "state": "GA",
            "xt_education_college": 0.45,
            "xt_race_white": 0.68,
            "xt_race_black": 0.15,
            "xt_race_hispanic": 0.10,
            "xt_age_senior": 0.17,
        }
        result = _extract_raw_demographics(poll)
        assert result is not None
        assert len(result) == 5  # one entry per mappable field

    def test_returns_floats(self):
        """Values in the output dict must be plain Python floats."""
        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA",
                "xt_education_college": 0.45}
        result = _extract_raw_demographics(poll)
        assert result is not None
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# W vector differentiation — xt_ vs no-xt_ polls
# ---------------------------------------------------------------------------


class TestXtWVectorDifferentiation:
    """Polls with xt_* data should produce different W vectors than polls
    that only use the state-level fallback."""

    def _run_single_poll(self, poll: dict, type_profiles: pd.DataFrame,
                         type_scores: np.ndarray, states: list[str]) -> np.ndarray:
        """Run a single-poll forecast and return theta_national."""
        n = len(states)
        county_priors = np.full(n, 0.5)
        county_votes = np.ones(n)
        result = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race={"test_race": [poll]},
            races=["test_race"],
            lam=0.5,
            mu=0.5,
            type_profiles=type_profiles,
        )
        return result["test_race"].theta_national

    def test_xt_poll_differs_from_no_xt_poll(self):
        """A poll with xt_education_college should produce a different theta_national
        than the same poll without any xt_ data, because the W vector is skewed
        toward high-college types.
        """
        J = 4
        type_profiles = _make_type_profiles(J)
        type_scores = _make_type_scores(J=J)
        states = ["GA"] * 3 + ["FL"] * 3

        base_poll = {"dem_share": 0.55, "n_sample": 800, "state": "GA"}
        xt_poll = {**base_poll, "xt_education_college": 0.70}  # high-college sample

        theta_base = self._run_single_poll(base_poll, type_profiles, type_scores, states)
        theta_xt = self._run_single_poll(xt_poll, type_profiles, type_scores, states)

        # The W vectors should be different, producing different theta estimates.
        assert not np.allclose(theta_base, theta_xt), (
            "Expected xt_ poll to produce a different theta_national than the "
            "plain state-level poll, but they were identical."
        )

    def test_poll_without_xt_still_succeeds(self):
        """Polls without xt_ data must not crash and must return valid predictions."""
        J = 4
        type_profiles = _make_type_profiles(J)
        type_scores = _make_type_scores(J=J)
        states = ["GA"] * 3 + ["FL"] * 3

        poll = {"dem_share": 0.52, "n_sample": 600, "state": "GA"}
        theta = self._run_single_poll(poll, type_profiles, type_scores, states)

        assert theta.shape == (J,)
        assert np.all(np.isfinite(theta))

    def test_xt_theta_is_valid(self):
        """theta_national from an xt_ poll must have finite, reasonable values."""
        J = 4
        type_profiles = _make_type_profiles(J)
        type_scores = _make_type_scores(J=J)
        states = ["GA"] * 3 + ["FL"] * 3

        poll = {
            "dem_share": 0.55, "n_sample": 800, "state": "GA",
            "xt_education_college": 0.60,
            "xt_race_white": 0.65,
        }
        theta = self._run_single_poll(poll, type_profiles, type_scores, states)

        assert theta.shape == (J,)
        assert np.all(np.isfinite(theta))
        # Dem share around 0.55 — theta values should stay in plausible range
        assert np.all(theta > 0.0) and np.all(theta < 1.0)

    def test_two_polls_same_dem_share_different_composition(self):
        """Two polls with the same dem_share but different xt_ composition
        should produce different W-weighted theta estimates.
        """
        J = 4
        type_profiles = _make_type_profiles(J)
        type_scores = _make_type_scores(J=J)
        states = ["GA"] * 3 + ["FL"] * 3

        # High-college sample
        poll_college = {
            "dem_share": 0.55, "n_sample": 800, "state": "GA",
            "xt_education_college": 0.70,
        }
        # Low-college sample (different composition, same top-line share)
        poll_noncollege = {
            "dem_share": 0.55, "n_sample": 800, "state": "GA",
            "xt_education_college": 0.20,
        }

        theta_college = self._run_single_poll(poll_college, type_profiles, type_scores, states)
        theta_noncollege = self._run_single_poll(poll_noncollege, type_profiles, type_scores, states)

        # Same dem_share but different sample composition → different theta
        assert not np.allclose(theta_college, theta_noncollege), (
            "Same dem_share but different xt_education_college should produce "
            "different theta_national vectors."
        )

    def test_xt_signal_survives_prepare_polls_when_reference_date_set(self):
        """The production path weights polls before building W vectors.

        A poll enriched with xt_* metadata should still differ from the same
        poll stripped to toplines after prepare_polls runs inside run_forecast.
        """
        J = 4
        type_profiles = _make_type_profiles(J)
        type_scores = _make_type_scores(J=J)
        states = ["GA"] * 3 + ["FL"] * 3
        county_priors = np.full(len(states), 0.5)
        county_votes = np.ones(len(states))

        base_poll = {
            "dem_share": 0.55,
            "n_sample": 800,
            "state": "GA",
            "date": "2026-03-15",
            "pollster": "TestPollster",
            "notes": "",
            "methodology": "mixed",
        }
        enriched_poll = {
            **base_poll,
            "xt_education_college": 0.70,
            "xt_vote_education_college": 0.64,
            "xt_race_black": 0.20,
            "xt_vote_race_black": 0.78,
        }

        stripped = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race={"test_race": [base_poll]},
            races=["test_race"],
            lam=0.5,
            mu=0.5,
            type_profiles=type_profiles,
            reference_date="2026-03-29",
        )["test_race"].theta_national
        enriched = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race={"test_race": [enriched_poll]},
            races=["test_race"],
            lam=0.5,
            mu=0.5,
            type_profiles=type_profiles,
            reference_date="2026-03-29",
        )["test_race"].theta_national

        assert not np.allclose(enriched, stripped), (
            "Expected xt-enriched poll to differ from stripped topline poll "
            "after prepare_polls, but the metadata was lost before W-building."
        )


# ---------------------------------------------------------------------------
# Tier fallback behaviour
# ---------------------------------------------------------------------------


class TestTierFallback:
    def test_no_type_profiles_still_works(self):
        """When type_profiles is None, xt_ fields should be ignored gracefully and
        the forecast should complete without error (plain W state fallback).
        """
        J = 3
        n = 4
        type_scores = np.eye(J + 1, J)[:n]
        county_priors = np.full(n, 0.5)
        states = ["TX", "TX", "CA", "CA"]
        county_votes = np.array([100.0, 200.0, 150.0, 300.0])

        polls = {"2026 TX Senate": [
            {"dem_share": 0.48, "n_sample": 700, "state": "TX",
             "xt_education_college": 0.42, "xt_race_white": 0.72},
        ]}

        result = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls,
            races=["2026 TX Senate"],
            type_profiles=None,  # No type_profiles — xt_ data is ignored
        )
        assert "2026 TX Senate" in result
        assert result["2026 TX Senate"].n_polls == 1
        assert np.all(np.isfinite(result["2026 TX Senate"].theta_national))
