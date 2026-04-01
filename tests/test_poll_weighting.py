"""Tests for poll weighting: time decay, pollster quality, and aggregation."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.propagation.propagate_polls import PollObservation
from src.propagation.poll_weighting import (
    _HE_DEM_SHARE_MAX,
    _HE_DEM_SHARE_MIN,
    _PRE_PRIMARY_DISCOUNT,
    _SB_MAX_MULTIPLIER,
    _SB_MIN_MULTIPLIER,
    _sb_score_to_multiplier,
    aggregate_polls,
    apply_all_weights,
    apply_house_effect_correction,
    apply_pollster_quality,
    apply_primary_discount,
    apply_time_decay,
    election_day_for_cycle,
    extract_grade_from_notes,
    grade_to_multiplier,
    load_polls_with_notes,
    reset_house_effect_cache,
    reset_sb_cache,
)


def _make_poll(
    dem_share: float = 0.50,
    n_sample: int = 1000,
    date: str = "2020-11-03",
    geography: str = "FL",
    pollster: str = "TestPollster",
    race: str = "2020 FL President",
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
# Time decay
# ---------------------------------------------------------------------------


class TestTimeDecay:
    def test_recent_poll_unchanged(self):
        """A poll from reference_date should have decay ~1.0."""
        poll = _make_poll(date="2020-11-03", n_sample=1000)
        result = apply_time_decay([poll], reference_date="2020-11-03")
        assert len(result) == 1
        assert result[0].n_sample == 1000

    def test_old_poll_reduced(self):
        """A poll 60 days old with half_life=30 -> n ~= n/4."""
        poll = _make_poll(date="2020-09-04", n_sample=1000)
        result = apply_time_decay([poll], reference_date="2020-11-03", half_life_days=30.0)
        # 60 days / 30 half-life = 2 half-lives -> decay = 0.25 -> n ~250
        assert result[0].n_sample == pytest.approx(250, abs=10)

    def test_half_life(self):
        """A poll exactly half_life old -> n ~= n/2."""
        poll = _make_poll(date="2020-10-04", n_sample=1000)
        result = apply_time_decay([poll], reference_date="2020-11-03", half_life_days=30.0)
        assert result[0].n_sample == pytest.approx(500, abs=10)

    def test_preserves_other_fields(self):
        """Geography, dem_share, pollster etc should be unchanged."""
        poll = _make_poll(
            date="2020-10-03",
            dem_share=0.48,
            geography="GA",
            pollster="ABC",
            race="GA Senate",
        )
        result = apply_time_decay([poll], reference_date="2020-11-03")
        assert result[0].geography == "GA"
        assert result[0].dem_share == 0.48
        assert result[0].pollster == "ABC"
        assert result[0].race == "GA Senate"

    def test_returns_copies(self):
        """Original polls should not be modified."""
        poll = _make_poll(date="2020-09-04", n_sample=1000)
        result = apply_time_decay([poll], reference_date="2020-11-03")
        assert poll.n_sample == 1000  # unchanged
        assert result[0].n_sample < 1000  # reduced

    def test_minimum_n_one(self):
        """Very old polls should have n_sample >= 1."""
        poll = _make_poll(date="2019-01-01", n_sample=100)
        result = apply_time_decay([poll], reference_date="2020-11-03")
        assert result[0].n_sample >= 1

    def test_no_date_unchanged(self):
        """Polls with no date should pass through unchanged."""
        poll = _make_poll(date="", n_sample=500)
        result = apply_time_decay([poll], reference_date="2020-11-03")
        assert result[0].n_sample == 500

    def test_future_poll_no_decay(self):
        """Polls after reference date get no decay."""
        poll = _make_poll(date="2020-12-01", n_sample=1000)
        result = apply_time_decay([poll], reference_date="2020-11-03")
        assert result[0].n_sample == 1000


# ---------------------------------------------------------------------------
# Pollster quality
# ---------------------------------------------------------------------------


class TestPollsterQuality:
    """Tests for 538-grade-based pollster quality (Silver Bulletin disabled)."""

    def test_a_plus_boost(self):
        """A+ grade should boost n_sample."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality(
            [poll], poll_notes=["grade=3.0"], use_silver_bulletin=False
        )
        assert result[0].n_sample > 1000  # 1.2x
        assert result[0].n_sample == 1200

    def test_d_grade_reduction(self):
        """D grade should reduce n to ~30%."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality(
            [poll], poll_notes=["grade=0.2"], use_silver_bulletin=False
        )
        assert result[0].n_sample == 300

    def test_no_grade_default(self):
        """No grade in notes -> 0.8x."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality(
            [poll], poll_notes=["method=Live Phone"], use_silver_bulletin=False
        )
        assert result[0].n_sample == 800

    def test_no_notes_default(self):
        """No notes at all -> 0.8x."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality([poll], poll_notes=None, use_silver_bulletin=False)
        assert result[0].n_sample == 800

    def test_extracts_grade_from_notes(self):
        """Parses 'grade=2.5' from semicolon-delimited notes."""
        notes = "method=Online Panel; rating_id=588; grade=2.5; pollscore=0.2; bias=0.1"
        grade = extract_grade_from_notes(notes)
        assert grade == "A"  # 2.4-2.7 maps to A

    def test_b_grade_multiplier(self):
        """B grade (numeric ~1.5-1.9) -> 0.9x."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality(
            [poll], poll_notes=["grade=1.5"], use_silver_bulletin=False
        )
        assert result[0].n_sample == 900

    def test_custom_multipliers(self):
        """Custom grade multiplier table should be respected."""
        poll = _make_poll(n_sample=1000)
        custom = {"A": 2.0}
        result = apply_pollster_quality(
            [poll],
            poll_notes=["grade=2.5"],
            grade_multipliers=custom,
            use_silver_bulletin=False,
        )
        assert result[0].n_sample == 2000


# ---------------------------------------------------------------------------
# Combined weighting
# ---------------------------------------------------------------------------


class TestApplyAllWeights:
    def test_combines_both(self):
        """apply_all_weights should apply both time decay and quality."""
        # Poll 30 days old (half_life=30 -> 0.5 decay), grade A (1.1x multiplier)
        # Silver Bulletin disabled so 538 grade from notes drives quality.
        poll = _make_poll(date="2020-10-04", n_sample=1000)
        result = apply_all_weights(
            [poll],
            reference_date="2020-11-03",
            half_life_days=30.0,
            poll_notes=["grade=2.5"],
            apply_quality=True,
            use_silver_bulletin=False,
        )
        # Time decay: 1000 * 0.5 = 500
        # Quality (A = 1.1): 500 * 1.1 = 550 -> but quality runs on decayed n
        # Actually: time decay first -> 500, then quality on 500 -> 500 * 1.1 = 550
        # But quality runs on the already-decayed n_sample, and grade=2.5 -> A -> 1.1x
        # So: round(500 * 1.1) = 550
        # Wait, time decay gives round(1000 * 0.5) = 500, then quality gives round(500 * 1.1) = 550
        assert result[0].n_sample == pytest.approx(550, abs=15)

    def test_quality_disabled(self):
        """apply_quality=False should skip pollster quality."""
        poll = _make_poll(date="2020-10-04", n_sample=1000)
        result = apply_all_weights(
            [poll],
            reference_date="2020-11-03",
            half_life_days=30.0,
            poll_notes=["grade=0.2"],  # D grade
            apply_quality=False,
        )
        # Only time decay: 30 days at half_life=30 -> 0.5 -> 500
        assert result[0].n_sample == pytest.approx(500, abs=10)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregatePolls:
    def test_single_poll_identity(self):
        """One poll -> same share and ~same n."""
        poll = _make_poll(dem_share=0.48, n_sample=800)
        share, n = aggregate_polls([poll])
        assert share == pytest.approx(0.48, abs=0.001)
        assert n == pytest.approx(800, abs=5)

    def test_equal_polls_average(self):
        """Two identical polls -> same share, ~2x n."""
        p1 = _make_poll(dem_share=0.50, n_sample=500)
        p2 = _make_poll(dem_share=0.50, n_sample=500)
        share, n = aggregate_polls([p1, p2])
        assert share == pytest.approx(0.50, abs=0.001)
        assert n == pytest.approx(1000, abs=20)

    def test_precise_poll_dominates(self):
        """Large-N poll should dominate small-N poll in the average."""
        small = _make_poll(dem_share=0.60, n_sample=100)
        large = _make_poll(dem_share=0.40, n_sample=10000)
        share, n = aggregate_polls([small, large])
        # Should be very close to 0.40 (the large poll's value)
        assert share == pytest.approx(0.40, abs=0.01)

    def test_empty_raises(self):
        """Empty poll list should raise ValueError."""
        with pytest.raises(ValueError, match="No polls"):
            aggregate_polls([])

    def test_different_shares_weighted_average(self):
        """Two polls with different shares and equal N -> midpoint."""
        p1 = _make_poll(dem_share=0.40, n_sample=1000)
        p2 = _make_poll(dem_share=0.60, n_sample=1000)
        share, n = aggregate_polls([p1, p2])
        # Should be close to 0.50 (equal weight)
        assert share == pytest.approx(0.50, abs=0.01)

    def test_combined_n_larger_than_individual(self):
        """Combined effective N should be larger than any individual poll."""
        p1 = _make_poll(dem_share=0.48, n_sample=500)
        p2 = _make_poll(dem_share=0.52, n_sample=700)
        _, n = aggregate_polls([p1, p2])
        assert n > 500


# ---------------------------------------------------------------------------
# Grade extraction helpers
# ---------------------------------------------------------------------------


class TestGradeExtraction:
    def test_extract_grade_numeric(self):
        assert extract_grade_from_notes("grade=2.9") == "A+"
        assert extract_grade_from_notes("grade=2.5") == "A"
        assert extract_grade_from_notes("grade=2.0") == "A/B"
        assert extract_grade_from_notes("grade=1.5") == "B"
        assert extract_grade_from_notes("grade=1.0") == "B/C"
        assert extract_grade_from_notes("grade=0.5") == "C"
        assert extract_grade_from_notes("grade=0.3") == "C/D"
        assert extract_grade_from_notes("grade=0.1") == "D"

    def test_extract_no_grade(self):
        assert extract_grade_from_notes("method=IVR") is None
        assert extract_grade_from_notes("") is None
        assert extract_grade_from_notes(None) is None

    def test_grade_in_middle_of_notes(self):
        notes = "method=Live Phone; grade=1.8; pollscore=-1.0"
        assert extract_grade_from_notes(notes) == "B"


class TestElectionDay:
    def test_known_cycles(self):
        assert election_day_for_cycle("2020") == "2020-11-03"
        assert election_day_for_cycle("2022") == "2022-11-08"

    def test_unknown_cycle_default(self):
        assert election_day_for_cycle("2030") == "2030-11-03"


# ---------------------------------------------------------------------------
# Silver Bulletin integration
# ---------------------------------------------------------------------------


class TestSbScoreToMultiplier:
    """Unit tests for the score→multiplier rescaling helper."""

    def test_score_zero_gives_min(self):
        assert _sb_score_to_multiplier(0.0) == pytest.approx(_SB_MIN_MULTIPLIER)

    def test_score_one_gives_max(self):
        assert _sb_score_to_multiplier(1.0) == pytest.approx(_SB_MAX_MULTIPLIER)

    def test_midpoint(self):
        expected = (_SB_MIN_MULTIPLIER + _SB_MAX_MULTIPLIER) / 2
        assert _sb_score_to_multiplier(0.5) == pytest.approx(expected)

    def test_monotone_increasing(self):
        assert _sb_score_to_multiplier(0.3) < _sb_score_to_multiplier(0.7)


class TestSilverBulletinUsed:
    """apply_pollster_quality should use Silver Bulletin when XLSX is present."""

    def setup_method(self):
        reset_sb_cache()

    def teardown_method(self):
        reset_sb_cache()

    def test_sb_quality_used_when_available(self):
        """When Silver Bulletin returns a quality score, it drives the multiplier."""
        poll = _make_poll(n_sample=1000, pollster="Emerson College")

        # Patch get_pollster_quality to return a known score (A = 0.93)
        with patch(
            "src.propagation.poll_quality._get_sb_quality",
            return_value=_sb_score_to_multiplier(0.93),
        ):
            result = apply_pollster_quality([poll], use_silver_bulletin=True)

        expected_multiplier = _sb_score_to_multiplier(0.93)
        expected_n = int(max(1, round(1000 * expected_multiplier)))
        assert result[0].n_sample == expected_n

    def test_sb_high_quality_boosts_n(self):
        """A+ pollster (score=1.0) should produce n_sample > original."""
        poll = _make_poll(n_sample=1000, pollster="NYT/Siena")

        with patch(
            "src.propagation.poll_quality._get_sb_quality",
            return_value=_sb_score_to_multiplier(1.0),
        ):
            result = apply_pollster_quality([poll], use_silver_bulletin=True)

        assert result[0].n_sample > 1000

    def test_sb_banned_pollster_low_weight(self):
        """Banned pollster (score=0.0) should produce minimum weight."""
        poll = _make_poll(n_sample=1000, pollster="Research America (banned)")

        with patch(
            "src.propagation.poll_quality._get_sb_quality",
            return_value=_sb_score_to_multiplier(0.0),
        ):
            result = apply_pollster_quality([poll], use_silver_bulletin=True)

        expected_n = int(max(1, round(1000 * _SB_MIN_MULTIPLIER)))
        assert result[0].n_sample == expected_n

    def test_sb_unknown_pollster_neutral(self):
        """Unknown pollster (score=0.5) should give a mid-range multiplier."""
        poll = _make_poll(n_sample=1000, pollster="Unknown Outfit")
        neutral_multiplier = _sb_score_to_multiplier(0.5)

        with patch(
            "src.propagation.poll_quality._get_sb_quality",
            return_value=neutral_multiplier,
        ):
            result = apply_pollster_quality([poll], use_silver_bulletin=True)

        expected_n = int(max(1, round(1000 * neutral_multiplier)))
        assert result[0].n_sample == expected_n


class TestSilverBulletinFallback:
    """When Silver Bulletin XLSX is missing, fall back to 538 grade from notes."""

    def setup_method(self):
        reset_sb_cache()

    def teardown_method(self):
        reset_sb_cache()

    def test_fallback_to_538_when_sb_unavailable(self):
        """FileNotFoundError from SB loader → fall back to notes-based 538 grade."""
        poll = _make_poll(n_sample=1000, pollster="SomePollster")

        # Silver Bulletin raises FileNotFoundError → None returned by _get_sb_quality
        with patch(
            "src.propagation.poll_quality._get_sb_quality",
            return_value=None,
        ):
            # notes carry a 538 A-grade (numeric 2.5 → "A" → 1.1x)
            result = apply_pollster_quality(
                [poll],
                poll_notes=["grade=2.5"],
                use_silver_bulletin=True,
            )

        # 538 A grade = 1.1x
        assert result[0].n_sample == 1100

    def test_sb_disabled_uses_538_grade(self):
        """use_silver_bulletin=False should bypass SB entirely and use 538 grades."""
        poll = _make_poll(n_sample=1000, pollster="TestPollster")
        result = apply_pollster_quality(
            [poll],
            poll_notes=["grade=2.5"],  # A grade → 1.1x
            use_silver_bulletin=False,
        )
        assert result[0].n_sample == 1100

    def test_fallback_no_grade_in_notes_uses_default(self):
        """No SB, no grade in notes → 0.8x default."""
        poll = _make_poll(n_sample=1000, pollster="TestPollster")

        with patch(
            "src.propagation.poll_quality._get_sb_quality",
            return_value=None,
        ):
            result = apply_pollster_quality(
                [poll],
                poll_notes=["method=IVR"],
                use_silver_bulletin=True,
            )

        assert result[0].n_sample == 800

    def test_apply_all_weights_sb_disabled_uses_notes(self):
        """apply_all_weights with use_silver_bulletin=False uses 538 grade."""
        poll = _make_poll(date="2020-11-03", n_sample=1000)
        result = apply_all_weights(
            [poll],
            reference_date="2020-11-03",
            poll_notes=["grade=3.0"],  # A+ → 1.2x; no time decay (same day)
            apply_quality=True,
            use_silver_bulletin=False,
        )
        assert result[0].n_sample == 1200

    def test_apply_all_weights_sb_enabled_uses_sb(self):
        """apply_all_weights propagates use_silver_bulletin=True to quality step."""
        poll = _make_poll(date="2020-11-03", n_sample=1000, pollster="Emerson College")
        # SB returns score=1.0 → multiplier = _SB_MAX_MULTIPLIER = 1.2
        with patch(
            "src.propagation.poll_quality._get_sb_quality",
            return_value=_SB_MAX_MULTIPLIER,
        ):
            result = apply_all_weights(
                [poll],
                reference_date="2020-11-03",
                apply_quality=True,
                use_silver_bulletin=True,
                apply_house_effects=False,  # isolate quality test from house effects
            )
        assert result[0].n_sample == int(max(1, round(1000 * _SB_MAX_MULTIPLIER)))


# ---------------------------------------------------------------------------
# House effect correction
# ---------------------------------------------------------------------------


class TestHouseEffectCorrection:
    """Tests for apply_house_effect_correction."""

    def setup_method(self):
        reset_house_effect_cache()

    def teardown_method(self):
        reset_house_effect_cache()

    # --- Basic correction direction and magnitude ---

    def test_positive_house_effect_reduces_dem_share(self):
        """Positive house effect (D-biased pollster) lowers dem_share."""
        poll = _make_poll(dem_share=0.50, pollster="Biased Left Pollster")
        # +2 pp house effect: 0.50 - 0.02 = 0.48
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"Biased Left Pollster": 2.0},
            bias_538={},
        )
        assert result[0].dem_share == pytest.approx(0.48, abs=1e-6)

    def test_negative_house_effect_raises_dem_share(self):
        """Negative house effect (R-biased pollster) raises dem_share."""
        poll = _make_poll(dem_share=0.50, pollster="Biased Right Pollster")
        # -3 pp house effect: 0.50 - (-0.03) = 0.53
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"Biased Right Pollster": -3.0},
            bias_538={},
        )
        assert result[0].dem_share == pytest.approx(0.53, abs=1e-6)

    def test_zero_house_effect_leaves_dem_share_unchanged(self):
        """Zero house effect should not modify dem_share."""
        poll = _make_poll(dem_share=0.52, pollster="Unbiased Pollster")
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"Unbiased Pollster": 0.0},
            bias_538={},
        )
        assert result[0].dem_share == pytest.approx(0.52, abs=1e-6)

    # --- Source priority ---

    def test_sb_takes_priority_over_538(self):
        """Silver Bulletin house effect takes priority over 538 bias_ppm."""
        poll = _make_poll(dem_share=0.50, pollster="Priority Pollster")
        # SB says +5 pp; 538 says +1 pp — SB should win
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"Priority Pollster": 5.0},
            bias_538={"Priority Pollster": 1.0},
        )
        assert result[0].dem_share == pytest.approx(0.45, abs=1e-6)

    def test_fallback_to_538_when_sb_absent(self):
        """When pollster not in SB, 538 bias_ppm should be used."""
        poll = _make_poll(dem_share=0.50, pollster="FiveThirtyEight Only Pollster")
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={"FiveThirtyEight Only Pollster": 2.0},
        )
        assert result[0].dem_share == pytest.approx(0.48, abs=1e-6)

    def test_no_correction_when_pollster_unknown(self):
        """Unknown pollster (not in either source) → dem_share unchanged."""
        poll = _make_poll(dem_share=0.55, pollster="Completely Unknown Org ZZZZZ")
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
        )
        assert result[0].dem_share == pytest.approx(0.55, abs=1e-6)

    # --- Clamping ---

    def test_clamp_at_upper_bound(self):
        """Result above _HE_DEM_SHARE_MAX should be clamped."""
        poll = _make_poll(dem_share=0.98, pollster="R Biased Extreme")
        # -5 pp house effect: 0.98 + 0.05 = 1.03 → clamp to 0.99
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"R Biased Extreme": -5.0},
            bias_538={},
        )
        assert result[0].dem_share == _HE_DEM_SHARE_MAX

    def test_clamp_at_lower_bound(self):
        """Result below _HE_DEM_SHARE_MIN should be clamped."""
        poll = _make_poll(dem_share=0.02, pollster="D Biased Extreme")
        # +5 pp house effect: 0.02 - 0.05 = -0.03 → clamp to 0.01
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"D Biased Extreme": 5.0},
            bias_538={},
        )
        assert result[0].dem_share == _HE_DEM_SHARE_MIN

    # --- Return value properties ---

    def test_returns_copies_not_originals(self):
        """Original polls should be unchanged after correction."""
        poll = _make_poll(dem_share=0.50, pollster="Test Pollster")
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"Test Pollster": 2.0},
            bias_538={},
        )
        assert poll.dem_share == 0.50  # original unchanged
        assert result[0].dem_share != 0.50  # copy was modified

    def test_n_sample_unchanged_by_correction(self):
        """House effect correction adjusts dem_share, not n_sample."""
        poll = _make_poll(dem_share=0.50, n_sample=1234, pollster="Test Pollster")
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={"Test Pollster": 1.5},
            bias_538={},
        )
        assert result[0].n_sample == 1234

    def test_empty_pollster_skipped(self):
        """Polls with no pollster name should pass through unchanged."""
        poll = _make_poll(dem_share=0.50, n_sample=500, pollster="")
        result = apply_house_effect_correction(
            [poll],
            sb_house_effects={},
            bias_538={},
        )
        assert result[0].dem_share == pytest.approx(0.50, abs=1e-6)


class TestHouseEffectsInPipeline:
    """Integration: house effect correction wired into apply_all_weights."""

    def setup_method(self):
        reset_house_effect_cache()
        reset_sb_cache()

    def teardown_method(self):
        reset_house_effect_cache()
        reset_sb_cache()

    def test_house_effects_applied_before_time_decay(self):
        """Correction is on dem_share, unaffected by time decay (which adjusts n_sample)."""
        # Poll same day as reference → no time decay; A+ grade → 1.2x quality
        poll = _make_poll(date="2020-11-03", dem_share=0.50, n_sample=1000, pollster="Biased Pollster")
        result = apply_all_weights(
            [poll],
            reference_date="2020-11-03",
            poll_notes=["grade=3.0"],  # A+ → 1.2x quality
            apply_quality=True,
            use_silver_bulletin=False,
            apply_house_effects=True,
            # Inject a known +2 pp house effect
        )
        # We need to inject the bias; patch the cache directly
        reset_house_effect_cache()
        with patch("src.propagation.house_effects._SB_HOUSE_EFFECTS", {"Biased Pollster": 2.0}), \
             patch("src.propagation.house_effects._538_BIAS", {}):
            result = apply_all_weights(
                [poll],
                reference_date="2020-11-03",
                poll_notes=["grade=3.0"],
                apply_quality=True,
                use_silver_bulletin=False,
                apply_house_effects=True,
            )
        # dem_share corrected: 0.50 - 0.02 = 0.48
        assert result[0].dem_share == pytest.approx(0.48, abs=1e-4)

    def test_house_effects_disabled_leaves_dem_share(self):
        """apply_house_effects=False bypasses correction entirely."""
        poll = _make_poll(date="2020-11-03", dem_share=0.50, n_sample=1000, pollster="Biased Pollster")
        with patch("src.propagation.house_effects._SB_HOUSE_EFFECTS", {"Biased Pollster": 5.0}), \
             patch("src.propagation.house_effects._538_BIAS", {}):
            result = apply_all_weights(
                [poll],
                reference_date="2020-11-03",
                apply_quality=False,
                apply_house_effects=False,
            )
        assert result[0].dem_share == pytest.approx(0.50, abs=1e-6)


class TestLoadHouseEffects:
    """Tests for the loaders in silver_bulletin_ratings."""

    def test_sb_house_effects_loads_dict(self):
        """load_pollster_house_effects returns a non-empty dict."""
        from src.assembly.silver_bulletin_ratings import load_pollster_house_effects
        he = load_pollster_house_effects()
        assert isinstance(he, dict)
        assert len(he) > 50

    def test_sb_house_effects_are_floats(self):
        """All house effect values should be floats."""
        from src.assembly.silver_bulletin_ratings import load_pollster_house_effects
        he = load_pollster_house_effects()
        for name, val in he.items():
            assert isinstance(val, float), f"Non-float house effect for {name!r}: {val!r}"

    def test_538_bias_loads_dict(self):
        """load_538_bias returns a non-empty dict."""
        from src.assembly.silver_bulletin_ratings import load_538_bias
        bias = load_538_bias()
        assert isinstance(bias, dict)
        assert len(bias) > 50

    def test_538_bias_are_floats(self):
        """All 538 bias values should be floats."""
        from src.assembly.silver_bulletin_ratings import load_538_bias
        bias = load_538_bias()
        for name, val in bias.items():
            assert isinstance(val, float), f"Non-float bias for {name!r}: {val!r}"

    def test_sb_file_not_found_raises(self, tmp_path):
        """Missing XLSX raises FileNotFoundError."""
        from src.assembly.silver_bulletin_ratings import load_pollster_house_effects
        with pytest.raises(FileNotFoundError):
            load_pollster_house_effects(tmp_path / "missing.xlsx")

    def test_538_file_not_found_raises(self, tmp_path):
        """Missing CSV raises FileNotFoundError."""
        from src.assembly.silver_bulletin_ratings import load_538_bias
        with pytest.raises(FileNotFoundError):
            load_538_bias(tmp_path / "missing.csv")


# ---------------------------------------------------------------------------
# Pre/post-primary discount
# ---------------------------------------------------------------------------


def _write_calendar(tmp_path: Path, rows: list[str]) -> Path:
    """Write a primary calendar CSV to tmp_path and return its path."""
    path = tmp_path / "primary_calendar_2026.csv"
    path.write_text("state,race_type,primary_date\n" + "\n".join(rows) + "\n")
    return path


class TestApplyPrimaryDiscount:
    """Unit tests for apply_primary_discount."""

    def test_pre_primary_poll_discounted(self, tmp_path):
        """A poll before the primary date gets n_sample reduced by discount_factor."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(
            n_sample=1000,
            date="2026-03-01",  # well before the Aug primary
            race="2026 FL Senate",
        )
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        # Default factor=0.5 -> 1000 * 0.5 = 500
        assert result[0].n_sample == 500

    def test_post_primary_poll_unchanged(self, tmp_path):
        """A poll after the primary date is not discounted."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(
            n_sample=1000,
            date="2026-09-01",  # after the Aug primary
            race="2026 FL Senate",
        )
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert result[0].n_sample == 1000

    def test_poll_on_primary_day_unchanged(self, tmp_path):
        """A poll conducted exactly on the primary date is not discounted (not before)."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(
            n_sample=1000,
            date="2026-08-18",
            race="2026 FL Senate",
        )
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert result[0].n_sample == 1000

    def test_missing_calendar_entry_no_change(self, tmp_path):
        """A poll with no matching calendar entry is unchanged."""
        # Calendar has FL Senate; poll is for GA Senate — no match
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(
            n_sample=1000,
            date="2026-03-01",
            race="2026 GA Senate",
        )
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert result[0].n_sample == 1000

    def test_missing_calendar_file_no_change(self, tmp_path):
        """When the calendar file does not exist, all polls pass through unchanged."""
        poll = _make_poll(n_sample=1000, date="2026-03-01", race="2026 FL Senate")
        nonexistent = tmp_path / "no_such_file.csv"
        result = apply_primary_discount([poll], primary_calendar_path=nonexistent)
        assert result[0].n_sample == 1000

    def test_custom_discount_factor(self, tmp_path):
        """The discount factor is configurable."""
        cal = _write_calendar(tmp_path, ["GA,Senate,2026-05-19"])
        poll = _make_poll(
            n_sample=1000,
            date="2026-01-15",
            race="2026 GA Senate",
        )
        result = apply_primary_discount([poll], primary_calendar_path=cal, discount_factor=0.25)
        assert result[0].n_sample == 250

    def test_governor_race_discounted(self, tmp_path):
        """Governor races are handled identically to Senate races."""
        cal = _write_calendar(tmp_path, ["OH,Governor,2026-05-05"])
        poll = _make_poll(
            n_sample=800,
            date="2026-02-01",
            race="2026 OH Governor",
        )
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert result[0].n_sample == 400  # 800 * 0.5

    def test_returns_copies_not_originals(self, tmp_path):
        """Original poll objects are not mutated."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(n_sample=1000, date="2026-03-01", race="2026 FL Senate")
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert poll.n_sample == 1000  # original unchanged
        assert result[0].n_sample == 500  # copy discounted

    def test_dem_share_unchanged_by_discount(self, tmp_path):
        """Discounting adjusts n_sample only; dem_share is preserved."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(
            dem_share=0.47,
            n_sample=1000,
            date="2026-03-01",
            race="2026 FL Senate",
        )
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert result[0].dem_share == pytest.approx(0.47)

    def test_poll_with_no_race_unchanged(self, tmp_path):
        """Polls with empty race field are passed through without modification."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(n_sample=1000, date="2026-03-01", race="")
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert result[0].n_sample == 1000

    def test_poll_with_no_date_unchanged(self, tmp_path):
        """Polls with empty date field are passed through without modification."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(n_sample=1000, date="", race="2026 FL Senate")
        result = apply_primary_discount([poll], primary_calendar_path=cal)
        assert result[0].n_sample == 1000

    def test_invalid_discount_factor_raises(self, tmp_path):
        """discount_factor outside (0, 1] raises ValueError."""
        poll = _make_poll(n_sample=1000, date="2026-03-01", race="2026 FL Senate")
        with pytest.raises(ValueError):
            apply_primary_discount([poll], discount_factor=0.0)
        with pytest.raises(ValueError):
            apply_primary_discount([poll], discount_factor=1.5)

    def test_minimum_n_one_after_discount(self, tmp_path):
        """Very small n_sample is clamped to at least 1 after discounting."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(n_sample=1, date="2026-03-01", race="2026 FL Senate")
        result = apply_primary_discount([poll], primary_calendar_path=cal, discount_factor=0.1)
        assert result[0].n_sample >= 1

    def test_multiple_races_mixed_pre_post_primary(self, tmp_path):
        """Mix of pre and post-primary polls: each gets appropriate treatment."""
        cal = _write_calendar(tmp_path, [
            "FL,Senate,2026-08-18",
            "GA,Senate,2026-05-19",
        ])
        pre_fl = _make_poll(n_sample=1000, date="2026-03-01", race="2026 FL Senate")
        post_fl = _make_poll(n_sample=1000, date="2026-09-01", race="2026 FL Senate")
        pre_ga = _make_poll(n_sample=800, date="2026-02-01", race="2026 GA Senate")
        post_ga = _make_poll(n_sample=800, date="2026-06-01", race="2026 GA Senate")

        result = apply_primary_discount(
            [pre_fl, post_fl, pre_ga, post_ga], primary_calendar_path=cal
        )
        assert result[0].n_sample == 500   # pre FL → discounted
        assert result[1].n_sample == 1000  # post FL → unchanged
        assert result[2].n_sample == 400   # pre GA → discounted
        assert result[3].n_sample == 800   # post GA → unchanged


class TestPrimaryDiscountInPipeline:
    """Integration: primary discount wired into apply_all_weights."""

    def test_pre_primary_discount_applied_in_pipeline(self, tmp_path):
        """apply_all_weights with use_primary_discount=True discounts pre-primary polls."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        # Poll on reference_date (no time decay), no quality (Silver Bulletin off, no grade),
        # no house effects — isolates the primary discount step.
        poll = _make_poll(
            n_sample=1000,
            date="2026-03-01",
            race="2026 FL Senate",
        )
        result = apply_all_weights(
            [poll],
            reference_date="2026-03-01",
            apply_house_effects=False,
            use_primary_discount=True,
            primary_calendar_path=cal,
            primary_discount_factor=0.5,
            apply_quality=False,
        )
        assert result[0].n_sample == 500

    def test_primary_discount_disabled_in_pipeline(self, tmp_path):
        """apply_all_weights with use_primary_discount=False skips the discount step."""
        cal = _write_calendar(tmp_path, ["FL,Senate,2026-08-18"])
        poll = _make_poll(
            n_sample=1000,
            date="2026-03-01",
            race="2026 FL Senate",
        )
        result = apply_all_weights(
            [poll],
            reference_date="2026-03-01",
            apply_house_effects=False,
            use_primary_discount=False,
            primary_calendar_path=cal,
            apply_quality=False,
        )
        assert result[0].n_sample == 1000

    def test_primary_discount_uses_default_calendar(self):
        """apply_all_weights without explicit path uses the project primary calendar."""
        # This exercises the default path resolution. The production calendar file
        # should exist; if it does, a 2026 FL Senate poll from early 2026 gets discounted.
        from pathlib import Path
        default_cal = (
            Path(__file__).parents[1] / "data" / "polls" / "primary_calendar_2026.csv"
        )
        if not default_cal.exists():
            pytest.skip("Primary calendar not present; skipping default-path integration test")

        poll = _make_poll(
            n_sample=1000,
            date="2026-01-01",
            race="2026 FL Senate",
        )
        result = apply_all_weights(
            [poll],
            reference_date="2026-01-01",
            apply_house_effects=False,
            use_primary_discount=True,
            primary_discount_factor=0.5,
            apply_quality=False,
        )
        # FL Senate primary is 2026-08-18, so Jan 1 is pre-primary → discounted
        assert result[0].n_sample == 500
