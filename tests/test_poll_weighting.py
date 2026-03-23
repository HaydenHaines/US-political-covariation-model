"""Tests for poll weighting: time decay, pollster quality, and aggregation."""

from __future__ import annotations

import math

import pytest

from src.propagation.propagate_polls import PollObservation
from src.propagation.poll_weighting import (
    aggregate_polls,
    apply_all_weights,
    apply_pollster_quality,
    apply_time_decay,
    election_day_for_cycle,
    extract_grade_from_notes,
    grade_to_multiplier,
    load_polls_with_notes,
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
    def test_a_plus_boost(self):
        """A+ grade should boost n_sample."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality([poll], poll_notes=["grade=3.0"])
        assert result[0].n_sample > 1000  # 1.2x
        assert result[0].n_sample == 1200

    def test_d_grade_reduction(self):
        """D grade should reduce n to ~30%."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality([poll], poll_notes=["grade=0.2"])
        assert result[0].n_sample == 300

    def test_no_grade_default(self):
        """No grade in notes -> 0.8x."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality([poll], poll_notes=["method=Live Phone"])
        assert result[0].n_sample == 800

    def test_no_notes_default(self):
        """No notes at all -> 0.8x."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality([poll], poll_notes=None)
        assert result[0].n_sample == 800

    def test_extracts_grade_from_notes(self):
        """Parses 'grade=2.5' from semicolon-delimited notes."""
        notes = "method=Online Panel; rating_id=588; grade=2.5; pollscore=0.2; bias=0.1"
        grade = extract_grade_from_notes(notes)
        assert grade == "A"  # 2.4-2.7 maps to A

    def test_b_grade_multiplier(self):
        """B grade (numeric ~1.5-1.9) -> 0.9x."""
        poll = _make_poll(n_sample=1000)
        result = apply_pollster_quality([poll], poll_notes=["grade=1.5"])
        assert result[0].n_sample == 900

    def test_custom_multipliers(self):
        """Custom grade multiplier table should be respected."""
        poll = _make_poll(n_sample=1000)
        custom = {"A": 2.0}
        result = apply_pollster_quality(
            [poll], poll_notes=["grade=2.5"], grade_multipliers=custom
        )
        assert result[0].n_sample == 2000


# ---------------------------------------------------------------------------
# Combined weighting
# ---------------------------------------------------------------------------


class TestApplyAllWeights:
    def test_combines_both(self):
        """apply_all_weights should apply both time decay and quality."""
        # Poll 30 days old (half_life=30 -> 0.5 decay), grade A (1.1x multiplier)
        poll = _make_poll(date="2020-10-04", n_sample=1000)
        result = apply_all_weights(
            [poll],
            reference_date="2020-11-03",
            half_life_days=30.0,
            poll_notes=["grade=2.5"],
            apply_quality=True,
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
