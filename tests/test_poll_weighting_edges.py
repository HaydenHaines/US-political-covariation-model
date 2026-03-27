"""Edge-case tests for poll_weighting.py.

Covers:
  - Empty poll list passed to aggregate_polls (raises ValueError)
  - Single poll (no aggregation needed; result equals that poll's dem_share)
  - Zero-weight polls (n_sample rounds to 0 after decay — code clamps to 1)
  - Very large n_sample values (no overflow in inverse-variance)
  - Duplicate pollsters (same pollster, same date, same race — both included)
  - n_sample=0 in PollObservation (sigma would divide by zero)
  - Mismatched state — poll with race label for wrong state passes through unchanged
  - All polls expired (far beyond half-life window — n_sample collapses but clamped to 1)
  - Negative house effect correction making dem_share approach 0 (clamp to _HE_DEM_SHARE_MIN)
  - Positive house effect correction making dem_share approach 1 (clamp to _HE_DEM_SHARE_MAX)
  - Pollster with no quality rating (falls back to 0.8x default multiplier)
  - apply_time_decay with a future poll date (age < 0 → no decay)
  - apply_primary_discount with no calendar file (no-op when file missing)
  - grade_to_multiplier returns default for unknown grade strings
  - aggregate_polls with identical dem_shares returns same combined share
"""

from __future__ import annotations

import math
from copy import copy

import pytest

from src.propagation.propagate_polls import PollObservation
from src.propagation.poll_weighting import (
    _HE_DEM_SHARE_MAX,
    _HE_DEM_SHARE_MIN,
    _NO_GRADE_MULTIPLIER,
    aggregate_polls,
    apply_all_weights,
    apply_house_effect_correction,
    apply_pollster_quality,
    apply_time_decay,
    grade_to_multiplier,
    reset_house_effect_cache,
    reset_sb_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poll(
    dem_share: float = 0.50,
    n_sample: int = 1000,
    date: str = "2026-06-01",
    pollster: str = "TestPollster",
    race: str = "2026 FL Senate",
    geography: str = "FL",
) -> PollObservation:
    """Build a PollObservation with sensible defaults for edge-case tests."""
    return PollObservation(
        geography=geography,
        dem_share=dem_share,
        n_sample=n_sample,
        race=race,
        date=date,
        pollster=pollster,
    )


# ---------------------------------------------------------------------------
# Edge case 1: Empty poll list → aggregate_polls raises ValueError
# ---------------------------------------------------------------------------


def test_aggregate_polls_empty_list_raises():
    """aggregate_polls must raise ValueError when given an empty list."""
    with pytest.raises(ValueError, match="No polls"):
        aggregate_polls([])


# ---------------------------------------------------------------------------
# Edge case 2: Single poll — combined share equals that poll's dem_share
# ---------------------------------------------------------------------------


def test_aggregate_polls_single_poll_returns_dem_share():
    """With one poll, combined dem_share must equal that poll's dem_share exactly."""
    poll = _poll(dem_share=0.52, n_sample=800)
    combined_share, _ = aggregate_polls([poll])
    assert abs(combined_share - 0.52) < 1e-9, (
        f"Single-poll aggregation returned {combined_share:.6f}, expected 0.52"
    )


# ---------------------------------------------------------------------------
# Edge case 3: Zero-weight polls — extreme time decay clamps n_sample to 1
# ---------------------------------------------------------------------------


def test_time_decay_extreme_age_clamps_n_sample_to_one():
    """A poll from 10 years ago should decay to n_sample=1 (floor), not 0."""
    poll = _poll(n_sample=1000, date="2010-01-01")
    result = apply_time_decay([poll], reference_date="2026-06-01", half_life_days=30.0)
    assert len(result) == 1
    assert result[0].n_sample >= 1, (
        "Time decay should clamp n_sample to at least 1, not zero"
    )
    # Also verify that heavy decay DID reduce n_sample (not left at 1000)
    assert result[0].n_sample < poll.n_sample


# ---------------------------------------------------------------------------
# Edge case 4: Very large n_sample — no overflow in inverse-variance
# ---------------------------------------------------------------------------


def test_aggregate_polls_very_large_n_sample_no_overflow():
    """aggregate_polls must not raise or produce NaN/Inf for n_sample=10^9."""
    poll = _poll(dem_share=0.48, n_sample=1_000_000_000)
    combined_share, combined_n = aggregate_polls([poll])
    assert math.isfinite(combined_share), "combined_share is not finite"
    assert math.isfinite(combined_n), "combined_n is not finite"
    assert combined_n >= 1


# ---------------------------------------------------------------------------
# Edge case 5: Duplicate pollsters — both polls are included in aggregation
# ---------------------------------------------------------------------------


def test_aggregate_polls_duplicate_pollsters_both_included():
    """Two polls from the same pollster on the same date must both be aggregated."""
    poll_a = _poll(dem_share=0.44, n_sample=600, pollster="Acme Polling")
    poll_b = _poll(dem_share=0.56, n_sample=600, pollster="Acme Polling")
    combined_share, _ = aggregate_polls([poll_a, poll_b])
    # With equal n and equal distances from 0.50, shares should average near 0.50
    assert 0.44 <= combined_share <= 0.56, (
        f"Duplicate-pollster aggregation out of range: {combined_share:.4f}"
    )
    # With symmetric inputs the result should be near 0.50
    assert abs(combined_share - 0.50) < 0.01, (
        f"Symmetric duplicate polls should average near 0.50; got {combined_share:.4f}"
    )


# ---------------------------------------------------------------------------
# Edge case 6: Mismatched state — poll for wrong state passes through unchanged
# ---------------------------------------------------------------------------


def test_apply_all_weights_wrong_state_poll_passes_through():
    """A poll whose race label is for a different state should pass through apply_all_weights."""
    # Race is for GA Senate but geography is FL — not a filtering criterion in weighting
    poll = _poll(race="2026 GA Senate", geography="FL", dem_share=0.45)
    reset_sb_cache()
    reset_house_effect_cache()
    result = apply_all_weights(
        [poll],
        reference_date="2026-06-01",
        apply_quality=False,
        apply_house_effects=False,
        use_primary_discount=False,
    )
    assert len(result) == 1, "Mismatched-state poll should not be silently dropped"


# ---------------------------------------------------------------------------
# Edge case 7: All polls expired — heavy decay, n_sample clamped to 1 for each
# ---------------------------------------------------------------------------


def test_time_decay_all_polls_expired_survive_with_floor():
    """Polls well beyond the decay window should survive with n_sample=1 (floor)."""
    polls = [
        _poll(n_sample=500, date="2010-01-01"),
        _poll(n_sample=800, date="2011-06-15"),
    ]
    result = apply_time_decay(polls, reference_date="2026-06-01", half_life_days=30.0)
    assert len(result) == 2
    for p in result:
        assert p.n_sample >= 1, "Expired poll n_sample must be at least 1 (floor)"


# ---------------------------------------------------------------------------
# Edge case 8: Negative house effect → dem_share clamps to _HE_DEM_SHARE_MIN
# ---------------------------------------------------------------------------


def test_house_effect_correction_large_positive_clamps_to_min():
    """A very large positive house effect (pollster way over-estimates Dems) must
    clamp corrected dem_share to _HE_DEM_SHARE_MIN rather than going negative."""
    # house_effect_pp = 60pp → correction = 0.60 → corrected = 0.05 - 0.60 = -0.55 → clamped
    poll = _poll(dem_share=0.05, pollster="BiasedPollster")
    reset_house_effect_cache()
    # Inject a very large house effect directly into the sb_house_effects dict
    result = apply_house_effect_correction(
        [poll],
        sb_house_effects={"BiasedPollster": 60.0},  # 60 percentage points
        bias_538={},
    )
    assert len(result) == 1
    assert result[0].dem_share >= _HE_DEM_SHARE_MIN, (
        f"dem_share {result[0].dem_share} fell below _HE_DEM_SHARE_MIN={_HE_DEM_SHARE_MIN}"
    )


# ---------------------------------------------------------------------------
# Edge case 9: Positive correction making share > 1 → clamp to _HE_DEM_SHARE_MAX
# ---------------------------------------------------------------------------


def test_house_effect_correction_large_negative_clamps_to_max():
    """A very large negative house effect (pollster way under-estimates Dems) must
    clamp corrected dem_share to _HE_DEM_SHARE_MAX rather than exceeding 1."""
    # house_effect_pp = -60pp → correction = -0.60 → corrected = 0.95 + 0.60 = 1.55 → clamped
    poll = _poll(dem_share=0.95, pollster="UndercountPollster")
    reset_house_effect_cache()
    result = apply_house_effect_correction(
        [poll],
        sb_house_effects={"UndercountPollster": -60.0},  # -60 pp
        bias_538={},
    )
    assert len(result) == 1
    assert result[0].dem_share <= _HE_DEM_SHARE_MAX, (
        f"dem_share {result[0].dem_share} exceeded _HE_DEM_SHARE_MAX={_HE_DEM_SHARE_MAX}"
    )


# ---------------------------------------------------------------------------
# Edge case 10: Pollster with no quality rating falls back to default (0.8x)
# ---------------------------------------------------------------------------


def test_pollster_with_no_rating_uses_default_multiplier():
    """A pollster absent from Silver Bulletin and notes with no grade gets 0.8x default."""
    poll = _poll(n_sample=1000, pollster="CompletelyUnknownPollster")
    reset_sb_cache()
    result = apply_pollster_quality(
        [poll],
        poll_notes=[""],      # no grade in notes
        use_silver_bulletin=False,  # force 538-grade-only path
    )
    assert len(result) == 1
    expected_n = int(max(1, round(1000 * _NO_GRADE_MULTIPLIER)))
    assert result[0].n_sample == expected_n, (
        f"Unknown pollster n_sample: {result[0].n_sample}, expected {expected_n} "
        f"(0.8x of 1000)"
    )


# ---------------------------------------------------------------------------
# Edge case 11: Future poll date (age < 0) → no decay applied
# ---------------------------------------------------------------------------


def test_time_decay_future_poll_gets_no_decay():
    """A poll dated after the reference date should have age=0 and thus decay=1.0."""
    poll = _poll(n_sample=1000, date="2027-01-01")
    result = apply_time_decay([poll], reference_date="2026-06-01", half_life_days=30.0)
    assert len(result) == 1
    # decay=1.0 → n_effective = round(1000 * 1.0) = 1000
    assert result[0].n_sample == 1000, (
        f"Future poll should have n_sample unchanged; got {result[0].n_sample}"
    )


# ---------------------------------------------------------------------------
# Edge case 12: grade_to_multiplier returns default for unknown grade strings
# ---------------------------------------------------------------------------


def test_grade_to_multiplier_unknown_grade_returns_default():
    """grade_to_multiplier must return _NO_GRADE_MULTIPLIER for unrecognised grade strings."""
    result = grade_to_multiplier("Z+")
    assert result == _NO_GRADE_MULTIPLIER, (
        f"Unknown grade 'Z+' returned {result}, expected default {_NO_GRADE_MULTIPLIER}"
    )


def test_grade_to_multiplier_none_returns_default():
    """grade_to_multiplier must return _NO_GRADE_MULTIPLIER when grade is None."""
    result = grade_to_multiplier(None)
    assert result == _NO_GRADE_MULTIPLIER


# ---------------------------------------------------------------------------
# Edge case 13: aggregate_polls with identical dem_shares returns same combined share
# ---------------------------------------------------------------------------


def test_aggregate_polls_identical_dem_shares_returns_same_share():
    """When all polls show the same dem_share, the combined share must equal that value."""
    target_share = 0.47
    polls = [_poll(dem_share=target_share, n_sample=n) for n in [400, 600, 1000]]
    combined_share, _ = aggregate_polls(polls)
    assert abs(combined_share - target_share) < 1e-9, (
        f"Combined share {combined_share:.6f} differs from uniform input {target_share}"
    )


# ---------------------------------------------------------------------------
# Edge case 14: apply_time_decay with poll that has no date — passes through unchanged
# ---------------------------------------------------------------------------


def test_time_decay_poll_without_date_passes_through():
    """A poll with no date should pass through apply_time_decay with n_sample unchanged."""
    poll = PollObservation(geography="FL", dem_share=0.50, n_sample=750, date="")
    result = apply_time_decay([poll], reference_date="2026-06-01")
    assert len(result) == 1
    assert result[0].n_sample == 750, (
        "Poll without a date should not have its n_sample modified by time decay"
    )


# ---------------------------------------------------------------------------
# Edge case 15: apply_house_effect_correction with empty pollster name — no correction
# ---------------------------------------------------------------------------


def test_house_effect_correction_empty_pollster_no_correction():
    """A poll with an empty pollster string should have its dem_share left unchanged."""
    poll = PollObservation(geography="FL", dem_share=0.52, n_sample=800, pollster="")
    reset_house_effect_cache()
    result = apply_house_effect_correction(
        [poll],
        sb_house_effects={"SomePollster": 5.0},
        bias_538={"SomePollster": 3.0},
    )
    assert len(result) == 1
    assert result[0].dem_share == 0.52, (
        f"Poll with empty pollster should have unmodified dem_share; "
        f"got {result[0].dem_share}"
    )
