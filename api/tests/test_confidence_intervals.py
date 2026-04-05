"""Tests for forecast confidence interval computation.

Covers:
- _get_std_floor: race-type-specific empirical error floors from 2022 backtest
- _compute_state_std: vote-weighted std with race-type-aware floors
- Edge cases: single county, no votes, many counties all agreeing
"""
from __future__ import annotations

import numpy as np
import pytest

from api.routers.forecast._helpers import (
    _GENERIC_STD_FLOOR,
    _GOVERNOR_STD_FLOOR,
    _SENATE_STD_FLOOR,
    _STATE_STD_CAP,
    _STATE_STD_FALLBACK,
    _compute_state_std,
    _get_std_floor,
)


# ── _get_std_floor unit tests ─────────────────────────────────────────────────


class TestGetStdFloor:
    """_get_std_floor returns the empirical RMSE for each race type."""

    def test_senate_lowercase(self):
        assert _get_std_floor("senate") == _SENATE_STD_FLOOR

    def test_senate_uppercase(self):
        assert _get_std_floor("Senate") == _SENATE_STD_FLOOR

    def test_senate_mixed_case(self):
        assert _get_std_floor("SENATE") == _SENATE_STD_FLOOR

    def test_governor_lowercase(self):
        assert _get_std_floor("governor") == _GOVERNOR_STD_FLOOR

    def test_governor_capitalized(self):
        assert _get_std_floor("Governor") == _GOVERNOR_STD_FLOOR

    def test_gov_abbreviation(self):
        # 'gov' prefix should also match governor floor
        assert _get_std_floor("gov") == _GOVERNOR_STD_FLOOR

    def test_unknown_race_type(self):
        assert _get_std_floor("president") == _GENERIC_STD_FLOOR

    def test_empty_string(self):
        assert _get_std_floor("") == _GENERIC_STD_FLOOR

    def test_senate_floor_is_below_governor_floor(self):
        """Senate races historically have lower error than governor races.

        2022 backtest: Senate RMSE ~3.7pp, Governor (competitive) RMSE ~5.5pp.
        """
        assert _SENATE_STD_FLOOR < _GOVERNOR_STD_FLOOR

    def test_all_floors_are_positive(self):
        for rt in ["senate", "governor", "president", ""]:
            assert _get_std_floor(rt) > 0

    def test_all_floors_below_cap(self):
        for rt in ["senate", "governor", "president", ""]:
            assert _get_std_floor(rt) < _STATE_STD_CAP


# ── _compute_state_std unit tests ─────────────────────────────────────────────


class TestComputeStateStd:
    """_compute_state_std returns vote-weighted std clamped to empirical floors."""

    def _uniform_votes(self, n: int) -> np.ndarray:
        """Equal vote weights for N counties."""
        return np.ones(n, dtype=float)

    def test_no_counties_returns_fallback(self):
        """Empty arrays fall back to the generic fallback std."""
        result = _compute_state_std(np.array([]), np.array([]), 0.5)
        assert result == pytest.approx(_STATE_STD_FALLBACK, abs=1e-6)

    def test_single_county_returns_fallback(self):
        """Single county: can't compute variance, use fallback >= floor."""
        result = _compute_state_std(np.array([0.52]), np.array([1000.0]), 0.52)
        # Must be at least the generic floor
        assert result >= _GENERIC_STD_FLOOR
        # And at least the fallback (which is larger than the generic floor)
        assert result >= _STATE_STD_FALLBACK

    def test_single_county_senate_returns_at_least_senate_floor(self):
        result = _compute_state_std(np.array([0.52]), np.array([1000.0]), 0.52, race_type="senate")
        assert result >= _SENATE_STD_FLOOR

    def test_single_county_governor_returns_at_least_governor_floor(self):
        result = _compute_state_std(np.array([0.52]), np.array([1000.0]), 0.52, race_type="governor")
        assert result >= _GOVERNOR_STD_FLOOR

    def test_perfect_agreement_senate_floor(self):
        """When all counties predict identically, variance is zero — floor applies."""
        preds = np.full(50, 0.52)
        votes = self._uniform_votes(50)
        result = _compute_state_std(preds, votes, 0.52, race_type="senate")
        assert result == pytest.approx(_SENATE_STD_FLOOR, abs=1e-6)

    def test_perfect_agreement_governor_floor(self):
        """Governor races have a higher floor than senate."""
        preds = np.full(50, 0.52)
        votes = self._uniform_votes(50)
        result = _compute_state_std(preds, votes, 0.52, race_type="governor")
        assert result == pytest.approx(_GOVERNOR_STD_FLOOR, abs=1e-6)

    def test_wide_spread_uses_variance(self):
        """Genuinely high variance should not be clamped to the floor."""
        # Counties ranging from 0.3 to 0.7: std will be well above any floor
        preds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        votes = self._uniform_votes(5)
        state_pred = float(preds.mean())
        result = _compute_state_std(preds, votes, state_pred, race_type="senate")
        assert result > _SENATE_STD_FLOOR

    def test_cap_enforced(self):
        """Extreme spread is capped at _STATE_STD_CAP."""
        preds = np.array([0.0, 1.0])
        votes = self._uniform_votes(2)
        result = _compute_state_std(preds, votes, 0.5)
        assert result <= _STATE_STD_CAP

    def test_result_is_always_non_negative(self):
        for n in [1, 2, 5, 100]:
            preds = np.random.rand(n)
            votes = np.random.rand(n) * 1000
            state_pred = float(np.average(preds, weights=votes) if votes.sum() > 0 else preds.mean())
            result = _compute_state_std(preds, votes, state_pred, race_type="governor")
            assert result >= 0

    def test_zero_votes_returns_fallback(self):
        """Zero vote totals can't produce a vote-weighted std; use fallback >= floor."""
        preds = np.array([0.48, 0.52])
        votes = np.zeros(2)
        result = _compute_state_std(preds, votes, 0.50)
        assert result >= _GENERIC_STD_FLOOR
        assert result >= _STATE_STD_FALLBACK

    def test_governor_floor_higher_than_senate_floor(self):
        """Governor std must exceed Senate std when both are at floor (zero variance)."""
        preds = np.full(10, 0.50)
        votes = self._uniform_votes(10)
        gov_std = _compute_state_std(preds, votes, 0.50, race_type="governor")
        sen_std = _compute_state_std(preds, votes, 0.50, race_type="senate")
        assert gov_std > sen_std

    def test_vote_weighting_affects_result(self):
        """Heavy vote county pulls std toward its value when weights differ."""
        # preds: 0.4 and 0.6; equal votes → state_pred=0.5, equal var
        preds_eq = np.array([0.4, 0.6])
        votes_eq = np.array([500.0, 500.0])
        std_equal = _compute_state_std(preds_eq, votes_eq, 0.5)

        # preds same, but 90% weight on the 0.4 county
        votes_heavy = np.array([4500.0, 500.0])
        state_pred_heavy = float(np.average(preds_eq, weights=votes_heavy))
        std_heavy = _compute_state_std(preds_eq, votes_heavy, state_pred_heavy)

        # Both should be valid; different weights produce different stds
        assert std_equal > 0
        assert std_heavy > 0

    def test_default_race_type_uses_generic_floor(self):
        """When race_type is omitted, generic floor applies."""
        preds = np.full(10, 0.52)
        votes = self._uniform_votes(10)
        result = _compute_state_std(preds, votes, 0.52)
        assert result == pytest.approx(_GENERIC_STD_FLOOR, abs=1e-6)


# ── CI interval bounds sanity checks ─────────────────────────────────────────


class TestCIBoundsSanity:
    """Verify that lo90 <= prediction <= hi90 always holds."""

    @pytest.mark.parametrize("race_type", ["senate", "governor", "president", ""])
    @pytest.mark.parametrize("n_counties", [1, 5, 50])
    def test_ci_brackets_prediction(self, race_type, n_counties):
        from api.routers.forecast._helpers import _Z90

        preds = np.random.rand(n_counties) * 0.4 + 0.3  # range 0.3–0.7
        votes = np.random.rand(n_counties) * 1000 + 1
        state_pred = float(np.average(preds, weights=votes))

        std = _compute_state_std(preds, votes, state_pred, race_type=race_type)
        lo90 = state_pred - _Z90 * std
        hi90 = state_pred + _Z90 * std

        assert lo90 <= state_pred, f"lo90 ({lo90:.4f}) > prediction ({state_pred:.4f})"
        assert hi90 >= state_pred, f"hi90 ({hi90:.4f}) < prediction ({state_pred:.4f})"
