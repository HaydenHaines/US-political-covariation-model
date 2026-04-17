"""Tests that API state lists match the canonical candidate config.

Regression test for S548: FL and OH special Senate elections were modeled
by the prediction pipeline (via candidates_2026.json) but not served by
the API because SENATE_2026_STATES was hardcoded to 33 Class II seats.
These tests ensure every configured race has a matching API entry.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from api.routers.governor._helpers import _GOVERNOR_INCUMBENT, GOVERNOR_2026_STATES
from api.routers.senate._helpers import _CLASS_II_INCUMBENT, SENATE_2026_STATES

_CANDIDATES_PATH = Path("data/config/candidates_2026.json")

# State abbreviation extracted from race keys like "2026 AK Senate"
_RACE_KEY_RE = re.compile(r"^2026 ([A-Z]{2}) (Senate|Governor)$")


def _load_candidates() -> dict:
    if not _CANDIDATES_PATH.exists():
        pytest.skip("candidates_2026.json not found (gitignored data)")
    with open(_CANDIDATES_PATH) as f:
        return json.load(f)


# ── Senate ────────────────────────────────────────────────────────────────


class TestSenateCompleteness:
    """Every senate race in candidates config must be served by the API."""

    def test_all_candidate_races_in_api_state_list(self):
        """No race defined in candidates_2026.json should be missing from
        SENATE_2026_STATES — this was the S548 bug."""
        candidates = _load_candidates()
        senate_races = candidates.get("senate", {})
        candidate_states = set()
        for key in senate_races:
            m = _RACE_KEY_RE.match(key)
            assert m, f"Unexpected race key format: {key}"
            candidate_states.add(m.group(1))

        missing = candidate_states - SENATE_2026_STATES
        assert not missing, (
            f"Senate races in candidates_2026.json but NOT in "
            f"SENATE_2026_STATES: {sorted(missing)}. "
            f"Add them to api/routers/senate/_helpers.py."
        )

    def test_all_api_states_have_incumbent_entry(self):
        """Every state in SENATE_2026_STATES must have an incumbent party."""
        missing = SENATE_2026_STATES - _CLASS_II_INCUMBENT.keys()
        assert not missing, (
            f"States in SENATE_2026_STATES but NOT in "
            f"_CLASS_II_INCUMBENT: {sorted(missing)}. "
            f"Add incumbent party to api/routers/senate/_helpers.py."
        )

    def test_incumbent_entries_match_state_list(self):
        """No orphan entries in _CLASS_II_INCUMBENT."""
        extra = _CLASS_II_INCUMBENT.keys() - SENATE_2026_STATES
        assert not extra, (
            f"States in _CLASS_II_INCUMBENT but NOT in "
            f"SENATE_2026_STATES: {sorted(extra)}. Remove stale entries."
        )


# ── Governor ──────────────────────────────────────────────────────────────


class TestGovernorCompleteness:
    """Governor API state list and incumbent map must be consistent."""

    def test_all_api_states_have_incumbent_entry(self):
        """Every state in GOVERNOR_2026_STATES must have an incumbent party."""
        missing = GOVERNOR_2026_STATES - _GOVERNOR_INCUMBENT.keys()
        assert not missing, (
            f"States in GOVERNOR_2026_STATES but NOT in "
            f"_GOVERNOR_INCUMBENT: {sorted(missing)}. "
            f"Add incumbent party to api/routers/governor/_helpers.py."
        )

    def test_incumbent_entries_match_state_list(self):
        """No orphan entries in _GOVERNOR_INCUMBENT."""
        extra = _GOVERNOR_INCUMBENT.keys() - GOVERNOR_2026_STATES
        assert not extra, (
            f"States in _GOVERNOR_INCUMBENT but NOT in "
            f"GOVERNOR_2026_STATES: {sorted(extra)}. Remove stale entries."
        )


# ── Cross-consistency ─────────────────────────────────────────────────────


class TestCrossConsistency:
    """Validate that state lists have sane sizes."""

    def test_senate_race_count_plausible(self):
        """Class II has 33 seats; specials may add 1-3 more."""
        assert 33 <= len(SENATE_2026_STATES) <= 37, (
            f"SENATE_2026_STATES has {len(SENATE_2026_STATES)} entries — "
            f"expected 33-37 (Class II + specials)."
        )

    def test_governor_race_count_plausible(self):
        """36 governorships are up in 2026."""
        assert 34 <= len(GOVERNOR_2026_STATES) <= 38, (
            f"GOVERNOR_2026_STATES has {len(GOVERNOR_2026_STATES)} entries — "
            f"expected 34-38."
        )
