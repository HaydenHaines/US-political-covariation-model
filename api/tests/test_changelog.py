"""Tests for the forecast changelog API endpoint."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.routers.forecast import get_forecast_changelog, SNAPSHOTS_DIR


@pytest.fixture
def mock_snapshots_dir():
    """Create a temporary snapshots directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Snapshot 1: initial
        s1 = {
            "date": "2026-03-22",
            "predictions": {
                "2026 FL Senate": 0.40,
                "2026 GA Senate": 0.45,
                "2026 MI Senate": 0.50,
            },
            "note": "Initial baseline",
        }
        (tmppath / "2026-03-22.json").write_text(json.dumps(s1))

        # Snapshot 2: FL and GA moved
        s2 = {
            "date": "2026-03-29",
            "predictions": {
                "2026 FL Senate": 0.42,
                "2026 GA Senate": 0.47,
                "2026 MI Senate": 0.501,  # < 0.2pp change, should be filtered
            },
            "note": "Weekly poll scrape",
        }
        (tmppath / "2026-03-29.json").write_text(json.dumps(s2))

        with patch("api.routers.forecast.SNAPSHOTS_DIR", tmppath):
            yield tmppath


def test_changelog_returns_entries(mock_snapshots_dir):
    result = get_forecast_changelog()
    assert len(result.entries) == 2
    assert result.current_snapshot_date == "2026-03-29"


def test_changelog_newest_first(mock_snapshots_dir):
    result = get_forecast_changelog()
    dates = [e.date for e in result.entries]
    assert dates == ["2026-03-29", "2026-03-22"]


def test_changelog_filters_small_changes(mock_snapshots_dir):
    result = get_forecast_changelog()
    latest = result.entries[0]
    races = {d.race for d in latest.diffs}
    # MI Senate changed < 0.2pp, should be filtered out
    assert "2026 MI Senate" not in races
    # FL and GA moved by 2pp, should be included
    assert "2026 FL Senate" in races
    assert "2026 GA Senate" in races


def test_changelog_delta_values(mock_snapshots_dir):
    result = get_forecast_changelog()
    latest = result.entries[0]
    fl_diff = next(d for d in latest.diffs if d.race == "2026 FL Senate")
    assert fl_diff.before == pytest.approx(0.40)
    assert fl_diff.after == pytest.approx(0.42)
    assert fl_diff.delta == pytest.approx(0.02)


def test_changelog_initial_entry_has_null_before(mock_snapshots_dir):
    result = get_forecast_changelog()
    initial = result.entries[-1]  # oldest = last (newest-first order)
    assert initial.date == "2026-03-22"
    for d in initial.diffs:
        assert d.before is None
        assert d.after is not None


def test_changelog_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("api.routers.forecast.SNAPSHOTS_DIR", Path(tmpdir)):
            result = get_forecast_changelog()
            assert result.entries == []
            assert result.current_snapshot_date is None


def test_changelog_nonexistent_dir():
    with patch("api.routers.forecast.SNAPSHOTS_DIR", Path("/nonexistent/path")):
        result = get_forecast_changelog()
        assert result.entries == []


def test_changelog_only_tracked_races(mock_snapshots_dir):
    """Only races in TRACKED_RACES should appear in changelog entries."""
    result = get_forecast_changelog()
    from api.routers.forecast import TRACKED_RACES
    for entry in result.entries:
        for d in entry.diffs:
            assert d.race in TRACKED_RACES
