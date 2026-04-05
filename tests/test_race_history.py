"""Tests for the GET /forecast/race-history endpoint.

Uses temporary directories to simulate forecast snapshots -- no real filesystem
state required.  The endpoint reads from SNAPSHOTS_DIR which is monkey-patched
via the module attribute.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.routers.forecast.race_history import (
    _load_race_history_from_snapshots,
    _race_to_slug,
    get_race_history,
)
import api.routers.forecast.race_history as race_history_module


class TestRaceToSlug:
    def test_senate_race_slug(self):
        assert _race_to_slug("2026 FL Senate") == "2026-fl-senate"

    def test_governor_race_slug(self):
        assert _race_to_slug("2026 TX Governor") == "2026-tx-governor"

    def test_lowercase_conversion(self):
        """Slug is always lowercase."""
        assert _race_to_slug("2026 NC Senate") == "2026-nc-senate"

    def test_spaces_become_hyphens(self):
        """All spaces are replaced with hyphens."""
        slug = _race_to_slug("2026 MN Senate")
        assert " " not in slug
        assert "-" in slug


class TestLoadRaceHistoryFromSnapshots:
    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        result = _load_race_history_from_snapshots(tmp_path / "nonexistent")
        assert result == []

    def test_returns_empty_for_empty_dir(self, tmp_path):
        result = _load_race_history_from_snapshots(tmp_path)
        assert result == []

    def test_single_snapshot_single_race(self, tmp_path):
        """One snapshot, one race -> one entry with one history point."""
        snap = {"date": "2026-03-01", "predictions": {"2026 FL Senate": 0.54}}
        (tmp_path / "2026-03-01.json").write_text(json.dumps(snap))
        result = _load_race_history_from_snapshots(tmp_path)

        assert len(result) == 1
        entry = result[0]
        assert entry["slug"] == "2026-fl-senate"
        assert len(entry["history"]) == 1
        assert entry["history"][0]["date"] == "2026-03-01"
        # margin = 0.54 - 0.5 = 0.04 (positive = Dem-favored)
        assert abs(entry["history"][0]["margin"] - 0.04) < 1e-5

    def test_margin_sign_convention(self, tmp_path):
        """Positive dem_share > 0.5 -> positive margin; < 0.5 -> negative."""
        snap = {
            "date": "2026-04-01",
            "predictions": {
                "2026 FL Senate": 0.56,   # Dem-favored
                "2026 TX Senate": 0.44,   # GOP-favored
                "2026 NJ Senate": 0.5,    # exactly tied
            },
        }
        (tmp_path / "2026-04-01.json").write_text(json.dumps(snap))
        result = _load_race_history_from_snapshots(tmp_path)

        by_slug = {r["slug"]: r["history"][0]["margin"] for r in result}
        assert by_slug["2026-fl-senate"] > 0
        assert by_slug["2026-tx-senate"] < 0
        assert by_slug["2026-nj-senate"] == 0.0

    def test_multiple_snapshots_history_sorted_chronologically(self, tmp_path):
        """History within each race is sorted oldest-first (by filename sort order)."""
        for date, share in [("2026-03-01", 0.51), ("2026-03-15", 0.53), ("2026-04-01", 0.55)]:
            snap = {"date": date, "predictions": {"2026 GA Senate": share}}
            (tmp_path / f"{date}.json").write_text(json.dumps(snap))

        result = _load_race_history_from_snapshots(tmp_path)
        assert len(result) == 1

        history = result[0]["history"]
        assert len(history) == 3
        dates = [h["date"] for h in history]
        assert dates == sorted(dates)

    def test_margin_changes_across_snapshots(self, tmp_path):
        """Margin values reflect the prediction at each snapshot date."""
        snap1 = {"date": "2026-03-01", "predictions": {"2026 MI Senate": 0.48}}
        snap2 = {"date": "2026-04-05", "predictions": {"2026 MI Senate": 0.53}}
        (tmp_path / "2026-03-01.json").write_text(json.dumps(snap1))
        (tmp_path / "2026-04-05.json").write_text(json.dumps(snap2))

        result = _load_race_history_from_snapshots(tmp_path)
        history = result[0]["history"]
        assert len(history) == 2
        # March: negative margin (GOP-favored); April: positive (Dem-favored)
        assert history[0]["margin"] < 0
        assert history[1]["margin"] > 0

    def test_multiple_races_all_returned(self, tmp_path):
        """All races present in any snapshot are returned."""
        snap = {
            "date": "2026-04-01",
            "predictions": {
                "2026 FL Senate": 0.50,
                "2026 GA Senate": 0.52,
                "2026 NC Senate": 0.49,
            },
        }
        (tmp_path / "2026-04-01.json").write_text(json.dumps(snap))
        result = _load_race_history_from_snapshots(tmp_path)

        slugs = {r["slug"] for r in result}
        assert "2026-fl-senate" in slugs
        assert "2026-ga-senate" in slugs
        assert "2026-nc-senate" in slugs

    def test_malformed_json_skipped(self, tmp_path):
        """Files with invalid JSON are skipped; others still process."""
        (tmp_path / "2026-03-01.json").write_text("{bad json!!")
        (tmp_path / "2026-04-01.json").write_text(json.dumps({
            "date": "2026-04-01",
            "predictions": {"2026 FL Senate": 0.51},
        }))
        result = _load_race_history_from_snapshots(tmp_path)
        assert len(result) == 1
        assert result[0]["history"][0]["date"] == "2026-04-01"

    def test_snapshot_missing_date_field_skipped(self, tmp_path):
        """Snapshots without a 'date' field are silently skipped."""
        bad = {"predictions": {"2026 FL Senate": 0.5}}
        good = {"date": "2026-04-01", "predictions": {"2026 FL Senate": 0.52}}
        (tmp_path / "2026-03-01.json").write_text(json.dumps(bad))
        (tmp_path / "2026-04-01.json").write_text(json.dumps(good))
        result = _load_race_history_from_snapshots(tmp_path)
        # Only the good snapshot counts, so one history point
        assert len(result) == 1
        assert len(result[0]["history"]) == 1

    def test_non_numeric_prediction_skipped(self, tmp_path):
        """Non-numeric prediction values in a snapshot are silently skipped."""
        snap = {
            "date": "2026-04-01",
            "predictions": {
                "2026 FL Senate": "not_a_number",
                "2026 GA Senate": 0.52,
            },
        }
        (tmp_path / "2026-04-01.json").write_text(json.dumps(snap))
        result = _load_race_history_from_snapshots(tmp_path)
        slugs = {r["slug"] for r in result}
        assert "2026-ga-senate" in slugs
        assert "2026-fl-senate" not in slugs


class TestGetRaceHistory:
    def test_returns_empty_when_no_snapshots_dir(self, monkeypatch, tmp_path):
        """Non-existent snapshots dir -> empty list."""
        monkeypatch.setattr(race_history_module, "SNAPSHOTS_DIR", tmp_path / "nonexistent")
        result = get_race_history()
        assert result == []

    def test_entry_schema(self, monkeypatch, tmp_path):
        """Each entry has the expected keys: slug and history."""
        snap = {
            "date": "2026-04-01",
            "predictions": {"2026 NC Senate": 0.61},
        }
        (tmp_path / "2026-04-01.json").write_text(json.dumps(snap))
        monkeypatch.setattr(race_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_race_history()

        assert len(result) == 1
        entry = result[0]
        assert set(entry.keys()) == {"slug", "history"}
        assert isinstance(entry["slug"], str)
        assert isinstance(entry["history"], list)

    def test_history_entry_schema(self, monkeypatch, tmp_path):
        """Each history entry has 'date' (str) and 'margin' (float)."""
        snap = {
            "date": "2026-04-01",
            "predictions": {"2026 NC Senate": 0.61},
        }
        (tmp_path / "2026-04-01.json").write_text(json.dumps(snap))
        monkeypatch.setattr(race_history_module, "SNAPSHOTS_DIR", tmp_path)
        result = get_race_history()

        history_entry = result[0]["history"][0]
        assert set(history_entry.keys()) == {"date", "margin"}
        assert isinstance(history_entry["date"], str)
        assert isinstance(history_entry["margin"], float)

    def test_real_snapshots_present(self, monkeypatch):
        """With the real snapshots directory, the endpoint returns data."""
        # Do NOT monkeypatch -- use the real SNAPSHOTS_DIR.
        result = get_race_history()
        # We have at least 3 real snapshots (2026-03-29, 2026-04-03, 2026-04-05)
        assert len(result) > 0
        # Every entry should have history
        assert all(len(r["history"]) > 0 for r in result)
        # Slugs should be lowercase, hyphen-separated
        for r in result:
            assert r["slug"] == r["slug"].lower()
            assert " " not in r["slug"]
