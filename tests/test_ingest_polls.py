"""Tests for src/assembly/ingest_polls.py — poll CSV ingestion.

Uses temporary CSV files to test loading, filtering, and validation logic.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

# We need to patch the CSV path since ingest_polls uses PROJECT_ROOT
import src.assembly.ingest_polls as ingest_mod
from src.propagation.propagate_polls import PollObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_polls(rows: list[dict], path: Path) -> Path:
    """Write poll rows to a CSV at the given path."""
    fieldnames = ["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"]
    # Also include any xt_ columns from the rows
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _setup_cycle_csv(tmp_path: Path, rows: list[dict], cycle: str = "test") -> None:
    """Create the expected polls directory and CSV for a given cycle."""
    polls_dir = tmp_path / "data" / "polls"
    polls_dir.mkdir(parents=True)
    _write_polls(rows, polls_dir / f"polls_{cycle}.csv")
    # Monkey-patch PROJECT_ROOT so ingest_polls finds our temp dir
    ingest_mod.PROJECT_ROOT = tmp_path


# ---------------------------------------------------------------------------
# _row_to_observation
# ---------------------------------------------------------------------------

class TestRowToObservation:
    def test_valid_row(self):
        row = {
            "race": "2026 FL Senate",
            "geography": "FL",
            "geo_level": "state",
            "dem_share": "0.48",
            "n_sample": "1000",
            "date": "2026-01-15",
            "pollster": "Siena",
            "notes": "",
        }
        obs = ingest_mod._row_to_observation(row, 2)
        assert obs is not None
        assert isinstance(obs, PollObservation)
        assert obs.geography == "FL"
        assert obs.dem_share == pytest.approx(0.48)
        assert obs.n_sample == 1000

    def test_missing_dem_share_returns_none(self):
        row = {"race": "test", "geography": "FL", "geo_level": "state",
               "dem_share": "", "n_sample": "500", "date": "", "pollster": ""}
        assert ingest_mod._row_to_observation(row, 1) is None

    def test_missing_n_sample_returns_none(self):
        row = {"race": "test", "geography": "FL", "geo_level": "state",
               "dem_share": "0.5", "n_sample": "", "date": "", "pollster": ""}
        assert ingest_mod._row_to_observation(row, 1) is None

    def test_invalid_dem_share_returns_none(self):
        row = {"race": "test", "geography": "FL", "geo_level": "state",
               "dem_share": "abc", "n_sample": "500", "date": "", "pollster": ""}
        assert ingest_mod._row_to_observation(row, 1) is None

    def test_dem_share_zero_returns_none(self):
        """dem_share must be strictly between 0 and 1."""
        row = {"race": "test", "geography": "FL", "geo_level": "state",
               "dem_share": "0.0", "n_sample": "500", "date": "", "pollster": ""}
        assert ingest_mod._row_to_observation(row, 1) is None

    def test_dem_share_one_returns_none(self):
        row = {"race": "test", "geography": "FL", "geo_level": "state",
               "dem_share": "1.0", "n_sample": "500", "date": "", "pollster": ""}
        assert ingest_mod._row_to_observation(row, 1) is None

    def test_negative_n_sample_returns_none(self):
        row = {"race": "test", "geography": "FL", "geo_level": "state",
               "dem_share": "0.5", "n_sample": "-10", "date": "", "pollster": ""}
        assert ingest_mod._row_to_observation(row, 1) is None

    def test_default_geo_level_is_state(self):
        row = {"race": "test", "geography": "FL", "geo_level": "",
               "dem_share": "0.5", "n_sample": "500", "date": "", "pollster": ""}
        obs = ingest_mod._row_to_observation(row, 1)
        assert obs.geo_level == "state"

    def test_float_n_sample_truncated(self):
        """n_sample='1000.5' should be parsed as 1000."""
        row = {"race": "test", "geography": "FL", "geo_level": "state",
               "dem_share": "0.5", "n_sample": "1000.5", "date": "", "pollster": ""}
        obs = ingest_mod._row_to_observation(row, 1)
        assert obs is not None
        assert obs.n_sample == 1000


# ---------------------------------------------------------------------------
# load_polls
# ---------------------------------------------------------------------------

class TestLoadPolls:
    def test_basic_load(self, tmp_path):
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.48", "n_sample": "1000", "date": "2026-01-15",
             "pollster": "Siena", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test")
        assert len(polls) == 1
        assert polls[0].race == "2026 FL Senate"

    def test_filter_by_race(self, tmp_path):
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.48", "n_sample": "1000", "date": "2026-01-15",
             "pollster": "Siena", "notes": ""},
            {"race": "2026 GA Senate", "geography": "GA", "geo_level": "state",
             "dem_share": "0.45", "n_sample": "800", "date": "2026-02-01",
             "pollster": "Quinnipiac", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test", race="FL Senate")
        assert len(polls) == 1
        assert polls[0].geography == "FL"

    def test_filter_by_geography(self, tmp_path):
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.48", "n_sample": "1000", "date": "2026-01-15",
             "pollster": "A", "notes": ""},
            {"race": "2026 GA Senate", "geography": "GA", "geo_level": "state",
             "dem_share": "0.45", "n_sample": "800", "date": "2026-02-01",
             "pollster": "B", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test", geography="GA")
        assert len(polls) == 1
        assert polls[0].geography == "GA"

    def test_filter_by_date_after(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "A", "notes": ""},
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.52", "n_sample": "600", "date": "2026-03-01",
             "pollster": "B", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test", after="2026-02-01")
        assert len(polls) == 1
        assert polls[0].date == "2026-03-01"

    def test_filter_by_date_before(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "A", "notes": ""},
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.52", "n_sample": "600", "date": "2026-03-01",
             "pollster": "B", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test", before="2026-02-01")
        assert len(polls) == 1
        assert polls[0].date == "2026-01-01"

    def test_filter_by_geo_level(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "A", "notes": ""},
            {"race": "test", "geography": "FL-03", "geo_level": "county",
             "dem_share": "0.52", "n_sample": "600", "date": "2026-01-01",
             "pollster": "B", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test", geo_level="county")
        assert len(polls) == 1
        assert polls[0].geo_level == "county"

    def test_sorted_by_date(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-03-01",
             "pollster": "A", "notes": ""},
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.52", "n_sample": "600", "date": "2026-01-01",
             "pollster": "B", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test")
        assert polls[0].date <= polls[1].date

    def test_skips_invalid_rows(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "Good", "notes": ""},
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "bad", "n_sample": "500", "date": "2026-01-01",
             "pollster": "Bad", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test")
        assert len(polls) == 1

    def test_missing_file_raises(self, tmp_path):
        ingest_mod.PROJECT_ROOT = tmp_path
        with pytest.raises(FileNotFoundError):
            ingest_mod.load_polls("nonexistent_cycle")

    def test_race_filter_is_case_insensitive(self, tmp_path):
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.48", "n_sample": "1000", "date": "2026-01-15",
             "pollster": "A", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls = ingest_mod.load_polls("test", race="fl senate")
        assert len(polls) == 1


# ---------------------------------------------------------------------------
# list_races
# ---------------------------------------------------------------------------

class TestListRaces:
    def test_returns_sorted_unique(self, tmp_path):
        rows = [
            {"race": "2026 GA Senate", "geography": "GA", "geo_level": "state",
             "dem_share": "0.45", "n_sample": "800", "date": "2026-01-01",
             "pollster": "A", "notes": ""},
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.48", "n_sample": "1000", "date": "2026-01-15",
             "pollster": "B", "notes": ""},
            {"race": "2026 GA Senate", "geography": "GA", "geo_level": "state",
             "dem_share": "0.46", "n_sample": "900", "date": "2026-02-01",
             "pollster": "C", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        races = ingest_mod.list_races("test")
        assert races == ["2026 FL Senate", "2026 GA Senate"]


# ---------------------------------------------------------------------------
# load_polls_with_crosstabs
# ---------------------------------------------------------------------------

class TestLoadPollsWithCrosstabs:
    def test_no_xt_columns(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "A", "notes": ""},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls, xt = ingest_mod.load_polls_with_crosstabs("test")
        assert len(polls) == 1
        assert xt == {}

    def test_with_xt_columns(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "A", "notes": "",
             "xt_race_white": "0.65", "xt_race_black": "0.12"},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls, xt = ingest_mod.load_polls_with_crosstabs("test")
        assert len(polls) == 1
        assert len(xt) == 1
        key = list(xt.keys())[0]
        assert "race" in [d["demographic_group"] for d in xt[key]]

    def test_xt_out_of_range_skipped(self, tmp_path):
        rows = [
            {"race": "test", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "A", "notes": "",
             "xt_race_white": "1.5"},  # > 1.0, should be skipped
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls, xt = ingest_mod.load_polls_with_crosstabs("test")
        assert len(polls) == 1
        assert xt == {}  # All xt entries were invalid

    def test_filters_applied_to_crosstabs(self, tmp_path):
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.50", "n_sample": "500", "date": "2026-01-01",
             "pollster": "A", "notes": "",
             "xt_race_white": "0.60"},
            {"race": "2026 GA Senate", "geography": "GA", "geo_level": "state",
             "dem_share": "0.45", "n_sample": "800", "date": "2026-02-01",
             "pollster": "B", "notes": "",
             "xt_race_white": "0.55"},
        ]
        _setup_cycle_csv(tmp_path, rows, "test")
        polls, xt = ingest_mod.load_polls_with_crosstabs("test", geography="FL")
        assert len(polls) == 1
        assert len(xt) == 1
