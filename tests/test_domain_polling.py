"""Tests for polling domain ingest."""
from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pytest

from src.db.domains.polling import ingest, _make_poll_id, POLL_ID_LENGTH


def _base_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def _write_poll_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SAMPLE_ROWS = [
    {"race": "FL Senate", "geography": "FL", "geo_level": "state",
     "dem_share": "0.45", "n_sample": "600", "date": "2026-01-15",
     "pollster": "Siena", "notes": "grade=A"},
    {"race": "FL Senate", "geography": "FL", "geo_level": "state",
     "dem_share": "0.47", "n_sample": "800", "date": "2026-02-01",
     "pollster": "Emerson", "notes": "grade=B+"},
]


def test_ingest_creates_polls_table(tmp_path):
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM polls WHERE cycle='2026'").fetchone()[0]
    assert n == 2


def test_poll_id_is_stable():
    id1 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Siena", "2026")
    id2 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Siena", "2026")
    assert id1 == id2
    assert len(id1) == POLL_ID_LENGTH  # first POLL_ID_LENGTH chars of SHA-256 hex


def test_poll_id_differs_on_different_pollster():
    id1 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Siena", "2026")
    id2 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Emerson", "2026")
    assert id1 != id2


def test_notes_preserved(tmp_path):
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    notes = con.execute("SELECT notes FROM polls WHERE pollster='Siena' AND cycle='2026'").fetchone()[0]
    assert notes == "grade=A"


def test_poll_notes_table_populated(tmp_path):
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM poll_notes WHERE note_type='grade'").fetchone()[0]
    assert n == 2


def test_poll_crosstabs_table_exists_empty(tmp_path):
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
    assert n == 0


def test_missing_csv_returns_empty(tmp_path):
    """Missing CSV creates empty tables (no error).

    Deliberate deviation from the spec's error-table which lists
    DomainIngestionError for missing sources: that rule applies to the
    model domain parquets (required for prediction). Polls are optional
    — a missing poll CSV is valid for historical cycles or future races
    not yet polled. This matches the existing /polls endpoint behavior
    (returns [] on FileNotFoundError).
    """
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
    assert n == 0


def test_invalid_dem_share_row_is_skipped(tmp_path):
    """Rows with out-of-range dem_share are filtered at CSV parse time."""
    rows = SAMPLE_ROWS + [
        {"race": "FL Senate", "geography": "FL", "geo_level": "state",
         "dem_share": "1.5", "n_sample": "600", "date": "2026-03-01",
         "pollster": "Bad Poll", "notes": ""},
    ]
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", rows)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM polls WHERE cycle='2026'").fetchone()[0]
    assert n == 2  # bad row skipped


def test_reingest_is_idempotent(tmp_path):
    """Calling ingest() twice for the same cycle must not double the rows."""
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    ingest(con, "2026", tmp_path)
    n_polls = con.execute("SELECT COUNT(*) FROM polls WHERE cycle='2026'").fetchone()[0]
    n_notes = con.execute("SELECT COUNT(*) FROM poll_notes").fetchone()[0]
    assert n_polls == 2  # not 4
    assert n_notes == 2  # not 4


def test_invalid_state_geography_raises(tmp_path):
    """State-level poll with unknown geography abbreviation should abort ingest."""
    from src.db.domains import DomainIngestionError
    rows = [
        {"race": "FL Senate", "geography": "XX", "geo_level": "state",
         "dem_share": "0.45", "n_sample": "600", "date": "2026-01-15",
         "pollster": "Siena", "notes": "grade=A"},
    ]
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", rows)
    con = _base_db()
    with pytest.raises(DomainIngestionError, match="unknown geography"):
        ingest(con, "2026", tmp_path)
