"""Tests for polling domain ingest."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pytest

from src.db.domains.polling import POLL_ID_LENGTH, _make_poll_id, ingest

_BASE_FIELDNAMES = [
    "race",
    "geography",
    "geo_level",
    "dem_share",
    "n_sample",
    "date",
    "pollster",
    "notes",
]


def _base_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def _write_poll_csv(path: Path, rows: list[dict], extra_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _BASE_FIELDNAMES + (extra_fields or [])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SAMPLE_ROWS = [
    {
        "race": "FL Senate",
        "geography": "FL",
        "geo_level": "state",
        "dem_share": "0.45",
        "n_sample": "600",
        "date": "2026-01-15",
        "pollster": "Siena",
        "notes": "grade=A",
    },
    {
        "race": "FL Senate",
        "geography": "FL",
        "geo_level": "state",
        "dem_share": "0.47",
        "n_sample": "800",
        "date": "2026-02-01",
        "pollster": "Emerson",
        "notes": "grade=B+",
    },
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
    notes = con.execute(
        "SELECT notes FROM polls WHERE pollster='Siena' AND cycle='2026'"
    ).fetchone()[0]
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
        {
            "race": "FL Senate",
            "geography": "FL",
            "geo_level": "state",
            "dem_share": "1.5",
            "n_sample": "600",
            "date": "2026-03-01",
            "pollster": "Bad Poll",
            "notes": "",
        },
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
        {
            "race": "FL Senate",
            "geography": "XX",
            "geo_level": "state",
            "dem_share": "0.45",
            "n_sample": "600",
            "date": "2026-01-15",
            "pollster": "Siena",
            "notes": "grade=A",
        },
    ]
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", rows)
    con = _base_db()
    with pytest.raises(DomainIngestionError, match="unknown geography"):
        ingest(con, "2026", tmp_path)


# ---------- Phase 4 Step 1: xt_* crosstab parsing ----------


_XT_FIELDS = ["xt_education_college", "xt_education_noncollege", "xt_race_white"]

_ROWS_WITH_XT = [
    {
        "race": "GA Senate",
        "geography": "GA",
        "geo_level": "state",
        "dem_share": "0.52",
        "n_sample": "800",
        "date": "2026-04-01",
        "pollster": "Quinnipiac",
        "notes": "grade=A",
        "xt_education_college": "0.55",
        "xt_education_noncollege": "0.45",
        "xt_race_white": "0.62",
    },
]


def test_xt_columns_parsed_into_crosstabs(tmp_path):
    """xt_* CSV columns are parsed into poll_crosstabs rows."""
    _write_poll_csv(
        tmp_path / "data" / "polls" / "polls_2026.csv",
        _ROWS_WITH_XT,
        extra_fields=_XT_FIELDS,
    )
    con = _base_db()
    ingest(con, "2026", tmp_path)

    rows = con.execute(
        "SELECT demographic_group, group_value, pct_of_sample "
        "FROM poll_crosstabs ORDER BY demographic_group, group_value"
    ).fetchall()
    assert len(rows) == 3
    assert rows[0][0] == "education" and rows[0][1] == "college"
    assert rows[0][2] == pytest.approx(0.55)
    assert rows[1][0] == "education" and rows[1][1] == "noncollege"
    assert rows[1][2] == pytest.approx(0.45)
    assert rows[2][0] == "race" and rows[2][1] == "white"
    assert rows[2][2] == pytest.approx(0.62)


def test_xt_crosstab_poll_id_matches_parent(tmp_path):
    """Crosstab rows reference the correct poll_id from the polls table."""
    _write_poll_csv(
        tmp_path / "data" / "polls" / "polls_2026.csv",
        _ROWS_WITH_XT,
        extra_fields=_XT_FIELDS,
    )
    con = _base_db()
    ingest(con, "2026", tmp_path)

    poll_ids = con.execute("SELECT DISTINCT poll_id FROM polls").fetchall()
    xt_poll_ids = con.execute("SELECT DISTINCT poll_id FROM poll_crosstabs").fetchall()
    assert poll_ids == xt_poll_ids


def test_backward_compat_no_xt_columns(tmp_path):
    """CSV without xt_* columns ingests cleanly with zero crosstab rows."""
    _write_poll_csv(tmp_path / "data" / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)

    n_polls = con.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
    n_xt = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
    assert n_polls == 2
    assert n_xt == 0


def test_empty_xt_values_are_skipped(tmp_path):
    """Rows with empty xt_* values produce no crosstab entries for those columns."""
    rows = [
        {
            "race": "GA Senate",
            "geography": "GA",
            "geo_level": "state",
            "dem_share": "0.52",
            "n_sample": "800",
            "date": "2026-04-01",
            "pollster": "Quinnipiac",
            "notes": "",
            "xt_education_college": "0.55",
            "xt_education_noncollege": "",
            "xt_race_white": "",
        },
    ]
    _write_poll_csv(
        tmp_path / "data" / "polls" / "polls_2026.csv",
        rows,
        extra_fields=_XT_FIELDS,
    )
    con = _base_db()
    ingest(con, "2026", tmp_path)

    n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
    assert n == 1  # only xt_education_college had a value


def test_nan_xt_values_are_skipped(tmp_path):
    """NaN values in xt_* columns are treated as missing."""
    rows = [
        {
            "race": "GA Senate",
            "geography": "GA",
            "geo_level": "state",
            "dem_share": "0.52",
            "n_sample": "800",
            "date": "2026-04-01",
            "pollster": "Quinnipiac",
            "notes": "",
            "xt_education_college": "nan",
            "xt_education_noncollege": "0.45",
            "xt_race_white": "NaN",
        },
    ]
    _write_poll_csv(
        tmp_path / "data" / "polls" / "polls_2026.csv",
        rows,
        extra_fields=_XT_FIELDS,
    )
    con = _base_db()
    ingest(con, "2026", tmp_path)

    n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
    assert n == 1  # only xt_education_noncollege is valid


def test_xt_reingest_is_idempotent(tmp_path):
    """Re-ingesting the same cycle does not duplicate crosstab rows."""
    _write_poll_csv(
        tmp_path / "data" / "polls" / "polls_2026.csv",
        _ROWS_WITH_XT,
        extra_fields=_XT_FIELDS,
    )
    con = _base_db()
    ingest(con, "2026", tmp_path)
    ingest(con, "2026", tmp_path)

    n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
    assert n == 3  # not 6


def test_mixed_rows_with_and_without_xt(tmp_path):
    """A CSV with some rows having xt_* values and others without works correctly."""
    rows = [
        {
            "race": "FL Senate",
            "geography": "FL",
            "geo_level": "state",
            "dem_share": "0.45",
            "n_sample": "600",
            "date": "2026-01-15",
            "pollster": "Siena",
            "notes": "",
            "xt_education_college": "",
            "xt_education_noncollege": "",
            "xt_race_white": "",
        },
        {
            "race": "GA Senate",
            "geography": "GA",
            "geo_level": "state",
            "dem_share": "0.52",
            "n_sample": "800",
            "date": "2026-04-01",
            "pollster": "Quinnipiac",
            "notes": "",
            "xt_education_college": "0.55",
            "xt_education_noncollege": "0.45",
            "xt_race_white": "0.62",
        },
    ]
    _write_poll_csv(
        tmp_path / "data" / "polls" / "polls_2026.csv",
        rows,
        extra_fields=_XT_FIELDS,
    )
    con = _base_db()
    ingest(con, "2026", tmp_path)

    n_polls = con.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
    n_xt = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
    assert n_polls == 2
    assert n_xt == 3  # only the GA poll has crosstab data
