"""Regression tests for polling domain ingestion (src/db/domains/polling.py).

Covers the bug found 2026-05-04: when polls table was empty at ingest time,
the DELETE-by-subquery pattern left stale poll_crosstabs rows in the DB.
The subsequent batch INSERT then failed with a duplicate-key ConstraintException,
silently aborting the crosstab write and leaving polls/notes committed but
crosstabs missing (or stale).

Root cause: the old delete pattern was
    DELETE FROM poll_crosstabs WHERE poll_id IN (SELECT poll_id FROM polls WHERE cycle=?)
which is a no-op when polls is empty.

Fix: parse all poll_ids from the CSV first, then delete by those IDs, then
delete remaining old polls for the cycle, then insert — in that order.
"""
from __future__ import annotations

import csv
import io
import textwrap
from pathlib import Path

import duckdb
import pytest

from src.db.domains.polling import (
    _make_poll_id,
    create_tables,
    ingest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CYCLE = "2026"

_CSV_HEADER = (
    "race,geography,geo_level,dem_share,n_sample,date,pollster,notes,"
    "xt_education_college,xt_education_noncollege,xt_race_white,xt_race_black\n"
)


def _make_csv(rows: list[dict]) -> str:
    """Build a minimal polls CSV string from a list of row dicts."""
    fields = [
        "race", "geography", "geo_level", "dem_share", "n_sample",
        "date", "pollster", "notes",
        "xt_education_college", "xt_education_noncollege",
        "xt_race_white", "xt_race_black",
    ]
    lines = [",".join(fields)]
    for r in rows:
        lines.append(",".join(str(r.get(f, "")) for f in fields))
    return "\n".join(lines) + "\n"


def _seed_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create all polling tables in an in-memory connection."""
    create_tables(con)


def _ingest_csv(con: duckdb.DuckDBPyConnection, csv_text: str, cycle: str = _CYCLE) -> None:
    """Write csv_text to a temp file and call ingest()."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        polls_dir = root / "data" / "polls"
        polls_dir.mkdir(parents=True)
        csv_path = polls_dir / f"polls_{cycle}.csv"
        csv_path.write_text(csv_text, encoding="utf-8")
        ingest(con, cycle, root)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_POLL_A = {
    "race": "2026 AZ Governor",
    "geography": "AZ",
    "geo_level": "state",
    "dem_share": "0.54",
    "n_sample": "800",
    "date": "2026-01-15",
    "pollster": "TestPoll",
    "notes": "src=test",
    "xt_education_college": "0.45",
    "xt_education_noncollege": "0.55",
    "xt_race_white": "0.62",
    "xt_race_black": "0.12",
}

_POLL_B = {
    "race": "2026 TX Governor",
    "geography": "TX",
    "geo_level": "state",
    "dem_share": "0.41",
    "n_sample": "600",
    "date": "2026-01-20",
    "pollster": "TestPoll",
    "notes": "",
    "xt_education_college": "",
    "xt_education_noncollege": "",
    "xt_race_white": "",
    "xt_race_black": "",
}


# ---------------------------------------------------------------------------
# Basic ingestion correctness
# ---------------------------------------------------------------------------


class TestPollingIngestBasic:
    def test_ingests_poll_rows(self):
        con = duckdb.connect(":memory:")
        _seed_db(con)
        _ingest_csv(con, _make_csv([_POLL_A, _POLL_B]))
        n = con.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
        assert n == 2

    def test_ingests_crosstab_rows(self):
        con = duckdb.connect(":memory:")
        _seed_db(con)
        _ingest_csv(con, _make_csv([_POLL_A]))
        n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
        assert n == 4  # college, noncollege, white, black

    def test_skips_empty_xt_columns(self):
        """Poll B has no crosstab columns — it contributes 0 crosstab rows."""
        con = duckdb.connect(":memory:")
        _seed_db(con)
        _ingest_csv(con, _make_csv([_POLL_B]))
        n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
        assert n == 0

    def test_notes_parsed_into_poll_notes(self):
        con = duckdb.connect(":memory:")
        _seed_db(con)
        _ingest_csv(con, _make_csv([_POLL_A]))
        n = con.execute("SELECT COUNT(*) FROM poll_notes WHERE note_type='src'").fetchone()[0]
        assert n == 1

    def test_poll_id_deterministic(self):
        """Same poll ingested twice should produce the same poll_id."""
        poll_id = _make_poll_id("2026 AZ Governor", "AZ", "2026-01-15", "TestPoll", "2026")
        con = duckdb.connect(":memory:")
        _seed_db(con)
        _ingest_csv(con, _make_csv([_POLL_A]))
        ids = con.execute("SELECT poll_id FROM polls").fetchdf()["poll_id"].tolist()
        assert ids == [poll_id]

    def test_idempotent_double_ingest(self):
        """Running ingest twice must not duplicate rows."""
        con = duckdb.connect(":memory:")
        _seed_db(con)
        csv_text = _make_csv([_POLL_A, _POLL_B])
        _ingest_csv(con, csv_text)
        _ingest_csv(con, csv_text)
        assert con.execute("SELECT COUNT(*) FROM polls").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0] == 4

    def test_dropped_poll_removed_on_reingest(self):
        """If a poll is removed from the CSV, re-ingesting clears it from the DB."""
        con = duckdb.connect(":memory:")
        _seed_db(con)
        _ingest_csv(con, _make_csv([_POLL_A, _POLL_B]))
        assert con.execute("SELECT COUNT(*) FROM polls").fetchone()[0] == 2
        # Remove POLL_B from CSV and re-ingest
        _ingest_csv(con, _make_csv([_POLL_A]))
        assert con.execute("SELECT COUNT(*) FROM polls").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Regression: stale crosstabs cause duplicate-key crash
# ---------------------------------------------------------------------------


class TestPollingIngestStaleCrosstabs:
    """Regression tests for the 2026-05-04 bug.

    Scenario: a prior ingest committed poll_crosstabs rows, then polls was
    cleared by some other mechanism.  On re-ingest the DELETE-by-subquery
    saw an empty polls table, deleted nothing, and the batch INSERT into
    poll_crosstabs raised a ConstraintException on the first duplicate key.
    Result: polls and poll_notes committed but poll_crosstabs silently empty/partial.
    """

    def _setup_stale_state(self, con: duckdb.DuckDBPyConnection) -> None:
        """Simulate the broken state: crosstabs present, polls empty."""
        _seed_db(con)
        # Do a clean first ingest
        _ingest_csv(con, _make_csv([_POLL_A]))
        # Manually clear polls to simulate the broken state
        con.execute("DELETE FROM polls")
        n_polls = con.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
        n_xtabs = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
        assert n_polls == 0, "Setup: polls must be empty"
        assert n_xtabs == 4, "Setup: stale crosstabs must be present"

    def test_second_ingest_does_not_raise(self):
        """Re-ingesting with stale crosstabs must not raise ConstraintException."""
        con = duckdb.connect(":memory:")
        self._setup_stale_state(con)
        # This must complete without raising
        _ingest_csv(con, _make_csv([_POLL_A]))

    def test_polls_fully_populated_after_second_ingest(self):
        """After re-ingest, polls table must contain all rows from the CSV."""
        con = duckdb.connect(":memory:")
        self._setup_stale_state(con)
        _ingest_csv(con, _make_csv([_POLL_A, _POLL_B]))
        n = con.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
        assert n == 2, f"Expected 2 polls, got {n}"

    def test_crosstabs_fully_populated_after_second_ingest(self):
        """After re-ingest, crosstabs must reflect ALL polls from CSV (not stale partial)."""
        con = duckdb.connect(":memory:")
        self._setup_stale_state(con)
        _ingest_csv(con, _make_csv([_POLL_A]))
        n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
        assert n == 4, (
            f"Expected 4 crosstab rows after re-ingest, got {n}. "
            "Stale rows from prior ingest may not have been deleted."
        )

    def test_no_orphaned_crosstabs_after_ingest(self):
        """After ingest, every crosstab row must have a parent in polls."""
        con = duckdb.connect(":memory:")
        self._setup_stale_state(con)
        _ingest_csv(con, _make_csv([_POLL_A]))
        orphans = con.execute("""
            SELECT COUNT(*) FROM poll_crosstabs
            WHERE poll_id NOT IN (SELECT poll_id FROM polls)
        """).fetchone()[0]
        assert orphans == 0, f"{orphans} orphaned crosstab rows found"

    def test_stale_crosstabs_from_different_poll_cleared(self):
        """Crosstabs for a poll dropped from the CSV must be cleaned up.

        Specifically: if an old ingest wrote crosstabs for POLL_A, and the new
        CSV only contains POLL_B, POLL_A's crosstabs must be gone after re-ingest.
        """
        con = duckdb.connect(":memory:")
        _seed_db(con)
        # First ingest: POLL_A only
        _ingest_csv(con, _make_csv([_POLL_A]))
        assert con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0] == 4

        # Clear polls to simulate the broken state
        con.execute("DELETE FROM polls")

        # Second ingest: POLL_B only (no crosstabs, but POLL_A's stale ones still exist)
        _ingest_csv(con, _make_csv([_POLL_B]))

        # POLL_A's crosstabs should be gone since POLL_A is not in the new CSV
        orphans = con.execute("""
            SELECT COUNT(*) FROM poll_crosstabs
            WHERE poll_id NOT IN (SELECT poll_id FROM polls)
        """).fetchone()[0]
        assert orphans == 0, f"{orphans} orphaned (stale) crosstab rows after ingest"


# ---------------------------------------------------------------------------
# validate_predictions catches empty polls
# ---------------------------------------------------------------------------


class TestValidatePredictionsPollingCheck:
    """validate_predictions should flag when polls is empty but predictions exist.

    This is a separate signal from the counties check: if the predictions table
    has data but polls is empty it doesn't crash the API, but it means the
    frontend will show zero polls everywhere.
    """

    def test_no_error_when_polls_populated(self):
        from src.db.validate import validate_predictions

        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE counties (
                county_fips VARCHAR PRIMARY KEY,
                state_abbr VARCHAR,
                county_name VARCHAR,
                total_votes_2024 INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE predictions (
                county_fips VARCHAR, race VARCHAR,
                version_id VARCHAR, forecast_mode VARCHAR,
                pred_dem_share DOUBLE
            )
        """)
        # Enough counties + predictions for the validator to run
        for i, state in enumerate(["MA", "IL", "RI", "NY", "NJ", "WA", "CA", "TX", "IN", "FL"]):
            fips = f"{i+1:05d}"
            con.execute("INSERT INTO counties VALUES (?, ?, ?, ?)", [fips, state, f"{state} Co", 50000])
            con.execute("INSERT INTO predictions VALUES (?, '2026 MA Senate', 'v1', 'local', 0.65)",
                        [fips])
        con.execute("""
            CREATE TABLE polls (
                poll_id VARCHAR PRIMARY KEY,
                race VARCHAR, geography VARCHAR, geo_level VARCHAR,
                dem_share FLOAT, n_sample INTEGER, date VARCHAR,
                pollster VARCHAR, notes VARCHAR, cycle VARCHAR NOT NULL
            )
        """)
        con.execute("INSERT INTO polls VALUES ('abc123', '2026 MA Senate', 'MA', 'state', 0.63, 800, '2026-01-01', 'TestPoll', '', '2026')")
        errors = validate_predictions(con)
        assert not any("POLLS TABLE EMPTY" in e for e in errors)
