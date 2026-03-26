"""Polling domain: polls, poll_crosstabs, poll_notes.

Ingests from polls_{cycle}.csv into DuckDB. Replaces CSV-at-request-time
parsing in /polls and /forecast/polls endpoints.

The `polls` table stores a `notes` VARCHAR column (raw notes string)
alongside the structured `poll_notes` table so endpoints can reconstruct
PollObservation objects via a simple SELECT.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from src.db.domains import DomainIngestionError, DomainSpec

log = logging.getLogger(__name__)

# Truncation length for poll_id SHA-256 hex digest. 16 hex chars = 64 bits
# of entropy — collision probability is negligible for any realistic poll count.
POLL_ID_LENGTH = 16

# Known US state abbreviations for cross-compliance validation.
_US_STATE_ABBRS: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
})

DOMAIN_SPEC = DomainSpec(
    name="polling",
    tables=["polls", "poll_crosstabs", "poll_notes"],
    description="Poll rows ingested from CSV; crosstabs and quality notes queryable via SQL",
    version_key="cycle",
)


class PollIngestRow(BaseModel):
    race: str
    geography: str
    geo_level: Literal["state", "county", "district"]
    dem_share: float = Field(ge=0.0, le=1.0)
    n_sample: int = Field(gt=0)
    date: str | None
    cycle: str


_DDL = """
CREATE TABLE IF NOT EXISTS polls (
    poll_id    VARCHAR NOT NULL,
    race       VARCHAR NOT NULL,
    geography  VARCHAR NOT NULL,
    geo_level  VARCHAR NOT NULL,
    dem_share  FLOAT   NOT NULL,
    n_sample   INTEGER NOT NULL,
    date       VARCHAR,
    pollster   VARCHAR,
    notes      VARCHAR,
    cycle      VARCHAR NOT NULL,
    PRIMARY KEY (poll_id)
);
CREATE TABLE IF NOT EXISTS poll_crosstabs (
    poll_id           VARCHAR NOT NULL,
    demographic_group VARCHAR NOT NULL,
    group_value       VARCHAR NOT NULL,
    dem_share         FLOAT,
    n_sample          INTEGER
);
CREATE TABLE IF NOT EXISTS poll_notes (
    poll_id    VARCHAR NOT NULL,
    note_type  VARCHAR NOT NULL,
    note_value VARCHAR NOT NULL
);
"""


def _cross_compliance(con: duckdb.DuckDBPyConnection, cycle: str) -> None:
    """Validate state-level poll geography values against known US abbreviations."""
    state_geos = con.execute(
        "SELECT DISTINCT geography FROM polls WHERE geo_level='state' AND cycle=?",
        [cycle],
    ).fetchdf()
    if state_geos.empty:
        return
    unknown = [g for g in state_geos["geography"].tolist() if g not in _US_STATE_ABBRS]
    if unknown:
        raise DomainIngestionError(
            "polling", f"polls_{cycle}.csv",
            f"state-level polls with unknown geography (first 5): {unknown[:5]}"
        )


def _make_poll_id(race: str, geography: str, date: str | None, pollster: str | None, cycle: str) -> str:
    """SHA-256 hex digest of pipe-delimited fields, truncated to 16 chars."""
    key = "|".join([race, geography, date or "", pollster or "", cycle])
    return hashlib.sha256(key.encode()).hexdigest()[:POLL_ID_LENGTH]


def _parse_note_kvs(notes_str: str) -> list[tuple[str, str]]:
    """Parse 'grade=A foo=bar' into [('grade','A'), ('foo','bar')]."""
    pairs = []
    for part in notes_str.split():
        if "=" in part:
            k, v = part.split("=", 1)
            pairs.append((k.strip(), v.strip()))
    return pairs


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create all polling domain tables (idempotent)."""
    con.execute(_DDL)


def ingest(con: duckdb.DuckDBPyConnection, cycle: str, project_root: Path) -> None:
    """Ingest polls_{cycle}.csv into DuckDB polling tables.

    Missing CSV is treated as an empty poll set — tables are created
    but left empty. Invalid rows (bad dem_share, zero n_sample) are
    skipped with a warning, matching the existing CSV parse behavior.
    """
    create_tables(con)

    # Delete child rows before parent — subquery reads polls before it is cleared
    con.execute("DELETE FROM poll_crosstabs WHERE poll_id IN (SELECT poll_id FROM polls WHERE cycle=?)", [cycle])
    con.execute("DELETE FROM poll_notes WHERE poll_id IN (SELECT poll_id FROM polls WHERE cycle=?)", [cycle])
    con.execute("DELETE FROM polls WHERE cycle=?", [cycle])

    path = project_root / "data" / "polls" / f"polls_{cycle}.csv"
    if not path.exists():
        log.warning("polls_%s.csv not found at %s; polling tables will be empty", cycle, path)
        return

    poll_rows = []
    note_rows = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_dem = row.get("dem_share", "").strip()
            raw_n = row.get("n_sample", "").strip()
            if not raw_dem or not raw_n:
                continue
            try:
                dem_share = float(raw_dem)
                n_sample = int(float(raw_n))
            except ValueError:
                continue
            if not (0.0 <= dem_share <= 1.0) or n_sample <= 0:
                continue

            race = row.get("race", "").strip()
            geography = row.get("geography", "").strip()
            geo_level = row.get("geo_level", "state").strip() or "state"
            date = row.get("date", "").strip() or None
            pollster = row.get("pollster", "").strip() or None
            notes = row.get("notes", "").strip()

            # Validate via Pydantic (skips rows that fail)
            try:
                PollIngestRow(
                    race=race, geography=geography, geo_level=geo_level,
                    dem_share=dem_share, n_sample=n_sample, date=date, cycle=cycle,
                )
            except Exception as exc:
                log.warning("Skipping invalid poll row (%s): %s", row, exc)
                continue

            poll_id = _make_poll_id(race, geography, date, pollster, cycle)
            poll_rows.append({
                "poll_id": poll_id, "race": race, "geography": geography,
                "geo_level": geo_level, "dem_share": dem_share, "n_sample": n_sample,
                "date": date, "pollster": pollster, "notes": notes, "cycle": cycle,
            })
            for note_type, note_value in _parse_note_kvs(notes):
                note_rows.append({"poll_id": poll_id, "note_type": note_type, "note_value": note_value})

    if poll_rows:
        df = pd.DataFrame(poll_rows)
        con.execute("INSERT INTO polls SELECT * FROM df")
        log.info("polls: ingested %d rows for cycle=%s", len(poll_rows), cycle)

    # Cross-compliance: validate state geography values
    if poll_rows:
        _cross_compliance(con, cycle)
        log.info("Polling domain cross-compliance passed for cycle=%s", cycle)

    if note_rows:
        ndf = pd.DataFrame(note_rows)
        con.execute("INSERT INTO poll_notes SELECT * FROM ndf")
        log.info("poll_notes: ingested %d rows", len(note_rows))
