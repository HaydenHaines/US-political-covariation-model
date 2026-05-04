"""Ingest Emerson College poll crosstab data from polls_2026.csv into DuckDB.

The scraper (scripts/scrape_emerson_crosstabs.py) populates polls_2026.csv with
two categories of xt_* columns per Emerson poll:

  Composition (pct_of_sample):
    xt_race_white, xt_race_black, xt_race_hispanic, xt_race_asian,
    xt_education_college, xt_education_noncollege, xt_age_senior, ...

  Per-group vote shares (dem_share for Tier 2):
    xt_vote_race_white, xt_vote_race_black, xt_vote_race_hispanic,
    xt_vote_race_asian, xt_vote_education_college, xt_vote_age_senior, ...

This tool reads both column families, pairs them, and upserts records into the
poll_crosstabs DuckDB table, enabling Tier 2 W vector construction (per-group
demographic observations) in the forecast engine.

Run order:
  1. uv run python scripts/scrape_emerson_crosstabs.py  → updates polls_2026.csv
  2. uv run python src/db/build_database.py             → builds DuckDB polls table
  3. uv run python tools/ingest_emerson_crosstabs.py    → populates poll_crosstabs

Step 2 creates poll_crosstabs rows with dem_share=NULL (from xt_* composition
only). Step 3 replaces those rows with fully-populated records including per-group
dem_share. Re-running step 2 will clear dem_share; re-run step 3 to restore.

Usage:
    uv run python tools/ingest_emerson_crosstabs.py
    uv run python tools/ingest_emerson_crosstabs.py --dry-run
    uv run python tools/ingest_emerson_crosstabs.py --db path/to/other.duckdb
    uv run python tools/ingest_emerson_crosstabs.py --cycle 2026
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# Resolve project root from this file's location (tools/ is one level below root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.domains.polling import _make_poll_id, create_tables
from src.prediction.forecast_engine import _extract_crosstabs_from_xt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB = PROJECT_ROOT / "data" / "wethervane.duckdb"
_DEFAULT_CYCLE = "2026"

# Emerson's pollster string as it appears in polls_2026.csv.
EMERSON_POLLSTER = "Emerson College"

# Temporary view name used for DuckDB DataFrame-based inserts (reuses the
# register/unregister pattern from polling.py to avoid heap corruption).
_INSERT_VIEW = "_tmp_emerson_insert"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_emerson_poll_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load Emerson College rows from a polls CSV that have at least one xt_* value.

    Skips rows from other pollsters and rows where every xt_* column is empty
    (no crosstab data to ingest).

    Args:
        csv_path: Path to the polls CSV (e.g. data/polls/polls_2026.csv).

    Returns:
        List of raw CSV row dicts (all values are strings).

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Polls CSV not found: {csv_path}")

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("pollster") != EMERSON_POLLSTER:
                continue
            # Only include rows that carry at least one xt_* composition value.
            # xt_vote_* columns are excluded here — they are dem_share data, not
            # composition.  We check whether there is any composition signal first.
            has_xt = any(
                k.startswith("xt_") and not k.startswith("xt_vote_") and v.strip()
                for k, v in row.items()
            )
            if has_xt:
                rows.append(row)

    log.info("Loaded %d Emerson rows with xt_* data from %s", len(rows), csv_path)
    return rows


# ---------------------------------------------------------------------------
# Crosstab record construction
# ---------------------------------------------------------------------------

def build_crosstab_records(csv_row: dict[str, str], poll_id: str) -> list[dict]:
    """Build poll_crosstabs records from a single Emerson poll CSV row.

    Delegates parsing to _extract_crosstabs_from_xt() so the same demographic
    extraction logic is used in both offline ingestion (this tool) and in-memory
    forecast processing (forecast_engine.py).  String values in the CSV row are
    handled by that function's float() conversion paths.

    For each xt_<group>_<value> composition column with a positive value:
      - pct_of_sample comes from the composition column.
      - dem_share comes from the matching xt_vote_<group>_<value> column when
        present; otherwise falls back to the poll's topline dem_share (same
        behaviour as the in-memory path in _extract_crosstabs_from_xt).
      - n_sample is the subsample size: int(total_n × pct_of_sample).

    Args:
        csv_row: Raw CSV row dict (string values).
        poll_id: Pre-computed poll_id for this row.

    Returns:
        List of dicts with keys matching poll_crosstabs DDL column order:
        poll_id, demographic_group, group_value, dem_share, n_sample, pct_of_sample.
        Empty list if no xt_* data found.

        Key order must match the table DDL so that the register/SELECT * insert
        in ingest_to_db() maps DataFrame columns to table columns by position.
    """
    crosstabs = _extract_crosstabs_from_xt(csv_row)
    if not crosstabs:
        return []

    try:
        total_n = int(float(csv_row.get("n_sample", "") or ""))
    except (ValueError, TypeError):
        total_n = None

    return [
        {
            # Column order MUST match poll_crosstabs DDL:
            # poll_id, demographic_group, group_value, dem_share, n_sample, pct_of_sample
            "poll_id": poll_id,
            "demographic_group": xt["demographic_group"],
            "group_value": xt["group_value"],
            "dem_share": xt["dem_share"],
            "n_sample": int(total_n * xt["pct_of_sample"]) if total_n else None,
            "pct_of_sample": xt["pct_of_sample"],
        }
        for xt in crosstabs
    ]


def _compute_poll_id(csv_row: dict[str, str], cycle: str) -> str:
    """Compute poll_id for a CSV row using the same hash as polling.py.

    Must stay in sync with src/db/domains/polling._make_poll_id so that
    poll_crosstabs.poll_id values match the polls.poll_id values inserted by
    polling.ingest().
    """
    return _make_poll_id(
        race=csv_row.get("race", ""),
        geography=csv_row.get("geography", ""),
        date=csv_row.get("date", "") or None,
        pollster=csv_row.get("pollster", "") or None,
        cycle=cycle,
    )


# ---------------------------------------------------------------------------
# DuckDB ingestion
# ---------------------------------------------------------------------------

def ingest_to_db(
    con: duckdb.DuckDBPyConnection,
    records: list[dict],
    dry_run: bool = False,
) -> int:
    """Upsert poll_crosstabs records into DuckDB.

    For each poll_id in records:
      1. DELETE existing poll_crosstabs rows for that poll_id.
      2. INSERT the new fully-populated rows.

    DELETE + INSERT (rather than UPDATE) handles both first-time inserts and
    re-runs after build_database.py has populated the table with dem_share=NULL
    rows.

    Args:
        con: Open DuckDB connection.
        records: List of dicts from build_crosstab_records().
        dry_run: If True, report what would be ingested without writing.

    Returns:
        Number of records inserted (or that would be inserted in dry_run mode).
    """
    if not records:
        log.info("No records to ingest.")
        return 0

    # Ensure poll_crosstabs (and sibling tables) exist.
    create_tables(con)

    if dry_run:
        affected_polls = len({r["poll_id"] for r in records})
        log.info(
            "DRY-RUN: would insert %d records for %d polls",
            len(records), affected_polls,
        )
        return len(records)

    # Group by poll_id so we can delete-then-insert atomically per poll.
    by_poll: dict[str, list[dict]] = {}
    for rec in records:
        by_poll.setdefault(rec["poll_id"], []).append(rec)

    total_inserted = 0
    for poll_id, rows in by_poll.items():
        con.execute("DELETE FROM poll_crosstabs WHERE poll_id = ?", [poll_id])
        df = pd.DataFrame(rows)
        con.register(_INSERT_VIEW, df)
        try:
            con.execute(f"INSERT INTO poll_crosstabs SELECT * FROM {_INSERT_VIEW}")
            total_inserted += len(rows)
        finally:
            con.unregister(_INSERT_VIEW)
        log.debug("poll_id=%s: inserted %d crosstab rows", poll_id, len(rows))

    return total_inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Emerson College poll crosstab data into DuckDB poll_crosstabs"
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        metavar="PATH",
        help="Path to DuckDB database (default: data/wethervane.duckdb)",
    )
    parser.add_argument(
        "--cycle",
        default=_DEFAULT_CYCLE,
        metavar="YEAR",
        help="Election cycle (default: 2026). Used to locate polls_{cycle}.csv "
             "and to compute poll_id hashes that match the polls table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be ingested without writing to DuckDB",
    )
    args = parser.parse_args(argv)

    csv_path = PROJECT_ROOT / "data" / "polls" / f"polls_{args.cycle}.csv"

    try:
        csv_rows = load_emerson_poll_rows(csv_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    if not csv_rows:
        log.info("No Emerson polls with xt_* data found in %s", csv_path)
        return 0

    # Build crosstab records for every Emerson poll with xt_* data.
    all_records: list[dict] = []
    for row in csv_rows:
        poll_id = _compute_poll_id(row, args.cycle)
        records = build_crosstab_records(row, poll_id)
        all_records.extend(records)
        if records:
            log.info(
                "  %s / %s %s → %d crosstab groups",
                row.get("geography"), row.get("race"), row.get("date"),
                len(records),
            )

    log.info(
        "Built %d crosstab records from %d Emerson poll rows",
        len(all_records), len(csv_rows),
    )

    if not all_records:
        log.info("Nothing to ingest.")
        return 0

    con = duckdb.connect(args.db)
    try:
        n = ingest_to_db(con, all_records, dry_run=args.dry_run)
    finally:
        con.close()

    if args.dry_run:
        log.info("Dry-run complete. Would insert %d records.", n)
    else:
        log.info("Done. Inserted %d poll_crosstabs records.", n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
