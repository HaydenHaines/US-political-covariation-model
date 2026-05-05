"""Ingest Quinnipiac University poll crosstab data from press release PDFs.

Downloads a Quinnipiac PDF from poll.qu.edu (or reads a local file), parses
the main election question (Q1: governor or senate race), extracts vote-share
data by demographic group, and upserts records into the poll_crosstabs DuckDB
table.

Data available in press release PDFs:
  - Vote share per party (republican, democrat, independent)
  - Vote share per gender (men, women)
  - Vote share per education (college, noncollege)  ← PA-style detailed polls
  - Vote share per race (white, black)              ← PA-style detailed polls
  - Vote share per age group (18_34, 35_49, 50_64, 65_plus)  ← PA-style only
  NOTE: pct_of_sample is stored as NULL — sample composition (unweighted base
  counts) is not published in Quinnipiac press release PDFs.

URL pattern: https://poll.qu.edu/images/polling/{state}/{code}.pdf
Verified 2026 coverage: NJ Governor (Oct/Sep 2025); PA Governor (Oct 2025, Feb 2026)

Usage:
    uv run python tools/ingest_quinnipiac_crosstabs.py \\
        --url https://poll.qu.edu/images/polling/pa/pa10012025_piss74.pdf \\
        --race "2026 PA Governor" \\
        --geography PA \\
        --date 2025-10-01 \\
        [--cycle 2026] \\
        [--dry-run]
    uv run python tools/ingest_quinnipiac_crosstabs.py \\
        --pdf /path/to/file.pdf \\
        --race "2026 NJ Governor" \\
        --geography NJ \\
        --date 2025-10-30 \\
        [--cycle 2026] \\
        [--dry-run]
    uv run python tools/ingest_quinnipiac_crosstabs.py \\
        --url ... --race ... --geography ... --date ... --db path/to/other.duckdb
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import pdfplumber
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.domains.polling import _make_poll_id, create_tables

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB = PROJECT_ROOT / "data" / "wethervane.duckdb"
_DEFAULT_CYCLE = "2026"
QUINNIPIAC_POLLSTER = "Quinnipiac University"

_INSERT_VIEW = "_tmp_qu_insert"

# Regex to find the Q1 election block: from "1. If the election for..."
# up to (but not including) the next sub-question or main question.
_Q1_BLOCK_RE = re.compile(
    r"(1\.\s+If the election.+?)(?=\n1a\.|\n2\.|\n1b\.)",
    re.DOTALL | re.IGNORECASE,
)

# Regex to extract Democrat candidate's name from question text.
# Handles: "were Mikie Sherrill the Democrat, Jack..." or "Josh Shapiro the Democrat and..."
_DEM_NAME_RE = re.compile(
    r"candidates were (.+?) the Democrat",
    re.IGNORECASE,
)

# Regex for parsing a percentage value like "55%", "39", "-", "--"
_PCT_RE = re.compile(r"^-+$|^(\d+(?:\.\d+)?)%?$")


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

def download_pdf(url: str, timeout: int = 30) -> bytes:
    """Download a PDF from a URL and return its raw bytes.

    Args:
        url: Full URL to the PDF.
        timeout: Request timeout in seconds.

    Returns:
        Raw PDF bytes.

    Raises:
        requests.HTTPError: If the response status is not 2xx.
        requests.RequestException: On network errors.
    """
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "WetherVane/1.0"})
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def _parse_pct(token: str) -> float | None:
    """Convert '55%', '39', '-', '--' to a [0,1] float or None."""
    token = token.strip().rstrip("%")
    if re.match(r"^-+$", token) or not token:
        return None
    try:
        v = float(token)
        return v / 100.0
    except ValueError:
        return None


def _column_map(headers: list[str], context: str) -> dict[int, tuple[str, str] | None]:
    """Map header token index → (demographic_group, group_value) or None.

    Args:
        headers: Tokenized column header line, e.g. ['Tot','Rep','Dem','Ind','Men','Wom','Yes','No'].
        context: The few lines of text preceding the header line (used to detect
                 whether 'Yes'/'No' mean college degree and whether we're in the
                 race/age sub-table where 'Men'/'Wom' are White sub-groups).

    Returns:
        Dict mapping column index to (group, value) pair, or None to skip.
    """
    is_race_age = any(h in ("18-34", "35-49", "50-64", "65+", "Wht", "Blk") for h in headers)
    has_coll_deg = bool(re.search(r"COLL\s*DEG|COLLEGE\s*DEG", context, re.IGNORECASE))

    mapping: dict[int, tuple[str, str] | None] = {}
    for i, h in enumerate(headers):
        if h in ("Tot",):
            mapping[i] = None
        elif h == "Rep":
            mapping[i] = ("party", "republican")
        elif h == "Dem":
            mapping[i] = ("party", "democrat")
        elif h == "Ind":
            mapping[i] = ("party", "independent")
        elif h == "Men":
            # Skip in race/age table (those are White Men, not overall Men)
            mapping[i] = None if is_race_age else ("gender", "men")
        elif h == "Wom":
            mapping[i] = None if is_race_age else ("gender", "women")
        elif h == "Yes":
            mapping[i] = ("education", "college") if has_coll_deg else None
        elif h == "No":
            mapping[i] = ("education", "noncollege") if has_coll_deg else None
        elif h == "18-34":
            mapping[i] = ("age", "18_34")
        elif h == "35-49":
            mapping[i] = ("age", "35_49")
        elif h == "50-64":
            mapping[i] = ("age", "50_64")
        elif h == "65+":
            mapping[i] = ("age", "65_plus")
        elif h == "Wht":
            mapping[i] = ("race", "white")
        elif h == "Blk":
            mapping[i] = ("race", "black")
        else:
            mapping[i] = None
    return mapping


def _is_data_row(tokens: list[str]) -> bool:
    """Return True if this looks like a candidate/response data row (has numeric values)."""
    if len(tokens) < 2:
        return False
    # At least one token after the first must look like a number or '-'
    return any(re.match(r"^-+$|^\d+%?$", t) for t in tokens[1:4])


def _parse_tables_from_block(block_text: str, dem_last_name: str) -> list[dict]:
    """Parse all crosstab tables in a question block and return democrat's row data.

    Scans for header lines (containing 'Tot' and at least one of 'Rep'/'18-34'),
    determines column mapping from header context, then looks for the Democratic
    candidate's row and extracts dem_share per demographic group.

    Args:
        block_text: Text of the Q1 election question block.
        dem_last_name: Last name of the Democratic candidate (for row detection).

    Returns:
        List of dicts with keys: demographic_group, group_value, dem_share.
        pct_of_sample is omitted (not available from press release PDFs).
    """
    lines = block_text.split("\n")
    results: list[dict] = []
    seen_groups: set[tuple[str, str]] = set()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()

        # Detect a crosstab header line: starts with 'Tot' or contains '18-34'
        if not tokens:
            i += 1
            continue

        is_std_header = tokens[0] == "Tot" and any(t in tokens for t in ("Rep", "Dem"))
        is_race_header = "18-34" in tokens or ("Wht" in tokens and "Blk" in tokens)

        if not (is_std_header or is_race_header):
            i += 1
            continue

        # Gather context (up to 5 lines before header)
        context = "\n".join(lines[max(0, i - 5) : i])
        col_map = _column_map(tokens, context)

        # Scan forward to find the Democrat's row
        j = i + 1
        found = False
        while j < len(lines) and j < i + 15:
            dline = lines[j].strip()
            dtokens = dline.split()
            if not dtokens:
                j += 1
                continue

            # Stop if we hit the next question or header
            if re.match(r"^\d+\w*\.", dtokens[0]) or dtokens[0] == "Tot":
                break

            # Check if this is the Democrat's row
            is_dem_row = (
                dem_last_name.lower() in dtokens[0].lower()
                or dtokens[0].lower().startswith(dem_last_name[:4].lower())
            ) and _is_data_row(dtokens)

            if is_dem_row:
                values = dtokens[1:]  # strip candidate name
                for col_idx, group_pair in col_map.items():
                    if group_pair is None:
                        continue
                    group, value = group_pair
                    if col_idx < len(values):
                        pct = _parse_pct(values[col_idx])
                        if pct is not None and (group, value) not in seen_groups:
                            results.append({
                                "demographic_group": group,
                                "group_value": value,
                                "dem_share": pct,
                            })
                            seen_groups.add((group, value))
                found = True
                break

            j += 1

        if not found:
            log.debug("No dem row found for header at line %d: %s", i, line)

        i += 1

    return results


def parse_pdf_crosstabs(pdf_bytes: bytes) -> tuple[list[dict], str | None]:
    """Parse a Quinnipiac press release PDF and extract election crosstab data.

    Finds the Q1 election question block, extracts the Democratic candidate's
    vote share across all demographic groups present in the PDF tables.

    Args:
        pdf_bytes: Raw PDF bytes (from download_pdf or open().read()).

    Returns:
        Tuple of (crosstab_records, dem_name_detected):
          - crosstab_records: List of dicts with demographic_group, group_value, dem_share.
            Empty list if no election question found or no data parsed.
          - dem_name_detected: Full name string of the Democratic candidate, or None.
    """
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        all_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Find Q1 block
    m = _Q1_BLOCK_RE.search(all_text)
    if not m:
        log.warning("No Q1 election question block found in PDF")
        return [], None

    block = m.group(1)

    # Normalize whitespace in block for name detection (names can span line breaks)
    block_flat = re.sub(r"\s+", " ", block)

    # Extract Democrat's name
    name_m = _DEM_NAME_RE.search(block_flat)
    if not name_m:
        log.warning("Could not identify Democratic candidate name from Q1 text")
        return [], None

    dem_full_name = name_m.group(1).strip()
    dem_last_name = dem_full_name.split()[-1]
    log.info("Democrat candidate: %s (matching rows starting with '%s')", dem_full_name, dem_last_name)

    records = _parse_tables_from_block(block, dem_last_name)
    return records, dem_full_name


# ---------------------------------------------------------------------------
# DuckDB ingestion
# ---------------------------------------------------------------------------

def build_crosstab_records(
    parsed_rows: list[dict],
    poll_id: str,
    total_n: int | None,
) -> list[dict]:
    """Build poll_crosstabs DB records from parsed crosstab rows.

    Args:
        parsed_rows: Output from parse_pdf_crosstabs() — list of {demographic_group,
                     group_value, dem_share}.
        poll_id: Pre-computed poll_id matching the polls table hash.
        total_n: Total poll sample size (n_sample), if known; else None.

    Returns:
        List of dicts with schema: poll_id, demographic_group, group_value,
        dem_share, n_sample, pct_of_sample (always NULL for Quinnipiac press PDFs).
    """
    return [
        {
            "poll_id": poll_id,
            "demographic_group": row["demographic_group"],
            "group_value": row["group_value"],
            "dem_share": row["dem_share"],
            "n_sample": None,   # Quinnipiac press PDFs don't publish per-group n
            "pct_of_sample": None,  # Sample composition not in press release PDFs
        }
        for row in parsed_rows
    ]


def ingest_to_db(
    con: duckdb.DuckDBPyConnection,
    records: list[dict],
    dry_run: bool = False,
) -> int:
    """Upsert poll_crosstabs records into DuckDB (delete-then-insert per poll_id).

    Args:
        con: Open DuckDB connection.
        records: List of dicts from build_crosstab_records().
        dry_run: If True, report what would be done without writing.

    Returns:
        Number of records inserted (or that would be inserted in dry-run mode).
    """
    if not records:
        log.info("No records to ingest.")
        return 0

    create_tables(con)

    if dry_run:
        affected_polls = len({r["poll_id"] for r in records})
        log.info("DRY-RUN: would insert %d records for %d polls", len(records), affected_polls)
        return len(records)

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
        description=(
            "Ingest Quinnipiac University poll crosstab data from a press release PDF "
            "into DuckDB poll_crosstabs."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", metavar="URL", help="URL to a Quinnipiac press release PDF")
    src.add_argument("--pdf", metavar="PATH", help="Local path to a Quinnipiac press release PDF")

    parser.add_argument(
        "--race", required=True, metavar="RACE",
        help="Race name, e.g. '2026 PA Governor'",
    )
    parser.add_argument(
        "--geography", required=True, metavar="GEO",
        help="State abbreviation, e.g. 'PA'",
    )
    parser.add_argument(
        "--date", required=True, metavar="DATE",
        help="Poll date YYYY-MM-DD (release date or field end date)",
    )
    parser.add_argument(
        "--pollster", default=QUINNIPIAC_POLLSTER, metavar="POLLSTER",
        help=f"Pollster name (default: '{QUINNIPIAC_POLLSTER}')",
    )
    parser.add_argument(
        "--n-sample", type=int, default=None, metavar="N",
        help="Total sample size (if known; used to compute n_sample per group)",
    )
    parser.add_argument(
        "--cycle", default=_DEFAULT_CYCLE, metavar="YEAR",
        help="Election cycle (default: 2026)",
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB), metavar="PATH",
        help="Path to DuckDB database (default: data/wethervane.duckdb)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be ingested without writing to DuckDB",
    )
    args = parser.parse_args(argv)

    # Load PDF
    if args.url:
        try:
            pdf_bytes = download_pdf(args.url)
        except requests.RequestException as exc:
            log.error("Failed to download PDF: %s", exc)
            return 1
    else:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            log.error("PDF file not found: %s", pdf_path)
            return 1
        pdf_bytes = pdf_path.read_bytes()

    # Parse PDF
    parsed_rows, dem_name = parse_pdf_crosstabs(pdf_bytes)

    if not parsed_rows:
        log.warning("No crosstab data extracted from PDF. Check that this is a Quinnipiac election poll PDF.")
        return 1

    log.info("Extracted %d demographic groups from PDF (Democrat: %s)", len(parsed_rows), dem_name or "?")
    for row in parsed_rows:
        log.info(
            "  %s / %s → dem_share=%.1f%%",
            row["demographic_group"], row["group_value"], row["dem_share"] * 100,
        )

    # Compute poll_id
    poll_id = _make_poll_id(
        race=args.race,
        geography=args.geography,
        date=args.date,
        pollster=args.pollster,
        cycle=args.cycle,
    )
    log.info("poll_id: %s (race=%s, geography=%s, date=%s)", poll_id, args.race, args.geography, args.date)

    # Build DB records
    db_records = build_crosstab_records(parsed_rows, poll_id, args.n_sample)

    # Ingest
    con = duckdb.connect(args.db)
    try:
        n = ingest_to_db(con, db_records, dry_run=args.dry_run)
    finally:
        con.close()

    if args.dry_run:
        log.info("Dry-run complete. Would insert %d records.", n)
    else:
        log.info("Done. Inserted %d poll_crosstabs records.", n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
