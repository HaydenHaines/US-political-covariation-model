"""
Scrape 2026 primary election polls from RealClearPolling.

Primary polls are within-party contests (e.g. the Republican primary for
Georgia's 2026 US Senate race).  They don't fit the existing two-party
``dem_share`` schema used for general election polls, so they get their own
output file with a richer candidate-level schema.

Output: ``data/polls/primary_polls_2026.csv`` with the columns:

    race_key, geography, geo_level, party, date, pollster, n_sample,
    candidates_json, is_primary, notes

``candidates_json`` is a JSON array of ``{"name": str, "pct": float}``
objects, sorted by ``pct`` descending, so downstream consumers can identify
the leader without having to re-sort.

Usage::

    uv run python scripts/scrape_rcp_primaries.py
    uv run python scripts/scrape_rcp_primaries.py --dry-run
    uv run python scripts/scrape_rcp_primaries.py --races "GA Senate R"

This script reuses the Next.js JSON extraction logic from
``scrape_2026_polls.py``.  The User-Agent and 2-second request delay are
inherited to avoid the 403 issues that bit the generic ballot scraper (S587).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the general-election scraper's JSON extraction, HTTP helper, and
# field parsers — primary polls share RCP's Next.js embedded-JSON format.
from scrape_2026_polls import (  # noqa: E402
    _RCP_AVERAGE_TYPES,
    RCP_BASE_URL,
    REQUEST_DELAY,
    _extract_rcp_polls_json,
    extract_pct,
    extract_sample_size,
    fetch_html,
    normalize_pollster,
    parse_poll_date,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "polls" / "primary_polls_2026.csv"

OUTPUT_COLUMNS = [
    "race_key",
    "geography",
    "geo_level",
    "party",
    "date",
    "pollster",
    "n_sample",
    "candidates_json",
    "is_primary",
    "notes",
]


# ---------------------------------------------------------------------------
# Primary race configuration
# ---------------------------------------------------------------------------
# Each entry maps a race_key to scraping config.  RCP's primary URL pattern:
#     /polls/{senate|governor}/{republican|democratic}-primary/{year}/{state}
# Note the URL has NO candidate suffix — unlike general-election matchup pages
# (e.g. "/ossoff-vs-carter"), primary pages aggregate all polled candidates at
# the state-level URL.
#
# URLs below were verified against live RCP on 2026-04-21 by scanning RCP's
# cross-links on a confirmed-live page.  If RCP adds new 2026 primaries, drop
# them in here; a missing URL returns [] from fetch_html with no side effects.
PRIMARY_RACE_CONFIG: dict[str, dict] = {
    "2026 GA Senate Republican Primary": {
        "state": "GA",
        "party": "R",
        "rcp_urls": ["/polls/senate/republican-primary/2026/georgia"],
    },
    "2026 TX Senate Republican Primary": {
        "state": "TX",
        "party": "R",
        "rcp_urls": ["/polls/senate/republican-primary/2026/texas"],
    },
    "2026 TX Senate Democratic Primary": {
        "state": "TX",
        "party": "D",
        "rcp_urls": ["/polls/senate/democratic-primary/2026/texas"],
    },
    "2026 IL Senate Democratic Primary": {
        "state": "IL",
        "party": "D",
        "rcp_urls": ["/polls/senate/democratic-primary/2026/illinois"],
    },
    "2026 GA Governor Republican Primary": {
        "state": "GA",
        "party": "R",
        "rcp_urls": ["/polls/governor/republican-primary/2026/georgia"],
    },
    "2026 OK Governor Republican Primary": {
        "state": "OK",
        "party": "R",
        "rcp_urls": ["/polls/governor/republican-primary/2026/oklahoma"],
    },
    "2026 SC Governor Republican Primary": {
        "state": "SC",
        "party": "R",
        "rcp_urls": ["/polls/governor/republican-primary/2026/south-carolina"],
    },
    "2026 RI Governor Democratic Primary": {
        "state": "RI",
        "party": "D",
        "rcp_urls": ["/polls/governor/democratic-primary/2026/rhode-island"],
    },
}


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
def scrape_rcp_primary(race_key: str, state: str, party: str, url: str) -> list[dict]:
    """Scrape a single RealClearPolling primary-election page.

    Primary polls list all candidates of a single party.  Unlike general
    election polls, we don't convert to a two-party share — every
    candidate's percentage is preserved in ``candidates_json`` so downstream
    consumers can reason about multi-way races, leader margins, etc.

    Returns a list of poll dicts (one per poll entry, excluding the
    RCP Average composite row).
    """
    full_url = f"{RCP_BASE_URL}{url}"
    logger.info("  RCP primary: %s", full_url)
    html = fetch_html(full_url)
    if not html:
        return []

    raw_polls = _extract_rcp_polls_json(html)
    if raw_polls is None:
        logger.warning("  RCP primary: could not extract poll JSON from %s", full_url)
        return []

    polls: list[dict] = []
    for entry in raw_polls:
        # Skip composite average rows — not actual polls.
        if entry.get("type") in _RCP_AVERAGE_TYPES or entry.get("pollster") == "rcp_average":
            continue

        pollster_raw = entry.get("pollster_group_name") or entry.get("pollster") or ""
        if not pollster_raw:
            continue

        # Prefer data_end_date (includes year); fall back to the short date.
        date_end = entry.get("data_end_date", "")
        if date_end:
            date_parsed = date_end.replace("/", "-")
        else:
            date_raw = entry.get("date", "")
            if date_raw and re.match(r"^\d{1,2}/\d{1,2}\s*-\s*\d{1,2}/\d{1,2}$", date_raw.strip()):
                date_raw = f"{date_raw}/2026"
            date_parsed = parse_poll_date(date_raw)

        sample_raw = entry.get("sampleSize", "")
        n_sample = extract_sample_size(sample_raw)
        sample_type = ""
        if sample_raw:
            s_upper = sample_raw.upper()
            if "LV" in s_upper:
                sample_type = "LV"
            elif "RV" in s_upper:
                sample_type = "RV"

        # Preserve every candidate with their raw percentage.
        candidates: list[dict] = []
        for cand in entry.get("candidate", []):
            name = str(cand.get("name", "")).strip()
            val = extract_pct(cand.get("value"))
            if name and val is not None:
                candidates.append({"name": name, "pct": val})

        if not candidates:
            continue
        # Sort by pct descending so the leader is always candidates[0].
        candidates.sort(key=lambda c: c["pct"], reverse=True)

        polls.append(
            {
                "race_key": race_key,
                "geography": state,
                "geo_level": "state",
                "party": party,
                "pollster_raw": pollster_raw.strip(),
                "pollster": normalize_pollster(pollster_raw),
                "date": date_parsed,
                "n_sample": n_sample,
                "candidates": candidates,
                "source": "rcp",
                "sample_type": sample_type,
                "is_primary": True,
            }
        )

    logger.info("  RCP primary: %d polls for %s", len(polls), race_key)
    return polls


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def primary_dedup_key(poll: dict) -> tuple:
    """Dedup key for primary polls: (race_key, date, normalized_pollster)."""
    return (poll["race_key"], poll.get("date", ""), poll["pollster"].lower())


def deduplicate_primaries(polls: list[dict]) -> list[dict]:
    """Drop duplicates across multiple RCP URLs that point at the same poll.

    When the same pollster/date/race appears in multiple scraped URLs (e.g. a
    two-way and a three-way matchup page both showing the same Emerson poll),
    keep the one with the most candidates — that's the richer record.
    """
    seen: dict[tuple, dict] = {}
    for p in polls:
        key = primary_dedup_key(p)
        if key not in seen:
            seen[key] = p
            continue
        # Prefer the poll with more candidates (richer record)
        if len(p.get("candidates", [])) > len(seen[key].get("candidates", [])):
            seen[key] = p
    n_removed = len(polls) - len(seen)
    if n_removed > 0:
        logger.info("Deduplication removed %d duplicate primary polls", n_removed)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Output DataFrame
# ---------------------------------------------------------------------------
def build_primary_output_df(polls: list[dict]) -> pd.DataFrame:
    """Convert primary poll dicts to the output CSV schema."""
    rows = []
    for p in polls:
        candidates = p.get("candidates", [])
        candidates_json = json.dumps(candidates)

        notes_parts: list[str] = []
        if candidates:
            leader = candidates[0]
            notes_parts.append(f"lead={leader['name']}@{leader['pct']:.1f}%")
        if p.get("sample_type"):
            notes_parts.append(p["sample_type"])
        notes_parts.append(f"src={p.get('source', 'rcp')}")

        rows.append(
            {
                "race_key": p["race_key"],
                "geography": p["geography"],
                "geo_level": p["geo_level"],
                "party": p["party"],
                "date": p.get("date", ""),
                "pollster": p["pollster"],
                "n_sample": p.get("n_sample", ""),
                "candidates_json": candidates_json,
                "is_primary": True,
                "notes": "; ".join(notes_parts),
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if df.empty:
        return df
    return df.sort_values(["race_key", "date"], ascending=[True, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _matches_filter(race_key: str, race_filter: list[str]) -> bool:
    """Case-insensitive substring match: any filter term in the race_key."""
    rk_lower = race_key.lower()
    return any(term.strip().lower() in rk_lower for term in race_filter)


def main():
    parser = argparse.ArgumentParser(description="Scrape 2026 primary election polls")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to CSV",
    )
    parser.add_argument(
        "--races",
        type=str,
        default=None,
        help=("Comma-separated race filter (substring match on race_key). Example: 'GA Senate R,TX Senate'"),
    )
    args = parser.parse_args()

    races = PRIMARY_RACE_CONFIG
    if args.races:
        race_filter = args.races.split(",")
        races = {k: v for k, v in PRIMARY_RACE_CONFIG.items() if _matches_filter(k, race_filter)}
        if not races:
            logger.error("No matching races found. Available: %s", list(PRIMARY_RACE_CONFIG.keys()))
            return

    all_polls: list[dict] = []
    request_count = 0

    for race_key, cfg in races.items():
        logger.info("--- Scraping %s ---", race_key)
        state = cfg["state"]
        party = cfg["party"]

        for url in cfg.get("rcp_urls", []):
            if request_count > 0:
                time.sleep(REQUEST_DELAY)
            polls = scrape_rcp_primary(race_key, state, party, url)
            all_polls.extend(polls)
            request_count += 1

    logger.info("=== Raw total: %d primary polls from RCP ===", len(all_polls))

    deduped = deduplicate_primaries(all_polls)
    logger.info("=== After dedup: %d primary polls ===", len(deduped))

    df = build_primary_output_df(deduped)

    logger.info("=== Per-race primary poll counts ===")
    for race_key in races:
        count = len(df[df["race_key"] == race_key])
        logger.info("  %s: %d polls", race_key, count)
    logger.info("  TOTAL: %d polls", len(df))

    if args.dry_run:
        logger.info("=== DRY RUN -- printing output ===")
        print(df.to_csv(index=False))
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing primary polls so historical data survives scraper
    # gaps (e.g., RCP removes a URL after the primary is decided).
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        merged = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["race_key", "date", "pollster"], keep="last"
        )
        merged = merged.sort_values(["race_key", "date"]).reset_index(drop=True)
        n_new = len(merged) - len(existing)
        logger.info(
            "Merged %d new polls into %d existing → %d total",
            max(n_new, 0),
            len(existing),
            len(merged),
        )
        df = merged

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Written %d primary polls to %s", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
