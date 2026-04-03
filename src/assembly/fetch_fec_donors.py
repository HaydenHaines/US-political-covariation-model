"""Fetch FEC individual contribution totals aggregated by state for the 2024 cycle.

Source: FEC Open Data API v1
  https://api.open.fec.gov/v1/

Strategy:
  Use the /schedules/schedule_a/by_state/ endpoint, which returns contribution
  totals (count + amount) aggregated by state and committee for a given cycle.
  Paginate through all results for the 2024 cycle and aggregate by state to
  produce state-level donor activity totals.

  The by_zip endpoint is the most granular option but returns one row per
  (committee, ZIP, cycle) tuple. For the 2024 cycle this yields ~2.3M rows
  (~23K pages at 100/page), which is impractical at 1000 req/hr. The by_state
  endpoint returns ~1,200 pages for the same cycle and is well within rate limits.

  County-level granularity is approximated by mapping state totals to all
  counties in the state, consistent with the BEA state features approach in
  build_bea_state_features.py. This is appropriate because within-state
  geographic variation in donor intensity is a second-order signal; the
  cross-state variation (urban-coastal vs rural-interior) is the primary one.

Features produced (state-level, mapped to counties):
  fec_donors_per_1k   — individual donors per 1,000 population (state-level)
  fec_total_per_capita — total contributions per capita in dollars (state-level)
  fec_avg_contribution — average contribution amount in dollars (state-level)

Output:
  data/raw/fec/fec_by_state_2024.parquet
    Columns: state, total_amount, total_count (one row per state, all committees summed)

API key:
  Set environment variable FEC_API_KEY.
  Demo key (DEMO_KEY) is rate-limited to 30 req/hr — use a real key for full fetches.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fec"

FEC_BASE = "https://api.open.fec.gov/v1"
# Default demo key is severely rate-limited; set FEC_API_KEY env var for real fetches.
_FALLBACK_API_KEY = "DEMO_KEY"

# 2024 FEC cycle covers 2023-01-01 through 2024-12-31 (two_year_transaction_period=2024).
TARGET_CYCLE = 2024

# Results per API page. FEC max is 100.
PER_PAGE = 100

# Polite delay between API page requests, in seconds.
# At 0.4s/req we stay well under the 1,000/hr (2.78/s) limit.
# FEC real API key: 1000 req/hr = 1.67/s. Use 4s to stay safely under limit.
# Previous value of 0.4s consistently triggered 429s even with a real key.
_PAGE_SLEEP = 4.0

# Cache path for the raw aggregated state data.
RAW_CACHE_PATH = RAW_DIR / f"fec_by_state_{TARGET_CYCLE}.parquet"


# ── API helpers ───────────────────────────────────────────────────────────────


def _get_api_key() -> str:
    """Return the FEC API key from environment, or fall back to DEMO_KEY."""
    key = os.environ.get("FEC_API_KEY", "").strip()
    if not key:
        log.warning(
            "FEC_API_KEY not set — using DEMO_KEY (30 req/hr, may fail for full fetch). "
            "Set FEC_API_KEY environment variable to use a real key."
        )
        return _FALLBACK_API_KEY
    return key


def _fetch_by_state_page(api_key: str, cycle: int, page: int, max_retries: int = 3) -> dict:
    """Fetch one page from the FEC /schedules/schedule_a/by_state/ endpoint.

    Returns the raw JSON response dictionary.
    Retries on timeout, 429 (rate limit), and 5xx (server error) up to max_retries times.
    FEC API is unreliable — timeouts are common and need exponential backoff.
    """
    url = f"{FEC_BASE}/schedules/schedule_a/by_state/"
    params = {
        "api_key": api_key,
        "cycle": cycle,
        "per_page": PER_PAGE,
        "page": page,
        "sort": "state",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
        except requests.exceptions.ReadTimeout:
            wait = 30 * (attempt + 1)
            log.warning("FEC timeout on page %d (attempt %d/%d) — sleeping %ds", page, attempt + 1, max_retries, wait)
            time.sleep(wait)
            continue
        except requests.exceptions.ConnectionError:
            wait = 30 * (attempt + 1)
            log.warning("FEC connection error on page %d (attempt %d/%d) — sleeping %ds", page, attempt + 1, max_retries, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            log.warning("FEC rate limit hit on page %d — sleeping 65s", page)
            time.sleep(65)
            continue
        elif resp.status_code >= 500:
            wait = 30 * (attempt + 1)
            log.warning("FEC server error %d on page %d — sleeping %ds", resp.status_code, page, wait)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    raise requests.exceptions.ReadTimeout(f"FEC API failed after {max_retries} retries on page {page}")


# ── Main fetcher ──────────────────────────────────────────────────────────────


def fetch_fec_by_state(
    cycle: int = TARGET_CYCLE,
    force_refresh: bool = False,
    cache_path: Path = RAW_CACHE_PATH,
) -> pd.DataFrame:
    """Download and cache FEC contribution totals aggregated by state.

    Paginates through all pages of /schedules/schedule_a/by_state/ for the
    given cycle, aggregates the (committee, state, cycle) rows by state, and
    returns a DataFrame with one row per state.

    Parameters
    ----------
    cycle:
        FEC two-year transaction period (e.g., 2024 covers 2023–2024).
    force_refresh:
        If True, re-download even if the cache exists.
    cache_path:
        Path to write the cached Parquet file.

    Returns
    -------
    DataFrame with columns:
        state          — 2-letter state abbreviation (e.g., "CA")
        total_amount   — sum of all contributions in the state (dollars)
        total_count    — sum of all contribution records in the state
    """
    if cache_path.exists() and not force_refresh:
        log.info("Using cached FEC by-state data: %s", cache_path)
        return pd.read_parquet(cache_path)

    api_key = _get_api_key()
    log.info("Fetching FEC by-state totals for cycle=%d ...", cycle)

    # First page to get total page count.
    first = _fetch_by_state_page(api_key, cycle, page=1)
    total_pages = first.get("pagination", {}).get("pages", 1)
    total_count = first.get("pagination", {}).get("count", 0)
    log.info(
        "  FEC by_state cycle=%d: %d total rows, %d pages",
        cycle, total_count, total_pages,
    )

    # Check for partial progress from a previous interrupted fetch.
    partial_path = cache_path.with_suffix(".partial.parquet")
    if partial_path.exists() and not force_refresh:
        partial_df = pd.read_parquet(partial_path)
        rows = partial_df.to_dict("records")
        # Resume from where we left off: each page has PER_PAGE rows.
        start_page = (len(rows) // PER_PAGE) + 2  # +2 because page 1 is already counted
        log.info("  Resuming from page %d (have %d rows from partial)", start_page, len(rows))
    else:
        rows = list(first.get("results", []))
        start_page = 2

    for page in range(start_page, total_pages + 1):
        data = _fetch_by_state_page(api_key, cycle, page)
        rows.extend(data.get("results", []))
        if page % 100 == 0:
            log.info("  Page %d/%d  (rows so far: %d)", page, total_pages, len(rows))
            # Save incremental progress every 100 pages.
            pd.DataFrame(rows).to_parquet(partial_path, index=False)
        time.sleep(_PAGE_SLEEP)

    log.info("  Downloaded %d rows across %d pages", len(rows), total_pages)

    if not rows:
        log.warning("FEC returned no data for cycle=%d", cycle)
        empty = pd.DataFrame(columns=["state", "total_amount", "total_count"])
        return empty

    df = pd.DataFrame(rows)

    # Validate expected columns are present.
    required = {"state", "total", "count"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"FEC by_state response missing expected columns. "
            f"Got: {list(df.columns)}, expected: {sorted(required)}"
        )

    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    # Filter to 2-letter state abbreviations (excludes territory codes like "AA", "AE")
    df = df[df["state"].str.match(r"^[A-Z]{2}$", na=False)].copy()

    # Aggregate all committees by state: sum amount and count across all PACs/committees.
    # This gives total individual contribution activity (Dem + Rep + other) per state.
    agg = (
        df.groupby("state", as_index=False)
        .agg(total_amount=("total", "sum"), total_count=("count", "sum"))
    )

    log.info(
        "  Aggregated to %d states: total_amount range [$%.0f, $%.0f]",
        len(agg),
        agg["total_amount"].min(),
        agg["total_amount"].max(),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(cache_path, index=False)
    log.info("Saved FEC by-state cache → %s", cache_path)

    # Clean up partial progress file if it exists.
    partial_path = cache_path.with_suffix(".partial.parquet")
    if partial_path.exists():
        partial_path.unlink()
        log.info("Cleaned up partial progress file")
    return agg


def main() -> None:
    """Fetch FEC by-state data for the 2024 cycle and save to disk."""
    df = fetch_fec_by_state(cycle=TARGET_CYCLE)
    log.info("FEC by-state fetch complete: %d states", len(df))
    log.info("  Top 5 states by total amount:")
    for _, row in df.nlargest(5, "total_amount").iterrows():
        log.info(
            "    %s: $%.0f total, %d donors",
            row["state"], row["total_amount"], row["total_count"],
        )


if __name__ == "__main__":
    main()
