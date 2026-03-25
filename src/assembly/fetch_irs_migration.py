"""
Stage 1 data assembly: fetch IRS SOI county-to-county migration flow data.

Source: IRS Statistics of Income (SOI) Division — irs.gov/statistics/soi-tax-stats
Data: County-to-county migration data derived from individual income tax returns
Scope: FL (FIPS 12), GA (FIPS 13), AL (FIPS 01) — flows involving our 3 target states

The IRS SOI program tracks taxpayer mobility by comparing addresses on successive
year tax returns. The result is a county-to-county edge list capturing the direction,
volume, and economic profile of population flows across the United States.

This data is qualitatively different from ACS or RCMS: it is relational (flows between
counties, not properties of counties in isolation). It can be used to build migration
networks, which are powerful inputs to community detection — counties that send and
receive migrants from the same places are likely to share social identity and
behavioral norms.

**Why inflow files only**:
Inflow files record who moved INTO each destination county (y2) from an origin (y1).
Outflow files are the symmetric counterpart — each inflow row has a matching outflow
row from the other county's perspective. We download inflow only to avoid double-
counting and because inflow directly tells us what communities are flowing INTO our
target region. Downstream feature computation can compute net flows as needed.

**Column semantics** (inflow files):
  y2_statefips   : destination state FIPS (2-digit string)
  y2_countyfips  : destination county FIPS (3-digit string)
  y1_statefips   : origin state FIPS (2-digit string)
  y1_countyfips  : origin county FIPS (3-digit string)
  y1_state       : origin state abbreviation
  y1_countyname  : origin county name
  n1             : number of tax returns filed (≈households)
  n2             : number of exemptions claimed (≈individuals)
  agi            : adjusted gross income, in $thousands

**Special FIPS codes** (aggregates — skipped):
  statefips 96  : Total flows (US + Foreign combined)
  statefips 97  : Total US flows (97/0 = all, 97/1 = same state, 97/3 = different state)
  statefips 98  : Foreign country flows
  Same-county   : Non-migrant returns (origin == destination) — "Non-migrants"

**Available year pairs**: CSV format available from 2011-2012 through 2021-2022 (11 pairs).
URL pattern: https://www.irs.gov/pub/irs-soi/countyinflow{YY}{YY}.csv

Output: data/raw/irs_migration.parquet
  Columns: origin_fips, dest_fips, n_returns, n_exemptions, agi, year_pair
  Rows: one per county-pair-year_pair combination (only flows where origin or
        destination is in FL, GA, or AL)
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "irs_migration.parquet"

# State list comes from config/model.yaml (all 50+DC by default).
# IRS migration data is national; we filter to flows touching our target states.
STATES: dict[str, str] = _cfg.STATES  # abbr → fips prefix

# Set of state FIPS codes we care about (for filtering)
TARGET_STATE_FIPS = frozenset(STATES.values())

# IRS SOI base URL for migration CSV files
BASE_URL = "https://www.irs.gov/pub/irs-soi"

# Aggregate / non-migrant FIPS codes to skip
# statefips >= 96 are summary rows, not actual county-to-county flows
AGGREGATE_STATE_FIPS_THRESHOLD = 96

# Available year pairs for inflow files (filing year Y to Y+1)
# Format: (start_year_2digit, end_year_2digit, label)
ALL_YEAR_PAIRS = [
    ("1112", "2011-2012"),
    ("1213", "2012-2013"),
    ("1314", "2013-2014"),
    ("1415", "2014-2015"),
    ("1516", "2015-2016"),
    ("1617", "2016-2017"),
    ("1718", "2017-2018"),
    ("1819", "2018-2019"),
    ("1920", "2019-2020"),
    ("2021", "2020-2021"),
    ("2122", "2021-2022"),
]

# Default: latest 3 year pairs for MVP
DEFAULT_YEAR_PAIRS = ALL_YEAR_PAIRS[-3:]

# Polite delay between downloads (seconds)
REQUEST_DELAY = 1.0

# Inflow file column names (IRS SOI format)
INFLOW_COLUMNS = [
    "y2_statefips",
    "y2_countyfips",
    "y1_statefips",
    "y1_countyfips",
    "y1_state",
    "y1_countyname",
    "n1",
    "n2",
    "agi",
]


def build_url(year_code: str) -> str:
    """Construct the IRS SOI URL for a given year-pair inflow file.

    Args:
        year_code: 4-digit string, e.g. "2122" for the 2021-2022 year pair.

    Returns:
        Full URL to the CSV file on irs.gov.
    """
    return f"{BASE_URL}/countyinflow{year_code}.csv"


def build_fips(statefips: str | int, countyfips: str | int) -> str:
    """Construct a 5-digit county FIPS code from state and county components.

    Args:
        statefips: State FIPS as integer or string (will be zero-padded to 2 digits).
        countyfips: County FIPS as integer or string (will be zero-padded to 3 digits).

    Returns:
        5-digit FIPS string, e.g. "12001".
    """
    return f"{int(statefips):02d}{int(countyfips):03d}"


def is_aggregate_row(statefips: int) -> bool:
    """Return True if the row is an IRS summary/aggregate row, not a real county.

    IRS SOI uses special state FIPS codes >= 96 for summary rows:
      96: Total (US + Foreign)
      97: Total US (intra-state, inter-state subtotals)
      98: Foreign countries

    Args:
        statefips: Integer state FIPS code.

    Returns:
        True if the row should be skipped.
    """
    return statefips >= AGGREGATE_STATE_FIPS_THRESHOLD


def is_nonmigrant_row(
    y1_statefips: int,
    y1_countyfips: int,
    y2_statefips: int,
    y2_countyfips: int,
) -> bool:
    """Return True if origin and destination are the same county (non-migrant row).

    IRS labels these rows "Non-migrants" — returns filed at the same county
    address in both years. They should not appear in a migration edge list.

    Args:
        y1_statefips: Origin state FIPS.
        y1_countyfips: Origin county FIPS.
        y2_statefips: Destination state FIPS.
        y2_countyfips: Destination county FIPS.

    Returns:
        True if origin == destination (same county, same state).
    """
    return y1_statefips == y2_statefips and y1_countyfips == y2_countyfips


def fetch_inflow_csv(year_code: str) -> pd.DataFrame | None:
    """Download and parse one IRS SOI inflow CSV file.

    Downloads the CSV from irs.gov, parses it with the known column schema,
    coerces FIPS and numeric columns to integers, and returns the raw DataFrame
    before any filtering.

    Args:
        year_code: 4-digit year code, e.g. "2122".

    Returns:
        Raw DataFrame with INFLOW_COLUMNS, or None on download/parse failure.
    """
    url = build_url(year_code)
    log.info("  Downloading %s...", url)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("  HTTP error for %s: %s", url, exc)
        return None

    try:
        df = pd.read_csv(
            io.StringIO(resp.text),
            header=0,
            names=INFLOW_COLUMNS,
            dtype=str,
            # IRS files sometimes have a trailing newline or blank rows at end
            skip_blank_lines=True,
        )
    except Exception as exc:
        log.warning("  Parse error for %s: %s", url, exc)
        return None

    # Coerce FIPS columns to numeric (non-numeric rows → NaN → dropped)
    for col in ("y1_statefips", "y1_countyfips", "y2_statefips", "y2_countyfips"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with unparseable FIPS (headers accidentally included, etc.)
    fips_cols = ["y1_statefips", "y1_countyfips", "y2_statefips", "y2_countyfips"]
    df = df.dropna(subset=fips_cols)
    df[fips_cols] = df[fips_cols].astype(int)

    # Coerce numeric measure columns
    for col in ("n1", "n2", "agi"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("  Downloaded %d raw rows", len(df))
    return df


def filter_inflow_df(df: pd.DataFrame, year_label: str) -> pd.DataFrame:
    """Filter a raw inflow DataFrame to the rows we need.

    Applies three filters in order:
      1. Skip aggregate rows (state FIPS >= 96 for either origin or destination)
      2. Skip non-migrant rows (same county in both years)
      3. Keep only rows where origin OR destination is in our target states

    Then constructs 5-digit FIPS codes for origin and destination and selects
    the output columns.

    Args:
        df: Raw inflow DataFrame from fetch_inflow_csv().
        year_label: Human-readable year pair string, e.g. "2021-2022".

    Returns:
        Filtered DataFrame with columns:
          origin_fips, dest_fips, n_returns, n_exemptions, agi, year_pair
    """
    _empty_output = pd.DataFrame(
        columns=["origin_fips", "dest_fips", "n_returns", "n_exemptions", "agi", "year_pair"]
    )

    if df.empty:
        return _empty_output

    n_raw = len(df)

    # 1. Drop aggregate rows (either side has special FIPS >= 96)
    agg_mask = (df["y1_statefips"] >= AGGREGATE_STATE_FIPS_THRESHOLD) | (
        df["y2_statefips"] >= AGGREGATE_STATE_FIPS_THRESHOLD
    )
    df = df[~agg_mask]
    n_after_agg = len(df)
    log.info(
        "  Dropped %d aggregate rows (statefips >= 96); %d remaining",
        n_raw - n_after_agg,
        n_after_agg,
    )

    # 2. Drop non-migrant rows (same county both years)
    nonmig_mask = (df["y1_statefips"] == df["y2_statefips"]) & (
        df["y1_countyfips"] == df["y2_countyfips"]
    )
    df = df[~nonmig_mask]
    n_after_nonmig = len(df)
    log.info(
        "  Dropped %d non-migrant rows; %d remaining",
        n_after_agg - n_after_nonmig,
        n_after_nonmig,
    )

    # 3. Keep only flows involving our target states
    target_fips_ints = {int(f) for f in TARGET_STATE_FIPS}
    state_mask = df["y1_statefips"].isin(target_fips_ints) | df["y2_statefips"].isin(
        target_fips_ints
    )
    df = df[state_mask]
    n_after_state = len(df)
    log.info(
        "  Kept %d rows involving FL/GA/AL (dropped %d out-of-scope)",
        n_after_state,
        n_after_nonmig - n_after_state,
    )

    if df.empty:
        return _empty_output

    # Construct 5-digit FIPS strings using vectorized formatting
    df = df.copy()
    df["origin_fips"] = df["y1_statefips"].astype(int).map(lambda x: f"{x:02d}") + df[
        "y1_countyfips"
    ].astype(int).map(lambda x: f"{x:03d}")
    df["dest_fips"] = df["y2_statefips"].astype(int).map(lambda x: f"{x:02d}") + df[
        "y2_countyfips"
    ].astype(int).map(lambda x: f"{x:03d}")
    df["year_pair"] = year_label

    # Select and rename output columns
    output = df[["origin_fips", "dest_fips", "n1", "n2", "agi", "year_pair"]].copy()
    output = output.rename(columns={"n1": "n_returns", "n2": "n_exemptions"})

    return output.reset_index(drop=True)


def fetch_year_pair(year_code: str, year_label: str) -> pd.DataFrame:
    """Fetch, parse, and filter one IRS inflow year pair.

    Combines fetch_inflow_csv() and filter_inflow_df() with logging.

    Args:
        year_code: 4-digit code, e.g. "2122".
        year_label: Human-readable label, e.g. "2021-2022".

    Returns:
        Filtered edge-list DataFrame, or empty DataFrame on failure.
    """
    log.info("Fetching year pair %s...", year_label)
    raw = fetch_inflow_csv(year_code)
    if raw is None or raw.empty:
        log.warning("  No data for year pair %s", year_label)
        return pd.DataFrame()

    filtered = filter_inflow_df(raw, year_label)
    log.info("  Year pair %s: %d edges retained", year_label, len(filtered))
    return filtered


def main(year_pairs: list[tuple[str, str]] | None = None) -> None:
    """Download IRS migration inflow files and save as a combined edge list.

    Args:
        year_pairs: List of (year_code, year_label) tuples to fetch.
            Defaults to DEFAULT_YEAR_PAIRS (latest 3 year pairs).
    """
    if year_pairs is None:
        year_pairs = DEFAULT_YEAR_PAIRS

    log.info(
        "Fetching IRS SOI county-to-county migration data for %d year pair(s)",
        len(year_pairs),
    )
    log.info("Target states: %s", list(STATES.keys()))
    log.info(
        "Year pairs: %s",
        [label for _, label in year_pairs],
    )
    log.info("Downloading INFLOW files only (outflow is symmetric counterpart).")

    frames: list[pd.DataFrame] = []
    for i, (year_code, year_label) in enumerate(year_pairs):
        df = fetch_year_pair(year_code, year_label)
        if not df.empty:
            frames.append(df)

        # Polite delay between downloads (skip after last)
        if i < len(year_pairs) - 1:
            time.sleep(REQUEST_DELAY)

    if not frames:
        log.error("No data retrieved for any year pair. Aborting.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Summary
    n_edges = len(combined)
    n_year_pairs = combined["year_pair"].nunique()
    n_origins = combined["origin_fips"].nunique()
    n_dests = combined["dest_fips"].nunique()
    log.info(
        "\nSummary: %d edges | %d year pair(s) | %d unique origins | %d unique dests",
        n_edges,
        n_year_pairs,
        n_origins,
        n_dests,
    )

    # Validate FIPS format
    for col in ("origin_fips", "dest_fips"):
        fips_ok = combined[col].str.match(r"^\d{5}$")
        if not fips_ok.all():
            bad = combined[~fips_ok][col].unique()
            log.warning("Non-5-digit FIPS in %s (dropping): %s", col, bad[:10])
            combined = combined[combined["origin_fips"].str.match(r"^\d{5}$")]
            combined = combined[combined["dest_fips"].str.match(r"^\d{5}$")]

    # Per-year summary
    for yp, grp in combined.groupby("year_pair"):
        log.info("  %s: %d edges", yp, len(grp))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        OUTPUT_PATH,
        len(combined),
        len(combined.columns),
    )


if __name__ == "__main__":
    main()
