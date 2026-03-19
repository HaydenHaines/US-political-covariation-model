"""
Stage 1 data assembly: fetch LEHD LODES 8 commuting flow data.

Source: LEHD Origin-Destination Employment Statistics (LODES) — census.gov
Data: LODES 8 OD (origin-destination) main files, all jobs (JT00)
Scope: FL (FIPS 12), GA (FIPS 13), AL (FIPS 01) — commuting flows involving target states

LODES provides daily commuting flow data at the census block level: how many people
live in one block and work in another. This is a proxy for "who you interact with" —
commuting networks reveal economic and social connections between counties. Counties
that exchange commuters share labor markets and social ties, making commuting networks
a powerful input for community detection.

**OD file format (gzipped CSV)**:
  w_geocode  : 15-digit work census block FIPS
  h_geocode  : 15-digit home census block FIPS
  S000       : total number of jobs (all workers)
  SA01       : workers age 29 or younger
  SA02       : workers age 30 to 54
  SA03       : workers age 55 or older
  SE01       : workers earning $1,250/month or less
  SE02       : workers earning $1,251 to $3,333/month
  SE03       : workers earning more than $3,333/month
  SI01       : workers in Goods Producing industries
  SI02       : workers in Trade, Transportation, and Utilities
  SI03       : workers in All Other Services
  createdate : date string

**Processing**:
1. Download gzipped CSV for each state+year combination
2. Extract county FIPS (first 5 digits) from 15-digit block FIPS
3. Aggregate S000 to county-to-county level by (home_county, work_county, year)
4. Filter to flows involving target states (FL/GA/AL)
5. Drop intra-county flows (home_county == work_county)
6. Save to parquet

Files are large (~50MB gzipped, ~500MB uncompressed for FL), so we use chunked
reading to keep memory usage bounded.

Output: data/raw/lodes_commuting.parquet
  Columns: home_fips, work_fips, total_jobs, year
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "lodes_commuting.parquet"

# Target states: abbreviation → (FIPS prefix, lowercase abbreviation for URL)
STATES = {"AL": ("01", "al"), "FL": ("12", "fl"), "GA": ("13", "ga")}

# Set of 2-digit state FIPS prefixes for filtering
TARGET_STATE_FIPS = frozenset(fips for fips, _ in STATES.values())

# LODES 8 base URL
BASE_URL = "https://lehd.ces.census.gov/data/lodes/LODES8"

# Default years to fetch (most recent complete LODES 8 data)
DEFAULT_YEARS = [2020, 2021, 2022]

# Chunked reading size for large CSV files
CHUNK_SIZE = 500_000

# Columns to read from the OD file (only what we need)
OD_USECOLS = ["w_geocode", "h_geocode", "S000"]

# Polite delay between downloads (seconds)
REQUEST_DELAY = 1.0


def build_url(state_lower: str, year: int) -> str:
    """Construct the LODES 8 OD main file URL for a given state and year.

    Args:
        state_lower: Lowercase 2-letter state abbreviation (e.g. "fl").
        year: Data year (e.g. 2021).

    Returns:
        Full URL to the gzipped CSV on census.gov.
    """
    return f"{BASE_URL}/{state_lower}/od/{state_lower}_od_main_JT00_{year}.csv.gz"


def extract_county_fips(geocode: str | int) -> str:
    """Extract the 5-digit county FIPS from a 15-digit census block FIPS.

    Census block FIPS codes are structured as:
      SS CCC TTTTTT BBB
      2-digit state + 3-digit county + 6-digit tract + 3/4-digit block

    The first 5 digits give the county FIPS.

    Args:
        geocode: 15-digit census block FIPS code (string or integer).

    Returns:
        5-digit county FIPS string, zero-padded.
    """
    s = str(geocode).zfill(15)
    return s[:5]


def aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a chunk of block-level OD data to county-level.

    Extracts county FIPS from the 15-digit geocodes, then groups by
    (home_county, work_county) and sums S000 (total jobs).

    Args:
        chunk: DataFrame with columns w_geocode, h_geocode, S000.

    Returns:
        Aggregated DataFrame with columns home_fips, work_fips, total_jobs.
    """
    chunk = chunk.copy()
    chunk["home_fips"] = chunk["h_geocode"].astype(str).str.zfill(15).str[:5]
    chunk["work_fips"] = chunk["w_geocode"].astype(str).str.zfill(15).str[:5]

    agg = (
        chunk.groupby(["home_fips", "work_fips"], as_index=False)["S000"]
        .sum()
        .rename(columns={"S000": "total_jobs"})
    )
    return agg


def filter_commuting_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter aggregated commuting flows to target states and remove intra-county.

    Applies two filters:
      1. Keep only flows where home OR work county is in a target state (FL/GA/AL)
      2. Drop intra-county flows (home_fips == work_fips)

    Args:
        df: Aggregated DataFrame with columns home_fips, work_fips, total_jobs.

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    n_raw = len(df)

    # 1. Keep only flows involving target states
    home_state = df["home_fips"].str[:2]
    work_state = df["work_fips"].str[:2]
    state_mask = home_state.isin(TARGET_STATE_FIPS) | work_state.isin(TARGET_STATE_FIPS)
    df = df[state_mask].copy()
    n_after_state = len(df)
    log.info(
        "  Kept %d flows involving FL/GA/AL (dropped %d out-of-scope)",
        n_after_state,
        n_raw - n_after_state,
    )

    # 2. Drop intra-county flows (commuters who live and work in the same county)
    intra_mask = df["home_fips"] == df["work_fips"]
    n_intra = intra_mask.sum()
    df = df[~intra_mask].copy()
    log.info(
        "  Dropped %d intra-county flows; %d inter-county flows remaining",
        n_intra,
        len(df),
    )

    return df.reset_index(drop=True)


def fetch_state_year(state_abbr: str, state_lower: str, year: int) -> pd.DataFrame:
    """Download and process one LODES OD file for a state+year combination.

    Uses chunked reading to handle large files without excessive memory usage.
    Each chunk is aggregated to county level before combining.

    Args:
        state_abbr: Uppercase state abbreviation (e.g. "FL") — for logging only.
        state_lower: Lowercase state abbreviation (e.g. "fl") — for URL construction.
        year: Data year.

    Returns:
        County-level aggregated DataFrame with columns home_fips, work_fips, total_jobs,
        or empty DataFrame on failure.
    """
    url = build_url(state_lower, year)
    log.info("  Downloading %s %d from %s...", state_abbr, year, url)

    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("  HTTP error for %s %d: %s", state_abbr, year, exc)
        return pd.DataFrame()

    # Write to a temporary file so pandas can read gzipped chunks
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as tmp:
        for data in resp.iter_content(chunk_size=8192):
            tmp.write(data)
        tmp_path = tmp.name

    try:
        chunk_frames: list[pd.DataFrame] = []
        chunk_num = 0

        for chunk in pd.read_csv(
            tmp_path,
            compression="gzip",
            usecols=OD_USECOLS,
            dtype={"w_geocode": str, "h_geocode": str, "S000": "Int64"},
            chunksize=CHUNK_SIZE,
        ):
            chunk_num += 1
            agg = aggregate_chunk(chunk)
            chunk_frames.append(agg)
            if chunk_num % 5 == 0:
                log.info("    Processing chunk %d...", chunk_num)

        log.info("    Processed %d chunk(s) for %s %d", chunk_num, state_abbr, year)
    except Exception as exc:
        log.warning("  Parse error for %s %d: %s", state_abbr, year, exc)
        return pd.DataFrame()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not chunk_frames:
        return pd.DataFrame()

    # Combine chunk aggregates and re-aggregate (same county pair may span chunks)
    combined = pd.concat(chunk_frames, ignore_index=True)
    combined = (
        combined.groupby(["home_fips", "work_fips"], as_index=False)["total_jobs"]
        .sum()
    )

    log.info(
        "  %s %d: %d county-pairs, %d total commuters",
        state_abbr,
        year,
        len(combined),
        combined["total_jobs"].sum(),
    )
    return combined


def main(years: list[int] | None = None) -> None:
    """Download LODES commuting data and save as a combined edge list.

    Args:
        years: List of data years to fetch. Defaults to DEFAULT_YEARS.
    """
    if years is None:
        years = DEFAULT_YEARS

    log.info("Fetching LODES 8 commuting data for %d state(s), %d year(s)", len(STATES), len(years))
    log.info("Target states: %s", list(STATES.keys()))
    log.info("Years: %s", years)

    frames: list[pd.DataFrame] = []
    download_count = 0

    for state_abbr, (state_fips, state_lower) in STATES.items():
        for year in years:
            df = fetch_state_year(state_abbr, state_lower, year)
            if not df.empty:
                df["year"] = year
                frames.append(df)

            download_count += 1
            # Polite delay between downloads (skip after last)
            total_downloads = len(STATES) * len(years)
            if download_count < total_downloads:
                time.sleep(REQUEST_DELAY)

    if not frames:
        log.error("No data retrieved for any state/year combination. Aborting.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Re-aggregate across states (a flow from FL county to GA county appears in both
    # state files — we need to deduplicate by summing, but actually each state file
    # only contains blocks *within* that state as the work location, so the same
    # county pair can appear from two different state downloads if home is in one
    # state and work in another). Actually, OD main files are per-state based on
    # the *work* state — so there should be no duplicates. But to be safe, aggregate.
    combined = (
        combined.groupby(["home_fips", "work_fips", "year"], as_index=False)["total_jobs"]
        .sum()
    )

    # Filter to target states and remove intra-county flows
    combined = filter_commuting_flows(combined)

    if combined.empty:
        log.error("No commuting flows remaining after filtering. Aborting.")
        return

    # Ensure correct types
    combined["total_jobs"] = combined["total_jobs"].astype(int)
    combined["year"] = combined["year"].astype(int)

    # Validate FIPS format
    for col in ("home_fips", "work_fips"):
        fips_ok = combined[col].str.match(r"^\d{5}$")
        if not fips_ok.all():
            bad = combined[~fips_ok][col].unique()
            log.warning("Non-5-digit FIPS in %s (dropping): %s", col, bad[:10])
            combined = combined[combined["home_fips"].str.match(r"^\d{5}$")]
            combined = combined[combined["work_fips"].str.match(r"^\d{5}$")]

    # Summary
    n_flows = len(combined)
    n_years = combined["year"].nunique()
    n_home = combined["home_fips"].nunique()
    n_work = combined["work_fips"].nunique()
    log.info(
        "\nSummary: %d flows | %d year(s) | %d unique home counties | %d unique work counties",
        n_flows,
        n_years,
        n_home,
        n_work,
    )

    # Per-year summary
    for yr, grp in combined.groupby("year"):
        log.info("  %d: %d flows, %d total commuters", yr, len(grp), grp["total_jobs"].sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved -> %s  (%d rows x %d cols)",
        OUTPUT_PATH,
        len(combined),
        len(combined.columns),
    )


if __name__ == "__main__":
    main()
