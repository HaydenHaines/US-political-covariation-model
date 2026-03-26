"""
Stage 1 data assembly: fetch BLS Quarterly Census of Employment and Wages (QCEW) data.

Source: Bureau of Labor Statistics (BLS) QCEW — data.bls.gov/cew/data/api/
Data: Annual average employment, wages, and establishment counts by NAICS industry sector
Scope: FL (FIPS 12), GA (FIPS 13), AL (FIPS 01) — county-level data

The QCEW program compiles employment and wage data from state unemployment insurance
programs. It covers ~97% of U.S. jobs in quarterly snapshots aggregated to annual
averages. County-level industry composition (manufacturing share, healthcare share, etc.)
is a powerful predictor of political behavior — deindustrialization, public-sector
dependency, and healthcare worker concentration all correlate with partisan lean.

**BLS QCEW API design**:
The public QCEW API does NOT require an API key. It serves county-level data by
industry sector (aggregation level "county") for a given year and NAICS code.

URL pattern:
  https://data.bls.gov/cew/data/api/{year}/1/industry/{industry}/county/all.csv

Where:
  {year}     : 4-digit year (e.g. 2022)
  1          : Quarter = 1 (annual average is quarter "A"; this API uses "1" for
               the annual file endpoint) — we use the annual slice
  {industry} : NAICS supersector code (2-digit) or "10" for all industries

**NAICS supersector codes** (2-digit aggregation):
  10  : Total, all industries (aggregate)
  11  : Natural Resources and Mining
  21  : Mining, Quarrying, and Oil and Gas Extraction (subset of 11)
  22  : Utilities
  23  : Construction
  31-33: Manufacturing (fetched as "31")  — BLS uses "31" for the 31-33 range
  42  : Wholesale Trade
  44-45: Retail Trade (fetched as "44") — BLS uses "44" for 44-45 range
  48-49: Transportation and Warehousing (fetched as "48")
  51  : Information
  52  : Finance and Insurance
  53  : Real Estate and Rental and Leasing
  54  : Professional and Technical Services
  55  : Management of Companies and Enterprises
  56  : Administrative and Waste Services
  61  : Educational Services
  62  : Health Care and Social Assistance
  71  : Arts, Entertainment, and Recreation
  72  : Accommodation and Food Services
  81  : Other Services
  92  : Public Administration (government)

**County file format** (annual average, CSV):
  area_fips         : 5-digit county FIPS (string)
  own_code          : Ownership code (0=total, 5=private, 1-3=govt)
  industry_code     : NAICS code string
  agglvl_code       : Aggregation level
  size_code         : Establishment size code
  year              : 4-digit year
  qtr               : Quarter code ("A" = annual average)
  disclosure_code   : "N" = not disclosable (suppressed)
  annual_avg_estabs : Annual average establishment count
  annual_avg_emplvl : Annual average employment level
  total_annual_wages: Total annual wages (dollars)
  taxable_annual_wages: Taxable wages
  annual_contributions: UI contributions
  annual_avg_wkly_wage: Average weekly wage
  avg_annual_pay    : Average annual pay

**Output**: data/raw/qcew_county.parquet
  Columns: county_fips, year, industry_code, own_code,
           annual_avg_estabs, annual_avg_emplvl, total_annual_wages

**Derived features** (computed in build_qcew_features.py):
  manufacturing_share    : manufacturing employment / total employment
  government_share       : government employment / total employment
  healthcare_share       : healthcare (NAICS 62) employment / total employment
  retail_share           : retail (NAICS 44-45) employment / total employment
  construction_share     : construction (NAICS 23) employment / total employment
  finance_share          : finance (NAICS 52) employment / total employment
  hospitality_share      : accommodation & food (NAICS 72) / total employment
  industry_diversity_hhi : Herfindahl-Hirschman Index of employment concentration
                           (lower = more diverse, higher = more concentrated)
  top_industry           : NAICS code of the largest sector by employment
  avg_annual_pay         : Average annual pay (total wages / total employment)
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

RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "qcew_county.parquet"

# Per-(year, industry) cache directory — enables idempotent re-runs.
# Each downloaded+filtered CSV is saved as a parquet shard here so the full
# fetcher can resume without re-downloading already-completed combinations.
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "qcew_cache"

# State list comes from config/model.yaml (all 50+DC by default).
# BLS QCEW API is national; we filter to our target state FIPS prefixes.
STATES: dict[str, str] = _cfg.STATES  # abbr → fips prefix

# Set of 2-digit state FIPS prefixes (for filtering)
TARGET_STATE_FIPS = frozenset(STATES.values())

# BLS QCEW API base URL (no API key required for public data)
BASE_URL = "https://data.bls.gov/cew/data/api"

# Years to fetch (annual averages)
DEFAULT_YEARS = [2020, 2021, 2022, 2023]

# Ownership code for "all ownerships" (private + government combined)
OWN_CODE_TOTAL = "0"

# Total industry code (all industries aggregate) — anchor for computing sector shares
TOTAL_INDUSTRY_CODE = "10"

# NAICS industry codes to fetch
# Key: friendly name → NAICS code string used in BLS API
# BLS uses the lower bound of multi-code ranges (31-33 → "31", 44-45 → "44", 48-49 → "48")
INDUSTRY_CODES: dict[str, str] = {
    "total": "10",           # All industries — anchor for shares
    "construction": "23",
    "manufacturing": "31",   # Covers NAICS 31-33
    "retail": "44",          # Covers NAICS 44-45
    "transportation": "48",  # Covers NAICS 48-49
    "finance": "52",
    "healthcare": "62",      # Health Care and Social Assistance
    "hospitality": "72",     # Accommodation and Food Services
    "government": "92",      # Public Administration
}

# Columns to keep from the raw CSV (reduces memory usage)
KEEP_COLUMNS = [
    "area_fips",
    "own_code",
    "industry_code",
    "year",
    "qtr",
    "disclosure_code",
    "annual_avg_estabs",
    "annual_avg_emplvl",
    "total_annual_wages",
    "annual_avg_wkly_wage",
    "avg_annual_pay",
]

# Polite delay between API requests (seconds)
REQUEST_DELAY = 1.0


def build_url(year: int, industry_code: str) -> str:
    """Construct the BLS QCEW API URL for county-level annual data.

    The QCEW "county" endpoint returns ALL counties in one file for a given
    year and industry code. This is the most efficient access pattern —
    one request per (year, industry) covers all 3,000+ US counties.

    Args:
        year: 4-digit data year (e.g. 2022).
        industry_code: NAICS code string (e.g. "10", "62").

    Returns:
        Full URL to the county CSV file on data.bls.gov.
    """
    return f"{BASE_URL}/{year}/A/industry/{industry_code}/county/all.csv"


def fetch_county_csv(year: int, industry_code: str) -> pd.DataFrame | None:
    """Download and parse one BLS QCEW county-level annual CSV file.

    Downloads the CSV from data.bls.gov, selects the columns we need,
    coerces numeric columns, and returns the raw DataFrame before state filtering.

    Args:
        year: 4-digit data year.
        industry_code: NAICS code string.

    Returns:
        Raw DataFrame with KEEP_COLUMNS (subset), or None on download/parse failure.
    """
    url = build_url(year, industry_code)
    log.info("  Downloading %s...", url)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("  HTTP error for %s: %s", url, exc)
        return None

    try:
        df = pd.read_csv(
            io.StringIO(resp.text),
            dtype=str,
            skip_blank_lines=True,
            low_memory=False,
        )
    except Exception as exc:
        log.warning("  Parse error for %s: %s", url, exc)
        return None

    if df.empty:
        log.warning("  Empty CSV for year=%d industry=%s", year, industry_code)
        return None

    # Strip whitespace from column names (BLS sometimes adds spaces)
    df.columns = [c.strip() for c in df.columns]

    # Select only the columns we need (gracefully handle missing columns)
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        log.warning("  Missing columns for year=%d industry=%s: %s", year, industry_code, missing)
    df = df[available].copy()

    # Coerce numeric columns
    numeric_cols = [
        "annual_avg_estabs",
        "annual_avg_emplvl",
        "total_annual_wages",
        "annual_avg_wkly_wage",
        "avg_annual_pay",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce year to int
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    log.info("  Downloaded %d rows", len(df))
    return df


def filter_county_df(
    df: pd.DataFrame,
    year: int,
    industry_code: str,
) -> pd.DataFrame:
    """Filter a raw QCEW county DataFrame to our target states and parameters.

    Applies three filters:
      1. Keep only annual-average rows (qtr == "A")
      2. Keep only total-ownership rows (own_code == "0")
      3. Keep only counties in FL, GA, or AL (area_fips starts with "01", "12", or "13")
      4. Drop suppressed rows (disclosure_code == "N")
      5. Drop non-county FIPS (area_fips like "SSSSS" but state-level or US-level)

    Args:
        df: Raw DataFrame from fetch_county_csv().
        year: Data year (for logging).
        industry_code: NAICS code (for logging).

    Returns:
        Filtered DataFrame with area_fips (5-digit county FIPS), own_code,
        industry_code, year, annual_avg_estabs, annual_avg_emplvl,
        total_annual_wages.
    """
    _empty = pd.DataFrame(
        columns=[
            "county_fips",
            "own_code",
            "industry_code",
            "year",
            "annual_avg_estabs",
            "annual_avg_emplvl",
            "total_annual_wages",
        ]
    )

    if df is None or df.empty:
        return _empty

    n_raw = len(df)

    # 1. Keep only annual average rows
    if "qtr" in df.columns:
        df = df[df["qtr"].astype(str).str.strip() == "A"].copy()
        log.info(
            "  [%d/%s] Annual-average filter: %d → %d rows",
            year, industry_code, n_raw, len(df),
        )

    # 2. Keep total ownership (own_code == "0")
    if "own_code" in df.columns:
        n_before = len(df)
        df = df[df["own_code"].astype(str).str.strip() == OWN_CODE_TOTAL].copy()
        log.info(
            "  [%d/%s] Own_code=0 filter: %d → %d rows",
            year, industry_code, n_before, len(df),
        )

    if df.empty:
        return _empty

    # 3. Keep only county-level FIPS (5 digits where last 3 are not "000")
    #    State-level records have area_fips like "01000", "12000" (county=000)
    #    US-level record has "US000"
    fips_col = "area_fips"
    if fips_col in df.columns:
        df[fips_col] = df[fips_col].astype(str).str.strip().str.zfill(5)
        # Valid county FIPS: 5 digits, numeric, state part in our target set, county != 000
        valid_fips = (
            df[fips_col].str.match(r"^\d{5}$")
            & (df[fips_col].str[:2].isin(TARGET_STATE_FIPS))
            & (df[fips_col].str[2:] != "000")
        )
        n_before = len(df)
        df = df[valid_fips].copy()
        log.info(
            "  [%d/%s] Target-state filter: %d → %d rows",
            year, industry_code, n_before, len(df),
        )

    if df.empty:
        return _empty

    # 4. Drop suppressed rows (disclosure_code == "N" means data withheld)
    if "disclosure_code" in df.columns:
        n_before = len(df)
        suppressed = df["disclosure_code"].astype(str).str.strip() == "N"
        n_suppressed = suppressed.sum()
        if n_suppressed > 0:
            log.info(
                "  [%d/%s] Dropping %d suppressed rows (disclosure_code=N)",
                year, industry_code, n_suppressed,
            )
            df = df[~suppressed].copy()

    if df.empty:
        return _empty

    # Rename area_fips → county_fips and select output columns
    df = df.rename(columns={"area_fips": "county_fips"})

    output_cols = [
        "county_fips",
        "own_code",
        "industry_code",
        "year",
        "annual_avg_estabs",
        "annual_avg_emplvl",
        "total_annual_wages",
    ]
    available_out = [c for c in output_cols if c in df.columns]
    return df[available_out].reset_index(drop=True)


def _cache_path(year: int, industry_code: str) -> Path:
    """Return the parquet cache path for a (year, industry_code) shard."""
    return CACHE_DIR / f"qcew_{year}_{industry_code}.parquet"


def fetch_industry_year(year: int, industry_name: str, industry_code: str) -> pd.DataFrame:
    """Fetch, parse, and filter QCEW data for one industry+year combination.

    Idempotent: if a parquet cache shard already exists for this (year,
    industry_code), the cached data is returned without making an HTTP request.

    Args:
        year: Data year.
        industry_name: Friendly name (for logging).
        industry_code: NAICS code string.

    Returns:
        Filtered county-level DataFrame, or empty DataFrame on failure.
    """
    cache = _cache_path(year, industry_code)
    if cache.exists():
        log.info(
            "  Cache hit: year=%d industry=%s (%s) — loading %s",
            year, industry_code, industry_name, cache.name,
        )
        return pd.read_parquet(cache)

    log.info("Fetching year=%d industry=%s (%s)...", year, industry_code, industry_name)
    raw = fetch_county_csv(year, industry_code)
    if raw is None or raw.empty:
        log.warning("  No data for year=%d industry=%s", year, industry_code)
        return pd.DataFrame()

    filtered = filter_county_df(raw, year, industry_code)
    log.info(
        "  year=%d industry=%s: %d county rows retained",
        year, industry_code, len(filtered),
    )

    # Persist shard for idempotent re-runs
    if not filtered.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(cache, index=False)
        log.info("  Cached → %s", cache.name)

    return filtered


def main(
    years: list[int] | None = None,
    industry_codes: dict[str, str] | None = None,
) -> None:
    """Download BLS QCEW county data and save combined parquet.

    Fetches all combinations of (year, industry_code), filters to FL/GA/AL
    counties, and saves the combined result.

    Args:
        years: List of data years to fetch. Defaults to DEFAULT_YEARS.
        industry_codes: Dict of {name: code}. Defaults to INDUSTRY_CODES.
    """
    if years is None:
        years = DEFAULT_YEARS
    if industry_codes is None:
        industry_codes = INDUSTRY_CODES

    log.info(
        "Fetching BLS QCEW data: %d year(s) × %d industries → %d requests",
        len(years),
        len(industry_codes),
        len(years) * len(industry_codes),
    )
    log.info("Target states: %s", list(STATES.keys()))
    log.info("Years: %s", years)
    log.info("Industries: %s", list(industry_codes.keys()))

    frames: list[pd.DataFrame] = []
    request_count = 0
    total_requests = len(years) * len(industry_codes)

    for year in years:
        for ind_name, ind_code in industry_codes.items():
            df = fetch_industry_year(year, ind_name, ind_code)
            if not df.empty:
                frames.append(df)

            request_count += 1
            if request_count < total_requests:
                time.sleep(REQUEST_DELAY)

    if not frames:
        log.error("No data retrieved for any year/industry combination. Aborting.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Validate FIPS format
    fips_ok = combined["county_fips"].str.match(r"^\d{5}$")
    if not fips_ok.all():
        bad = combined[~fips_ok]["county_fips"].unique()
        log.warning("Non-5-digit FIPS found (dropping): %s", bad[:10])
        combined = combined[fips_ok]

    # Ensure correct types
    for col in ("annual_avg_estabs", "annual_avg_emplvl", "total_annual_wages"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["year"] = combined["year"].astype(int)

    # Summary
    n_rows = len(combined)
    n_counties = combined["county_fips"].nunique()
    n_years = combined["year"].nunique()
    n_industries = combined["industry_code"].nunique()
    log.info(
        "\nSummary: %d rows | %d counties | %d year(s) | %d industry codes",
        n_rows, n_counties, n_years, n_industries,
    )
    for yr, grp in combined.groupby("year"):
        log.info("  %d: %d rows", yr, len(grp))

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(RAW_OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        RAW_OUTPUT_PATH, len(combined), len(combined.columns),
    )


if __name__ == "__main__":
    main()
