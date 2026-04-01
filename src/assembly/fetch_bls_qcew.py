"""
Stage 1 data assembly: fetch BLS Quarterly Census of Employment and Wages (QCEW) data.

Source: Bureau of Labor Statistics (BLS) QCEW — data.bls.gov/cew/data/files/
Data: Annual average employment, wages, and establishment counts by NAICS industry sector
Scope: All 50 states + DC — national county-level data

The QCEW program compiles employment and wage data from state unemployment insurance
programs. It covers ~97% of U.S. jobs in quarterly snapshots aggregated to annual
averages. County-level industry composition (manufacturing share, healthcare share, etc.)
is a powerful predictor of political behavior — deindustrialization, public-sector
dependency, and healthcare worker concentration all correlate with partisan lean.

**BLS QCEW bulk download design**:
BLS publishes annual singlefile ZIP archives (one per year, ~74-80MB each) at:

  https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip

Each ZIP contains a single CSV (e.g. 2022.annual.singlefile.csv) with all counties,
all ownership codes, all industries, all aggregation levels. We download one ZIP per
year, extract the CSV in-memory, and filter to:
  - qtr == "A" (annual average)
  - area_fips is a 5-digit county FIPS (not state-level "SS000" or "US000")
  - area_fips state prefix is in TARGET_STATE_FIPS (all 50+DC)
  - industry_code in our target set
  - own_code == "0" for industry_code "10" (total, all ownerships aggregated by BLS)
  - For sector codes, own_code is NOT "0" in the singlefile (BLS only pre-aggregates
    totals at the all-industry level). We sum across own_codes 1+2+3+5 to reconstruct
    total-ownership employment per sector. This matches what the old per-industry
    API endpoint returned as own_code=0.

**NAICS supersector codes** (2-digit aggregation):
  10  : Total, all industries (aggregate) — own_code=0 available directly
  23  : Construction
  31  : Manufacturing (covers NAICS 31-33)
  44  : Retail Trade (covers NAICS 44-45)
  48  : Transportation and Warehousing (covers NAICS 48-49)
  52  : Finance and Insurance
  62  : Health Care and Social Assistance
  72  : Accommodation and Food Services
  92  : Public Administration (government)

**Singlefile CSV format** (annual average, partial columns used):
  area_fips         : 5-digit county FIPS (string)
  own_code          : Ownership code (0=all totals, 1=federal, 2=state, 3=local, 5=private)
  industry_code     : NAICS code string (can be 2-digit supersector or full 6-digit)
  agglvl_code       : Aggregation level (70=county total, 74=county supersector, etc.)
  year              : 4-digit year
  qtr               : Quarter code ("A" = annual average)
  disclosure_code   : "N" = not disclosable (suppressed)
  annual_avg_estabs : Annual average establishment count
  annual_avg_emplvl : Annual average employment level
  total_annual_wages: Total annual wages (dollars)
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
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]

RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "qcew_county.parquet"

# Per-year cache directory — enables idempotent re-runs.
# Each processed annual singlefile is saved here so subsequent runs skip
# already-downloaded years.
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "qcew_cache"

# State list comes from config/model.yaml (all 50+DC by default).
# BLS QCEW singlefile is national; we filter to our target state FIPS prefixes.
STATES: dict[str, str] = _cfg.STATES  # abbr → fips prefix

# Set of 2-digit state FIPS prefixes (for filtering)
TARGET_STATE_FIPS = frozenset(STATES.values())

# BLS QCEW bulk download base URL (no API key required)
BULK_BASE_URL = "https://data.bls.gov/cew/data/files"

# BLS QCEW per-industry API base URL (kept for build_url() backward compat / tests)
BASE_URL = "https://data.bls.gov/cew/data/api"

# Years to fetch (annual averages)
DEFAULT_YEARS = [2020, 2021, 2022, 2023]

# Ownership code for "all ownerships" (private + government combined)
OWN_CODE_TOTAL = "0"

# Total industry code (all industries aggregate) — anchor for computing sector shares
TOTAL_INDUSTRY_CODE = "10"

# NAICS industry codes to fetch
# Key: friendly name → NAICS code string as it appears in the BLS singlefile CSV
# Note: manufacturing, retail, and transportation use range codes in the singlefile
# (e.g. "31-33" not "31") — these are the agglvl=74 supersector codes BLS uses.
INDUSTRY_CODES: dict[str, str] = {
    "total": "10",           # All industries — anchor for shares (own_code=0, agglvl=70)
    "construction": "23",
    "manufacturing": "31-33",  # NAICS 31-33 range — BLS singlefile uses "31-33"
    "retail": "44-45",          # NAICS 44-45 range — BLS singlefile uses "44-45"
    "transportation": "48-49",  # NAICS 48-49 range — BLS singlefile uses "48-49"
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

# Polite delay between annual file downloads (seconds)
REQUEST_DELAY = 2.0

# Ownership codes that appear in the singlefile for non-total-ownership rows
# We sum these to reconstruct the total-ownership figure for sector codes.
# own_code=0 (pre-aggregated total) only exists for industry_code="10" in singlefiles.
# For sector codes (23, 31, 44, 48, 52, 62, 72, 92) we sum:
#   own_code=1 (federal govt) + 2 (state govt) + 3 (local govt) + 5 (private)
OWN_CODES_TO_SUM = frozenset({"1", "2", "3", "5"})

# FIPS validation: county FIPS are exactly 5 digits; county suffix "000" means state-level
FIPS_PATTERN = r"^\d{5}$"
STATE_LEVEL_COUNTY_SUFFIX = "000"

# Numeric columns that need coercion from string in the raw CSV
NUMERIC_COLUMNS = [
    "annual_avg_estabs",
    "annual_avg_emplvl",
    "total_annual_wages",
    "annual_avg_wkly_wage",
    "avg_annual_pay",
]

# Output columns for assembled county-level parquet
OUTPUT_COLUMNS = [
    "county_fips",
    "own_code",
    "industry_code",
    "year",
    "annual_avg_estabs",
    "annual_avg_emplvl",
    "total_annual_wages",
]

# Suppression indicator in BLS disclosure_code column
SUPPRESSED_CODE = "N"

# Annual average quarter code in BLS data
ANNUAL_QTR_CODE = "A"


def build_url(year: int, industry_code: str) -> str:
    """Construct the BLS QCEW API URL for county-level annual data.

    Note: This endpoint has historically been unreliable (timeouts, 404s).
    The fetcher now uses bulk singlefile downloads instead; this function is
    retained for backward compatibility with tests.

    Args:
        year: 4-digit data year (e.g. 2022).
        industry_code: NAICS code string (e.g. "10", "62").

    Returns:
        Full URL to the county CSV file on data.bls.gov.
    """
    return f"{BASE_URL}/{year}/A/industry/{industry_code}/county/all.csv"


def _singlefile_url(year: int) -> str:
    """Return the BLS QCEW annual singlefile ZIP URL for a given year."""
    return f"{BULK_BASE_URL}/{year}/csv/{year}_annual_singlefile.zip"


def _cache_path(year: int) -> Path:
    """Return the parquet cache path for a processed annual singlefile."""
    return CACHE_DIR / f"qcew_{year}_all_industries.parquet"


def download_singlefile(year: int) -> bytes | None:
    """Download the BLS QCEW annual singlefile ZIP for a given year.

    Args:
        year: 4-digit data year.

    Returns:
        Raw ZIP bytes, or None on failure.
    """
    url = _singlefile_url(year)
    log.info("  Downloading %s...", url)
    try:
        resp = requests.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        data = resp.content
        log.info("  Downloaded %.1f MB", len(data) / 1_048_576)
        return data
    except requests.RequestException as exc:
        log.warning("  HTTP error for %s: %s", url, exc)
        return None


def _select_and_coerce_columns(
    df: pd.DataFrame,
    context: str,
) -> pd.DataFrame:
    """Select KEEP_COLUMNS and coerce numeric types from a raw BLS CSV DataFrame.

    Shared helper used by both parse_singlefile() and fetch_county_csv().
    Strips whitespace from column names, selects target columns, and coerces
    numeric fields from string dtype (BLS CSVs are read as all-string).

    Args:
        df: Raw DataFrame read directly from a BLS CSV.
        context: Human-readable label for warning messages (e.g. "year=2022").

    Returns:
        DataFrame with KEEP_COLUMNS subset and numeric columns coerced.
    """
    df.columns = [c.strip() for c in df.columns]

    available = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        log.warning("  Missing columns for %s: %s", context, missing)
    df = df[available].copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df


def _open_csv_from_zip(zip_bytes: bytes, year: int) -> tuple[zipfile.ZipFile, str] | None:
    """Open a QCEW annual singlefile ZIP and locate the CSV entry inside it.

    Args:
        zip_bytes: Raw ZIP archive bytes.
        year: Data year (for logging).

    Returns:
        (ZipFile, csv_filename) tuple, or None if the ZIP is malformed or has no CSV.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        log.warning("  Bad ZIP for year=%d: %s", year, exc)
        return None

    names = zf.namelist()
    if not names:
        log.warning("  Empty ZIP for year=%d", year)
        return None

    csv_names = [n for n in names if n.endswith(".csv")]
    if not csv_names:
        log.warning("  No CSV in ZIP for year=%d (files: %s)", year, names)
        return None

    return zf, csv_names[0]


def parse_singlefile(zip_bytes: bytes, year: int) -> pd.DataFrame | None:
    """Extract and parse the CSV from a QCEW annual singlefile ZIP.

    Args:
        zip_bytes: Raw ZIP archive bytes.
        year: Data year (for logging).

    Returns:
        Parsed DataFrame with KEEP_COLUMNS subset, or None on failure.
    """
    result = _open_csv_from_zip(zip_bytes, year)
    if result is None:
        return None
    zf, csv_name = result

    log.info("  Parsing %s...", csv_name)
    try:
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                dtype=str,
                skip_blank_lines=True,
                low_memory=False,
                encoding="latin-1",
            )
    except Exception as exc:
        log.warning("  Parse error for year=%d: %s", year, exc)
        return None

    if df.empty:
        log.warning("  Empty CSV in ZIP for year=%d", year)
        return None

    df = _select_and_coerce_columns(df, context=f"year={year}")
    log.info("  Parsed %d rows from %s", len(df), csv_name)
    return df


def _empty_output_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the standard output column schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _filter_annual_rows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Keep only annual-average rows (qtr == "A").

    BLS singlefiles include quarterly and annual rows. We only want "A"
    (annual average), which aggregates all four quarters into a single row.
    """
    if "qtr" not in df.columns:
        return df
    n_before = len(df)
    df = df[df["qtr"].astype(str).str.strip() == ANNUAL_QTR_CODE].copy()
    log.info("  [%d] Annual-average filter: %d → %d rows", year, n_before, len(df))
    return df


def _filter_county_fips(df: pd.DataFrame, fips_col: str, year: int) -> pd.DataFrame:
    """Keep only county-level FIPS codes for our target states.

    Drops:
    - State-level aggregates (e.g. "01000" — county suffix "000")
    - National aggregate ("US000")
    - States outside TARGET_STATE_FIPS
    - Any non-numeric or non-5-digit FIPS

    Normalizes FIPS to zero-padded 5-character strings in place.
    """
    if fips_col not in df.columns:
        return df
    df[fips_col] = df[fips_col].astype(str).str.strip().str.zfill(5)
    valid_fips = (
        df[fips_col].str.match(FIPS_PATTERN)
        & df[fips_col].str[:2].isin(TARGET_STATE_FIPS)
        & (df[fips_col].str[2:] != STATE_LEVEL_COUNTY_SUFFIX)
    )
    n_before = len(df)
    df = df[valid_fips].copy()
    log.info("  [%d] County FIPS filter: %d → %d rows", year, n_before, len(df))
    return df


def _filter_target_industries(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Keep only rows whose industry_code is in our configured target set."""
    if "industry_code" not in df.columns:
        return df
    target_codes = set(INDUSTRY_CODES.values())
    n_before = len(df)
    df = df[df["industry_code"].astype(str).str.strip().isin(target_codes)].copy()
    log.info("  [%d] Industry filter: %d → %d rows", year, n_before, len(df))
    return df


def _drop_suppressed(df: pd.DataFrame, year: int, label: str) -> pd.DataFrame:
    """Drop rows where disclosure_code == "N" (data suppressed by BLS).

    Suppressed rows have withheld values to protect employer privacy. Including
    them in sums would introduce systematic downward bias for small-county sectors.
    """
    if "disclosure_code" not in df.columns:
        return df
    suppressed = df["disclosure_code"].astype(str).str.strip() == SUPPRESSED_CODE
    n_dropped = suppressed.sum()
    if n_dropped > 0:
        log.info("  [%d] Dropped %d suppressed %s rows", year, n_dropped, label)
    return df[~suppressed].copy()


def _aggregate_total_industry(df_total: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Extract pre-aggregated total-industry rows (own_code=0) from the singlefile.

    BLS pre-aggregates industry_code="10" (all industries) with own_code=0 at
    agglvl=70 (county total). These rows exist directly in the singlefile and
    represent the total employment/wages across all sectors and ownerships.

    Returns the filtered rows, or None if none survive.
    """
    if df_total.empty:
        return None

    total_own0 = df_total[df_total["own_code"] == OWN_CODE_TOTAL].copy()
    total_own0 = _drop_suppressed(total_own0, year, "total-industry")
    if total_own0.empty:
        return None

    total_own0["own_code"] = OWN_CODE_TOTAL
    log.info("  [%d] Total-industry rows (own_code=0): %d", year, len(total_own0))
    return total_own0


def _aggregate_sector_rows(
    df_sectors: pd.DataFrame,
    fips_col: str,
    year: int,
) -> pd.DataFrame | None:
    """Sum sector employment across ownership codes to produce total-ownership rows.

    For sector codes (23, 31-33, 44-45, etc.), the BLS singlefile does NOT
    include pre-aggregated own_code=0 rows. We reconstruct them by summing
    employment and wages across own_codes 1, 2, 3, 5 (federal, state, local,
    private). This matches what the old per-industry API returned as own_code=0.

    Suppressed rows (disclosure_code="N") are dropped before summing to avoid
    introducing systematic bias.

    Returns the aggregated DataFrame, or None if no data survives.
    """
    if df_sectors.empty:
        return None

    df_sectors = _drop_suppressed(df_sectors, year, "sector")
    df_sectors = df_sectors[df_sectors["own_code"].isin(OWN_CODES_TO_SUM)].copy()

    if df_sectors.empty:
        return None

    # Sum employment and wages by (county, industry, year) across ownership codes
    agg_cols = {
        col: "sum"
        for col in ("annual_avg_estabs", "annual_avg_emplvl", "total_annual_wages")
        if col in df_sectors.columns
    }
    sector_agg = (
        df_sectors
        .groupby([fips_col, "industry_code", "year"], as_index=False)
        .agg(agg_cols)
    )
    sector_agg["own_code"] = OWN_CODE_TOTAL  # label reconstructed total as own_code=0
    log.info("  [%d] Sector aggregated rows: %d", year, len(sector_agg))
    return sector_agg


def filter_singlefile(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter and aggregate a parsed QCEW singlefile DataFrame.

    Produces a county x industry DataFrame equivalent to what the old
    per-industry API returned (own_code == "0", all industries).

    Processing steps:
      1. Keep only annual-average rows (qtr == "A")
      2. Keep only county-level FIPS (5-digit numeric, county != "000",
         state prefix in TARGET_STATE_FIPS)
      3. Keep only our target industry codes (INDUSTRY_CODES values)
      4. For industry_code "10" (total): keep own_code == "0" rows directly
         (BLS pre-aggregates these in the singlefile at agglvl=70)
      5. For sector industry codes: sum employment/wages across
         own_codes {1, 2, 3, 5} to reconstruct total-ownership figures,
         dropping suppressed rows (disclosure_code == "N") before summing
      6. Return with own_code set to "0" for all rows (consistent with the
         old API output format)

    Args:
        df: Parsed DataFrame from parse_singlefile().
        year: Data year (for logging).

    Returns:
        Filtered/aggregated county-level DataFrame with columns:
          county_fips, own_code, industry_code, year,
          annual_avg_estabs, annual_avg_emplvl, total_annual_wages
    """
    if df is None or df.empty:
        return _empty_output_frame()

    fips_col = "area_fips"

    # Steps 1-3: row-level filters
    df = _filter_annual_rows(df, year)
    if df.empty:
        return _empty_output_frame()

    df = _filter_county_fips(df, fips_col, year)
    if df.empty:
        return _empty_output_frame()

    df = _filter_target_industries(df, year)
    if df.empty:
        return _empty_output_frame()

    if "own_code" in df.columns:
        df["own_code"] = df["own_code"].astype(str).str.strip()

    # Steps 4-5: separate total-industry (pre-aggregated) from sector codes (needs summing)
    is_total = df["industry_code"].astype(str).str.strip() == TOTAL_INDUSTRY_CODE
    frames_out = [
        f for f in [
            _aggregate_total_industry(df[is_total].copy(), year),
            _aggregate_sector_rows(df[~is_total].copy(), fips_col, year),
        ]
        if f is not None
    ]

    if not frames_out:
        return _empty_output_frame()

    combined = pd.concat(frames_out, ignore_index=True)
    combined = combined.rename(columns={fips_col: "county_fips"})

    available_out = [c for c in OUTPUT_COLUMNS if c in combined.columns]
    return combined[available_out].reset_index(drop=True)


def fetch_county_csv(year: int, industry_code: str) -> pd.DataFrame | None:
    """Download and parse one BLS QCEW county-level annual CSV file.

    NOTE: This function uses the per-industry API endpoint which has been
    unreliable. Prefer fetch_year_singlefile() for production use. This
    function is retained for backward compatibility with tests.

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

    df = _select_and_coerce_columns(df, context=f"year={year} industry={industry_code}")
    log.info("  Downloaded %d rows", len(df))
    return df


def filter_county_df(
    df: pd.DataFrame,
    year: int,
    industry_code: str,
) -> pd.DataFrame:
    """Filter a raw QCEW county DataFrame to our target states and parameters.

    Applies filters:
      1. Keep only annual-average rows (qtr == "A")
      2. Keep only total-ownership rows (own_code == "0")
      3. Keep only counties in configured states (all 50+DC)
      4. Drop suppressed rows (disclosure_code == "N")
      5. Drop non-county FIPS (state-level "SS000" or "US000")

    Args:
        df: Raw DataFrame from fetch_county_csv().
        year: Data year (for logging).
        industry_code: NAICS code (for logging).

    Returns:
        Filtered DataFrame with area_fips (5-digit county FIPS), own_code,
        industry_code, year, annual_avg_estabs, annual_avg_emplvl,
        total_annual_wages.
    """
    if df is None or df.empty:
        return _empty_output_frame()

    fips_col = "area_fips"
    n_raw = len(df)

    # 1. Keep only annual average rows
    if "qtr" in df.columns:
        df = df[df["qtr"].astype(str).str.strip() == ANNUAL_QTR_CODE].copy()
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
        return _empty_output_frame()

    # 3. Keep only county-level FIPS (5 digits where last 3 are not "000")
    #    State-level records have area_fips like "01000", "12000" (county=000)
    #    US-level record has "US000"
    if fips_col in df.columns:
        df[fips_col] = df[fips_col].astype(str).str.strip().str.zfill(5)
        valid_fips = (
            df[fips_col].str.match(FIPS_PATTERN)
            & (df[fips_col].str[:2].isin(TARGET_STATE_FIPS))
            & (df[fips_col].str[2:] != STATE_LEVEL_COUNTY_SUFFIX)
        )
        n_before = len(df)
        df = df[valid_fips].copy()
        log.info(
            "  [%d/%s] Target-state filter: %d → %d rows",
            year, industry_code, n_before, len(df),
        )

    if df.empty:
        return _empty_output_frame()

    # 4. Drop suppressed rows (disclosure_code == "N" means data withheld)
    df = _drop_suppressed(df, year, label=f"industry={industry_code}")

    if df.empty:
        return _empty_output_frame()

    # Rename area_fips → county_fips and select output columns
    df = df.rename(columns={fips_col: "county_fips"})
    available_out = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[available_out].reset_index(drop=True)


def fetch_year_singlefile(year: int) -> pd.DataFrame:
    """Download, parse, and filter QCEW data for all industries in one year.

    Idempotent: if a parquet cache shard already exists for this year,
    the cached data is returned without making an HTTP request.

    Downloads the annual singlefile ZIP (~74-80MB), parses the CSV, and
    filters to county-level rows for our target states and industry codes.

    Args:
        year: Data year.

    Returns:
        Filtered county-level DataFrame, or empty DataFrame on failure.
    """
    cache = _cache_path(year)
    if cache.exists():
        log.info("  Cache hit: year=%d — loading %s", year, cache.name)
        return pd.read_parquet(cache)

    log.info("Fetching year=%d annual singlefile...", year)
    zip_bytes = download_singlefile(year)
    if zip_bytes is None:
        log.warning("  Download failed for year=%d", year)
        return pd.DataFrame()

    raw = parse_singlefile(zip_bytes, year)
    if raw is None or raw.empty:
        log.warning("  Parse failed for year=%d", year)
        return pd.DataFrame()

    filtered = filter_singlefile(raw, year)
    log.info("  year=%d: %d county-industry rows retained", year, len(filtered))

    if not filtered.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(cache, index=False)
        log.info("  Cached → %s", cache.name)

    return filtered


def fetch_industry_year(year: int, industry_name: str, industry_code: str) -> pd.DataFrame:
    """Fetch QCEW data for one industry+year combination.

    This is a thin wrapper around fetch_year_singlefile() that extracts
    a single industry from the year's cached data. Provided for backward
    compatibility with code that iterates (year, industry) pairs.

    Args:
        year: Data year.
        industry_name: Friendly name (for logging).
        industry_code: NAICS code string.

    Returns:
        Filtered county-level DataFrame for the requested industry, or empty
        DataFrame on failure.
    """
    year_df = fetch_year_singlefile(year)
    if year_df.empty:
        return pd.DataFrame()

    result = year_df[
        year_df["industry_code"].astype(str).str.strip() == str(industry_code)
    ].copy()
    log.info(
        "  year=%d industry=%s (%s): %d rows",
        year, industry_code, industry_name, len(result),
    )
    return result


def main(
    years: list[int] | None = None,
    industry_codes: dict[str, str] | None = None,
) -> None:
    """Download BLS QCEW county data and save combined parquet.

    Downloads one annual singlefile ZIP per year, filters to all configured
    states (all 50+DC) and target industry codes, and saves the combined
    result to data/raw/qcew_county.parquet.

    Args:
        years: List of data years to fetch. Defaults to DEFAULT_YEARS.
        industry_codes: Dict of {name: code}. Defaults to INDUSTRY_CODES
            (used only for logging; singlefile approach fetches all industries
            in one pass per year).
    """
    if years is None:
        years = DEFAULT_YEARS
    if industry_codes is None:
        industry_codes = INDUSTRY_CODES

    log.info(
        "Fetching BLS QCEW data (singlefile bulk downloads): %d year(s)",
        len(years),
    )
    log.info("Target states: %d states", len(STATES))
    log.info("Years: %s", years)
    log.info("Industries: %s", list(industry_codes.keys()))

    frames: list[pd.DataFrame] = []

    for i, year in enumerate(years):
        df = fetch_year_singlefile(year)
        if not df.empty:
            frames.append(df)
        # Polite delay between downloads (skip after last year)
        if i < len(years) - 1:
            time.sleep(REQUEST_DELAY)

    if not frames:
        log.error("No data retrieved for any year. Aborting.")
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
