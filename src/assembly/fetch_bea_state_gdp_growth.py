"""Fetch BEA state-level real GDP and personal income, compute growth rates, map to counties.

Uses the BEA Regional API to pull annual real GDP (SAGDP9N) and personal income
(SAINC1) for all US states, 2018–present. Growth rates capture economic *momentum*,
which predicts electoral behavior better than levels — a state that was always poor
behaves differently from one that recently lost manufacturing jobs.

Features produced (state-level, mapped uniformly to all counties in the state):

  bea_gdp_growth_1yr   — most recent 1-year growth rate: (GDP_t - GDP_t-1) / GDP_t-1
  bea_gdp_growth_2yr   — 2-year compound annual growth rate from t-2 to t
  bea_income_growth_1yr — personal income 1-year growth rate (same computation)

Growth rate interpretation: +0.03 = 3% annual real GDP growth; negative values
indicate contraction (recession signal).

All three features are mapped by 2-digit FIPS prefix (same approach as
build_bea_state_features.py). Missing states are filled with the national median
so the feature matrix never has gaps.

API endpoints used:
  SAGDP9N — Real GDP by state (millions of chained 2017 dollars)
    https://apps.bea.gov/api/data?UserID=...&method=GetData&datasetname=Regional
        &TableName=SAGDP9N&LineCode=1&GeoFips=STATE&Year=ALL&ResultFormat=JSON
  SAINC1  — State personal income summary
    LineCode 3 = Per capita personal income (dollars)

Cache:
  data/raw/bea_growth/sagdp9n_states.parquet — raw API response, all years
  data/raw/bea_growth/sainc1_states.parquet  — raw API response, all years

Output:
  data/assembled/county_bea_growth_features.parquet
  Columns: county_fips, bea_gdp_growth_1yr, bea_gdp_growth_2yr, bea_income_growth_1yr

FIPS format: all county_fips values are 5-char zero-padded strings throughout.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bea_growth"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_bea_growth_features.parquet"

BEA_BASE = "https://apps.bea.gov/api/data"

# SAGDP9N — Real GDP by state in millions of chained 2017 dollars
#   LineCode 1 = All industry total
GDP_TABLE = "SAGDP9N"
GDP_LINE_CODE = 1

# SAINC1 — State personal income summary
#   LineCode 3 = Per capita personal income (dollars)
INCOME_TABLE = "SAINC1"
INCOME_LINE_CODE = 3

# Fetch data for all available years (BEA returns whichever are available)
FETCH_YEAR = "ALL"

# Number of years to fetch for growth computation (we need at least 3: t, t-1, t-2)
# Using Year=ALL is more robust and lets us pick the latest available year
MIN_YEARS_REQUIRED = 3

# Request timeout in seconds — BEA can be slow for all-state all-year queries
REQUEST_TIMEOUT_SECS = 120

# Output column names — kept as module-level constants so callers can import them
COL_GDP_GROWTH_1YR = "bea_gdp_growth_1yr"
COL_GDP_GROWTH_2YR = "bea_gdp_growth_2yr"
COL_INCOME_GROWTH_1YR = "bea_income_growth_1yr"

FEATURE_COLS = [COL_GDP_GROWTH_1YR, COL_GDP_GROWTH_2YR, COL_INCOME_GROWTH_1YR]

# Maps state name (as returned by BEA API) to 2-digit FIPS prefix.
# This is the same mapping as in build_bea_state_features.py — kept local to
# avoid a cross-module import dependency that would complicate testing.
_STATE_NAME_TO_FIPS_PREFIX: dict[str, str] = {
    "Alabama": "01",
    "Alaska": "02",
    "Arizona": "04",
    "Arkansas": "05",
    "California": "06",
    "Colorado": "08",
    "Connecticut": "09",
    "Delaware": "10",
    "District of Columbia": "11",
    "Florida": "12",
    "Georgia": "13",
    "Hawaii": "15",
    "Idaho": "16",
    "Illinois": "17",
    "Indiana": "18",
    "Iowa": "19",
    "Kansas": "20",
    "Kentucky": "21",
    "Louisiana": "22",
    "Maine": "23",
    "Maryland": "24",
    "Massachusetts": "25",
    "Michigan": "26",
    "Minnesota": "27",
    "Mississippi": "28",
    "Missouri": "29",
    "Montana": "30",
    "Nebraska": "31",
    "Nevada": "32",
    "New Hampshire": "33",
    "New Jersey": "34",
    "New Mexico": "35",
    "New York": "36",
    "North Carolina": "37",
    "North Dakota": "38",
    "Ohio": "39",
    "Oklahoma": "40",
    "Oregon": "41",
    "Pennsylvania": "42",
    "Rhode Island": "44",
    "South Carolina": "45",
    "South Dakota": "46",
    "Tennessee": "47",
    "Texas": "48",
    "Utah": "49",
    "Vermont": "50",
    "Virginia": "51",
    "Washington": "53",
    "West Virginia": "54",
    "Wisconsin": "55",
    "Wyoming": "56",
}


# ── API key handling ──────────────────────────────────────────────────────────


def _get_api_key() -> str:
    """Return the BEA API key from environment, or raise with a helpful message."""
    key = os.environ.get("BEA_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "BEA_API_KEY environment variable is not set.\n"
            "Get a free key at https://apps.bea.gov/API/signup/\n"
            "Then set it: export BEA_API_KEY=your_key_here"
        )
    return key


# ── API fetching ──────────────────────────────────────────────────────────────


def _fetch_bea_state_series(
    table_name: str,
    line_code: int,
    api_key: str,
) -> pd.DataFrame:
    """Fetch a BEA Regional table for all states, all available years.

    Returns a tidy DataFrame with columns:
      GeoFips, GeoName, TimePeriod (int year), DataValue (float)

    BEA suppressed/missing values like "(D)", "(NA)", "--" are converted to NaN.
    State-level aggregate rows (GeoFips ending in "000") are included — these
    ARE the state rows for GeoFips=STATE requests.

    Parameters
    ----------
    table_name:
        BEA table name, e.g. "SAGDP9N" or "SAINC1".
    line_code:
        The LineCode integer (e.g. 1 for total GDP, 3 for per-capita income).
    api_key:
        Valid BEA API key.

    Raises
    ------
    requests.HTTPError
        If the HTTP request fails (4xx/5xx).
    ValueError
        If the BEA response structure is unexpected or contains an API error.
    """
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "Regional",
        "TableName": table_name,
        "LineCode": str(line_code),
        "GeoFips": "STATE",
        "Year": FETCH_YEAR,
        "ResultFormat": "JSON",
    }

    log.info("Fetching BEA %s LineCode=%d Year=%s ...", table_name, line_code, FETCH_YEAR)
    resp = requests.get(BEA_BASE, params=params, timeout=REQUEST_TIMEOUT_SECS)
    resp.raise_for_status()

    payload = resp.json()

    if "BEAAPI" not in payload:
        raise ValueError(
            f"Unexpected BEA response structure: top-level keys = {list(payload.keys())}"
        )

    beaapi = payload["BEAAPI"]
    if "Results" not in beaapi:
        error = beaapi.get("Error", {})
        raise ValueError(f"BEA API error (no Results key): {error}")

    results = beaapi["Results"]
    # Inline API errors arrive as 200 responses with an Error key inside Results
    if "Error" in results and "Data" not in results:
        raise ValueError(f"BEA API inline error: {results['Error']}")

    data = results.get("Data", [])
    if not data:
        log.warning("BEA returned no data for table=%s LineCode=%d", table_name, line_code)
        return pd.DataFrame(columns=["GeoFips", "GeoName", "TimePeriod", "DataValue"])

    df = pd.DataFrame(data)

    # Keep only the columns we need; BEA returns extra metadata columns
    df = df[["GeoFips", "GeoName", "TimePeriod", "DataValue"]].copy()

    # TimePeriod comes back as a string like "2022" or "2022Q1" — extract the year
    df["TimePeriod"] = df["TimePeriod"].astype(str).str[:4].pipe(
        pd.to_numeric, errors="coerce"
    )

    # DataValue arrives as a string with commas and suppression codes
    df["DataValue"] = df["DataValue"].apply(_parse_data_value)

    n_states = df["GeoFips"].nunique()
    n_years = df["TimePeriod"].nunique()
    log.info(
        "  Got %d rows for %s LineCode=%d (%d states, %d years)",
        len(df),
        table_name,
        line_code,
        n_states,
        n_years,
    )
    return df


def _parse_data_value(val: object) -> float:
    """Parse a BEA DataValue string to float.

    BEA uses several suppression codes instead of null:
      (D) — data withheld to avoid disclosure
      (NA) — not available
      (L) — value < $50,000
      (S) — value < $500
      (X) — not applicable
      -- — missing

    All of these are returned as NaN; callers handle them via median imputation.
    """
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().replace(",", "")
    if s in ("(D)", "(NA)", "(L)", "(S)", "(X)", "--", ""):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ── Cached fetch wrappers ─────────────────────────────────────────────────────


def fetch_state_gdp_series(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch and cache state real GDP series (SAGDP9N, LineCode 1).

    Cache path: data/raw/bea_growth/sagdp9n_states.parquet

    Returns tidy DataFrame: GeoFips, GeoName, TimePeriod (int), DataValue (float)
    """
    cache_path = RAW_DIR / "sagdp9n_states.parquet"
    if cache_path.exists() and not force_refresh:
        log.info("Using cached BEA GDP series: %s", cache_path)
        return pd.read_parquet(cache_path)

    api_key = _get_api_key()
    df = _fetch_bea_state_series(GDP_TABLE, GDP_LINE_CODE, api_key)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Saved BEA GDP series → %s (%d rows)", cache_path, len(df))
    return df


def fetch_state_income_series(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch and cache state per-capita personal income series (SAINC1, LineCode 3).

    Cache path: data/raw/bea_growth/sainc1_states.parquet

    Returns tidy DataFrame: GeoFips, GeoName, TimePeriod (int), DataValue (float)
    """
    cache_path = RAW_DIR / "sainc1_states.parquet"
    if cache_path.exists() and not force_refresh:
        log.info("Using cached BEA income series: %s", cache_path)
        return pd.read_parquet(cache_path)

    api_key = _get_api_key()
    df = _fetch_bea_state_series(INCOME_TABLE, INCOME_LINE_CODE, api_key)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Saved BEA income series → %s (%d rows)", cache_path, len(df))
    return df


# ── Growth rate computation ───────────────────────────────────────────────────


def compute_growth_rates(
    raw_df: pd.DataFrame,
    state_name_to_fips: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute 1-year and 2-year growth rates from a tidy BEA state series.

    Growth rate formula:
      1-year: (GDP_t  - GDP_{t-1}) / GDP_{t-1}
      2-year CAGR: (GDP_t / GDP_{t-2})^(1/2) - 1

    The most recent year with non-null data is used as t. If t-1 or t-2 are
    missing (suppressed by BEA or not yet released), those growth rates are NaN
    and will be filled with the national median downstream.

    Parameters
    ----------
    raw_df:
        Tidy DataFrame with columns GeoFips, GeoName, TimePeriod (int), DataValue.
        As returned by fetch_state_gdp_series() or fetch_state_income_series().
    state_name_to_fips:
        Optional override for the state-name → FIPS-prefix mapping (useful in
        tests that inject synthetic state names).

    Returns
    -------
    DataFrame with columns: fips_prefix (str), growth_1yr (float), growth_2yr (float).
    One row per state. States with insufficient data have NaN in growth columns.
    """
    if state_name_to_fips is None:
        state_name_to_fips = _STATE_NAME_TO_FIPS_PREFIX

    df = raw_df.copy()
    df = df.dropna(subset=["TimePeriod", "DataValue"])

    # Map state name to 2-digit FIPS prefix
    df["fips_prefix"] = df["GeoName"].map(state_name_to_fips)
    unmapped = df[df["fips_prefix"].isna()]["GeoName"].unique().tolist()
    if unmapped:
        # Non-states (e.g., "United States", territories) are expected in the
        # GeoFips=STATE response but have no county FIPS mapping — log and skip.
        log.debug("Skipping unmapped GeoNames: %s", unmapped)

    df = df.dropna(subset=["fips_prefix"])

    if df.empty:
        log.warning("No states with FIPS mapping after filtering — returning empty DataFrame")
        return pd.DataFrame(columns=["fips_prefix", "growth_1yr", "growth_2yr"])

    # Pivot to wide: index=fips_prefix, columns=year, values=DataValue
    pivot = df.pivot_table(
        index="fips_prefix",
        columns="TimePeriod",
        values="DataValue",
        aggfunc="first",  # each state/year combination should be unique
    )
    pivot.columns = pivot.columns.astype(int)
    pivot.columns.name = None

    available_years = sorted(pivot.columns)
    if len(available_years) < MIN_YEARS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_YEARS_REQUIRED} years of data to compute growth rates; "
            f"got {len(available_years)} ({available_years})."
        )

    # Find the most recent year where at least half the states have data.
    # This guards against a year that just started and has only 1-2 preliminary values.
    latest_year = _find_latest_usable_year(pivot, available_years)
    year_t = latest_year
    year_t1 = year_t - 1
    year_t2 = year_t - 2

    log.info("Computing growth rates: t=%d  t-1=%d  t-2=%d", year_t, year_t1, year_t2)

    # Build result DataFrame — one row per state
    result = pd.DataFrame(index=pivot.index)

    gdp_t = pivot.get(year_t)
    gdp_t1 = pivot.get(year_t1)
    gdp_t2 = pivot.get(year_t2)

    # 1-year growth: safe against zero/missing denominator
    if gdp_t is not None and gdp_t1 is not None:
        valid = gdp_t1.notna() & (gdp_t1 != 0) & gdp_t.notna()
        result["growth_1yr"] = float("nan")
        result.loc[valid, "growth_1yr"] = (gdp_t[valid] - gdp_t1[valid]) / gdp_t1[valid]
    else:
        result["growth_1yr"] = float("nan")

    # 2-year CAGR: (GDP_t / GDP_{t-2})^0.5 - 1
    if gdp_t is not None and gdp_t2 is not None:
        valid = gdp_t2.notna() & (gdp_t2 > 0) & gdp_t.notna()
        result["growth_2yr"] = float("nan")
        result.loc[valid, "growth_2yr"] = (gdp_t[valid] / gdp_t2[valid]) ** 0.5 - 1.0
    else:
        result["growth_2yr"] = float("nan")

    result = result.reset_index()
    log.info(
        "Growth rates computed: %d states, 1yr mean=%.3f, 2yr mean=%.3f",
        len(result),
        result["growth_1yr"].mean(),
        result["growth_2yr"].mean(),
    )
    return result[["fips_prefix", "growth_1yr", "growth_2yr"]]


def _find_latest_usable_year(pivot: pd.DataFrame, available_years: list[int]) -> int:
    """Return the most recent year where ≥ half the states have non-null data.

    The BEA often releases preliminary estimates that cover only a few states.
    Starting from the latest year and working back, we pick the first year
    where coverage is ≥ 50% (26 of 51 states). This prevents a half-released
    year from becoming the reference year and producing mostly-NaN growth rates.
    """
    min_coverage = len(pivot) / 2
    for year in reversed(available_years):
        n_valid = pivot[year].notna().sum()
        if n_valid >= min_coverage:
            return year
    # Fallback: use second-to-last year if even the oldest is sparse
    # (should never happen with real BEA data but guards against test edge cases)
    return available_years[-2] if len(available_years) >= 2 else available_years[-1]


# ── County mapping ────────────────────────────────────────────────────────────


def build_county_bea_growth_features(
    county_fips: list[str],
    force_refresh: bool = False,
    _gdp_df: pd.DataFrame | None = None,
    _income_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map state-level BEA GDP and income growth rates to counties via FIPS prefix.

    Each county inherits its state's economic growth rates. Counties in states
    with missing or suppressed data are imputed with the national median so the
    output covers all input counties without gaps.

    Parameters
    ----------
    county_fips:
        List of 5-char zero-padded county FIPS strings (e.g., "12001").
    force_refresh:
        If True, ignore cached BEA API responses and re-fetch. Default False.
    _gdp_df:
        Pre-fetched raw GDP DataFrame (for testing without live API calls).
    _income_df:
        Pre-fetched raw income DataFrame (for testing without live API calls).

    Returns
    -------
    DataFrame with columns: county_fips, bea_gdp_growth_1yr, bea_gdp_growth_2yr,
    bea_income_growth_1yr. One row per county in county_fips.

    Notes
    -----
    - All county_fips values must be exactly 5 characters after zero-padding.
    - Duplicate FIPS in the input are preserved (not deduplicated).
    - National medians (computed from present states) fill any missing states.
    """
    # Fetch or accept injected data (injection used in tests)
    if _gdp_df is None:
        _gdp_df = fetch_state_gdp_series(force_refresh=force_refresh)
    if _income_df is None:
        _income_df = fetch_state_income_series(force_refresh=force_refresh)

    gdp_growth = compute_growth_rates(_gdp_df)
    income_growth = compute_growth_rates(_income_df)

    # Build county DataFrame and validate FIPS format
    df = pd.DataFrame({"county_fips": county_fips})
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    if not df["county_fips"].str.len().eq(5).all():
        bad = df[df["county_fips"].str.len() != 5]["county_fips"].tolist()[:5]
        raise ValueError(f"county_fips must be 5-char strings. Bad examples: {bad}")

    # Index growth tables by fips_prefix for O(1) lookup
    gdp_1yr = gdp_growth.set_index("fips_prefix")["growth_1yr"]
    gdp_2yr = gdp_growth.set_index("fips_prefix")["growth_2yr"]
    inc_1yr = income_growth.set_index("fips_prefix")["growth_1yr"]

    state_prefix = df["county_fips"].str[:2]

    # Map state-level growth rates to counties
    df[COL_GDP_GROWTH_1YR] = state_prefix.map(gdp_1yr)
    df[COL_GDP_GROWTH_2YR] = state_prefix.map(gdp_2yr)
    df[COL_INCOME_GROWTH_1YR] = state_prefix.map(inc_1yr)

    # Fill missing states with national median
    # (e.g., territories not in the BEA STATE query, or suppressed data)
    for col in FEATURE_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = float(df[col].median())
            df[col] = df[col].fillna(median_val)
            log.info(
                "%d counties lack BEA %s data — filled with national median %.4f",
                n_missing,
                col,
                median_val,
            )

    log.info(
        "Built BEA growth features: %d counties, "
        "gdp_growth_1yr=[%.3f, %.3f], gdp_growth_2yr=[%.3f, %.3f], "
        "income_growth_1yr=[%.3f, %.3f]",
        len(df),
        df[COL_GDP_GROWTH_1YR].min(),
        df[COL_GDP_GROWTH_1YR].max(),
        df[COL_GDP_GROWTH_2YR].min(),
        df[COL_GDP_GROWTH_2YR].max(),
        df[COL_INCOME_GROWTH_1YR].min(),
        df[COL_INCOME_GROWTH_1YR].max(),
    )

    return df[["county_fips"] + FEATURE_COLS].reset_index(drop=True)


# ── Main entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """Fetch BEA state GDP/income growth rates and save county-level features.

    Loads all county FIPS from the ACS spine (required to be present), fetches
    BEA state series via API (with caching), computes growth rates, maps to
    counties, and writes data/assembled/county_bea_growth_features.parquet.

    Run as:
        uv run python -m src.assembly.fetch_bea_state_gdp_growth
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    acs_path = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
    if not acs_path.exists():
        raise FileNotFoundError(
            f"ACS spine not found: {acs_path}. "
            "Run src/assembly/build_county_acs_features.py first."
        )

    log.info("Loading ACS county spine from %s", acs_path)
    acs = pd.read_parquet(acs_path)
    county_fips = acs["county_fips"].astype(str).str.zfill(5).tolist()
    log.info("Found %d counties in ACS spine", len(county_fips))

    features = build_county_bea_growth_features(county_fips)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %d county BEA growth features → %s", len(features), OUTPUT_PATH)

    for col in FEATURE_COLS:
        log.info(
            "  %s — Q1=%.4f, median=%.4f, Q3=%.4f, n_valid=%d",
            col,
            *features[col].quantile([0.25, 0.5, 0.75]),
            features[col].notna().sum(),
        )


if __name__ == "__main__":
    main()
