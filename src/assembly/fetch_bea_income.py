"""Fetch BEA Local Area Personal Income data and produce county-level income composition features.

Source: BEA Regional Data API
  https://apps.bea.gov/api/data/

Dataset: CAINC4 — Personal income by major component

Line codes used (CAINC4):
  LineCode 10: Personal income (total)
  LineCode 45: Net earnings by place of residence
  LineCode 47: Personal current transfer receipts
  LineCode 46: Dividends, interest, and rent

Note: CAINC1 (County Personal Income Summary) only has 3 line codes (personal income,
population, per capita personal income) and does NOT support multiple LineCode values
in a single request (APIErrorCode 41). CAINC4 has the income component breakdown.

Output:
  data/assembled/bea_county_income.parquet
  Columns: county_fips, earnings_share, transfers_share, investment_share

  Where:
    earnings_share    = net_earnings / personal_income
    transfers_share   = transfers / personal_income
    investment_share  = dividends_interest_rent / personal_income

Cache:
  data/raw/bea/cainc4_national_{year}.parquet  — all US counties, all line codes, one file
  data/raw/bea/cainc4_{fips_prefix}_{year}.parquet  — per-state subset (state-level cache)

API key:
  Set environment variable BEA_API_KEY.
  Free key available at https://apps.bea.gov/API/signup/
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.core import config as _cfg

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bea"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

BEA_BASE = "https://apps.bea.gov/api/data/"

# BEA table and line codes
# CAINC4 = "Personal income by major component" — has earnings, transfers, dividends
BEA_TABLE = "CAINC4"

LINE_PERSONAL_INCOME = 10      # Personal income (total)
LINE_NET_EARNINGS = 45         # Net earnings by place of residence
LINE_TRANSFERS = 47            # Personal current transfer receipts
LINE_DIVIDENDS_INTEREST = 46   # Dividends, interest, and rent

# Preferred year; falls back to FALLBACK_YEAR if primary year is unavailable
PRIMARY_YEAR = 2022
FALLBACK_YEAR = 2021

# FIPS prefix → state abbreviation (e.g. "12" → "FL")
STATE_ABBR: dict[str, str] = _cfg.STATE_ABBR

# All configured state FIPS prefixes (50 states + DC = 51)
TARGET_FIPS_PREFIXES = set(STATE_ABBR.keys())


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


def _fetch_cainc4_one_linecode(
    line_code: int,
    year: int,
    api_key: str,
) -> pd.DataFrame:
    """Fetch CAINC4 data for ALL US counties for a single LineCode and year.

    The BEA Regional API rejects requests with multiple LineCode values
    (APIErrorCode 41: "Multiple parameter values were supplied for a parameter
    that only allows single values"). Each LineCode must be fetched separately.

    Returns a DataFrame with columns:
      GeoFips, GeoName, LineCode, TimePeriod, DataValue
    """
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "Regional",
        "TableName": BEA_TABLE,
        "LineCode": str(line_code),
        "GeoFips": "COUNTY",
        "Year": str(year),
        "ResultFormat": "JSON",
    }

    log.info("  Fetching BEA %s LineCode=%d year=%d ...", BEA_TABLE, line_code, year)
    resp = requests.get(BEA_BASE, params=params, timeout=120)
    resp.raise_for_status()

    payload = resp.json()

    if "BEAAPI" not in payload:
        raise ValueError(f"Unexpected BEA response structure: {list(payload.keys())}")

    beaapi = payload["BEAAPI"]
    if "Results" not in beaapi:
        error = beaapi.get("Error", {})
        raise ValueError(f"BEA API error: {error}")

    results = beaapi["Results"]
    # Inline API errors are returned as 200 responses with an Error key
    if "Error" in results and "Data" not in results:
        raise ValueError(f"BEA API error: {results['Error']}")

    data = results.get("Data", [])
    if not data:
        log.warning(
            "BEA returned no data for %s LineCode=%d year=%d",
            BEA_TABLE,
            line_code,
            year,
        )
        return pd.DataFrame(
            columns=["GeoFips", "GeoName", "LineCode", "TimePeriod", "DataValue"]
        )

    df = pd.DataFrame(data)
    # BEA API returns a "Code" column (e.g. "CAINC4-10") instead of "LineCode".
    # Add a LineCode column matching the requested line code for downstream pivoting.
    df["LineCode"] = str(line_code)
    return df


def _fetch_cainc4_national(year: int, api_key: str) -> pd.DataFrame:
    """Fetch CAINC4 data for ALL US counties for all required LineCodes.

    Makes exactly 4 API calls (one per LineCode) and caches the combined
    result in data/raw/bea/cainc4_national_{year}.parquet.  All subsequent
    calls for the same year read from that cache — avoiding 51 × 4 = 204
    redundant API calls when iterating over states.

    Returns a DataFrame with columns:
      GeoFips, GeoName, LineCode, TimePeriod, DataValue
    """
    national_cache = RAW_DIR / f"cainc4_national_{year}.parquet"
    if national_cache.exists():
        log.info("Using national BEA cache: %s", national_cache)
        return pd.read_parquet(national_cache)

    line_codes = [
        LINE_PERSONAL_INCOME,
        LINE_NET_EARNINGS,
        LINE_TRANSFERS,
        LINE_DIVIDENDS_INTEREST,
    ]

    frames: list[pd.DataFrame] = []
    for lc in line_codes:
        lc_df = _fetch_cainc4_one_linecode(lc, year, api_key)
        frames.append(lc_df)

    if not any(len(f) > 0 for f in frames):
        log.warning("BEA returned no national data for %s year=%d", BEA_TABLE, year)
        return pd.DataFrame(
            columns=["GeoFips", "GeoName", "LineCode", "TimePeriod", "DataValue"]
        )

    national = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    national.to_parquet(national_cache, index=False)
    n_counties = national["GeoFips"].nunique() if len(national) > 0 else 0
    log.info(
        "Saved national BEA cache → %s (%d rows, %d counties)",
        national_cache,
        len(national),
        n_counties,
    )
    return national


def _fetch_cainc4_state(
    fips_prefix: str,
    year: int,
    api_key: str,
) -> pd.DataFrame:
    """Fetch CAINC4 data for all counties in one state for a given year.

    Reads from the national COUNTY cache (fetching it if needed) and filters
    to the target state.

    Returns a DataFrame with columns:
      GeoFips, GeoName, LineCode, TimePeriod, DataValue
    """
    state_fips_2 = fips_prefix.zfill(2)
    log.info("Fetching BEA %s state_fips=%s year=%d ...", BEA_TABLE, state_fips_2, year)

    national = _fetch_cainc4_national(year, api_key)

    if national.empty:
        log.warning(
            "BEA returned no data for state_fips=%s year=%d", state_fips_2, year
        )
        return pd.DataFrame(
            columns=["GeoFips", "GeoName", "LineCode", "TimePeriod", "DataValue"]
        )

    # Filter to this state's counties
    national["GeoFips"] = national["GeoFips"].astype(str).str.zfill(5)
    df = national[national["GeoFips"].str[:2] == state_fips_2].copy()

    # Exclude state-level aggregates (county code 000)
    df = df[df["GeoFips"].str[2:] != "000"].copy()

    log.info("  state=%s year=%d: %d data rows", state_fips_2, year, len(df))
    return df


def fetch_cainc1_state_cached(
    fips_prefix: str,
    year: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and cache CAINC4 data for one state/year.

    Cache path: data/raw/bea/cainc4_{fips_prefix}_{year}.parquet

    Note: function name kept as fetch_cainc1_state_cached for backward
    compatibility with callers and tests.
    """
    cache_path = RAW_DIR / f"cainc4_{fips_prefix}_{year}.parquet"

    if cache_path.exists() and not force_refresh:
        log.info("Using cached BEA data: %s", cache_path)
        return pd.read_parquet(cache_path)

    api_key = _get_api_key()
    df = _fetch_cainc4_state(fips_prefix, year, api_key)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Saved BEA cache → %s (%d rows)", cache_path, len(df))
    return df


# ── Feature computation ───────────────────────────────────────────────────────


def _parse_data_value(val: str | float) -> float:
    """Parse BEA DataValue string to float. Returns NaN for suppressed/missing."""
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().replace(",", "")
    if s in ("(D)", "(NA)", "(L)", "(S)", "(X)", "--", ""):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def compute_income_shares(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute income composition shares from raw CAINC4 data.

    Args:
        raw_df: DataFrame with columns GeoFips, LineCode, DataValue
                (as returned by _fetch_cainc4_state or from cache).

    Returns:
        DataFrame with columns: county_fips, earnings_share, transfers_share,
        investment_share. Counties with zero or missing personal_income are
        excluded (NaN shares would be meaningless).
    """
    df = raw_df.copy()
    df["LineCode"] = pd.to_numeric(df["LineCode"], errors="coerce").astype("Int64")
    df["value"] = df["DataValue"].apply(_parse_data_value)
    df["county_fips"] = df["GeoFips"].astype(str).str.zfill(5)

    # Pivot: one row per county, one column per line code
    pivot = df.pivot_table(
        index="county_fips",
        columns="LineCode",
        values="value",
        aggfunc="first",
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # Rename columns to descriptive names (use .get() so missing lines don't crash)
    line_map = {
        LINE_PERSONAL_INCOME: "personal_income",
        LINE_NET_EARNINGS: "net_earnings",
        LINE_TRANSFERS: "transfers",
        LINE_DIVIDENDS_INTEREST: "dividends_interest_rent",
    }
    pivot = pivot.rename(columns=line_map)

    # Ensure all required columns exist (fill missing with NaN)
    for col in line_map.values():
        if col not in pivot.columns:
            pivot[col] = float("nan")

    # Compute shares — handle zero/missing personal_income
    def safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        result = pd.Series(float("nan"), index=denominator.index, dtype=float)
        valid = denominator.notna() & (denominator != 0) & numerator.notna()
        result[valid] = numerator[valid] / denominator[valid]
        return result

    pivot["earnings_share"] = safe_share(
        pivot["net_earnings"], pivot["personal_income"]
    )
    pivot["transfers_share"] = safe_share(
        pivot["transfers"], pivot["personal_income"]
    )
    pivot["investment_share"] = safe_share(
        pivot["dividends_interest_rent"], pivot["personal_income"]
    )

    # Keep only counties with valid personal_income
    pivot = pivot[
        pivot["personal_income"].notna() & (pivot["personal_income"] != 0)
    ].copy()

    return pivot[
        ["county_fips", "earnings_share", "transfers_share", "investment_share"]
    ].reset_index(drop=True)


# ── State filtering ───────────────────────────────────────────────────────────


def filter_to_target_states(df: pd.DataFrame) -> pd.DataFrame:
    """Filter county_fips column to configured target states (all 50 + DC)."""
    return df[df["county_fips"].str[:2].isin(TARGET_FIPS_PREFIXES)].copy()


# ── Main assembly ─────────────────────────────────────────────────────────────


def build_bea_income_features(
    year: int | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build county-level BEA income composition features for all US counties.

    Fetches CAINC4 for each state. Falls back from PRIMARY_YEAR to FALLBACK_YEAR
    if the primary year returns no data.

    Returns DataFrame with columns:
      county_fips, earnings_share, transfers_share, investment_share
    """
    if year is None:
        year = PRIMARY_YEAR

    frames: list[pd.DataFrame] = []

    for fips_prefix in sorted(TARGET_FIPS_PREFIXES):
        state_abbr = STATE_ABBR[fips_prefix]
        try:
            raw = fetch_cainc1_state_cached(fips_prefix, year, force_refresh)
            if raw.empty:
                log.warning(
                    "No data for state=%s year=%d, trying fallback year %d",
                    state_abbr,
                    year,
                    FALLBACK_YEAR,
                )
                raw = fetch_cainc1_state_cached(
                    fips_prefix, FALLBACK_YEAR, force_refresh
                )
        except Exception as exc:
            log.error(
                "Failed to fetch BEA data for state=%s year=%d: %s — trying fallback year %d",
                state_abbr,
                year,
                exc,
                FALLBACK_YEAR,
            )
            raw = fetch_cainc1_state_cached(fips_prefix, FALLBACK_YEAR, force_refresh)

        shares = compute_income_shares(raw)
        shares = filter_to_target_states(shares)
        log.info("  %s: %d counties with income shares", state_abbr, len(shares))
        frames.append(shares)

    if not frames:
        log.warning("No BEA data assembled for any state.")
        return pd.DataFrame(
            columns=["county_fips", "earnings_share", "transfers_share", "investment_share"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["county_fips"]).reset_index(drop=True)

    log.info(
        "BEA income features: %d counties total  earnings_share mean=%.3f  "
        "transfers_share mean=%.3f  investment_share mean=%.3f",
        len(result),
        result["earnings_share"].mean(),
        result["transfers_share"].mean(),
        result["investment_share"].mean(),
    )
    return result


def main() -> None:
    df = build_bea_income_features()

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSEMBLED_DIR / "bea_county_income.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved → %s (%d counties)", out, len(df))

    # Summary stats
    for col in ["earnings_share", "transfers_share", "investment_share"]:
        if col in df.columns:
            log.info(
                "  %s: mean=%.3f  min=%.3f  max=%.3f  n_valid=%d",
                col,
                df[col].mean(),
                df[col].min(),
                df[col].max(),
                df[col].notna().sum(),
            )


if __name__ == "__main__":
    main()
