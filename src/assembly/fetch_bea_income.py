"""Fetch BEA Local Area Personal Income data and produce county-level income composition features.

Source: BEA Regional Data API
  https://apps.bea.gov/api/data/

Dataset: CAINC1 — County and MSA Personal Income Summary

Line codes used:
  LineCode 1: Personal income (total)
  LineCode 3: Net earnings by place of residence
  LineCode 6: Personal current transfer receipts
  LineCode 7: Dividends, interest, and rent

Output:
  data/assembled/bea_county_income.parquet
  Columns: county_fips, earnings_share, transfers_share, investment_share

  Where:
    earnings_share    = net_earnings / personal_income
    transfers_share   = transfers / personal_income
    investment_share  = dividends_interest_rent / personal_income

Cache:
  data/raw/bea/cainc1_{fips_prefix}_{year}.parquet  (one file per state FIPS prefix per year)

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

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bea"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

BEA_BASE = "https://apps.bea.gov/api/data/"

# CAINC1 line codes
LINE_PERSONAL_INCOME = 1       # Total personal income
LINE_NET_EARNINGS = 3          # Net earnings by place of residence
LINE_TRANSFERS = 6             # Personal current transfer receipts
LINE_DIVIDENDS_INTEREST = 7    # Dividends, interest, and rent

# Preferred year; falls back to FALLBACK_YEAR if primary year is unavailable
PRIMARY_YEAR = 2022
FALLBACK_YEAR = 2021

# FIPS prefix → state abbreviation (e.g. "12" → "FL")
STATE_ABBR: dict[str, str] = _cfg.STATE_ABBR

# Target state FIPS prefixes: FL=12, GA=13, AL=01
TARGET_FIPS_PREFIXES = set(STATE_ABBR.keys())  # {"12", "13", "01"}


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


def _fetch_cainc1_state(
    fips_prefix: str,
    year: int,
    api_key: str,
) -> pd.DataFrame:
    """Fetch CAINC1 data for all counties in one state for a given year.

    Returns a DataFrame with columns:
      GeoFips, GeoName, LineCode, TimePeriod, DataValue
    """
    # BEA GeoFips wildcard: use state FIPS + "000" for all counties in state
    # e.g. FL counties = "12*" or use state + "000" pattern
    # The BEA API accepts GeoFips as a comma-separated list or a state wildcard.
    # State-level wildcard: "{state_fips}*" fetches all counties in that state.
    state_fips_2 = fips_prefix.zfill(2)

    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "Regional",
        "TableName": "CAINC1",
        "LineCode": f"{LINE_PERSONAL_INCOME},{LINE_NET_EARNINGS},{LINE_TRANSFERS},{LINE_DIVIDENDS_INTEREST}",
        "GeoFips": f"COUNTY",
        "Year": str(year),
        "ResultFormat": "JSON",
    }

    log.info("Fetching BEA CAINC1 state_fips=%s year=%d ...", state_fips_2, year)
    resp = requests.get(BEA_BASE, params=params, timeout=120)
    resp.raise_for_status()

    payload = resp.json()

    # BEA API can return error inside a 200 response
    if "BEAAPI" not in payload:
        raise ValueError(f"Unexpected BEA response structure: {list(payload.keys())}")

    beaapi = payload["BEAAPI"]
    if "Results" not in beaapi:
        # Check for error
        error = beaapi.get("Error", {})
        raise ValueError(f"BEA API error: {error}")

    results = beaapi["Results"]
    data = results.get("Data", [])
    if not data:
        log.warning("BEA returned no data for state_fips=%s year=%d", state_fips_2, year)
        return pd.DataFrame(columns=["GeoFips", "GeoName", "LineCode", "TimePeriod", "DataValue"])

    df = pd.DataFrame(data)

    # Filter to this state's counties: GeoFips starts with state_fips_2 and is 5 chars
    df = df[
        df["GeoFips"].astype(str).str.zfill(5).str[:2] == state_fips_2
    ].copy()

    # Exclude state-level aggregates (county code 000)
    df["GeoFips"] = df["GeoFips"].astype(str).str.zfill(5)
    df = df[df["GeoFips"].str[2:] != "000"].copy()

    log.info("  state=%s year=%d: %d data rows", state_fips_2, year, len(df))
    return df


def fetch_cainc1_state_cached(
    fips_prefix: str,
    year: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and cache CAINC1 data for one state/year.

    Cache path: data/raw/bea/cainc1_{fips_prefix}_{year}.parquet
    """
    cache_path = RAW_DIR / f"cainc1_{fips_prefix}_{year}.parquet"

    if cache_path.exists() and not force_refresh:
        log.info("Using cached BEA data: %s", cache_path)
        return pd.read_parquet(cache_path)

    api_key = _get_api_key()
    df = _fetch_cainc1_state(fips_prefix, year, api_key)

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
    """Compute income composition shares from raw CAINC1 data.

    Args:
        raw_df: DataFrame with columns GeoFips, LineCode, DataValue
                (as returned by _fetch_cainc1_state or from cache).

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
    pivot = pivot[pivot["personal_income"].notna() & (pivot["personal_income"] != 0)].copy()

    return pivot[
        ["county_fips", "earnings_share", "transfers_share", "investment_share"]
    ].reset_index(drop=True)


# ── State filtering ───────────────────────────────────────────────────────────


def filter_to_target_states(df: pd.DataFrame) -> pd.DataFrame:
    """Filter county_fips column to FL (12), GA (13), AL (01) only."""
    return df[
        df["county_fips"].str[:2].isin(TARGET_FIPS_PREFIXES)
    ].copy()


# ── Main assembly ─────────────────────────────────────────────────────────────


def build_bea_income_features(
    year: int | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build county-level BEA income composition features for FL, GA, AL.

    Fetches CAINC1 for each state. Falls back from PRIMARY_YEAR to FALLBACK_YEAR
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
                    state_abbr, year, FALLBACK_YEAR,
                )
                raw = fetch_cainc1_state_cached(fips_prefix, FALLBACK_YEAR, force_refresh)
        except Exception as exc:
            log.error(
                "Failed to fetch BEA data for state=%s year=%d: %s — trying fallback year %d",
                state_abbr, year, exc, FALLBACK_YEAR,
            )
            raw = fetch_cainc1_state_cached(fips_prefix, FALLBACK_YEAR, force_refresh)

        shares = compute_income_shares(raw)
        shares = filter_to_target_states(shares)
        log.info(
            "  %s: %d counties with income shares", state_abbr, len(shares)
        )
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
