"""Fetch county-level ACS 5-year 2022 data for FL, GA, and AL.

Uses the same Census API variables as fetch_acs.py (tract level) but changes
the geography to county:* so the results can be joined to county_community_assignments.

Output:
    data/assembled/acs_counties_2022.parquet

Reference: docs/references/data-sources/census-acs-tract.md
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.core import config as _cfg

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

API_KEY = os.getenv("CENSUS_API_KEY")
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

# State list comes from config/model.yaml (all 50+DC by default).
# To restrict to a subset for testing, edit geography.states in that file.
STATES: dict[str, str] = _cfg.STATES  # abbr → fips prefix

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"

# ── Variable manifest ─────────────────────────────────────────────────────────
# Identical to fetch_acs.py VARIABLES — same Census API codes, county geography.

VARIABLES: dict[str, str] = {
    # Racial / ethnic composition (B03002)
    "B03002_001E": "pop_total",
    "B03002_001M": "pop_total_moe",
    "B03002_003E": "pop_white_nh",
    "B03002_003M": "pop_white_nh_moe",
    "B03002_004E": "pop_black",
    "B03002_004M": "pop_black_moe",
    "B03002_006E": "pop_asian",
    "B03002_006M": "pop_asian_moe",
    "B03002_012E": "pop_hispanic",
    "B03002_012M": "pop_hispanic_moe",
    # Median age (B01002)
    "B01002_001E": "median_age",
    "B01002_001M": "median_age_moe",
    # Median household income (B19013)
    "B19013_001E": "median_hh_income",
    "B19013_001M": "median_hh_income_moe",
    # Housing tenure (B25003)
    "B25003_001E": "housing_units",
    "B25003_001M": "housing_units_moe",
    "B25003_002E": "housing_owner",
    "B25003_002M": "housing_owner_moe",
    # Commute mode (B08301)
    "B08301_001E": "commute_total",
    "B08301_001M": "commute_total_moe",
    "B08301_002E": "commute_car",
    "B08301_002M": "commute_car_moe",
    "B08301_010E": "commute_transit",
    "B08301_010M": "commute_transit_moe",
    "B08301_021E": "commute_wfh",
    "B08301_021M": "commute_wfh_moe",
    # Educational attainment 25+ (B15003)
    "B15003_001E": "educ_total",
    "B15003_001M": "educ_total_moe",
    "B15003_022E": "educ_bachelors",
    "B15003_022M": "educ_bachelors_moe",
    "B15003_023E": "educ_masters",
    "B15003_023M": "educ_masters_moe",
    "B15003_024E": "educ_professional",
    "B15003_024M": "educ_professional_moe",
    "B15003_025E": "educ_doctorate",
    "B15003_025M": "educ_doctorate_moe",
    # Occupation: management/professional split by sex (C24010)
    "C24010_001E": "occ_total",
    "C24010_001M": "occ_total_moe",
    "C24010_003E": "occ_mgmt_male",
    "C24010_003M": "occ_mgmt_male_moe",
    "C24010_039E": "occ_mgmt_female",
    "C24010_039M": "occ_mgmt_female_moe",
}

ESTIMATE_COLS = [name for name in VARIABLES.values() if not name.endswith("_moe")]


# ── Fetch ─────────────────────────────────────────────────────────────────────


def fetch_state_counties(state_fips: str, api_vars: list[str]) -> pd.DataFrame:
    """Fetch ACS 5-year 2022 data for all counties in one state.

    Geography: for=county:*&in=state:{state_fips}
    The API returns a JSON array where row 0 is headers and rows 1..N are data.
    geo identifiers (state, county) are appended automatically.
    """
    if len(api_vars) > 50:
        raise ValueError(
            f"Census API limit is 50 variables per request; got {len(api_vars)}. "
            "Split into multiple requests."
        )
    params: dict = {
        "get": ",".join(api_vars),
        "for": "county:*",
        "in": f"state:{state_fips}",
    }
    if API_KEY:
        params["key"] = API_KEY

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    headers, rows = data[0], data[1:]
    return pd.DataFrame(rows, columns=headers)


# ── Transform ─────────────────────────────────────────────────────────────────


def build_county_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 5-digit county FIPS from the state/county geo columns."""
    df["county_fips"] = df["state"] + df["county"]
    return df.drop(columns=["state", "county"])


def cast_numeric(df: pd.DataFrame, api_vars: list[str]) -> pd.DataFrame:
    """Cast ACS columns to float; replace the -666666666 null sentinel with NaN."""
    for col in api_vars:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-666666666, float("nan"))
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename API codes to friendly names defined in VARIABLES."""
    return df.rename(columns=VARIABLES)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    api_vars = list(VARIABLES.keys())
    log.info(
        "Fetching %d ACS county variables for %d states (key=%s)",
        len(api_vars),
        len(STATES),
        "set" if API_KEY else "unset (anonymous)",
    )

    frames: list[pd.DataFrame] = []
    for state_abbr, state_fips in STATES.items():
        log.info("  %s (FIPS %s)...", state_abbr, state_fips)
        df = fetch_state_counties(state_fips, api_vars)
        log.info("    %d counties returned", len(df))
        frames.append(df)
        time.sleep(0.5)  # polite rate limiting

    combined = pd.concat(frames, ignore_index=True)
    combined = build_county_fips(combined)
    combined = cast_numeric(combined, api_vars)
    combined = rename_columns(combined)

    n_counties = len(combined)
    log.info("Total: %d counties x %d columns", n_counties, len(combined.columns))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "acs_counties_2022.parquet"
    combined.to_parquet(out_path, index=False)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
