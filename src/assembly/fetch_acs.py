"""
Stage 1 data assembly: fetch ACS 5-year 2022 data at census tract level.

Fetches core demographic, economic, and social variables for all census
tracts in Florida (12), Georgia (13), and Alabama (01).

Variables are chosen for community detection (Stage 2) — non-political
features that distinguish community types. Raw counts only; derived
ratios (% owner-occupied, % college-educated, etc.) are computed in
the feature engineering step.

Output:
    data/assembled/acs_tracts_2022.parquet

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

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

API_KEY = os.getenv("CENSUS_API_KEY")
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

STATES = {"AL": "01", "FL": "12", "GA": "13"}

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"

# MOE threshold: flag tracts where MOE / |estimate| > 30% on any primary feature
# See ADR-004 and census-acs-tract.md for rationale. Do not exclude — just surface.
MOE_THRESHOLD = 0.30

# ── Variable manifest ─────────────────────────────────────────────────────────
# Format: {api_code: friendly_name}
# All primary variables include a paired MOE column (_moe suffix).
# The Census API returns -666666666 for suppressed/unavailable values; these
# are replaced with NaN during casting (see cast_numeric()).

VARIABLES: dict[str, str] = {
    # Racial / ethnic composition (B03002)
    # Used to distinguish: Black Belt, Hispanic, White rural, diverse urban tracts
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
    # Used to distinguish retiree tracts from working-age and student tracts
    "B01002_001E": "median_age",
    "B01002_001M": "median_age_moe",
    # Median household income (B19013)
    # Primary economic stratifier
    "B19013_001E": "median_hh_income",
    "B19013_001M": "median_hh_income_moe",
    # Housing tenure (B25003)
    # Owner-occupied % is a proxy for stability / community rootedness
    "B25003_001E": "housing_units",
    "B25003_001M": "housing_units_moe",
    "B25003_002E": "housing_owner",
    "B25003_002M": "housing_owner_moe",
    # Commute mode (B08301)
    # Transit use → urban density; WFH → professional class
    "B08301_001E": "commute_total",
    "B08301_001M": "commute_total_moe",
    "B08301_002E": "commute_car",
    "B08301_002M": "commute_car_moe",
    "B08301_010E": "commute_transit",
    "B08301_010M": "commute_transit_moe",
    "B08301_021E": "commute_wfh",
    "B08301_021M": "commute_wfh_moe",
    # Educational attainment 25+ (B15003)
    # College-educated % is the primary class/community-type signal
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
    # Male + female management combined → knowledge-worker community signal
    "C24010_001E": "occ_total",
    "C24010_001M": "occ_total_moe",
    "C24010_003E": "occ_mgmt_male",
    "C24010_003M": "occ_mgmt_male_moe",
    "C24010_039E": "occ_mgmt_female",
    "C24010_039M": "occ_mgmt_female_moe",
}

# Names of estimate-only columns (no _moe suffix) — used for MOE flagging
ESTIMATE_COLS = [name for name in VARIABLES.values() if not name.endswith("_moe")]


# ── Fetch ─────────────────────────────────────────────────────────────────────


def fetch_state_tracts(state_fips: str, api_vars: list[str]) -> pd.DataFrame:
    """Fetch ACS 5-year 2022 data for all tracts in one state.

    The API returns a JSON array where row 0 is headers and rows 1..N are data.
    geo identifiers (state, county, tract) are appended automatically by the API.
    """
    if len(api_vars) > 50:
        raise ValueError(
            f"Census API limit is 50 variables per request; got {len(api_vars)}. "
            "Split into multiple requests."
        )
    params = {
        "get": ",".join(api_vars),
        "for": "tract:*",
        "in": f"state:{state_fips}",
        "key": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    headers, rows = data[0], data[1:]
    return pd.DataFrame(rows, columns=headers)


# ── Transform ─────────────────────────────────────────────────────────────────


def build_geoid(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 11-digit tract GEOID from the state/county/tract geo columns.

    Census tract FIPS = [2-digit state][3-digit county][6-digit tract]
    This is the join key for all other data sources (VEST crosswalk, ARDA, etc.)
    """
    df["tract_geoid"] = df["state"] + df["county"] + df["tract"]
    return df.drop(columns=["state", "county", "tract"])


def cast_numeric(df: pd.DataFrame, api_vars: list[str]) -> pd.DataFrame:
    """Cast ACS columns to float and replace the -666666666 null sentinel with NaN.

    The Census Bureau uses -666666666 for suppressed or unavailable estimates.
    Leaving it as-is would corrupt any downstream aggregation.
    """
    for col in api_vars:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-666666666, float("nan"))
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename API codes to friendly names defined in VARIABLES."""
    return df.rename(columns=VARIABLES)


def add_moe_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag columns for high-uncertainty estimates.

    Flags any tract where MOE / |estimate| > MOE_THRESHOLD (30%).
    Flags are surfaced for post-Stage-2 review; tracts are never excluded.

    Per census-acs-tract.md: store the flag, don't model the error.
    """
    flags_added = 0
    for api_code, friendly_name in VARIABLES.items():
        if not api_code.endswith("M"):
            continue
        est_code = api_code[:-1] + "E"
        est_name = VARIABLES.get(est_code)
        if not (est_name and est_name in df.columns and friendly_name in df.columns):
            continue

        est = df[est_name].abs()
        moe = df[friendly_name].abs()
        df[f"{est_name}_high_moe"] = (moe / est.replace(0, float("nan"))) > MOE_THRESHOLD
        flags_added += 1

    log.info("Added %d MOE flag columns", flags_added)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not API_KEY:
        raise EnvironmentError(
            "CENSUS_API_KEY not set. "
            "Add it to .env or export as environment variable. "
            "Free key: https://api.census.gov/data/key_signup.html"
        )

    api_vars = list(VARIABLES.keys())
    log.info("Fetching %d ACS variables for %d states", len(api_vars), len(STATES))

    frames: list[pd.DataFrame] = []
    for state_abbr, state_fips in STATES.items():
        log.info("  %s (FIPS %s)...", state_abbr, state_fips)
        df = fetch_state_tracts(state_fips, api_vars)
        log.info("    %d tracts returned", len(df))
        frames.append(df)
        time.sleep(0.5)  # polite API rate limiting

    combined = pd.concat(frames, ignore_index=True)
    combined = build_geoid(combined)
    combined = cast_numeric(combined, api_vars)
    combined = rename_columns(combined)
    combined = add_moe_flags(combined)

    # Summary
    n_tracts = len(combined)
    high_moe_cols = [c for c in combined.columns if c.endswith("_high_moe")]
    n_flagged = combined[high_moe_cols].any(axis=1).sum()
    log.info(
        "Total: %d tracts × %d columns | %d tracts (%.1f%%) have ≥1 high-MOE field",
        n_tracts,
        len(combined.columns),
        n_flagged,
        100.0 * n_flagged / n_tracts,
    )

    # Persist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "acs_tracts_2022.parquet"
    combined.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
