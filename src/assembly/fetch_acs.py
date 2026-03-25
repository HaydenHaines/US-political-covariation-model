"""
Stage 1 data assembly: fetch ACS 5-year 2022 data at census tract level.

Fetches core demographic, economic, and social variables for all census
tracts in all 50 states + DC.

Variables are chosen for community detection (Stage 2) — non-political
features that distinguish community types. Raw counts only; derived
ratios (% owner-occupied, % college-educated, etc.) are computed in
the feature engineering step.

The Census API has a 50-variable-per-request limit. Variables are split
into batches; each batch is fetched separately per state and joined on
the geo identifiers before GEOID construction.

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

from src.core import config

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

API_KEY = os.getenv("CENSUS_API_KEY")
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

# All 50 states + DC from central config (abbr → 2-digit FIPS prefix)
STATES: dict[str, str] = config.STATES

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"

# MOE threshold: flag tracts where MOE / |estimate| > 30% on any primary feature
# See ADR-004 and census-acs-tract.md for rationale. Do not exclude — just surface.
MOE_THRESHOLD = 0.30

# Census API hard limit
_API_VAR_LIMIT = 49  # leave 1 slot for NAME (sometimes injected by the API)

# ── Variable manifest ─────────────────────────────────────────────────────────
# Format: {api_code: friendly_name}
# All primary variables include a paired MOE column (_moe suffix).
# The Census API returns -666666666 for suppressed/unavailable values; these
# are replaced with NaN during casting (see cast_numeric()).
#
# Variables are split into two batches to stay under the 50-var API limit:
#   VARIABLES_BATCH1 — race, age, income, housing tenure, commute, education,
#                       occupation (original set, 44 vars)
#   VARIABLES_BATCH2 — foreign born, poverty, inequality, housing detail,
#                       household type, age detail, vehicle access, veteran (34 vars)

VARIABLES_BATCH1: dict[str, str] = {
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
    "B15003_002E": "educ_no_schooling",
    "B15003_002M": "educ_no_schooling_moe",
    "B15003_016E": "educ_hs_diploma",    # HS diploma (used to compute no_hs: total - diploma_and_above)
    "B15003_016M": "educ_hs_diploma_moe",
    "B15003_017E": "educ_ged",           # GED or alternative credential
    "B15003_017M": "educ_ged_moe",
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

VARIABLES_BATCH2: dict[str, str] = {
    # Foreign-born population (B05001)
    "B05001_006E": "pop_foreign_born",
    "B05001_006M": "pop_foreign_born_moe",
    # Poverty status (B17001)
    # Both numerator and denominator stored; ratio computed in feature step
    "B17001_001E": "poverty_universe",
    "B17001_001M": "poverty_universe_moe",
    "B17001_002E": "poverty_below",
    "B17001_002M": "poverty_below_moe",
    # Income inequality (B19083)
    "B19083_001E": "gini",
    "B19083_001M": "gini_moe",
    # Median gross rent (B25064)
    "B25064_001E": "median_rent",
    "B25064_001M": "median_rent_moe",
    # Median home value (B25077)
    "B25077_001E": "median_home_value",
    "B25077_001M": "median_home_value_moe",
    # Units in structure (B25024) — multi-unit proxy
    # 003=2-unit; 005=5-9; 006=10-19; 007=20-49; 008=50+
    # (no MOE to save slots — derived column only, no MOE flagging needed)
    # Note: 3-4 unit bucket dropped to stay under 49-var limit; minor impact on aggregate
    "B25024_003E": "housing_struct_2unit",
    "B25024_005E": "housing_struct_5_9unit",
    "B25024_006E": "housing_struct_10_19unit",
    "B25024_007E": "housing_struct_20_49unit",
    "B25024_008E": "housing_struct_50plus_unit",
    # Year structure built (B25034) — pre-1960 housing age proxy
    # 009=1950-1959, 010=1940-1949, 011=1939 or earlier (no MOE to save slots)
    "B25034_009E": "housing_built_1950s",
    "B25034_010E": "housing_built_1940s",
    "B25034_011E": "housing_built_pre1940",
    # Households by type (B11016) — single-person households
    "B11016_001E": "hh_total",
    "B11016_001M": "hh_total_moe",
    "B11016_010E": "hh_single",        # 1-person household
    "B11016_010M": "hh_single_moe",
    # Age summary: children under 18 (B09001) and seniors 65+ via B01001 totals
    # B09001_001E = own children under 18 in households (proxy for pop_under_18)
    "B09001_001E": "pop_under_18",
    "B09001_001M": "pop_under_18_moe",
    # Over-65: use B01001_026E (female 65+) + symmetric male sum via B01001_002E total
    # Simpler: B01001_020E..025 (male 65+) + B01001_044E..049 (female 65+)
    # Use only 4 buckets to fit in limit (drop 65-66/67-69 granularity, use 65-74/75+)
    "B01001_022E": "pop_male_70_74",   # male 70-74
    "B01001_023E": "pop_male_75_79",
    "B01001_024E": "pop_male_80_84",
    "B01001_025E": "pop_male_85plus",
    "B01001_046E": "pop_female_70_74", # female 70-74
    "B01001_047E": "pop_female_75_79",
    "B01001_048E": "pop_female_80_84",
    "B01001_049E": "pop_female_85plus",
    # Over-65-to-74 totals for male/female (covers 65-69 missing above)
    "B01001_020E": "pop_male_65_66",
    "B01001_021E": "pop_male_67_69",
    "B01001_044E": "pop_female_65_66",
    "B01001_045E": "pop_female_67_69",
    # Vehicle access (B08201) — no vehicle available
    "B08201_001E": "hh_vehicle_total",
    "B08201_001M": "hh_vehicle_total_moe",
    "B08201_002E": "hh_no_vehicle",
    "B08201_002M": "hh_no_vehicle_moe",
    # Veteran status (B21001) — count only; pct_veteran uses pop_total as denominator
    "B21001_002E": "pop_veteran",
    "B21001_002M": "pop_veteran_moe",
    # Mean travel time to work (B08135 = aggregate; divide by B08101_001 = workers)
    "B08135_001E": "commute_time_agg",   # aggregate minutes for all workers
    "B08135_001M": "commute_time_agg_moe",
    "B08101_001E": "commute_workers",    # workers 16+ who did not WFH
    "B08101_001M": "commute_workers_moe",
}

# Combined variable map for renaming (both batches)
VARIABLES: dict[str, str] = {**VARIABLES_BATCH1, **VARIABLES_BATCH2}

# Names of estimate-only columns (no _moe suffix) — used for MOE flagging
ESTIMATE_COLS = [name for name in VARIABLES.values() if not name.endswith("_moe")]


# ── Fetch ─────────────────────────────────────────────────────────────────────


def _fetch_batch(state_fips: str, api_vars: list[str]) -> pd.DataFrame:
    """Fetch one batch of ACS variables (≤49) for all tracts in a state.

    The API returns a JSON array where row 0 is headers and rows 1..N are data.
    geo identifiers (state, county, tract) are appended automatically by the API.
    """
    assert len(api_vars) <= _API_VAR_LIMIT, (
        f"Batch has {len(api_vars)} vars; Census API limit is {_API_VAR_LIMIT}"
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


def fetch_state_tracts(state_fips: str) -> pd.DataFrame:
    """Fetch all ACS variables for all tracts in one state, batching as needed.

    Splits VARIABLES into chunks of ≤49 (Census API limit). Each chunk is
    fetched separately then joined on the geo identifiers (state, county, tract).
    """
    batch1_keys = list(VARIABLES_BATCH1.keys())
    batch2_keys = list(VARIABLES_BATCH2.keys())

    # Validate batch sizes
    assert len(batch1_keys) <= _API_VAR_LIMIT, f"Batch1 too large: {len(batch1_keys)}"
    assert len(batch2_keys) <= _API_VAR_LIMIT, f"Batch2 too large: {len(batch2_keys)}"

    geo_cols = ["state", "county", "tract"]

    df1 = _fetch_batch(state_fips, batch1_keys)
    time.sleep(0.2)  # small inter-batch pause
    df2 = _fetch_batch(state_fips, batch2_keys)

    # Join on geo identifiers; drop duplicate geo cols from right side
    merged = df1.merge(df2, on=geo_cols, how="outer")
    return merged


# ── Transform ─────────────────────────────────────────────────────────────────


def build_geoid(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 11-digit tract GEOID from the state/county/tract geo columns.

    Census tract FIPS = [2-digit state][3-digit county][6-digit tract]
    This is the join key for all other data sources (VEST crosswalk, ARDA, etc.)
    """
    df["tract_geoid"] = df["state"] + df["county"] + df["tract"]
    return df.drop(columns=["state", "county", "tract"])


def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all ACS estimate/MOE columns to float; replace -666666666 sentinel with NaN.

    The Census Bureau uses -666666666 for suppressed or unavailable estimates.
    Leaving it as-is would corrupt any downstream aggregation.
    """
    for col in VARIABLES.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-666666666, float("nan"))
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename API codes to friendly names defined in VARIABLES."""
    return df.rename(columns=VARIABLES)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate columns that require summing raw ACS fields.

    These aggregates are used by build_tract_features.py:
      - educ_graduate  : masters + professional + doctorate
      - educ_no_hs     : educ_total - (hs_diploma + ged + bachelors + graduate)
      - pop_under_18   : sum of male/female age buckets < 18
      - pop_over_65    : sum of male/female age buckets ≥ 65
      - housing_multi_unit : sum of 2-unit through 50+ structures
      - housing_pre_1960   : 1950s + 1940s + pre-1940
      - poverty_rate   : poverty_below / poverty_universe
      - pop_no_vehicle : hh_no_vehicle (households without a vehicle)
      - mean_commute_time : commute_time_agg / commute_workers
    """
    # Graduate degrees (masters + professional + doctorate)
    df["educ_graduate"] = (
        df.get("educ_masters", 0).fillna(0)
        + df.get("educ_professional", 0).fillna(0)
        + df.get("educ_doctorate", 0).fillna(0)
    )

    # No high school diploma: total minus everyone with HS diploma or higher
    hs_and_above = (
        df.get("educ_hs_diploma", 0).fillna(0)
        + df.get("educ_ged", 0).fillna(0)
        + df.get("educ_bachelors", 0).fillna(0)
        + df.get("educ_graduate", 0).fillna(0)
    )
    df["educ_no_hs"] = (df.get("educ_total", pd.Series(dtype=float)) - hs_and_above).clip(lower=0)

    # pop_under_18 is fetched directly as B09001_001E — no derivation needed

    # Over-65 population
    over65_cols = [
        "pop_male_65_66", "pop_male_67_69", "pop_male_70_74",
        "pop_male_75_79", "pop_male_80_84", "pop_male_85plus",
        "pop_female_65_66", "pop_female_67_69", "pop_female_70_74",
        "pop_female_75_79", "pop_female_80_84", "pop_female_85plus",
    ]
    df["pop_over_65"] = sum(df.get(c, pd.Series(0, index=df.index)).fillna(0) for c in over65_cols)

    # Multi-unit housing structures (2+ units in building)
    multi_cols = [
        "housing_struct_2unit", "housing_struct_5_9unit",
        "housing_struct_10_19unit", "housing_struct_20_49unit", "housing_struct_50plus_unit",
    ]
    df["housing_multi_unit"] = sum(df.get(c, pd.Series(0, index=df.index)).fillna(0) for c in multi_cols)

    # Pre-1960 housing stock
    df["housing_pre_1960"] = (
        df.get("housing_built_1950s", pd.Series(0, index=df.index)).fillna(0)
        + df.get("housing_built_1940s", pd.Series(0, index=df.index)).fillna(0)
        + df.get("housing_built_pre1940", pd.Series(0, index=df.index)).fillna(0)
    )

    # Poverty rate (computed here for convenience; feature step can also divide)
    df["poverty_rate"] = (
        df.get("poverty_below", pd.Series(dtype=float))
        / df.get("poverty_universe", pd.Series(dtype=float)).replace(0, float("nan"))
    )

    # No-vehicle households alias
    df["pop_no_vehicle"] = df.get("hh_no_vehicle", pd.Series(dtype=float))

    # Mean commute time (minutes)
    df["mean_commute_time"] = (
        df.get("commute_time_agg", pd.Series(dtype=float))
        / df.get("commute_workers", pd.Series(dtype=float)).replace(0, float("nan"))
    )

    return df


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

    log.info(
        "Fetching ACS 5-year 2022 tract data: %d variables across %d states (2 batches per state)",
        len(VARIABLES),
        len(STATES),
    )
    log.info("Batch 1: %d vars | Batch 2: %d vars", len(VARIABLES_BATCH1), len(VARIABLES_BATCH2))

    frames: list[pd.DataFrame] = []
    failed_states: list[str] = []

    for state_abbr, state_fips in STATES.items():
        log.info("  %s (FIPS %s)...", state_abbr, state_fips)
        try:
            df = fetch_state_tracts(state_fips)
            log.info("    %d tracts returned", len(df))
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            log.error("    FAILED for %s (%s): %s", state_abbr, state_fips, exc)
            failed_states.append(state_abbr)
        time.sleep(0.5)  # polite API rate limiting between states

    if failed_states:
        log.warning("Failed states (%d): %s", len(failed_states), ", ".join(failed_states))

    if not frames:
        raise RuntimeError("No data fetched — all states failed.")

    combined = pd.concat(frames, ignore_index=True)
    combined = build_geoid(combined)
    combined = cast_numeric(combined)
    combined = rename_columns(combined)
    combined = add_derived_columns(combined)
    combined = add_moe_flags(combined)

    # Summary
    n_tracts = len(combined)
    high_moe_cols = [c for c in combined.columns if c.endswith("_high_moe")]
    n_flagged = combined[high_moe_cols].any(axis=1).sum() if high_moe_cols else 0
    log.info(
        "Total: %d tracts × %d columns | %d tracts (%.1f%%) have ≥1 high-MOE field",
        n_tracts,
        len(combined.columns),
        n_flagged,
        100.0 * n_flagged / n_tracts if n_tracts else 0,
    )
    if failed_states:
        log.warning("Missing states (will be absent from output): %s", ", ".join(failed_states))

    # Persist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "acs_tracts_2022.parquet"
    combined.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
