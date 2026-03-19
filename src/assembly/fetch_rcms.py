"""
Stage 1 data assembly: fetch RCMS 2020 county-level religious congregation data.

Source: Association of Religion Data Archives (ARDA) — thearda.com
Data: Religious Congregations and Membership Study (RCMS) 2020
Scope: FL (FIPS 12), GA (FIPS 13), AL (FIPS 01) — 293 counties total

The ARDA serves RCMS 2020 data through an interactive county map tool at:
  https://www.thearda.com/us-religion/maps/us-county-maps

Data is scraped by constructing parameterized GET requests. The URL parameters are:
  st   = state abbreviation (AL, FL, GA)
  rt   = report type: 2 = Major Religious Groups (traditions)
  code = group code: 9999=All, 1=Evangelical, 2=Mainline, 3=Catholic,
                     6=Black Protestant, 4=Other, 5=Orthodox
  t    = variable + year: "0y2020"=Congregations, "1y2020"=Adherents,
                          "2y2020"=Adherence Rate per 1,000
  m1   = encoded parameter string: "{t_prefix}_{rt}_{code}_{year}"

Variables fetched per county:
  total_adherents      : RCMS count of all religious adherents
  evang_adherents      : Evangelical Protestant adherents
  mainline_adherents   : Mainline Protestant adherents
  catholic_adherents   : Catholic adherents
  black_prot_adherents : Black Protestant adherents
  total_congregations  : Total congregation count (all groups)
  adherence_rate       : Adherents per 1,000 residents (ARDA computed)

Derived features (computed in build_features.py):
  evangelical_share    : evang / total_adherents
  mainline_share       : mainline / total_adherents
  catholic_share       : catholic / total_adherents
  black_prot_share     : black_protestant / total_adherents
  congregations_per_1000 : total_congregations / (adherence_rate / 1000 * county_pop)
  religious_adherence_rate : adherence_rate (pass-through)

NaN handling:
  - Counties missing from ARDA (typically very small or suppressed) are retained
    with NaN values. Downstream imputation uses county-level ACS population data.
  - Not all counties appear in every RCMS tradition; missing entries → NaN.

Output: data/raw/rcms_county.parquet
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "rcms_county.parquet"

# States to fetch: abbreviation → FIPS prefix
STATES = {"AL": "01", "FL": "12", "GA": "13"}

# ARDA base URL for county map data
BASE_URL = "https://www.thearda.com/us-religion/maps/us-county-maps"

# Groups to fetch: friendly_name → ARDA group code
GROUPS: dict[str, str] = {
    "all": "9999",
    "evangelical": "1",
    "mainline": "2",
    "catholic": "3",
    "black_protestant": "6",
}

# Variables to fetch: friendly_name → ARDA t-parameter
VARIABLES: dict[str, str] = {
    "adherents": "1y2020",
    "congregations": "0y2020",
    "adherence_rate": "2y2020",
}

# Variable prefix for m1 encoding (maps t-value prefix → m1 prefix)
_T_PREFIX: dict[str, str] = {"0": "0", "1": "1", "2": "2"}

# Polite delay between requests (seconds)
REQUEST_DELAY = 1.0


def _make_m1(t_value: str, code: str) -> str:
    """Encode the m1 parameter from the t variable code and group code.

    ARDA's m1 format: "{t_prefix}_{rt}_{code}_{year}"
    where rt=2 (Major Religious Groups), t_prefix is the first char of t_value,
    and year is extracted from t_value.

    Examples:
        t_value="1y2020", code="9999" → "1_2_9999_2020"
        t_value="0y2020", code="1"    → "0_2_1_2020"
    """
    t_prefix, year = t_value.split("y")
    return f"{t_prefix}_2_{code}_{year}"


def _parse_map_data(html: str) -> dict[str, float]:
    """Extract county FIPS → value pairs from ARDA county map HTML.

    The data is embedded as a JavaScript array:
        county_map_divXXXX_data = [
            { id: "12001", value: 123.45 },
            ...
        ]

    Returns a dict mapping 5-digit county FIPS string to float value.
    Missing FIPS codes indicate counties with suppressed or no data.
    """
    entries = re.findall(r'\{ id: "(\d{5})",\s*value: ([\d.]+)', html)
    return {fips: float(val) for fips, val in entries}


def fetch_series(state: str, group_code: str, t_value: str) -> dict[str, float]:
    """Fetch one data series for one state from ARDA county map.

    Args:
        state: Two-letter state abbreviation (e.g. "AL")
        group_code: ARDA group code (e.g. "9999", "1")
        t_value: ARDA variable+year code (e.g. "1y2020")

    Returns:
        Dict mapping 5-digit county FIPS → value (may be empty on failure).
    """
    m1 = _make_m1(t_value, group_code)
    params = {
        "m1": m1,
        "color": "orange",
        "st": state,
        "rt": "2",
        "code": group_code,
        "t": t_value,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return _parse_map_data(resp.text)
    except requests.RequestException as exc:
        log.warning("  ARDA request failed for %s/%s/%s: %s", state, group_code, t_value, exc)
        return {}


def fetch_state(state: str) -> pd.DataFrame:
    """Fetch all RCMS variables for all counties in one state.

    Loops through the GROUPS × VARIABLES matrix, polling ARDA once per
    combination. Applies REQUEST_DELAY between requests to be polite.

    Returns a DataFrame with columns:
        county_fips   : 5-digit string
        state_abbr    : 2-letter string
        {var}_{group} : float, e.g. "adherents_all", "congregations_evangelical"
    """
    data: dict[str, dict] = {}  # county_fips → {col: value}

    # Fetch plan:
    #   - adherents: all groups (total + evangelical + mainline + catholic + black_protestant)
    #   - congregations: all-groups total only
    #   - adherence_rate: all-groups total only
    for var_name, t_value in VARIABLES.items():
        for grp_name, grp_code in GROUPS.items():
            # congregations and adherence_rate only needed for the "all" aggregate
            if var_name != "adherents" and grp_name != "all":
                continue

            col_name = f"{var_name}_{grp_name}" if grp_name != "all" else f"{var_name}_total"
            log.info("    %s | %s | %s → %s", state, var_name, grp_name, col_name)

            series = fetch_series(state, grp_code, t_value)
            for fips, val in series.items():
                if fips not in data:
                    data[fips] = {}
                data[fips][col_name] = val

            time.sleep(REQUEST_DELAY)

    if not data:
        log.warning("  No data retrieved for %s", state)
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "county_fips"
    df = df.reset_index()
    df["state_abbr"] = state

    # Reorder columns
    fixed_cols = ["county_fips", "state_abbr"]
    data_cols = sorted(c for c in df.columns if c not in fixed_cols)
    return df[fixed_cols + data_cols]


def main() -> None:
    log.info("Fetching RCMS 2020 county data from ARDA for %d states", len(STATES))
    log.info("Groups: %s", list(GROUPS.keys()))
    log.info("Variables: %s", list(VARIABLES.keys()))
    log.info("Note: congregations and adherence_rate fetched for 'all' group only.")
    log.info("      adherents fetched for all groups.")

    frames: list[pd.DataFrame] = []
    for state_abbr in STATES:
        log.info("  Fetching %s...", state_abbr)
        df = fetch_state(state_abbr)
        if df.empty:
            log.warning("  %s: no data returned", state_abbr)
            continue
        log.info("  %s: %d counties, %d columns", state_abbr, len(df), len(df.columns))
        frames.append(df)

    if not frames:
        log.error("No data retrieved for any state. Aborting.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Validate county FIPS format
    fips_ok = combined["county_fips"].str.match(r"^\d{5}$")
    if not fips_ok.all():
        bad = combined[~fips_ok]["county_fips"].unique()
        log.warning("Non-5-digit FIPS codes found (dropping): %s", bad)
        combined = combined[fips_ok]

    # Validate state prefixes match state_abbr
    fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
    derived_state = combined["county_fips"].str[:2].map(fips_to_abbr)
    mismatch = combined["state_abbr"] != derived_state
    if mismatch.any():
        log.warning(
            "%d rows have FIPS/state_abbr mismatch (retaining; derived may be more accurate):",
            mismatch.sum(),
        )

    # Summary
    n_counties = len(combined)
    n_states = combined["state_abbr"].nunique()
    data_cols = [c for c in combined.columns if c not in ("county_fips", "state_abbr")]
    nan_pct = combined[data_cols].isna().sum() / len(combined) * 100
    log.info(
        "\nSummary: %d counties across %d states | %d data columns",
        n_counties, n_states, len(data_cols),
    )
    for col, pct in nan_pct[nan_pct > 0].items():
        log.info("  %-40s  %.1f%% NaN", col, pct)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    log.info("\nSaved → %s  (%d rows × %d cols)", OUTPUT_PATH, len(combined), len(combined.columns))


if __name__ == "__main__":
    main()
