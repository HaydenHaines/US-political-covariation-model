"""
Stage 1 data assembly: fetch County Health Rankings (CHR) analytic data.

Source: County Health Rankings & Roadmaps (RWJF / University of Wisconsin)
Data: CHR Annual Analytic Data CSV
URL pattern: https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data{YEAR}.csv
Scope: All 50 states + DC (national coverage) — ~3,000+ counties

The County Health Rankings program publishes annual county-level health data
compiled from dozens of administrative and survey sources. The analytic CSV
contains both the measure values and their confidence intervals, with one
row per county.

**Column structure in CHR analytic CSV:**
  Column 4 (index 3): statecode (2-digit string)
  Column 5 (index 4): countycode (3-digit string)
  Column 6 (index 5): county (name)
  Column 7 (index 6): state (abbreviation)
  Data columns: labeled by measure name with suffixes _rawvalue, _cilow, _cihigh, etc.

**Variables fetched per county (raw values only):**
  v001_rawvalue  : Premature Death (YPLL rate per 100K, age-adjusted)
  v009_rawvalue  : Adult Smoking (% smokers)
  v011_rawvalue  : Adult Obesity (% obese)
  v049_rawvalue  : Excessive Drinking (% excessive drinkers)
  v003_rawvalue  : Uninsured (% uninsured)
  v004_rawvalue  : Primary Care Physicians (rate per 100K)
  v062_rawvalue  : Mental Health Providers (rate per 100K)
  v063_rawvalue  : Median Household Income ($)
  v024_rawvalue  : Children in Poverty (%)
  v069_rawvalue  : Insufficient Sleep (% of adults)
  v070_rawvalue  : Physical Inactivity (%)
  v042_rawvalue  : Severe Housing Problems (% households)
  v044_rawvalue  : Driving Alone to Work (% commuters)
  v058_rawvalue  : High School Completion (%)
  v023_rawvalue  : Some College (% with some post-secondary education)
  v085_rawvalue  : Life Expectancy (years)
  v060_rawvalue  : Diabetes Prevalence (%)
  v036_rawvalue  : Poor Mental Health Days (avg days/month)

**NaN handling:**
  - Counties with suppressed or unavailable data retain NaN values.
  - State-median imputation is applied in build_county_health_features.py.

**CLI:**
  --year YEAR   : Data year to fetch (default: try 2024, fall back to 2023)
  --force       : Re-download even if output file already exists (idempotent by default)

Output: data/raw/county_health_rankings/chr_{year}.parquet
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.core import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "county_health_rankings"

# All states + DC from central config (abbreviation → FIPS prefix)
STATES: dict[str, str] = cfg.get_state_fips()

# Set of state FIPS prefixes for filtering (all 50 states + DC)
TARGET_STATE_FIPS = frozenset(STATES.values())

# CHR analytic data URL pattern (try year in descending order)
CHR_URL_TEMPLATE = (
    "https://www.countyhealthrankings.org/sites/default/files/media/document/"
    "analytic_data{year}.csv"
)

# Default year to attempt; will fall back to FALLBACK_YEAR on failure
DEFAULT_YEAR = 2024
FALLBACK_YEAR = 2023

# Polite request timeout (seconds)
REQUEST_TIMEOUT = 60

# CHR measure codes → friendly column names (raw values only)
# Based on the standard CHR analytic data dictionary (2023/2024 editions).
# NOTE: v043_rawvalue (violent_crime_rate) was removed from CHR analytic data
# in 2023; it is intentionally absent from this mapping.
CHR_MEASURES: dict[str, str] = {
    "v001_rawvalue": "premature_death_rate",
    "v009_rawvalue": "adult_smoking_pct",
    "v011_rawvalue": "adult_obesity_pct",
    "v049_rawvalue": "excessive_drinking_pct",
    "v003_rawvalue": "uninsured_pct",
    "v004_rawvalue": "primary_care_physicians_rate",
    "v062_rawvalue": "mental_health_providers_rate",
    "v063_rawvalue": "median_household_income",
    "v024_rawvalue": "children_in_poverty_pct",
    "v069_rawvalue": "insufficient_sleep_pct",
    "v070_rawvalue": "physical_inactivity_pct",
    "v042_rawvalue": "severe_housing_problems_pct",
    "v044_rawvalue": "drive_alone_pct",
    "v058_rawvalue": "high_school_completion_pct",
    "v023_rawvalue": "some_college_pct",
    "v085_rawvalue": "life_expectancy",
    "v060_rawvalue": "diabetes_prevalence_pct",
    "v036_rawvalue": "poor_mental_health_days",
}

# Output column names (in order)
OUTPUT_COLUMNS = (
    ["county_fips", "state_abbr", "county_name", "data_year"]
    + list(CHR_MEASURES.values())
)


def build_url(year: int) -> str:
    """Construct the CHR analytic CSV URL for a given data year.

    Args:
        year: 4-digit integer data year (e.g. 2024).

    Returns:
        Full URL string to the CHR analytic CSV.
    """
    return CHR_URL_TEMPLATE.format(year=year)


def fetch_raw_csv(year: int) -> str | None:
    """Download the CHR analytic CSV for a given year.

    Args:
        year: 4-digit data year.

    Returns:
        Raw CSV text string, or None on failure.
    """
    url = build_url(year)
    log.info("  Fetching CHR %d from: %s", year, url)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        log.warning("  CHR request failed for year %d: %s", year, exc)
        return None


def parse_chr_csv(csv_text: str, year: int) -> pd.DataFrame:
    """Parse a raw CHR analytic CSV into a structured DataFrame.

    The CHR analytic CSV has a two-row header:
      - Row 0: measure descriptions (human-readable labels)
      - Row 1: column names (v001_rawvalue, etc.)
    Data starts at row 2.

    Performs:
    1. Skip the first header row (descriptions), use second as column names
    2. Build county_fips from statecode + countycode
    3. Filter to FL/GA/AL counties only (excluding state-level summary rows)
    4. Extract measure columns, rename to friendly names
    5. Coerce measure values to float

    Args:
        csv_text: Raw CSV text from the CHR website.
        year: Data year (attached as data_year column).

    Returns:
        DataFrame with OUTPUT_COLUMNS, one row per county.
        Empty DataFrame if parsing fails.
    """
    if not csv_text:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        # CHR analytic CSV: row 0 is descriptions, row 1 is column names, data starts row 2
        df = pd.read_csv(
            io.StringIO(csv_text),
            skiprows=1,  # skip the description row; row 1 becomes header
            dtype=str,
            low_memory=False,
        )
    except Exception as exc:
        log.error("  Failed to parse CHR CSV: %s", exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Normalize column names: lowercase and strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    log.info("  Parsed CSV: %d rows × %d columns", len(df), len(df.columns))

    # Identify FIPS columns — CHR uses 'statecode' and 'countycode'
    # The state-level summary rows have countycode == '000'
    statecode_col = _find_column(df, ["statecode", "state_code", "statefips"])
    countycode_col = _find_column(df, ["countycode", "county_code", "countyfips"])

    if statecode_col is None or countycode_col is None:
        log.error(
            "  Could not identify FIPS columns. Available: %s",
            list(df.columns[:20]),
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Build 5-digit county_fips from statecode (2-digit) + countycode (3-digit)
    # Zero-pad each component
    df["county_fips"] = (
        df[statecode_col].str.strip().str.zfill(2)
        + df[countycode_col].str.strip().str.zfill(3)
    )

    # Filter: exclude state-level rows (countycode == '000' or '0')
    county_mask = df[countycode_col].str.strip().str.lstrip("0").ne("") & (
        df[countycode_col].str.strip() != "000"
    )
    df = df[county_mask].copy()

    # Filter to target states (FL=12, GA=13, AL=01)
    state_prefix_mask = df["county_fips"].str[:2].isin(TARGET_STATE_FIPS)
    n_before = len(df)
    df = df[state_prefix_mask].copy()
    n_after = len(df)
    log.info(
        "  State filter: %d → %d rows (kept all 50 states + DC)",
        n_before, n_after,
    )

    if df.empty:
        log.warning("  No counties found after filtering")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Validate FIPS format
    fips_ok = df["county_fips"].str.match(r"^\d{5}$", na=False)
    n_bad = (~fips_ok).sum()
    if n_bad > 0:
        log.warning("  Dropping %d rows with non-5-digit FIPS", n_bad)
        df = df[fips_ok]

    # Map state prefix → abbreviation
    fips_to_abbr = {v: k for k, v in STATES.items()}
    df["state_abbr"] = df["county_fips"].str[:2].map(fips_to_abbr)

    # County name
    county_name_col = _find_column(df, ["county", "county_name", "countyname"])
    df["county_name"] = df[county_name_col].str.strip() if county_name_col else ""

    # Data year
    df["data_year"] = year

    # Extract and rename measure columns
    result_rows = df[["county_fips", "state_abbr", "county_name", "data_year"]].copy()
    result_rows = result_rows.reset_index(drop=True)

    for chr_col, friendly_col in CHR_MEASURES.items():
        if chr_col in df.columns:
            raw_series = pd.to_numeric(df[chr_col].reset_index(drop=True), errors="coerce")
            result_rows[friendly_col] = raw_series
        else:
            log.warning(
                "  CHR column '%s' not found — setting '%s' to NaN",
                chr_col, friendly_col,
            )
            result_rows[friendly_col] = float("nan")

    # Ensure all OUTPUT_COLUMNS are present
    for col in OUTPUT_COLUMNS:
        if col not in result_rows.columns:
            result_rows[col] = float("nan")

    return result_rows[OUTPUT_COLUMNS].reset_index(drop=True)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name found in df, or None.

    Args:
        df: DataFrame with normalized (lowercase, stripped) column names.
        candidates: List of possible column names to check, in priority order.

    Returns:
        First matching column name, or None if none found.
    """
    for name in candidates:
        if name in df.columns:
            return name
    return None


def main(year: int | None = None, force: bool = False) -> None:
    """Download CHR analytic data and save to parquet.

    Idempotent: skips download if the output file already exists unless
    ``force=True`` is passed.

    Attempts to download the requested year; falls back to FALLBACK_YEAR
    if the primary year is unavailable.

    Args:
        year: 4-digit data year. Defaults to DEFAULT_YEAR with FALLBACK_YEAR fallback.
        force: If True, re-download even when output file already exists.
    """
    if year is None:
        year = DEFAULT_YEAR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"chr_{year}.parquet"

    if output_path.exists() and not force:
        log.info(
            "CHR %d already downloaded at %s — skipping (pass --force to re-download)",
            year, output_path,
        )
        return

    log.info("Fetching County Health Rankings analytic data for year %d", year)
    log.info("Target: national coverage (%d states + DC)", len(STATES) - 1)
    log.info("Measures: %d", len(CHR_MEASURES))

    csv_text = fetch_raw_csv(year)

    if csv_text is None and year == DEFAULT_YEAR:
        log.warning(
            "Year %d not available; falling back to %d", year, FALLBACK_YEAR
        )
        time.sleep(1.0)
        csv_text = fetch_raw_csv(FALLBACK_YEAR)
        if csv_text is not None:
            year = FALLBACK_YEAR

    if csv_text is None:
        log.error(
            "Could not retrieve CHR data for year %d (or fallback %d). Aborting.",
            year, FALLBACK_YEAR,
        )
        return

    log.info("Parsing CHR CSV for year %d...", year)
    df = parse_chr_csv(csv_text, year)

    if df.empty:
        log.error("No data after parsing. Aborting.")
        return

    # Summary
    n_counties = len(df)
    n_states = df["state_abbr"].nunique()
    state_counts = df.groupby("state_abbr").size().to_dict()
    measure_cols = [c for c in df.columns if c not in ("county_fips", "state_abbr", "county_name", "data_year")]

    log.info(
        "\nSummary: %d counties across %d states | year: %d",
        n_counties, n_states, year,
    )
    for state, count in sorted(state_counts.items()):
        log.info("  %s: %d counties", state, count)

    # NaN audit
    nan_counts = df[measure_cols].isna().sum()
    n_any_nan = (nan_counts > 0).sum()
    if n_any_nan > 0:
        log.info("\nNaN counts per measure (non-zero only):")
        for col, n in nan_counts[nan_counts > 0].items():
            pct = 100 * n / n_counties
            log.info("  %-35s  %d (%.1f%%)", col, n, pct)

    df.to_parquet(output_path, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        output_path, len(df), len(df.columns),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch County Health Rankings analytic data (national, all 50 states + DC)."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=f"Data year to fetch (default: {DEFAULT_YEAR}, falls back to {FALLBACK_YEAR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if output file already exists",
    )
    args = parser.parse_args()
    main(year=args.year, force=args.force)
