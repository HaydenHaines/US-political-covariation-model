"""
Stage 1 feature engineering: compute county-level health features from CHR data.

Takes the raw County Health Rankings analytic data from fetch_county_health_rankings.py
and produces a clean feature DataFrame suitable for community characterization
and correlation with partisan lean.

Health outcomes are strongly correlated with political realignment. The "deaths
of despair" literature (Case & Deaton) and public health research show that
counties with poor health outcomes (high premature death, high uninsured rates,
low access to primary care) have shifted significantly toward Republicans since
2016. Health features complement the CDC mortality and COVID vaccination data
already in the pipeline.

**Features produced (18 total):**
  premature_death_rate          : Years of Potential Life Lost rate per 100K (age-adj)
  adult_smoking_pct             : % adults who currently smoke
  adult_obesity_pct             : % adults with BMI ≥ 30
  excessive_drinking_pct        : % adults reporting excessive alcohol consumption
  uninsured_pct                 : % population under 65 without health insurance
  primary_care_physicians_rate  : Primary care physicians per 100K population
  mental_health_providers_rate  : Mental health providers per 100K population
  median_household_income       : Median household income (dollars)
  children_in_poverty_pct       : % children in poverty
  insufficient_sleep_pct        : % adults reporting insufficient sleep
  physical_inactivity_pct       : % adults physically inactive
  severe_housing_problems_pct   : % households with ≥1 severe housing problem
  drive_alone_pct               : % commuters driving alone to work
  high_school_completion_pct    : % adults with high school diploma or equivalent
  some_college_pct              : % adults with some post-secondary education
  life_expectancy               : Average life expectancy (years)
  diabetes_prevalence_pct       : % adults with diabetes diagnosis
  poor_mental_health_days       : Average number of poor mental health days/month

NOTE: violent_crime_rate (v043) was removed from CHR analytic data as of 2023.
It is no longer present in CHR downloads and has been removed from this pipeline.

**NaN handling:**
  - Counties with suppressed or unavailable CHR data retain NaN values.
  - State-median imputation fills missing values. Matches the strategy used
    in build_covid_features.py, build_cdc_mortality_features.py, and build_qcew_features.py.

Input:  data/raw/county_health_rankings/chr_{year}.parquet
        (uses latest year available; prefers 2024, falls back to 2023)
Output: data/assembled/county_health_features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "county_health_rankings"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_health_features.parquet"

# Preferred data year (will search for latest available)
PREFERRED_YEAR = 2024
FALLBACK_YEAR = 2023

# Health feature column names (all measures from CHR analytic data)
CHR_FEATURE_COLS = [
    "premature_death_rate",
    "adult_smoking_pct",
    "adult_obesity_pct",
    "excessive_drinking_pct",
    "uninsured_pct",
    "primary_care_physicians_rate",
    "mental_health_providers_rate",
    "median_household_income",
    "children_in_poverty_pct",
    "insufficient_sleep_pct",
    "physical_inactivity_pct",
    "severe_housing_problems_pct",
    "drive_alone_pct",
    "high_school_completion_pct",
    "some_college_pct",
    "life_expectancy",
    "diabetes_prevalence_pct",
    "poor_mental_health_days",
]


def find_latest_chr_file(input_dir: Path, preferred_year: int, fallback_year: int) -> Path | None:
    """Locate the most recent CHR parquet file in the raw data directory.

    Searches for chr_{preferred_year}.parquet first, then chr_{fallback_year}.parquet.

    Args:
        input_dir: Directory containing chr_{year}.parquet files.
        preferred_year: First year to try.
        fallback_year: Year to try if preferred not found.

    Returns:
        Path to the CHR parquet file, or None if not found.
    """
    for year in (preferred_year, fallback_year):
        path = input_dir / f"chr_{year}.parquet"
        if path.exists():
            log.info("  Found CHR data at: %s", path)
            return path
    return None


def compute_chr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate CHR health features from the raw parquet DataFrame.

    Selects the county_fips, state_abbr, and all CHR feature columns.
    Coerces all feature values to float and clips physiologically impossible
    percentage values to [0, 100].

    Args:
        df: DataFrame from data/raw/county_health_rankings/chr_{year}.parquet
            with columns: county_fips, state_abbr, and CHR measure columns.

    Returns:
        DataFrame with columns: county_fips, state_abbr, data_year + CHR_FEATURE_COLS
        One row per county.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["county_fips", "state_abbr", "data_year"] + CHR_FEATURE_COLS
        )

    result = pd.DataFrame({
        "county_fips": df["county_fips"],
        "state_abbr": df["state_abbr"],
        "data_year": df["data_year"] if "data_year" in df.columns else None,
    })

    # Percentage features that must be in [0, 100]
    pct_cols = {
        "adult_smoking_pct",
        "adult_obesity_pct",
        "excessive_drinking_pct",
        "uninsured_pct",
        "children_in_poverty_pct",
        "insufficient_sleep_pct",
        "physical_inactivity_pct",
        "severe_housing_problems_pct",
        "drive_alone_pct",
        "high_school_completion_pct",
        "some_college_pct",
        "diabetes_prevalence_pct",
    }

    for col in CHR_FEATURE_COLS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if col in pct_cols:
                values = values.clip(lower=0, upper=100)
            result[col] = values.reset_index(drop=True)
        else:
            log.warning(
                "  Feature column '%s' not found in input; setting to NaN", col
            )
            result[col] = float("nan")

    return result.reset_index(drop=True)


def impute_chr_state_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN CHR feature values with state-level medians.

    State is derived from the first 2 digits of county_fips. Matches the
    imputation strategy used for COVID vaccination, CDC mortality, and QCEW
    features throughout the pipeline.

    Args:
        df: DataFrame from compute_chr_features() with county_fips, state_abbr,
            and CHR_FEATURE_COLS.

    Returns:
        DataFrame with NaN values filled by state-level medians.
        NaN values remain only if the entire state is missing data for that feature.
    """
    df = df.copy()
    df["state_fips"] = df["county_fips"].str[:2]

    for col in CHR_FEATURE_COLS:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        state_medians = df.groupby("state_fips")[col].median()
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "state_fips"].map(state_medians)

        n_remaining = df[col].isna().sum()
        log.info(
            "  CHR %-35s  %d NaN → imputed %d, remaining %d",
            col, n_missing, n_missing - n_remaining, n_remaining,
        )

    df = df.drop(columns=["state_fips"])
    return df


def main() -> None:
    """Compute CHR health features from raw data and save to parquet.

    Reads the latest available CHR parquet file from data/raw/county_health_rankings/,
    computes health features, imputes missing values with state medians,
    and saves to data/assembled/county_health_features.parquet.
    """
    input_path = find_latest_chr_file(INPUT_DIR, PREFERRED_YEAR, FALLBACK_YEAR)

    if input_path is None:
        log.error(
            "CHR data not found in %s.\n"
            "Run: uv run python src/assembly/fetch_county_health_rankings.py",
            INPUT_DIR,
        )
        return

    log.info("Loading CHR data from %s...", input_path)
    raw = pd.read_parquet(input_path)
    log.info("Loaded: %d counties × %d cols", len(raw), len(raw.columns))

    # Compute features
    features = compute_chr_features(raw)

    data_year = int(features["data_year"].iloc[0]) if "data_year" in features.columns else "unknown"
    log.info("Data year: %s", data_year)

    # Report NaN before imputation
    nan_counts = features[CHR_FEATURE_COLS].isna().sum()
    n_nan_cols = (nan_counts > 0).sum()
    if n_nan_cols > 0:
        log.info("\nNaN counts before imputation (%d columns):", n_nan_cols)
        for col, n in nan_counts[nan_counts > 0].items():
            pct = 100 * n / len(features)
            log.info("  %-35s  %d (%.1f%%)", col, n, pct)

    # Impute with state-level medians
    log.info("\nImputing with state-level medians...")
    features = impute_chr_state_medians(features)

    # Final NaN audit
    remaining_nan = features[CHR_FEATURE_COLS].isna().sum().sum()
    if remaining_nan > 0:
        log.warning(
            "%d NaN values remain after imputation (possibly entire state missing data)",
            remaining_nan,
        )

    # Summary
    n_counties = len(features)
    n_states = features["state_abbr"].nunique()
    state_counts = features.groupby("state_abbr").size().to_dict()
    log.info(
        "\nSummary: %d counties across %d states | year: %s",
        n_counties, n_states, data_year,
    )
    for state, count in sorted(state_counts.items()):
        log.info("  %s: %d counties", state, count)

    # Feature distribution summary
    log.info("\nFeature ranges (non-NaN values, all counties):")
    for col in CHR_FEATURE_COLS:
        valid = features[col].dropna()
        if len(valid) > 0:
            q1, med, q3 = valid.quantile([0.25, 0.5, 0.75])
            log.info("  %-35s  Q1=%.2f  median=%.2f  Q3=%.2f", col, q1, med, q3)

    # Final column order
    out_cols = ["county_fips", "state_abbr", "data_year"] + CHR_FEATURE_COLS
    output = features[out_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        OUTPUT_PATH, len(output), len(output.columns),
    )


if __name__ == "__main__":
    main()
