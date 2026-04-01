"""Build a consolidated national county features file joining all assembled data sources.

Reads:
    data/assembled/county_acs_features.parquet         (3,144 counties × 15 cols)
    data/assembled/county_rcms_features.parquet        (3,141 counties × 8 cols)
    data/assembled/county_qcew_features.parquet        (12,768 rows county_fips×year × 12 cols)
    data/assembled/county_health_features.parquet      (3,143 counties × 33+ cols, expanded CHR)
    data/assembled/county_migration_features.parquet   (3,127 counties × 5 cols)
    data/assembled/county_urbanicity_features.parquet  (3,135 counties × 4 cols)
    data/assembled/county_sci_features.parquet         (3,200+ counties × 5 cols)
    data/assembled/county_broadband_features.parquet   (3,100+ counties × 6 cols)
    data/assembled/county_bea_features.parquet         (3,149 counties × 5 cols)
    data/assembled/county_cdc_mortality_features.parquet (3,078 counties × 9 cols)
    data/assembled/county_covid_features.parquet       (3,208 counties × 5 cols)
    data/assembled/va_disability_features.parquet      (3,100+ counties × 4 cols)
    data/assembled/usda_typology_features.parquet      (3,100+ counties × 14 cols)
    data/assembled/transportation_features.parquet     (3,100+ counties × 8 cols)
    data/assembled/county_bea_growth_features.parquet  (3,144 counties × 4 cols)

Outputs:
    data/assembled/county_features_national.parquet  (3,100+ counties × ~90 cols)

Broadband features (from build_acs_broadband_features.py; ACS B28002):
    pct_broadband, pct_no_internet, pct_satellite, pct_cable_fiber, broadband_gap

Derived ACS features (from build_county_acs_features.py):
    pct_white_nh, pct_black, pct_asian, pct_hispanic
    median_age, median_hh_income, log_median_hh_income
    pct_bachelors_plus, pct_graduate
    pct_owner_occupied, pct_wfh, pct_transit, pct_management
    pop_total

RCMS features (joined on county_fips):
    evangelical_share, mainline_share, catholic_share, black_protestant_share
    congregations_per_1000, religious_adherence_rate

QCEW features (industry mix, aggregated to 2023; joined on county_fips):
    manufacturing_share, government_share, healthcare_share, retail_share,
    construction_share, finance_share, hospitality_share, industry_diversity_hhi
    Note: top_industry (categorical) and avg_annual_pay (ACS overlap) are dropped.

CHR features (County Health Rankings; joined on county_fips):
    premature_death_rate, adult_smoking_pct, adult_obesity_pct,
    excessive_drinking_pct, uninsured_pct, primary_care_physicians_rate,
    mental_health_providers_rate, children_in_poverty_pct, insufficient_sleep_pct,
    physical_inactivity_pct, severe_housing_problems_pct, drive_alone_pct,
    life_expectancy, diabetes_prevalence_pct, poor_mental_health_days
    Note: median_household_income, high_school_completion_pct, some_college_pct
    are dropped (overlap with ACS).

Counties missing RCMS/QCEW/CHR data are imputed with state-level medians for
each feature, consistent with the strategy in build_features.py.

FIPS format: string "01001" throughout (zero-padded, 5 characters).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
RCMS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_rcms_features.parquet"
QCEW_PATH = PROJECT_ROOT / "data" / "assembled" / "county_qcew_features.parquet"
CHR_PATH = PROJECT_ROOT / "data" / "assembled" / "county_health_features.parquet"
MIGRATION_PATH = PROJECT_ROOT / "data" / "assembled" / "county_migration_features.parquet"
URBANICITY_PATH = PROJECT_ROOT / "data" / "assembled" / "county_urbanicity_features.parquet"
SCI_PATH = PROJECT_ROOT / "data" / "assembled" / "county_sci_features.parquet"
BROADBAND_PATH = PROJECT_ROOT / "data" / "assembled" / "county_broadband_features.parquet"
BEA_PATH = PROJECT_ROOT / "data" / "assembled" / "county_bea_features.parquet"
BEA_STATE_PATH = PROJECT_ROOT / "data" / "assembled" / "county_bea_state_features.parquet"
CDC_MORTALITY_PATH = PROJECT_ROOT / "data" / "assembled" / "county_cdc_mortality_features.parquet"
COVID_PATH = PROJECT_ROOT / "data" / "assembled" / "county_covid_features.parquet"
VA_PATH = PROJECT_ROOT / "data" / "assembled" / "va_disability_features.parquet"
USDA_PATH = PROJECT_ROOT / "data" / "assembled" / "usda_typology_features.parquet"
TRANSPORT_PATH = PROJECT_ROOT / "data" / "assembled" / "transportation_features.parquet"
BEA_GROWTH_PATH = PROJECT_ROOT / "data" / "assembled" / "county_bea_growth_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"

# How many malformed FIPS to show in the error log before truncating.
MAX_FIPS_IN_LOG = 5

# Columns logged in the per-feature summary at the end of main().
SUMMARY_LOG_COLS = [
    "pct_white_nh",
    "pct_black",
    "pct_hispanic",
    "pct_bachelors_plus",
    "median_hh_income",
    "evangelical_share",
    "catholic_share",
    "pct_broadband",
    "pct_no_internet",
]

RCMS_FEATURE_COLS = [
    "evangelical_share",
    "mainline_share",
    "catholic_share",
    "black_protestant_share",
    "congregations_per_1000",
    "religious_adherence_rate",
]

# QCEW industry-mix features (top_industry dropped: categorical; avg_annual_pay: ACS overlap)
QCEW_FEATURE_COLS = [
    "manufacturing_share",
    "government_share",
    "healthcare_share",
    "retail_share",
    "construction_share",
    "finance_share",
    "hospitality_share",
    "industry_diversity_hhi",
]

# Migration features (IRS SOI inflow/outflow; derived from county_migration_features.parquet)
MIGRATION_FEATURE_COLS = [
    "net_migration_rate",
    "avg_inflow_income",
    "migration_diversity",
    "inflow_outflow_ratio",
]

# Urbanicity features (Census Gazetteer + ACS pop_total)
URBANICITY_FEATURE_COLS = [
    "log_pop_density",
    "land_area_sq_mi",
    "pop_per_sq_mi",
]

# SCI features (Facebook Social Connectedness Index)
SCI_FEATURE_COLS = [
    "network_diversity",
    "pct_sci_instate",
    "sci_top5_mean_dem_share",
    "sci_geographic_reach",
]

# CHR features (median_household_income, high_school_completion_pct, some_college_pct
# dropped: overlap with ACS; state_abbr and data_year are metadata, not features)
CHR_FEATURE_COLS = [
    "premature_death_rate",
    "adult_smoking_pct",
    "adult_obesity_pct",
    "excessive_drinking_pct",
    "uninsured_pct",
    "primary_care_physicians_rate",
    "mental_health_providers_rate",
    "children_in_poverty_pct",
    "insufficient_sleep_pct",
    "physical_inactivity_pct",
    "severe_housing_problems_pct",
    "drive_alone_pct",
    "life_expectancy",
    "diabetes_prevalence_pct",
    "poor_mental_health_days",
]

# Expanded CHR features (from chr_2024 raw data; separate list to handle
# graceful fallback if CHR hasn't been rebuilt with expanded measures yet)
CHR_EXPANDED_COLS = [
    "drug_overdose_deaths_rate",
    "suicide_rate",
    "firearm_fatalities_rate",
    "food_insecurity_pct",
    "social_associations_rate",
    "voter_turnout_pct",
    "disconnected_youth_pct",
    "residential_segregation",
    "homeownership_pct",
    "census_participation_pct",
    "free_reduced_lunch_pct",
    "alcohol_impaired_driving_deaths_pct",
    "injury_deaths_rate",
    "single_parent_households_pct",
    "homicide_rate",
]

# Broadband / internet access features (ACS B28002)
BROADBAND_FEATURE_COLS = [
    "pct_broadband",
    "pct_no_internet",
    "pct_satellite",
    "pct_cable_fiber",
    "broadband_gap",
]

# BEA income/GDP features (Bureau of Economic Analysis, county-level)
BEA_FEATURE_COLS = [
    "pci",
    "pci_growth",
    "gdp_per_capita",
    "gdp_growth",
]

# BEA state-level GDP and income features (mapped from state to county via FIPS prefix)
# These are macro-economic context signals distinct from the county-level income shares.
BEA_STATE_FEATURE_COLS = [
    "bea_state_gdp_millions",
    "bea_state_income_per_capita",
]

# CDC mortality features (exclude reserved all-NaN cols: heart_disease_rate, cancer_rate,
# suicide_rate)
CDC_MORTALITY_FEATURE_COLS = [
    "drug_overdose_rate",
    "despair_death_rate",
    "covid_death_rate",
    "allcause_age_adj_rate",
    "excess_mortality_ratio",
]

# COVID vaccination features
COVID_FEATURE_COLS = [
    "vax_complete_pct",
    "vax_booster_pct",
    "vax_dose1_pct",
]

# VA disability features
VA_FEATURE_COLS = [
    "va_disability_per_1000",
    "va_disability_pct_100rated",
    "va_disability_pct_young",
]

# USDA economic typology features (binary flags + ordinal)
USDA_FEATURE_COLS = [
    "High_Farming_2025",
    "High_Mining_2025",
    "High_Manufacturing_2025",
    "High_Government_2025",
    "High_Recreation_2025",
    "Nonspecialized_2025",
    "Low_PostSecondary_Ed_2025",
    "Low_Employment_2025",
    "Population_Loss_2025",
    "Housing_Stress_2025",
    "Retirement_Destination_2025",
    "Persistent_Poverty_1721",
    "Industry_Dependence_2025",
]

# BEA state-level GDP and income growth rates (year-over-year momentum signals)
BEA_GROWTH_FEATURE_COLS = [
    "bea_gdp_growth_1yr",
    "bea_gdp_growth_2yr",
    "bea_income_growth_1yr",
]

# DOT transportation features (aggregated from tract-level)
TRANSPORT_FEATURE_COLS = [
    "transport_pop_density",
    "transport_job_density",
    "transport_intersection_density",
    "transport_pct_local_roads",
    "transport_broadband",
    "transport_dead_end_proportion",
    "transport_circuity_avg",
]


def _merge_feature_block(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    *,
    allow_partial_cols: bool = False,
    fill_zero: bool = False,
    skip_cols_in_df: bool = False,
) -> pd.DataFrame:
    """Merge one feature source into the accumulating county dataframe.

    Encapsulates the repeated validate → select cols → merge → impute pattern so
    each data-source block in build_national_features() is 1-2 lines instead of ~15.

    Parameters
    ----------
    df:
        The accumulating merged dataframe (ACS spine + previously merged sources).
    source_df:
        The incoming feature dataframe to merge in.
    feature_cols:
        The declared list of feature columns for this source (e.g. RCMS_FEATURE_COLS).
    label:
        Human-readable name used in log messages (e.g. "RCMS", "BEA").
    allow_partial_cols:
        When True, silently restrict feature_cols to columns that actually exist in
        source_df. Useful for sources whose schema may vary (BEA, CDC, COVID, VA,
        Transport). When False (default), all declared feature_cols must be present.
    fill_zero:
        When True, impute NaN values with 0 rather than the state median. Used for
        binary flag sources like USDA where absence means "not classified".
    skip_cols_in_df:
        When True, skip any feature_cols that are already present in df before
        merging. Used for the expanded-CHR pass to avoid duplicate columns.

    Returns
    -------
    Updated dataframe with source features merged and imputed.
    """
    # Validate FIPS format on source before touching the spine
    if not source_df["county_fips"].str.len().eq(5).all():
        raise ValueError(f"{label} county_fips must be 5-char zero-padded strings")

    # Determine which columns to bring in from source_df
    cols_to_merge = feature_cols
    if allow_partial_cols:
        cols_to_merge = [c for c in feature_cols if c in source_df.columns]
    if skip_cols_in_df:
        cols_to_merge = [c for c in cols_to_merge if c not in df.columns]

    if not cols_to_merge:
        log.info("No new %s columns to merge — skipping", label)
        return df

    df = df.merge(source_df[["county_fips"] + cols_to_merge], on="county_fips", how="left")

    # After merging, check which declared cols actually landed in df
    # (allow_partial_cols may have reduced the set further)
    actual_feature_cols = [c for c in cols_to_merge if c in df.columns]

    if not actual_feature_cols:
        return df

    n_missing = df[actual_feature_cols[0]].isna().sum()
    if n_missing == 0:
        return df

    if fill_zero:
        for col in actual_feature_cols:
            df[col] = df[col].fillna(0)
        log.info("%d counties lack %s data — filled missing flags with 0", n_missing, label)
    else:
        log.info("%d counties lack %s data — imputing with state-level medians", n_missing, label)
        df = _impute_state_medians(df, actual_feature_cols)

    return df


def _impute_state_medians(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Impute NaN values in cols using state-level medians (state derived from county_fips).

    Falls back to national median for counties whose entire state has NaN values
    (e.g., Connecticut 2022 planning regions which have no RCMS match because
    RCMS still uses the pre-2022 county structure).
    """
    df = df.copy()
    state_fips = df["county_fips"].str[:2]

    for col in cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        state_medians = df.groupby(state_fips)[col].median()
        mask = df[col].isna()
        df.loc[mask, col] = state_fips[mask].map(state_medians)

        # Fall back to national median for counties where state median is also NaN
        still_missing = df[col].isna()
        if still_missing.any():
            national_median = df[col].median()
            df.loc[still_missing, col] = national_median
            log.info(
                "  %-35s  %d NaN → state imputed, then %d → national median (%.3f)",
                col,
                n_missing,
                still_missing.sum(),
                national_median,
            )
        else:
            n_remaining = df[col].isna().sum()
            log.info(
                "  %-35s  %d NaN → imputed %d, remaining %d",
                col,
                n_missing,
                n_missing - n_remaining,
                n_remaining,
            )

    return df


def build_national_features(
    acs: pd.DataFrame,
    rcms: pd.DataFrame,
    qcew: pd.DataFrame | None = None,
    chr_df: pd.DataFrame | None = None,
    migration: pd.DataFrame | None = None,
    urbanicity: pd.DataFrame | None = None,
    sci: pd.DataFrame | None = None,
    broadband: pd.DataFrame | None = None,
    bea: pd.DataFrame | None = None,
    bea_state: pd.DataFrame | None = None,
    cdc_mortality: pd.DataFrame | None = None,
    covid: pd.DataFrame | None = None,
    va: pd.DataFrame | None = None,
    usda: pd.DataFrame | None = None,
    transport: pd.DataFrame | None = None,
    bea_growth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join all assembled county feature sources into a single feature matrix.

    All sources are left-joined onto the ACS spine (which defines the county set).
    Missing values are imputed with state-level medians, falling back to national
    median for counties in states with no data for a given feature.

    Returns
    -------
    DataFrame with county_fips, pop_total, and all available feature columns.
    """
    # Validate ACS and RCMS FIPS format up front — both are required, non-optional sources.
    # All optional sources are validated inside _merge_feature_block.
    if not acs["county_fips"].str.len().eq(5).all():
        raise ValueError("ACS county_fips must be 5-char zero-padded strings")
    if not rcms["county_fips"].str.len().eq(5).all():
        raise ValueError("RCMS county_fips must be 5-char zero-padded strings")

    # Left join: keep all ACS counties, bring in RCMS where available
    merged = acs.merge(rcms[["county_fips"] + RCMS_FEATURE_COLS], on="county_fips", how="left")
    n_missing_rcms = merged[RCMS_FEATURE_COLS[0]].isna().sum()
    if n_missing_rcms > 0:
        log.info("%d counties lack RCMS data — imputing with state-level medians", n_missing_rcms)
        merged = _impute_state_medians(merged, RCMS_FEATURE_COLS)

    # ── QCEW industry features ───────────────────────────────────────────────
    # QCEW is multi-year; aggregate to the latest available year before merging.
    if qcew is not None:
        latest_year = qcew["year"].max()
        log.info("Aggregating QCEW to latest year: %d", latest_year)
        qcew_latest = qcew[qcew["year"] == latest_year][["county_fips"] + QCEW_FEATURE_COLS].copy()
        merged = _merge_feature_block(merged, qcew_latest, QCEW_FEATURE_COLS, "QCEW")

    # ── CHR health features ──────────────────────────────────────────────────
    if chr_df is not None:
        merged = _merge_feature_block(merged, chr_df, CHR_FEATURE_COLS, "CHR")

    # ── Migration features ───────────────────────────────────────────────────
    if migration is not None:
        merged = _merge_feature_block(merged, migration, MIGRATION_FEATURE_COLS, "Migration")

    # ── Urbanicity features ──────────────────────────────────────────────────
    if urbanicity is not None:
        merged = _merge_feature_block(merged, urbanicity, URBANICITY_FEATURE_COLS, "Urbanicity")

    # ── SCI social connectedness features ────────────────────────────────────
    if sci is not None:
        merged = _merge_feature_block(merged, sci, SCI_FEATURE_COLS, "SCI")

    # ── Broadband / internet access features ────────────────────────────────
    if broadband is not None:
        merged = _merge_feature_block(merged, broadband, BROADBAND_FEATURE_COLS, "Broadband")

    # ── BEA income/GDP features ─────────────────────────────────────────────
    # allow_partial_cols=True: BEA schema may vary; only merge columns present in source.
    if bea is not None:
        merged = _merge_feature_block(merged, bea, BEA_FEATURE_COLS, "BEA", allow_partial_cols=True)

    # ── BEA state-level GDP and income features ──────────────────────────────
    # State-level macro signals (GDP in millions, income per capita) mapped from
    # state to county via FIPS prefix. Complements county-level BEA income shares.
    if bea_state is not None:
        merged = _merge_feature_block(
            merged,
            bea_state,
            BEA_STATE_FEATURE_COLS,
            "BEA state",
            allow_partial_cols=True,
        )

    # ── BEA state-level GDP and income growth features ────────────────────
    # Year-over-year growth rates capturing economic momentum at the state level.
    if bea_growth is not None:
        merged = _merge_feature_block(
            merged,
            bea_growth,
            BEA_GROWTH_FEATURE_COLS,
            "BEA growth",
            allow_partial_cols=True,
        )

    # ── CDC mortality features ──────────────────────────────────────────────
    # allow_partial_cols=True: some mortality columns are reserved/all-NaN in some runs.
    if cdc_mortality is not None:
        merged = _merge_feature_block(
            merged,
            cdc_mortality,
            CDC_MORTALITY_FEATURE_COLS,
            "CDC mortality",
            allow_partial_cols=True,
        )

    # ── COVID vaccination features ──────────────────────────────────────────
    if covid is not None:
        merged = _merge_feature_block(
            merged,
            covid,
            COVID_FEATURE_COLS,
            "COVID vaccination",
            allow_partial_cols=True,
        )

    # ── VA disability features ──────────────────────────────────────────────
    if va is not None:
        merged = _merge_feature_block(
            merged,
            va,
            VA_FEATURE_COLS,
            "VA disability",
            allow_partial_cols=True,
        )

    # ── USDA economic typology features ─────────────────────────────────────
    # fill_zero=True: binary flags — absence of a flag means "not classified", not missing.
    if usda is not None:
        merged = _merge_feature_block(
            merged,
            usda,
            USDA_FEATURE_COLS,
            "USDA typology",
            allow_partial_cols=True,
            fill_zero=True,
        )

    # ── DOT transportation features ─────────────────────────────────────────
    if transport is not None:
        merged = _merge_feature_block(
            merged,
            transport,
            TRANSPORT_FEATURE_COLS,
            "Transportation",
            allow_partial_cols=True,
        )

    # ── Expanded CHR features (if available in the assembled CHR file) ──────
    # These come from the same chr_df but are columns added by the expanded fetch.
    # skip_cols_in_df=True avoids re-merging columns already brought in by the CHR pass above.
    if chr_df is not None:
        available_expanded = [c for c in CHR_EXPANDED_COLS if c in chr_df.columns]
        if available_expanded:
            merged = _merge_feature_block(
                merged,
                chr_df,
                CHR_EXPANDED_COLS,
                "expanded CHR",
                allow_partial_cols=True,
                skip_cols_in_df=True,
            )
            n_added = sum(c in merged.columns for c in CHR_EXPANDED_COLS)
            log.info("Added %d expanded CHR features", n_added)

    return merged.reset_index(drop=True)


def _load_optional_source(path: Path, label: str) -> pd.DataFrame | None:
    """Load a parquet file if it exists, returning None and a warning if it does not.

    All optional feature sources follow the same load-or-skip pattern. This helper
    eliminates the 12 near-identical if/else blocks in main().
    """
    if not path.exists():
        log.warning("%s features not found at %s — skipping", label, path)
        return None
    log.info("Loading %s features from %s", label, path)
    df = pd.read_parquet(path)
    log.info("  %d rows × %d cols", len(df), len(df.columns))
    return df


def _log_quality_report(features: pd.DataFrame) -> None:
    """Log NaN counts per column and flag any malformed FIPS values.

    Called after build_national_features() to surface data quality issues before
    writing to disk. Warnings here indicate imputation gaps or source schema drift.
    """
    n_na_any = features.isnull().any(axis=1).sum()
    log.info(
        "Built %d county feature rows × %d columns | %d counties with ≥1 NaN",
        len(features),
        len(features.columns),
        n_na_any,
    )

    if n_na_any > 0:
        log.warning("Columns with remaining NaN:")
        for col in features.columns:
            n = features[col].isna().sum()
            if n > 0:
                log.warning("  %-35s  %d NaN", col, n)

    bad_fips = features[~features["county_fips"].str.match(r"^\d{5}$")]
    if len(bad_fips):
        log.error(
            "%d rows with malformed county_fips: %s",
            len(bad_fips),
            bad_fips["county_fips"].tolist()[:MAX_FIPS_IN_LOG],
        )


def _log_feature_summary(features: pd.DataFrame) -> None:
    """Log Q1/median/Q3 for key demographic and political features.

    Quick sanity check that imputation hasn't produced out-of-range values
    and that the feature distributions look reasonable.
    """
    log.info("\nFeature summary (all counties):")
    for col in SUMMARY_LOG_COLS:
        if col not in features.columns:
            continue
        q1, med, q3 = features[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-30s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)


def main() -> None:
    # ── Required sources (always present) ───────────────────────────────────
    log.info("Loading ACS county features from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)
    log.info("  %d counties × %d cols", len(acs), len(acs.columns))

    log.info("Loading RCMS county features from %s", RCMS_PATH)
    rcms = pd.read_parquet(RCMS_PATH)
    log.info("  %d counties × %d cols", len(rcms), len(rcms.columns))

    # ── Optional sources (skip gracefully if file is absent) ────────────────
    qcew = _load_optional_source(QCEW_PATH, "QCEW county")
    chr_df = _load_optional_source(CHR_PATH, "CHR county")
    migration = _load_optional_source(MIGRATION_PATH, "Migration county")
    urbanicity = _load_optional_source(URBANICITY_PATH, "Urbanicity county")
    sci = _load_optional_source(SCI_PATH, "SCI county")
    broadband = _load_optional_source(BROADBAND_PATH, "Broadband county")
    bea = _load_optional_source(BEA_PATH, "BEA county")
    bea_state = _load_optional_source(BEA_STATE_PATH, "BEA state GDP/income")
    cdc_mortality = _load_optional_source(CDC_MORTALITY_PATH, "CDC mortality county")
    covid = _load_optional_source(COVID_PATH, "COVID vaccination county")
    va = _load_optional_source(VA_PATH, "VA disability county")
    usda = _load_optional_source(USDA_PATH, "USDA typology county")
    transport = _load_optional_source(TRANSPORT_PATH, "Transportation county")
    bea_growth = _load_optional_source(BEA_GROWTH_PATH, "BEA growth county")

    # ── Build ────────────────────────────────────────────────────────────────
    features = build_national_features(
        acs,
        rcms,
        qcew=qcew,
        chr_df=chr_df,
        migration=migration,
        urbanicity=urbanicity,
        sci=sci,
        broadband=broadband,
        bea=bea,
        bea_state=bea_state,
        cdc_mortality=cdc_mortality,
        covid=covid,
        va=va,
        usda=usda,
        transport=transport,
        bea_growth=bea_growth,
    )

    # ── Quality checks, save, summary ────────────────────────────────────────
    _log_quality_report(features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)

    _log_feature_summary(features)


if __name__ == "__main__":
    main()
