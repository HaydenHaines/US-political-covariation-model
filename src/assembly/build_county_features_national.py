"""Build a consolidated national county features file joining ACS + RCMS + QCEW + CHR.

Reads:
    data/assembled/county_acs_features.parquet         (3,144 counties × 15 cols)
    data/assembled/county_rcms_features.parquet        (3,141 counties × 8 cols)
    data/assembled/county_qcew_features.parquet        (12,768 rows county_fips×year × 12 cols)
    data/assembled/county_health_features.parquet      (3,143 counties × 21 cols)

Outputs:
    data/assembled/county_features_national.parquet  (3,100+ counties × ~45 cols)

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

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
RCMS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_rcms_features.parquet"
QCEW_PATH = PROJECT_ROOT / "data" / "assembled" / "county_qcew_features.parquet"
CHR_PATH = PROJECT_ROOT / "data" / "assembled" / "county_health_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"

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
) -> pd.DataFrame:
    """Join ACS, RCMS, QCEW, and CHR county features into a single national DataFrame.

    Parameters
    ----------
    acs:
        Output of build_county_acs_features.py — county_fips + 14 derived ACS cols.
    rcms:
        Output of build_features.py RCMS section — county_fips + state_abbr + 6 RCMS cols.
    qcew:
        Output of build_qcew_features.py — county_fips × year rows (12,768) with 8 industry
        share features. Aggregated to 2023 (latest year) inside this function. Optional.
    chr_df:
        Output of build_county_health_features.py — county_fips + 15 CHR health features.
        metadata cols (state_abbr, data_year) and ACS-overlap cols are dropped. Optional.

    Returns
    -------
    DataFrame with county_fips, pop_total, 13 ACS ratio features, 6 RCMS features,
    8 QCEW industry-mix features (if provided), and 15 CHR health features (if provided).
    Missing values for all joined sources are imputed with state-level medians.
    """
    # Validate FIPS format
    if not acs["county_fips"].str.len().eq(5).all():
        raise ValueError("ACS county_fips must be 5-char zero-padded strings")
    if not rcms["county_fips"].str.len().eq(5).all():
        raise ValueError("RCMS county_fips must be 5-char zero-padded strings")

    # Left join: keep all ACS counties, bring in RCMS where available
    rcms_cols = ["county_fips"] + RCMS_FEATURE_COLS
    merged = acs.merge(rcms[rcms_cols], on="county_fips", how="left")

    n_missing_rcms = merged[RCMS_FEATURE_COLS[0]].isna().sum()
    if n_missing_rcms > 0:
        log.info(
            "%d counties lack RCMS data — imputing with state-level medians",
            n_missing_rcms,
        )
        merged = _impute_state_medians(merged, RCMS_FEATURE_COLS)

    # ── QCEW industry features ───────────────────────────────────────────────
    if qcew is not None:
        if not qcew["county_fips"].str.len().eq(5).all():
            raise ValueError("QCEW county_fips must be 5-char zero-padded strings")

        # Aggregate to latest available year (2023)
        latest_year = qcew["year"].max()
        log.info("Aggregating QCEW to latest year: %d", latest_year)
        qcew_latest = qcew[qcew["year"] == latest_year][["county_fips"] + QCEW_FEATURE_COLS].copy()

        merged = merged.merge(qcew_latest, on="county_fips", how="left")

        n_missing_qcew = merged[QCEW_FEATURE_COLS[0]].isna().sum()
        if n_missing_qcew > 0:
            log.info(
                "%d counties lack QCEW data — imputing with state-level medians",
                n_missing_qcew,
            )
            merged = _impute_state_medians(merged, QCEW_FEATURE_COLS)

    # ── CHR health features ──────────────────────────────────────────────────
    if chr_df is not None:
        if not chr_df["county_fips"].str.len().eq(5).all():
            raise ValueError("CHR county_fips must be 5-char zero-padded strings")

        chr_cols = ["county_fips"] + CHR_FEATURE_COLS
        merged = merged.merge(chr_df[chr_cols], on="county_fips", how="left")

        n_missing_chr = merged[CHR_FEATURE_COLS[0]].isna().sum()
        if n_missing_chr > 0:
            log.info(
                "%d counties lack CHR data — imputing with state-level medians",
                n_missing_chr,
            )
            merged = _impute_state_medians(merged, CHR_FEATURE_COLS)

    return merged.reset_index(drop=True)


def main() -> None:
    log.info("Loading ACS county features from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)
    log.info("  %d counties × %d cols", len(acs), len(acs.columns))

    log.info("Loading RCMS county features from %s", RCMS_PATH)
    rcms = pd.read_parquet(RCMS_PATH)
    log.info("  %d counties × %d cols", len(rcms), len(rcms.columns))

    # Load QCEW (industry) features if available
    qcew: pd.DataFrame | None = None
    if QCEW_PATH.exists():
        log.info("Loading QCEW county features from %s", QCEW_PATH)
        qcew = pd.read_parquet(QCEW_PATH)
        log.info("  %d rows × %d cols (county × year)", len(qcew), len(qcew.columns))
    else:
        log.warning("QCEW features not found at %s — skipping", QCEW_PATH)

    # Load CHR (County Health Rankings) features if available
    chr_df: pd.DataFrame | None = None
    if CHR_PATH.exists():
        log.info("Loading CHR county features from %s", CHR_PATH)
        chr_df = pd.read_parquet(CHR_PATH)
        log.info("  %d counties × %d cols", len(chr_df), len(chr_df.columns))
    else:
        log.warning("CHR features not found at %s — skipping", CHR_PATH)

    features = build_national_features(acs, rcms, qcew=qcew, chr_df=chr_df)

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

    # Sanity check: FIPS format
    bad_fips = features[~features["county_fips"].str.match(r"^\d{5}$")]
    if len(bad_fips):
        log.error("%d rows with malformed county_fips: %s", len(bad_fips), bad_fips["county_fips"].tolist()[:5])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)

    # Summary stats for key features
    log.info("\nFeature summary (all counties):")
    summary_cols = ["pct_white_nh", "pct_black", "pct_hispanic", "pct_bachelors_plus",
                    "median_hh_income", "evangelical_share", "catholic_share"]
    for col in summary_cols:
        if col not in features.columns:
            continue
        q1, med, q3 = features[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-30s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)


if __name__ == "__main__":
    main()
