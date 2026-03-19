"""
Stage 1 feature engineering: compute NMF-ready tract-level features from ACS data,
and county-level RCMS religious features for community detection enrichment.

Takes raw ACS counts from fetch_acs.py and produces a normalized feature matrix
suitable for NMF community detection in Stage 2.

Two-stage separation is enforced here: this script uses ONLY non-political data
(ACS demographics, economics, housing, commute, education, occupation, religion).
Election data lives in vest_tracts_2020.parquet and enters at Stage 3.

ACS Features computed (12 total):
  Race/ethnicity:  pct_white_nh, pct_black, pct_asian, pct_hispanic
  Economics:       log_median_income, pct_mgmt_occ
  Housing:         pct_owner_occ
  Commute:         pct_car_commute, pct_transit_commute, pct_wfh_commute
  Education:       pct_college_plus
  Age:             median_age

RCMS Features computed (6 total, county-level):
  Religious composition: evangelical_share, mainline_share, catholic_share,
                         black_protestant_share
  Congregational density: congregations_per_1000
  Religious participation: religious_adherence_rate

RCMS features are county-level only (RCMS has no sub-county data). They are
saved as a separate output to preserve the tract-level resolution of ACS features.
Downstream NMF can optionally join county RCMS features onto tracts via county FIPS.

NaN handling:
  Tract features:
  - Tracts with pop_total = 0 are flagged `is_uninhabited` and all ratio
    features set to NaN. Stage 2 should exclude them.
  - Other NaN values (suppressed ACS data, zero denominators) are imputed
    with the state-level median for that feature. Imputation is counted per
    tract in `n_features_imputed` for post-hoc review.

  County RCMS features:
  - Missing group adherents (e.g. Catholic in small rural counties with no
    RCMS-surveyed Catholic congregations) are treated as 0 adherents before
    computing shares. This is conservative but appropriate: absence from RCMS
    means negligible organized presence, not unknown.
  - Counties missing from RCMS entirely (0 in the data) retain NaN shares.
    These are imputed with state-level medians.

Income treatment:
  - ACS top-codes median household income at $250,001. log() reduces the
    ceiling effect and compresses the right tail.
  - log_median_income = log(median_hh_income); floor at log(10000) before
    imputation to avoid imputing log of a suppressed near-zero value.

Inputs:
  data/assembled/acs_tracts_2022.parquet
  data/raw/rcms_county.parquet          (optional; skipped if absent)
Outputs:
  data/assembled/tract_features.parquet        — (N_tracts, 12) ACS features
  data/assembled/county_rcms_features.parquet  — (293, 6) RCMS features
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "acs_tracts_2022.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "tract_features.parquet"

RCMS_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "rcms_county.parquet"
RCMS_OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_rcms_features.parquet"

# RCMS-derived feature names (county-level output)
RCMS_FEATURE_COLS = [
    "evangelical_share",
    "mainline_share",
    "catholic_share",
    "black_protestant_share",
    "congregations_per_1000",
    "religious_adherence_rate",
]

# Feature names in output order — determines column order in the NMF matrix
FEATURE_COLS = [
    "pct_white_nh",
    "pct_black",
    "pct_asian",
    "pct_hispanic",
    "log_median_income",
    "pct_mgmt_occ",
    "pct_owner_occ",
    "pct_car_commute",
    "pct_transit_commute",
    "pct_wfh_commute",
    "pct_college_plus",
    "median_age",
]


# ── Feature computation ───────────────────────────────────────────────────────


def compute_features(acs: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 12 interpretable community-detection features from ACS raw counts.

    All ratio features are bounded [0, 1]. Median age and log_median_income
    are continuous and will need min-max scaling before NMF in Stage 2.
    """
    df = pd.DataFrame({"tract_geoid": acs["tract_geoid"]})

    # Uninhabited flag — zero-population tracts produce NaN for all ratios
    df["is_uninhabited"] = acs["pop_total"] == 0

    # Race / ethnicity (denominator: total population)
    pop = acs["pop_total"].replace(0, np.nan)
    df["pct_white_nh"] = acs["pop_white_nh"] / pop
    df["pct_black"] = acs["pop_black"] / pop
    df["pct_asian"] = acs["pop_asian"] / pop
    df["pct_hispanic"] = acs["pop_hispanic"] / pop

    # Income (log-transform to compress right skew and soften the $250,001 top-code)
    df["log_median_income"] = np.log(acs["median_hh_income"].clip(lower=1))

    # Occupation: management/professional fraction (male + female combined)
    occ = acs["occ_total"].replace(0, np.nan)
    df["pct_mgmt_occ"] = (acs["occ_mgmt_male"] + acs["occ_mgmt_female"]) / occ

    # Housing tenure: owner-occupied fraction
    units = acs["housing_units"].replace(0, np.nan)
    df["pct_owner_occ"] = acs["housing_owner"] / units

    # Commute mode (denominator: total workers with a commute response)
    commute = acs["commute_total"].replace(0, np.nan)
    df["pct_car_commute"] = acs["commute_car"] / commute
    df["pct_transit_commute"] = acs["commute_transit"] / commute
    df["pct_wfh_commute"] = acs["commute_wfh"] / commute

    # Education: bachelor's or higher fraction of 25+ adults
    educ = acs["educ_total"].replace(0, np.nan)
    grad_degrees = (
        acs["educ_bachelors"]
        + acs["educ_masters"]
        + acs["educ_professional"]
        + acs["educ_doctorate"]
    )
    df["pct_college_plus"] = grad_degrees / educ

    # Median age: already a continuous ratio, no transformation needed
    df["median_age"] = acs["median_age"]

    return df


# ── NaN imputation ────────────────────────────────────────────────────────────


def impute_state_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaN feature values with the state-level median for that feature.

    State is derived from the first 2 digits of tract_geoid. State-level
    imputation preserves geographic context better than global medians —
    Florida's income distribution differs substantially from Alabama's.

    Uninhabited tracts are excluded from the median computation and remain NaN
    after imputation; Stage 2 should filter them out entirely.
    """
    df = df.copy()
    df["state_fips"] = df["tract_geoid"].str[:2]

    df["n_features_imputed"] = 0

    for col in FEATURE_COLS:
        if df[col].isna().sum() == 0:
            continue

        n_before = df[col].isna().sum()

        # Compute state medians from populated tracts only
        state_medians = (
            df.loc[~df["is_uninhabited"], ["state_fips", col]]
            .groupby("state_fips")[col]
            .median()
        )

        # Impute non-uninhabited tracts with their state median
        mask = df[col].isna() & ~df["is_uninhabited"]
        df.loc[mask, col] = df.loc[mask, "state_fips"].map(state_medians)

        n_after = df[col].isna().sum()
        n_imputed = n_before - n_after
        if n_imputed > 0:
            df.loc[mask, "n_features_imputed"] += 1
            log.info("  %-25s  %d NaN → imputed %d, remaining %d (uninhabited)",
                     col, n_before, n_imputed, n_after)

    df = df.drop(columns=["state_fips"])
    return df


# ── RCMS feature computation ──────────────────────────────────────────────────


def compute_rcms_features(rcms: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 6 interpretable religious community features from RCMS 2020 raw counts.

    All share features are bounded [0, 1]. Congregations per 1,000 and
    adherence rate are continuous and will need scaling before NMF in Stage 2.

    Missing group adherents (NaN in RCMS for small counties) are treated as 0
    before computing shares. This is conservative but appropriate: absence from
    RCMS means negligible organized presence. Counties with no RCMS data at all
    (adherents_total = NaN) retain NaN for all derived features.

    Args:
        rcms: DataFrame from data/raw/rcms_county.parquet

    Returns:
        DataFrame with county_fips, state_abbr, and RCMS_FEATURE_COLS.
    """
    df = pd.DataFrame({"county_fips": rcms["county_fips"], "state_abbr": rcms["state_abbr"]})

    # Total adherents (denominator for shares)
    total = rcms["adherents_total"].replace(0, np.nan)

    # For group-level adherents: NaN → 0 before computing share
    # Rationale: RCMS surveys all counties; NaN for a specific tradition means
    # that tradition had no reported congregations (negligible presence), not
    # that the data is missing entirely.
    evang = rcms["adherents_evangelical"].fillna(0)
    mainline = rcms["adherents_mainline"].fillna(0)
    catholic = rcms["adherents_catholic"].fillna(0)
    black_prot = rcms["adherents_black_protestant"].fillna(0)

    # Religious tradition shares (fraction of total adherents)
    df["evangelical_share"] = evang / total
    df["mainline_share"] = mainline / total
    df["catholic_share"] = catholic / total
    df["black_protestant_share"] = black_prot / total

    # Congregational density: congregations per 1,000 adherents
    # Note: ARDA provides adherence_rate (adherents per 1,000 pop), not pop directly.
    # Compute congregations_per_1000 as congregations / (total_adherents / 1000).
    # This is independent of county population and captures organizational density.
    df["congregations_per_1000"] = rcms["congregations_total"] / (total / 1000)

    # Religious adherence rate: adherents per 1,000 residents (ARDA computed)
    df["religious_adherence_rate"] = rcms["adherence_rate_total"]

    return df


def impute_rcms_state_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaN RCMS feature values with state-level medians.

    State is derived from the first 2 digits of county_fips. Matches the
    imputation strategy used for ACS tract features to maintain consistency.

    Counties with adherents_total = NaN (entirely missing from RCMS) are
    imputed; the county is small enough that state median is appropriate.
    """
    df = df.copy()
    df["state_fips"] = df["county_fips"].str[:2]

    for col in RCMS_FEATURE_COLS:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        state_medians = df.groupby("state_fips")[col].median()
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "state_fips"].map(state_medians)

        n_remaining = df[col].isna().sum()
        log.info(
            "  RCMS %-30s  %d NaN → imputed %d, remaining %d",
            col, n_missing, n_missing - n_remaining, n_remaining,
        )

    df = df.drop(columns=["state_fips"])
    return df


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    acs = pd.read_parquet(INPUT_PATH)
    log.info("Loaded ACS: %d tracts × %d cols", len(acs), len(acs.columns))

    # Compute features
    features = compute_features(acs)

    # Report NaN distribution before imputation
    nan_counts = features[FEATURE_COLS].isna().sum()
    log.info("\nNaN counts before imputation:")
    for col, n in nan_counts[nan_counts > 0].items():
        pct = 100 * n / len(features)
        log.info("  %-25s  %d (%.1f%%)", col, n, pct)

    # Impute
    log.info("\nImputing with state-level medians...")
    features = impute_state_medians(features)

    # Final NaN audit
    remaining_nan = features[FEATURE_COLS].isna().sum().sum()
    n_uninhabited = features["is_uninhabited"].sum()
    n_imputed_any = (features["n_features_imputed"] > 0).sum()
    log.info(
        "\nSummary: %d tracts | %d uninhabited (NaN retained) | "
        "%d had ≥1 imputed feature | %d NaN remaining (all uninhabited)",
        len(features), n_uninhabited, n_imputed_any, remaining_nan,
    )

    # Clip ratios to [0, 1] — rounding artifacts can push ratios just above 1
    ratio_cols = [c for c in FEATURE_COLS if c.startswith("pct_")]
    for col in ratio_cols:
        over = (features[col] > 1).sum()
        if over:
            log.warning("  %s: %d values > 1.0 (clipping)", col, over)
        features[col] = features[col].clip(0, 1)

    # Final column order: key + features + metadata
    out_cols = ["tract_geoid", "is_uninhabited", "n_features_imputed"] + FEATURE_COLS
    output = features[out_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s  (%d tracts × %d feature cols)", OUTPUT_PATH, len(output), len(FEATURE_COLS))

    # Feature distribution summary
    log.info("\nFeature ranges (populated tracts):")
    populated = output[~output["is_uninhabited"]]
    for col in FEATURE_COLS:
        q1, med, q3 = populated[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-25s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)

    # ── RCMS county-level features ────────────────────────────────────────────
    if not RCMS_INPUT_PATH.exists():
        log.warning(
            "\nRCMS data not found at %s. Skipping county RCMS features.\n"
            "Run: uv run python src/assembly/fetch_rcms.py",
            RCMS_INPUT_PATH,
        )
        return

    log.info("\nBuilding county RCMS features from %s...", RCMS_INPUT_PATH)
    rcms_raw = pd.read_parquet(RCMS_INPUT_PATH)
    log.info("Loaded RCMS: %d counties × %d cols", len(rcms_raw), len(rcms_raw.columns))

    rcms_features = compute_rcms_features(rcms_raw)

    # Report NaN before imputation
    rcms_nan = rcms_features[RCMS_FEATURE_COLS].isna().sum()
    log.info("RCMS NaN counts before imputation:")
    for col, n in rcms_nan[rcms_nan > 0].items():
        pct = 100 * n / len(rcms_features)
        log.info("  %-30s  %d (%.1f%%)", col, n, pct)

    # Impute with state-level medians
    log.info("Imputing RCMS with state-level medians...")
    rcms_features = impute_rcms_state_medians(rcms_features)

    # Clip shares to [0, 1]
    share_cols = [c for c in RCMS_FEATURE_COLS if c.endswith("_share")]
    for col in share_cols:
        over = (rcms_features[col] > 1).sum()
        if over:
            log.warning("  RCMS %s: %d values > 1.0 (clipping)", col, over)
        rcms_features[col] = rcms_features[col].clip(0, 1)

    # Final column order
    rcms_out_cols = ["county_fips", "state_abbr"] + RCMS_FEATURE_COLS
    rcms_output = rcms_features[rcms_out_cols]

    RCMS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rcms_output.to_parquet(RCMS_OUTPUT_PATH, index=False)
    log.info(
        "Saved → %s  (%d counties × %d RCMS feature cols)",
        RCMS_OUTPUT_PATH, len(rcms_output), len(RCMS_FEATURE_COLS),
    )

    # RCMS feature distribution summary
    log.info("\nRCMS feature ranges (all counties):")
    for col in RCMS_FEATURE_COLS:
        q1, med, q3 = rcms_output[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-30s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)


if __name__ == "__main__":
    main()
