"""
Stage 1 feature engineering: compute county-level industry composition features from QCEW.

Takes the raw QCEW county data from fetch_bls_qcew.py and produces a clean feature
DataFrame suitable for community characterization and correlation with partisan lean.

Industry composition is a core predictor of political behavior: manufacturing decline
correlates with Republican realignment, healthcare-heavy counties track with Democratic
lean, and government employment creates distinct political dynamics around public-sector
unions and federal spending.

Features computed (10 total per year):
  manufacturing_share     : manufacturing employment / total employment
  government_share        : public administration employment / total employment
  healthcare_share        : health care & social assistance employment / total employment
  retail_share            : retail trade employment / total employment
  construction_share      : construction employment / total employment
  finance_share           : finance & insurance employment / total employment
  hospitality_share       : accommodation & food services employment / total employment
  industry_diversity_hhi  : Herfindahl-Hirschman Index of employment concentration
                            (sum of squared sector shares; lower = more diverse)
  top_industry            : NAICS code of largest sector by employment (string)
  avg_annual_pay          : total wages / total employment (average annual pay)

NaN handling:
  - Counties missing total employment retain NaN for all share features.
  - Counties with total employment = 0 produce NaN shares (avoids division by zero).
  - Downstream imputation uses state-level medians (consistent with RCMS/COVID strategy).

Input:  data/raw/qcew_county.parquet
Output: data/assembled/county_qcew_features.parquet
  One row per (county_fips, year) — downstream join on those two columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.assembly.fetch_bls_qcew import TOTAL_INDUSTRY_CODE
from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "qcew_county.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_qcew_features.parquet"

# NAICS codes for each feature sector (must match INDUSTRY_CODES in fetch_bls_qcew.py)
SECTOR_CODES: dict[str, str] = {
    "manufacturing": "31",
    "government": "92",
    "healthcare": "62",
    "retail": "44",
    "construction": "23",
    "finance": "52",
    "hospitality": "72",
    "transportation": "48",
}

# Feature column names in the output
QCEW_FEATURE_COLS = [
    "manufacturing_share",
    "government_share",
    "healthcare_share",
    "retail_share",
    "construction_share",
    "finance_share",
    "hospitality_share",
    "industry_diversity_hhi",
    "top_industry",
    "avg_annual_pay",
]

# Sector columns used for HHI calculation (must match _share columns above)
HHI_SECTOR_COLS = [
    "manufacturing_share",
    "government_share",
    "healthcare_share",
    "retail_share",
    "construction_share",
    "finance_share",
    "hospitality_share",
]


def pivot_qcew(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the raw QCEW edge-list into a wide county × sector DataFrame.

    The raw data has one row per (county_fips, year, industry_code). This
    function pivots to one row per (county_fips, year) with separate columns
    for each sector's employment.

    Args:
        df: Raw QCEW DataFrame from data/raw/qcew_county.parquet.

    Returns:
        Wide DataFrame with columns:
          county_fips, year, empl_{sector}, total_empl, total_wages
        where {sector} is the NAICS code string.
    """
    if df.empty:
        return pd.DataFrame()

    # Keep only relevant columns
    df = df[
        ["county_fips", "year", "industry_code", "annual_avg_emplvl", "total_annual_wages"]
    ].copy()

    # Pivot employment by industry_code
    empl_wide = df.pivot_table(
        index=["county_fips", "year"],
        columns="industry_code",
        values="annual_avg_emplvl",
        aggfunc="first",
    ).reset_index()

    # Flatten column names from multi-index
    empl_wide.columns.name = None
    empl_wide = empl_wide.rename(
        columns={code: f"empl_{code}" for code in empl_wide.columns if code not in ("county_fips", "year")}
    )

    # Get total wages from the "total" industry row
    wages_total = (
        df[df["industry_code"] == TOTAL_INDUSTRY_CODE]
        .groupby(["county_fips", "year"])["total_annual_wages"]
        .first()
        .reset_index()
        .rename(columns={"total_annual_wages": "total_wages"})
    )

    result = empl_wide.merge(wages_total, on=["county_fips", "year"], how="left")
    log.info("Pivoted to %d county-year rows", len(result))
    return result


def compute_shares(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute employment share features from the wide-format QCEW DataFrame.

    For each sector, divides sector employment by total employment. Handles
    zero-division by producing NaN (not 0 or inf).

    Args:
        wide: Wide-format DataFrame from pivot_qcew().

    Returns:
        DataFrame with county_fips, year, and sector share columns.
    """
    if wide.empty:
        return pd.DataFrame()

    result = wide[["county_fips", "year"]].copy()

    total_empl_col = f"empl_{TOTAL_INDUSTRY_CODE}"
    total_empl = wide.get(total_empl_col, pd.Series(dtype=float))

    for sector_name, naics_code in SECTOR_CODES.items():
        empl_col = f"empl_{naics_code}"
        share_col = f"{sector_name}_share"

        if empl_col in wide.columns and total_empl_col in wide.columns:
            sector_empl = pd.to_numeric(wide[empl_col], errors="coerce")
            total = pd.to_numeric(wide[total_empl_col], errors="coerce")
            # NaN where total is 0 or missing; clip to [0, 1]
            result[share_col] = (sector_empl / total.replace(0, float("nan"))).clip(0, 1)
        else:
            log.warning("  Column %s not found; setting %s to NaN", empl_col, share_col)
            result[share_col] = float("nan")

    return result


def compute_hhi(shares: pd.DataFrame) -> pd.Series:
    """Compute Herfindahl-Hirschman Index of employment concentration.

    HHI = sum of squared sector shares across the tracked sectors.
    Range: [0, 1] where 0 = perfectly diversified, 1 = single-sector economy.
    Counties with all-NaN shares produce NaN HHI.

    Args:
        shares: DataFrame with sector share columns.

    Returns:
        Series of HHI values (same index as shares).
    """
    sector_cols = [c for c in HHI_SECTOR_COLS if c in shares.columns]
    if not sector_cols:
        return pd.Series(float("nan"), index=shares.index)

    share_matrix = shares[sector_cols].clip(0, 1)
    hhi = (share_matrix**2).sum(axis=1)

    # If all sector shares are NaN for a county, HHI is NaN (not 0)
    all_nan = share_matrix.isna().all(axis=1)
    hhi[all_nan] = float("nan")

    return hhi


def compute_top_industry(wide: pd.DataFrame) -> pd.Series:
    """Find the NAICS code of the largest sector by employment (excluding total).

    Args:
        wide: Wide-format DataFrame from pivot_qcew().

    Returns:
        Series of NAICS code strings (e.g. "62" for healthcare-dominant).
        Counties with no sector data produce NaN.
    """
    sector_empl_cols = {
        naics_code: f"empl_{naics_code}"
        for naics_code in SECTOR_CODES.values()
        if f"empl_{naics_code}" in wide.columns
    }

    if not sector_empl_cols:
        return pd.Series(float("nan"), index=wide.index, dtype=object)

    empl_matrix = wide[[col for col in sector_empl_cols.values()]].copy()
    empl_matrix.columns = list(sector_empl_cols.keys())

    # idxmax gives the column (NAICS code) with the highest value per row
    # skipna=True: rows with all-NaN will produce NaN naturally via .where()
    top = empl_matrix.idxmax(axis=1, skipna=True)

    # Where all values are NaN, set to NaN
    all_nan = empl_matrix.isna().all(axis=1)
    top = top.where(~all_nan, other=float("nan"))

    return top.astype(object)


def compute_avg_pay(wide: pd.DataFrame) -> pd.Series:
    """Compute average annual pay from total wages and total employment.

    Args:
        wide: Wide-format DataFrame from pivot_qcew().

    Returns:
        Series of average annual pay values (dollars).
    """
    total_empl_col = f"empl_{TOTAL_INDUSTRY_CODE}"
    if "total_wages" not in wide.columns or total_empl_col not in wide.columns:
        return pd.Series(float("nan"), index=wide.index)

    total_empl = pd.to_numeric(wide[total_empl_col], errors="coerce")
    total_wages = pd.to_numeric(wide["total_wages"], errors="coerce")

    return (total_wages / total_empl.replace(0, float("nan"))).clip(lower=0)


def compute_qcew_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all QCEW county-level industry features.

    Full pipeline: pivot → shares → HHI → top_industry → avg_pay.

    Args:
        df: Raw QCEW DataFrame from data/raw/qcew_county.parquet.

    Returns:
        Feature DataFrame with columns: county_fips, year, + QCEW_FEATURE_COLS.
        One row per (county_fips, year).
    """
    if df.empty:
        return pd.DataFrame(columns=["county_fips", "year"] + QCEW_FEATURE_COLS)

    # Step 1: Pivot to wide format
    wide = pivot_qcew(df)
    if wide.empty:
        return pd.DataFrame(columns=["county_fips", "year"] + QCEW_FEATURE_COLS)

    # Step 2: Compute sector employment shares
    shares = compute_shares(wide)

    # Step 3: HHI concentration index
    shares["industry_diversity_hhi"] = compute_hhi(shares)

    # Step 4: Top industry by employment
    shares["top_industry"] = compute_top_industry(wide).values

    # Step 5: Average annual pay
    shares["avg_annual_pay"] = compute_avg_pay(wide).values

    # Reorder to standard output columns
    out_cols = ["county_fips", "year"] + QCEW_FEATURE_COLS
    available_out = [c for c in out_cols if c in shares.columns]
    return shares[available_out].reset_index(drop=True)


def impute_qcew_state_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN QCEW feature values with state-level medians, within year.

    State is derived from the first 2 digits of county_fips. Imputes within
    each (state, year) group separately, so time trends aren't smoothed away.
    Consistent with the imputation strategy used for RCMS and COVID features.

    Args:
        df: DataFrame from compute_qcew_features().

    Returns:
        DataFrame with NaN values filled by state×year medians.
    """
    df = df.copy()
    df["state_fips"] = df["county_fips"].str[:2]

    numeric_feature_cols = [c for c in QCEW_FEATURE_COLS if c != "top_industry"]

    for col in numeric_feature_cols:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        state_year_medians = df.groupby(["state_fips", "year"])[col].median()
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask].apply(
            lambda row, c=col, m=state_year_medians: m.get((row["state_fips"], row["year"]), float("nan")),
            axis=1,
        )

        n_remaining = df[col].isna().sum()
        log.info(
            "  QCEW %-28s  %d NaN → imputed %d, remaining %d",
            col, n_missing, n_missing - n_remaining, n_remaining,
        )

    df = df.drop(columns=["state_fips"])
    return df


def main() -> None:
    """Compute QCEW industry composition features and save to parquet.

    Reads data/raw/qcew_county.parquet (produced by fetch_bls_qcew.py),
    computes 10 industry features per county-year, imputes missing values,
    and saves to data/assembled/county_qcew_features.parquet.
    """
    if not INPUT_PATH.exists():
        log.error(
            "QCEW raw data not found at %s.\n"
            "Run: uv run python src/assembly/fetch_bls_qcew.py",
            INPUT_PATH,
        )
        return

    log.info("Loading QCEW county data from %s...", INPUT_PATH)
    raw = pd.read_parquet(INPUT_PATH)
    log.info("Loaded: %d rows × %d cols", len(raw), len(raw.columns))
    log.info("  Years: %s", sorted(raw["year"].unique()))
    log.info("  Counties: %d unique", raw["county_fips"].nunique())
    log.info("  Industry codes: %s", sorted(raw["industry_code"].unique()))

    # Compute features
    log.info("\nComputing features...")
    features = compute_qcew_features(raw)
    log.info("Features computed: %d county-year rows × %d cols", len(features), len(features.columns))

    if features.empty:
        log.error("No features computed. Aborting.")
        return

    # NaN audit before imputation
    numeric_feat_cols = [c for c in QCEW_FEATURE_COLS if c != "top_industry" and c in features.columns]
    nan_counts = features[numeric_feat_cols].isna().sum()
    log.info("\nNaN counts before imputation:")
    for col, n in nan_counts[nan_counts > 0].items():
        pct = 100 * n / len(features)
        log.info("  %-30s  %d (%.1f%%)", col, n, pct)

    # Impute with state×year medians
    log.info("\nImputing with state×year medians...")
    features = impute_qcew_state_medians(features)

    # Final NaN audit
    remaining_nan = features[numeric_feat_cols].isna().sum().sum()
    if remaining_nan > 0:
        log.warning(
            "%d NaN values remain after imputation (possibly entire state/year missing data)",
            remaining_nan,
        )

    # Summary
    n_county_years = len(features)
    n_counties = features["county_fips"].nunique()
    n_years = features["year"].nunique()
    state_counts = features["county_fips"].str[:2].value_counts().to_dict()
    log.info(
        "\nSummary: %d county-year rows | %d unique counties | %d year(s)",
        n_county_years, n_counties, n_years,
    )
    fips_to_abbr = _cfg.STATE_ABBR  # maps "01" → "AL", "12" → "FL", etc. (all 50+DC)
    for fips_pref, count in sorted(state_counts.items()):
        abbr = fips_to_abbr.get(fips_pref, fips_pref)
        log.info("  %s: %d county-year rows", abbr, count)

    # Feature distribution summary
    log.info("\nFeature ranges (latest year):")
    latest_year = features["year"].max()
    latest = features[features["year"] == latest_year]
    for col in numeric_feat_cols:
        if col in latest.columns:
            q1, med, q3 = latest[col].quantile([0.25, 0.5, 0.75])
            log.info("  %-30s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)

    # Final column order
    out_cols = ["county_fips", "year"] + QCEW_FEATURE_COLS
    available_out = [c for c in out_cols if c in features.columns]
    output = features[available_out]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        OUTPUT_PATH, len(output), len(output.columns),
    )


if __name__ == "__main__":
    main()
