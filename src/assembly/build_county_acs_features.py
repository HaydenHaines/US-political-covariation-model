"""Build derived demographic ratios from raw county ACS data.

Reads data/assembled/acs_counties_2022.parquet (raw counts) and outputs
data/assembled/county_acs_features.parquet with the following derived columns:

    pct_white_nh        pop_white_nh / pop_total
    pct_black           pop_black / pop_total
    pct_asian           pop_asian / pop_total
    pct_hispanic        pop_hispanic / pop_total
    median_age          pass-through
    median_hh_income    pass-through
    pct_bachelors_plus  (bachelors + masters + professional + doctorate) / educ_total
    pct_owner_occupied  housing_owner / housing_units
    pct_wfh             commute_wfh / commute_total
    pct_management      (occ_mgmt_male + occ_mgmt_female) / occ_total

Also retains county_fips and pop_total for population-weighted aggregation downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "acs_counties_2022.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numerator by denominator; return NaN where denominator is zero or NaN."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute derived ratios from raw ACS count columns.

    Parameters
    ----------
    raw:
        DataFrame produced by fetch_acs_county.py — county_fips + raw ACS counts.

    Returns
    -------
    DataFrame with county_fips, pop_total, and all derived feature columns.
    """
    df = raw[["county_fips"]].copy()

    # Population totals for weighting
    df["pop_total"] = raw["pop_total"]

    # Racial / ethnic composition
    df["pct_white_nh"] = _safe_ratio(raw["pop_white_nh"], raw["pop_total"])
    df["pct_black"] = _safe_ratio(raw["pop_black"], raw["pop_total"])
    df["pct_asian"] = _safe_ratio(raw["pop_asian"], raw["pop_total"])
    df["pct_hispanic"] = _safe_ratio(raw["pop_hispanic"], raw["pop_total"])

    # Age and income pass-throughs
    df["median_age"] = raw["median_age"]
    df["median_hh_income"] = raw["median_hh_income"]

    # Educational attainment: bachelor's degree or higher
    educ_higher = (
        raw["educ_bachelors"]
        + raw["educ_masters"]
        + raw["educ_professional"]
        + raw["educ_doctorate"]
    )
    df["pct_bachelors_plus"] = _safe_ratio(educ_higher, raw["educ_total"])

    # Housing tenure: owner-occupied
    df["pct_owner_occupied"] = _safe_ratio(raw["housing_owner"], raw["housing_units"])

    # Commute: work from home
    df["pct_wfh"] = _safe_ratio(raw["commute_wfh"], raw["commute_total"])

    # Occupation: management / professional
    occ_mgmt = raw["occ_mgmt_male"] + raw["occ_mgmt_female"]
    df["pct_management"] = _safe_ratio(occ_mgmt, raw["occ_total"])

    return df.reset_index(drop=True)


def main() -> None:
    log.info("Loading raw ACS county data from %s", INPUT_PATH)
    raw = pd.read_parquet(INPUT_PATH)
    log.info("  %d counties, %d columns", len(raw), len(raw.columns))

    features = build_features(raw)

    n_na = features.isnull().any(axis=1).sum()
    log.info(
        "Built %d county feature rows x %d columns | %d counties with at least one NaN",
        len(features),
        len(features.columns),
        n_na,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
