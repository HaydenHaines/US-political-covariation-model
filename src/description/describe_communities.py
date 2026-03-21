"""Describe discovered communities by overlaying ACS demographics.

For each community, compute population-weighted means of the ACS features
plus total population, land area, and turnout by election type.

Two modes:
  - Tract mode (original): joins on ``tract_geoid`` using tract-level features.
  - County mode (new):    joins on ``county_fips`` using county_acs_features.parquet
                          + county_rcms_features.parquet, then writes
                          data/communities/community_profiles.parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# The 12 ACS feature names from build_features.py (tract-level, legacy)
DEMOGRAPHIC_COLS = [
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

# County-level ACS feature columns (from build_county_acs_features.py)
COUNTY_ACS_COLS = [
    "pct_white_nh",
    "pct_black",
    "pct_asian",
    "pct_hispanic",
    "median_age",
    "median_hh_income",
    "pct_bachelors_plus",
    "pct_owner_occupied",
    "pct_wfh",
    "pct_management",
]

# County-level RCMS feature columns (from fetch_rcms.py / build_features.py)
COUNTY_RCMS_COLS = [
    "evangelical_share",
    "mainline_share",
    "catholic_share",
    "black_protestant_share",
    "congregations_per_1000",
    "religious_adherence_rate",
]

_ROOT = Path(__file__).resolve().parents[2]
# Input/output paths (tract-level, used by original main())
ASSIGNMENTS_PATH = _ROOT / "data" / "communities" / "community_assignments.parquet"
FEATURES_PATH = _ROOT / "data" / "assembled" / "tract_features.parquet"
SHIFTS_PATH = _ROOT / "data" / "shifts" / "tract_shifts.parquet"
OUTPUT_PATH = _ROOT / "data" / "communities" / "community_profiles.parquet"

# County-mode paths (used by build_county_community_profiles / county_main())
COUNTY_ASSIGNMENTS_PATH = _ROOT / "data" / "communities" / "county_community_assignments.parquet"
COUNTY_ACS_PATH = _ROOT / "data" / "assembled" / "county_acs_features.parquet"
COUNTY_RCMS_PATH = _ROOT / "data" / "assembled" / "county_rcms_features.parquet"
COUNTY_SHIFTS_PATH = _ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
COUNTY_OUTPUT_PATH = _ROOT / "data" / "communities" / "community_profiles.parquet"


def build_community_profiles(
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    shifts: pd.DataFrame,
) -> pd.DataFrame:
    """Join assignments with features and shifts, then aggregate per community.

    Parameters
    ----------
    assignments:
        DataFrame with columns ``tract_geoid`` and ``community_id``.
    features:
        DataFrame with ``tract_geoid``, ``pop_total``, and the columns in
        ``DEMOGRAPHIC_COLS``.
    shifts:
        DataFrame with ``tract_geoid`` and one or more shift columns.

    Returns
    -------
    DataFrame with one row per community_id.  Columns include ``community_id``,
    ``n_tracts``, ``pop_total`` (sum), all ``DEMOGRAPHIC_COLS`` (population-
    weighted means), and mean of each shift column.
    """
    # Identify shift columns (everything except the key)
    shift_cols = [c for c in shifts.columns if c != "tract_geoid"]

    # Normalise assignments column name (pipeline writes "community", not "community_id")
    if "community" in assignments.columns and "community_id" not in assignments.columns:
        assignments = assignments.rename(columns={"community": "community_id"})

    # pop_total may be absent from the assembled features; fall back to unweighted means
    feat_cols = ["tract_geoid"] + DEMOGRAPHIC_COLS
    if "pop_total" in features.columns:
        feat_cols = ["tract_geoid", "pop_total"] + DEMOGRAPHIC_COLS

    # Merge all tables on tract_geoid
    merged = (
        assignments
        .merge(features[feat_cols], on="tract_geoid", how="left")
        .merge(shifts[["tract_geoid"] + shift_cols], on="tract_geoid", how="left")
    )
    if "pop_total" not in merged.columns:
        merged["pop_total"] = 1.0  # unweighted: treat each tract equally

    records = []
    for community_id, group in merged.groupby("community_id"):
        pop = group["pop_total"].fillna(0)
        total_pop = pop.sum()

        row: dict = {
            "community_id": community_id,
            "n_tracts": len(group),
            "pop_total": total_pop,
        }

        # Population-weighted means for demographic columns
        for col in DEMOGRAPHIC_COLS:
            if total_pop > 0:
                row[col] = (group[col] * pop).sum() / total_pop
            else:
                row[col] = group[col].mean()

        # Simple means for shift columns
        for col in shift_cols:
            row[col] = group[col].mean()

        records.append(row)

    profiles = pd.DataFrame(records)
    # Ensure consistent column ordering
    col_order = ["community_id", "n_tracts", "pop_total"] + DEMOGRAPHIC_COLS + shift_cols
    profiles = profiles[[c for c in col_order if c in profiles.columns]]
    return profiles.reset_index(drop=True)


def build_county_community_profiles(
    assignments: pd.DataFrame,
    acs_features: pd.DataFrame,
    rcms_features: pd.DataFrame,
    shifts: pd.DataFrame,
) -> pd.DataFrame:
    """Build community profiles from county-level data.

    Parameters
    ----------
    assignments:
        DataFrame with ``county_fips`` and ``community_id``.
    acs_features:
        DataFrame from build_county_acs_features.py — county_fips, pop_total,
        and COUNTY_ACS_COLS.
    rcms_features:
        DataFrame from fetch_rcms.py — county_fips and COUNTY_RCMS_COLS.
    shifts:
        DataFrame with ``county_fips`` and shift columns.

    Returns
    -------
    DataFrame with one row per community_id containing population-weighted
    demographic means and simple-mean shift profiles.
    """
    # Normalise community_id column name
    if "community" in assignments.columns and "community_id" not in assignments.columns:
        assignments = assignments.rename(columns={"community": "community_id"})

    # Identify available ACS and RCMS cols (only include what exists in the frame)
    acs_cols = [c for c in COUNTY_ACS_COLS if c in acs_features.columns]
    rcms_cols = [c for c in COUNTY_RCMS_COLS if c in rcms_features.columns]
    shift_cols = [c for c in shifts.columns if c != "county_fips"]

    # Merge everything on county_fips
    merged = assignments[["county_fips", "community_id"]].copy()
    merged["county_fips"] = merged["county_fips"].astype(str).str.zfill(5)

    acs_sel = acs_features[["county_fips", "pop_total"] + acs_cols].copy()
    acs_sel["county_fips"] = acs_sel["county_fips"].astype(str).str.zfill(5)
    merged = merged.merge(acs_sel, on="county_fips", how="left")

    if rcms_cols:
        rcms_sel = rcms_features[["county_fips"] + rcms_cols].copy()
        rcms_sel["county_fips"] = rcms_sel["county_fips"].astype(str).str.zfill(5)
        merged = merged.merge(rcms_sel, on="county_fips", how="left")

    if shift_cols:
        shift_sel = shifts[["county_fips"] + shift_cols].copy()
        shift_sel["county_fips"] = shift_sel["county_fips"].astype(str).str.zfill(5)
        merged = merged.merge(shift_sel, on="county_fips", how="left")

    all_demo_cols = acs_cols + rcms_cols

    records = []
    for community_id, group in merged.groupby("community_id"):
        pop = group["pop_total"].fillna(0)
        total_pop = pop.sum()

        row: dict = {
            "community_id": int(community_id),
            "n_counties": len(group),
            "pop_total": float(total_pop),
        }

        # Population-weighted means for demographic columns
        for col in all_demo_cols:
            if col not in group.columns:
                continue
            if total_pop > 0:
                row[col] = float((group[col] * pop).sum() / total_pop)
            else:
                row[col] = float(group[col].mean())

        # Simple means for shift columns
        for col in shift_cols:
            if col in group.columns:
                row[col] = float(group[col].mean())

        records.append(row)

    profiles = pd.DataFrame(records)
    col_order = (
        ["community_id", "n_counties", "pop_total"]
        + all_demo_cols
        + shift_cols
    )
    profiles = profiles[[c for c in col_order if c in profiles.columns]]
    return profiles.reset_index(drop=True)


def main() -> None:
    """Load data, build profiles, save to parquet (tract-level legacy mode)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("Loading community assignments from %s", ASSIGNMENTS_PATH)
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)

    log.info("Loading tract features from %s", FEATURES_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    log.info("Loading tract shifts from %s", SHIFTS_PATH)
    shifts = pd.read_parquet(SHIFTS_PATH)

    profiles = build_community_profiles(assignments, features, shifts)
    log.info("Built %d community profiles", len(profiles))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s", OUTPUT_PATH)


def county_main() -> None:
    """Build community profiles from county-level ACS + RCMS + shift data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("Loading county community assignments from %s", COUNTY_ASSIGNMENTS_PATH)
    assignments = pd.read_parquet(COUNTY_ASSIGNMENTS_PATH)

    log.info("Loading county ACS features from %s", COUNTY_ACS_PATH)
    acs_features = pd.read_parquet(COUNTY_ACS_PATH)

    log.info("Loading county RCMS features from %s", COUNTY_RCMS_PATH)
    rcms_features = pd.read_parquet(COUNTY_RCMS_PATH)

    log.info("Loading county shifts from %s", COUNTY_SHIFTS_PATH)
    shifts = pd.read_parquet(COUNTY_SHIFTS_PATH)

    profiles = build_county_community_profiles(assignments, acs_features, rcms_features, shifts)
    log.info(
        "Built %d county community profiles with %d columns",
        len(profiles),
        len(profiles.columns),
    )

    COUNTY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(COUNTY_OUTPUT_PATH, index=False)
    log.info("Saved -> %s", COUNTY_OUTPUT_PATH)


if __name__ == "__main__":
    county_main()
