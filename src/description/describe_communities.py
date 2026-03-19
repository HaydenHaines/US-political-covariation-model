"""Describe discovered communities by overlaying ACS demographics.

For each community, compute population-weighted means of the 12 ACS features
plus total population, land area, and turnout by election type.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# The 12 ACS feature names from build_features.py
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

_ROOT = Path(__file__).resolve().parents[2]
# Input/output paths
ASSIGNMENTS_PATH = _ROOT / "data" / "communities" / "community_assignments.parquet"
FEATURES_PATH = _ROOT / "data" / "assembled" / "tract_features.parquet"
SHIFTS_PATH = _ROOT / "data" / "shifts" / "tract_shifts.parquet"
OUTPUT_PATH = _ROOT / "data" / "communities" / "community_profiles.parquet"


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


def main() -> None:
    """Load data, build profiles, save to parquet."""
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
    log.info("Saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
