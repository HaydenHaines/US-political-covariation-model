"""Describe discovered electoral types by overlaying county demographics.

For each type, compute score-weighted means of county demographics within
that type. Each county is weighted by its absolute score on that type
(soft membership strength).

Two inputs:
  - type_assignments: county_fips, type_0_score..type_{J-1}_score, dominant_type
  - demographics:     county_fips, year, pct_white_nh, ... (long format)

Optional:
  - rcms_features:    county_fips, evangelical_share, ... (wide format)

Output (type_profiles.parquet):
  type_id, n_counties, pop_total, <all demographic means>, [<rcms means>]

Usage:
    python -m src.description.describe_types
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# RCMS feature columns produced by fetch_rcms.py / build_features.py
RCMS_COLS = [
    "evangelical_share",
    "mainline_share",
    "catholic_share",
    "black_protestant_share",
    "congregations_per_1000",
    "religious_adherence_rate",
]

_ROOT = Path(__file__).resolve().parents[2]

# Default paths for CLI
TYPE_ASSIGNMENTS_PATH = _ROOT / "data" / "communities" / "type_assignments.parquet"
DEMOGRAPHICS_INTERP_PATH = _ROOT / "data" / "assembled" / "demographics_interpolated.parquet"
DEMOGRAPHICS_ACS_PATH = _ROOT / "data" / "assembled" / "county_acs_features.parquet"
RCMS_PATH = _ROOT / "data" / "assembled" / "county_rcms_features.parquet"
OUTPUT_PATH = _ROOT / "data" / "communities" / "type_profiles.parquet"


def describe_types(
    type_assignments: pd.DataFrame,
    demographics: pd.DataFrame,
    election_year: int | None = None,
    rcms_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build demographic profiles for each electoral type.

    Parameters
    ----------
    type_assignments:
        DataFrame with county_fips, type_0_score..type_{J-1}_score columns,
        and dominant_type (integer index of the highest-abs-score type).
    demographics:
        Long-format DataFrame with county_fips, year, and numeric demographic
        columns (e.g. pct_white_nh, median_age, ...).  Produced by
        interpolate_demographics.py or build_county_acs_features.py.
    election_year:
        If provided, filter demographics to this year's rows.
        If None, use the latest available year.
    rcms_features:
        Optional wide-format DataFrame with county_fips and RCMS columns
        (evangelical_share, etc.).  If None, RCMS columns are omitted from
        the output.

    Returns
    -------
    DataFrame with one row per type (type_id 0..J-1).  Columns:
        type_id, n_counties, pop_total, <demographic means>, [<rcms means>]
    """
    # Identify score columns (type_0_score, type_1_score, ...)
    score_cols = sorted(
        [c for c in type_assignments.columns if c.endswith("_score") and c.startswith("type_")],
        key=lambda c: int(c.split("_")[1]),
    )
    j = len(score_cols)

    # Resolve election year for demographics
    if "year" in demographics.columns:
        if election_year is not None:
            demo_year = demographics[demographics["year"] == election_year].copy()
        else:
            latest_year = demographics["year"].max()
            demo_year = demographics[demographics["year"] == latest_year].copy()
    else:
        # No year column — use as-is (e.g. ACS features, single snapshot)
        demo_year = demographics.copy()

    # Normalise county_fips to 5-digit string for join safety
    ta = type_assignments.copy()
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)
    demo_year["county_fips"] = demo_year["county_fips"].astype(str).str.zfill(5)

    # Demographic columns: everything except county_fips and year
    skip = {"county_fips", "year"}
    demo_cols = [c for c in demo_year.columns if c not in skip]

    # Merge type assignments with demographics
    merged = ta.merge(demo_year[["county_fips"] + demo_cols], on="county_fips", how="left")

    # Merge RCMS if provided
    rcms_cols_used: list[str] = []
    if rcms_features is not None:
        available_rcms = [c for c in RCMS_COLS if c in rcms_features.columns]
        if available_rcms:
            rcms = rcms_features[["county_fips"] + available_rcms].copy()
            rcms["county_fips"] = rcms["county_fips"].astype(str).str.zfill(5)
            merged = merged.merge(rcms, on="county_fips", how="left")
            rcms_cols_used = available_rcms

    all_feature_cols = demo_cols + rcms_cols_used

    records = []
    for type_idx in range(j):
        score_col = score_cols[type_idx]

        # Weight each county by its absolute score on this type
        weights = merged[score_col].abs().fillna(0.0)
        total_weight = weights.sum()

        # n_counties: count where dominant_type == type_idx
        n_counties = int((merged["dominant_type"] == type_idx).sum())

        # pop_total: sum of pop_total weighted by score (or raw pop sum if no pop col)
        if "pop_total" in merged.columns:
            pop = merged["pop_total"].fillna(0.0)
            if total_weight > 0:
                total_pop = float((pop * weights).sum() / total_weight)
            else:
                total_pop = float(pop.sum())
        else:
            total_pop = float(n_counties)

        row: dict = {
            "type_id": int(type_idx),
            "n_counties": n_counties,
            "pop_total": total_pop,
        }

        # Score-weighted mean of every feature column
        for col in all_feature_cols:
            if col not in merged.columns or col == "pop_total":
                continue
            col_vals = merged[col].fillna(0.0)
            if total_weight > 0:
                row[col] = float((col_vals * weights).sum() / total_weight)
            else:
                row[col] = float(col_vals.mean())

        records.append(row)

    profiles = pd.DataFrame(records)

    # Consistent column ordering
    col_order_raw = ["type_id", "n_counties", "pop_total"] + demo_cols + rcms_cols_used
    # Deduplicate while preserving order
    seen: set[str] = set()
    col_order: list[str] = []
    for c in col_order_raw:
        if c not in seen and c in profiles.columns:
            col_order.append(c)
            seen.add(c)
    profiles = profiles[col_order]
    return profiles.reset_index(drop=True)


def main() -> None:
    """Load type assignments + demographics, build type profiles, save parquet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("Loading type assignments from %s", TYPE_ASSIGNMENTS_PATH)
    type_assignments = pd.read_parquet(TYPE_ASSIGNMENTS_PATH)

    # Prefer interpolated demographics; fall back to ACS features
    if DEMOGRAPHICS_INTERP_PATH.exists():
        log.info("Loading interpolated demographics from %s", DEMOGRAPHICS_INTERP_PATH)
        demographics = pd.read_parquet(DEMOGRAPHICS_INTERP_PATH)
    elif DEMOGRAPHICS_ACS_PATH.exists():
        log.info(
            "Interpolated demographics not found; falling back to %s", DEMOGRAPHICS_ACS_PATH
        )
        demographics = pd.read_parquet(DEMOGRAPHICS_ACS_PATH)
    else:
        raise FileNotFoundError(
            f"No demographics file found. Tried:\n  {DEMOGRAPHICS_INTERP_PATH}\n  {DEMOGRAPHICS_ACS_PATH}"
        )

    # Optional RCMS
    rcms_features: pd.DataFrame | None = None
    if RCMS_PATH.exists():
        log.info("Loading RCMS features from %s", RCMS_PATH)
        rcms_features = pd.read_parquet(RCMS_PATH)
    else:
        log.info("RCMS features not found at %s — skipping", RCMS_PATH)

    profiles = describe_types(type_assignments, demographics, rcms_features=rcms_features)
    log.info(
        "Built %d type profiles with %d columns", len(profiles), len(profiles.columns)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
