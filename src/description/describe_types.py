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
URBANICITY_PATH = _ROOT / "data" / "assembled" / "county_urbanicity_features.parquet"
MIGRATION_PATH = _ROOT / "data" / "assembled" / "county_migration_features.parquet"
BEA_PATH = _ROOT / "data" / "assembled" / "bea_county_income.parquet"
FEC_PATH = _ROOT / "data" / "assembled" / "fec_county_contributions.parquet"
OUTPUT_PATH = _ROOT / "data" / "communities" / "type_profiles.parquet"


def _filter_demographics_to_year(
    demographics: pd.DataFrame,
    election_year: int | None,
) -> pd.DataFrame:
    """Return the demographics slice for the requested year.

    If ``election_year`` is None, uses the latest year in the DataFrame.
    If there is no ``year`` column (e.g. a single-snapshot ACS file), returns
    the entire DataFrame unchanged.
    """
    if "year" not in demographics.columns:
        return demographics.copy()
    if election_year is not None:
        return demographics[demographics["year"] == election_year].copy()
    return demographics[demographics["year"] == demographics["year"].max()].copy()


def _merge_feature_sources(
    merged: pd.DataFrame,
    rcms_features: pd.DataFrame | None,
    extra_features: list[pd.DataFrame] | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge RCMS and extra feature DataFrames into ``merged``.

    Returns the updated DataFrame, plus lists of RCMS and extra column names
    that were actually added (for downstream column ordering).
    """
    rcms_cols_used: list[str] = []
    if rcms_features is not None:
        available_rcms = [c for c in RCMS_COLS if c in rcms_features.columns]
        if available_rcms:
            rcms = rcms_features[["county_fips"] + available_rcms].copy()
            rcms["county_fips"] = rcms["county_fips"].astype(str).str.zfill(5)
            merged = merged.merge(rcms, on="county_fips", how="left")
            rcms_cols_used = available_rcms

    extra_cols_used: list[str] = []
    if extra_features:
        existing_cols = set(merged.columns)
        for extra_df in extra_features:
            extra_df = extra_df.copy()
            extra_df["county_fips"] = extra_df["county_fips"].astype(str).str.zfill(5)
            # Only add columns not already present (skip county_fips and duplicates)
            new_cols = [
                c for c in extra_df.columns
                if c != "county_fips" and c not in existing_cols
            ]
            if new_cols:
                merged = merged.merge(
                    extra_df[["county_fips"] + new_cols], on="county_fips", how="left"
                )
                extra_cols_used.extend(new_cols)
                existing_cols.update(new_cols)

    return merged, rcms_cols_used, extra_cols_used


def _weighted_feature_row(
    type_idx: int,
    score_col: str,
    merged: pd.DataFrame,
    all_feature_cols: list[str],
) -> dict:
    """Compute one type's score-weighted demographic profile row.

    Weights = abs(score) for each county.  NaN values in individual feature
    columns are excluded from both numerator and denominator so sparse features
    (e.g. Alaska Census Areas missing ACS data) are not dragged toward zero.
    """
    weights = merged[score_col].abs().fillna(0.0)
    total_weight = weights.sum()
    n_counties = int((merged["dominant_type"] == type_idx).sum())

    # Population: score-weighted mean of pop_total (or raw county count if absent)
    if "pop_total" in merged.columns:
        pop = merged["pop_total"].fillna(0.0)
        total_pop = float((pop * weights).sum() / total_weight) if total_weight > 0 else float(pop.sum())
    else:
        total_pop = float(n_counties)

    row: dict = {
        "type_id": int(type_idx),
        "n_counties": n_counties,
        "pop_total": total_pop,
    }

    for col in all_feature_cols:
        if col not in merged.columns or col == "pop_total":
            continue
        mask = merged[col].notna()
        if mask.sum() == 0:
            row[col] = float("nan")
            continue
        col_vals = merged.loc[mask, col]
        w = weights[mask]
        w_sum = w.sum()
        row[col] = float((col_vals * w).sum() / w_sum) if w_sum > 0 else float(col_vals.mean())

    return row


def describe_types(
    type_assignments: pd.DataFrame,
    demographics: pd.DataFrame,
    election_year: int | None = None,
    rcms_features: pd.DataFrame | None = None,
    extra_features: list[pd.DataFrame] | None = None,
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
    extra_features:
        Optional list of wide-format DataFrames, each with county_fips and
        additional numeric columns.  Merged left on county_fips.  Columns
        already present in demographics or RCMS are skipped to avoid
        duplication.

    Returns
    -------
    DataFrame with one row per type (type_id 0..J-1).  Columns:
        type_id, n_counties, pop_total, <demographic means>, [<rcms means>],
        [<extra feature means>]
    """
    # Identify score columns (type_0_score, type_1_score, ...)
    score_cols = sorted(
        [c for c in type_assignments.columns if c.endswith("_score") and c.startswith("type_")],
        key=lambda c: int(c.split("_")[1]),
    )
    j = len(score_cols)

    # Slice demographics to the requested year
    demo_year = _filter_demographics_to_year(demographics, election_year)

    # Normalise county_fips to 5-digit string for join safety
    ta = type_assignments.copy()
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)
    demo_year["county_fips"] = demo_year["county_fips"].astype(str).str.zfill(5)

    # Demographic columns: everything except county_fips and year
    demo_cols = [c for c in demo_year.columns if c not in {"county_fips", "year"}]

    # Merge type assignments with demographics, then RCMS and extras
    merged = ta.merge(demo_year[["county_fips"] + demo_cols], on="county_fips", how="left")
    merged, rcms_cols_used, extra_cols_used = _merge_feature_sources(
        merged, rcms_features, extra_features
    )

    all_feature_cols = demo_cols + rcms_cols_used + extra_cols_used

    records = [
        _weighted_feature_row(type_idx, score_cols[type_idx], merged, all_feature_cols)
        for type_idx in range(j)
    ]
    profiles = pd.DataFrame(records)

    # Consistent column ordering, deduplicated
    col_order_raw = ["type_id", "n_counties", "pop_total"] + demo_cols + rcms_cols_used + extra_cols_used
    seen: set[str] = set()
    col_order: list[str] = []
    for c in col_order_raw:
        if c not in seen and c in profiles.columns:
            col_order.append(c)
            seen.add(c)
    return profiles[col_order].reset_index(drop=True)


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

    # Collect extra feature sources (each optional — loaded if file exists)
    extra_features: list[pd.DataFrame] = []
    extra_paths = {
        "ACS extras": DEMOGRAPHICS_ACS_PATH,
        "urbanicity": URBANICITY_PATH,
        "migration": MIGRATION_PATH,
        "BEA income": BEA_PATH,
        "FEC contributions": FEC_PATH,
    }
    for label, path in extra_paths.items():
        if path.exists():
            log.info("Loading %s features from %s", label, path)
            extra_features.append(pd.read_parquet(path))
        else:
            log.info("%s features not found at %s — skipping", label, path)

    profiles = describe_types(
        type_assignments,
        demographics,
        rcms_features=rcms_features,
        extra_features=extra_features if extra_features else None,
    )
    log.info(
        "Built %d type profiles with %d columns", len(profiles), len(profiles.columns)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
