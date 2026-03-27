"""
Build district-level type composition vectors from county type assignments.

For each congressional district, computes the type membership vector by
aggregating county memberships weighted by the county-district overlap fraction
from the crosswalk.

The key equation:
  W_district[j] = sum_i (overlap_fraction[i,d] * W_county[i,j])

where i indexes counties, d indexes districts, and j indexes types.

This enables the model to treat districts as linear combinations of types,
just like it does for counties -- but districts are defined by the crosswalk
mapping, not by direct type assignment.

Inputs:
  data/districts/county_district_crosswalk.parquet (from build_district_crosswalk)
  data/communities/type_assignments.parquet (county-level type scores)

Output: data/districts/district_type_compositions.parquet
  - district_id: str, format "SS-DD"
  - type_0 through type_99: float, soft membership scores (sum to 1.0 per district)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
DISTRICTS_DIR = PROJECT_ROOT / "data" / "districts"
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"

# Number of KMeans types in the county model
N_TYPES = 100


def load_crosswalk(path: Path | None = None) -> pd.DataFrame:
    """Load the county-district crosswalk."""
    if path is None:
        path = DISTRICTS_DIR / "county_district_crosswalk.parquet"
    df = pd.read_parquet(path)
    log.info("Loaded crosswalk: %d rows, %d districts", len(df), df["district_id"].nunique())
    return df


def load_county_types(path: Path | None = None) -> pd.DataFrame:
    """Load county-level type assignments (soft membership scores).

    Returns DataFrame with county_fips and type_0..type_99 columns.
    Type score columns are renamed from type_N_score to type_N for cleanliness.
    """
    if path is None:
        path = COMMUNITIES_DIR / "type_assignments.parquet"
    df = pd.read_parquet(path)

    # Rename type_N_score → type_N
    score_cols = [c for c in df.columns if c.endswith("_score")]
    rename_map = {c: c.replace("_score", "") for c in score_cols}
    df = df.rename(columns=rename_map)

    type_cols = [f"type_{j}" for j in range(N_TYPES)]
    available_cols = [c for c in type_cols if c in df.columns]
    if len(available_cols) < N_TYPES:
        log.warning(
            "Expected %d type columns, found %d", N_TYPES, len(available_cols)
        )

    log.info("Loaded county types: %d counties, %d types", len(df), len(available_cols))
    return df[["county_fips"] + available_cols]


def build_district_types(
    crosswalk: pd.DataFrame,
    county_types: pd.DataFrame,
) -> pd.DataFrame:
    """Compute district-level type compositions from county types and crosswalk.

    Args:
        crosswalk: DataFrame with county_fips, district_id, overlap_fraction.
        county_types: DataFrame with county_fips and type_0..type_99 columns.

    Returns:
        DataFrame with district_id and type_0..type_99 columns.
        Type scores are renormalized to sum to 1.0 per district.
    """
    type_cols = [c for c in county_types.columns if c.startswith("type_")]

    # Ensure county_fips types match for join
    crosswalk = crosswalk.copy()
    county_types = county_types.copy()
    crosswalk["county_fips"] = crosswalk["county_fips"].astype(str).str.zfill(5)
    county_types["county_fips"] = county_types["county_fips"].astype(str).str.zfill(5)

    # Join crosswalk with county types
    merged = crosswalk.merge(county_types, on="county_fips", how="inner")

    n_missing = crosswalk["county_fips"].nunique() - merged["county_fips"].nunique()
    if n_missing > 0:
        missing_fips = set(crosswalk["county_fips"]) - set(merged["county_fips"])
        log.warning(
            "%d counties in crosswalk have no type assignments (e.g., %s)",
            n_missing,
            list(missing_fips)[:5],
        )

    # Weight type scores by overlap fraction
    for col in type_cols:
        merged[col] = merged[col] * merged["overlap_fraction"]

    # Aggregate to district level
    district_types = merged.groupby("district_id")[type_cols].sum().reset_index()

    # Renormalize so type scores sum to 1.0 per district
    row_sums = district_types[type_cols].sum(axis=1)
    for col in type_cols:
        district_types[col] = district_types[col] / row_sums.where(row_sums > 0, other=1)

    # Validate
    final_sums = district_types[type_cols].sum(axis=1)
    bad_rows = ~final_sums.between(0.999, 1.001)
    if bad_rows.any():
        log.warning(
            "%d districts have type scores not summing to 1.0",
            bad_rows.sum(),
        )

    log.info(
        "District type compositions: %d districts, %d types",
        len(district_types),
        len(type_cols),
    )

    return district_types


def main() -> None:
    """Build and save district-level type compositions."""
    DISTRICTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DISTRICTS_DIR / "district_type_compositions.parquet"

    crosswalk = load_crosswalk()
    county_types = load_county_types()
    district_types = build_district_types(crosswalk, county_types)

    district_types.to_parquet(output_path, index=False)
    log.info("Saved district types to %s (%d rows)", output_path, len(district_types))


if __name__ == "__main__":
    main()
