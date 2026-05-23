"""Regenerate tract_type_assignments.parquet from current county type artifacts.

The full T.3 tract discovery pipeline writes tract-level soft type assignments,
but that pipeline depends on data/shifts/tract_shifts_national.parquet. Some
verification checkouts have the T.1 tract election data and the current J=100
county type model, but not that intermediate shift matrix. In that case this
script restores the API/test contract by projecting each tract to its county's
current type vector.

Inputs:
  - data/assembled/tract_elections.parquet
  - data/communities/county_type_assignments_full.parquet, preferred
    or data/communities/type_assignments.parquet

Output:
  - data/communities/tract_type_assignments.parquet

Usage:
    .venv/bin/python scripts/regenerate_tract_type_assignments.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"
TRACT_ELECTIONS_PATH = PROJECT_ROOT / "data" / "assembled" / "tract_elections.parquet"
COUNTY_ASSIGNMENTS_FULL_PATH = COMMUNITIES_DIR / "county_type_assignments_full.parquet"
COUNTY_ASSIGNMENTS_PATH = COMMUNITIES_DIR / "type_assignments.parquet"
OUTPUT_PATH = COMMUNITIES_DIR / "tract_type_assignments.parquet"


def _score_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("type_") and c.endswith("_score")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def load_county_assignments() -> pd.DataFrame:
    """Load the current county-level J=100 type assignments."""
    path = (
        COUNTY_ASSIGNMENTS_FULL_PATH
        if COUNTY_ASSIGNMENTS_FULL_PATH.exists()
        else COUNTY_ASSIGNMENTS_PATH
    )
    if not path.exists():
        raise FileNotFoundError(
            "No county type assignments found at "
            f"{COUNTY_ASSIGNMENTS_FULL_PATH} or {COUNTY_ASSIGNMENTS_PATH}"
        )

    assignments = pd.read_parquet(path)
    score_cols = _score_cols(assignments)
    if not score_cols:
        raise ValueError(f"No type score columns found in {path}")

    assignments = assignments.copy()
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
    if "dominant_type" not in assignments.columns:
        assignments["dominant_type"] = assignments[score_cols].values.argmax(axis=1)
    if "super_type" not in assignments.columns:
        assignments["super_type"] = 0

    return assignments[["county_fips", *score_cols, "dominant_type", "super_type"]]


def load_tract_counties() -> pd.DataFrame:
    """Load distinct tract GEOIDs and derive county FIPS from Census GEOID."""
    if not TRACT_ELECTIONS_PATH.exists():
        raise FileNotFoundError(f"Tract elections not found: {TRACT_ELECTIONS_PATH}")

    tracts = pd.read_parquet(TRACT_ELECTIONS_PATH, columns=["tract_geoid"])
    tracts = tracts.drop_duplicates(subset="tract_geoid").copy()
    tracts["tract_geoid"] = tracts["tract_geoid"].astype(str).str.zfill(11)
    tracts["county_fips"] = tracts["tract_geoid"].str[:5]
    return tracts


def regenerate() -> pd.DataFrame:
    """Build and write tract assignments aligned to the current county model."""
    county_assignments = load_county_assignments()
    tracts = load_tract_counties()

    merged = tracts.merge(county_assignments, on="county_fips", how="left")
    score_cols = _score_cols(merged)
    missing = merged["dominant_type"].isna()
    if missing.any():
        missing_counties = sorted(merged.loc[missing, "county_fips"].dropna().unique())
        preview = ", ".join(missing_counties[:10])
        print(
            f"WARNING: {missing.sum()} tracts lack county type assignments "
            f"({len(missing_counties)} counties; first: {preview}); using uniform type scores."
        )
        uniform_score = 1.0 / len(score_cols)
        merged.loc[missing, score_cols] = uniform_score
        merged.loc[missing, "dominant_type"] = 0
        merged.loc[missing, "super_type"] = 0

    output = merged[["tract_geoid", *score_cols, "dominant_type", "super_type"]].copy()
    output["dominant_type"] = output["dominant_type"].astype(int)
    output["super_type"] = output["super_type"].astype(int)

    COMMUNITIES_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    return output


def main() -> None:
    output = regenerate()
    score_cols = _score_cols(output)
    print(
        f"Wrote {len(output):,} tract assignments with {len(score_cols)} type scores "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
