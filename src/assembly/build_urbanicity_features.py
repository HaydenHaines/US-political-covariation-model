"""Build urbanicity features for FL, GA, AL counties.

Sources:
  - Census Bureau 2020 Gazetteer file (county land area in sq meters)
    https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.txt
  - data/assembled/county_acs_features.parquet (pop_total per county)

Features produced:
    log_pop_density   log10(pop_total / land_area_sq_mi) — key urbanicity signal
    land_area_sq_mi   county land area (ALAND / 2589988.11)
    pop_per_sq_mi     raw population density (pop_total / land_area_sq_mi)

Output: data/assembled/county_urbanicity_features.parquet
Columns: county_fips, log_pop_density, land_area_sq_mi, pop_per_sq_mi
"""

from __future__ import annotations

import io
import logging
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
GAZETTEER_CACHE = PROJECT_ROOT / "data" / "raw" / "census_gazetteer_counties.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_urbanicity_features.parquet"

GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2020_Gazetteer/2020_Gaz_counties_national.zip"
)
GAZETTEER_ZIP_ENTRY = "2020_Gaz_counties_national.txt"

# Sq meters → sq miles conversion factor
SQ_M_PER_SQ_MI = 2_589_988.11

# FL=12, GA=13, AL=01
TARGET_FIPS_PREFIXES = {"01", "12", "13"}


def _load_gazetteer() -> pd.DataFrame:
    """Return DataFrame with columns: county_fips (5-char str), aland_sq_m (float).

    Downloads the Census 2020 county gazetteer file once and caches it to
    data/raw/census_gazetteer_counties.txt.
    """
    if GAZETTEER_CACHE.exists():
        log.info("Using cached gazetteer: %s", GAZETTEER_CACHE)
        raw_text = GAZETTEER_CACHE.read_text(encoding="latin-1")
    else:
        log.info("Downloading Census 2020 county gazetteer ZIP from %s", GAZETTEER_URL)
        resp = requests.get(GAZETTEER_URL, timeout=120)
        resp.raise_for_status()
        # Extract the txt from the zip in-memory
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(GAZETTEER_ZIP_ENTRY) as fh:
                raw_text = fh.read().decode("latin-1")
        GAZETTEER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        GAZETTEER_CACHE.write_text(raw_text, encoding="latin-1")
        log.info("Cached gazetteer -> %s", GAZETTEER_CACHE)

    df = pd.read_csv(StringIO(raw_text), sep="\t", low_memory=False, dtype=str, encoding="latin-1")

    # Normalize column names: strip whitespace
    df.columns = df.columns.str.strip()

    # GEOID is the 5-digit county FIPS; ALAND is land area in sq meters
    df["county_fips"] = df["GEOID"].astype(str).str.zfill(5)
    df["aland_sq_m"] = pd.to_numeric(df["ALAND"], errors="coerce")

    log.info("Gazetteer loaded: %d counties nationwide", len(df))
    return df[["county_fips", "aland_sq_m"]]


def build_urbanicity_features(
    acs: pd.DataFrame,
    gazetteer: pd.DataFrame,
) -> pd.DataFrame:
    """Compute urbanicity features for FL/GA/AL counties.

    Parameters
    ----------
    acs:
        DataFrame from county_acs_features.parquet — must have county_fips + pop_total.
    gazetteer:
        DataFrame from _load_gazetteer() — must have county_fips + aland_sq_m.

    Returns
    -------
    DataFrame with county_fips, log_pop_density, land_area_sq_mi, pop_per_sq_mi.
    """
    # Filter to FL/GA/AL
    acs_fga = acs[acs["county_fips"].str[:2].isin(TARGET_FIPS_PREFIXES)].copy()
    gaz_fga = gazetteer[gazetteer["county_fips"].str[:2].isin(TARGET_FIPS_PREFIXES)].copy()

    merged = acs_fga[["county_fips", "pop_total"]].merge(
        gaz_fga[["county_fips", "aland_sq_m"]],
        on="county_fips",
        how="inner",
    )

    # Land area in square miles
    # Replace zero land area with NaN to avoid division by zero / log(-inf)
    aland_safe = merged["aland_sq_m"].replace(0, np.nan)
    merged["land_area_sq_mi"] = aland_safe / SQ_M_PER_SQ_MI

    # Raw population density
    merged["pop_per_sq_mi"] = merged["pop_total"] / merged["land_area_sq_mi"]

    # Log10 population density (safe: zero land area → NaN, not -inf)
    merged["log_pop_density"] = np.log10(merged["pop_per_sq_mi"].replace(0, np.nan))

    result = merged[
        ["county_fips", "log_pop_density", "land_area_sq_mi", "pop_per_sq_mi"]
    ].reset_index(drop=True)

    return result


def main() -> None:
    log.info("Loading ACS features from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)
    log.info("  %d counties in ACS file", len(acs))

    gazetteer = _load_gazetteer()

    features = build_urbanicity_features(acs, gazetteer)

    n_na = features.isnull().any(axis=1).sum()
    log.info(
        "Built %d county urbanicity rows x %d columns | %d counties with at least one NaN",
        len(features),
        len(features.columns),
        n_na,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s", OUTPUT_PATH)

    # Sanity check: most urban and most rural
    sorted_df = features.sort_values("log_pop_density", ascending=False)
    log.info("\n--- 5 most urban counties (by log_pop_density) ---")
    for _, row in sorted_df.head(5).iterrows():
        log.info(
            "  %s  log_density=%.3f  pop_per_sq_mi=%.1f  area=%.1f sq mi",
            row["county_fips"],
            row["log_pop_density"],
            row["pop_per_sq_mi"],
            row["land_area_sq_mi"],
        )

    log.info("\n--- 5 most rural counties (by log_pop_density) ---")
    for _, row in sorted_df.tail(5).iterrows():
        log.info(
            "  %s  log_density=%.3f  pop_per_sq_mi=%.1f  area=%.1f sq mi",
            row["county_fips"],
            row["log_pop_density"],
            row["pop_per_sq_mi"],
            row["land_area_sq_mi"],
        )


if __name__ == "__main__":
    main()
