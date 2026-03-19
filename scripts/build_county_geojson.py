# scripts/build_county_geojson.py
"""Download Census TIGER/Line county shapefiles and generate FL+GA+AL GeoJSON.

Output: web/public/counties-fl-ga-al.geojson
Properties per feature: county_fips, county_name (if crosswalk present)
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
TARGET_STATES = {"01", "12", "13"}  # AL, FL, GA
OUT_PATH = PROJECT_ROOT / "web" / "public" / "counties-fl-ga-al.geojson"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"


def main() -> None:
    print(f"Downloading TIGER/Line county shapefile...")
    gdf = gpd.read_file(TIGER_URL)
    print(f"Downloaded {len(gdf)} counties (all US)")

    # Filter to FL/GA/AL
    gdf = gdf[gdf["STATEFP"].isin(TARGET_STATES)].copy()
    print(f"Filtered to {len(gdf)} counties (FL+GA+AL)")

    # Build county_fips
    gdf["county_fips"] = gdf["STATEFP"] + gdf["COUNTYFP"]

    # Simplify geometry (tolerance ~0.001 degrees ≈ ~100m at these latitudes)
    gdf["geometry"] = gdf["geometry"].simplify(0.001, preserve_topology=True)

    # Join county names from crosswalk if available
    keep_cols = ["county_fips", "geometry"]
    if CROSSWALK_PATH.exists():
        xwalk = pd.read_csv(CROSSWALK_PATH, dtype=str)
        xwalk["county_fips"] = xwalk["county_fips"].str.zfill(5)
        gdf = gdf.merge(xwalk, on="county_fips", how="left")
        missing = gdf["county_name"].isna().sum()
        assert missing == 0, f"{missing} counties missing county_name after crosswalk merge"
        keep_cols = ["county_fips", "county_name", "geometry"]

    gdf = gdf[keep_cols].set_geometry("geometry")

    # Reproject to WGS84 (EPSG:4326) — required for Deck.gl
    gdf = gdf.to_crs(epsg=4326)

    # Write GeoJSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT_PATH, driver="GeoJSON")
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Saved {len(gdf)} counties to {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
