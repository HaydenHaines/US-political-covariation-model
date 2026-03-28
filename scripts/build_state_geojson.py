# scripts/build_state_geojson.py
"""Generate state-level GeoJSON and per-state tract community polygon splits.

Outputs:
  1. web/public/states-us.geojson — 51 US state polygons with race metadata
  2. web/public/tracts/{STATE}.geojson — per-state tract community polygons

State polygons come from Census TIGER/Line cartographic boundaries (500k).
Tract splits are produced by spatial-joining the national tract community
polygon file against state boundaries using centroids.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIGER_STATE_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip"
)
TIGER_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "tiger_states"
TRACT_GEOJSON = PROJECT_ROOT / "web" / "public" / "tracts-us.geojson"
STATE_OUT = PROJECT_ROOT / "web" / "public" / "states-us.geojson"
TRACT_OUT_DIR = PROJECT_ROOT / "web" / "public" / "tracts"

# All 50 states + DC (FIPS codes)
VALID_STATE_FIPS = {
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56",
}

# 2026 election metadata
SENATE_2026_STATES = {
    "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS", "KY",
    "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE", "NH", "NJ",
    "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",
}

GOVERNOR_2026_STATES = {
    "AK", "AL", "AZ", "AR", "CA", "CO", "CT", "FL", "GA", "HI", "IA",
    "ID", "IL", "KS", "MD", "MA", "ME", "MI", "MN", "NE", "NV", "NH",
    "NM", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "VT", "WI", "WY",
}

# Geometry simplification: states get heavier simplification (web perf),
# tract splits inherit their existing geometry unchanged.
STATE_SIMPLIFY_TOLERANCE = 0.01  # ~1km at mid-latitudes; keeps output <700KB


def _download_state_shapefile() -> gpd.GeoDataFrame:
    """Download and cache Census TIGER/Line state cartographic boundaries."""
    TIGER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TIGER_CACHE_DIR / "cb_2020_us_state_500k.zip"
    shp_dir = TIGER_CACHE_DIR / "cb_2020_us_state_500k"

    if not shp_dir.exists():
        if not zip_path.exists():
            print(f"Downloading state shapefile from Census...")
            urlretrieve(TIGER_STATE_URL, zip_path)
            print(f"Saved to {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(shp_dir)
        print(f"Extracted to {shp_dir}")
    else:
        print(f"Using cached state shapefile at {shp_dir}")

    return gpd.read_file(shp_dir)


def build_state_geojson() -> gpd.GeoDataFrame:
    """Part A: Build state-level GeoJSON with race metadata."""
    gdf = _download_state_shapefile()

    # Filter to 50 states + DC
    gdf = gdf[gdf["STATEFP"].isin(VALID_STATE_FIPS)].copy()
    print(f"Filtered to {len(gdf)} states/DC")

    # Add properties
    gdf["state_fips"] = gdf["STATEFP"]
    gdf["state_abbr"] = gdf["STUSPS"]
    gdf["state_name"] = gdf["NAME"]
    gdf["has_senate_2026"] = gdf["state_abbr"].isin(SENATE_2026_STATES)
    gdf["has_governor_2026"] = gdf["state_abbr"].isin(GOVERNOR_2026_STATES)

    # Simplify geometry for web performance
    gdf["geometry"] = gdf["geometry"].simplify(
        STATE_SIMPLIFY_TOLERANCE, preserve_topology=True
    )

    # Keep only needed columns, reproject to WGS84
    keep = [
        "state_fips", "state_abbr", "state_name",
        "has_senate_2026", "has_governor_2026", "geometry",
    ]
    gdf = gdf[keep].set_geometry("geometry").to_crs(epsg=4326)

    STATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(STATE_OUT, driver="GeoJSON")
    size_kb = STATE_OUT.stat().st_size / 1024
    print(f"Saved {len(gdf)} states to {STATE_OUT} ({size_kb:.0f} KB)")

    return gdf


def build_tract_splits(states_gdf: gpd.GeoDataFrame) -> None:
    """Part B: Split national tract GeoJSON into per-state files."""
    if not TRACT_GEOJSON.exists():
        raise FileNotFoundError(f"National tract GeoJSON not found: {TRACT_GEOJSON}")

    print(f"Reading national tract GeoJSON...")
    tracts = gpd.read_file(TRACT_GEOJSON)
    print(f"Loaded {len(tracts)} community polygons")

    # Compute centroids for spatial join (in same CRS)
    tracts = tracts.to_crs(epsg=4326)
    states_gdf = states_gdf.to_crs(epsg=4326)

    tracts_centroids = tracts.copy()
    tracts_centroids["geometry"] = tracts_centroids.geometry.representative_point()

    # Spatial join: assign each tract polygon to a state
    joined = gpd.sjoin(
        tracts_centroids,
        states_gdf[["state_abbr", "geometry"]],
        how="left",
        predicate="within",
    )

    # Handle any tracts that didn't land in a state (ocean borders, etc.)
    unassigned = joined["state_abbr"].isna().sum()
    if unassigned > 0:
        print(f"Warning: {unassigned} polygons not assigned to any state (dropping)")
        joined = joined.dropna(subset=["state_abbr"])

    # Restore original geometries (not centroids) for output
    joined["geometry"] = tracts.loc[joined.index, "geometry"]

    # Write per-state files
    TRACT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    original_cols = [c for c in tracts.columns if c != "geometry"] + ["geometry"]

    state_groups = joined.groupby("state_abbr")
    for state_abbr, group in state_groups:
        # Keep only the original tract properties + geometry
        out_cols = [c for c in original_cols if c in group.columns]
        state_gdf = group[out_cols].set_geometry("geometry")
        out_path = TRACT_OUT_DIR / f"{state_abbr}.geojson"
        state_gdf.to_file(out_path, driver="GeoJSON")
        size_kb = out_path.stat().st_size / 1024
        print(f"  {state_abbr}: {len(state_gdf)} polygons ({size_kb:.0f} KB)")

    print(f"Wrote {len(state_groups)} state tract files to {TRACT_OUT_DIR}")


def main() -> None:
    print("=== Part A: State GeoJSON ===")
    states_gdf = build_state_geojson()

    print("\n=== Part B: Per-State Tract Splits ===")
    build_tract_splits(states_gdf)


if __name__ == "__main__":
    main()
