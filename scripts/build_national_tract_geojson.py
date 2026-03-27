"""Build national dissolved tract GeoJSON for the frontend map.

Downloads the Census Bureau's TIGER 2020 cartographic boundary file (500k
simplification), joins it with national tract assignments, dissolves adjacent
same-type tracts into community polygons per state, and exports GeoJSON.

Usage:
    cd /home/hayden/projects/wethervane
    uv run python scripts/build_national_tract_geojson.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.viz.bubble_dissolve import bubble_dissolve  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────

TIGER_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_tract_500k.zip"
TIGER_DIR = PROJECT_ROOT / "data" / "raw" / "tiger"
TIGER_ZIP = TIGER_DIR / "cb_2020_us_tract_500k.zip"
TIGER_SHP_DIR = TIGER_DIR / "cb_2020_us_tract_500k"

ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "tracts" / "national_tract_assignments.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "tracts" / "tract_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "web" / "public" / "tracts-us.geojson"

# Demographic columns to embed as type-level averages in the GeoJSON
DEMO_COLS = [
    "median_hh_income",
    "pct_ba_plus",
    "pct_white_nh",
    "pct_black",
    "pct_hispanic",
    "evangelical_share",
]

SIMPLIFY_TOLERANCE = 1000  # metres in EPSG:5070
MIN_AREA_SQKM = 0.1

# States/territories to exclude (no electoral votes, limited data)
EXCLUDE_FIPS = {"60", "66", "69", "72", "78"}  # AS, GU, MP, PR, VI


def download_tiger() -> Path:
    """Download and extract the national tract cartographic boundary file."""
    TIGER_DIR.mkdir(parents=True, exist_ok=True)

    # Find the .shp file — could be in subdir or directly extracted
    shp_candidates = list(TIGER_DIR.glob("**/cb_2020_us_tract_500k.shp"))
    if shp_candidates:
        print(f"TIGER shapefile already exists: {shp_candidates[0]}")
        return shp_candidates[0]

    if not TIGER_ZIP.exists():
        print(f"Downloading TIGER tract boundaries (~100MB)...")
        print(f"  URL: {TIGER_URL}")
        urllib.request.urlretrieve(TIGER_URL, TIGER_ZIP)
        size_mb = TIGER_ZIP.stat().st_size / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.1f} MB")
    else:
        print(f"TIGER zip already exists: {TIGER_ZIP}")

    # Extract
    print("Extracting...")
    TIGER_SHP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TIGER_ZIP, "r") as zf:
        zf.extractall(TIGER_SHP_DIR)

    shp_candidates = list(TIGER_SHP_DIR.glob("*.shp"))
    if not shp_candidates:
        print("ERROR: No .shp file found after extraction", file=sys.stderr)
        sys.exit(1)

    print(f"  Extracted to: {shp_candidates[0]}")
    return shp_candidates[0]


def load_super_type_names() -> dict[int, str]:
    """Try to load tract super-type names from DuckDB, fall back to generic."""
    try:
        import duckdb

        db_path = PROJECT_ROOT / "data" / "wethervane.duckdb"
        if db_path.exists():
            db = duckdb.connect(str(db_path), read_only=True)
            # Check if super_types has a model column
            cols = db.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'super_types'"
            ).fetchdf()["column_name"].tolist()

            if "model" in cols:
                st_df = db.execute(
                    "SELECT super_type_id, display_name FROM super_types WHERE model = 'tract'"
                ).fetchdf()
                if len(st_df) > 0:
                    return dict(zip(st_df["super_type_id"], st_df["display_name"]))

            # Fall through to generic names
            db.close()
    except Exception:
        pass

    # Descriptive names for 8 tract super-types.
    # Derived from Ward HAC on ACS demographic profiles of 100 fine types.
    # Profile summary (ACS 2022, 81,129 tracts, S=8, silhouette=0.350):
    # ST 0: 65% Hispanic, 18% no-hs, low income ($63K), high Catholic, D+37%
    # ST 1: 62% Black, low income ($52K), high poverty, high Black-Protestant, D+64%
    # ST 2: 84% White, avg age 57 (retirement peak), 38% over-65, swing D+0%
    # ST 3: 82% White, heavy evangelical (43%), rural, working class, R+33%
    # ST 4: 54% White/16% Black/20% Hispanic, mixed suburb, moderate income, D+8%
    # ST 5: 48% Asian, highest income ($122K), 49% college, high Catholic, D+35%
    # ST 6: 76% White, $134K income, 57% college, high WFH, affluent suburb, D+3%
    # ST 7: 56% White/11% Black/18% Hispanic, $100K income, 56% college, D+41%
    return {
        0: "Hispanic Working Community",
        1: "Black Urban Neighborhood",
        2: "White Retirement Town",
        3: "Rural Evangelical Heartland",
        4: "Multiracial Outer Suburb",
        5: "Asian-American Professional",
        6: "Affluent White Suburb",
        7: "Urban Knowledge District",
    }


def compute_type_demographics(assignments: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Compute average demographic stats per type from tract features.

    Returns a dict mapping type_id -> {col: value} for embedding in GeoJSON.
    Only computes if the features file exists; returns empty dict otherwise.
    """
    if not FEATURES_PATH.exists():
        print(f"  WARNING: Features file not found at {FEATURES_PATH} — skipping demographics")
        return {}

    # Only load columns that actually exist in the file
    all_cols = pd.read_parquet(FEATURES_PATH, columns=["GEOID"]).columns.tolist()
    avail_cols = [c for c in DEMO_COLS if c in all_cols]
    features = pd.read_parquet(FEATURES_PATH, columns=["GEOID"] + avail_cols)
    print(f"  Loaded {len(features):,} rows from tract_features.parquet ({len(avail_cols)} demo cols)")

    # Join dominant_type onto features
    merged = features.merge(
        assignments[["GEOID", "dominant_type"]],
        on="GEOID",
        how="inner",
    )
    print(f"  Matched {len(merged):,} tracts for demographic averages")

    # Compute mean per type; round for compact JSON
    type_means = merged.groupby("dominant_type")[avail_cols].mean()

    def _round(val: float, ndigits: int) -> float | None:
        return round(float(val), ndigits) if pd.notna(val) else None

    result: dict[int, dict[str, float]] = {}
    for type_id, row in type_means.iterrows():
        record: dict[str, float | None] = {}
        if "median_hh_income" in row.index:
            record["median_hh_income"] = _round(row["median_hh_income"], 0)
        if "pct_ba_plus" in row.index:
            record["pct_ba_plus"] = _round(row["pct_ba_plus"], 3)
        if "pct_white_nh" in row.index:
            record["pct_white_nh"] = _round(row["pct_white_nh"], 3)
        if "pct_black" in row.index:
            record["pct_black"] = _round(row["pct_black"], 3)
        if "pct_hispanic" in row.index:
            record["pct_hispanic"] = _round(row["pct_hispanic"], 3)
        if "evangelical_share" in row.index:
            record["evangelical_share"] = _round(row["evangelical_share"], 1)
        result[int(type_id)] = record  # type: ignore[assignment]

    print(f"  Computed demographics for {len(result)} types")
    return result


def main() -> None:
    t0 = time.time()

    # 1. Download/locate TIGER data
    shp_path = download_tiger()

    # 2. Load national tract shapefile
    print("\nLoading national tract shapefile...")
    tracts_gdf = gpd.read_file(shp_path)
    print(f"  Loaded {len(tracts_gdf):,} tract geometries")

    # 3. Load assignments
    print("Loading tract assignments...")
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)
    print(f"  Loaded {len(assignments):,} tract assignments")

    # 4. Join
    # Ensure GEOID is string in both
    tracts_gdf["GEOID"] = tracts_gdf["GEOID"].astype(str)
    assignments["GEOID"] = assignments["GEOID"].astype(str)

    joined = tracts_gdf.merge(
        assignments[["GEOID", "dominant_type", "super_type"]],
        on="GEOID",
        how="inner",
    )
    print(f"  Matched {len(joined):,} / {len(assignments):,} assigned tracts "
          f"({len(joined) / len(assignments) * 100:.1f}%)")

    # 5. Filter out territories
    joined["state_fips"] = joined["GEOID"].str[:2]
    n_before = len(joined)
    joined = joined[~joined["state_fips"].isin(EXCLUDE_FIPS)]
    n_excluded = n_before - len(joined)
    if n_excluded > 0:
        print(f"  Excluded {n_excluded:,} territory tracts")

    # 6. Load super-type names
    st_names = load_super_type_names()
    print(f"  Super-type names: {st_names}")

    # 6b. Compute type-level demographic averages for tooltip enrichment
    print("\nComputing type-level demographic averages...")
    type_demo = compute_type_demographics(assignments)

    # 7. Dissolve by state
    states = sorted(joined["state_fips"].unique())
    print(f"\nDissolving {len(joined):,} tracts across {len(states)} states...")

    all_communities = []
    total_input_tracts = 0

    for i, fips in enumerate(states):
        state_gdf = joined[joined["state_fips"] == fips].copy()
        n_tracts = len(state_gdf)
        total_input_tracts += n_tracts

        t_state = time.time()
        try:
            result = bubble_dissolve(
                state_gdf[["geometry", "dominant_type", "super_type"]],
                min_area_sqkm=MIN_AREA_SQKM,
                simplify_tolerance=SIMPLIFY_TOLERANCE,
                super_type_names=st_names,
                type_demographics=type_demo,
            )
            elapsed = time.time() - t_state
            print(f"  [{i+1:2d}/{len(states)}] FIPS {fips}: "
                  f"{n_tracts:,} tracts -> {len(result):,} polygons "
                  f"({elapsed:.1f}s)")
            all_communities.append(result)
        except Exception as e:
            print(f"  [{i+1:2d}/{len(states)}] FIPS {fips}: "
                  f"ERROR with {n_tracts:,} tracts: {e}")
            # Try with larger min_area to reduce complexity
            try:
                result = bubble_dissolve(
                    state_gdf[["geometry", "dominant_type", "super_type"]],
                    min_area_sqkm=1.0,
                    simplify_tolerance=SIMPLIFY_TOLERANCE,
                    super_type_names=st_names,
                    type_demographics=type_demo,
                )
                elapsed = time.time() - t_state
                print(f"         Retry succeeded: {len(result):,} polygons "
                      f"({elapsed:.1f}s)")
                all_communities.append(result)
            except Exception as e2:
                print(f"         Retry also failed: {e2}", file=sys.stderr)

    # 8. Combine
    if not all_communities:
        print("ERROR: No communities produced!", file=sys.stderr)
        sys.exit(1)

    combined = gpd.pd.concat(all_communities, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    # 9. Export
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(OUTPUT_PATH, driver="GeoJSON")

    file_size = OUTPUT_PATH.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    elapsed_total = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Input tracts:       {total_input_tracts:,}")
    print(f"  Output polygons:    {len(combined):,}")
    print(f"  Compression ratio:  {total_input_tracts / len(combined):.1f}x")
    print(f"  Unique types:       {combined['type_id'].nunique()}")
    print(f"  Unique super-types: {combined['super_type'].nunique()}")
    print(f"  File size:          {file_size_mb:.1f} MB")
    print(f"  Output path:        {OUTPUT_PATH}")
    print(f"  Total time:         {elapsed_total:.0f}s")

    if file_size_mb > 30:
        print(f"\n  WARNING: File is {file_size_mb:.1f} MB (>30 MB target)")
        print("  Consider increasing simplify_tolerance or min_area_sqkm")


if __name__ == "__main__":
    main()
