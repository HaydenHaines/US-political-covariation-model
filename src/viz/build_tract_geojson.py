"""Build tract-level GeoJSON for Kepler.gl visualization.

Downloads TIGER 2022 tract shapefiles (if not already present), joins them
to NMF community membership weights, computes derived fields, and exports a
GeoJSON file ready for Kepler.gl interactive visualization.

Usage:
    uv run python src/viz/build_tract_geojson.py
"""

from __future__ import annotations

import math
import os
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIGER_DIR = PROJECT_ROOT / "data" / "raw" / "tiger"
MEMBERSHIPS_PATH = PROJECT_ROOT / "data" / "communities" / "tract_memberships_k7.parquet"
VIZ_DIR = PROJECT_ROOT / "data" / "viz"
OUTPUT_PATH = VIZ_DIR / "tract_memberships_k7.geojson"
OUTPUT_PATH_SIMPLIFIED = VIZ_DIR / "tract_memberships_k7_simplified.geojson"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FIPS = ["01", "12", "13"]  # AL, FL, GA

TIGER_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_{fips}_tract.zip"
)

COMMUNITY_LABELS: dict[str, str] = {
    "c1": "c1: White rural homeowner (older+WFH)",
    "c2": "c2: Black urban (transit+income)",
    "c3": "c3: Knowledge worker (mgmt+WFH+college)",
    "c4": "c4: Asian",
    "c5": "c5: Working-class homeowner (owner-occ)",
    "c6": "c6: Hispanic low-income",
    "c7": "c7: Generic suburban baseline",
}

COMMUNITY_COLS = list(COMMUNITY_LABELS.keys())  # ['c1', ..., 'c7']
K = len(COMMUNITY_COLS)  # 7

SIMPLIFY_TOLERANCE_PRIMARY = 0.0001   # ~10 m; primary output
SIMPLIFY_TOLERANCE_FALLBACK = 0.001   # ~100 m; fallback if primary > 100 MB

GEOJSON_SIZE_THRESHOLD_MB = 100


# ---------------------------------------------------------------------------
# Step 1 — Download TIGER 2022 tract shapefiles
# ---------------------------------------------------------------------------


def download_tiger_shapefiles() -> None:
    """Download and unzip TIGER 2022 tract shapefiles for AL, FL, GA.

    Skips states where the zip already exists on disk.
    """
    TIGER_DIR.mkdir(parents=True, exist_ok=True)

    for fips in STATE_FIPS:
        zip_path = TIGER_DIR / f"tl_2022_{fips}_tract.zip"
        extract_dir = TIGER_DIR / f"tl_2022_{fips}_tract"

        if zip_path.exists() and extract_dir.exists():
            print(f"  [skip] TIGER {fips}: already downloaded and extracted")
            continue

        if not zip_path.exists():
            url = TIGER_URL_TEMPLATE.format(fips=fips)
            print(f"  [download] {url}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            zip_path.write_bytes(response.content)
            print(f"  [saved] {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

        if not extract_dir.exists():
            print(f"  [extract] {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            print(f"  [extracted] -> {extract_dir}")


# ---------------------------------------------------------------------------
# Step 2 — Load and concatenate tract geometries
# ---------------------------------------------------------------------------


def load_geometries() -> gpd.GeoDataFrame:
    """Load and concatenate TIGER 2022 tract shapefiles for all three states."""
    gdfs = []
    for fips in STATE_FIPS:
        shp_path = TIGER_DIR / f"tl_2022_{fips}_tract" / f"tl_2022_{fips}_tract.shp"
        gdf = gpd.read_file(shp_path, columns=["GEOID", "geometry"])
        print(f"  [loaded] {fips}: {len(gdf):,} tracts  CRS={gdf.crs}")
        gdfs.append(gdf)

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs,
    )
    print(f"  [combined] {len(combined):,} total tract geometries")
    return combined


# ---------------------------------------------------------------------------
# Step 3 — Join memberships
# ---------------------------------------------------------------------------


def join_memberships(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Inner-join NMF membership weights to tract geometries."""
    memberships = pd.read_parquet(MEMBERSHIPS_PATH)
    print(f"  [memberships] {len(memberships):,} rows loaded")

    # Inner join: geometry GEOID == memberships tract_geoid
    merged = gdf.merge(
        memberships,
        left_on="GEOID",
        right_on="tract_geoid",
        how="inner",
    )
    print(f"  [joined] {len(merged):,} matched tracts")

    # For uninhabited tracts, set c1-c7 to 0.0 (not NaN)
    uninhabited_mask = merged["is_uninhabited"].fillna(False).astype(bool)
    n_uninhabited = uninhabited_mask.sum()
    for col in COMMUNITY_COLS:
        merged.loc[uninhabited_mask, col] = 0.0
    print(f"  [uninhabited] zeroed {n_uninhabited} uninhabited tracts")

    return merged


# ---------------------------------------------------------------------------
# Step 4 — Compute dominant community and entropy
# ---------------------------------------------------------------------------


def compute_derived_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add dominant_community label and membership_entropy."""
    weights = gdf[COMMUNITY_COLS].values  # shape (N, 7)

    # Dominant community: label of the highest-weight component
    # For uninhabited (all zeros) tracts, argmax returns 0 — that's fine.
    dominant_idx = np.argmax(weights, axis=1)
    col_names = np.array(COMMUNITY_COLS)
    dominant_keys = col_names[dominant_idx]
    gdf["dominant_community"] = [COMMUNITY_LABELS[k] for k in dominant_keys]

    # Membership entropy: -sum(w * log(w)) / log(K), normalized to [0, 1]
    # Treat w=0 as 0*log(0) = 0 (standard entropy convention)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = np.where(weights > 0, np.log(weights), 0.0)
    raw_entropy = -np.sum(weights * log_w, axis=1)
    norm_entropy = raw_entropy / math.log(K)

    # Clamp to [0, 1] to handle any floating-point edge cases
    norm_entropy = np.clip(norm_entropy, 0.0, 1.0)
    gdf["membership_entropy"] = norm_entropy

    return gdf


# ---------------------------------------------------------------------------
# Step 5 — Simplify geometry
# ---------------------------------------------------------------------------


def reproject_and_simplify(
    gdf: gpd.GeoDataFrame,
    tolerance: float,
) -> gpd.GeoDataFrame:
    """Reproject to EPSG:4326 and simplify geometry."""
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance=tolerance, preserve_topology=True
    )
    return gdf


# ---------------------------------------------------------------------------
# Step 6 — Export
# ---------------------------------------------------------------------------


def prepare_export_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select, rename, and round columns for GeoJSON export."""
    out = gdf.copy()

    # Round membership weights to 4 dp
    for col in COMMUNITY_COLS:
        out[col] = out[col].round(4)

    # Round entropy to 3 dp
    out["membership_entropy"] = out["membership_entropy"].round(3)

    # Add state_fips derived from tract_geoid
    out["state_fips"] = out["tract_geoid"].str[:2]

    # Ensure is_uninhabited is a plain Python bool (JSON serializable)
    out["is_uninhabited"] = out["is_uninhabited"].astype(bool)

    # Select final columns (geometry is preserved by GeoDataFrame)
    keep_cols = (
        ["tract_geoid", "is_uninhabited"]
        + COMMUNITY_COLS
        + ["dominant_community", "membership_entropy", "state_fips", "geometry"]
    )
    out = out[keep_cols]

    return out


def export_geojson(gdf: gpd.GeoDataFrame, path: Path) -> float:
    """Write GeoDataFrame to GeoJSON and return file size in MB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    size_mb = path.stat().st_size / 1e6
    return size_mb


# ---------------------------------------------------------------------------
# Step 7 — Print summary
# ---------------------------------------------------------------------------


def print_summary(gdf: gpd.GeoDataFrame, output_path: Path, size_mb: float) -> None:
    """Print post-export summary statistics."""
    print()
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Output file : {output_path}")
    print(f"File size   : {size_mb:.2f} MB")
    print(f"Features    : {len(gdf):,}")

    # Dominant community distribution
    print()
    print("Dominant community distribution:")
    dist = gdf["dominant_community"].value_counts()
    for label, count in dist.items():
        pct = 100 * count / len(gdf)
        print(f"  {label:<50s}  {count:>5,}  ({pct:.1f}%)")

    # Mean entropy (populated tracts only)
    populated = gdf[~gdf["is_uninhabited"]]
    mean_entropy = populated["membership_entropy"].mean()
    print()
    print(f"Mean membership entropy (populated tracts): {mean_entropy:.4f}")
    print(f"  (0 = pure single-community, 1 = perfectly uniform across all 7)")

    # Validate c1-c7 in [0, 1]
    all_valid = True
    for col in COMMUNITY_COLS:
        col_min = gdf[col].min()
        col_max = gdf[col].max()
        if col_min < 0 or col_max > 1:
            print(f"  [WARN] {col} out of range: min={col_min:.6f}, max={col_max:.6f}")
            all_valid = False
    if all_valid:
        print()
        print("Validation: all c1-c7 values confirmed in [0, 1]")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n=== build_tract_geojson.py ===\n")

    # Step 1
    print("Step 1: Download TIGER 2022 shapefiles")
    download_tiger_shapefiles()

    # Step 2
    print("\nStep 2: Load tract geometries")
    gdf = load_geometries()

    # Step 3
    print("\nStep 3: Join membership weights")
    gdf = join_memberships(gdf)

    # Step 4
    print("\nStep 4: Compute derived columns")
    gdf = compute_derived_columns(gdf)

    # Step 5 (primary tolerance)
    print(f"\nStep 5: Reproject to EPSG:4326 and simplify (tolerance={SIMPLIFY_TOLERANCE_PRIMARY})")
    gdf_simplified = reproject_and_simplify(gdf, tolerance=SIMPLIFY_TOLERANCE_PRIMARY)

    # Step 6
    print("\nStep 6: Export GeoJSON")
    export_gdf = prepare_export_gdf(gdf_simplified)
    size_mb = export_geojson(export_gdf, OUTPUT_PATH)
    print(f"  [exported] {OUTPUT_PATH}  ({size_mb:.2f} MB)")

    # If primary output exceeds 100 MB, also produce a more-simplified version
    if size_mb > GEOJSON_SIZE_THRESHOLD_MB:
        print(
            f"\n  [note] Primary output is {size_mb:.1f} MB (> {GEOJSON_SIZE_THRESHOLD_MB} MB)."
        )
        print(
            f"  Producing additional simplified version "
            f"(tolerance={SIMPLIFY_TOLERANCE_FALLBACK})..."
        )
        gdf_extra = reproject_and_simplify(gdf, tolerance=SIMPLIFY_TOLERANCE_FALLBACK)
        export_gdf_extra = prepare_export_gdf(gdf_extra)
        size_mb_extra = export_geojson(export_gdf_extra, OUTPUT_PATH_SIMPLIFIED)
        print(f"  [exported] {OUTPUT_PATH_SIMPLIFIED}  ({size_mb_extra:.2f} MB)")

    # Step 7
    print_summary(export_gdf, OUTPUT_PATH, size_mb)


if __name__ == "__main__":
    main()
