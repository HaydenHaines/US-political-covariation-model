"""Build Queen contiguity spatial weights matrix from census tract geometries.

Constructs a spatial adjacency graph for use as the connectivity constraint in
AgglomerativeClustering. Uses Queen contiguity (shared edge OR vertex counts as
a neighbor). Handles island tracts (zero neighbors) by connecting each to its
nearest neighbor via centroid distance.

Outputs:
    data/communities/adjacency.npz  — scipy sparse CSR matrix (N x N)

Usage:
    uv run python src/discovery/build_adjacency.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.sparse import csr_matrix, issparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIGER_DIR = PROJECT_ROOT / "data" / "raw" / "tiger"
OUTPUT_PATH = PROJECT_ROOT / "data" / "communities" / "adjacency.npz"

STATE_FIPS = ["01", "12", "13"]  # AL, FL, GA


# ── Core functions ─────────────────────────────────────────────────────────────


def build_queen_adjacency(
    gdf: gpd.GeoDataFrame,
) -> tuple[csr_matrix, list[str]]:
    """Build a Queen contiguity sparse adjacency matrix from tract geometries.

    Parameters
    ----------
    gdf:
        GeoDataFrame with at least a ``geometry`` column. If a ``tract_geoid``
        column is present it is used for the returned geoid list; otherwise the
        integer index is used.

    Returns
    -------
    W : csr_matrix
        Symmetric binary adjacency matrix of shape (N, N).
    geoids : list[str]
        Ordered list of tract identifiers corresponding to matrix rows/cols.
    """
    from libpysal.weights import Queen

    n = len(gdf)
    log.info("Building Queen adjacency for %d tracts …", n)

    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)

    # Convert libpysal weights to scipy sparse CSR
    sparse = w.sparse  # already CSR
    # libpysal sparse may contain float weights; binarise to 0/1
    sparse = (sparse > 0).astype(np.int8)
    # Ensure strict symmetry (Queen should already be symmetric, but guard)
    sparse = (sparse + sparse.T)
    sparse = (sparse > 0).astype(np.int8)
    W = csr_matrix(sparse)

    # Build geoid list in GDF row order
    if "tract_geoid" in gdf.columns:
        geoids = list(gdf["tract_geoid"].astype(str))
    else:
        geoids = [str(i) for i in gdf.index]

    # Neighbor stats
    degrees = np.array(W.sum(axis=1)).flatten()
    island_count = int((degrees == 0).sum())
    log.info(
        "Queen adjacency built: mean neighbors=%.2f, min=%d, max=%d, islands=%d",
        degrees.mean(),
        int(degrees.min()),
        int(degrees.max()),
        island_count,
    )

    return W, geoids


def handle_islands(W: csr_matrix, gdf: gpd.GeoDataFrame) -> csr_matrix:
    """Connect island tracts (zero Queen neighbors) to their nearest neighbor.

    For each island, computes Euclidean centroid distance to every other tract
    and adds a symmetric edge to the nearest one.

    Parameters
    ----------
    W:
        Adjacency matrix produced by :func:`build_queen_adjacency`.
    gdf:
        The same GeoDataFrame passed to ``build_queen_adjacency`` (same row
        order).

    Returns
    -------
    csr_matrix
        Updated adjacency matrix with islands resolved.
    """
    degrees = np.array(W.sum(axis=1)).flatten()
    island_indices = np.where(degrees == 0)[0]

    if len(island_indices) == 0:
        log.info("No island tracts detected — nothing to fix.")
        return W

    log.info("Connecting %d island tract(s) to nearest neighbor …", len(island_indices))

    # Work with a lil_matrix for efficient row assignment
    W_lil = W.tolil()

    # Compute centroids; reproject to a metric CRS for distance accuracy
    gdf_proj = gdf.to_crs("EPSG:3857")
    centroids = np.column_stack(
        [gdf_proj.geometry.centroid.x, gdf_proj.geometry.centroid.y]
    )

    for idx in island_indices:
        centroid = centroids[idx]
        dists = np.linalg.norm(centroids - centroid, axis=1)
        dists[idx] = np.inf  # exclude self
        nearest = int(np.argmin(dists))
        W_lil[idx, nearest] = 1
        W_lil[nearest, idx] = 1
        log.debug("  Island %d connected to nearest neighbor %d", idx, nearest)

    W_fixed = csr_matrix(W_lil)
    # Final symmetry normalisation
    W_fixed = (W_fixed + W_fixed.T)
    W_fixed = (W_fixed > 0).astype(np.int8)
    W_fixed = csr_matrix(W_fixed)

    remaining = int((np.array(W_fixed.sum(axis=1)).flatten() == 0).sum())
    log.info("Island handling complete. Remaining islands: %d", remaining)

    return W_fixed


# ── Main ───────────────────────────────────────────────────────────────────────


def _load_tiger_tracts() -> gpd.GeoDataFrame:
    """Load TIGER 2022 tract shapefiles for AL, FL, GA."""
    gdfs = []
    for fips in STATE_FIPS:
        extract_dir = TIGER_DIR / f"tl_2022_{fips}_tract"
        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(
                f"TIGER shapefile not found at {extract_dir}. "
                "Run src/viz/build_tract_geojson.py first to download TIGER data."
            )
        gdf = gpd.read_file(shp_files[0])
        gdf = gdf.rename(columns={"GEOID": "tract_geoid"})
        gdfs.append(gdf[["tract_geoid", "geometry"]])
        log.info("  Loaded %d tracts for FIPS %s", len(gdf), fips)

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=gdfs[0].crs
    )
    log.info("Total tracts loaded: %d", len(combined))
    return combined


def main() -> None:
    """Build adjacency matrix from TIGER shapefiles and save to disk."""
    import pandas as pd  # local import to keep module-level imports lean

    log.info("Loading TIGER tract geometries …")
    gdf = _load_tiger_tracts()

    log.info("Building Queen contiguity adjacency …")
    W, geoids = build_queen_adjacency(gdf)

    log.info("Handling island tracts …")
    W = handle_islands(W, gdf)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    from scipy.sparse import save_npz
    save_npz(str(OUTPUT_PATH), W)
    log.info("Adjacency matrix saved to %s", OUTPUT_PATH)

    # Save geoid ordering alongside the matrix
    geoid_path = OUTPUT_PATH.with_suffix(".geoids.txt")
    geoid_path.write_text("\n".join(geoids))
    log.info("Geoid ordering saved to %s", geoid_path)


if __name__ == "__main__":
    main()
