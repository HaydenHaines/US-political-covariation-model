"""Tests for community discovery pipeline (adjacency, clustering, borders).

Uses synthetic geometries and shift data — no real data files needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.discovery.build_adjacency import (
    build_queen_adjacency,
    handle_islands,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def grid_gdf():
    """4x4 grid of square tract polygons with geoids."""
    import geopandas as gpd
    geoids, geoms = [], []
    for r in range(4):
        for c in range(4):
            geoids.append(f"120010{r:02d}{c:02d}0")
            geoms.append(box(c, r, c + 1, r + 1))
    return gpd.GeoDataFrame({"tract_geoid": geoids, "geometry": geoms}, crs="EPSG:4326")


@pytest.fixture(scope="module")
def grid_with_island(grid_gdf):
    """Grid plus one disconnected island tract."""
    import geopandas as gpd
    island = gpd.GeoDataFrame({
        "tract_geoid": ["12001009990"],
        "geometry": [box(100, 100, 101, 101)],
    }, crs="EPSG:4326")
    result = pd.concat([grid_gdf, island], ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQueenAdjacency:
    def test_returns_sparse_matrix(self, grid_gdf):
        from scipy.sparse import issparse
        W, geoids = build_queen_adjacency(grid_gdf)
        assert issparse(W)

    def test_shape_matches_tract_count(self, grid_gdf):
        W, geoids = build_queen_adjacency(grid_gdf)
        assert W.shape == (16, 16)

    def test_corner_has_3_neighbors(self, grid_gdf):
        """Corner tract in 4x4 grid has 3 Queen neighbors (side + diagonal)."""
        W, geoids = build_queen_adjacency(grid_gdf)
        corner_idx = 0  # (0,0)
        assert W[corner_idx].nnz == 3

    def test_center_has_8_neighbors(self, grid_gdf):
        """Interior tract in 4x4 grid has 8 Queen neighbors."""
        W, geoids = build_queen_adjacency(grid_gdf)
        # (1,1) = row 1 * 4 + col 1 = index 5
        assert W[5].nnz == 8

    def test_symmetric(self, grid_gdf):
        W, geoids = build_queen_adjacency(grid_gdf)
        diff = W - W.T
        assert diff.nnz == 0


class TestIslandHandling:
    def test_island_gets_connected(self, grid_with_island):
        W, geoids = build_queen_adjacency(grid_with_island)
        W_fixed = handle_islands(W, grid_with_island)
        island_idx = len(geoids) - 1
        assert W_fixed[island_idx].nnz >= 1

    def test_no_islands_remain(self, grid_with_island):
        W, _ = build_queen_adjacency(grid_with_island)
        W_fixed = handle_islands(W, grid_with_island)
        # Every row should have at least 1 neighbor
        for i in range(W_fixed.shape[0]):
            assert W_fixed[i].nnz >= 1
