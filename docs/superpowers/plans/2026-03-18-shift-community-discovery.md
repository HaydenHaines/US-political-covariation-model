# Shift-Based Community Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover electoral communities from spatially correlated vote shifts across FL+GA+AL census tracts, replacing the shelved NMF-on-demographics approach.

**Architecture:** Build 9-dimensional shift vectors per tract (D/R/turnout deltas across presidential and midterm election pairs), then run hierarchical agglomerative clustering with Queen spatial contiguity constraint. Validate by temporal holdout against 2024. Compare predictive power to the shelved NMF communities.

**Tech Stack:** Python, pandas, geopandas, libpysal, scikit-learn, scipy, kneed, parquet

**Spec:** `docs/superpowers/specs/2026-03-18-shift-based-community-discovery-design.md`

---

## File Map

### New files to create

| File | Responsibility |
|------|---------------|
| `src/assembly/build_shift_vectors.py` | Compute per-tract 9-dim shift vectors from VEST + MEDSL data |
| `src/discovery/build_adjacency.py` | Queen contiguity graph from tract geometries, island handling |
| `src/discovery/cluster_communities.py` | Hierarchical agglomerative clustering, dendrogram, elbow cut |
| `src/discovery/score_borders.py` | Border gradient sharpness between adjacent communities |
| `src/discovery/__init__.py` | Package init |
| `src/description/describe_communities.py` | Census overlay, demographic profiles per community |
| `src/description/compare_to_nmf.py` | Side-by-side comparison with shelved NMF communities |
| `src/description/__init__.py` | Package init |
| `src/validation/validate_holdout.py` | Temporal holdout validation against 2024 |
| `tests/test_shift_vectors.py` | Tests for shift vector construction |
| `tests/test_discovery.py` | Tests for adjacency, clustering, border scoring |
| `tests/test_description.py` | Tests for community description and NMF comparison |
| `tests/test_holdout.py` | Tests for holdout validation logic |

### Existing files to modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add libpysal, kneed dependencies |
| `CLAUDE.md` | Update for architectural pivot, add ADR-005 reference |
| `docs/ASSUMPTIONS_LOG.md` | Update two-stage separation assumption status |

---

## Task 0: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `src/discovery/__init__.py`, `src/description/__init__.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Open `pyproject.toml` and add `libpysal` and `kneed` to the `[project.dependencies]` list. `geopandas`, `scipy`, and `scikit-learn` should already be present — verify.

- [ ] **Step 2: Install and verify**

Run: `cd /home/hayden/projects/US-political-covariation-model && uv sync`
Expected: all deps install successfully

- [ ] **Step 3: Create package directories**

```bash
mkdir -p src/discovery src/description
touch src/discovery/__init__.py src/description/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/discovery/__init__.py src/description/__init__.py
git commit -m "chore: add libpysal, kneed deps; create discovery/description packages"
```

---

## Task 1: Build Shift Vectors

**Files:**
- Create: `src/assembly/build_shift_vectors.py`
- Test: `tests/test_shift_vectors.py`

### Context for implementer

The existing data pipeline produces these parquet files:
- `data/assembled/vest_tracts_2016.parquet` — columns include `tract_geoid`, `pres_dem_share_2016`, `pres_total_2016`
- `data/assembled/vest_tracts_2018.parquet` — columns include `tract_geoid`, `gov_dem_share_2018`, `gov_total_2018`
- `data/assembled/vest_tracts_2020.parquet` (from `fetch_vest.py`) — `tract_geoid`, `pres_dem_share_2020`, `pres_total_2020`
- `data/assembled/medsl_county_2022_governor.parquet` — `county_fips`, `gov_dem_share_2022`, `gov_total_2022`
- `data/assembled/medsl_county_2024_president.parquet` — `county_fips`, `pres_dem_share_2024`, `pres_total_2024`

For 2022/2024 (county-level only), every tract in a county gets the same shift. For AL midterm (2018 uncontested), all 3 midterm dimensions are set to 0.0 (structural zero, not missing).

The output should be `data/shifts/tract_shifts.parquet` with columns:
- `tract_geoid` (11-digit string)
- `pres_d_shift_16_20`, `pres_r_shift_16_20`, `pres_turnout_shift_16_20`
- `pres_d_shift_20_24`, `pres_r_shift_20_24`, `pres_turnout_shift_20_24`
- `mid_d_shift_18_22`, `mid_r_shift_18_22`, `mid_turnout_shift_18_22`

D share shift = later_dem_share - earlier_dem_share.
R share shift = (1 - later_dem_share) - (1 - earlier_dem_share) = earlier_dem_share - later_dem_share (= -D shift in two-party). Keep both: third-party vote exists in 2016.
Turnout shift = later_total / later_vap - earlier_total / earlier_vap. If VAP is unavailable, use raw vote totals as proxy (acceptable for MVP since we're comparing shifts, not levels).

- [ ] **Step 1: Write failing tests**

Create `tests/test_shift_vectors.py`:

```python
"""Tests for shift vector construction.

Uses synthetic DataFrames to verify shift math, AL midterm zeroing,
county-level fallback for MEDSL data, and output shape/columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_shift_vectors import (
    compute_presidential_shifts,
    compute_midterm_shifts,
    build_shift_vectors,
    SHIFT_COLS,
    AL_FIPS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vest_2016():
    """Synthetic VEST 2016 tract-level data (3 tracts, 2 states)."""
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pres_dem_share_2016": [0.60, 0.55, 0.40],
        "pres_total_2016": [1000, 800, 500],
    })


@pytest.fixture
def vest_2020():
    """Synthetic VEST 2020 tract-level data."""
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pres_dem_share_2020": [0.62, 0.50, 0.38],
        "pres_total_2020": [1100, 850, 520],
    })


@pytest.fixture
def medsl_2024():
    """Synthetic MEDSL 2024 county-level data."""
    return pd.DataFrame({
        "county_fips": ["12001", "01001"],
        "pres_dem_share_2024": [0.58, 0.35],
        "pres_total_2024": [2100, 540],
    })


@pytest.fixture
def vest_2018():
    """Synthetic VEST 2018 tract-level data (FL+GA only, AL uncontested)."""
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200"],
        "gov_dem_share_2018": [0.55, 0.48],
        "gov_total_2018": [900, 700],
    })


@pytest.fixture
def medsl_2022():
    """Synthetic MEDSL 2022 county-level data."""
    return pd.DataFrame({
        "county_fips": ["12001"],
        "gov_dem_share_2022": [0.52, ],
        "gov_total_2022": [1700],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPresidentialShifts:
    def test_d_shift_is_difference(self, vest_2016, vest_2020):
        result = compute_presidential_shifts(vest_2016, vest_2020, "16_20")
        expected = 0.62 - 0.60  # tract 12001000100
        assert abs(result.loc[result.tract_geoid == "12001000100", "pres_d_shift_16_20"].iloc[0] - expected) < 1e-9

    def test_output_has_all_columns(self, vest_2016, vest_2020):
        result = compute_presidential_shifts(vest_2016, vest_2020, "16_20")
        for col in ["pres_d_shift_16_20", "pres_r_shift_16_20", "pres_turnout_shift_16_20"]:
            assert col in result.columns

    def test_no_nans_for_matched_tracts(self, vest_2016, vest_2020):
        result = compute_presidential_shifts(vest_2016, vest_2020, "16_20")
        assert not result[["pres_d_shift_16_20", "pres_r_shift_16_20"]].isna().any().any()


class TestMidtermShifts:
    def test_al_tracts_get_zero(self, vest_2018, medsl_2022):
        result = compute_midterm_shifts(vest_2018, medsl_2022)
        al_rows = result[result.tract_geoid.str.startswith("01")]
        if len(al_rows) > 0:
            assert (al_rows["mid_d_shift_18_22"] == 0.0).all()
            assert (al_rows["mid_r_shift_18_22"] == 0.0).all()
            assert (al_rows["mid_turnout_shift_18_22"] == 0.0).all()

    def test_fl_tracts_have_nonzero_shift(self, vest_2018, medsl_2022):
        result = compute_midterm_shifts(vest_2018, medsl_2022)
        fl_rows = result[result.tract_geoid.str.startswith("12")]
        assert not (fl_rows["mid_d_shift_18_22"] == 0.0).all()


class TestBuildShiftVectors:
    def test_output_columns(self, vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024):
        result = build_shift_vectors(vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024)
        assert "tract_geoid" in result.columns
        assert len(SHIFT_COLS) == 9
        for col in SHIFT_COLS:
            assert col in result.columns

    def test_output_has_all_tracts(self, vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024):
        result = build_shift_vectors(vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024)
        assert len(result) == 3  # all tracts from vest_2020
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/hayden/projects/US-political-covariation-model && uv run pytest tests/test_shift_vectors.py -v`
Expected: ImportError — `src.assembly.build_shift_vectors` does not exist yet

- [ ] **Step 3: Implement build_shift_vectors.py**

Create `src/assembly/build_shift_vectors.py`. Key functions:

- `compute_presidential_shifts(early_df, late_df, label) -> pd.DataFrame` — joins on `tract_geoid`, computes D/R/turnout deltas
- `compute_midterm_shifts(vest_2018, medsl_2022) -> pd.DataFrame` — expands county-level 2022 to tracts, zeros AL tracts, computes deltas
- `build_shift_vectors(...) -> pd.DataFrame` — uses `vest_2020` tract_geoid list as the authoritative spine. Left-joins all shift DataFrames onto this spine. AL tracts will have NaN midterm shifts from the join — fill these with 0.0 (structural zero, see spec). This ensures all tracts appear in output even when midterm data is missing for AL.
- `main()` — loads parquets, calls functions, saves to `data/shifts/tract_shifts.parquet`. Creates `data/shifts/` directory if needed (`output_path.parent.mkdir(parents=True, exist_ok=True)`).

Note: normalization (zero-mean, unit-variance, sqrt(2) midterm scaling) lives in `cluster_communities.normalize_shifts()`, NOT in this module. This module produces raw shift values only.

Constants: `SHIFT_COLS` (list of 9 column names), `AL_FIPS = "01"`, `STATES`, `PROJECT_ROOT`.

For county→tract expansion of MEDSL data: join on `county_fips = tract_geoid[:5]`.
For R share: compute as `1 - dem_share` (two-party simplification; third-party signal captured by D+R not summing to 1.0 in VEST data where third-party votes exist).
For turnout shift: use raw vote totals as proxy (total_later - total_earlier) / total_earlier. This is a relative change, not absolute VAP-normalized, acceptable for MVP.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shift_vectors.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/assembly/build_shift_vectors.py tests/test_shift_vectors.py
git commit -m "feat: build_shift_vectors — 9-dim electoral shift vectors per tract"
```

---

## Task 2: Build Adjacency Graph

**Files:**
- Create: `src/discovery/build_adjacency.py`
- Test: `tests/test_discovery.py` (first section)

### Context for implementer

This module builds a Queen contiguity spatial weights matrix from tract geometries. Uses `libpysal.weights.Queen`. Must handle "island" tracts (zero neighbors) by connecting them to nearest neighbor via `libpysal.weights.KNN(k=1)`. The output is a scipy sparse matrix compatible with sklearn's `AgglomerativeClustering(connectivity=...)`.

Tract geometries come from Census TIGER/Line shapefiles. The existing `src/viz/build_tract_geojson.py` already downloads and processes these. Alternatively, use `geopandas.read_file()` on the TIGER shapefiles in `data/raw/tiger/`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_discovery.py`:

```python
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
    return pd.concat([grid_gdf, island], ignore_index=True)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_discovery.py::TestQueenAdjacency -v`
Expected: ImportError

- [ ] **Step 3: Implement build_adjacency.py**

Create `src/discovery/build_adjacency.py`. Key functions:

- `build_queen_adjacency(gdf: gpd.GeoDataFrame) -> tuple[scipy.sparse.csr_matrix, list[str]]` — builds Queen contiguity via `libpysal.weights.Queen.from_dataframe(gdf)`, converts to scipy sparse via `w.sparse`, returns (matrix, geoid_list). Log tract count and neighbor stats.
- `handle_islands(W: csr_matrix, gdf: gpd.GeoDataFrame) -> csr_matrix` — finds rows with zero neighbors, for each island computes centroid distance to all other tracts, connects to nearest. Returns updated sparse matrix. Log island count.
- `main()` — loads tract geometries from TIGER shapefiles, calls above functions, saves sparse matrix to `data/communities/adjacency.npz`. Creates output directory: `output_path.parent.mkdir(parents=True, exist_ok=True)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_discovery.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/build_adjacency.py tests/test_discovery.py
git commit -m "feat: build_adjacency — Queen contiguity with island handling"
```

---

## Task 3: Cluster Communities

**Files:**
- Create: `src/discovery/cluster_communities.py`
- Modify: `tests/test_discovery.py` (add clustering tests)

### Context for implementer

Uses `sklearn.cluster.AgglomerativeClustering` with the Queen adjacency as connectivity constraint. Must set `compute_distances=True` to enable dendrogram reconstruction. The scipy linkage matrix must be manually built from sklearn's `children_` and `distances_` arrays — sklearn does not produce it directly. Format: `[left_child, right_child, distance, n_samples_in_cluster]` per merge.

Elbow detection uses `kneed.KneeLocator` with `curve="convex"`, `direction="decreasing"`, `S=1.0`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_discovery.py`:

```python
from src.discovery.cluster_communities import (
    cluster_at_threshold,
    build_linkage_matrix,
    find_elbow,
    normalize_shifts,
)


class TestNormalizeShifts:
    def test_output_zero_mean(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        normed = normalize_shifts(shifts, n_presidential_dims=6)
        assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-10)

    def test_presidential_cols_unit_variance(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        normed = normalize_shifts(shifts, n_presidential_dims=6)
        # Presidential dims should have variance ~1.0 (standard normalization)
        assert np.allclose(normed[:, :6].var(axis=0), 1.0, atol=0.15)

    def test_midterm_cols_scaled_variance(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        normed = normalize_shifts(shifts, n_presidential_dims=6)
        # Midterm dims should have variance ~2.0 (scaled by sqrt(2), so var = 2)
        assert np.allclose(normed[:, 6:].var(axis=0), 2.0, atol=0.3)


class TestClusterAtThreshold:
    def test_returns_labels(self, grid_gdf):
        from scipy.sparse import csr_matrix
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels, model = cluster_at_threshold(shifts, W, threshold=5.0)
        assert len(labels) == n
        assert labels.min() >= 0

    def test_more_clusters_at_lower_threshold(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels_fine, _ = cluster_at_threshold(shifts, W, threshold=1.0)
        labels_coarse, _ = cluster_at_threshold(shifts, W, threshold=10.0)
        assert len(set(labels_fine)) >= len(set(labels_coarse))


class TestBuildLinkageMatrix:
    def test_linkage_shape(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        _, model = cluster_at_threshold(shifts, W, n_clusters=1)  # force full merge
        linkage = build_linkage_matrix(model)
        assert linkage.shape == (n - 1, 4)  # n-1 merges, 4 columns

    def test_distances_monotonic(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        _, model = cluster_at_threshold(shifts, W, n_clusters=1)  # force full merge
        linkage = build_linkage_matrix(model)
        distances = linkage[:, 2]
        assert np.all(distances[1:] >= distances[:-1])


class TestFindElbow:
    def test_returns_valid_threshold(self):
        # Synthetic variance curve with a clear elbow
        n_communities = np.array([200, 150, 100, 80, 60, 50, 45, 42, 40, 39])
        variances = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.10, 1.50])
        elbow_k = find_elbow(n_communities, variances)
        assert elbow_k is not None
        assert 40 <= elbow_k <= 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_discovery.py::TestNormalizeShifts -v`
Expected: ImportError

- [ ] **Step 3: Implement cluster_communities.py**

Create `src/discovery/cluster_communities.py`. Key functions:

- `normalize_shifts(shifts: np.ndarray, n_presidential_dims: int = 6) -> np.ndarray` — zero-mean, unit-variance per column, then scale midterm columns by sqrt(2)
- `cluster_at_threshold(shifts: np.ndarray, connectivity: csr_matrix, threshold: float | None = None, n_clusters: int | None = None) -> tuple[np.ndarray, AgglomerativeClustering]` — run sklearn AgglomerativeClustering with `linkage="ward"`, `connectivity=connectivity`, `compute_distances=True`. Pass either `distance_threshold` or `n_clusters` (mutually exclusive; exactly one must be set). Use `n_clusters=1` to force a full-merge dendrogram. Return (labels, model).
- `build_linkage_matrix(model: AgglomerativeClustering) -> np.ndarray` — construct scipy-format linkage `[left, right, distance, count]` from `model.children_` and `model.distances_`. Count for leaf = 1, for merge = sum of children's counts.
- `find_elbow(n_communities: np.ndarray, variances: np.ndarray) -> int | None` — use `kneed.KneeLocator(n_communities, variances, curve="convex", direction="decreasing", S=1.0)`. Return knee x-value.
- `sweep_thresholds(shifts, connectivity, linkage, n_steps=50) -> tuple[np.ndarray, np.ndarray]` — cut dendrogram at `n_steps` evenly spaced heights, compute weighted mean within-cluster variance at each. Return (n_communities_array, variance_array).
- `main()` — loads shift vectors + adjacency, normalizes, clusters at full merge, builds linkage, sweeps thresholds, finds elbow, clusters at elbow threshold, saves `community_assignments.parquet` + `dendrogram.pkl`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_discovery.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/cluster_communities.py tests/test_discovery.py
git commit -m "feat: cluster_communities — hierarchical spatial clustering with elbow detection"
```

---

## Task 4: Score Borders

**Files:**
- Create: `src/discovery/score_borders.py`
- Modify: `tests/test_discovery.py` (add border tests)

### Context for implementer

For every pair of adjacent communities, compute the mean shift-vector Euclidean distance between tracts on either side of the boundary. This produces a "border sharpness" score. High = hard demographic wall, low = soft transition.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_discovery.py`:

```python
from src.discovery.score_borders import compute_border_gradients


class TestBorderGradients:
    def test_output_columns(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, geoids = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        # Make 2 communities: top half vs bottom half
        labels = np.array([0]*8 + [1]*8)
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert "community_a" in result.columns
        assert "community_b" in result.columns
        assert "gradient" in result.columns
        assert "n_boundary_pairs" in result.columns

    def test_identical_communities_have_zero_gradient(self, grid_gdf):
        n = len(grid_gdf)
        # All tracts have identical shifts
        shifts = np.ones((n, 9))
        W, geoids = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels = np.array([0]*8 + [1]*8)
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert (result["gradient"] == 0.0).all()

    def test_different_communities_have_positive_gradient(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = np.zeros((n, 9))
        shifts[:8, :] = 1.0  # community 0 shifted right
        shifts[8:, :] = -1.0  # community 1 shifted left
        W, geoids = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels = np.array([0]*8 + [1]*8)
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert (result["gradient"] > 0.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_discovery.py::TestBorderGradients -v`
Expected: ImportError

- [ ] **Step 3: Implement score_borders.py**

Create `src/discovery/score_borders.py`. Key function:

- `compute_border_gradients(labels: np.ndarray, shifts: np.ndarray, W: csr_matrix, geoids: list[str]) -> pd.DataFrame` — for each nonzero entry (i,j) in W where labels[i] != labels[j], compute Euclidean distance between shifts[i] and shifts[j]. Group by (community_a, community_b) pair (sorted), compute mean distance and count. Return DataFrame with columns: community_a, community_b, gradient, n_boundary_pairs.
- `main()` — loads assignments + shifts + adjacency, computes gradients, saves to `data/communities/border_gradients.parquet`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_discovery.py::TestBorderGradients -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/score_borders.py tests/test_discovery.py
git commit -m "feat: score_borders — boundary gradient sharpness between communities"
```

---

## Task 5: Describe Communities

**Files:**
- Create: `src/description/describe_communities.py`
- Test: `tests/test_description.py`

### Context for implementer

Overlay ACS demographics onto discovered communities. For each community, compute population-weighted means of the 12 ACS features plus total population, land area, and turnout by election type.

- [ ] **Step 1: Write failing tests**

Create `tests/test_description.py`:

```python
"""Tests for community description (census overlay)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.description.describe_communities import (
    build_community_profiles,
    DEMOGRAPHIC_COLS,
)


@pytest.fixture
def assignments():
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "community_id": [0, 0, 1],
    })


@pytest.fixture
def features():
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pop_total": [5000, 3000, 2000],  # unequal for weighted mean testing
        "pct_white_nh": [0.60, 0.65, 0.80],
        "pct_black": [0.25, 0.20, 0.10],
        "pct_asian": [0.05, 0.05, 0.02],
        "pct_hispanic": [0.10, 0.10, 0.08],
        "log_median_income": [10.5, 10.8, 10.0],
        "pct_mgmt_occ": [0.30, 0.35, 0.20],
        "pct_owner_occ": [0.55, 0.60, 0.70],
        "pct_car_commute": [0.70, 0.65, 0.85],
        "pct_transit_commute": [0.15, 0.20, 0.02],
        "pct_wfh_commute": [0.10, 0.10, 0.08],
        "pct_college_plus": [0.40, 0.45, 0.20],
        "median_age": [35.0, 38.0, 42.0],
    })


@pytest.fixture
def shifts():
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pres_d_shift_16_20": [0.02, 0.03, -0.02],
        "pres_d_shift_20_24": [-0.01, -0.02, -0.05],
    })


class TestBuildCommunityProfiles:
    def test_one_row_per_community(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        assert len(result) == 2  # community 0 and 1

    def test_has_demographic_columns(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        for col in DEMOGRAPHIC_COLS:
            assert col in result.columns

    def test_has_community_id(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        assert "community_id" in result.columns

    def test_has_tract_count(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        assert "n_tracts" in result.columns
        assert result.loc[result.community_id == 0, "n_tracts"].iloc[0] == 2

    def test_demographics_are_population_weighted(self, assignments, features, shifts):
        result = build_community_profiles(assignments, features, shifts)
        # Community 0 = tracts 12001000100 (pop 5000) + 12001000200 (pop 3000)
        # Population-weighted pct_white_nh = (0.60*5000 + 0.65*3000) / 8000 = 0.61875
        comm0 = result.loc[result.community_id == 0]
        expected = (0.60 * 5000 + 0.65 * 3000) / 8000
        assert abs(comm0["pct_white_nh"].iloc[0] - expected) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_description.py -v`
Expected: ImportError

- [ ] **Step 3: Implement describe_communities.py**

Create `src/description/describe_communities.py`. Key functions:

- `build_community_profiles(assignments: pd.DataFrame, features: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame` — join assignments with features and shifts on `tract_geoid`, group by `community_id`, compute mean of demographic cols and shift cols, count tracts. Return one row per community.
- `main()` — loads community_assignments.parquet, tract_features.parquet, tract_shifts.parquet, calls build_community_profiles, saves `data/communities/community_profiles.parquet`.

Constants: `DEMOGRAPHIC_COLS` (the 12 ACS feature names from `build_features.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_description.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/description/describe_communities.py tests/test_description.py
git commit -m "feat: describe_communities — census overlay and community profiles"
```

---

## Task 6: Compare to NMF Baseline

**Files:**
- Create: `src/description/compare_to_nmf.py`
- Modify: `tests/test_description.py` (add comparison tests)

### Context for implementer

Load the shelved NMF community assignments from `data/propagation/community_weights_tract.parquet` (K=7 soft assignments). Convert to hard assignment (argmax). Compare against shift-discovered communities using within-community shift variance on the holdout period.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_description.py`:

```python
from src.description.compare_to_nmf import (
    nmf_hard_assignment,
    within_community_variance,
    random_spatial_variance,
)


class TestNmfHardAssignment:
    def test_assigns_to_max_weight(self):
        weights = pd.DataFrame({
            "tract_geoid": ["12001000100", "12001000200"],
            "c0": [0.7, 0.1],
            "c1": [0.2, 0.8],
            "c2": [0.1, 0.1],
        })
        result = nmf_hard_assignment(weights, component_cols=["c0", "c1", "c2"])
        assert result.loc[result.tract_geoid == "12001000100", "nmf_community"].iloc[0] == 0
        assert result.loc[result.tract_geoid == "12001000200", "nmf_community"].iloc[0] == 1


class TestWithinCommunityVariance:
    def test_identical_tracts_have_zero_variance(self):
        shifts = np.ones((10, 9))
        labels = np.array([0]*5 + [1]*5)
        var = within_community_variance(shifts, labels)
        assert var == 0.0

    def test_different_tracts_have_positive_variance(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        labels = np.array([0]*50 + [1]*50)
        var = within_community_variance(shifts, labels)
        assert var > 0.0

    def test_known_variance(self):
        """Two clusters of 2 points each, hand-calculable variance."""
        shifts = np.array([
            [1.0, 0.0],  # cluster 0
            [3.0, 0.0],  # cluster 0  -> centroid [2,0], distances 1,1, var=1.0
            [0.0, 1.0],  # cluster 1
            [0.0, 3.0],  # cluster 1  -> centroid [0,2], distances 1,1, var=1.0
        ])
        labels = np.array([0, 0, 1, 1])
        var = within_community_variance(shifts, labels)
        # Both clusters have var=1.0, equal size -> weighted mean = 1.0
        assert abs(var - 1.0) < 1e-9


class TestRandomSpatialVariance:
    def test_positive_for_nonuniform_shifts(self):
        from scipy.sparse import csr_matrix
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(16, 9))
        # Simple chain adjacency for 16 nodes
        rows = list(range(15)) + list(range(1, 16))
        cols = list(range(1, 16)) + list(range(15))
        W = csr_matrix(([1]*30, (rows, cols)), shape=(16, 16))
        var = random_spatial_variance(shifts, W, n_communities=4, n_trials=10)
        assert var > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_description.py::TestNmfHardAssignment -v`
Expected: ImportError

- [ ] **Step 3: Implement compare_to_nmf.py**

Create `src/description/compare_to_nmf.py`. Key functions:

- `nmf_hard_assignment(weights_df: pd.DataFrame, component_cols: list[str]) -> pd.DataFrame` — argmax across component columns, return DataFrame with tract_geoid + nmf_community.
- `within_community_variance(shifts: np.ndarray, labels: np.ndarray) -> float` — for each community k, compute mean squared distance from each member's shift vector to the community centroid: `var_k = mean(||x_i - mu_k||^2)`. Return the size-weighted mean across communities: `sum(n_k * var_k) / sum(n_k)`.
- `random_spatial_variance(shifts, W, n_communities, n_trials=100) -> float` — baseline: randomly partition the adjacency graph into n_communities contiguous chunks, compute mean within-community variance. Return mean across trials.
- `main()` — loads NMF weights, shift-discovered assignments, shift vectors. Computes within-community variance for both. Prints comparison table. Saves `data/validation/nmf_comparison.parquet`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_description.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/description/compare_to_nmf.py tests/test_description.py
git commit -m "feat: compare_to_nmf — side-by-side shift vs NMF community comparison"
```

---

## Task 7: Temporal Holdout Validation

**Files:**
- Create: `src/validation/validate_holdout.py`
- Test: `tests/test_holdout.py`

### Context for implementer

Training: discover communities from 2016→2020 presidential shift + 2018→2022 midterm shift (6 dims).
Holdout: 2020→2024 presidential shift (3 dims, unseen during training).

Metrics: (1) within-community variance of holdout shift, (2) community-level prediction correlation, (3) comparison to NMF baseline and random spatial baseline.

- [ ] **Step 1: Write failing tests**

Create `tests/test_holdout.py`:

```python
"""Tests for temporal holdout validation."""
from __future__ import annotations

import numpy as np
import pytest

from src.validation.validate_holdout import (
    split_training_holdout,
    community_level_prediction_accuracy,
)
from src.description.compare_to_nmf import within_community_variance


class TestSplitTrainingHoldout:
    def test_training_has_6_dims(self):
        """Training = pres 16->20 (cols 0-2) + midterm 18->22 (cols 6-8) = 6 dims."""
        shifts_9d = np.random.default_rng(42).normal(size=(100, 9))
        train, holdout = split_training_holdout(shifts_9d)
        assert train.shape == (100, 6)

    def test_holdout_has_3_dims(self):
        """Holdout = pres 20->24 (cols 3-5) = 3 dims."""
        shifts_9d = np.random.default_rng(42).normal(size=(100, 9))
        train, holdout = split_training_holdout(shifts_9d)
        assert holdout.shape == (100, 3)

    def test_holdout_is_cols_3_to_5(self):
        """Verify holdout contains exactly the 20->24 presidential shift."""
        shifts_9d = np.arange(900).reshape(100, 9).astype(float)
        _, holdout = split_training_holdout(shifts_9d)
        np.testing.assert_array_equal(holdout, shifts_9d[:, 3:6])


class TestCommunityLevelPrediction:
    def test_perfect_prediction_gives_correlation_1(self):
        # Communities where training shift == holdout shift
        training_means = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        holdout_means = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        corr, mae = community_level_prediction_accuracy(training_means, holdout_means)
        assert abs(corr - 1.0) < 1e-6

    def test_returns_positive_mae(self):
        rng = np.random.default_rng(42)
        training_means = rng.normal(size=(20, 3))
        holdout_means = rng.normal(size=(20, 3))
        corr, mae = community_level_prediction_accuracy(training_means, holdout_means)
        assert mae > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_holdout.py -v`
Expected: ImportError

- [ ] **Step 3: Implement validate_holdout.py**

Create `src/validation/validate_holdout.py`. Key functions:

- `split_training_holdout(shifts_9d: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — SHIFT_COLS order is [pres_16_20 (0-2), pres_20_24 (3-5), mid_18_22 (6-8)]. Training = cols 0-2 + cols 6-8 (non-contiguous: `np.concatenate([shifts[:, 0:3], shifts[:, 6:9]], axis=1)`). Holdout = cols 3-5 (`shifts[:, 3:6]`). Return (train_6d, holdout_3d). CRITICAL: cols 3-5 (the 2024 shift) must NOT appear in training.
- `community_level_prediction_accuracy(training_means: np.ndarray, holdout_means: np.ndarray) -> tuple[float, float]` — compute Pearson correlation and MAE between community-level mean training D-shift and holdout D-shift. Return (correlation, mae).
- `main()` — full holdout pipeline:
  1. Load 9-dim shift vectors
  2. Split into training (6d) and holdout (3d)
  3. Normalize training shifts
  4. Load adjacency, cluster on training shifts only
  5. Compute within-community variance on holdout shifts
  6. Compute community-level prediction accuracy
  7. Compare to NMF baseline and random spatial baseline
  8. Print report, save `data/validation/holdout_2024_results.parquet`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_holdout.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/validation/validate_holdout.py tests/test_holdout.py
git commit -m "feat: validate_holdout — temporal holdout against 2024 presidential shift"
```

---

## Task 8: Documentation Update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/ASSUMPTIONS_LOG.md`
- Create: `docs/adr/005-shift-based-community-discovery.md`

- [ ] **Step 1: Update CLAUDE.md**

Update the Architecture section to reflect the shift-based approach. In the Conventions section under "Research Integrity", change "Two-stage separation is sacred" to note it was validated (R^2~0.66) and superseded by shift-based discovery — falsifiability now via temporal holdout. Update the directory map to include `src/discovery/`, `src/description/`, `data/shifts/`. Add ADR-005 to the Key Decisions Log.

- [ ] **Step 2: Update ASSUMPTIONS_LOG.md**

Find the assumption about two-stage separation. Change its status from "Active" to "Validated and Superseded" with a note: "R^2~0.66 confirmed the hypothesis. New approach discovers communities directly from electoral shift, validated by temporal holdout. See ADR-005."

- [ ] **Step 3: Write ADR-005**

Create `docs/adr/005-shift-based-community-discovery.md` following the template at `docs/adr/000-template.md`. Key points:
- Context: proof-of-concept validated that community structure predicts electoral covariance
- Decision: invert the approach — define communities by electoral shift correlation, use demographics for description
- Consequences: two-stage separation no longer applies; falsifiability shifts to temporal holdout; existing NMF code shelved
- Status: Accepted

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/ASSUMPTIONS_LOG.md docs/adr/005-shift-based-community-discovery.md
git commit -m "docs: ADR-005 shift-based community discovery, update CLAUDE.md and assumptions"
```

---

## Task 9: Autonomous TODO — Historical VEST Data Expansion

**Files:**
- Create: `src/assembly/fetch_vest_2012_2014.py`
- Modify: `src/assembly/build_shift_vectors.py` (extend to 15 dims when data available)

This task is designed for autonomous execution and should be written as a standalone TODO doc at `docs/TODO-vest-expansion.md` with full acceptance criteria, not implemented now.

- [ ] **Step 1: Write the autonomous TODO document**

Create `docs/TODO-vest-expansion.md` with:
- Goal: pull VEST 2012 + 2014 data, crosswalk to 2020 census tracts
- Acceptance criteria: `vest_tracts_2012.parquet` and `vest_tracts_2014.parquet` exist in `data/assembled/` with same schema as existing VEST files
- Key challenge: 2010→2020 tract boundary crosswalk (Census provides relationship files)
- "Do not do" list: don't modify the clustering pipeline; just produce the data files. `build_shift_vectors.py` will be extended separately to consume them.
- Expected output format: tract_geoid, pres_dem_share_2012, pres_total_2012 (and gov_ equivalents for 2014)

- [ ] **Step 2: Write data source research TODO**

Create `docs/TODO-data-source-research.md` with the full list from spec TODO 2 (RCMS, LODES, IRS SOI, FCC, USDA, school districts, property values, SCI). Acceptance criteria: produce `docs/DATA_SOURCES_EXPANSION.md`.

- [ ] **Step 3: Write local election data research TODO**

Create `docs/TODO-local-election-research.md` with spec TODO 3 (state election offices FL/GA/AL, OpenElections project, academic datasets). Acceptance criteria: produce `docs/LOCAL_ELECTION_DATA.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/TODO-vest-expansion.md docs/TODO-data-source-research.md docs/TODO-local-election-research.md
git commit -m "docs: autonomous TODOs for VEST expansion, data sources, and local elections"
```

---

## Execution Order and Dependencies

```
Task 0 (setup) ─────────────────────────────┐
Task 1 (shift vectors) ◄────────────────────┤
Task 2 (adjacency) ◄────────────────────────┤
Task 3 (clustering) ◄── Task 1 + Task 2     │
Task 4 (borders) ◄── Task 3                 │
Task 5 (description) ◄── Task 3             │
Task 6 (NMF comparison) ◄── Task 3          │
Task 7 (holdout validation) ◄── Task 1 + Task 2 (independent of 3-6, does its own clustering)
Task 8 (docs) ◄── any time after Task 3
Task 9 (autonomous TODOs) ◄── any time, independent
```

Tasks 2, 1 can run in parallel after Task 0. Tasks 4, 5, 6 can run in parallel after Task 3. Task 9 is independent.
