# Sub-County Type Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover electoral types at census tract level (9,393 tracts) with a configurable experiment framework, and visualize as bubble-dissolved community polygons.

**Architecture:** Areal interpolation maps precinct votes onto census tracts. A feature registry provides tagged features (electoral + nonpolitical). A YAML-driven experiment runner selects features, runs KMeans, validates via holdout. Bubble dissolve merges adjacent same-type tracts into community polygons.

**Tech Stack:** Python (geopandas, libpysal, networkx, scikit-learn, shapely), Census TIGER/Line, VEST shapefiles, NYTimes precinct data

**Spec:** `docs/superpowers/specs/2026-03-21-sub-county-type-discovery-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `scripts/fetch_tiger_tracts.py` | Download TIGER/Line 2020 tract shapefiles for FL+GA+AL |
| `src/tracts/__init__.py` | Package init |
| `src/tracts/interpolate_precincts.py` | Areal interpolation: precinct shapefile → tract-level vote allocation |
| `src/tracts/build_tract_features.py` | Compute all tract-level features (electoral + demographic) |
| `src/tracts/feature_registry.py` | Feature metadata registry — name, category, subcategory, source |
| `src/experiments/__init__.py` | Package init |
| `src/experiments/run_experiment.py` | YAML config → feature selection → scaling → clustering → validation → output |
| `src/experiments/compare_runs.py` | Side-by-side comparison of two experiment runs (ARI, NMI, correspondence) |
| `src/viz/bubble_dissolve.py` | Merge adjacent same-type tracts into community polygons |
| `tests/test_interpolation.py` | Tests for areal interpolation |
| `tests/test_tract_features.py` | Tests for feature engineering + registry |
| `tests/test_experiment_runner.py` | Tests for experiment framework |
| `tests/test_bubble_dissolve.py` | Tests for dissolve algorithm |
| `experiments/tract_political_only.yaml` | Config: political features only |
| `experiments/tract_nonpolitical_only.yaml` | Config: nonpolitical features only |

---

## Task 0: Download TIGER/Line Tract Shapefiles

**Files:**
- Create: `scripts/fetch_tiger_tracts.py`

- [ ] **Step 1: Write the download script**

Download TIGER/Line 2020 tract shapefiles for FL (12), GA (13), AL (01):
```
https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{fips}_tract.zip
```

Save to `data/raw/tiger/`. Unzip each into `data/raw/tiger/tl_2020_{fips}_tract/`.

- [ ] **Step 2: Run the script**

```bash
uv run python scripts/fetch_tiger_tracts.py
```

Expected: 3 zip files downloaded, unzipped, shapefiles present.

- [ ] **Step 3: Verify tract counts**

```python
import geopandas as gpd
for fips in ["01", "12", "13"]:
    gdf = gpd.read_file(f"data/raw/tiger/tl_2020_{fips}_tract/")
    print(f"{fips}: {len(gdf)} tracts")
# Expected: AL ~1,181, FL ~4,245, GA ~1,969 ≈ 9,393 total
```

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_tiger_tracts.py
git commit -m "feat: add TIGER/Line tract shapefile downloader"
```

---

## Task 1: Areal Interpolation (Precinct → Tract)

**Files:**
- Create: `src/tracts/__init__.py`
- Create: `src/tracts/interpolate_precincts.py`
- Create: `tests/test_interpolation.py`

- [ ] **Step 1: Write failing tests**

`tests/test_interpolation.py` — use synthetic geometries (rectangles that partially overlap):

Key test cases:
- `test_full_overlap` — precinct entirely within one tract → 100% allocation
- `test_partial_overlap` — precinct spans two tracts 60/40 → votes allocated 60/40
- `test_no_overlap` — precinct outside all tracts → no allocation (dropped)
- `test_multiple_precincts_one_tract` — two precincts contribute to same tract → summed
- `test_output_columns` — output has tract GEOID, votes_dem, votes_rep, votes_total
- `test_crs_reprojection` — input in EPSG:4326 → reprojected to EPSG:5070 for area calculation
- `test_zero_area_precinct` — degenerate geometry handled (no division by zero)
- `test_dem_share_computed` — dem_share = votes_dem / votes_total

Create synthetic GeoDataFrames with shapely boxes for testing — no real data needed.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_interpolation.py -v
```

- [ ] **Step 3: Implement interpolation**

`src/tracts/interpolate_precincts.py`:

```python
def interpolate_precincts_to_tracts(
    precinct_gdf: gpd.GeoDataFrame,
    tract_gdf: gpd.GeoDataFrame,
    vote_columns: list[str] = ["votes_dem", "votes_rep", "votes_total"],
) -> pd.DataFrame:
    """Allocate precinct votes to tracts proportional to area overlap."""
```

Key implementation details:
- Reproject both to EPSG:5070 (NAD83 Conus Albers, meters) for accurate area
- `buffer(0)` on both to fix invalid geometries
- `gpd.overlay(how="intersection")` for geometric intersection
- Area fraction = intersection_area / precinct_area
- Allocate each vote column by area fraction
- Aggregate to tract GEOID via groupby sum
- Compute dem_share = votes_dem / votes_total

Also implement a CLI that processes all available elections:

```python
def interpolate_all_elections(
    vest_dir: Path,       # data/raw/vest/
    nyt_dir: Path,        # data/raw/nyt_precinct/
    tiger_dir: Path,      # data/raw/tiger/
    output_dir: Path,     # data/tracts/
) -> None:
```

For each state + year:
1. Load precinct shapefile (VEST zip or NYTimes GeoJSON/TopoJSON)
2. Load tract shapefile (TIGER)
3. Run interpolation
4. Save to `data/tracts/tract_votes_{year}.parquet`

VEST shapefile loading: unzip on-the-fly, read with geopandas. Vote columns vary by year — inspect `.dbf` columns for patterns like `G16PRERTRU` (2016 president Republican Trump), `G16PREDCLI` (2016 president Democrat Clinton).

NYTimes loading: decompress `.geojson.gz` or `.topojson.gz`, read with geopandas.

CLI: `python -m src.tracts.interpolate_precincts`

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_interpolation.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/tracts/ tests/test_interpolation.py
git commit -m "feat: areal interpolation pipeline (precinct → tract)"
```

---

## Task 2: VEST Column Mapping

**Files:**
- Create: `src/tracts/vest_columns.py`

VEST uses cryptic column names like `G16PRERTRU`, `G18GOVDDES`. We need a mapping layer.

- [ ] **Step 1: Write column mapper**

`src/tracts/vest_columns.py`:

```python
def parse_vest_columns(columns: list[str], state: str, year: int) -> dict[str, str]:
    """Map VEST column names to standardized names.

    VEST convention: G{YY}{RACE}{PARTY}{CANDIDATE}
    - G16PRERTRU → 2016 president Republican Trump
    - G16PREDCLI → 2016 president Democrat Clinton
    - G18GOVDgil → 2018 governor Democrat Gillum
    """
```

Return dict mapping VEST column → standardized name (e.g., `votes_dem`, `votes_rep`, `votes_total`).

This needs to inspect actual VEST shapefiles to determine the column pattern for each state/year. Write a discovery function that reads the shapefile and auto-detects the relevant columns.

- [ ] **Step 2: Test against real VEST data**

```bash
uv run python -c "
import geopandas as gpd
gdf = gpd.read_file('data/raw/vest/fl_2016.zip')
print(gdf.columns.tolist())
"
```

Verify the column mapper handles the actual format.

- [ ] **Step 3: Commit**

```bash
git add src/tracts/vest_columns.py
git commit -m "feat: VEST column name parser for precinct vote extraction"
```

---

## Task 3: Tract Feature Engineering

**Files:**
- Create: `src/tracts/feature_registry.py`
- Create: `src/tracts/build_tract_features.py`
- Create: `tests/test_tract_features.py`

- [ ] **Step 1: Write failing tests**

`tests/test_tract_features.py`:

Key test cases:
- `test_registry_has_all_categories` — electoral, demographic, religion categories present
- `test_registry_feature_count` — at least 50 features registered
- `test_select_by_category` — `select_features(category="electoral")` returns only electoral features
- `test_select_by_subcategory` — `select_features(subcategory="presidential_shifts")` works
- `test_build_electoral_features` — given tract votes, computes shifts, lean, turnout, density
- `test_build_demographic_features` — given ACS tract data, computes all demographic features
- `test_state_centering` — non-presidential shifts have zero state mean after centering
- `test_shift_computation` — log-odds shift computed correctly
- `test_white_working_class` — wwc = white_nh_pct × (1 - ba_plus_pct) interaction term
- `test_output_shape` — combined features have expected columns for each category

- [ ] **Step 2: Implement feature registry**

`src/tracts/feature_registry.py`:

```python
@dataclass
class FeatureSpec:
    name: str
    category: str          # electoral, demographic, religion
    subcategory: str       # presidential_shifts, race_ethnicity, etc.
    source: str            # vest, nyt, acs_tract, rcms_county
    year: int | None       # None for static features
    description: str

FEATURE_REGISTRY: dict[str, FeatureSpec] = { ... }

def select_features(
    category: str | None = None,
    subcategory: str | None = None,
    exclude_year: int | None = None,  # for holdout leakage prevention
) -> list[str]:
    """Return feature names matching the filter criteria."""
```

Register all ~60 features from the spec.

- [ ] **Step 3: Implement tract feature builder**

`src/tracts/build_tract_features.py`:

```python
def build_electoral_features(
    tract_votes: dict[int, pd.DataFrame],  # {year: tract_votes_df}
    tract_gdf: gpd.GeoDataFrame,           # for area calculation (vote density)
    state_fips: dict[str, str],             # tract FIPS → state FIPS prefix
) -> pd.DataFrame:
    """Compute electoral features: shifts, lean, turnout, density."""

def build_demographic_features(
    acs_tract_data: pd.DataFrame,          # ACS 5-year tract data
) -> pd.DataFrame:
    """Compute demographic features from ACS tract data."""

def build_religion_features(
    rcms_county: pd.DataFrame,             # County RCMS features
    tract_fips: pd.Series,                 # tract FIPS for county mapping
) -> pd.DataFrame:
    """Map county RCMS to tracts (proxy)."""

def build_all_features(config: dict) -> pd.DataFrame:
    """Build complete feature matrix based on experiment config."""
```

Shift computation: same log-odds math as county model (`src/assembly/build_county_shifts_multiyear.py`).

State-centering: subtract state mean from non-presidential shifts.

ACS tract data: use existing `fetch_acs.py` (tract-level fetcher already exists in the codebase, just never run).

CLI: `python -m src.tracts.build_tract_features`
Output: `data/tracts/tract_features.parquet` (9,393 × ~60 columns)

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest tests/test_tract_features.py -v
git add src/tracts/feature_registry.py src/tracts/build_tract_features.py tests/test_tract_features.py
git commit -m "feat: tract feature engineering with registry and state-centering"
```

---

## Task 4: Experiment Runner Framework

**Files:**
- Create: `src/experiments/__init__.py`
- Create: `src/experiments/run_experiment.py`
- Create: `tests/test_experiment_runner.py`
- Create: `experiments/tract_political_only.yaml`
- Create: `experiments/tract_nonpolitical_only.yaml`

- [ ] **Step 1: Write failing tests**

`tests/test_experiment_runner.py`:

Key test cases:
- `test_load_config` — YAML config parsed into ExperimentConfig dataclass
- `test_feature_selection_from_config` — config enables/disables feature categories correctly
- `test_feature_weighting` — category weights applied to feature matrix
- `test_holdout_exclusion` — features from holdout year excluded automatically
- `test_kmeans_produces_assignments` — assigns each tract to a type
- `test_min_tracts_per_type` — no type has fewer than min threshold
- `test_output_directory_created` — experiment creates timestamped output dir
- `test_config_frozen` — config.yaml copied to output dir
- `test_validation_json_created` — validation metrics saved
- `test_meta_yaml_created` — metadata (J, holdout_r, timestamp, git_commit) saved

Use small synthetic data (100 tracts × 10 features) for fast tests.

- [ ] **Step 2: Implement experiment runner**

`src/experiments/run_experiment.py`:

```python
@dataclass
class ExperimentConfig:
    name: str
    description: str
    geography_level: str
    features: dict          # parsed from YAML features section
    clustering: dict        # algorithm, j_candidates, etc.
    nesting: dict           # s_candidates, method
    visualization: dict     # bubble_dissolve, min_polygon_area
    holdout: dict           # pairs, metric, min_threshold

def load_config(path: Path) -> ExperimentConfig:
    """Load and validate experiment YAML config."""

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Execute a complete experiment run."""
    # 1. Load tract features from data/tracts/tract_features.parquet
    # 2. Select features by config (category enables + subcategory includes)
    # 3. Exclude holdout-year features (leakage prevention)
    # 4. Apply category weights
    # 5. Min-max scale to [0,1]
    # 6. Apply presidential_weight multiplier
    # 7. J selection: sweep j_candidates, evaluate holdout r for each
    # 8. Run KMeans with best J
    # 9. Compute soft membership (inverse-distance)
    # 10. Nest into super-types
    # 11. Run validation (coherence, holdout accuracy)
    # 12. Save outputs to timestamped directory
```

CLI: `python -m src.experiments.run_experiment --config experiments/tract_political_only.yaml`

- [ ] **Step 3: Write experiment configs**

`experiments/tract_political_only.yaml` — political features only, ~27 dims
`experiments/tract_nonpolitical_only.yaml` — nonpolitical features only, ~29 dims

(Use the configs from the spec verbatim)

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest tests/test_experiment_runner.py -v
git add src/experiments/ tests/test_experiment_runner.py experiments/
git commit -m "feat: YAML-driven experiment runner with holdout validation"
```

---

## Task 5: Bubble Dissolve

**Files:**
- Create: `src/viz/bubble_dissolve.py`
- Create: `tests/test_bubble_dissolve.py`

- [ ] **Step 1: Write failing tests**

`tests/test_bubble_dissolve.py`:

Key test cases (use synthetic tract grids):
- `test_adjacent_same_type_merge` — 4 adjacent tracts of type A → 1 polygon
- `test_non_adjacent_same_type_separate` — 2 type-A tracts separated by type-B → 2 polygons
- `test_different_types_not_merged` — adjacent tracts of different types stay separate
- `test_small_polygon_filtered` — polygon below min_area_sqkm excluded
- `test_output_has_type_columns` — output GeoDataFrame has type_id, super_type, n_tracts, area_sqkm
- `test_geometry_valid` — all output polygons are valid (no self-intersections)
- `test_single_tract_type` — a type with one tract → one small polygon (not filtered if above min area)

Build test data: 4×4 grid of square tracts with known type assignments.

- [ ] **Step 2: Implement bubble dissolve**

`src/viz/bubble_dissolve.py`:

```python
def bubble_dissolve(
    tract_gdf: gpd.GeoDataFrame,      # must have geometry + dominant_type + super_type
    min_area_sqkm: float = 0.1,
    simplify_tolerance: float = 0.001,
) -> gpd.GeoDataFrame:
    """Merge adjacent same-type tracts into community polygons."""
    # 1. Build Queen contiguity graph (libpysal)
    # 2. For each type, find connected components (networkx)
    # 3. Dissolve geometry for each connected component (unary_union)
    # 4. Optionally simplify (Douglas-Peucker) for smaller file size
    # 5. Filter by min area
    # 6. Return GeoDataFrame with type_id, super_type, n_tracts, area_sqkm
```

CLI: `python -m src.viz.bubble_dissolve --input data/experiments/{name}/assignments.parquet --output data/experiments/{name}/dissolved_communities.geojson`

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_bubble_dissolve.py -v
git add src/viz/bubble_dissolve.py tests/test_bubble_dissolve.py
git commit -m "feat: bubble dissolve — merge adjacent same-type tracts into community polygons"
```

---

## Task 6: Comparison Framework

**Files:**
- Create: `src/experiments/compare_runs.py`

- [ ] **Step 1: Implement comparison**

`src/experiments/compare_runs.py`:

```python
def compare_runs(
    run_a_path: Path,     # data/experiments/{name_a}_latest/
    run_b_path: Path,     # data/experiments/{name_b}_latest/
) -> ComparisonResult:
    """Compare two experiment runs on the same tract set."""
    # 1. Load assignments from both runs
    # 2. Compute adjusted_rand_score, normalized_mutual_info_score
    # 3. Build type correspondence matrix (crosstab of dominant types)
    # 4. Save comparison JSON + correspondence matrix
```

CLI: `python -m src.experiments.compare_runs --run-a tract_political_only --run-b tract_nonpolitical_only`

- [ ] **Step 2: Write tests, commit**

```bash
git add src/experiments/compare_runs.py
git commit -m "feat: experiment comparison framework (ARI, NMI, type correspondence)"
```

---

## Task 7: Run Foundational Experiments

- [ ] **Step 1: Download TIGER tracts**

```bash
uv run python scripts/fetch_tiger_tracts.py
```

- [ ] **Step 2: Run areal interpolation for all elections**

```bash
uv run python -m src.tracts.interpolate_precincts
```

This processes: VEST 2016/2018/2020 × 3 states + NYTimes 2024 × 3 states.
Expected runtime: 10-30 minutes (geometric overlays are expensive).

- [ ] **Step 3: Fetch ACS tract data**

```bash
uv run python src/assembly/fetch_acs.py
```

(Existing script, never run — fetches tract-level ACS for FL+GA+AL)

- [ ] **Step 4: Build tract features**

```bash
uv run python -m src.tracts.build_tract_features
```

- [ ] **Step 5: Run political-only experiment**

```bash
uv run python -m src.experiments.run_experiment --config experiments/tract_political_only.yaml
```

- [ ] **Step 6: Run nonpolitical-only experiment**

```bash
uv run python -m src.experiments.run_experiment --config experiments/tract_nonpolitical_only.yaml
```

- [ ] **Step 7: Compare runs**

```bash
uv run python -m src.experiments.compare_runs --run-a tract_political_only --run-b tract_nonpolitical_only
```

- [ ] **Step 8: Review results and commit**

Review the dissolved community GeoJSON files, validation metrics, and comparison ARI. Commit all data artifacts and results.

```bash
git add experiments/ src/ tests/
git commit -m "feat: foundational tract experiments — political vs nonpolitical type discovery"
```

---

## Task Dependencies

```
Task 0 (TIGER download) ─┐
                          ├── Task 1 (areal interpolation) ─┐
Task 2 (VEST columns) ───┘                                  │
                                                             ├── Task 3 (features) ── Task 4 (experiment runner)
                                                             │                                │
                                                             │                                ├── Task 5 (bubble dissolve)
                                                             │                                ├── Task 6 (comparison)
                                                             │                                └── Task 7 (run experiments)
```

**Parallelizable:** Tasks 0+2 (TIGER download + VEST column mapping) are independent. Task 5 (bubble dissolve) and Task 6 (comparison) are independent of each other.
