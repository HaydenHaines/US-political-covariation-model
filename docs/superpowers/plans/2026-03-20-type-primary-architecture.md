# Type-Primary Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pivot the model from HAC geographic communities (K=10 blobs) to SVD+varimax electoral types (J=15-25) as the primary predictive engine, producing a stained glass county map.

**Architecture:** SVD+varimax on 293-county shift vectors discovers J electoral types. Covariance constructed from type demographic profiles (Economist-inspired). Census interpolation provides time-matched demographics. Counties colored by dominant type for stained glass visualization.

**Tech Stack:** Python (scikit-learn, scipy, pandas, numpy), Census API, FastAPI, DuckDB, Next.js + Deck.gl

**Spec:** `docs/superpowers/specs/2026-03-20-type-primary-architecture-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/assembly/fetch_census_decennial.py` | Fetch 2000/2010/2020 decennial census + ACS for FL+GA+AL counties |
| `src/assembly/interpolate_demographics.py` | Linear interpolation of demographics between census years |
| `src/discovery/run_type_discovery.py` | SVD + varimax type discovery pipeline |
| `src/discovery/select_j.py` | J selection via leave-one-pair-out CV |
| `src/discovery/nest_types.py` | Hierarchical nesting of fine types into super-types |
| `src/covariance/construct_type_covariance.py` | Economist-inspired covariance construction from type profiles |
| `src/description/describe_types.py` | Overlay demographics on discovered types |
| `src/prediction/predict_2026_types.py` | Type-based 2026 predictions |
| `tests/test_census_decennial.py` | Tests for census fetch + interpolation |
| `tests/test_type_discovery.py` | Tests for SVD+varimax, J selection, nesting |
| `tests/test_type_covariance.py` | Tests for covariance construction + validation |
| `tests/test_type_prediction.py` | Tests for type-based prediction pipeline |

### Modified Files
| File | Changes |
|------|---------|
| `config/model.yaml` | Add type discovery config (j_candidates, algorithm, super_type_count, lambda_shrinkage) |
| `src/db/build_database.py` | Add type tables + interpolated demographics to DuckDB |
| `api/routers/communities.py` | Rename/refactor to serve types instead of communities |
| `api/routers/forecast.py` | Update to use type-based predictions |
| `api/models.py` | Update Pydantic models for types |
| `web/` (frontend components) | Update map coloring, community panel → type panel |
| `CLAUDE.md` | Update architecture, conventions, commands |

---

## Task 0: Config + Branch Setup

**Files:**
- Modify: `config/model.yaml`

- [ ] **Step 1: Verify on feature branch**

```bash
cd /home/hayden/projects/US-political-covariation-model
git branch  # should be on feat/type-primary-architecture
```

- [ ] **Step 2: Add type discovery config to model.yaml**

Add under a new `type_discovery:` key:

```yaml
type_discovery:
  algorithm: svd_varimax
  fallback_algorithms: [archetypal_analysis, semi_nmf]
  j_candidates: [8, 10, 12, 15, 18, 20, 22, 25]
  super_type_count_candidates: [5, 6, 7, 8]
  lambda_shrinkage: 0.75
  covariance_acceptance_threshold: 0.4
  holdout_r_minimum: 0.85
  holdout_r_target: 0.90
census:
  decennial_years: [2000, 2010, 2020]
  interpolation: true
  api_urls:
    sf1_2000: "https://api.census.gov/data/2000/dec/sf1"
    sf3_2000: "https://api.census.gov/data/2000/dec/sf3"
    sf1_2010: "https://api.census.gov/data/2010/dec/sf1"
    acs5_2010: "https://api.census.gov/data/2010/acs/acs5"
    dhc_2020: "https://api.census.gov/data/2020/dec/dhc"
    acs5_2020: "https://api.census.gov/data/2020/acs/acs5"
```

- [ ] **Step 3: Commit**

```bash
git add config/model.yaml
git commit -m "config: add type discovery and census interpolation settings"
```

---

## Task 1: Census Decennial Fetcher

**Files:**
- Create: `src/assembly/fetch_census_decennial.py`
- Create: `tests/test_census_decennial.py`

- [ ] **Step 1: Write failing tests for census fetch**

`tests/test_census_decennial.py` — test the variable crosswalk, URL construction, and data parsing. Mock HTTP responses.

Key test cases:
- `test_build_url_2000_sf1` — verifies correct URL for 2000 SF1 variables
- `test_build_url_2010_sf1` — verifies correct URL for 2010 SF1 variables
- `test_build_url_2020_dhc` — verifies correct URL for 2020 DHC variables
- `test_parse_census_response` — verifies JSON response → DataFrame conversion
- `test_variable_crosswalk_completeness` — all 7 measures have variables for all 3 years
- `test_fips_column_created` — county_fips is zero-padded 5-digit string
- `test_income_not_in_decennial` — 2010/2020 income comes from ACS, not SF1/DHC
- `test_output_columns` — output has standardized column names regardless of year

Run: `uv run pytest tests/test_census_decennial.py -v`
Expected: All tests FAIL (module not found)

- [ ] **Step 2: Implement census fetcher**

`src/assembly/fetch_census_decennial.py`:

The variable crosswalk (all verified with live API calls):

```python
CROSSWALK = {
    2000: {
        "sf1": {
            "url": "https://api.census.gov/data/2000/dec/sf1",
            "vars": {
                "P001001": "pop_total",
                "P004005": "pop_white_nh",
                "P004006": "pop_black",
                "P004008": "pop_asian",
                "P004002": "pop_hispanic",
                "P013001": "median_age",
                "H001001": "housing_total",
                "H004002": "housing_owner_mortgage",
                "H004003": "housing_owner_free",
            },
        },
        "sf3": {
            "url": "https://api.census.gov/data/2000/dec/sf3",
            "vars": {
                "P053001": "median_hh_income",  # 1999 dollars
                "P037001": "educ_total",
                "P037015": "educ_ba_male", "P037016": "educ_ma_male",
                "P037017": "educ_prof_male", "P037018": "educ_doc_male",
                "P037032": "educ_ba_female", "P037033": "educ_ma_female",
                "P037034": "educ_prof_female", "P037035": "educ_doc_female",
                "P030001": "commute_total",
                "P030003": "commute_car",
                "P030005": "commute_transit",
                "P030016": "commute_wfh",
            },
        },
    },
    2010: {
        "sf1": {
            "url": "https://api.census.gov/data/2010/dec/sf1",
            "vars": {
                "P001001": "pop_total",
                "P005003": "pop_white_nh",
                "P005004": "pop_black",
                "P005006": "pop_asian",
                "P005010": "pop_hispanic",
                "P013001": "median_age",
                "H001001": "housing_total",
                "H004002": "housing_owner_mortgage",
                "H004003": "housing_owner_free",
            },
        },
        "acs5": {
            "url": "https://api.census.gov/data/2010/acs/acs5",
            "vars": {
                "B19013_001E": "median_hh_income",  # 2010 dollars
                "B15002_001E": "educ_total",
                "B15002_015E": "educ_ba_male", "B15002_016E": "educ_ma_male",
                "B15002_017E": "educ_prof_male", "B15002_018E": "educ_doc_male",
                "B15002_032E": "educ_ba_female", "B15002_033E": "educ_ma_female",
                "B15002_034E": "educ_prof_female", "B15002_035E": "educ_doc_female",
                "B08301_001E": "commute_total",
                "B08301_003E": "commute_car",
                "B08301_010E": "commute_transit",
                "B08301_021E": "commute_wfh",
            },
        },
    },
    2020: {
        "dhc": {
            "url": "https://api.census.gov/data/2020/dec/dhc",
            "vars": {
                "P1_001N": "pop_total",
                "P5_003N": "pop_white_nh",
                "P5_004N": "pop_black",
                "P5_006N": "pop_asian",
                "P5_010N": "pop_hispanic",
                "P13_001N": "median_age",
                "H1_001N": "housing_total",
                "H10_002N": "housing_owner",
            },
        },
        "acs5": {
            "url": "https://api.census.gov/data/2020/acs/acs5",
            "vars": {
                "B19013_001E": "median_hh_income",  # 2020 dollars
                "B15002_001E": "educ_total",
                "B15002_015E": "educ_ba_male", "B15002_016E": "educ_ma_male",
                "B15002_017E": "educ_prof_male", "B15002_018E": "educ_doc_male",
                "B15002_032E": "educ_ba_female", "B15002_033E": "educ_ma_female",
                "B15002_034E": "educ_prof_female", "B15002_035E": "educ_doc_female",
                "B08301_001E": "commute_total",
                "B08301_003E": "commute_car",
                "B08301_010E": "commute_transit",
                "B08301_021E": "commute_wfh",
            },
        },
    },
}
```

Main function: `fetch_decennial(year: int) -> pd.DataFrame`
- Fetches from both endpoints for the given year
- Merges on county FIPS
- Standardizes column names (same names regardless of year)
- Computes derived columns: `housing_owner` = mortgage + free (for 2000/2010)
- Computes `educ_bachelors_plus` = sum of 8 education cells
- Saves to `data/assembled/census_{year}.parquet`

CLI: `python -m src.assembly.fetch_census_decennial --year 2000`
Or `--all` to fetch all three years.

Gotchas to handle:
- 2000 housing owner = `H004002 + H004003` (split); 2020 = `H10_002N` (single)
- Income in 1999/2010/2020 dollars — store raw, CPI-adjust in interpolation step
- Census 2000 SF3 API can be flaky — add retry logic (3 retries, 5s backoff)
- NHGIS fallback: if SF3 fails after retries, log error with NHGIS download instructions

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_census_decennial.py -v
```
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/assembly/fetch_census_decennial.py tests/test_census_decennial.py
git commit -m "feat: add decennial census fetcher (2000/2010/2020) with variable crosswalk"
```

---

## Task 2: Census Interpolation

**Files:**
- Create: `src/assembly/interpolate_demographics.py`
- Add tests to: `tests/test_census_decennial.py`

- [ ] **Step 1: Write failing tests for interpolation**

Add to `tests/test_census_decennial.py`:

Key test cases:
- `test_interpolate_midpoint` — 2005 = 50% 2000 + 50% 2010
- `test_interpolate_weighted` — 2008 = 80% 2010 + 20% 2000
- `test_interpolate_at_census_year` — 2010 = 100% 2010
- `test_interpolate_pre_2000` — 1998 uses Census 2000 flat
- `test_interpolate_post_2020` — 2022 uses Census 2020 flat
- `test_income_cpi_adjusted` — 2000 income (1999$) adjusted to common base year
- `test_output_for_all_election_years` — produces row for every election year in model.yaml
- `test_county_count` — 293 counties per year

- [ ] **Step 2: Implement interpolation**

`src/assembly/interpolate_demographics.py`:

```python
def interpolate_demographics(
    election_year: int,
    census_data: dict[int, pd.DataFrame],  # {2000: df, 2010: df, 2020: df}
) -> pd.DataFrame:
    """Linearly interpolate demographics between nearest census years."""
    census_years = sorted(census_data.keys())

    if election_year <= census_years[0]:
        return census_data[census_years[0]].copy()
    if election_year >= census_years[-1]:
        return census_data[census_years[-1]].copy()

    # Find bracketing census years
    earlier = max(y for y in census_years if y <= election_year)
    later = min(y for y in census_years if y > election_year)

    weight_later = (election_year - earlier) / (later - earlier)

    # Interpolate numeric columns
    df_earlier = census_data[earlier]
    df_later = census_data[later]
    # ... merge on county_fips, interpolate numeric cols
```

Also: `build_all_election_year_demographics()` — reads config for all election years, produces one parquet per year or a single long-format parquet keyed by (county_fips, year).

CPI adjustment: use BLS CPI-U annual averages (hardcode the 3 values: CPI 1999, 2010, 2020 relative to 2020 base). Adjust income before interpolation so the linear blend is in constant dollars.

Output: `data/assembled/demographics_interpolated.parquet` — columns: county_fips, year, pct_white_nh, pct_black, pct_asian, pct_hispanic, median_age, median_hh_income_2020, pct_bachelors_plus, pct_owner_occupied, pct_wfh, pct_transit, pct_car.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_census_decennial.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/assembly/interpolate_demographics.py tests/test_census_decennial.py
git commit -m "feat: add demographic interpolation between decennial census years"
```

---

## Task 3: SVD + Varimax Type Discovery

**Files:**
- Create: `src/discovery/run_type_discovery.py`
- Create: `tests/test_type_discovery.py`

- [ ] **Step 1: Write failing tests**

`tests/test_type_discovery.py`:

Key test cases:
- `test_svd_varimax_basic` — 50-county synthetic shift matrix → J=3 types, verify shapes
- `test_varimax_rotation_orthogonal` — rotation matrix is orthogonal
- `test_scores_shape` — rotated scores shape is (n_counties, J)
- `test_loadings_shape` — rotated loadings shape is (J, n_dims)
- `test_dominant_type_assignment` — each county gets argmax(abs(scores)) as dominant type
- `test_type_sizes_asymmetric` — synthetic data with one large group and one small → types reflect this
- `test_explained_variance` — total explained variance preserved after rotation

- [ ] **Step 2: Implement type discovery**

`src/discovery/run_type_discovery.py`:

```python
def discover_types(
    shift_matrix: np.ndarray,
    j: int,
    random_state: int = 42,
) -> TypeDiscoveryResult:
    """SVD + varimax rotation on county shift vectors."""
    # 1. Center
    X = shift_matrix - shift_matrix.mean(axis=0)

    # 2. Truncated SVD
    svd = TruncatedSVD(n_components=j, random_state=random_state)
    scores = svd.fit_transform(X)
    loadings = svd.components_
    explained_variance = svd.explained_variance_ratio_

    # 3. Varimax rotation
    rotated_scores, rotation = varimax(scores)
    rotated_loadings = loadings @ rotation.T

    # 4. Dominant type assignment
    dominant_types = np.argmax(np.abs(rotated_scores), axis=1)

    return TypeDiscoveryResult(
        scores=rotated_scores,
        loadings=rotated_loadings,
        dominant_types=dominant_types,
        explained_variance=explained_variance,
        rotation_matrix=rotation,
    )
```

Include the varimax function from the spec. Use a dataclass for `TypeDiscoveryResult`.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_type_discovery.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/discovery/run_type_discovery.py tests/test_type_discovery.py
git commit -m "feat: SVD + varimax type discovery on county shift vectors"
```

---

## Task 4: J Selection via Leave-One-Pair-Out CV

**Files:**
- Create: `src/discovery/select_j.py`
- Add tests to: `tests/test_type_discovery.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_type_discovery.py`:

- `test_select_j_returns_best` — synthetic data where J=5 is clearly optimal → returns 5
- `test_select_j_holdout_r_computed` — verify holdout r computation for one pair
- `test_select_j_respects_max` — J never exceeds degrees-of-freedom limit
- `test_select_j_results_dataframe` — returns DataFrame with j, mean_holdout_r, explained_var columns

- [ ] **Step 2: Implement J selection**

`src/discovery/select_j.py`:

```python
def select_j(
    shift_matrix: np.ndarray,
    election_pairs: list[tuple[str, str, str]],  # [(d_col, r_col, turnout_col), ...]
    j_candidates: list[int],
    random_state: int = 42,
) -> JSelectionResult:
    """Leave-one-election-pair-out CV to select optimal J."""
    results = []
    for j in j_candidates:
        # Check degrees of freedom
        n_counties, n_dims = shift_matrix.shape
        n_params = j * (n_counties + n_dims)
        if n_counties * n_dims / n_params < 1.5:
            continue

        holdout_rs = []
        for pair_idx, pair_cols in enumerate(election_pairs):
            # Hold out 3 columns for this pair
            train_cols = [i for i in range(n_dims) if i not in pair_col_indices]
            holdout_cols = pair_col_indices

            # Fit SVD+varimax on training dims
            result = discover_types(shift_matrix[:, train_cols], j, random_state)

            # Predict holdout: reconstruct via type means
            # ... compute holdout Pearson r
            holdout_rs.append(r)

        results.append({"j": j, "mean_holdout_r": np.mean(holdout_rs), ...})

    # Select J maximizing mean holdout r
    best = max(results, key=lambda x: x["mean_holdout_r"])
    return JSelectionResult(best_j=best["j"], all_results=pd.DataFrame(results))
```

Read `election_pairs` from `config/model.yaml` — the pairs are already defined there (presidential_pairs, governor_pairs, senate_pairs). Map each pair to its 3 column indices in the shift matrix.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_type_discovery.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/discovery/select_j.py tests/test_type_discovery.py
git commit -m "feat: J selection via leave-one-pair-out cross-validation"
```

---

## Task 5: Hierarchical Type Nesting

**Files:**
- Create: `src/discovery/nest_types.py`
- Add tests to: `tests/test_type_discovery.py`

- [ ] **Step 1: Write failing tests**

- `test_nest_types_count` — J=20 types nested into S=6 super-types → 6 groups
- `test_nest_types_all_assigned` — every fine type maps to exactly one super-type
- `test_nest_types_silhouette` — silhouette scores computed for each S candidate
- `test_nest_types_mapping` — returns dict mapping fine_type_id → super_type_id

- [ ] **Step 2: Implement nesting**

`src/discovery/nest_types.py`:

```python
def nest_types(
    type_loadings: np.ndarray,  # J × D rotated loadings
    s_candidates: list[int] = [5, 6, 7, 8],
) -> NestingResult:
    """Cluster fine types into super-types via Ward HAC."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    results = []
    for s in s_candidates:
        hac = AgglomerativeClustering(n_clusters=s, linkage="ward")
        labels = hac.fit_predict(type_loadings)
        score = silhouette_score(type_loadings, labels)
        results.append({"s": s, "labels": labels, "silhouette": score})

    # Return all results for manual review; recommend highest silhouette
    best = max(results, key=lambda x: x["silhouette"])
    mapping = {i: int(best["labels"][i]) for i in range(len(best["labels"]))}
    return NestingResult(mapping=mapping, all_results=results)
```

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_type_discovery.py -v
git add src/discovery/nest_types.py tests/test_type_discovery.py
git commit -m "feat: hierarchical nesting of fine types into super-types"
```

---

## Task 6: Type Description (Demographics Overlay)

**Files:**
- Create: `src/description/describe_types.py`
- Create: `tests/test_type_description.py`

- [ ] **Step 1: Write failing tests**

- `test_describe_types_columns` — output has type_id, all demographic columns, n_counties
- `test_describe_types_weighted` — demographics are population-weighted within type
- `test_describe_types_uses_interpolated` — when election_year given, uses interpolated demographics
- `test_describe_types_all_types_present` — every type from discovery appears in output

- [ ] **Step 2: Implement type description**

`src/description/describe_types.py`:

Build type profiles by taking population-weighted means of county demographics within each type. Uses SVD scores as weights (absolute value for weighting, sign preserved separately).

Inputs: county demographic DataFrame (from interpolation), SVD scores, dominant type assignments.
Output: `data/communities/type_profiles.parquet` — J rows × demographic columns.

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_type_description.py -v
git add src/description/describe_types.py tests/test_type_description.py
git commit -m "feat: type description with time-matched demographics overlay"
```

---

## Task 7: Type Covariance Construction

**Files:**
- Create: `src/covariance/construct_type_covariance.py`
- Create: `tests/test_type_covariance.py`

- [ ] **Step 1: Write failing tests**

- `test_covariance_shape` — J × J matrix
- `test_covariance_symmetric` — matrix is symmetric
- `test_covariance_positive_definite` — all eigenvalues > 0
- `test_covariance_diagonal_ones` — correlation matrix has 1s on diagonal
- `test_shrinkage_floor` — minimum off-diagonal >= (1-lambda) with lambda=0.75
- `test_validation_against_observed` — correlation of off-diagonals computed correctly
- `test_hybrid_fallback_triggered` — when off-diagonal r < 0.4, hybrid is used
- `test_hybrid_fallback_not_triggered` — when r >= 0.4, constructed matrix returned as-is

- [ ] **Step 2: Implement covariance construction**

`src/covariance/construct_type_covariance.py`:

Follow the spec exactly:
1. Build type profile matrix from demographics + RCMS + FEC
2. Min-max scale
3. Pearson correlation
4. Floor negatives (configurable via OQ-N1)
5. Shrink toward all-1s (lambda from config)
6. Spectral truncation for PD
7. Validate against observed comovement
8. Hybrid fallback if r < threshold

Also implement `validate_covariance()` as a separate function.

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_type_covariance.py -v
git add src/covariance/construct_type_covariance.py tests/test_type_covariance.py
git commit -m "feat: Economist-inspired type covariance construction with validation"
```

---

## Task 8: Type-Based Prediction Pipeline

**Files:**
- Create: `src/prediction/predict_2026_types.py`
- Create: `tests/test_type_prediction.py`

- [ ] **Step 1: Write failing tests**

- `test_predict_produces_county_estimates` — 293 rows output
- `test_predict_dem_share_bounded` — all predictions in [0, 1]
- `test_predict_has_uncertainty` — CI columns present
- `test_predict_uses_type_covariance` — mock covariance affects output
- `test_predict_multiple_races` — FL Senate, FL Governor, GA Governor, GA Senate all work
- `test_poll_update_shifts_predictions` — feeding a poll changes the predictions

- [ ] **Step 2: Implement type-based prediction**

`src/prediction/predict_2026_types.py`:

Adapted from `predict_2026_hac.py` but uses:
- Type membership (SVD scores) instead of community assignments
- Type covariance instead of community covariance
- Type-level prior means instead of community-level

Key function: `predict_race(race, poll_data, type_scores, type_covariance, type_priors) -> pd.DataFrame`

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_type_prediction.py -v
git add src/prediction/predict_2026_types.py tests/test_type_prediction.py
git commit -m "feat: type-based 2026 prediction pipeline"
```

---

## Task 9: DuckDB Schema Update

**Files:**
- Modify: `src/db/build_database.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_db_builder.py`:
- `test_types_table_exists` — DuckDB has `types` table
- `test_county_type_assignments_table` — has soft membership columns
- `test_super_types_table` — has nesting mapping
- `test_type_covariance_table` — has J×J matrix
- `test_interpolated_demographics_table` — has per-year demographics

- [ ] **Step 2: Extend build_database.py**

Add tables:
- `types` — type_id, super_type_id, display_name, demographic profile columns
- `county_type_assignments` — county_fips, type scores (one column per type)
- `super_types` — super_type_id, display_name, member_type_ids
- `type_covariance` — J×J matrix as type_i, type_j, correlation columns
- `demographics_interpolated` — county_fips, year, all demographic columns

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_db_builder.py -v
git add src/db/build_database.py tests/test_db_builder.py
git commit -m "feat: add type tables and interpolated demographics to DuckDB"
```

---

## Task 10: API Updates

**Files:**
- Modify: `api/routers/communities.py` → refactor for types
- Modify: `api/routers/forecast.py`
- Modify: `api/models.py`
- Modify: `api/main.py`

- [ ] **Step 1: Update Pydantic models**

Update `api/models.py` to add `TypeSummary`, `TypeDetail`, `SuperTypeSummary` response models.

- [ ] **Step 2: Update communities router to serve types**

Rename or refactor `GET /communities` → `GET /types` (keep `/communities` as alias for backward compat).
`GET /types/{id}` returns type detail with member counties, demographics, shift profile.

- [ ] **Step 3: Update forecast router**

Update `POST /forecast/poll` to use `predict_2026_types` instead of `predict_2026_hac`.
Load type covariance and scores at startup in `api/main.py` lifespan.

- [ ] **Step 4: Run API tests, commit**

```bash
uv run pytest api/tests/ -v
git add api/
git commit -m "feat: update API to serve type-based data and predictions"
```

---

## Task 11: Frontend Stained Glass Map

**Files:**
- Modify: `web/` — map layer, community panel → type panel, color scheme

- [ ] **Step 1: Update map coloring**

Change the Deck.gl GeoJsonLayer `getFillColor` to use dominant super-type ID instead of community ID. Use 5-8 distinct colors for super-types.

- [ ] **Step 2: Update type panel**

Replace CommunityPanel with TypePanel showing:
- Type name and super-type
- Demographic profile (from API)
- Member counties list
- Shift history sparkline

- [ ] **Step 3: Update forecast view**

Wire forecast to use updated API endpoints.

- [ ] **Step 4: Rebuild and deploy**

```bash
cd web && npm run build
cp -r .next/static .next/standalone/.next/static
cp -r public .next/standalone/public
systemctl --user restart bedrock-frontend
```

- [ ] **Step 5: Verify with Playwright**

Navigate to `https://bedrock.hhaines.duckdns.org` and verify:
- Map shows stained glass pattern (many individual counties, few colors)
- Type panel displays correctly
- Forecast updates work

- [ ] **Step 6: Commit**

```bash
git add web/
git commit -m "feat: stained glass map with type-based coloring"
```

---

## Task 12: Validation Suite

**Files:**
- Create: `src/validation/validate_types.py`
- Create: `tests/test_type_validation.py`

- [ ] **Step 1: Implement type validation**

`src/validation/validate_types.py`:

Functions:
- `type_coherence(scores, shift_matrix, holdout_cols)` — within vs between type variance
- `type_stability(shift_matrix, window_a_cols, window_b_cols, j)` — subspace angle
- `covariance_validation(constructed, observed)` — off-diagonal correlation
- `holdout_accuracy(scores, shift_matrix, holdout_cols)` — Pearson r on holdout
- `generate_type_validation_report()` — runs all checks, saves report

- [ ] **Step 2: Write tests, run, commit**

```bash
uv run pytest tests/test_type_validation.py -v
git add src/validation/validate_types.py tests/test_type_validation.py
git commit -m "feat: type validation suite (coherence, stability, covariance, holdout)"
```

---

## Task 13: Run Full Pipeline + Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run census fetch**

```bash
uv run python -m src.assembly.fetch_census_decennial --all
```

- [ ] **Step 2: Run interpolation**

```bash
uv run python -m src.assembly.interpolate_demographics
```

- [ ] **Step 3: Run J selection sweep**

```bash
uv run python -m src.discovery.select_j
```

Review output — select best J.

- [ ] **Step 4: Run type discovery with selected J**

```bash
uv run python -m src.discovery.run_type_discovery
```

- [ ] **Step 5: Run type description**

```bash
uv run python -m src.description.describe_types
```

- [ ] **Step 6: Run covariance construction**

```bash
uv run python -m src.covariance.construct_type_covariance
```

- [ ] **Step 7: Run prediction**

```bash
uv run python -m src.prediction.predict_2026_types
```

- [ ] **Step 8: Rebuild DuckDB**

```bash
uv run python src/db/build_database.py --reset
```

- [ ] **Step 9: Run full validation**

```bash
uv run python -m src.validation.validate_types
```

- [ ] **Step 10: Run all tests**

```bash
uv run pytest -v
```

All must pass.

- [ ] **Step 11: Update CLAUDE.md**

Update architecture section, key decisions log, directory map, commands, and conventions to reflect the type-primary architecture. Remove references to K=10 HAC communities as primary. Add ADR-006 reference.

- [ ] **Step 12: Final commit**

```bash
git add -A
git commit -m "feat: complete type-primary architecture pivot — end-to-end pipeline"
```

---

## Task Dependencies

```
Task 0 (config) ──┬── Task 1 (census fetch) ── Task 2 (interpolation) ──┐
                   │                                                      │
                   ├── Task 3 (SVD+varimax) ── Task 4 (J selection) ──── Task 6 (description) ── Task 7 (covariance)
                   │                            │                                                  │
                   │                            └── Task 5 (nesting)                               │
                   │                                                                               │
                   │                                                              Task 8 (prediction) ── Task 9 (DuckDB)
                   │                                                                               │
                   │                                                              Task 10 (API) ── Task 11 (frontend)
                   │
                   └── Task 12 (validation) depends on Tasks 3,4,7
                       Task 13 (full pipeline) depends on ALL above
```

**Parallelizable:** Tasks 1-2 (census) and Tasks 3-5 (type discovery) can run in parallel.
