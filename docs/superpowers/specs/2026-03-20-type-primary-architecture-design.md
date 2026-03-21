# Design Spec: Type-Primary Architecture Pivot

**Date:** 2026-03-20
**Status:** Draft
**Supersedes:** HAC community-primary architecture (ADR-005 community layer remains but role changes)
**Requires:** ADR-006 (type-primary pivot — to be written during implementation)

---

## Problem Statement

The current model produces K=10 geographically contiguous community blobs via Ward HAC with spatial contiguity constraint. The resulting map looks like 10 "alternative states" — large monolithic regions. This is wrong. The desired output is a **stained glass window**: many small geographic units (counties now, small tract clusters later), each colored by one of J electoral types that represent genuinely distinct voting behavior patterns.

The current architecture makes geographic communities the primary predictive engine (covariance estimated at community level, predictions flow through communities). The pivot makes **types** the primary engine.

---

## Design Goals

1. **Stained glass visualization**: Each county is an individual piece of glass, colored by its dominant electoral type
2. **Types as the predictive engine**: Covariance estimation, poll propagation, and prediction flow through type structure
3. **Asymmetric type sizes are expected**: "Rural Conservative" may cover 40% of counties; "Asian Professional" may cover 2%
4. **Hierarchical interpretability**: J=15-25 fine types nest into 5-8 super-types for public communication
5. **Census interpolation**: Time-matched demographics for each election year via decennial census interpolation
6. **Extensible to tracts**: Architecture supports future tract-level expansion where small spatial clusters replace counties as the unit of analysis

---

## Architecture

### Layer Roles (Inverted from Current)

| | Current | New |
|--|---------|-----|
| **Primary** | K=10 geographic communities (HAC) | J=15-25 electoral types (SVD + varimax on county shifts) |
| **Secondary** | J=7 types (NMF on community profiles) | Geographic clusters (HAC, deferred to tract phase) |
| **Covariance** | Stan factor model on K communities | Constructed from type demographic profiles (Economist approach) |
| **Prediction** | Community-level estimates | Type-level estimates, mapped to counties via soft membership |
| **Visualization** | Counties colored by community blob | Counties colored by dominant type (stained glass) |

### Pipeline (Revised)

```
1. Data Assembly
   ├── Election returns (MEDSL pres 2000-2024, Algara gov 2002-2018, Senate 2002-2022)
   ├── Decennial Census (2000, 2010, 2020) + interpolation for election years
   ├── ACS 5-year (2022, existing)
   ├── RCMS, IRS migration, FEC donor density (existing + new)
   └── NYTimes precinct data (2020, 2024) — for future tract expansion

2. Shift Vector Computation
   └── Log-odds shift vectors per county. Current: 54 dims breakdown:
       ├── 5 presidential pairs (00→04, 04→08, 08→12, 12→16, 16→20) × 3 (D/R/turnout) = 15
       ├── 4 governor pairs (02→06, 06→10, 10→14, 14→18) × 3 = 12
       ├── 8 Senate pairs (02→08, 04→10, ...) × 3 = 24
       └── Holdout: pres 20→24 × 3 = 3 (excluded from training = 51 training dims)
       Extend with older governor cycles (1994, 1998) for additional training dims.

3. Type Discovery (NEW — replaces HAC as primary)
   ├── SVD + varimax on 293 × 51 county shift matrix (training dims only)
   │   ├── Truncated SVD → J components
   │   ├── Varimax rotation → interpretable type loadings
   │   └── County scores = soft membership (can be negative = anti-correlated)
   ├── J selection via leave-one-election-pair-out CV, sweep J=10..25
   ├── Hierarchical nesting: cluster J fine types into 5-8 super-types
   └── Stability analysis: compare types from 2000-2012 window vs 2012-2024 window

4. Type Description
   ├── Overlay time-matched demographics (interpolated census) onto types
   ├── Overlay RCMS, IRS migration, FEC donor density
   ├── Name types from demographic + behavioral character
   └── Super-type naming for public communication

5. Type Covariance Construction (NEW — Economist-inspired)
   ├── ASSUMPTION: Types with similar demographic/behavioral profiles comove electorally.
   │   This is a proxy, not an identity. Validated empirically in Step 5c.
   ├── a. Assemble type profile vectors (~15-20 features: demographics, urbanicity,
   │      evangelical %, donor density, etc.)
   ├── b. Compute Pearson correlation across types from profiles → J × J matrix
   │      ├── Floor negative correlations to zero (debatable — see OQ-N1)
   │      ├── Shrink: C_final = lambda * C_demographic + (1-lambda) * C_ones
   │      │   └── lambda ≈ 0.75 (75% demographic similarity, 25% national swing)
   │      └── Ensure positive definiteness (spectral truncation)
   ├── c. Validate against observed comovement in historical elections
   │      ├── Compute actual type-level shifts per election → sample covariance
   │      ├── Correlation of off-diagonal elements between constructed and observed
   │      ├── THRESHOLD: off-diagonal r >= 0.4 → accept constructed matrix
   │      └── If r < 0.4 → trigger hybrid (see 5d)
   ├── d. Hybrid fallback (if validation fails):
   │      ├── Use constructed matrix as prior mean
   │      ├── Bayesian update: C_hybrid = w * C_constructed + (1-w) * C_observed
   │      ├── w = max(0.3, r_offdiag) — scale trust by how well construction matches
   │      └── Re-validate; if still poor, fall back to pure Stan factor model (existing code)
   └── e. Scale for three uses: polling bias, prior uncertainty, random walk innovation

6. Poll Propagation
   ├── Reverse random walk anchored at election day (Economist approach)
   ├── Binomial-logit likelihood on raw poll counts
   ├── Type soft membership decomposes state polls into type-level signals
   └── Covariance propagates signal across types

7. Prediction
   ├── Type-level estimates × county membership weights → county predictions
   ├── Dual output: vote share + turnout
   └── Uncertainty from posterior + covariance structure

8. Validation
   ├── Leave-one-election-pair-out CV for J selection
   ├── Type coherence: within-type variance vs between-type variance on holdout shifts
   ├── Type stability: subspace angle between time-window NMF solutions
   ├── Effective number of types (entropy of membership weights)
   ├── Calibration: 80% CI coverage target 75-85%
   └── Variation partitioning: demographics vs types vs overlap vs residual
```

### Census Interpolation

**Problem:** Elections span 2000-2024 but demographics are pinned to ACS 2022. A county undergoing rapid demographic change (e.g., exurban growth) gets mischaracterized when its 2004 election is described with 2022 demographics.

**Solution:** Fetch decennial census (SF1/SF3) for 2000, 2010, and 2020 at county level for FL+GA+AL. For any election year, interpolate linearly between the two nearest census points:

```
weight_later = (election_year - earlier_census) / (later_census - earlier_census)
demographics(election_year) = (1 - weight_later) * census_earlier + weight_later * census_later
```

Examples:
- 2008 election → 80% Census 2010 + 20% Census 2000
- 2014 election → 40% Census 2020 + 60% Census 2010
- 2002 election → 20% Census 2010 + 80% Census 2000

Note: This is interpolation, not prediction — both census endpoints are known values used to estimate the demographic state at any interior point. A 2008 estimate uses the 2010 census (which hadn't occurred yet in 2008) because we're reconstructing what conditions *were*, not what was *known* at the time.

**Variables to fetch** (matching existing ACS pipeline):
- Population by race/ethnicity (total, white NH, Black, Asian, Hispanic)
- Median age (may need approximation from age brackets in SF1)
- Median household income (SF3 for 2000; ACS for 2010+)
- Housing tenure (owner/renter)
- Educational attainment (SF3 for 2000)
- Commute mode (SF3 for 2000)

**Priority rule:** If existing pipeline data (e.g., ACS 5-year estimates) already provides better temporal resolution for a variable, use that instead of interpolated decennial data. The interpolation is a floor, not a ceiling.

**Edge cases:**
- Pre-2000 elections: use Census 2000 flat (no extrapolation backward)
- Post-2020 elections: use Census 2020 flat until 2030 census (ACS 5-year may be better here — prefer ACS when available)
- FIPS code changes between censuses: use crosswalk (Census Bureau provides these)

**Use cases:**
- **Type description** (immediate): describe what a type looked like during each election era
- **Type covariance construction** (immediate): type profiles use time-matched demographics
- **Potential model feature** (future): demographics as covariates in type discovery or covariance estimation

### Microdonation / Donor Density Feature

**Source:** FEC individual contributions data (existing `fetch_fec_contributions.py` partially implements this)

**Two signals:**
1. **Donor density** (pct of county population who made any political donation): Type descriptor, not shift dimension. Correlates with political engagement intensity. Available post-2004 (ActBlue founding), reliable post-2012 (microdonation explosion).
2. **Partisan donor ratio shift** (ActBlue share of total donors, shifted across cycles): Experimental shift dimension for post-2012 elections only. Controls for platform adoption effect. Must be clearly separated from pre-2012 data where infrastructure didn't exist.

**Usage:**
- Donor density → type profile feature for covariance construction
- Partisan donor ratio shift → experimental: test whether adding it to shift vectors improves holdout accuracy for post-2012 election pairs

### Visualization: Stained Glass Map

**Current:** Counties colored by community blob ID (10 colors, 10 giant regions)

**New:** Counties colored by dominant super-type (5-8 colors, 293 individual pieces)

Each county is its own piece of glass. The color comes from its dominant super-type assignment. Fine-type information available on hover/click. The map should look like a stained glass window — many small regions, few colors, patterns that emerge from the political geography (rural types spanning large areas, urban types in small concentrated spots).

**Hover/click detail:**
- County name, state
- Dominant type and super-type
- Full soft membership vector (pie chart or bar)
- Time-matched demographics for selected election
- Shift history sparkline

**Future (tract level):** When tracts replace counties, individual tracts become the glass pieces. At that scale, optional spatial smoothing (small HAC clusters of 3-10 tracts) may improve visual coherence without losing the stained glass character.

---

## Type Discovery: Technical Details

### Algorithm: SVD + Varimax Rotation (Primary)

SVD + varimax is chosen because log-odds shift vectors contain negative values (-2 to +2 range). NMF requires non-negative input, and offsetting destroys the signal. SVD handles the full range naturally. Varimax rotation produces interpretable type loadings analogous to factor analysis.

**MODEL CHOICE RISK:** If SVD + varimax produces types that are hard to interpret or that don't validate well, revisit with: (a) archetypal analysis (convex combinations, handles negatives, membership sums to 1), (b) semi-NMF via `nimfa` (negative H, non-negative W), or (c) HDBSCAN on UMAP-reduced shifts. This is a first-revisit point if results are unsatisfactory.

```python
from sklearn.decomposition import TruncatedSVD
from scipy.stats import ortho_group
import numpy as np

# 1. Center the shift matrix (important for SVD interpretability)
X = shift_matrix - shift_matrix.mean(axis=0)  # 293 × D (D = training dims)

# 2. Truncated SVD
svd = TruncatedSVD(n_components=J, random_state=42)
scores = svd.fit_transform(X)       # 293 × J county scores (membership)
loadings = svd.components_           # J × D type shift profiles

# 3. Varimax rotation for interpretability
def varimax(Phi, gamma=1.0, max_iter=500, tol=1e-6):
    """Varimax rotation of factor loading matrix."""
    p, k = Phi.shape
    R = np.eye(k)
    for _ in range(max_iter):
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0)))
        )
        R_new = u @ vt
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new
    return Phi @ R, R

# Apply varimax to county scores
rotated_scores, rotation = varimax(scores)  # 293 × J rotated membership
rotated_loadings = loadings @ rotation.T     # J × D rotated type profiles
```

**Interpretation of scores:**
- Positive score: county is aligned with this type's shift pattern
- Negative score: county shifts in the *opposite* direction from this type
- Near-zero: county is unrelated to this type
- For visualization (dominant type coloring), use `argmax(abs(rotated_scores))` with sign indicating alignment

**Dominant type assignment for stained glass map:**
- Each county's dominant type = the type with highest absolute score
- Color = super-type of the dominant type
- Sign preserved in metadata (aligned vs anti-aligned)

### J Selection Protocol

1. For each candidate J in {8, 10, 12, 15, 18, 20, 22, 25}:
   a. For each election pair p in the 17 available pairs:
      - Hold out pair p (3 dims: D-shift, R-shift, turnout-shift)
      - Fit SVD + varimax on remaining dims (D-3 training dims)
      - Predict held-out shifts: project held-out columns onto the rotated type space,
        reconstruct via type-level means weighted by county scores
      - Compute holdout Pearson r
   b. Average holdout r across all held-out pairs
2. Select J maximizing mean holdout r, subject to:
   - Explained variance ratio > 0.70 (types capture meaningful structure)
   - No single component explains > 40% of variance (avoid dominance)
   - Scree plot elbow consistent with selected J

**Degrees of freedom note:** SVD parameters scale as J*(293+D), similar to NMF. With D=51 training dims: J=20 → 6,880 params from 14,943 data points (ratio 2.2:1, acceptable). J=25 → 8,600 params (ratio 1.7:1, borderline). J > 25 not recommended for 293 counties.

### Hierarchical Nesting

After J is selected and SVD + varimax is fit:

1. Compute type centroids from rotated loadings (J × D shift profiles)
2. Cluster types into S super-types using Ward HAC (no spatial constraint — types are abstract)
3. Sweep S = 5, 6, 7, 8; select for interpretability (manual review + silhouette score)
4. Each fine type maps to exactly one super-type
5. Super-types are named from their constituent fine types' demographic character

### Type Stability Analysis

Critical validation — tests whether types are structural or period-dependent:

1. Split shift dimensions into two windows:
   - Window A: 2000-2012 pairs (presidential + governor + Senate)
   - Window B: 2012-2024 pairs (presidential + governor + Senate)
2. Fit SVD + varimax separately on each window
3. Compute subspace angle between rotated score matrices W_A and W_B
4. If angle > 30°: types are period-dependent, model may not generalize
5. If angle < 15°: types are structurally stable, high confidence

---

## Type Covariance: Technical Details (Economist-Inspired)

### Construction (Not Estimation)

The key insight from the Economist model: **construct** the type covariance matrix from type profiles rather than **estimating** it from too-few election observations.

**Core assumption:** Types with similar demographic and behavioral profiles comove electorally. This is a proxy — demographic similarity is not identical to electoral comovement. The assumption is validated empirically in the validation step below, with a quantitative threshold and fallback.

```python
import numpy as np

# 1. Assemble type profile matrix: J types × F features
#    Features: time-matched demographics (interpolated census), urbanicity,
#    evangelical %, donor density, prior vote share
#    Each type's profile = population-weighted mean of its member counties' features
type_profiles = build_type_profiles(rotated_scores, demographics, rcms, fec)  # J × F

# 2. Min-max scale each feature to [0, 1]
type_profiles_scaled = minmax_scale(type_profiles, axis=0)

# 3. Compute correlation across types (treating features as observations)
C_demo = np.corrcoef(type_profiles_scaled)  # J × J

# 4. Floor negative correlations (optional — see OQ-N1)
#    NOTE: with lambda=0.75 shrinkage, minimum correlation = (1-0.75)*1 = 0.25
#    This means every type pair has at least 25% correlation (national swing floor)
C_demo = np.maximum(C_demo, 0)

# 5. Shrink toward all-1s (national swing component)
lam = 0.75
C_final = lam * C_demo + (1 - lam) * np.ones((J, J))

# 6. Ensure positive definiteness
eigvals, eigvecs = np.linalg.eigh(C_final)
eigvals = np.maximum(eigvals, 1e-6)
C_final = eigvecs @ np.diag(eigvals) @ eigvecs.T

# 7. Create base covariance (uniform sigma, modulated by correlation)
sigma_base = 0.07  # logit-scale standard deviation per type
Sigma_base = sigma_base**2 * C_final
```

### Validation of Constructed Matrix

The constructed matrix must predict observed comovement:

1. For each historical election, compute actual type-level shifts (weighted mean of county shifts within each type, using SVD scores as weights)
2. Compute sample covariance of these type-level shifts across elections
3. Compare constructed matrix to sample covariance:
   - Frobenius norm of difference
   - **Correlation of off-diagonal elements (primary metric)**
   - Rank-ordering: do types we predict are most correlated actually comove most?

**Acceptance threshold:** Off-diagonal correlation r >= 0.4 between constructed and observed matrices → accept constructed matrix.

**Hybrid fallback (if r < 0.4):**
1. Use constructed matrix as prior mean
2. Blend: `C_hybrid = w * C_constructed + (1-w) * C_observed`
3. `w = max(0.3, r_offdiag)` — scale trust by how well construction matches observation
4. Ensure positive definiteness after blending
5. If hybrid still poor (r < 0.3 after blending), fall back to Stan factor model (existing code in `src/covariance/`), which estimates covariance directly from election observations

### Stan/R Role Going Forward

The Economist-inspired covariance construction is **deterministic Python** — no Stan sampling required for the covariance matrix itself. This simplifies the critical path:

- **Stan factor model**: retained in codebase as comparison/fallback. Not on the critical path unless hybrid fallback is triggered.
- **R + Stan MRP**: remains deferred to post-MVP (same as current architecture). The Python Gaussian/Kalman update is sufficient for state-level poll propagation.
- **Uncertainty**: The constructed covariance is a point estimate (no posterior distribution on the matrix itself). Prediction uncertainty comes from: (a) poll measurement error, (b) the base covariance scaling for prior uncertainty, and (c) the random walk innovation term. This matches the Economist's approach — they also pass a fixed covariance to Stan.
- **If full Bayesian covariance uncertainty is needed later**: the Stan factor model code exists and can be adapted to operate on types instead of communities.

---

## Data Sources: New and Extended

### New: Decennial Census (2000, 2010, 2020)

| Census | API | Key Tables | Notes |
|--------|-----|------------|-------|
| 2000 | `https://api.census.gov/data/2000/sf1` | P003 (race), P012 (age/sex) | Income/education in SF3 (`/2000/sf3`) |
| 2010 | `https://api.census.gov/data/2010/dec/sf1` | P5 (race/Hispanic), P13 (age) | Income/education in ACS 2010 5-year |
| 2020 | `https://api.census.gov/data/2020/dec/dhc` | P5 (race/Hispanic), P14 (age/sex) | Detailed DHC tables |

Variable crosswalk will be needed — table numbers changed between censuses.

### New: FEC Donor Density

Extend existing `fetch_fec_contributions.py`:
- Count unique donors per county per cycle (not dollar amounts)
- Compute donor density: unique_donors / county_population
- Compute partisan ratio: ActBlue_donors / total_donors
- Available 2004-2024; reliable signal 2012+

### Extended: Algara & Amlani Governor Pre-2000

Currently using 2002-2018. Can extend to 1994, 1998 (and earlier) for FL+GA+AL. Adds 2-4 more shift pairs to the training set.

### Future: NYTimes Precinct Data

- 2020 presidential (MIT license): FL + GA full coverage, AL unusable
- 2024 presidential (C-UDA non-commercial): FL + GA + AL full coverage
- With shapefiles — enables tract-level analysis when that phase begins

### Future: CES/CCES Survey Data

Individual-level validated vote + county geography. Essential for validating type assignments against individual behavior. ~60K respondents per wave. Priority for post-MVP validation.

---

## Open Questions

| ID | Question | When to Answer |
|----|----------|----------------|
| OQ-N1 | Should we floor negative type correlations? The Economist does, but our types may have genuinely inverse relationships (rural evangelical vs urban progressive). Preserving negatives may improve prediction. Note: with lambda=0.75, minimum correlation after shrinkage = 0.25 regardless of flooring. | During covariance construction |
| OQ-N2 | What lambda for shrinkage toward all-1s? Economist uses 0.75. Our types are more granular than states — may need different lambda. | Empirical sweep |
| OQ-N3 | Should donor density use total donors or microdonors only (< $200)? Microdonors are the behavioral signal; large donors are a different population. | During FEC data exploration |
| OQ-N4 | If SVD + varimax produces types that are hard to interpret or validate poorly, which fallback algorithm? Candidates: archetypal analysis, semi-NMF (nimfa), HDBSCAN on UMAP. See Model Choice Risk. | After initial J sweep |
| OQ-N5 | Should pre-2000 governor data be included in shift vectors? Adds training dims but introduces very old political patterns (pre-realignment). | After type stability analysis |
| OQ-N6 | Census 2000 SF3 API reliability — has been flaky in recent years. NHGIS (bulk CSV downloads) is a reliable fallback for income/education variables. | During census data fetch implementation |

---

## Migration Path

### What Changes

| Component | Current | New | Migration |
|-----------|---------|-----|-----------|
| `src/discovery/cluster.py` | Ward HAC → K communities | Still exists but secondary | Retain for future tract smoothing |
| `src/discovery/run_county_clustering.py` | Primary pipeline entry | Secondary | New `run_type_discovery.py` becomes primary |
| `src/description/describe_communities.py` | Overlays on communities | Overlays on types | Refactor to accept type assignments |
| `src/covariance/` | Stan factor model on communities | Demographic construction on types | New `construct_type_covariance.py` |
| `src/propagation/propagate_polls.py` | Gaussian update via community Σ | Gaussian update via type Σ | Update to use type membership + type Σ |
| `src/prediction/predict_2026_hac.py` | HAC K=10 predictions | Type-based predictions | New `predict_2026_types.py` |
| `api/` | Serves community-based data | Serves type-based data | Update endpoints + DuckDB schema |
| `web/` | Community choropleth | Type choropleth (stained glass) | Update map layer + color scheme |
| `config/model.yaml` | K, min_community_size | J, J_sweep_range, S_super_types, lambda_shrinkage | Extend config |
| `src/db/build_database.py` | Community tables | Type tables + census interpolation | Extend schema |

### What Stays

- Shift vector computation (`src/assembly/build_county_shifts_multiyear.py`) — unchanged
- Election data fetchers — unchanged
- RCMS, IRS migration, ACS fetchers — unchanged (census interpolation adds to, doesn't replace)
- Stan factor model code — retained for comparison/validation
- API framework (FastAPI + DuckDB) — endpoints change, architecture stays
- Frontend framework (Next.js + Deck.gl) — rendering changes, structure stays

### Backward Compatibility

The current HAC pipeline is retained as `county_baseline` in the model versioning scheme. The new type-primary model becomes `current`. Both can be loaded and compared.

---

## Success Criteria

1. **Stained glass map**: Each of 293 counties individually colored by type — visually distinct from current blob map
2. **Holdout accuracy**: Type-based holdout r >= 0.85 (minimum threshold); r >= 0.90 (target, matching current HAC). If r falls between 0.85-0.90, acceptable only if type interpretability is clearly superior to HAC blobs. Below 0.85: revisit algorithm choice (see Model Choice Risk in Type Discovery).
3. **Type interpretability**: At least 80% of types can be named by a human reviewer from their demographic profile
4. **Type stability**: Subspace angle between time-window solutions < 25°
5. **Covariance validity**: Constructed type covariance predicts actual type comovement (off-diagonal correlation > 0.5)
6. **Census interpolation**: Demographics match era-appropriate values (manual spot-check: Miami-Dade 2004 vs 2020 should show meaningful Hispanic population growth)
7. **End-to-end predictions**: 2026 county-level predictions with uncertainty intervals, propagated through type covariance

---

## Estimated Scope

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| Census interpolation (fetch + interpolate) | Medium | Census API (NHGIS fallback for 2000 SF3) |
| Type discovery (SVD + varimax + J selection + nesting) | Large | Shift vectors (existing) |
| Type description (demographics overlay) | Medium | Census interpolation |
| Type covariance construction | Medium | Type profiles |
| Covariance validation | Small | Historical elections (existing) |
| Poll propagation update | Medium | Type covariance |
| Prediction pipeline update | Medium | Type membership + covariance |
| DuckDB schema update | Small | Type assignments |
| API endpoint updates | Medium | DuckDB schema |
| Frontend stained glass map | Medium | API endpoints |
| FEC donor density | Small | FEC API |
| Algara governor extension (pre-2000) | Small | Algara data (downloaded) |
| Validation suite | Medium | All above |
