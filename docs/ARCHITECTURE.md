# Architecture Design Document: US Political Covariation Model

**Status:** Living document (last updated March 2026)
**Scope:** Full technical specification for the type-primary electoral covariation model
**Geography:** National (all 50 states + DC)
**Target:** Functional predictions by October 2026 midterms

> **⚠️ MIGRATION IN PROGRESS (2026-03-27):** This document describes the current county-primary production architecture. A migration to **tract-primary architecture with a voter behavior layer** is approved and in progress. See `docs/superpowers/specs/2026-03-27-tract-primary-behavior-layer-design.md` for the target design. Key changes:
> - Unit of analysis: counties (3,154) → tracts (~81K)
> - Data source: MEDSL + Algara/Amlani → DRA block data (all 51 states, 2008-2024)
> - New Layer 2: Voter behavior layer with turnout ratio (τ) and residual choice shift (δ) per type, decomposing presidential vs off-cycle behavior
> - Governor/Senate results move from type discovery inputs → behavior layer training data
> - County frontend layer retired; tracts become sole map view
> - State-centering of off-cycle shifts documented as proxy for candidate effect removal (future improvement)
>
> Until migration is complete, the county model described below remains production. Do not delete county infrastructure.

---

## Table of Contents

1. [Overall Architecture and Philosophy](#1-overall-architecture-and-philosophy)
2. [Data Assembly Component](#2-data-assembly-component)
3. [Shift Vector Computation](#3-shift-vector-computation)
4. [Type Discovery (KMeans)](#4-type-discovery-kmeans)
5. [Hierarchical Nesting and Type Description](#5-hierarchical-nesting-and-type-description)
6. [Covariance Construction](#6-covariance-construction)
7. [Poll Ingestion and Type-Level Propagation](#7-poll-ingestion-and-type-level-propagation)
8. [Prediction and Interpretation](#8-prediction-and-interpretation)
9. [Validation Framework](#9-validation-framework)
10. [Identified Gaps](#10-identified-gaps)

---

## 1. Overall Architecture and Philosophy

### Design Principles

Four non-negotiable design principles govern every component of this system:

**1. Types from shifts, demographics for description.** Electoral types are discovered directly from how counties shift across elections (shift vectors), not from demographic data. Demographics enter the model only to describe discovered types and construct the covariance matrix. This separation ensures types capture genuine electoral structure rather than demographic proxies.

**2. Loosely coupled components.** Each pipeline stage reads from and writes to `data/` subdirectories. Components communicate through files on disk (Parquet, JSON, CSV), not direct function calls or in-memory objects. Any stage can be re-run independently without re-running the full pipeline.

**3. Reproducibility.** Every intermediate output is saved to `data/` subdirectories. Random seeds are fixed and logged. All data transformations are scripted -- no manual steps between raw data and outputs. The pipeline can be re-run from scratch and produce identical results.

**4. Falsification built in.** Types discovered from pre-2024 shifts are validated against held-out 2024 shifts. If types fail to predict the holdout, the model fails cleanly. Historical approaches (NMF on demographics, HAC geographic blobs) are retained as comparison baselines. Negative results are documented, not hidden.

### Pipeline Overview (current county production)

```
[Data Assembly] --> [Shift Vectors] --> [Type Discovery] --> [Hierarchical Nesting] --> [Type Description]
  src/assembly/     src/discovery/      src/discovery/        src/discovery/             src/description/
  Python            Python              Python (KMeans)       Python (Ward HAC)          Python

--> [Covariance Construction] --> [Poll Propagation] --> [Prediction] --> [Validation]
       src/covariance/              src/propagation/      src/prediction/   src/validation/
       Python                       Python                Python            Python
```

### Target Pipeline (tract-primary, IN PROGRESS)

```
[DRA Block→Tract] --> [Shift Vectors] --> [Type Discovery] --> [Behavior Layer] --> [Covariance]
  src/assembly/        src/discovery/      src/discovery/       src/behavior/        src/covariance/
  Block→tract agg      Pres + off-cycle    KMeans J=100         τ + δ per type       Ledoit-Wolf on
  All 51 states        shifts (separate)   Run once             From pres vs         tract shifts
                       Off-cycle state-                         off-cycle results
                       centered

--> [Prediction] --> [Validation] --> [Frontend]
     src/prediction/   src/validation/     web/
     Ridge priors +    LOPO CV +           Tract community
     behavior adj +    backtest τ/δ        polygons only
     Bayesian poll
```

Components communicate through artifacts in `data/`:

```
data/
  raw/                  # Original downloaded files (never modified)
  assembled/            # Cleaned county-level Parquet, census interpolated, shifts
  shifts/               # County shift vectors (log-odds)
  communities/          # Type assignments, super-type hierarchy, membership weights
  covariance/           # Demographic-derived covariance matrices
  polls/                # Cleaned poll data, crosstabs
  predictions/          # Model outputs with credible intervals
  validation/           # Holdout sets, metrics, comparison tables
```

### What "Electoral Type" Means

An electoral type is a **latent archetype** discovered from how counties shift across elections. It is not a geographic region, not a demographic category, and not a political party proxy. It is a pattern of correlated electoral shifts -- counties that move similarly across multiple election pairs belong to the same type, even when geographically distant.

Each tract (target architecture) or county (current production) has a **soft membership vector** across J=100 types, computed as temperature-scaled inverse-distance to KMeans centroids (T=10, row-normalized to sum to 1). A tract is never "one thing." Types nest **hierarchically**: J=100 fine types group into super-types via Ward HAC on demographic profiles (not centroids — centroids produce degenerate clustering at J=100). Super-types are the public-facing "colors" of the stained glass map.

**Examples from the FL+GA+AL pilot:**
- A cross-state rural type spans counties in AL, FL, and GA that share low density, older populations, and similar shift trajectories
- The Atlanta metro professional type clusters suburban GA counties with $78K median income and 40% BA+ education
- The Miami-Dade Hispanic enclave type captures 3 South FL counties with 51% Hispanic population and a distinctive Cuban American shift pattern
- The Alabama Black Belt type spans 9 AL counties with 68% Black population and $31K median income

The critical claim is that counties sharing a type will **covary politically**, even when geographically separated. This is validated: 10 of 20 types span multiple states, and holdout r=0.778 confirms predictive power.

### Dual-Output Model

Every community type produces **two parameters** per election:

1. **Turnout rate** -- the fraction of eligible voters who vote. Captures mobilization, enthusiasm, access, and suppression effects.
2. **Vote share conditional on turnout** -- the D/R split among those who actually vote. Captures persuasion, partisan lean, and issue positioning.

Covariance operates on **both dimensions jointly**, producing a 2K x 2K covariance matrix (where K is the number of types). This enables decomposition of apparent county-level shifts into three distinct mechanisms, following Grimmer & Hersh (2021, *Science Advances*):

- **Persuasion:** People who voted in both elections changed their vote choice.
- **Differential turnout:** Different people showed up -- some previous voters stayed home, some new voters participated.
- **Population change:** The county's residents changed through migration, aging, death, and new eligible voters.

Most election models treat the county-level vote margin as a single number. By modeling turnout and vote share as separate but correlated quantities at the type level, this model can distinguish between "Type X shifted 3 points toward Democrats because voters were persuaded" and "Type X appeared to shift 3 points because its turnout rate dropped differentially among Republican-leaning members."

---

## 2. Data Assembly Component

**Technology:** Python (pandas, geopandas, cenpy/pytidycensus equivalents)
**Source:** `src/assembly/`
**Output:** `data/assembled/`

### Data Sources

| Source | Resolution | Temporal Coverage | Access Method | Pipeline Step |
|--------|-----------|-------------------|---------------|---------------|
| **MEDSL county returns** | County | 2000-2024 (7 presidential + 5 midterm) | Download from [MIT Election Data + Science Lab](https://electionlab.mit.edu/data) | `download_medsl.py` |
| **Census / ACS** | Tract, block group, county | 2000, 2010, 2020 (decennial); 2005-2024 (ACS 5-year) | Census API via [NHGIS](https://www.nhgis.org/) or cenpy | `download_census.py` |
| **RCMS religion** | County | 2000, 2010, 2020 | Download from [ARDA](https://www.thearda.com/) (Religious Congregations and Membership Study) | `download_rcms.py` |
| **IRS SOI migration** | County-to-county | 2011-2022 (annual) | Download from [IRS SOI](https://www.irs.gov/statistics/soi-tax-stats-migration-data) | `download_irs_migration.py` |
| **Census LODES commuting** | County-to-county (block available) | 2002-2021 | Download from [LEHD](https://lehd.ces.census.gov/) | `download_lodes.py` |
| **Facebook SCI** | County-to-county, ZIP-to-ZIP | Snapshot (~2020) | Download from [HDX](https://data.humdata.org/dataset/social-connectedness-index) | `download_sci.py` |
| **BLS QCEW** | County | 2000-2024 (quarterly) | Download from [BLS](https://www.bls.gov/qcew/) | `download_qcew.py` |
| **Opportunity Insights social capital** | County, ZIP | 2022 | Download from [Opportunity Insights](https://opportunityinsights.org/) | `download_social_capital.py` |
| **CES/CCES** | Individual (geocoded) | 2006-2024 (biennial) | Download from [Harvard Dataverse](https://cces.gov.harvard.edu/) | `download_ces.py` |
| **538 poll archive** | National, state, district | 2000-2024 | GitHub: [fivethirtyeight/data](https://github.com/fivethirtyeight/data) | `download_polls.py` |
| **FL/GA/AL voter files** | Individual (geocoded) | Varies by state | State-specific (FL: public request; GA/AL: see Gaps) | `download_voter_files.py` |

### Feature Engineering

Census/ACS variables are organized into five domains following the OAC methodology (Singleton & Longley 2015):

1. **Demographic structure:** Age distribution, racial/ethnic composition, household size, foreign-born share
2. **Socioeconomic:** Education attainment, income distribution, occupation mix (QCEW), poverty rate
3. **Housing:** Owner/renter ratio, housing age, housing value, density, structure type
4. **Religious/cultural:** RCMS denominational shares (evangelical Protestant, mainline Protestant, Catholic, historically Black Protestant, other), Opportunity Insights social capital indices
5. **Connectivity:** SCI summary statistics per county, commuting self-containment, net migration rate

All features are standardized (range or z-score) before community detection. Correlations are checked; highly collinear features (r > 0.95) are reduced via PCA within domains.

### Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `data/assembled/counties.parquet` | Parquet | 226 rows (FL+GA+AL counties), all features across five domains |
| `data/assembled/sci_pairs.parquet` | Parquet | County-to-county SCI values (226 x 226 = 51,076 pairs, sparse) |
| `data/assembled/migration_pairs.parquet` | Parquet | County-to-county IRS migration flows |
| `data/assembled/commuting_pairs.parquet` | Parquet | County-to-county LODES commuting flows |
| `data/assembled/elections.parquet` | Parquet | County x election matrix (226 x ~12 elections), vote share + turnout |
| `data/assembled/feature_metadata.json` | JSON | Feature names, domains, transformations applied, sources |

---

## 3. Shift Vector Computation

**Technology:** Python (numpy, scipy)
**Source:** `src/discovery/shift_vectors.py`
**Output:** `data/shifts/`

### Method

For each county and each consecutive election pair (e.g., 2016 pres -> 2020 pres, 2018 gov -> 2022 gov), compute the log-odds shift:

```
shift = logit(dem_share_later) - logit(dem_share_earlier)
```

where `dem_share = dem_votes / total_votes` (all candidates, not two-party). Epsilon clipping at 0.01/0.99 for uncontested-adjacent races.

### Presidential Weighting and State-Centering

Raw shift vectors produce state-isolated types because governor races dominate and differ by state. The solution discovered empirically:

1. **Presidential shifts are weighted 2.5x** -- presidential races correlate across state lines (same candidates everywhere), enabling cross-state type discovery
2. **Governor/Senate shifts are state-centered** (demeaned within each state) -- removes state-level baseline so within-state differentiation comes through without forcing types to be state-specific

This was the empirical sweet spot: raw all-shifts produced state-isolated types; presidential-only lost governor/Senate signal for within-state differentiation.

### Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `data/shifts/county_shifts_multiyear.parquet` | Parquet | 293 x D shift matrix (D = training + holdout dimensions) |

---

## 4. Type Discovery (KMeans)

**Technology:** Python (scikit-learn KMeans)
**Source:** `src/discovery/run_type_discovery.py`
**Output:** `data/communities/`

### Method

KMeans clustering on the weighted, state-centered shift matrix discovers J=20 electoral types. Each county is assigned to the nearest centroid (hard assignment for clustering), then soft membership is computed as inverse-distance to all centroids (row-normalized to sum to 1).

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
labels = kmeans.fit_predict(weighted_shifts)

# Soft membership via inverse distance
distances = kmeans.transform(weighted_shifts)  # 293 x 20
inv_dist = 1.0 / (distances + epsilon)
soft_membership = inv_dist / inv_dist.sum(axis=1, keepdims=True)
```

### Why KMeans (not HAC or NMF)

- **HAC with spatial constraint** (ADR-005): Produced K=10 "alternative states" -- giant geographic blobs, not the stained glass pattern of cross-state types
- **NMF on demographics** (original approach): Requires non-negative input; log-odds shifts are negative. Indirectly discovers types via demographic proxies rather than electoral behavior
- **KMeans**: No spatial constraint (types can cross state lines), handles negative values naturally, produces balanced cluster sizes, and is computationally simple

### Validation

- **Holdout**: Train on pre-2024 shift dimensions, test against 2024 presidential shifts. Current: r=0.778
- **Cross-state coverage**: 10 of 20 types span multiple states, confirming types capture electoral behavior rather than geographic proximity
- **Size balance**: Types range from 3 to 31 counties -- no single type dominates

---

## 5. Hierarchical Nesting and Type Description

**Technology:** Python (scipy Ward HAC, pandas)
**Source:** `src/discovery/`, `src/description/`
**Output:** `data/communities/`

### Hierarchical Nesting

Ward HAC on the J=20 KMeans centroids (no spatial constraint) produces 6-8 super-types. Super-types are the public-facing "colors" of the stained glass map.

Hierarchical consistency is enforced: every fine type maps to exactly one super-type.

### Type Description

After discovering types from shifts, overlay time-matched demographics to characterize them:

1. **Census interpolation**: Decennial census (2000/2010/2020) linearly interpolated for election years. CPI-adjusted income.
2. **Demographic overlay**: For each type, compute weighted-mean demographics across member counties. Features include: population density, racial composition, median income, education attainment, age distribution.
3. **RCMS religious data**: Denominational shares overlaid on types.
4. **IRS migration**: Net migration rates overlaid on types.

Types are **named** from their demographic and behavioral character, not discovered from it. Example: a type with 68% Black population, $31K income, and 9 Alabama counties is named "Alabama Black Belt" -- but those demographics did not influence the clustering.

### Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `data/communities/county_type_assignments.parquet` | Parquet | 293 x J soft assignment matrix |
| `data/communities/type_centroids.parquet` | Parquet | J x D centroid matrix |
| `data/communities/hierarchy.json` | JSON | Type-to-super-type mapping |
| `data/communities/type_profiles.parquet` | Parquet | Descriptive statistics for each type (demographics, shift history) |

---

## 6. Covariance Construction

**Technology:** Python (numpy)
**Source:** `src/covariance/construct_type_covariance.py`
**Output:** `data/covariance/`

### The Problem

We have J=20 electoral types and need a J x J covariance matrix to propagate poll signals between types. Direct estimation from election returns is unreliable with J >> T (number of elections).

### Method: Economist-Inspired Demographic Correlation

Following Heidemanns, Gelman, Morris (2020), the covariance matrix is constructed from type demographic profiles rather than estimated from elections:

#### Step 1: Build Type Demographic Profiles

Using time-matched demographics (interpolated decennial census), compute weighted-mean demographic features for each type. Features include population density, racial composition, median income, education attainment, age distribution.

#### Step 2: Pearson Correlation

Compute the J x J Pearson correlation matrix from type demographic profiles. Types with similar demographics are assumed to covary politically.

#### Step 3: Shrinkage Toward National Swing

Shrink the correlation matrix toward the all-1s matrix (every type moves together = national swing). This regularizes extreme correlations and embeds the prior that there is a common national component to electoral shifts.

```
Sigma_shrunk = (1 - alpha) * Sigma_demographic + alpha * ones_matrix
```

#### Step 4: PD Enforcement

Ensure the resulting matrix is positive definite via eigenvalue clipping.

### Why Not Stan Factor Model?

The Stan factor model (Section 4 in the original architecture) required sampling and was sensitive to the K >> T problem. The Economist-inspired approach is:
- **Deterministic**: No MCMC sampling, no convergence diagnostics
- **Principled**: Grounded in the same logic as the Economist's state-level model
- **Validated**: Correlation between constructed covariance and observed historical comovement serves as a check

The Stan factor model is retained as an ultimate fallback if the demographic correlation approach fails validation (off-diagonal r < 0.4 between constructed and observed covariance).

### Key References

- Heidemanns, Gelman, Morris (2020). "An Updated Dynamic Bayesian Forecasting Model for the US Presidential Election." [HDSR](https://hdsr.mitpress.mit.edu/pub/nw1dzd02)
- TheEconomist/us-potus-model: [GitHub](https://github.com/TheEconomist/us-potus-model)

### Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `data/covariance/type_covariance.parquet` | Parquet | J x J covariance matrix |
| `data/covariance/type_demographic_profiles.parquet` | Parquet | J x F demographic profile matrix |

---

## 7. Poll Ingestion and Type-Level Propagation

**Technology:** Python for scraping/cleaning, R + Stan for the Bayesian model (adapted from [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model)), cmdstanpy as the Python-Stan bridge
**Source:** `src/propagation/` (R + Stan), `src/assembly/` (poll cleaning, Python)
**Output:** `data/polls/`, `data/predictions/`

### Poll Sources

| Source | Type | Access | Notes |
|--------|------|--------|-------|
| 538 poll archive | National, state, district | [GitHub](https://github.com/fivethirtyeight/data) | Historical archive, structured CSV |
| RealClearPolitics | National, state | Web scraping | Current cycle polls |
| CES/CCES individual data | Individual-level (geocoded) | [Harvard Dataverse](https://cces.gov.harvard.edu/) | 50,000+ respondents per wave, with county FIPS. Gold standard for MRP. |
| State/district polls | State, CD | Various | Aggregated from 538 and RCP |
| Crosstab data | Demographic subgroups within polls | Extracted from poll reports | Age x race x education breakdowns when available |

### Poll Corrections

Following the Economist Stan model (Heidemanns, Gelman, Morris 2020), all poll observations are corrected for systematic biases before entering the propagation model. These corrections are estimated **within the Stan model** as parameters with priors, not applied as fixed pre-processing steps:

1. **Partisan non-response bias:** When one party's supporters are more enthusiastic, they answer polls at higher rates, creating phantom swings (Gelman et al. 2016). Corrected by poststratifying on party identification when individual-level data is available, and by estimating a time-varying non-response parameter in the model.

2. **Mode adjustment:** Phone polls, online polls, IVR polls, and in-person polls have systematic differences. Each mode gets an estimated bias parameter.

3. **Population adjustment:** Registered voter polls vs. likely voter polls vs. adult population polls differ systematically. Adjustment parameters are estimated per population type.

4. **House effects:** Each polling firm has a persistent lean. Estimated as firm-level random effects with a shrinkage prior (half-normal on the SD).

### The Propagation Model (Core Architecture)

This is the central inference engine. It is adapted from [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model), with the critical modification that the 51 state units are replaced by J electoral types.

#### Generative Model

**Prior (anchoring to previous elections + fundamentals):**

The prior for each type's election-day position is informed by:

- Previous election result for that type (strongest signal, per voter stability evidence)
- Demographic drift since the last election (ACS changes in education, age, race composition)
- Economic fundamentals (following Erikson & Wlezien 2012, *The Timeline of Presidential Elections*; Sides & Vavreck 2013, *The Gamble*): national GDP growth, unemployment, presidential approval
- The prior is **tight**, reflecting the voter stability literature: campaigns have minimal persuasive effects (Kalla & Broockman 2018), and most poll variation is noise

**Latent state (reverse random walk):**

Following Linzer (2013), the latent type-level political state is modeled as a **reverse random walk** anchored at Election Day and walking backward in time:

```
theta_k(t) = theta_k(t+1) + eta_k(t)
eta(t) ~ MVN(0, Sigma_innovation)
```

where `Sigma_innovation` is the innovation covariance derived from the factor model (Section 4). The reverse walk ensures that the model is anchored to the actual election outcome (for hindcasting) or to the prior (for forecasting), with polls progressively refining the estimate as they accumulate.

**Observation model (spectral unmixing):**

Each poll p observes a noisy mixture of type-level signals:

```
y_p = sum_k( w_pk * theta_k ) + bias_p + epsilon_p
```

where:
- `y_p` is the observed poll result
- `w_pk` is the weight of type k in the polled population (derived from the demographic composition of the poll's geographic area and the type assignment matrix W)
- `theta_k` is the latent political state of type k
- `bias_p` captures house effects, mode effects, and population adjustment
- `epsilon_p` is sampling noise (known from poll sample size)

**This is the linear spectral unmixing model from remote sensing** (Rasti et al. 2024; see [HySUPP](https://github.com/BehnoodRasti/HySUPP)). The "endmembers" are community types; the "abundances" are known population shares; the "observed spectrum" is the poll result. The key difference from remote sensing: our "abundances" (type population shares) are known from census data, so we are solving for the endmember values (type-level political states) from multiple mixed observations.

**Crosstab sub-model:**

When polls report demographic breakdowns (e.g., vote share among 18-29 year olds, or among Black respondents), these provide **tighter constraints** because the type-composition weights `w_pk` are more narrowly defined. A crosstab for "Black voters in Florida" has a type composition that is heavily concentrated on types with high Black population share, making the unmixing problem more determined.

```
y_p_crosstab = sum_k( w_pk_crosstab * theta_k ) + bias_p + epsilon_p
```

where `w_pk_crosstab` is the type composition of the specific demographic subgroup in the polled geography.

**CES integration via MRP:**

The Cooperative Election Study (CES) provides individual-level vote choice data with demographic covariates and geographic identifiers. This is integrated via MRP (Multilevel Regression and Poststratification):

```
P(vote_D | demographics_i, community_type_k) = logit^{-1}(
    alpha + X_i * beta + gamma_state + delta_type_k
)
```

where `delta_type_k` is a random effect for community type with a structured prior.

The structured prior on type random effects follows Gao et al. (2021, *Bayesian Analysis*): an ICAR (Intrinsic Conditional Autoregressive) prior where the adjacency structure is defined by type similarity in the community network. This provides spatial smoothing across types without estimating K free parameters.

**Implementation uses:**
- [ccesMRPprep](https://github.com/kuriwaki/ccesMRPprep) (Kuriwaki): R package for preparing CES data for MRP, including poststratification table construction
- [ccesMRPrun](https://github.com/kuriwaki/ccesMRPrun) (Kuriwaki): R package for running MRP on CES data with brms/rstanarm
- Structured priors from [alexgao09/structuredpriorsmrp_public](https://github.com/alexgao09/structuredpriorsmrp_public): Implementation of Gao et al. 2021

### Poll Accumulation Mechanism

The model's behavior changes over the election cycle:

| Phase | Polls Available | Model Behavior |
|-------|----------------|----------------|
| **Early cycle** (12+ months out) | Few/none | Prior dominates. Type-level estimates are essentially previous election + fundamentals + demographic drift. Wide credible intervals. |
| **Mid cycle** (6-12 months) | National + some state polls | Systematic deviations from prior begin to emerge. National swing factor identified. Type-level estimates begin to differentiate. |
| **Late cycle** (0-6 months) | Dense state + some district polls + crosstabs | Type-level shifts identifiable for major types. Crosstabs tighten estimates. Credible intervals narrow. |
| **Election night** | Actual results (partial, then full) | Results decomposed into type-level contributions. Full posterior on type-level parameters. |

### Key Repos to Adapt

| Repository | License | Language | What to Adapt |
|-----------|---------|----------|---------------|
| [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model) | MIT | R + Stan | Core state-space model structure, poll correction framework, reverse random walk. Replace 51 states with K community types. |
| [markjrieke/2024-potus](https://github.com/markjrieke/2024-potus) | MIT | R + Stan | Estimated covariance parameters, updated polling methodology for 2024 cycle. |
| [alexgao09/structuredpriorsmrp_public](https://github.com/alexgao09/structuredpriorsmrp_public) | -- | R + Stan | ICAR structured priors for MRP random effects. Apply to community type random effects. |
| [fonnesbeck/election_pycast](https://github.com/fonnesbeck/election_pycast) | -- | Python + PyMC | Dynamic Bayesian model in PyMC. Reference for Python-native implementation if cmdstanpy bridge proves unwieldy. |
| [kuriwaki/ccesMRPprep](https://github.com/kuriwaki/ccesMRPprep) | MIT | R | CES data preparation for MRP. |
| [kuriwaki/ccesMRPrun](https://github.com/kuriwaki/ccesMRPrun) | MIT | R | MRP execution with brms/rstanarm on CES data. |

---

## 8. Prediction and Interpretation

**Technology:** Python for aggregation and visualization, R for mapping
**Source:** `src/prediction/`, `src/viz/`
**Output:** `data/predictions/`

### Outputs

#### County-Level Predictions

For each of the 293 FL+GA+AL counties, the model produces:

- **Vote share** (two-party Democratic share) with 80% and 95% credible intervals
- **Turnout** (as fraction of voting-eligible population) with credible intervals
- **Type decomposition:** Which types contribute what fraction of the county's predicted vote share and turnout

#### Aggregated Predictions

County-level predictions are aggregated to:

- **Congressional districts** (using county-to-CD crosswalk; for split counties, use tract-level type assignments if available)
- **State-level** totals (FL, GA, AL)
- **Custom geographies** (media markets, MSAs)

#### Uncertainty Decomposition

Total prediction uncertainty decomposes into four sources:

| Source | Description | How Estimated |
|--------|------------|---------------|
| **Polling noise** | Sampling error in poll observations | Known from poll sample sizes; propagated through observation model |
| **Type assignment uncertainty** | Counties' type compositions are estimated, not known | Bootstrap or posterior from NMF/Leiden stability analysis |
| **Covariance estimation uncertainty** | Factor loadings and innovation variance are estimated from 12 elections | Posterior from Stan factor model |
| **Innovation uncertainty** | Future type-level shifts are stochastic | Innovation covariance from factor model, propagated through random walk |

### Shift Narratives (the Distinctive Output)

This is what distinguishes this model from standard election forecasts. Instead of saying "Duval County shifted 2 points toward Democrats," the model says:

**Type-level shift table:**

| Type | Weight in Duval | Vote Share Shift | Turnout Shift | Contribution to Duval Shift |
|------|----------------|-----------------|---------------|---------------------------|
| Military-suburban | 0.30 | +0.5 | -1.2 | +0.15 |
| Urban-Black-institutional | 0.25 | +1.0 | +3.5 | +0.25 |
| New-South-professional | 0.20 | +2.5 | +0.5 | +0.50 |
| Coastal-retirement | 0.15 | -1.0 | -0.5 | -0.15 |
| Other | 0.10 | +0.0 | +0.0 | +0.00 |
| **Total** | **1.00** | -- | -- | **+0.75** |

**Per-county decomposition of shifts into type contributions** -- this is the spectral unmixing output. Each county's observed shift is decomposed into the weighted sum of its constituent types' shifts.

**Turnout decomposition** following Grimmer & Hersh (2021):

| Mechanism | Contribution to Duval Shift |
|-----------|---------------------------|
| Persuasion (same voters, different choice) | +0.30 |
| Differential turnout (different voters showed up) | +0.35 |
| Population change (different people live there) | +0.10 |
| **Total apparent shift** | **+0.75** |

### Conditional Forecasting

The type-level structure enables mechanistic conditional forecasts:

- "If Cuban American communities shift as polls suggest (Type X moves +5 toward R), Florida shifts by Y points."
- "If Black turnout returns to 2012 levels in GA (Type Z turnout increases by 8 points), Georgia flips by probability P."
- "If suburban-professional types continue their 2016-2020 trend, FL-CD13 shifts by Z points."

These are **mechanistic scenarios**, not pure extrapolation. Each scenario specifies which type-level parameters change and by how much, and the model propagates those changes through the county composition structure.

---

## 9. Validation Framework

**Technology:** Python + R
**Source:** `src/validation/`
**Output:** `data/validation/`

### Three Baselines

Every model run is compared against three baselines. If the community-type model does not beat all three, it is not contributing useful structure.

| Baseline | Description | Implementation |
|----------|-------------|----------------|
| **1. Demographic linear model** | OLS regression of county vote share on ACS demographic variables (education, race, income, age, urbanicity) + state fixed effects. No community structure. | Python: scikit-learn `LinearRegression` with state dummies |
| **2. Uniform swing** | Apply the national popular vote swing uniformly to all counties' previous-election results. The simplest possible model. | Python: single scalar applied to all counties |
| **3. Demographic MRP** | Standard MRP (multilevel regression and poststratification) using CES data with demographic grouping variables but **no community type variable**. Uses ccesMRPprep + ccesMRPrun with `(1|state) + (1|age) + (1|race) + (1|education)` but not `(1|community_type)`. | R: brms or rstanarm via ccesMRPrun |

### Hindcast Validation

**Leave-one-election-out** for all elections 2000-2024:

For each election t in {2000, 2002, 2004, ..., 2024}:
1. Estimate community types from non-political data (this does not change across folds, since types use no political data).
2. Estimate covariance from all elections **except** t.
3. Set priors from elections before t.
4. Simulate a "polling environment" for election t using polls available before election day.
5. Generate predictions for election t.
6. Compare to actual results.

**Specific historical tests** (chosen because they stress-test the model's ability to capture known dynamics):

| Test Case | What It Tests |
|-----------|--------------|
| **FL 2000** (Bush v. Gore) | Can the model handle a razor-thin election with unusual turnout patterns? |
| **FL 2008/2012** (Obama coalition) | Does the model capture the Obama coalition's distinctive type composition? |
| **GA 2020** (Biden flip) | Can the model capture a state flip driven by suburban-professional + Black-urban type shifts? |
| **FL Cuban American shift 2016-2024** | Can the model detect and propagate a type-specific shift (Cuban American communities moving sharply toward R) across multiple elections? |
| **2022 midterm** | Does the model generalize from presidential to midterm elections (different turnout patterns, potentially different covariance structure)? |

### Metrics

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| **County RMSE (vote share)** | < demographic baseline | Average prediction error across counties |
| **County RMSE (turnout)** | < demographic baseline | Average turnout prediction error |
| **Differential swing correlation** | r > 0.5 between predicted and actual type-level swings | Does the model correctly predict *which types shift* and *in which direction*? |
| **Type-level shift stability** | Cross-election correlation of type shifts > 0.7 | Are the same types consistently behaving similarly across elections? |
| **Calibration** | 80% CI covers actual result 75-85% of the time | Are credible intervals well-calibrated? |
| **Information gain from polls** | Measurable reduction in RMSE as polls accumulate | Do polls actually improve predictions through the covariance structure? |

### Falsification Criteria

The model is considered **falsified** (hypothesis rejected) if any of the following hold:

| # | Criterion | What It Would Mean |
|---|-----------|-------------------|
| 1 | Community types do not beat demographics (Baseline 3 wins) | Non-political community structure adds no information beyond standard demographic predictors. The hypothesis that community types capture something beyond demographics is wrong. |
| 2 | Cross-border structure adds < 1% RMSE improvement over within-state-only types | The SCI-based cross-state community structure is not meaningfully different from state-level demographics. The "communities cross state borders" claim is not supported. |
| 3 | Type-level behavior is less stable across elections than county-level behavior | The types are not capturing stable behavioral patterns. The abstraction is losing signal rather than gaining it. |
| 4 | 2000-2016 covariance fails to predict 2020/2024 shifts | The covariance structure is not stable enough for forecasting. The factor model is fitting noise in historical data rather than capturing genuine structure. |
| 5 | Adding the turnout dimension does not improve vote share predictions | The joint modeling of turnout and vote share is not adding value. The additional complexity is not justified. |

### Variation Partitioning

Following metacommunity ecology methodology, use `vegan::varpart()` (R) to decompose county-level political variation into:

- **E (Environment):** Fraction explained by demographics alone (ACS variables)
- **S (Space/Structure):** Fraction explained by community structure alone (type assignments)
- **E intersection S:** Fraction explained by spatially structured demographics (demographics that covary with community structure)
- **Residual:** Unexplained variation

This directly answers: "How much does community structure add beyond demographics?" If E alone explains 90% and S adds only 1%, the community types are not pulling their weight.

**Implementation:** Use `vegan::varpart()` with county vote share as the response, ACS demographic variables as the E matrix, and community type dummy variables (or soft assignment vectors) as the S matrix. The partial R-squared values give the decomposition.

### Reference

Economist backtesting scripts: `final_2008.R`, `final_2012.R`, `final_2016.R` in [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model). These provide tested code for running the Economist model on historical elections with known outcomes.

---

*MVP section removed -- the type-primary pipeline (KMeans J=20) is the current production implementation. See Phase 1 deliverables in `docs/ROADMAP.md`.*

---

## 10. Identified Gaps

Known gaps in the current design that require resolution before or during implementation:

| Gap | Severity | Description | Mitigation Strategy |
|-----|----------|-------------|---------------------|
| **GA voter file access** | Medium | Georgia voter file requires a fee and institutional affiliation for full access. Individual-level turnout history needed for Grimmer-Hersh decomposition. | Defer to post-MVP. Use aggregate county-level turnout from MEDSL for MVP. Explore academic data-sharing agreements. |
| **AL voter file access** | Medium | Alabama voter file access is restrictive. Less critical than GA because AL is less politically competitive. | Defer to post-MVP. AL primarily included for geographic contiguity and SCI network completeness, not as a primary prediction target. |
| **Tract-level community assignment** | Medium | Crosstab mapping requires knowing the type composition of demographic subgroups within counties. This requires tract-level type assignments, but MVP community detection operates at county level. | For MVP, approximate by assuming uniform type composition within counties. Post-MVP, run geodemographic classification at tract level within heterogeneous counties and construct tract-level type assignments. |
| **Soft assignment validation** | Low-Medium | There is no ground truth for soft type assignments. How do we validate that a county is "35% suburban-professional and 25% urban-Black-institutional"? | Validate indirectly: (a) compare NMF-derived soft assignments to Leiden-derived ones; (b) check that soft assignments correlate with known tract-level demographic variation within counties; (c) verify that using soft vs. hard assignments improves prediction. |
| **Midterm vs. presidential covariance** | Medium | The covariance structure may differ between presidential and midterm elections (different turnout patterns, different issue salience). With only 5 midterms in the 2000-2024 window, estimating a separate midterm covariance is infeasible. | Start by assuming the same factor structure with an election-type indicator. If midterm predictions are systematically worse, investigate a modified covariance for midterms. The factor model can include a midterm-specific loading adjustment. |
| **Turnout ground truth at type level** | Medium | Type-level turnout is not directly observed; it is inferred from county-level turnout via soft assignments. This inference is circular if the type assignments were partly determined by turnout-correlated features. | Mitigated by the two-stage separation: types are discovered from non-political data, so turnout does not influence type assignment. Validate by comparing inferred type-level turnout against voter-file-derived turnout estimates (FL voter file provides individual turnout history). |
| **Narrative generation** | Low | Shift narratives (Section 6) are currently conceived as manual interpretation of model outputs. Automated narrative generation would require NLG infrastructure. | Keep manual for MVP and v1. Consider LLM-based narrative generation post-v1 (cf. the EPJ Data Science paper on LLM-based geodemographic naming). |
| **LODES engineering complexity** | Low-Medium | Census LODES origin-destination files are large (block-level), require aggregation to county level, and have annual vintage changes. Processing pipeline is non-trivial. | Defer to post-MVP. IRS SOI migration data + Facebook SCI provide adequate network layers for initial community detection. Add LODES when the pipeline is stable. |
| **Model identifiability with K >> T** | Medium | With K = 50 types and T = 12 elections, the type-level parameters are not individually identifiable from election data alone. The factor model addresses this by reducing dimensionality, but individual type estimates may have wide posteriors. | Accept wide posteriors on individual type parameters. The model's value is in the covariance structure (factor loadings), not individual type point estimates. Validate that the covariance structure produces useful predictions despite individual-type uncertainty. |
| **Temporal alignment of data sources** | Low | ACS, RCMS, SCI, and election data have different temporal cadences. ACS is 5-year rolling; RCMS is decennial; SCI is a snapshot; elections are biennial. | Use temporally closest available data for each election. Log all temporal mismatches in `data/assembled/feature_metadata.json`. For MVP, use a single cross-section (most recent available) and note the approximation. |

---

## Appendix A: Key References (Consolidated)

### Election Modeling

- Linzer (2013). "Dynamic Bayesian Forecasting of Presidential Elections in the States." *JASA*, 108(501), 124-134. [PDF](https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf)
- Heidemanns, Gelman, Morris (2020). "An Updated Dynamic Bayesian Forecasting Model for the US Presidential Election." [HDSR](https://hdsr.mitpress.mit.edu/pub/nw1dzd02)
- Ghitza & Gelman (2013). "Deep Interactions with MRP." *AJPS*. [PDF](https://sites.stat.columbia.edu/gelman/research/published/misterp.pdf)
- Gao et al. (2021). "Improving MRP with Structured Priors." *Bayesian Analysis*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9203002/)
- Erikson & Wlezien (2012). *The Timeline of Presidential Elections*. University of Chicago Press.
- Sides & Vavreck (2013). *The Gamble*. Princeton University Press.

### Voter Stability

- Gelman, Goel, Rivers, Rothschild (2016). "The Mythical Swing Voter." *QJPS*, 11, 103-130.
- Kalla & Broockman (2018). "The Minimal Persuasive Effects of Campaign Contact in General Elections." *APSR*, 112(1), 148-166.
- Shirani-Mehr, Rothschild, Goel, Gelman (2018). "Disentangling Bias and Variance in Election Polls." *JASA*, 113(522), 607-614.

### Composition vs. Conversion

- Grimmer, Hersh, et al. (2021). "Not by Turnout Alone: Measuring the Sources of Electoral Change, 2012 to 2016." *Science Advances*.

### Community Detection and Geodemographics

- Traag, Waltman, van Eck (2019). "From Louvain to Leiden." *Scientific Reports*. [Nature](https://www.nature.com/articles/s41598-019-41695-z)
- Peixoto (2014). "Hierarchical Block Structures and High-Resolution Model Selection in Large Networks." *Physical Review X*.
- Bailey et al. (2018). "Social Connectedness: Measurement, Determinants, and Effects." *JEP*. [AEA](https://www.aeaweb.org/articles?id=10.1257/jep.32.3.259)
- Singleton & Longley (2015). "Creating the 2011 Area Classification for Output Areas." [UCL](https://discovery.ucl.ac.uk/id/eprint/1498873/)
- Jeub et al. (2018). "Multiresolution Consensus Clustering in Networks." *Scientific Reports*. [Nature](https://www.nature.com/articles/s41598-018-21352-7)
- Cai et al. "Graph Regularized Nonnegative Matrix Factorization." [arXiv:1111.0885](https://arxiv.org/pdf/1111.0885)

### Cross-Disciplinary Methods

- Rasti et al. (2024). "Image Processing and Machine Learning for Hyperspectral Unmixing." *IEEE TGRS*. [HySUPP](https://github.com/BehnoodRasti/HySUPP)
- Liang et al. (2025). Region2Vec-GAT. [GitHub](https://github.com/GeoDS/region2vec-GAT)
- Zhang (2022). "Improving Commuting Zones Using the Louvain Community Detection Algorithm." [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165176522003093)

### Factor Analysis of Election Data

- PCA of US Presidential Elections. [aidanem.com](https://www.aidanem.com/us-presidential-elections-pca.html)
- Partisanship & Nationalization (1872-2020). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0261379421001050)

### Open-Source Election Models

- TheEconomist/us-potus-model: [GitHub](https://github.com/TheEconomist/us-potus-model) (MIT license, R + Stan)
- markjrieke/2024-potus: [GitHub](https://github.com/markjrieke/2024-potus) (MIT license, R + Stan)
- fonnesbeck/election_pycast: [GitHub](https://github.com/fonnesbeck/election_pycast) (Python + PyMC)
- alexgao09/structuredpriorsmrp_public: [GitHub](https://github.com/alexgao09/structuredpriorsmrp_public) (R + Stan)
- kuriwaki/ccesMRPprep: [GitHub](https://github.com/kuriwaki/ccesMRPprep) (MIT, R)
- kuriwaki/ccesMRPrun: [GitHub](https://github.com/kuriwaki/ccesMRPrun) (MIT, R)

### Ecology (Variation Partitioning)

- `vegan::varpart()`: [CRAN](https://cran.r-project.org/web/packages/vegan/). Standard tool for decomposing variation into environmental vs. spatial components.

## Appendix B: Technology Stack Summary

| Role | Primary Tool | Language | Backup |
|------|-------------|----------|--------|
| Data wrangling | pandas, geopandas | Python | -- |
| Census data access | cenpy, NHGIS downloads | Python | tidycensus (R) |
| Community detection (network) | leidenalg | Python | graph-tool (nSBM), infomap |
| Community detection (matrix) | sklearn NMF | Python | custom graph-regularized NMF |
| Graph construction | python-igraph | Python | networkx (prototyping only) |
| PCA / factor analysis | sklearn PCA | Python | statsmodels DynamicFactorMQ |
| Bayesian modeling | Stan via cmdstanpy | Python/Stan | PyMC v5 |
| State-space filtering (MVP) | FilterPy | Python | pykalman |
| MRP | brms, rstanarm via ccesMRPrun | R/Stan | PyMC MRP |
| Spatial analysis | PySAL (libpysal, esda) | Python | spdep (R) |
| Variation partitioning | vegan::varpart() | R | -- |
| Posterior diagnostics | ArviZ | Python | -- |
| Visualization | matplotlib, plotly | Python | ggplot2 (R) |
| Data format (intermediate) | Parquet (pyarrow) | -- | CSV |
| Data format (covariance) | Parquet or Arrow IPC | -- | NetCDF |
| Environment (Python) | pyproject.toml | -- | -- |
| Environment (R) | renv.lock | -- | -- |
