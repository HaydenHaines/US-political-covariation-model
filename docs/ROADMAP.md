# Project Roadmap: Robust Path Forward

**Status:** Active
**Last updated:** 2026-03-21
**Perspective shift:** Hobby proof-of-concept → publicly available political modeling platform

---

## Governing Principles

*These replace the MVP-era "minimize complexity" mandate. Every architectural and modeling decision should be evaluated against them.*

1. **Build it right, not fast.** Use the correct model even if it takes longer. Wrong foundations compound.
2. **Build it expandable.** Every component should have a clear interface so it can be swapped, extended, or scaled without rewriting everything around it.
3. **Types are primary.** Electoral types are discovered directly from county shift vectors via KMeans. Types are the predictive engine: covariance, poll propagation, and prediction all flow through type structure. Geographic communities (HAC blobs) are deferred to the tract-level phase.
4. **J is the hardest problem.** The number of types J determines everything downstream. J selection must be principled — maximize holdout predictive accuracy subject to balanced type sizes — not heuristic. Current: J=20 via KMeans.
5. **The model answers one question publicly:** *What will happen in 2026?* Everything else (community discovery, type classification, covariance estimation) is infrastructure that makes that answer defensible.

---

## Name

Working name: **Bedrock**
Concept: the structural foundation beneath surface-level election results. Communities are bedrock; individual elections are weather.
Domain target: `bedrock.vote`
*(Final name/domain decision deferred — build the thing first.)*

---

## Architecture: Type-Primary Model

```
Electoral types (Primary layer)           Super-types (Interpretability layer)
──────────────────────────────            ────────────────────────────────────
KMeans on weighted shift vectors          Ward HAC on KMeans centroids
→ J=20 electoral archetypes               → 6-8 super-types for public communication
→ soft membership via inverse-distance    → no spatial constraint
→ types span states (10/20 cross-state)   → super-types are the "colors" of the map
→ presidential shifts ×2.5 weight         → examples: "Rural Conservative"
→ governor/Senate state-centered                     "Metro Professional"

National model: J ≈ 40-80, super-types ≈ 12-20
FL+GA+AL pilot: J = 20,    super-types ≈ 6-8
```

**Why types-primary?**
The HAC community-primary approach (ADR-005) produced "alternative states" — 10 giant geographic blobs that were essentially alternative administrative boundaries. Types-primary (ADR-006) discovers abstract archetypes that cross state lines, producing the stained glass pattern of many small units colored by behavioral type. Types carry covariance and prediction; geographic communities are deferred to the tract-level phase where sub-county resolution makes spatial clustering meaningful.

**Key empirical discovery:** Raw all-shifts produced state-isolated types (governor races dominate and differ by state). Presidential-only lost within-state differentiation. Presidential weighting at 2.5x with state-centered governor/Senate shifts was the sweet spot — enabling cross-state correlation while preserving within-state signal.

---

## Modeling Decisions (Settled)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Shift math | **Log-odds**: `logit(p_later) − logit(p_earlier)` | Correct geometry for bounded shares. Amplifies shifts at extremes — a rural area shifting from 10% to 15% Dem is geometrically as significant as a swing county shifting 50→55%. Epsilon clipping at 0.01/0.99 for uncontested-adjacent races. |
| Vote share denominator | **Total votes** (all candidates) | Captures full electoral environment including third-party bleeding. Two-party share distorts 2000 (Nader) and 2016 (Johnson/Stein). |
| Type assignment | **Soft membership** via inverse-distance to KMeans centroids (row-normalized to sum to 1) | Counties are mixtures of types, not hard-labeled. Soft assignment captures within-county heterogeneity and enables smooth prediction surfaces. |
| Training races | **Presidential + gubernatorial + Senate** (all available) | Political races have so few data points that every valid cycle is signal. Senate races add cycles; candidate effects are a known confound but are not a reason to exclude — they add variance, not systematic bias. |
| Cross-state communities | **Yes** | Communities are defined by how they move, not by administrative lines. The Great Plains may have very long cross-state communities; most of the country will have localized blobs. |
| Fixed K | **Yes** | K is a model parameter, not a product feature. Finding the right K is the hardest problem; it should be solved once per model generation, not exposed to end users. |
| Uncontested races | **Drop the election pair** for shift calculation; retain the **turnout data point** | A shift from an uncontested race is uninterpretable. But the turnout for that race is a valid local data point (how many people voted in an uncontested race reveals partisan mobilization). Log separately as a turnout feature, not a shift dimension. |
| Temporal weighting | **Equal weighting for now; migration data to explain divergence** | Don't deweight older cycles until we understand why communities diverge. The correct explanation for temporal drift is migration (residents change) and demographics (income, death, generational replacement). Build migration and demographic features to explain drift rather than paper over it with decay weights. |
| Covariance estimation | **Economist-inspired demographic correlation** | Pearson correlation from type demographic profiles, shrunk toward all-1s (national swing), PD enforced. Deterministic Python, no sampling. Stan factor model retained as fallback. |
| Presidential weighting | **2.5x weight** on presidential shift dimensions | Presidential races correlate across state lines; governor/Senate races are state-specific. Weighting presidential shifts enables cross-state type discovery. Empirically discovered: raw all-shifts isolate by state; pres-only loses governor signal. |
| State-centering | **Governor/Senate shifts demeaned within each state** | Removes state-level baseline from non-presidential races, so within-state differentiation comes through without forcing types to be state-specific. |
| Type discovery method | **KMeans J=20** on weighted, state-centered shift vectors | KMeans produces balanced clusters (3-31 counties) without spatial constraint. 10/20 types span multiple states. Holdout r=0.778. |
| Hierarchical nesting | **Ward HAC on KMeans centroids** → 6-8 super-types | No spatial constraint. Super-types provide public interpretability ("stained glass map colors"). |
| Census interpolation | **Linear interpolation** between 2000/2010/2020 decennial census | Provides time-matched demographics for type description and covariance construction. CPI-adjusted income. |
| Turnout modeling | **Keep the triplet** (D-shift, R-shift, turnout-shift); compare 2D vs. 3D as a validation experiment | Note: R-shift = −D-shift identically, so the triplet has 2 independent dims + turnout. Run both framings and compare holdout correlation before committing. |

---

## Open Questions (Logged, Not Yet Answered)

| ID | Question | Why It Matters | When to Answer |
|----|----------|---------------|----------------|
| OQ-001 | Louisiana/California jungle primaries in national expansion | Non-D-vs-R general elections break the shift framework | At national expansion phase |
| OQ-002 | How to incorporate turnout data from dropped uncontested-race pairs | Uncontested turnout as a feature, not a shift dimension | Phase 1 modeling |
| OQ-003 | 2D vs. 3D triplet: does turnout as a third dim help or hurt holdout accuracy? | Affects all shift math going forward | Phase 1 validation |
| OQ-004 | Senate race candidate effect contamination: how large is it relative to community signal? | Determines whether Senate cycles add noise or signal | Phase 1 validation |
| OQ-005 | Right K for FL+GA+AL county model: what does holdout accuracy say? | K selection is the hardest problem; needs empirical answer | **RESOLVED**: J=20 KMeans, holdout r=0.778 |
| OQ-006 | Right J for type classification: how many types are interpretable and stable? | J defines what users see and name | **RESOLVED**: J=20 fine types, 6-8 super-types |
| OQ-007 | Population weighting in clustering: should large counties pull community centroids more? | Equal-county vs. population-weighted HAC gives different community shapes | **SUPERSEDED**: KMeans uses equal-county weighting; population weighting deferred to national phase |

---

## Tech Stack (Revised)

### Modeling
- **Python** — data pipeline, community discovery, feature engineering
- **Stan (via cmdstanpy)** — Bayesian covariance model, type estimation
- **R (via cmdstanr)** — MRP, advanced poll propagation (post-Phase 2)

### Data Layer
- **DuckDB** — replaces flat parquets as the query layer. One `.duckdb` file per model version; queryable via SQL from Python and directly from FastAPI. Parquets remain as intermediate pipeline artifacts but DuckDB is the source of truth for the API.

### API
- **FastAPI** — exposes model outputs as REST endpoints. Serves community assignments, shift profiles, predictions, covariance queries. Stateless except for DuckDB reads.

### Frontend
- **React** — app shell, routing, state, controls
- **Deck.gl** — geographic rendering (county/community polygon choropleth, WebGL)
- **Observable Plot** — charts, linked views, shift explorers, scatter plots

### Deployment
- **Fly.io or Railway** — FastAPI container + static React build. ~$5–10/month.

### Config
- **`config/model.yaml`** — all model parameters (K, J, shift_type, vote_share_type, states, holdout pairs, uncontested policy, etc.). No hardcoded constants in pipeline scripts.

---

## Model Versioning Scheme

Each model run produces a versioned snapshot. The following roles are maintained:

| Role | Description | Retention |
|------|-------------|-----------|
| `current` | Active model; what the API and frontend serve | Always kept |
| `previous` | Last model before current; rollback target | Always kept |
| `previous_gen_best` | Best model from the prior architectural generation | Always kept |
| `county_baseline` | The 3-cycle county model (r=0.93–0.98); permanent reference point | Always kept |
| `archived/{date}` | Historical snapshots | Keep last 3; prune older |

Structure:
```
data/models/
├── current/          → symlink to versioned dir
├── previous/         → symlink
├── previous_gen_best/→ symlink
├── county_baseline/  → versioned dir (frozen)
└── versions/
    ├── county_3cycle_20260319/
    ├── county_multiyear_20260319/
    └── county_multiyear_logodds_{date}/   ← next generation
```

Each versioned dir contains:
- `assignments.parquet` — community assignments
- `shifts.parquet` — shift vectors (log-odds)
- `covariance.parquet` — Σ matrix
- `predictions.parquet` — forward predictions
- `meta.yaml` — K, J, config used, validation metrics, git commit hash

---

## Phase 0: Foundational Refactor

*Must complete before any new feature work. This is the last time we build on the wrong foundation.*

### 0.1 — Config system
- Create `config/model.yaml` with all current hardcoded parameters
- Update all pipeline scripts to read from config rather than hardcoding
- Add `config/model_national.yaml` placeholder for future national run

### 0.2 — Shift math: log-odds + total vote share
- Update `src/assembly/build_county_shifts_multiyear.py` to use `logit(dem/total) − logit(dem_prev/total_prev)`
- Update `src/assembly/fetch_medsl_county_presidential.py` to use `totalvotes` denominator (not D+R only)
- Update `src/assembly/fetch_algara_amlani.py` same
- Update `src/assembly/fetch_2022_governor.py` and `fetch_2024_president.py` same
- Re-run full county shift pipeline; output to new versioned dir
- Verify Miami-Dade 2020→2024 log-odds shift captures Hispanic realignment

### 0.3 — DuckDB data layer
- Write `src/db/build_database.py` — reads versioned model dirs, ingests into `data/bedrock.duckdb`
- Schema: `counties`, `communities`, `shifts`, `predictions`, `model_versions`
- Add DuckDB queries to replace pandas parquet loads in validation and viz scripts
- FastAPI will query DuckDB directly

### 0.4 — Model versioning
- Implement the versioned directory structure above
- Tag `county_baseline` (current 3-cycle model) as frozen
- Tag `county_multiyear_v1` (current 30-dim model) as `previous_gen_best`
- New log-odds model will become `current` after Phase 1

### 0.5 — Two-layer architecture in code
- Formalize `Layer1Community` (geographic blob from HAC) and `Layer2Type` (abstract archetype from NMF/k-means)
- Add type-classification step to pipeline: after community discovery, cluster community profiles into J types
- Update docs to reflect two-layer framing throughout

### 0.6 — Integration tests
- Write `tests/integration/test_full_county_pipeline.py`
- Runs full pipeline on FL+GA+AL synthetic mini-dataset (100 counties, 5 years)
- Asserts: correct output shape, no NaN, log-odds range check, DuckDB queryable

---

## Phase 1: County Model v2 (Production-Quality, FL+GA+AL)

*The county-level model becomes a complete, defensible artifact ready for visualization.*

### Deliverables
- [x] Log-odds shift vectors for all 293 FL+GA+AL counties, all available cycles
- [x] Presidential x2.5 weighting + state-centered governor/Senate shifts
- [ ] Senate races added to training (MEDSL Senate data via Harvard Dataverse)
- [x] J selection: KMeans J=20, holdout r=0.778, 10/20 types cross state lines
- [x] Soft type assignments via inverse-distance to KMeans centroids, stored in DuckDB
- [x] Hierarchical nesting: 6-8 super-types via Ward HAC on centroids
- [x] Economist-inspired covariance matrix constructed from demographic profiles
- [x] Type descriptions: time-matched census demographics overlaid on discovered types
- [ ] Turnout feature from dropped uncontested pairs (OQ-002 answered)
- [ ] 3D vs 2D triplet comparison experiment (OQ-003 answered)
- [x] 2026 county-level predictions updated with type-primary structure
- [x] Stained glass map live at bedrock.hhaines.duckdns.org (293 counties colored by super-type)
- [x] Validation report: holdout r=0.778, balanced type sizes (3-31), cross-state types confirmed
- [ ] Historical VEST 2012/2014 expansion for richer shift vectors

---

## Phase 2: API + Visualization Engine

*Something people can actually use. The modeling becomes visible.*

### API (FastAPI + DuckDB)
- `GET /communities` — list communities with profiles
- `GET /communities/{id}` — community detail: counties, shift history, type membership, 2026 prediction
- `GET /counties/{fips}` — county detail: community assignment, all shift dims, demographics
- `GET /shifts?metric=pres_d_shift_16_20&state=FL` — ranked counties by any metric
- `GET /forecast/2026?race=FL_Senate` — community-level prediction with uncertainty
- `GET /types` — list types with profiles and member communities
- `POST /forecast/scenario` — feed a poll, get updated community predictions

### Frontend (React + Deck.gl + Observable Plot)

**View 1: Community Map**
- County choropleth colored by community or type
- Click county → community detail panel
- Metric selector (any shift dimension, or predicted 2026 outcome)
- Time slider showing shift evolution across election pairs

**View 2: Shift Explorer**
- Counties or communities ranked by any metric
- Scatter plot: any two shift dimensions, colored by community/type
- Filter by state, community, type
- Observable Plot charts linked to the map (hover map → highlight chart)

**View 3: 2026 Forecast**
- Community-level predictions with uncertainty
- "Feed a poll" interface: input a state-level poll number, see how predictions update
- Historical accuracy display: how did the community model do in 2020/2022/2024?

**View 4: Community Profiles**
- For each community: shift history, demographic character, type membership
- Comparison between communities of the same type across geographies

---

## Phase 3: Community-Scale Model (FL+GA+AL)

*Move from counties to true sub-county communities where data allows.*

### Key work
- Sub-county communities using VEST tract data (2016–2020 where available)
- Areal interpolation pipeline for pre-2016 VEST data (programmatic, not manual)
- Updated community discovery at tract level where data supports it; fall back to county otherwise
- Candidate effect estimation: decompose election outcomes into community baseline + national environment + candidate residual
- Sabermetrics layer: rank candidates by CTOV (Candidate Total Over Value), estimate drag/lift
- Poll propagation updated for tract-level communities
- Updated 2026 predictions

---

## Phase 4: National

*The full map.*

### Key work
- All 50 states, all 3,143 counties
- MEDSL presidential + governor data for all states (already structured to support this)
- National type discovery: KMeans with presidential weighting + state-centered non-presidential shifts
- J selection at national scale (J ≈ 40-80, research needed)
- Super-type classification at national scale (12-20 super-types)
- Handle non-standard primaries (OQ-001)
- National 2026 predictions: all Senate races, governor races
- National visualization: the map that shows the country's actual political topology

---

## What This Replaces

- `docs/FUTURE_DEV.md` — superseded by this document for roadmap purposes; the detailed feature specs in FUTURE_DEV.md remain valid and will be incorporated into Phase 3 and Phase 4 planning
- The MVP-era "minimize operational complexity" mandate
- The assumption that the county model is a stepping stone to be thrown away — it is a production artifact in its own right

---

## Immediate Next Step

**Phase 1 is substantially complete.** The type-primary pipeline runs end-to-end with KMeans J=20, holdout r=0.778, and a live stained glass map.

Remaining Phase 1 work:
1. Historical VEST 2012/2014 expansion (Task 9 from shift-community-discovery plan)
2. Senate races added to training data
3. Turnout feature from uncontested pairs
4. 3D vs 2D triplet comparison experiment

Next major milestone: **Phase 2 refinement** — polish API endpoints, add interactive controls to the stained glass map, and begin Phase 3 tract-level planning.
