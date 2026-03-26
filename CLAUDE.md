# Project: WetherVane

A political modeling platform that discovers electoral communities directly from spatially correlated shift patterns, estimates how those communities covary, and propagates polling signals through the covariance structure to produce forward predictions. Public-facing; 538-style audience.

**Core insight:** beneath the noise of individual elections is a structural landscape of communities that move together politically. Those communities cross administrative boundaries, persist across decades, and can be discovered purely from how places shift. Understanding this structure — not just the surface results — is what makes prediction defensible.

**Governing principles (updated 2026-03-26):**
- Build it right, not fast. Use the correct model even if it takes longer.
- Build it expandable. Every component has a clear interface.
- Types are primary. KMeans on shift vectors discovers J types. Counties/tracts get soft membership. Types carry covariance and prediction.
- Governor/Senate shifts must be state-centered before cross-state clustering. Presidential shifts carry cross-state signal.
- J selection must be principled (holdout accuracy), not heuristic.
- **θ is the fundamental inference target.** Type means θ are what the model estimates. State/county outcomes are downstream products of θ, not the primary objects of inference.
- **Polls are observations of W·θ.** A poll tells us about the type composition of the polled geography. The model learns θ from it and propagates that inference everywhere those types exist — regardless of state lines.
- **Deviations from expected θ are candidate effects.** Σ + fundamentals generate an expected θ. Posterior deviations are candidate-specific draws (Trump/Rust Belt, W/Hispanic). These are detectable from early polling and propagatable to unpolled geographies.
- The public question is: *What will happen in 2026?* The 2028 question is: *What is each type doing, and why?*

## CRITICAL: For Autonomous Agents

**DO NOT:**
- Revert to SVD+varimax, NMF, or HAC as the primary clustering algorithm. KMeans is the production algorithm. See ADR-006.
- Use raw (non-state-centered) governor/Senate shifts for cross-state clustering. This creates state-isolated types.
- Use the old community_assignments or HAC K=10 model for anything except historical comparison.
- Modify the county-level model without running `uv run python -m src.validation.validate_types` and comparing to baseline.
- Run tract-level experiments without population weighting (tracts with <500 voters are noise).
- Trust the standard holdout r without checking LOO. Standard metric inflates by ~0.22 due to type self-prediction.

**BASELINE METRICS (beat these or don't merge):**
- County holdout r: 0.698 (J=100, StandardScaler+pw=8, national 3,154 counties)
- County holdout LOO r: 0.448 (type-mean baseline, S196; honest generalization metric)
- County holdout LOO r (Ridge): 0.533 (Ridge scores+county_mean, J=100, S197)
- County holdout LOO r (Ridge+all): 0.671 (Ridge scores+county_mean+54 features from 7 sources, N=3,106, S203) — NEW BEST
- County covariance val r: 0.915 (observed LW-regularized, S196; was 0.556 with demographic construction)
- County coherence: 0.783
- County RMSE: 0.073
- County Ridge LOO RMSE: 0.084 (S197)
- County Ridge+Demo LOO RMSE: 0.059 (S197)
- Tract holdout r: 0.632 (J=100, 35 dims, S192)

**Data sources on disk (gitignored, do NOT re-download):**
- `data/raw/fivethirtyeight/` — 538 data (887MB), pollster ratings, polls
- `data/raw/fekrazad/` — 49-state tract-level RLCR vote allocations (320MB)
- `data/raw/dra-block-data/` — block-level 2008-2024 election data
- `data/raw/vest/` — precinct shapefiles 2016-2020
- `data/raw/nyt_precinct/` — NYTimes 2020+2024 precinct data
- `data/raw/tiger/` — TIGER/Line 2020 tract shapefiles
- `data/raw/facebook_sci/` — Facebook Social Connectedness Index (234MB, 10.3M county pairs)
- `data/raw/qcew_county.parquet` — BLS QCEW industry data (104K rows, 3,192 counties, 2020-2023)
- `research/economist-model/` — Economist 2020 model (MIT license)

See `docs/ROADMAP.md` for the full path forward and `docs/TODO-autonomous-improvements.md` for the autonomous improvement queue.

## Self-Improvement Protocol

### When to Update CLAUDE.md
- Architectural decision made --> add to Architecture / Key Decisions Log
- Convention established --> add to Conventions
- Build/test/deploy command confirmed --> add to Commands
- Constraint or gotcha discovered --> add to Constraints
- Project structure changes --> update Directory Map
- New data source integrated --> update Data Sources reference

### When to Update the Reference Web
- Needed the same external information twice --> create a reference file + index entry
- Starting a new pipeline stage --> capture the references you're about to use
- Reference file grows past 150 lines --> split it
- Approach rejected --> move to `_deprecated/` with a note
- See `docs/references/GOVERNANCE.md` for full rules

### When to Flag a Skill
- Same type of task recurs across sessions --> flag to user: "this should become a skill"
- Same reference file consulted 3+ times for the same operation --> flag it
- After using a skill 2-3 times, propose concrete improvements based on what actually happened

### When to Update Memory Files
- Mistake made and corrected --> `memory/lessons.md`
- Debugging technique works --> `memory/debugging.md`
- Workflow improvement found --> `memory/workflow.md`
- User preference expressed --> `memory/preferences.md`

### When to Prune
- Info contradicts codebase --> update this file
- Memory file exceeds 150 lines --> split or summarize
- Something proven wrong --> remove it

---

## Architecture

Two-resolution electoral model (ADR-006):

### County-Level Model (production, live at wethervane.hhaines.duckdns.org)

**Algorithm:** KMeans J=100 on StandardScaler-normalized shifts with presidential weight=8.0 + state-centered governor/Senate shifts (33 dims, 2008+). All 50 states + DC, 3,154 counties.
**Key insight:** Governor/Senate shifts are state-specific races — must be state-centered before cross-state clustering. Presidential shifts carry the cross-state signal. StandardScaler normalizes feature scales; presidential weight=8.0 amplifies the strongest cross-state signal (optimal via LOO sweep over pw=[2..12]).
**Soft membership:** Temperature-scaled inverse distance (T=10, production default). T=10 reduces calibration MAE by ~37% vs T=1.
**Holdout r:** 0.698 (type-mean prior), 0.672 (county-level prior). Coherence=0.783. Covariance val r=0.556. RMSE=0.073.
**Super-types:** 5 super-types (Hispanic Urban, Rural Retiree, Catholic College, Evangelical Coalition, Asian-Pacific Exurban) via Ward HAC on demographic profiles (not centroids — centroids produce degenerate clustering at J=100).

### Tract-Level Model (experimental, toggle on frontend)

**Algorithm:** KMeans J=100, combined features (31 dims: 12 electoral + 19 demographic). Presidential weight=1.0 (applied post-StandardScaler).
**Data source:** DRA (Dave's Redistricting) block-level data aggregated to tracts via GEOID[:11] groupby — NO areal interpolation needed. 84,415 tracts national (all 50 states + DC), 867K tract-race rows.
**Holdout r:** 0.632 (S192). Improved from 0.628 (S191) by adding 4 religion features (national RCMS, 100% coverage). S191: 0.586→0.628 from earlier pres shifts, ACS demographics, J=40→100.
**Population weighting:** Tracts with <500 voters excluded (3,286 dropped → 81,129 tracts).
**Super-types:** 5, derived from Ward HAC on ACS demographic profiles (not KMeans centroids — centroids produce degenerate clustering). Names: Black Urban, Hispanic & Diverse Working, White Mainstream, Urban Professional, Affluent Suburban.
**Visualization:** Bubble dissolve merges adjacent same-type tracts into ~39,561 community polygons (17MB GeoJSON).
**Key findings (S191):** Presidential weighting before StandardScaler has NO EFFECT (bug). Optimal pw=1.0 for combined features. Gov shifts at 54% coverage hurt (excluded). Electoral+Demo beats both electoral-only and demo-only. J keeps improving monotonically but plateaus ~J=100-150.

### Full pipeline:
1. **Data Assembly** -- Historical election returns (presidential 2000-2024, governor 2002-2022, Senate 2002-2022), Decennial Census (2000/2010/2020) with interpolation, ACS 2022, RCMS, IRS migration, FEC donor density. DRA block data for tract-level.
2. **Shift Vector Computation** -- Log-odds shifts. County: 33 training dims (2008+). Tract: 31 dims (12 electoral + 19 demographic).
3. **Type Discovery** -- KMeans clustering. County: presidential×2.5 weighted (J=55). Tract: combined features, pw=1.0 (J=100).
4. **Hierarchical Nesting** -- County: Ward HAC on centroids → super-types. Tract: Ward HAC on demographic profiles (centroids are degenerate).
5. **Type Description** -- Time-matched demographics overlaid on types. Named descriptively.
6. **Type Covariance Construction** -- Observed Ledoit-Wolf regularized correlation from election shifts (primary). Demographic construction retained as fallback. LOEO val_r=0.995.
7. **Poll Propagation** -- Gaussian Bayesian update via type covariance Σ.
8. **Prediction** -- Type-level estimates × soft membership → county/tract predictions.
9. **Validation** -- Leave-one-pair-out CV, calibration, type coherence, type stability.
10. **Visualization** -- County: stained glass map (293 pieces). Tract: bubble dissolve (~2,500 community polygons).

**Historical approaches (shelved, retained for comparison):**
- HAC community-primary (K=10, ADR-005): retained as `county_baseline` in model versioning
- NMF-on-demographics (K=7): original two-stage approach, R²~0.66

**Separate silo: Political Sabermetrics** -- Advanced analytics for politician performance. Shares data infrastructure with the shift discovery pipeline but has its own compute pipeline. Decomposes election outcomes into district baseline + national environment + candidate effect. See `docs/SABERMETRICS_ARCHITECTURE.md`.

See `docs/ARCHITECTURE.md` for the full technical specification.

## Tech Stack

- **Python** -- Data assembly, community detection, feature engineering, prediction, visualization
- **R** -- MRP (multilevel regression and poststratification), poll propagation
- **Stan** -- Bayesian modeling bridge between Python and R ecosystems
- **Key Python packages**: pandas, numpy, scikit-learn, cmdstanpy, pymc (evaluation), geopandas, matplotlib/plotly
- **Key R packages**: brms, rstanarm, tidyverse, survey, lme4
- **Data formats**: Parquet (intermediate data), CSV (raw inputs), NetCDF or Arrow (covariance matrices)
- **Environment**: pyproject.toml (Python), renv.lock (R)

## Directory Map

```
wethervane/
├── CLAUDE.md                          # This file — project conventions for Claude Code
├── README.md                          # Project overview and orientation
├── docs/
│   ├── VISION.md                      # Long-term vision and goals
│   ├── ARCHITECTURE.md                # Full technical architecture
│   ├── ASSUMPTIONS_LOG.md             # Explicit assumptions and their status
│   ├── DATA_SOURCES.md                # Data source catalog and access notes
│   ├── SOURCE_LOG.md                  # Academic and reference sources
│   ├── DECISIONS_LOG.md               # Detailed decision records
│   ├── adr/                           # Architecture Decision Records
│   └── references/                    # Local reference web (see GOVERNANCE.md)
│       ├── GOVERNANCE.md              # Rules for adding/removing references
│       ├── RESOURCE_INDEX.md          # Stage-to-resource lookup table (start here)
│       ├── stan/                      # Stan language, model patterns, cmdstanpy
│       ├── data-sources/              # Census, ACS, election returns, surveys
│       ├── methods/                   # MRP, NMF, Bayesian workflow
│       ├── r-ecosystem/               # brms, rstanarm, survey package patterns
│       └── _deprecated/               # Rejected approaches with notes
├── research/
│   ├── voter-stability-evidence.md    # Literature on voter behavior stability
│   ├── cross-disciplinary-methods.md  # Methods borrowed from other fields
│   ├── community-detection-research.md # Community detection algorithms review
│   ├── methods_and_tools_research.md  # Tools and frameworks evaluation
│   └── moneyball-politician-analytics.md # Political sabermetrics research direction
├── src/
│   ├── assembly/                      # Data assembly pipeline (Python)
│   ├── discovery/                     # Shift-based community discovery (Python)
│   │   ├── shift_vectors.py           # Compute electoral shift vectors from election returns
│   │   ├── adjacency.py               # Build Queen-contiguity spatial adjacency graphs
│   │   └── cluster.py                 # Hierarchical agglomerative clustering with spatial constraints
│   ├── description/                   # Community characterization (Python)
│   │   └── overlay_demographics.py    # Describe discovered communities via ACS/RCMS/LODES/IRS
│   ├── detection/                     # Historical NMF community type discovery (shelved, comparison only)
│   ├── covariance/                    # Historical covariance estimation (Python + Stan)
│   ├── propagation/                   # Poll propagation model (R + Stan)
│   │   ├── stan/                      # Stan model files (.stan)
│   │   └── mrp/                       # MRP implementation (R scripts)
│   ├── prediction/                    # Prediction and interpretation (Python)
│   ├── validation/                    # Validation framework (Python + R)
│   ├── viz/                           # Visualization (Python)
│   └── sabermetrics/                  # Political sabermetrics silo (Python)
│       ├── ingest.py                  # Data download and ID crosswalk
│       ├── baselines.py               # District baseline computation
│       ├── residuals.py               # Candidate residual / CTOV computation
│       ├── legislative.py             # Legislative effectiveness stats
│       └── composites.py              # Career summaries, fit scores, scouting
├── data/                              # All data artifacts (gitignored)
│   ├── raw/                           # Original downloaded data
│   ├── assembled/                     # Cleaned and harmonized county-level data
│   ├── communities/                   # Community type assignments
│   ├── covariance/                    # Estimated covariance matrices
│   ├── polls/                         # Polling data
│   ├── predictions/                   # Model outputs
│   ├── validation/                    # Holdout sets and validation results
│   └── sabermetrics/                  # Politician stat records and composites
├── api/                               # FastAPI backend (Phase 2)
│   ├── main.py                        # App factory, lifespan (DB + sigma + weights)
│   ├── db.py                          # DuckDB dependency
│   ├── models.py                      # Pydantic response models
│   ├── routers/                       # Endpoint implementations
│   │   ├── meta.py                    # GET /health, GET /model/version
│   │   ├── communities.py             # GET /communities, GET /communities/{id}
│   │   ├── counties.py                # GET /counties
│   │   └── forecast.py                # GET /forecast, POST /forecast/poll (stub)
│   └── tests/                         # API unit tests (in-memory DuckDB fixtures)
├── web/                               # Next.js frontend (Phase 2, in progress)
│   └── public/
│       └── counties-fl-ga-al.geojson  # FL+GA+AL county polygons for Deck.gl
├── notebooks/                         # Exploratory Jupyter notebooks
├── tests/                             # Unit and integration tests
├── scripts/                           # One-off and utility scripts
│   ├── fetch_fips_crosswalk.py        # Download Census FIPS→county name mapping
│   └── build_county_geojson.py        # Generate FL+GA+AL GeoJSON from TIGER/Line
├── pyproject.toml                     # Python project config and dependencies
└── renv.lock                          # R dependency lockfile
```

## Conventions

### Research Integrity
- **Types defined by electoral behavior**: Types are discovered directly from county-level shift vectors via KMeans clustering. No demographic inputs to discovery. See ADR-005 (shift-based) and ADR-006 (type-primary pivot).
- **Falsifiability via leave-one-pair-out CV**: Hold out each election pair in turn, predict held-out shifts via type structure. If types fail to predict, the model fails cleanly.
- **Demographics are descriptive + covariance construction**: After discovering types from shifts, overlay time-matched demographics (interpolated decennial census) to characterize types. Demographics also inform the type covariance matrix (Economist-inspired construction), but do NOT influence type discovery.
- **Assumptions are explicit**: Every modeling assumption is logged in `docs/ASSUMPTIONS_LOG.md` with its status (untested / supported / refuted).
- **Falsification over confirmation**: Design validation to try to break the model, not confirm it. Negative results are documented, not hidden.
- **Reproducibility**: All data transformations are scripted. No manual steps between raw data and outputs. Random seeds are pinned.

### Data
- **Free data only for MVP**: Census, ACS, election returns, FEC, religious congregation data -- all publicly available at no cost.
- **County-level primary**: The unit of analysis is the county (3,154 across all 50 states + DC). Tract-level refinement deferred to post-MVP when precinct data coverage improves.
- **Soft assignment**: Counties have mixed membership across types via KMeans inverse-distance scores. Scores are always in [0,1], row-normalized to sum to 1.
- **Census interpolation**: Decennial census (2000/2010/2020) linearly interpolated for election years. Provides time-matched demographics for type description and covariance construction.

### Code
- **Python formatting**: ruff for linting and formatting
- **R formatting**: styler package conventions
- **Stan models**: one .stan file per model, documented parameter blocks
- **Naming**: snake_case everywhere (Python, R, file names)
- **Data flow**: each pipeline stage reads from and writes to `data/` subdirectories; stages are independently re-runnable

### Dual Output
- The model produces two estimates per community-type-county combination: **vote share** (D/R split) and **turnout** (participation rate). These are modeled jointly because they covary.

## Code Quality Rule (MANDATORY)

**Every touch improves the code.** When you modify a file, leave it better than you found it. This is not optional.

Specifically:
- **No hacks.** If a solution feels hacky, stop and find the right one. If there isn't time, document the debt explicitly with `# DEBT:` and a one-line explanation — but this should be rare, not routine.
- **No copy-paste.** If you're about to duplicate logic, extract it. Three similar lines are better than a premature abstraction, but three similar *blocks* are a mandatory extraction.
- **No magic numbers.** Every literal that isn't self-evident (0, 1, "", True, None) gets a named constant or comes from config/data files.
- **No God objects.** If a file does more than one thing, it's a candidate for splitting. If it's over 400 lines, it almost certainly needs splitting. The only exception is data files.
- **No hardcoded data in code.** Thresholds, hyperparameters, and lookup tables belong in config files or data files — not inline in pipeline modules.
- **Comments explain WHY, not WHAT.** Don't comment obvious code. Do comment non-obvious modeling decisions, workarounds, and constraints. Remove dead comments.
- **Tests test behavior, not implementation.** If a test would break when you refactor internals without changing behavior, it's testing the wrong thing.

If you encounter code that violates these rules in a file you're modifying, fix it — even if it wasn't part of your task. If fixing it would be a large detour (>30 min), add a `# DEBT:` comment and create a TODO instead.

The goal: every file in the active codebase should be code you'd be proud to show in a portfolio. This is a public-facing research project.

## Commands

```bash
# Python environment
pip install -e .                                    # Install project in dev mode

# Data ingestion (run once; downloads raw data from MEDSL/VEST)
python src/assembly/fetch_vest_multi_year.py        # VEST 2016/2018/2020 → tract-level
python src/assembly/fetch_2022_governor.py          # MEDSL 2022 governor → county-level
python src/assembly/fetch_2024_president.py         # MEDSL 2024 president → county-level

# Community back-calculation (extend prior chain)
python src/assembly/estimate_2022_community_shares.py  # Stan prior → 2022 governor
python src/assembly/estimate_2024_community_shares.py  # 2022 prior → 2024 president

# Validation (out-of-sample tests)
python src/validation/validate_2020.py              # 2020 holdout: within-state corr
python src/validation/validate_2022.py              # 2022 governor: 2020 prior → actuals
python src/validation/validate_2024.py              # 2024 president: 2022 prior → actuals

# Census interpolation
python -m src.assembly.fetch_census_decennial --all         # Decennial census 2000/2010/2020 → data/assembled/census_{year}.parquet
python -m src.assembly.interpolate_demographics             # Interpolate for election years → data/assembled/demographics_interpolated.parquet

# Type-primary pipeline (current)
python -m src.discovery.select_j                            # J selection sweep via leave-one-pair-out CV
python -m src.discovery.run_type_discovery                  # KMeans → type assignments
python -m src.description.describe_types                    # Overlay time-matched demographics on types
python -m src.covariance.construct_type_covariance          # Observed LW-regularized covariance (primary)
python -m src.prediction.predict_2026_types                 # Type-based 2026 predictions
python -m src.validation.validate_types                     # Type validation report

# Legacy HAC pipeline (retained for comparison)
python src/prediction/predict_2026.py               # All 2026 races (old HAC pipeline)

# Multi-year county data pipeline
python src/assembly/fetch_medsl_county_presidential.py      # MEDSL county pres 2000–2024 (Harvard Dataverse)
python src/assembly/fetch_algara_amlani.py                  # Algara/Amlani governor 2002–2018 (Harvard Dataverse)
python src/assembly/build_county_shifts_multiyear.py        # 54-dim county shift vectors → data/shifts/county_shifts_multiyear.parquet
python -m src.validation.validate_county_holdout_multiyear  # Compare multi-year vs 3-cycle baseline

# Quality
ruff check src/ api/                               # Lint Python
ruff format src/ api/                              # Format Python
python src/assembly/fetch_irs_migration.py          # IRS migration flows (latest 3 year pairs) → data/raw/irs_migration.parquet
pytest                                              # Run all Python tests (src + api)

# Phase 2 — one-time setup (run before building DuckDB)
python scripts/fetch_fips_crosswalk.py             # Download Census FIPS→county name → data/raw/fips_county_crosswalk.csv
python scripts/build_county_geojson.py             # Generate FL+GA+AL county GeoJSON → web/public/counties-fl-ga-al.geojson
python src/db/build_database.py --reset             # Build/rebuild data/wethervane.duckdb

# Phase 2 — run API locally
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000          # API at http://localhost:8000/api/docs
```

## API–Frontend Contract

The API is the contract boundary between model pipeline and frontend. The frontend hardcodes nothing about model shape — all names, counts, and demographics come from API endpoints. See `docs/superpowers/specs/2026-03-21-api-frontend-contract-design.md`.

**Key rules:**
- Frontend reads super-type names and colors from `/super-types` API, never hardcoded
- Demographics render generically from `Record<string, number>` — new features auto-display
- Race strings are opaque labels; frontend groups by `state_abbr`
- `build_database.py` validates contract on exit (required tables, columns, referential integrity)
- API `/health` reports `contract: "ok"` or `"degraded"`
- Integration tests in `tests/test_api_contract.py` validate the full DuckDB→API chain

**If you change the model pipeline:** Run `uv run pytest tests/test_api_contract.py -v` to verify the frontend won't break.

## Known Tech Debt

### Poll Ingestion — Rich Ingestion Model Needed
**DEBT:** Current poll ingestion treats every poll as a scalar `(dem_share, n_sample)` pair. This leaves significant information on the table. Polls with crosstabs tell us *which types* were sampled heavily — a poll that oversampled college-educated voters should pull harder on types with high college-educated membership. The correct fix is to construct poll-specific W vectors from crosstab demographic composition rather than using a generic state-level W. This requires rebuilding the entire poll ingestion pipeline to ingest, store, and process crosstab data, and mapping crosstab demographic groups to the type structure. Until this is done, poll propagation is informationally impoverished.

### ~~Poll Propagation — Multi-Poll Collapse is Mathematically Wrong~~
**RESOLVED 2026-03-26** (feat/forecasting, commit e710dc6): `predict_race` now accepts `polls: list[tuple[float, int, str]]`. Each poll stacks as its own W row. `update_forecast_with_multi_polls` passes the full weighted list instead of collapsing. Do not reopen unless a new mathematical issue is identified.

### ~~Poll Propagation — Fragile State Extraction in API~~
**RESOLVED 2026-03-26** (feat/forecasting, commit e710dc6): `_forecast_poll_types` now passes `polls=[(poll.dem_share, poll.n, poll.state)]` directly. `_extract_state_from_race` deleted. `state_filter` is now output-only. Do not reopen unless the polls interface changes.

### ~~API Contract — `state_pred` Inconsistency~~
**RESOLVED 2026-03-26** (feat/forecasting, commit e710dc6): `_forecast_poll_types` computes `state_pred_val` as mean `pred_dem_share` across counties matching `poll.state`. Both pipelines now populate the field. Do not reopen unless the computation method needs revision.

### Covariance — Cross-Race Underrepresentation
**DEBT:** The type covariance Σ is built primarily from presidential shift data. Cross-race covariance between Senate/governor races and presidential races may be understated. Types are race-agnostic by design, but the Σ structure reflects presidential comovement more than Senate or governor comovement. Worth revisiting once Senate/governor shift data is more fully integrated into covariance construction.

### Fundamentals — State-Level Signal Investigation Needed
**DEBT:** Fundamentals modeling is currently conceived as a national signal ("it's the economy, stupid"). However, regional economic conditions likely vary in how they hit different types — a manufacturing downturn hits Rust Belt working-class types differently than coastal knowledge-worker types. Worth investigating whether BLS, BEA regional Fed, or BEA state-level data can support state- or type-level fundamentals signals rather than a single national scalar.

### Forecast Tab — Map Performance on Recalculate
**DEBT:** The "Recalculate" button in the Forecast tab re-renders the full map choropleth. At national scale (3,154 counties + 81K tracts) this is a heavy draw. Consider rendering a state-only zoomed view when a state is selected in the Forecast tab, rather than re-rendering the full national map on every recalculate.

## Constraints

- **Free data only**: Census, ACS, election returns, congregation data, public polls. No paid subscriptions.
- **October 2026 target**: Functional public prediction tool for the 2026 midterms. Hard external deadline.
- **FL+GA+AL pilot first**: County model and visualization ship before national expansion. But architecture must support national from day one.
- **Build it right, not fast**: Operational complexity is no longer a constraint to minimize. Correct models and expandable architecture take priority.
- **Public-facing**: Code quality, documentation, and methodology must be publication-ready. Assume others will read and attempt to replicate.
- **Hybrid stack**: Python + R + Stan. Stan is the bridge — both cmdstanpy and cmdstanr compile the same .stan files. FastAPI exposes outputs; React + Deck.gl consumes them.
- **No proprietary models**: All inference is transparent and reproducible.

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-10 | Python + R + Stan hybrid stack | Python for data wrangling and ML ecosystem; R for MRP (dominant in political science); Stan as the shared Bayesian engine both can call |
| 2026-03-10 | Community-type soft assignment (mixed membership) | Counties are not monolithic -- a county can be 40% "rural evangelical" and 30% "college-town professional." Hard clustering loses this signal |
| 2026-03-10 | FL+GA+AL proof-of-concept geography (226 counties) | Three-state cluster with diverse political geography (urban/rural, Black Belt, retirement communities, college towns). Large enough to test, small enough to iterate |
| 2026-03-10 | Dual output: vote share + turnout | Turnout variation is a major source of prediction error in political models. Modeling it jointly with vote share through shared community structure captures an effect most models miss |
| 2026-03-10 | Two-stage separation (non-political detection, political validation) | Core falsifiability mechanism. If community types discovered from religion/class/neighborhood do not predict political covariance, the hypothesis fails cleanly |
| 2026-03-10 | Free public data only for MVP | Census, ACS, election returns, congregation data, public polls. No budget for proprietary data. Sufficient for proof of concept |
| 2026-03-18 | K=7 NMF community types (canonical) | Separates Asian from Knowledge Worker/WFH — critical for geographic analysis. R²=0.661 (Strong). ~60% generic baseline is a valid finding, not a flaw. |
| 2026-03-18 | Multi-election validation complete (2016+2018+2020) | R²=0.689/0.636/0.661 — hypothesis confirmed across three cycles. c6 (Hispanic) realignment (+4.9%→+1.2% swing) detected. c6 and c4 negatively correlated — key signal for Stan factor model. Historical election set: 2016 pres + 2018 gov (FL+GA only; AL uncontested) + 2020 pres. |
| 2026-03-18 | Tikhonov ridge regression for 2022/2024 community back-calculation | Bayesian approach (bayesian_poll_update with Stan Sigma) fails for county-level back-calc: data precision (~670K per county × 67 counties) dominates prior precision (~500) by 1340:1 — posterior exits [0,1]. Vote-normalized Tikhonov solves `min_{0≤θ≤1} ||W_w·θ - y_w||² + λ||θ - θ₀||²`. Adaptive λ search finds minimum regularization keeping estimates physical. Prior chain: Stan(2016-2020) → 2022 governor → 2024 president. AL excluded from 2022 (MEDSL data quality); falls back to 2020 prior. |
| 2026-03-18 | Stage 4 poll propagation implemented in Python (not R+Stan MRP) | Lightweight Gaussian Bayesian update in `src/propagation/propagate_polls.py`: mu_post = Sigma_post(Sigma_prior⁻¹ mu_prior + Σ_polls Hᵀσ⁻²y). Uses Stan covariance matrix but does not require running Stan for each forecast. Full MRP (R+Stan) deferred to post-MVP; Python approach sufficient for state-level polls. |
| 2026-03-18 | 2026 forward prediction pipeline complete (placeholder polls) | `src/assembly/ingest_polls.py` + `data/polls/polls_2026.csv` + `src/prediction/predict_2026.py`. FL Senate: polls 43.9% → model 39.5% (R+21); GA Senate: polls 50.5% → model 43.4% (R+13). Model applies structural downward correction to FL consistent with 2022/2024 validation showing ~5pp poll overestimate of Democrats. County predictions saved to `data/predictions/county_predictions_2026.parquet`. |
| 2026-03-18 | RCMS 2020 religious data integrated (county level) | `src/assembly/fetch_rcms.py` scrapes ARDA county map tool (parameterized GET, no login required). Produces `data/raw/rcms_county.parquet` (293 counties, 7 data cols). `build_features.py` computes 6 derived features → `data/assembled/county_rcms_features.parquet`. RCMS is county-level only; ACS tract features remain in separate file. 38 new tests in `tests/test_rcms.py`. |
| 2026-03-18 | IRS SOI county-to-county migration edge list integrated | `src/assembly/fetch_irs_migration.py` downloads IRS SOI inflow CSVs (default: latest 3 year pairs: 2019-2020, 2020-2021, 2021-2022). Filters to flows involving FL/GA/AL, skips aggregate rows (statefips ≥ 96) and non-migrant same-county rows. Produces `data/raw/irs_migration.parquet` (edge list: origin_fips, dest_fips, n_returns, n_exemptions, agi, year_pair). Inflow-only design avoids double-counting. 48 new tests in `tests/test_irs_migration.py`. Feature computation from edge list deferred to post-MVP. |
| 2026-03-18 | Shift-based community discovery replaces NMF-on-demographics | Communities now discovered directly from spatially correlated electoral shift vectors (9-dim: D/R/turnout changes across three election pairs), using hierarchical agglomerative clustering with Queen spatial contiguity. The historical two-stage approach (NMF on ACS+RCMS → political validation) achieved R²~0.66 but imposed an indirect discovery path. Shift-based approach is more direct: communities are defined by how they move politically, not demographic proxies. Falsifiability moves from "do demographics predict covariance?" to "do communities from pre-2024 shifts predict 2024 shifts?" See ADR-005. |
| 2026-03-19 | County level is the primary model engine; tracts are a future refinement layer | Tract-level clustering produces r=-0.14 on holdout (data artifact: 6/9 shift dims are county-level MEDSL, so all tracts within a county receive identical values). County-level clustering produces r=0.93–0.98. County is the realistic common denominator for 2000–2024 data; tract-level refinement deferred to post-MVP when VEST-style harmonized precinct data for pre-2016 cycles becomes available. |
| 2026-03-19 | County model extended to 2000–2024 via MEDSL presidential + Algara/Amlani governor | 3-cycle baseline (2016–2024) extended to 12 cycles (2000–2024). Adds 4 presidential pairs (2000→2004, …, 2012→2016) and 5 governor pairs (2002→2006, …, 2018→2022). Training dims: 6 → 30. Sources: MEDSL doi:10.7910/DVN/VOQCHQ; Algara/Amlani doi:10.7910/DVN/DGUMFI. AL 2018 governor was contested (Ivey vs Maddox, ~60/40) — real data used, no structural zeros. |
| 2026-03-19 | Multi-year holdout r=0.87 (k=5, training col 16→20 vs holdout 20→24); 3-cycle baseline r=0.98 | Delta ~-0.11. Multi-year model is slightly weaker on short-run holdout: 30 dims average over 20 years of electoral history, pulling communities toward older patterns rather than recent momentum. This is expected and acceptable — temporal depth and structural robustness are the multi-year model's value, not short-run autocorrelation. Both approaches confirm strong community structure in FL/GA/AL electoral geography. |
| 2026-03-19 | Phase 1 K selection: K=10 (holdout r=0.9027) | K selection sweep over K=5..30 with spatial Ward HAC and min community size 8. K=10 maximizes holdout Pearson r between community-mean pres_d_shift_16_20 (training) and pres_d_shift_20_24 (holdout). 3-cycle baseline at K=10 was r=0.941; multi-year model gives r=0.9027. |
| 2026-03-19 | Phase 1 NMF types: J=7 | J sweep over 5-8; J=7 chosen for interpretability and consistency with project history. Type weights stored in county_type_assignments.parquet. |
| 2026-03-19 | Phase 1 Stan Σ: county HAC model, T=5 elections | Stan rank-1 factor model fit on 5 elections (2016 pres, 2018 gov, 2020 pres, 2022 gov, 2024 pres). k_ref selected dynamically as most Democratic community. Σ stored at data/covariance/county_community_sigma.parquet. DuckDB has community_sigma table. |
| 2026-03-19 | Phase 2 API architecture: FastAPI + DuckDB read-only + in-process Bayesian update | FastAPI opens wethervane.duckdb read-only at startup; loads K×K sigma, mu_prior, and weight matrices into app.state. All data endpoints are simple SQL queries via Depends(get_db). POST /forecast/poll calls bayesian_update() from src/prediction/predict_2026_hac.py (HAC K=10 pipeline) — NOT propagate_polls.py (old NMF K=7). Test isolation via create_app(lifespan_override=_noop_lifespan) factory + in-memory DuckDB fixture. |
| 2026-03-19 | Phase 2 frontend: Next.js App Router + Deck.gl + Observable Plot | Persistent choropleth map (left) + tabbed right panel. Tab bar holds View 3 (Forecast) now; Views 2 and 4 slot in as future tabs. Community age slider (3–10 training pairs) deferred to future phase. Visual style: Clean Academic (light background, Georgia serif headings, #2166ac/#d73027 partisan colors). |
| 2026-03-20 | Type-primary architecture pivot (ADR-006) | Types become the primary predictive engine, replacing HAC geographic communities. KMeans on 293 county shift vectors (33 training dims, 2008+, presidential×2.5 weighted) discovers J=43 types (J=20 was initial; refined to J=43 via leave-one-pair-out CV). Types carry covariance and prediction. Geographic communities deferred to tract phase. Motivated by: HAC K=10 produced "alternative states" (10 giant blobs), not the stained glass pattern of many small units colored by behavioral type. |
| 2026-03-22 | J=20→43 based on leave-one-pair-out CV sweep | Formal CV across J=12..50. Optimal: J=43 (r=0.779 CV mean). Extended sweep to J=50 shows plateau ~43. Performance gain: holdout r 0.778→0.818 (+4%). Integrated into production pipeline. |
| 2026-03-20 | KMeans for type discovery (not SVD/NMF) | SVD+varimax produced degenerate 2-type solution (r=0.35). NMF requires non-negative input. KMeans achieves holdout r=0.778 with presidential×2.5 weighting. |
| 2026-03-20 | Economist-inspired covariance construction (not Stan estimation) | Construct J×J type covariance from demographic profiles (Pearson correlation + shrinkage toward all-1s + PD enforcement), following Heidemanns/Gelman/Morris 2020. Validated against observed comovement; hybrid fallback if off-diagonal r < 0.4. Stan factor model retained as ultimate fallback. Deterministic Python, no sampling needed. |
| 2026-03-20 | Census interpolation for time-matched demographics | Decennial census 2000/2010/2020 at county level, linearly interpolated for election years. Demographics describe types and construct covariance. ACS 2022 preferred where it provides better temporal resolution. CPI-adjusted income before interpolation. |
| 2026-03-20 | Hierarchical type nesting: J fine types → 5-8 super-types | Fine types give model resolution; super-types give public interpretability. Ward HAC on type loadings (no spatial constraint). Super-types are the "colors" of the stained glass map. |
| 2026-03-20 | 293 counties in FL+GA+AL (not 226) | All three states included. Prior references to 226 counties were FL+GA only (pre-AL inclusion). |
