# Project: WetherVane

A political modeling platform that discovers electoral communities directly from spatially correlated shift patterns, estimates how those communities covary, and propagates polling signals through the covariance structure to produce forward predictions. Public-facing; 538-style audience.

**Core insight:** beneath the noise of individual elections is a structural landscape of communities that move together politically. Those communities cross administrative boundaries, persist across decades, and can be discovered purely from how places shift. Understanding this structure — not just the surface results — is what makes prediction defensible.

**Governing principles (updated 2026-03-27):**
- Build it right, not fast. Use the correct model even if it takes longer.
- Build it expandable. Every component has a clear interface.
- **Types are structural and permanent.** KMeans on shift vectors discovers J types. Tracts get soft membership. Types carry covariance and prediction. Everything downstream is inference *conditioned on* types — types are the nouns, everything else is verbs and adjectives.
- **The model models community behavior, not elections.** Type discovery answers "who moves together." The voter behavior layer answers "how does each community express itself in different electoral contexts." Predictions are a downstream product of community behavior, not the primary object of modeling.
- **Tracts are the sole unit of analysis.** County layer is being retired. DRA block data aggregated to tracts (~81K) provides the granularity needed for pure type signal. See spec: `docs/superpowers/specs/2026-03-27-tract-primary-behavior-layer-design.md`.
- Off-cycle shifts are state-centered before cross-state clustering (proxy for candidate effect removal — future improvement). Presidential shifts carry cross-state signal.
- J selection must be principled (holdout accuracy), not heuristic.
- **θ is the fundamental inference target.** Type means θ are what the model estimates. State/tract outcomes are downstream products of θ, not the primary objects of inference.
- **Polls are observations of W·θ.** A poll tells us about the type composition of the polled geography. The model learns θ from it and propagates that inference everywhere those types exist — regardless of state lines.
- **Deviations from expected θ are candidate effects.** Σ + fundamentals generate an expected θ. Posterior deviations are candidate-specific draws (Trump/Rust Belt, W/Hispanic). These are detectable from early polling and propagatable to unpolled geographies.
- The public question is: *What will happen in 2026?* The 2028 question is: *What is each type doing, and why?*

## CRITICAL: For Autonomous Agents

**DO NOT:**
- Revert to SVD+varimax, NMF, or HAC as the primary clustering algorithm. KMeans is the production algorithm. See ADR-006.
- Use raw (non-state-centered) off-cycle shifts for cross-state clustering. This creates state-isolated types.
- Use the old community_assignments or HAC K=10 model for anything except historical comparison.
- Delete county infrastructure until tract-primary model is validated and deployed. The county model is still production.
- Run tract-level experiments without population weighting (tracts with <500 voters are noise).
- Trust the standard holdout r without checking LOO. Standard metric inflates by ~0.22 due to type self-prediction.
- Feed governor/Senate results into type discovery dimensions. They are training data for the behavior layer only.

## Gotchas

- **GeoJSON must be rebuilt after retraining.** The tract community polygon GeoJSON (`web/public/tracts-us.geojson`) embeds type_id and super_type from the model. After any retrain that changes J, type assignments, or super-types, run `uv run python scripts/build_national_tract_geojson.py`. Failure to rebuild causes choropleth color mismatch — polygons reference stale type IDs that don't match API predictions. (Source: S245, stale J=100 GeoJSON vs J=130 model.)
- **Religious adherence rate is per-1,000, not a fraction.** RCMS data uses "adherents per 1,000 population" convention. Display as `value / 10` with "%" suffix. Do NOT pass through formatPct (which multiplies by 100). (Source: S245, Type 66 showed 53,383%.)
- **Hardcoded model parameters in frontend/data artifacts are a recurring source of bugs.** When J changes, super-type count changes, or column naming changes, stale artifacts break silently. Schedule periodic audits using the hardcoded-values skill. (Source: S245, also S243 column naming mismatch.)
- **DRA tract assignments have duplicate GEOIDs.** The clustering pipeline may produce 112K rows for 81K unique tracts. Always `drop_duplicates(subset="GEOID")` before using as an index. (Source: S245.)

**BASELINE METRICS (beat these or don't merge):**
- County holdout r: 0.698 (J=100, StandardScaler+pw=8, national 3,154 counties)
- County holdout LOO r: 0.448 (type-mean baseline, S196; honest generalization metric)
- County holdout LOO r (Ridge): 0.533 (Ridge scores+county_mean, J=100, S197)
- County holdout LOO r (Ridge+all): 0.731 (Ridge scores+county_mean+40 pruned features, PCA(n=15,whiten=True) before KMeans, N=3,106, S306) — NEW BEST
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

> **⚠️ MIGRATION IN PROGRESS (2026-03-27):** Migrating from county-primary to tract-primary architecture with new voter behavior layer. See spec: `docs/superpowers/specs/2026-03-27-tract-primary-behavior-layer-design.md`. Until migration is complete, the county model remains the production system. Do not delete county infrastructure until tract model is validated and deployed.

### Target Architecture (IN PROGRESS)

Four-layer tract-primary model:

**Layer 1 — Type Discovery (run once):** KMeans on tract-level shift vectors from DRA block data (all 51 states, 2008-2024). Presidential shifts + state-centered off-cycle shifts as separate dimensions. Off-cycle state-centering is a proxy for candidate effect removal (future improvement). ~81K tracts, J=100, soft membership via temperature-scaled inverse distance (T=10).

**Layer 2 — Voter Behavior Layer (NEW):** Per-type parameters learned from historical data:
- τ (turnout ratio): off-cycle turnout / presidential turnout per type. Captures which communities don't show up in midterms.
- δ (residual choice shift): off-cycle Dem share minus expected share from turnout reweighting alone. Captures genuine preference shifts beyond turnout composition.
- Binary cycle type: presidential vs off-cycle (turnout is ballot-level, not race-level).
- Governor/Senate results are training data for τ and δ, NOT inputs to type discovery.

**Layer 3 — Covariance:** Ledoit-Wolf regularized covariance on observed tract-level presidential shifts. Same methodology, finer granularity.

**Layer 4 — Prediction:** Ridge priors (tract-level) → behavior adjustment (τ + δ for cycle type) → Bayesian poll update through Σ → tract predictions → vote-weighted state aggregation.

**Frontend:** Tract community polygons as sole map view. County layer removed.

### Current Production (county-primary, being replaced)

**Algorithm:** KMeans J=100 on StandardScaler-normalized shifts with presidential weight=8.0 + state-centered governor/Senate shifts (33 dims, 2008+). All 50 states + DC, 3,154 counties.
**Holdout r:** 0.698 (type-mean prior). Coherence=0.783. RMSE=0.073.
**Known limitation:** No cycle-type awareness. Ridge priors trained on 2024 presidential outcomes. Off-cycle races predicted with presidential-shaped electorate, systematically overestimating R in midterms.

### Historical approaches (shelved, retained for comparison):
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
├── docs/          # ARCHITECTURE.md, DECISIONS_LOG.md, ROADMAP.md, DATA_SOURCES.md, adr/, references/
├── research/      # Background literature and method comparisons
├── src/           # Pipeline: assembly/, discovery/, description/, covariance/, prediction/, validation/, sabermetrics/
├── data/          # (gitignored) raw/, assembled/, communities/, covariance/, polls/, predictions/
├── api/           # FastAPI + DuckDB backend
├── web/           # Next.js + Deck.gl frontend
├── notebooks/     # Exploratory notebooks
├── tests/         # Unit and integration tests
└── scripts/       # One-off utilities
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
- **Free data only**: Census, ACS, election returns, FEC, religious congregation data -- all publicly available at no cost.
- **Tracts are the unit of analysis** (migration in progress): ~81K tracts from DRA block data. County layer is production until tract model is validated and deployed.
- **Soft assignment**: Tracts/counties have mixed membership across types via KMeans inverse-distance scores. Scores are always in [0,1], row-normalized to sum to 1.
- **Census interpolation**: Decennial census (2000/2010/2020) linearly interpolated for election years. Provides time-matched demographics for type description and covariance construction.

### Code
- **Python formatting**: ruff for linting and formatting
- **R formatting**: styler package conventions
- **Stan models**: one .stan file per model, documented parameter blocks
- **Naming**: snake_case everywhere (Python, R, file names)
- **Data flow**: each pipeline stage reads from and writes to `data/` subdirectories; stages are independently re-runnable

### Dual Output
- The model produces two estimates per community-type-county combination: **vote share** (D/R split) and **turnout** (participation rate). These are modeled jointly because they covary.

## Code Quality Rules (MANDATORY)

**Every touch improves the code.** When you modify a file, leave it better than you found it. This is not optional. If you encounter a violation in a file you're already modifying, fix it. If fixing it would be a large detour (>30 min), drop a `# DEBT:` comment and create a TODO instead.

### Structure
- **No monolithic files.** Files over 400 lines are a mandatory split candidate. One file = one clear responsibility. The only exception is data files.
- **No monolithic functions.** Each function does one thing. If you need "and" to describe what it does, it's two functions.
- **No God objects.** Classes/modules that know too much or do too much are a split candidate.
- **No dead code.** Commented-out blocks, unused variables, unreachable branches — delete them.

### Values & Configuration
- **No magic numbers or strings.** Every literal that isn't self-evident (0, 1, "", True, None) gets a named constant or lives in a config file.
- **No hardcoded parameters in pipeline code.** Thresholds, hyperparameters, lookup tables, model knobs → config files or data files, not inline.
- **No hardcoded data in the frontend.** Names, counts, colors come from the API — never hardcoded. See API–Frontend Contract section.

### Duplication & Abstraction
- **DRY.** Every piece of logic has one unambiguous home. Three similar lines is fine; three similar *blocks* is a mandatory extraction.
- **YAGNI.** Don't build for hypothetical future requirements. The right abstraction is what the task actually needs — no speculative layers.
- **No copy-paste inheritance.** If you duplicated something to "customize it slightly," that's a DRY violation.

### Naming
- **Names are documentation.** `calculate_type_prior()` not `calc()`. Names explain what a value *means*, not how it's stored.
- **No unexplained abbreviations.** `dem_share` is fine. `dsh` is not.
- **Booleans are questions.** `is_off_cycle`, `has_poll_data` — not `flag`, `mode`, `status`.

### Functions & Interfaces
- **One level of abstraction per function.** A function that orchestrates calls should not also contain raw math. A function that does math should not also do I/O.
- **No surprising side effects.** A function named `get_type_weights()` should not write to disk.
- **Fail loud and early.** Validate at system boundaries (file loads, external APIs, user input). Don't silently swallow bad state.

### Comments
- **Comment like a Freshman CS student will review it.** Assume the reader is smart but has never seen this codebase, doesn't know the political science, and has no context for why the model works the way it does. Every non-obvious decision — the math, the modeling choice, the workaround — gets a plain-English explanation. If you had to think for more than 10 seconds about why the code does what it does, that's a comment.
- **Comments explain WHY, not WHAT.** The code says what. Comments say why it must be that way, what was tried before, what constraint forced this design.
- **No stale comments.** A comment that no longer matches the code is worse than no comment. Update or delete.
- **`# DEBT:`** is the only acceptable marker for known violations — with a one-line explanation of what's wrong and why it's not fixed yet.

### Testing
- **Tests test behavior, not implementation.** A test that breaks when you refactor internals without changing behavior is testing the wrong thing.
- **No mocking internals you own.** Mock external boundaries (APIs, filesystems), not your own functions.
- **Every bug gets a regression test.** If it broke once, prove it can't break again.

The goal: every file should be code you'd be proud to show in a portfolio. This is a public-facing research project.

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

### Poll Ingestion — Rich Ingestion Model (Partially Resolved)
**PARTIALLY RESOLVED 2026-03-30** (Phase 4 rich poll ingestion, S251–S252): Tiered W vector construction implemented in `src/prediction/poll_enrichment.py`. Three tiers: Tier 1 (crosstab-based W, structure ready but no crosstab data yet), Tier 2 (LV/RV propensity + methodology-based dimension adjustments via `data/config/poll_method_adjustments.json`), Tier 3 (state-level W fallback). Pipeline wired into `forecast_engine.py` via `w_builder` callable and `type_profiles` param. Poll quality weighting (`prepare_polls()`) applies time decay, pollster grade, and house effects. Integration test (S253): avg 2.64pp shift vs unweighted baseline across 7 Senate races. Core and full W vector modes currently produce identical results because `method_reach_profiles` only has one active entry (`online_panel: log_pop_density_shift`).

**REMAINING DEBT:** (1) No crosstab data ingested yet — Tier 1 is structural scaffolding only. (2) `method_reach_profiles` now has profiles for phone_live, phone_ivr, sms, mail, online_panel, and unknown (issue #86 resolved 2026-04-01); values are research-based estimates, not GA-tuned. (3) No GA tuning of propensity coefficients. See `docs/TODO-autonomous-improvements.md` for the full enrichment TODO list.

### Covariance — Cross-Race Underrepresentation
**RESOLVED (2026-04-01, branch research/covariance-cross-race):** Audited in `scripts/audit_covariance_cross_race.py`. Key findings:
- Per-race covariances (pres/gov/senate) are nearly orthogonal to each other (r~0.12-0.17). Presidential, governor, and Senate elections tap different comovement structures.
- LOEO r: current combined=0.9943, pres-only=0.8495, gov-only=0.9267, senate-only=0.9252. Combined approach wins because it has 20 election pairs vs 5-8.
- The production Σ most resembles presidential covariance (r=0.46) even though pres is only 25% of dims, because presidential shifts are lower-noise.
- **Reweighting does not help.** No tested weight scheme outperforms the current equal-weight combined approach.
- **Downgraded to low priority.** The current approach is adequate. Per-type cross-race divergence (~10-15 types with row_mad>0.50) is real but belongs in the voter behavior layer (τ/δ), not the covariance matrix.
- One data quality finding: early governor pairs (1994→1998, 1998→2002) have ~10x the variance of presidential pairs, suggesting noisy/uncontested races. Worth filtering.
- Full report: `docs/research/covariance-cross-race-audit.md`

### Fundamentals — State-Level Signal Investigation Needed
**DEBT:** Fundamentals modeling is currently conceived as a national signal ("it's the economy, stupid"). However, regional economic conditions likely vary in how they hit different types — a manufacturing downturn hits Rust Belt working-class types differently than coastal knowledge-worker types. Worth investigating whether BLS, BEA regional Fed, or BEA state-level data can support state- or type-level fundamentals signals rather than a single national scalar.

## Constraints

- **Free data only**: Census, ACS, election returns, congregation data, public polls. No paid subscriptions.
- **October 2026 target**: Functional public prediction tool for the 2026 midterms. Hard external deadline.
- **FL+GA+AL pilot first**: County model and visualization ship before national expansion. But architecture must support national from day one.
- **Build it right, not fast**: Operational complexity is no longer a constraint to minimize. Correct models and expandable architecture take priority.
- **Public-facing**: Code quality, documentation, and methodology must be publication-ready. Assume others will read and attempt to replicate.
- **Hybrid stack**: Python + R + Stan. Stan is the bridge — both cmdstanpy and cmdstanr compile the same .stan files. FastAPI exposes outputs; React + Deck.gl consumes them.
- **No proprietary models**: All inference is transparent and reproducible.

## Key Decisions Log

Full log in `docs/DECISIONS_LOG.md`. Covers all decisions from 2026-03-10 through current, including architecture pivots (ADR-006 type-primary), algorithm choices (KMeans, J=43), and pipeline decisions.
