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
- **`data/tracts/_deprecated_j130/` contains stale J=130 artifacts.** After the tract-primary migration (T.1-T.7), the current model uses J=100 data from `data/communities/` (type assignments, priors) and `data/covariance/` (type covariance). Stale J=130 files were moved to `_deprecated_j130/` in S508 to prevent accidental loading. The API was silently using these stale files for live poll propagation until S504. Always verify J consistency when loading type data. (Source: S504, S508.)
- **Super-type names in DuckDB must match GeoJSON.** The `build_national_tract_geojson.py` script generates its own super-type names based on demographic composition (e.g., "Black Urban Neighborhood"). These names are embedded in the tract GeoJSON. DuckDB `super_types.display_name` must be updated to match after rebuilding GeoJSON, or the API and map will show different names. After rebuilding tracts: also run `build_state_geojson.py` and update DuckDB names. (Source: 2026-04-10, Black Belt types labeled "white".)
- **Type volatility data must be regenerated after retraining.** `web/public/type-volatility.json` contains per-type average |shift| across presidential pairs. After any retrain that changes type assignments, regenerate with the script block in the commit that added it (computes from `county_shifts_multiyear.parquet` + `type_assignments.parquet`). Stale volatility data causes wrong gold borders on the map. (Source: 2026-04-10.)

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
- Governor Ridge holdout r: 0.696 (train ≤2018, predict 2022, N=32 states, S494)
- Governor Ridge holdout r (with state econ): 0.754 (QCEW employment/wage, sensitivity=0.5, 2026-04-10)
- Governor Ridge holdout bias: +2.2pp (vs presidential +4.6pp)
- Governor Ridge holdout direction accuracy: 87.5%

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

See `~/projects/claude-workspace-meta/process/self-improvement-protocol.md` for the full protocol. Key triggers: architectural decisions → decisions log, gotchas → this file, memory files > 150 lines → split/summarize.

---

## Architecture

### Production Architecture (tract-primary, deployed S341)

Four-layer tract-primary model:

**Layer 1 — Type Discovery (run once):** KMeans on tract-level shift vectors from DRA block data (all 51 states, 2008-2024). Presidential shifts + state-centered off-cycle shifts as separate dimensions. Off-cycle state-centering is a proxy for candidate effect removal (future improvement). ~81K tracts, J=100, soft membership via temperature-scaled inverse distance (T=10).

**Layer 2 — Voter Behavior Layer (NEW):** Per-type parameters learned from historical data:
- τ (turnout ratio): off-cycle turnout / presidential turnout per type. Captures which communities don't show up in midterms.
- δ (residual choice shift): off-cycle Dem share minus expected share from turnout reweighting alone. **DISABLED (S501):** temporal stability analysis shows governor δ has cross-year r=0.091 — cycle-specific noise, not a type property. Infrastructure retained; route choice-level signal through polls instead.
- Binary cycle type: presidential vs off-cycle (turnout is ballot-level, not race-level).
- Governor/Senate results are training data for τ and δ, NOT inputs to type discovery.

**Layer 3 — Covariance:** Ledoit-Wolf regularized covariance on observed tract-level presidential shifts. Same methodology, finer granularity.

**Layer 4 — Prediction:** Ridge priors (tract-level) → behavior adjustment (τ + δ for cycle type) → Bayesian poll update through Σ → tract predictions → vote-weighted state aggregation.

**Frontend:** Tract community polygons as sole map view. County layer removed.

### Historical approaches (shelved, retained for comparison):
- County-primary KMeans (pre-S341): J=100, 3,154 counties, holdout r=0.698. Superseded by tract model.
- HAC community-primary (K=10, ADR-005): retained as `county_baseline`
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
- **Tracts are the unit of analysis**: ~81K tracts from DRA block data. County layer retained for ensemble features and Ridge priors.
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
# Setup
pip install -e .                                    # Install project in dev mode

# Core pipeline (tract-primary)
python -m src.discovery.run_type_discovery                  # KMeans → type assignments
python -m src.description.describe_types                    # Overlay demographics on types
python -m src.covariance.construct_type_covariance          # Observed LW-regularized covariance
python -m src.prediction.predict_2026_types                 # Type-based 2026 predictions
python -m src.validation.validate_types                     # Type validation report

# Data rebuild
python src/db/build_database.py --reset             # Build/rebuild DuckDB
python scripts/build_county_geojson.py              # Rebuild county GeoJSON
python scripts/build_national_tract_geojson.py      # Rebuild tract GeoJSON (after retrain)

# Quality
uv run pytest                                       # Run all tests (4,099)
ruff check src/ api/ && ruff format src/ api/       # Lint + format

# API
uvicorn api.main:app --reload --port 8000           # Local API at http://localhost:8000/api/docs
```

See `docs/ARCHITECTURE.md` for full pipeline documentation including data ingestion and validation commands.

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

### Poll Ingestion — Rich W Vectors (Partially Resolved)
Tiered W vector construction in `src/prediction/poll_enrichment.py`. Tier 1 (crosstab) ready but no data. Tier 2 (methodology adjustments) active. Tier 3 (state fallback) active. Remaining: crosstab data, GA tuning of propensity coefficients. See `docs/TODO-autonomous-improvements.md`.

### Covariance — Cross-Race (RESOLVED 2026-04-01)
Equal-weight combined approach wins (LOEO r=0.9943). Reweighting does not help. Full report: `docs/research/covariance-cross-race-audit.md`.

### Fundamentals — State-Level (RESOLVED 2026-04-10, #88)
QCEW employment/wage signal in `src/prediction/state_economics.py`. Governor backtest +0.010 r.

## Constraints

- **Free data only**: Census, ACS, election returns, congregation data, public polls. No paid subscriptions.
- **October 2026 target**: Functional public prediction tool for the 2026 midterms. Hard external deadline.
- **National scope**: All 50 states + DC. Originally FL+GA+AL pilot, expanded nationally S199+.
- **Build it right, not fast**: Operational complexity is no longer a constraint to minimize. Correct models and expandable architecture take priority.
- **Public-facing**: Code quality, documentation, and methodology must be publication-ready. Assume others will read and attempt to replicate.
- **Hybrid stack**: Python + R + Stan. Stan is the bridge — both cmdstanpy and cmdstanr compile the same .stan files. FastAPI exposes outputs; React + Deck.gl consumes them.
- **No proprietary models**: All inference is transparent and reproducible.

## Key Decisions Log

Full log in `docs/DECISIONS_LOG.md`. Covers all decisions from 2026-03-10 through current, including architecture pivots (ADR-006 type-primary), algorithm choices (KMeans, J=43), and pipeline decisions.
