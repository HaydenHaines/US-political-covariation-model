# Project: US Political Covariation Model

A research model that discovers community types from non-political data (religion, class/occupation, neighborhood characteristics), estimates how those community types covary politically using historical election results, and propagates polling information through the learned covariance structure to produce community-level political estimates. The proof-of-concept geography is FL+GA+AL (226 counties), targeting functional predictions by the October 2026 midterms.

**Core insight:** communities that share social identity and behavioral patterns will covary politically, even when geographically separated. Detecting these communities from non-political data and then separately estimating their political covariance avoids circular reasoning and produces a falsifiable model.

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

Six-stage pipeline with strict separation between community detection (non-political) and political validation:

1. **Data Assembly** -- Ingest and harmonize census, ACS, religious congregation, occupation, and neighborhood data at the county level
2. **Community Detection** -- Discover latent community types from non-political features using soft assignment (mixed membership)
3. **Covariance Estimation** -- Estimate how community types covary politically using historical election data (Stan)
4. **Poll Propagation** -- Propagate current polling data through the community covariance structure using MRP (R + Stan)
5. **Prediction / Interpretation** -- Generate county-level dual estimates: vote share and turnout
6. **Validation** -- Holdout backtesting, cross-validation, calibration checks

**Separate silo: Political Sabermetrics** -- Advanced analytics for politician performance. Shares data infrastructure with the covariance pipeline but has its own compute pipeline. Decomposes election outcomes into district baseline + national environment + candidate effect. When the covariance model is available, candidate effects are further decomposed by community type (CTOV). See `docs/SABERMETRICS_ARCHITECTURE.md`.

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
US-political-covariation-model/
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
│   ├── detection/                     # Community type discovery (Python)
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
├── notebooks/                         # Exploratory Jupyter notebooks
├── tests/                             # Unit and integration tests
├── scripts/                           # One-off and utility scripts
├── pyproject.toml                     # Python project config and dependencies
└── renv.lock                          # R dependency lockfile
```

## Conventions

### Research Integrity
- **Two-stage separation is sacred**: Community detection uses ONLY non-political data. Political data enters ONLY at the covariance estimation stage. This is the core falsifiability mechanism -- if non-political community structure does not predict political covariance, the hypothesis is wrong.
- **Assumptions are explicit**: Every modeling assumption is logged in `docs/ASSUMPTIONS_LOG.md` with its status (untested / supported / refuted).
- **Falsification over confirmation**: Design validation to try to break the model, not confirm it. Negative results are documented, not hidden.
- **Reproducibility**: All data transformations are scripted. No manual steps between raw data and outputs. Random seeds are pinned.

### Data
- **Free data only for MVP**: Census, ACS, election returns, FEC, religious congregation data -- all publicly available at no cost.
- **Census tract resolution**: The unit of analysis is the census tract (9,393 in FL+GA+AL per 2022 ACS). See ADR-004. County-level data may be used where tract-level is unavailable, but all community assignments and predictions are at tract level.
- **Soft assignment**: Counties have mixed membership across community types (probability vectors), not hard cluster labels.

### Code
- **Python formatting**: ruff for linting and formatting
- **R formatting**: styler package conventions
- **Stan models**: one .stan file per model, documented parameter blocks
- **Naming**: snake_case everywhere (Python, R, file names)
- **Data flow**: each pipeline stage reads from and writes to `data/` subdirectories; stages are independently re-runnable

### Dual Output
- The model produces two estimates per community-type-county combination: **vote share** (D/R split) and **turnout** (participation rate). These are modeled jointly because they covary.

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

# 2026 forward prediction
python src/prediction/predict_2026.py               # All 2026 races
python src/prediction/predict_2026.py --race "FL Senate"   # Single race
python src/prediction/predict_2026.py --top-counties 10    # More counties shown

# Visualization
python src/viz/build_blended_map.py                 # Community blend map → data/viz/

# Quality
ruff check src/                                     # Lint Python
ruff format src/                                    # Format Python
pytest                                              # Run Python tests (117 tests)
```

## Known Tech Debt

(None yet -- project is pre-implementation.)

## Constraints

- **Free data only for MVP**: No paid data subscriptions. Census, ACS, election returns, congregation data, and public polls are all freely available.
- **Personal research project**: Solo developer, hobby pace. Architecture decisions should minimize operational complexity.
- **October 2026 target**: Functional prediction system for the 2026 midterm elections. This is a hard external deadline.
- **Proof-of-concept geography**: FL+GA+AL only (226 counties). National expansion is a post-MVP goal.
- **Hybrid stack complexity**: Python + R + Stan requires careful interface design. Stan is the bridge -- both cmdstanpy and rstan/cmdstanr can compile and run the same .stan files.
- **No proprietary models**: All modeling code is transparent. No black-box APIs for core inference.

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
