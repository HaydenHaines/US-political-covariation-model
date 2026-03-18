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
# TODO: finalize package manager (uv or pip)
# pip install -e .                             # Install project in dev mode

# Pipeline stages (placeholder — modules not yet implemented)
# python -m src.assembly.run                   # Stage 1: Assemble county-level data
# python -m src.detection.run                  # Stage 2: Discover community types
# python -m src.covariance.run                 # Stage 3: Estimate political covariance
# Rscript src/propagation/mrp/run.R            # Stage 4: MRP poll propagation
# python -m src.prediction.run                 # Stage 5: Generate predictions
# python -m src.validation.run                 # Stage 6: Validate against holdout

# Quality
# ruff check src/                              # Lint Python
# ruff format src/                             # Format Python
# pytest                                       # Run Python tests

# R environment
# Rscript -e "renv::restore()"                 # Restore R dependencies
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
