# US Political Covariation Model

A political modeling platform that discovers electoral types directly from how counties shift across elections, estimates how those types covary using demographic profiles, and propagates polling signals through the covariance structure to produce county-level predictions. The key insight is that beneath the noise of individual elections lies a structural landscape of communities that move together politically -- communities that cross administrative boundaries, persist across decades, and can be discovered purely from spatially correlated electoral shifts.

## Core Hypothesis

Counties that shift similarly across elections -- regardless of geographic distance -- share underlying structural characteristics that make them covary politically. These "electoral types" can be discovered directly from shift vectors (log-odds changes in vote share across election pairs), and their covariance structure (constructed from demographic profiles) can propagate sparse polling information to produce county-level predictions.

This hypothesis is **falsifiable by design**: types are discovered from pre-2024 electoral shifts, then tested against held-out 2024 shifts. If types fail to predict the holdout, the model fails cleanly. Current holdout correlation: **r = 0.778** across 293 counties.

## Architecture

The system is a nine-stage pipeline where types are discovered from electoral shifts, then described and connected via demographics:

```
Data Assembly --> Shift Vectors --> Type Discovery --> Hierarchical Nesting --> Type Description
   (Python)        (Python)         (Python/KMeans)     (Python/Ward HAC)       (Python)

--> Covariance Construction --> Poll Propagation --> Prediction --> Validation
        (Python)                   (Python)           (Python)      (Python)
```

**Stage 1 -- Data Assembly.** Ingest historical election returns (presidential 2000-2024, governor 2002-2022), decennial census (2000/2010/2020), ACS 2022, RCMS religious data, IRS migration flows. Output: county-level data for 293 FL+GA+AL counties.

**Stage 2 -- Shift Vector Computation.** Compute log-odds shift vectors for each county across all election pairs. Presidential shifts are weighted 2.5x (they correlate across state lines); governor/Senate shifts are state-centered (demeaned within each state) to capture within-state differentiation without isolating types by state.

**Stage 3 -- Type Discovery (KMeans J=20).** Cluster counties into J=20 electoral types using KMeans on the weighted, state-centered shift vectors. Types represent abstract archetypes of electoral behavior -- counties that shift similarly belong to the same type regardless of geography. 10 of 20 types span multiple states.

**Stage 4 -- Hierarchical Nesting.** Ward HAC on KMeans centroids (no spatial constraint) produces 6-8 super-types for public interpretability. Super-types are the "colors" of the stained glass map.

**Stage 5 -- Type Description.** Overlay time-matched demographics (interpolated decennial census), RCMS religious data, and IRS migration on discovered types. Types are named from their demographic character, not discovered from it.

**Stage 6 -- Covariance Construction.** Economist-inspired approach (Heidemanns/Gelman/Morris 2020): construct J x J type covariance from demographic profiles via Pearson correlation, shrunk toward all-1s (national swing prior), with PD enforcement. Deterministic Python, no sampling needed.

**Stage 7 -- Poll Propagation.** Gaussian Bayesian update distributes state-level poll signal to types via the covariance structure. Full MRP (R+Stan) deferred to post-MVP.

**Stage 8 -- Prediction.** Type-level estimates multiplied by county soft-membership weights (inverse-distance to KMeans centroids, row-normalized to sum to 1) produce county-level predictions. Dual output: vote share + turnout.

**Stage 9 -- Validation.** Holdout backtesting (train on pre-2024 shifts, test on 2024), type coherence analysis, and calibration diagnostics.

## Key Innovation: Dual Output

Most political prediction models estimate vote share only, treating turnout as exogenous or ignoring it. This model estimates vote share and turnout jointly through the same community covariance structure. This matters because turnout variation is one of the largest sources of prediction error in elections -- communities that shift in partisan preference often shift in participation simultaneously, and the two are driven by related social dynamics.

## Proof of Concept

The initial implementation covers **Florida, Georgia, and Alabama** (293 counties). This three-state region was chosen for its political diversity: major metro areas, rural counties, the Black Belt, retirement communities, college towns, military-adjacent communities, and Cuban-American enclaves. It is large enough to test the covariance structure meaningfully and small enough to iterate quickly.

## Technology Stack

| Layer | Technology | Role | Status |
|-------|-----------|------|--------|
| Data assembly | Python (pandas, geopandas, pyarrow) | Ingestion, cleaning, harmonization | Working |
| Shift vectors | Python (numpy, scipy) | Log-odds shift computation with presidential weighting | Working |
| Type discovery | Python (scikit-learn KMeans) | Electoral type clustering (J=20) | Working |
| Hierarchical nesting | Python (scipy Ward HAC) | Super-type grouping (6-8) | Working |
| Type description | Python (pandas) | Demographic overlay on discovered types | Working |
| Covariance construction | Python (numpy) | Economist-inspired demographic correlation | Working |
| Poll propagation (MVP) | Python (numpy, scipy) | Gaussian/Kalman update | Working |
| Poll propagation (full) | R + Stan (cmdstanr, brms) | Full MRP | Scaffolded, deferred |
| Prediction | Python | Type-level estimates x county weights | Working |
| Validation | Python | Holdout backtesting, calibration | Working |
| Visualization | Next.js + Deck.gl | Stained glass map, interactive frontend | Working |
| API | FastAPI + DuckDB | REST endpoints for model data | Working |
| Sabermetrics | Python | Politician performance analytics | Scaffolded, not started |

## Project Status

**Type-primary architecture implemented end-to-end.** The pipeline runs from data assembly through KMeans type discovery, hierarchical nesting, demographic description, covariance construction, and county-level prediction. Stained glass map live at bedrock.hhaines.duckdns.org.

### Stage summary

| Stage | Status | Key result |
|-------|--------|------------|
| 1 — Data Assembly | Complete | MEDSL presidential 2000-2024, Algara/Amlani governor 2002-2018, Census decennial 2000/2010/2020, ACS 2022, RCMS 2020, IRS migration |
| 2 — Shift Vectors | Complete | Log-odds shifts with presidential x2.5 weighting + state-centered governor/Senate |
| 3 — Type Discovery | Complete | KMeans J=20; 10/20 types span multiple states; balanced sizes (3-31 counties per type) |
| 4 — Hierarchical Nesting | Complete | 6-8 super-types via Ward HAC on centroids |
| 5 — Type Description | Complete | Time-matched census demographics overlaid on discovered types |
| 6 — Covariance Construction | Complete | Economist-inspired demographic correlation with shrinkage |
| 7 — Poll Propagation | MVP complete | Gaussian/Kalman update; full MRP (R+Stan) deferred |
| 8 — Prediction | Complete | 2026 forecast pipeline running with placeholder polls |
| 9 — Validation | Complete | Holdout r=0.778 (train pre-2024, test 2024); catches FL poll overestimate |

### Primary gaps

- **Test coverage**: 641 tests covering assembly, discovery, covariance, propagation, API, and frontend
- **Additional data sources**: RCMS 2020 religious data integrated (293 counties x 6 features). IRS migration edge list integrated (county-to-county flows, 2019-2022). Still pending: LODES commuting flows, Facebook SCI, FEC donor density
- **Real poll data**: `data/polls/polls_2026.csv` contains synthetic placeholder polls; real 2026 polls must replace these as the cycle advances
- **Full MRP**: R+Stan propagation pipeline is scaffolded but not implemented; Python Gaussian update is sufficient for the October 2026 target
- **Historical VEST expansion**: 2012/2014 VEST data would add election pairs for richer shift vectors
- **Sabermetrics**: all five sabermetrics source files contain only function signatures; no implemented logic yet

## Electoral Types (KMeans J=20)

KMeans clustering on presidential-weighted, state-centered shift vectors discovers 20 electoral types. Each county receives soft membership via inverse-distance to KMeans centroids (row-normalized to sum to 1). Types nest into 6-8 super-types via Ward HAC for public interpretability.

### Example types (illustrative)

| Type | Geography | Key demographics | Notable pattern |
|------|-----------|-----------------|-----------------|
| Cross-state rural | AL + FL + GA rural counties | Low density, older, white | Spans 3 states -- validates cross-border hypothesis |
| Atlanta metro professional | Metro Atlanta suburbs | $78K median income, 40% BA+ | High-education suburban cluster |
| Miami-Dade Hispanic enclave | 3 South FL counties | 51% Hispanic | Captures Cuban American shift distinctly |
| Alabama Black Belt | 9 AL counties | 68% Black, $31K income | Historical plantation belt pattern |

10 of 20 types span multiple states, confirming that electoral behavior crosses administrative boundaries. Type sizes are balanced (3-31 counties per type), avoiding the "alternative states" problem of earlier HAC K=10 approach.

Key discovery: presidential shifts at 2.5x weight enable cross-state correlation while state-centered governor/Senate shifts provide within-state differentiation. Raw all-shifts produced state-isolated types; presidential-only lost governor signal; this weighting was the empirical sweet spot.

## Validation Results

Types are discovered from pre-2024 shift vectors, then validated against held-out 2024 presidential shifts:

| Metric | Value | Notes |
|--------|-------|-------|
| Holdout Pearson r | 0.778 | Train on pre-2024 shifts, predict 2024 pres D-share shift |
| Cross-state types | 10/20 | Types spanning multiple states validate cross-border hypothesis |
| Type size range | 3-31 counties | Balanced -- no single type dominates |
| Super-types | 6-8 | Ward HAC on centroids for public interpretability |

### Historical approaches (retained for comparison)

| Approach | Holdout metric | Notes |
|----------|---------------|-------|
| KMeans J=20 (current) | r=0.778 | Presidential x2.5 + state-centered governor |
| HAC K=10 geographic blobs | r=0.903 | High autocorrelation but "alternative states" not types |
| NMF K=7 on demographics | R²=0.661 | Indirect: demographics -> political validation |
| 3-cycle county baseline | r=0.941 | Simple autocorrelation, no community structure |

The KMeans approach trades raw holdout correlation for interpretable, cross-state types that capture genuine electoral structure rather than geographic proximity.

## Repository Structure

```
docs/           Detailed documentation (architecture, assumptions, data sources, decisions)
research/       Literature review and methods research
src/            Source code organized by pipeline stage
  assembly/     Data ingestion, census interpolation, shift computation
  discovery/    KMeans type discovery, shift vectors, adjacency
  description/  Demographic overlay on discovered types
  detection/    Historical NMF discovery (shelved, comparison only)
  covariance/   Economist-inspired covariance construction
  propagation/  Poll propagation (Python MVP + scaffolded R+Stan MRP)
  prediction/   2026 forecast generation
  validation/   Holdout backtesting and calibration
  viz/          Visualization
  sabermetrics/ Politician analytics (scaffolded)
api/            FastAPI backend (REST endpoints, DuckDB)
web/            Next.js frontend (stained glass map)
data/           Data artifacts (gitignored)
notebooks/      Exploratory analysis
tests/          Test suite (641 tests across all pipeline stages)
scripts/        Utility scripts
```

See `docs/ARCHITECTURE.md` for the full technical specification. See `docs/ASSUMPTIONS_LOG.md` for explicit modeling assumptions and their status. See `docs/DECISIONS_LOG.md` for a complete record of architectural decisions and their rationale.

## Quick Start

```bash
# Install
pip install -e .    # or: uv sync

# Run type-primary pipeline
python -m src.discovery.run_type_discovery          # KMeans type discovery
python -m src.description.describe_types            # Demographic overlay
python -m src.covariance.construct_type_covariance  # Covariance construction
python -m src.prediction.predict_2026_types         # 2026 predictions
python -m src.validation.validate_types             # Validation report

# Build DuckDB + run API
python src/db/build_database.py --reset
uvicorn api.main:app --reload --port 8000

# Lint
ruff check src/ api/
ruff format src/ api/

# Tests
pytest
```

See `CLAUDE.md` for the full command reference including individual pipeline stage commands.

## License

To be decided.
