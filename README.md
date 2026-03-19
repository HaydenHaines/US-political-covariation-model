# US Political Covariation Model

A Bayesian model that discovers latent community types from non-political data -- religious affiliation, class and occupation structure, neighborhood characteristics -- and estimates how those community types covary in their political behavior. By learning the covariance structure from historical elections, the model can propagate sparse polling information through communities that share social identity, producing county-level estimates of both vote share and turnout. The key premise is that communities sharing social structure will move together politically, even when they are geographically distant, and that this covariance is detectable without using political data to define the communities in the first place.

## Core Hypothesis

Political behavior at the community level is better predicted by shared social identity and behavioral patterns (religion, class, occupation, neighborhood type) than by geography or broad demographics alone. Communities that share these non-political characteristics will exhibit correlated political shifts -- a pattern that can be learned from historical data and exploited for prediction.

This hypothesis is **falsifiable by design**: the model discovers community types entirely from non-political data, then separately tests whether those types predict political covariance. If they do not, the hypothesis fails cleanly.

## Architecture

The system is a six-stage pipeline with a strict firewall between community detection and political modeling:

```
Data Assembly --> Community Detection --> Covariance Estimation --> Poll Propagation --> Prediction --> Validation
   (Python)          (Python)            (Python + Stan)           (Python)             (Python)     (Python)
```

**Stage 1 -- Data Assembly.** Ingest and harmonize census tract-level data from the Census, American Community Survey, and election return sources. Output: a clean tract-by-feature matrix for FL, GA, and AL (9,393 tracts across 226 counties). Sources: ACS 5-year (demographics, income, education, commute), VEST (precinct election returns 2016-2020), MEDSL (county returns 2022-2024).

**Stage 2 -- Community Detection.** Discover latent community types from the non-political feature matrix using Non-negative Matrix Factorization (NMF). Tracts receive soft assignments (probability vectors across community types), not hard cluster labels, because real tracts contain mixtures of community types. Canonical solution: K=7.

**Stage 3 -- Covariance Estimation.** Using historical election returns, estimate how the discovered community types covary in their political behavior. This is where political data enters the model for the first time. Implemented in Stan (cmdstanpy) as an F=1 factor model. Validated against 2016, 2018, and 2020 elections.

**Stage 4 -- Poll Propagation.** Propagate current polling data through the community covariance structure. Implemented as an analytical Gaussian/Kalman filter update: when a poll captures opinion in one geography, the covariance structure informs estimates for related community types elsewhere. Full MRP (R+Stan) is scaffolded but deferred to post-MVP.

**Stage 5 -- Prediction and Interpretation.** Combine propagated estimates with community-type assignments to produce county-level predictions. The model outputs two quantities jointly: **vote share** (partisan split) and **turnout** (participation rate).

**Stage 6 -- Validation.** Holdout backtesting against known election results, cross-validation, and calibration diagnostics. Designed to stress-test the model, not confirm it.

## Key Innovation: Dual Output

Most political prediction models estimate vote share only, treating turnout as exogenous or ignoring it. This model estimates vote share and turnout jointly through the same community covariance structure. This matters because turnout variation is one of the largest sources of prediction error in elections -- communities that shift in partisan preference often shift in participation simultaneously, and the two are driven by related social dynamics.

## Proof of Concept

The initial implementation covers **Florida, Georgia, and Alabama** (226 counties, 9,393 census tracts). This three-state region was chosen for its political diversity: major metro areas, rural counties, the Black Belt, retirement communities, college towns, military-adjacent communities, and Cuban-American enclaves. It is large enough to test the covariance structure meaningfully and small enough to iterate quickly.

## Technology Stack

| Layer | Technology | Role | Status |
|-------|-----------|------|--------|
| Data assembly | Python (pandas, geopandas, pyarrow) | Ingestion, cleaning, harmonization | Working |
| Community detection | Python (scikit-learn NMF) | Latent community type discovery | Working |
| Covariance estimation | Python + Stan (cmdstanpy) | Bayesian factor model | Working |
| Poll propagation (MVP) | Python (numpy, scipy) | Gaussian/Kalman update | Working |
| Poll propagation (full) | R + Stan (cmdstanr, brms) | Full MRP | Scaffolded, deferred |
| Prediction | Python | Combining estimates, generating outputs | Working |
| Validation | Python | Backtesting, calibration, diagnostics | Working |
| Visualization | Python (matplotlib, plotly) | Maps, diagnostics, results | Partial |
| Sabermetrics | Python | Politician performance analytics | Scaffolded, not started |

## Project Status

**Substantially implemented through Stage 5.** The pipeline runs end-to-end from data assembly through 2026 forecast generation. Architecture is complete and validated.

### Stage summary

| Stage | Status | Key result |
|-------|--------|------------|
| 1 — Data Assembly | Complete | ACS tracts, VEST 2016-2020, MEDSL 2022-2024 fetched and assembled |
| 2 — Community Detection | Complete | K=7 NMF canonical solution; 9,393 tract soft assignments |
| 3 — Covariance Estimation | Complete | R²=0.689/0.636/0.661 across 2016/2018/2020; hypothesis confirmed |
| 4 — Poll Propagation | MVP complete | Gaussian/Kalman update; full MRP (R+Stan) deferred |
| 5 — Prediction | Complete | 2026 forecast pipeline running with placeholder polls |
| 6 — Validation | Complete through 2024 | Catches ~5pp FL poll overestimate of Democrats |

### Primary gaps

- **Test coverage**: 117 real tests covering assembly, detection, covariance, propagation, and sabermetrics modules
- **Additional data sources**: current community detection uses only 12 ACS demographic features; RCMS religious data, IRS migration flows, LODES commuting flows, and Facebook SCI are designed-in but not yet integrated
- **Real poll data**: `data/polls/polls_2026.csv` contains synthetic placeholder polls; real 2026 polls must replace these as the cycle advances
- **Full MRP**: R+Stan propagation pipeline is scaffolded but not implemented; Python Gaussian update is sufficient for the October 2026 target
- **Sabermetrics**: all five sabermetrics source files contain only function signatures; no implemented logic yet

## Community Types (K=7)

NMF applied to ACS features identifies seven latent community types. Each tract receives a probability vector (soft assignment) across these types, not a hard label.

| ID | Label | Key features |
|----|-------|-------------|
| c1 | White rural homeowner | Older, work-from-home, low density |
| c2 | Black urban | Transit use, lower income, high density |
| c3 | Knowledge worker | Management occupations, WFH, college-educated |
| c4 | Asian | Distinct from knowledge worker; different geographic logic |
| c5 | Working-class homeowner | Blue-collar occupations, homeownership |
| c6 | Hispanic low-income | Lower income, distinct from c2 trajectory |
| c7 | Generic suburban baseline | ~60% of tracts; genuine Deep South suburban heterogeneity |

Community ordering by Democratic vote share is stable across all three historical elections: c7 < c1 < c5 < c3 < c4 < c6 < c2.

Key finding: c6 (Hispanic) shows detectable realignment (+4.9pp→+1.2pp Democratic swing, 2016→2020). c6 and c4 (Asian) are negatively correlated over this period -- c4 shifted Democratic as c6 shifted Republican.

## Validation Results

The F=1 Stan factor model was fit on 2016+2018+2020 election returns and validated via holdout:

| Election | R² | Notes |
|----------|-----|-------|
| 2016 presidential | 0.689 | Training set |
| 2018 gubernatorial | 0.636 | FL+GA only (AL race uncontested) |
| 2020 presidential | 0.661 | Training set |
| 2022 gubernatorial | — | Back-calculated via Tikhonov ridge; extends prior chain |
| 2024 presidential | — | Back-calculated via Tikhonov ridge (genuine 2-year lag validation) |

The validation pipeline (`src/validation/`) confirms that the model catches the ~5pp Democratic poll overestimate in Florida that is visible in both 2022 and 2024 actual results.

## Repository Structure

```
docs/           Detailed documentation (architecture, assumptions, data sources, decisions)
research/       Literature review and methods research
src/            Source code organized by pipeline stage
  assembly/     Data ingestion and feature engineering
  detection/    NMF community type discovery
  covariance/   Stan factor model and covariance estimation
  propagation/  Poll propagation (Python MVP + scaffolded R+Stan MRP)
  prediction/   2026 forecast generation
  validation/   Holdout backtesting and calibration
  viz/          Visualization (partial)
  sabermetrics/ Politician analytics (scaffolded)
data/           Data artifacts (gitignored)
notebooks/      Exploratory analysis
tests/          Test suite (117 tests across all pipeline stages)
scripts/        Utility scripts
```

See `docs/ARCHITECTURE.md` for the full technical specification. See `docs/ASSUMPTIONS_LOG.md` for explicit modeling assumptions and their status. See `docs/DECISIONS_LOG.md` for a complete record of architectural decisions and their rationale.

## Quick Start

```bash
# Install
pip install -e .    # or: uv sync

# Run end-to-end (requires data files in data/)
python src/prediction/predict_2026.py

# Run specific validation
python src/validation/validate_2024.py

# Lint
ruff check src/
ruff format src/

# Tests
pytest
```

See `CLAUDE.md` for the full command reference including individual pipeline stage commands.

## License

To be decided.
