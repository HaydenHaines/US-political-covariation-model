# Autonomous Improvement TODOs — Bedrock Model

These tasks can be picked up independently by autonomous sessions. **Work them in priority order.** Each task should be a single session's work, with tests and a commit on `feat/type-primary-architecture`.

**Baseline model (2026-03-22):**
- Algorithm: KMeans J=43
- Features: Presidential×2.5 + state-centered gov/Senate (33 dims, 2008+)
- Holdout r: 0.818
- Calibration MAE: 0.061 (T=10 soft membership)
- Covariance validation r: 0.216
- County RMSE (2024): 8.66pp
- Feature count: 41 numeric (rank-deficient for J=43)

**Validation command:** `uv run python -m src.validation.validate_types`
**Test command:** `uv run pytest tests/ -q --tb=short`
**Pipeline rebuild:** After any model change, re-run type discovery → nesting → description → covariance → DuckDB rebuild → restart services.

---

## Priority 1 — Fix the Prediction Pipeline (biggest error source)

These are the highest-impact improvements. The prediction pipeline has known structural problems.

- [ ] **P1.1: County-level priors instead of type means** — Current prediction uses type-mean Dem share as the prior for every county in that type. This is wrong — a 70% Dem county and a 45% Dem county in the same type get the same prior. County's own historical baseline (e.g., average of last 3 presidential results) should be the prior; types only determine *comovement* (how much a county moves when its type moves). This is the single biggest error source. RMSE=8.66pp, 8/10 worst errors are Black Belt counties assigned to mixed-race types where the type mean is 20pp off from the county's actual level. **No external deps. Unblocked.**

- [ ] **P1.2: Covariance rank reduction via PCA** — Observed covariance has rank ~18 (from 19 elections). Constructed covariance has rank 37+. The model hallucinates more structure than the data supports. Compress constructed covariance to rank ~18 via PCA before shrinkage. Should improve covariance validation r from 0.216. **No external deps. Unblocked.**

- [ ] **P1.3: Negative correlation preservation** — Currently flooring negative correlations to zero in covariance construction. Rural evangelical vs urban progressive types may genuinely inverse-correlate, and zeroing this out loses signal. Test with `floor_negatives=False`. Research: does the Economist floor negatives? **No external deps. Unblocked.**

---

## Priority 2 — Expand Feature Space (fixes rank deficiency)

41 features for 43 types = near-singular covariance. Need ≥43 features for full rank. These tasks add features from data already on disk or freely available.

- [ ] **P2.1: Extend governor shift data pre-2000** — Algara & Amlani data already downloaded, goes back to 1865. Currently using 2002-2018. Add 1994 and 1998 governor pairs. Low effort, no downloads needed. Adds 2 training dims → 35 total.

- [ ] **P2.2: Urbanicity feature (Economist-style)** — Compute `avg_log_pop_within_5_miles` per county. Better than raw density for distinguishing suburban from exurban. Code exists at `src/assembly/build_urbanicity_features.py` (26 tests) but data may need integration into describe_types pipeline.

- [ ] **P2.3: FEC donor density feature** — Requires FEC_API_KEY (free from api.data.gov/signup/). Fetcher exists at `src/assembly/fetch_fec_contributions.py` (29 tests in worktree). Microdonation rate per county as a type discriminator. **BLOCKED on API key — ask Hayden via Telegram if not set.**

- [ ] **P2.4: BEA income composition** — Requires BEA_API_KEY (free from apps.bea.gov/API/signup/). Fetcher exists at `src/assembly/fetch_bea_income.py` (38 tests). Income from wages vs transfers vs investments. **BLOCKED on API key — ask Hayden via Telegram if not set.**

- [ ] **P2.5: IRS migration features** — Code exists at `src/assembly/build_irs_migration_features.py`. Net migration rate, income diversity, flow concentration per county. Data already fetched. Needs integration into describe_types.

After completing P2 tasks, re-run shrinkage lambda tuning — the rank deficiency was the reason lambda had no effect (S162). With ≥43 features it may matter.

---

## Priority 3 — Clustering Algorithm Experiments

KMeans at J=43 with r=0.818 is solid. These experiments may find marginal gains but are lower priority than fixing prediction and covariance. **Run one per session. If it doesn't beat baseline, document why and move on.**

- [ ] **P3.1: Gaussian Mixture Models** — GMM gives proper probabilistic soft membership (vs inverse-distance hack at T=10). Test with full and diagonal covariance. Compare soft scores and holdout r. Most likely to improve on KMeans because it addresses a known weakness (our soft membership is a post-hoc approximation).

- [ ] **P3.2: MiniBatch KMeans stability** — Test centroid stability via bootstrap (100 random starts). Measure how often counties switch types across runs. If unstable, consider ensemble averaging. Lower priority — current model seems stable in practice.

- [ ] **P3.3: Spectral clustering** — Test with k-nearest-neighbors affinity, J=43. Can find non-convex clusters KMeans misses. Research: spectral clustering in electoral geography.

- [ ] **P3.4: HDBSCAN with auto-J** — Discovers J from data density. Interesting but risky — may produce very different J than 43. Counties in sparse regions become "noise."

- [ ] **P3.5: Turnout as separate dimension** — Turnout-shift is redundant with 2-party D/R shifts. Test: cluster on turnout separately, merge with partisan types. Research: does turnout structure add predictive value for midterms?

---

## Priority 4 — Validation & Analysis

- [ ] **P4.1: Variation partitioning** — Decompose holdout variance: how much do types explain vs demographics alone vs their overlap? Important for understanding whether types add value beyond demographics.

- [ ] **P4.2: Type stability on sub-windows** — Compare types from 2008-2016 vs 2016-2024. Current full-window stability fails (89.8° angular distance). Sub-windows should be more stable. Documents whether types are durable or period-specific.

---

## Priority 5 — Frontend & Visualization

These make the product more useful but don't improve model accuracy.

- [ ] **P5.1: Type-aware tooltips** — County hover shows: name, type name, Dem share, key demographics. Highest-value UX improvement.

- [ ] **P5.2: Type naming in legend** — Replace "Super-Type N" with descriptive names from DuckDB super_types table. Small effort.

- [ ] **P5.3: Type comparison table** — Side-by-side demographic profiles of selected types. Sortable columns.

- [ ] **P5.4: Shift Explorer view** — Scatter plot of counties colored by type, x/y axes selectable (e.g., pct_white vs pres_d_shift). For exploration and storytelling.

---

## Priority 6 — Data Source Research (research only, no implementation)

These are research tasks — web search and evaluation, not code. Output should be a findings doc in `docs/` with a recommendation on whether to proceed.

- [ ] **P6.1: County-level presidential returns pre-2000** — Need to extend to 1948+ for parity with Economist. Sources: ICPSR, Dave Leip, OpenElections, Wikipedia. Don't pay for data.

- [ ] **P6.2: CES/CCES survey data** — Individual-level validated vote + county geography. ~60K respondents/wave. Harvard Dataverse. For type validation (do self-reported party ID match type assignments?).

- [ ] **P6.3: VEST 2012/2014 precinct data quality** — Check VEST GitHub for FL/GA/AL coverage. Needed for future tract-level model.

- [ ] **P6.4: FL/GA voter file availability** — Public voter files could validate type assignments against registered party affiliation. Research format, access, academic precedent.

- [ ] **P6.5: Facebook Social Connectedness Index** — For propagation validation (do types that are socially connected also covary politically?). Not for discovery.

---

## Priority 7 — Infrastructure

- [ ] **P7.1: Real 2026 poll data** — Replace placeholder `polls_2026.csv`. Scrape from 538/RCP/Economist. Critical before October 2026 launch.

- [ ] **P7.2: Model versioning** — Tag current model as `type_primary_v1`. Freeze baseline so experiments compare against a pinned version, not a moving target.

- [ ] **P7.3: CI/CD validation** — GitHub Action runs `validate_types` on push to feat/type-primary-architecture. Fail if holdout r drops below 0.80.

---

## Completed (archive)

<details>
<summary>Click to expand completed tasks</summary>

### Immediate Fixes (all done)
- [x] Fix DuckDB county_type_assignments wiring
- [x] Fix super_types table
- [x] Fix stained glass map rendering
- [x] Name the types (S160)
- [x] Compute real type priors (S160)

### Experiments (all done)
- [x] Presidential weight sweep (S161) — plateau 2.0-4.0, current 2.5 fine
- [x] J sweep with formal CV (S162) — J=43 optimal, integrated
- [x] Population-weighted KMeans (S162) — hurts (-0.037), keep equal weighting
- [x] Temporal weighting (S162) — all decay schemes hurt, keep equal
- [x] Shrinkage lambda tuning (S162) — flat across all lambda, rank-deficient features are the root cause

### Validation (all done)
- [x] Write ADR-006 (S161)
- [x] Fix validate_types training dims (S161)
- [x] Calibration analysis (S161) — MAE=0.117, rural +15pp bias, urban -9pp
- [x] Sharpen soft membership (S161) — T=10 reduces MAE 37%
- [x] County spot checks (S163) — Black Belt systematic underprediction found

### Data Sources (downloaded)
- [x] NYTimes 2020 precinct data — MIT license
- [x] NYTimes 2024 precinct data — C-UDA license

### Documentation (all done)
- [x] README.md, ARCHITECTURE.md, ROADMAP.md, ASSUMPTIONS_LOG.md
</details>
