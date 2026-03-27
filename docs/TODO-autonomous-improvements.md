# Autonomous Improvement TODOs — Bedrock Model

These tasks can be picked up independently by autonomous sessions. **Work them in priority order.** Each task should be a single session's work, with tests and a commit on `feat/type-primary-architecture`.

**Baseline model (2026-03-22 S164 updated):**
- Algorithm: KMeans J=43
- Features: Presidential×2.5 + state-centered gov/Senate (33 dims, 2008+)
- Holdout r: 0.828 (county-level prior), 0.818 (type-mean prior)
- Calibration MAE: 0.061 (T=10 soft membership)
- Covariance validation r: 0.210 (3-dimensional observed structure)
- County RMSE (2024): **2.67pp** (county-prior), 9.00pp (type-mean prior)
- Feature count: 42 numeric (including migration features)

**Validation command:** `uv run python -m src.validation.validate_types`
**Test command:** `uv run pytest tests/ -q --tb=short`
**Pipeline rebuild:** After any model change, re-run type discovery → nesting → description → covariance → DuckDB rebuild → restart services.

---

## Priority 1 — Fix the Prediction Pipeline (biggest error source)

These are the highest-impact improvements. The prediction pipeline has known structural problems.

- [x] **P1.1: County-level priors instead of type means** — DONE S164. County RMSE: 9.00pp → 2.67pp (70% improvement). Black Belt RMSE: 12.73pp → 3.13pp. See docs/spot-check-county-priors-S164.md.

- [x] **P1.2: Covariance rank reduction via PCA** — DONE S164. Code added (`_rank_reduce`, `max_rank` param) but inactive. Experiment showed validation r is flat across ALL parameter combos. Root cause: demographic similarity ≠ electoral comovement. Observed structure is 3-dimensional. See docs/covariance-experiment-S164.md.

- [x] **P1.3: Negative correlation preservation** — DONE S164. floor_negatives=False makes no difference (0.205 vs 0.202). Dead end — confirmed in same experiment.

---

## Priority 2 — Expand Feature Space (fixes rank deficiency)

41 features for 43 types = near-singular covariance. Need ≥43 features for full rank. These tasks add features from data already on disk or freely available.

- [x] **P2.1: Extend governor shift data pre-2000** — DONE S164. Added 1994 and 1998 governor pairs from Algara data. Training dims: 54 → 60 (+6). Pre-2008, so not used in KMeans but available for county priors.

- [x] **P2.2: Urbanicity feature (Economist-style)** — DONE (pre-S164). `log_pop_density`, `land_area_sq_mi`, `pop_per_sq_mi` already integrated into type_profiles.parquet. The Economist-style `avg_log_pop_within_5_miles` is unnecessary — raw log density already distinguishes urban/suburban/rural effectively.

- [ ] **P2.6: LDS/Mormon adherents feature** — Current RCMS fetch uses "Major Religious Groups" (rt=2) which lumps LDS into "Other." ARDA has denomination-level data — need to find the LDS-specific denomination code and fetch county-level LDS adherent counts separately. Politically distinctive: UT/ID/NV/AZ/WY LDS populations create unique shift patterns (anti-Trump 2016, snap-back 2020+). McMullin 2016 is the canonical signal. Feature: `lds_share = lds_adherents / total_pop`. May also want to check ARDA's denomination-level download (CSV export) for RCMS 2020 which lists 366 individual denominations including "Church of Jesus Christ of Latter-day Saints." Hayden's data scientist friend flagged Mormonism as important (2026-03-27).

- [ ] **P2.3: FEC donor density feature** — Requires FEC_API_KEY (free from api.data.gov/signup/). Fetcher exists at `src/assembly/fetch_fec_contributions.py` (29 tests in worktree). Microdonation rate per county as a type discriminator. **BLOCKED on API key — ask Hayden via Telegram if not set.**

- [ ] **P2.4: BEA income composition** — Requires BEA_API_KEY (free from apps.bea.gov/API/signup/). Fetcher exists at `src/assembly/fetch_bea_income.py` (38 tests). Income from wages vs transfers vs investments. **BLOCKED on API key — ask Hayden via Telegram if not set.**

- [x] **P2.5: IRS migration features** — DONE (pre-S164). 4 features integrated into type profiles. 39 tests.

After completing P2 tasks, re-run shrinkage lambda tuning — the rank deficiency was the reason lambda had no effect (S162). With ≥43 features it may matter.

---

## Priority 3 — Clustering Algorithm Experiments

KMeans at J=43 with r=0.818 is solid. These experiments may find marginal gains but are lower priority than fixing prediction and covariance. **Run one per session. If it doesn't beat baseline, document why and move on.**

- [x] **P3.1: Gaussian Mixture Models** — DONE S164. GMM diagonal beats KMeans at J=43 (0.847 vs 0.832 holdout_r, +1.5%) but KMeans wins at J=50. Marginal gains. Recommendation: keep KMeans. See docs/gmm-experiment-S164.md.

- [x] **P3.2: KMeans stability** — DONE S165. Holdout r = 0.919 ± 0.004 across 50 random seeds. Stable for production. See docs/kmeans-stability-experiment-S165.md.

- [x] **P3.3: Spectral clustering** — DONE S175. Noise-level difference: spectral wins at J=43 (+0.006) but loses at J=50 (-0.009). No systematic advantage. KMeans confirmed. See docs/spectral-experiment-S175.md.

- [x] **P3.4: HDBSCAN with auto-J** — DONE S178. Fails decisively (r=0.53 vs KMeans 0.84). Finds state-level density, not electoral types. All clustering experiments complete except P3.5.

- [x] **P3.5: Turnout as separate dimension** — DONE S186. Current x1.0 weight is optimal. Dropping turnout costs -0.021r; up-weighting trades partisan accuracy for turnout with no net gain. Turnout-only r=0.79. ALL P3.x experiments COMPLETE. See docs/turnout-dimension-experiment-S186.md.

- [ ] **P3.6: PCA/dimensionality reduction before KMeans** — Recommended by external data scientist (2026-03-27). Currently KMeans runs on raw 33-dim shift vectors. PCA could reduce noise dimensions (presidential shifts across years are highly correlated) and concentrate signal. Experiment: run PCA on shift matrix, sweep n_components (5-25), re-run KMeans J=100 on reduced space, compare holdout r to baseline 0.698. The Economist model uses a factor model on shifts (similar concept). Also try UMAP as a nonlinear alternative.

---

## Priority 4 — Validation & Analysis

- [x] **P4.1: Variation partitioning** — DONE S175. Types R²=0.685, Demo R²=0.548, Combined R²=0.777. Unique to types: 22.9%, Unique to demo: 9.1%, Shared: 45.6%, Residual: 22.3%. Types add substantial unique value. See docs/variation-partitioning-S175.md.

- [x] **P4.2: Type stability on sub-windows** — DONE S175. Cross-window ARI=0.113, NMI=0.582, county stability=32.4%. Seed ARI=0.440. Types partially stable — core structure preserved but assignments drift across eras. Full window outperforms sub-windows (r=0.854 vs 0.804/0.838). See docs/type-stability-subwindows-S175.md.

---

## Priority 5 — Frontend & Visualization

These make the product more useful but don't improve model accuracy.

- [x] **P5.1: Type-aware tooltips** — DONE S188. County hover shows type name, super-type, political lean, income, education, white NH %, density. Dark tooltip styling.

- [x] **P5.2: Type naming in legend** — DONE S188. 55 national type names from demographic z-scores. Data-driven super-type names. No "Type N" fallbacks.

- [x] **P5.3: Type comparison table** — DONE S189. "Compare" tab with side-by-side demographics for 2-4 types. Sortable columns. Subtle heatmap highlighting.

- [x] **P5.4: Shift Explorer view** — DONE S189. "Explore" tab with Observable Plot scatter. 55 types as dots, colored by super-type, sized by county count. Selectable X/Y axes.

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

- [x] **P7.1: Real 2026 poll data** — DONE S176+S213. Cron-based poll scraper for 18 national races (6 original + 12 competitive Senate/Governor). 108 polls scraped. Silver Bulletin pollster ratings (539 pollsters). House effects correction (S230).

- [x] **P7.2: Model versioning** — DONE S175. Tagged `type-primary-v1.0` on main. KMeans J=43, holdout r=0.828, county RMSE 2.67pp, 1511 tests.

- [x] **P7.3: CI/CD validation** — DONE S175. GitHub Actions: lint + test + test count threshold.

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
