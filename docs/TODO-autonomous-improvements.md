# Autonomous Improvement TODOs — Bedrock Model

These tasks can be picked up independently by autonomous sessions to incrementally improve the model. Ordered roughly by priority. Each task should be a single session's work, with tests and a commit.

---

## Immediate Fixes

- [x] **Fix DuckDB county_type_assignments wiring** — DONE. Parquet files aligned, all tables populated.
- [x] **Fix super_types table** — DONE. Generated from nesting result.
- [x] **Fix stained glass map rendering** — DONE. Map shows 293 counties in 8 super-type colors.
- [x] **Name the types** — DONE (S160). All 20 types + 8 super-types have descriptive names (e.g., "Black Belt Rural (GA)", "Atlanta Metro Professional"). Stored in DuckDB types + super_types tables.
- [x] **Compute real type priors** — DONE (S160). Priors computed from 2024 actuals. Range: 0.113 (Rural White AL Panhandle) to 0.637 (Deep Black Belt AL). Stored in data/communities/type_priors.parquet.

---

## Iterative Model Improvement (Autonomous Research + Experiment Loop)

**This is the core autonomous workflow.** Each session should:
1. Research a specific improvement hypothesis (web search for papers, methods, datasets)
2. Implement the experiment on a feature branch
3. Run validation and compare to baseline (holdout r=0.778, coherence=0.673)
4. If better: merge. If worse: document why and discard.
5. Update this TODO with findings.

**Baseline model (2026-03-21):**
- Algorithm: KMeans J=20
- Features: Presidential×2.5 + state-centered gov/Senate (33 dims, 2008+)
- Holdout r: 0.778
- Coherence: 0.673
- Covariance validation: 0.449

### How we got here (the pattern to replicate)

This session discovered the optimal approach through empirical iteration:

| Attempt | Algorithm | Data | Holdout r | Issue |
|---------|-----------|------|-----------|-------|
| 1 | SVD+varimax (spec'd) | All 54 dims | 0.35 | Degenerate — 2 giant types |
| 2 | NMF (offset) | All 33 dims | 0.16-0.22 | Worse than SVD |
| 3 | KMeans | All 33 dims (2008+) | 0.77 | State-isolated types |
| 4 | HAC (no spatial) | All 33 dims | 0.76 | Same state isolation |
| 5 | KMeans | Presidential only (15 dims) | 0.70 | Cross-state but lost signal |
| 6 | State-centered KMeans | All 33 dims | 0.72 | Still mostly single-state |
| 7 | **Presidential×2.5 + state-centered** | **33 dims** | **0.778** | **Sweet spot** |

**Key insight:** Governor/Senate shifts are state-specific races. Using them raw creates state blocs. State-centering removes the state-level mean, keeping only within-state differentiation. Presidential weighting at 2.5× ensures cross-state signal dominates type discovery while governor/Senate adds local detail.

**This pattern of hypothesis → experiment → validation → iterate is what autonomous sessions should do.**

### Research-Driven Experiments (pick one per session)

Each of these should start with web research (search for recent papers, blog posts, 538/Economist methodology updates) to inform the experiment design.

- [x] **Presidential weight sweep** — DONE (S161). Tested weights [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]. Best: 3.0 (r=0.655 vs 2.5's r=0.647, +0.009). Broad plateau from 2.0-4.0. Cross-state types: 10 at w=2.5-3.5, 13 at w=4.0-5.0. Current 2.5 is within plateau; consider bumping to 3.0 for marginal gain. Results: data/validation/presidential_weight_sweep.csv

- [ ] **J sweep with formal CV** — Current J=20 was empirically good but not CV-selected with the new feature weighting. Run leave-one-pair-out CV across J=12..30. Research: how do geodemographic classification systems (OAC, PRIZM) select segment count?

- [ ] **Population-weighted KMeans** — Large counties (Miami-Dade: 2.7M, Fulton: 1M) are equally weighted with rural counties (5K). Test `sample_weight=county_population` in KMeans. Research: does the Census use population weighting in their geodemographic classifications?

- [ ] **Temporal weighting** — Weight 2016→2020 shifts at 2× and 2012→2016 at 1.5× to emphasize recent patterns. Research: do the Economist or 538 models use temporal decay? What decay functions work best?

- [ ] **MiniBatch KMeans for stability** — Standard KMeans can be sensitive to initialization. Test MiniBatchKMeans with 100 random starts, measure centroid stability via bootstrap. Research: what's the state of the art for assessing cluster stability?

- [ ] **Spectral clustering** — KMeans assumes spherical clusters. Spectral clustering can find non-convex communities. Test with k-nearest-neighbors affinity. Research: has spectral clustering been applied to electoral geography?

- [ ] **Gaussian Mixture Models** — GMM gives proper probabilistic soft membership (vs our inverse-distance hack). Test with full/diagonal covariance. Compare soft scores. Research: GMM vs KMeans for geodemographic classification.

- [ ] **HDBSCAN with auto-J** — Discovers J automatically from data density. Counties in sparse regions become "noise" (genuinely heterogeneous). Research: HDBSCAN applications in political science / geography.

- [ ] **Add turnout as a separate clustering dimension** — Currently D-shift and R-shift are in the feature space. Turnout-shift is too but it's =1-(D+R), so it's redundant with 2-party data. Test: separate turnout clustering → merge with partisan types. Research: does turnout clustering add predictive value for midterm elections?

- [ ] **Negative correlation preservation** — Currently flooring negatives to zero in covariance. Test with floor_negatives=False. Rural evangelical vs urban progressive may genuinely inverse-correlate. Research: does the Economist floor negatives? Do other models?

- [ ] **Shrinkage lambda tuning** — Current lambda=0.75 from Economist. Their types are 51 states; ours are 20 county types — different granularity. Sweep lambda=0.3..0.95. Research: what's the theoretical basis for the shrinkage parameter?

- [ ] **Urbanicity feature (Economist-style)** — Compute `avg_log_pop_within_5_miles` per county. Much better than raw density for distinguishing suburban from exurban. Use as type profile feature. Research: how did the Economist compute this? What data source?

- [ ] **Add FEC donor density to type profiles** — Microdonation rate (donors/population) as a type discriminator. Post-2012 only. Research: what threshold defines "microdonation"? Is there academic literature on donation rates as political engagement proxy?

### Data Source Research (web search → evaluate → integrate)

- [ ] **Search for county-level presidential returns pre-2000** — Need to extend to 1948+ for parity with Economist model. Sources to investigate: ICPSR, Dave Leip (license restrictions?), OpenElections project, Wikipedia county results tables. Don't pay for data.

- [ ] **Evaluate VEST 2012/2014 precinct data quality** — Already planned as Task 9 in original shift-community plan. Check VEST GitHub for FL/GA/AL coverage and data quality notes.

- [ ] **Research voter file availability** — FL and GA have public voter files. Could validate type assignments against actual registered party affiliation (where available). Research: what's available, what format, any academic papers using FL voter file for ecological inference?

- [ ] **Search for free sub-county demographic data** — ACS provides tract-level but only back to 2009. For 2000, need Census 2000 tract data. Check if API works at tract level for SF1/SF3. Research: NHGIS tract-level extracts.

---

## Data Source Expansion

- [x] **NYTimes 2020 precinct data** — DOWNLOADED. 264MB at `data/raw/nyt_precinct/precincts_2020_national.geojson.gz`. MIT license.
- [x] **NYTimes 2024 precinct data** — DOWNLOADED. CSV (2.1MB) + TopoJSON (183MB) at `data/raw/nyt_precinct/`. C-UDA license (non-commercial, attribution required).
- [ ] **Extend governor shift data pre-2000** — Algara & Amlani has data back to 1865. Currently using 2002-2018. Add 1994, 1998. Low effort (data already downloaded).
- [ ] **FEC donor density feature** — Extend `fetch_fec_contributions.py`. Count unique donors per county per cycle. Reliable signal 2012+.
- [ ] **CES/CCES survey data** — Individual-level validated vote + county geography. ~60K respondents/wave. Harvard Dataverse. For type validation.
- [ ] **BEA Local Area Personal Income** — Income composition (wages vs transfers vs investments). Free API.
- [ ] **Facebook Social Connectedness Index** — For propagation validation, not discovery.

---

## Validation & Analysis

- [x] **Write ADR-006** — DONE (S161). docs/adr/006-type-primary-kmeans-architecture.md
- [x] **Fix validate_types to use training dims only** — DONE (S161). Added --min-year flag (default 2008). Training dims now correctly 33, not 54. Stability still fails (89.8°) — genuine finding: types from 2008-2016 vs 2016-2024 are quite different.
- [ ] **Type stability on recent sub-windows** — Compare 2008-2016 vs 2016-2024 types. Expected to be more stable than full 2000-2024.
- [ ] **County prediction spot checks** — Pinellas FL, Cobb GA, DeKalb GA, Miami-Dade FL. Do predictions match known trends?
- [x] **Calibration analysis** — DONE (S161). scripts/calibration_analysis.py + 30 tests. 2024: MAE=0.117, r=0.787, bias=+0.025 Dem. Rural types +15pp, urban -9pp. Worst: Clayton GA (pred 0.386, actual 0.843). 2020 LOO: MAE=0.165, r=0.741. Key insight: inverse-distance weighting too smooth — predictions compressed to 0.32-0.45 range.
- [ ] **Sharpen soft membership** — S161 experiment in progress. Test temperature-scaled inverse distance to reduce cross-type averaging. Expected to reduce MAE significantly.
- [ ] **Variation partitioning** — How much holdout variance do types explain vs demographics alone vs overlap?

---

## Visualization & Frontend

- [ ] **Type-aware tooltips** — County hover: name, type name, Dem share, key demographics.
- [ ] **Shift Explorer view** — Scatter plot of counties, colored by type, x/y selectable.
- [ ] **Type comparison table** — Side-by-side demographic profiles. Sortable.
- [ ] **Type naming in legend** — Replace "Super-Type N" with descriptive names.

---

## Infrastructure

- [ ] **Real 2026 poll data** — Replace placeholder polls_2026.csv. Scrape from 538/RCP.
- [ ] **Model versioning** — Tag current model as `type_primary_v1`. Freeze as baseline.
- [ ] **CI/CD validation** — GitHub Action runs validate_types on push. Fail below thresholds.

---

## Documentation

- [x] **Update README.md** — In progress (dispatched agent).
- [x] **Update ARCHITECTURE.md** — In progress.
- [x] **Update ROADMAP.md** — In progress.
- [x] **Update ASSUMPTIONS_LOG.md** — In progress.
