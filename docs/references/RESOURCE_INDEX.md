# Resource Index

Stage key: `S1` Data Assembly · `S2` Community Detection · `S3` Covariance Estimation · `S4` Poll Propagation · `S5` Prediction · `S6` Validation · `ALL` Cross-cutting

Add an entry here **at the same time** as creating a reference file. See `GOVERNANCE.md` for rules.

---

## Cross-cutting (`ALL`)

| File | What it covers | Tags |
|------|---------------|------|
| [skills/anthropic-skills-guidance.md](skills/anthropic-skills-guidance.md) | Skill type taxonomy (9 types), key design tips, gotchas pattern, reference→skill pipeline | `ALL` |

---

## S1 — Data Assembly

| File | What it covers | Tags |
|------|---------------|------|
| [data-sources/census-acs-tract.md](data-sources/census-acs-tract.md) | ACS 5-year API, key tables, MOE flagging, vintage convention | `S1` |
| [data-sources/vest-election-crosswalk.md](data-sources/vest-election-crosswalk.md) | VEST precinct→block→tract crosswalk, download, aggregation strategy | `S1`, `S3` |

**Known gaps to fill as work begins:**
- Religious congregation survey methodology and access (ARDA)

---

## S2 — Community Detection

| File | What it covers | Tags |
|------|---------------|------|
| [methods/nmf-community-detection.md](methods/nmf-community-detection.md) | K selection empirical results, generic baseline interpretation, visualization strategy, NNLS pathology, gotchas | `S2`, `S3` |
| [methods/hdbscan-future-option.md](methods/hdbscan-future-option.md) | HDBSCAN as NMF alternative — explicit noise labeling for heterogeneous tracts, implementation pattern, tradeoffs | `S2` |
| [methods/geographic-visualization.md](methods/geographic-visualization.md) | Deck.gl/Kepler.gl for interactive viz; Google Earth Engine for spatial analysis (watersheds/terrain/geology); KML for quick checks; TIGER geometry join; gotchas | `S5`, `S6` |
| [methods/kepler-gl-loading-guide.md](methods/kepler-gl-loading-guide.md) | Step-by-step: load tract_memberships_k7.geojson into Kepler.gl, set up community intensity layers, tooltips, layer duplication | `S5` |

**Known gaps:**
- Feature normalization conventions for county-level data (tract-level now resolved)

---

## S3 — Covariance Estimation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- Stan hierarchical model patterns for covariance estimation
- cmdstanpy interface: data passing, sampling, diagnostics
- Covariance matrix parameterization in Stan (LKJ prior)

---

## S4 — Poll Propagation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- MRP methodology: multilevel model + poststratification frame construction
- brms/rstanarm syntax for MRP models
- Poststratification frame construction from ACS microdata

---

## S5 — Prediction / Interpretation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- Joint vote share + turnout prediction from shared community structure
- Uncertainty propagation from Stan posteriors through to county-level estimates

---

## S6 — Validation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- Holdout backtesting design for election prediction
- Calibration diagnostics for probabilistic forecasts
- Proper scoring rules for dual-output (vote share + turnout) predictions

---

## _deprecated

| File | What it covered | Why deprecated |
|------|----------------|----------------|
| *(none yet)* | | |
