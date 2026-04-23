# Data Expansion Findings — 2026-04-22

**Source issue:** https://github.com/HaydenHaines/wethervane/issues/95
**Research dispatched by:** agentic-box researcher-visionary (issue label → role routing)
**Scope:** Close out the open research items in GH#95 — VEST 2012/2014 priority, USDA ERS rural-urban codes, FL/GA/AL sub-federal data.

---

## TL;DR

1. **VEST 2014 is HIGHER priority than GH#95 assumes.** DRA block data *does not* cover the 2014 cycle for 47 of 51 states (only MA/MN/NC/TX have `E_14_*` columns). DRA also has zero 2010 coverage. The GH#95 note "DRA now provides all 51 states 2008-2024 at block level" is incorrect for midterm years.
2. **USDA ERS RUCC is low-value, low-cost.** One 3,235-row Excel download, trivial to ingest. Expected signal is redundant with existing urbanicity features (population density, ACS). Worth a 1-hour integration + LOO r check; if +0 keep as a display/segmentation variable, don't promote to feature set.
3. **FL/GA/AL sub-federal data is available but requires meaningful cleanup work.** OpenElections + state SoS sites + Klarner's State Legislative Election Returns (1967–2024) are the three cleanest paths. Expected LOO-r impact is modest given existing 26-dim tract shift vectors; defer unless a pilot on FL state House shows meaningfully-different type assignments.

---

## 1. VEST 2012 / 2014 — Reassessment

### What I checked

Scanned the year coverage in DRA block data across all 51 downloaded states at `data/raw/dra-block-data/*/v*/election_data_block_*.csv`, extracting the `E_NN_` prefixes from each header row.

### Findings

```
States WITH 2014 in DRA:     MA MN NC TX  (4)
States WITHOUT 2014 in DRA:  47 (everything else)
States WITH 2010 in DRA:     none (0)
```

Representative headers:
- **FL:** `E_08_PRES E_12_PRES E_16_PRES E_18_GOV/SEN/AG/TREAS E_20_PRES E_22_SEN/GOV/AG/TREAS/CONG E_24_PRES/SEN`
- **GA:** `E_08_PRES E_12_PRES E_16_PRES/SEN E_18_GOV/AG/LTG E_20_PRES/SEN/SEN_SPEC E_21_SEN_ROFF/SPECROFF E_22_... E_24_PRES/CONG`
- **AL:** `E_08_PRES E_12_PRES E_16_PRES/SEN E_17_SEN_SPEC E_18_GOV/AG/LTG E_20_PRES/SEN E_22_... E_24_PRES/CONG`

No state in DRA has 2014 gubernatorial, Senate, or House data, with the four exceptions above. No state has 2010 data at all.

### Implications

GH#95 said *"VEST 2012/2014 is lower priority unless DRA is missing those years for key states."* The premise is wrong: DRA is missing 2014 for virtually all states.

| Cycle | DRA coverage | Only-via-VEST? | Model relevance |
|---|---|---|---|
| 2010 midterm | 0 states | Yes | Tea Party wave — first midterm post-Obama. Strong signal for Obama→Trump counties. |
| 2012 PRES | 51 states | No (DRA has it) | Low priority; already in the shift pipeline. |
| 2014 midterm | 4 states | **Yes for 47 states** | Obama-era second midterm; precedes 2016 realignment. Would add a second off-cycle shift dim to tract vectors. |

### Recommendation

**Promote VEST 2014 from Low → Medium priority.** Rationale:
- 47 states would be new data, not duplicative of DRA
- Adds a midterm shift dim (2014→2018 gov/sen) that currently doesn't exist at tract resolution
- 2014 is structurally informative: it's the last pre-Trump midterm and captures the tail of the Obama-era realignment for tracts that later flipped
- Existing `docs/TODO-vest-expansion.md` already scopes crosswalk mechanics (2010→2020 tract relationship file + areal interpolation) and acceptance criteria

**Keep VEST 2012 at Low priority.** DRA already covers it at block resolution; VEST would be a validation dataset at best.

**Open question for model owner:** Is the incremental off-cycle shift dim from 2014 worth the ingestion cost? Quick sanity check: run the type-discovery pipeline on current 26 cols + 6 mocked-zero 2014 dims and see if holdout r degrades — if current pipeline is already near ceiling, 2014 may not move the needle. Log a follow-up in `TODO-vest-expansion.md`.

**Out-of-scope but noted:** VEST 2010 would be the only path to midterm data for that year. Deferred unless a specific ADR calls for extending the training window pre-2012.

---

## 2. USDA ERS Rural-Urban Continuum Codes

### What's available

- **2023 RUCC:** 3,235 counties, single Excel file at https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/documentation
- 9 codes: metro (1–3) / nonmetro (4–9) graded by population + metro adjacency
- 2023 revision raised the urban-area threshold from 2,500 → 5,000 per 2020 Census Bureau redefinition
- Historical 2013, 2003, 1993, 1983, 1974 vintages also available (for change-over-time, not needed for the description layer)
- Public domain, no license friction

**Sibling product:** Rural-Urban Commuting Area (**RUCA**) Codes go to census-tract resolution — same office, similar 10-code scheme, updated 2010 (most recent public tract-level release). For the tract-primary model, RUCA is the actually-useful variant.

### Assessment

**Signal overlap risk is high.** wethervane already has:
- ACS population density (implicit in feature pipeline)
- DOT transportation typology (`data/raw/dot_transportation_typology.csv`)
- Type discovery from shift vectors — which naturally separates urban/suburban/rural behavior
- NCHS urban-rural class mentioned in `TODO-data-source-research.md` (not yet integrated)

RUCA/RUCC would be a categorical urbanicity stratifier, not a new behavioral signal. BEA income (merged S339) moved LOO r by +0.002; approval ratings were negligible (S339); FEC was net-negative (S340). RUCC is likely in the same band: near-zero to +0.001.

### Recommendation

**Integrate RUCA (tract-level), not RUCC (county-level).** One script, ~1 hr. Use it as:
- A segmentation/display variable in the frontend (already stated as "Useful as a stratification variable and sanity check" in `DATA_SOURCES.md`)
- A sanity check on type urbanicity post-clustering

**Don't add RUCA to the feature matrix unless the pilot shows > +0.002 LOO r.** Follow the same rejection discipline applied to FEC/approval.

**Script stub to add:**
```
src/assembly/fetch_ruca.py        # downloads 2010 RUCA tract-level .xlsx, converts to parquet
data/raw/ers/ruca_2010_tract.parquet
```

Action: add a line item to `TODO-data-source-research.md` → "RUCA (not RUCC) for tract urbanicity segmentation; integrate as display variable only, pending +LOO r pilot."

---

## 3. FL/GA/AL Sub-Federal Election Data

### Inventory

| Source | Coverage | Access | Notes |
|---|---|---|---|
| **OpenElections** (openelections.net) | All 3 states, federal + statewide + state legislative, 2000–present coverage uneven | GitHub CSVs by state-year | Standardized schema. Quality varies — some state-years fully certified, others stubs. Best starting point for machine-readable data. |
| **FL DoS Division of Elections** | Precinct-level, 2012–present; county-level back to 1978 | Bulk download per election | Most complete source for FL. Requires format normalization across cycles. |
| **GA SoS** | Precinct + county, 2012–2024 | Bulk download | Unique data: runoff + special-election runoffs (2021 Senate, others). Already partially in DRA for GA (E_21_SEN_ROFF). |
| **AL SoS** | County-level archive | Bulk download | Lowest granularity of the three. Precinct-level requires county-by-county FOIA or scraping. |
| **Redistricting Data Hub** | Precinct shapes merged with 2016/18/20 results, all 3 states | API + bulk | Useful for shapefile joins but limited to recent cycles. |
| **Klarner State Legislative Election Returns (1967–2024)** | Every state leg race nationally, **400K+ observations** | Harvard Dataverse | The research gold standard for state-leg data. County- and district-level; candidate-level. This is the *biggest lift/biggest reward* source for adding state-leg signal. |

### Assessment

**The question isn't availability — it's whether state-leg shifts add information beyond federal+statewide shifts.**

From current tract model (S306, S339): county-level LOO r = 0.731 via J=100 + 40 features + PCA. Tract model (T.5) hits LOO r = 0.986 on training tracts. We are near the feature-engineering ceiling for the description layer. The remaining signal gains have been coming from the **electoral shift dimensions themselves** (26 cols after DRA migration), not from new description features.

If state-leg races add *new shift dims* (not new description features), the theoretical case is stronger:
- State-leg races are more heterogeneous within a state (more district-specific candidate effects → noisier but with orthogonal signal)
- They fire in every cycle (including off-years in some states), extending temporal coverage
- They reveal split-ticket behavior more precisely than a single gov/senate race

But the noise concern is real: uncontested races (common for FL/GA/AL state leg), district boundary changes each cycle, and small-N precincts all add cleaning cost.

### Recommendation

**Tiered pilot, defer full integration:**

**Tier 1 (low cost, do now):** Add Klarner SLER data as a county-level *validation* dataset. Not a feature input, just a holdout — "does the covariance structure predict state-leg shifts as well as it predicts federal shifts?" Yes = model generalizes. No = federal-only types miss structure.

**Tier 2 (pilot, 1 sprint):** For FL only, ingest OpenElections state House + state Senate 2018/2020/2022 precinct-level. Crosswalk to tracts. Add 3 shift dims. Re-run type discovery at J=100. Compare holdout r and LOO r.
- **Gate:** if LOO r improves ≥ +0.003 (3× BEA's gain), expand to GA + AL.
- **Gate:** if LOO r flat or negative, stop — federal+statewide already captures the community-level signal.

**Tier 3 (defer):** Full state-leg ingestion for all 50 states. Only pursue if Tier 2 is strongly positive.

### Output docs to create (follow-up, not this session)

- `docs/LOCAL_ELECTION_DATA.md` — expand the inventory above with schemas and scraping scripts
- Not doing it here — that's an ingestion sprint, not a research deliverable

---

## Open questions left for the model owner

1. **Training window extension:** Does the model benefit more from (a) VEST 2014 adding a 2014 off-cycle dim at tract level for 47 new states, or (b) Klarner state-leg adding federal-parallel shift dims at county level for 50 states? Both are midterm-cycle gap fills but at different resolutions.
2. **LOO r ceiling:** The issue states "feature engineering ceiling reached at LOO r=0.734 (county)" and tract LOO r=0.986 is on training tracts, not honest holdout. Should the next evaluation focus on cross-tract generalization (hold-out-state or hold-out-CBSA LOO) before more features?
3. **RUCC vs RUCA:** Any preference for county-level (RUCC) over tract-level (RUCA) urbanicity codes given the tract-primary model direction?

---

## What I changed in this session

- Added this memo: `docs/data-expansion-findings-2026-04-22.md`
- No code changes, no artifact changes, no model retraining
- No updates to `TODO-vest-expansion.md` / `TODO-local-election-research.md` / `TODO-data-source-research.md` — left for the model owner to action once priorities are re-set with these findings

---

## Sources

- [VEST Harvard Dataverse](https://dataverse.harvard.edu/dataverse/electionscience)
- [Harvard Election Data Archive (HEDA)](https://projects.iq.harvard.edu/eda/home) — 2002-2012 + most-states 2014 precinct data
- [HEDA 2014 precinct dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B51MPX)
- [USDA ERS Rural-Urban Continuum Codes](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes)
- [USDA ERS Rural-Urban Commuting Area Codes (RUCA)](https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes)
- [OpenElections](https://openelections.net/)
- [Florida DoS Precinct Results](https://dos.fl.gov/elections/data-statistics/elections-data/precinct-level-election-results/)
- [Georgia SoS Election Results](https://sos.ga.gov/page/georgia-election-results)
- [Redistricting Data Hub](https://redistrictingdatahub.org/data/download-data/)
