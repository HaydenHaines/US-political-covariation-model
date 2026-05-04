# P6.1: Pre-2000 County Presidential Returns — Source Evaluation

**Date:** 2026-05-04  
**Task:** WetherVane P6.1 (TODO-autonomous-improvements.md)  
**Goal:** Identify a free, FIPS-compatible, county-level presidential returns source covering 1948–2000 to extend the shift-pair training window.

---

## TL;DR

- **Recommended source: Algara & Amlani (Harvard Dataverse, CC0)** — county-level presidential returns 1868–2020, FIPS codes, `.Rdata` format, free and already partially on disk.
- The presidential file (`dataverse_shareable_presidential_county_returns_1868_2020.Rdata`) is in the same Dataverse dataset (doi:10.7910/DVN/DGUMFI) as the governor and senate files already cached in `data/raw/algara_amlani/`.
- FIPS codes are 5-digit zero-padded strings — schema-compatible with `county_fips` in existing parquets.
- Implementation is low-complexity: copy the pattern from `src/assembly/fetch_algara_amlani.py`.
- No viable free alternative exists for pre-2000 data at county granularity.

---

## Source Evaluation

### 1. ICPSR 08611 — Electoral Data for Counties 1840–1972 (Clubb/Flanigan/Zingale)

| Attribute | Detail |
|-----------|--------|
| Coverage | Presidential + House, 1840–1972 |
| Format | Stata, SAS, SPSS (tab-delimited available) |
| License | Free for ICPSR member institutions only; no open license |
| County ID | **NOT FIPS** — variable V3 is an independent numbering system |
| Access | Requires institutional login |
| Verdict | **Unusable** — county ID crosswalk to FIPS required; access restricted |

ICPSR's processor note explicitly states V3 is "not necessarily equivalent to the FIPS code." Joining to WetherVane's `county_fips` schema would require a separate crosswalk file (e.g., Carl Klarner's county-name-to-FIPS crosswalk). Adds significant complexity for no gain over Algara.

### 2. ICPSR 00013 — General Election Data 1950–1990

| Attribute | Detail |
|-----------|--------|
| Coverage | Presidential + Senate + Governor + House, 1950–1990 |
| Format | Tab-delimited, R, SAS, SPSS, Stata |
| License | Free for ICPSR member institutions only |
| County ID | FIPS status **unclear** from documentation |
| Access | Requires institutional login |
| Verdict | **Unusable** — institutional access barrier; FIPS status unverified |

Covers the needed range (1952–1988 presidential), but the same access restriction as 08611 applies. Even if FIPS codes are present, institutional login is required for non-members.

### 3. Dave Leip's Atlas of U.S. Presidential Elections

| Attribute | Detail |
|-----------|--------|
| Coverage | Presidential, 1824–2024, county level |
| Format | CSV, Excel |
| License | **Commercial** — individual $79–$113, site $315–$451 |
| County ID | Includes state + county FIPS codes |
| Access | Purchase required; no free academic tier |
| Verdict | **Unusable** — paid. Task constraint says "Don't pay for data." |

Harvard Library and several universities have site licenses, but WetherVane has no institutional affiliation to leverage. Note: a Harvard Dataverse copy (DVN/SUCQ52) exists, but it appears to be a licensed institutional mirror, not freely downloadable.

### 4. OpenElections

| Attribute | Detail |
|-----------|--------|
| Coverage | Primary focus: 2000+; accepts pre-2000 submissions but no systematic coverage |
| Format | CSV |
| License | CC BY |
| County ID | County FIPS where available |
| Access | Public GitHub repos |
| Verdict | **Unusable for pre-2000** — project explicitly works backward from 2000; sparse and state-by-state for older years |

OpenElections is excellent for 2000+ but not a viable source for systematic 1948–2000 county returns. A partial sweep of state repos found no organized coverage of the target years.

### 5. Algara & Amlani — Harvard Dataverse (doi:10.7910/DVN/DGUMFI) ✓ RECOMMENDED

| Attribute | Detail |
|-----------|--------|
| Coverage | Presidential, 1868–2020; Senate, 1908–2020; Governor, 1865–2020 |
| Format | `.Rdata` (readable via `pyreadr`) |
| License | **CC0 1.0** (public domain, unrestricted use) |
| County ID | Standard 5-digit FIPS, zero-padded (confirmed from `fetch_algara_amlani.py`) |
| Access | Free, no registration required |
| Presidential file | `dataverse_shareable_presidential_county_returns_1868_2020.Rdata` |
| Already on disk? | Gov + Senate files cached; presidential file is one additional download |
| Verdict | **Recommended** |

**Key evidence of FIPS compatibility:** The existing `fetch_algara_amlani.py` at line 147 does `year_df["fips"].astype(str).str.zfill(5)` to produce the `county_fips` column. The presidential file uses the same dataset schema.

**Data sourcing note:** Algara/Amlani sourced raw data from CQ Press and ICPSR United States Historical Election Returns. Their FIPS assignment for pre-1970 elections (1948–1968) handles extinct counties by assigning the modern FIPS of the successor county or retaining the ICPSR code. Some extinct county entries may appear in 1948–1968 that do not exist in the current MEDSL 2000+ parquets — an inner-join in the shift builder will drop these silently (same behavior as the existing pipeline for missing counties).

### 6. Wikipedia Tabular Scraping

| Attribute | Detail |
|-----------|--------|
| Coverage | Results pages exist for all presidential elections |
| Format | HTML tables (inconsistent structure per election/state) |
| License | CC BY-SA (attribution + share-alike required) |
| County ID | County names, not FIPS — requires name-to-FIPS crosswalk |
| Effort | Very high — per-state, per-year scraping + name normalization |
| Verdict | **Unusable** — engineering cost disproportionate; data quality unverifiable |

---

## FIPS Code Compatibility Assessment

**Current WetherVane schema** (`county_fips`): 5-character zero-padded FIPS string (e.g., `"01001"` for Autauga County, AL).

**Algara presidential data**: Uses `fips` column, zero-padded to 5 digits via `str.zfill(5)`. Confirmed equivalent from governor fetch code.

**Known FIPS boundary complications for 1948–2000:**
- Alaska (admitted 1959): First appears in 1960 election with borough/census area FIPS codes. Some codes differ from 2000+ MEDSL codes. Recommend excluding Alaska for pre-1972 elections or mapping to 2000 equivalents.
- Virginia independent cities: Several changed city/county status during this period. Small count, low impact.
- Dade County FL → Miami-Dade (2007): County remained FIPS `12086` throughout, no issue.
- Pre-FIPS elections (1948–1968): FIPS codes were institutionalized ~1970. For counties dissolved before 1970, Algara assigns the modern FIPS of the successor or the ICPSR code. An inner join with the 2000+ county list naturally excludes any extinct-county rows.

**Bottom line:** For 1976–2000 (pairs: 1976→1980 through 1996→2000), FIPS alignment is clean. For 1948–1972 (pairs: 1948→1952 through 1968→1972), expect ~20–50 counties in Alaska and extinct/restructured jurisdictions that will be dropped by inner join. Acceptable precision loss.

---

## Recommended Implementation Path

**Shift pairs unlocked by adding Algara presidential data:**

| Pair | New training dims |
|------|------------------|
| 1948→1952 | 3 (d/r/turnout shift) |
| 1952→1956 | 3 |
| 1956→1960 | 3 |
| 1960→1964 | 3 |
| 1964→1968 | 3 |
| 1968→1972 | 3 |
| 1972→1976 | 3 |
| 1976→1980 | 3 |
| 1980→1984 | 3 |
| 1984→1988 | 3 |
| 1988→1992 | 3 |
| 1992→1996 | 3 |
| 1996→2000 | 3 |

13 new pairs × 3 dims = **39 new training dimensions** (if all are used). In practice, the current model uses 5 presidential pairs; whether to use all 13 or a windowed subset is an architectural decision for the developer/architect role.

**Estimated implementation complexity: Low (2–4 hours)**

1. Add `src/assembly/fetch_algara_presidential.py` following the pattern of `fetch_algara_amlani.py`.
   - Download `dataverse_shareable_presidential_county_returns_1868_2020.Rdata` from same DOI.
   - Filter to years 1948–2000, transform to `medsl_county_presidential_{year}.parquet` column schema.
   - Write one parquet per election year to `data/assembled/`.

2. Update `config.py` to add pre-2000 presidential pairs to `PRES_PAIRS` (or a separate `PRES_PAIRS_HISTORICAL` constant).

3. Update `build_county_shifts_multiyear.py` to ingest the new parquets and compute shift columns for the new pairs.

4. Run existing test suite — no new tests strictly required for the fetch module (mirror the governor test pattern from `tests/test_fetch_algara_amlani.py`).

**Prerequisite:** The `pyreadr` package is already in the project's venv (used by the governor fetch). No new dependencies required.

---

## Open Questions

1. **R object name in presidential `.Rdata` file**: The governor file contains `gov_elections_release`. The presidential file likely contains `pres_elections_release` or similar. Confirm by loading the file with `pyreadr.read_r()` and calling `.keys()`.

2. **Column structure of presidential file**: Likely mirrors the governor schema (`fips`, `election_year`, `office`, `democratic_raw_votes`, `republican_raw_votes`, `raw_county_vote_totals`), but confirm before writing the fetch module.

3. **Alaska pre-1972 handling**: Decide whether to exclude Alaska for elections before Alaska's FIPS codes stabilized (~1972), or map to 2000-era equivalents. Recommend exclusion for simplicity.

4. **How many historical pairs to use**: The model currently uses 5 consecutive presidential pairs (2000–2020). Adding 13 more back to 1948 may introduce signal from structurally different eras (pre-Southern realignment, etc.). The architect/researcher role should decide on windowing or differential weighting.

5. **Training vs. holdout designation**: Currently 2020→2024 is the holdout pair. If historical pairs are added, consider whether any pre-2000 pair should serve as a validation split.

---

## Suggested Follow-ups

- **Task: Implement `fetch_algara_presidential.py`** — Low complexity, follows existing pattern. Unlocks 39 new training dims.
- **Task: Experiment — pre-2000 pairs impact on holdout r** — After implementation, run a sweep adding pairs one window at a time (5→7→10→13) and measure holdout r change.
- **Decision: Handle Alaska pre-1972** — Consult with architect on whether to exclude or map. Recommend exclusion as default.
