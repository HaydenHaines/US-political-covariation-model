# Pre-2000 County Presidential Returns — P6.1 Research Brief

**Date:** 2026-05-12
**Task:** WetherVane P6.1 (TODO-autonomous-improvements.md)
**Requester:** agentic-box research-assistant role
**Sources:** Codebase inspection, `docs/p61-pre2000-county-returns-research.md` (2026-05-04), `docs/research/pre2000-presidential-data-sources.md` (2026-03-27)

---

## TL;DR

- **Status: Implementation already complete.** The Algara & Amlani presidential file (1868–2020, CC0, Harvard Dataverse doi:10.7910/DVN/DGUMFI) is on disk, 14 parquets for 1948–2000 are generated, and `src/assembly/fetch_algara_presidential.py` exists.
- Data is **schema-compatible** with the existing MEDSL pipeline: `county_fips` (5-char zero-padded), `pres_dem_share_{year}` = dem / total_all_candidates.
- **Task note:** P6.1 description says "logit-transformed D/(D+R) shares" — the actual WetherVane shift format uses raw dem/total shares (not logit, not 2-party). The Algara data provides raw vote counts, so either formula is computable, and the existing pipeline uses dem/total.
- Recommendation: **Proceed.** The next step is integrating the parquets into the shift-pair builder, not further data acquisition.

---

## Findings

### Data Status

| File | Status |
|------|--------|
| `data/raw/algara_amlani/dataverse_shareable_presidential_county_returns_1868_2020.Rdata` | **On disk** (1.4 MB, cached 2026-05-04) |
| `data/assembled/algara_county_presidential_{1948..2000}.parquet` | **All 14 generated** |
| `src/assembly/fetch_algara_presidential.py` | **Exists** — downloads, filters, produces parquets |

### Parquet Schema (verified)

```
county_fips           str    5-char zero-padded FIPS  (matches county_fips everywhere)
state_abbr            str    2-char abbreviation
pres_dem_{year}       float  raw Democratic votes
pres_rep_{year}       float  raw Republican votes
pres_total_{year}     float  total_all_candidates (falls back to dem+rep if total=0)
pres_dem_share_{year} float  dem / total_all_candidates
```

Spot check (1948): 3,096 counties, 0 NaN values. 1948 Alabama county 01001 shows Truman 47.4%, 2,445 total votes including Thurmond — denominator is all-candidate total, consistent with MEDSL schema.

Spot check (1996): 3,114 counties, 0 NaN values. Clean.

### Source Evaluation Summary

Prior research (`docs/p61-pre2000-county-returns-research.md`, 2026-05-04) evaluated six sources. Key verdicts:

| Source | Verdict | Reason |
|--------|---------|--------|
| ICPSR 08611 (1840–1972) | Unusable | County ID ≠ FIPS; institutional access required |
| ICPSR 00013 (1950–1990) | Unusable | FIPS status unverified; institutional access required |
| Dave Leip (1824–2024) | Unusable | Commercial license; "Don't pay for data" constraint |
| OpenElections | Unusable | 2000+ focus; pre-2000 coverage sparse |
| **Algara & Amlani (1868–2020)** | **Recommended** | CC0, FIPS-compatible, already on disk |
| Wikipedia scraping | Unusable | Engineering cost disproportionate; quality unverifiable |

See `docs/research/pre2000-presidential-data-sources.md` for the earlier deeper evaluation of ICPSR and Dave Leip alternatives (written before Algara presidential data was identified as the solution).

### Shift Pairs Unlocked

13 new presidential pairs × 3 dims (d-shift, r-shift, turnout-shift) = **39 new training dimensions** available:

```
1948→1952, 1952→1956, 1956→1960, 1960→1964, 1964→1968,
1968→1972, 1972→1976, 1976→1980, 1980→1984, 1984→1988,
1988→1992, 1992→1996, 1996→2000
```

The 1992–1996 gap noted in the March 2026 research is fully resolved — Algara & Amlani covers it. There is no gap.

### Feasibility Assessment

**Schema compatibility:** Full. Parquets use identical column names and types as MEDSL. The existing multi-cycle shift builder (`build_county_shifts_multiyear.py`) can ingest them without column mapping.

**FIPS alignment:** Clean for 1972–2000. For 1948–1968, ~20–50 Alaska/extinct-county rows are excluded by the fetch script (pre-1972 Alaska FIPS instability). Inner join in the shift builder silently drops these — same behavior as existing pipeline for missing counties. Acceptable precision loss.

**Third-party handling (1948, 1968, 1980, 1992):** `pres_dem_share` uses total_all_candidates as denominator. In Thurmond (1948), Wallace (1968), Anderson (1980), Perot (1992) years, this dilutes both D and R shares equally — shifts between adjacent pairs are still computed on a consistent basis. This matches the existing MEDSL approach and is the documented WetherVane convention (Assumptions Log A003).

**Pre-realignment signal:** Pairs before 1968 capture the pre-Southern realignment era when the South was heavily Democratic. These pairs will have large d-shifts (negative in the South post-1964) and may destabilize the type-discovery model unless temporally weighted or windowed. This is an **architectural decision** not a data quality problem — the developer/architect role should decide on windowing strategy before enabling all 13 pairs.

---

## Open Questions

1. **Which pre-2000 pairs to enable:** All 13? A windowed subset (post-realignment only: 1972→2000)? Use all but down-weight via the `temporal_weight` config parameter? The current model uses equal weights across all pairs. This decision belongs to the architect.
2. **Validation split:** Currently 2020→2024 is the holdout. Should any pre-2000 pair serve as a secondary holdout for out-of-sample validation of historical performance?
3. **Alaska pre-1972:** `fetch_algara_presidential.py` excludes Alaska for 1948–1968. Confirm this is the intended handling or add borough-level mapping.

---

## Suggested Follow-ups

- **Queue: Integrate Algara parquets into `build_county_shifts_multiyear.py`** — Low complexity. The fetch module exists; only the shift builder and `config.py` need updating to consume the new parquets. Estimated 1–2 hours.
- **Queue: Experiment — pre-2000 pairs impact on holdout r** — After integration, sweep pairs window (5→7→10→13 pairs) and measure holdout r change. Specifically test whether pre-1972 pairs help or hurt.
- **Decision needed: Temporal windowing strategy** — Flag for architect before enabling all 13 pairs.
