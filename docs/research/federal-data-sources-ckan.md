# Federal Data Sources from data.gov (CKAN)

**Date:** 2026-03-27
**Context:** Hayden pointed at CKAN/data.gov. Researched what county-level datasets could improve the WetherVane ensemble.

## Already Integrated (Have These)

| Source | Feature | Status |
|--------|---------|--------|
| ACS (Census) | Demographics, income, education, housing, broadband | Integrated |
| QCEW (BLS) | Industry composition by county | Integrated |
| RCMS (ARDA) | Religious adherence (major groups) | Integrated |
| County Health Rankings | Health behaviors, outcomes | Integrated |
| IRS SOI | Migration flows | Integrated |
| Facebook SCI | Social connectedness | Integrated |
| FEC | Donor density | Fetcher exists, **now unblocked** (API key added) |
| BEA | Income composition | Fetcher exists, **now unblocked** (API key added) |

## High-Value New Sources (from data.gov/CKAN)

### 1. CDC PLACES: Local Data for Better Health (County Data 2023)
- **What:** 36 chronic disease measures, health behaviors, prevention metrics — model-based estimates for ALL US counties
- **Why it matters:** Health behaviors (smoking, obesity, exercise) are strongly correlated with political lean. We have County Health Rankings but PLACES has more granular measures.
- **Format:** CSV, JSON, API
- **URL:** data.gov search "PLACES county"
- **Priority:** MEDIUM — may overlap with existing CHR data. Check for unique measures.

### 2. CDC Provisional County-Level Drug Overdose Death Counts
- **What:** Opioid/drug mortality rates by county
- **Why it matters:** Opioid crisis counties shifted R dramatically 2012-2020. This captures "deaths of despair" geography that overlaps with but isn't identical to existing health data.
- **Format:** CSV
- **Priority:** HIGH — unique signal not in current features

### 3. VA Disability Compensation Recipients by County
- **What:** Number of veterans receiving disability benefits per county
- **Why it matters:** Veteran density is politically distinctive — high-veteran counties lean R but with specific VA-related policy sensitivity. Military base proximity creates economic dependency.
- **Format:** CSV (FY2024 available)
- **Priority:** MEDIUM — proxy for veteran/military community density

### 4. USDA Atlas of Rural and Small-Town America
- **What:** Comprehensive rural indicators: population change, age structure, farm dependency, mining dependency, recreation dependency, federal lands, persistent poverty, retirement destination
- **Why it matters:** Rural-Urban Continuum Codes and Urban Influence Codes are already used, but the Atlas has ~80 additional county-level variables including economic typology (farming, mining, manufacturing, government, recreation, nonspecialized).
- **Format:** CSV/Excel
- **URL:** ers.usda.gov
- **Priority:** HIGH — economic typology codes (farming vs mining vs government-dependent) would directly help distinguish rural types

### 5. CDC Social Determinants of Health (SDOH) Measures
- **What:** ACS-derived social determinants aggregated at county level — housing burden, transportation, food access, education
- **Why it matters:** Pre-computed composite measures that may capture signal we're missing from raw ACS variables.
- **Format:** CSV
- **Priority:** LOW — likely redundant with our ACS features

### 6. USDA SNAP County-Level Data
- **What:** SNAP participation rates by county (via USDA Food & Nutrition Service)
- **Why it matters:** Transfer dependency is a distinct political signal. SNAP participation rate captures "government-dependent" communities differently than income alone.
- **Format:** Not on data.gov directly — available from USDA FNS or the Atlas of Rural America
- **Priority:** MEDIUM — partially captured by BEA transfer income (once integrated)

## Sources Worth Investigating Further

| Source | What | Where to Find |
|--------|------|---------------|
| FBI UCR / NIBRS | Crime rates by county | crime-data-explorer.fr.cloud.gov |
| ATF FFL data | Gun dealer density by county (proxy for gun culture) | atf.gov/resource-center |
| USDA NASS | Crop production, farm income by county | quickstats.nass.usda.gov |
| DOE EIA | Energy production by county (oil, gas, coal, renewables) | eia.gov |
| DOL OEWS | Occupation mix by county | bls.gov |
| SSA | Disability insurance recipients by county | ssa.gov |
| HUD | Fair market rents, housing vouchers by county | huduser.gov |
| DOT FHWA | Commute patterns, VMT by county | fhwa.dot.gov |

## Recommendation

**Immediate (unblocked, fetchers exist):** FEC + BEA — running now.

**Next batch (highest unique signal):**
1. USDA Atlas economic typology codes (farming/mining/government/recreation)
2. CDC drug overdose death counts
3. VA veteran density

**Later (diminishing returns):**
4. PLACES health measures (check for CHR overlap first)
5. FBI crime data
6. SNAP participation
