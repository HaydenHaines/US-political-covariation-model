# ADR-004: Census Tract as Primary Unit of Analysis

## Status
Accepted

## Context
The model's foundational structural decision is the geographic unit at which community types are detected and political behavior is measured. Three candidates were evaluated.

**Counties** (226 in FL+GA+AL — original working assumption)
The natural first choice. Election data is readily available at county level, ACS coverage is good, and aggregation is simple. But counties are administrative units, not social units. A county containing a city or college town will average that community's signal into the surrounding rural area. The result reflects no real community — it is a population-weighted average of several distinct ones. This is the **Modifiable Areal Unit Problem (MAUP)**: statistical results change depending on which arbitrary boundaries are used. For a model explicitly trying to discover communities that administrative boundaries miss, using county boundaries as the unit of analysis is incoherent with the stated goal.

**Census Blocks** (~1M in FL+GA+AL)
The finest grain the Census reports. VEST election data is available at block level via precinct-to-block spatial allocation. But the **American Community Survey (ACS)** — which provides the rich non-political features this model needs (occupation, education, housing, income, commute patterns, household composition) — is **not available at block level in any usable form**. The ACS is a probability sample of ~3.5M households nationally. Most census blocks have zero or one ACS respondents, making estimates meaningless. Only the decennial Census reports at block level, and it captures only basic counts: population, race, housing units, Hispanic origin. That is insufficient for community detection. Blocks are the right unit for spatial joins; they cannot be the analytical unit.

**Census Tracts** (~9,400 in FL+GA+AL)
Deliberately designed by the Census Bureau to approximate socially homogeneous neighborhoods — typically 1,200–8,000 people, drawn to capture coherent residential areas. ACS 5-year estimates are available at tract level with manageable margins of error for most variables. VEST election data, reported at block level, can be aggregated to tracts via FIPS joins. Religious congregation data (ARDA) is available at congregation level with coordinates, enabling spatial joins to tracts. Tract granularity is sufficient to distinguish Black Belt rural communities from college-town communities within the same county — distinctions that county-level analysis would average away.

One additional property of census tracts is directly valuable as a validation mechanism: a geographically "transitional" tract — one that sits physically between two distinct community areas — should naturally receive high mixed-membership across both neighboring community types under soft assignment. If statistical intermediacy correlates with geographic intermediacy, that is evidence the detected communities reflect real social structure rather than model artifacts.

## Decision
**Census tract is the primary unit of analysis.**

- **9,393 tracts** in FL+GA+AL per 2022 ACS 5-year (FL: 5,160 · GA: 2,796 · AL: 1,437)
- **Features**: ACS 5-year estimates at tract level
- **Election data**: VEST precinct-to-block joins, aggregated to tract via census block FIPS codes
- **Congregation data**: ARDA coordinates spatially joined to tracts via geopandas
- **MOE handling**: High-MOE tracts are flagged at ingestion. They are not excluded and their error distributions are not incorporated into the model. Post-Stage 2 review will assess whether high-MOE tracts cluster into spurious community assignments. If they do, exclusion thresholds will be set at that point.
- **Pipeline parameterization**: Geographic unit is a parameter of Stage 1, not a hard-coded architectural assumption. Moving to block groups (finer) or counties (coarser) in a future iteration requires Stage 1 data changes only.

---

### Primary Data Sources

**ACS 5-Year Estimates (U.S. Census Bureau)**
- Base URL: `https://api.census.gov/data/`
- Example tract-level call: `https://api.census.gov/data/2022/acs/acs5?get=B19013_001E&for=tract:*&in=state:01,12,13&key=YOUR_KEY`
- State FIPS: Alabama = 01, Florida = 12, Georgia = 13
- Access: Free public API. API key registration at https://api.census.gov/data/key_signup.html (instant, free)
- Python: `cenpy` package or direct `requests` calls
- Key tables for Stage 1:
  - `B01001` — Sex by age
  - `B03002` — Hispanic or Latino origin by race
  - `B08301` — Means of transportation to work (commute mode as proxy for urban character)
  - `B19013` — Median household income
  - `B23001` — Sex by age by employment status
  - `B25001` / `B25003` — Housing units / tenure (owner vs. renter)
  - `C24010` — Sex by occupation (detailed occupation categories)
  - `DP02` — Social characteristics (education, marital status, household type)
  - `DP03` — Economic characteristics (income, poverty, commute)
- Vintage: Use **2019 ACS 5-year** (2015–2019) as the baseline feature set for training. Use **2022 ACS 5-year** (2018–2022) for current-cycle analysis. Do not mix vintages within a model run.
- MOE fields: Every estimate `*_E` has a corresponding margin of error `*_M`. Fetch both.
- Documentation: https://www.census.gov/programs-surveys/acs/technical-documentation/table-shells.html

**VEST Election Returns (Voting and Election Science Team, University of Florida)**
- Dataverse URL: https://dataverse.harvard.edu/dataverse/vest
- Coverage: 2016, 2018, 2020, 2022 general elections, all 50 states
- Format: Shapefile or GeoPackage. Contains precinct boundary polygons + vote totals by candidate and party.
- Access: Free download via Harvard Dataverse. No account required for download.
- Citation format: `Voting and Election Science Team, [YEAR], "[State] [Year] Precinct-Level Election Results", Harvard Dataverse, V[version]`
- States needed: Florida (VEST ID: florida_[year]_vest), Georgia (georgia_[year]_vest), Alabama (alabama_[year]_vest)
- Aggregation strategy: VEST provides a `GEOID20` (2020 census block FIPS) or similar block-level join field. Aggregate to tract by summing vote totals within the 11-digit tract FIPS prefix of each block's GEOID.
- Note: Some VEST releases use precinct-level geometries with block apportionment notes rather than block-level records. Review the data dictionary for each state/year before assuming aggregation strategy.

**ARDA Religious Congregation Data (Association of Religion Data Archives)**
- URL: https://www.thearda.com/
- Key dataset: **Religious Congregations and Membership Study (RCMS)** — decennial, most recent: 2020
- County-level aggregate counts by denomination are the standard release. Congregation-level coordinate data availability varies by year and release — review the 2020 RCMS documentation before assuming coordinates are available.
- Access: Free download, account registration required (free)
- Fallback: If congregation-level coordinates are unavailable, use county-level congregation counts as features (less granular but still meaningful signal).
- Secondary source: **ARDA U.S. Congregational Life Survey** and **National Congregations Study** for congregation-level characteristics if spatial data is needed.

---

## Consequences

**What becomes easier:**
- Sub-county community variation is visible. A college town inside a rural county is no longer averaged into surrounding communities.
- Transitional tracts provide a built-in validation mechanism for soft assignment: geographic and statistical intermediacy should correlate if communities are real.
- VEST block-level spatial precision is fully utilized rather than discarded in county aggregation.
- Congregation data can be spatially joined at neighborhood rather than county level.

**What becomes more difficult:**
- Stage 1 data engineering is more complex. ACS tract-level calls require pagination. MOE management requires fetching and storing parallel MOE fields for every estimate. VEST aggregation to tract requires a spatial join or FIPS string manipulation step.
- ACS margins of error are higher at tract level than county level, especially for low-population rural tracts. Flagging logic must be built into Stage 1 ingestion.
- The model has ~42x more units (9,393 vs. 226). Stage 2 and Stage 3 compute increases accordingly, though it remains feasible on a single machine.
- Existing political science literature benchmarks are typically at county or state level. Validating against those benchmarks requires aggregating tract-level outputs upward, adding a post-processing step.
- ADR-003 references "226 counties" as the proof-of-concept scope. That language refers to the geographic footprint (FL+GA+AL), not the unit of analysis. ADR-003 remains valid; this ADR supersedes the implied county-level unit.
