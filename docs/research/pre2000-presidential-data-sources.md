# Pre-2000 County-Level Presidential Election Data Sources

Research date: 2026-03-27
Context: WetherVane currently has MEDSL county presidential data from 2000-2024. Goal is to extend back to at least 1960, ideally 1948, for parity with the Economist model.

---

## 1. ICPSR (Inter-university Consortium for Political and Social Research)

ICPSR is the gold standard for historical US election data. Three datasets are relevant:

### ICPSR 0001: United States Historical Election Returns, 1824-1968
- **Years**: 1824-1968
- **Granularity**: County-level
- **Offices**: President, Governor, US Senator, US Representative
- **Coverage**: >90% of all elections for those offices
- **Format**: Fixed-width ASCII with SAS/SPSS/Stata setup files. County names standardized, county ID numbers added. State-level files (some states split across multiple files due to volume).
- **Third-party candidates**: Yes — all parties and candidates included
- **License**: Free to users at ICPSR member institutions. Non-members pay ~$825/dataset admin fee. Researcher Passport account required.
- **URL**: https://www.icpsr.umich.edu/web/ICPSR/studies/1
- **Quality notes**: Professional curation, standardized county names (including historical name changes). This is the most widely cited source for pre-1970 county election data in political science literature.
- **FIPS warning**: Uses ICPSR county codes, NOT FIPS codes. ICPSR codes are 4-digit (3-digit base + 1 digit for historical changes). Crosswalk needed — see https://github.com/kjhealy/icpsr_fips

### ICPSR 0013: General Election Data for the United States, 1950-1990
- **Years**: 1950-1990
- **Granularity**: County-level
- **Offices**: President, Governor, US Senator, US Representative, plus one additional statewide office per state
- **Format**: TSV, R data file, Stata, SAS, SPSS, plus PDF codebook
- **Third-party candidates**: Yes — all parties and candidates
- **License**: Same as ICPSR 0001 (free at member institutions)
- **URL**: https://www.icpsr.umich.edu/web/ICPSR/studies/13
- **Quality notes**: National files (not state-by-state like 0001). Overlaps with 0001 for 1950-1968 and provides continuity through 1990. TSV format is much easier to work with than 0001's fixed-width.

### ICPSR 8611: Electoral Data for Counties, Presidential and Congressional Races, 1840-1972
- **Years**: 1840-1972
- **Granularity**: County-level
- **Offices**: President and US House
- **Format**: Multiple formats (TSV, Stata, SAS, R, SPSS) with codebook
- **Third-party candidates**: Major and "significant" minor party candidates, with residual collapsed into "other" category. Also includes total vote counts and turnout estimates.
- **License**: Same as above (free at member institutions)
- **URL**: https://www.icpsr.umich.edu/web/ICPSR/studies/8611
- **Quality notes**: Created by Clubb, Flanigan, and Zingale. More analytical dataset — includes vote percentages and turnout. The "significant minor party" threshold means very small parties may be in "other."
- **DOI**: https://doi.org/10.3886/ICPSR08611.v1

### ICPSR Access Strategy
- If Hayden has any university affiliation (alumni library access, adjunct, etc.), ICPSR data is free.
- Without affiliation: ~$825/dataset or contact ICPSR-help@umich.edu for individual researcher pricing.
- Some ICPSR data has been deposited on openICPSR (free) — worth checking if any of these studies have open deposits.

---

## 2. Dave Leip's Atlas of US Presidential Elections

- **Years**: County-level data from **1892** to present (2024)
- **Granularity**: County-level (and state-level, which is free)
- **Format**: Excel (.xlsx) and CSV files
- **Third-party candidates**: Yes — all candidates and parties
- **License**: Commercial. County-level data requires purchase.
  - Individual license: single-person use
  - Site license (30-seat): ~$435/year
  - Redistribution requires separate contract
  - Per-dataset pricing not publicly listed — must visit store page
- **Store URL**: https://uselectionatlas.org/BOTTOM/store_data.php
- **Quality notes**: Considered the most comprehensive single source for county-level presidential data. Compiled from official sources. Widely used by journalists and researchers. Clean, well-structured data.
- **Institutional access**: Many universities (Harvard, Cornell, UVA, Penn, Oregon, Michigan State) have site licenses. If Hayden has library access at any of these, the data may be downloadable.
- **Harvard Dataverse mirror**: Some Leip data has been deposited at Harvard Dataverse by licensed institutions (DOI: 10.7910/DVN/SUCQ52), but access is restricted to those institutions.

### Assessment
Best single commercial source. Covers 1892-2024 at county level with all candidates. Cost is the main barrier. Worth it if ICPSR proves too difficult to access or clean.

---

## 3. OpenElections Project (openelections.net)

- **Years**: Primarily 2000+, with some states going further back
- **Granularity**: Precinct-level (primary focus), county-level (secondary)
- **Format**: CSV files in GitHub repos (one repo per state)
- **Third-party candidates**: Yes — from official certified results
- **License**: Free / open source
- **GitHub**: https://github.com/openelections
- **Nationwide repo**: https://github.com/openelections/openelections-data-us
- **Quality notes**:
  - Volunteer-driven project, coverage is uneven across states
  - Pre-2000 data is sparse — most states only have data from 2000+
  - Data comes from official certified results (high quality where available)
  - New England states report at town level, not county
  - Alaska has no county equivalents
  - Candidate/party names not fully standardized across states

### Assessment
Not useful for pre-2000 presidential data at scale. Too incomplete for our needs. Useful as a cross-validation source for 2000+ data we already have.

---

## 4. MIT Election Data + Science Lab (MEDSL)

- **County presidential data**: **2000-2024 only** (this is what we already use)
- **State-level presidential**: 1976-2024
- **Granularity**: County for 2000+, state/constituency for pre-2000
- **Format**: Tab-delimited (TSV)
- **Third-party candidates**: Yes
- **License**: Free, CC-BY
- **Harvard Dataverse**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ
- **GitHub**: https://github.com/MEDSL

### Assessment
MEDSL does NOT have county-level presidential data before 2000. Their pre-2000 data is state-level only. We already have their complete county presidential offering.

---

## 5. Wikipedia / Other Free Sources

### Wikipedia
- Wikipedia has state-level presidential results for every election, but county-level data is not systematically available in structured format.
- Some individual state Wikipedia articles have county result tables, but scraping these would be labor-intensive and error-prone.
- **Not a viable source** for systematic county-level data.

### USGS National Atlas / helloworlddata
- **Years**: 2004, 2008, 2012 only
- **URL**: https://github.com/helloworlddata/us-presidential-election-county-results
- **Format**: CSV
- **License**: Public domain (USGS)
- **Assessment**: Too recent, doesn't help with pre-2000 goal.

### tonmcg GitHub
- **Years**: 2008-2024
- **URL**: https://github.com/tonmcg/US_County_Level_Election_Results_08-24
- **Format**: CSV with FIPS codes
- **License**: Unclear
- **Assessment**: Too recent for our needs.

### john-guerra GitHub
- **URL**: https://github.com/john-guerra/US_Elections_Results
- County-level presidential results, but coverage/years unclear from search results.
- Worth checking but unlikely to have deep historical data.

---

## 6. Harvard Dataverse

### MEDSL County Presidential Returns 2000-2020
- DOI: 10.7910/DVN/VOQCHQ
- Already using this data — no pre-2000 county data here.

### Harvard Election Data Archive (HEDA)
- **URL**: https://projects.iq.harvard.edu/eda/home
- Contains state, county, and district-level election returns for recent elections.
- Focus is on recent data, not deep historical.
- **Assessment**: Not a source for pre-2000 county presidential data.

### Dave Leip Data on Harvard Dataverse
- DOI: 10.7910/DVN/SUCQ52
- Licensed data — restricted to Harvard affiliates.

### Economist Model Data
- The Economist's election model (cloned to `research/economist-model/`) constructs covariance matrices from demographics, not raw election returns. Their county-level election data sources are worth examining in their code/data to see what they used for pre-2000.

---

## Recommendation: Acquisition Strategy

### Priority 1: ICPSR 0013 (1950-1990, TSV format)
- **Why first**: TSV format is easiest to ingest. Covers 1952-1988 presidential elections (6 cycles we're missing). National files, not state-by-state. Overlaps with our 2000+ data for validation.
- **Gap**: Leaves 1992 and 1996 uncovered (between ICPSR 0013's 1990 endpoint and our 2000 start).

### Priority 2: ICPSR 8611 (1840-1972)
- **Why second**: Extends back to 1948 (and beyond). Overlaps with 0013 for cross-validation. Has turnout data. Multiple formats available.
- **Caveat**: "Significant" minor party threshold — verify what this means for years like 1948 (Thurmond/Dixiecrats), 1968 (Wallace), 1992 (Perot).

### Priority 3: Dave Leip (1892-2024, if ICPSR access fails)
- **Why fallback**: Single consistent source covering entire range. Clean data. But costs money and has redistribution restrictions.

### Filling the 1992-1996 Gap
- ICPSR 0013 ends at 1990, our MEDSL data starts at 2000.
- Options: (a) ICPSR 0001 won't help (ends 1968). (b) Dave Leip covers it. (c) OpenElections may have some states. (d) Check if ICPSR has additional datasets covering the 1990s.
- **Best bet**: Dave Leip for 1992+1996 if ICPSR doesn't cover it, or search ICPSR catalog for a 1990s update to study 0013.

### Data Engineering Considerations
- **FIPS crosswalk**: ICPSR uses its own 4-digit county codes. Need https://github.com/kjhealy/icpsr_fips or https://github.com/vbehnam/ICPSR_FIPS to map to modern FIPS.
- **County changes over time**: Counties have been created, merged, split, and renamed. Pre-1960 data will have counties that no longer exist. Need a temporal FIPS crosswalk.
- **Alaska**: No counties pre-statehood (1959). Borough system adopted later. Handle carefully.
- **Virginia independent cities**: Historically reported separately from counties. Our current data likely handles this but verify consistency.
- **Format normalization**: ICPSR data will need transformation to match our existing MEDSL schema (year, state, county_fips, candidate, party, candidatevotes, totalvotes).

### Target Timeline After Acquisition
With ICPSR 0013 + 8611 + existing MEDSL:
- **1948-1968**: ICPSR 8611 and/or 0001
- **1952-1988**: ICPSR 0013 (presidential years: 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988)
- **1992-1996**: Gap — needs Dave Leip or alternative source
- **2000-2024**: MEDSL (current)

This gives us 1948-2024 coverage (19 presidential cycles) with a potential 1992-1996 gap to fill.
