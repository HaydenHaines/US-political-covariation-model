# CES/CCES Survey Data Research for Type Validation

**Date**: 2026-03-27
**Purpose**: Evaluate the Cooperative Election Study (CES, formerly CCES) for validating
WetherVane's KMeans electoral types against self-reported party ID, ideology, and issue positions.

---

## 1. Overview

The Cooperative Election Study (CES) is one of the largest political surveys in the US,
administered by YouGov. Half the questionnaire is "Common Content" asked of all respondents;
half is "Team Content" designed by participating research teams. Conducted annually since 2006.
Renamed from "Cooperative Congressional Election Study" (CCES) to "Cooperative Election Study"
(CES) in 2020.

- **Principal Investigators**: Stephen Ansolabehere (Harvard), Brian Schaffner (Tufts),
  Sam Luks (YouGov)
- **Home**: https://cces.gov.harvard.edu/
- **Dataverse**: https://dataverse.harvard.edu/dataverse/cces

## 2. Years Available and Sample Sizes

### Even Years (election years — larger samples, pre+post waves)

| Year | Sample Size (approx) |
|------|---------------------|
| 2006 | 36,421 |
| 2008 | 32,800 |
| 2010 | 55,400 |
| 2012 | 54,535 |
| 2014 | 56,200 |
| 2016 | 64,600 |
| 2018 | 60,000 |
| 2020 | 61,000 |
| 2022 | 60,000 |
| 2024 | 60,000 |

### Odd Years (smaller samples, no post-election wave)

| Year | Sample Size (approx) |
|------|---------------------|
| 2007 | 9,999 |
| 2009 | 13,800 |
| 2011 | 20,150 |
| 2013 | 16,400 |
| 2015 | 14,250 |
| 2017 | 18,200 |
| 2019 | 18,000 |
| 2021 | ~24,000 |
| 2023 | 24,500 |

### Cumulative Dataset

The **Cumulative CES Common Content** dataset (maintained by Shiro Kuriwaki) stacks all years
with standardized variable coding:
- **DOI**: 10.7910/DVN/II2DB6
- **n = 641,955** respondents (2006-2023 as of latest release)
- **103 variables** with harmonized coding across years
- Guide (v10): covers 2006-2024

## 3. Geographic Granularity

### Available Geographic Variables

The cumulative dataset includes:

| Variable | Description | Notes |
|----------|-------------|-------|
| `state` / `st` | State name / abbreviation | All years |
| `county_fips` | 5-digit county FIPS code | Available in cumulative file |
| `zipcode` | 5-digit ZIP code | Available in cumulative file |
| `cd` | Congressional district (at time of survey) | Available in cumulative file |
| `cd_up` | Congressional district (updated to current redistricting) | Available in cumulative file |
| `dist` / `dist_up` | District number variants | Available in cumulative file |

### Key Finding: County FIPS IS Available

**The CES cumulative dataset includes `county_fips` as a standard variable.** This is the
critical finding for our use case — we can directly join CES respondents to our county-level
electoral types without needing a crosswalk.

YouGov geocodes respondents from registration data and/or self-reported location, providing
both ZIP code and county FIPS. The geographic placement comes from YouGov's matching process
and voter file linkage.

### Caveats on County Geography

- **Coverage varies by year**: Earlier years (2006-2007) may have lower county_fips fill rates
  compared to later years. Need to check missingness after download.
- **ZIP-to-county ambiguity**: Some ZIP codes span multiple counties. YouGov resolves this
  through voter file matching, but some respondents may have imprecise county assignment.
- **Small-county sample sizes**: With ~60K respondents nationally, many of the 3,154 counties
  will have zero or very few respondents in a single year. Aggregating across years helps.
  The cumulative file's 641K respondents gives a much better county-level distribution.

### If County FIPS Were Missing (Fallback)

If county_fips has significant missingness, fallback options:
1. **ZIP-to-county crosswalk**: HUD USPS crosswalk (updated quarterly) maps ZIPs to counties
   with allocation factors. Available at https://www.huduser.gov/portal/datasets/usps_crosswalk.html
2. **Congressional district to county**: Census Bureau relationship files provide CD-to-county
   crosswalks with population weights. IPUMS NHGIS provides block-level crosswalks.
   NBER population-based crosswalks (1790-2020) are also available.

## 4. Key Variables for Type Validation

### Party Identification
- `pid3`: 3-point party ID (Democrat / Republican / Independent)
- `pid3_leaner`: 3-point with leaners folded in
- `pid7`: 7-point party ID scale (Strong D → Strong R)

### Ideology
- `ideo5`: 5-point ideology scale (Very Liberal → Very Conservative)

### Vote Choice
- `voted_pres`: Presidential vote choice (even years with presidential election)
- `voted_pres_party`: Party of presidential vote
- `voted_sen`, `voted_sen_party`: Senate vote
- `voted_gov`, `voted_gov_party`: Governor vote
- `voted_rep`, `voted_rep_party`: House vote

### Validated Vote
- YouGov matches respondents to voter files to produce **validated turnout** (whether they
  actually voted, not just self-reported). This is a major advantage over most surveys.

### Demographics
- Gender, age, race/ethnicity, Hispanic identifier
- Education level, family income, marital status
- News interest
- Employment, religion (in individual year files, not all in cumulative)

### Issue Positions (in individual year files)
The Common Content includes policy questions on:
- Abortion, gun control, immigration, healthcare/ACA
- Government spending, taxes, environment/climate
- Same-sex marriage, affirmative action
- Defense spending, trade policy

**Note**: Issue positions are NOT all standardized in the cumulative file. For issue-level
analysis, you need to download individual year files and handle coding differences manually.
The **Cumulative CES Policy Preferences** dataset (DOI: 10.7910/DVN/OSXDQO) provides some
harmonized policy variables.

## 5. Access Requirements

### Download
- **No login required** for public datasets on Harvard Dataverse
- Free to download directly from the Dataverse web interface
- API access also available without authentication for public datasets

### License
- Harvard Dataverse defaults to **CC0 Public Domain Dedication**
- Individual CES datasets should be checked for specific terms, but the data is publicly
  available and widely used without restrictive licensing

### Citation Requirements
- Standard academic citation expected. The CES website provides recommended citations.
- Cite both the specific year's dataset AND the cumulative dataset if using it.

### Redistribution of Derived Data
- **Derived/aggregated data**: No restrictions on publishing county-level aggregates or
  model outputs derived from CES data. This is standard practice in political science.
- **Raw microdata**: Redistribution of the raw respondent-level data is discouraged
  (use Dataverse links instead). Our use case (aggregating to county types) is fine.

## 6. File Formats and Sizes

### Cumulative Dataset
| Format | Extension | Notes |
|--------|-----------|-------|
| Stata | `.dta` | Readable in Python via `pandas.read_stata()` |
| R | `.Rds` | R-native format |
| Arrow/Feather | `.feather` | Fast columnar format, readable via `pyarrow` |

Estimated size: ~200-400 MB for the full cumulative file (641K rows x 103 cols).

### Individual Year Files
Each year's Common Content is also available separately on Dataverse, typically in:
- Stata `.dta` format
- SPSS `.sav` format (some years)
- Tab-delimited `.tab` format (Dataverse native)

### Online Exploration
- **Crunch** (crunch.io): Free web-based cross-tabulation interface for browsing CES data
  without downloading. Useful for quick variable exploration.

## 7. How to Download

### Option A: Direct from Dataverse (recommended for us)

**Cumulative file**:
```
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/II2DB6
```

Click "Access Dataset" → download the `.dta` or `.feather` file.

**API download** (no auth needed for public data):
```bash
# Get file list
curl "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/II2DB6"

# Download specific file by ID
curl -L -O -J "https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}"
```

### Option B: pyDataverse (Python API client)
```python
from pyDataverse.api import NativeApi, DataAccessApi

base_url = "https://dataverse.harvard.edu/"
api = NativeApi(base_url)
data_api = DataAccessApi(base_url)

DOI = "doi:10.7910/DVN/II2DB6"
dataset = api.get_dataset(DOI)
# Then iterate files and download
```

### Option C: wget (for bulk/recursive downloads)
```bash
wget -r -e robots=off -nH --cut-dirs=3 --content-disposition \
  "https://dataverse.harvard.edu/api/datasets/:persistentId/dirindex?persistentId=doi:10.7910/DVN/II2DB6"
```

### Individual Year Datasets
- 2022: DOI 10.7910/DVN/PR4L8P
- 2020: DOI 10.7910/DVN/E9N6PH
- Each year listed at https://cces.gov.harvard.edu/data

## 8. Existing Research: CES + County-Level Analysis

### Direct Precedent: Tausanovitch & Warshaw — American Ideology Project

The most directly relevant prior work. Tausanovitch & Warshaw combine ~1 million survey
respondents (including CES/CCES) with MRP to estimate ideology at the **county level**
(among other geographies).

- **Dataset**: "Subnational ideology and presidential vote estimates (v2022)"
  DOI: 10.7910/DVN/BQKU4M
- **Geographies estimated**: States, counties, cities, congressional districts, state
  legislative districts, ZIP code tabulation areas, school districts
- **Method**: Bayesian group-level IRT + MRP using Stan
- **Surveys used**: CCES, National Annenberg Election Survey (NAES), and others
- **Citation**: Tausanovitch & Warshaw (2013), "Measuring Constituent Policy Preferences
  in Congress, State Legislatures, and Cities," Journal of Politics 75(2): 330-342.
- **Website**: https://americanideologyproject.com/

**Relevance to WetherVane**: Their county-level ideology estimates could serve as an
alternative validation source for our types — or we could use their methodology as a
template. However, our approach is simpler: we just need to aggregate raw CES responses
within each type cluster, not build a full MRP model.

### Kuriwaki MRP Toolkit

Shiro Kuriwaki (Harvard) maintains an R ecosystem for CES+MRP:
- `ccesMRPprep`: Data cleaning and preparation for MRP with CES data
- `ccesMRPrun`: Model fitting companion package
- NSF-funded (Grant 1926424)
- Includes county FIPS in processing pipeline

### CDC PLACES

The CDC's PLACES project uses MRP on BRFSS data to produce county-level health estimates,
demonstrating the broader viability of survey-to-county estimation at scale.

## 9. Feasibility Assessment for WetherVane Type Validation

### Approach: Direct Aggregation (no MRP needed)

Because CES includes `county_fips`, we can:

1. Download the cumulative file (641K respondents, 2006-2023)
2. Join each respondent to their county's electoral type assignment
3. Aggregate `pid3`/`pid7`, `ideo5`, and vote choice within each of the 100 types
4. Compare: Do respondents in "blue-shift" types self-identify as more Democratic/liberal?
   Do "red-shift" types report more Republican/conservative identity?

### Expected Sample Sizes per Type

- 641K respondents / 100 types = ~6,400 per type on average
- Distribution will be uneven (urban types will have more respondents)
- Even sparse types should have 100+ respondents across all years combined
- This is more than sufficient for meaningful aggregates

### Specific Validation Questions We Can Answer

1. **Party ID gradient**: Do types ordered by Democratic vote share show a monotonic
   gradient in self-reported party ID? (pid3, pid7)
2. **Ideology alignment**: Does ideo5 track with the partisan lean of each type?
3. **Issue position clustering**: Do types with similar electoral behavior also share
   similar policy preferences? (requires individual year files)
4. **Validated vote vs. type prediction**: Compare validated turnout patterns across types
5. **Demographic profiles**: Characterize each type's demographic composition from CES
   and compare with ACS-based features we already use
6. **Super-type validation**: Do the 8 tract super-types / 5 county super-types show
   distinct survey profiles?

### Implementation Complexity: Low-Medium

- Download: 1 file (~300 MB)
- Join: Simple county_fips merge
- Analysis: Groupby aggregations on pid3/pid7/ideo5 by type
- Visualization: Heatmaps, box plots by type
- Timeline: 1-2 sessions to implement

### Limitations

- **Ecological inference**: Aggregating individual survey responses to type-level means
  is valid for description but doesn't prove individual-level behavior
- **Temporal mismatch**: Our types are defined on 2008-2024 elections; CES respondents
  from 2006-2007 predate some of our electoral data
- **YouGov sample**: Online panel, not probability sample. Weighted to be representative
  but may have biases in hard-to-reach populations
- **County assignment noise**: Some respondents may be assigned to wrong county via
  ZIP-to-county imputation

## 10. Recommended Next Steps

1. **Download the cumulative file** (.dta or .feather format) from Dataverse
2. **Check county_fips completeness** — what % of rows have valid county FIPS?
3. **Join to type assignments** — merge on county_fips to our J=100 type labels
4. **Produce type profiles** — aggregate pid3, pid7, ideo5 by type
5. **Visualize** — types ordered by Dem vote share vs. mean ideology/party ID
6. **Optional**: Download 2020 and 2024 individual files for issue position analysis
7. **Optional**: Compare against Tausanovitch-Warshaw county ideology estimates as
   a second validation source (DOI: 10.7910/DVN/BQKU4M)

---

## Sources

- [CES Home](https://cces.gov.harvard.edu/)
- [CES Dataverse](https://dataverse.harvard.edu/dataverse/cces)
- [Cumulative CES Common Content](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/II2DB6)
- [Cumulative CES Policy Preferences](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OSXDQO)
- [CES 2022](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PR4L8P)
- [CES 2020](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/E9N6PH)
- [kuriwaki/cces_cumulative (GitHub)](https://github.com/kuriwaki/cces_cumulative)
- [ccesMRPprep (R package)](https://github.com/kuriwaki/ccesMRPprep)
- [Cumulative Guide v10 (PDF)](https://csmweb-prod-02.ist.berkeley.edu/sdaweb/docs/ces-cumulative-2024-v10/DOC/guide_cumulative_2006-2024.pdf)
- [American Ideology Project](https://americanideologyproject.com/)
- [Tausanovitch-Warshaw Subnational Estimates](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BQKU4M)
- [NHGIS Geographic Crosswalks](https://www.nhgis.org/geographic-crosswalks)
- [NBER County-CD Crosswalks (1790-2020)](https://www.nber.org/papers/w32206)
- [Census Relationship Files](https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.2020.html)
- [HUD USPS ZIP-County Crosswalk](https://www.huduser.gov/portal/datasets/usps_crosswalk.html)
- [pyDataverse](https://github.com/gdcc/pyDataverse)
- [Dataverse Data Access API](https://guides.dataverse.org/en/5.6/api/dataaccess.html)
- [Tufts CES Data Downloads](https://tischcollege.tufts.edu/research-faculty/research-centers/cooperative-election-study/data-downloads)
