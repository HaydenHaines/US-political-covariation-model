# CES/CCES Data for Type Model Validation

**Researched**: S498, 2026-04-08
**Verdict**: Highly feasible at county level. County FIPS available, ~500 validated voters per type per election year.

## Data Access
- **URL**: https://dataverse.harvard.edu/dataverse/cces
- **Cumulative file**: DOI `10.7910/DVN/II2DB6` — 701,955 respondents, 2006-2024
- **Format**: .dta (Stata), .csv, .feather — free download, no registration
- **Size**: ~200MB for cumulative file

## Geographic Identifiers
- `state` / `st` — State FIPS / abbreviation
- `cd` / `dist` — Congressional district
- `zipcode` — Respondent's ZIP
- `county_fips` — 5-digit county FIPS (imputed from ZIP, ~95% accuracy)
- **No tract or block group identifier**

## Validated Vote Data
- Validated **turnout** via Catalist voter file matching (2006-2022, 2024 planned)
- `vv_turnout_gvm` — confirmed voted / no record / no voter file (~90% match rate)
- Vote **choice** is self-reported in post-election wave (among confirmed voters — gold standard)
- `voted_pres_party` — D/R/Third/Other (2008-2024)
- `voted_gov_party`, `voted_sen_party` — governor and Senate party vote

## Sample Sizes (even years, election years)
| Year | Sample |
|------|--------|
| 2008 | ~32,800 |
| 2010 | ~55,400 |
| 2012 | ~54,535 |
| 2016 | ~64,600 |
| 2018 | ~60,000 |
| 2020 | ~61,000 |
| 2022 | ~60,000 |
| 2024 | ~60,000 |

## Expected Sample per Type (J=100)
- ~600 respondents/type/year at county-matched level (average)
- Urban mega-types: 1,000+; rural niche types: ~50
- Precision: ±3-5pp on type-level D-share (sufficient for validation)

## Validation Pipeline
1. Download cumulative_2006-2024.dta from Harvard Dataverse
2. Filter: `vv_turnout_gvm == "Voted"` AND `voted_pres_party` non-missing
3. Join on `county_fips` to WV county-type assignment table
4. Aggregate by type: `cces_dem_share = mean(voted_pres_party == "Democratic")`
5. Compare against WV type-level predictions / historical type means
6. Extend to governor/Senate for behavior layer (τ/δ) validation

## Key Limitations
1. No tract identifier — county is finest reliable geography
2. Vote choice is self-reported (only turnout is validated)
3. ~20 respondents/county/year — per-county estimates have large error bars
4. YouGov opt-in panel, not probability sample (weights provided)
5. 2024 voter validation not yet released (V11 planned)

## Sources
- CCES Dataverse: https://dataverse.harvard.edu/dataverse/cces
- Cumulative Guide: berkeley.edu CES guide PDF
- HUD ZIP-to-Tract Crosswalk: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
- ccesMRPprep R package: https://github.com/kuriwaki/ccesMRPprep
