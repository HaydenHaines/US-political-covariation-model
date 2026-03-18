---
source: https://api.census.gov/data/
captured: 2026-03-18
version: ACS 5-year 2022 (covers 2018-2022); ACS 5-year 2019 (covers 2015-2019)
---

# Census ACS at Tract Level

## API Entry Point
`https://api.census.gov/data/{year}/acs/acs5`

Free. API key required (instant, free): https://api.census.gov/data/key_signup.html

## Stage 1 Query Pattern
```
GET /data/2022/acs/acs5
  ?get=B19013_001E,B19013_001M,C24010_001E,...
  &for=tract:*
  &in=state:01,12,13   ← Alabama=01, Florida=12, Georgia=13
  &key=YOUR_KEY
```

## Key Tables for This Project

| Table | Content | Notes |
|---|---|---|
| `B01001` | Sex by age | Population structure |
| `B03002` | Hispanic/Latino origin by race | Racial composition |
| `B08301` | Commute mode | Urban/rural proxy |
| `B19013` | Median household income | Economic character |
| `B23001` | Employment status by age/sex | Labor force participation |
| `B25001`/`B25003` | Housing units / tenure | Owner vs. renter |
| `C24010` | Occupation by sex | Industry/class proxy |
| `DP02` | Social characteristics | Education, household type |
| `DP03` | Economic characteristics | Income, poverty, commute |

Every estimate field (`_E` suffix) has a margin of error field (`_M` suffix). **Fetch both.**

## MOE Flagging Rule
Flag tracts where `MOE / estimate > 0.30` on any primary feature variable. Store the flag as a boolean column. Do not exclude or model the error — just surface it for post-Stage 2 review.

## Vintage Convention
- Use **2019 5-year** (2015–2019) as the training baseline
- Use **2022 5-year** (2018–2022) for current-cycle analysis
- Never mix vintages within a single model run

## Python Access
`cenpy` package wraps the API. Alternatively, use `requests` directly — the API is simple enough that direct calls are fine for batch pulls.

## Actual Tract Counts (2022 5-year ACS)
From first live pull (2026-03-18):
- Alabama: **1,437** tracts
- Florida: **5,160** tracts
- Georgia: **2,796** tracts
- **Total: 9,393 tracts** across FL+GA+AL

The "~4,200 tracts" figure cited in early planning was wrong — that's closer to the county count for the full US. The actual tract count is 9,393.

## Gotchas

**1. The "any high-MOE" flag rate is misleading at 99%.**
When you flag tracts where any variable has MOE/estimate > 30%, nearly every tract fires — because small-count variables (WFH commuters, master's degree holders, specific racial groups in homogeneous tracts) routinely have huge MOE/estimate ratios. This is expected, not alarming.

Flag rates from first pull:
- Reliable anchor variables: `pop_total` 3.2%, `housing_units` 2.4%, `median_age` 4.8%
- Small-count race variables: `pop_asian` 69.6%, `pop_black` 72.5%, `pop_hispanic` 80.6%
- Small-count education: `educ_masters` 92.3%, `educ_doctorate` 71.5%
- Commute: `commute_wfh` 93.8%, `commute_transit` 38.7%

**Implication for Stage 2**: Evaluate MOE flags per-feature, not as a tract-level "any" flag. The features most relevant to community detection (pop_total, housing_units, median_age, median_hh_income) are mostly reliable. Small-count variables at the raw count level will look unreliable but will be more stable when expressed as percentages of a large denominator.

**2. API null sentinel is -666666666.**
The Census API returns -666666666 (not NaN, not None) for suppressed or unavailable estimates. Must be replaced with NaN before any arithmetic. Handled in `cast_numeric()` in `fetch_acs.py`.
