# YouGov/Economist Crosstab Investigation

**Date:** 2026-04-06 (S346)
**Status:** VIABLE — high-value for generic ballot crosstabs

## Summary

The Economist/YouGov weekly tracker publishes full crosstab PDFs with demographic breakdowns. These are **national** polls (~1,600 adults), not state-level, but contain generic congressional ballot (question #41) with rich demographics.

## PDF Source

**CDN URL pattern:** `https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_<hash>.pdf`

Note: The old CDN (`docs.cdn.yougov.com`) redirects. New URLs use CloudFront.

### Known 2026 PDFs

| Date Range | N | URL |
|------------|---|-----|
| Jan 2-5 | 1,551 | `econTabReport_aY8mpiN.pdf` |
| Jan 9-12 | 1,602 | `econTabReport_qsNv5iE.pdf` |
| Jan 16-19 | 1,722 | `econTabReport_z9wtNZI.pdf` |
| Jan 23-26 | 1,684 | `econTabReport_8FWGyNz.pdf` |
| Feb 6-9 | 1,730 | `econTabReport_vNnwPx2.pdf` |
| Mar 6-9 | 1,563 | `econTabReport_EcCnfRV.pdf` |
| Mar 13-16 | 1,595 | `econTabReport_CwWXhS2.pdf` |
| Mar 20-23 | 1,665 | `econTabReport_o84FoNw.pdf` |

Published weekly. Hash appears random (not date-derived).

## PDF Format (examined Mar 20-23 issue)

64 pages. Table of contents on pages 1-2. Each question gets 1-2 pages.

**Generic Congressional Vote (question 41, page 59):**

Format: demographics as COLUMN headers, response options as ROWS.

```
41. GenericCongressionalVote
IftheelectionsforU.S.Congresswerebeingheldtoday,whowouldyouvotefor...

                          Sex         Race              Age              Education
              Total Male Female White Black Hispanic 18-29 30-44 45-64 65+ Nodegree Collegegrad
Dem           39%   33%  44%   34%   65%   41%      39%   41%   33%  46%  34%      48%
Rep           36%   43%  29%   43%    6%   34%      29%   31%   40%  40%  38%      32%
Other          2%    2%   1%    2%    1%    2%       0%    1%    4%   1%   1%       2%
Not sure      11%   10%  12%    9%   14%    9%      13%   12%   12%   6%  11%      10%
Would not     13%   12%  13%   12%   15%   14%      19%   15%   11%   7%  16%       7%
UnweightedN (1664) (784)(880) (1096)(208) (252)    (352) (437) (509)(366)(1064)   (600)
```

Second half of table:
```
              2024Vote     Reg    Ideology         MAGA      PartyID
              Harris Trump Voters Lib  Mod  Con  Supporter  Dem  Ind  Rep
Dem           89%    5%   45%    84%  41%   5%    2%       91%  28%   2%
Rep            2%   84%   42%     4%  21%  82%   85%        1%  21%  87%
```

**Key differences from Marist:**
- Demographics are columns, not rows
- Multiple demographic dimensions on same row (Sex + Race + Age + Education in one table)
- Second table below with different demographic columns (Vote + Ideology + Party)
- UnweightedN row provides sample sizes per group
- Text spacing is inconsistent (words run together in headers)

## Value for WetherVane

**Generic ballot crosstabs** — per-group vote shares for:
- Race: White, Black, Hispanic (3 groups)
- Age: 18-29, 30-44, 45-64, 65+ (4 groups)
- Education: No degree, College grad (2 groups)
- Gender: Male, Female
- 2024 Vote: Harris, Trump
- Party: Dem, Ind, Rep
- Ideology: Liberal, Moderate, Conservative

These map directly to our xt_vote_* columns and would significantly enrich the generic ballot adjustment calculation.

**Frequency:** Weekly data = time series of per-group vote preferences. Can track how each demographic group shifts over the cycle.

## Next Steps

1. Build YouGov PDF parser (different format from Marist — column-oriented tables)
2. Download all 2026 issues
3. Extract generic ballot crosstabs → time series
4. Feed into generic ballot adjustment (currently uses topline only)
5. Potentially use for national-level W vector calibration

## Priority

Medium-high. The weekly frequency makes this the richest crosstab source by volume. But it's national-only (no state-level), so it supplements rather than replaces Marist/Emerson state polls.
