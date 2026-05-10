# Quinnipiac pct_of_sample NULL Fix

**Date:** 2026-05-09  
**Implemented in:** commit `6e87d3b` (2026-05-06)  
**Affected code:** `src/prediction/forecast_engine.py:_extract_crosstabs_from_xt()`

## Problem

Quinnipiac University press-release PDFs publish per-group vote shares (e.g., 52% of White voters choose Dem) but do NOT publish per-group sample sizes. This means Quinnipiac rows in `data/polls/polls_2026.csv` have:

- **`xt_vote_<group>_<value>` columns present** (e.g., `xt_vote_race_white=0.52`)
- **`xt_<group>_<value>` columns absent** (sample composition unknown — no `pct_of_sample`)

Before the fix, `_extract_crosstabs_from_xt()` only ran a **first pass** that looked for `xt_<group>_<value>` (sample composition) columns. Quinnipiac had none of these, so the function returned an empty list. All 5 Quinnipiac rows contributed zero Tier 2 crosstab signal despite having meaningful per-group vote shares.

The guard `if pct <= 0: continue` (line 295) was not the direct culprit — Quinnipiac entries never even entered the first pass, since they had no `xt_*` composition columns.

## Root Cause

`_extract_crosstabs_from_xt()` required sample composition data (`xt_<group>_<value>`) to generate any crosstab observations. Pollsters that publish only per-group vote shares (no sample size breakdown) were silently excluded from Tier 2.

## Fix

A **second pass** was added after the first pass (commit `6e87d3b`). It collects any `xt_vote_<group>_<value>` entries not already captured by the first pass, setting `pct_of_sample=None` to signal that sample composition is unknown:

```python
seen = {(c["demographic_group"], c["group_value"]) for c in crosstabs}
for key, value in poll.items():
    if not key.startswith("xt_vote_"):
        continue
    # ... parse demographic_group, group_value from key
    crosstabs.append({
        "demographic_group": parts[0],
        "group_value": parts[1],
        "pct_of_sample": None,   # composition unknown
        "dem_share": parsed,
    })
```

`build_W_from_crosstabs()` in `poll_enrichment.py` already handled `pct_of_sample=None` by falling back to `sub_n = max(n_sample, 1)` — using the full poll sample as the denominator. This gives the largest sigma (most uncertainty) per observation, which is the appropriate conservative choice when sub-group sample sizes are unavailable.

## Before/After Delta (compare_xt_impact_v2.py, 2026-05-09)

**Quinnipiac-affected race:** `2026 PA Governor` (4 Quinnipiac polls out of 8 total)

| Run | PA Governor pred | vs stripped |
|-----|-----------------|-------------|
| Stripped (no xt_) | 0.5567 | — |
| **Live enriched (with fix)** | **0.5953** | **+3.869pp** |
| Tier 2 bypass | 0.5862 | +2.954pp |

The fix moves PA Governor's forecast 3.87pp toward Dems, reflecting Quinnipiac's strong per-group signal (race/education vote shares). The live path shows a slightly larger effect than the bypass because `prepare_polls` applies additional quality weighting.

**System-wide summary (19 races with xt_ data):**

| Metric | Mean |Δ| | Max |Δ| | Median |Δ| |
|--------|---------|---------|----------|
| Live enriched vs stripped | 1.758pp | 4.779pp | 1.473pp |
| Tier 2 bypass vs stripped | 1.457pp | 3.192pp | 1.386pp |

θ_national max absolute diff (live vs stripped): **0.7965** across J=100 types.

## Test Coverage

All 4196 tests pass. Relevant tests:

- `tests/test_ingest_quinnipiac_crosstabs.py` — 409 tests for the ingestion pipeline
- `tests/test_forecast_engine.py` — covers `_extract_crosstabs_from_xt()` paths

## Follow-Up Items

- **Uniform pct fallback**: The task originally suggested using `1.0/num_groups` as a uniform estimate when `pct_of_sample` is None. The full-`n_sample` fallback chosen instead is more conservative (larger sigma) but doesn't penalize groups that happen to be fewer in the crosstab table. The tradeoff is acceptable for now.
- **Other pollsters**: The second pass generalizes to any pollster that publishes `xt_vote_*` without `xt_*` composition. TIPP Insights and Trafalgar also benefit (commit `6e87d3b` added xt data for multiple pollsters simultaneously).
