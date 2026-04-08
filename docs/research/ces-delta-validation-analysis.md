# CES Empirical δ vs Model δ — Validation Analysis

**Date**: 2026-04-08 (S500)
**Status**: Complete — findings inform behavior layer design

## Summary

The CES/CCES survey data provides the first independent measurement of per-type
behavioral shifts (δ) between presidential and off-cycle races. Comparing CES
empirical δ against the model's computed δ reveals that **the current behavior
layer δ is uncorrelated with actual voter behavior**.

## Key Findings

### 1. Model δ does not predict real behavior

| Metric | CES Governor δ | Model δ | Correlation |
|--------|---------------|---------|-------------|
| Mean | +1.06pp | -1.01pp | r = -0.008 |
| Std | 5.96pp | 5.17pp | (no signal) |
| Mean abs | 4.45pp | 2.76pp | |

**The correlation between model δ and CES empirical δ is essentially zero
(r = -0.008 for governor, r = +0.030 for Senate).** The model's δ has no
predictive relationship with how types actually shift between race types.

### 2. CES δ is substantial and type-specific

- Governor δ ranges from -15.9pp (Type 76) to +18.5pp (Type 40)
- Senate δ ranges from -14.8pp (Type 88) to +13.5pp (Type 59)
- These are not noise — they reflect genuine cross-over voting patterns

### 3. Governor δ ≠ Senate δ

Governor and Senate δ values per type are only moderately correlated (r = 0.39).
Types do not shift uniformly across all off-cycle races. This means a single δ
per type is insufficient — the behavior layer needs race-type-specific parameters.

### 4. CES presidential alignment is strong

- Pres-Gov correlation: r = 0.87
- Pres-Sen correlation: r = 0.94

Senate is more nationalized (higher pres correlation, lower δ variance). Governor
races have more ticket-splitting (lower pres correlation, higher δ variance).

## Implications for Behavior Layer

### What's wrong with current δ

The model computes δ from county-level election returns: `δ = off-cycle D-share -
expected D-share from turnout reweighting`. This confounds:

1. **Turnout composition** — which voters show up (τ handles this partially)
2. **Candidate quality** — strong/weak candidates shift results
3. **Genuine preference shifts** — some types actually vote differently in off-cycle

County-level aggregation can't distinguish these. CES has individual-level validated
voter data, which directly measures #3.

### What would work better

1. **Use CES δ directly as the behavioral prior** — 79 types with empirical δ from
   248K presidential + 118K governor voters. This replaces the model's computed δ
   entirely.

2. **Separate governor δ and Senate δ** — they're only r=0.39 correlated. A single
   off-cycle δ is a poor approximation.

3. **Consider hierarchical pooling** — 20 types have <100 governor respondents.
   Shrink low-N δ estimates toward the global mean to reduce noise.

4. **Temporal stability check needed** — CES data spans 2006-2024. Per-year δ
   stability analysis would reveal if δ is durable or cycle-specific.

## Data Location

- `data/validation/ces_governor_delta.csv` — per-type governor δ
- `data/validation/ces_senate_delta.csv` — per-type Senate δ
- `data/validation/ces_governor_delta_summary.json` — summary stats
- `data/validation/ces_senate_delta_summary.json` — summary stats
