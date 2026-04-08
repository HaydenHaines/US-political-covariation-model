# CES δ Backtest — 2022 Governor Actuals (S501)

**Date:** 2026-04-08
**Branch:** feat/ces-delta-integration
**Purpose:** Test whether CES-derived empirical δ improves governor predictions
compared to model-computed δ and no-δ baselines.

## Background

S499-S500 established that the type model correlates r=0.894 with CES
external validation (248K validated voters), but the model-computed δ has
r=-0.008 with CES empirical δ (no signal). This prompted replacing model δ
with CES δ.

## Results (30 states, ≥80% county coverage)

| Baseline | r | RMSE | Bias | Dir |
|---|---|---|---|---|
| No δ (current production) | **0.839** | **7.8pp** | +1.3pp | **25/30** |
| + CES governor δ (248K voters) | 0.777 | 9.4pp | +1.4pp | 25/30 |
| + Model δ (tract-computed) | 0.793 | 8.6pp | +0.6pp | 23/30 |
| 2024 Presidential baseline | 0.824 | 8.1pp | +0.8pp | 24/30 |

**CES δ hurts.** Adding CES δ increases RMSE by 1.6pp and drops r by 0.062.
The model δ also hurts but less (r drops 0.046). No-δ with blended governor
priors and incumbency heuristic remains the best approach.

## Analysis

The CES type-level validation (r=0.894) is strong — the types genuinely capture
how people vote. But the *differential* (δ = governor - presidential) introduces
variance that hurts state-level predictions when applied through county type scores.

**Why CES δ fails despite good type validation:**

1. **Double-counting.** The blended governor Ridge priors already incorporate
   governor-specific patterns (70% governor data). Adding δ on top double-counts
   the governor signal.

2. **Soft assignment amplification.** δ is per-type (J=100), but counties have
   soft membership across many types. The weighted sum of δ values amplifies noise
   from poorly-estimated types, especially small ones (21 types have no CES data).

3. **Cross-cycle averaging.** CES δ pools 2006-2024 cycles, mixing different
   political environments. The average δ may not reflect any single cycle well.

4. **The backtest is indirect.** Comparing 2026 model predictions to 2022 actuals
   conflates model structure with temporal environment. δ may help in a true
   holdout but not in this indirect comparison.

## Conclusion

The behavior layer δ (whether model-computed or CES-derived) does not improve
governor predictions with the current architecture. The blended governor priors
+ incumbency heuristic capture most of the governor-specific signal without
introducing the variance that δ adds.

**The race-specific δ infrastructure is retained** (governor and senate get
different CES δ values) for future use when:
- More sophisticated application methods exist (e.g., only types with N > 500 CES respondents)
- A true holdout experiment (train ≤2020, predict 2022) validates δ
- Hierarchical partial pooling regularizes small-type δ estimates
