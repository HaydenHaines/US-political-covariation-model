# Model Improvement Experiments — S196 (2026-03-25)

## Context
Hayden requested "top priority is further improving the model." Session S196 ran 5 experiments to identify improvement avenues for the county model (holdout r=0.698, covariance val r=0.556).

## Key Discovery: Metric Inflation
The standard holdout_accuracy_county_prior metric inflates by ~0.22 due to type self-prediction. Counties in small types (10 singletons, 22 types with <=3 members at J=100) contribute disproportionately because their type mean ≈ their actual value.

| Metric | Standard | LOO | Inflation |
|--------|----------|-----|-----------|
| Holdout r (county-prior) | 0.672 | 0.448 | +0.224 |
| RMSE | 0.073 | 0.091 | -0.018 |

**LOO is now reported in validate_types output.** Future optimization should target the LOO metric.

## Experiment 1: Observed-Primary Covariance ✅ IMPLEMENTED
**Hypothesis:** Use Ledoit-Wolf regularized observed electoral correlation as primary, instead of demographic Pearson correlation.

**Result:** LOEO val_r = 0.995 (vs baseline 0.556). The observed electoral comovement structure is highly stable across elections — T-1 training predicts held-out at r=0.995.

**Implemented:** `construct_type_covariance.py` now uses observed LW as primary, with demographic construction as fallback.

## Experiment 2: Hierarchical Shrinkage ❌ NO IMPROVEMENT
**Hypothesis:** Boundary counties (low dominant type score) should trust the type adjustment less and use their own momentum instead.

**Result:** Baseline wins. No partial-pooling strategy beats flat type adjustment. County momentum (training trend) doesn't generalize to holdout. The type structure already captures all predictable cross-sectional variance.

| Strategy | r | vs Baseline |
|----------|---|-------------|
| Baseline (α=1.0) | 0.672 | — |
| α = dominant_score | 0.547 | -0.125 |
| α = dominant_score² | 0.476 | -0.196 |
| fixed α=0.9 | 0.640 | -0.032 |

## Experiment 3: Senate Pair Quality Filtering 🟡 MARGINAL
**Hypothesis:** Removing noisy Senate election pairs improves type discovery.

**Result:** Presidential-only (9 dims) marginally beats 33-dim baseline on standard metric (+0.007). On LOO with seed stability, the difference vanishes (both ~0.39 mean LOO r).

| Strategy | Dims | Standard r | Coherence |
|----------|------|-----------|-----------|
| Baseline (all 33) | 33 | 0.698 | 0.783 |
| Drop all Senate | 18 | 0.694 | 0.789 |
| Presidential only | 9 | 0.705 | 0.830 |

**Seed stability (5 seeds):** Pres-only J=70-100 mean LOO r=0.39±0.03 vs baseline 0.38±0.02. Not significant.

**Not implemented.** The coherence improvement (0.83 vs 0.78) is interesting but doesn't translate to meaningful holdout improvement.

## Experiment 4: State-Level Bias Correction ❌ NOT VIABLE
Per-state analysis revealed systematic biases (Alaska RMSE=0.38, Idaho r=0.05), but:
- Training-derived state effects are all ~zero (types already capture state patterns)
- The bias is 2024-specific (Hispanic realignment, etc.), not predictable from training

## Experiment 5: Ridge Regression Prediction ✅ COMPLETED (late)
**Hypothesis:** Ridge regression from type scores + county training mean may outperform simple type-mean adjustment.

**Result:** Ridge LOO r=0.447 vs type-mean LOO r=0.396 (+0.051). Alpha=6.55 (moderate regularization). This is meaningful — the type scores contain predictive signal that simple weighted-mean adjustment doesn't fully exploit.

**Not yet implemented.** Would require changes to the prediction pipeline. Worth pursuing in a future session.

## Summary of Improvements
| Change | Before | After | Status |
|--------|--------|-------|--------|
| Covariance construction | val_r=0.556 | val_r=0.915 | ✅ Merged |
| LOO validation metric | not reported | LOO r=0.448 | ✅ Merged |

## Recommendations for Future Work
1. **LOO-optimized J selection**: Re-sweep J using LOO metric. Current J=100 was optimized for inflated metric.
2. **Prediction method research**: Ridge/Lasso regression from type scores may outperform type-mean adjustment on LOO.
3. **Cross-resolution alignment**: Add county type info to tract model (not yet tested).
4. **Pres-only as option**: While not a clear win, monitor whether governor/Senate data helps as more elections are added.
