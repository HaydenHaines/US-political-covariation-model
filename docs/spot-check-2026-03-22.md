# County Spot Check & Covariance Diagnostic — 2026-03-22 (S163)

## Summary

Ran prediction spot checks on 4 bellwether counties and a covariance diagnostic. Found both a prediction pipeline issue and confirmed the covariance rank-deficiency root cause.

## Bellwether County Results

| County | FIPS | Dominant Type | 2024 Actual | Model Predicted | Error |
|--------|------|--------------|-------------|-----------------|-------|
| Pinellas, FL | 12103 | 24 (Southern Rural Conservative) | 46.9% D | ~47% | ~0pp |
| Cobb, GA | 13067 | 6 (Suburban Professional) | 56.9% D | 65.3% | D+8.4pp |
| DeKalb, GA | 13089 | 6 (Suburban Professional) | 81.9% D | 65.1% | R-16.7pp |
| Miami-Dade, FL | 12086 | 40 (singleton) | 43.9% D | 44.1% | D+0.2pp |

### Key Findings

1. **DeKalb and Cobb share Type 6** — correct for shift prediction (both suburban Atlanta, respond to same forces) but produces identical 2026 predictions (67-70%) despite voting 25pp apart. Types predict comovement, not levels.

2. **Miami-Dade is a success** — singleton type correctly isolates its unique rightward-drifting Hispanic profile. 0.1pp error.

3. **Pinellas** — assigned to "Southern Rural Conservative" with 99.8% confidence. Type name misleading for an urban swing county but prediction is numerically OK.

4. **Type assignments are near-hard** at T=10 — scores of 0.998-1.000. Soft membership is operationally minimal.

## Worst 10 Prediction Errors (2024 Presidential)

RMSE across 293 counties: **8.66 pp**. Systematic pattern: majority-Black counties underpredicted.

8 of 10 worst errors are Black Belt / majority-Black counties assigned to types anchored by majority-white demographics. The model predicts 33-42% Dem for counties that vote 65-85% Dem.

**Root cause:** KMeans types are discovered from shift vectors (how counties move), not from racial composition. A majority-Black county that shifts similarly to a majority-white county gets the same type — but they vote at very different baseline levels. The type-level prediction uses the type mean as the prior, which averages Black and white county baselines within the type.

## Covariance Diagnostic

**Validation r = 0.216** — unchanged from 0.221 before feature expansion (31→37 features).

### Why features aren't helping

1. **Observed covariance rank = 18** (from 19 elections) vs constructed rank = 37. Most of the constructed correlation structure has no empirical support.

2. **Systematic over-prediction of correlation**: Constructed off-diagonal mean = 0.561, observed mean = 0.420. The constructed matrix predicts moderate correlation (0.5-0.9) for type pairs that have near-zero observed correlation.

3. **The worst mismatches** are Type 21 vs many other types: observed ≈ 0 but constructed = 0.77-0.93. Demographically similar but electorally independent.

4. **Negative correlations exist** (down to -0.15 observed) but are floored to 0 in construction.

### Implications

- Adding more demographic features helps with rank deficiency but doesn't fix the fundamental issue: **demographic similarity ≠ electoral comovement**.
- The covariance construction works well for strongly correlated types (observed ~0.99 → constructed 0.7-0.97) but fails for weakly/un-correlated pairs.
- The hybrid fallback (blending with observed covariance when validation_r < 0.4) is correctly triggered but the blend still inherits the constructed matrix's overestimation.

## Recommendations

### Short-term (this session)
- [x] Continue expanding features to full rank (≥43)
- [ ] Add IRS migration features (agent in progress)
- [ ] Test `floor_negatives=False` — negative correlations are real signal

### Medium-term (next sessions)
1. **Fix prediction pipeline** — use county-level priors (each county's own historical baseline), not type-level means. Types should only determine HOW counties move together (covariance), not WHERE they start.
2. **Factor model for covariance** — reduce constructed matrix to rank ~18 via PCA to match observed dimensionality. This would naturally suppress the over-predicted weak correlations.
3. **Feature selection** — not all demographics predict comovement. Use Lasso or mutual information to select the ~15 features that actually correlate with observed off-diagonals.
4. **Consider adding % Black as a direct covariance feature** — the Black Belt prediction errors suggest this single feature would cut RMSE substantially.
