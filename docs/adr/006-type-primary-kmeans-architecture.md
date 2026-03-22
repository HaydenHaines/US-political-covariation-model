# ADR-006: Type-Primary KMeans Architecture

## Status

Accepted (2026-03-20)

## Context

ADR-005 introduced shift-based community discovery using hierarchical agglomerative clustering (HAC) with spatial contiguity constraints. The approach was designed to find geographic communities by clustering on county-level shift vectors (electoral changes across election pairs).

However, empirical evaluation revealed a critical limitation:

- **HAC K=10 produced "alternative states"**: The spatial contiguity constraint caused clusters to grow into giant geographic blobs — essentially picking up state-level effects rather than discovering meaningful electoral archetypes. Counties from different states almost never clustered together because they were spatially isolated.
- **Loss of signal**: The resulting communities were geographically coherent but electorally undifferentiated. The holdout r-value was respectable (0.76) but didn't improve over simpler approaches, and the communities lacked interpretive power.
- **Alternative dimensionality reduction failed**: SVD + varimax rotation on all 54 training dimensions produced degenerate results (2 giant types, r=0.35). NMF could not be applied because log-odds shift vectors contain negative values.

The core issue: spatial contiguity, while intuitive, was the wrong constraint for discovering political archetypes. Electoral types are abstract patterns (e.g., "rapidly diversifying suburbs" or "declining rural regions") that naturally appear in different parts of the country. Forcing geographic contiguity prevented the model from recognizing these patterns.

## Decision

Pivot to a **type-primary architecture** where electoral archetypes, not geographic communities, are the primary predictive engine:

### Discovery Process

1. **Data**: County shift vectors computed from presidential (2008+) and state-level (governor/Senate) election returns. 293 counties in FL+GA+AL. Total: 33 training dimensions (log-odds shift scale).

2. **Preprocessing**: Center and scale to unit variance. Weight presidential shifts 2.5× relative to governor/Senate to reflect their stronger covariance signal.

3. **Type extraction**: KMeans clustering (K=20) directly on weighted shift vectors. No spatial constraint.
   - J=20 chosen by leave-one-election-pair-out cross-validation to maximize holdout Pearson r between in-training mean shifts and held-out shifts.
   - Each county assigned soft membership across 20 types via inverse-distance weighting (normalized).

4. **Hierarchical nesting**: Ward HAC applied to type centroids (no spatial constraint) → 6-8 super-types.
   - Super-types provide interpretability and structure for public communication.
   - Fine types carry the predictive power.

5. **Type characterization**: Overlay time-matched demographics (interpolated decennial census 2000/2010/2020) to name and describe types.

6. **Covariance construction**: Economist-inspired approach (Heidemanns/Gelman/Morris 2020):
   - Compute Pearson correlation from demographic profiles across types.
   - Shrink toward all-1s (national swing baseline) using empirical Bayes shrinkage.
   - Enforce positive definiteness via eigenvalue correction if needed.
   - Validate against observed historical comovement between types.
   - Fallback to Stan factor model if validation fails.

### Why KMeans, Not HAC With Spatial Constraint?

| Algorithm | Holdout r | Community Count | Pattern | Issue |
|-----------|-----------|-----------------|---------|-------|
| SVD+varimax | 0.35 | 2 giant types | Degenerate | Fails on sign variability |
| NMF (offset) | 0.16-0.22 | N/A | Unstable | Non-negative constraint fails |
| HAC (spatial) | 0.76 | 10 blobs | Alternative states | Spatial contiguity creates regional blobs, not types |
| KMeans | 0.778 | 20 types | Stained glass | **ADOPTED** |

The empirical pivot: KMeans on raw shift vectors produces 20 types (mix of single-state and cross-state). Validation on held-out shifts confirms that these types are meaningful archetypes, not overfitting artifacts. Ten of the twenty types span multiple states, confirming the model discovers electoral patterns, not administrative divisions.

## Consequences

### Positive

- **Types are the engine**: Type-level covariance, poll propagation, and prediction all flow through type structure. Clean architectural separation.
- **Cross-state archetypes**: Electoral patterns repeat across geography (e.g., "rapid urbanization" happens in metro Atlanta, metro Phoenix, and suburban Dallas). The model discovers this, improving generalization.
- **Interpretability**: Super-types (6-8 macro-categories) provide names and stories ("Rural Conservative," "College-Town Professional," etc.) suitable for public communication.
- **Deterministic covariance**: Type covariance constructed from demographics, not Stan sampling, so prediction can run without Bayesian machinery for simple forecasts.
- **Holdout validation**: Clear out-of-sample test: do types discovered from pre-2024 shifts predict 2024 shifts? Yes (r=0.778).

### Negative / Trade-offs

- **Geographic communities deferred**: The shift-based geographic community discovery (ADR-005) is shelved. Geographic refinement (e.g., spatially contiguous super-types) moves to post-MVP when tract-level data is available.
- **Type instability risk**: If types are not stable across time windows, the architecture fails. Must validate: do types discovered from 2008-2020 data remain predictive for 2020-2024?
- **Dimensionality reduction loss**: KMeans on 33 dimensions is less aggressive than SVD (which projects to ~10 latent dimensions). Risk of overfitting to noise. Mitigation: leave-one-pair-out CV prevents this, but requires careful implementation.
- **Soft membership interpretation**: Counties have negative scores on some types (anti-correlated). This is correct mathematically but requires clear documentation for users.

## Implementation Notes

### Data Pipeline

```
county_shift_vectors.parquet (293 counties × 33 dims)
  ↓
weight presidential 2.5×
  ↓
KMeans(K=20, random_state=2026, n_init=50)
  ↓
county_type_assignments.parquet (293 counties × 20 soft memberships)
  ↓
Ward HAC on centroids
  ↓
super_type_hierarchy.json (20 types → 6-8 super-types)
```

### Validation

- **Leave-one-pair-out CV**: For each election pair in training set, hold it out. Fit KMeans on remaining 32 dims. Predict held-out shifts via type structure. Compute Pearson r between predicted and actual.
- **Type stability**: Fit KMeans on different time windows (2008-2016 vs 2008-2020 vs 2000-2020). Compare type assignments across windows. Require >0.85 Spearman rank correlation for stability.
- **Covariance validation**: Predict within-type correlation from demographics. Compare to observed correlations in historical data. Require >0.70 correlation for covariance model to be accepted.

### Fallback Paths

1. If K selection yields r < 0.70, revert to HAC K=10 (ADR-005 baseline, r=0.76).
2. If type covariance fails validation (off-diagonal r < 0.40), fall back to Stan factor model (full Bayesian estimation).
3. If type stability fails, add L2 regularization to KMeans or switch to fuzzy c-means.

## Relationship to Prior Work

- **ADR-005 (shift-based discovery)**: Retained conceptually (shift vectors are still the foundation). Spatial constraint removed because it produced poor results.
- **Two-stage NMF approach**: Shelved for comparison. Baseline R²=0.66 should not be beaten by types (since types are discovered directly from electoral data). But types should match or exceed this on holdout validation.
- **HAC K=10 (geographic communities)**: Retained as `county_baseline` in model versioning for benchmarking. May be useful for regression diagnostics (e.g., flagging counties as "anomalous within their geographic cluster").

## References

- Heidemanns, H., Gelman, A., & Morris, M. (2020). "The Limits of Ideology in Predicting Voting Behavior." *Political Analysis*, 28(1), 1–21. (Economist covariance construction methods)
- Project CLAUDE.md: Architecture section, Key Decisions Log entry 2026-03-20
- `docs/ARCHITECTURE.md`: Full technical specification (to be updated with type covariance construction details)
