# Delta (δ) Regularization Approaches for Behavior Layer

**Source**: Gemma 4 research (S499, 2026-04-08). Validated against political science literature.

## Problem Statement
100 type-level δ estimates from 1 governor cycle = massive overfitting. Adding raw δ hurts r (0.801 → 0.790). Variance exceeds signal.

## Recommended Approaches (priority order)

### 1. Hierarchical Partial Pooling (Best Fit)
Model δ_i ~ N(μ_δ, σ_δ²) across types. Individual δ_i shrunk toward global mean μ_δ. Types with sparse data get shrunk more. This is the standard approach in MRP literature (Gelman & Hill).

**Implementation**: PyMC or Stan. Estimate μ_δ and σ_δ from the data. Strong prior on μ_δ ~ N(0, τ²) since theory suggests net shift should be near zero.

### 2. PCA on Type-Level δ
Reduce 100 δ values to top K=3-5 principal components. Assumes δ variation is explained by a few latent dimensions (urban/rural, education, etc.). Dramatically reduces parameter count.

**Implementation**: Compute δ per type from governor data. SVD → keep top K components. Apply adjustment in reduced space.

### 3. Lasso / Horseshoe Prior (Sparsity)
If only a small subset of types genuinely shift in governor races, Lasso drives irrelevant types' δ to exactly zero. Horseshoe prior is the Bayesian equivalent.

**Practical note**: For J=100 with 1 training cycle, Lasso will likely shrink most δ to zero, effectively applying a sparse correction only where signal is strong.

## Key Insight
The current approach uses point estimates for δ — treating each type's shift as independent. Partial pooling assumes types are drawn from a common political process, which matches WetherVane's core assumption (types ARE the common structure). The model should leverage type structure for regularization, not fight it.

## When to Implement
When additional governor cycles are available (ideally 2+ cycles), or as a research experiment with the current single cycle. The blended priors (70/30) are the pragmatic interim solution — this research applies to the "build it right" phase.
