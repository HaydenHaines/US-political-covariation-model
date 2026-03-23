# Covariance Construction Experiment — S164 (2026-03-22)

## Question
Can we improve covariance validation (currently r=0.216) by:
1. Reducing constructed rank via PCA (P1.2)
2. Preserving negative correlations (P1.3)
3. Tuning lambda shrinkage

## Setup
- J=43 types, 41 demographic features, 19 elections
- Validation: off-diagonal Pearson r between constructed and observed correlation matrices
- Sweep: lambda ∈ {0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0} × floor_negatives ∈ {True, False} × max_rank ∈ {None, 18, 15, 10}

## Results

| Config | val_r | eff_rank |
|--------|-------|----------|
| All lambda/floor/rank combos | 0.202–0.205 | 1.0–2.2 |

**Validation r is completely flat across ALL parameter combinations.**

## Key Findings

1. **Effective rank is always ~2** after shrinkage. Even at lambda=1.0 (pure demographics), the effective rank is only 2.2. The shrinkage toward all-1s dominates the spectrum.

2. **PCA rank reduction has no effect** because the matrix is already effectively low-rank after shrinkage.

3. **Negative correlations make no difference** (0.205 vs 0.202 — within noise).

4. **Observed covariance has 3 dominant eigenvalues** (21.7, 13.6, 6.8), with the rest near zero. Electoral comovement in FL/GA/AL is essentially 3-dimensional. The observed matrix is not PD (13 negative eigenvalues, all ≈0 — rank deficiency from 43 types but only 19 elections).

5. **Observed off-diagonal range**: [-0.154, 0.999], mean=0.420.

## Root Cause

The constructed covariance from demographic profiles does not correlate with observed electoral comovement. This is not a parameter problem — it's a **feature selection problem**. The demographics we're using (population, race, education, religion, etc.) don't capture the dimensions along which types actually co-move in elections.

This was diagnosed in S163: "demographic similarity ≠ electoral comovement."

## What Would Help

1. **Use observed covariance directly** (with PD repair via eigentruncation). Sample covariance from 19 elections is rank-deficient but captures real structure.
2. **Find features that predict comovement** — e.g., media market overlap, economic exposure similarity, partisan sorting metrics. These would make the constructed covariance meaningful.
3. **Hybrid approach** — blend observed (for dimensions where we have data) with constructed (for regularization).

## Code Changes (retained)

- `_rank_reduce()` function added to `construct_type_covariance.py`
- `max_rank` parameter available but inactive in config
- 9 new tests for rank reduction
- Infrastructure ready for when better features are available
