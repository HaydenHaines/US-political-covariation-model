---
source: empirical — derived from FL+GA+AL 2022 ACS tract analysis
captured: 2026-03-18
---

# NMF Community Detection — Patterns and Findings

Covers K selection, component quality evaluation, and visualization strategy
for the mixed-membership NMF approach used in Stage 2.

## K Selection

NMF reconstruction error decreases monotonically with K — there is no clean
elbow in this data. K choice is a modeling judgment, not a data-driven answer.

**Empirical results across K=4..11 (FL+GA+AL, 9,295 populated tracts):**

| K | NNLS R² | Largest bucket | Notes |
|---|---------|---------------|-------|
| 4 | 0.615 | 44% | Balanced; fuses everything into racial types |
| 6 | 0.681 | 52% | Best R²; fuses Asian+Knowledge Worker |
| 7 | 0.661 | 60% | Cleanest separation; separates Asian from KW |
| 8 | 0.624 | 56% | Adds walkable-urban-professional distinction |
| 9 | 0.647 | 56% | Artifact WFH+Hispanic fusion component |
| 11 | 0.719 | 97.6% | Broken — nearly all tracts in one bucket |

**R² is non-monotonic in K** because each K finds a different NMF local
optimum. Some solutions align better with political variation than others.
Do not use R² alone to choose K.

**Selected: K=7.** Rationale:
- Properly separates Asian demographic from Knowledge Worker/WFH type —
  these have completely different geographic logics (Atlanta suburbs vs.
  university towns) and must not be fused for geographic analysis
- Generic bucket (60%) is larger than K=8 (56%) but K=8 produces no
  dominant tracts for its knowledge worker and walkable-urban-professional
  components, making them unidentifiable in single-election validation
- K=7 has stable NNLS estimates for 5 of 7 components; only Black urban
  and Knowledge Worker exceed [0,1] due to sparse dominance (expected)
- Passes the geographic coherence test: each component has a plausible
  geographic concentration (Black Belt, South FL, retirement corridors,
  university towns, generic suburban ring)

## The "Generic Baseline" Component

Every NMF solution at every K produces one component that sits near the
origin of the feature space — demographically near-average tracts that
cannot be explained by any distinctive named type. This is not a modeling
artifact; it reflects real heterogeneity in the population.

**This is a finding, not a failure.** ~56-60% of Deep South suburban
tracts are genuinely heterogeneous — not particularly white, not
particularly affluent, not transit-oriented, not Hispanic, just average
suburban. Forcing these into a named community type would be dishonest.

The NMF membership entropy already encodes this distinction:
- Low entropy (near 0) = tract clearly belongs to one community type
- High entropy (near max) = tract is a genuine demographic mixture
- Average entropy at K=7: 1.234 / 1.946 max (63%)

## Visualization Strategy for Geographic Analysis

**Do not map dominant community type** — the generic baseline swamps
everything and hides the interesting geographic patterns.

**Map membership intensity per community** — for each community type k,
show the membership weight w_ik as a continuous color scale. This:
- Shows clean geographic blobs where each community concentrates
- Lets heterogeneous tracts naturally appear "light" on every map
- Avoids forcing every tract into a discrete category
- Supports geographic questions (borders, rivers, regional spillover)
  by showing continuous gradients rather than hard boundaries

You'll produce K maps, one per community type. The heterogeneous
tracts will be low-intensity on all of them simultaneously — the
geographic expression of "this is just suburbs."

## NNLS Validation Pathology

Single-election NNLS regression of dem_share on membership weights produces
pathological θ values (> 100%) for sparse components — those with few
dominant tracts (e.g., Knowledge Worker with 16 dominant tracts at K=7).

This is expected: NNLS has no upper bound constraint, and sparse components
have tiny weights in most tracts, so the solver inflates θ to compensate.

**Interpretation:** R² from NNLS (0.661 at K=7) measures the maximum
predictive power of the community basis — valid even when individual θ
are unrealistic. The Stage 3 Bayesian model will provide regularized
estimates via informative priors. Do not interpret individual NNLS θ
values for sparse components as meaningful vote shares.

## Gotchas

**1. nndsvda initialization is required — do not use random init.**
`init="random"` with `max_iter=500` fails to converge for almost all K
values. nndsvda (Non-Negative Double SVD with average fill) converges in
<1,400 iterations and produces stable, reproducible results.
```python
NMF(n_components=k, init="nndsvda", max_iter=3000, tol=1e-4)
```

**2. Row-normalize W before using as membership weights.**
Raw NMF W matrix rows do not sum to 1. Divide each row by its sum so
that w_ik represents the fraction of tract i belonging to community k.
Required for Stage 3 regression and the membership intensity maps.

**3. Uninhabited tracts (pop=0) must be excluded before fitting.**
98 of 9,393 FL+GA+AL tracts have pop_total=0. Fit NMF on populated tracts
only; write NaN memberships for uninhabited tracts in the output.
