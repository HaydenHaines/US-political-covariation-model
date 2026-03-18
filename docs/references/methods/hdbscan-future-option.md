---
source: methodological note — not yet implemented
captured: 2026-03-18
status: FUTURE OPTION — consider if NMF community types prove unstable
---

# HDBSCAN — Future Alternative to NMF for Community Detection

## Why This Was Noted

NMF always produces a "generic baseline" component that dominates 50-60%
of tracts — a near-zero component capturing demographically average
(heterogeneous) tracts. This is mathematically unavoidable with NMF
because NMF cannot subtract from the mean the way PCA can.

HDBSCAN was identified as the most principled alternative if we ever want
to explicitly separate "identifiable community types" from "genuinely
heterogeneous / unclustered" tracts without a dominant baseline bucket.

## What HDBSCAN Does Differently

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
with Noise) finds clusters only where data points are sufficiently dense.
Points in low-density regions are explicitly labeled as **noise** (label
= -1) rather than forced into a cluster.

For this project:
- Dense demographic regions → named community types
- Sparse / heterogeneous tracts → labeled "unclustered" (not a named type)
- No K to specify — cluster count emerges from data density
- No dominant baseline bucket — the "noise" label is the honest answer
  for demographically average tracts

## Typical Implementation Pattern

```python
import umap
import hdbscan

# 1. Reduce dimensionality first (HDBSCAN struggles in 12-dim space)
reducer = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0,
                    random_state=42)
X_reduced = reducer.fit_transform(X_scaled)  # X_scaled = MinMax-normalized features

# 2. Fit HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10,
                             cluster_selection_method='eom')
labels = clusterer.fit_predict(X_reduced)
# labels == -1 → noise/heterogeneous
# labels >= 0 → community type index

# 3. Soft membership probabilities (optional)
soft_probs = hdbscan.all_points_membership_vectors(clusterer)
```

Key parameters to tune:
- `min_cluster_size`: minimum tracts to form a named community (~50-100)
- `min_samples`: controls noise sensitivity (higher = more noise)
- UMAP `n_neighbors`: controls local vs global structure (15-50 range)

## Tradeoffs vs NMF

| | NMF | HDBSCAN |
|--|-----|---------|
| Generic baseline | Always present | Replaced by explicit noise label |
| K choice | Required | Automatic |
| Soft assignment | Natural (W matrix) | Via `all_points_membership_vectors` |
| Reproducibility | Deterministic (nndsvda) | Stochastic (UMAP has random state) |
| Interpretability | Component profiles (H matrix) | Cluster centroids / exemplars |
| Stage 3 compatibility | Direct (W rows = membership priors) | Requires converting noise → zero weights |

## When to Consider Switching

- If NMF community types do not show geographic coherence in the
  membership intensity maps (see `nmf-community-detection.md`)
- If the 60% generic baseline is obscuring meaningful political patterns
  in Stage 3 covariance estimation
- If we want to produce a public-facing map that shows "where do
  identifiable communities live" vs "where is just suburbs"

## Dependencies to Add

```toml
# pyproject.toml
umap-learn>=0.5
hdbscan>=0.8
```

Note: hdbscan requires numba; add `numba>=0.57` if not already present.
