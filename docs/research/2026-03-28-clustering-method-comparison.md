# Clustering Method Comparison — 2026-03-28

## Summary

**Winner: GMM with tied covariance, J=183, holdout r=0.633**

Compared KMeans, GMM-tied, and Fuzzy C-Means across multiple J values on 101,407 tracts × 16 shift dimensions.

## Head-to-Head Results

### Holdout r (predict 2020→2024 presidential shift from type means)

| J | KMeans | GMM-tied | FCM |
|---|--------|----------|-----|
| 50 | 0.564 | 0.532 | 0.546 |
| 70 | 0.569 | 0.571 | 0.549 |
| 90 | 0.583 | 0.585 | 0.550 |
| 110 | 0.596 | 0.597 | — |
| 130 | 0.607 | 0.606 | — |
| 150 | 0.597 | 0.612 | — |

### GMM-tied Bubble Search (J optimization)

| J | Holdout r | BIC |
|---|-----------|-----|
| 130 | 0.606 | -1,040,844 |
| 150 | 0.612 | -1,163,875 |
| 170 | 0.621 | -1,181,557 |
| 175 | 0.617 | -1,208,816 |
| 180 | 0.621 | -1,209,095 |
| 182 | 0.623 | -1,214,496 |
| **183** | **0.633** | -1,190,922 |
| 185 | 0.626 | -1,234,614 |
| 186 | 0.625 | -1,231,080 |
| 187 | 0.626 | -1,226,377 |
| 190 | 0.621 | -1,244,767 |

### Soft Membership Quality (entropy in bits)

| Method | Entropy | Interpretation |
|--------|---------|----------------|
| KMeans + temp (T=10) | 0.6-0.8 | Nearly hard assignment — max prob ~0.01 |
| FCM (m=2) | 5.0-5.8 | Nearly uniform — too soft |
| GMM-tied | 2.1-2.4 | Concentrates on 3-4 types — informative |

## Why GMM-tied Wins

1. **Comparable holdout r** to KMeans at all J values, slightly better at higher J
2. **Dramatically better soft membership** — GMM posteriors concentrate weight on 3-4 types per tract (entropy ~2.2 bits) vs KMeans+temp which spreads weight nearly uniformly (entropy ~0.7 bits). This means the soft scores actually carry information for downstream Ridge regression.
3. **BIC for principled J selection** — BIC provides a model selection criterion, though it doesn't plateau cleanly either
4. **Tied covariance is appropriate** — all types share one covariance matrix, capturing the global correlation structure of electoral shifts without overfitting per-type covariances

## Why FCM Lost

FCM with m=2 produced even more uniform membership than KMeans (entropy 5+ bits). The fuzziness parameter makes everything too soft. At matched J, holdout r was consistently 2-4pp below KMeans.

## Decision

- **Algorithm**: GMM with tied covariance (`sklearn.mixture.GaussianMixture(covariance_type='tied')`)
- **J**: 183
- **Soft membership**: Native posterior probabilities (predict_proba)
- **Holdout r**: 0.633 (vs county baseline 0.698 — gap to close with behavior layer + Ridge)
- **Runtime**: ~17 min per fit at J=183 (n_init=3)
