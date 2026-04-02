# PCA Before KMeans — Experiment Results

**Issue:** #131  
**Date:** 2026-04-01  
**Script:** `scripts/experiment_pca_before_kmeans.py`  

## Research Questions

1. Does PCA before KMeans improve LOO holdout accuracy?
2. How many PCs capture most variance? (scree data below)
3. Does PCA help reduce noise in governor/Senate shifts?

## Baselines (to beat)

- **Standard holdout r:** 0.6982 (J=100, no PCA, StandardScaler+pw=8)
- **LOO r (type-mean):** 0.4485 (honest generalization metric, S196)

## Scree Plot Data

Cumulative variance explained by each PCA component on the
30%→100% training matrix (33 dims total).

| PC | Individual Var | Cumulative Var |
|----|---------------|----------------|
| PC 1 | 0.3015 | 0.3015 |
| PC 2 | 0.2370 | 0.5385 |
| PC 3 | 0.1523 | 0.6908 |
| PC 4 | 0.1258 | 0.8167 |
| PC 5 | 0.0918 | 0.9084 |
| PC 6 | 0.0557 | 0.9641 |
| PC 7 | 0.0062 | 0.9703 |
| PC 8 | 0.0043 | 0.9746 |
| PC 9 | 0.0040 | 0.9787 |
| PC10 | 0.0034 | 0.9821 |
| PC11 | 0.0030 | 0.9851 |
| PC12 | 0.0028 | 0.9879 |
| PC13 | 0.0021 | 0.9901 |
| PC14 | 0.0020 | 0.9921 |
| PC15 | 0.0017 | 0.9938 |
| PC16 | 0.0014 | 0.9952 |
| PC17 | 0.0012 | 0.9964 |
| PC18 | 0.0010 | 0.9975 |
| PC19 | 0.0008 | 0.9983 |
| PC20 | 0.0007 | 0.9990 |
| PC21 | 0.0006 | 0.9996 |
| PC22 | 0.0004 | 1.0000 |
| PC23 | 0.0000 | 1.0000 |
| PC24 | 0.0000 | 1.0000 |
| PC25 | 0.0000 | 1.0000 |
| PC26 | 0.0000 | 1.0000 |
| PC27 | 0.0000 | 1.0000 |
| PC28 | 0.0000 | 1.0000 |
| PC29 | 0.0000 | 1.0000 |
| PC30 | 0.0000 | 1.0000 |

- **80% variance:** PC4
- **90% variance:** PC5
- **95% variance:** PC6
- **99% variance:** PC13

## Results Table

Standard holdout r inflates ~0.22 vs LOO (S196); LOO r is the honest metric.

| Method | n_comp | Var Explained | Holdout r | LOO r | Δ Holdout | Δ LOO |
|--------|--------|--------------|-----------|-------|----------|-------|
| baseline_no_pca      |     33 |        1.000 | 0.6982 | 0.4485 | +0.0000 | +0.0000 |
| PCA                  |      5 |        0.908 | 0.6617 | 0.4379 | -0.0365 | -0.0105 |
| PCA                  |     10 |        0.982 | 0.6849 | 0.4488 | -0.0134 | +0.0003 |
| PCA                  |     15 |        0.994 | 0.6415 | 0.4176 | -0.0567 | -0.0308 |
| PCA                  |     20 |        0.999 | 0.7016 | 0.4524 | +0.0034 | +0.0039 |
| PCA                  |     25 |        1.000 | 0.6982 | 0.4485 | +0.0000 | -0.0000 |
| PCA                  |     30 |        1.000 | 0.6982 | 0.4485 | +0.0000 | -0.0000 |
| PCA_whiten           |      5 |        0.908 | 0.6550 | 0.4106 | -0.0433 | -0.0378 |
| PCA_whiten           |     10 |        0.982 | 0.6517 | 0.4253 | -0.0465 | -0.0231 |
| PCA_whiten           |     15 |        0.994 | 0.6105 | 0.4986 | -0.0877 | +0.0502 |
| PCA_whiten           |     20 |        0.999 | 0.5877 | 0.4630 | -0.1105 | +0.0145 |
| PCA_whiten           |     25 |        1.000 | 0.5996 | 0.4988 | -0.0987 | +0.0503 |
| PCA_whiten           |     30 |        1.000 | 0.5083 | 0.4166 | -0.1900 | -0.0319 |

## Key Findings

**Best standard holdout r:** PCA n_comp=20, r=0.7016 (Δ=+0.0034)

**Best LOO r:** PCA_whiten n_comp=25, r=0.4988 (Δ=+0.0503)

**PC structure:** The 33-dim training matrix has a dramatic drop-off after PC6 (96.4% variance). PCs 7-22 together account for only 3.6%, with PCs 23+ being zero (linear combinations of prior PCs). The 6-component structure reflects strong cross-election correlation of presidential shifts.

**PCA (no whitening) sweep (6 settings):** Results are erratic — no monotonic relationship between n_components and accuracy. Only 2 of 6 beat the LOO baseline, and both are marginal (Δ ≤ +0.004). PCA without whitening keeps PC1 dominant (30% variance), so KMeans still clusters primarily along the national swing axis.

**Whitening sweep (6 settings):** Much stronger results. Best: n=25, LOO r=0.4988 (Δ=+0.0503). Whitening rescales all PCs to unit variance before KMeans, so the Euclidean metric treats the college-town realignment signal (PC7, 0.6% variance) the same as the national swing (PC1, 30%). This allows KMeans to discover type distinctions based on structural deviations from the dominant national trend, rather than clustering primarily by swing magnitude.

**Seed stability check (5 seeds):** The whitening result is robust. Across seeds 0, 1, 7, 42, 99:
- Whitening n=25: LOO r range 0.476–0.532, mean ≈ 0.497
- Baseline no PCA: LOO r range 0.419–0.456, mean ≈ 0.440
- Mean improvement: +0.057 (consistent across random initializations)

## Recommendation

**INVESTIGATE whitening further before adopting.** The whitening finding (+0.050 LOO r, robust across seeds) is meaningful but contradicts the earlier exp_pca_clustering.py study (issue #93, which only measured standard holdout r). The whitening improvement is genuine: it works because treating all PCs equally allows KMeans to cluster on structural deviations rather than national swing magnitude.

**However**, whitening with n=25 still degrades standard holdout r (0.600 vs 0.698 baseline — a -0.098 drop). This divergence between standard r and LOO r is unusual and warrants investigation before production adoption. Possible explanations:
1. The whitened types may be more behaviorally coherent within each type (better LOO) but noisier in absolute level (worse holdout r).
2. The whitening at n=30 collapses to holdout_r=0.508 (very poor), suggesting a cliff near the rank limit of the training matrix.

**If adopting**, suggested setting: `pca_components: 25` with `whiten: True` in `config/model.yaml`. This requires updating `discover_types()` to expose the `whiten` parameter.

**To confirm before adopting:** Run Ridge+demo LOO r (the highest-quality metric, currently 0.695) with whitening to see if the improvement persists through the Ridge layer. The type-mean LOO r is a coarser signal.

## Notes

- This experiment is read-only: no production files are modified.
- `random_state=42` for PCA throughout. Seed stability check used seeds 0, 1, 7, 42, 99 with `n_init=10`.
- Shift data: `data/shifts/county_shifts_multiyear.parquet` (gitignored).
- LOO r uses the same county_prior formula as `holdout_accuracy_county_prior_loo()` in `src/validation/holdout_accuracy.py`. County priors are in raw log-odds shift space (not scaled), matching the production validation pipeline.
- Previous experiment (`exp_pca_clustering.py`, issue #93) found PCA without whitening offered no improvement using standard holdout r. This experiment adds LOO r and whitening, revealing a genuine whitening benefit that the earlier study missed because it only measured standard (inflated) holdout r.
- Results saved to `data/experiments/pca_before_kmeans_results.parquet`.
