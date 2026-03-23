# GMM vs KMeans Experiment — S164 (2026-03-22)

## Question
Does Gaussian Mixture Model give better type discovery than KMeans? GMM provides proper Bayesian soft membership (posterior probabilities) vs KMeans' inverse-distance hack (T=10).

## Setup
- 293 counties, 33 training dims (presidential×2.5 weighted, 2008+)
- Holdout: 3 dims (pres_d/r/turnout_shift_20_24)
- J ∈ {20, 30, 43, 50}
- Methods: KMeans T=10, GMM full/diagonal/spherical covariance
- Both type-mean and county-level prior holdout r computed

## Results

### holdout_r (type-mean prior)

| Method | J=20 | J=30 | J=43 | J=50 |
|--------|------|------|------|------|
| KMeans T=10 | 0.771 | **0.813** | 0.832 | **0.851** |
| GMM full | 0.772 | 0.801 | 0.835 | 0.845 |
| GMM diag | **0.791** | 0.805 | **0.847** | 0.846 |
| GMM spherical | 0.789 | 0.800 | 0.834 | 0.849 |

### county_prior_r

| Method | J=20 | J=30 | J=43 | J=50 |
|--------|------|------|------|------|
| KMeans T=10 | 0.787 | **0.824** | 0.843 | **0.860** |
| GMM full | 0.780 | 0.808 | 0.841 | 0.852 |
| GMM diag | **0.801** | 0.815 | **0.850** | 0.853 |
| GMM spherical | 0.797 | 0.810 | 0.841 | 0.854 |

## Key Findings

1. **GMM diagonal beats KMeans at J=20 and J=43** (+2% and +1.5% holdout_r respectively). This is the most interesting result — proper soft membership gives a measurable improvement.

2. **KMeans wins at J=30 and J=50.** At high J, the KMeans inverse-distance soft membership with T=10 seems to regularize better than GMM's probabilistic assignment.

3. **GMM full covariance underperforms** at all J values tested. With only 293 counties and 33 dims, the full J×33×33 covariance per component is overfit.

4. **County-level priors improve ALL methods** by +0.01 to +0.02. This is independent of the clustering algorithm.

5. **The optimal (J, method) combination is J=50 KMeans T=10** (holdout_r=0.851, county_prior_r=0.860), but this may be overfitting by having too many types for 293 counties.

## Recommendation

**Do not switch from KMeans to GMM.** The gains are marginal (+1.5% at J=43), KMeans is simpler and faster, and KMeans wins at higher J. If we want to improve:
- Increasing J to 50 gives more than switching to GMM (+2% vs +1.5%)
- County-level priors give a universal +1-2% regardless of method
- The real bottleneck is features, not the clustering algorithm

## Future Work
- Test GMM diagonal at optimal J via leave-one-pair-out CV (not just J=43)
- Compare BIC/AIC for GMM model selection vs holdout-based J selection
- If we expand to national (3,000+ counties), GMM may become more valuable with more data
