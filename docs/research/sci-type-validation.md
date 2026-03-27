# SCI-Type Validation Results

## Research Question

Do counties that are socially connected (high Facebook SCI) also tend
to belong to the same electoral type? If so, this validates that our
types capture real community structure, not just statistical artifacts.

## Data Summary

- **Counties**: 3,154
- **County pairs analyzed**: 4,753,986
- **Electoral types (J)**: 100

## Finding 1: Same-Type Counties Are More Socially Connected

| Metric | Same Type | Different Type | Ratio |
|--------|-----------|----------------|-------|
| Mean SCI | 224,885 | 21,404 | 10.51x |
| Mean log10(SCI) | 3.925 | 3.399 | +0.526 |

## Finding 2: SCI Correlates with Type Similarity

Correlation between log10(SCI) and cosine similarity of soft type
membership vectors:

- **Pearson r**: 0.1401 (p=0.00e+00)
- **Spearman r**: 0.1897 (p=0.00e+00)

## Finding 3: Effect Persists Across State Lines

Same-state pairs are both closer AND more likely to share a type.
The critical test: does the SCI-type relationship hold for
cross-state pairs?

- **% same-type among same-state pairs**: 10.1%
- **% same-type among cross-state pairs**: 1.3%

Cross-state only:

- Mean SCI (same type, cross-state): 23,435
- Mean SCI (diff type, cross-state): 6,925
- **Cross-state SCI ratio**: 3.38x

## Finding 4: SCI Signal Beyond Geographic Proximity

Partial correlation of log(SCI) vs type cosine similarity,
controlling for log(geodesic distance):

- **Partial r**: 0.0755 (p=0.00e+00)

This measures whether SCI adds information about type similarity
beyond what geographic distance alone provides.

## Finding 5: SCI-Type Relationship by Distance Bin

Holding distance roughly constant, same-type pairs still have
higher SCI than different-type pairs:

| Distance | N pairs | % Same Type | Mean SCI (same) | Mean SCI (diff) | Ratio | Pearson r |
|----------|---------|-------------|-----------------|-----------------|-------|-----------|
| 0-100km | 27,213 | 14.5% | 3,309,168 | 2,308,293 | 1.43x | 0.158 |
| 100-250km | 135,648 | 7.2% | 245,402 | 160,518 | 1.53x | 0.167 |
| 250-500km | 418,179 | 3.1% | 52,815 | 27,282 | 1.94x | 0.133 |
| 500-1000km | 1,225,950 | 1.5% | 8,641 | 6,235 | 1.39x | 0.056 |
| 1000-2000km | 2,017,735 | 1.0% | 3,608 | 2,851 | 1.27x | 0.024 |
| 2000-5000km | 911,531 | 0.9% | 2,775 | 2,172 | 1.28x | 0.035 |

## Interpretation

A positive SCI-type correlation validates that our electoral types
capture real social community structure. Counties in the same type
are not just statistically similar -- they are socially connected
in measurable ways.

The distance controls are critical: nearby counties are trivially
both more connected (shared commute networks) and more similar
(shared media markets, economies). The partial correlation and
within-bin analysis isolate the SCI signal that goes beyond
geographic proximity.

## Methodology

- **SCI data**: Facebook Social Connectedness Index (county pairs,
  Jan 2026 snapshot). Higher SCI = more Facebook friendships between
  two counties, scaled for population.
- **Type similarity**: Cosine similarity of J=100 soft membership
  vectors (temperature-scaled inverse distance to KMeans centroids).
- **Distance control**: Haversine distance between Census 2020
  county centroids.
- **Partial correlation**: Standard OLS residualization of both
  log(SCI) and cosine_sim on log(distance), then Pearson r of
  residuals.