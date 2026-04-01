# Covariance Cross-Race Audit (Issue #87)

**Date:** 2026-04-01  
**Script:** `scripts/audit_covariance_cross_race.py`  
**Branch:** `research/covariance-cross-race`

---

## Question

Does the type covariance Σ systematically understate Senate/governor comovement
patterns because presidential shift data dominates its construction?

---

## Setup

The current covariance pipeline (`construct_type_covariance.py`) builds Σ from
**all 60 training dimensions** (pres + gov + senate), grouped by election pair
(every 3 dims = 1 pair). After StandardScaler normalization, each of the 20
election pairs contributes equally: 5 presidential (25%), 7 governor (35%),
8 Senate (40%). Presidential weighting (pw=8.0) is only applied during KMeans
type discovery, **not** during covariance construction.

Key data: 3,154 counties, J=100 types, 60 shift dims (5 pres + 7 gov + 8 senate pairs × 3 each).

---

## Findings

### 1. Raw variance: governor shifts dominate without scaling

| Race   | N dims | % of total variance |
|--------|--------|---------------------|
| pres   | 15     | 4.9% (0.030 per dim) |
| gov    | 21     | 59.8% (0.257 per dim) |
| senate | 24     | 35.3% (0.133 per dim) |

Presidential shifts have dramatically lower raw variance (0.03 vs 0.26 for gov).
Without StandardScaler this would make governors dominate KMeans distance. After
scaling, each dim has variance=1 and pct of total = n_dims/60.

### 2. Per-race covariance matrices are nearly uncorrelated with each other

Computing separate LW-regularized J=100 covariance matrices for each race type:

| Race   | Mean off-diag corr | vs pres (cross-race r) |
|--------|-------------------|------------------------|
| pres   | 0.2823            | —                      |
| gov    | 0.1314            | **r=0.1161**           |
| senate | 0.2088            | **r=0.1653**           |

**Gov vs pres: r=0.12. Senate vs pres: r=0.17.** These are near-zero correlations.
This means presidential comovement patterns and off-cycle comovement patterns are
nearly orthogonal at the type level.

### 3. The production covariance is most correlated with presidential (r=0.46)

Despite presidential pairs being 25% of dims (post-scaling), the production
covariance correlates strongest with the pres-only covariance (r=0.46 vs 0.09
for gov and 0.12 for senate). This is consistent with the hypothesis that
presidential shifts have lower noise relative to signal at the type level,
causing presidential structure to dominate the combined covariance.

### 4. Per-type divergence is large (mean row MAD ~0.33)

The mean row-wise absolute deviation between pres and gov covariance matrices
is **0.33** (gov_vs_pres) and **0.28** (sen_vs_pres). For comparison, off-diagonal
correlations range roughly [-0.6, 0.6], so a MAD of 0.30 represents substantial
divergence. About 10-15 types show row_mad > 0.50, indicating their comovement
behavior genuinely differs across race types.

### 5. LOEO validation — combined approach is clearly best

| Strategy              | LOEO r | N elections |
|-----------------------|--------|-------------|
| Current (all equal)   | **0.9943** | 20 |
| Presidential + Senate | 0.9722 | 13 |
| Governor only         | 0.9267 | 7  |
| Senate only           | 0.9252 | 8  |
| Presidential only     | 0.8495 | 5  |

The combined approach (all 20 election pairs) achieves LOEO r=0.9943, far
outperforming any single-race subset. This is the most important finding.

The drop from 0.9943 to 0.8495 (pres-only) is primarily a **sample size effect**:
LOEO needs many elections to be reliable, and with T=5 (pres-only) each held-out
election represents 20% of the training data, causing much higher variance.
More election pairs = more stable covariance estimates.

### 6. Reweighted blends show no improvement over current production

All reweighting schemes (pres x2, x4, x8, pres+sen, pres-only blend) produce
covariances that correlate with the production covariance at r=0.36-0.46.
None of the blend strategies yields a higher LOEO r than the current approach
(which would require LOEO > 0.9943).

---

## Interpretation

The cross-race covariances are genuinely nearly orthogonal (r~0.12-0.17),
which means:

1. **Presidential, governor, and Senate elections tap different comovement
   structures at the type level.** This is not noise — it reflects that
   off-cycle elections have different candidate effects, different turnout
   compositions, and different issue salience that scramble type comovement
   patterns relative to presidential elections.

2. **The combined covariance is doing something useful precisely because it
   averages across these orthogonal signals.** The high LOEO r (0.9943 vs
   0.8495 pres-only) comes from having 20 elections to estimate a stable
   J=100 covariance, not from any race-type signal alignment.

3. **The current Σ is primarily estimating "what types swing together across
   many kinds of elections."** This is actually the right target for a
   propagation prior — you want to know "if type X swings Dem, does type Y
   also swing Dem?" across all electoral contexts, not just presidential.

4. **Issue #87's concern was correct in substance but inverted in implication.**
   Presidential shifts do have less variance and less comovement information
   per pair than off-cycle races. But the fix is not to upweight presidential —
   it's that the current mixed-race approach already averages the presidential
   signal with the richer off-cycle signal. Presidential-only would be worse.

---

## Recommendation

**Status: Issue #87 downgraded to low priority.**

The current covariance construction approach is adequate. Key reasons:

- LOEO r = 0.9943 is extremely high, meaning the structure is stable
- Adding presidential weighting to covariance construction does not help
- The cross-race divergence is real but the combined estimate is more
  stable than any single-race estimate

**One genuine improvement worth exploring (separate issue):** The per-type
divergence analysis shows ~10-15 types with row_mad > 0.50 between presidential
and governor covariance rows. These types have genuinely different comovement
in off-cycle vs presidential elections. This is likely explained by the
**voter behavior layer** (τ and δ parameters) — it's the correct place to
model this divergence, not the covariance matrix. Covariance should represent
structural type relationships; cycle-specific behavior belongs in τ/δ.

**One data quality issue identified:** Governor shifts have ~10x the raw
variance of presidential shifts, suggesting noisy/uncontested races in the
early gov pairs (1994→1998, 1998→2002). Consider filtering heavily uncontested
gubernatorial cycles from the training data. This is a data hygiene issue
independent of covariance construction.

---

## Supporting Data Files

- `scripts/audit_covariance_cross_race.py` — full audit script
- `data/covariance/type_covariance.parquet` — current production Σ (J=100)
- `data/covariance/type_correlation.parquet` — current production correlation

---

## Related

- CLAUDE.md "Covariance — Cross-Race Underrepresentation" debt note
- `docs/superpowers/specs/2026-03-27-tract-primary-behavior-layer-design.md` (τ/δ layer)
- `src/covariance/construct_type_covariance.py` (production covariance code)
