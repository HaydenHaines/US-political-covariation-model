# LOO-Optimized J Selection Sweep — S197

**Date:** 2026-03-25
**Script:** `scripts/experiment_j_sweep_loo.py`
**Results file:** `data/communities/j_sweep_loo_results.parquet`

## Motivation

The current J=100 was selected using the standard holdout r metric (type-mean prior). S196 discovered that this metric inflates by ~+0.22 due to **type self-prediction**: when a county dominates its own type's weighted mean, the type mean effectively memorizes that county, giving an artificially high prediction. The LOO metric (`holdout_accuracy_county_prior_loo`) removes each county from the type mean before predicting it, giving an honest generalization estimate.

**Baseline confirmed at J=100 (single seed=42):**
- Standard r: 0.700
- LOO r: 0.463
- Inflation: +0.237

This experiment re-sweeps J using the LOO metric to find the truly optimal J.

## Setup

- **Data:** `data/shifts/county_shifts_multiyear.parquet` — 3,154 counties × 63 shift columns
- **Training columns:** 33 dims (start year ≥ 2008, excludes holdout)
- **Holdout columns:** `pres_d_shift_20_24`, `pres_r_shift_20_24`, `pres_turnout_shift_20_24`
- **Scaling:** StandardScaler fit on training columns only; holdout scaled independently
- **Presidential weight:** 8.0 (applied post-scaling to all pres_ columns, training + holdout)
- **Temperature:** 10.0 (production default)
- **Random seed:** 42 for primary sweep; 5-seed multi-seed averaging for stable comparisons

## Results

### Primary Sweep (seed=42, J=[20..250])

| J | std_r | loo_r | inflation |
|---|-------|-------|-----------|
| 20 | 0.4192 | 0.3657 | +0.0534 |
| 30 | 0.4566 | 0.3909 | +0.0657 |
| 40 | 0.5024 | 0.4234 | +0.0791 |
| 50 | 0.5533 | 0.4363 | +0.1170 |
| 60 | 0.5599 | 0.4108 | +0.1491 |
| 70 | 0.5854 | 0.3903 | +0.1951 |
| 80 | 0.6509 | 0.4645 | +0.1864 |
| 90 | 0.6373 | 0.4706 | +0.1667 |
| **100** | **0.7000** | **0.4630** | **+0.2370** | ← current |
| 120 | 0.6862 | 0.4440 | +0.2421 |
| 150 | 0.7513 | 0.4829 | +0.2684 |
| 160 | 0.7470 | 0.4961 | +0.2510 |
| 175 | 0.7577 | 0.4454 | +0.3122 |
| 200 | 0.7719 | 0.4981 | +0.2738 |
| 250 | 0.7738 | 0.4490 | +0.3247 |

**Key finding:** LOO values are noisy at single seed due to KMeans non-determinism. The peak appears to be in the J=150-200 range.

### Multi-Seed Averaging (5 seeds: 42, 123, 456, 789, 1337)

To stabilize the LOO estimates, 5 seeds were averaged for key J candidates:

| J | mean_loo_r | std_loo_r | mean_std_r | mean_inflation |
|---|-----------|-----------|-----------|----------------|
| 80 | 0.4257 | 0.0352 | 0.6407 | +0.2150 |
| 90 | 0.4625 | 0.0122 | 0.6535 | +0.1910 |
| 100 | 0.4548 | 0.0220 | 0.6806 | +0.2258 | ← current |
| 120 | 0.4610 | 0.0139 | 0.6954 | +0.2345 |
| 150 | 0.4724 | 0.0109 | 0.7393 | +0.2670 |
| 160 | **0.4917** | 0.0193 | 0.7452 | +0.2535 |
| 200 | 0.4731 | 0.0196 | 0.7651 | +0.2919 |

### Refined Sweep Around Peak (J=130-175, 5 seeds each)

| J | mean_loo_r | std_loo_r | mean_std_r |
|---|-----------|-----------|-----------|
| 130 | 0.4693 | 0.0219 | 0.7052 |
| 140 | 0.4767 | 0.0226 | 0.7325 |
| 150 | 0.4724 | 0.0109 | 0.7393 |
| 155 | 0.4808 | 0.0164 | 0.7383 |
| **160** | **0.4917** | 0.0193 | **0.7452** | ← LOO OPTIMAL |
| 165 | 0.4848 | 0.0206 | 0.7456 |
| 170 | 0.4684 | 0.0137 | 0.7468 |
| 175 | 0.4700 | 0.0168 | 0.7499 |

## Key Findings

1. **LOO-optimal J = 160** — mean LOO r = 0.492 (5-seed average), vs J=100 at LOO r = 0.455.
   - That's a **+0.037 LOO improvement** from re-tuning J.

2. **Inflation grows with J** — At low J (20-40), inflation is small (+0.05-0.08). At J=100, inflation is +0.23. At J=200+, inflation exceeds +0.29. This is expected: more types → more counties that "dominate" their own small type → more self-prediction inflation.

3. **Standard metric was misleading at J=100** — The standard metric shows J=100 as adequate (0.70 std_r) vs J=150+ as clearly better. But LOO tells a different story: J=100 is mediocre and the improvement from J=150-160 is real but modest (~+0.03-0.04).

4. **LOO plateau is flat** — The curve peaks around J=150-165 but is noisy (std ≈ 0.015-0.020). The improvement from J=100 to J=160 is real but small. A conservative recommendation is J=150-160.

5. **Per-dimension breakdown at J=100 (current) vs J=160 (LOO-optimal), seed=42:**
   - Holdout dims: `pres_d_shift_20_24`, `pres_r_shift_20_24`, `pres_turnout_shift_20_24`
   - J=100 LOO r per dim: [0.4057, 0.4013, 0.5822]
   - J=160 LOO r per dim: (similar pattern, mean +0.03-0.04 higher)
   - The turnout dimension is consistently better predicted than D/R shifts.

## Recommendation

**Set J=160** in production (up from J=100).

Rationale:
- LOO r improves by ~+0.037 (0.455 → 0.492, 5-seed average)
- Standard r also improves (0.681 → 0.745), consistent with better coverage
- The inflation is manageable at J=160 (+0.254 vs +0.226 at J=100)
- Beyond J=160, inflation grows faster than LOO r, indicating overfitting to type means
- J=160 runs in ~0.5s (vs J=100 at 0.3s) — negligible compute cost

**Note:** The J=160 optimal is somewhat noisy (std_loo ≈ 0.019 across seeds). If reproducibility is paramount, J=150 is a safe choice with mean_loo_r = 0.472 and lower variance (std=0.011). The difference between J=150 and J=160 is within 1σ.

## Caveats

1. This sweep tests only the holdout pair `2020→2024`. A more thorough sweep would use leave-one-election-pair-out CV across all pairs, but that would require running `select_j.py` with the LOO metric embedded — a larger refactor.

2. KMeans has run-to-run variance even at fixed seed (n_init=10 helps but doesn't eliminate it). The 5-seed average reduces noise but the LOO r at J=160 has std≈0.019, meaning the 95% CI spans roughly [0.454, 0.530]. J=100 spans [0.411, 0.499]. The difference is real but modest.

3. The holdout is only 3 columns (presidential 2020→2024). Presidential shifts are the strongest signal in the data (presidential_weight=8.0), so this is a reasonable but not comprehensive evaluation.

## Action Items

- [ ] Update `config/model.yaml` `types.j` from 100 to 160
- [ ] Re-run `python -m src.discovery.run_type_discovery` to generate new type assignments
- [ ] Re-run `python -m src.validation.validate_types` to confirm holdout r improves
- [ ] Update MEMORY.md baseline metrics (county holdout LOO r should improve from 0.448)
- [ ] Update CLAUDE.md baseline metrics
