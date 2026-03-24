# Type Stability: Sub-Window Analysis (P4.2)

**Session**: S175
**Date**: 2026-03-23
**Question**: Do the same electoral community types emerge from 2008-2016 data as from 2016-2024 data?

---

## Setup

| Parameter | Value |
|-----------|-------|
| Algorithm | KMeans |
| J (types) | 43 |
| Counties | 293 |
| Presidential weight | 2.5x |
| Gov/Senate | state-centered |
| Temperature (T) | 10.0 |
| Sub-window split | year 2016 |
| Min training year | 2008 |
| Holdout | `pres_*_shift_20_24` |
| Bootstrap seeds | 50 |

### Early Sub-Window (2008-2015): 24 dimensions

Columns: pres_d_shift_08_12, pres_r_shift_08_12, pres_turnout_shift_08_12, pres_d_shift_12_16, pres_r_shift_12_16, pres_turnout_shift_12_16 ... (24 total)

### Late Sub-Window (2016+): 9 dimensions

Columns: pres_d_shift_16_20, pres_r_shift_16_20, pres_turnout_shift_16_20, gov_d_shift_18_22, gov_r_shift_18_22, gov_turnout_shift_18_22, sen_d_shift_16_22, sen_r_shift_16_22, sen_turnout_shift_16_22

### Full Window: 33 dimensions (production baseline)

---

## Results

### Cross-Window Agreement

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Adjusted Rand Index (ARI) | **0.1131** | 1.0 = identical, 0.0 = random |
| Normalized Mutual Information (NMI) | **0.5822** | 1.0 = identical partitions |
| County stability (Hungarian-matched) | **32.4%** | % counties in same type after optimal relabeling |

### Seed Stability Baseline (Full Window, 50 seeds)

| Metric | Value |
|--------|-------|
| Mean pairwise ARI | **0.4403** |
| Std pairwise ARI | 0.0436 |
| Pairs evaluated | 1225 |

> The seed ARI measures how consistently KMeans finds the same types when only
> the random initialization changes. This is the ceiling for what "perfectly
> stable" types would look like under this algorithm.
>
> **Cross-window ARI / Seed ARI = 0.26x**
> (1.0 = sub-window types are as stable as seed variation alone)

### Holdout r (2020→2024 Presidential Shifts)

| Model | Holdout r | Notes |
|-------|-----------|-------|
| Early window (2008-2015) | **0.8041** | Older data predicts modern shifts |
| Late window (2016+) | **0.8378** | More recent data |
| Full window (production) | **0.8541** | Combined 33-dim model |

---

## Type Size Distributions

### Early Window (24 dims) — Top 15 types by size:

```
  type_00 (  14 counties): ##############################
  type_01 (  13 counties): ###########################
  type_02 (  13 counties): ###########################
  type_03 (  12 counties): #########################
  type_04 (  12 counties): #########################
  type_05 (  11 counties): #######################
  type_06 (  11 counties): #######################
  type_07 (  11 counties): #######################
  type_08 (  11 counties): #######################
  type_09 (  10 counties): #####################
  type_10 (  10 counties): #####################
  type_11 (  10 counties): #####################
  type_12 (   9 counties): ###################
  type_13 (   9 counties): ###################
  type_14 (   9 counties): ###################
  ... (28 more types)
```

### Late Window (9 dims) — Top 15 types by size:

```
  type_00 (  29 counties): ##############################
  type_01 (  27 counties): ###########################
  type_02 (  16 counties): ################
  type_03 (  14 counties): ##############
  type_04 (  12 counties): ############
  type_05 (  12 counties): ############
  type_06 (  11 counties): ###########
  type_07 (  11 counties): ###########
  type_08 (  11 counties): ###########
  type_09 (  10 counties): ##########
  type_10 (   9 counties): #########
  type_11 (   8 counties): ########
  type_12 (   8 counties): ########
  type_13 (   8 counties): ########
  type_14 (   7 counties): #######
  ... (28 more types)
```

---

## Interpretation

UNSTABLE: Cross-window ARI (0.113) approaches chance. Types appear to be period-specific artifacts rather than durable structures.

### What This Means for Forecasting

- **Cross-window ARI = 0.113** vs **seed ARI = 0.440**
  (ratio: 0.26x)

- Holdout r order: early (0.804) vs late (0.838) vs full (0.854)
  Late > Early as expected: recent data is more predictive of recent shifts.

- **Implication for 2026 predictions**: Types show significant period drift. Consider using time-weighted clustering (see experiment_temporal_weighting.py) for 2026 predictions.

---

## Methodology Notes

- ARI and NMI are computed on hard cluster assignments (argmax of soft membership).
- County stability uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
  to find the optimal bijective mapping between early-window type labels and
  late-window type labels before computing the match rate.
- Bootstrap seed stability uses pairwise ARI over 50 runs × 49 / 2 = 1225 pairs.
- Holdout r is mean Pearson r across the 3 holdout columns (D, R, turnout shifts).
- All clustering uses the production weighting: pres_* columns × 2.5, gov/sen state-centered.
