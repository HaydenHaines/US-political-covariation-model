# Governor Forecast Backtest — 2022 Actuals vs 2026 Model (S492)

**Date:** 2026-04-07  
**Branch:** feat/governor-backtest  
**Purpose:** Validate whether the S492 behavior layer (δ adjustment) improves
governor predictions relative to a naive presidential baseline.

## Methodology

This is an **indirect backtest**: the model targets 2026, not 2022.  The model's
structural priors come from historical county-level shift patterns, so comparing
2026 predictions to 2022 actuals tests whether the model's understanding of state
partisan structure is correct — not its ability to predict a specific cycle.

A true holdout (train on ≤2020, predict 2022) would be more rigorous but requires
a full retrain.  This comparison is a quick diagnostic.

**Three baselines compared against 2022 actual two-party Dem governor share:**

1. **Model (current)** — 2026 county predictions (with δ behavior adjustment),
   aggregated vote-weighted to state, plus +4pp incumbency heuristic for
   non-open seats (matching API logic in `api/routers/governor/_helpers.py`).
2. **2024 Presidential baseline** — 2024 presidential state-level two-party
   Dem share.  Simplest possible structural baseline.
3. **Model without δ** — same as (1) but with the per-county δ shift reversed
   before aggregation.  Tests whether the behavior layer actually helped.

## Summary Metrics

| Baseline | r | RMSE | Bias | Dir |
|---|---|---|---|---|
| Model (current, with δ+inc) | 0.790 | 9.3pp | +2.4pp | 27/34 |
| 2024 Presidential baseline | 0.770 | 8.8pp | +1.7pp | 25/34 |
| Model without δ (with inc) | 0.801 | 9.8pp | +3.2pp | 29/34 |

- **r** = Pearson correlation with 2022 actuals
- **RMSE** = root mean squared error
- **Bias** = mean(predicted − actual): positive = over-predicts D
- **Dir** = states where D/R winner correctly predicted

## δ Behavior Layer Impact

- Mean δ shift applied: -0.81pp
- Std: 2.97pp
- Range: [-10.72, +3.79]pp

## Per-State Comparison

| State | Actual | Model | Pres | NoDelta | Err(M) | Err(P) | Err(ND) |
|---|---|---|---|---|---|---|---|
| WY | 0.0% | 6.8% | 25.8% | 3.1% | +6.8pp | +25.8pp | +3.1pp |
| AL | 17.8% | 34.9% | 34.1% | 38.1% | +17.0pp | +16.3pp | +20.2pp |
| ID | 28.9% | 27.7% | 30.4% | 26.6% | -1.2pp | +1.5pp | -2.3pp |
| AK | 30.7% | 39.0% | 41.4% | 43.9% | +8.4pp | +10.8pp | +13.3pp |
| AR | 34.5% | 35.0% | 33.6% | 35.0% | +0.5pp | -1.0pp | +0.5pp |
| SD | 36.2% | 33.4% | 34.2% | 36.8% | -2.8pp | -2.0pp | +0.6pp |
| OH | 37.5% | 52.7% | 43.9% | 51.4% | +15.3pp | +6.5pp | +14.0pp |
| OK | 37.7% | 33.4% | 31.9% | 30.0% | -4.3pp | -5.8pp | -7.7pp |
| NE | 37.8% | 16.1% | 39.1% | 17.1% | -21.7pp | +1.3pp | -20.7pp |
| CA | 38.4% | 53.3% | 58.5% | 53.3% | +14.9pp | +20.1pp | +14.9pp |
| FL | 40.2% | 50.9% | 43.0% | 48.0% | +10.6pp | +2.8pp | +7.7pp |
| VT | 40.5% | 37.6% | 63.2% | 36.9% | -2.9pp | +22.7pp | -3.6pp |
| IA | 40.5% | 47.0% | 42.3% | 47.3% | +6.5pp | +1.7pp | +6.8pp |
| SC | 41.2% | 45.1% | 40.4% | 46.2% | +3.9pp | -0.8pp | +5.0pp |
| NH | 42.1% | 40.0% | 50.7% | 38.5% | -2.1pp | +8.5pp | -3.6pp |
| TX | 44.6% | 47.9% | 42.5% | 48.4% | +3.3pp | -2.1pp | +3.8pp |
| NY | 46.4% | 74.3% | 56.2% | 73.9% | +27.9pp | +9.9pp | +27.6pp |
| GA | 46.8% | 50.9% | 48.5% | 49.1% | +4.1pp | +1.7pp | +2.3pp |
| NV | 48.7% | 45.8% | 47.5% | 46.8% | -2.9pp | -1.2pp | -1.9pp |
| KS | 51.1% | 48.1% | 41.0% | 47.5% | -3.0pp | -10.1pp | -3.6pp |
| WI | 51.7% | 57.7% | 48.7% | 57.9% | +6.0pp | -3.0pp | +6.1pp |
| OR | 51.9% | 52.2% | 55.3% | 53.0% | +0.3pp | +3.4pp | +1.1pp |
| AZ | 53.2% | 58.8% | 46.4% | 60.9% | +5.6pp | -6.8pp | +7.8pp |
| MN | 54.0% | 66.0% | 50.9% | 65.2% | +12.0pp | -3.0pp | +11.2pp |
| MI | 55.2% | 52.6% | 48.3% | 55.9% | -2.6pp | -6.9pp | +0.7pp |
| CT | 55.8% | 53.5% | 56.4% | 52.7% | -2.4pp | +0.6pp | -3.1pp |
| IL | 56.4% | 46.4% | 54.4% | 45.7% | -10.1pp | -2.1pp | -10.7pp |
| ME | 56.7% | 52.2% | 52.2% | 52.9% | -4.5pp | -4.5pp | -3.7pp |
| RI | 57.1% | 55.0% | 55.4% | 56.3% | -2.1pp | -1.7pp | -0.8pp |
| PA | 57.5% | 67.1% | 48.7% | 73.9% | +9.6pp | -8.9pp | +16.4pp |
| CO | 59.9% | 61.7% | 54.2% | 63.0% | +1.8pp | -5.7pp | +3.1pp |
| HI | 63.2% | 57.1% | 60.6% | 55.7% | -6.1pp | -2.6pp | -7.5pp |
| MA | 64.8% | 69.9% | 61.2% | 75.9% | +5.1pp | -3.6pp | +11.0pp |
| MD | 66.8% | 58.4% | 62.6% | 69.1% | -8.4pp | -4.1pp | +2.4pp |

## Notable Errors (|error| > 10pp)

- **NY**: actual=46.4%, model=74.3%, error=+27.9pp
- **NE**: actual=37.8%, model=16.1%, error=-21.7pp
- **AL**: actual=17.8%, model=34.9%, error=+17.0pp
- **OH**: actual=37.5%, model=52.7%, error=+15.3pp
- **CA**: actual=38.4%, model=53.3%, error=+14.9pp
- **MN**: actual=54.0%, model=66.0%, error=+12.0pp
- **FL**: actual=40.2%, model=50.9%, error=+10.6pp
- **IL**: actual=56.4%, model=46.4%, error=-10.1pp

## Analysis

**Best baseline by correlation:** NoDelta (r=0.801)

**δ behavior layer improved accuracy for:** 20/34 states

**Key observations:**

- The model uses presidential-trained Ridge priors with no cycle-type awareness.
  This means it structurally predicts governor races with a presidential electorate,
  which tends to amplify national environment signals and miss incumbency dynamics.
- Positive bias means the model over-predicts the Democratic share relative to 2022 actuals.
  2022 was a good D cycle for governors; 2024 presidential was R-tilted. A model using
  2024 presidential priors should show systematic D under-prediction, not over-prediction.
- The incumbency heuristic (+4pp toward the incumbent party) partially corrects for
  the cycle-type mismatch but does not address structural range compression.

## Limitations

1. **Indirect comparison**: 2026 model vs 2022 actuals. Not a true holdout.
2. **Incumbency mismatch**: The 2026 incumbency map differs from 2022. For example,
   MD (Hogan R→Moore D) would show different model corrections in 2022 vs 2026.
3. **National environment**: 2022 had a particular national environment (inflation,
   Biden midterms) that the 2026 model does not attempt to replicate for 2022.
4. **Sample size**: 34 states (2022 data coverage) out of 36 2026 governor races.

## Next Steps

- Implement cycle-type awareness in the prediction pipeline (governor vs presidential)
- True holdout: retrain on ≤2020 data, predict 2022 governor results
- Expand δ estimation to weight governor-specific off-cycle shifts more heavily
- Consider separate Ridge priors for governor vs presidential context
