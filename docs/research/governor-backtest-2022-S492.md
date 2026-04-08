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
| Model (current, with δ+inc) | 0.715 | 10.4pp | +3.4pp | 26/34 |
| 2024 Presidential baseline | 0.770 | 8.8pp | +1.7pp | 25/34 |
| Model without δ (with inc) | 0.729 | 10.5pp | +4.0pp | 27/34 |

- **r** = Pearson correlation with 2022 actuals
- **RMSE** = root mean squared error
- **Bias** = mean(predicted − actual): positive = over-predicts D
- **Dir** = states where D/R winner correctly predicted

## δ Behavior Layer Impact

- Mean δ shift applied: -0.64pp
- Std: 2.78pp
- Range: [-10.25, +3.39]pp

## Per-State Comparison

| State | Actual | Model | Pres | NoDelta | Err(M) | Err(P) | Err(ND) |
|---|---|---|---|---|---|---|---|
| WY | 0.0% | 24.4% | 25.8% | 25.1% | +24.4pp | +25.8pp | +25.1pp |
| AL | 17.8% | 34.7% | 34.1% | 37.2% | +16.9pp | +16.3pp | +19.4pp |
| ID | 28.9% | 34.8% | 30.4% | 32.8% | +5.9pp | +1.5pp | +3.9pp |
| AK | 30.7% | 34.5% | 41.4% | 39.7% | +3.8pp | +10.8pp | +9.1pp |
| AR | 34.5% | 34.2% | 33.6% | 34.5% | -0.3pp | -1.0pp | +0.0pp |
| SD | 36.2% | 32.3% | 34.2% | 33.6% | -3.9pp | -2.0pp | -2.6pp |
| OH | 37.5% | 54.2% | 43.9% | 54.9% | +16.7pp | +6.5pp | +17.4pp |
| OK | 37.7% | 28.4% | 31.9% | 28.9% | -9.3pp | -5.8pp | -8.8pp |
| NE | 37.8% | 25.6% | 39.1% | 26.5% | -12.2pp | +1.3pp | -11.3pp |
| CA | 38.4% | 59.4% | 58.5% | 59.6% | +21.0pp | +20.1pp | +21.2pp |
| FL | 40.2% | 52.4% | 43.0% | 49.1% | +12.1pp | +2.8pp | +8.9pp |
| VT | 40.5% | 43.1% | 63.2% | 41.8% | +2.6pp | +22.7pp | +1.3pp |
| IA | 40.5% | 47.0% | 42.3% | 45.3% | +6.5pp | +1.7pp | +4.8pp |
| SC | 41.2% | 44.3% | 40.4% | 46.5% | +3.1pp | -0.8pp | +5.3pp |
| NH | 42.1% | 40.0% | 50.7% | 40.1% | -2.1pp | +8.5pp | -2.0pp |
| TX | 44.6% | 48.3% | 42.5% | 47.5% | +3.7pp | -2.1pp | +2.9pp |
| NY | 46.4% | 70.6% | 56.2% | 69.1% | +24.2pp | +9.9pp | +22.8pp |
| GA | 46.8% | 52.8% | 48.5% | 50.8% | +6.0pp | +1.7pp | +4.0pp |
| NV | 48.7% | 46.3% | 47.5% | 47.7% | -2.3pp | -1.2pp | -0.9pp |
| KS | 51.1% | 41.8% | 41.0% | 41.7% | -9.4pp | -10.1pp | -9.5pp |
| WI | 51.7% | 57.9% | 48.7% | 58.6% | +6.2pp | -3.0pp | +6.8pp |
| OR | 51.9% | 55.6% | 55.3% | 54.7% | +3.7pp | +3.4pp | +2.8pp |
| AZ | 53.2% | 59.5% | 46.4% | 61.6% | +6.3pp | -6.8pp | +8.4pp |
| MN | 54.0% | 66.7% | 50.9% | 65.7% | +12.8pp | -3.0pp | +11.7pp |
| MI | 55.2% | 54.2% | 48.3% | 56.9% | -1.0pp | -6.9pp | +1.7pp |
| CT | 55.8% | 52.7% | 56.4% | 51.5% | -3.2pp | +0.6pp | -4.3pp |
| IL | 56.4% | 46.7% | 54.4% | 47.3% | -9.7pp | -2.1pp | -9.1pp |
| ME | 56.7% | 52.2% | 52.2% | 48.8% | -4.5pp | -4.5pp | -7.9pp |
| RI | 57.1% | 54.4% | 55.4% | 55.9% | -2.7pp | -1.7pp | -1.2pp |
| PA | 57.5% | 69.2% | 48.7% | 76.3% | +11.7pp | -8.9pp | +18.8pp |
| CO | 59.9% | 60.7% | 54.2% | 59.0% | +0.8pp | -5.7pp | -0.9pp |
| HI | 63.2% | 62.4% | 60.6% | 61.1% | -0.8pp | -2.6pp | -2.0pp |
| MA | 64.8% | 69.9% | 61.2% | 73.1% | +5.1pp | -3.6pp | +8.2pp |
| MD | 66.8% | 49.4% | 62.6% | 59.7% | -17.3pp | -4.1pp | -7.1pp |

## Notable Errors (|error| > 10pp)

- **WY**: actual=0.0%, model=24.4%, error=+24.4pp
- **NY**: actual=46.4%, model=70.6%, error=+24.2pp
- **CA**: actual=38.4%, model=59.4%, error=+21.0pp
- **MD**: actual=66.8%, model=49.4%, error=-17.3pp
- **AL**: actual=17.8%, model=34.7%, error=+16.9pp
- **OH**: actual=37.5%, model=54.2%, error=+16.7pp
- **MN**: actual=54.0%, model=66.7%, error=+12.8pp
- **NE**: actual=37.8%, model=25.6%, error=-12.2pp
- **FL**: actual=40.2%, model=52.4%, error=+12.1pp
- **PA**: actual=57.5%, model=69.2%, error=+11.7pp

## Analysis

**Best baseline by correlation:** Presidential (r=0.770)

**δ behavior layer improved accuracy for:** 16/34 states

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
