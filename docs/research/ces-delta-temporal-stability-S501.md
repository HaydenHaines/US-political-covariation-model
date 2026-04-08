# CES δ Temporal Stability Analysis (S501)

**Date:** 2026-04-08
**Purpose:** Determine whether governor δ (governor Dem share - presidential Dem share
per type) is a durable type property or cycle-specific noise.

## Method

Computed per-type δ for each midterm governor cycle using CES validated voters:
- δ = governor Dem two-party share - nearest presidential Dem share for same type
- Minimum 30 respondents per type per race-year
- Paired: 2006→2008, 2010→2008, 2014→2012, 2018→2016, 2022→2020

## Per-Cycle δ Statistics

| Year | Pres Ref | Types | Mean δ | Std δ |
|------|----------|-------|--------|-------|
| 2006 | 2008 | 60 | +8.06pp | 11.90pp |
| 2010 | 2008 | 62 | -1.61pp | 8.91pp |
| 2014 | 2012 | 62 | +0.06pp | 8.34pp |
| 2018 | 2016 | 64 | +1.09pp | 8.03pp |
| 2022 | 2020 | 61 | -1.59pp | 5.95pp |

Note: 2006 has an unusually large D-favorable δ (mean +8.06pp), likely reflecting
the D wave year compared to the 2008 presidential baseline which was also D-favorable.

## Cross-Year δ Correlations

|      | 2006  | 2010  | 2014  | 2018  | 2022  |
|------|-------|-------|-------|-------|-------|
| 2006 | —     | 0.364 | 0.195 | 0.023 | -0.058 |
| 2010 | 0.364 | —     | 0.526 | -0.245 | -0.355 |
| 2014 | 0.195 | 0.526 | —     | 0.063 | 0.047 |
| 2018 | 0.023 | -0.245 | 0.063 | —     | 0.347 |
| 2022 | -0.058 | -0.355 | 0.047 | 0.347 | —     |

**Mean cross-year stability: r = 0.091**
**Range: [-0.355, 0.526]**

## Key Finding

**Governor δ is almost entirely cycle-specific noise, not durable type behavior.**

- The mean cross-year correlation (r=0.091) is indistinguishable from zero.
- Adjacent cycles show modest correlations (2010-2014: r=0.526, 2018-2022: r=0.347)
  but these decay rapidly with distance, suggesting they capture slowly-evolving
  candidate environment rather than permanent type structure.
- Distant cycles are uncorrelated or negatively correlated (2010-2022: r=-0.355).

## Implications for the Behavior Layer

1. **Pooled δ (what CES backtest used) is noise averaging.** Pooling δ across
   5 cycles with r≈0 between them produces values that capture no cycle-specific
   signal and have attenuated any true temporal signal.

2. **δ is not a type property.** Unlike τ (which likely reflects durable demographic
   and engagement patterns), δ appears to be driven by cycle-specific factors:
   individual candidates, local issues, campaign quality, national environment.

3. **The behavior layer should model τ only, not δ.** τ captures the structural
   difference in who shows up for off-cycle elections. δ captures who they vote
   for, which changes every cycle based on candidate-specific factors.

4. **Candidate effects (the sabermetrics module) are the right home for δ.**
   Once enough polling data arrives, per-race candidate effects will naturally
   capture the cycle-specific δ that the structural model cannot.

## Recommendation

- **Disable δ in predictions entirely.** Not just in the county pipeline (already
  disabled) but also in the tract pipeline. τ-only behavior adjustment may still
  be worth testing.
- **Reclassify δ research as complete.** The behavior layer spec's δ component
  is answered: δ is not a type-level property. It's a race-level property.
- **Direct future effort toward poll integration.** The cycle-specific signal
  that δ was supposed to capture is better captured by actual polls.

---

## Appendix: τ Temporal Stability (for comparison)

τ (turnout engagement ratio) was analyzed with the same methodology.
CES-derived τ proxy = governor respondent count / presidential respondent count
per type, normalized by overall ratio.

### Cross-Year τ Correlations

|      | 2006  | 2010  | 2014  | 2018  | 2022  |
|------|-------|-------|-------|-------|-------|
| 2006 | —     | 0.752 | 0.441 | 0.708 | 0.578 |
| 2010 | 0.752 | —     | 0.607 | 0.723 | 0.507 |
| 2014 | 0.441 | 0.607 | —     | 0.526 | 0.521 |
| 2018 | 0.708 | 0.723 | 0.526 | —     | 0.628 |
| 2022 | 0.578 | 0.507 | 0.521 | 0.628 | —     |

**Mean cross-year τ stability: r = 0.599**
**Range: [0.441, 0.752]**

### Comparison

| Parameter | Mean Stability | Interpretation |
|-----------|---------------|----------------|
| τ (turnout) | r = 0.599 | Durable type property — worth modeling |
| δ (choice) | r = 0.091 | Cycle noise — not worth modeling at type level |

τ is ~7x more stable than δ across election cycles. This confirms that turnout
engagement patterns (who shows up for off-cycle elections) are a genuine structural
property of electoral types, while choice shifts (who they vote for when they
show up) are driven by cycle-specific candidates and issues.
