# Design Spec: Tract-Primary Architecture + Voter Behavior Layer

**Date**: 2026-03-27
**Status**: APPROVED — awaiting implementation plan
**Scope**: Full architectural migration from county-primary to tract-primary model, plus new voter behavior layer decomposing turnout and choice effects by cycle type.

## Motivation

The current model treats all elections as interchangeable. Ridge priors are trained on 2024 presidential outcomes. Type discovery mixes presidential, governor, and Senate shifts as undifferentiated dimensions. There is no concept of "this is a midterm" — so when predicting 2026 governor/Senate races, the model uses a presidential-shaped electorate as its baseline. This systematically overestimates Republican performance in off-cycle years because R-leaning types turn out at lower rates when a presidential race isn't on the ballot.

The fix is not a polling adjustment or a flat correction. It is a structural decomposition: communities are permanent (discovered once from all elections), but their *activation pattern* varies by electoral context. A voter behavior layer learns, per type, the turnout and choice differentials between presidential and off-cycle years. This is the difference between "modeling elections" and "modeling community behavior."

Simultaneously, the county layer (3,154 units) is being retired in favor of tracts (~81K units). Tracts are more homogeneous, yield purer type assignments, and DRA block-level data provides complete coverage (all 51 states, 2008-2024) at this granularity. There is no remaining data constraint that requires county-level aggregation.

## Architecture

### Layer 1: Type Discovery (run once)

**Input**: Tract-level shift vectors computed from DRA block data aggregated to tracts.

**Shift dimensions**: Presidential shifts (e.g., 2008→2012, 2012→2016, 2016→2020, 2020→2024) AND off-cycle shifts (e.g., 2018 gov→2022 gov, 2016 Senate→2022 Senate) as **separate columns**. Including off-cycle shifts in discovery lets the model distinguish communities that behave identically in presidential years but differently off-cycle (e.g., military communities with stable turnout vs college towns with variable turnout).

**Off-cycle shift preprocessing**: State-centered before clustering. This removes the state-level mean so within-state variation is preserved but state-specific candidate effects don't pollute cross-state type discovery. Presidential shifts remain raw (they carry cross-state signal).

> **Modeling assumption**: State-centering is a proxy for candidate effect removal. A proper candidate effect model (decomposing district baseline + national environment + candidate-specific draw) is planned but not yet implemented. When candidate effects are modeled explicitly, state-centering of off-cycle shifts should be replaced with candidate-effect-residualized shifts. Document this dependency in the codebase.

**Algorithm**: KMeans with StandardScaler normalization. Presidential weight parameter (currently 8.0) applied post-standardization. J selected via leave-one-pair-out CV (current production: J=100).

**Output**: Soft membership scores per tract (temperature-scaled inverse distance, T=10). These are the permanent structural assignments. Everything downstream consumes them; nothing re-derives them.

**Population weighting**: Tracts with <500 voters excluded from discovery (current threshold; ~3,300 tracts dropped).

### Layer 2: Voter Behavior Layer (NEW)

**Purpose**: For each type, learn how that community expresses itself differently in presidential vs off-cycle elections. Decomposes into two parameters:

#### Parameter 1: Turnout Ratio (τ)

```
τ_type = weighted_mean(off-cycle_total_votes / presidential_total_votes)
```

Weighted by type membership score across tracts. Tells us: "Type 47 retains 72% of its presidential turnout in off-cycle years."

**Data source**: DRA block data aggregated to tracts. Total votes columns (`E_XX_PRES_Total`, `E_XX_GOV_Total`, `E_XX_SEN_Total`) provide raw turnout counts. Average across available off-cycle elections (2010-2022) vs presidential elections (2008-2024) for stability.

**Cycle type definition**: Binary — presidential vs off-cycle. Turnout is ballot-level (you show up and vote on everything presented), so there is no meaningful distinction between "showed up for the governor race" vs "showed up for the Senate race" in the same election year. A voter who appears in a 2018 election voted on both the governor and Senate races on their ballot.

#### Parameter 2: Residual Choice Shift (δ)

```
δ_type = observed_off-cycle_dem_share - expected_dem_share_from_turnout_reweighting
```

After accounting for which types drop off (via τ), compute what the expected Dem share *would be* if vote preferences stayed constant but only turnout changed. The residual between that expected share and the actual observed off-cycle Dem share is δ — the genuine choice shift that turnout alone cannot explain.

**Computation**:
1. For each tract in each off-cycle election, compute expected Dem share by reweighting the presidential-year type composition by τ values.
2. Compare to actual observed off-cycle Dem share.
3. Weight residuals by type membership to get per-type δ.

If δ ≈ 0 for most types, turnout composition is the full story. If δ is significantly nonzero, there are real preference shifts in off-cycle contexts (e.g., less nationalized voting, more local-issue-driven).

#### Training Data

All DRA block data across all available states and election cycles. The behavior layer trains on the *results* (levels), not the shifts (changes between elections of the same type). This answers "what does this community do in a midterm" rather than "how does it change between midterms."

Presidential elections used: 2008, 2012, 2016, 2020, 2024 (5 cycles).
Off-cycle elections used: All available governor and Senate races from 2010-2022 (varies by state; typically 2-4 per state).

### Layer 3: Covariance (existing, recomputed at tract level)

Ledoit-Wolf regularized covariance on observed tract-level presidential shifts. Same methodology as current county model, just computed from tract data.

### Layer 4: Prediction (existing, modified)

For a 2026 race:

1. **Base prediction**: Ridge model trained on tract-level features → tract-level priors (analogous to current county Ridge priors, but at tract granularity).

2. **Behavior adjustment**: Apply τ and δ to adjust priors for the cycle type:
   - Reweight type composition of each tract's electorate by τ (which types show up less in off-cycle)
   - Apply δ choice shift per type
   - This produces cycle-adjusted tract-level priors

3. **Bayesian poll update**: Polls propagate through type covariance Σ, exactly as today. The poll update operates on the cycle-adjusted priors, not the raw presidential-year priors.

4. **Aggregation**: Tract predictions → state predictions via vote-weighted sum.

### Frontend Changes

- **Default view**: Tract map (bubble-dissolved community polygons). No toggle needed.
- **County layer**: Removed entirely. County choropleth, county detail pages, and county-level API endpoints retired.
- **Forecast tab**: Operates on tract-level predictions. State-level display uses vote-weighted aggregation from tracts.
- **County pages**: Redirect or remove. If retained, they aggregate tract predictions within the county boundary but are not a primary model output.

## Data Pipeline

```
DRA block CSVs (all 51 states, v06/v07)
  → Aggregate to tracts (GEOID[:11] groupby sum)
  → Tract shift vectors (presidential + state-centered off-cycle, separate dims)
  → KMeans type discovery (Layer 1, run once)
  → Behavior layer training (Layer 2, τ + δ per type)
  → Covariance estimation (Layer 3, Ledoit-Wolf on presidential shifts)
  → Ridge model training (tract-level features → priors)
  → 2026 predictions (Layer 4: priors + behavior + polls)
  → DuckDB + API + frontend
```

## DRA Data Coverage

**Block-level data on disk**: All 51 states/DC. 40 states at v07, 11 at v06 (AL, AR, CT, DE, IA, ID, MI, ND, OR, SD, WY). Both versions include 2024 presidential data.

**Race coverage**: Every state has presidential (2008-2024), governor (1-4 cycles), and Senate (2-5 cycles) results at block level with total votes + Dem + Rep columns.

**Tract mapping**: Census block GEOID[:11] maps directly to tract GEOID. No areal interpolation needed.

## What This Replaces

| Component | Current (county-primary) | New (tract-primary) |
|-----------|-------------------------|---------------------|
| Unit of analysis | 3,154 counties | ~81K tracts |
| Data source | MEDSL county + Algara/Amlani | DRA block→tract |
| Type discovery inputs | 33 dims (pres + gov + Senate shifts, all mixed) | Presidential shifts + state-centered off-cycle shifts, separate dims |
| Cycle awareness | None | Behavior layer (τ + δ) |
| Ridge target | 2024 presidential Dem share | 2024 presidential Dem share (behavior layer adjusts downstream for off-cycle) |
| Covariance source | County-level observed shifts | Tract-level observed shifts |
| Frontend default | County choropleth | Tract community polygons |
| Governor/Senate role | Discovery input dimensions | Behavior layer training data only |

## Key Design Decisions

1. **Communities discovered once from all election types** — presidential + off-cycle shifts as separate dimensions. Types are structural and permanent.
2. **Behavior layer decomposes into turnout (τ) + residual choice shift (δ)** — two parameters per type, learned from historical data.
3. **Binary cycle type: presidential vs off-cycle** — turnout is ballot-level, not race-level. No distinction between governor and Senate in the same election year.
4. **Governor/Senate results consumed only by behavior layer** — they are training data for τ and δ, not inputs to type discovery.
5. **State-centering of off-cycle shifts is a proxy for candidate effect removal** — documented as modeling assumption, to be replaced by explicit candidate effect model in future.
6. **County layer eliminated** — tracts are the sole unit of analysis. County aggregation is a presentation convenience, not a model output.

## Validation Strategy

1. **Type discovery**: Leave-one-pair-out CV on tract-level shifts. Beat current county holdout r (0.698) or understand why tract is lower and whether the behavior layer compensates.
2. **Behavior layer**: Backtest on held-out off-cycle elections. Hold out 2022 governor/Senate, train τ and δ on 2010-2018, predict 2022. Compare to naive (presidential-prior-only) baseline.
3. **End-to-end**: For each 2026 race, does the behavior-adjusted prediction match polling better than the unadjusted prediction? GA Senate is the canary — the model should show it as competitive without poll input.
4. **Regression**: Full test suite must pass. API contract tests must pass. No reduction in test count.

## Risks

- **Tract model has historically underperformed county** (holdout r 0.632 vs 0.698). The current tract model uses combined electoral+demographic features in 31 dims. The new architecture uses DRA data (richer electoral coverage, 2008-2024) which may close this gap. If it doesn't, the behavior layer's per-type adjustments may compensate by correcting systematic cycle-type bias.
- **Compute**: 81K tracts × 100 types is 25x more data than counties. Bayesian update, Ridge training, and API responses will be slower. May need caching or precomputation strategies.
- **DRA v06/v07 schema differences**: 11 states are at v06. Column naming may differ. Need a normalization step in the ingestion pipeline.
- **Tract boundary changes**: 2020 Census redrew tract boundaries. DRA uses 2020 geography for all years (block-level allocation handles redistricting). Verify this assumption.

## Non-Goals

- Candidate effect modeling (future work; state-centering is the current proxy)
- House district forecasts (depends on tract→district crosswalk, separate effort)
- Multi-cycle behavior parameters (e.g., "this is a first-term-midterm vs second-term-midterm") — binary is sufficient for now
- Precinct-level modeling (blocks aggregate to tracts; no need to go finer)
