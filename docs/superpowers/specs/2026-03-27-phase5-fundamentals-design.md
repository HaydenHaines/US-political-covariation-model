# Phase 5: Fundamentals Model — Design Spec
**Date:** 2026-03-27
**Status:** Design / Pre-implementation
**Author:** Claude (research synthesis)
**Context:** Forecast tab Phase 5 per `2026-03-26-forecast-tab-design.md`

---

## Overview

The fundamentals model adds a non-polling, structural prior component to the WetherVane forecast. It answers: *given the current economic and political environment, what should we expect to happen in November, before any race-specific polls are considered?*

This is analogous to what 538's "Lite/Classic/Deluxe" modes do and what the Economist's fundamentals model does — but integrated into the type-covariance framework so it produces a **type-level expected signal vector** rather than a single scalar. The key design requirement is that fundamentals feed into the same Bayesian update pipeline that polls use, not as a side channel.

---

## Existing Architecture (current state)

The current 3-layer forecast stack is:

```
1. County prior     ridge_county_priors.parquet (Ridge+HGB ensemble, LOO r=0.711)
                    anchored to 2024 presidential Dem share per county

2. Generic ballot   generic_ballot.py → flat national shift applied to all county priors
                    source: polls_2026.csv rows with geo_level="national"
                    shift = gb_avg - PRES_DEM_SHARE_2024_NATIONAL (currently +3.7pp D)

3. Race polls       POST /forecast/polls → Bayesian update through type covariance Σ
                    W matrix derived from state-level type scores (or crosstab-adjusted)
```

The fundamentals model inserts between layers 1 and 2, or replaces the generic ballot entirely:

```
1. County prior     (unchanged)

2. Fundamentals     structural expected environment shift for this cycle
                    → type-level signal vector f ∈ ℝ^J (or national scalar as fallback)
                    → replaces or supplements flat generic ballot shift

3. Race polls       (unchanged — Bayesian update through Σ)
```

---

## Design Goals

1. **Defensible predictions before polls exist.** Early-cycle forecasts (January 2026) rely almost entirely on structural factors. The model should be credible then.

2. **Type-level resolution if data supports it.** A national scalar is acceptable as a first version. A type-level vector (e.g., manufacturing downturn hits working-class types harder than knowledge-worker types) is the aspirational target.

3. **Composable with existing pipeline.** Fundamentals output should enter the forecast through the same additive shift mechanism as the generic ballot. No new inference engines needed for v1.

4. **Graceful degradation.** If fundamentals data is unavailable or stale, the model falls back to the current generic ballot behavior. No crashes.

5. **Transparent and explainable.** Every fundamentals component should be displayable to users: "Based on X economic indicator and Y approval rating, the model expects a Z-point Dem advantage in a neutral district."

---

## Data Sources (all free)

### Tier 1: Core inputs (high value, freely available now)

#### 1. Presidential Approval Rating
- **What:** Current president's net approval rating as of election eve.
- **Why it matters:** Presidential approval is the single strongest fundamentals predictor of midterm outcomes. Since 1946, the president's party loses an average of ~28 House seats in midterms; approval rating explains most of the variance.
- **Historical range:** Net approval correlates with seat change at roughly r ≈ 0.7 in presidential cycles.
- **Sources:**
  - **FiveThirtyEight aggregate** — historical approval averages by president/date available in `data/raw/fivethirtyeight/` (already on disk, gitignored). Check for `approval_topline.csv` or similar.
  - **Real Clear Politics** — free daily average, scrapable HTML table. No API, but layout is stable.
  - **Wikipedia List of Presidential Approval Ratings** — well-maintained historical table; good for historical calibration.
  - **Polling CSV** — can add a row type `race="Presidential Approval"` to `polls_2026.csv` to feed manually until scraper is built.
- **Format for ingestion:** Scalar float (net approval, e.g. -15.0 for 15 points underwater).

#### 2. Economic Indicators via FRED API
- **What:** Macroeconomic indicators with strong historical correlation to midterm outcomes.
- **Why it matters:** Voters punish or reward the incumbent party for economic conditions. GDP growth, unemployment, and real income growth are the standard "fundamentals" inputs.
- **FRED API:** Free, no auth required for basic access. `https://api.stlouisfed.org/fred/series/observations?series_id=GDPC1&api_key=YOUR_KEY&file_type=json`. Key is free, sign up at fred.stlouisfed.org.
- **Key series:**
  - `GDPC1` — Real GDP (quarterly). Use Q2 growth in election year (available by August).
  - `UNRATE` — National unemployment rate (monthly). Use month before election.
  - `CPIAUCSL` — CPI (monthly). Compute YoY inflation as of election month.
  - `DSPIC96` — Real disposable personal income (monthly). Use election-year growth.
  - `UMCSENT` — University of Michigan Consumer Sentiment Index (monthly). Captures forward-looking economic expectations.
- **Format for ingestion:** One value per series (the relevant observation for the election cycle), loaded into a `FundamentalsSnapshot` dataclass.

#### 3. Historical Midterm Patterns (structural baseline)
- **What:** Historical relationship between presidential party, approval, economic conditions, and seat/vote-share outcomes.
- **Why it matters:** Even without current polling, historical patterns provide a prior on how much the in-party typically loses.
- **Sources (all on disk or downloadable):**
  - MEDSL/MIT historical election returns (already on disk): 2000–2024 presidential + governor + Senate. This is the calibration set.
  - 538 historical House results (in `data/raw/fivethirtyeight/`): check for `house_party_seat_count_1789_2023.csv` or similar.
  - Hand-curated lookup table of presidential approval + GDP growth + Dem/Rep House share (1946–2024): 20 data points, fully public from Congressional Research Service or Wikipedia. This is the historical regression dataset.
- **Format for ingestion:** CSV in `data/raw/fundamentals/midterm_history.csv` with columns: `year`, `pres_party`, `pres_net_approval_oct`, `gdp_q2_growth`, `unemployment_oct`, `house_dem_share_change`.

#### 4. Generic Ballot Polls (already integrated)
- **What:** National generic congressional ballot polling.
- **Why it matters:** The generic ballot is itself a fundamentals-adjacent signal — it reflects the general national environment before any race-specific information.
- **Current status:** Already ingested via `generic_ballot.py`. The flat `+3.7pp D` shift is currently the only "fundamentals" component.
- **Role in Phase 5:** Remains as an input to the fundamentals model but is no longer applied as a standalone flat shift. Instead, it becomes one signal among several that the fundamentals model synthesizes.

---

### Tier 2: State/regional inputs (enable type-level resolution)

These are the data sources that enable the fundamentals signal to hit types differently rather than applying a flat national shift. Required for type-vector fundamentals; not required for scalar fundamentals.

#### 5. BLS State-Level Unemployment (LAUS)
- **What:** Monthly unemployment rate by state (Local Area Unemployment Statistics).
- **URL:** `https://www.bls.gov/web/laus/laumstrk.htm` (current), bulk download at `https://download.bls.gov/pub/time.series/la/`
- **Format:** State-level monthly unemployment rate. No API key required for bulk downloads.
- **Type-level use:** States with high unemployment have more counties of working-class types. State unemployment differentials create type-level economic signals without requiring county-level data.
- **Availability:** Monthly, published ~4 weeks after reference month. October data available ~late November, so this is usable for election-eve forecast updates.

#### 6. BEA State-Level Personal Income Growth
- **What:** Annual per-capita personal income by state.
- **URL:** `https://apps.bea.gov/iTable/iTable.cfm` (interactive) or API: `https://apps.bea.gov/api/signup/` (free key).
- **BEA API key:** Free registration at apps.bea.gov. Returns JSON. Dataset: `Regional`, TableName: `SAINC1`.
- **Type-level use:** Income growth differentials by state feed into the type-level signal via the type-state score matrix.
- **Limitation:** Annual, so election-year data is available as a preliminary estimate.

#### 7. University of Michigan State-Level Consumer Sentiment
- **Not available at state level for free.** Michigan state-level data requires paid access. Skip for v1; use national UMCSENT only.

#### 8. Regional Fed Surveys (Philadelphia/Dallas/Kansas City/Richmond Fed)
- **What:** Business activity and expectations surveys from regional Federal Reserve banks.
- **URL:** Varies. Philadelphia Fed: `https://www.philadelphiafed.org/surveys-and-data/regional-economic-analysis/manufacturing-business-outlook-survey`. All free.
- **Coverage:** Only covers specific Fed districts, not all 50 states. Good proxy for manufacturing regions (Midwest, mid-Atlantic).
- **Type-level use:** Manufacturing activity index feeds directly into Rust Belt working-class types.
- **Availability:** Monthly, published quickly.

---

### Tier 3: Deferred (blocked or low ROI for 2026)

| Source | Status | Why Deferred |
|--------|--------|--------------|
| FEC fundraising (candidate-level) | Blocked — needs FEC API key that's not currently set up | Phase 6 (candidate effects) |
| Gallup presidential approval tracking | Proprietary at high frequency; historical data accessible | Use RCP aggregate instead |
| Presidential approval crosstabs by type | Proprietary (YouGov, Morning Consult) | Use aggregate approval only |
| Stock market indices (S&P 500) | Free (Yahoo Finance or FRED `SP500`), but weak midterm predictor vs. unemployment | Add in v2 if calibration warrants |
| Right track/wrong track (RCP) | Free but unstable scraping target | Manual entry for 2026 cycle |

---

## Model Structure

### v1: National Scalar (recommended starting point)

A single-equation OLS or linear regression on historical midterm cycles, producing a national shift in expected Dem share relative to the structural baseline.

```python
# Conceptual model
fundamentals_shift_pp = (
    beta_approval * net_approval           # e.g., -15 net → −0.8pp D per point
    + beta_gdp * gdp_q2_growth_pct         # e.g., 2.0% growth → +1.2pp D
    + beta_unemployment * delta_unemployment  # change from 2y prior
    + beta_cpi * yoy_inflation_pct         # e.g., 4% → −0.5pp D
    + intercept                            # in-party penalty baseline (~−2pp)
)
```

**Calibration:** Fit on 1994–2022 midterms (7–8 data points depending on series availability). Use LOO validation. With only ~8 cycles, regularization (Ridge) is essential to prevent overfit.

**Key historical regression table for calibration:**

| Year | Pres. Party | Net Approval Oct | GDP Q2 % | Unemployment Oct | CPI YoY | Dem House share Δ |
|------|-------------|-----------------|-----------|-----------------|---------|-------------------|
| 1994 | D (Clinton) | -5              | 4.0       | 5.8             | 2.7     | −6.4pp            |
| 1998 | D (Clinton) | +62             | 2.6       | 4.5             | 1.5     | +1.0pp            |
| 2002 | R (Bush)    | +63             | 1.0       | 5.7             | 1.5     | −0.3pp            |
| 2006 | R (Bush)    | -18             | 0.8       | 4.4             | 1.3     | +4.0pp            |
| 2010 | D (Obama)   | -8              | 2.5       | 9.7             | 1.2     | −6.6pp            |
| 2014 | D (Obama)   | -8              | 4.6       | 5.9             | 1.7     | −2.0pp            |
| 2018 | R (Trump)   | -12             | 3.5       | 3.7             | 2.5     | +3.0pp            |
| 2022 | D (Biden)   | -15             | -1.6      | 3.5             | 8.2     | −0.6pp            |

*Note: All values approximate from public sources; calibrate against actual election results.*

**Output:** A scalar `fundamentals_shift` in Dem share units, applied identically to all county priors (same mechanism as current generic ballot shift).

**Integration point:** Replaces or combines with `generic_ballot.py`. The simplest v1 integration: compute `fundamentals_shift` as above, then average with the current generic ballot shift weighted by a configurable `fundamentals_weight` parameter.

### v2: Type-Level Vector (requires state data)

Instead of a single national scalar, produce a J-vector of type-level expected shifts.

```
f ∈ ℝ^J    where f[k] = expected Dem shift for type k given current fundamentals
```

Construction:
1. Compute state-level economic signals (unemployment, income growth) for all 50 states.
2. Build a state-level signal vector `s ∈ ℝ^{50}`.
3. Use the type-state score matrix `A ∈ ℝ^{J×50}` (where `A[k,s]` = fraction of type k's counties in state s, weighted by county size) to project to type level: `f = A · s`.
4. Add national components (approval, GDP) as a uniform offset across all types.

This produces a type-level fundamentals vector that hits rural Midwest working-class types differently from coastal knowledge-worker types when, e.g., manufacturing employment declines.

**Type-state score matrix:** Can be computed from `type_assignments.parquet` and `counties.parquet` at model load time. No additional data required.

**Integration point:** Feed `f` into the Bayesian update as an additional observation before polls:
```
θ_fundamentals = θ_prior + f    (additive shift to type means)
```

Then poles update from `θ_fundamentals` as the new prior. This is composable with the existing `predict_race` function.

---

## Integration Architecture

### Where fundamentals enters the forecast pipeline

```
compute_county_priors()          # county historical baselines
        ↓
apply_fundamentals_shift()       # NEW: applies type-level or national fundamentals
        ↓
apply_gb_shift() [optional]      # existing generic ballot — may be folded into fundamentals
        ↓
predict_race() → Bayesian update with polls
```

For v1, `apply_fundamentals_shift()` is architecturally identical to `apply_gb_shift()` in `generic_ballot.py`. The function signature:

```python
def compute_fundamentals_shift(
    approval_net: float,           # presidential net approval (pp)
    gdp_q2_growth: float,          # Q2 real GDP growth (%)
    unemployment: float,           # current unemployment rate
    cpi_yoy: float,                # YoY CPI inflation (%)
    in_party: str,                 # "D" or "R" — president's party
    gb_shift: float | None = None, # optional generic ballot override
    model_weights: dict | None = None,  # override fitted coefficients
) -> FundamentalsInfo:             # analogous to GenericBallotInfo
```

### New module: `src/prediction/fundamentals.py`

Responsibilities:
- `FundamentalsSnapshot` dataclass — holds all raw inputs
- `load_fundamentals_snapshot()` — loads from `data/fundamentals/snapshot_2026.json`
- `compute_fundamentals_shift()` — applies regression model
- `apply_fundamentals_shift()` — modifies county priors (or type priors for v2)
- `FundamentalsInfo` dataclass — result (analogous to `GenericBallotInfo`)

### New data file: `data/fundamentals/snapshot_2026.json`

Manual + semi-automated input file:
```json
{
  "cycle": 2026,
  "in_party": "D",
  "approval_net_oct": -12.0,
  "gdp_q2_growth_pct": 1.8,
  "unemployment_oct": 4.1,
  "cpi_yoy_oct": 3.2,
  "consumer_sentiment": 68.5,
  "source_notes": {
    "approval": "RCP average 2026-10-15",
    "gdp": "BEA advance estimate 2026-07-25",
    "unemployment": "BLS LAUS 2026-11-06",
    "cpi": "BLS CPI 2026-10-10"
  }
}
```

Manual update cadence: once at model initialization (January 2026), then updated when major economic releases come in (Q2 GDP in late July, September CPI, October jobs report). Not a live feed for v1.

### Historical calibration file: `data/fundamentals/midterm_history.csv`

One row per midterm cycle (1994–2022). Compiled once from public sources. Used to fit `FundamentalsModel` coefficients at initialization.

### New data assembly script: `src/assembly/fetch_fred_fundamentals.py`

Semi-automated FRED fetch. Takes a FRED API key (from `.env`), fetches the 5 key series, extracts the election-relevant observation, and writes to `data/fundamentals/fred_latest.json`.

```bash
FRED_API_KEY=your_key uv run python -m src.assembly.fetch_fred_fundamentals --cycle 2026
```

### Existing API endpoint changes

**`GET /forecast/generic-ballot`** — extend or replace with a broader fundamentals endpoint.

Option A (backward-compatible): Keep `/forecast/generic-ballot` but add a new endpoint:
```
GET /forecast/fundamentals
```
Returns a `FundamentalsInfo` response with all component contributions displayed.

Option B: Rename `/forecast/generic-ballot` to `/forecast/environment` and return a combined struct. Higher-value change but breaks existing API contract.

**Recommendation:** Option A for v1. Add `/forecast/fundamentals` without removing existing endpoint. Migrate frontend later.

**`POST /forecast/polls` and `POST /forecast/poll`** — add `fundamentals_weight` parameter:
```python
class MultiPollInput(BaseModel):
    ...
    fundamentals_weight: float = 1.0  # 0 = ignore fundamentals; 1 = full weight
```

---

## Bayesian Prior Structure

### Where fundamentals fits in the full prior hierarchy

```
Structural prior (θ_0)     type means from Ridge+HGB county model, aggregated to type level
         ↓
Fundamentals update        shift type means by f ∈ ℝ^J (uniform for v1 national scalar)
         ↓
θ_fundamentals             updated type means before polls
         ↓
Poll Bayesian update       Σ_post⁻¹ = Σ_prior⁻¹ + Wᵀ R⁻¹ W
                           μ_post   = Σ_post (Σ_prior⁻¹ μ_fundamentals + Wᵀ R⁻¹ y)
         ↓
θ_post                     final type posterior
         ↓
county_pred = W_c · θ_post + county_residual
```

The fundamentals shift modifies `μ_prior` before the poll update. This is architecturally clean: fundamentals are just another source of information about the national environment that shifts the type prior. Polls then update from that shifted prior. The posterior is the correctly-weighted combination.

### Section weighting (from Forecast Tab design)

The existing forecast tab design specifies user-configurable section weights:
- Model prior: weight_prior (default 0.5)
- Fundamentals: weight_fundamentals (default ~0.2)
- National polls: weight_national (default 0.15)
- State polls: weight_state (default 0.15)

For v1, implement the simpler Option A from `2026-03-26-forecast-tab-design.md`: scale effective N by section weight. The fundamentals contribution is scaled by `weight_fundamentals` before being combined with the poll-driven posterior.

---

## Weighting Fundamentals vs. Polls Over Time

A critical design question: as Election Day approaches and more polls accumulate, fundamentals should be downweighted. Two approaches:

### Option A: Fixed weight with override
User controls section weights via sliders. Model recommends higher fundamentals weight early (e.g., 0.4 in January) and lower late (e.g., 0.1 in October). Default weights are time-sensitive.

### Option B: Linzer-style time decay
Inspired by Linzer (2013) "Dynamic Bayesian Forecasting." Fundamentals define the long-run prior; the effective weight of polls increases as a function of `(days_until_election)^{-0.5}` or similar.

```python
ELECTION_DAY = date(2026, 11, 3)
days_remaining = (ELECTION_DAY - date.today()).days
fundamentals_effective_weight = max(0.1, 0.5 * (days_remaining / 365))
```

At 365 days out: 50% fundamentals weight. At 30 days out: ~8% fundamentals weight. At 7 days out: effectively zero.

**Recommendation:** Option B for v2, Option A for v1 (manual slider is sufficient for the initial release; auto-decay is a refinement).

---

## Implementation Steps

### Step 1: Historical calibration dataset (0.5 days)
1. Create `data/raw/fundamentals/midterm_history.csv` with 1994–2022 data.
2. Sources: Wikipedia presidential approval, BEA GDP, BLS unemployment, BLS CPI, MIT election returns.
3. All public, no API required.
4. Fit simple OLS/Ridge regression. Log coefficients and LOO performance in `docs/experiments/`.

### Step 2: `src/prediction/fundamentals.py` (1 day)
1. `FundamentalsSnapshot` and `FundamentalsInfo` dataclasses.
2. `load_fundamentals_snapshot()` reads from `data/fundamentals/snapshot_2026.json`.
3. `compute_fundamentals_shift()` applies fitted coefficients.
4. `apply_fundamentals_shift()` applies shift to county priors (wrapper over existing `apply_gb_shift`).
5. Tests in `tests/test_fundamentals.py`.

### Step 3: FRED fetch script (0.5 days)
1. `src/assembly/fetch_fred_fundamentals.py`.
2. FRED API key in `.env` as `FRED_API_KEY`.
3. Fetches GDPC1, UNRATE, CPIAUCSL, DSPIC96, UMCSENT.
4. Writes `data/fundamentals/fred_latest.json` with election-relevant observations.
5. Manual trigger (not cron) for v1.

### Step 4: API endpoint (0.5 days)
1. Add `GET /forecast/fundamentals` to `forecast.py`.
2. Response model `FundamentalsResponse` in `api/models.py`.
3. Loads snapshot from disk; applies model; returns shift + component breakdown.
4. Tests in `api/tests/`.

### Step 5: Integrate with `predict_race` (0.5 days)
1. Modify `predict_2026_types.py` to call `compute_fundamentals_shift()` and `apply_fundamentals_shift()` before generic ballot step.
2. Add `use_fundamentals: bool = True` flag for backward compatibility.
3. Update `POST /forecast/polls` in API to use fundamentals-shifted priors.
4. If fundamentals snapshot not found, fall back to generic ballot only (existing behavior).

### Step 6: Frontend integration (1 day)
1. Fetch `/forecast/fundamentals` in Forecast tab.
2. Display as a "Fundamentals" section in the Data Panel (currently inactive/greyed).
3. Show component breakdown: approval contribution, GDP contribution, etc.
4. Wire fundamentals weight slider from section weighting UI.

### Step 7: Type-level vector (v2, future)
1. Compute type-state score matrix from `type_assignments.parquet`.
2. Fetch BLS LAUS state unemployment.
3. Project state signals through type-state matrix to produce `f ∈ ℝ^J`.
4. Replace flat national shift with type-level vector in `apply_fundamentals_shift()`.

---

## Scope Estimate

| Step | Effort | Complexity |
|------|--------|------------|
| Historical calibration dataset | 0.5 days | Low |
| `fundamentals.py` module | 1 day | Low-medium |
| FRED fetch script | 0.5 days | Low |
| API endpoint | 0.5 days | Low |
| Integration with `predict_race` | 0.5 days | Medium |
| Frontend integration | 1 day | Medium |
| **v1 total** | **4 days** | |
| Type-level vector (v2) | +2 days | High |
| Auto-decay time weighting (v2) | +1 day | Medium |
| BLS state scraping + projection | +1 day | Medium |
| **v2 total** | **8 days** | |

---

## Risks and Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Small historical N (only 8 midterms since 1994) limits regression reliability | High | Use Ridge regularization; set very wide credible intervals; be explicit in UI about uncertainty |
| Generic ballot and fundamentals are correlated (both measure national environment) | Medium | Treat generic ballot as one input to fundamentals model rather than a separate layer; or explicitly decompose into "structural" vs "polling" components |
| FRED API key registration takes time | Low | FRED key is free and instant. Fallback: manual JSON entry for all inputs. |
| Fundamentals overpower polls when election is near | Medium | Time-decay weighting (v2) or clear documentation that sliders should shift toward polls in October |
| State-level fundamentals data lags election day | Low | Use 2-months-prior data; document lag explicitly in `source_notes` |

---

## Open Questions

| ID | Question | Notes |
|----|----------|-------|
| F-001 | Should fundamentals replace the generic ballot or combine with it? | Option A (combine) is safer for v1; option B (replace) is cleaner long-term. Current generic ballot shift (+3.7pp D) already reflects some fundamentals signal. |
| F-002 | What historical window for regression calibration? 1946–2022 (19 midterms) or 1994–2022 (8)? | Broader historical window adds N but includes pre-TV, pre-national-media eras where fundamentals-outcome relationship differed. Recommendation: 1974–2022 (12 midterms) as primary, 1946–2022 as sensitivity check. |
| F-003 | Should the model explicitly model Senate vs. House outcomes separately? | Senate races have structural advantages (incumbency, 1/3 of seats up) that differ from House generic ballot. For v1, treat all 2026 races uniformly. |
| F-004 | Type-level fundamentals: should manufacturing types be hit differently from agricultural types during an inflation shock? | Yes, but requires type-level economic characterization. Already available from QCEW industry data in county features. Phase 2 research item. |
| F-005 | How to handle the 2022 anomaly? Biden had -15 net approval + 8% inflation but only lost ~5 House seats, dramatically outperforming structural models. | Use 2022 as a case study for "candidate quality / recruitment differentials" as an upper-layer adjustment. Flag as high-residual in the calibration. Suggests fundamentals alone are not sufficient — reinforces need for polling layer. |

---

## Relationship to Other Components

- **Generic ballot (`generic_ballot.py`):** Phase 5 subsumes this. The generic ballot poll average becomes one input to `compute_fundamentals_shift()` rather than the sole source of the national shift.
- **Forecast tab (`2026-03-26-forecast-tab-design.md`):** The "Fundamentals" section in the Data Panel is the frontend representation of this model. Currently inactive/greyed — Phase 5 activates it.
- **Candidate effects (Sabermetrics silo):** Fundamentals produce `θ_expected`. Posterior deviations from `θ_expected` are candidate effects. The fundamentals model is a prerequisite for quantifying candidate effects. See `docs/SABERMETRICS_ARCHITECTURE.md`.
- **Phase 5 (ROADMAP.md) / θ Inference Engine:** The ROADMAP.md Phase 5 refers to the full 2028 θ inference engine. This spec is the forecasting "Phase 5" from the Forecast Tab design spec — a different, earlier milestone. To avoid confusion, consider renaming this to "Forecast Phase 5: Fundamentals" or "Forecast Component: Fundamentals Model."
