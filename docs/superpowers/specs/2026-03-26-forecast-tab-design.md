# Forecast Tab Design — WetherVane
**Date:** 2026-03-26
**Status:** Planning / In Progress
**Branch:** feat/forecasting

---

## Vision

The Forecast tab answers the public-facing question: *What will happen in 2026?*

It is a configurable prediction interface. The structural model provides the baseline; users layer in data inputs (fundamentals, polls) and adjust section weights to see how predictions respond. All poll ingestion flows through the type covariance — races, geographies, and data sources are all handled as signals that update the shared type structure.

---

## UI Layout

### Controls (top, sequential)

1. **State dropdown** — selects a state; map pans and highlights selected state in white
2. **Year dropdown** — filters by election cycle (e.g., 2026)
3. **Election dropdown** — selects a specific race (e.g., GA Senate); narrows available poll options

### Data Panel (collapsible, below controls)

A collapsible section with a caret toggle. Contains data **sections**, each with independently configurable weight. Sections are expandable in future phases (candidate data, district-level data, etc.).

**Current sections:**
- **Model prior** — the structural Ridge+HGB county prior (always present; represents the model's structural expectation before any external signal)
- **Fundamentals** — national economic/climate signal ("it's the economy, stupid"); modifies expected demographic lean. Separate model to be built. Can be weighted to zero.
- **National polls** — polls with national scope for the selected race or related races
- **State polls** — polls scoped to the selected state; may be for the selected race or a correlated race (e.g., governor's race informing Senate prediction)

### Section Weighting

- Each section has a **weight slider** (default: recommended weights, e.g. 50% model+fundamentals, 25% national polls, 25% state polls)
- Weights are user-configurable; defaults are our "recommended" blend
- Intra-section reconciliation (how polls within a section are combined) is **non-configurable** — handled by our algorithm (see below)

### Recalculate Button

Pressing "Recalculate" reruns the prediction with current weights and selected data, then re-renders the map choropleth.

**Performance note:** At national scale, full re-render is expensive. Consider rendering a **state-only zoomed choropleth** when a state is selected, rather than re-rendering all 3,154 counties on every recalculate. Defer decision until we measure render time in practice.

---

## Data Model

### Poll Ingestion (current — simple)

Polls enter as `(dem_share, n_sample, state, race, date, pollster)`. W vector is derived from generic state-level type scores.

### Poll Ingestion (target — rich)

**DEBT:** See CLAUDE.md Known Tech Debt — "Poll Ingestion — Rich Ingestion Model Needed."

When crosstab data is available, the poll-specific W vector should be constructed from crosstab demographic composition rather than generic state-level type scores. A poll that oversampled college-educated voters should pull harder on types with high college-educated membership. This requires:
1. Storing crosstab data per poll in the database
2. Mapping crosstab demographic groups to the type structure
3. Computing a demographically-informed W per poll

Until rich ingestion is built, polls use generic state-level W (current behavior).

### All Polls Go Through Type Covariance

Every data input — regardless of race, geography, or section — updates the shared type means via the Bayesian update:

```
Σ_post⁻¹ = Σ_prior⁻¹ + Wᵀ R⁻¹ W
μ_post   = Σ_post (Σ_prior⁻¹ μ_prior + Wᵀ R⁻¹ y)
```

A governor's poll in Georgia provides information about Georgia's type composition. That updates the type means θ. Since types are race-agnostic, the updated θ propagates to the Senate race prediction via the same type scores. Cross-race information transfer is automatic through the covariance structure.

### Section Weight Implementation

Section weights determine how much each section's polls contribute to the combined W matrix before the Bayesian update. Two implementation options:

**Option A — Scale effective N by weight:** Multiply each poll's `n_sample` by its section weight before constructing R. Simple; works with existing update math.

**Option B — Sequential updates with weight decay:** Run separate Bayesian updates per section; posterior from one becomes prior for the next. More complex but more principled for sections with very different signal types.

Decision deferred. Option A is simpler and sufficient for initial implementation.

### Intra-Section Reconciliation (non-configurable)

How polls within a section are combined into a single section-level signal:

- **Time decay:** More recent polls weighted higher (exponential decay, half-life TBD — currently 60 days in `poll_weighting.py`)
- **Quality weighting:** Silver Bulletin ratings → effective N multiplier (currently 0.3x–1.2x)
- **Stacked Bayesian update:** Polls are NOT collapsed to a single effective poll first (see CLAUDE.md debt item on multi-poll collapse). Each poll is a separate row in W.

---

## Output Display

**Start with what we currently generate:**
- `pred_dem_share` — county-level Democratic share point estimate
- `pred_lo90` / `pred_hi90` — 90% credible interval
- `pred_std` — posterior standard deviation

**Evolve toward what's useful (TBD):**
- State-level aggregated prediction (population-weighted mean of county predictions)
- Win probability P(D > 0.5) at state level — requires integrating over posterior predictive distribution
- How far polls are moving the prediction vs the structural prior (delta display)

The output display design should be driven by what the model can defensibly produce, not by what looks good. Don't let UI requirements constrain model architecture.

---

## Candidate Effects (interpretive layer, future)

When posterior θ deviates from expected θ (Σ + fundamentals), that deviation is a candidate effect:

```
candidate_effect[k] = θ_posterior[k] - θ_expected[k]
```

The Forecast tab should eventually surface this as an interpretive layer: "This candidate is over/underperforming with [type] relative to structural expectations." This connects to the Sabermetrics silo (CTOV, candidate drag/lift) but operates at the type level rather than the race level.

**Critical distinction to preserve in the display:** A deviation from expected θ could be a *candidate effect* (idiosyncratic to this candidate, will not persist) or a *structural shift* (the type has genuinely relocated its partisan baseline). These require different interpretations and different responses in the model. A candidate effect should not update the structural prior. A structural shift should. See VISION.md for the full framing.

For 2026, the Forecast tab does not need to display candidate effects — but the architecture should not preclude it. The posterior θ should always be available for comparison against the expected θ baseline.

---

## Fundamentals Model (separate, future)

Fundamentals are a **national signal** that modifies expected demographic lean for a given cycle. Conceptually: economic conditions, presidential approval, historical midterm patterns.

**Investigation needed:** Whether state- or regional-level fundamentals data exists (BLS, BEA, regional Fed data) that would allow the fundamentals signal to hit types differently by region. A manufacturing downturn hits Rust Belt working-class types differently than coastal knowledge-worker types. If the data supports it, fundamentals should produce a type-level signal vector rather than a single national scalar.

**Data candidates for national fundamentals:**
- Consumer confidence index
- GDP growth
- Presidential approval (national)
- Historical midterm environment (party in power, seat exposure)

**Data candidates for regional/state fundamentals:**
- BLS state unemployment by industry
- BEA state-level income growth
- Regional Fed sentiment surveys

Fundamentals model is out of scope for initial Forecast tab implementation. The section should be present in the UI but inactive/greyed until the model is built.

---

## Implementation Phases

### Phase 1 — Fix Known Bugs (prerequisite)
- [ ] Fix multi-poll collapse: stack W rows instead of aggregating to single effective poll
- [ ] Fix fragile state extraction: pass `poll.state` explicitly to `predict_race()`
- [ ] Fix `state_pred = None` in type pipeline: compute W·μ_post and populate field

### Phase 2 — Forecast Tab Scaffold
- [ ] State / year / election dropdowns wired to available races in DuckDB
- [ ] Map pan + highlight on state selection
- [ ] Model-prior-only forecast display (no polls, no fundamentals) — structural baseline
- [ ] Recalculate button → re-renders choropleth (state-only view if performance is a concern)

### Phase 3 — Poll Integration
- [ ] Data panel with collapsible sections (national polls, state polls)
- [ ] Pre-loaded polls from ingestion pipeline visible per race
- [ ] Section weight sliders with recommended defaults
- [ ] Intra-section reconciliation (stacked update, time decay, quality weighting)
- [ ] Cross-race poll propagation working via type covariance

### Phase 4 — Rich Poll Ingestion
- [ ] Crosstab storage in DuckDB per poll
- [ ] Crosstab → type W vector construction
- [ ] Ingestion pipeline rebuilt for rich poll format
- [ ] See CLAUDE.md debt item for full scope

### Phase 5 — Fundamentals Section
- [ ] National fundamentals model built and validated
- [ ] State/regional fundamentals investigation
- [ ] Fundamentals section activated in Forecast tab UI

---

## Open Questions

| ID | Question | Notes |
|----|----------|-------|
| FT-001 | Section weight UI: sliders or text inputs? | Sliders feel more interactive but text inputs are more precise |
| FT-002 | Should recommended weights vary by how many polls are available? | E.g., if only 1 poll exists, should state polls weight higher? |
| FT-003 | State-only choropleth vs full national on recalculate? | Measure render time before deciding |
| FT-004 | Does the Forecast tab replace the existing GET /forecast endpoint output, or augment it? | Currently GET /forecast reads stored predictions; Forecast tab adds dynamic updates on top |
| FT-005 | Win probability: Monte Carlo sampling or analytical? | Analytical is faster; MC gives full distribution |
| FT-006 | State-level fundamentals data availability | Investigate BLS/BEA/regional Fed before committing to national-only |
