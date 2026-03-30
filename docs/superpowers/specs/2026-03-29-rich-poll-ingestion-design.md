# Design Spec: Rich Poll Ingestion Model

**Date:** 2026-03-29
**Status:** APPROVED
**Scope:** Poll quality integration into forecast engine + tiered W vector construction with demographic inference + TODO documentation for crosstab scraping and undersampled group identification.
**Approach:** Build for the ideal case (raw unweighted data + crosstabs), degrade gracefully via off-ramps when data isn't available. Ship Tier 3 (adjusted topline-only) now. Tiers 1-2 are interfaces defined now, populated when crosstab data arrives.

---

## Why This Exists

The current forecast engine treats every poll as a scalar `(dem_share, n_sample)` observation of the average type composition of an entire state. Two pieces of information are left on the table:

1. **Poll quality is ignored.** The `poll_weighting.py` module implements house effect correction, time decay, pollster grade adjustment, and pre-primary discounting — but none of it is wired into `forecast_engine.py`. Every poll gets equal treatment regardless of pollster quality or recency.

2. **All polls observe the same generic state-level W vector.** A Trafalgar LV poll of Georgia and an Emerson RV poll of Georgia get identical W rows in the Bayesian update, despite sampling systematically different slices of the electorate. Pollsters typically weight their samples on race, age, gender, and education — but NOT on religion, urbanicity, or income. And even within weighted dimensions, an LV screen systematically filters out low-propensity voter types.

The rich ingestion model fixes both: quality-adjusted precision (σ) and poll-specific type composition (W). The math is unchanged — `θ = (W'Σ⁻¹W + λI)⁻¹(W'Σ⁻¹y + λ·θ_prior)` — we just feed it better W and Σ.

### Governing Principle: Off-Ramps

The system is designed for the ideal data tier (raw unweighted sample + full crosstabs). When that data isn't available, each missing piece has a specific fallback:

- No raw data → assume the pollster did a reasonable job on the things they claim to weight on
- No crosstabs → infer type composition from methodology signals and non-weighted dimensions only
- No methodology info → use generic state-level W (current behavior)

We do NOT try to second-guess the pollster on dimensions they already corrected for. We only apply adjustments for things they haven't done.

---

## Architecture

### Layer 1: Poll Quality Integration

Wire the existing `poll_weighting.py` pipeline into `forecast_engine.py`.

**Current state:** `forecast_engine._build_poll_arrays()` receives raw poll dicts and computes sigma from raw `n_sample`. The quality weighting code in `poll_weighting.py` is disconnected.

**New flow:**

```
polls_by_race (raw dicts from DB)
    ↓
prepare_polls() — new function in forecast_engine.py
    ↓
  1. Convert raw dicts → PollObservation objects
  2. Call apply_all_weights() from poll_weighting.py
     - House effect correction on dem_share
     - Pre-primary discount on n_sample
     - Time decay on n_sample
     - Pollster grade adjustment on n_sample
  3. Convert back to dicts with adjusted dem_share + n_sample
    ↓
_build_poll_arrays() — unchanged interface, now receives quality-adjusted values
```

**Integration point:** `prepare_polls()` is called in `run_forecast()` before `_build_poll_arrays()`. The function is a pure transformation — no side effects, no state.

**What stays the same:** The math in `estimate_theta_national` and `estimate_delta_race` is untouched. They receive better σ values but the equations don't change.

### Layer 2: Tiered W Vector Construction

Replace `build_W_state()` with `build_W_poll()` that constructs poll-specific W vectors at the best available data tier.

**Three tiers, each with a defined off-ramp:**

#### Tier 1 — Raw Unweighted Sample + Crosstabs (ideal, future)

We know exactly who was sampled: 28% Black (raw, before pollster weighting), 12% evangelical, skewed urban. W vector constructed directly from the sample's demographic-to-type mapping. No inference needed.

*Off-ramp if unavailable: fall to Tier 2.*

#### Tier 2 — Weighted Topline + Crosstabs (future, when scraping is built)

Pollster reports `dem_share=53%` and provides sub-group breakdowns (e.g., Black voters D+40, White college D+2). We can't undo their weighting, but crosstabs give us direct sub-group observations.

Each crosstab group maps to types via demographic similarity. Critically, crosstab sub-groups become **separate observations** — a poll with 4 crosstab groups produces 4 rows in the W matrix, each with its own y value. The existing Bayesian update math supports this (W is n_observations × J). A single poll with crosstabs is dramatically more informative than a single topline number.

*Off-ramp if unavailable: fall to Tier 3.*

#### Tier 3 — Weighted Topline Only (what we build and ship now)

Just `dem_share, n_sample, state, methodology`. We trust the pollster did a reasonable job on demographics they claim to weight on (race, age, gender, education). We apply adjustments only for things they *haven't* done:

**Adjustment 1: LV/RV Type Screen**

An LV screen systematically excludes low-propensity voter types. From type profiles, compute a "propensity score" per type using turnout-correlated proxies: median_age, pct_owner_occupied, pct_bachelors_plus.

- LV polls: downweight low-propensity types in W
- RV polls: slight downweight (RV still excludes unregistered)
- No methodology noted: no adjustment

The propensity model is a config-driven linear combination — not a trained model. Coefficients from political science literature on voter turnout correlates.

**Adjustment 2: Non-Weighted Dimensions**

Pollsters weight on race/age/gender/education but NOT on religion, urbanicity, or income. For these unweighted dimensions, the poll's sample reflects the pollster's *reach* — phone polls skew urban, online panels skew younger.

Infer polling method reach from methodology notes or pollster defaults:
- Online panel → slight urban skew
- Phone (IVR) → older skew (captured in LV adjustment)
- Phone (live caller) → closest to representative
- Unknown → no adjustment

Adjustments shift W on unweighted dimensions only. Magnitude is intentionally small — refinements, not overrides.

**Adjustment 3: Pollster Quality + House Effects (existing, wired in)**

The existing `apply_all_weights()` pipeline from Layer 1. House effect correction on dem_share, quality/recency adjustment on n_sample.

**What Tier 3 explicitly does NOT do:**
- Re-infer race/age/education composition (pollster already weighted on those)
- Try to undo the pollster's weighting
- Apply large adjustments — the off-ramp philosophy means Tier 3 trusts the pollster

### Dispatch Function

```python
def build_W_poll(
    poll: dict,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
    poll_crosstabs: dict | None = None,
    raw_sample_demographics: dict | None = None,
    w_vector_mode: str = "core",  # "core" | "full"
) -> np.ndarray:
    """Construct poll-specific W vector at the best available tier."""

    if raw_sample_demographics is not None:
        return build_W_from_raw_sample(...)      # Tier 1

    if poll_crosstabs is not None:
        return build_W_from_crosstabs(...)        # Tier 2

    return build_W_with_adjustments(...)           # Tier 3
```

### W Vector Mode Comparison

Two modes for the Tier 3 non-weighted dimension adjustment:

- `"core"`: Adjusts on religion only (the single biggest dimension pollsters don't weight on)
- `"full"`: Adjusts on religion + urbanicity + income

Both modes use identical tier logic and code paths. The difference is only which dimensions feed the non-weighted adjustment in `build_W_with_adjustments()`.

After implementation, run the full forecast pipeline twice (once per mode), compare LOO r and RMSE, set the winner as default. The mode is a parameter on `run_forecast()`.

---

## Data Flow

### Inputs

| Data | Source | Status |
|------|--------|--------|
| Poll topline (dem_share, n_sample, state, date, pollster) | `polls` table in DuckDB | Available — 109 polls |
| Poll methodology (LV/RV/online/phone) | `poll_notes` table, parsed from notes column | Partially available — notes contain "LV", "RV" markers |
| Pollster quality ratings | 538 `pollster-ratings-combined.csv` + Silver Bulletin XLSX | Available |
| House effects | Silver Bulletin + 538 bias_ppm | Available |
| State demographics | ACS via `county_demographics` table, aggregated to state level | Available |
| Type demographic profiles | `type_profiles.parquet` from `describe_types.py` | Available |
| Type propensity proxies | median_age, pct_owner_occupied, pct_bachelors_plus per type | Available in type profiles |
| State-level type weights | `type_scores` aggregated by state (vote-weighted) | Available — computed from `county_type_assignments` |
| Poll crosstabs | `poll_crosstabs` table | Schema exists, table empty — populated by future scraping |
| Raw unweighted sample data | Direct from pollster | Not available — requires future data partnerships |

### Outputs

The enriched poll data feeds into the existing Bayesian update. The output interface is unchanged:

- `W_all`: (n_observations × J) matrix — now with poll-specific rows instead of generic state rows
- `y_all`: (n_observations,) — dem_share values (house-effect-corrected for Tier 3)
- `sigma_all`: (n_observations,) — quality-adjusted poll noise
- These feed directly into `estimate_theta_national()` and `estimate_delta_race()`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/prediction/poll_enrichment.py` | Tiered W vector construction: `build_W_poll()`, `build_W_with_adjustments()`, `build_W_from_crosstabs()`, `build_W_from_raw_sample()` |
| `src/prediction/propensity_model.py` | LV/RV type-screen propensity scoring: config-driven linear combination of type demographics → propensity score per type |
| `data/config/poll_method_adjustments.json` | Config file: LV/RV adjustment factors, polling method reach profiles, propensity model coefficients |

### Modified Files

| File | Change |
|------|--------|
| `src/prediction/forecast_engine.py` | Add `prepare_polls()` for quality integration; replace `build_W_state()` calls with `build_W_poll()` in `_build_poll_arrays()`; add `w_vector_mode` parameter to `run_forecast()` |
| `src/prediction/predict_2026_types.py` | Pass `w_vector_mode` through to `run_forecast()` call in `run()` |

### Unchanged Files

| File | Why Unchanged |
|------|--------------|
| `src/prediction/national_environment.py` | Math unchanged — just receives better W and σ |
| `src/prediction/candidate_effects.py` | Math unchanged — just receives better residuals |
| `src/propagation/poll_weighting.py` | Already built — called by new `prepare_polls()`, not modified |
| `src/assembly/ingest_polls.py` | Already supports xt_* columns — no changes needed |
| `src/db/domains/polling.py` | Crosstab schema already defined — no changes needed |

---

## Testing Strategy

### Unit Tests

| Test | What It Validates |
|------|------------------|
| `test_prepare_polls` | Quality weighting integration: house effect shifts dem_share, time decay reduces n_sample, pollster grade adjusts n_sample |
| `test_build_W_tier_dispatch` | Tier selection: raw data → Tier 1, crosstabs → Tier 2, topline only → Tier 3 |
| `test_build_W_tier3_lv_adjustment` | LV screen downweights low-propensity types relative to RV/no-screen |
| `test_build_W_tier3_religion_adjustment` | Non-weighted dimension shifts W when poll state has strong religious composition |
| `test_build_W_tier2_crosstabs` | Crosstab groups produce multiple W rows with correct type mappings |
| `test_build_W_tier1_raw` | Raw sample demographics map directly to type weights |
| `test_w_vector_mode_core_vs_full` | Core mode uses fewer dimensions than full mode |
| `test_propensity_model` | Propensity scores correlate with known turnout proxies |

### Integration Test

Run the full forecast pipeline on all 33 senate races in both modes (`core` and `full`), compare against the current (no-adjustment) baseline:
- LOO r (leave-one-out correlation)
- RMSE
- Per-race prediction shift magnitude (sanity check: Tier 3 adjustments should be small, not revolutionizing predictions)

### Validation Criteria

Tier 3 adjustments should:
- Shift predictions by < 2pp on average (these are refinements, not overhauls)
- Not degrade LOO r relative to baseline
- Show larger effects on races with many LV polls vs RV polls (the LV adjustment should be detectable)

---

## Future Work (TODO Items)

### TODO-POLL-1: Crosstab Scraping Pipeline

**Priority:** High — enables Tier 2 W vectors, the biggest information gain.

**Scope:** Per-pollster integrations for extracting demographic breakdowns from original poll releases (PDFs, pollster websites). No known structured API or aggregator exists.

**Priority scraping targets** (by volume in our data):
- Cygnal — multiple GA/FL/MI polls
- Emerson College — GA/NC polls, tends to publish detailed crosstabs
- Trafalgar Group — GA poll, distinctive methodology
- Quantus Insights — GA polls
- TIPP Insights — large sample polls, likely has crosstabs

**Output:** Populate `poll_crosstabs` table with `demographic_group`, `group_value`, `pct_of_sample`, and (when available) `dem_share` per sub-group.

**Design note:** Each pollster has a different release format. Expect N separate parsers. Prioritize pollsters that publish structured data (HTML tables) over PDF-only releases.

### TODO-POLL-2: Undersampled Group Identification

**Priority:** Medium — requires Tier 2 or tract-level analysis to be meaningful.

**Core insight:** Demographic representation does not equal type representation. A poll weighted to 33% Black in Georgia can still miss that Black voters in Atlanta (Type 29: Black-Belt Mid-Income) behave differently from Black voters in rural SW Georgia (Type 50: Black-Belt Rural Unchurched). These are different types with different political behavior. The question is: did the poll sample both groups, or are we hearing from one tract that checks the demographic box?

**Approach:**
1. For each poll, compute "type coverage" — which types in the polled state are likely represented given the poll's sample composition and methodology
2. Compare against the state's full type distribution (vote-weighted)
3. Identify "coverage gaps" — types that are likely underrepresented
4. Two outputs:
   - **σ adjustment:** Inflate per-type uncertainty for underrepresented types (wider credible intervals)
   - **API reporting:** Surface coverage gaps in forecast response so the UI can show "this forecast has limited data for [type X] communities"

**Key factor:** Sample size. On n=10,000 polls you generally get one of everybody. On n=300 polls you've missed a lot of tracts. The coverage gap analysis should be sample-size-aware.

**Requires:** Tract-level type assignments + demographic profiles at sub-state geographic level (available from existing tract model).

### TODO-POLL-3: House Effects as Type Signal

**Priority:** Low — requires TODO-POLL-1 (crosstab data) to validate.

**Current approach:** House effects are treated as pollster bias — corrected away before the Bayesian update. A pollster showing R+2 relative to peers has their dem_share adjusted D-ward by 2pp.

**Alternative interpretation:** Persistent house effects may reflect which types a pollster systematically reaches. Trafalgar's R-lean may mean they reach rural evangelical types that other pollsters miss — that's information about type composition, not error to correct.

**Future work:**
1. Decompose house effects into type-reach profiles per pollster (requires sufficient poll history per pollster)
2. Use pollster-specific type-reach as a prior on their W vector construction
3. Stop correcting house effects away — instead, let the W vector capture the information

**Risk:** This interacts with the house effect correction in Layer 1. If we both correct dem_share AND infer type composition from the house effect, we double-count. The resolution is to *replace* house effect correction with type-reach inference once the latter is validated — but that requires confidence in the type-reach estimates, which requires crosstab data.

**For now:** Keep house effect correction as-is. Flag this TODO for revisiting once Tier 2 is operational and we have crosstab data to validate type-reach hypotheses.

---

## Configuration

### `data/config/poll_method_adjustments.json`

```json
{
  "lv_propensity_coefficients": {
    "median_age": 0.3,
    "pct_owner_occupied": 0.4,
    "pct_bachelors_plus": 0.3,
    "_note": "Linear combination → propensity score per type. Higher = more likely to pass LV screen."
  },
  "lv_downweight_factor": 0.5,
  "rv_downweight_factor": 0.8,
  "method_reach_profiles": {
    "online_panel": {
      "log_pop_density_shift": 0.05,
      "_note": "Online panels slightly overrepresent urban/suburban types"
    },
    "phone_ivr": {
      "_note": "IVR skews older — captured by LV adjustment, no separate correction"
    },
    "phone_live": {
      "_note": "Closest to representative — no adjustment"
    },
    "unknown": {
      "_note": "No adjustment — assume representative"
    }
  },
  "w_vector_dimensions": {
    "core": ["evangelical_share", "catholic_share", "mainline_share"],
    "full": [
      "evangelical_share", "catholic_share", "mainline_share",
      "log_pop_density", "median_hh_income", "pct_owner_occupied"
    ]
  }
}
```

All adjustment magnitudes are intentionally conservative. These are refinements to the W vector, not overrides. The config file is the single source of truth for all tunable parameters.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | Our Rule |
|---|---|---|
| Re-inferring demographics pollsters already weighted on | Double-counting — pollster already corrected for race/age/education | Only adjust non-weighted dimensions |
| Large W vector adjustments from thin methodology signals | Overfitting to noisy method categories | Cap adjustment magnitudes; Tier 3 shifts should average < 2pp |
| Treating house effects as pure bias | Loses information about pollster type-reach | TODO-POLL-3 flags this for future work; don't "fix" it now |
| Building Tier 1/2 code without data to test it | Untestable code rots | Define interfaces only; implement when crosstab data arrives |
| Hardcoding adjustment factors | Stale after recalibration | All factors in config JSON, not in code |
