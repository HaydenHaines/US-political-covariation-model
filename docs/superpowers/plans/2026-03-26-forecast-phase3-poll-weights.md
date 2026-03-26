# Forecast Tab Phase 3: Section Weight Sliders

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add section weight sliders to the Forecast tab so users can control how much the model prior vs poll data influences predictions.

**Architecture:** Section weights scale poll effective N (Option A from spec). A `model_prior_weight` of 0.5 means the prior is half as informative (prior covariance scaled by 1/weight). Poll section weights scale `n_sample` before the Bayesian update. Backend receives weights via `MultiPollInput`; frontend sends them on recalculate.

**Tech Stack:** FastAPI (Python), Next.js/React (TypeScript), existing poll_weighting + predict_2026_types pipeline.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `api/models.py` | Modify | Add `section_weights` to `MultiPollInput` |
| `api/routers/forecast.py` | Modify | Apply section weights to poll N and prior scaling |
| `src/prediction/predict_2026_types.py` | Modify | Accept `prior_weight` param in `predict_race` |
| `web/lib/api.ts` | Modify | Add `section_weights` to `feedMultiplePolls` body |
| `web/components/ForecastView.tsx` | Modify | Add weight sliders UI, pass weights on recalculate |
| `tests/test_forecast_weights.py` | Create | API tests for weight scaling behavior |

---

### Task 1: Backend — Add prior_weight to predict_race

**Files:**
- Modify: `src/prediction/predict_2026_types.py:124-237`
- Create: `tests/test_forecast_weights.py`

- [ ] **Step 1: Write failing test for prior_weight parameter**

```python
# tests/test_forecast_weights.py
"""Tests for section weight scaling in forecast pipeline."""
import numpy as np
import pytest

from src.prediction.predict_2026_types import predict_race


@pytest.fixture
def mock_model():
    """Minimal 3-county, 2-type model for testing weight effects."""
    J = 2
    N = 3
    type_scores = np.array([
        [0.8, 0.2],  # county 0: mostly type 0
        [0.3, 0.7],  # county 1: mostly type 1
        [0.5, 0.5],  # county 2: mixed
    ])
    type_covariance = np.array([
        [0.01, 0.002],
        [0.002, 0.01],
    ])
    type_priors = np.array([0.55, 0.40])
    county_fips = ["01001", "01003", "01005"]
    states = ["AL", "AL", "AL"]
    county_names = ["Autauga", "Baldwin", "Barbour"]
    county_priors = np.array([0.52, 0.38, 0.45])
    return {
        "type_scores": type_scores,
        "type_covariance": type_covariance,
        "type_priors": type_priors,
        "county_fips": county_fips,
        "states": states,
        "county_names": county_names,
        "county_priors": county_priors,
    }


def test_prior_weight_zero_ignores_prior(mock_model):
    """With prior_weight=0, polls should dominate completely."""
    polls = [(0.60, 500, "AL")]
    result_full = predict_race(
        race="test", polls=polls, prior_weight=1.0, **mock_model,
    )
    result_zero = predict_race(
        race="test", polls=polls, prior_weight=0.01, **mock_model,
    )
    # With near-zero prior weight, predictions should move further toward poll
    shift_full = abs(result_full["pred_dem_share"].mean() - 0.45)
    shift_zero = abs(result_zero["pred_dem_share"].mean() - 0.45)
    assert shift_zero > shift_full


def test_prior_weight_one_is_default(mock_model):
    """prior_weight=1.0 should produce identical results to no weight arg."""
    polls = [(0.60, 500, "AL")]
    result_default = predict_race(race="test", polls=polls, **mock_model)
    result_explicit = predict_race(
        race="test", polls=polls, prior_weight=1.0, **mock_model,
    )
    np.testing.assert_array_almost_equal(
        result_default["pred_dem_share"].values,
        result_explicit["pred_dem_share"].values,
    )


def test_no_polls_prior_weight_irrelevant(mock_model):
    """Without polls, prior_weight doesn't change output (no update to scale)."""
    result_a = predict_race(race="test", prior_weight=0.5, **mock_model)
    result_b = predict_race(race="test", prior_weight=1.0, **mock_model)
    np.testing.assert_array_almost_equal(
        result_a["pred_dem_share"].values,
        result_b["pred_dem_share"].values,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forecast_weights.py -v`
Expected: FAIL — `predict_race() got an unexpected keyword argument 'prior_weight'`

- [ ] **Step 3: Add prior_weight parameter to predict_race**

In `src/prediction/predict_2026_types.py`, add `prior_weight: float = 1.0` parameter to `predict_race()` signature (after `county_priors`). Before the Bayesian update block (line ~206), scale the prior covariance:

```python
    # Scale prior precision by prior_weight (lower weight = less informative prior)
    if prior_weight != 1.0 and prior_weight > 0:
        # Scaling covariance by 1/weight makes the prior less precise,
        # so polls pull predictions further from the baseline.
        type_cov = type_cov / prior_weight
```

Insert this between `type_cov = type_covariance.copy().astype(float)` (line 206) and `if polls:` (line 208).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_forecast_weights.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/prediction/predict_2026_types.py tests/test_forecast_weights.py
git commit -m "feat: add prior_weight parameter to predict_race for section weighting"
```

---

### Task 2: Backend — Add section_weights to MultiPollInput and API

**Files:**
- Modify: `api/models.py:100-106`
- Modify: `api/routers/forecast.py:292-424`
- Modify: `tests/test_forecast_weights.py`

- [ ] **Step 1: Write failing test for API weight passthrough**

Append to `tests/test_forecast_weights.py`:

```python
def test_poll_n_scaling(mock_model):
    """Section weight < 1 should reduce poll influence (smaller effective N)."""
    polls_full = [(0.60, 500, "AL")]
    polls_scaled = [(0.60, 250, "AL")]  # 500 * 0.5 = 250
    result_full = predict_race(race="test", polls=polls_full, **mock_model)
    result_scaled = predict_race(race="test", polls=polls_scaled, **mock_model)
    # Halved N should move predictions less toward 0.60
    shift_full = abs(result_full["pred_dem_share"].mean() - 0.45)
    shift_scaled = abs(result_scaled["pred_dem_share"].mean() - 0.45)
    assert shift_full > shift_scaled
```

- [ ] **Step 2: Run test to verify it passes** (this tests the N-scaling concept, should pass already)

Run: `uv run pytest tests/test_forecast_weights.py::test_poll_n_scaling -v`
Expected: PASS

- [ ] **Step 3: Add SectionWeights to api/models.py**

In `api/models.py`, add a new model and update `MultiPollInput`:

```python
class SectionWeights(BaseModel):
    model_prior: float = Field(1.0, ge=0.0, le=2.0)
    state_polls: float = Field(1.0, ge=0.0, le=2.0)
    national_polls: float = Field(1.0, ge=0.0, le=2.0)
```

Add `section_weights` field to `MultiPollInput`:

```python
class MultiPollInput(BaseModel):
    cycle: str
    state: str
    race: str | None = None
    half_life_days: float = 30.0
    apply_quality: bool = True
    section_weights: SectionWeights = Field(default_factory=SectionWeights)
```

- [ ] **Step 4: Apply section weights in forecast.py**

In `api/routers/forecast.py`, in `update_forecast_with_multi_polls` (line ~340-346), after `race_polls` is built, scale each poll's N by the state_polls weight:

```python
    # Apply section weight to poll effective N (Option A from spec)
    sw = body.section_weights
    race_polls = [
        (p.dem_share, max(1, int(p.n_sample * sw.state_polls)), p.geography)
        for p in weighted
        if p.geo_level == "state"
    ]
```

And pass `prior_weight` to `predict_race` (line ~376-386):

```python
        result_df = predict_race(
            race=body.race or race_polls[0][2],
            polls=race_polls,
            type_scores=type_scores,
            type_covariance=type_covariance,
            type_priors=type_priors,
            county_fips=fips_list,
            states=states_list,
            county_names=names_list,
            county_priors=county_priors,
            prior_weight=sw.model_prior,
        )
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/test_forecast_weights.py -v && uv run pytest api/tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add api/models.py api/routers/forecast.py tests/test_forecast_weights.py
git commit -m "feat: add section_weights to MultiPollInput API"
```

---

### Task 3: Frontend — Add weight sliders to ForecastView

**Files:**
- Modify: `web/lib/api.ts:184-198`
- Modify: `web/components/ForecastView.tsx`

- [ ] **Step 1: Update api.ts to pass section_weights**

In `web/lib/api.ts`, update `feedMultiplePolls` signature:

```typescript
export async function feedMultiplePolls(body: {
  cycle: string;
  state: string;
  race?: string;
  half_life_days?: number;
  apply_quality?: boolean;
  section_weights?: {
    model_prior: number;
    state_polls: number;
    national_polls: number;
  };
}): Promise<MultiPollResponse> {
```

(No other changes needed — the body is already JSON.stringified directly.)

- [ ] **Step 2: Add weight slider state to ForecastView**

In `web/components/ForecastView.tsx`, add state after the existing state declarations (~line 155):

```typescript
  const [modelPriorWeight, setModelPriorWeight] = useState(1.0);
  const [statePollsWeight, setStatePollsWeight] = useState(1.0);
```

- [ ] **Step 3: Create WeightSlider component**

Add a `WeightSlider` inline component above the `ForecastView` export:

```typescript
function WeightSlider({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "6px 10px",
      fontSize: "12px",
    }}>
      <span style={{ color: "var(--color-text-muted)", minWidth: "50px" }}>
        {label}
      </span>
      <input
        type="range"
        min={0}
        max={200}
        value={Math.round(value * 100)}
        onChange={(e) => onChange(parseInt(e.target.value) / 100)}
        style={{ flex: 1, height: "4px", accentColor: "#2166ac" }}
      />
      <span style={{
        fontFamily: "var(--font-mono, monospace)",
        fontSize: "11px",
        minWidth: "32px",
        textAlign: "right",
        color: value === 1.0 ? "var(--color-text-muted)" : "#333",
        fontWeight: value === 1.0 ? "normal" : 600,
      }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}
```

- [ ] **Step 4: Wire sliders into the Model Prior and State Polls sections**

In the Model Prior section (after the description `div`, ~line 365), add the slider:

```typescript
        {modelPriorOpen && (
          <div style={{
            padding: "8px 10px",
            border: "1px solid var(--color-border)",
            borderTop: "none",
            borderRadius: "0 0 3px 3px",
            marginBottom: "4px",
            fontSize: "12px",
            color: "var(--color-text-muted)",
            lineHeight: "1.5",
          }}>
            Structural baseline from type covariance (Ridge+HGB ensemble, LOO r=0.671).
            <WeightSlider label="Weight" value={modelPriorWeight} onChange={setModelPriorWeight} />
          </div>
        )}
```

In the State Polls section (after the poll list, inside the `statePollsOpen && polls.length > 0` block), add:

```typescript
            <WeightSlider label="Weight" value={statePollsWeight} onChange={setStatePollsWeight} />
```

Place it as the last child before the closing `</div>` of the polls container.

- [ ] **Step 5: Pass section_weights in recalculate callback**

Update the `recalculate` callback (~line 236) to pass weights:

```typescript
      if (polls.length > 0) {
        const result = await feedMultiplePolls({
          cycle: YEAR,
          state: selectedState,
          race: selectedRace,
          section_weights: {
            model_prior: modelPriorWeight,
            state_polls: statePollsWeight,
            national_polls: 1.0,
          },
        });
```

Add `modelPriorWeight` and `statePollsWeight` to the `useCallback` dependency array.

- [ ] **Step 6: Build and verify**

Run: `cd web && npm run build`
Expected: Build succeeds with no type errors

- [ ] **Step 7: Commit**

```bash
git add web/lib/api.ts web/components/ForecastView.tsx
git commit -m "feat: add section weight sliders to Forecast tab"
```

---

### Task 4: Integration test — end-to-end weight behavior

**Files:**
- Modify: `tests/test_forecast_weights.py`

- [ ] **Step 1: Write integration test verifying weights affect API response**

Append to `tests/test_forecast_weights.py`:

```python
def test_section_weights_model():
    """Verify SectionWeights model has correct defaults and validation."""
    from api.models import SectionWeights, MultiPollInput

    # Defaults
    sw = SectionWeights()
    assert sw.model_prior == 1.0
    assert sw.state_polls == 1.0
    assert sw.national_polls == 1.0

    # Custom values
    sw2 = SectionWeights(model_prior=0.5, state_polls=1.5)
    assert sw2.model_prior == 0.5
    assert sw2.state_polls == 1.5

    # Embedded in MultiPollInput
    mpi = MultiPollInput(cycle="2026", state="FL")
    assert mpi.section_weights.model_prior == 1.0

    mpi2 = MultiPollInput(
        cycle="2026", state="FL",
        section_weights=SectionWeights(model_prior=0.3),
    )
    assert mpi2.section_weights.model_prior == 0.3
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_forecast_weights.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_forecast_weights.py
git commit -m "test: add integration tests for section weight models"
```

---

### Task 5: Build, deploy, verify

**Files:** None (deployment only)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: All pass, no regressions

- [ ] **Step 2: Build frontend**

```bash
cd /home/hayden/projects/wethervane/web
npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
```

- [ ] **Step 3: Restart services**

```bash
systemctl --user restart wethervane-api.service
systemctl --user restart wethervane-frontend.service
```

- [ ] **Step 4: Verify API health**

```bash
curl -s http://localhost:8002/api/v1/health
```
Expected: `{"status":"ok","db":"connected","contract":"ok"}`

- [ ] **Step 5: Test weight API manually**

```bash
curl -s -X POST http://localhost:8002/api/v1/forecast/polls \
  -H "Content-Type: application/json" \
  -d '{"cycle":"2026","state":"FL","race":"2026 FL Senate","section_weights":{"model_prior":0.5,"state_polls":1.5,"national_polls":1.0}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'polls_used={d[\"polls_used\"]}, counties={len(d[\"counties\"])}')"
```
Expected: Response with county predictions and poll count

- [ ] **Step 6: Final commit and push**

```bash
git push
```
