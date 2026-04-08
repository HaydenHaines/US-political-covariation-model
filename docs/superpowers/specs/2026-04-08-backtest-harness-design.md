# WetherVane Backtest Harness — Design Spec

**Date:** 2026-04-08
**Status:** Approved, pending implementation

---

## Overview

The backtest harness evaluates WetherVane's model quality by running it against historical information states and comparing its outputs to known election results. Unlike unit tests (which verify mechanics) or sanity checks (which catch broken predictions), the backtest harness answers the hard question: *how good are we, and where do we systematically go wrong?*

A single backtest run reconstructs the world as it existed on October 31 of a target election year — using only polls, fundamentals, and model artifacts available at that time — produces a full forecast, then compares that forecast to actual results across four levels: state vote share, county residuals, type-level systematic bias, and chamber seat outcomes. Repeated across multiple election cycles, error patterns emerge: overestimation of Democrats in the Mountain West, a consistent candidate-effect miss in competitive Senate races, a type cluster that always breaks late in one direction.

This is a research tool, not a CI gate. Outputs are structured artifacts (parquet + JSON) that persist across runs and a human-readable markdown report generated from those artifacts.

---

## Goals

1. **Detect broken predictions** — large errors (>10pp) that indicate something is mechanically wrong
2. **Surface systematic bias** — consistent directional errors across cycles that indicate modeling assumptions are wrong
3. **Enable error classification** — distinguish candidate effects, structural misses, and one-off outliers
4. **Track model improvement** — re-run after pipeline changes to see if errors improved or regressed
5. **Build toward honest historical simulation** — Phase 3 adds epoch-retrained types; Phase 1–2 use current types as a faster starting point

---

## Non-Goals

- This is not a CI regression test. It does not block deploys.
- It does not retrain the model (until Phase 3).
- It does not generate public-facing forecasts.
- It does not replace the Tier 1 build-time gate or Tier 2 sanity suite from `MODEL_VERIFICATION_PLAN.md`.

---

## Architecture

### Module: `src/backtest/`

Six files, each with one clear responsibility:

```
src/backtest/
  __init__.py
  cli.py          # Entry point: `python -m src.backtest run/report`
  inputs.py       # Reconstruct historical information state
  runner.py       # Run forecast_engine with historical inputs
  actuals.py      # Load known election results
  errors.py       # Compute ErrorArtifact from predictions vs actuals
  report.py       # Generate markdown report from ErrorArtifacts
  catalog.py      # Manage data/backtest/ directory and run index
```

### Data Directory: `data/backtest/` (gitignored)

```
data/backtest/
  index.json                        # catalog of all completed runs
  2024-10-31/                       # directory named by cutoff date
    senate/
      state_errors.parquet
      county_errors.parquet
      type_errors.parquet
      chamber_errors.parquet
      summary.json
    presidential/
      state_errors.parquet
      county_errors.parquet
      type_errors.parquet
      chamber_errors.parquet
      summary.json
    governor/
      ...
  2022-10-31/
    ...
  2020-10-31/
    ...
  2018-10-31/
    ...
```

### Report Directory: `docs/backtest/` (committed)

```
docs/backtest/
  report_2024-10-31.md    # single-cycle report
  report_2022-10-31.md
  report_all.md           # cross-cycle comparison (regenerated)
```

### Data Flow

```
build_historical_inputs(year, race_types, cutoff_date)
  → HistoricalInputs
    → run_backtest(inputs)
      → BacktestRun (state/county/type predictions)
        → load_actuals(year, race_types)
          → compute_errors(run, actuals)
            → ErrorArtifact
              → ReportGenerator([artifacts])
                → docs/backtest/report_YYYY.md
```

---

## Component Specifications

### `inputs.py` — HistoricalInputs

Reconstructs the information state that existed on a given cutoff date for a given election year. All downstream components receive a `HistoricalInputs` object; none of them query raw data sources directly.

**Interface:**

```python
@dataclass
class HistoricalInputs:
    year: int
    race_types: list[str]          # "senate", "governor", "presidential"
    cutoff_date: date              # default: October 31 of year
    polls: pd.DataFrame            # polls with poll_date <= cutoff_date
    fundamentals: dict             # economic/approval signals as of cutoff month
    races: list[dict]              # race metadata: state_abbr, office, candidates, race_id

def build_historical_inputs(
    year: int,
    race_types: list[str],
    cutoff_date: date | None = None,  # defaults to October 31 of year
) -> HistoricalInputs:
    ...
```

**Poll reconstruction:**
- Source: `data/raw/fivethirtyeight/` (887MB 538 archive, covers multiple cycles)
- Filter: `poll_date <= cutoff_date` AND `cycle_year == year`
- Format: Same schema as `data/polls/polls_2026.csv` so `forecast_engine.py` accepts it without modification
- Coverage must be verified per cycle before Phase 2 implementation (see Data Sources section)

**Fundamentals reconstruction:**
- Presidential approval: Historical values from `data/raw/` approval archive (fetched by `scripts/fetch_approval_rating.py` — verify historical range)
- Economic indicators: FRED historical data (fetched by `scripts/fetch_fred_fundamentals.py` — verify historical range)
- Snapshot: Use last available value at or before `cutoff_date`
- Format: Same schema as `data/fundamentals/snapshot_2026.json`

**Race metadata:**
- For Senate: load from MEDSL historical data filtered to `year`
- For Presidential: 51 state races, standard format
- For Governor: load from Algara/Amlani (2002–2018) + MEDSL (2022+) filtered to `year`

---

### `runner.py` — BacktestRunner

Calls the existing `forecast_engine.py` with a `HistoricalInputs` object and returns structured prediction outputs. This file should contain minimal logic — its job is to translate `HistoricalInputs` into the format `forecast_engine.py` expects and translate the outputs into `BacktestRun`.

**Interface:**

```python
@dataclass
class BacktestRun:
    inputs: HistoricalInputs
    state_predictions: pd.DataFrame    # columns: state_abbr, race_id, race_type,
                                       #          pred_dem_share, pred_rep_share
    county_predictions: pd.DataFrame   # columns: fips, race_id, race_type,
                                       #          pred_dem_share
    type_predictions: pd.DataFrame     # columns: type_id, race_id, race_type,
                                       #          mean_pred_dem_share, n_counties
    chamber_prediction: dict           # {race_type: {pred_dem_seats: int,
                                       #              pred_rep_seats: int}}

def run_backtest(
    inputs: HistoricalInputs,
    use_epoch_types: bool = False,     # Phase 3 flag; raises NotImplementedError until then
) -> BacktestRun:
    ...
```

**Type assignment behavior (Phase 1–2):**
- Uses current type assignments from `data/communities/` and `data/covariance/`
- This is a known limitation: types were trained on data that includes the target year
- Documented explicitly in `BacktestRun` metadata and in every report
- Phase 3 introduces `use_epoch_types=True` to retrain types per epoch

**Forecast engine wiring:**
- Calls `forecast_engine.py` the same way `predict_2026_types.py` does, but passes `inputs.polls` and `inputs.fundamentals` instead of current-cycle data
- Does not modify `forecast_engine.py` — inputs are translated at the boundary

---

### `actuals.py` — ActualsLoader

Loads known election results for a given year and race type, normalized to the same schema as predictions so error computation is a simple join.

**Interface:**

```python
@dataclass
class Actuals:
    state: pd.DataFrame     # columns: state_abbr, race_id, race_type,
                            #          actual_dem_share, actual_rep_share
    county: pd.DataFrame    # columns: fips, race_id, race_type,
                            #          actual_dem_share
    chamber: dict           # {race_type: {actual_dem_seats: int,
                            #              actual_rep_seats: int,
                            #              actual_dem_majority: bool}}

def load_actuals(year: int, race_types: list[str]) -> Actuals:
    ...
```

**Data sources by race type:**
- Presidential: `data/raw/` MEDSL county presidential (2000–2024)
- Senate: MEDSL county Senate data (`src/assembly/fetch_medsl_county_senate.py` output)
- Governor: Algara/Amlani 2002–2018 (`data/raw/algara_amlani/`) + MEDSL 2022+ governor
- Chamber outcomes: `api/data/historical_results.json` (seat counts by cycle)

**Two-party vote share:**
All actual and predicted shares are normalized to two-party (dem / (dem + rep)) for comparability. Third-party votes are excluded from error computation with a note in the report when significant (e.g., 2016 presidential, Libertarian 3%+).

---

### `errors.py` — ErrorArtifact

Takes a `BacktestRun` and `Actuals` and computes signed and absolute errors at all four levels. This is the core analytical output.

**Interface:**

```python
@dataclass
class ErrorArtifact:
    year: int
    race_types: list[str]
    cutoff_date: date
    types_are_epoch_trained: bool   # False for Phase 1-2; True for Phase 3

    state_errors: pd.DataFrame      # see schema below
    county_errors: pd.DataFrame     # see schema below
    type_errors: pd.DataFrame       # see schema below
    chamber_errors: pd.DataFrame    # see schema below
    summary: dict                   # aggregate metrics per race_type

def compute_errors(run: BacktestRun, actuals: Actuals) -> ErrorArtifact:
    ...
```

**Error sign convention:**
```
error = predicted_dem_share - actual_dem_share
```
- Positive error = overestimated Democrats (underestimated Republicans)
- Negative error = underestimated Democrats (overestimated Republicans)

This convention is used consistently across all levels.

**State errors schema:**
```
state_abbr, race_id, race_type, office, pred_dem_share, actual_dem_share,
error, abs_error, pred_winner, actual_winner, call_correct (bool)
```

**County errors schema:**
```
fips, state_abbr, race_id, race_type, pred_dem_share, actual_dem_share,
error, abs_error
```

**Type errors schema:**
Type-level errors summarize average error across all counties of each type, for each race. Types with consistent directional errors across multiple races reveal structural model assumptions that don't hold.
```
type_id, race_id, race_type, mean_error, mean_abs_error,
n_counties, pop_weighted_error
```

**Chamber errors schema:**
```
race_type, pred_dem_seats, pred_rep_seats, actual_dem_seats, actual_rep_seats,
seat_error (pred_dem - actual_dem), pred_majority, actual_majority,
majority_correct (bool)
```

**Summary dict structure (per race_type):**
```json
{
  "senate": {
    "state_mae": 2.4,
    "state_bias": 0.8,
    "state_rmse": 3.1,
    "county_mae": 3.2,
    "county_bias": 0.6,
    "call_accuracy": 0.82,
    "majority_correct": true,
    "seat_error": 2,
    "n_races": 33,
    "n_counties": 3154
  }
}
```

---

### `report.py` — ReportGenerator

Takes a list of `ErrorArtifact` instances (one or more cycles) and produces a markdown report. Sections scale to what's available: a single-cycle report gets state tables and type bias; a multi-cycle report adds cross-cycle patterns.

**Interface:**

```python
def generate_report(
    artifacts: list[ErrorArtifact],
    output_path: Path,
    title: str | None = None,
) -> None:
    ...
```

**Single-cycle report sections:**
1. **Header** — cycle, race types covered, cutoff date, type-training note (current vs epoch)
2. **Summary table** — race_type × metric (MAE, bias, RMSE, call accuracy, seat error)
3. **Chamber outcomes** — predicted vs actual seat balance per race type
4. **State outliers** — top 10 largest absolute errors per race type, with error and direction
5. **Type systematic bias** — types with |mean_error| > 3pp, ranked by magnitude, with note on which states they concentrate in
6. **State error table** — full state-level error table, sortable by race type

**Multi-cycle report additions:**
7. **Cross-cycle bias summary** — for each race type, mean bias by cycle (are we consistently high on Dems? Getting better or worse over time?)
8. **Persistent state outliers** — states where the error is large AND consistent in direction across cycles (structural miss vs. candidate effect)
9. **Type stability** — types with consistent directional error across cycles (points toward wrong θ estimate, not candidate noise)
10. **Improvement tracking** — if re-run after a model change, show delta in MAE/bias vs prior run

**Formatting conventions:**
- Errors displayed in percentage points (pp), not fractions
- Positive errors displayed as "D+X" (overestimated Dems), negative as "R+X"
- Tables use markdown pipe format
- State winner calls displayed as "✓ Correct" / "✗ Missed"

---

### `catalog.py` — BacktestCatalog

Manages the `data/backtest/` directory. Provides a clean API for listing, loading, and comparing completed runs.

**Interface:**

```python
class BacktestCatalog:
    root: Path  # data/backtest/

    def list_runs(self) -> list[dict]:
        # Returns index.json entries: year, cutoff_date, race_types, run_timestamp

    def load_artifact(self, cutoff_date: date, race_type: str) -> ErrorArtifact:
        # Load from parquet/JSON in data/backtest/YYYY-MM-DD/race_type/

    def save_artifact(self, artifact: ErrorArtifact) -> None:
        # Write parquet/JSON, update index.json

    def load_all(self, race_type: str | None = None) -> list[ErrorArtifact]:
        # Load all completed runs, optionally filtered by race type
```

**`index.json` structure:**
```json
[
  {
    "cutoff_date": "2024-10-31",
    "year": 2024,
    "race_types": ["senate", "presidential"],
    "types_are_epoch_trained": false,
    "run_timestamp": "2026-04-10T14:32:00Z",
    "summary": {
      "senate": {"state_mae": 2.4, "state_bias": 0.8},
      "presidential": {"state_mae": 1.9, "state_bias": -0.3}
    }
  }
]
```

---

## Error Metrics Reference

| Metric | Formula | Interpretation |
|---|---|---|
| Bias | mean(error) | Systematic directional miss. Positive = overestimating Dems. |
| MAE | mean(|error|) | Average miss magnitude, in pp. Primary accuracy metric. |
| RMSE | sqrt(mean(error²)) | Penalizes large misses more than MAE. |
| Call accuracy | correct_winner / n_races | Fraction of races where predicted winner was correct. |
| Seat error | pred_dem_seats - actual_dem_seats | Chamber-level accuracy. |
| Type bias | mean(error) by type_id | Types with large type_bias have wrong θ estimates. |

All metrics computed at state level unless prefixed with "county_" or "type_".

---

## Data Sources and Coverage

| Source | Location | Cycles Covered | Notes |
|---|---|---|---|
| 538 polls archive | `data/raw/fivethirtyeight/` | Verify per cycle | 887MB; covers Senate + Governor + Presidential |
| MEDSL county presidential | `data/raw/` | 2000–2024 | Already in pipeline |
| MEDSL county Senate | assemby output | Verify per cycle | `fetch_medsl_county_senate.py` |
| Algara/Amlani governor | `data/raw/` | 2002–2018 | Harvard Dataverse |
| MEDSL governor 2022+ | assembly output | 2022+ | `fetch_2022_governor.py` |
| Historical results (chamber) | `api/data/historical_results.json` | Multiple | Seat counts by cycle |
| FRED fundamentals | scripts output | Verify historical range | `fetch_fred_fundamentals.py` |
| Presidential approval | scripts output | Verify historical range | `fetch_approval_rating.py` |

**Coverage verification is a prerequisite for Phase 2.** Before implementing 2018–2022 `HistoricalInputs`, confirm that 538 poll data, FRED historical values, and approval ratings extend to those cycles with adequate density.

---

## Phase Plan

### Phase 1 — 2024 backtest, Senate + Presidential

**Goal:** One full cycle working end-to-end. Establishes the module structure and proves the error computation is correct before adding complexity.

**Scope:**
- `HistoricalInputs` for 2024: 538 polls with `poll_date <= 2024-10-31`, FRED/approval as of October 2024
- `BacktestRunner` wired to current `forecast_engine.py`
- `ActualsLoader` for 2024 Senate + Presidential (MEDSL + `historical_results.json`)
- `ErrorArtifact` at all four levels
- `ReportGenerator` (single-cycle sections only)
- `BacktestCatalog` (save/load/index)

**Success criterion:** The 2024 Senate report produces state-level errors with plausible MAE (expecting 2–5pp range), and the chamber seat count is within 3 seats of actual.

**Explicit limitation in report:** *"Type assignments used in this backtest were trained on data including 2024 results. State-level errors are therefore a lower bound on true out-of-sample error. See Phase 3 for epoch-trained comparison."*

---

### Phase 2 — Multi-cycle (2018, 2020, 2022), all race types

**Goal:** Pattern detection across cycles. This is where the interesting findings emerge.

**Prerequisites:**
- Phase 1 complete and validated
- Data coverage confirmed for each historical cycle (polls, fundamentals)

**Scope:**
- Extend `HistoricalInputs` to 2018, 2020, 2022
- Add Governor race type to all components
- Extend `ActualsLoader` for Algara/Amlani + MEDSL Senate archives
- Add cross-cycle report sections to `ReportGenerator`
- `BacktestCatalog.load_all()` powers cross-cycle comparison

**Analytical targets:**
- Which race types does the model predict best? (Senate vs Governor vs Presidential)
- Are there states where errors are large and consistent across cycles? (structural miss)
- Are there types with consistent directional bias? (wrong θ)
- Does error correlate with poll density? (unpolled races worse than polled?)

---

### Phase 3 — Epoch-retrained types

**Goal:** Honest historical simulation. Each cycle is backtest with types trained only on data available before that election.

**Prerequisites:** Phase 2 complete. This is a significant compute investment.

**Scope:**
- For each target year Y, retrain KMeans types using shift vectors from cycles through Y-2 (excluding Y-1 since those results weren't available in October of Y)
- Store epoch models in `data/backtest/epoch_models/YYYY/`
- `BacktestRunner(use_epoch_types=True)` loads the appropriate epoch model
- Compare epoch-trained errors to current-type errors (Phase 2) for the same cycles
- Type stability metric: how much do type assignments change across epochs?

**Key question this answers:** Are our types stable across eras, or are they overfitting to the full historical record? Large differences between Phase 2 and Phase 3 errors indicate overfitting.

---

## CLI Entry Point

A dedicated `src/backtest/cli.py` (invoked as `python -m src.backtest`) handles all entry points. This keeps run and report commands unified under one interface:

```bash
# Run 2024 backtest (Senate + Presidential)
uv run python -m src.backtest run --year 2024 --race-types senate presidential

# Run all configured cycles
uv run python -m src.backtest run --all

# Generate single-cycle report from existing artifacts
uv run python -m src.backtest report --cutoff 2024-10-31

# Generate cross-cycle report from all existing artifacts
uv run python -m src.backtest report --all --output docs/backtest/report_all.md
```

`cli.py` delegates immediately to `runner.py`, `report.py`, and `catalog.py` — it contains no logic of its own.

---

## Integration with Existing Infrastructure

- **`forecast_engine.py`** — called by `runner.py` without modification. Historical inputs are translated at the boundary.
- **`data/predictions/metrics_log.jsonl`** — backtest summary metrics appended per run, enabling slow-drift detection alongside live forecast metrics.
- **`MODEL_VERIFICATION_PLAN.md` Tier 2** — the backtest harness is a separate concern from Tier 2 sanity tests. Tier 2 catches broken predictions; the backtest harness measures accuracy. Both should remain.
- **`docs/backtest/`** — committed to the repo so error patterns are versioned alongside code changes.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Error sign convention | `predicted - actual` | Positive = overestimated Dems. Consistent across all levels. |
| Two-party normalization | Always | Makes errors comparable across different third-party environments. |
| Information cutoff default | October 31 | Captures the last major polling period before Election Day. Configurable. |
| Type assignments Phase 1–2 | Current (not epoch-trained) | Faster to build; limitation is documented in every report. |
| Output format | Parquet (errors) + JSON (summary) | Parquet for large county/type tables; JSON for lightweight summary loading. |
| Report location | `docs/backtest/` | Committed alongside code so error patterns are versioned. |
| Artifact location | `data/backtest/` | Gitignored; generated from code. |
| Forecast engine coupling | Thin translation layer | `runner.py` translates inputs; `forecast_engine.py` is unchanged. Backtest stays in sync with live forecast automatically. |
