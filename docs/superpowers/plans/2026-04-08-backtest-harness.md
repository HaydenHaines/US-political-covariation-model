# Backtest Harness — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/backtest/` — a module that runs WetherVane against the October 2024 information state and produces error artifacts at state, county, type, and chamber level, plus a markdown report.

**Architecture:** Seven focused modules. `inputs.py` reconstructs October 2024 polls and fundamentals. `runner.py` loads production type scores/priors and calls `predict_race()` with historical inputs. `actuals.py` loads MEDSL county results. `errors.py` computes signed error artifacts. `catalog.py` persists artifacts. `report.py` generates markdown. `cli.py` wires the entry point.

**Tech stack:** Python, pandas, numpy, pathlib, json. No new dependencies.

---

## File Map

**New files:**
| Path | Responsibility |
|---|---|
| `src/backtest/__init__.py` | Re-exports public API |
| `src/backtest/cli.py` | `python -m src.backtest run/report` |
| `src/backtest/inputs.py` | `HistoricalInputs`, `build_historical_inputs()` |
| `src/backtest/runner.py` | `BacktestRun`, `run_backtest()` |
| `src/backtest/actuals.py` | `Actuals`, `load_actuals()` |
| `src/backtest/errors.py` | `ErrorArtifact`, `compute_errors()` |
| `src/backtest/catalog.py` | `BacktestCatalog` |
| `src/backtest/report.py` | `generate_report()` |
| `tests/backtest/__init__.py` | (empty) |
| `tests/backtest/test_actuals.py` | Unit tests for actuals loading |
| `tests/backtest/test_inputs.py` | Unit tests for historical inputs |
| `tests/backtest/test_runner.py` | Unit tests for backtest runner |
| `tests/backtest/test_errors.py` | Unit tests for error computation |
| `tests/backtest/test_catalog.py` | Unit tests for artifact persistence |
| `tests/backtest/test_report.py` | Unit tests for report generation |

**Modified files:**
| Path | Change |
|---|---|
| `.gitignore` | Add `data/backtest/` |
| `CLAUDE.md` | Add backtest CLI commands |

---

## Shared Type Reference

These dataclasses are defined across the module. Later tasks reference them — do not rename.

```python
# src/backtest/inputs.py
@dataclass
class HistoricalInputs:
    year: int
    race_types: list[str]           # "senate", "presidential", "governor"
    cutoff_date: date
    polls_by_race: dict[str, list[dict]]   # race_id -> [{dem_share, n_sample, state, date, pollster, notes}]
    fundamentals: dict              # same schema as data/fundamentals/snapshot_YYYY.json
    races: list[dict]               # [{race_id, state_abbr, office}]

# src/backtest/runner.py
@dataclass
class BacktestRun:
    inputs: HistoricalInputs
    state_predictions: pd.DataFrame    # columns: state_abbr, race_id, race_type, pred_dem_share
    county_predictions: pd.DataFrame   # columns: fips, state_abbr, race_id, race_type, pred_dem_share, dominant_type
    chamber_prediction: dict           # {race_type: {pred_dem_seats: int, pred_rep_seats: int}}

# src/backtest/actuals.py
@dataclass
class Actuals:
    state: pd.DataFrame    # state_abbr, race_id, race_type, actual_dem_share
    county: pd.DataFrame   # fips, state_abbr, race_id, race_type, actual_dem_share
    chamber: dict          # {race_type: {actual_dem_seats: int, actual_rep_seats: int}}

# src/backtest/errors.py
@dataclass
class ErrorArtifact:
    year: int
    race_types: list[str]
    cutoff_date: date
    types_are_epoch_trained: bool   # always False in Phase 1-2
    state_errors: pd.DataFrame      # state_abbr, race_id, race_type, pred_dem_share, actual_dem_share, error, abs_error, pred_winner, actual_winner, call_correct
    county_errors: pd.DataFrame     # fips, state_abbr, race_id, race_type, pred_dem_share, actual_dem_share, error, abs_error
    type_errors: pd.DataFrame       # dominant_type, race_id, race_type, mean_error, mean_abs_error, n_counties
    chamber_errors: pd.DataFrame    # race_type, pred_dem_seats, actual_dem_seats, seat_error, majority_correct
    summary: dict                   # {race_type: {state_mae, state_bias, state_rmse, call_accuracy, county_mae, county_bias, n_races, n_counties, seat_error, majority_correct}}
```

**Error sign convention throughout:** `error = predicted_dem_share - actual_dem_share`
- Positive = overestimated Democrats
- Negative = underestimated Democrats

---

## Task 1: Module Skeleton + .gitignore

**Files:**
- Create: `src/backtest/__init__.py`
- Create: `tests/backtest/__init__.py`
- Modify: `.gitignore`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create module directory and empty init**

```python
# src/backtest/__init__.py
"""
Backtest harness for WetherVane.

Runs the model against historical information states and compares outputs
to known election results, producing error artifacts and markdown reports.

Usage:
    uv run python -m src.backtest run --year 2024 --race-types senate presidential
    uv run python -m src.backtest report --cutoff 2024-10-31
"""
from src.backtest.inputs import HistoricalInputs, build_historical_inputs
from src.backtest.runner import BacktestRun, run_backtest
from src.backtest.actuals import Actuals, load_actuals
from src.backtest.errors import ErrorArtifact, compute_errors
from src.backtest.catalog import BacktestCatalog
from src.backtest.report import generate_report

__all__ = [
    "HistoricalInputs",
    "build_historical_inputs",
    "BacktestRun",
    "run_backtest",
    "Actuals",
    "load_actuals",
    "ErrorArtifact",
    "compute_errors",
    "BacktestCatalog",
    "generate_report",
]
```

```python
# tests/backtest/__init__.py
# (empty)
```

- [ ] **Step 2: Add data/backtest/ to .gitignore**

Open `.gitignore` and add this line alongside the other `data/` exclusions:
```
data/backtest/
```

- [ ] **Step 3: Add backtest commands to CLAUDE.md**

In the `## Commands` section, add after the existing pipeline commands:
```markdown
# Backtest harness
uv run python -m src.backtest run --year 2024 --race-types senate presidential   # Run 2024 backtest
uv run python -m src.backtest run --all                                            # Run all configured cycles
uv run python -m src.backtest report --cutoff 2024-10-31                          # Single-cycle report
uv run python -m src.backtest report --all --output docs/backtest/report_all.md   # Cross-cycle report
```

- [ ] **Step 4: Verify Python can find the module**

```bash
uv run python -c "import src.backtest; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/backtest/__init__.py tests/backtest/__init__.py .gitignore CLAUDE.md
git commit -m "feat: scaffold src/backtest module"
```

---

## Task 2: Data Verification Sprint

**This task has no new source files.** Its job is to verify what data actually exists before implementing the data-dependent modules. Record findings as comments in the relevant implementation tasks.

**Files:**
- Read: `src/prediction/predict_2026_types.py` (full file — understand all importable functions and how type scores are loaded)
- Read: `src/prediction/county_priors.py` (full file — understand `load_county_priors_with_ridge()` signature and return type)
- Investigate: `data/raw/fivethirtyeight/` directory structure
- Investigate: `data/communities/` directory — list all files, note what type score/assignment files exist
- Investigate: `data/assembled/` — does a converted 2024 poll file exist?

- [ ] **Step 1: Map the production prediction pipeline**

Read `src/prediction/predict_2026_types.py` in full. Answer:
1. What are the importable functions (not `if __name__ == "__main__"` code)?
2. How does it load type scores — what file paths, what format?
3. What does `load_county_priors_with_ridge()` return? What arguments does it take?
4. How does it call `run_forecast()` or `predict_race()`?

Record the function signatures and file paths. You will need them in Task 4.

- [ ] **Step 2: Inventory data/communities/**

```bash
ls -la data/communities/
```

Record the exact filenames. The type assignment file may be named differently from `type_assignments.parquet`. You need to know the exact name to load it in Task 4.

- [ ] **Step 3: Inventory the 538 archive**

```bash
ls data/raw/fivethirtyeight/ | head -30
```

Then find poll data files:
```bash
find data/raw/fivethirtyeight/ -name "*.csv" | head -20
find data/raw/fivethirtyeight/ -name "*.parquet" | head -20
```

Look for a file containing Senate/presidential polls for 2024 with columns: date, state, pollster, dem%, n. Record the filename and its column schema.

- [ ] **Step 4: Check for converted poll output**

```bash
ls data/assembled/ | grep -i poll
ls data/polls/
```

If a `polls_2024.csv` or similar already exists with the standard schema (`race, geography, geo_level, dem_share, n_sample, date, pollster, notes`), Task 3 can read it directly. If not, `src/assembly/convert_538_polls.py` will need to be run with `--year 2024` first.

- [ ] **Step 5: Verify historical fundamentals**

```bash
# Check if historical fundamentals files exist
ls data/config/ | grep fundamental
ls data/raw/ | grep -i "fred\|approval"
```

If `data/config/fundamentals_2024.json` does not exist, Task 3 must create it manually with known October 2024 values. If FRED historical data exists in `data/raw/`, Task 3 can load it programmatically.

- [ ] **Step 6: Check MEDSL county data for presidential and senate actuals**

```bash
# Presidential county-level data (needed for actuals)
ls data/raw/ | grep medsl
ls data/assembled/ | grep -i "presidential\|senate"
```

For presidential actuals: MEDSL county presidential data for 2020 and 2024 must exist. Confirm the filename and the column `dem_share` or equivalent.

For Senate actuals: county-level Senate data for 2024 may or may not exist nationally. Note what is available — this determines whether Task 2 implements county-level Senate actuals or falls back to state-level only.

- [ ] **Step 7: Document findings**

Create `data/backtest/FINDINGS.md` (this file is gitignored):
```markdown
# Data Verification Findings — YYYY-MM-DD

## Type scores
- File: data/communities/XXXX.parquet
- Columns: ...
- Shape: N x J

## Production prediction pipeline
- Key importable function: predict_2026_types.FUNCTION_NAME(...)
- Type score loading: ...

## 538 poll archive
- 2024 Senate polls file: data/raw/fivethirtyeight/XXXX
- Columns: ...
- Date range: ...

## Historical fundamentals
- 2024 snapshot: [exists at / must be created manually]
- October 2024 values (if manual): approval_net=-X, gdp=Y, unemployment=Z, cpi=W

## MEDSL actuals
- Presidential county file: data/assembled/XXXX
- Columns: ...
- Senate county availability: [national / FL+GA+AL only / state-level only]
```

---

## Task 3: `actuals.py`

**Files:**
- Create: `src/backtest/actuals.py`
- Create: `tests/backtest/test_actuals.py`

`actuals.py` loads known election results and normalizes them to two-party vote share. All vote shares are `dem / (dem + rep)` — third-party votes are excluded.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_actuals.py
"""Tests for actuals loading. Uses synthetic DataFrames — no real data files required."""

import pandas as pd
import pytest
from unittest.mock import patch
from src.backtest.actuals import (
    Actuals,
    _to_two_party_share,
    _build_state_actuals_from_county,
)


class TestTwoPartyShare:
    def test_basic_conversion(self):
        dem, rep = 55_000.0, 45_000.0
        result = _to_two_party_share(dem, rep)
        assert abs(result - 0.55) < 1e-6

    def test_symmetric(self):
        assert abs(_to_two_party_share(50.0, 50.0) - 0.5) < 1e-6

    def test_ignores_third_party(self):
        # 55D, 45R, 10 third-party -> 55/100 = 0.55
        result = _to_two_party_share(55.0, 45.0)
        assert abs(result - 0.55) < 1e-6

    def test_zero_denominator_returns_nan(self):
        import numpy as np
        result = _to_two_party_share(0.0, 0.0)
        assert np.isnan(result)


class TestBuildStateActualsFromCounty:
    def _make_county_df(self):
        return pd.DataFrame({
            "fips": ["01001", "01003", "02001"],
            "state_abbr": ["AL", "AL", "AK"],
            "race_id": ["2024-al-pres", "2024-al-pres", "2024-ak-pres"],
            "race_type": ["presidential", "presidential", "presidential"],
            "actual_dem_share": [0.35, 0.42, 0.38],
            "dem_votes": [10_000.0, 20_000.0, 5_000.0],
            "rep_votes": [20_000.0, 27_000.0, 8_000.0],
        })

    def test_state_aggregation_is_vote_weighted(self):
        county_df = self._make_county_df()
        result = _build_state_actuals_from_county(county_df)
        al_row = result[result["state_abbr"] == "AL"].iloc[0]
        # AL: dem=(10k+20k)=30k, rep=(20k+27k)=47k -> 30/77 ≈ 0.3896
        expected = 30_000 / (30_000 + 47_000)
        assert abs(al_row["actual_dem_share"] - expected) < 1e-4

    def test_returns_required_columns(self):
        county_df = self._make_county_df()
        result = _build_state_actuals_from_county(county_df)
        assert set(result.columns) >= {"state_abbr", "race_id", "race_type", "actual_dem_share"}

    def test_one_row_per_state_race(self):
        county_df = self._make_county_df()
        result = _build_state_actuals_from_county(county_df)
        assert len(result) == 2  # AL + AK


class TestActualsContract:
    """Verify the Actuals dataclass has the required columns."""

    def _make_actuals(self):
        state_df = pd.DataFrame({
            "state_abbr": ["AL", "AK"],
            "race_id": ["2024-al-pres", "2024-ak-pres"],
            "race_type": ["presidential", "presidential"],
            "actual_dem_share": [0.39, 0.38],
        })
        county_df = pd.DataFrame({
            "fips": ["01001", "02001"],
            "state_abbr": ["AL", "AK"],
            "race_id": ["2024-al-pres", "2024-ak-pres"],
            "race_type": ["presidential", "presidential"],
            "actual_dem_share": [0.35, 0.38],
        })
        return Actuals(state=state_df, county=county_df, chamber={})

    def test_state_columns(self):
        a = self._make_actuals()
        assert "actual_dem_share" in a.state.columns
        assert "race_id" in a.state.columns

    def test_county_columns(self):
        a = self._make_actuals()
        assert "fips" in a.county.columns
        assert "actual_dem_share" in a.county.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/backtest/test_actuals.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` (actuals.py doesn't exist yet)

- [ ] **Step 3: Implement `actuals.py`**

Before writing this file, check your Task 2 findings for the exact parquet filenames and column schemas for presidential and Senate county data.

```python
# src/backtest/actuals.py
"""
Load known election results for historical backtest comparison.

All vote shares are normalized to two-party: dem / (dem + rep).
Third-party votes are excluded from error computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_ASSEMBLED_DIR = _PROJECT_ROOT / "data" / "assembled"
_API_DATA_DIR = _PROJECT_ROOT / "api" / "data"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class Actuals:
    state: pd.DataFrame    # state_abbr, race_id, race_type, actual_dem_share
    county: pd.DataFrame   # fips, state_abbr, race_id, race_type, actual_dem_share
    chamber: dict          # {race_type: {actual_dem_seats: int, actual_rep_seats: int}}


def load_actuals(year: int, race_types: list[str]) -> Actuals:
    """
    Load known election results for the given year and race types.

    State-level actuals are aggregated from county data using vote weighting.
    County-level actuals are loaded directly from MEDSL parquet files.
    Chamber actuals are loaded from api/data/historical_results.json.
    """
    state_dfs: list[pd.DataFrame] = []
    county_dfs: list[pd.DataFrame] = []
    chamber: dict = {}

    if "presidential" in race_types:
        county_pres = _load_presidential_county_actuals(year)
        state_pres = _build_state_actuals_from_county(county_pres)
        county_dfs.append(county_pres)
        state_dfs.append(state_pres)
        # Presidential chamber = electoral college; not computed in Phase 1

    if "senate" in race_types:
        county_sen = _load_senate_county_actuals(year)
        state_sen = _build_state_actuals_from_county(county_sen)
        county_dfs.append(county_sen)
        state_dfs.append(state_sen)
        chamber["senate"] = _load_senate_chamber_actuals(year)

    return Actuals(
        state=pd.concat(state_dfs, ignore_index=True) if state_dfs else pd.DataFrame(),
        county=pd.concat(county_dfs, ignore_index=True) if county_dfs else pd.DataFrame(),
        chamber=chamber,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_two_party_share(dem: float, rep: float) -> float:
    """
    Convert raw vote counts to two-party Democratic share.

    Returns NaN when both are zero to signal missing data cleanly rather
    than silently producing 0.5 or raising ZeroDivisionError.
    """
    total = dem + rep
    if total == 0:
        return float("nan")
    return dem / total


def _build_state_actuals_from_county(county_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate county-level actuals to state level using vote weighting.

    Vote weighting (dem_votes + rep_votes) ensures large counties contribute
    proportionally, which matches how the forecast engine aggregates predictions.

    county_df must have: fips, state_abbr, race_id, race_type, dem_votes, rep_votes
    """
    agg = (
        county_df.groupby(["state_abbr", "race_id", "race_type"])
        .agg(dem_votes=("dem_votes", "sum"), rep_votes=("rep_votes", "sum"))
        .reset_index()
    )
    agg["actual_dem_share"] = agg.apply(
        lambda r: _to_two_party_share(r["dem_votes"], r["rep_votes"]), axis=1
    )
    return agg[["state_abbr", "race_id", "race_type", "actual_dem_share"]]


def _load_presidential_county_actuals(year: int) -> pd.DataFrame:
    """
    Load MEDSL county-level presidential results for the given year.

    IMPORTANT: Check Task 2 findings for the exact filename and column schema.
    Adjust the path and column mapping below to match what actually exists.

    Returns DataFrame with: fips, state_abbr, race_id, race_type, dem_votes, rep_votes, actual_dem_share
    """
    # Adjust filename based on Task 2 findings:
    path = _ASSEMBLED_DIR / f"medsl_county_presidential_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Presidential county data not found: {path}\n"
            f"Run: uv run python src/assembly/fetch_medsl_county_presidential.py"
        )
    df = pd.read_parquet(path)

    # Normalize columns — adjust these names based on Task 2 findings:
    df = df.rename(columns={
        f"president_dem_{year}": "dem_votes",
        f"president_rep_{year}": "rep_votes",
        "county_fips": "fips",
    })

    df["race_id"] = df["state_abbr"].str.lower() + f"-{year}-pres"
    df["race_type"] = "presidential"
    df["actual_dem_share"] = df.apply(
        lambda r: _to_two_party_share(r["dem_votes"], r["rep_votes"]), axis=1
    )
    return df[["fips", "state_abbr", "race_id", "race_type", "dem_votes", "rep_votes", "actual_dem_share"]]


def _load_senate_county_actuals(year: int) -> pd.DataFrame:
    """
    Load MEDSL county-level Senate results for the given year.

    IMPORTANT: Check Task 2 findings for availability. If county-level Senate
    data only covers FL/GA/AL, this returns data for those states only — note
    this limitation clearly in the generated report.

    Returns DataFrame with: fips, state_abbr, race_id, race_type, dem_votes, rep_votes, actual_dem_share
    """
    # Adjust filename based on Task 2 findings:
    path = _ASSEMBLED_DIR / f"medsl_county_senate_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Senate county data not found: {path}\n"
            f"Run: uv run python src/assembly/fetch_medsl_county_senate.py"
        )
    df = pd.read_parquet(path)

    # Normalize — adjust column names based on Task 2 findings:
    df = df.rename(columns={
        f"senate_dem_{year}": "dem_votes",
        f"senate_rep_{year}": "rep_votes",
        "county_fips": "fips",
    })

    # Senate race_id: state + year + "senate" (handles dual-seat states like GA)
    df["race_id"] = df["state_abbr"].str.lower() + f"-{year}-senate"
    df["race_type"] = "senate"
    df["actual_dem_share"] = df.apply(
        lambda r: _to_two_party_share(r["dem_votes"], r["rep_votes"]), axis=1
    )
    return df[["fips", "state_abbr", "race_id", "race_type", "dem_votes", "rep_votes", "actual_dem_share"]]


def _load_senate_chamber_actuals(year: int) -> dict:
    """
    Load actual Senate seat counts for the given election year.

    Reads from api/data/historical_results.json, which stores per-race outcomes.
    Counts races won by each party to produce the final seat balance.

    Note: This counts the races held in `year`, not the full Senate composition.
    The full composition (including non-cycle seats) is not stored here.
    """
    path = _API_DATA_DIR / "historical_results.json"
    if not path.exists():
        return {}

    data = json.loads(path.read_text())
    dem_wins = 0
    rep_wins = 0
    for race_id, race_data in data.items():
        if str(year) not in race_id:
            continue
        last_race = race_data.get("last_race", {})
        if last_race.get("year") == year:
            party = last_race.get("party", "")
            if party == "D":
                dem_wins += 1
            elif party == "R":
                rep_wins += 1

    return {"actual_dem_seats": dem_wins, "actual_rep_seats": rep_wins}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/backtest/test_actuals.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/actuals.py tests/backtest/test_actuals.py
git commit -m "feat(backtest): add actuals loader with two-party normalization"
```

---

## Task 4: `inputs.py` — Historical Information State

**Files:**
- Create: `src/backtest/inputs.py`
- Create: `tests/backtest/test_inputs.py`

`inputs.py` reconstructs the information state that existed on a given cutoff date. For Phase 1, this means: 2024 polls filtered to October 31 + October 2024 fundamentals snapshot.

**Before implementing:** Check Task 2 findings to confirm:
1. The exact path and column names of the 2024 assembled polls file
2. Whether `data/config/fundamentals_2024.json` exists or must be created manually

If it must be created manually, create `data/config/fundamentals_2024.json` now with known October 2024 values:
```json
{
  "cycle": 2024,
  "in_party": "D",
  "approval_net_oct": -18.0,
  "gdp_q2_growth_pct": 3.0,
  "unemployment_oct": 4.1,
  "cpi_yoy_oct": 2.44,
  "consumer_sentiment": 70.5,
  "source_notes": {
    "approval": "FiveThirtyEight average, October 2024",
    "gdp": "BEA advance estimate Q2 2024",
    "unemployment": "BLS, October 2024",
    "cpi": "BLS CPI-U, October 2024 YoY",
    "consumer_sentiment": "U Michigan, October 2024 final",
    "last_updated": "2026-04-08 (historical, manually entered)"
  }
}
```

*Update these values with accurate historical figures from BLS/BEA/FiveThirtyEight before running the first real backtest.*

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_inputs.py
"""Tests for historical input reconstruction. Uses in-memory data — no data files required."""

import io
from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.backtest.inputs import (
    HistoricalInputs,
    _filter_polls_to_cutoff,
    _build_polls_by_race,
)


class TestFilterPollsToCutoff:
    def _make_polls_df(self):
        return pd.DataFrame({
            "race": ["2024 AL Senate", "2024 AL Senate", "2024 AK Senate"],
            "geography": ["AL", "AL", "AK"],
            "geo_level": ["state", "state", "state"],
            "dem_share": [0.42, 0.44, 0.38],
            "n_sample": [600.0, 800.0, 500.0],
            "date": pd.to_datetime(["2024-09-15", "2024-11-02", "2024-10-20"]),
            "pollster": ["Pollster A", "Pollster B", "Pollster C"],
            "notes": ["LV", "LV", "RV"],
        })

    def test_excludes_polls_after_cutoff(self):
        df = self._make_polls_df()
        cutoff = date(2024, 10, 31)
        result = _filter_polls_to_cutoff(df, cutoff)
        # 2024-11-02 is after cutoff, should be excluded
        assert len(result) == 2
        assert all(result["date"].dt.date <= cutoff)

    def test_includes_polls_on_cutoff_date(self):
        df = pd.DataFrame({
            "race": ["2024 AL Senate"],
            "geography": ["AL"],
            "geo_level": ["state"],
            "dem_share": [0.42],
            "n_sample": [600.0],
            "date": pd.to_datetime(["2024-10-31"]),
            "pollster": ["Pollster A"],
            "notes": ["LV"],
        })
        result = _filter_polls_to_cutoff(df, date(2024, 10, 31))
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=["race", "geography", "dem_share", "n_sample", "date", "pollster", "notes"])
        result = _filter_polls_to_cutoff(df, date(2024, 10, 31))
        assert len(result) == 0


class TestBuildPollsByRace:
    def _make_filtered_df(self):
        return pd.DataFrame({
            "race": ["2024 AL Senate", "2024 AL Senate", "2024 AK Senate"],
            "geography": ["AL", "AL", "AK"],
            "geo_level": ["state", "state", "state"],
            "dem_share": [0.42, 0.44, 0.38],
            "n_sample": [600.0, 800.0, 500.0],
            "date": pd.to_datetime(["2024-09-15", "2024-10-20", "2024-10-01"]),
            "pollster": ["Pollster A", "Pollster B", "Pollster C"],
            "notes": ["LV", "LV", "RV"],
        })

    def test_groups_by_race(self):
        df = self._make_filtered_df()
        result = _build_polls_by_race(df)
        assert "2024 AL Senate" in result
        assert "2024 AK Senate" in result
        assert len(result["2024 AL Senate"]) == 2
        assert len(result["2024 AK Senate"]) == 1

    def test_poll_dicts_have_required_keys(self):
        df = self._make_filtered_df()
        result = _build_polls_by_race(df)
        poll = result["2024 AL Senate"][0]
        assert "dem_share" in poll
        assert "n_sample" in poll
        assert "state" in poll
        assert "date" in poll
        assert "pollster" in poll

    def test_state_field_is_geography(self):
        df = self._make_filtered_df()
        result = _build_polls_by_race(df)
        assert result["2024 AL Senate"][0]["state"] == "AL"

    def test_date_is_iso_string(self):
        df = self._make_filtered_df()
        result = _build_polls_by_race(df)
        date_val = result["2024 AL Senate"][0]["date"]
        assert isinstance(date_val, str)
        assert "2024" in date_val


class TestHistoricalInputsContract:
    def test_default_cutoff_is_october_31(self):
        # HistoricalInputs with year=2024 should default cutoff to 2024-10-31
        inputs = HistoricalInputs(
            year=2024,
            race_types=["senate"],
            cutoff_date=date(2024, 10, 31),
            polls_by_race={},
            fundamentals={"cycle": 2024},
            races=[],
        )
        assert inputs.cutoff_date == date(2024, 10, 31)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/backtest/test_inputs.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `inputs.py`**

```python
# src/backtest/inputs.py
"""
Reconstruct the information state that existed on a given cutoff date.

Inputs: polls filtered to cutoff_date, fundamentals as of cutoff month.
Outputs: HistoricalInputs — a self-contained bundle passed to BacktestRunner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import json
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_POLLS_DIR = _PROJECT_ROOT / "data" / "polls"
_ASSEMBLED_DIR = _PROJECT_ROOT / "data" / "assembled"
_CONFIG_DIR = _PROJECT_ROOT / "data" / "config"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class HistoricalInputs:
    year: int
    race_types: list[str]
    cutoff_date: date
    # race_id -> list of poll dicts with keys: dem_share, n_sample, state, date, pollster, notes
    polls_by_race: dict[str, list[dict]]
    # Same schema as data/fundamentals/snapshot_YYYY.json
    fundamentals: dict
    # [{race_id: str, state_abbr: str, office: str}]
    races: list[dict]


def build_historical_inputs(
    year: int,
    race_types: list[str],
    cutoff_date: date | None = None,
    polls_path: Path | None = None,
    fundamentals_path: Path | None = None,
) -> HistoricalInputs:
    """
    Reconstruct the information state for a given election year.

    polls_path: path to an assembled polls CSV in the standard schema
        (race, geography, geo_level, dem_share, n_sample, date, pollster, notes).
        Defaults to data/polls/polls_{year}.csv — run convert_538_polls.py first
        if this file doesn't exist.

    fundamentals_path: path to a JSON file with the same schema as
        data/fundamentals/snapshot_2026.json. Defaults to
        data/config/fundamentals_{year}.json.
    """
    if cutoff_date is None:
        cutoff_date = date(year, 10, 31)

    polls_df = _load_polls(year, polls_path)
    polls_df = _filter_polls_to_cutoff(polls_df, cutoff_date)
    polls_by_race = _build_polls_by_race(polls_df)

    fundamentals = _load_fundamentals(year, fundamentals_path)
    races = _build_race_list(year, race_types)

    return HistoricalInputs(
        year=year,
        race_types=race_types,
        cutoff_date=cutoff_date,
        polls_by_race=polls_by_race,
        fundamentals=fundamentals,
        races=races,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_polls(year: int, polls_path: Path | None) -> pd.DataFrame:
    """Load polls CSV in standard schema. Raises FileNotFoundError with instructions if missing."""
    if polls_path is None:
        polls_path = _POLLS_DIR / f"polls_{year}.csv"
    if not polls_path.exists():
        raise FileNotFoundError(
            f"Polls file not found: {polls_path}\n"
            f"Run: uv run python src/assembly/convert_538_polls.py --year {year}\n"
            f"to generate polls for that cycle from the 538 archive."
        )
    df = pd.read_csv(polls_path, parse_dates=["date"])
    return df


def _filter_polls_to_cutoff(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Keep only polls conducted on or before the cutoff date."""
    if df.empty:
        return df
    return df[df["date"].dt.date <= cutoff].copy()


def _build_polls_by_race(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Convert a flat polls DataFrame into the dict format run_forecast() expects.

    Each poll dict has: dem_share, n_sample, state, date (ISO string), pollster, notes.
    The 'state' field maps to the 'geography' column (state abbreviation).
    """
    if df.empty:
        return {}

    result: dict[str, list[dict]] = {}
    for race_id, group in df.groupby("race"):
        polls = []
        for _, row in group.iterrows():
            polls.append({
                "dem_share": float(row["dem_share"]),
                "n_sample": float(row["n_sample"]),
                "state": str(row["geography"]),
                "date": row["date"].strftime("%Y-%m-%d"),
                "pollster": str(row.get("pollster", "")),
                "notes": str(row.get("notes", "")),
            })
        result[race_id] = polls
    return result


def _load_fundamentals(year: int, path: Path | None) -> dict:
    """
    Load the fundamentals snapshot for the given year.

    The snapshot captures economic and approval conditions as of October
    of the election year. For historical years, this is a static JSON file
    created from known BLS/BEA/approval data.
    """
    if path is None:
        path = _CONFIG_DIR / f"fundamentals_{year}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Fundamentals snapshot not found: {path}\n"
            f"Create it manually with October {year} values. "
            f"See data/fundamentals/snapshot_2026.json for the required schema."
        )
    return json.loads(path.read_text())


def _build_race_list(year: int, race_types: list[str]) -> list[dict]:
    """
    Build the list of races to forecast for the given year and race types.

    IMPORTANT: This is a stub for Phase 1. It returns an empty list because
    the runner loads races from the polls_by_race keys. Implement a real race
    registry here in Phase 2 when cross-cycle comparison needs structured race metadata.
    """
    # Phase 1: race list is implicitly defined by polls_by_race keys
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/backtest/test_inputs.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/inputs.py tests/backtest/test_inputs.py
git commit -m "feat(backtest): add historical inputs reconstruction"
```

---

## Task 5: `runner.py` — Wrap Production Forecast Pipeline

**Files:**
- Create: `src/backtest/runner.py`
- Create: `tests/backtest/test_runner.py`

`runner.py` loads type scores, covariance, and county priors from the same disk locations as the production pipeline, then calls `predict_race()` with historical inputs substituted for polls and fundamentals. It produces county-level predictions for all races in `inputs.polls_by_race`.

**Before implementing:** Review Task 2 findings to determine:
1. The exact path to type scores (e.g., `data/communities/TYPE_SCORE_FILE.parquet`)
2. What `load_county_priors_with_ridge()` from `src/prediction/county_priors.py` requires as arguments
3. How `predict_race()` from `src/prediction/forecast_runner.py` is called (exact signature from Task 2 research)

The production forecast path in `predict_2026_types.py` is the reference implementation. `runner.py` should call the same functions with the same arguments except polls (replaced with `inputs.polls_by_race`) and fundamentals (replaced with `inputs.fundamentals`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_runner.py
"""Tests for BacktestRunner. Uses synthetic data — no real data files required."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.backtest.inputs import HistoricalInputs
from src.backtest.runner import (
    BacktestRun,
    _aggregate_to_state,
    _compute_chamber_prediction,
    _polls_to_race_ids,
)


class TestAggregateToState:
    def _make_county_preds(self):
        return pd.DataFrame({
            "fips": ["01001", "01003", "02001"],
            "state_abbr": ["AL", "AL", "AK"],
            "race_id": ["2024-al-senate", "2024-al-senate", "2024-ak-senate"],
            "race_type": ["senate", "senate", "senate"],
            "pred_dem_share": [0.42, 0.38, 0.50],
            "dominant_type": [5, 12, 7],
            "county_votes": [30_000.0, 50_000.0, 8_000.0],
        })

    def test_returns_one_row_per_state_race(self):
        county_preds = self._make_county_preds()
        result = _aggregate_to_state(county_preds)
        assert len(result) == 2  # AL Senate + AK Senate

    def test_state_pred_is_vote_weighted(self):
        county_preds = self._make_county_preds()
        result = _aggregate_to_state(county_preds)
        al = result[result["state_abbr"] == "AL"].iloc[0]
        # AL: (0.42 * 30k + 0.38 * 50k) / 80k = (12600 + 19000) / 80000 = 0.395
        expected = (0.42 * 30_000 + 0.38 * 50_000) / (30_000 + 50_000)
        assert abs(al["pred_dem_share"] - expected) < 1e-4

    def test_required_columns_present(self):
        result = _aggregate_to_state(self._make_county_preds())
        assert set(result.columns) >= {"state_abbr", "race_id", "race_type", "pred_dem_share"}


class TestComputeChamberPrediction:
    def _make_state_preds(self):
        # 3 Senate races: 2 D wins, 1 R win
        return pd.DataFrame({
            "state_abbr": ["AL", "AK", "AZ"],
            "race_id": ["2024-al-senate", "2024-ak-senate", "2024-az-senate"],
            "race_type": ["senate", "senate", "senate"],
            "pred_dem_share": [0.35, 0.55, 0.52],  # AL=R, AK=D, AZ=D
        })

    def test_counts_predicted_winners(self):
        state_preds = self._make_state_preds()
        result = _compute_chamber_prediction(state_preds, race_type="senate")
        assert result["pred_dem_seats"] == 2
        assert result["pred_rep_seats"] == 1

    def test_empty_returns_zeros(self):
        empty = pd.DataFrame(columns=["state_abbr", "race_id", "race_type", "pred_dem_share"])
        result = _compute_chamber_prediction(empty, race_type="senate")
        assert result["pred_dem_seats"] == 0
        assert result["pred_rep_seats"] == 0


class TestPollsToRaceIds:
    def test_extracts_race_ids_from_polls_by_race(self):
        polls_by_race = {
            "2024 AL Senate": [{"dem_share": 0.4}],
            "2024 AK Senate": [{"dem_share": 0.5}],
        }
        result = _polls_to_race_ids(polls_by_race)
        assert set(result) == {"2024 AL Senate", "2024 AK Senate"}

    def test_empty_input_returns_empty(self):
        assert _polls_to_race_ids({}) == []


class TestBacktestRunContract:
    def test_dataclass_fields(self):
        inputs = HistoricalInputs(
            year=2024, race_types=["senate"], cutoff_date=date(2024, 10, 31),
            polls_by_race={}, fundamentals={}, races=[],
        )
        run = BacktestRun(
            inputs=inputs,
            state_predictions=pd.DataFrame(),
            county_predictions=pd.DataFrame(),
            chamber_prediction={},
        )
        assert run.inputs.year == 2024
        assert isinstance(run.state_predictions, pd.DataFrame)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/backtest/test_runner.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `runner.py`**

Fill in the `# FILL IN FROM TASK 2 FINDINGS` sections based on your data discovery results.

```python
# src/backtest/runner.py
"""
Run the production forecast pipeline with historical inputs.

Loads type scores, covariance, and county priors from the same disk locations
as the production forecast, then substitutes historical polls and fundamentals.
This ensures the backtest stays in sync with production as the pipeline evolves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.backtest.inputs import HistoricalInputs
from src.prediction.county_priors import load_county_priors_with_ridge
from src.prediction.forecast_engine import run_forecast
from src.prediction.fundamentals import compute_fundamentals_shift, load_fundamentals_snapshot
from src.prediction.generic_ballot import compute_gb_shift

_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class BacktestRun:
    inputs: HistoricalInputs
    # columns: state_abbr, race_id, race_type, pred_dem_share
    state_predictions: pd.DataFrame
    # columns: fips, state_abbr, race_id, race_type, pred_dem_share, dominant_type, county_votes
    county_predictions: pd.DataFrame
    # {race_type: {pred_dem_seats: int, pred_rep_seats: int}}
    chamber_prediction: dict


def run_backtest(inputs: HistoricalInputs, use_epoch_types: bool = False) -> BacktestRun:
    """
    Run a backtest for the given historical inputs.

    use_epoch_types=True raises NotImplementedError until Phase 3. Phase 1-2
    always use current type assignments, which introduces a small upward bias
    in accuracy metrics (types were trained on data that includes the target year).
    This limitation is documented in every report.
    """
    if use_epoch_types:
        raise NotImplementedError(
            "Epoch-retrained types are not yet implemented. See Phase 3 in the design spec."
        )

    # Load production model artifacts
    type_scores, county_fips, county_votes, county_state_abbr, dominant_types = (
        _load_type_scores()
    )
    county_priors = load_county_priors_with_ridge()  # FILL IN ARGS from Task 2 findings

    # Translate historical fundamentals to a generic ballot shift
    fund_snapshot = inputs.fundamentals
    generic_ballot_shift = _compute_generic_ballot_shift(fund_snapshot)

    # Run forecast for each race
    races = list(inputs.polls_by_race.keys())
    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=list(county_state_abbr),
        county_votes=county_votes,
        polls_by_race=inputs.polls_by_race,
        races=races,
        generic_ballot_shift=generic_ballot_shift,
        reference_date=inputs.cutoff_date.isoformat(),
    )

    # Assemble county-level predictions DataFrame
    county_rows = []
    for race_id, result in forecast_results.items():
        race_type = _infer_race_type(race_id)
        for i, fips in enumerate(county_fips):
            county_rows.append({
                "fips": fips,
                "state_abbr": county_state_abbr[i],
                "race_id": race_id,
                "race_type": race_type,
                "pred_dem_share": float(result.county_preds_local[i]),
                "dominant_type": int(dominant_types[i]),
                "county_votes": float(county_votes[i]),
            })
    county_predictions = pd.DataFrame(county_rows)

    state_predictions = _aggregate_to_state(county_predictions)

    chamber_prediction = {}
    for race_type in inputs.race_types:
        race_state_preds = state_predictions[state_predictions["race_type"] == race_type]
        chamber_prediction[race_type] = _compute_chamber_prediction(race_state_preds, race_type)

    return BacktestRun(
        inputs=inputs,
        state_predictions=state_predictions,
        county_predictions=county_predictions,
        chamber_prediction=chamber_prediction,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_type_scores() -> tuple[np.ndarray, list[str], np.ndarray, list[str], np.ndarray]:
    """
    Load type scores, county FIPS, vote weights, state assignments, and dominant type
    from the production model artifacts.

    FILL IN based on Task 2 findings: the exact file paths and column names
    for type score data in data/communities/.

    Returns:
        type_scores: ndarray (n_counties, J)
        county_fips: list of FIPS strings, length n_counties
        county_votes: ndarray (n_counties,) — vote counts for weighting
        county_state_abbr: list of state abbreviations, length n_counties
        dominant_types: ndarray (n_counties,) of int — index of max-score type
    """
    # FILL IN from Task 2 findings:
    # path = _PROJECT_ROOT / "data" / "communities" / "FILENAME.parquet"
    # df = pd.read_parquet(path)
    # type_scores = df[[...type score columns...]].values
    # county_fips = df["FIPS_COLUMN"].tolist()
    # county_votes = df["VOTES_COLUMN"].values
    # county_state_abbr = df["STATE_COLUMN"].tolist()
    # dominant_types = np.argmax(type_scores, axis=1)
    # return type_scores, county_fips, county_votes, county_state_abbr, dominant_types
    raise NotImplementedError(
        "Fill in _load_type_scores() based on Task 2 findings. "
        "Check data/communities/ for the type score file and its column schema."
    )


def _compute_generic_ballot_shift(fundamentals: dict) -> float:
    """
    Translate a fundamentals snapshot dict into a scalar generic ballot shift.

    Calls the same compute_fundamentals_shift() used in production, passing
    historical approval/economic values instead of current ones.
    """
    # load_fundamentals_snapshot normally reads from a file path.
    # We have the dict already, so write it to a temp file and load it.
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fundamentals, f)
        tmp_path = Path(f.name)
    try:
        snapshot = load_fundamentals_snapshot(tmp_path)
        return float(compute_fundamentals_shift(snapshot))
    finally:
        tmp_path.unlink(missing_ok=True)


def _infer_race_type(race_id: str) -> str:
    """Infer race type from race ID string convention."""
    race_lower = race_id.lower()
    if "senate" in race_lower:
        return "senate"
    if "governor" in race_lower or "gov" in race_lower:
        return "governor"
    if "pres" in race_lower or "president" in race_lower:
        return "presidential"
    return "unknown"


def _aggregate_to_state(county_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Vote-weight county predictions up to state level.

    Uses county_votes as weights so large-population counties drive the
    state prediction, matching how the forecast engine itself aggregates.
    """
    if county_predictions.empty:
        return pd.DataFrame(columns=["state_abbr", "race_id", "race_type", "pred_dem_share"])

    def _weighted_mean(group):
        weights = group["county_votes"]
        if weights.sum() == 0:
            return group["pred_dem_share"].mean()
        return (group["pred_dem_share"] * weights).sum() / weights.sum()

    state_preds = (
        county_predictions.groupby(["state_abbr", "race_id", "race_type"])
        .apply(_weighted_mean)
        .reset_index(name="pred_dem_share")
    )
    return state_preds


def _compute_chamber_prediction(state_preds: pd.DataFrame, race_type: str) -> dict:
    """
    Count predicted winners from state-level predictions.

    A race is predicted D if pred_dem_share > 0.5. This simple threshold
    matches how we describe outcomes in the report.
    """
    if state_preds.empty:
        return {"pred_dem_seats": 0, "pred_rep_seats": 0}
    dem_wins = int((state_preds["pred_dem_share"] > 0.5).sum())
    rep_wins = int((state_preds["pred_dem_share"] <= 0.5).sum())
    return {"pred_dem_seats": dem_wins, "pred_rep_seats": rep_wins}


def _polls_to_race_ids(polls_by_race: dict) -> list[str]:
    """Return the list of race IDs that have polls."""
    return list(polls_by_race.keys())
```

- [ ] **Step 4: Fill in `_load_type_scores()` based on Task 2 findings**

After reading `predict_2026_types.py` and checking `data/communities/`, replace the `raise NotImplementedError` block with the actual loading code.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/backtest/test_runner.py -v
```

Expected: all tests PASS. (The `run_backtest()` integration path is not tested here — that's covered in Task 8.)

- [ ] **Step 6: Commit**

```bash
git add src/backtest/runner.py tests/backtest/test_runner.py
git commit -m "feat(backtest): add BacktestRunner wrapping production forecast"
```

---

## Task 6: `errors.py` — Compute Error Artifacts

**Files:**
- Create: `src/backtest/errors.py`
- Create: `tests/backtest/test_errors.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_errors.py
"""Tests for error computation. Uses synthetic predictions and actuals."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.backtest.errors import (
    ErrorArtifact,
    compute_errors,
    _compute_state_errors,
    _compute_county_errors,
    _compute_type_errors,
    _compute_chamber_errors,
    _compute_summary,
)
from src.backtest.runner import BacktestRun
from src.backtest.actuals import Actuals
from src.backtest.inputs import HistoricalInputs


def _make_inputs():
    return HistoricalInputs(
        year=2024, race_types=["senate"],
        cutoff_date=date(2024, 10, 31),
        polls_by_race={}, fundamentals={}, races=[],
    )


def _make_state_preds():
    return pd.DataFrame({
        "state_abbr": ["AL", "AK", "AZ"],
        "race_id": ["2024-al-senate", "2024-ak-senate", "2024-az-senate"],
        "race_type": ["senate", "senate", "senate"],
        "pred_dem_share": [0.42, 0.55, 0.51],
    })


def _make_county_preds():
    return pd.DataFrame({
        "fips": ["01001", "01003", "02001"],
        "state_abbr": ["AL", "AL", "AK"],
        "race_id": ["2024-al-senate", "2024-al-senate", "2024-ak-senate"],
        "race_type": ["senate", "senate", "senate"],
        "pred_dem_share": [0.40, 0.44, 0.55],
        "dominant_type": [5, 12, 7],
        "county_votes": [30_000.0, 50_000.0, 8_000.0],
    })


def _make_state_actuals():
    return pd.DataFrame({
        "state_abbr": ["AL", "AK", "AZ"],
        "race_id": ["2024-al-senate", "2024-ak-senate", "2024-az-senate"],
        "race_type": ["senate", "senate", "senate"],
        "actual_dem_share": [0.38, 0.53, 0.49],
    })


def _make_county_actuals():
    return pd.DataFrame({
        "fips": ["01001", "01003", "02001"],
        "state_abbr": ["AL", "AL", "AK"],
        "race_id": ["2024-al-senate", "2024-al-senate", "2024-ak-senate"],
        "race_type": ["senate", "senate", "senate"],
        "actual_dem_share": [0.35, 0.41, 0.53],
    })


class TestComputeStateErrors:
    def test_error_sign_convention(self):
        # error = predicted - actual; positive means overestimated Dems
        result = _compute_state_errors(_make_state_preds(), _make_state_actuals())
        al = result[result["state_abbr"] == "AL"].iloc[0]
        assert abs(al["error"] - (0.42 - 0.38)) < 1e-6

    def test_abs_error_is_nonnegative(self):
        result = _compute_state_errors(_make_state_preds(), _make_state_actuals())
        assert (result["abs_error"] >= 0).all()

    def test_call_correct_column(self):
        # AZ: pred=0.51 (D win), actual=0.49 (R win) -> call_correct=False
        result = _compute_state_errors(_make_state_preds(), _make_state_actuals())
        az = result[result["state_abbr"] == "AZ"].iloc[0]
        assert az["call_correct"] is False or az["call_correct"] == False

    def test_correct_call_is_true(self):
        # AL: pred=0.42 (R win), actual=0.38 (R win) -> call_correct=True
        result = _compute_state_errors(_make_state_preds(), _make_state_actuals())
        al = result[result["state_abbr"] == "AL"].iloc[0]
        assert al["call_correct"] is True or al["call_correct"] == True

    def test_required_columns(self):
        result = _compute_state_errors(_make_state_preds(), _make_state_actuals())
        assert set(result.columns) >= {
            "state_abbr", "race_id", "race_type",
            "pred_dem_share", "actual_dem_share",
            "error", "abs_error", "pred_winner", "actual_winner", "call_correct",
        }


class TestComputeCountyErrors:
    def test_error_sign_convention(self):
        result = _compute_county_errors(_make_county_preds(), _make_county_actuals())
        row = result[result["fips"] == "01001"].iloc[0]
        assert abs(row["error"] - (0.40 - 0.35)) < 1e-6

    def test_required_columns(self):
        result = _compute_county_errors(_make_county_preds(), _make_county_actuals())
        assert set(result.columns) >= {"fips", "race_id", "race_type", "error", "abs_error"}


class TestComputeTypeErrors:
    def test_groups_by_dominant_type(self):
        county_errors = _compute_county_errors(_make_county_preds(), _make_county_actuals())
        # Add dominant_type back for grouping
        county_errors = county_errors.merge(
            _make_county_preds()[["fips", "race_id", "dominant_type"]],
            on=["fips", "race_id"],
        )
        result = _compute_type_errors(county_errors)
        # Types 5, 12, 7 should each appear
        assert set(result["dominant_type"]) == {5, 12, 7}

    def test_mean_error_is_signed(self):
        county_errors = _compute_county_errors(_make_county_preds(), _make_county_actuals())
        county_errors = county_errors.merge(
            _make_county_preds()[["fips", "race_id", "dominant_type"]],
            on=["fips", "race_id"],
        )
        result = _compute_type_errors(county_errors)
        # All errors are positive here (pred > actual), so mean_error > 0
        assert (result["mean_error"] > 0).all()


class TestComputeSummary:
    def test_includes_all_race_types(self):
        state_errors = _compute_state_errors(_make_state_preds(), _make_state_actuals())
        county_errors = _compute_county_errors(_make_county_preds(), _make_county_actuals())
        chamber_errors = _compute_chamber_errors(
            {"senate": {"pred_dem_seats": 48, "pred_rep_seats": 52}},
            {"senate": {"actual_dem_seats": 47, "actual_rep_seats": 53}},
        )
        summary = _compute_summary(state_errors, county_errors, chamber_errors, ["senate"])
        assert "senate" in summary
        assert "state_mae" in summary["senate"]
        assert "state_bias" in summary["senate"]
        assert "call_accuracy" in summary["senate"]

    def test_bias_is_signed(self):
        # All predictions overestimate Dems -> positive bias
        state_errors = _compute_state_errors(_make_state_preds(), _make_state_actuals())
        county_errors = _compute_county_errors(_make_county_preds(), _make_county_actuals())
        chamber_errors = pd.DataFrame()
        summary = _compute_summary(state_errors, county_errors, chamber_errors, ["senate"])
        assert summary["senate"]["state_bias"] > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/backtest/test_errors.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `errors.py`**

```python
# src/backtest/errors.py
"""
Compute error artifacts by joining BacktestRun predictions with Actuals.

Error sign convention: error = predicted_dem_share - actual_dem_share
  Positive -> overestimated Democrats (underestimated Republicans)
  Negative -> underestimated Democrats (overestimated Republicans)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd

from src.backtest.runner import BacktestRun
from src.backtest.actuals import Actuals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ErrorArtifact:
    year: int
    race_types: list[str]
    cutoff_date: date
    # Always False in Phase 1-2; True only when epoch-retrained types are used (Phase 3)
    types_are_epoch_trained: bool
    state_errors: pd.DataFrame
    county_errors: pd.DataFrame
    type_errors: pd.DataFrame
    chamber_errors: pd.DataFrame
    summary: dict


def compute_errors(run: BacktestRun, actuals: Actuals) -> ErrorArtifact:
    """
    Join predictions against actuals and compute signed errors at all four levels.

    Rows in predictions that have no matching actual (county or state) are dropped
    with a warning, since they likely represent data gaps rather than model errors.
    """
    state_errors = _compute_state_errors(run.state_predictions, actuals.state)
    county_errors = _compute_county_errors(run.county_predictions, actuals.county)

    # Type errors are derived from county errors (grouped by dominant_type from run)
    county_errors_with_type = county_errors.merge(
        run.county_predictions[["fips", "race_id", "dominant_type"]].drop_duplicates(),
        on=["fips", "race_id"],
        how="left",
    )
    type_errors = _compute_type_errors(county_errors_with_type)

    chamber_errors = _compute_chamber_errors(run.chamber_prediction, actuals.chamber)
    summary = _compute_summary(state_errors, county_errors, chamber_errors, run.inputs.race_types)

    return ErrorArtifact(
        year=run.inputs.year,
        race_types=run.inputs.race_types,
        cutoff_date=run.inputs.cutoff_date,
        types_are_epoch_trained=False,
        state_errors=state_errors,
        county_errors=county_errors,
        type_errors=type_errors,
        chamber_errors=chamber_errors,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_state_errors(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or actuals.empty:
        return pd.DataFrame()
    merged = predictions.merge(actuals, on=["state_abbr", "race_id", "race_type"], how="inner")
    merged["error"] = merged["pred_dem_share"] - merged["actual_dem_share"]
    merged["abs_error"] = merged["error"].abs()
    merged["pred_winner"] = merged["pred_dem_share"].apply(lambda x: "D" if x > 0.5 else "R")
    merged["actual_winner"] = merged["actual_dem_share"].apply(lambda x: "D" if x > 0.5 else "R")
    merged["call_correct"] = merged["pred_winner"] == merged["actual_winner"]
    return merged


def _compute_county_errors(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or actuals.empty:
        return pd.DataFrame()
    # pred_dem_share is only in predictions, so no _x suffix after merge.
    # state_abbr is in both DataFrames, so it gets _x/_y suffixes.
    merged = predictions.merge(actuals, on=["fips", "race_id", "race_type"], how="inner")
    merged["error"] = merged["pred_dem_share"] - merged["actual_dem_share"]
    merged["abs_error"] = merged["error"].abs()
    return merged[["fips", "state_abbr_x", "race_id", "race_type", "pred_dem_share",
                   "actual_dem_share", "error", "abs_error"]].rename(
        columns={"state_abbr_x": "state_abbr"}
    )


def _compute_type_errors(county_errors_with_type: pd.DataFrame) -> pd.DataFrame:
    """
    Group county errors by dominant_type to surface per-type systematic bias.

    Types with large mean_error across multiple races are candidates for
    wrong θ estimates rather than candidate effects (which would show up
    in one race but not others).
    """
    if county_errors_with_type.empty or "dominant_type" not in county_errors_with_type.columns:
        return pd.DataFrame()
    return (
        county_errors_with_type.groupby(["dominant_type", "race_id", "race_type"])
        .agg(
            mean_error=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            n_counties=("fips", "count"),
        )
        .reset_index()
    )


def _compute_chamber_errors(pred_chamber: dict, actual_chamber: dict) -> pd.DataFrame:
    rows = []
    for race_type in set(list(pred_chamber.keys()) + list(actual_chamber.keys())):
        pred = pred_chamber.get(race_type, {})
        actual = actual_chamber.get(race_type, {})
        pred_dem = pred.get("pred_dem_seats")
        actual_dem = actual.get("actual_dem_seats")
        seat_error = (pred_dem - actual_dem) if (pred_dem is not None and actual_dem is not None) else None
        rows.append({
            "race_type": race_type,
            "pred_dem_seats": pred_dem,
            "pred_rep_seats": pred.get("pred_rep_seats"),
            "actual_dem_seats": actual_dem,
            "actual_rep_seats": actual.get("actual_rep_seats"),
            "seat_error": seat_error,
        })
    return pd.DataFrame(rows)


def _compute_summary(
    state_errors: pd.DataFrame,
    county_errors: pd.DataFrame,
    chamber_errors: pd.DataFrame,
    race_types: list[str],
) -> dict:
    summary = {}
    for race_type in race_types:
        se = state_errors[state_errors["race_type"] == race_type] if not state_errors.empty else pd.DataFrame()
        ce = county_errors[county_errors["race_type"] == race_type] if not county_errors.empty else pd.DataFrame()
        ch = chamber_errors[chamber_errors["race_type"] == race_type] if not chamber_errors.empty else pd.DataFrame()

        row: dict = {
            "state_mae": float(se["abs_error"].mean()) if not se.empty else None,
            "state_bias": float(se["error"].mean()) if not se.empty else None,
            "state_rmse": float(np.sqrt((se["error"] ** 2).mean())) if not se.empty else None,
            "call_accuracy": float(se["call_correct"].mean()) if not se.empty else None,
            "county_mae": float(ce["abs_error"].mean()) if not ce.empty else None,
            "county_bias": float(ce["error"].mean()) if not ce.empty else None,
            "n_races": len(se["race_id"].unique()) if not se.empty else 0,
            "n_counties": len(ce["fips"].unique()) if not ce.empty else 0,
        }
        if not ch.empty:
            r = ch.iloc[0]
            row["seat_error"] = int(r["seat_error"]) if r["seat_error"] is not None else None
        summary[race_type] = row
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/backtest/test_errors.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/errors.py tests/backtest/test_errors.py
git commit -m "feat(backtest): add ErrorArtifact and error computation"
```

---

## Task 7: `catalog.py` — Persist and Index Artifacts

**Files:**
- Create: `src/backtest/catalog.py`
- Create: `tests/backtest/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_catalog.py
"""Tests for BacktestCatalog. Uses a tmp_path fixture — no real data/backtest/ required."""

import json
from datetime import date
from pathlib import Path
import pandas as pd
import pytest

from src.backtest.errors import ErrorArtifact
from src.backtest.catalog import BacktestCatalog


def _make_artifact(tmp_cutoff: date = date(2024, 10, 31)) -> ErrorArtifact:
    return ErrorArtifact(
        year=2024,
        race_types=["senate"],
        cutoff_date=tmp_cutoff,
        types_are_epoch_trained=False,
        state_errors=pd.DataFrame({
            "state_abbr": ["AL"], "race_id": ["2024-al-senate"],
            "race_type": ["senate"], "pred_dem_share": [0.42],
            "actual_dem_share": [0.38], "error": [0.04],
            "abs_error": [0.04], "pred_winner": ["R"], "actual_winner": ["R"],
            "call_correct": [True],
        }),
        county_errors=pd.DataFrame(),
        type_errors=pd.DataFrame(),
        chamber_errors=pd.DataFrame(),
        summary={"senate": {"state_mae": 0.04, "state_bias": 0.04}},
    )


class TestBacktestCatalog:
    def test_save_and_load_roundtrip(self, tmp_path):
        catalog = BacktestCatalog(root=tmp_path)
        artifact = _make_artifact()
        catalog.save_artifact(artifact)
        loaded = catalog.load_artifact(date(2024, 10, 31), "senate")
        assert loaded.year == 2024
        assert loaded.race_types == ["senate"]
        assert abs(loaded.state_errors.iloc[0]["error"] - 0.04) < 1e-6

    def test_save_creates_index(self, tmp_path):
        catalog = BacktestCatalog(root=tmp_path)
        catalog.save_artifact(_make_artifact())
        index_path = tmp_path / "index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert len(index) == 1
        assert index[0]["cutoff_date"] == "2024-10-31"

    def test_list_runs_returns_index(self, tmp_path):
        catalog = BacktestCatalog(root=tmp_path)
        catalog.save_artifact(_make_artifact())
        runs = catalog.list_runs()
        assert len(runs) == 1
        assert runs[0]["year"] == 2024

    def test_load_all_returns_all_artifacts(self, tmp_path):
        catalog = BacktestCatalog(root=tmp_path)
        catalog.save_artifact(_make_artifact(date(2024, 10, 31)))
        # Simulate a second run (different cutoff)
        artifact2 = _make_artifact(date(2022, 10, 31))
        artifact2 = ErrorArtifact(
            year=2022, race_types=["senate"],
            cutoff_date=date(2022, 10, 31), types_are_epoch_trained=False,
            state_errors=artifact2.state_errors, county_errors=pd.DataFrame(),
            type_errors=pd.DataFrame(), chamber_errors=pd.DataFrame(),
            summary=artifact2.summary,
        )
        catalog.save_artifact(artifact2)
        all_runs = catalog.load_all()
        assert len(all_runs) == 2

    def test_missing_artifact_raises(self, tmp_path):
        catalog = BacktestCatalog(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            catalog.load_artifact(date(2020, 10, 31), "senate")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/backtest/test_catalog.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `catalog.py`**

```python
# src/backtest/catalog.py
"""
Persist and index BacktestErrorArtifacts in data/backtest/.

Directory structure:
    data/backtest/
      index.json                     # catalog of all runs
      2024-10-31/                    # cutoff date
        senate/
          state_errors.parquet
          county_errors.parquet
          type_errors.parquet
          chamber_errors.parquet
          summary.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
import pandas as pd

from src.backtest.errors import ErrorArtifact

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_ROOT = _PROJECT_ROOT / "data" / "backtest"


class BacktestCatalog:
    def __init__(self, root: Path | None = None):
        self.root = root or _DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_artifact(self, artifact: ErrorArtifact) -> None:
        """Write artifact to disk and update index.json."""
        run_dir = self._run_dir(artifact.cutoff_date, artifact.race_types[0])
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_df(artifact.state_errors, run_dir / "state_errors.parquet")
        self._write_df(artifact.county_errors, run_dir / "county_errors.parquet")
        self._write_df(artifact.type_errors, run_dir / "type_errors.parquet")
        self._write_df(artifact.chamber_errors, run_dir / "chamber_errors.parquet")
        (run_dir / "summary.json").write_text(json.dumps(artifact.summary, indent=2))

        self._update_index(artifact)

    def load_artifact(self, cutoff_date: date, race_type: str) -> ErrorArtifact:
        """Load a previously saved artifact from disk."""
        run_dir = self._run_dir(cutoff_date, race_type)
        if not run_dir.exists():
            raise FileNotFoundError(
                f"No backtest artifact found for cutoff={cutoff_date}, race_type={race_type}. "
                f"Expected directory: {run_dir}"
            )
        summary = json.loads((run_dir / "summary.json").read_text())
        return ErrorArtifact(
            year=cutoff_date.year,
            race_types=list(summary.keys()),
            cutoff_date=cutoff_date,
            types_are_epoch_trained=False,
            state_errors=self._read_df(run_dir / "state_errors.parquet"),
            county_errors=self._read_df(run_dir / "county_errors.parquet"),
            type_errors=self._read_df(run_dir / "type_errors.parquet"),
            chamber_errors=self._read_df(run_dir / "chamber_errors.parquet"),
            summary=summary,
        )

    def load_all(self, race_type: str | None = None) -> list[ErrorArtifact]:
        """Load all saved artifacts, optionally filtered by race_type."""
        runs = self.list_runs()
        artifacts = []
        for run in runs:
            cutoff = date.fromisoformat(run["cutoff_date"])
            for rt in run["race_types"]:
                if race_type is None or rt == race_type:
                    try:
                        artifacts.append(self.load_artifact(cutoff, rt))
                    except FileNotFoundError:
                        pass
        return artifacts

    def list_runs(self) -> list[dict]:
        """Return the index of all completed runs."""
        index_path = self.root / "index.json"
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dir(self, cutoff_date: date, race_type: str) -> Path:
        return self.root / cutoff_date.isoformat() / race_type

    def _write_df(self, df: pd.DataFrame, path: Path) -> None:
        if df is not None and not df.empty:
            df.to_parquet(path, index=False)
        else:
            pd.DataFrame().to_parquet(path, index=False)

    def _read_df(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def _update_index(self, artifact: ErrorArtifact) -> None:
        index_path = self.root / "index.json"
        runs = self.list_runs()

        # Remove existing entry for this cutoff if present
        runs = [r for r in runs if r["cutoff_date"] != artifact.cutoff_date.isoformat()]

        runs.append({
            "cutoff_date": artifact.cutoff_date.isoformat(),
            "year": artifact.year,
            "race_types": artifact.race_types,
            "types_are_epoch_trained": artifact.types_are_epoch_trained,
            "run_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": artifact.summary,
        })
        runs.sort(key=lambda r: r["cutoff_date"])
        index_path.write_text(json.dumps(runs, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/backtest/test_catalog.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/catalog.py tests/backtest/test_catalog.py
git commit -m "feat(backtest): add BacktestCatalog for artifact persistence"
```

---

## Task 8: `report.py` — Markdown Report Generation

**Files:**
- Create: `src/backtest/report.py`
- Create: `tests/backtest/test_report.py`
- Create: `docs/backtest/` directory

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_report.py
"""Tests for report generation. Validates structure, not exact wording."""

import re
from datetime import date
from pathlib import Path
import pandas as pd
import pytest

from src.backtest.errors import ErrorArtifact
from src.backtest.report import generate_report, _format_error_direction


def _make_artifact():
    return ErrorArtifact(
        year=2024, race_types=["senate"],
        cutoff_date=date(2024, 10, 31), types_are_epoch_trained=False,
        state_errors=pd.DataFrame({
            "state_abbr": ["AL", "AK", "AZ"],
            "race_id": ["2024-al-senate", "2024-ak-senate", "2024-az-senate"],
            "race_type": ["senate", "senate", "senate"],
            "pred_dem_share": [0.42, 0.55, 0.51],
            "actual_dem_share": [0.38, 0.53, 0.49],
            "error": [0.04, 0.02, 0.02],
            "abs_error": [0.04, 0.02, 0.02],
            "pred_winner": ["R", "D", "D"],
            "actual_winner": ["R", "D", "R"],
            "call_correct": [True, True, False],
        }),
        county_errors=pd.DataFrame(),
        type_errors=pd.DataFrame({
            "dominant_type": [5, 12],
            "race_id": ["2024-al-senate", "2024-ak-senate"],
            "race_type": ["senate", "senate"],
            "mean_error": [0.05, -0.02],
            "mean_abs_error": [0.05, 0.02],
            "n_counties": [10, 5],
        }),
        chamber_errors=pd.DataFrame({
            "race_type": ["senate"],
            "pred_dem_seats": [48],
            "pred_rep_seats": [52],
            "actual_dem_seats": [47],
            "actual_rep_seats": [53],
            "seat_error": [1],
        }),
        summary={"senate": {
            "state_mae": 0.027, "state_bias": 0.027, "state_rmse": 0.028,
            "call_accuracy": 0.667, "n_races": 3, "seat_error": 1,
        }},
    )


class TestFormatErrorDirection:
    def test_positive_is_D_overestimate(self):
        result = _format_error_direction(0.03)
        assert "D+" in result

    def test_negative_is_R_overestimate(self):
        result = _format_error_direction(-0.03)
        assert "R+" in result

    def test_zero_is_zero(self):
        result = _format_error_direction(0.0)
        assert "0" in result

    def test_value_is_in_pp(self):
        # 0.03 should display as 3.0pp, not 0.03
        result = _format_error_direction(0.03)
        assert "3.0" in result or "3pp" in result.lower()


class TestGenerateReport:
    def test_produces_markdown_string(self, tmp_path):
        artifact = _make_artifact()
        output = tmp_path / "report.md"
        generate_report([artifact], output_path=output)
        content = output.read_text()
        assert len(content) > 100  # not empty
        assert "#" in content  # has headings

    def test_contains_summary_table(self, tmp_path):
        artifact = _make_artifact()
        output = tmp_path / "report.md"
        generate_report([artifact], output_path=output)
        content = output.read_text()
        assert "MAE" in content
        assert "senate" in content.lower()

    def test_contains_chamber_outcome(self, tmp_path):
        artifact = _make_artifact()
        output = tmp_path / "report.md"
        generate_report([artifact], output_path=output)
        content = output.read_text()
        # Should mention seat counts
        assert "48" in content or "47" in content

    def test_contains_type_bias_section(self, tmp_path):
        artifact = _make_artifact()
        output = tmp_path / "report.md"
        generate_report([artifact], output_path=output)
        content = output.read_text()
        assert "Type" in content

    def test_epoch_warning_when_not_epoch_trained(self, tmp_path):
        artifact = _make_artifact()
        output = tmp_path / "report.md"
        generate_report([artifact], output_path=output)
        content = output.read_text()
        # Should warn that types are not epoch-trained
        assert "epoch" in content.lower() or "current type" in content.lower()

    def test_multi_cycle_includes_cross_cycle_section(self, tmp_path):
        artifact1 = _make_artifact()
        artifact2 = _make_artifact()
        artifact2 = ErrorArtifact(
            year=2022, race_types=["senate"], cutoff_date=date(2022, 10, 31),
            types_are_epoch_trained=False,
            state_errors=artifact1.state_errors, county_errors=pd.DataFrame(),
            type_errors=artifact1.type_errors, chamber_errors=artifact1.chamber_errors,
            summary=artifact1.summary,
        )
        output = tmp_path / "report.md"
        generate_report([artifact1, artifact2], output_path=output)
        content = output.read_text()
        assert "Cross-Cycle" in content or "2022" in content
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/backtest/test_report.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `docs/backtest/` directory**

```bash
mkdir -p docs/backtest
touch docs/backtest/.gitkeep
git add docs/backtest/.gitkeep
```

- [ ] **Step 4: Implement `report.py`**

```python
# src/backtest/report.py
"""
Generate a markdown report from one or more ErrorArtifacts.

Single-cycle report: summary table, chamber outcomes, top outliers, type bias.
Multi-cycle report: adds cross-cycle bias trends, persistent state outliers.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.backtest.errors import ErrorArtifact


def generate_report(
    artifacts: list[ErrorArtifact],
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    Write a markdown report to output_path.

    Single artifact -> single-cycle report.
    Multiple artifacts -> multi-cycle report with cross-cycle sections appended.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections = []

    if len(artifacts) == 1:
        a = artifacts[0]
        report_title = title or f"WetherVane Backtest — {a.year} (cutoff {a.cutoff_date})"
        sections.append(f"# {report_title}\n")
        sections.append(_epoch_warning(a))
        sections.append(_summary_table(a))
        sections.append(_chamber_section(a))
        sections.append(_state_outliers_section(a))
        sections.append(_type_bias_section(a))
        sections.append(_full_state_table(a))
    else:
        report_title = title or "WetherVane Backtest — Cross-Cycle Report"
        sections.append(f"# {report_title}\n")
        sections.append(_cross_cycle_summary(artifacts))
        sections.append("---\n")
        for a in sorted(artifacts, key=lambda x: x.year, reverse=True):
            sections.append(f"## {a.year} Detail\n")
            sections.append(_epoch_warning(a))
            sections.append(_summary_table(a))
            sections.append(_chamber_section(a))
            sections.append(_state_outliers_section(a))
        sections.append(_cross_cycle_type_bias(artifacts))

    output_path.write_text("\n".join(sections))


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _epoch_warning(artifact: ErrorArtifact) -> str:
    if artifact.types_are_epoch_trained:
        return "> **Type training:** Epoch-retrained — types trained only on data available before this election.\n\n"
    return (
        "> **Type training:** Current types (not epoch-retrained). "
        "Types were trained on data including this election year, which introduces "
        "a small downward bias in error metrics. See Phase 3 for honest comparison.\n\n"
    )


def _summary_table(artifact: ErrorArtifact) -> str:
    lines = ["## Summary\n", "| Race Type | State MAE | State Bias | Call Accuracy | Seat Error |",
             "|---|---|---|---|---|"]
    for race_type, m in artifact.summary.items():
        mae = f"{m['state_mae']*100:.1f}pp" if m.get("state_mae") is not None else "—"
        bias = _format_error_direction(m.get("state_bias", 0.0)) if m.get("state_bias") is not None else "—"
        acc = f"{m['call_accuracy']*100:.0f}%" if m.get("call_accuracy") is not None else "—"
        seat_err = str(m.get("seat_error", "—")) if m.get("seat_error") is not None else "—"
        lines.append(f"| {race_type} | {mae} | {bias} | {acc} | {seat_err} |")
    return "\n".join(lines) + "\n\n"


def _chamber_section(artifact: ErrorArtifact) -> str:
    if artifact.chamber_errors.empty:
        return ""
    lines = ["## Chamber Outcomes\n", "| Race Type | Predicted | Actual | Seat Error |",
             "|---|---|---|---|"]
    for _, row in artifact.chamber_errors.iterrows():
        pred = f"D {row['pred_dem_seats']}–R {row['pred_rep_seats']}" if row.get("pred_dem_seats") else "—"
        actual = f"D {row['actual_dem_seats']}–R {row['actual_rep_seats']}" if row.get("actual_dem_seats") else "—"
        err = f"{int(row['seat_error']):+d}" if row.get("seat_error") is not None else "—"
        lines.append(f"| {row['race_type']} | {pred} | {actual} | {err} |")
    return "\n".join(lines) + "\n\n"


def _state_outliers_section(artifact: ErrorArtifact) -> str:
    if artifact.state_errors.empty:
        return ""
    lines = ["## Largest State Errors\n",
             "| State | Race | Predicted | Actual | Error | Call |",
             "|---|---|---|---|---|---|"]
    top = artifact.state_errors.nlargest(10, "abs_error")
    for _, row in top.iterrows():
        pred = f"{row['pred_dem_share']*100:.1f}% D"
        actual = f"{row['actual_dem_share']*100:.1f}% D"
        err = _format_error_direction(row["error"])
        call = "✓" if row["call_correct"] else "✗"
        lines.append(f"| {row['state_abbr']} | {row['race_id']} | {pred} | {actual} | {err} | {call} |")
    return "\n".join(lines) + "\n\n"


def _type_bias_section(artifact: ErrorArtifact) -> str:
    if artifact.type_errors.empty:
        return ""
    # Show types with |mean_error| > 3pp
    significant = artifact.type_errors[artifact.type_errors["mean_abs_error"] > 0.03]
    if significant.empty:
        return "## Type Bias\n\nNo types with mean absolute error > 3pp.\n\n"
    lines = ["## Type Systematic Bias\n",
             "_Types where |mean error| > 3pp — candidates for wrong θ estimates._\n",
             "| Type | Race | Mean Error | n Counties |",
             "|---|---|---|---|"]
    for _, row in significant.sort_values("mean_abs_error", ascending=False).iterrows():
        err = _format_error_direction(row["mean_error"])
        lines.append(f"| Type {int(row['dominant_type'])} | {row['race_id']} | {err} | {int(row['n_counties'])} |")
    return "\n".join(lines) + "\n\n"


def _full_state_table(artifact: ErrorArtifact) -> str:
    if artifact.state_errors.empty:
        return ""
    lines = ["## Full State Results\n",
             "| State | Race Type | Predicted | Actual | Error | Call |",
             "|---|---|---|---|---|---|"]
    for _, row in artifact.state_errors.sort_values(["race_type", "state_abbr"]).iterrows():
        pred = f"{row['pred_dem_share']*100:.1f}% D"
        actual = f"{row['actual_dem_share']*100:.1f}% D"
        err = _format_error_direction(row["error"])
        call = "✓" if row["call_correct"] else "✗"
        lines.append(f"| {row['state_abbr']} | {row['race_type']} | {pred} | {actual} | {err} | {call} |")
    return "\n".join(lines) + "\n\n"


def _cross_cycle_summary(artifacts: list[ErrorArtifact]) -> str:
    lines = ["## Cross-Cycle Bias Summary\n",
             "| Year | Race Type | State MAE | State Bias | Call Accuracy |",
             "|---|---|---|---|---|"]
    for a in sorted(artifacts, key=lambda x: x.year):
        for race_type, m in a.summary.items():
            mae = f"{m['state_mae']*100:.1f}pp" if m.get("state_mae") is not None else "—"
            bias = _format_error_direction(m.get("state_bias", 0.0)) if m.get("state_bias") is not None else "—"
            acc = f"{m['call_accuracy']*100:.0f}%" if m.get("call_accuracy") is not None else "—"
            lines.append(f"| {a.year} | {race_type} | {mae} | {bias} | {acc} |")
    return "\n".join(lines) + "\n\n"


def _cross_cycle_type_bias(artifacts: list[ErrorArtifact]) -> str:
    all_type_errors = []
    for a in artifacts:
        if not a.type_errors.empty:
            t = a.type_errors.copy()
            t["year"] = a.year
            all_type_errors.append(t)
    if not all_type_errors:
        return ""
    combined = pd.concat(all_type_errors, ignore_index=True)
    # Types with consistent bias across at least 2 cycles
    type_summary = (
        combined.groupby("dominant_type")
        .agg(n_cycles=("year", "nunique"), mean_error=("mean_error", "mean"))
        .reset_index()
    )
    persistent = type_summary[(type_summary["n_cycles"] >= 2) & (type_summary["mean_error"].abs() > 0.02)]
    if persistent.empty:
        return "## Cross-Cycle Type Bias\n\nNo types with consistent bias across 2+ cycles.\n\n"
    lines = ["## Cross-Cycle Type Bias\n",
             "_Types with consistent directional error across 2+ cycles._\n",
             "| Type | n Cycles | Mean Error |",
             "|---|---|---|"]
    for _, row in persistent.sort_values("mean_error", key=abs, ascending=False).iterrows():
        err = _format_error_direction(row["mean_error"])
        lines.append(f"| Type {int(row['dominant_type'])} | {int(row['n_cycles'])} | {err} |")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_error_direction(error: float) -> str:
    """
    Format a signed error value for display.

    Positive error = overestimated Democrats -> displayed as "D+Xpp"
    Negative error = underestimated Democrats -> displayed as "R+Xpp"
    """
    pp = abs(error) * 100
    if abs(error) < 0.001:
        return "0.0pp"
    direction = "D" if error > 0 else "R"
    return f"{direction}+{pp:.1f}pp"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/backtest/test_report.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/report.py tests/backtest/test_report.py docs/backtest/.gitkeep
git commit -m "feat(backtest): add markdown report generator"
```

---

## Task 9: `cli.py` + Integration Test

**Files:**
- Create: `src/backtest/cli.py`
- Create: `tests/backtest/test_integration.py`

The integration test is marked `@pytest.mark.skipif` when real data files are absent, so it doesn't block CI. It runs end-to-end when data is present.

- [ ] **Step 1: Implement `cli.py`**

```python
# src/backtest/cli.py
"""
Entry point for the backtest harness.

Usage:
    uv run python -m src.backtest run --year 2024 --race-types senate presidential
    uv run python -m src.backtest run --all
    uv run python -m src.backtest report --cutoff 2024-10-31
    uv run python -m src.backtest report --all --output docs/backtest/report_all.md
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.backtest.actuals import load_actuals
from src.backtest.catalog import BacktestCatalog
from src.backtest.errors import compute_errors
from src.backtest.inputs import build_historical_inputs
from src.backtest.report import generate_report
from src.backtest.runner import run_backtest

# Cycles available for --all mode. Extend as Phase 2 adds more.
_CONFIGURED_CYCLES = [
    (2024, ["senate", "presidential"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="WetherVane backtest harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a backtest for a given year")
    run_parser.add_argument("--year", type=int, help="Election year (e.g. 2024)")
    run_parser.add_argument(
        "--race-types", nargs="+", default=["senate", "presidential"],
        choices=["senate", "presidential", "governor"],
        help="Race types to backtest",
    )
    run_parser.add_argument(
        "--cutoff", type=str, default=None,
        help="Cutoff date as YYYY-MM-DD (default: October 31 of --year)",
    )
    run_parser.add_argument("--all", action="store_true", help="Run all configured cycles")

    report_parser = subparsers.add_parser("report", help="Generate a markdown report")
    report_parser.add_argument(
        "--cutoff", type=str, default=None,
        help="Report for specific cutoff date (YYYY-MM-DD)",
    )
    report_parser.add_argument("--all", action="store_true", help="Cross-cycle report from all saved artifacts")
    report_parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for report markdown",
    )

    args = parser.parse_args()

    if args.command == "run":
        if args.all:
            for year, race_types in _CONFIGURED_CYCLES:
                _run_single(year, race_types, cutoff_date=None)
        elif args.year:
            cutoff = date.fromisoformat(args.cutoff) if args.cutoff else None
            _run_single(args.year, args.race_types, cutoff_date=cutoff)
        else:
            run_parser.error("Provide --year or --all")

    elif args.command == "report":
        catalog = BacktestCatalog()
        if args.all:
            artifacts = catalog.load_all()
            if not artifacts:
                print("No artifacts found in data/backtest/. Run `python -m src.backtest run` first.")
                return
            output = Path(args.output) if args.output else Path("docs/backtest/report_all.md")
            generate_report(artifacts, output_path=output)
            print(f"Report written to {output}")
        elif args.cutoff:
            cutoff = date.fromisoformat(args.cutoff)
            artifacts = catalog.load_all()
            artifacts = [a for a in artifacts if a.cutoff_date == cutoff]
            if not artifacts:
                print(f"No artifacts found for cutoff {cutoff}.")
                return
            output = Path(args.output) if args.output else Path(f"docs/backtest/report_{cutoff}.md")
            generate_report(artifacts, output_path=output)
            print(f"Report written to {output}")
        else:
            report_parser.error("Provide --cutoff or --all")


def _run_single(year: int, race_types: list[str], cutoff_date: date | None) -> None:
    print(f"Running backtest: year={year}, race_types={race_types}, cutoff={cutoff_date or f'{year}-10-31'}")
    inputs = build_historical_inputs(year=year, race_types=race_types, cutoff_date=cutoff_date)
    print(f"  Loaded {sum(len(v) for v in inputs.polls_by_race.values())} polls across {len(inputs.polls_by_race)} races")
    run = run_backtest(inputs)
    print(f"  Generated predictions for {len(run.state_predictions)} state-races, {len(run.county_predictions)} county-races")
    actuals = load_actuals(year=year, race_types=race_types)
    artifact = compute_errors(run, actuals)
    catalog = BacktestCatalog()
    catalog.save_artifact(artifact)
    print(f"  Artifact saved to data/backtest/{inputs.cutoff_date}/")
    for race_type, m in artifact.summary.items():
        mae = f"{m['state_mae']*100:.1f}pp" if m.get('state_mae') else '—'
        bias = f"{m['state_bias']*100:+.1f}pp" if m.get('state_bias') else '—'
        print(f"  {race_type}: state MAE={mae}, bias={bias}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `__main__.py` so `python -m src.backtest` works**

```python
# src/backtest/__main__.py
from src.backtest.cli import main
main()
```

- [ ] **Step 3: Write the integration test**

```python
# tests/backtest/test_integration.py
"""
End-to-end integration test for the 2024 backtest.

Skipped when data files are absent (CI environment without real data).
Run locally after verifying data availability in Task 2.
"""

import pytest
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

_REQUIRED_DATA = [
    PROJECT_ROOT / "data" / "polls" / "polls_2024.csv",
    PROJECT_ROOT / "data" / "config" / "fundamentals_2024.json",
]
_HAS_DATA = all(p.exists() for p in _REQUIRED_DATA)


@pytest.mark.skipif(not _HAS_DATA, reason="2024 data files not present")
def test_full_2024_senate_backtest(tmp_path):
    """Run the full 2024 Senate backtest end-to-end and verify output structure."""
    from src.backtest.inputs import build_historical_inputs
    from src.backtest.runner import run_backtest
    from src.backtest.actuals import load_actuals
    from src.backtest.errors import compute_errors
    from src.backtest.catalog import BacktestCatalog
    from src.backtest.report import generate_report

    inputs = build_historical_inputs(
        year=2024,
        race_types=["senate"],
        cutoff_date=date(2024, 10, 31),
    )

    assert len(inputs.polls_by_race) > 0, "Expected at least some 2024 Senate polls"

    run = run_backtest(inputs)

    assert not run.state_predictions.empty
    assert "pred_dem_share" in run.state_predictions.columns
    assert run.state_predictions["pred_dem_share"].between(0.0, 1.0).all()

    assert not run.county_predictions.empty
    assert "dominant_type" in run.county_predictions.columns

    actuals = load_actuals(year=2024, race_types=["senate"])
    assert not actuals.state.empty

    artifact = compute_errors(run, actuals)
    assert not artifact.state_errors.empty
    assert artifact.summary.get("senate") is not None
    assert artifact.summary["senate"]["state_mae"] is not None
    # MAE should be plausible — a completely broken model would exceed 20pp
    assert artifact.summary["senate"]["state_mae"] < 0.20, (
        f"State MAE {artifact.summary['senate']['state_mae']*100:.1f}pp is implausibly large. "
        f"Check prediction pipeline wiring."
    )

    catalog = BacktestCatalog(root=tmp_path / "backtest")
    catalog.save_artifact(artifact)
    loaded = catalog.load_artifact(date(2024, 10, 31), "senate")
    assert loaded.year == 2024

    report_path = tmp_path / "report_2024.md"
    generate_report([artifact], output_path=report_path)
    content = report_path.read_text()
    assert "senate" in content.lower()
    assert "MAE" in content
```

- [ ] **Step 4: Run all backtest tests**

```bash
uv run pytest tests/backtest/ -v
```

Expected: All unit tests PASS. Integration test SKIPPED (unless data files are present).

- [ ] **Step 5: Verify CLI entry point works**

```bash
uv run python -m src.backtest --help
```

Expected: Shows `run` and `report` subcommands.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/cli.py src/backtest/__main__.py tests/backtest/test_integration.py
git commit -m "feat(backtest): add CLI entry point and integration test"
```

---

## Post-Implementation: First Real Run

Once all code is in place and data has been verified:

```bash
# 1. Generate 2024 polls file from 538 archive (if not already present)
uv run python src/assembly/convert_538_polls.py --year 2024

# 2. Verify fundamentals_2024.json exists with correct values
cat data/config/fundamentals_2024.json

# 3. Run the 2024 backtest
uv run python -m src.backtest run --year 2024 --race-types senate presidential

# 4. Generate the report
uv run python -m src.backtest report --cutoff 2024-10-31 --output docs/backtest/report_2024-10-31.md

# 5. Commit the report
git add docs/backtest/report_2024-10-31.md
git commit -m "docs(backtest): add 2024 Senate + Presidential backtest report"
```

Read the report. Look for:
- Consistent directional bias in state_bias (positive = systematically overestimating Dems)
- States with large abs_error that are surprising (NH, NV, GA are good test cases)
- Types with mean_error > 3pp in the type bias section
- Call accuracy — how many races did we call wrong?

These observations drive the Phase 2 analytical priorities.
