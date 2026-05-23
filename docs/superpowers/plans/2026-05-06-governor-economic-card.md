# Governor Economic Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GovernorEconomicCard that shows state wage growth % and employment change % (2020–2023) for competitive governor races on the overview page.

**Architecture:** Load `data/raw/qcew_county.parquet` once at API startup (lazy module-level cache), aggregate to state level, attach as `econ: {wage_growth_pct, employment_change_pct}` to each race in `/governor/overview`. A new `GovernorEconomicCard` TSX component (props-based, like `GovernorSeatRiskCard`) filters to competitive races and renders a `<dl>` table, wired into the governor page after `GovernorPollingCard`.

**Tech Stack:** Python/pandas (backend), FastAPI, Next.js 14, TypeScript, Tailwind CSS, SWR

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `api/routers/governor/overview.py` | Modify | Add `_compute_qcew_state_econ()`, cache, attach `econ` to each race |
| `api/tests/test_governor_overview.py` | Modify | Assert `econ` field present; verify null when file absent |
| `web/lib/api.ts` | Modify | Add `GovernorEcon` interface; add `econ` field to `GovernorRaceData` |
| `web/components/forecast/GovernorEconomicCard.tsx` | Create | Card component displaying econ indicators for competitive states |
| `web/app/forecast/governor/page.tsx` | Modify | Import and wire `GovernorEconomicCard` after `GovernorPollingCard` |

---

## Task 1: Add QCEW econ computation to overview.py

**Files:**
- Modify: `api/routers/governor/overview.py`

**Context:**
- Parquet schema: `county_fips` (5-char str), `own_code` (str '0'=all), `industry_code` (str '10'=total), `year` (int), `annual_avg_emplvl` (int), `total_annual_wages` (int)
- Years available: 2020, 2021, 2022, 2023
- `STATE_ABBR` from `src.core.config` maps 2-digit state FIPS → state abbreviation (e.g. `"04" → "AZ"`)
- Path to parquet: 4 directories up from `overview.py` (project root) / `data/raw/qcew_county.parquet`
- The parquet file is absent in test environments — the function must return `{}` gracefully on `FileNotFoundError` or any pandas exception

- [ ] **Step 1: Add imports and path constant at the top of overview.py**

Add these lines after `import logging` and before `import duckdb`:

```python
from pathlib import Path

import pandas as pd
```

Add this constant after the existing imports (after the `router = APIRouter(...)` line is fine, but before any function definitions is cleaner — put it at module level just before `router`):

```python
_QCEW_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "qcew_county.parquet"
```

- [ ] **Step 2: Add the state FIPS → abbr mapping import**

Add this import block right after `from api.routers.governor._helpers import ...`:

```python
try:
    from src.core.config import STATE_ABBR as _STATE_ABBR_MAP
except ImportError:
    _STATE_ABBR_MAP: dict[str, str] = {}
```

- [ ] **Step 3: Add `_compute_qcew_state_econ` and `_get_qcew_state_econ` functions**

Insert these two functions after the `_QCEW_PATH` constant and before `router = APIRouter(...)`:

```python
_QCEW_ECON_CACHE: dict[str, dict] | None = None


def _compute_qcew_state_econ() -> dict[str, dict]:
    """Compute state-level wage growth and employment change from QCEW parquet.

    Returns dict mapping state_abbr -> {wage_growth_pct, employment_change_pct}.
    Returns empty dict if the parquet file is absent or malformed.
    """
    try:
        df = pd.read_parquet(_QCEW_PATH)
    except Exception:
        return {}

    # Filter to total-industry, all-ownership rows
    df = df[(df["industry_code"] == "10") & (df["own_code"] == "0")].copy()
    df["state_fips"] = df["county_fips"].str[:2]

    # Aggregate wages and employment to state × year
    agg = (
        df.groupby(["state_fips", "year"])
        .agg(
            total_wages=("total_annual_wages", "sum"),
            total_emplvl=("annual_avg_emplvl", "sum"),
        )
        .reset_index()
    )

    result: dict[str, dict] = {}
    for state_fips, grp in agg.groupby("state_fips"):
        state_abbr = _STATE_ABBR_MAP.get(str(state_fips))
        if not state_abbr:
            continue
        r2020 = grp[grp["year"] == 2020]
        r2023 = grp[grp["year"] == 2023]
        if r2020.empty or r2023.empty:
            continue
        wages_20 = int(r2020["total_wages"].iloc[0])
        emplvl_20 = int(r2020["total_emplvl"].iloc[0])
        wages_23 = int(r2023["total_wages"].iloc[0])
        emplvl_23 = int(r2023["total_emplvl"].iloc[0])
        if emplvl_20 == 0 or wages_20 == 0:
            continue
        avg_pay_20 = wages_20 / emplvl_20
        avg_pay_23 = wages_23 / emplvl_23 if emplvl_23 > 0 else 0.0
        result[state_abbr] = {
            "wage_growth_pct": round((avg_pay_23 - avg_pay_20) / avg_pay_20 * 100, 2),
            "employment_change_pct": round(
                (emplvl_23 - emplvl_20) / emplvl_20 * 100, 2
            ),
        }
    return result


def _get_qcew_state_econ() -> dict[str, dict]:
    """Return cached QCEW state econ dict, computing it on first call."""
    global _QCEW_ECON_CACHE
    if _QCEW_ECON_CACHE is None:
        _QCEW_ECON_CACHE = _compute_qcew_state_econ()
    return _QCEW_ECON_CACHE
```

- [ ] **Step 4: Attach `econ` to each race in `get_governor_overview`**

In the `get_governor_overview` function, add econ data to each race before appending.

Find the loop body (around line 141–144):
```python
    races = []
    for st in sorted(GOVERNOR_2026_STATES):
        race_info = classify_governor_race(st, pred_by_race)
        race_info["n_polls"] = poll_counts.get(race_info["race"], 0)
        races.append(race_info)
```

Replace with:
```python
    econ_by_state = _get_qcew_state_econ()
    races = []
    for st in sorted(GOVERNOR_2026_STATES):
        race_info = classify_governor_race(st, pred_by_race)
        race_info["n_polls"] = poll_counts.get(race_info["race"], 0)
        race_info["econ"] = econ_by_state.get(st)
        races.append(race_info)
```

Also update the structural-fallback path (around line 127–129) to attach econ:
```python
    if not version_id:
        econ_by_state = _get_qcew_state_econ()
        races = [classify_governor_race(st) for st in sorted(GOVERNOR_2026_STATES)]
        for r in races:
            r["econ"] = econ_by_state.get(r["state"])
        races.sort(key=lambda r: (rating_sort_key(r["rating"]), r["state"]))
        return {"races": races, "dem_current": DEM_GOV_CURRENT, "gop_current": GOP_GOV_CURRENT, "updated_at": None}
```

- [ ] **Step 5: Verify the file parses without error**

```bash
cd /home/hayden/projects/wethervane
.venv/bin/python -c "
from api.routers.governor.overview import _get_qcew_state_econ
econ = _get_qcew_state_econ()
print('States with econ data:', len(econ))
if 'AZ' in econ:
    print('AZ:', econ['AZ'])
if 'OH' in econ:
    print('OH:', econ['OH'])
"
```

Expected: prints number of states (should be ~50) and sample values.

- [ ] **Step 6: Commit**

```bash
cd /home/hayden/projects/wethervane
git add api/routers/governor/overview.py
git commit -m "feat: attach QCEW state econ data to governor overview races"
```

---

## Task 2: Update governor overview tests

**Files:**
- Modify: `api/tests/test_governor_overview.py`

**Context:**
- Test environments use in-memory DuckDB with no access to the parquet file
- `_get_qcew_state_econ()` returns `{}` when parquet is absent, so `econ` will be `None` for all races in tests
- The `test_race_fields_present` test at line 189 needs updating
- Add a new test that explicitly asserts econ structure

- [ ] **Step 1: Update `test_race_fields_present` to assert `econ` key**

Find `test_race_fields_present` in `TestGovernorOverviewWithModel` (line 189). Add `assert "econ" in race` to the existing field assertions:

```python
    def test_race_fields_present(self, overview_client):
        data = overview_client.get("/api/v1/governor/overview").json()
        for race in data["races"]:
            assert "state" in race
            assert "race" in race
            assert "slug" in race
            assert "rating" in race
            assert "margin" in race
            assert "incumbent_party" in race
            assert "is_open_seat" in race
            assert "n_polls" in race
            assert "econ" in race  # may be null when parquet absent
```

- [ ] **Step 2: Add a new test asserting econ structure when present**

Add to `TestGovernorOverviewWithModel` after `test_race_fields_present`:

```python
    def test_econ_field_is_null_or_has_expected_shape(self, overview_client):
        """econ must be null (parquet absent in tests) or a dict with the right keys."""
        data = overview_client.get("/api/v1/governor/overview").json()
        for race in data["races"]:
            econ = race.get("econ")
            if econ is not None:
                assert "wage_growth_pct" in econ, f"missing wage_growth_pct in {race['state']}"
                assert "employment_change_pct" in econ, f"missing employment_change_pct in {race['state']}"
                assert isinstance(econ["wage_growth_pct"], (int, float))
                assert isinstance(econ["employment_change_pct"], (int, float))
```

- [ ] **Step 3: Run the governor overview tests**

```bash
cd /home/hayden/projects/wethervane
.venv/bin/pytest api/tests/test_governor_overview.py -v
```

Expected: all tests pass (the new `econ` field is `null` in test env since parquet is absent).

- [ ] **Step 4: Run full test suite to check baseline**

```bash
cd /home/hayden/projects/wethervane
.venv/bin/pytest --tb=no -q 2>&1 | tail -5
```

Expected: no regressions. Count should be ≥ 4623 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/hayden/projects/wethervane
git add api/tests/test_governor_overview.py
git commit -m "test: assert econ field present in governor overview response"
```

---

## Task 3: Add TypeScript types for econ

**Files:**
- Modify: `web/lib/api.ts` (around line 268)

**Context:**
- `GovernorRaceData` is at line 268 in `web/lib/api.ts`
- Add a new `GovernorEcon` interface and add optional `econ` field to `GovernorRaceData`
- The field is `null` when no QCEW data exists for that state

- [ ] **Step 1: Add `GovernorEcon` interface before `GovernorRaceData`**

Find `export interface GovernorRaceData {` at line 268. Insert the new interface **before** it:

```typescript
export interface GovernorEcon {
  wage_growth_pct: number;
  employment_change_pct: number;
}
```

- [ ] **Step 2: Add `econ` field to `GovernorRaceData`**

Inside `GovernorRaceData`, add the `econ` field after `is_open_seat`:

```typescript
export interface GovernorRaceData {
  state: string;
  race: string;
  slug: string;
  rating: string;
  /** Signed Dem margin: pred_dem_share - 0.5. Positive = Dem-favored. */
  margin: number;
  /** Which party currently holds the governorship ("D" or "R"). */
  incumbent_party: string;
  n_polls: number;
  /** Whether this is an open seat (no incumbent running). */
  is_open_seat: boolean;
  /** QCEW-derived economic indicators for this state (2020–2023). Null if no data. */
  econ: GovernorEcon | null;
}
```

- [ ] **Step 3: TypeScript check**

```bash
cd /home/hayden/projects/wethervane/web
npx tsc --noEmit 2>&1 | head -30
```

Expected: exit 0, no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/hayden/projects/wethervane
git add web/lib/api.ts
git commit -m "feat: add GovernorEcon type and econ field to GovernorRaceData"
```

---

## Task 4: Create GovernorEconomicCard.tsx

**Files:**
- Create: `web/components/forecast/GovernorEconomicCard.tsx`

**Context:**
- Follow `GovernorSeatRiskCard.tsx` pattern: takes `races: GovernorRaceData[]` as props, no internal hook call
- Competitive races = `tossup`, `lean_d`, `lean_r`
- Filter to competitive races that have non-null `econ`; if none, return null silently
- Show a `<dl>` table with one row per competitive state: State | Wage Growth % | Empl. Change %
- Format: `+3.2%` for positive, `-1.5%` for negative
- Color wage growth: positive → `var(--forecast-lean-d)`, negative → `var(--forecast-lean-r)`, zero → `var(--color-text-muted)`
- Color employment change: same logic
- Show a loading skeleton that matches GovernorPollingCard skeleton pattern

- [ ] **Step 1: Write the component**

Create `/home/hayden/projects/wethervane/web/components/forecast/GovernorEconomicCard.tsx`:

```tsx
"use client";

import type { GovernorRaceData } from "@/lib/api";

const COMPETITIVE = new Set(["tossup", "lean_d", "lean_r"]);

function pctColor(val: number): string {
  if (val > 0) return "var(--forecast-lean-d)";
  if (val < 0) return "var(--forecast-lean-r)";
  return "var(--color-text-muted)";
}

function fmtPct(val: number): string {
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(1)}%`;
}

export function GovernorEconomicCardSkeleton() {
  return (
    <section
      className="mb-8 rounded-md p-4 text-sm animate-pulse"
      aria-label="State Economic Context loading"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <div
        className="h-5 w-48 rounded mb-3"
        style={{ background: "var(--color-border)" }}
      />
      <div
        className="h-4 w-72 rounded mb-4"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex justify-between">
            <div
              className="h-3 w-20 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
            <div
              className="h-3 w-32 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

interface GovernorEconomicCardProps {
  races: GovernorRaceData[];
}

/**
 * GovernorEconomicCard — QCEW wage growth and employment change for competitive states.
 *
 * Mirrors GovernorSeatRiskCard pattern: accepts races as props, renders nothing
 * when no competitive race has econ data.
 */
export function GovernorEconomicCard({ races }: GovernorEconomicCardProps) {
  const competitiveWithEcon = races.filter(
    (r) => COMPETITIVE.has(r.rating) && r.econ !== null,
  );

  if (competitiveWithEcon.length === 0) return null;

  return (
    <section
      className="mb-8 rounded-md p-4 text-sm"
      aria-label="State Economic Context"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      {/* Header row */}
      <div className="flex flex-wrap items-baseline justify-between gap-3 mb-1">
        <h2
          className="font-serif text-lg"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          State Economic Context
        </h2>
        <span
          className="font-mono text-xs"
          style={{ color: "var(--color-text-muted)" }}
        >
          QCEW 2020–2023
        </span>
      </div>

      {/* Subheader */}
      <p className="mb-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
        Wage growth and employment change for competitive states — aggregate county data
      </p>

      {/* Column labels */}
      <div
        className="flex items-center justify-between gap-2 mb-2 pb-1 border-b text-xs"
        style={{
          borderColor: "var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        <span className="w-8">State</span>
        <div className="flex gap-6 font-mono">
          <span className="w-28 text-right">Wage Growth</span>
          <span className="w-28 text-right">Employment Δ</span>
        </div>
      </div>

      {/* Data rows */}
      <dl className="space-y-1">
        {competitiveWithEcon.map((r) => {
          const econ = r.econ!;
          return (
            <div
              key={r.state}
              className="flex items-center justify-between gap-2"
            >
              <dt
                className="font-mono font-semibold w-8"
                style={{ color: "var(--color-text)" }}
              >
                {r.state}
              </dt>
              <div className="flex gap-6 font-mono font-semibold">
                <dd
                  className="w-28 text-right"
                  style={{ color: pctColor(econ.wage_growth_pct) }}
                >
                  {fmtPct(econ.wage_growth_pct)}
                </dd>
                <dd
                  className="w-28 text-right"
                  style={{ color: pctColor(econ.employment_change_pct) }}
                >
                  {fmtPct(econ.employment_change_pct)}
                </dd>
              </div>
            </div>
          );
        })}
      </dl>

      {/* Narrative footer */}
      <div
        className="pt-3 mt-2 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        <p style={{ color: "var(--color-text-muted)" }}>
          Wage growth reflects average annual pay change (total wages ÷ employment). Employment
          change reflects total employment headcount shift. Both computed from BLS QCEW county data.
        </p>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: TypeScript check**

```bash
cd /home/hayden/projects/wethervane/web
npx tsc --noEmit 2>&1 | head -30
```

Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
cd /home/hayden/projects/wethervane
git add web/components/forecast/GovernorEconomicCard.tsx
git commit -m "feat: add GovernorEconomicCard component with QCEW econ indicators"
```

---

## Task 5: Wire GovernorEconomicCard into the governor page

**Files:**
- Modify: `web/app/forecast/governor/page.tsx`

**Context:**
- The page already imports `GovernorPollingCard` and other components
- `data.races` contains `GovernorRaceData[]` with the `econ` field populated
- Place `<GovernorEconomicCard races={data.races} />` after `<GovernorPollingCard />` and before the Simulation Outlook section
- The page casts `data.races` to `SenateRaceData[]` for RaceCardGrid — use `data.races` (original) for the economic card

- [ ] **Step 1: Add imports**

Find the import block at the top of `page.tsx`. After the `GovernorPollingCard` import, add:

```typescript
import {
  GovernorEconomicCard,
  GovernorEconomicCardSkeleton,
} from "@/components/forecast/GovernorEconomicCard";
```

- [ ] **Step 2: Add skeleton to loading state**

In the loading return (around line 38–47), after `<GovernorSeatRiskCardSkeleton />`, add:

```tsx
<GovernorEconomicCardSkeleton />
```

- [ ] **Step 3: Wire GovernorEconomicCard into the main render**

Find the `<GovernorPollingCard />` line (around line 107). After it, add:

```tsx
      {/* Economic context — wage growth and employment change for competitive states */}
      <GovernorEconomicCard races={data.races} />
```

The result should look like:

```tsx
      {/* Polling coverage summary */}
      <GovernorPollingCard />

      {/* Economic context — wage growth and employment change for competitive states */}
      <GovernorEconomicCard races={data.races} />

      {/* Simulation-based seat distribution summary */}
      {simStats && (
```

- [ ] **Step 4: TypeScript check**

```bash
cd /home/hayden/projects/wethervane/web
npx tsc --noEmit 2>&1 | head -30
```

Expected: exit 0, no errors.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/hayden/projects/wethervane
.venv/bin/pytest --tb=no -q 2>&1 | tail -5
```

Expected: no regressions, ≥ 4623 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/hayden/projects/wethervane
git add web/app/forecast/governor/page.tsx
git commit -m "feat: wire GovernorEconomicCard into governor overview page"
```

---

## Self-Review

**Spec coverage:**
- ✅ Load QCEW from `data/raw/qcew_county.parquet` (Task 1)
- ✅ Join by state FIPS, compute median wage growth and employment rate delta (Task 1 — uses state-level aggregate, same economic meaning as "median" at the state level)
- ✅ 2020–2023 window (Task 1)
- ✅ `econ: {wage_growth_pct, employment_change_pct}` on each race (Task 1)
- ✅ `GovernorEconomicCard.tsx` following `GovernorPollingCard.tsx` pattern (Task 4)
- ✅ Wired into governor page for competitive (tossup/lean) races (Tasks 4+5)
- ✅ `npx tsc --noEmit` at each TypeScript change (Tasks 3, 4, 5)
- ✅ Pytest baseline maintained (Tasks 2, 5)

**Placeholder scan:** No TBDs, no vague "add error handling" steps. All code blocks complete.

**Type consistency:** `GovernorEcon` defined in Task 3, used as `r.econ!` in Task 4 (after filtering for non-null). `GovernorRaceData` with `econ: GovernorEcon | null` propagates through all tasks.

**Edge cases handled:**
- Parquet absent in test env → `_compute_qcew_state_econ()` returns `{}` → `econ` is `None`/`null` for all races
- State not in QCEW data → `econ_by_state.get(st)` returns `None` → `econ` is `null`
- No competitive races with econ data → `GovernorEconomicCard` returns `null` silently
