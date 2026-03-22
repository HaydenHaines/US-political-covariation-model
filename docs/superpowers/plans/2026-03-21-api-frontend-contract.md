# API–Frontend Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frontend fully data-driven so model pipeline changes never break the visualizer.

**Architecture:** The API is the contract boundary. The frontend fetches all model metadata (super-type names, colors, demographics) from API endpoints and hardcodes nothing about model shape. Pipeline validation catches schema violations at build time. Integration tests verify the full chain.

**Tech Stack:** Next.js (frontend), FastAPI + DuckDB (API), Python (pipeline), pytest (tests)

**Spec:** `docs/superpowers/specs/2026-03-21-api-frontend-contract-design.md`

---

### Task 1: Add `validate_contract()` to build_database.py

**Files:**
- Modify: `src/db/build_database.py`
- Test: `tests/test_build_database.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_database.py
"""Tests for DuckDB contract validation."""
import duckdb
import pytest

from src.db.build_database import validate_contract


def _make_valid_db() -> duckdb.DuckDBPyConnection:
    """Build a minimal in-memory DuckDB that passes contract validation."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE counties (county_fips VARCHAR, state_abbr VARCHAR, county_name VARCHAR)")
    con.execute("INSERT INTO counties VALUES ('12001', 'FL', 'Alachua')")
    con.execute("CREATE TABLE super_types (super_type_id INTEGER, display_name VARCHAR)")
    con.execute("INSERT INTO super_types VALUES (0, 'Test Super Type')")
    con.execute("CREATE TABLE types (type_id INTEGER, super_type_id INTEGER, display_name VARCHAR)")
    con.execute("INSERT INTO types VALUES (0, 0, 'Test Type')")
    con.execute("CREATE TABLE county_type_assignments (county_fips VARCHAR, dominant_type INTEGER, super_type INTEGER)")
    con.execute("INSERT INTO county_type_assignments VALUES ('12001', 0, 0)")
    return con


def test_valid_db_passes():
    con = _make_valid_db()
    errors = validate_contract(con)
    con.close()
    assert errors == []


def test_missing_table_detected():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE counties (county_fips VARCHAR, state_abbr VARCHAR, county_name VARCHAR)")
    errors = validate_contract(con)
    con.close()
    assert any("MISSING TABLE: super_types" in e for e in errors)


def test_missing_column_detected():
    con = _make_valid_db()
    con.execute("DROP TABLE super_types")
    con.execute("CREATE TABLE super_types (super_type_id INTEGER)")  # missing display_name
    errors = validate_contract(con)
    con.close()
    assert any("MISSING COLUMN: super_types.display_name" in e for e in errors)


def test_orphan_super_type_detected():
    con = _make_valid_db()
    con.execute("DELETE FROM county_type_assignments")
    con.execute("INSERT INTO county_type_assignments VALUES ('12001', 0, 99)")  # super_type 99 not in super_types
    errors = validate_contract(con)
    con.close()
    assert any("ORPHAN super_type" in e for e in errors)


def test_orphan_dominant_type_detected():
    con = _make_valid_db()
    con.execute("DELETE FROM county_type_assignments")
    con.execute("INSERT INTO county_type_assignments VALUES ('12001', 99, 0)")  # dominant_type 99 not in types
    errors = validate_contract(con)
    con.close()
    assert any("ORPHAN dominant_type" in e for e in errors)


def test_optional_predictions_not_required():
    """Predictions table is optional — missing it should not cause errors."""
    con = _make_valid_db()
    errors = validate_contract(con)
    con.close()
    assert not any("predictions" in e for e in errors)


def test_optional_predictions_validated_if_present():
    """If predictions exists, its columns are validated."""
    con = _make_valid_db()
    con.execute("CREATE TABLE predictions (county_fips VARCHAR)")  # missing race, pred_dem_share
    errors = validate_contract(con)
    con.close()
    assert any("MISSING COLUMN: predictions.race" in e for e in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_database.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate_contract'`

- [ ] **Step 3: Implement `validate_contract` in build_database.py**

Add this function before the `build()` function in `src/db/build_database.py`:

```python
def validate_contract(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Validate DuckDB matches the API-frontend contract.

    Returns a list of violation strings. Empty list = pass.
    See docs/superpowers/specs/2026-03-21-api-frontend-contract-design.md
    """
    errors: list[str] = []

    required = {
        "super_types": ["super_type_id", "display_name"],
        "types": ["type_id", "super_type_id", "display_name"],
        "county_type_assignments": ["county_fips", "dominant_type", "super_type"],
        "counties": ["county_fips", "state_abbr", "county_name"],
    }

    optional = {
        "predictions": ["county_fips", "race", "pred_dem_share"],
    }

    def _check_table(table: str, columns: list[str], is_required: bool) -> None:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            if is_required:
                errors.append(f"MISSING TABLE: {table}")
            return
        actual_cols = set(con.execute(f"SELECT * FROM \"{table}\" LIMIT 0").fetchdf().columns)
        for col in columns:
            if col not in actual_cols:
                errors.append(f"MISSING COLUMN: {table}.{col}")

    for table, columns in required.items():
        _check_table(table, columns, is_required=True)
    for table, columns in optional.items():
        _check_table(table, columns, is_required=False)

    # Referential integrity (only if required tables exist)
    if not any("MISSING TABLE" in e for e in errors):
        orphans = con.execute("""
            SELECT DISTINCT cta.super_type
            FROM county_type_assignments cta
            LEFT JOIN super_types st ON cta.super_type = st.super_type_id
            WHERE st.super_type_id IS NULL AND cta.super_type IS NOT NULL
        """).fetchdf()
        if not orphans.empty:
            ids = orphans["super_type"].tolist()
            errors.append(f"ORPHAN super_type values in county_type_assignments: {ids}")

        orphan_types = con.execute("""
            SELECT DISTINCT cta.dominant_type
            FROM county_type_assignments cta
            LEFT JOIN types t ON cta.dominant_type = t.type_id
            WHERE t.type_id IS NULL AND cta.dominant_type IS NOT NULL
        """).fetchdf()
        if not orphan_types.empty:
            ids = orphan_types["dominant_type"].tolist()
            errors.append(f"ORPHAN dominant_type values in county_type_assignments: {ids}")

    return errors
```

Then add the validation call at the end of the `build()` function, just before `con.close()` (after the summary query block):

```python
    # ── Contract validation ────────────────────────────────────────────────────
    errors = validate_contract(con)
    if errors:
        for e in errors:
            log.error("CONTRACT VIOLATION: %s", e)
        con.close()
        import sys
        sys.exit(1)
    log.info("Contract validation passed")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_build_database.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/db/build_database.py tests/test_build_database.py
git commit -m "feat: add contract validation to build_database.py"
```

---

### Task 2: Add API startup contract check and degraded health

**Files:**
- Modify: `api/main.py:72-146` (lifespan function)
- Modify: `api/routers/meta.py` (health endpoint)
- Modify: `api/models.py` (HealthResponse)
- Test: existing `api/tests/test_meta.py`

- [ ] **Step 1: Write the failing test**

Add to `api/tests/test_meta.py` (or a new test if the file doesn't cover health):

```python
def test_health_reports_degraded_without_types(client_no_types):
    """Health endpoint reports degraded when contract tables are missing."""
    resp = client_no_types.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["contract"] == "degraded"
```

Add the `client_no_types` fixture to `api/tests/conftest.py`:

```python
@pytest.fixture
def client_no_types():
    """TestClient with a DB that has no type-primary tables."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE counties (county_fips VARCHAR PRIMARY KEY, state_abbr VARCHAR, state_fips VARCHAR, county_name VARCHAR)")
    con.execute("INSERT INTO counties VALUES ('12001', 'FL', '12', 'Alachua')")
    con.execute("CREATE TABLE model_versions (version_id VARCHAR PRIMARY KEY, role VARCHAR, k INTEGER, j INTEGER, shift_type VARCHAR, vote_share_type VARCHAR, n_training_dims INTEGER, n_holdout_dims INTEGER, holdout_r VARCHAR, geography VARCHAR, description VARCHAR, created_at TIMESTAMP)")
    con.execute("INSERT INTO model_versions VALUES ('test_v1', 'current', 3, 7, 'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')")
    con.execute("CREATE TABLE community_assignments (county_fips VARCHAR, community_id INTEGER, k INTEGER, version_id VARCHAR, PRIMARY KEY(county_fips, k, version_id))")
    con.execute("INSERT INTO community_assignments VALUES ('12001', 0, 3, 'test_v1')")
    con.execute("CREATE TABLE community_sigma (community_id_row INTEGER, community_id_col INTEGER, sigma_value DOUBLE, version_id VARCHAR)")
    con.execute("INSERT INTO community_sigma VALUES (0, 0, 0.01, 'test_v1')")
    # Deliberately NO types, super_types, or county_type_assignments tables

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = con
    test_app.state.version_id = "test_v1"
    test_app.state.K = 3
    test_app.state.sigma = np.eye(3) * 0.01
    test_app.state.mu_prior = np.full(3, 0.42)
    test_app.state.state_weights = pd.DataFrame()
    test_app.state.county_weights = pd.DataFrame()
    test_app.state.contract_ok = False  # simulates missing tables

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c
    con.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest api/tests/test_meta.py -v -k degraded`
Expected: FAIL — `contract` key not in response / fixture not found

- [ ] **Step 3: Implement**

In `api/models.py`, add `contract` field to `HealthResponse`:

```python
class HealthResponse(BaseModel):
    status: str
    db: str
    contract: str = "ok"
```

In `api/main.py` lifespan, after loading type data (before `yield`), add:

```python
    # ── Contract check ─────────────────────────────────────────────────────────
    contract_ok = True
    for table_name in ["super_types", "types", "county_type_assignments"]:
        try:
            result = app.state.db.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            if not result or result[0] == 0:
                log.warning("CONTRACT: missing table %s — frontend will show degraded state", table_name)
                contract_ok = False
        except Exception:
            contract_ok = False
    app.state.contract_ok = contract_ok
    log.info("Contract status: %s", "ok" if contract_ok else "degraded")
```

In `api/routers/meta.py`, update health endpoint:

```python
@router.get("/health", response_model=HealthResponse)
def health(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "error"
    contract = "ok" if getattr(request.app.state, "contract_ok", True) else "degraded"
    return HealthResponse(status="ok", db=db_status, contract=contract)
```

Update the existing `client` fixture in `conftest.py` to set `contract_ok = True`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest api/tests/ -v`
Expected: All pass (existing + new degraded test)

- [ ] **Step 5: Commit**

```bash
git add api/main.py api/routers/meta.py api/models.py api/tests/
git commit -m "feat: add API startup contract check and degraded health reporting"
```

---

### Task 3: Make MapShell.tsx fully data-driven

This is the largest task — removes all hardcoded model knowledge from the map component.

**Files:**
- Rewrite: `web/components/MapShell.tsx`
- Modify: `web/components/TypePanel.tsx` (remove import of deleted exports)

- [ ] **Step 1: Rewrite MapShell.tsx**

Replace the entire file. Key changes:
- Delete `SUPER_TYPE_COLORS`, `COUNTY_SUPER_TYPE_NAMES`, `TRACT_SUPER_TYPE_NAMES`, `SUPER_TYPE_NAMES`, `COMMUNITY_COLORS`
- Add `PALETTE` (15-color const array — visual concern only)
- Add `superTypeMap` state built from `/super-types` API response
- `getColor` reads from `PALETTE[super_type_id % PALETTE.length]`
- Tooltip reads name from `superTypeMap` (county view) or GeoJSON `super_type_name` property (tract view), falling back to `"Type {id}"`
- Legend built from `superTypeMap` entries, filtered to IDs present in data
- Toggle button no longer mutates globals

```tsx
"use client";
import { useState, useEffect, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchCounties, fetchSuperTypes, type CountyRow } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";
import { TypePanel } from "@/components/TypePanel";

// 15-color perceptually-distinct palette. Assigned by super_type_id.
// Purely a visual concern — the model does not know about colors.
export const PALETTE: [number, number, number][] = [
  [31, 119, 180],   // 0: blue
  [255, 127, 14],   // 1: orange
  [44, 160, 44],    // 2: green
  [214, 39, 40],    // 3: red
  [148, 103, 189],  // 4: purple
  [140, 86, 75],    // 5: brown
  [227, 119, 194],  // 6: pink
  [127, 127, 127],  // 7: gray
  [188, 189, 34],   // 8: olive
  [23, 190, 207],   // 9: teal
  [174, 199, 232],  // 10: light blue
  [255, 187, 120],  // 11: light orange
  [152, 223, 138],  // 12: light green
  [255, 152, 150],  // 13: light red
  [197, 176, 213],  // 14: light purple
];

export function getColorForSuperType(superTypeId: number): [number, number, number] {
  if (superTypeId < 0) return [180, 180, 180];
  return PALETTE[superTypeId % PALETTE.length];
}

const INITIAL_VIEW = {
  longitude: -84.5,
  latitude: 31.5,
  zoom: 5.8,
  pitch: 0,
  bearing: 0,
};

export interface SuperTypeInfo {
  name: string;
  color: [number, number, number];
}

export default function MapShell() {
  const { selectedCommunityId, setSelectedCommunityId, selectedTypeId, setSelectedTypeId } = useMapContext();
  const [geojson, setGeojson] = useState<any>(null);
  const [tractGeojson, setTractGeojson] = useState<any>(null);
  const [countyMap, setCountyMap] = useState<Record<string, CountyRow>>({});
  const [superTypeMap, setSuperTypeMap] = useState<Map<number, SuperTypeInfo>>(new Map());
  const [hasTypeData, setHasTypeData] = useState(false);
  const [showTracts, setShowTracts] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/counties-fl-ga-al.geojson").then((r) => r.json()),
      fetchCounties(),
      fetchSuperTypes().catch(() => []),
      fetch("/tract-communities.geojson").then((r) => r.json()).catch(() => null),
    ]).then(([geo, counties, superTypes, tractGeo]) => {
      if (tractGeo) setTractGeojson(tractGeo);

      // Build super-type map from API
      const stMap = new Map<number, SuperTypeInfo>();
      superTypes.forEach((st: any) => {
        stMap.set(st.super_type_id, {
          name: st.display_name,
          color: getColorForSuperType(st.super_type_id),
        });
      });
      setSuperTypeMap(stMap);

      // Build county map
      const map: Record<string, CountyRow> = {};
      let typeDataPresent = false;
      counties.forEach((c: CountyRow) => {
        map[c.county_fips] = c;
        if (c.super_type !== null) typeDataPresent = true;
      });
      setCountyMap(map);
      setHasTypeData(typeDataPresent);

      // Enrich GeoJSON
      const enriched = {
        ...geo,
        features: geo.features.map((f: any) => ({
          ...f,
          properties: {
            ...f.properties,
            community_id: map[f.properties.county_fips]?.community_id ?? -1,
            dominant_type: map[f.properties.county_fips]?.dominant_type ?? -1,
            super_type: map[f.properties.county_fips]?.super_type ?? -1,
          },
        })),
      };
      setGeojson(enriched);
    });
  }, []);

  const getColor = useCallback(
    (f: any): [number, number, number, number] => {
      const st: number = f.properties?.super_type ?? -1;
      const dt: number = f.properties?.dominant_type ?? f.properties?.type_id ?? -1;

      if (st >= 0 || hasTypeData) {
        const isSelected = selectedTypeId !== null && dt === selectedTypeId;
        const base = getColorForSuperType(st);
        return [...base, isSelected ? 255 : 180] as [number, number, number, number];
      }
      // Fallback for legacy community data (no type data)
      return [180, 180, 180, 120] as [number, number, number, number];
    },
    [selectedTypeId, hasTypeData, showTracts]
  );

  const getLineWidth = useCallback(
    (f: any): number => {
      if (hasTypeData) {
        const dt: number = f.properties?.dominant_type ?? -1;
        return selectedTypeId !== null && dt === selectedTypeId ? 800 : 200;
      }
      const cid: number = f.properties?.community_id ?? -1;
      return selectedCommunityId !== null && cid === selectedCommunityId ? 800 : 200;
    },
    [selectedCommunityId, selectedTypeId, hasTypeData]
  );

  const getSuperTypeName = useCallback(
    (superTypeId: number, feature?: any): string => {
      // County view: read from API-populated map
      if (!showTracts) {
        return superTypeMap.get(superTypeId)?.name ?? `Type ${superTypeId}`;
      }
      // Tract view: read from GeoJSON property, fall back to map, then generic
      const geoName = feature?.properties?.super_type_name;
      if (geoName) return geoName;
      return superTypeMap.get(superTypeId)?.name ?? `Type ${superTypeId}`;
    },
    [showTracts, superTypeMap]
  );

  const activeData = showTracts && tractGeojson ? tractGeojson : geojson;
  const layerId = showTracts && tractGeojson ? "tract-communities" : "counties";

  const layers = activeData
    ? [
        new GeoJsonLayer({
          id: layerId,
          data: activeData,
          pickable: true,
          stroked: true,
          filled: true,
          getFillColor: getColor as any,
          getLineColor: [80, 80, 80, 120],
          getLineWidth,
          lineWidthUnits: "meters",
          updateTriggers: {
            getFillColor: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts],
            getLineWidth: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts],
          },
          onHover: ({ object, x, y }: any) => {
            if (object) {
              if (showTracts && tractGeojson) {
                const st = object.properties?.super_type;
                const tid = object.properties?.type_id;
                const n = object.properties?.n_tracts;
                const area = object.properties?.area_sqkm;
                const stName = getSuperTypeName(st, object);
                setTooltip({ x, y, text: `${stName}\nType ${tid} · ${n} tracts · ${Math.round(area)} km²` });
              } else {
                const name = object.properties?.county_name || object.properties?.county_fips;
                if (hasTypeData) {
                  const st = object.properties?.super_type;
                  const dt = object.properties?.dominant_type;
                  const stName = getSuperTypeName(st, object);
                  setTooltip({ x, y, text: `${name}\n${stName} (Type ${dt})` });
                } else {
                  const cid = object.properties?.community_id;
                  setTooltip({ x, y, text: `${name}\nCommunity ${cid}` });
                }
              }
            } else {
              setTooltip(null);
            }
          },
          onClick: ({ object }: any) => {
            if (object) {
              if (hasTypeData) {
                const dt = object.properties?.dominant_type;
                if (dt !== undefined && dt >= 0) {
                  setSelectedTypeId(dt === selectedTypeId ? null : dt);
                  setSelectedCommunityId(null);
                }
              } else {
                const cid = object.properties?.community_id;
                if (cid !== undefined && cid >= 0) {
                  setSelectedCommunityId(cid === selectedCommunityId ? null : cid);
                  setSelectedTypeId(null);
                }
              }
            }
          },
        }),
      ]
    : [];

  // Build legend from API data — only show super-types that appear in counties
  const activeSuperTypeIds = new Set<number>();
  if (hasTypeData && !showTracts) {
    Object.values(countyMap).forEach((c) => {
      if (c.super_type !== null) activeSuperTypeIds.add(c.super_type);
    });
  } else if (showTracts && tractGeojson) {
    tractGeojson.features?.forEach((f: any) => {
      const st = f.properties?.super_type;
      if (st != null && st >= 0) activeSuperTypeIds.add(st);
    });
  }

  const legendEntries = Array.from(activeSuperTypeIds)
    .sort((a, b) => a - b)
    .map((id) => ({
      id,
      color: getColorForSuperType(id),
      label: superTypeMap.get(id)?.name ?? `Type ${id}`,
    }));

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller={true}
        layers={layers}
        style={{ background: "#e8ecf0" }}
      />

      {tooltip && (
        <div style={{
          position: "absolute",
          left: tooltip.x + 12,
          top: tooltip.y + 12,
          background: "white",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
          padding: "6px 10px",
          fontSize: "12px",
          fontFamily: "var(--font-sans)",
          pointerEvents: "none",
          whiteSpace: "pre-line",
          boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
        }}>
          {tooltip.text}
        </div>
      )}

      {/* County/Tract toggle */}
      {tractGeojson && (
        <button
          onClick={() => setShowTracts((prev) => !prev)}
          style={{
            position: "absolute",
            top: 12,
            left: 16,
            background: showTracts ? "#2166ac" : "white",
            color: showTracts ? "white" : "#333",
            border: "1px solid var(--color-border)",
            borderRadius: "4px",
            padding: "6px 14px",
            fontSize: "12px",
            fontFamily: "var(--font-sans)",
            cursor: "pointer",
            fontWeight: 600,
            boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
          }}
        >
          {showTracts ? "Tract Communities" : "County Types"} ▾
        </button>
      )}

      {/* Legend */}
      {legendEntries.length > 0 && (
        <div style={{
          position: "absolute",
          bottom: 24,
          left: 16,
          background: "white",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
          padding: "8px 12px",
          fontSize: "11px",
          fontFamily: "var(--font-sans)",
        }}>
          {legendEntries.map((entry) => (
            <div key={entry.id} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
              <div style={{
                width: 12, height: 12, borderRadius: 2,
                background: `rgb(${entry.color.join(",")})`,
              }} />
              <span style={{ color: "var(--color-text-muted)" }}>{entry.label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Side panels */}
      {selectedCommunityId !== null && !hasTypeData && (
        <CommunityPanel
          communityId={selectedCommunityId}
          onClose={() => setSelectedCommunityId(null)}
        />
      )}

      {selectedTypeId !== null && hasTypeData && (
        <TypePanel
          typeId={selectedTypeId}
          superTypeMap={superTypeMap}
          onClose={() => setSelectedTypeId(null)}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 2: Update TypePanel.tsx to receive superTypeMap as prop**

**Line 5** — Replace import:
```tsx
// OLD:
import { SUPER_TYPE_COLORS, SUPER_TYPE_NAMES } from "@/components/MapShell";
// NEW:
import { getColorForSuperType, type SuperTypeInfo } from "@/components/MapShell";
```

**Lines 7-10** — Update Props interface:
```tsx
interface Props {
  typeId: number;
  superTypeMap: Map<number, SuperTypeInfo>;
  onClose: () => void;
}
```

**Line 75** — Update function signature:
```tsx
// OLD:
export function TypePanel({ typeId, onClose }: Props) {
// NEW:
export function TypePanel({ typeId, superTypeMap, onClose }: Props) {
```

**Lines 87-89** — Replace color lookup:
```tsx
// OLD:
const superColor = detail
  ? SUPER_TYPE_COLORS[detail.super_type_id % SUPER_TYPE_COLORS.length]
  : [127, 127, 127];
// NEW:
const superColor = detail
  ? getColorForSuperType(detail.super_type_id)
  : [127, 127, 127];
```

**Lines 126-128** — Replace name lookup:
```tsx
// OLD:
{SUPER_TYPE_NAMES[detail.super_type_id] && (
  <> &middot; {SUPER_TYPE_NAMES[detail.super_type_id]}</>
)}
// NEW:
{superTypeMap.get(detail.super_type_id)?.name && (
  <> &middot; {superTypeMap.get(detail.super_type_id)!.name}</>
)}
```

- [ ] **Step 3: Verify the build succeeds**

Run: `cd web && npx next build`
Expected: `✓ Compiled successfully`

- [ ] **Step 4: Commit**

```bash
git add web/components/MapShell.tsx web/components/TypePanel.tsx
git commit -m "feat: make MapShell fully data-driven — no hardcoded model knowledge"
```

---

### Task 4: Make TypePanel demographics rendering generic

**Files:**
- Modify: `web/components/TypePanel.tsx`

- [ ] **Step 1: Replace hardcoded demographics with generic renderer**

Replace the 7 hardcoded `DemographicRow` calls (lines 162-168) with a loop that renders all keys from the `demographics` dict:

```tsx
// Add this constant at module level (before the component)
const DEMO_DISPLAY: Record<string, { label: string; fmt: "pct" | "dollar" | "num" }> = {
  median_hh_income: { label: "Median income", fmt: "dollar" },
  median_age: { label: "Median age", fmt: "num" },
  pct_white_nh: { label: "White (non-Hispanic)", fmt: "pct" },
  pct_black: { label: "Black", fmt: "pct" },
  pct_hispanic: { label: "Hispanic", fmt: "pct" },
  pct_asian: { label: "Asian", fmt: "pct" },
  pct_bachelors_plus: { label: "Bachelor's+", fmt: "pct" },
  pct_owner_occupied: { label: "Owner-occupied", fmt: "pct" },
  pct_wfh: { label: "Work from home", fmt: "pct" },
  pct_transit: { label: "Transit commuters", fmt: "pct" },
  pct_car: { label: "Car commuters", fmt: "pct" },
  evangelical_share: { label: "Evangelical", fmt: "pct" },
  mainline_share: { label: "Mainline Protestant", fmt: "pct" },
  catholic_share: { label: "Catholic", fmt: "pct" },
  black_protestant_share: { label: "Black Protestant", fmt: "pct" },
  congregations_per_1000: { label: "Congregations/1K", fmt: "num" },
  religious_adherence_rate: { label: "Religious adherence", fmt: "num" },
};

// Skip these keys in demographics rendering (raw counts, not useful for display)
const DEMO_SKIP = new Set([
  "pop_total", "pop_white_nh", "pop_black", "pop_asian", "pop_hispanic",
  "housing_total", "housing_owner", "educ_total", "educ_bachelors_plus",
  "commute_total", "commute_car", "commute_transit", "commute_wfh",
  "n_counties",
]);

function prettifyKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\bpct\b/g, "%")
    .replace(/\bhh\b/g, "household")
    .replace(/^./, (c) => c.toUpperCase());
}

function inferFormat(key: string): "pct" | "dollar" | "num" {
  if (key.startsWith("pct_") || key.endsWith("_share")) return "pct";
  if (key.includes("income")) return "dollar";
  return "num";
}
```

Then replace the demographics section in the JSX:

```tsx
{/* Demographics */}
{Object.keys(detail.demographics).length > 0 && (
  <div style={{ marginBottom: "16px" }}>
    <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
      Demographics
    </p>
    {Object.entries(detail.demographics)
      .filter(([key]) => !DEMO_SKIP.has(key))
      .map(([key, value]) => {
        const display = DEMO_DISPLAY[key];
        return (
          <DemographicRow
            key={key}
            label={display?.label ?? prettifyKey(key)}
            value={value}
            fmt={display?.fmt ?? inferFormat(key)}
          />
        );
      })}
  </div>
)}
```

- [ ] **Step 2: Verify the build succeeds**

Run: `cd web && npx next build`
Expected: `✓ Compiled successfully`

- [ ] **Step 3: Commit**

```bash
git add web/components/TypePanel.tsx
git commit -m "feat: generic demographics rendering — new features auto-display"
```

---

### Task 5: Fix ForecastView to use state_abbr from data

**Files:**
- Modify: `web/components/ForecastView.tsx`

- [ ] **Step 1: Replace race string parsing with state_abbr grouping**

Replace lines 89-90:

```tsx
// OLD:
const selectedState = selectedRace.split(" ")[1] ?? "";
const stateRows = displayRows.filter((r) => r.state_abbr === selectedState);
```

With:

```tsx
// Derive state from the data, never parse race strings
const stateRows = displayRows.filter((r) => r.race === selectedRace);
const selectedState = stateRows.length > 0 ? stateRows[0].state_abbr : "";
```

Also fix line 109 — `r.replace("_", " ")` is unnecessary since race strings already have spaces:

```tsx
// OLD:
{races.map((r) => <option key={r} value={r}>{r.replace("_", " ")}</option>)}
// NEW:
{races.map((r) => <option key={r} value={r}>{r}</option>)}
```

- [ ] **Step 2: Verify the build succeeds**

Run: `cd web && npx next build`
Expected: `✓ Compiled successfully`

- [ ] **Step 3: Commit**

```bash
git add web/components/ForecastView.tsx
git commit -m "fix: ForecastView uses state_abbr from data, never parses race strings"
```

---

### Task 6: Add super_type_name to tract GeoJSON

**Files:**
- Modify: `src/viz/bubble_dissolve.py`

- [ ] **Step 1: Add `super_type_name` property to dissolved communities**

In `bubble_dissolve.py`, the `communities.append()` call (around line 90-97) builds the feature dict. It currently includes `super_type` as an integer. Add `super_type_name` by looking up the name from the assignments DataFrame or a separate lookup.

Since `bubble_dissolve.py` takes a `--assignments` parquet which has `super_type` but not `super_type_name`, the simplest approach is to accept an optional `--super-type-names` JSON file or embed the name from the `super_types.parquet` file:

Add a `--super-types` argument:

```python
p.add_argument(
    "--super-types",
    type=Path,
    default=None,
    help="Path to super_types.parquet for display names. If omitted, names default to 'Type {id}'.",
)
```

Load the names in `main()`:

```python
st_names: dict[int, str] = {}
if args.super_types and args.super_types.exists():
    st_df = pd.read_parquet(args.super_types)
    st_names = dict(zip(st_df["super_type_id"], st_df["display_name"]))
```

Then in the community dict construction, add:

```python
"super_type_name": st_names.get(super_type, f"Type {super_type}"),
```

- [ ] **Step 2: Verify the output GeoJSON includes the property**

Run: `uv run python -m src.viz.bubble_dissolve --help`
Expected: shows `--super-types` argument

- [ ] **Step 3: Commit**

```bash
git add src/viz/bubble_dissolve.py
git commit -m "feat: include super_type_name in tract GeoJSON for frontend"
```

---

### Task 7: Integration test — contract end-to-end

**Files:**
- Create: `tests/test_api_contract.py`

- [ ] **Step 1: Write the integration tests**

These use the real DuckDB file and a test app instance with the real lifespan.

```python
# tests/test_api_contract.py
"""Integration tests: validate DuckDB→API→frontend contract.

These tests use the REAL bedrock.duckdb to catch pipeline/API mismatches.
Skip gracefully if the DB file doesn't exist (CI without data).
"""
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

DB_PATH = Path("data/bedrock.duckdb")

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="data/bedrock.duckdb not found — skip contract integration tests",
)


@pytest.fixture(scope="module")
def real_client():
    """TestClient backed by the real bedrock.duckdb."""
    from api.main import create_app, lifespan

    app = create_app()  # uses real lifespan
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_db():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


# ── Schema tests ──────────────────────────────────────────────────────────

def test_duckdb_contract(real_db):
    """Required tables and columns exist in bedrock.duckdb."""
    from src.db.build_database import validate_contract

    errors = validate_contract(real_db)
    assert errors == [], f"Contract violations: {errors}"


# ── API response shape tests ──────────────────────────────────────────────

def test_super_types_response(real_client):
    resp = real_client.get("/api/v1/super-types")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0, "super-types must not be empty"
    for st in data:
        assert "super_type_id" in st
        assert "display_name" in st
        assert isinstance(st["display_name"], str)
        assert len(st["display_name"]) > 0


def test_counties_reference_valid_super_types(real_client):
    super_types = {st["super_type_id"] for st in real_client.get("/api/v1/super-types").json()}
    counties = real_client.get("/api/v1/counties").json()
    for c in counties:
        if c["super_type"] is not None:
            assert c["super_type"] in super_types, \
                f"County {c['county_fips']} has super_type={c['super_type']} not in super-types"


def test_forecast_has_state_abbr(real_client):
    rows = real_client.get("/api/v1/forecast").json()
    if not rows:
        pytest.skip("No forecast data")
    for r in rows:
        assert "state_abbr" in r
        assert isinstance(r["state_abbr"], str)
        assert len(r["state_abbr"]) == 2


def test_type_detail_has_dynamic_dicts(real_client):
    types = real_client.get("/api/v1/types").json()
    if not types:
        pytest.skip("No types in database")
    detail = real_client.get(f"/api/v1/types/{types[0]['type_id']}").json()
    assert isinstance(detail["demographics"], dict)
    if detail["shift_profile"] is not None:
        assert isinstance(detail["shift_profile"], dict)
        # Verify no non-numeric metadata leaked into shift_profile
        for key, val in detail["shift_profile"].items():
            assert isinstance(val, (int, float)), \
                f"shift_profile[{key}] is {type(val).__name__}, expected number"


# ── Cross-layer consistency ───────────────────────────────────────────────

def test_super_type_coverage(real_client):
    super_type_ids = {st["super_type_id"] for st in real_client.get("/api/v1/super-types").json()}
    county_super_types = {c["super_type"] for c in real_client.get("/api/v1/counties").json() if c["super_type"] is not None}
    assert county_super_types <= super_type_ids, \
        f"Counties reference super-types not in API: {county_super_types - super_type_ids}"
    unused = super_type_ids - county_super_types
    if unused:
        import warnings
        warnings.warn(f"Super-types with no counties: {unused}")


def test_health_reports_contract_status(real_client):
    resp = real_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "contract" in data
    assert data["contract"] in ("ok", "degraded")
```

- [ ] **Step 2: Run the integration tests**

Run: `uv run pytest tests/test_api_contract.py -v`
Expected: All pass (or skip if DB not present)

- [ ] **Step 3: Commit**

```bash
git add tests/test_api_contract.py
git commit -m "feat: add contract integration tests — DuckDB→API→frontend chain"
```

---

### Task 8: Update test conftest, add super_types table, rebuild and verify

**Files:**
- Modify: `api/tests/conftest.py` (add `super_types` table to test DB)
- Verify: all existing API tests still pass

- [ ] **Step 1: Update test DB to include super_types table**

In `api/tests/conftest.py`, in the `_build_test_db()` function, add after the `types` table creation:

```python
    con.execute("""
        CREATE TABLE super_types (
            super_type_id INTEGER PRIMARY KEY,
            display_name VARCHAR
        )
    """)
    con.execute("INSERT INTO super_types VALUES (0, 'Rural & Conservative')")
    con.execute("INSERT INTO super_types VALUES (1, 'Suburban & Moderate')")
```

Also set `contract_ok = True` on the test app state in the `client` fixture:

```python
    test_app.state.contract_ok = True
```

- [ ] **Step 2: Run all API tests**

Run: `uv run pytest api/tests/ -v`
Expected: All pass

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: All tests pass (existing + new)

- [ ] **Step 4: Rebuild frontend and restart services**

```bash
cd web && npx next build
cp -r public .next/standalone/web/
cp -r .next/static .next/standalone/web/.next/
systemctl --user restart bedrock-api bedrock-frontend
```

- [ ] **Step 5: Verify live endpoints**

```bash
curl -s http://localhost:8002/api/v1/health
# Expected: {"status":"ok","db":"connected","contract":"ok"}

curl -s http://localhost:8002/api/v1/super-types | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d)} super-types, first: {d[0][\"display_name\"]}')"
# Expected: 5 super-types, first: Southern Rural Conservative
```

- [ ] **Step 6: Commit**

```bash
git add api/tests/conftest.py
git commit -m "chore: align test fixtures with contract (add super_types table)"
```
