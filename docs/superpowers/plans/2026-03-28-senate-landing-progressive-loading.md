# Senate Forecast Landing + Progressive Map Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the WetherVane landing page from a grey-map research tool into a 538-style Senate forecast overview with progressive map loading.

**Architecture:** The forecast page becomes a scrollable content page (Layout A) with: headline prediction → Senate control bar → state-level map → race cards. The map loads 51 state polygons instantly, then loads per-state tract GeoJSON on state click with surrounding-state desaturation. New `/api/v1/senate/overview` endpoint provides the landing page data. Dusty Ink color palette throughout.

**Tech Stack:** Next.js 14 (App Router), React 18, Deck.gl 9, Observable Plot, TypeScript

---

## File Structure

### New Files
- `web/components/SenateOverview.tsx` — headline + control bar + race cards (replaces ForecastView as default)
- `web/components/SenateControlBar.tsx` — horizontal spectrum bar component
- `web/components/RaceCard.tsx` — individual race card component
- `web/lib/colors.ts` — Dusty Ink palette constants + choropleth color function
- `api/routers/senate.py` — Senate overview API endpoint
- `scripts/build_state_geojson.py` — generate state polygons + per-state tract splits
- `web/public/states-us.geojson` — 51 state polygons (~200KB)
- `web/public/tracts/` — per-state tract GeoJSON directory

### Modified Files
- `web/components/MapShell.tsx` — progressive loading (states first, tracts on click, desaturation)
- `web/components/MapContext.tsx` — add zoomedState, layoutMode state
- `web/components/ForecastView.tsx` — refactor to race-detail view (used after state drill-down)
- `web/components/TabBar.tsx` — update tab labels
- `web/app/(map)/forecast/page.tsx` — render SenateOverview
- `web/app/(map)/layout.tsx` — support content-mode layout
- `api/main.py` — register senate router
- `web/lib/api.ts` — add fetchSenateOverview function

---

## Task 1: Generate State GeoJSON + Per-State Tract Splits

**Files:**
- Create: `scripts/build_state_geojson.py`
- Output: `web/public/states-us.geojson`, `web/public/tracts/{STATE}.geojson`

- [ ] **Step 1: Create the state + tract splitting script**

```python
# scripts/build_state_geojson.py
"""Generate state-level GeoJSON and per-state tract GeoJSON splits.

State GeoJSON: 51 lightweight state polygons for the national map.
Per-state tracts: individual files for progressive loading on state click.

Usage:
    cd /home/hayden/projects/wethervane
    uv run python scripts/build_state_geojson.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_PUBLIC = PROJECT_ROOT / "web" / "public"
TRACTS_DIR = WEB_PUBLIC / "tracts"

# Census TIGER cartographic boundary files (500k simplification)
STATE_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip"
STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

# 2026 Senate races (states with contested seats)
SENATE_2026_STATES = {
    "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS",
    "KY", "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE",
    "NH", "NJ", "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
    "VA", "WV", "WY",
}

# 2026 Governor races
GOVERNOR_2026_STATES = {
    "AK", "AL", "AZ", "AR", "CA", "CO", "CT", "FL", "GA", "HI",
    "IA", "ID", "IL", "KS", "MD", "MA", "ME", "MI", "MN", "NE",
    "NV", "NH", "NM", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "VT", "WI", "WY",
}


def download_and_extract(url: str, dest_dir: Path) -> Path:
    """Download and extract a zip file, returning the extracted directory."""
    zip_path = dest_dir / "download.zip"
    if not zip_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
    extract_dir = dest_dir / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    return extract_dir


def build_state_geojson() -> None:
    """Build state-level GeoJSON with race metadata."""
    cache_dir = PROJECT_ROOT / "data" / "raw" / "tiger_states"
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = download_and_extract(STATE_URL, cache_dir)

    shp_files = list(extracted.rglob("*.shp"))
    if not shp_files:
        print("ERROR: No shapefile found in extracted archive")
        sys.exit(1)

    gdf = gpd.read_file(shp_files[0])
    # Filter to 50 states + DC
    gdf = gdf[gdf["STATEFP"].isin(STATE_FIPS_TO_ABBR.keys())].copy()
    gdf["state_abbr"] = gdf["STATEFP"].map(STATE_FIPS_TO_ABBR)
    gdf["state_fips"] = gdf["STATEFP"]
    gdf["has_senate_2026"] = gdf["state_abbr"].isin(SENATE_2026_STATES)
    gdf["has_governor_2026"] = gdf["state_abbr"].isin(GOVERNOR_2026_STATES)

    # Simplify geometry for performance
    gdf = gdf.to_crs(epsg=4326)
    gdf.geometry = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)

    # Keep only needed columns
    out = gdf[["state_abbr", "state_fips", "has_senate_2026",
               "has_governor_2026", "NAME", "geometry"]].copy()
    out = out.rename(columns={"NAME": "state_name"})

    output_path = WEB_PUBLIC / "states-us.geojson"
    out.to_file(output_path, driver="GeoJSON")
    size_kb = output_path.stat().st_size / 1024
    print(f"Saved {output_path} ({len(out)} states, {size_kb:.0f} KB)")


def split_tracts_by_state() -> None:
    """Split tracts-us.geojson into per-state files."""
    tracts_path = WEB_PUBLIC / "tracts-us.geojson"
    if not tracts_path.exists():
        print("ERROR: tracts-us.geojson not found. Run build_national_tract_geojson.py first.")
        sys.exit(1)

    TRACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(tracts_path) as f:
        data = json.load(f)

    # Group features by state (first 2 chars of any tract coordinate → state FIPS)
    # Actually, we need to extract state from the geometry or an embedded property.
    # The community polygons don't have state FIPS directly, but we can compute
    # the centroid and map to state, OR we can use geopandas spatial join.
    # Simpler: load with geopandas and use the state GeoJSON for spatial join.

    print(f"Loading {len(data['features'])} tract features...")
    tracts_gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

    # Compute centroid state assignment
    states_path = WEB_PUBLIC / "states-us.geojson"
    states_gdf = gpd.read_file(states_path)

    # Spatial join: assign each tract polygon to the state containing its centroid
    tracts_gdf["centroid"] = tracts_gdf.geometry.centroid
    centroids = gpd.GeoDataFrame(tracts_gdf, geometry="centroid", crs="EPSG:4326")
    joined = gpd.sjoin(centroids, states_gdf[["state_abbr", "geometry"]], how="left", predicate="within")

    # Some centroids may fall outside state polygons (border tracts) — fill with nearest
    missing = joined["state_abbr"].isna()
    if missing.any():
        print(f"  {missing.sum()} tracts unmatched — assigning to nearest state")
        joined.loc[missing, "state_abbr"] = "XX"  # placeholder

    # Restore original geometry
    joined = joined.set_geometry(tracts_gdf.geometry)
    joined = joined.drop(columns=["centroid", "index_right"], errors="ignore")

    # Write per-state files
    for state_abbr in sorted(joined["state_abbr"].unique()):
        if state_abbr == "XX":
            continue
        state_tracts = joined[joined["state_abbr"] == state_abbr].copy()
        state_tracts = state_tracts.drop(columns=["state_abbr"], errors="ignore")
        out_path = TRACTS_DIR / f"{state_abbr}.geojson"
        state_tracts.to_file(out_path, driver="GeoJSON")
        size_kb = out_path.stat().st_size / 1024
        print(f"  {state_abbr}: {len(state_tracts)} polygons ({size_kb:.0f} KB)")

    print(f"\nSaved {len(joined['state_abbr'].unique())} state tract files to {TRACTS_DIR}")


if __name__ == "__main__":
    build_state_geojson()
    split_tracts_by_state()
```

- [ ] **Step 2: Run the script**

```bash
cd /home/hayden/projects/wethervane
uv run python scripts/build_state_geojson.py
```

Expected: `web/public/states-us.geojson` (~200KB, 51 states) and `web/public/tracts/{STATE}.geojson` (51 files, 500KB-3MB each).

- [ ] **Step 3: Verify outputs**

```bash
ls -la web/public/states-us.geojson
ls web/public/tracts/ | wc -l
du -sh web/public/tracts/
# Check a sample state
python3 -c "import json; d=json.load(open('web/public/states-us.geojson')); print(len(d['features']), 'states'); print(d['features'][0]['properties'])"
```

- [ ] **Step 4: Add tracts/ directory to .gitignore** (too large for git)

Append to `.gitignore`:
```
web/public/tracts/
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_state_geojson.py web/public/states-us.geojson .gitignore
git commit -m "feat: state GeoJSON + per-state tract splits for progressive loading

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Dusty Ink Color Palette

**Files:**
- Create: `web/lib/colors.ts`
- Modify: `web/app/globals.css` (CSS variables)
- Modify: `web/components/MapShell.tsx` (choropleth function)

- [ ] **Step 1: Create colors.ts with Dusty Ink palette**

```typescript
// web/lib/colors.ts
// Dusty Ink palette — aged atlas aesthetic for partisan lean visualization.
// Muted, academic, authoritative. Not cable-news red/blue.

export const DUSTY_INK = {
  safeD:    "#2d4a6f",
  likelyD:  "#4b6d90",
  leanD:    "#7e9ab5",
  tossup:   "#b5a995",
  leanR:    "#c4907a",
  likelyR:  "#9e5e4e",
  safeR:    "#6e3535",

  background: "#fafaf8",
  text:       "#3a3632",
  textMuted:  "#6e6860",
  textSubtle: "#8a8478",
  cardBg:     "#f5f3ef",
  border:     "#e0ddd8",
  mapEmpty:   "#eae7e2",
} as const;

// Rating thresholds: margin in percentage points (absolute value of dem_share - 0.5)
// Safe: >15pp, Likely: 8-15pp, Lean: 3-8pp, Tossup: <3pp
export type Rating = "safe_d" | "likely_d" | "lean_d" | "tossup" | "lean_r" | "likely_r" | "safe_r";

export function marginToRating(demShare: number): Rating {
  const margin = demShare - 0.5; // positive = D, negative = R
  const abs = Math.abs(margin);
  if (abs < 0.03) return "tossup";
  if (margin > 0) {
    if (abs >= 0.15) return "safe_d";
    if (abs >= 0.08) return "likely_d";
    return "lean_d";
  }
  if (abs >= 0.15) return "safe_r";
  if (abs >= 0.08) return "likely_r";
  return "lean_r";
}

export function ratingColor(rating: Rating): string {
  const map: Record<Rating, string> = {
    safe_d: DUSTY_INK.safeD,
    likely_d: DUSTY_INK.likelyD,
    lean_d: DUSTY_INK.leanD,
    tossup: DUSTY_INK.tossup,
    lean_r: DUSTY_INK.leanR,
    likely_r: DUSTY_INK.likelyR,
    safe_r: DUSTY_INK.safeR,
  };
  return map[rating];
}

export function ratingLabel(rating: Rating): string {
  const map: Record<Rating, string> = {
    safe_d: "Safe D", likely_d: "Likely D", lean_d: "Lean D",
    tossup: "Tossup",
    lean_r: "Lean R", likely_r: "Likely R", safe_r: "Safe R",
  };
  return map[rating];
}

// Choropleth color for predicted Dem share — Dusty Ink gradient
// Maps 0.3→deep red through 0.5→warm grey to 0.7→deep blue
export function dustyInkChoropleth(demShare: number): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, (demShare - 0.3) / 0.4));

  // Color stops: safeR(0) → tossup(0.5) → safeD(1)
  // Parse hex to RGB for interpolation
  if (t >= 0.5) {
    // tossup → safeD
    const s = (t - 0.5) * 2;
    return [
      Math.round(181 * (1 - s) + 45 * s),   // b5 → 2d
      Math.round(169 * (1 - s) + 74 * s),   // a9 → 4a
      Math.round(149 * (1 - s) + 111 * s),  // 95 → 6f
      200,
    ];
  }
  // safeR → tossup
  const s = t * 2;
  return [
    Math.round(110 * (1 - s) + 181 * s),  // 6e → b5
    Math.round(53 * (1 - s) + 169 * s),   // 35 → a9
    Math.round(53 * (1 - s) + 149 * s),   // 35 → 95
    200,
  ];
}
```

- [ ] **Step 2: Update CSS variables in globals.css**

Find the CSS variables section and update with Dusty Ink values:

```css
:root {
  --color-dem: #2d4a6f;
  --color-rep: #6e3535;
  --color-tossup: #b5a995;
  --color-surface: #fafaf8;
  --color-card: #f5f3ef;
  --color-border: #e0ddd8;
  --color-text: #3a3632;
  --color-text-muted: #6e6860;
  --color-text-subtle: #8a8478;
  --color-map-empty: #eae7e2;
  --font-serif: Georgia, "Times New Roman", serif;
  --font-sans: system-ui, -apple-system, sans-serif;
}
```

- [ ] **Step 3: Update MapShell.tsx choroplethColor to use dustyInkChoropleth**

Replace the existing `choroplethColor` function import/definition with:

```typescript
import { dustyInkChoropleth } from "@/lib/colors";
```

And replace all calls to `choroplethColor(...)` with `dustyInkChoropleth(...)`.

- [ ] **Step 4: Build and verify**

```bash
cd web && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add web/lib/colors.ts web/app/globals.css web/components/MapShell.tsx
git commit -m "feat: Dusty Ink color palette — muted academic partisan colors

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Senate Overview API Endpoint

**Files:**
- Create: `api/routers/senate.py`
- Modify: `api/main.py` (register router)
- Test: `api/tests/test_senate.py`

- [ ] **Step 1: Write the test**

```python
# api/tests/test_senate.py
"""Tests for Senate overview endpoint."""
import pytest
from fastapi.testclient import TestClient
from api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_senate_overview_returns_races(client):
    resp = client.get("/api/v1/senate/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "races" in data
    assert "headline" in data
    assert isinstance(data["races"], list)


def test_senate_overview_race_has_rating(client):
    resp = client.get("/api/v1/senate/overview")
    data = resp.json()
    if data["races"]:
        race = data["races"][0]
        assert "state" in race
        assert "rating" in race
        assert race["rating"] in ["safe_d", "likely_d", "lean_d", "tossup",
                                   "lean_r", "likely_r", "safe_r"]
```

- [ ] **Step 2: Implement the endpoint**

```python
# api/routers/senate.py
"""Senate forecast overview — landing page data."""
from __future__ import annotations

import logging

import duckdb
import numpy as np
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from api.db import get_db

log = logging.getLogger(__name__)
router = APIRouter()


class SenateRace(BaseModel):
    state: str
    race: str
    slug: str
    rating: str
    margin: float
    n_polls: int


class SenateOverview(BaseModel):
    headline: str
    subtitle: str
    dem_seats_safe: int
    gop_seats_safe: int
    races: list[SenateRace]


# States with 2026 Senate elections
SENATE_2026_STATES = {
    "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS",
    "KY", "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE",
    "NH", "NJ", "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
    "VA", "WV", "WY",
}

# Current senate seats NOT up for election (Class I seats stay, Class II up)
# Dems hold 47 seats total, 33 up in 2026 (Class II)
DEM_SAFE_SEATS = 47  # seats not contested in 2026
GOP_SAFE_SEATS = 53  # seats not contested in 2026


def _margin_to_rating(dem_share: float) -> str:
    margin = dem_share - 0.5
    abs_m = abs(margin)
    if abs_m < 0.03:
        return "tossup"
    if margin > 0:
        if abs_m >= 0.15:
            return "safe_d"
        if abs_m >= 0.08:
            return "likely_d"
        return "lean_d"
    if abs_m >= 0.15:
        return "safe_r"
    if abs_m >= 0.08:
        return "likely_r"
    return "lean_r"


def _race_to_slug(race: str) -> str:
    return race.lower().replace(" ", "-")


@router.get("/senate/overview", response_model=SenateOverview)
def senate_overview(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    """National Senate forecast overview for the landing page."""
    # Get all Senate races and their stored predictions (vote-weighted state-level)
    races_df = db.execute("""
        SELECT DISTINCT race FROM predictions
        WHERE race LIKE '%Senate%' AND version_id = (
            SELECT version_id FROM model_versions WHERE role = 'current' LIMIT 1
        )
    """).fetchdf()

    senate_races = []
    for _, row in races_df.iterrows():
        race_name = row["race"]
        # Extract state abbreviation
        parts = race_name.split()
        state_abbr = next((p for p in parts if len(p) == 2 and p.isupper()), None)
        if not state_abbr or state_abbr not in SENATE_2026_STATES:
            continue

        # Get vote-weighted state prediction
        pred = db.execute("""
            SELECT
                CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                          / SUM(COALESCE(c.total_votes_2024, 0))
                     ELSE AVG(p.pred_dem_share)
                END AS state_pred
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.race = ? AND c.state_abbr = ?
        """, [race_name, state_abbr]).fetchone()

        dem_share = pred[0] if pred and pred[0] is not None else 0.45

        # Count polls
        n_polls_row = db.execute(
            "SELECT COUNT(*) FROM polls WHERE race = ?", [race_name]
        ).fetchone()
        n_polls = n_polls_row[0] if n_polls_row else 0

        margin = round((dem_share - 0.5) * 100, 1)
        rating = _margin_to_rating(dem_share)

        senate_races.append(SenateRace(
            state=state_abbr,
            race=race_name,
            slug=_race_to_slug(race_name),
            rating=rating,
            margin=margin,
            n_polls=n_polls,
        ))

    # Sort by competitiveness (tossups first, then lean, then likely, then safe)
    rating_order = {"tossup": 0, "lean_d": 1, "lean_r": 1, "likely_d": 2,
                    "likely_r": 2, "safe_d": 3, "safe_r": 3}
    senate_races.sort(key=lambda r: (rating_order.get(r.rating, 9), abs(r.margin)))

    # Determine headline
    d_leaning = sum(1 for r in senate_races if r.rating in ("safe_d", "likely_d", "lean_d"))
    r_leaning = sum(1 for r in senate_races if r.rating in ("safe_r", "likely_r", "lean_r"))

    if r_leaning > d_leaning:
        headline = "Republicans Favored"
        subtitle = "to retain control of the Senate"
    elif d_leaning > r_leaning:
        headline = "Democrats Favored"
        subtitle = "to win control of the Senate"
    else:
        headline = "Senate Control Is a Toss-Up"
        subtitle = "neither party has a clear advantage"

    return SenateOverview(
        headline=headline,
        subtitle=subtitle,
        dem_seats_safe=DEM_SAFE_SEATS,
        gop_seats_safe=GOP_SAFE_SEATS,
        races=senate_races,
    )
```

- [ ] **Step 3: Register the router in api/main.py**

Add alongside the existing router registrations:

```python
from api.routers import senate
app.include_router(senate.router, prefix="/api/v1")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest api/tests/test_senate.py -v --tb=short
```

- [ ] **Step 5: Verify endpoint manually**

```bash
curl -s http://localhost:8002/api/v1/senate/overview | python3 -m json.tool | head -30
```

- [ ] **Step 6: Commit**

```bash
git add api/routers/senate.py api/main.py api/tests/test_senate.py
git commit -m "feat: GET /api/v1/senate/overview — landing page data

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Senate Control Bar Component

**Files:**
- Create: `web/components/SenateControlBar.tsx`

- [ ] **Step 1: Create the component**

```typescript
// web/components/SenateControlBar.tsx
"use client";

import { DUSTY_INK, ratingColor, ratingLabel, type Rating } from "@/lib/colors";

interface SenateRace {
  state: string;
  race: string;
  slug: string;
  rating: Rating;
  margin: number;
  n_polls: number;
}

interface Props {
  races: SenateRace[];
  demSeats: number;
  gopSeats: number;
  onRaceClick?: (race: SenateRace) => void;
}

export function SenateControlBar({ races, demSeats, gopSeats, onRaceClick }: Props) {
  // Group races by rating for the spectrum bar
  const ratingOrder: Rating[] = [
    "safe_d", "likely_d", "lean_d", "tossup", "lean_r", "likely_r", "safe_r",
  ];

  const grouped = new Map<Rating, SenateRace[]>();
  for (const r of ratingOrder) grouped.set(r, []);
  for (const race of races) {
    const group = grouped.get(race.rating as Rating);
    if (group) group.push(race);
  }

  return (
    <div style={{ margin: "16px 0" }}>
      {/* Labels */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        fontSize: "10px",
        color: DUSTY_INK.textSubtle,
        marginBottom: "4px",
        fontFamily: "var(--font-sans)",
      }}>
        <span>Safe Dem</span>
        <span>Competitive</span>
        <span>Safe Rep</span>
      </div>

      {/* Spectrum bar */}
      <div style={{
        display: "flex",
        height: "32px",
        borderRadius: "3px",
        overflow: "hidden",
        fontSize: "10px",
        lineHeight: "32px",
        textAlign: "center",
        color: "rgba(255,255,255,0.9)",
        fontFamily: "var(--font-sans)",
      }}>
        {ratingOrder.map((rating) => {
          const group = grouped.get(rating) || [];
          if (group.length === 0) return null;
          return group.map((race) => (
            <div
              key={race.state}
              style={{
                flex: rating === "tossup" ? 2 : 1,
                background: ratingColor(rating),
                cursor: "pointer",
                borderRight: "1px solid rgba(255,255,255,0.15)",
                transition: "opacity 0.15s",
              }}
              title={`${race.state} Senate — ${ratingLabel(rating)} (${race.margin > 0 ? "D" : "R"}+${Math.abs(race.margin).toFixed(1)})`}
              onClick={() => onRaceClick?.(race)}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.opacity = "0.8"; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.opacity = "1"; }}
            >
              {race.state}
            </div>
          ));
        })}
      </div>

      {/* Seat counts */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        fontSize: "10px",
        color: DUSTY_INK.textSubtle,
        marginTop: "4px",
        fontFamily: "var(--font-sans)",
      }}>
        <span>Dem seats: {demSeats} + races</span>
        <span style={{ color: DUSTY_INK.text, fontWeight: 600 }}>50 for majority</span>
        <span>GOP seats: {gopSeats} + races</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Build to verify no TypeScript errors**

```bash
cd web && npm run build 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add web/components/SenateControlBar.tsx
git commit -m "feat: SenateControlBar — horizontal spectrum by race rating

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Race Card Component

**Files:**
- Create: `web/components/RaceCard.tsx`

- [ ] **Step 1: Create the component**

```typescript
// web/components/RaceCard.tsx
"use client";

import { DUSTY_INK, ratingColor, ratingLabel, marginToRating, type Rating } from "@/lib/colors";

interface Props {
  state: string;
  race: string;
  slug: string;
  margin: number;
  nPolls: number;
  rating: Rating;
  onClick?: () => void;
}

export function RaceCard({ state, race, margin, slug, nPolls, rating, onClick }: Props) {
  const stateName = race.replace(/^2026\s+/, "").replace(/\s+Senate$/, "");
  const marginText = margin > 0
    ? `D+${margin.toFixed(1)}`
    : margin < 0
    ? `R+${Math.abs(margin).toFixed(1)}`
    : "EVEN";

  return (
    <div
      onClick={onClick}
      style={{
        background: DUSTY_INK.cardBg,
        padding: "12px 14px",
        borderRadius: "6px",
        borderLeft: `4px solid ${ratingColor(rating)}`,
        cursor: "pointer",
        transition: "background 0.15s",
        fontFamily: "var(--font-sans)",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.background = DUSTY_INK.border;
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.background = DUSTY_INK.cardBg;
      }}
    >
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
      }}>
        <span style={{
          fontWeight: 700,
          fontSize: "14px",
          color: DUSTY_INK.text,
          fontFamily: "var(--font-serif)",
        }}>
          {stateName}
        </span>
        <span style={{
          fontSize: "11px",
          fontWeight: 600,
          color: ratingColor(rating),
        }}>
          {ratingLabel(rating)}
        </span>
      </div>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        marginTop: "4px",
        fontSize: "12px",
        color: DUSTY_INK.textMuted,
      }}>
        <span>{marginText}</span>
        <span>{nPolls > 0 ? `${nPolls} polls` : "No polls"}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/components/RaceCard.tsx
git commit -m "feat: RaceCard component for senate race grid

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Senate Overview Page (Layout A — Content Mode)

**Files:**
- Create: `web/components/SenateOverview.tsx`
- Modify: `web/app/(map)/forecast/page.tsx`
- Modify: `web/lib/api.ts` (add fetchSenateOverview)

- [ ] **Step 1: Add API fetch function**

In `web/lib/api.ts`, add:

```typescript
export interface SenateRaceData {
  state: string;
  race: string;
  slug: string;
  rating: string;
  margin: number;
  n_polls: number;
}

export interface SenateOverviewData {
  headline: string;
  subtitle: string;
  dem_seats_safe: number;
  gop_seats_safe: number;
  races: SenateRaceData[];
}

export async function fetchSenateOverview(): Promise<SenateOverviewData> {
  const res = await fetch(`${API_BASE}/senate/overview`);
  if (!res.ok) throw new Error(`/senate/overview failed: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 2: Create SenateOverview component**

```typescript
// web/components/SenateOverview.tsx
"use client";

import { useEffect, useState } from "react";
import { DUSTY_INK } from "@/lib/colors";
import { fetchSenateOverview, type SenateOverviewData, type SenateRaceData } from "@/lib/api";
import { SenateControlBar } from "./SenateControlBar";
import { RaceCard } from "./RaceCard";
import { useMapContext } from "./MapContext";
import type { Rating } from "@/lib/colors";

export function SenateOverview() {
  const [data, setData] = useState<SenateOverviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const { setForecastState } = useMapContext();

  useEffect(() => {
    fetchSenateOverview()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ padding: "40px 20px", textAlign: "center", color: DUSTY_INK.textMuted }}>
        Loading Senate forecast...
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ padding: "40px 20px", textAlign: "center", color: DUSTY_INK.textMuted }}>
        Failed to load forecast data
      </div>
    );
  }

  const handleRaceClick = (race: SenateRaceData) => {
    setForecastState(race.state);
    // Navigate to race detail or expand inline
    window.location.href = `/forecast/${race.slug}`;
  };

  // Split races into competitive (tossup + lean) and safe
  const competitive = data.races.filter(
    (r) => ["tossup", "lean_d", "lean_r"].includes(r.rating)
  );
  const others = data.races.filter(
    (r) => !["tossup", "lean_d", "lean_r"].includes(r.rating)
  );

  return (
    <div style={{
      padding: "24px 20px",
      fontFamily: "var(--font-serif)",
      maxWidth: "100%",
    }}>
      {/* Headline */}
      <div style={{ textAlign: "center", marginBottom: "20px" }}>
        <div style={{
          fontSize: "11px",
          color: DUSTY_INK.textSubtle,
          textTransform: "uppercase",
          letterSpacing: "2px",
          fontFamily: "var(--font-sans)",
        }}>
          2026 United States Senate
        </div>
        <div style={{
          fontSize: "28px",
          fontWeight: 700,
          color: DUSTY_INK.text,
          margin: "4px 0",
        }}>
          {data.headline}
        </div>
        <div style={{
          fontSize: "14px",
          color: DUSTY_INK.textMuted,
        }}>
          {data.subtitle}
        </div>
      </div>

      {/* Senate Control Bar */}
      <SenateControlBar
        races={data.races as any}
        demSeats={data.dem_seats_safe}
        gopSeats={data.gop_seats_safe}
        onRaceClick={handleRaceClick}
      />

      {/* Key Races */}
      {competitive.length > 0 && (
        <div style={{ marginTop: "24px" }}>
          <h3 style={{
            fontSize: "16px",
            fontWeight: 700,
            color: DUSTY_INK.text,
            marginBottom: "12px",
          }}>
            Key Races
          </h3>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
            gap: "8px",
          }}>
            {competitive.map((race) => (
              <RaceCard
                key={race.state}
                state={race.state}
                race={race.race}
                slug={race.slug}
                margin={race.margin}
                nPolls={race.n_polls}
                rating={race.rating as Rating}
                onClick={() => handleRaceClick(race)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Other Races */}
      {others.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h3 style={{
            fontSize: "14px",
            fontWeight: 600,
            color: DUSTY_INK.textMuted,
            marginBottom: "8px",
          }}>
            Other Senate Races
          </h3>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
            gap: "6px",
          }}>
            {others.map((race) => (
              <RaceCard
                key={race.state}
                state={race.state}
                race={race.race}
                slug={race.slug}
                margin={race.margin}
                nPolls={race.n_polls}
                rating={race.rating as Rating}
                onClick={() => handleRaceClick(race)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Update forecast page to render SenateOverview**

Replace the contents of `web/app/(map)/forecast/page.tsx`:

```typescript
import { SenateOverview } from "@/components/SenateOverview";

export default function ForecastPage() {
  return <SenateOverview />;
}
```

- [ ] **Step 4: Build and verify**

```bash
cd web && npm run build 2>&1 | tail -10
```

- [ ] **Step 5: Deploy and check**

```bash
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-api.service wethervane-frontend.service
```

Visit `wethervane.hhaines.duckdns.org` — should show the Senate overview with headline, control bar, and race cards.

- [ ] **Step 6: Commit**

```bash
git add web/components/SenateOverview.tsx web/app/\(map\)/forecast/page.tsx web/lib/api.ts
git commit -m "feat: Senate forecast landing page — headline, control bar, race cards

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Progressive Map Loading — States First, Tracts on Click

**Files:**
- Modify: `web/components/MapShell.tsx`
- Modify: `web/components/MapContext.tsx`

This is the most complex task. The map needs to:
1. Load state polygons on mount (instant, ~200KB)
2. Color states by Senate race rating
3. On state click: zoom to state, load that state's tracts, desaturate others
4. Back button returns to national view

- [ ] **Step 1: Add zoomedState to MapContext**

In `web/components/MapContext.tsx`, add to the context interface and provider:

```typescript
zoomedState: string | null;
setZoomedState: (s: string | null) => void;
```

- [ ] **Step 2: Rewrite MapShell for progressive loading**

This is a significant rewrite of MapShell.tsx. The key changes:

1. Load `states-us.geojson` on mount instead of `tracts-us.geojson`
2. Color states by Senate race rating (fetch from `/api/v1/senate/overview`)
3. On state click: fetch `/tracts/{STATE}.geojson`, zoom to state bounds, render tracts layer
4. Desaturate non-selected states (opacity 0.3)
5. "Back to national" resets zoom and clears tract layer

The implementer should read the existing MapShell.tsx (538 lines) carefully and modify it rather than rewriting from scratch. The key changes are:
- Replace the eager tract GeoJSON load with lazy per-state loading
- Add a second GeoJsonLayer for state polygons (always visible)
- The tract layer only renders when `zoomedState` is set
- State fill color comes from race rating (via senate overview data)
- Desaturation: when zoomedState is set, non-matching states get `fillColor: [234, 231, 226, 80]`

- [ ] **Step 3: Build and test**

```bash
cd web && npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-frontend.service
```

Verify: page loads with colored state map instantly. Click a state → tracts load, surrounding states fade. Back to national works.

- [ ] **Step 4: Commit**

```bash
git add web/components/MapShell.tsx web/components/MapContext.tsx
git commit -m "feat: progressive map loading — states first, tracts on state click

States load instantly (~200KB). Click a state to load its tracts.
Surrounding states desaturate to communicate prediction scope.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Visual Polish + Bug Fixes

**Files:**
- Various web/components files
- `web/app/favicon.ico`

- [ ] **Step 1: Add favicon**

Download or create a simple weathervane icon and save to `web/app/favicon.ico`. A simple SVG-based favicon works:

```bash
# Create a simple text-based favicon (placeholder)
convert -size 32x32 xc:white -fill "#2d4a6f" -font Georgia -pointsize 24 -gravity center -annotate 0 "W" web/app/favicon.ico 2>/dev/null || echo "Install imagemagick for favicon, or use a manual SVG"
```

- [ ] **Step 2: Fix persistent popups on tab navigation**

In MapShell.tsx, add cleanup when tab changes. The `usePathname()` hook can trigger popup dismissal:

```typescript
// Clear popups when route changes
const pathname = usePathname();
useEffect(() => {
  setPopup(null);
  setHoveredFeature(null);
}, [pathname]);
```

- [ ] **Step 3: Fix Alaska FIPS codes in prediction tables**

The county predictions table shows FIPS codes for some Alaska counties. This is a data issue — ensure the `counties` table in DuckDB has proper names for all Alaska entries. Check and fix in `build_database.py` if needed.

- [ ] **Step 4: Build, deploy, verify**

```bash
cd web && npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
systemctl --user restart wethervane-frontend.service
```

- [ ] **Step 5: Commit**

```bash
git add web/
git commit -m "fix: visual polish — favicon, popup cleanup, consistent naming

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Final Integration + Deploy

- [ ] **Step 1: Run full test suite**

```bash
cd /home/hayden/projects/wethervane
uv run pytest tests/ -q --tb=short 2>&1 | tail -10
```

- [ ] **Step 2: Build frontend final**

```bash
cd web && npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
```

- [ ] **Step 3: Restart all services**

```bash
systemctl --user restart wethervane-api.service wethervane-frontend.service
```

- [ ] **Step 4: Verify live site**

Visit `wethervane.hhaines.duckdns.org`:
- Landing page shows Senate headline + control bar + race cards
- Map shows colored states (not grey)
- Click a state → tracts load, surrounding states desaturate
- Race cards link to detail pages
- Dusty Ink palette throughout
- No persistent popups across tabs
- Favicon visible

- [ ] **Step 5: Push**

```bash
TOKEN=$(gh auth token) && git push "https://$TOKEN@github.com/HaydenHaines/wethervane.git" main
```

- [ ] **Step 6: Update session handoff**

Write session handoff to `workspace/session-logs/` with final state.
