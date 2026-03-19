# Phase 2: API Layer + Frontend — Design Spec

**Date:** 2026-03-19
**Status:** Approved
**Scope:** Views 1 (Community Map) + 3 (2026 Forecast) — Views 2 and 4 deferred
**Target domain:** bedrock.vote (production VPS deployment)

---

## Goal

Build the public-facing Bedrock platform: a FastAPI backend serving electoral model data from DuckDB, and a Next.js frontend with a persistent choropleth map and tabbed data panel. Ship Views 1 (Community Map) and 3 (2026 Forecast) for the 2026 midterm cycle.

---

## Architecture & Service Layout

Three processes behind a single Caddy reverse proxy:

```
bedrock.vote
     │
  [Caddy]  :80/:443  — TLS termination (Let's Encrypt auto), routing
     ├── /api/*  →  FastAPI  :8000
     └── /*      →  Next.js  :3000
```

**FastAPI** (`api/` directory)
- Opens `data/bedrock.duckdb` as a read-only connection at startup
- Serves all data endpoints under `/api/v1/`
- Python 3.11 + uvicorn
- No auth for now — all reads, no writes
- OpenAPI docs auto-generated at `/api/docs`

**Next.js** (`web/` directory)
- App Router (Next.js 14+)
- Server components for static metadata; client components for Deck.gl and poll input
- County GeoJSON for FL+GA+AL served as static file from `web/public/counties-fl-ga-al.geojson`

**DuckDB**
- Not a service — `data/bedrock.duckdb` mounted read-only into the API container
- Single file, read-only connection, no connection pool needed

**Orchestration**
- `docker-compose.yml` at project root starts all three services
- `Caddyfile` at project root defines routing rules

### Directory structure additions

```
US-political-covariation-model/
├── api/
│   ├── main.py                  # FastAPI app factory, lifespan (DB open/close)
│   ├── db.py                    # DuckDB connection dependency
│   ├── routers/
│   │   ├── communities.py       # /communities endpoints
│   │   ├── counties.py          # /counties endpoint
│   │   ├── forecast.py          # /forecast endpoints including feed-a-poll
│   │   └── meta.py              # /health, /model/version
│   ├── models.py                # Pydantic response models
│   ├── tests/
│   │   ├── conftest.py          # TestClient + synthetic DuckDB fixture
│   │   ├── test_communities.py
│   │   ├── test_counties.py
│   │   ├── test_forecast.py
│   │   └── test_meta.py
│   ├── requirements.txt
│   └── Dockerfile
├── web/
│   ├── app/
│   │   ├── layout.tsx           # Root layout: persistent map shell + tab panel
│   │   ├── page.tsx             # Redirects to /forecast (default view)
│   │   ├── forecast/
│   │   │   └── page.tsx         # View 3: 2026 Forecast
│   │   └── globals.css
│   ├── components/
│   │   ├── MapShell.tsx         # Persistent Deck.gl map (client component)
│   │   ├── CommunityPanel.tsx   # Side panel: profile, sparkline, county list
│   │   ├── TabBar.tsx           # Top tab navigation for the panel
│   │   ├── ForecastView.tsx     # View 3 content
│   │   ├── FeedAPoll.tsx        # Poll input card (client component)
│   │   └── MapContext.tsx       # selectedCommunityId React context
│   ├── public/
│   │   └── counties-fl-ga-al.geojson   # Static county boundaries (FL+GA+AL)
│   ├── lib/
│   │   └── api.ts               # Typed fetch wrappers for API calls
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── Caddyfile
└── docker-compose.yml
```

---

## API Design

All endpoints under `/api/v1/`. All responses JSON. CORS open for now (bedrock.vote + localhost dev).

### Endpoints

#### `GET /api/v1/health`
```json
{ "status": "ok", "db": "connected" }
```

#### `GET /api/v1/model/version`
```json
{
  "version_id": "county_multiyear_logodds_20260319",
  "k": 10,
  "j": 7,
  "holdout_r": 0.9027,
  "shift_type": "logodds",
  "date_created": "2026-03-19"
}
```

#### `GET /api/v1/communities`
Returns all K communities with summary stats for choropleth legend and panel list.
```json
[
  {
    "community_id": 0,
    "n_counties": 28,
    "states": ["FL", "GA"],
    "dominant_type_id": 3,
    "mean_pred_dem_share": 0.41
  }
]
```

#### `GET /api/v1/communities/{community_id}`
Full community profile for the side panel.
```json
{
  "community_id": 0,
  "n_counties": 28,
  "states": ["FL", "GA"],
  "dominant_type_id": 3,
  "counties": [
    { "county_fips": "12001", "state_abbr": "FL", "pred_dem_share": 0.39 }
  ],
  "shift_profile": {
    "pres_d_shift_00_04": -0.02,
    "pres_d_shift_04_08": 0.04
    // ... all training shift cols
  }
}
```
Returns 404 if community_id not found.

#### `GET /api/v1/counties`
All counties with community assignment — drives choropleth coloring.
```json
[
  { "county_fips": "12001", "state_abbr": "FL", "community_id": 0 }
]
```

#### `GET /api/v1/forecast`
2026 county-level predictions. Query params: `race` (e.g. `FL_Senate`), `state` (e.g. `FL`). Both optional; returns all predictions if omitted.
```json
[
  {
    "county_fips": "12001",
    "state_abbr": "FL",
    "race": "FL_Senate",
    "pred_dem_share": 0.39,
    "pred_std": 0.04,
    "pred_lo90": 0.33,
    "pred_hi90": 0.45,
    "state_pred": 0.44,
    "poll_avg": 0.46
  }
]
```

#### `POST /api/v1/forecast/poll`
Feed-a-poll: runs Bayesian community update, returns updated county predictions.

Request:
```json
{
  "state": "FL",
  "race": "FL_Senate",
  "dem_share": 0.47,
  "n": 600
}
```

Response: same shape as `GET /forecast` but with updated `pred_dem_share` values reflecting the poll signal propagated through community structure. Calls `src/propagation/propagate_polls.py` logic directly (Python import, not subprocess).

---

## Navigation Model

The map is the persistent canvas. The right panel is the content area with a tab bar at the top. For Phase 2, one tab: **Forecast**. Future views (2, 4) drop in as additional tabs without changing the map.

```
┌─────────────────────────────────┬────────────────────────────┐
│                                 │  [Forecast] [···future···]  │
│                                 │─────────────────────────────│
│       Community Map             │                             │
│       (Deck.gl, persistent)     │   Tab content               │
│                                 │   (View 3 / future)         │
│                                 │                             │
└─────────────────────────────────┴────────────────────────────┘
```

Routes:
- `/` → redirects to `/forecast`
- `/forecast` → map + Forecast tab active
- (future) `/communities`, `/sabermetrics` → map stays, panel switches

Community side panel (triggered by map click) overlays the right panel temporarily — it is not a tab. Closing the panel returns to the active tab view.

---

## View 1: Community Map

**Choropleth layer** (`Deck.gl GeoJsonLayer`)
- On mount: fetch `/api/v1/counties` and load `counties-fl-ga-al.geojson` in parallel; merge community_id into GeoJSON features client-side
- Color by community_id — 10-color muted palette (colorblind-safe, Clean Academic style)
- Hover: tooltip with county name + community id
- Click: sets `selectedCommunityId` in context → all counties in that community highlight (brighter fill); community side panel opens

**Community side panel** (slides in, overlays right panel)
- Community id + county count + states at top
- Shift sparkline (Observable Plot): community's mean shift value across training election pairs — the political fingerprint
- Dominant NMF type label
- Scrollable county list: name, state, predicted 2026 dem share (first available race)
- Close button: clears `selectedCommunityId`, panel closes

**Map controls**
- Zoom/pan (Deck.gl defaults)
- Legend strip bottom-left: 10 color swatches
- Reset button clears community selection

**Data fetched on demand:** `/api/v1/communities/{id}` called when panel opens (not pre-fetched for all communities).

---

## View 3: 2026 Forecast

**Race selector** (top of panel, below tab bar)
- Dropdown listing available races from `/api/v1/forecast` distinct races
- Default: first available race (FL_Senate)

**State summary card**
- Predicted statewide dem share (population-weighted mean of county predictions)
- Margin label: Solid R / Lean R / Toss-up / Lean D / Solid D
- Partisan color bar

**Forecast bar chart** (Observable Plot)
- Counties sorted by predicted dem share (left = most R, right = most D)
- Bars colored by partisan lean
- Y-axis: 0–100% dem share; reference line at 50%

**County table**
- County name, state, predicted dem share, 90% interval, community assignment
- Sortable by margin

**Feed-a-Poll card**
- Label: *"What if the polls show..."*
- Dem share input (slider 0–100% with text display) + sample size input (default 600)
- "Update" button → `POST /forecast/poll` → bar chart and table update in-place
- Small methodological note: *"Bayesian update propagated through community covariance structure"*
- "Reset to baseline" link restores original predictions

**Visual style:** `#2166ac` Democrat blue, `#d73027` Republican red (Economist-style, not neon). Georgia serif for race titles; system sans-serif for data.

---

## Testing

### API (pytest + FastAPI TestClient)

`api/tests/conftest.py` provides:
- A synthetic in-memory DuckDB with the same schema as `bedrock.duckdb` (populated with ~10 fake counties across 3 communities, following the pattern in `tests/test_db_builder.py`)
- A `TestClient` fixture wrapping the FastAPI app with the test DB injected

One test file per router:
- `test_meta.py`: health returns 200, version returns correct fields
- `test_communities.py`: list returns K items, detail returns correct shape, 404 on unknown id
- `test_counties.py`: returns all counties, each has community_id
- `test_forecast.py`: baseline predictions return correct shape; `POST /forecast/poll` returns updated values distinct from baseline

### Frontend (Playwright e2e)

`web/e2e/` directory:
- Map loads and county polygons render
- Click a county → side panel appears with community data, community counties highlight
- Forecast tab loads race predictions and bar chart
- Feed-a-Poll: enter dem share, click Update → predicted values change from baseline

No React component unit tests for Phase 2 — e2e coverage is sufficient.

---

## Future Features (not in Phase 2 scope)

### Community Age Slider (View 1)
A slider in map controls (range: 3–10 training pairs) that re-renders the choropleth showing how communities change with different historical depth. Requires:
- Pre-computing HAC community assignments at each training depth (3, 4, ..., 10 pairs) and storing in DuckDB with an `n_pairs` key
- `GET /api/v1/communities?n_pairs=5` filter on the communities and counties endpoints
- DuckDB schema must include `n_pairs` column on `community_assignments` table from day one (even if only populated for n_pairs=10 initially)

### Views 2 and 4 as Panel Tabs
When Views 2 (Community Profiles deep-dive) and 4 (Political Sabermetrics) are built, they slot into the tab bar alongside Forecast. No structural changes to the map or layout needed.

---

## Out of Scope (Phase 2)

- Auth / rate limiting (deferred until API goes public)
- National geography (FL+GA+AL only)
- Tract-level resolution
- View 2 (Community Profiles), View 4 (Sabermetrics)
- MRP full pipeline (poll propagation uses existing Gaussian Bayesian update)
- Mobile layout (desktop-first for Phase 2)
