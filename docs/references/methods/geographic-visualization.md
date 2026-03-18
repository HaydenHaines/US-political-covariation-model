---
source: research synthesis — 2026-03-18
status: not yet implemented; tooling decision recorded here
---

# Geographic Visualization and Spatial Analysis

Two distinct problems with different right tools:
1. **Interactive visualization** — display community membership blobs, toggle layers
2. **Spatial analysis** — test whether community boundaries follow rivers, geology, borders

---

## Visualization: Deck.gl / Kepler.gl

**Primary recommendation for interactive community maps.**

Deck.gl is Uber's WebGL-based geospatial framework. Kepler.gl is a
no-code UI built on top of it for rapid exploration.

**Why it fits:**
- GPU-accelerated; handles 9,300 census tract polygons without stutter
- React-native — embeds cleanly into a custom frontend
- Supports continuous color scales for membership intensity per community
- Layer toggling built in; can show/hide each community type independently
- Click-to-inspect callbacks: show a tract's full 7-component membership vector
- Future overlays (election results, turnout, etc.) are just additional layers

**Recommended implementation path:**
1. **Phase 1 (today):** Use Kepler.gl for Jupyter — load tracts + membership
   weights, explore interactively, no code required
2. **Phase 2 (production):** Deck.gl `PolygonLayer` in React app with toggle UI
3. **Phase 3 (future):** County zoom, election result overlay, time-series animation

**Data format:** Deck.gl expects GeoJSON. Convert parquet → GeoJSON in Python:
```python
import geopandas as gpd
tracts = gpd.read_parquet("data/communities/tract_memberships_k7.parquet")
# join to TIGER tract geometries, then:
tracts.to_file("data/viz/tract_memberships_k7.geojson", driver="GeoJSON")
```

**Visualization strategy (see nmf-community-detection.md):**
Map membership intensity per community as a continuous color scale — NOT
dominant type. Each community gets its own map layer. Heterogeneous tracts
appear light on every layer simultaneously.

**Resources:**
- https://deck.gl/docs
- https://docs.kepler.gl/

---

## Static Exploration: Google Earth + KML

For quick visual sanity checks — "does this blob follow that river?" —
you can export community membership as KML with polygon fill opacity
proportional to membership weight and load it in Google Earth (consumer app).

```python
import simplekml
# set polygon fill color/opacity per tract based on membership weight
```

This is exploration-only, not a development platform. Good for:
- Confirming geographic coherence of community types before building full viz
- Eyeballing whether blobs respect obvious geographic features

---

## Spatial Analysis: Google Earth Engine (GEE)

**Recommended for the analytical question: do communities follow rivers,
geological features, state borders, or other physical geography?**

GEE is a planetary-scale geospatial computation platform (free for
non-commercial research). It is NOT a viz platform — it's an analysis
platform. Its catalog includes datasets directly relevant to this project:

- **USGS National Hydrography Dataset** — river networks and watershed basins
- **NLCD land cover** — urban/suburban/rural/forest/wetland classification
- **SRTM elevation** — terrain, ridgelines, coastal plains
- **EPA Ecoregions** — ecological zones that often predict human settlement patterns
- **Census TIGER tract boundaries** — can load as FeatureCollections

**Key analysis: do community boundaries follow watershed basins?**
```python
import ee
ee.Initialize()

# Load tract memberships as a FeatureCollection
tracts = ee.FeatureCollection("path/to/uploaded/tracts")

# Load watershed boundaries from GEE catalog
watersheds = ee.FeatureCollection("USGS/WBD/2017/HUC10")

# Spatial join: assign each tract to its watershed
joined = tracts.map(lambda t: t.set(
    "watershed_id", watersheds.filterBounds(t.geometry()).first().get("huc10")
))

# Compute within-watershed vs between-watershed membership variance
# Low within / high between = community follows watershed boundaries
```

**Setup:**
- Account at https://earthengine.google.com (free for research)
- Python: `pip install earthengine-api`
- Authenticate: `ee.Authenticate()`

**Gotchas:**
- GEE uploads are slow for large FeatureCollections; use Cloud Storage intermediary
- GEE's visualization (getMapId/foliumMap) is limited; use for analysis, not final viz
- Free tier has compute quotas; batch exports to Google Drive for large analyses

---

## Recommended Architecture

```
parquet memberships
        │
        ▼
   Python pipeline
   (geopandas join to TIGER tract boundaries)
        │
        ├──► GeoJSON → Deck.gl React app (interactive viz)
        │
        ├──► GEE FeatureCollection upload → spatial analysis
        │    (watershed/terrain/geology correlations)
        │
        └──► KML export → Google Earth (quick visual check)
```

---

## Gotchas

**1. TIGER tract geometries are separate from the membership parquet.**
The membership files contain tract_geoid but no geometry. Download TIGER
2022 tract shapefiles for FL+GA+AL and join on tract_geoid before any
geographic rendering. Use geopandas for the join; project to EPSG:4326
(WGS84) for web display.

**2. 9k GeoJSON polygons at full detail is ~50-100MB.**
Simplify geometry for web rendering:
```python
tracts["geometry"] = tracts["geometry"].simplify(tolerance=0.0001)
```
Tolerance 0.0001 degrees (~10m) is imperceptible at county zoom and
reduces file size 60-80%.

**3. React-Leaflet is NOT recommended at this scale.**
Standard Leaflet rendering of 9k polygons is slow (30s+ render times
reported). Use Deck.gl which does WebGL/GPU rendering.
