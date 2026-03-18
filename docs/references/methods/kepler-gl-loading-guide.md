# Kepler.gl Loading Guide — Tract Community Memberships

Reference for loading `data/viz/tract_memberships_k7.geojson` into Kepler.gl
for interactive exploration of NMF community type weights across FL + GA + AL
census tracts.

---

## 1. Open Kepler.gl

Navigate to **https://kepler.gl/demo** in a modern browser (Chrome or Firefox recommended).

All data processing happens client-side in your browser. Nothing is uploaded to any server.

---

## 2. Load the GeoJSON file

1. On the landing screen, click **"Load Map"** or drag a file onto the canvas.
2. Drag `data/viz/tract_memberships_k7.geojson` from your file manager directly onto the Kepler.gl window.
3. Kepler.gl will parse the file and auto-create an initial layer. With ~9,393 features at 25 MB, this takes a few seconds.

---

## 3. Set up a fill-color layer for a single community (e.g., c7)

1. In the left panel, click the layer named something like **"tract_memberships_k7"**.
2. Under **Fill Color**, click the color swatch and switch from a fixed color to **"Color Based On"**.
3. Select the column: **`c7`** (or whichever community you want to visualize).
4. Set the color scale type to **"Quantize"** (divides the [0,1] range into discrete bins — useful for choropleth maps). Alternatively, use **"Sequential"** for a smooth continuous gradient.
5. Choose a color palette:
   - Single-community maps: **"Yellow-Orange-Red"** or **"Blues"** work well.
   - For the dominant-community layer: **"Paired"** or a custom 7-color palette.
6. Under **Stroke**, turn off the outline (set to 0 width) to reduce visual noise at this scale, or set it to a very thin neutral gray.

Repeat this for each community column (`c1` through `c7`) to create 7 single-community layers.

---

## 4. Add a tooltip showing all 7 weights + entropy

1. In the layer panel, click **"Interactions"** (the cursor icon at the top of the left panel).
2. Enable **"Tooltip"**.
3. Under **"Select fields to show in tooltip"**, add:
   - `tract_geoid`
   - `dominant_community`
   - `c1` through `c7` (all seven weights)
   - `membership_entropy`
   - `is_uninhabited`
4. Hover over any tract on the map to see the full community composition and entropy score.

---

## 5. Duplicate the layer for all 7 communities

To build a 7-layer setup (one per community, togglable):

1. In the layer panel, click the **three-dot menu** (`...`) on the layer you already configured.
2. Select **"Duplicate Layer"**.
3. On the duplicate, change **Fill Color** → **"Color Based On"** to the next community column (`c2`, then `c3`, etc.).
4. Rename each layer for clarity by clicking the layer name:
   - `c1 — White rural homeowner`
   - `c2 — Black urban`
   - `c3 — Knowledge worker`
   - `c4 — Asian`
   - `c5 — Working-class homeowner`
   - `c6 — Hispanic low-income`
   - `c7 — Generic suburban baseline`
5. Use the eye icon on each layer to toggle visibility. Show one layer at a time for clear choropleth exploration.

---

## 6. Visualize dominant community (categorical)

1. Create a new layer (or repurpose one).
2. Under **Fill Color** → **"Color Based On"**, select **`dominant_community`**.
3. Kepler.gl will auto-assign a distinct color per category label.
4. You can manually reassign colors to be semantically meaningful (e.g., blue for `c2: Black urban`, yellow for `c3: Knowledge worker`).

---

## 7. Visualize membership entropy

Entropy ranges from 0 (tract dominated by a single community) to 1 (weight spread evenly across all 7).

1. In a layer, set **Fill Color** → **"Color Based On"** → **`membership_entropy`**.
2. Use a **diverging** color scale (e.g., "Yellow-Green-Blue") with low entropy = yellow, high entropy = blue.
3. High-entropy tracts (blue) are transitional or mixed communities — often politically competitive.

---

## 8. Export a saved state

After configuring your layers, click the **floppy disk icon** (top right) → **"Export Map"** → **"Export as JSON"** to save your Kepler.gl configuration. This saves layer settings and color scales but not the data itself. You can reload it later by dragging the config JSON onto kepler.gl and supplying the GeoJSON again.

---

## Field reference

| Field | Type | Description |
|---|---|---|
| `tract_geoid` | string | 11-digit census tract FIPS |
| `state_fips` | string | 2-digit state FIPS (`01`=AL, `12`=FL, `13`=GA) |
| `is_uninhabited` | bool | True for 98 uninhabited tracts (all c-weights = 0) |
| `c1` | float [0,1] | White rural homeowner (older+WFH) |
| `c2` | float [0,1] | Black urban (transit+income) |
| `c3` | float [0,1] | Knowledge worker (mgmt+WFH+college) |
| `c4` | float [0,1] | Asian |
| `c5` | float [0,1] | Working-class homeowner (owner-occ) |
| `c6` | float [0,1] | Hispanic low-income |
| `c7` | float [0,1] | Generic suburban baseline |
| `dominant_community` | string | Label of highest-weight community |
| `membership_entropy` | float [0,1] | Normalized Shannon entropy of membership weights |
