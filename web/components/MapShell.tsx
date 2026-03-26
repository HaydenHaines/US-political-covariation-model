"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import DeckGL from "@deck.gl/react";
import { WebMercatorViewport } from "@deck.gl/core";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchCounties, fetchSuperTypes, fetchTypes, type CountyRow, type TypeSummary } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";
import { TypePanel } from "@/components/TypePanel";

// ── Tooltip helpers ──────────────────────────────────────────────────────────

/** Format dem share as political lean: "D+5.2" or "R+3.1" */
function formatLean(share: number | null | undefined): string {
  if (share == null) return "";
  const pct = share * 100;
  if (pct >= 50) return `D+${(pct - 50).toFixed(1)}`;
  return `R+${(50 - pct).toFixed(1)}`;
}

/** Format income as "$XX,XXX" */
function formatIncome(income: number | null | undefined): string {
  if (income == null) return "";
  return `$${Math.round(income).toLocaleString("en-US")}`;
}

/** Convert log_pop_density to a readable category */
function densityCategory(logDensity: number | null | undefined): string {
  if (logDensity == null) return "";
  const density = Math.exp(logDensity);
  if (density >= 2000) return "Urban";
  if (density >= 500) return "Suburban";
  if (density >= 100) return "Exurban";
  return "Rural";
}

/** Format a percentage value (0–1) as "XX%" */
function formatPct(val: number | null | undefined): string {
  if (val == null) return "";
  return `${Math.round(val * 100)}%`;
}

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

export interface SuperTypeInfo {
  name: string;
  color: [number, number, number];
}

export interface TractContext {
  nTracts: number;
  areaSqkm: number;
  superTypeName: string;
}

const INITIAL_VIEW = {
  longitude: -98.0,
  latitude: 39.0,
  zoom: 4.0,
  pitch: 0,
  bearing: 0,
};

/** Partisan lean color scale: blue (D) → white (toss-up) → red (R). */
function choroplethColor(demShare: number): [number, number, number, number] {
  // Center at 0.5; map 0.3–0.7 to full color range (clip beyond that)
  const t = Math.max(0, Math.min(1, (demShare - 0.3) / 0.4));
  if (t >= 0.5) {
    // 0.5→1.0: white→blue
    const s = (t - 0.5) * 2;
    return [
      Math.round(255 * (1 - s * 0.75)),
      Math.round(255 * (1 - s * 0.44)),
      255,
      200,
    ];
  }
  // 0.0→0.5: red→white
  const s = t * 2;
  return [
    255,
    Math.round(255 * s),
    Math.round(255 * s),
    200,
  ];
}

/** Extract all [lon, lat] pairs from a GeoJSON geometry. */
function extractCoords(geometry: any): [number, number][] {
  if (!geometry) return [];
  const recurse = (arr: any): [number, number][] => {
    if (!Array.isArray(arr)) return [];
    if (typeof arr[0] === "number") return [[arr[0], arr[1]]];
    return arr.flatMap(recurse);
  };
  return recurse(geometry.coordinates);
}

export default function MapShell() {
  const {
    selectedCommunityId, setSelectedCommunityId,
    selectedTypeId, setSelectedTypeId,
    forecastState, forecastChoropleth,
  } = useMapContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewState, setViewState] = useState<any>(INITIAL_VIEW);
  const [geojson, setGeojson] = useState<any>(null);
  const [tractGeojson, setTractGeojson] = useState<any>(null);
  const [countyMap, setCountyMap] = useState<Record<string, CountyRow>>({});
  const [superTypeMap, setSuperTypeMap] = useState<Map<number, SuperTypeInfo>>(new Map());
  const [typeNameMap, setTypeNameMap] = useState<Map<number, string>>(new Map());
  const [typeDataMap, setTypeDataMap] = useState<Map<number, TypeSummary>>(new Map());
  const [hasTypeData, setHasTypeData] = useState(false);
  const [showTracts, setShowTracts] = useState(false);
  const [tractLoading, setTractLoading] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [tractContext, setTractContext] = useState<TractContext | null>(null);

  // Pan to selected forecast state whenever it changes
  useEffect(() => {
    if (!forecastState || !geojson || Object.keys(countyMap).length === 0) return;
    let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
    let found = false;
    for (const feature of geojson.features) {
      const fips = feature.properties?.county_fips;
      if (countyMap[fips]?.state_abbr !== forecastState) continue;
      found = true;
      for (const [lon, lat] of extractCoords(feature.geometry)) {
        if (lon < minLon) minLon = lon;
        if (lon > maxLon) maxLon = lon;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      }
    }
    if (!found) return;
    const container = containerRef.current;
    const width = container?.clientWidth ?? 800;
    const height = container?.clientHeight ?? 600;
    try {
      const vp = new WebMercatorViewport({ width, height });
      const { longitude, latitude, zoom } = vp.fitBounds(
        [[minLon, minLat], [maxLon, maxLat]],
        { padding: 48 }
      );
      setViewState((prev: any) => ({ ...prev, longitude, latitude, zoom, transitionDuration: 600 }));
    } catch {
      // fitBounds can throw for degenerate bounds (single-county states etc.); ignore
    }
  }, [forecastState, geojson, countyMap]);

  useEffect(() => {
    Promise.all([
      fetch("/counties-us.geojson").then((r) => r.json()),
      fetchCounties(),
      fetchSuperTypes().catch(() => []),
      fetchTypes().catch(() => []),
    ]).then(([geo, counties, superTypes, types]) => {

      // Build super-type map from API
      const stMap = new Map<number, SuperTypeInfo>();
      superTypes.forEach((st: any) => {
        stMap.set(st.super_type_id, {
          name: st.display_name,
          color: getColorForSuperType(st.super_type_id),
        });
      });
      setSuperTypeMap(stMap);

      // Build fine type name map and full type data map from API
      const tnMap = new Map<number, string>();
      const tdMap = new Map<number, TypeSummary>();
      (types as TypeSummary[]).forEach((t) => {
        tnMap.set(t.type_id, t.display_name);
        tdMap.set(t.type_id, t);
      });
      setTypeNameMap(tnMap);
      setTypeDataMap(tdMap);

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

  // Lazy-load tract GeoJSON on first toggle (15MB, too large for initial load)
  const loadTracts = useCallback(() => {
    if (tractGeojson || tractLoading) return;
    setTractLoading(true);
    fetch("/tracts-us.geojson")
      .then((r) => r.json())
      .then((tractGeo) => {
        setTractGeojson(tractGeo);
        setTractLoading(false);
        setShowTracts(true);
      })
      .catch(() => setTractLoading(false));
  }, [tractGeojson, tractLoading]);

  const getColor = useCallback(
    (f: any): [number, number, number, number] => {
      // Forecast choropleth mode: color by dem_share
      if (forecastChoropleth) {
        const fips: string = f.properties?.county_fips ?? "";
        const share = forecastChoropleth.get(fips);
        if (share !== undefined) return choroplethColor(share);
        return [200, 200, 200, 120]; // no data
      }

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
    [selectedTypeId, hasTypeData, showTracts, forecastChoropleth]
  );

  const getLineColor = useCallback(
    (f: any): [number, number, number, number] => {
      // Highlight selected forecast state with white outline
      if (forecastState) {
        const fips: string = f.properties?.county_fips ?? "";
        if (countyMap[fips]?.state_abbr === forecastState) {
          return [255, 255, 255, 220];
        }
      }
      return [80, 80, 80, 120];
    },
    [forecastState, countyMap]
  );

  const getLineWidth = useCallback(
    (f: any): number => {
      // Forecast state highlight: thicker borders
      if (forecastState) {
        const fips: string = f.properties?.county_fips ?? "";
        if (countyMap[fips]?.state_abbr === forecastState) return 600;
      }
      if (hasTypeData) {
        const dt: number = f.properties?.dominant_type ?? -1;
        return selectedTypeId !== null && dt === selectedTypeId ? 800 : 200;
      }
      const cid: number = f.properties?.community_id ?? -1;
      return selectedCommunityId !== null && cid === selectedCommunityId ? 800 : 200;
    },
    [selectedCommunityId, selectedTypeId, hasTypeData, forecastState, countyMap]
  );

  const getSuperTypeName = useCallback(
    (superTypeId: number, feature?: any): string => {
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
          getLineColor: getLineColor as any,
          getLineWidth,
          lineWidthUnits: "meters",
          updateTriggers: {
            getFillColor: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts, forecastChoropleth],
            getLineColor: [forecastState, countyMap],
            getLineWidth: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts, forecastState, countyMap],
          },
          onHover: ({ object, x, y }: any) => {
            if (object) {
              if (showTracts && tractGeojson) {
                const st = object.properties?.super_type;
                const tid = object.properties?.type_id;
                const n = object.properties?.n_tracts;
                const area = object.properties?.area_sqkm;
                const stName = getSuperTypeName(st, object);
                // Demographic properties embedded at type level
                const income = formatIncome(object.properties?.median_hh_income);
                const college = formatPct(object.properties?.pct_ba_plus);
                const white = formatPct(object.properties?.pct_white_nh);
                const black = formatPct(object.properties?.pct_black);
                const hispanic = formatPct(object.properties?.pct_hispanic);
                // Build rich tooltip
                const lines: string[] = [
                  `${stName}  ·  Type ${tid}`,
                  `${n} tracts · ${Math.round(area)} km²`,
                ];
                const stats: string[] = [];
                if (income) stats.push(`Income: ${income}`);
                if (college) stats.push(`College+: ${college}`);
                if (stats.length > 0) lines.push(stats.join("  ·  "));
                const raceStats: string[] = [];
                if (white) raceStats.push(`White NH: ${white}`);
                if (black) raceStats.push(`Black: ${black}`);
                if (hispanic) raceStats.push(`Hispanic: ${hispanic}`);
                if (raceStats.length > 0) lines.push(raceStats.join("  ·  "));
                setTooltip({ x, y, text: lines.join("\n") });
              } else {
                const fips = object.properties?.county_fips;
                const name = object.properties?.county_name || fips;
                if (hasTypeData) {
                  const dt = object.properties?.dominant_type;
                  const st = object.properties?.super_type;
                  const stName = getSuperTypeName(st, object);
                  const typeName = typeNameMap.get(dt) ?? `Type ${dt}`;
                  // Enrich with predictions and type demographics
                  const county = countyMap[fips];
                  const typeInfo = typeDataMap.get(dt);
                  const lean = formatLean(county?.pred_dem_share);
                  const income = formatIncome(typeInfo?.median_hh_income);
                  const college = formatPct(typeInfo?.pct_bachelors_plus);
                  const white = formatPct(typeInfo?.pct_white_nh);
                  const density = densityCategory(typeInfo?.log_pop_density);
                  // Build tooltip lines
                  const lines: string[] = [`${name}`, `${typeName}  ·  ${stName}`];
                  if (lean) lines.push(`Lean: ${lean}`);
                  const stats: string[] = [];
                  if (income) stats.push(`Income: ${income}`);
                  if (college) stats.push(`College+: ${college}`);
                  if (white) stats.push(`White NH: ${white}`);
                  if (density) stats.push(`Density: ${density}`);
                  if (stats.length > 0) lines.push(stats.join("  ·  "));
                  setTooltip({ x, y, text: lines.join("\n") });
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
                const dt = object.properties?.dominant_type ?? object.properties?.type_id;
                if (dt !== undefined && dt >= 0) {
                  setSelectedTypeId(dt === selectedTypeId ? null : dt);
                  setSelectedCommunityId(null);
                  // Capture tract community context when clicking in tract view
                  if (showTracts && object.properties?.n_tracts != null) {
                    setTractContext({
                      nTracts: object.properties.n_tracts,
                      areaSqkm: object.properties.area_sqkm ?? 0,
                      superTypeName: getSuperTypeName(object.properties?.super_type, object),
                    });
                  } else {
                    setTractContext(null);
                  }
                }
              } else {
                const cid = object.properties?.community_id;
                if (cid !== undefined && cid >= 0) {
                  setSelectedCommunityId(cid === selectedCommunityId ? null : cid);
                  setSelectedTypeId(null);
                  setTractContext(null);
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

  // For tract view, extract super-type names from GeoJSON features
  const tractSuperTypeNames = new Map<number, string>();
  if (showTracts && tractGeojson) {
    tractGeojson.features?.forEach((f: any) => {
      const st = f.properties?.super_type;
      const name = f.properties?.super_type_name;
      if (st != null && name && !tractSuperTypeNames.has(st)) {
        tractSuperTypeNames.set(st, name);
      }
    });
  }

  const legendEntries = Array.from(activeSuperTypeIds)
    .sort((a, b) => a - b)
    .map((id) => ({
      id,
      color: getColorForSuperType(id),
      label: showTracts
        ? (tractSuperTypeNames.get(id) ?? `Type ${id}`)
        : (superTypeMap.get(id)?.name ?? `Type ${id}`),
    }));

  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%", height: "100%" }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }: any) => setViewState(vs)}
        controller={true}
        layers={layers}
        style={{ background: "#e8ecf0" }}
      />

      {tooltip && (() => {
        const lines = tooltip.text.split("\n");
        const [headline, ...rest] = lines;
        return (
          <div style={{
            position: "absolute",
            left: tooltip.x + 14,
            top: tooltip.y + 14,
            background: "rgba(20, 24, 32, 0.93)",
            borderRadius: "6px",
            padding: "8px 12px",
            fontSize: "12px",
            fontFamily: "var(--font-sans)",
            pointerEvents: "none",
            boxShadow: "0 4px 12px rgba(0,0,0,0.35)",
            minWidth: "160px",
            maxWidth: "280px",
          }}>
            <div style={{ color: "#f0f4f8", fontWeight: 600, fontSize: "13px", marginBottom: "4px" }}>
              {headline}
            </div>
            {rest.map((line, i) => {
              if (line.startsWith("Lean:")) {
                const lean = line.replace("Lean: ", "");
                const isDem = lean.startsWith("D");
                return (
                  <div key={i} style={{ color: isDem ? "#6baed6" : "#fc8d59", fontWeight: 700, fontSize: "13px", marginBottom: "4px" }}>
                    {lean}
                  </div>
                );
              }
              return (
                <div key={i} style={{ color: "#b0bec5", fontSize: "11px", marginTop: i === 0 ? 0 : "2px" }}>
                  {line}
                </div>
              );
            })}
          </div>
        );
      })()}

      {/* County/Tract toggle — always shown, lazy-loads tract data on first click */}
      <button
        className="map-toggle-btn"
        onClick={() => {
          if (tractGeojson) {
            setShowTracts((prev) => !prev);
          } else {
            loadTracts();
          }
        }}
        disabled={tractLoading}
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
          cursor: tractLoading ? "wait" : "pointer",
          fontWeight: 600,
          boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
          opacity: tractLoading ? 0.7 : 1,
        }}
      >
        {tractLoading ? "Loading tracts…" : showTracts ? "Tract Communities" : "County Types"} ▾
      </button>

      {/* Forecast choropleth legend — replaces super-type legend when active */}
      {forecastChoropleth && (
        <div
          className="map-legend"
          style={{
            position: "absolute",
            bottom: 24,
            left: 16,
            background: "white",
            border: "1px solid var(--color-border)",
            borderRadius: "4px",
            padding: "8px 12px",
            fontSize: "11px",
            fontFamily: "var(--font-sans)",
          }}
        >
          {[
            { label: "Strong D (60%+)", color: choroplethColor(0.65) },
            { label: "Lean D (50–60%)", color: choroplethColor(0.55) },
            { label: "Toss-up (~50%)", color: choroplethColor(0.50) },
            { label: "Lean R (40–50%)", color: choroplethColor(0.45) },
            { label: "Strong R (<40%)", color: choroplethColor(0.35) },
          ].map(({ label, color }) => (
            <div key={label} className="map-legend-item" style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
              <div style={{ width: 12, height: 12, borderRadius: 2, background: `rgba(${color[0]},${color[1]},${color[2]},${color[3] / 255})` }} />
              <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Super-type legend — hidden when forecast choropleth is active */}
      {!forecastChoropleth && legendEntries.length > 0 && (
        <div
          className="map-legend"
          style={{
            position: "absolute",
            bottom: 24,
            left: 16,
            background: "white",
            border: "1px solid var(--color-border)",
            borderRadius: "4px",
            padding: "8px 12px",
            fontSize: "11px",
            fontFamily: "var(--font-sans)",
          }}
        >
          {legendEntries.map((entry) => (
            <div key={entry.id} className="map-legend-item" style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
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
          tractContext={tractContext}
          onClose={() => { setSelectedTypeId(null); setTractContext(null); }}
        />
      )}
    </div>
  );
}
