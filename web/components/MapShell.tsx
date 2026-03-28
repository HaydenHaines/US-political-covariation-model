"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import DeckGL from "@deck.gl/react";
import { WebMercatorViewport } from "@deck.gl/core";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchCounties, fetchSuperTypes, fetchTypes, type CountyRow, type TypeSummary } from "@/lib/api";
import { dustyInkChoropleth } from "@/lib/colors";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";
import { TypePanel } from "@/components/TypePanel";
import { type TractFeatureProps } from "@/components/TractPanel";
import { TractPopup, type TractPopupData } from "@/components/TractPopup";

// ── Tooltip helpers ──────────────────────────────────────────────────────────

/** Format income as "$XX,XXX" */
function formatIncome(income: number | null | undefined): string {
  if (income == null) return "";
  return `$${Math.round(income).toLocaleString("en-US")}`;
}

/** Format a percentage value (0–1) as "XX%" */
function formatPct(val: number | null | undefined): string {
  if (val == null) return "";
  return `${Math.round(val * 100)}%`;
}

// Semantically assigned palette for the 8 tract super-types.
// Colors chosen to be perceptually distinct, non-partisan, and readable on both light and dark backgrounds.
// Indices correspond to tract super_type_id values 0–7; additional slots retained for safety.
export const PALETTE: [number, number, number][] = [
  [220, 120,  55],  // 0: Hispanic Working Community       — warm amber-orange
  [115,  45, 140],  // 1: Black Urban Neighborhood          — deep violet-purple
  [220, 110, 110],  // 2: White Retirement Town             — muted rose-salmon
  [170,  35,  50],  // 3: Rural Evangelical Heartland       — deep crimson (distinct from partisan red)
  [ 38, 145, 145],  // 4: Multiracial Outer Suburb          — teal-cyan
  [195, 155,  25],  // 5: Asian-American Professional       — deep gold-amber
  [ 65, 140, 210],  // 6: Affluent White Suburb             — sky blue (lighter than slate)
  [ 40, 140,  85],  // 7: Urban Knowledge District          — emerald green
  // Overflow slots (defensive — model currently uses 0–7 only)
  [140,  86,  75],  // 8
  [227, 119, 194],  // 9
  [188, 189,  34],  // 10
  [ 23, 190, 207],  // 11
  [174, 199, 232],  // 12
  [255, 187, 120],  // 13
  [152, 223, 138],  // 14
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

/** Partisan lean color scale: Dusty Ink muted academic palette. */
const choroplethColor = dustyInkChoropleth;

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
  const [typeDataMap, setTypeDataMap] = useState<Map<number, TypeSummary>>(new Map());
  const [hasTypeData, setHasTypeData] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [tractContext, setTractContext] = useState<TractContext | null>(null);
  const [tractPopup, setTractPopup] = useState<TractPopupData | null>(null);

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

      // Build full type data map from API (used by TractPopup for n_counties, mean_pred_dem_share)
      const tdMap = new Map<number, TypeSummary>();
      (types as TypeSummary[]).forEach((t) => {
        tdMap.set(t.type_id, t);
      });
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

  // Eagerly load tract GeoJSON on mount — tract view is always active
  useEffect(() => {
    fetch("/tracts-us.geojson")
      .then((r) => r.json())
      .then(setTractGeojson)
      .catch(() => {/* tract layer stays null; map shows empty */});
  }, []);

  const getColor = useCallback(
    (f: any): [number, number, number, number] => {
      // Forecast choropleth mode: color by dem_share
      if (forecastChoropleth) {
        // Community polygons are keyed by type_id, not county_fips
        const typeId: string = String(f.properties?.type_id ?? f.properties?.dominant_type ?? "");
        const share = forecastChoropleth.get(typeId);
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
    [selectedTypeId, hasTypeData, forecastChoropleth]
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
      // Tract view: read from GeoJSON property, fall back to API map, then generic
      const geoName = feature?.properties?.super_type_name;
      if (geoName) return geoName;
      return superTypeMap.get(superTypeId)?.name ?? `Type ${superTypeId}`;
    },
    [superTypeMap]
  );

  // Tract community polygons are the sole map view
  const activeData = tractGeojson;
  const layerId = "tract-communities";

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
            getFillColor: [selectedCommunityId, selectedTypeId, hasTypeData, forecastChoropleth],
            getLineColor: [forecastState, countyMap],
            getLineWidth: [selectedCommunityId, selectedTypeId, hasTypeData, forecastState, countyMap],
          },
          onHover: ({ object, x, y }: any) => {
            if (object) {
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
              setTooltip(null);
            }
          },
          onClick: ({ object, x, y }: any) => {
            if (object) {
              if (object.properties?.n_tracts != null) {
                // Tract view: show popup at click location
                const props = object.properties;
                const current = tractPopup;
                if (current && current.feature.type_id === props.type_id && current.feature.n_tracts === props.n_tracts) {
                  // Click same tract polygon again — dismiss popup
                  setTractPopup(null);
                } else {
                  setTractPopup({
                    feature: {
                      type_id: props.type_id,
                      super_type: props.super_type,
                      super_type_name: getSuperTypeName(props.super_type, object),
                      n_tracts: props.n_tracts,
                      area_sqkm: props.area_sqkm ?? 0,
                      median_hh_income: props.median_hh_income,
                      pct_ba_plus: props.pct_ba_plus,
                      pct_white_nh: props.pct_white_nh,
                      pct_black: props.pct_black,
                      pct_hispanic: props.pct_hispanic,
                      evangelical_share: props.evangelical_share,
                    },
                    x,
                    y,
                  });
                }
                // Clear type/community selections when clicking a tract polygon
                setSelectedTypeId(null);
                setSelectedCommunityId(null);
                setTractContext(null);
              }
            } else {
              // Click on empty map space — dismiss tract popup
              setTractPopup(null);
            }
          },
        }),
      ]
    : [];

  // Build legend: collect super-type IDs from tract GeoJSON features and extract names
  const activeSuperTypeIds = new Set<number>();
  const tractSuperTypeNames = new Map<number, string>();
  if (tractGeojson) {
    tractGeojson.features?.forEach((f: any) => {
      const st = f.properties?.super_type;
      const name = f.properties?.super_type_name;
      if (st != null && st >= 0) {
        activeSuperTypeIds.add(st);
        if (name && !tractSuperTypeNames.has(st)) tractSuperTypeNames.set(st, name);
      }
    });
  }

  const legendEntries = Array.from(activeSuperTypeIds)
    .sort((a, b) => a - b)
    .map((id) => ({
      id,
      color: getColorForSuperType(id),
      label: tractSuperTypeNames.get(id) ?? superTypeMap.get(id)?.name ?? `Type ${id}`,
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

      {/* Forecast choropleth legend — replaces super-type legend when active */}
      {forecastChoropleth && (
        <div
          className="map-legend"
          style={{
            position: "absolute",
            bottom: 24,
            left: 16,
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "4px",
            padding: "8px 12px",
            fontSize: "11px",
            fontFamily: "var(--font-sans)",
          }}
        >
          {[
            { label: "Strong D (D+10+)", color: choroplethColor(0.65) },
            { label: "Lean D (D+0 to D+10)", color: choroplethColor(0.55) },
            { label: "Toss-up (EVEN)", color: choroplethColor(0.50) },
            { label: "Lean R (R+0 to R+10)", color: choroplethColor(0.45) },
            { label: "Strong R (R+10+)", color: choroplethColor(0.35) },
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
            background: "var(--color-surface)",
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

      {tractPopup !== null && (
        <TractPopup
          data={tractPopup}
          nCounties={typeDataMap.get(tractPopup.feature.type_id)?.n_counties ?? null}
          meanDemShare={typeDataMap.get(tractPopup.feature.type_id)?.mean_pred_dem_share ?? null}
          onClose={() => setTractPopup(null)}
        />
      )}
    </div>
  );
}
