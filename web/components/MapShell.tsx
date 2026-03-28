"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import DeckGL from "@deck.gl/react";
import { FlyToInterpolator } from "@deck.gl/core";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchCounties, fetchSuperTypes, fetchTypes, type CountyRow, type TypeSummary } from "@/lib/api";
import { dustyInkChoropleth, ratingColor, type Rating } from "@/lib/colors";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";
import { TypePanel } from "@/components/TypePanel";
import { type TractFeatureProps } from "@/components/TractPanel";
import { TractPopup, type TractPopupData } from "@/components/TractPopup";
import { DashboardOverlay } from "@/components/DashboardOverlay";

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

/** Parse a hex color string (#rrggbb) to [r, g, b]. */
function hexToRgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

/** Compute bounding box [minLng, minLat, maxLng, maxLat] from a GeoJSON geometry. */
function bboxFromGeometry(geometry: any): [number, number, number, number] {
  let minLng = Infinity, maxLng = -Infinity, minLat = Infinity, maxLat = -Infinity;
  const recurse = (arr: any) => {
    if (!Array.isArray(arr)) return;
    if (typeof arr[0] === "number") {
      const [lng, lat] = arr;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      return;
    }
    arr.forEach(recurse);
  };
  recurse(geometry.coordinates);
  return [minLng, minLat, maxLng, maxLat];
}

/** Compute a reasonable zoom level from a bounding box span. */
function zoomFromBbox(minLng: number, minLat: number, maxLng: number, maxLat: number): number {
  const lngSpan = maxLng - minLng;
  const latSpan = maxLat - minLat;
  const span = Math.max(lngSpan, latSpan);
  // log2(360/span) gives roughly the zoom for that span to fill the viewport
  return Math.min(9, Math.max(4, Math.log2(360 / span) - 0.5));
}

/** Extract state abbreviation from a race string like "2026 GA Senate". */
function stateFromRace(race: string): string | null {
  const m = race.match(/\d{4}\s+([A-Z]{2})\s+/);
  return m ? m[1] : null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

const TRANSITION_MS = 800;

export default function MapShell() {
  const {
    selectedCommunityId, setSelectedCommunityId,
    selectedTypeId, setSelectedTypeId,
    forecastState, forecastChoropleth,
    zoomedState, setZoomedState,
    layoutMode,
  } = useMapContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewState, setViewState] = useState<any>(INITIAL_VIEW);

  // State-level data (loaded on mount, ~700KB)
  const [stateGeo, setStateGeo] = useState<any>(null);
  const [stateRatings, setStateRatings] = useState<Map<string, string>>(new Map());

  // Per-state tract data (loaded on state click)
  const [stateTracts, setStateTracts] = useState<any>(null);
  const [loadingTracts, setLoadingTracts] = useState(false);

  // Existing data structures retained for tract interaction
  const [superTypeMap, setSuperTypeMap] = useState<Map<number, SuperTypeInfo>>(new Map());
  const [typeDataMap, setTypeDataMap] = useState<Map<number, TypeSummary>>(new Map());
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [tractContext, setTractContext] = useState<TractContext | null>(null);
  const [tractPopup, setTractPopup] = useState<TractPopupData | null>(null);

  // ── On mount: load states + senate ratings + type metadata ──────────────
  useEffect(() => {
    // State polygons — instant, ~700KB
    fetch("/states-us.geojson").then((r) => r.json()).then(setStateGeo);

    // Senate ratings for state coloring
    fetch(`${API_BASE}/senate/overview`)
      .then((r) => r.json())
      .then((data) => {
        const ratings = new Map<string, string>();
        for (const race of data.races) {
          // API currently has a bug where race.state is always "AK";
          // extract the real state from the race string instead
          const abbr = stateFromRace(race.race) ?? race.state;
          const color = ratingColor(race.rating as Rating);
          if (color) ratings.set(abbr, color);
        }
        setStateRatings(ratings);
      })
      .catch(() => {/* senate ratings are non-critical */});

    // Super-type + type metadata (needed for tract tooltips/popups)
    Promise.all([
      fetchSuperTypes().catch(() => []),
      fetchTypes().catch(() => []),
    ]).then(([superTypes, types]) => {
      const stMap = new Map<number, SuperTypeInfo>();
      superTypes.forEach((st: any) => {
        stMap.set(st.super_type_id, {
          name: st.display_name,
          color: getColorForSuperType(st.super_type_id),
        });
      });
      setSuperTypeMap(stMap);

      const tdMap = new Map<number, TypeSummary>();
      (types as TypeSummary[]).forEach((t) => tdMap.set(t.type_id, t));
      setTypeDataMap(tdMap);
    });
  }, []);

  // ── Pan to forecast state (from Forecast tab) ──────────────────────────
  useEffect(() => {
    if (!forecastState || !stateGeo) return;
    const feature = stateGeo.features.find(
      (f: any) => f.properties?.state_abbr === forecastState
    );
    if (!feature) return;
    // When forecast tab selects a state, zoom into it and load tracts
    handleStateClick(forecastState);
  }, [forecastState, stateGeo]);

  // ── State click: load tracts + zoom ────────────────────────────────────
  const handleStateClick = useCallback((stateAbbr: string) => {
    setZoomedState(stateAbbr);
    setLoadingTracts(true);
    setTractPopup(null);

    fetch(`/tracts/${stateAbbr}.geojson`)
      .then((r) => r.json())
      .then(setStateTracts)
      .catch(() => setStateTracts(null))
      .finally(() => setLoadingTracts(false));

    // Zoom to state bounding box
    if (stateGeo) {
      const feature = stateGeo.features.find(
        (f: any) => f.properties?.state_abbr === stateAbbr
      );
      if (feature) {
        const [minLng, minLat, maxLng, maxLat] = bboxFromGeometry(feature.geometry);
        setViewState((prev: any) => ({
          ...prev,
          longitude: (minLng + maxLng) / 2,
          latitude: (minLat + maxLat) / 2,
          zoom: zoomFromBbox(minLng, minLat, maxLng, maxLat),
          transitionDuration: TRANSITION_MS,
          transitionInterpolator: new FlyToInterpolator(),
        }));
      }
    }
  }, [stateGeo, setZoomedState]);

  // ── Back to national ───────────────────────────────────────────────────
  const handleBackToNational = useCallback(() => {
    setZoomedState(null);
    setStateTracts(null);
    setTractPopup(null);
    setViewState((prev: any) => ({
      ...prev,
      ...INITIAL_VIEW,
      transitionDuration: TRANSITION_MS,
      transitionInterpolator: new FlyToInterpolator(),
    }));
  }, [setZoomedState]);

  const getSuperTypeName = useCallback(
    (superTypeId: number, feature?: any): string => {
      const geoName = feature?.properties?.super_type_name;
      if (geoName) return geoName;
      return superTypeMap.get(superTypeId)?.name ?? `Type ${superTypeId}`;
    },
    [superTypeMap]
  );

  // ── Build layers ──────────────────────────────────────────────────────
  const layers: any[] = [];

  // Layer 1: State polygons (always visible)
  if (stateGeo) {
    layers.push(
      new GeoJsonLayer({
        id: "states",
        data: stateGeo,
        filled: true,
        stroked: true,
        getFillColor: ((f: any) => {
          const abbr = f.properties?.state_abbr;
          // Desaturate non-zoomed states when zoomed in
          if (zoomedState && abbr !== zoomedState) {
            return [234, 231, 226, 60];
          }
          // Zoomed-into state: transparent fill (tracts will show through)
          if (zoomedState && abbr === zoomedState) {
            return [0, 0, 0, 0];
          }
          // National view: color by senate rating
          const color = stateRatings.get(abbr);
          if (color) {
            const [r, g, b] = hexToRgb(color);
            return [r, g, b, 180];
          }
          return [234, 231, 226, 180]; // no senate race — neutral
        }) as any,
        getLineColor: ((f: any) => {
          const abbr = f.properties?.state_abbr;
          if (zoomedState && abbr === zoomedState) {
            return [100, 95, 88, 200]; // bold border on zoomed state
          }
          if (zoomedState) {
            return [180, 175, 168, 40]; // faded border on other states
          }
          return [180, 175, 168, 120]; // normal border
        }) as any,
        lineWidthMinPixels: 1,
        pickable: !zoomedState, // only clickable in national view
        onClick: (info: any) => {
          const abbr = info.object?.properties?.state_abbr;
          if (abbr) handleStateClick(abbr);
        },
        onHover: ({ object, x, y }: any) => {
          if (!zoomedState && object) {
            const abbr = object.properties?.state_abbr;
            const name = object.properties?.state_name;
            setTooltip({ x, y, text: name ?? abbr });
          } else if (!zoomedState) {
            setTooltip(null);
          }
        },
        updateTriggers: {
          getFillColor: [zoomedState, stateRatings],
          getLineColor: [zoomedState],
        },
      })
    );
  }

  // Layer 2: Tract polygons (only when zoomed into a state)
  if (zoomedState && stateTracts) {
    layers.push(
      new GeoJsonLayer({
        id: "tracts",
        data: stateTracts,
        filled: true,
        stroked: true,
        getFillColor: ((f: any) => {
          // Forecast choropleth mode: color by dem_share
          if (forecastChoropleth) {
            const typeId = String(f.properties?.type_id ?? "");
            const share = forecastChoropleth.get(typeId);
            if (share !== undefined) return choroplethColor(share);
            return [200, 200, 200, 120];
          }
          // Default: super-type coloring
          const st = f.properties?.super_type ?? -1;
          const base = getColorForSuperType(st);
          return [...base, 180];
        }) as any,
        getLineColor: [200, 195, 188, 40],
        lineWidthMinPixels: 0.5,
        pickable: true,
        onHover: ({ object, x, y }: any) => {
          if (object) {
            const st = object.properties?.super_type;
            const tid = object.properties?.type_id;
            const n = object.properties?.n_tracts;
            const area = object.properties?.area_sqkm;
            const stName = getSuperTypeName(st, object);
            const income = formatIncome(object.properties?.median_hh_income);
            const college = formatPct(object.properties?.pct_ba_plus);
            const white = formatPct(object.properties?.pct_white_nh);
            const black = formatPct(object.properties?.pct_black);
            const hispanic = formatPct(object.properties?.pct_hispanic);
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
              const props = object.properties;
              const current = tractPopup;
              if (current && current.feature.type_id === props.type_id && current.feature.n_tracts === props.n_tracts) {
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
              setSelectedTypeId(null);
              setSelectedCommunityId(null);
              setTractContext(null);
            }
          } else {
            setTractPopup(null);
          }
        },
        updateTriggers: {
          getFillColor: [forecastChoropleth],
        },
      })
    );
  }

  // ── Build legend from tract features when zoomed ──────────────────────
  const activeSuperTypeIds = new Set<number>();
  const tractSuperTypeNames = new Map<number, string>();
  if (zoomedState && stateTracts) {
    stateTracts.features?.forEach((f: any) => {
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

      {/* Back to national button */}
      {zoomedState && (
        <button
          onClick={handleBackToNational}
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            zIndex: 10,
            padding: "6px 12px",
            borderRadius: 4,
            border: "1px solid #e0ddd8",
            background: "#fafaf8",
            color: "#3a3632",
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "var(--font-sans)",
          }}
        >
          &larr; Back to national
        </button>
      )}

      {/* Loading indicator for tract fetch */}
      {loadingTracts && (
        <div
          style={{
            position: "absolute",
            top: 12,
            left: zoomedState ? 160 : 12,
            zIndex: 10,
            padding: "4px 10px",
            borderRadius: 4,
            background: "rgba(20, 24, 32, 0.75)",
            color: "#f0f4f8",
            fontSize: 11,
            fontFamily: "var(--font-sans)",
          }}
        >
          Loading tracts...
        </div>
      )}

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

      {/* Super-type legend — shown when zoomed with tracts, no forecast choropleth */}
      {!forecastChoropleth && zoomedState && legendEntries.length > 0 && (
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

      {/* Senate rating legend — national view only */}
      {!forecastChoropleth && !zoomedState && stateRatings.size > 0 && (
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
            { label: "Safe D", color: "#2d4a6f" },
            { label: "Likely D", color: "#4b6d90" },
            { label: "Lean D", color: "#7e9ab5" },
            { label: "Tossup", color: "#b5a995" },
            { label: "Lean R", color: "#c4907a" },
            { label: "Likely R", color: "#9e5e4e" },
            { label: "Safe R", color: "#6e3535" },
            { label: "No race", color: "#eae7e2" },
          ].map(({ label, color }) => (
            <div key={label} className="map-legend-item" style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
              <div style={{ width: 12, height: 12, borderRadius: 2, background: color }} />
              <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Side panels */}
      {selectedCommunityId !== null && (
        <CommunityPanel
          communityId={selectedCommunityId}
          onClose={() => setSelectedCommunityId(null)}
        />
      )}

      {selectedTypeId !== null && (
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

      {/* Dashboard overlay panel — slides in from right when in dashboard mode */}
      {layoutMode === "dashboard" && <DashboardOverlay />}
    </div>
  );
}
