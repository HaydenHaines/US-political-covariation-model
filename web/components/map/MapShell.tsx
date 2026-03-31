"use client";

/**
 * MapShell — deck.gl map container.
 *
 * Responsibilities:
 *  - Load state GeoJSON + senate ratings on mount
 *  - Handle state click → load tracts + fly to state
 *  - Build deck.gl layers (states + tracts)
 *  - Delegate rendering to focused sub-components:
 *    MapTooltip, MapLegend, MapControls
 *  - Render side panels (CommunityPanel, TypePanel, TractPopup, DashboardOverlay)
 *
 * This component is intentionally free of formatting/color logic — those
 * live in map-utils.ts and map-palette.ts respectively.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import DeckGL from "@deck.gl/react";
import { FlyToInterpolator } from "@deck.gl/core";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchSuperTypes, fetchTypes, type TypeSummary } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";
import { TypePanel } from "@/components/TypePanel";
import { type TractFeatureProps } from "@/components/TractPanel";
import { TractPopup, type TractPopupData } from "@/components/TractPopup";
import { DashboardOverlay } from "@/components/DashboardOverlay";

import { MapTooltip } from "./MapTooltip";
import { MapLegend, type LegendEntry } from "./MapLegend";
import { MapControls } from "./MapControls";
import {
  formatIncome,
  formatPct,
  hexToRgb,
  bboxFromGeometry,
  zoomFromBbox,
  INITIAL_VIEW_STATE,
  TRANSITION_MS,
} from "./map-utils";
import { getColorForSuperType, dustyInkChoropleth } from "./map-palette";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SuperTypeInfo {
  name: string;
  color: [number, number, number];
}

export interface TractContext {
  nTracts: number;
  areaSqkm: number;
  superTypeName: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

const choroplethColor = dustyInkChoropleth;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface MapShellProps {
  /**
   * Controls the default rendering mode when no forecastChoropleth is active.
   *  - "types"    : Stained-glass super-type coloring (default, used on /explore/map)
   *  - "forecast" : State-rating coloring at national view; neutral grey tracts when zoomed.
   *               Used on /forecast/* pages where competitive ratings are the primary signal.
   */
  defaultOverlayMode?: "types" | "forecast";
}

export default function MapShell({ defaultOverlayMode = "types" }: MapShellProps) {
  const {
    selectedCommunityId, setSelectedCommunityId,
    selectedTypeId, setSelectedTypeId,
    forecastState, setForecastState, forecastChoropleth,
    zoomedState, setZoomedState,
    layoutMode,
  } = useMapContext();

  const containerRef = useRef<HTMLDivElement>(null);
  const [viewState, setViewState] = useState<Record<string, unknown>>(INITIAL_VIEW_STATE);

  // State-level data (loaded on mount)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [stateGeo, setStateGeo] = useState<Record<string, unknown> | null>(null);
  const [stateRatings, setStateRatings] = useState<Map<string, string>>(new Map());

  // Per-state tract data (loaded on state click)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [stateTracts, setStateTracts] = useState<Record<string, unknown> | null>(null);
  const [loadingTracts, setLoadingTracts] = useState(false);

  // Type metadata for tract tooltips and popups
  const [superTypeMap, setSuperTypeMap] = useState<Map<number, SuperTypeInfo>>(new Map());
  const [typeDataMap, setTypeDataMap] = useState<Map<number, TypeSummary>>(new Map());

  // Hover tooltip
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  // Tract interaction state
  const [tractContext, setTractContext] = useState<TractContext | null>(null);
  const [tractPopup, setTractPopup] = useState<TractPopupData | null>(null);

  // ── On mount: load states + senate ratings + type metadata ────────────────
  useEffect(() => {
    fetch("/states-us.geojson").then((r) => r.json()).then(setStateGeo);

    fetch(`${API_BASE}/senate/overview`)
      .then((r) => r.json())
      .then((data: { state_colors?: Record<string, string> }) => {
        const colors = new Map<string, string>();
        if (data.state_colors) {
          for (const [st, hex] of Object.entries(data.state_colors)) {
            colors.set(st, hex);
          }
        }
        setStateRatings(colors);
      })
      .catch(() => { /* senate colors are non-critical */ });

    Promise.all([
      fetchSuperTypes().catch(() => []),
      fetchTypes().catch(() => []),
    ]).then(([superTypes, types]) => {
      const stMap = new Map<number, SuperTypeInfo>();
      (superTypes as Array<{ super_type_id: number; display_name: string }>).forEach((st) => {
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

  // ── Pan to forecast state when Forecast tab selects one ──────────────────
  useEffect(() => {
    if (!forecastState || !stateGeo) return;
    handleStateClick(forecastState);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forecastState, stateGeo]);

  // ── State click: load tracts + zoom ──────────────────────────────────────
  const handleStateClick = useCallback(
    (stateAbbr: string) => {
      setZoomedState(stateAbbr);
      setLoadingTracts(true);
      setTractPopup(null);
      // On forecast pages, notify the forecast panel which state was clicked
      // so it can navigate to the corresponding race detail page.
      if (defaultOverlayMode === "forecast") {
        setForecastState(stateAbbr);
      }

      fetch(`/tracts/${stateAbbr}.geojson`)
        .then((r) => r.json())
        .then(setStateTracts)
        .catch(() => setStateTracts(null))
        .finally(() => setLoadingTracts(false));

      if (stateGeo) {
        const feature = (stateGeo as { features: Array<{ properties: Record<string, string>; geometry: { coordinates: unknown } }> })
          .features.find((f) => f.properties?.state_abbr === stateAbbr);
        if (feature) {
          const [minLng, minLat, maxLng, maxLat] = bboxFromGeometry(feature.geometry);
          setViewState((prev) => ({
            ...prev,
            longitude: (minLng + maxLng) / 2,
            latitude: (minLat + maxLat) / 2,
            zoom: zoomFromBbox(minLng, minLat, maxLng, maxLat),
            transitionDuration: TRANSITION_MS,
            transitionInterpolator: new FlyToInterpolator(),
          }));
        }
      }
    },
    [stateGeo, setZoomedState, setForecastState, defaultOverlayMode]
  );

  // ── Back to national ──────────────────────────────────────────────────────
  const handleBackToNational = useCallback(() => {
    setZoomedState(null);
    setStateTracts(null);
    setTractPopup(null);
    setViewState((prev) => ({
      ...prev,
      ...INITIAL_VIEW_STATE,
      transitionDuration: TRANSITION_MS,
      transitionInterpolator: new FlyToInterpolator(),
    }));
  }, [setZoomedState]);

  // ── Zoom controls ─────────────────────────────────────────────────────────
  const handleZoomIn = useCallback(() => {
    setViewState((prev) => ({
      ...prev,
      zoom: Math.min(14, ((prev.zoom as number) ?? 4) + 1),
      transitionDuration: 200,
    }));
  }, []);

  const handleZoomOut = useCallback(() => {
    setViewState((prev) => ({
      ...prev,
      zoom: Math.max(2, ((prev.zoom as number) ?? 4) - 1),
      transitionDuration: 200,
    }));
  }, []);

  const handleResetView = useCallback(() => {
    setViewState((prev) => ({
      ...prev,
      ...INITIAL_VIEW_STATE,
      transitionDuration: TRANSITION_MS,
      transitionInterpolator: new FlyToInterpolator(),
    }));
  }, []);

  // ── Super-type name resolution ────────────────────────────────────────────
  const getSuperTypeName = useCallback(
    (superTypeId: number, feature?: { properties?: Record<string, unknown> }): string => {
      const geoName = feature?.properties?.super_type_name;
      if (typeof geoName === "string") return geoName;
      return superTypeMap.get(superTypeId)?.name ?? `Type ${superTypeId}`;
    },
    [superTypeMap]
  );

  // ── Build deck.gl layers ──────────────────────────────────────────────────
  const layers: unknown[] = [];

  // Layer 1: State polygons (always visible)
  if (stateGeo) {
    layers.push(
      new GeoJsonLayer({
        id: "states",
        data: stateGeo as never,
        filled: true,
        stroked: true,
        getFillColor: ((f: { properties?: Record<string, string> }) => {
          const abbr = f.properties?.state_abbr;
          if (zoomedState && abbr !== zoomedState) return [234, 231, 226, 60];
          if (zoomedState && abbr === zoomedState) return [0, 0, 0, 0];
          const color = abbr ? stateRatings.get(abbr) : undefined;
          if (color) {
            const [r, g, b] = hexToRgb(color);
            return [r, g, b, 180];
          }
          return [234, 231, 226, 180];
        }) as never,
        getLineColor: ((f: { properties?: Record<string, string> }) => {
          const abbr = f.properties?.state_abbr;
          if (zoomedState && abbr === zoomedState) return [100, 95, 88, 200];
          if (zoomedState) return [180, 175, 168, 40];
          return [180, 175, 168, 120];
        }) as never,
        lineWidthMinPixels: 1,
        pickable: !zoomedState,
        onClick: (info: { object?: { properties?: Record<string, string> } }) => {
          const abbr = info.object?.properties?.state_abbr;
          if (abbr) handleStateClick(abbr);
        },
        onHover: ({ object, x, y }: { object?: { properties?: Record<string, string> }; x: number; y: number }) => {
          if (!zoomedState && object) {
            const abbr = object.properties?.state_abbr;
            const name = object.properties?.state_name;
            setTooltip({ x, y, text: name ?? abbr ?? "" });
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
        data: stateTracts as never,
        filled: true,
        stroked: true,
        getFillColor: ((f: { properties?: Record<string, unknown> }) => {
          if (forecastChoropleth) {
            const typeId = String(f.properties?.type_id ?? "");
            const share = forecastChoropleth.get(typeId);
            if (share !== undefined) return choroplethColor(share);
            return [200, 200, 200, 120];
          }
          // On forecast pages, color tracts by their type's partisan lean
          // using the type metadata already loaded from the API.
          if (defaultOverlayMode === "forecast") {
            const typeId = f.properties?.type_id as number | undefined;
            if (typeId != null) {
              const typeData = typeDataMap.get(typeId);
              if (typeData?.mean_pred_dem_share != null) {
                return choroplethColor(typeData.mean_pred_dem_share);
              }
            }
            return [200, 195, 188, 140];
          }
          const st = (f.properties?.super_type as number) ?? -1;
          const base = getColorForSuperType(st);
          return [...base, 180];
        }) as never,
        getLineColor: [200, 195, 188, 40],
        lineWidthMinPixels: 0.5,
        pickable: true,
        onHover: ({ object, x, y }: { object?: { properties?: Record<string, unknown> }; x: number; y: number }) => {
          if (object) {
            const props = object.properties ?? {};
            const st = props.super_type as number;
            const tid = props.type_id;
            const n = props.n_tracts;
            const area = props.area_sqkm as number;
            const stName = getSuperTypeName(st, object);
            const income = formatIncome(props.median_hh_income as number | null);
            const college = formatPct(props.pct_ba_plus as number | null);
            const white = formatPct(props.pct_white_nh as number | null);
            const black = formatPct(props.pct_black as number | null);
            const hispanic = formatPct(props.pct_hispanic as number | null);

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
        onClick: ({ object, x, y }: { object?: { properties?: Record<string, unknown> }; x: number; y: number }) => {
          if (object) {
            const props = object.properties ?? {};
            if (props.n_tracts != null) {
              const current = tractPopup;
              if (
                current &&
                current.feature.type_id === props.type_id &&
                current.feature.n_tracts === props.n_tracts
              ) {
                setTractPopup(null);
              } else {
                // TractFeatureProps uses optional (undefined) fields, not null
                const toOptNum = (v: unknown): number | undefined =>
                  typeof v === "number" ? v : undefined;
                setTractPopup({
                  feature: {
                    type_id: props.type_id as number,
                    super_type: props.super_type as number,
                    super_type_name: getSuperTypeName(props.super_type as number, object),
                    n_tracts: props.n_tracts as number,
                    area_sqkm: (props.area_sqkm as number) ?? 0,
                    median_hh_income: toOptNum(props.median_hh_income),
                    pct_ba_plus: toOptNum(props.pct_ba_plus),
                    pct_white_nh: toOptNum(props.pct_white_nh),
                    pct_black: toOptNum(props.pct_black),
                    pct_hispanic: toOptNum(props.pct_hispanic),
                    evangelical_share: toOptNum(props.evangelical_share),
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
          getFillColor: [forecastChoropleth, defaultOverlayMode, typeDataMap],
        },
      })
    );
  }

  // ── Build legend entries from loaded tract features ───────────────────────
  const activeSuperTypeIds = new Set<number>();
  const tractSuperTypeNames = new Map<number, string>();

  if (zoomedState && stateTracts) {
    (stateTracts as { features: Array<{ properties?: Record<string, unknown> }> })
      .features?.forEach((f) => {
        const st = f.properties?.super_type as number;
        const name = f.properties?.super_type_name as string | undefined;
        if (st != null && st >= 0) {
          activeSuperTypeIds.add(st);
          if (name && !tractSuperTypeNames.has(st)) tractSuperTypeNames.set(st, name);
        }
      });
  }

  // API display_name is the canonical source of truth for super-type labels.
  // GeoJSON-embedded names are only a fallback for cases where the API hasn't loaded yet.
  const legendEntries: LegendEntry[] = Array.from(activeSuperTypeIds)
    .sort((a, b) => a - b)
    .map((id) => ({
      id,
      color: getColorForSuperType(id),
      label: superTypeMap.get(id)?.name ?? tractSuperTypeNames.get(id) ?? `Type ${id}`,
    }));

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%", height: "100%" }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }: { viewState: Record<string, unknown> }) => setViewState(vs)}
        controller={true}
        layers={layers as never}
        style={{ background: "#e8ecf0" }}
      />

      <MapControls
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onResetView={handleResetView}
        zoomedState={zoomedState}
        onBackToNational={handleBackToNational}
        loadingTracts={loadingTracts}
      />

      {tooltip && <MapTooltip x={tooltip.x} y={tooltip.y} text={tooltip.text} />}

      <MapLegend
        forecastChoropleth={forecastChoropleth}
        zoomedState={zoomedState}
        entries={legendEntries}
        hasStateRatings={stateRatings.size > 0}
        overlayMode={defaultOverlayMode}
      />

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

      {layoutMode === "dashboard" && <DashboardOverlay />}
    </div>
  );
}
