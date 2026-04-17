/**
 * MiniMapInner — deck.gl state choropleth for the landing page.
 *
 * Renders all 51 US states colored by their Senate forecast rating.
 * Non-interactive except for the outer click handler that navigates to /forecast.
 *
 * Loaded via next/dynamic with ssr: false in MiniMap.tsx.
 */

"use client";

import { useState, useEffect, useMemo } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";
import { useRouter } from "next/navigation";

/** Properties attached to each state feature in states-us.geojson. */
interface StateFeatureProperties {
  state_abbr?: string;
  state_name?: string;
  [key: string]: unknown;
}

interface StateFeatureCollection {
  type: "FeatureCollection";
  features: {
    type: "Feature";
    properties: StateFeatureProperties;
    geometry: unknown;
  }[];
}

/**
 * CONUS-centered initial view.
 *
 * Latitude shifted north to 39.5° (from 38.5°) so the northeast corner
 * (Maine ~47°N, Vermont/New Hampshire ~43-44°N) is not clipped on the
 * 480×300 viewport. Zoom reduced slightly to 2.9 to keep the southern
 * states (Florida, Texas) fully visible at the same time.
 *
 * Issue #99: ME/NH/VT were cropped at the original 38.5°/zoom 3.0 settings.
 */
const INITIAL_VIEW_STATE = {
  longitude: -98.0,
  latitude: 39.5,
  zoom: 3.4,
  pitch: 0,
  bearing: 0,
};

/**
 * Convert a CSS hex color like "#2d4a6f" to an RGBA array for deck.gl.
 * Returns [0, 0, 0, 180] for unrecognized values.
 */
function hexToRgba(hex: string, alpha = 200): [number, number, number, number] {
  const clean = hex.replace("#", "");
  if (clean.length !== 6) return [0, 0, 0, alpha];
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return [r, g, b, alpha];
}

/**
 * Slightly darken an RGBA array for border/stroke.
 */
function darken(rgba: [number, number, number, number], factor = 0.65): [number, number, number, number] {
  return [
    Math.round(rgba[0] * factor),
    Math.round(rgba[1] * factor),
    Math.round(rgba[2] * factor),
    180,
  ];
}

interface MiniMapInnerProps {
  stateColors: Record<string, string>;
}

export function MiniMapInner({ stateColors }: MiniMapInnerProps) {
  const router = useRouter();
  const [geo, setGeo] = useState<StateFeatureCollection | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    fetch("/states-us.geojson")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<StateFeatureCollection>;
      })
      .then(setGeo)
      .catch(() => setLoadError(true));
  }, []);

  // Memoize the color lookup so stateColors reference changes don't thrash the layer
  const colorsByAbbr = useMemo(() => stateColors, [stateColors]);

  const layer = useMemo(() => {
    if (!geo) return null;

    return new GeoJsonLayer<StateFeatureProperties>({
      id: "mini-map-states",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      data: geo as any,
      filled: true,
      stroked: true,
      getFillColor: (f) => {
        const abbr = f.properties?.state_abbr ?? "";
        const hex = colorsByAbbr[abbr] ?? "#eae7e2";
        return hexToRgba(hex, 200);
      },
      getLineColor: (f) => {
        const abbr = f.properties?.state_abbr ?? "";
        const hex = colorsByAbbr[abbr] ?? "#eae7e2";
        return darken(hexToRgba(hex, 200));
      },
      lineWidthMinPixels: 0.8,
      pickable: false,
      updateTriggers: {
        getFillColor: [colorsByAbbr],
        getLineColor: [colorsByAbbr],
      },
    });
  }, [geo, colorsByAbbr]);

  if (loadError) {
    return (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 13,
          color: "var(--color-text-muted)",
        }}
      >
        Map unavailable.
      </div>
    );
  }

  if (!geo) {
    return (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 13,
          color: "var(--color-text-muted)",
        }}
      >
        Loading map…
      </div>
    );
  }

  return (
    <div
      style={{ position: "relative", width: "100%", height: "100%", cursor: "pointer" }}
      onClick={() => router.push("/forecast")}
      title="View Senate Forecast"
      role="button"
      aria-label="View full Senate forecast map"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") router.push("/forecast");
      }}
    >
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={false}
        layers={layer ? [layer] : []}
        style={{ background: "#e8ecf0", pointerEvents: "none" }}
      />
    </div>
  );
}
