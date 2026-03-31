"use client";

/**
 * MapPageClient — client component that owns the full-screen map page state.
 *
 * Provides MapContext, loads the deck.gl MapShell dynamically (ssr:false),
 * and renders overlay controls on top of the map.
 */

import dynamic from "next/dynamic";
import { useState } from "react";
import { MapProvider, useMapContext } from "@/components/MapContext";
import { MapOverlayToggle, type MapOverlay } from "@/components/explore/MapOverlayToggle";
import { ThemeToggle } from "@/components/ThemeToggle";

// deck.gl must not be SSR'd
const MapShell = dynamic(() => import("@/components/map/MapShell"), { ssr: false });

// ---------------------------------------------------------------------------
// Inner component — consumes MapContext
// ---------------------------------------------------------------------------

function MapPageInner() {
  const { setForecastChoropleth, overlayMode, setOverlayMode } = useMapContext();
  const [overlay, setOverlay] = useState<MapOverlay>(overlayMode);

  function handleOverlayChange(next: MapOverlay) {
    setOverlay(next);
    setOverlayMode(next);
    if (next === "types") {
      // Clear choropleth — MapShell will revert to super-type coloring
      setForecastChoropleth(null);
    }
    // "forecast" mode: choropleth data is loaded inside MapShell/ForecastView
    // when the user selects a race. Switching to "forecast" mode here
    // does not immediately push choropleth data — the legend will just show
    // the scale until a state/race is active.
  }

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        // Fill viewport below the nav bar (nav is h-12 = 3rem = 48px)
        height: "calc(100vh - 3rem)",
        overflow: "hidden",
      }}
    >
      {/* Full-viewport map */}
      <MapShell />

      {/* Top-right controls */}
      <div
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          zIndex: 20,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <MapOverlayToggle value={overlay} onChange={handleOverlayChange} />
        <ThemeToggle />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exported wrapper — provides MapContext
// ---------------------------------------------------------------------------

export function MapPageClient() {
  return (
    <MapProvider>
      <MapPageInner />
    </MapProvider>
  );
}
