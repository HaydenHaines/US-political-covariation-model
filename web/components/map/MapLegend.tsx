"use client";

/**
 * MapLegend — floating legend panel in the bottom-left corner of the map.
 *
 * Renders one of three legend variants depending on map state:
 *  1. Forecast choropleth — partisan lean color scale (when forecastChoropleth is active)
 *  2. Super-type legend — visible tract super-types in the zoomed state
 *  3. Senate ratings — national state colors when no state is zoomed
 *
 * Super-type names come from the `entries` prop (derived from live tract
 * features) so they're always in sync with the loaded GeoJSON, not hardcoded.
 */

import { dustyInkChoropleth } from "@/lib/config/palette";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LegendEntry {
  id: number;
  color: [number, number, number];
  label: string;
}

interface MapLegendProps {
  /** When set, renders the partisan-lean choropleth scale. */
  forecastChoropleth: Map<string, number> | null;
  /** State abbreviation of the currently zoomed state, or null for national. */
  zoomedState: string | null;
  /** Legend entries derived from loaded tract features (API names take priority). */
  entries: LegendEntry[];
  /** Whether state ratings data has loaded (controls national legend visibility). */
  hasStateRatings: boolean;
  /**
   * Overlay mode — controls which legend is shown when zoomed into a state.
   *  - "types"    : show the super-type stained-glass legend (default)
   *  - "forecast" : show the senate ratings legend even when zoomed (tracts are neutral grey)
   */
  overlayMode?: "types" | "forecast";
}

// ---------------------------------------------------------------------------
// Shared style
// ---------------------------------------------------------------------------

const LEGEND_STYLE: React.CSSProperties = {
  position: "absolute",
  bottom: 24,
  left: 16,
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: "4px",
  padding: "8px 12px",
  fontSize: "11px",
  fontFamily: "var(--font-sans)",
  zIndex: 10,
};

const ITEM_STYLE: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "6px",
  marginBottom: "2px",
};

function Swatch({ color, style }: { color: string; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        width: 12,
        height: 12,
        borderRadius: 2,
        background: color,
        flexShrink: 0,
        ...style,
      }}
    />
  );
}

function GeoCaveat() {
  return (
    <p style={{ color: "var(--color-text-subtle)", fontSize: 10, margin: "6px 0 0", lineHeight: 1.3 }}>
      Area reflects geography, not population.
    </p>
  );
}

// ---------------------------------------------------------------------------
// Sub-legends
// ---------------------------------------------------------------------------

const FORECAST_STOPS = [
  { label: "Strong D (D+10+)", share: 0.65 },
  { label: "Lean D (D+0 to D+10)", share: 0.55 },
  { label: "Toss-up (EVEN)", share: 0.50 },
  { label: "Lean R (R+0 to R+10)", share: 0.45 },
  { label: "Strong R (R+10+)", share: 0.35 },
];

function ForecastLegend() {
  return (
    <div className="map-legend" style={LEGEND_STYLE}>
      {FORECAST_STOPS.map(({ label, share }) => {
        const [r, g, b, a] = dustyInkChoropleth(share);
        return (
          <div key={label} className="map-legend-item" style={ITEM_STYLE}>
            <Swatch color={`rgba(${r},${g},${b},${a / 255})`} />
            <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
          </div>
        );
      })}
      <GeoCaveat />
    </div>
  );
}

function SuperTypeLegend({ entries }: { entries: LegendEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <div className="map-legend" style={LEGEND_STYLE}>
      {entries.map((entry) => (
        <div key={entry.id} className="map-legend-item" style={ITEM_STYLE}>
          <Swatch color={`rgb(${entry.color.join(",")})`} />
          <span style={{ color: "var(--color-text-muted)" }}>{entry.label}</span>
        </div>
      ))}
    </div>
  );
}

const SENATE_TIERS = [
  { label: "Safe D",   color: "#2d4a6f" },
  { label: "Likely D", color: "#4b6d90" },
  { label: "Lean D",   color: "#7e9ab5" },
  { label: "Tossup",   color: "#b5a995" },
  { label: "Lean R",   color: "#c4907a" },
  { label: "Likely R", color: "#9e5e4e" },
  { label: "Safe R",   color: "#6e3535" },
  { label: "No race",  color: "#eae7e2" },
];

function SenateLegend() {
  return (
    <div className="map-legend" style={LEGEND_STYLE}>
      {SENATE_TIERS.map(({ label, color }) => (
        <div key={label} className="map-legend-item" style={ITEM_STYLE}>
          <Swatch color={color} />
          <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
        </div>
      ))}
      <GeoCaveat />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function MapLegend({
  forecastChoropleth,
  zoomedState,
  entries,
  hasStateRatings,
  overlayMode = "types",
}: MapLegendProps) {
  if (forecastChoropleth) {
    return <ForecastLegend />;
  }

  // In forecast mode, tracts render neutral grey — show the senate ratings
  // legend even when zoomed so users always see the competitive-race color key.
  if (zoomedState && overlayMode === "types") {
    return <SuperTypeLegend entries={entries} />;
  }

  if (hasStateRatings) {
    return <SenateLegend />;
  }

  return null;
}
