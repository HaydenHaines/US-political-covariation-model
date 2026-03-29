/**
 * MiniMap — Senate forecast state choropleth for the landing page.
 *
 * Wraps MiniMapInner with next/dynamic (ssr: false) so deck.gl is never
 * evaluated during SSR.
 *
 * Usage:
 *   <MiniMap stateColors={data.state_colors} />
 */

"use client";

import dynamic from "next/dynamic";

/** Inner map component — loaded client-side only (deck.gl requires window). */
const MiniMapInner = dynamic(
  () =>
    import("./MiniMapInner").then((m) => ({
      default: m.MiniMapInner,
    })),
  {
    ssr: false,
    loading: () => (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 13,
          color: "var(--color-text-muted)",
          background: "#e8ecf0",
        }}
      >
        Loading map…
      </div>
    ),
  },
);

interface MiniMapProps {
  /** Map from state abbreviation (e.g. "TX") to hex color from senate overview. */
  stateColors: Record<string, string>;
}

/**
 * Aspect ratio: US map is roughly 1.6:1 (width:height) for CONUS.
 * We constrain to max 480px wide; height is derived from aspect ratio.
 */
const MAP_WIDTH = 480;
const MAP_HEIGHT = Math.round(MAP_WIDTH / 1.6);

export function MiniMap({ stateColors }: MiniMapProps) {
  return (
    <div
      style={{
        width: MAP_WIDTH,
        height: MAP_HEIGHT,
        maxWidth: "100%",
        borderRadius: 8,
        overflow: "hidden",
        border: "1px solid var(--color-border)",
        background: "#e8ecf0",
        position: "relative",
      }}
    >
      <MiniMapInner stateColors={stateColors} />
    </div>
  );
}
