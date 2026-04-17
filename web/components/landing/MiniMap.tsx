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

export function MiniMap({ stateColors }: MiniMapProps) {
  return (
    <div
      className="w-full mx-auto"
      style={{ aspectRatio: "1.6/1", position: "relative" }}
    >
      <MiniMapInner stateColors={stateColors} />
    </div>
  );
}
