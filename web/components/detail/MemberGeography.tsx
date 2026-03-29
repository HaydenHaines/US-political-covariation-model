/**
 * MemberGeography — mini map showing member counties for a type.
 *
 * Wraps MemberGeographyInner with next/dynamic (ssr: false) so deck.gl,
 * which requires browser globals, is never evaluated during SSR.
 *
 * Usage:
 *   <MemberGeography typeId={42} superTypeId={3} counties={data.counties} />
 */

"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { TypeCounty } from "@/lib/types";

/** Inner map component — loaded client-side only (deck.gl requires window). */
const MemberGeographyInner = dynamic(
  () =>
    import("./MemberGeographyInner").then((m) => ({
      default: m.MemberGeographyInner,
    })),
  {
    ssr: false,
    loading: () => (
      <div
        className="animate-pulse"
        style={{
          height: "100%",
          width: "100%",
          borderRadius: 6,
          background: "var(--color-border)",
        }}
      />
    ),
  },
);

/** Height of the map container in pixels. */
const MAP_HEIGHT = 280;

interface MemberGeographyProps {
  typeId: number;
  superTypeId: number;
  counties: TypeCounty[];
}

export function MemberGeography({
  typeId,
  superTypeId,
  counties,
}: MemberGeographyProps) {
  // Build the member FIPS set; memoised on typeId since counties is stable per page load.
  const memberFips = useMemo(
    () => new Set(counties.map((c) => c.county_fips)) as ReadonlySet<string>,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [typeId],
  );

  return (
    <div
      style={{
        height: MAP_HEIGHT,
        borderRadius: 6,
        overflow: "hidden",
        border: "1px solid var(--color-border)",
        background: "#e8ecf0",
        position: "relative",
      }}
    >
      <MemberGeographyInner
        typeId={typeId}
        superTypeId={superTypeId}
        memberFips={memberFips}
      />
    </div>
  );
}
