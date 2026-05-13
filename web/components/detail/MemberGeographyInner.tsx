/**
 * MemberGeographyInner — the deck.gl rendering core for MemberGeography.
 *
 * This file contains all deck.gl code so it can be dynamically imported with
 * ssr: false by the outer MemberGeography wrapper.
 *
 * Renders a mini choropleth map: member counties highlighted in the type's
 * super-type color, all other counties shown in muted gray.
 */

"use client";

import { useState, useEffect, useMemo } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";

/**
 * Properties attached to each county feature in counties-us.geojson.
 * Must satisfy GeoJsonProperties (Record<string, unknown>) for deck.gl.
 */
interface CountyFeatureProperties {
  county_fips?: string;
  county_name?: string;
  [key: string]: unknown;
}

/**
 * Typed GeoJSON collection for the counties file.
 * Declared as `unknown` geometry so it's assignable to deck.gl's `data` prop
 * without widening the properties type. We cast via `as unknown` when passing.
 */
interface CountyFeatureCollection {
  type: "FeatureCollection";
  features: {
    type: "Feature";
    properties: CountyFeatureProperties;
    geometry: unknown;
  }[];
}

/** Initial national view — centered on CONUS. */
const INITIAL_VIEW_STATE = {
  longitude: -98.0,
  latitude: 38.5,
  zoom: 3.2,
  pitch: 0,
  bearing: 0,
};

interface MemberGeographyInnerProps {
  typeId: number;
  superTypeId: number;
  /** FIPS codes of counties that belong to this type. */
  memberFips: ReadonlySet<string>;
}

export function MemberGeographyInner({
  typeId,
  superTypeId,
  memberFips,
}: MemberGeographyInnerProps) {
  const [geo, setGeo] = useState<CountyFeatureCollection | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    fetch("/counties-us.geojson")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<CountyFeatureCollection>;
      })
      .then(setGeo)
      .catch(() => setLoadError(true));
  }, []);

  const accentColor = useMemo(
    () => getSuperTypeColor(superTypeId),
    [superTypeId],
  );

  const layer = useMemo(() => {
    if (!geo) return null;

    return new GeoJsonLayer<CountyFeatureProperties>({
      id: `member-geography-${typeId}`,
      // Cast through unknown: our CountyFeatureCollection has `geometry: unknown`
      // which is intentionally narrower than GeoJSON's Geometry union. The
      // callbacks only access `properties`, so the geometry type mismatch is safe.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      data: geo as any,
      filled: true,
      stroked: true,
      // Highlighted members: super-type accent color, fully opaque
      // Background counties: muted warm gray, semi-transparent
      getFillColor: (f) => {
        const fips = f.properties?.county_fips ?? "";
        const isMember = memberFips.has(fips);
        if (isMember) {
          return [...accentColor, 210] as [number, number, number, number];
        }
        return [220, 217, 212, 80] as [number, number, number, number];
      },
      getLineColor: (f) => {
        const fips = f.properties?.county_fips ?? "";
        const isMember = memberFips.has(fips);
        if (isMember) {
          return [
            Math.round(accentColor[0] * 0.7),
            Math.round(accentColor[1] * 0.7),
            Math.round(accentColor[2] * 0.7),
            180,
          ] as [number, number, number, number];
        }
        return [200, 196, 190, 30] as [number, number, number, number];
      },
      lineWidthMinPixels: 0.5,
      pickable: false,
      updateTriggers: {
        getFillColor: [memberFips, accentColor],
        getLineColor: [memberFips, accentColor],
      },
    });
  }, [geo, typeId, accentColor, memberFips]);

  if (loadError) {
    return (
      <div
        style={{
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
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layer ? [layer] : []}
        style={{ background: "#e8ecf0" }}
      />
      {/* Color key */}
      <div
        style={{
          position: "absolute",
          bottom: 8,
          left: 8,
          display: "flex",
          alignItems: "center",
          gap: 6,
          background: "rgba(250,250,248,0.9)",
          border: "1px solid #e0ddd8",
          borderRadius: 4,
          padding: "4px 8px",
          fontSize: 11,
          fontFamily: "var(--font-sans)",
          color: "var(--color-text-muted)",
          pointerEvents: "none",
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 10,
            height: 10,
            borderRadius: 2,
            background: rgbToHex(accentColor),
          }}
        />
        <span>Member counties</span>
      </div>
    </div>
  );
}
