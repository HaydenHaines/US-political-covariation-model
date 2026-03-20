"use client";
import { useState, useEffect, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchCounties, type CountyRow } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";

const COMMUNITY_COLORS: [number, number, number][] = [
  [78, 121, 167],
  [89, 161, 79],
  [176, 122, 161],
  [255, 157, 167],
  [156, 117, 95],
  [242, 142, 43],
  [186, 176, 172],
  [255, 210, 0],
  [148, 103, 189],
  [140, 162, 82],
];

const INITIAL_VIEW = {
  longitude: -84.5,
  latitude: 31.5,
  zoom: 5.8,
  pitch: 0,
  bearing: 0,
};

export default function MapShell() {
  const { selectedCommunityId, setSelectedCommunityId } = useMapContext();
  const [geojson, setGeojson] = useState<any>(null);
  const [countyMap, setCountyMap] = useState<Record<string, CountyRow>>({});
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/counties-fl-ga-al.geojson").then((r) => r.json()),
      fetchCounties(),
    ]).then(([geo, counties]) => {
      const map: Record<string, CountyRow> = {};
      counties.forEach((c) => (map[c.county_fips] = c));
      setCountyMap(map);
      const enriched = {
        ...geo,
        features: geo.features.map((f: any) => ({
          ...f,
          properties: {
            ...f.properties,
            community_id: map[f.properties.county_fips]?.community_id ?? -1,
          },
        })),
      };
      setGeojson(enriched);
    });
  }, []);

  const getColor = useCallback(
    (f: any): [number, number, number, number] => {
      const cid: number = f.properties?.community_id ?? -1;
      const isSelected = selectedCommunityId !== null && cid === selectedCommunityId;
      const base = cid >= 0 && cid < COMMUNITY_COLORS.length ? COMMUNITY_COLORS[cid] : [180, 180, 180];
      return isSelected ? [...base, 255] as [number, number, number, number] : [...base, 180] as [number, number, number, number];
    },
    [selectedCommunityId]
  );

  const getLineWidth = useCallback(
    (f: any): number => {
      const cid: number = f.properties?.community_id ?? -1;
      return selectedCommunityId !== null && cid === selectedCommunityId ? 800 : 200;
    },
    [selectedCommunityId]
  );

  const layers = geojson
    ? [
        new GeoJsonLayer({
          id: "counties",
          data: geojson,
          pickable: true,
          stroked: true,
          filled: true,
          getFillColor: getColor as any,
          getLineColor: [80, 80, 80, 120],
          getLineWidth,
          lineWidthUnits: "meters",
          updateTriggers: {
            getFillColor: [selectedCommunityId],
            getLineWidth: [selectedCommunityId],
          },
          onHover: ({ object, x, y }: any) => {
            if (object) {
              const name = object.properties?.county_name || object.properties?.county_fips;
              const cid = object.properties?.community_id;
              setTooltip({ x, y, text: `${name}\nCommunity ${cid}` });
            } else {
              setTooltip(null);
            }
          },
          onClick: ({ object }: any) => {
            if (object) {
              const cid = object.properties?.community_id;
              if (cid !== undefined && cid >= 0) {
                setSelectedCommunityId(cid === selectedCommunityId ? null : cid);
              }
            }
          },
        }),
      ]
    : [];

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller={true}
        layers={layers}
        style={{ background: "#e8ecf0" }}
      />

      {tooltip && (
        <div style={{
          position: "absolute",
          left: tooltip.x + 12,
          top: tooltip.y + 12,
          background: "white",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
          padding: "6px 10px",
          fontSize: "12px",
          fontFamily: "var(--font-sans)",
          pointerEvents: "none",
          whiteSpace: "pre-line",
          boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
        }}>
          {tooltip.text}
        </div>
      )}

      <div style={{
        position: "absolute",
        bottom: 24,
        left: 16,
        background: "white",
        border: "1px solid var(--color-border)",
        borderRadius: "4px",
        padding: "8px 12px",
        fontSize: "11px",
        fontFamily: "var(--font-sans)",
      }}>
        {COMMUNITY_COLORS.map((color, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
            <div style={{
              width: 12, height: 12, borderRadius: 2,
              background: `rgb(${color.join(",")})`,
            }} />
            <span style={{ color: "var(--color-text-muted)" }}>Community {i}</span>
          </div>
        ))}
      </div>

      {selectedCommunityId !== null && (
        <CommunityPanel
          communityId={selectedCommunityId}
          onClose={() => setSelectedCommunityId(null)}
        />
      )}
    </div>
  );
}
