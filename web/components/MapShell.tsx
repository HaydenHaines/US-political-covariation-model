"use client";
import { useState, useEffect, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchCounties, fetchSuperTypes, fetchTypes, type CountyRow, type TypeSummary } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";
import { TypePanel } from "@/components/TypePanel";

// 15-color perceptually-distinct palette. Assigned by super_type_id.
// Purely a visual concern — the model does not know about colors.
export const PALETTE: [number, number, number][] = [
  [31, 119, 180],   // 0: blue
  [255, 127, 14],   // 1: orange
  [44, 160, 44],    // 2: green
  [214, 39, 40],    // 3: red
  [148, 103, 189],  // 4: purple
  [140, 86, 75],    // 5: brown
  [227, 119, 194],  // 6: pink
  [127, 127, 127],  // 7: gray
  [188, 189, 34],   // 8: olive
  [23, 190, 207],   // 9: teal
  [174, 199, 232],  // 10: light blue
  [255, 187, 120],  // 11: light orange
  [152, 223, 138],  // 12: light green
  [255, 152, 150],  // 13: light red
  [197, 176, 213],  // 14: light purple
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
  longitude: -84.5,
  latitude: 31.5,
  zoom: 5.8,
  pitch: 0,
  bearing: 0,
};

export default function MapShell() {
  const { selectedCommunityId, setSelectedCommunityId, selectedTypeId, setSelectedTypeId } = useMapContext();
  const [geojson, setGeojson] = useState<any>(null);
  const [tractGeojson, setTractGeojson] = useState<any>(null);
  const [countyMap, setCountyMap] = useState<Record<string, CountyRow>>({});
  const [superTypeMap, setSuperTypeMap] = useState<Map<number, SuperTypeInfo>>(new Map());
  const [typeNameMap, setTypeNameMap] = useState<Map<number, string>>(new Map());
  const [hasTypeData, setHasTypeData] = useState(false);
  const [showTracts, setShowTracts] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [tractContext, setTractContext] = useState<TractContext | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/counties-fl-ga-al.geojson").then((r) => r.json()),
      fetchCounties(),
      fetchSuperTypes().catch(() => []),
      fetch("/tract-communities.geojson").then((r) => r.json()).catch(() => null),
      fetchTypes().catch(() => []),
    ]).then(([geo, counties, superTypes, tractGeo, types]) => {
      if (tractGeo) setTractGeojson(tractGeo);

      // Build super-type map from API
      const stMap = new Map<number, SuperTypeInfo>();
      superTypes.forEach((st: any) => {
        stMap.set(st.super_type_id, {
          name: st.display_name,
          color: getColorForSuperType(st.super_type_id),
        });
      });
      setSuperTypeMap(stMap);

      // Build fine type name map from API
      const tnMap = new Map<number, string>();
      (types as TypeSummary[]).forEach((t) => {
        tnMap.set(t.type_id, t.display_name);
      });
      setTypeNameMap(tnMap);

      // Build county map
      const map: Record<string, CountyRow> = {};
      let typeDataPresent = false;
      counties.forEach((c: CountyRow) => {
        map[c.county_fips] = c;
        if (c.super_type !== null) typeDataPresent = true;
      });
      setCountyMap(map);
      setHasTypeData(typeDataPresent);

      // Enrich GeoJSON
      const enriched = {
        ...geo,
        features: geo.features.map((f: any) => ({
          ...f,
          properties: {
            ...f.properties,
            community_id: map[f.properties.county_fips]?.community_id ?? -1,
            dominant_type: map[f.properties.county_fips]?.dominant_type ?? -1,
            super_type: map[f.properties.county_fips]?.super_type ?? -1,
          },
        })),
      };
      setGeojson(enriched);
    });
  }, []);

  const getColor = useCallback(
    (f: any): [number, number, number, number] => {
      const st: number = f.properties?.super_type ?? -1;
      const dt: number = f.properties?.dominant_type ?? f.properties?.type_id ?? -1;

      if (st >= 0 || hasTypeData) {
        const isSelected = selectedTypeId !== null && dt === selectedTypeId;
        const base = getColorForSuperType(st);
        return [...base, isSelected ? 255 : 180] as [number, number, number, number];
      }
      // Fallback for legacy community data (no type data)
      return [180, 180, 180, 120] as [number, number, number, number];
    },
    [selectedTypeId, hasTypeData, showTracts]
  );

  const getLineWidth = useCallback(
    (f: any): number => {
      if (hasTypeData) {
        const dt: number = f.properties?.dominant_type ?? -1;
        return selectedTypeId !== null && dt === selectedTypeId ? 800 : 200;
      }
      const cid: number = f.properties?.community_id ?? -1;
      return selectedCommunityId !== null && cid === selectedCommunityId ? 800 : 200;
    },
    [selectedCommunityId, selectedTypeId, hasTypeData]
  );

  const getSuperTypeName = useCallback(
    (superTypeId: number, feature?: any): string => {
      if (!showTracts) {
        return superTypeMap.get(superTypeId)?.name ?? `Type ${superTypeId}`;
      }
      // Tract view: read from GeoJSON property, fall back to map, then generic
      const geoName = feature?.properties?.super_type_name;
      if (geoName) return geoName;
      return superTypeMap.get(superTypeId)?.name ?? `Type ${superTypeId}`;
    },
    [showTracts, superTypeMap]
  );

  const activeData = showTracts && tractGeojson ? tractGeojson : geojson;
  const layerId = showTracts && tractGeojson ? "tract-communities" : "counties";

  const layers = activeData
    ? [
        new GeoJsonLayer({
          id: layerId,
          data: activeData,
          pickable: true,
          stroked: true,
          filled: true,
          getFillColor: getColor as any,
          getLineColor: [80, 80, 80, 120],
          getLineWidth,
          lineWidthUnits: "meters",
          updateTriggers: {
            getFillColor: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts],
            getLineWidth: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts],
          },
          onHover: ({ object, x, y }: any) => {
            if (object) {
              if (showTracts && tractGeojson) {
                const st = object.properties?.super_type;
                const tid = object.properties?.type_id;
                const n = object.properties?.n_tracts;
                const area = object.properties?.area_sqkm;
                const stName = getSuperTypeName(st, object);
                setTooltip({ x, y, text: `${stName}\nType ${tid} · ${n} tracts · ${Math.round(area)} km²` });
              } else {
                const name = object.properties?.county_name || object.properties?.county_fips;
                if (hasTypeData) {
                  const st = object.properties?.super_type;
                  const dt = object.properties?.dominant_type;
                  const stName = getSuperTypeName(st, object);
                  const typeName = typeNameMap.get(dt) ?? `Type ${dt}`;
                  setTooltip({ x, y, text: `${name}\n${typeName}\n${stName}` });
                } else {
                  const cid = object.properties?.community_id;
                  setTooltip({ x, y, text: `${name}\nCommunity ${cid}` });
                }
              }
            } else {
              setTooltip(null);
            }
          },
          onClick: ({ object }: any) => {
            if (object) {
              if (hasTypeData) {
                const dt = object.properties?.dominant_type ?? object.properties?.type_id;
                if (dt !== undefined && dt >= 0) {
                  setSelectedTypeId(dt === selectedTypeId ? null : dt);
                  setSelectedCommunityId(null);
                  // Capture tract community context when clicking in tract view
                  if (showTracts && object.properties?.n_tracts != null) {
                    setTractContext({
                      nTracts: object.properties.n_tracts,
                      areaSqkm: object.properties.area_sqkm ?? 0,
                      superTypeName: getSuperTypeName(object.properties?.super_type, object),
                    });
                  } else {
                    setTractContext(null);
                  }
                }
              } else {
                const cid = object.properties?.community_id;
                if (cid !== undefined && cid >= 0) {
                  setSelectedCommunityId(cid === selectedCommunityId ? null : cid);
                  setSelectedTypeId(null);
                  setTractContext(null);
                }
              }
            }
          },
        }),
      ]
    : [];

  // Build legend from API data — only show super-types that appear in counties
  const activeSuperTypeIds = new Set<number>();
  if (hasTypeData && !showTracts) {
    Object.values(countyMap).forEach((c) => {
      if (c.super_type !== null) activeSuperTypeIds.add(c.super_type);
    });
  } else if (showTracts && tractGeojson) {
    tractGeojson.features?.forEach((f: any) => {
      const st = f.properties?.super_type;
      if (st != null && st >= 0) activeSuperTypeIds.add(st);
    });
  }

  const legendEntries = Array.from(activeSuperTypeIds)
    .sort((a, b) => a - b)
    .map((id) => ({
      id,
      color: getColorForSuperType(id),
      label: superTypeMap.get(id)?.name ?? `Type ${id}`,
    }));

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

      {/* County/Tract toggle */}
      {tractGeojson && (
        <button
          onClick={() => setShowTracts((prev) => !prev)}
          style={{
            position: "absolute",
            top: 12,
            left: 16,
            background: showTracts ? "#2166ac" : "white",
            color: showTracts ? "white" : "#333",
            border: "1px solid var(--color-border)",
            borderRadius: "4px",
            padding: "6px 14px",
            fontSize: "12px",
            fontFamily: "var(--font-sans)",
            cursor: "pointer",
            fontWeight: 600,
            boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
          }}
        >
          {showTracts ? "Tract Communities" : "County Types"} ▾
        </button>
      )}

      {/* Legend */}
      {legendEntries.length > 0 && (
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
          {legendEntries.map((entry) => (
            <div key={entry.id} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
              <div style={{
                width: 12, height: 12, borderRadius: 2,
                background: `rgb(${entry.color.join(",")})`,
              }} />
              <span style={{ color: "var(--color-text-muted)" }}>{entry.label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Side panels */}
      {selectedCommunityId !== null && !hasTypeData && (
        <CommunityPanel
          communityId={selectedCommunityId}
          onClose={() => setSelectedCommunityId(null)}
        />
      )}

      {selectedTypeId !== null && hasTypeData && (
        <TypePanel
          typeId={selectedTypeId}
          superTypeMap={superTypeMap}
          tractContext={tractContext}
          onClose={() => { setSelectedTypeId(null); setTractContext(null); }}
        />
      )}
    </div>
  );
}
