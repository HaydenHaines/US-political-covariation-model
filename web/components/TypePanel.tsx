"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { fetchTypeDetail, type TypeDetail, type TypeCounty } from "@/lib/api";
import { getColorForSuperType, type SuperTypeInfo, type TractContext } from "@/components/MapShell";
import { DEMO_DISPLAY, DEMO_SKIP, prettifyKey, inferFormat } from "@/lib/typeDisplay";

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    setIsMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return isMobile;
}

interface Props {
  typeId: number;
  superTypeMap: Map<number, SuperTypeInfo>;
  tractContext?: TractContext | null;
  onClose: () => void;
}

function ShiftSparkline({ shiftProfile }: { shiftProfile: Record<string, number> }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    const entries = Object.entries(shiftProfile)
      .filter(([k]) => k.startsWith("pres_d_shift_"))
      .map(([key, value]) => {
        const parts = key.split("_");
        const year = parseInt("20" + parts[parts.length - 1], 10);
        return { year, shift: value, label: key.replace("pres_d_shift_", "") };
      })
      .sort((a, b) => a.year - b.year);

    if (entries.length === 0) return;

    const plot = Plot.plot({
      width: 260,
      height: 80,
      marginLeft: 28,
      marginRight: 4,
      marginTop: 8,
      marginBottom: 20,
      x: { label: null, tickFormat: (d: number) => `'${String(d).slice(2)}` },
      y: { label: "shift", grid: true, zero: true },
      marks: [
        Plot.ruleY([0], { stroke: "#ccc" }),
        Plot.line(entries, {
          x: "year",
          y: "shift",
          stroke: entries.some((e) => e.shift > 0) ? "var(--color-dem)" : "var(--color-rep)",
          strokeWidth: 1.5,
        }),
        Plot.dot(entries, { x: "year", y: "shift", r: 2.5, fill: "var(--color-text)" }),
      ],
    });

    ref.current.innerHTML = "";
    ref.current.appendChild(plot);

    return () => {
      ref.current?.removeChild(plot);
    };
  }, [shiftProfile]);

  return <div ref={ref} />;
}

function DemographicRow({ label, value, fmt }: { label: string; value: number | undefined; fmt?: "pct" | "dollar" | "num" }) {
  if (value == null) return null;
  let display: string;
  if (fmt === "dollar") display = `$${Math.round(value).toLocaleString()}`;
  else if (fmt === "pct") display = `${(value * 100).toFixed(1)}%`;
  else display = value.toFixed(1);
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: "12px" }}>
      <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span style={{ fontWeight: 500 }}>{display}</span>
    </div>
  );
}


export function TypePanel({ typeId, superTypeMap, tractContext, onClose }: Props) {
  const [detail, setDetail] = useState<TypeDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const isMobile = useIsMobile();

  useEffect(() => {
    setLoading(true);
    fetchTypeDetail(typeId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [typeId]);

  // Reset collapsed state when a new type is selected
  useEffect(() => {
    setCollapsed(false);
  }, [typeId]);

  const superColor = detail
    ? getColorForSuperType(detail.super_type_id)
    : [127, 127, 127];

  // On mobile: use a fixed bottom-sheet so the panel doesn't cover the map.
  // Collapsed = just the header strip; expanded = up to 55vh scrollable sheet.
  const mobileStyle: React.CSSProperties = isMobile ? {
    position: "fixed",
    top: "auto",
    bottom: 0,
    right: 0,
    left: 0,
    width: "100%",
    height: collapsed ? "auto" : "min(55vh, 500px)",
    borderLeft: "none",
    borderTop: "2px solid var(--color-border)",
    boxShadow: "0 -4px 16px rgba(0,0,0,0.15)",
    zIndex: 200,
  } : {};

  return (
    <div style={{
      position: "absolute",
      top: 0,
      right: 0,
      width: "320px",
      height: "100%",
      background: "var(--color-surface)",
      borderLeft: "1px solid var(--color-border)",
      display: "flex",
      flexDirection: "column",
      boxShadow: "-2px 0 8px rgba(0,0,0,0.08)",
      zIndex: 10,
      ...mobileStyle,
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 16px 12px",
        borderBottom: "1px solid var(--color-border)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <div style={{
              width: 14, height: 14, borderRadius: 3,
              background: `rgb(${superColor.join(",")})`,
            }} />
            <h3 style={{ margin: 0, fontFamily: "var(--font-serif)", fontSize: "16px" }}>
              {detail?.display_name ?? `Type ${typeId}`}
            </h3>
          </div>
          {detail && (
            <p style={{ margin: "4px 0 0", fontSize: "12px", color: "var(--color-text-muted)" }}>
              {detail.n_counties} counties
              {superTypeMap.get(detail.super_type_id)?.name && (
                <> &middot; {superTypeMap.get(detail.super_type_id)!.name}</>
              )}
            </p>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          {isMobile && (
            <button
              onClick={() => setCollapsed((prev) => !prev)}
              aria-label={collapsed ? "Expand panel" : "Collapse panel"}
              style={{ border: "none", background: "none", cursor: "pointer", fontSize: "18px", color: "var(--color-text-muted)", lineHeight: 1, padding: "0 4px" }}
            >
              {collapsed ? "▼" : "▲"}
            </button>
          )}
          <button
            onClick={onClose}
            aria-label="Close panel"
            style={{ border: "none", background: "none", cursor: "pointer", fontSize: "20px", color: "var(--color-text-muted)", lineHeight: 1 }}
          >&times;</button>
        </div>
      </div>

      {/* Body: hidden when collapsed on mobile */}
      {(!isMobile || !collapsed) && (
        <>
      {/* Tract community context — shown when clicked from tract view */}
      {tractContext && (
        <div style={{
          padding: "10px 16px",
          borderBottom: "1px solid var(--color-border)",
          background: "var(--color-bg-muted, #f7f8fa)",
        }}>
          <p style={{ margin: "0 0 4px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
            This community
          </p>
          <p style={{ margin: 0, fontSize: "13px" }}>
            {tractContext.nTracts} tracts &middot; {Math.round(tractContext.areaSqkm)} km²
          </p>
        </div>
      )}

      {loading && (
        <div style={{ padding: "20px", color: "var(--color-text-muted)", fontSize: "13px" }}>
          Loading...
        </div>
      )}

      {detail && !loading && (
        <div style={{ flex: 1, overflow: "auto", padding: "12px 16px" }}>
          {/* Narrative description */}
          {detail.narrative && (
            <div style={{ marginBottom: "16px" }}>
              <p style={{
                margin: 0,
                fontSize: "13px",
                lineHeight: "1.5",
                color: "var(--color-text)",
              }}>
                {detail.narrative}
              </p>
            </div>
          )}

          {/* Shift sparkline */}
          {detail.shift_profile && Object.keys(detail.shift_profile).length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
                Shift pattern
              </p>
              <ShiftSparkline shiftProfile={detail.shift_profile} />
            </div>
          )}

          {/* Demographics */}
          {Object.keys(detail.demographics).length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
                Demographics
              </p>
              {Object.entries(detail.demographics)
                .filter(([key]) => !DEMO_SKIP.has(key))
                .map(([key, value]) => {
                  const display = DEMO_DISPLAY[key];
                  return (
                    <DemographicRow
                      key={key}
                      label={display?.label ?? prettifyKey(key)}
                      value={value}
                      fmt={display?.fmt ?? inferFormat(key)}
                    />
                  );
                })}
            </div>
          )}

          {/* County list */}
          <div>
            <p style={{ margin: "0 0 8px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Counties ({detail.n_counties})
            </p>
            {detail.counties.map((county: TypeCounty) => (
              <div key={county.county_fips} style={{
                padding: "5px 0",
                borderBottom: "1px solid var(--color-border)",
                fontSize: "13px",
              }}>
                {county.county_name
                  ? `${county.county_name}, ${county.state_abbr}`
                  : `${county.county_fips} (${county.state_abbr})`}
              </div>
            ))}
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}
