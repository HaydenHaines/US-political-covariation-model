"use client";
import { useEffect, useState } from "react";
import { getColorForSuperType } from "@/components/MapShell";

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

/** Properties embedded in each tract GeoJSON feature by bubble_dissolve. */
export interface TractFeatureProps {
  type_id: number;
  super_type: number;
  super_type_name: string;
  n_tracts: number;
  area_sqkm: number;
  // Demographics (type-level averages)
  median_hh_income?: number;
  pct_ba_plus?: number;
  pct_white_nh?: number;
  pct_black?: number;
  pct_hispanic?: number;
  evangelical_share?: number;
}

interface Props {
  tract: TractFeatureProps;
  onClose: () => void;
}

// Display configuration for tract demographics
const TRACT_DEMO_ROWS: { key: keyof TractFeatureProps; label: string; fmt: "pct" | "dollar" | "num" }[] = [
  { key: "median_hh_income", label: "Median income", fmt: "dollar" },
  { key: "pct_ba_plus", label: "Bachelor's+", fmt: "pct" },
  { key: "pct_white_nh", label: "White (non-Hispanic)", fmt: "pct" },
  { key: "pct_black", label: "Black", fmt: "pct" },
  { key: "pct_hispanic", label: "Hispanic", fmt: "pct" },
  { key: "evangelical_share", label: "Evangelical", fmt: "pct" },
];

function formatValue(value: number, fmt: "pct" | "dollar" | "num"): string {
  if (fmt === "dollar") return `$${Math.round(value).toLocaleString()}`;
  if (fmt === "pct") return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(1);
}

export function TractPanel({ tract, onClose }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const isMobile = useIsMobile();
  const superColor = getColorForSuperType(tract.super_type);

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
              {tract.super_type_name}
            </h3>
          </div>
          <p style={{ margin: "4px 0 0", fontSize: "12px", color: "var(--color-text-muted)" }}>
            Tract Type {tract.type_id}
          </p>
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

      {/* Body */}
      {(!isMobile || !collapsed) && (
        <div style={{ flex: 1, overflow: "auto", padding: "12px 16px" }}>
          {/* Community size */}
          <div style={{ marginBottom: "16px" }}>
            <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Community
            </p>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: "12px" }}>
              <span style={{ color: "var(--color-text-muted)" }}>Tracts</span>
              <span style={{ fontWeight: 500 }}>{tract.n_tracts.toLocaleString()}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: "12px" }}>
              <span style={{ color: "var(--color-text-muted)" }}>Area</span>
              <span style={{ fontWeight: 500 }}>{Math.round(tract.area_sqkm).toLocaleString()} km&sup2;</span>
            </div>
          </div>

          {/* Demographics */}
          <div style={{ marginBottom: "16px" }}>
            <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Demographics
            </p>
            {TRACT_DEMO_ROWS.map(({ key, label, fmt }) => {
              const value = tract[key];
              if (value == null) return null;
              return (
                <div key={key} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: "12px" }}>
                  <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
                  <span style={{ fontWeight: 500 }}>{formatValue(value as number, fmt)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
