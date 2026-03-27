"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { fetchCommunityDetail, type CommunityDetail, type CommunityDemographics } from "@/lib/api";
import { formatMargin, marginColor } from "@/lib/typeDisplay";

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
  communityId: number;
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
      y: { label: "Δ D%", grid: true, zero: true },
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

function DemographicRow({ label, value, fmt }: { label: string; value: number | null; fmt?: "pct" | "dollar" | "num" }) {
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

function DemographicsSection({ d }: { d: CommunityDemographics }) {
  return (
    <div style={{ marginBottom: "16px" }}>
      <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
        Demographics
      </p>
      <DemographicRow label="Median income" value={d.median_hh_income} fmt="dollar" />
      <DemographicRow label="Median age" value={d.median_age} fmt="num" />
      <DemographicRow label="White (non-Hispanic)" value={d.pct_white_nh} fmt="pct" />
      <DemographicRow label="Black" value={d.pct_black} fmt="pct" />
      <DemographicRow label="Hispanic" value={d.pct_hispanic} fmt="pct" />
      <DemographicRow label="Asian" value={d.pct_asian} fmt="pct" />
      <DemographicRow label="Bachelor's+" value={d.pct_bachelors_plus} fmt="pct" />
      <DemographicRow label="Owner-occupied" value={d.pct_owner_occupied} fmt="pct" />
      <DemographicRow label="Work from home" value={d.pct_wfh} fmt="pct" />
      <DemographicRow label="Management" value={d.pct_management} fmt="pct" />
      {(d.evangelical_share != null || d.catholic_share != null) && (
        <>
          <div style={{ margin: "8px 0 4px", borderTop: "1px solid var(--color-border)" }} />
          <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
            Religious composition
          </p>
          <DemographicRow label="Evangelical" value={d.evangelical_share} fmt="pct" />
          <DemographicRow label="Mainline Protestant" value={d.mainline_share} fmt="pct" />
          <DemographicRow label="Catholic" value={d.catholic_share} fmt="pct" />
          <DemographicRow label="Black Protestant" value={d.black_protestant_share} fmt="pct" />
          <DemographicRow label="Adherence rate" value={d.religious_adherence_rate} fmt="pct" />
        </>
      )}
    </div>
  );
}

export function CommunityPanel({ communityId, onClose }: Props) {
  const [detail, setDetail] = useState<CommunityDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const isMobile = useIsMobile();

  useEffect(() => {
    setLoading(true);
    fetchCommunityDetail(communityId)
      .then(setDetail)
      .finally(() => setLoading(false));
  }, [communityId]);

  useEffect(() => {
    setCollapsed(false);
  }, [communityId]);

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
          <h3 style={{ margin: 0, fontFamily: "var(--font-serif)", fontSize: "16px" }}>
            {detail?.display_name ?? `Community ${communityId}`}
          </h3>
          {detail && (
            <p style={{ margin: "4px 0 0", fontSize: "12px", color: "var(--color-text-muted)" }}>
              {detail.n_counties} counties · {detail.states.join(", ")}
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
          >×</button>
        </div>
      </div>

      {(!isMobile || !collapsed) && (
        <>
      {loading && (
        <div style={{ padding: "20px", color: "var(--color-text-muted)", fontSize: "13px" }}>
          Loading…
        </div>
      )}

      {detail && !loading && (
        <div style={{ flex: 1, overflow: "auto", padding: "12px 16px" }}>
          {/* Shift sparkline */}
          <div style={{ marginBottom: "16px" }}>
            <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Presidential swing pattern
            </p>
            <ShiftSparkline shiftProfile={detail.shift_profile} />
          </div>

          {/* Demographics */}
          {detail.demographics && <DemographicsSection d={detail.demographics} />}

          {/* County list */}
          <div>
            <p style={{ margin: "0 0 8px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
              Counties ({detail.n_counties})
            </p>
            {detail.counties.map((county) => {
              const share = county.pred_dem_share;
              return (
                <div key={county.county_fips} style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "5px 0",
                  borderBottom: "1px solid var(--color-border)",
                  fontSize: "13px",
                }}>
                  <span>{county.county_name ?? county.county_fips}</span>
                  {share !== null && (
                    <span style={{ color: marginColor(share), fontWeight: "600" }}>
                      {formatMargin(share)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}
