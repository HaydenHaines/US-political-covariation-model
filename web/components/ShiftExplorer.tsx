"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { fetchTypeScatterData, type TypeScatterPoint } from "@/lib/api";
import { PALETTE } from "@/components/MapShell";

// ── Axis option configuration ─────────────────────────────────────────────────

const DEMO_KEYS: Array<{ key: string; label: string }> = [
  { key: "median_hh_income", label: "Median income" },
  { key: "median_age", label: "Median age" },
  { key: "pct_bachelors_plus", label: "Bachelor's+" },
  { key: "pct_white_nh", label: "White (non-Hispanic)" },
  { key: "pct_black", label: "Black" },
  { key: "pct_hispanic", label: "Hispanic" },
  { key: "pct_asian", label: "Asian" },
  { key: "pct_owner_occupied", label: "Owner-occupied" },
  { key: "pct_wfh", label: "Work from home" },
  { key: "pct_transit", label: "Transit commuters" },
  { key: "evangelical_share", label: "Evangelical" },
  { key: "mainline_share", label: "Mainline Protestant" },
  { key: "catholic_share", label: "Catholic" },
  { key: "black_protestant_share", label: "Black Protestant" },
  { key: "congregations_per_1000", label: "Congregations/1K" },
  { key: "log_pop_density", label: "Pop density (log)" },
  { key: "mean_pred_dem_share", label: "Predicted D share" },
];

// Keys that are shift columns (used to decide whether to draw zero rule)
function isShiftKey(key: string): boolean {
  return key.includes("_shift_") || key.startsWith("shift_");
}

function prettifyKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\bpct\b/g, "%")
    .replace(/\bhh\b/g, "household")
    .replace(/^./, (c) => c.toUpperCase());
}

// Get value for a key from either demographics or shift_profile
function getValue(
  point: TypeScatterPoint,
  key: string
): number | undefined {
  if (key in point.demographics) return point.demographics[key];
  if (key in point.shift_profile) return point.shift_profile[key];
  return undefined;
}

// ── Super-type legend ─────────────────────────────────────────────────────────

interface SuperTypeLegendEntry {
  super_type_id: number;
  name: string;
}

function SuperTypeLegend({
  entries,
}: {
  entries: SuperTypeLegendEntry[];
}) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "8px 16px",
        padding: "8px 0 4px",
      }}
    >
      {entries.map((e) => {
        const [r, g, b] = PALETTE[e.super_type_id % PALETTE.length];
        return (
          <div
            key={e.super_type_id}
            style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px" }}
          >
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: `rgb(${r},${g},${b})`,
                flexShrink: 0,
              }}
            />
            <span style={{ color: "var(--color-text-muted)" }}>{e.name}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

interface TooltipState {
  point: TypeScatterPoint;
  xKey: string;
  yKey: string;
  x: number;
  y: number;
}

function Tooltip({ state }: { state: TooltipState }) {
  const { point, xKey, yKey, x, y } = state;
  const xVal = getValue(point, xKey);
  const yVal = getValue(point, yKey);

  function fmt(key: string, val: number | undefined): string {
    if (val == null) return "—";
    if (key.includes("income")) return `$${Math.round(val).toLocaleString()}`;
    if (key.startsWith("pct_") || key.endsWith("_share")) return `${(val * 100).toFixed(1)}%`;
    if (key.includes("_shift_")) return val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
    return val.toFixed(2);
  }

  return (
    <div
      style={{
        position: "fixed",
        left: x + 12,
        top: y - 8,
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: 4,
        padding: "8px 10px",
        fontSize: "12px",
        pointerEvents: "none",
        zIndex: 100,
        boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
        maxWidth: 200,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{point.display_name}</div>
      <div style={{ color: "var(--color-text-muted)", marginBottom: 4 }}>
        {point.n_counties} counties
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span style={{ color: "var(--color-text-muted)" }}>{prettifyKey(xKey)}</span>
        <span style={{ fontWeight: 500 }}>{fmt(xKey, xVal)}</span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span style={{ color: "var(--color-text-muted)" }}>{prettifyKey(yKey)}</span>
        <span style={{ fontWeight: 500 }}>{fmt(yKey, yVal)}</span>
      </div>
    </div>
  );
}

// ── Scatter plot ──────────────────────────────────────────────────────────────

function ScatterPlot({
  data,
  xKey,
  yKey,
  onHover,
}: {
  data: TypeScatterPoint[];
  xKey: string;
  yKey: string;
  onHover: (state: TooltipState | null, event?: MouseEvent) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLDivElement>(null);
  const [plotWidth, setPlotWidth] = useState(388);

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(([entry]) => setPlotWidth(Math.floor(entry.contentRect.width)));
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!ref.current || data.length === 0) return;

    const plotData = data
      .map((d) => ({
        ...d,
        xVal: getValue(d, xKey),
        yVal: getValue(d, yKey),
        color: PALETTE[d.super_type_id % PALETTE.length],
      }))
      .filter((d) => d.xVal != null && d.yVal != null);

    const maxN = Math.max(...plotData.map((d) => d.n_counties));

    const marks: Plot.Markish[] = [];

    if (isShiftKey(yKey)) marks.push(Plot.ruleY([0], { stroke: "#ccc", strokeDasharray: "3,3" }));
    if (isShiftKey(xKey)) marks.push(Plot.ruleX([0], { stroke: "#ccc", strokeDasharray: "3,3" }));

    marks.push(
      Plot.dot(plotData, {
        x: "xVal",
        y: "yVal",
        r: (d) => 3 + 8 * Math.sqrt(d.n_counties / maxN),
        fill: (d) => `rgb(${(d.color as [number,number,number]).join(",")})`,
        fillOpacity: 0.8,
        stroke: "white",
        strokeWidth: 0.5,
        tip: false,
      })
    );

    const effectiveWidth = plotWidth > 0 ? plotWidth : 388;

    const plot = Plot.plot({
      width: effectiveWidth,
      height: 300,
      marginLeft: 48,
      marginRight: 8,
      marginTop: 12,
      marginBottom: 40,
      style: {
        background: "transparent",
        color: "var(--color-text)",
        fontSize: "11px",
      },
      x: {
        label: prettifyKey(xKey),
        labelOffset: 36,
        grid: true,
        tickFormat: xKey.includes("income")
          ? (d: number) => `$${(d / 1000).toFixed(0)}K`
          : xKey.startsWith("pct_") || xKey.endsWith("_share")
          ? (d: number) => `${Math.round(d * 100)}%`
          : undefined,
      },
      y: {
        label: prettifyKey(yKey),
        grid: true,
        tickFormat: yKey.includes("income")
          ? (d: number) => `$${(d / 1000).toFixed(0)}K`
          : yKey.startsWith("pct_") || yKey.endsWith("_share")
          ? (d: number) => `${Math.round(d * 100)}%`
          : undefined,
      },
      marks,
    });

    ref.current.innerHTML = "";
    ref.current.appendChild(plot);

    // Tooltip via mousemove on the plot SVG
    const svg = ref.current.querySelector("svg");
    if (svg) {
      const handleMouseMove = (e: MouseEvent) => {
        const svgRect = svg.getBoundingClientRect();
        const mx = e.clientX - svgRect.left;
        const my = e.clientY - svgRect.top;

        // Find nearest point using plotData coordinates
        // We use the SVG viewBox to map pixel coords to data coords
        // Simpler: just find nearest by euclidean in pixel space using plot dimensions
        const plotLeft = 48;
        const plotRight = effectiveWidth - 8;
        const plotTop = 12;
        const plotBottom = 300 - 40;

        const xVals = plotData.map((d) => d.xVal as number);
        const yVals = plotData.map((d) => d.yVal as number);
        const xMin = Math.min(...xVals);
        const xMax = Math.max(...xVals);
        const yMin = Math.min(...yVals);
        const yMax = Math.max(...yVals);

        const xScale = (val: number) =>
          plotLeft + ((val - xMin) / (xMax - xMin || 1)) * (plotRight - plotLeft);
        const yScale = (val: number) =>
          plotBottom - ((val - yMin) / (yMax - yMin || 1)) * (plotBottom - plotTop);

        let closest: typeof plotData[0] | null = null;
        let minDist = Infinity;
        for (const d of plotData) {
          const px = xScale(d.xVal as number);
          const py = yScale(d.yVal as number);
          const dist = Math.hypot(mx - px, my - py);
          if (dist < minDist) {
            minDist = dist;
            closest = d;
          }
        }

        if (closest && minDist < 30) {
          onHover(
            { point: closest, xKey, yKey, x: e.clientX, y: e.clientY },
            e
          );
        } else {
          onHover(null);
        }
      };

      const handleMouseLeave = () => onHover(null);

      svg.addEventListener("mousemove", handleMouseMove as EventListener);
      svg.addEventListener("mouseleave", handleMouseLeave);

      return () => {
        svg.removeEventListener("mousemove", handleMouseMove as EventListener);
        svg.removeEventListener("mouseleave", handleMouseLeave);
        ref.current?.removeChild(plot);
      };
    }

    return () => {
      if (ref.current?.contains(plot)) ref.current.removeChild(plot);
    };
  }, [data, xKey, yKey, onHover, plotWidth]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <div ref={ref} />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function ShiftExplorer() {
  const [data, setData] = useState<TypeScatterPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [xKey, setXKey] = useState("pct_bachelors_plus");
  const [yKey, setYKey] = useState("pres_d_shift_20_24");
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  useEffect(() => {
    fetchTypeScatterData()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Derive available shift keys from data
  const shiftKeys = data.length > 0
    ? Object.keys(data[0].shift_profile).sort()
    : [];

  // Default yKey: prefer pres_d_shift_20_24 if available
  const resolvedYKey = shiftKeys.includes(yKey) || yKey in (data[0]?.demographics ?? {})
    ? yKey
    : shiftKeys[0] ?? yKey;

  // Super-type legend entries (unique, sorted)
  const superTypeEntries: SuperTypeLegendEntry[] = [];
  const seenSuperTypes = new Set<number>();
  for (const d of data) {
    if (!seenSuperTypes.has(d.super_type_id)) {
      seenSuperTypes.add(d.super_type_id);
      superTypeEntries.push({ super_type_id: d.super_type_id, name: `Super-type ${d.super_type_id + 1}` });
    }
  }
  superTypeEntries.sort((a, b) => a.super_type_id - b.super_type_id);

  const handleHover = (state: TooltipState | null) => {
    setTooltip(state);
  };

  const selectStyle: React.CSSProperties = {
    fontSize: "12px",
    padding: "4px 8px",
    border: "1px solid var(--color-border)",
    borderRadius: 3,
    background: "var(--color-surface)",
    color: "var(--color-text)",
    flex: 1,
  };

  return (
    <div style={{ padding: "12px 16px", minWidth: 0, overflow: "hidden" }}>
      <h3 style={{
        margin: "0 0 4px",
        fontFamily: "var(--font-serif)",
        fontSize: "16px",
      }}>
        Shift Explorer
      </h3>
      <p style={{ margin: "0 0 12px", fontSize: "12px", color: "var(--color-text-muted)" }}>
        Each dot is one of the 100 electoral types. Size = county count.
      </p>

      {loading && (
        <div style={{ color: "var(--color-text-muted)", fontSize: "13px", padding: "20px 0" }}>
          Loading...
        </div>
      )}

      {error && (
        <div style={{ color: "var(--color-rep)", fontSize: "13px", padding: "8px 0" }}>
          Error: {error}
        </div>
      )}

      {!loading && !error && (
        <>
          {/* Axis controls */}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginBottom: "10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "12px", color: "var(--color-text-muted)", width: 20 }}>X</span>
              <select
                value={xKey}
                onChange={(e) => setXKey(e.target.value)}
                style={selectStyle}
                aria-label="X-axis variable"
              >
                <optgroup label="Demographics">
                  {DEMO_KEYS.map(({ key, label }) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </optgroup>
                {shiftKeys.length > 0 && (
                  <optgroup label="Shifts">
                    {shiftKeys.map((key) => (
                      <option key={key} value={key}>{prettifyKey(key)}</option>
                    ))}
                  </optgroup>
                )}
              </select>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "12px", color: "var(--color-text-muted)", width: 20 }}>Y</span>
              <select
                value={resolvedYKey}
                onChange={(e) => setYKey(e.target.value)}
                style={selectStyle}
                aria-label="Y-axis variable"
              >
                {shiftKeys.length > 0 && (
                  <optgroup label="Shifts">
                    {shiftKeys.map((key) => (
                      <option key={key} value={key}>{prettifyKey(key)}</option>
                    ))}
                  </optgroup>
                )}
                <optgroup label="Demographics">
                  {DEMO_KEYS.map(({ key, label }) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </optgroup>
              </select>
            </div>
          </div>

          {/* Scatter plot */}
          <ScatterPlot
            data={data}
            xKey={xKey}
            yKey={resolvedYKey}
            onHover={handleHover}
          />

          {/* Legend */}
          <SuperTypeLegend entries={superTypeEntries} />

          {/* Count */}
          <p style={{ margin: "4px 0 0", fontSize: "11px", color: "var(--color-text-muted)" }}>
            {data.length} types shown
          </p>
        </>
      )}

      {/* Tooltip */}
      {tooltip && <Tooltip state={tooltip} />}
    </div>
  );
}
