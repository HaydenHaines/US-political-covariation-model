"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import type { TypeScatterPoint } from "@/lib/api";
import { formatMargin } from "@/lib/typeDisplay";
import { PALETTE } from "@/components/MapShell";

// ── Helpers ───────────────────────────────────────────────────────────────────

// Keys that are shift columns (used to decide whether to draw zero rule)
export function isShiftKey(key: string): boolean {
  return key.includes("_shift_") || key.startsWith("shift_");
}

export function prettifyKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\bpct\b/g, "%")
    .replace(/\bhh\b/g, "household")
    .replace(/^./, (c) => c.toUpperCase());
}

// Get value for a key from either demographics or shift_profile
export function getValue(
  point: TypeScatterPoint,
  key: string
): number | undefined {
  if (key in point.demographics) return point.demographics[key];
  if (key in point.shift_profile) return point.shift_profile[key];
  return undefined;
}

// ── Super-type legend ─────────────────────────────────────────────────────────

export interface SuperTypeLegendEntry {
  super_type_id: number;
  name: string;
}

export function SuperTypeLegend({
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

export interface TooltipState {
  point: TypeScatterPoint;
  xKey: string;
  yKey: string;
  x: number;
  y: number;
}

export function Tooltip({ state }: { state: TooltipState }) {
  const { point, xKey, yKey, x, y } = state;
  const xVal = getValue(point, xKey);
  const yVal = getValue(point, yKey);

  function fmt(key: string, val: number | undefined): string {
    if (val == null) return "—";
    if (key.includes("income")) return `$${Math.round(val).toLocaleString()}`;
    if (key === "mean_pred_dem_share") return formatMargin(val);
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

const PLOT_HEIGHT = 400;

export interface ShiftScatterPlotProps {
  data: TypeScatterPoint[];
  xKey: string;
  yKey: string;
  onHover: (state: TooltipState | null, event?: MouseEvent) => void;
  onClick?: (typeId: number) => void;
  selectedIds?: Set<number>;
}

export function ShiftScatterPlot({
  data,
  xKey,
  yKey,
  onHover,
  onClick,
  selectedIds,
}: ShiftScatterPlotProps) {
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

    // Draw selected rings behind the dots so selection is visually obvious
    const selectedData = selectedIds ? plotData.filter((d) => selectedIds.has(d.type_id)) : [];
    if (selectedData.length > 0) {
      marks.push(
        Plot.dot(selectedData, {
          x: "xVal",
          y: "yVal",
          r: (d) => 5 + 8 * Math.sqrt(d.n_counties / maxN),
          fill: "none",
          stroke: "var(--color-text, #333)",
          strokeWidth: 2.5,
          tip: false,
        })
      );
    }

    marks.push(
      Plot.dot(plotData, {
        x: "xVal",
        y: "yVal",
        r: (d) => 3 + 8 * Math.sqrt(d.n_counties / maxN),
        fill: (d) => `rgb(${(d.color as [number,number,number]).join(",")})`,
        fillOpacity: (d) => (selectedIds && selectedIds.size > 0 && !selectedIds.has(d.type_id) ? 0.35 : 0.8),
        stroke: "white",
        strokeWidth: 0.5,
        tip: false,
      })
    );

    const effectiveWidth = plotWidth > 0 ? plotWidth : 388;

    const plot = Plot.plot({
      width: effectiveWidth,
      height: PLOT_HEIGHT,
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
    const node = ref.current;

    // Tooltip via mousemove on the plot SVG.
    // We compute pixel coordinates manually from plot margins because Observable Plot
    // doesn't expose a data-to-pixel scale externally.
    const svg = ref.current.querySelector("svg");
    if (svg) {
      // Reusable helper: find the nearest plotData point to a pixel position (mx, my)
      // within the plot's inner frame. Returns null if no point is within 30px.
      const findNearest = (mx: number, my: number) => {
        const plotLeft = 48;
        const plotRight = effectiveWidth - 8;
        const plotTop = 12;
        const plotBottom = PLOT_HEIGHT - 40;

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

        return minDist < 30 ? closest : null;
      }

      const handleMouseMove = (e: MouseEvent) => {
        const svgRect = svg.getBoundingClientRect();
        const mx = e.clientX - svgRect.left;
        const my = e.clientY - svgRect.top;
        const nearest = findNearest(mx, my);
        if (nearest) {
          onHover({ point: nearest, xKey, yKey, x: e.clientX, y: e.clientY }, e);
        } else {
          onHover(null);
        }
      };

      const handleMouseLeave = () => onHover(null);

      const handleClick = (e: MouseEvent) => {
        if (!onClick) return;
        const svgRect = svg.getBoundingClientRect();
        const mx = e.clientX - svgRect.left;
        const my = e.clientY - svgRect.top;
        const nearest = findNearest(mx, my);
        if (nearest) onClick(nearest.type_id);
      };

      svg.addEventListener("mousemove", handleMouseMove as EventListener);
      svg.addEventListener("mouseleave", handleMouseLeave);
      svg.addEventListener("click", handleClick as EventListener);
      svg.style.cursor = "pointer";

      return () => {
        svg.removeEventListener("mousemove", handleMouseMove as EventListener);
        svg.removeEventListener("mouseleave", handleMouseLeave);
        svg.removeEventListener("click", handleClick as EventListener);
        node?.removeChild(plot);
      };
    }

    return () => {
      if (node?.contains(plot)) node.removeChild(plot);
    };
  }, [data, xKey, yKey, onHover, onClick, selectedIds, plotWidth]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <div ref={ref} />
    </div>
  );
}
