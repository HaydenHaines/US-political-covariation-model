"use client";

import { useState, useMemo, useCallback, useRef } from "react";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { Group } from "@visx/group";
import { useTooltip, TooltipWithBounds } from "@visx/tooltip";
import { SlidersHorizontal, X } from "lucide-react";
import { useTypeScatter } from "@/lib/hooks/use-type-scatter";
import { getFieldConfig } from "@/lib/config/display";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import type { TypeScatterPoint } from "@/lib/types";

// ---------------------------------------------------------------------------
// Axis field options
// ---------------------------------------------------------------------------

/** Demographic fields available as scatter axes, with human-readable labels. */
const AXIS_FIELDS: { key: string; label: string }[] = [
  { key: "mean_pred_dem_share",  label: "Predicted Dem share" },
  { key: "pct_white_nh",        label: "White (non-Hispanic)" },
  { key: "pct_black",           label: "Black" },
  { key: "pct_hispanic",        label: "Hispanic" },
  { key: "pct_asian",           label: "Asian" },
  { key: "pct_bachelors_plus",  label: "Bachelor's degree+" },
  { key: "median_hh_income",    label: "Median household income" },
  { key: "log_pop_density",     label: "Log population density" },
  { key: "evangelical_share",   label: "Evangelical share" },
  { key: "catholic_share",      label: "Catholic share" },
  { key: "median_age",          label: "Median age" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getPointValue(point: TypeScatterPoint, field: string): number | null {
  // Top-level fields on scatter data (not in demographics sub-object)
  if (field === "mean_pred_dem_share") {
    const v = (point as unknown as Record<string, unknown>)[field];
    return typeof v === "number" ? v : null;
  }
  const v = point.demographics[field];
  return typeof v === "number" ? v : null;
}

function formatAxisValue(field: string, value: number): string {
  const cfg = getFieldConfig(field);
  if (cfg.format === "percent") return `${(value * 100).toFixed(0)}%`;
  if (cfg.format === "margin") return `${(value * 100).toFixed(0)}%`;
  if (cfg.format === "currency") return `$${Math.round(value / 1_000)}K`;
  return value.toFixed(1);
}

// ---------------------------------------------------------------------------
// Margins / dimensions
// ---------------------------------------------------------------------------

const MARGIN = { top: 20, right: 20, bottom: 48, left: 56 };

interface TooltipData {
  point: TypeScatterPoint;
  x: number;
  y: number;
}

interface ScatterPlotProps {
  width?: number;
  height?: number;
}

/**
 * Interactive scatter plot for the types explorer.
 *
 * X and Y axes are user-selectable from dropdowns.
 * Desktop (≥768px): dropdowns inline above the chart.
 * Mobile (<768px): dropdowns in a bottom sheet triggered by a floating button.
 * Dots are colored by super-type, sized by county count, with hover tooltips.
 */
export function ScatterPlot({ width = 640, height = 420 }: ScatterPlotProps) {
  const { data: points, isLoading } = useTypeScatter();
  const [xField, setXField] = useState("pct_white_nh");
  const [yField, setYField] = useState("mean_pred_dem_share");
  const [bottomSheetOpen, setBottomSheetOpen] = useState(false);

  const { tooltipData, tooltipLeft, tooltipTop, showTooltip, hideTooltip } =
    useTooltip<TooltipData>();

  // Long-press timer ref for touch-based tooltip activation (≥400ms hold)
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleTouchStart = useCallback(
    (d: { p: TypeScatterPoint; xv: number; yv: number }, svgX: number, svgY: number) => {
      longPressTimer.current = setTimeout(() => {
        showTooltip({
          tooltipData: { point: d.p, x: d.xv, y: d.yv },
          tooltipLeft: svgX + MARGIN.left,
          tooltipTop: svgY + MARGIN.top,
        });
      }, 400);
    },
    [showTooltip],
  );

  const handleTouchEnd = useCallback(() => {
    if (longPressTimer.current !== null) {
      clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
    }
  }, []);

  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  // Compute axis scales from the data
  const { xScale, yScale, plotPoints } = useMemo(() => {
    if (!points || points.length === 0) {
      return { xScale: null, yScale: null, plotPoints: [] };
    }

    const resolved = points
      .map((p) => {
        const xv = getPointValue(p, xField);
        const yv = getPointValue(p, yField);
        return xv !== null && yv !== null ? { p, xv, yv } : null;
      })
      .filter((d): d is { p: TypeScatterPoint; xv: number; yv: number } => d !== null);

    if (resolved.length === 0) return { xScale: null, yScale: null, plotPoints: [] };

    const xs = resolved.map((d) => d.xv);
    const ys = resolved.map((d) => d.yv);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);

    // Pad by 5% on each side
    const xPad = (xMax - xMin) * 0.05 || 0.01;
    const yPad = (yMax - yMin) * 0.05 || 0.01;

    const xScale = scaleLinear({
      domain: [xMin - xPad, xMax + xPad],
      range: [0, innerWidth],
    });

    const yScale = scaleLinear({
      domain: [yMin - yPad, yMax + yPad],
      range: [innerHeight, 0],
    });

    return { xScale, yScale, plotPoints: resolved };
  }, [points, xField, yField, innerWidth, innerHeight]);

  const handleMouseEnter = useCallback(
    (d: { p: TypeScatterPoint; xv: number; yv: number }, svgX: number, svgY: number) => {
      showTooltip({
        tooltipData: { point: d.p, x: d.xv, y: d.yv },
        tooltipLeft: svgX + MARGIN.left,
        tooltipTop: svgY + MARGIN.top,
      });
    },
    [showTooltip],
  );

  if (isLoading) {
    return (
      <div className="text-sm" style={{ color: "var(--color-text-muted)", padding: "24px 0" }}>
        Loading scatter data…
      </div>
    );
  }

  if (!points || points.length === 0) {
    return (
      <div className="text-sm" style={{ color: "var(--color-text-muted)", padding: "24px 0" }}>
        Scatter data unavailable.
      </div>
    );
  }

  const xLabel = AXIS_FIELDS.find((f) => f.key === xField)?.label ?? xField;
  const yLabel = AXIS_FIELDS.find((f) => f.key === yField)?.label ?? yField;

  return (
    <div>
      {/* Desktop: axis selection dropdowns (≥768px) */}
      <div className="hidden md:flex flex-wrap gap-4 mb-4 items-center text-sm">
        <label className="flex items-center gap-2">
          <span style={{ color: "var(--color-text-muted)" }}>X axis:</span>
          <select
            value={xField}
            onChange={(e) => setXField(e.target.value)}
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: "var(--color-border)",
              background: "var(--color-surface)",
              color: "var(--color-text)",
            }}
          >
            {AXIS_FIELDS.map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2">
          <span style={{ color: "var(--color-text-muted)" }}>Y axis:</span>
          <select
            value={yField}
            onChange={(e) => setYField(e.target.value)}
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: "var(--color-border)",
              background: "var(--color-surface)",
              color: "var(--color-text)",
            }}
          >
            {AXIS_FIELDS.map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Mobile: current axis labels + button to open bottom sheet (<768px) */}
      <div className="flex md:hidden items-center justify-between mb-3 text-sm">
        <span style={{ color: "var(--color-text-muted)" }}>
          X: <strong>{xLabel}</strong> · Y: <strong>{yLabel}</strong>
        </span>
        <button
          onClick={() => setBottomSheetOpen(true)}
          className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm min-h-[44px]"
          style={{
            borderColor: "var(--color-border)",
            background: "var(--color-surface)",
            color: "var(--color-text)",
          }}
          aria-label="Change chart axes"
        >
          <SlidersHorizontal size={14} aria-hidden />
          Axes
        </button>
      </div>

      {/* Mobile: bottom sheet overlay */}
      {bottomSheetOpen && (
        <div
          className="fixed inset-0 z-50 md:hidden"
          onClick={() => setBottomSheetOpen(false)}
          aria-modal="true"
          role="dialog"
          aria-label="Axis selectors"
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/40" />

          {/* Sheet */}
          <div
            className="absolute bottom-0 left-0 right-0 rounded-t-2xl p-6 space-y-5"
            style={{
              background: "var(--color-surface)",
              boxShadow: "0 -4px 24px rgba(0,0,0,0.15)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Handle */}
            <div className="mx-auto w-10 h-1 rounded-full bg-[var(--color-border)]" />

            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-base" style={{ color: "var(--color-text)" }}>
                Chart Axes
              </h3>
              <button
                onClick={() => setBottomSheetOpen(false)}
                className="p-2 min-h-[44px] min-w-[44px] flex items-center justify-center rounded-md"
                style={{ color: "var(--color-text-muted)" }}
                aria-label="Close axis selector"
              >
                <X size={18} aria-hidden />
              </button>
            </div>

            <label className="block space-y-1.5">
              <span className="text-sm font-medium" style={{ color: "var(--color-text-muted)" }}>
                X axis
              </span>
              <select
                value={xField}
                onChange={(e) => setXField(e.target.value)}
                className="block w-full rounded border px-3 py-2.5 text-sm min-h-[44px]"
                style={{
                  borderColor: "var(--color-border)",
                  background: "var(--color-bg)",
                  color: "var(--color-text)",
                }}
              >
                {AXIS_FIELDS.map((f) => (
                  <option key={f.key} value={f.key}>
                    {f.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="block space-y-1.5">
              <span className="text-sm font-medium" style={{ color: "var(--color-text-muted)" }}>
                Y axis
              </span>
              <select
                value={yField}
                onChange={(e) => setYField(e.target.value)}
                className="block w-full rounded border px-3 py-2.5 text-sm min-h-[44px]"
                style={{
                  borderColor: "var(--color-border)",
                  background: "var(--color-bg)",
                  color: "var(--color-text)",
                }}
              >
                {AXIS_FIELDS.map((f) => (
                  <option key={f.key} value={f.key}>
                    {f.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              onClick={() => setBottomSheetOpen(false)}
              className="w-full rounded-md py-3 text-sm font-semibold min-h-[44px]"
              style={{
                background: "var(--color-text)",
                color: "var(--color-bg)",
              }}
            >
              Done
            </button>
          </div>
        </div>
      )}

      {/* SVG plot */}
      <div style={{ position: "relative" }}>
        <svg width={width} height={height} style={{ display: "block", maxWidth: "100%" }}>
          <Group left={MARGIN.left} top={MARGIN.top}>
            {/* Grid lines */}
            {xScale &&
              xScale.ticks(5).map((tick) => (
                <line
                  key={`xgrid-${tick}`}
                  x1={xScale(tick)}
                  x2={xScale(tick)}
                  y1={0}
                  y2={innerHeight}
                  stroke="var(--color-border)"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
              ))}
            {yScale &&
              yScale.ticks(5).map((tick) => (
                <line
                  key={`ygrid-${tick}`}
                  x1={0}
                  x2={innerWidth}
                  y1={yScale(tick)}
                  y2={yScale(tick)}
                  stroke="var(--color-border)"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
              ))}

            {/* Data points */}
            {xScale &&
              yScale &&
              plotPoints.map(({ p, xv, yv }) => {
                const cx = xScale(xv);
                const cy = yScale(yv);
                const color = rgbToHex(getSuperTypeColor(p.super_type_id));
                // Radius: base 5, scaled slightly by county count (log)
                const r = Math.max(4, Math.min(10, 4 + Math.log1p(p.n_counties) * 0.5));
                return (
                  <circle
                    key={p.type_id}
                    cx={cx}
                    cy={cy}
                    r={r}
                    fill={color}
                    fillOpacity={0.75}
                    stroke={color}
                    strokeWidth={1}
                    style={{ cursor: "pointer", transition: "r 0.1s" }}
                    onMouseEnter={() => handleMouseEnter({ p, xv, yv }, cx, cy)}
                    onMouseLeave={hideTooltip}
                    onTouchStart={() => handleTouchStart({ p, xv, yv }, cx, cy)}
                    onTouchEnd={handleTouchEnd}
                    onTouchMove={handleTouchEnd}
                  />
                );
              })}

            {/* Axes */}
            {xScale && (
              <AxisBottom
                top={innerHeight}
                scale={xScale}
                numTicks={5}
                tickFormat={(v) => formatAxisValue(xField, Number(v))}
                stroke="var(--color-border)"
                tickStroke="var(--color-border)"
                tickLabelProps={{ fontSize: 11, fill: "var(--color-text-muted)" }}
                label={xLabel}
                labelProps={{
                  fontSize: 12,
                  fill: "var(--color-text-muted)",
                  textAnchor: "middle",
                  dy: "2.5em",
                }}
              />
            )}
            {yScale && (
              <AxisLeft
                scale={yScale}
                numTicks={5}
                tickFormat={(v) => formatAxisValue(yField, Number(v))}
                stroke="var(--color-border)"
                tickStroke="var(--color-border)"
                tickLabelProps={{ fontSize: 11, fill: "var(--color-text-muted)" }}
                label={yLabel}
                labelProps={{
                  fontSize: 12,
                  fill: "var(--color-text-muted)",
                  textAnchor: "middle",
                  dy: "-2.5em",
                }}
              />
            )}
          </Group>
        </svg>

        {/* Hover tooltip */}
        {tooltipData && (
          <TooltipWithBounds
            left={tooltipLeft}
            top={tooltipTop}
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: 6,
              padding: "8px 12px",
              fontSize: 13,
              color: "var(--color-text)",
              pointerEvents: "none",
              zIndex: 50,
              boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            }}
          >
            <div className="font-semibold mb-1">
              {tooltipData.point.display_name}
            </div>
            <div style={{ color: "var(--color-text-muted)" }}>
              {xLabel}: {formatAxisValue(xField, tooltipData.x)}
            </div>
            <div style={{ color: "var(--color-text-muted)" }}>
              {yLabel}: {formatAxisValue(yField, tooltipData.y)}
            </div>
            <div style={{ color: "var(--color-text-muted)" }}>
              {tooltipData.point.n_counties} counties
            </div>
          </TooltipWithBounds>
        )}
      </div>

      {/* Legend */}
      <div className="mt-2 text-xs" style={{ color: "var(--color-text-muted)" }}>
        Dot size proportional to county count. Colors indicate super-type family.
      </div>
    </div>
  );
}
