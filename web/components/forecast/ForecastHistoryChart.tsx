"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { scaleLinear, scaleTime } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { curveMonotoneX } from "@visx/curve";
import { useTooltip, TooltipWithBounds, defaultStyles } from "@visx/tooltip";
import { localPoint } from "@visx/event";
import { useRaceHistory } from "@/lib/hooks/use-race-history";
import { PALETTE } from "@/lib/config/palette";
import type { RaceMarginPoint } from "@/lib/api";

// ── Design tokens ──────────────────────────────────────────────────────────

const TICK_COLOR = "var(--color-text-muted, #888)";
const GRID_COLOR = "var(--color-border, #e2e8f0)";
const MARGIN = { top: 16, right: 16, bottom: 40, left: 52 };

// ── Helpers ────────────────────────────────────────────────────────────────

function parseDate(d: string): Date {
  return new Date(d + "T00:00:00");
}

function formatDate(d: Date): string {
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatMarginPp(margin: number): string {
  const pp = margin * 100;
  const sign = pp >= 0 ? "+" : "";
  return `${sign}${pp.toFixed(1)} pp`;
}

function lineColor(latestMargin: number): string {
  if (latestMargin > 0) return PALETTE.DEM_PRIMARY;
  if (latestMargin < 0) return PALETTE.GOP_PRIMARY;
  return PALETTE.TOSSUP;
}

// ── Types ──────────────────────────────────────────────────────────────────

interface ParsedPoint {
  date: Date;
  margin: number;
}

interface TooltipData {
  point: ParsedPoint;
}

interface ForecastHistoryChartProps {
  /** Race slug used to filter margin history from the shared hook. */
  slug: string;
  /** Chart SVG width in px. Defaults to 480. */
  width?: number;
}

// ── Component ──────────────────────────────────────────────────────────────

/**
 * Full-size forecast history chart for a senate race detail page.
 *
 * Shows the race's dem margin (dem_share − 0.5) over snapshot dates,
 * with a zero reference line, labeled axes, and an interactive tooltip.
 *
 * Data comes from the shared useRaceHistory hook; only the series matching
 * `slug` is rendered.
 */
export function ForecastHistoryChart({ slug, width = 480 }: ForecastHistoryChartProps) {
  const { historyBySlug, isLoading, error } = useRaceHistory();

  const height = Math.max(180, Math.round(width * 0.38));
  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  const { showTooltip, hideTooltip, tooltipData, tooltipLeft, tooltipTop, tooltipOpen } =
    useTooltip<TooltipData>();

  const { parsedPoints, xScale, yScale, color } = useMemo(() => {
    const rawHistory: RaceMarginPoint[] = historyBySlug.get(slug) ?? [];

    if (rawHistory.length < 2) {
      return { parsedPoints: [], xScale: null, yScale: null, color: PALETTE.TOSSUP };
    }

    const pts: ParsedPoint[] = rawHistory.map((d) => ({
      date: parseDate(d.date),
      margin: d.margin,
    }));

    const allDates = pts.map((p) => p.date.getTime());
    const allMargins = pts.map((p) => p.margin);

    const minDate = new Date(Math.min(...allDates));
    const maxDate = new Date(Math.max(...allDates));
    minDate.setDate(minDate.getDate() - 3);
    maxDate.setDate(maxDate.getDate() + 3);

    const maxAbs = Math.max(
      Math.abs(Math.min(...allMargins)),
      Math.abs(Math.max(...allMargins)),
      0.02,
    );
    const yExtent = maxAbs * 1.25;

    return {
      parsedPoints: pts,
      xScale: scaleTime({ domain: [minDate, maxDate], range: [0, innerWidth] }),
      yScale: scaleLinear({ domain: [-yExtent, yExtent], range: [innerHeight, 0] }),
      color: lineColor(pts[pts.length - 1].margin),
    };
  }, [historyBySlug, slug, innerWidth, innerHeight]);

  // ── Loading state ─────────────────────────────────────────────────────────

  if (isLoading) {
    return (
      <div
        className="w-full rounded-md animate-pulse"
        style={{
          height,
          background: "var(--color-surface, #f8f9fa)",
          border: "1px solid var(--color-border, #e2e8f0)",
        }}
      />
    );
  }

  // ── Error state ───────────────────────────────────────────────────────────

  if (error) {
    return (
      <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
        Could not load forecast history.
      </p>
    );
  }

  // ── Empty state (< 2 snapshots) ───────────────────────────────────────────

  if (parsedPoints.length < 2 || !xScale || !yScale) {
    return (
      <p
        className="text-sm rounded-md px-4 py-3"
        style={{
          color: "var(--color-text-muted)",
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
        }}
      >
        Forecast history requires at least two snapshots.
      </p>
    );
  }

  // ── Zero reference line y coordinate ─────────────────────────────────────

  const zeroY = yScale(0);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="relative">
      <svg
        width={width}
        height={height}
        aria-label="Forecast margin history"
        role="img"
        style={{ overflow: "visible" }}
      >
        <Group top={MARGIN.top} left={MARGIN.left}>
          {/* Horizontal grid lines */}
          {yScale.ticks(5).map((tick) => (
            <line
              key={tick}
              x1={0}
              x2={innerWidth}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke={GRID_COLOR}
              strokeWidth={1}
              strokeDasharray="3,3"
            />
          ))}

          {/* Zero reference line (50/50 divider) */}
          <line
            x1={0}
            x2={innerWidth}
            y1={zeroY}
            y2={zeroY}
            stroke={GRID_COLOR}
            strokeWidth={1.5}
          />

          {/* Margin trend line */}
          <LinePath
            data={parsedPoints.map((p) => ({
              x: xScale(p.date),
              y: yScale(p.margin),
            }))}
            x={(d) => d.x}
            y={(d) => d.y}
            stroke={color}
            strokeWidth={2.5}
            curve={curveMonotoneX}
            strokeLinecap="round"
          />

          {/* Interactive hover dots */}
          {parsedPoints.map((p, i) => (
            <circle
              key={i}
              cx={xScale(p.date)}
              cy={yScale(p.margin)}
              r={4}
              fill={color}
              opacity={0}
              style={{ cursor: "crosshair" }}
              onMouseMove={(e) => {
                const coords = localPoint(e) ?? { x: 0, y: 0 };
                showTooltip({
                  tooltipData: { point: p },
                  tooltipLeft: coords.x + MARGIN.left,
                  tooltipTop: coords.y + MARGIN.top,
                });
              }}
              onMouseLeave={hideTooltip}
            >
              {/* Enlarged hit area for easier hover */}
              <title>{`${formatDate(p.date)}: ${formatMarginPp(p.margin)}`}</title>
            </circle>
          ))}

          {/* Terminal dot — latest snapshot */}
          <circle
            cx={xScale(parsedPoints[parsedPoints.length - 1].date)}
            cy={yScale(parsedPoints[parsedPoints.length - 1].margin)}
            r={3.5}
            fill={color}
            style={{ pointerEvents: "none" }}
          />

          {/* Axes */}
          <AxisBottom
            top={innerHeight}
            scale={xScale}
            numTicks={Math.min(6, Math.floor(innerWidth / 80))}
            tickFormat={(d) => formatDate(d as Date)}
            stroke={GRID_COLOR}
            tickStroke={GRID_COLOR}
            tickLabelProps={{ fill: TICK_COLOR, fontSize: 11, textAnchor: "middle" }}
          />
          <AxisLeft
            scale={yScale}
            numTicks={5}
            tickFormat={(v) => {
              const pp = (v as number) * 100;
              const sign = pp >= 0 ? "+" : "";
              return `${sign}${pp.toFixed(0)}`;
            }}
            stroke={GRID_COLOR}
            tickStroke={GRID_COLOR}
            tickLabelProps={{ fill: TICK_COLOR, fontSize: 11, textAnchor: "end", dx: -4 }}
          />
        </Group>
      </svg>

      {/* Legend */}
      <div className="flex gap-4 mt-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
        <span style={{ color, fontWeight: 600 }}>— Dem margin</span>
        <span>above zero = Dem-favored · below zero = GOP-favored</span>
      </div>

      {/* Tooltip */}
      {tooltipOpen && tooltipData && (
        <TooltipWithBounds
          top={tooltipTop}
          left={tooltipLeft}
          style={{
            ...defaultStyles,
            background: "var(--color-surface, #fff)",
            border: "1px solid var(--color-border, #e2e8f0)",
            color: "var(--color-text, #1a1a1a)",
            fontSize: 12,
            padding: "8px 10px",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 2 }}>
            {formatDate(tooltipData.point.date)}
          </div>
          <div style={{ color }}>
            Dem margin: {formatMarginPp(tooltipData.point.margin)}
          </div>
        </TooltipWithBounds>
      )}
    </div>
  );
}
