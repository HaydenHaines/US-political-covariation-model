"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { scaleLinear, scaleTime } from "@visx/scale";
import { curveMonotoneX } from "@visx/curve";
import { PALETTE } from "@/lib/config/palette";
import type { RaceMarginPoint } from "@/lib/api";

// Design tokens

/** Color for a Dem-leaning race (latest margin > 0). */
const DEM_COLOR = PALETTE.DEM_SECONDARY;
/** Color for a GOP-leaning race (latest margin < 0). */
const GOP_COLOR = PALETTE.GOP_SECONDARY;
/** Color when race is exactly at 50/50. */
const NEUTRAL_COLOR = PALETTE.TOSSUP;

// Constant padding -- zero margins so the sparkline fills its SVG bounds.
const PADDING = { top: 3, right: 2, bottom: 3, left: 2 };

// Helpers

function parseDate(d: string): Date {
  // Append midnight UTC so all dates parse consistently regardless of timezone.
  return new Date(d + "T00:00:00");
}

/**
 * Pick the line color based on the latest margin in the series.
 * Returns DEM_COLOR for positive, GOP_COLOR for negative, NEUTRAL_COLOR for zero.
 */
function colorForMargin(margin: number): string {
  if (margin > 0) return DEM_COLOR;
  if (margin < 0) return GOP_COLOR;
  return NEUTRAL_COLOR;
}

// Component

interface SparklineChartProps {
  /** Chronologically ordered margin history for a single race. */
  history: RaceMarginPoint[];
  /** Total SVG width in px. Defaults to 120. */
  width?: number;
  /** Total SVG height in px. Defaults to 30. */
  height?: number;
  /** Accessible label for screen readers. */
  ariaLabel?: string;
}

/**
 * Inline sparkline showing a race's margin movement over time.
 *
 * Designed for use inside race cards on the senate overview page.
 * Intentionally minimal -- no axes, no tick labels, no legend.
 *
 * Graceful degradation:
 *   - 0 points: renders nothing (null)
 *   - 1 point:  renders a single centered dot
 *   - 2+ points: renders the full line + endpoint dot
 *
 * Color follows the latest margin: blue = Dem-favored, terracotta = GOP-favored.
 */
export function SparklineChart({
  history,
  width = 120,
  height = 30,
  ariaLabel,
}: SparklineChartProps) {
  const innerWidth = width - PADDING.left - PADDING.right;
  const innerHeight = height - PADDING.top - PADDING.bottom;

  const { xScale, yScale, parsedPoints, lineColor } = useMemo(() => {
    if (!history || history.length === 0) {
      return { xScale: null, yScale: null, parsedPoints: [], lineColor: NEUTRAL_COLOR };
    }

    const points = history.map((d) => ({
      date: parseDate(d.date),
      margin: d.margin,
    }));

    const latestMargin = points[points.length - 1].margin;
    const color = colorForMargin(latestMargin);

    if (points.length === 1) {
      // Single point: no scales needed for line drawing, just center the dot.
      return { xScale: null, yScale: null, parsedPoints: points, lineColor: color };
    }

    const allDates = points.map((p) => p.date.getTime());
    const allMargins = points.map((p) => p.margin);

    // Symmetric y-domain: center on 0 with equal padding on each side.
    // This ensures the zero line (D/R divider) is always visible at mid-height.
    const maxAbs = Math.max(Math.abs(Math.min(...allMargins)), Math.abs(Math.max(...allMargins)), 0.01);
    // Add 20% headroom so data points are not clipped at the edges.
    const yExtent = maxAbs * 1.2;

    return {
      xScale: scaleTime({
        domain: [Math.min(...allDates), Math.max(...allDates)],
        range: [0, innerWidth],
      }),
      yScale: scaleLinear({
        domain: [-yExtent, yExtent],
        range: [innerHeight, 0],
      }),
      parsedPoints: points,
      lineColor: color,
    };
  }, [history, innerWidth, innerHeight]);

  // No data -- render nothing
  if (parsedPoints.length === 0) return null;

  // Single data point -- centered dot only
  if (parsedPoints.length === 1) {
    return (
      <svg
        width={width}
        height={height}
        aria-label={ariaLabel ?? "Race margin"}
        role="img"
        style={{ display: "block", overflow: "visible" }}
      >
        <circle
          cx={width / 2}
          cy={height / 2}
          r={3}
          fill={lineColor}
          opacity={0.85}
        />
      </svg>
    );
  }

  if (!xScale || !yScale) return null;

  // Zero-margin reference line y-coordinate
  const zeroY = yScale(0);
  const lastPoint = parsedPoints[parsedPoints.length - 1];

  return (
    <svg
      width={width}
      height={height}
      aria-label={ariaLabel ?? "Race margin trend"}
      role="img"
      style={{ display: "block", overflow: "visible" }}
    >
      <Group top={PADDING.top} left={PADDING.left}>
        {/* Zero-margin reference line -- subtle dashed divider */}
        <line
          x1={0}
          x2={innerWidth}
          y1={zeroY}
          y2={zeroY}
          stroke="var(--color-border, #e0ddd8)"
          strokeWidth={0.75}
          strokeDasharray="2,2"
        />

        {/* Margin trend line */}
        <LinePath
          data={parsedPoints.map((p) => ({
            x: xScale(p.date.getTime()),
            y: yScale(p.margin),
          }))}
          x={(d) => d.x}
          y={(d) => d.y}
          stroke={lineColor}
          strokeWidth={1.75}
          curve={curveMonotoneX}
          strokeLinecap="round"
        />

        {/* Terminal dot -- latest data point */}
        <circle
          cx={xScale(lastPoint.date.getTime())}
          cy={yScale(lastPoint.margin)}
          r={2.5}
          fill={lineColor}
        />
      </Group>
    </svg>
  );
}
