"use client";

import { useMemo } from "react";
import { ParentSize } from "@visx/responsive";
import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { scaleLinear, scaleTime } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { curveMonotoneX } from "@visx/curve";
import { useSeatHistory } from "@/lib/hooks/use-seat-history";
import { PALETTE } from "@/lib/config/palette";

// ── Design tokens ──────────────────────────────────────────────────────────

const TICK_COLOR = "var(--color-text-muted, #888)";
const GRID_COLOR = "var(--color-border, #e0ddd8)";
const MAJORITY_COLOR = "var(--color-text-subtle, #8a8478)";

const MARGIN = { top: 16, right: 24, bottom: 40, left: 44 };

// Senate has 100 seats; the y-axis range covers the plausible outcome space.
// Most forecasts will land in [44, 56] for the competitive party.
const Y_DOMAIN_MIN = 44;
const Y_DOMAIN_MAX = 56;
const MAJORITY_THRESHOLD = 50;

// ── Helpers ────────────────────────────────────────────────────────────────

function parseDate(d: string): Date {
  // Append midnight UTC so all dates parse consistently regardless of timezone.
  return new Date(d + "T00:00:00");
}

function formatAxisDate(d: Date): string {
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ── Component ──────────────────────────────────────────────────────────────

/**
 * Senate seat balance timeline.
 *
 * Displays a line chart showing how projected D and R seat counts change over
 * time as new polls arrive.  Fetches data from GET /api/v1/forecast/seat-history
 * via SWR.
 *
 * Designed to sit below the BalanceBar on the senate overview page.
 */
export function SeatBalanceTimeline() {
  return (
    <ParentSize>
      {({ width }) => <SeatBalanceTimelineInner width={Math.max(width, 200)} />}
    </ParentSize>
  );
}

function SeatBalanceTimelineInner({ width }: { width: number }) {
  const { data, error, isLoading } = useSeatHistory();

  const height = Math.max(180, Math.round(width * 0.35));
  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  // Parse dates and compute scales from the snapshot data.
  const { xScale, yScale, parsedData } = useMemo(() => {
    if (!data || data.length < 2) {
      return { xScale: null, yScale: null, parsedData: [] };
    }

    const parsed = data.map((d) => ({
      ...d,
      _date: parseDate(d.date),
    }));

    const allDates = parsed.map((d) => d._date);
    const minDate = new Date(Math.min(...allDates.map((d) => d.getTime())));
    const maxDate = new Date(Math.max(...allDates.map((d) => d.getTime())));
    // Add a 3-day pad on each side so edge points aren't clipped.
    minDate.setDate(minDate.getDate() - 3);
    maxDate.setDate(maxDate.getDate() + 3);

    // Widen the y-domain if any value falls outside the default range.
    const allSeats = parsed.flatMap((d) => [d.dem_projected, d.gop_projected]);
    const rawMin = Math.min(...allSeats);
    const rawMax = Math.max(...allSeats);
    const yMin = Math.min(Y_DOMAIN_MIN, rawMin - 1);
    const yMax = Math.max(Y_DOMAIN_MAX, rawMax + 1);

    return {
      xScale: scaleTime({ domain: [minDate, maxDate], range: [0, innerWidth] }),
      yScale: scaleLinear({ domain: [yMin, yMax], range: [innerHeight, 0] }),
      parsedData: parsed,
    };
  }, [data, innerWidth, innerHeight]);

  // ── Loading / error / empty states ───────────────────────────────────────

  if (isLoading) {
    return (
      <div
        className="w-full rounded-md animate-pulse"
        style={{
          height,
          background: "var(--color-surface, #f5f3ef)",
          border: "1px solid var(--color-border, #e0ddd8)",
        }}
      />
    );
  }

  if (error) {
    return null; // Fail silently — this is a supplementary chart
  }

  // Need at least two points to draw a line.
  if (!data || data.length < 2 || !xScale || !yScale) {
    return null;
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const majorityY = yScale(MAJORITY_THRESHOLD);

  return (
    <div>
      <p
        className="text-sm font-medium mb-2"
        style={{ color: "var(--color-text-muted)" }}
      >
        Projected seats over time
      </p>

      <svg
        width={width}
        height={height}
        aria-label="Senate seat balance timeline"
        role="img"
        style={{ overflow: "visible" }}
      >
        <Group top={MARGIN.top} left={MARGIN.left}>
          {/* Horizontal grid lines */}
          {yScale.ticks(4).map((tick) => (
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

          {/* 50-seat majority threshold */}
          <line
            x1={0}
            x2={innerWidth}
            y1={majorityY}
            y2={majorityY}
            stroke={MAJORITY_COLOR}
            strokeWidth={1.5}
            strokeDasharray="5,4"
          />
          <text
            x={innerWidth + 4}
            y={majorityY + 4}
            fontSize={10}
            fill={MAJORITY_COLOR}
            dominantBaseline="middle"
          >
            50
          </text>

          {/* Dem line */}
          <LinePath
            data={parsedData.map((d) => ({ x: xScale(d._date), y: yScale(d.dem_projected) }))}
            x={(d) => d.x}
            y={(d) => d.y}
            stroke={PALETTE.DEM_PRIMARY}
            strokeWidth={2.5}
            curve={curveMonotoneX}
            strokeLinecap="round"
          />

          {/* GOP line */}
          <LinePath
            data={parsedData.map((d) => ({ x: xScale(d._date), y: yScale(d.gop_projected) }))}
            x={(d) => d.x}
            y={(d) => d.y}
            stroke={PALETTE.GOP_PRIMARY}
            strokeWidth={2.5}
            curve={curveMonotoneX}
            strokeLinecap="round"
          />

          {/* Data points — small dots at each snapshot */}
          {parsedData.map((d, i) => (
            <g key={i}>
              <circle
                cx={xScale(d._date)}
                cy={yScale(d.dem_projected)}
                r={3.5}
                fill={PALETTE.DEM_PRIMARY}
              />
              <circle
                cx={xScale(d._date)}
                cy={yScale(d.gop_projected)}
                r={3.5}
                fill={PALETTE.GOP_PRIMARY}
              />
            </g>
          ))}

          {/* Axes */}
          <AxisBottom
            top={innerHeight}
            scale={xScale}
            numTicks={Math.min(data.length, Math.floor(innerWidth / 80))}
            tickFormat={(d) => formatAxisDate(d as Date)}
            stroke={GRID_COLOR}
            tickStroke={GRID_COLOR}
            tickLabelProps={{ fill: TICK_COLOR, fontSize: 11, textAnchor: "middle" }}
          />
          <AxisLeft
            scale={yScale}
            numTicks={4}
            tickFormat={(v) => String(v as number)}
            stroke={GRID_COLOR}
            tickStroke={GRID_COLOR}
            tickLabelProps={{ fill: TICK_COLOR, fontSize: 11, textAnchor: "end", dx: -4 }}
          />
        </Group>
      </svg>

      {/* Legend */}
      <div className="flex gap-4 mt-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
        <span style={{ color: PALETTE.DEM_PRIMARY, fontWeight: 600 }}>— Dem seats</span>
        <span style={{ color: PALETTE.GOP_PRIMARY, fontWeight: 600 }}>— GOP seats</span>
        <span style={{ color: MAJORITY_COLOR }}>· · majority threshold (50)</span>
      </div>
    </div>
  );
}
