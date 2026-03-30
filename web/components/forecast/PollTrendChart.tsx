"use client";

import { useCallback, useMemo, useState } from "react";
import { Group } from "@visx/group";
import { LinePath } from "@visx/shape";
import { scaleLinear, scaleTime } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { curveMonotoneX } from "@visx/curve";
import { useTooltip, TooltipWithBounds, defaultStyles } from "@visx/tooltip";
import { localPoint } from "@visx/event";
import { usePollTrend } from "@/lib/hooks/use-poll-trend";
import type { PollTrendPoll } from "@/lib/types";

// ── Design tokens ──────────────────────────────────────────────────────────

const DEM_COLOR = "#4b6d90";
const GOP_COLOR = "#9e5e4e";
const DEM_TREND_COLOR = "#2166ac";
const GOP_TREND_COLOR = "#c0392b";
const TICK_COLOR = "var(--color-text-muted, #888)";
const GRID_COLOR = "var(--color-border, #e2e8f0)";

const MARGIN = { top: 16, right: 16, bottom: 40, left: 48 };

// ── Types ──────────────────────────────────────────────────────────────────

interface TooltipData {
  poll: PollTrendPoll;
  x: number;
  y: number;
}

interface PollTrendChartProps {
  slug: string;
  width?: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function parseDate(d: string): Date {
  return new Date(d + "T00:00:00");
}

function formatDate(d: Date): string {
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

// ── Component ──────────────────────────────────────────────────────────────

/**
 * Poll trend chart for a race detail page.
 *
 * Shows individual poll dots (sized by sample_size) and a 30-day
 * weighted-moving-average trend line for each party.
 *
 * Fetches data via SWR from GET /api/v1/forecast/race/{slug}/poll-trend.
 * Renders "No polls available" when the race has no polls.
 */
export function PollTrendChart({ slug, width = 560 }: PollTrendChartProps) {
  const { data, error, isLoading } = usePollTrend(slug);

  const height = Math.max(220, Math.round(width * 0.4));
  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  const { showTooltip, hideTooltip, tooltipData, tooltipLeft, tooltipTop, tooltipOpen } =
    useTooltip<TooltipData>();

  // Parse poll dates once
  const pollsWithDates = useMemo(() => {
    if (!data?.polls.length) return [];
    return data.polls
      .filter((p) => p.date)
      .map((p) => ({ ...p, _date: parseDate(p.date) }));
  }, [data]);

  const trendWithDates = useMemo(() => {
    if (!data?.trend) return null;
    return {
      dates: data.trend.dates.map(parseDate),
      dem: data.trend.dem_trend,
      rep: data.trend.rep_trend,
    };
  }, [data]);

  // Compute scales
  const { xScale, yScale } = useMemo(() => {
    if (!pollsWithDates.length) return { xScale: null, yScale: null };

    const allDates = pollsWithDates.map((p) => p._date);
    if (trendWithDates) allDates.push(...trendWithDates.dates);

    const minDate = new Date(Math.min(...allDates.map((d) => d.getTime())));
    const maxDate = new Date(Math.max(...allDates.map((d) => d.getTime())));
    // Add a small date padding so edge dots aren't clipped
    minDate.setDate(minDate.getDate() - 7);
    maxDate.setDate(maxDate.getDate() + 7);

    const allShares = pollsWithDates.flatMap((p) => [p.dem_share, p.rep_share ?? 1 - p.dem_share]);
    const rawMin = Math.min(...allShares);
    const rawMax = Math.max(...allShares);
    // Pad by 3pp, clamp to [0.25, 0.75]
    const yMin = Math.max(0.25, rawMin - 0.03);
    const yMax = Math.min(0.75, rawMax + 0.03);

    return {
      xScale: scaleTime({ domain: [minDate, maxDate], range: [0, innerWidth] }),
      yScale: scaleLinear({ domain: [yMin, yMax], range: [innerHeight, 0] }),
    };
  }, [pollsWithDates, trendWithDates, innerWidth, innerHeight]);

  // Dot radius proportional to sample size (log scale), clamped to [3, 9]
  const dotRadius = useCallback((sampleSize: number | null): number => {
    if (!sampleSize || sampleSize <= 0) return 4;
    const r = 2 + Math.log10(sampleSize) * 2;
    return Math.min(9, Math.max(3, r));
  }, []);

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<SVGCircleElement>, poll: PollTrendPoll) => {
      const coords = localPoint(event) ?? { x: 0, y: 0 };
      showTooltip({
        tooltipData: { poll, x: coords.x, y: coords.y },
        tooltipLeft: coords.x,
        tooltipTop: coords.y,
      });
    },
    [showTooltip],
  );

  // ── Loading / error / empty states ───────────────────────────────────────

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

  if (error) {
    return (
      <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
        Could not load poll data.
      </p>
    );
  }

  if (!data || data.polls.length === 0) {
    return (
      <p
        className="text-sm rounded-md px-4 py-3"
        style={{
          color: "var(--color-text-muted)",
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
        }}
      >
        No polls available yet for this race.
      </p>
    );
  }

  if (!xScale || !yScale) return null;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="relative">
      <svg
        width={width}
        height={height}
        aria-label="Poll trend chart"
        role="img"
        style={{ overflow: "visible" }}
      >
        <Group top={MARGIN.top} left={MARGIN.left}>
          {/* Grid lines at y-axis ticks */}
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

          {/* 50% reference line */}
          {yScale.domain()[0] <= 0.5 && yScale.domain()[1] >= 0.5 && (
            <line
              x1={0}
              x2={innerWidth}
              y1={yScale(0.5)}
              y2={yScale(0.5)}
              stroke={GRID_COLOR}
              strokeWidth={1.5}
            />
          )}

          {/* Dem trend line */}
          {trendWithDates && (
            <LinePath
              data={trendWithDates.dates.map((d, i) => ({ x: xScale(d), y: yScale(trendWithDates.dem[i]) }))}
              x={(d) => d.x}
              y={(d) => d.y}
              stroke={DEM_TREND_COLOR}
              strokeWidth={2.5}
              curve={curveMonotoneX}
              strokeLinecap="round"
            />
          )}

          {/* Rep trend line */}
          {trendWithDates && (
            <LinePath
              data={trendWithDates.dates.map((d, i) => ({ x: xScale(d), y: yScale(trendWithDates.rep[i]) }))}
              x={(d) => d.x}
              y={(d) => d.y}
              stroke={GOP_TREND_COLOR}
              strokeWidth={2.5}
              curve={curveMonotoneX}
              strokeLinecap="round"
            />
          )}

          {/* Poll dots — Dem share */}
          {pollsWithDates.map((poll, i) => (
            <circle
              key={`dem-${i}`}
              cx={xScale(poll._date)}
              cy={yScale(poll.dem_share)}
              r={dotRadius(poll.sample_size)}
              fill={DEM_COLOR}
              opacity={0.7}
              style={{ cursor: "pointer" }}
              onMouseMove={(e) => handleMouseMove(e, poll)}
              onMouseLeave={hideTooltip}
            />
          ))}

          {/* Poll dots — Rep share */}
          {pollsWithDates.map((poll, i) => {
            const repShare = poll.rep_share ?? 1 - poll.dem_share;
            return (
              <circle
                key={`rep-${i}`}
                cx={xScale(poll._date)}
                cy={yScale(repShare)}
                r={dotRadius(poll.sample_size)}
                fill={GOP_COLOR}
                opacity={0.7}
                style={{ cursor: "pointer" }}
                onMouseMove={(e) => handleMouseMove(e, { ...poll, dem_share: repShare })}
                onMouseLeave={hideTooltip}
              />
            );
          })}

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
            tickFormat={(v) => `${((v as number) * 100).toFixed(0)}%`}
            stroke={GRID_COLOR}
            tickStroke={GRID_COLOR}
            tickLabelProps={{ fill: TICK_COLOR, fontSize: 11, textAnchor: "end", dx: -4 }}
          />
        </Group>
      </svg>

      {/* Legend */}
      <div className="flex gap-4 mt-1 text-xs" style={{ color: "var(--color-text-muted)" }}>
        <span style={{ color: DEM_TREND_COLOR, fontWeight: 600 }}>— Dem trend</span>
        <span style={{ color: GOP_TREND_COLOR, fontWeight: 600 }}>— Rep trend</span>
        <span>· dots = individual polls (size = sample size)</span>
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
            {tooltipData.poll.pollster ?? "Unknown pollster"}
          </div>
          <div>{tooltipData.poll.date}</div>
          <div style={{ color: DEM_TREND_COLOR }}>Dem: {formatPct(tooltipData.poll.dem_share)}</div>
          {tooltipData.poll.rep_share !== null && tooltipData.poll.rep_share !== undefined && (
            <div style={{ color: GOP_TREND_COLOR }}>Rep: {formatPct(tooltipData.poll.rep_share)}</div>
          )}
          {tooltipData.poll.sample_size !== null && tooltipData.poll.sample_size !== undefined && (
            <div>n = {tooltipData.poll.sample_size.toLocaleString()}</div>
          )}
        </TooltipWithBounds>
      )}
    </div>
  );
}
