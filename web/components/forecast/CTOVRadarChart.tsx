"use client";

/**
 * CTOVRadarChart — spider/radar chart of a candidate's party-relative CTOV badge scores.
 *
 * Each axis represents a badge dimension.  Distance from center = |score| / maxAbsScore,
 * so the chart always fills to the edge for the strongest dimension.  Positive and negative
 * scores are differentiated by the "+" or "-" prefix on the label — both plot outward on
 * their axis since we're showing *magnitude* of deviation (party-relative), not direction.
 *
 * Design: Dusty Ink color scheme, SVG-based, uses @visx/group for center transform.
 * No @visx/radar (doesn't exist) — polygon drawn manually from SVG path.
 */

import { Group } from "@visx/group";

/** Minimum |score| to include a dimension — filters out near-zero noise. */
const NOISE_THRESHOLD = 0.005;

/** Maximum dimensions to display on the radar. */
const MAX_DIMS = 8;

/** Minimum dimensions required to render anything. */
const MIN_DIMS = 3;

/** Radii for reference circles as fractions of the outer radius. */
const REFERENCE_CIRCLES = [0.33, 0.66, 1.0];

interface CTOVRadarChartProps {
  badgeScores: Record<string, number>;
  party: string;
  /** Outer diameter in pixels (default 120). */
  size?: number;
}

/**
 * Pick the top N dimensions by absolute score, filtering noise.
 *
 * Outlier values (|score| > 2) exist in the data for rarely-matched dimensions
 * like "Devout Community" and "Affluent Transplant" — these are data anomalies
 * not meaningful signal.  We cap at 2 to prevent them dominating the chart.
 */
function selectDimensions(
  scores: Record<string, number>,
): Array<{ name: string; score: number; absScore: number }> {
  return Object.entries(scores)
    .map(([name, raw]) => {
      // Cap outliers at ±2 — beyond that range the data is unreliable.
      const score = Math.max(-2, Math.min(2, raw));
      return { name, score, absScore: Math.abs(score) };
    })
    .filter((d) => d.absScore > NOISE_THRESHOLD)
    .sort((a, b) => b.absScore - a.absScore)
    .slice(0, MAX_DIMS);
}

/**
 * Compute the (x, y) endpoint for an axis at angle θ with given radius.
 *
 * SVG angles: 0 = right, positive = clockwise (standard SVG coordinate system).
 * We start at -π/2 (top) so the first axis points straight up.
 */
function polarToXY(angle: number, radius: number): { x: number; y: number } {
  return {
    x: radius * Math.cos(angle),
    y: radius * Math.sin(angle),
  };
}

/** Map party code to fill and stroke color. */
function partyColor(party: string): { fill: string; stroke: string } {
  if (party === "D") {
    return {
      fill: "rgba(var(--color-dem-rgb, 59, 130, 246), 0.15)",
      stroke: "var(--color-dem, #3b82f6)",
    };
  }
  if (party === "R") {
    return {
      fill: "rgba(var(--color-rep-rgb, 239, 68, 68), 0.15)",
      stroke: "var(--color-rep, #ef4444)",
    };
  }
  return {
    fill: "rgba(148, 163, 184, 0.15)",
    stroke: "rgba(148, 163, 184, 0.6)",
  };
}

/**
 * Truncate a dimension label to fit near the axis end.
 * Prioritizes first word (e.g. "Hispanic" from "Hispanic Appeal").
 */
function shortLabel(name: string): string {
  if (name.length <= 10) return name;
  const words = name.split(" ");
  // Two-word names: keep both if combined ≤ 14 chars, else first only
  if (words.length === 2 && words[0].length + words[1].length + 1 <= 14) {
    return name;
  }
  return words[0];
}

/**
 * Radar chart component.
 *
 * Returns null when there isn't enough data to draw a meaningful chart
 * (fewer than MIN_DIMS dimensions above the noise threshold).
 */
export function CTOVRadarChart({ badgeScores, party, size = 120 }: CTOVRadarChartProps) {
  const dims = selectDimensions(badgeScores);

  if (dims.length < MIN_DIMS) return null;

  const N = dims.length;
  const maxAbsScore = dims[0].absScore; // sorted descending, so first is max
  const cx = size / 2;
  const cy = size / 2;

  // Leave room for labels around the outside.
  const labelPadding = 18;
  const outerRadius = cx - labelPadding;

  // Angular step: evenly distribute N axes, starting at top (-π/2).
  const angleStep = (2 * Math.PI) / N;
  const startAngle = -Math.PI / 2;

  // Compute axis angles and polygon points.
  const axes = dims.map((dim, i) => {
    const angle = startAngle + i * angleStep;
    const normalizedMag = dim.absScore / maxAbsScore; // 0–1, where 1 = outerRadius
    const r = normalizedMag * outerRadius;
    const labelR = outerRadius + 10; // label sits just beyond the outer circle
    return {
      dim,
      angle,
      // Point on the polygon (centered at 0,0 since we use Group for center transform)
      px: r * Math.cos(angle),
      py: r * Math.sin(angle),
      // Outer axis endpoint
      ax: outerRadius * Math.cos(angle),
      ay: outerRadius * Math.sin(angle),
      // Label position
      lx: labelR * Math.cos(angle),
      ly: labelR * Math.sin(angle),
    };
  });

  // Build SVG polygon points string.
  const polygonPoints = axes.map((a) => `${a.px},${a.py}`).join(" ");

  // Build reference circle radii.
  const refCircles = REFERENCE_CIRCLES.map((frac) => outerRadius * frac);

  const { fill, stroke } = partyColor(party);

  return (
    <svg
      width={size}
      height={size}
      role="img"
      aria-label={`Radar chart of ${party} candidate badge scores`}
      style={{ display: "block", overflow: "visible" }}
    >
      {/* All drawing centered on (cx, cy) via Group transform */}
      <Group top={cy} left={cx}>
        {/* Reference circles */}
        {refCircles.map((r, i) => (
          <circle
            key={i}
            cx={0}
            cy={0}
            r={r}
            fill="none"
            stroke="rgba(148, 163, 184, 0.25)"
            strokeWidth={0.75}
          />
        ))}

        {/* Axis spokes: center → outer edge */}
        {axes.map(({ angle, ax, ay }, i) => (
          <line
            key={i}
            x1={0}
            y1={0}
            x2={ax}
            y2={ay}
            stroke="rgba(148, 163, 184, 0.25)"
            strokeWidth={0.75}
          />
        ))}

        {/* Score polygon */}
        <polygon
          points={polygonPoints}
          fill={fill}
          stroke={stroke}
          strokeWidth={1.25}
          strokeLinejoin="round"
        />

        {/* Axis labels */}
        {axes.map(({ dim, lx, ly, angle }, i) => {
          // Determine text-anchor based on horizontal position.
          const textAnchor =
            Math.abs(Math.cos(angle)) < 0.15
              ? "middle"
              : Math.cos(angle) > 0
              ? "start"
              : "end";

          // Shift label vertically: top labels shift up, bottom down.
          const dominantBaseline =
            Math.abs(Math.sin(angle)) < 0.15
              ? "middle"
              : Math.sin(angle) > 0
              ? "hanging"
              : "auto";

          const prefix = dim.score >= 0 ? "+" : "-";
          const label = `${prefix}${shortLabel(dim.name)}`;

          return (
            <text
              key={i}
              x={lx}
              y={ly}
              textAnchor={textAnchor}
              dominantBaseline={dominantBaseline}
              style={{
                fontSize: "0.5rem",
                fill: "var(--color-text-muted, rgb(100, 116, 139))",
                userSelect: "none",
              }}
            >
              {label}
            </text>
          );
        })}

        {/* Center dot */}
        <circle cx={0} cy={0} r={1.5} fill="rgba(148, 163, 184, 0.5)" />
      </Group>
    </svg>
  );
}
