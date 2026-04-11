"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { scaleLinear } from "@visx/scale";
import { useMediaQuery } from "@/lib/hooks/use-media-query";
import { PALETTE } from "@/lib/config/palette";

// Normal distribution quantile (probit) via rational approximation.
// Accurate to ~1e-5 for p in (0.0001, 0.9999).
function normalQuantile(p: number): number {
  // Beasley-Springer-Moro algorithm
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e1, 2.209460984245205e2,
    -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2,
    -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1,
    -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1,
    2.445134137142996, 3.754408661907416,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number;
  if (p < pLow) {
    const s = Math.sqrt(-2 * Math.log(p));
    q = (((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) /
        ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1);
  } else if (p <= pHigh) {
    const r = p - 0.5;
    const s = r * r;
    q = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r /
        (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
  } else {
    const s = Math.sqrt(-2 * Math.log(1 - p));
    q = -(((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) /
         ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1);
  }
  return q;
}

interface QuantileDotplotProps {
  /** Model predicted dem share (0-1). */
  predDemShare: number;
  /** Standard deviation from model (0-1 scale). Uses 0.08 default if null. */
  predStd?: number | null;
  /** Number of dots (default 100). */
  nDots?: number;
  width?: number;
  height?: number;
}

const DOT_RADIUS = 5;
const COLS = 20;

/**
 * Quantile dotplot showing model-predicted outcome distribution.
 *
 * Desktop (≥768px): 100 dots. Mobile (<768px): 50 dots to reduce visual density.
 * Each dot is one scenario (percentile); blue = Democrat wins (>0.5),
 * red = Republican wins (<0.5).
 */
export function QuantileDotplot({
  predDemShare,
  predStd,
  nDots = 100,
  width = 400,
  height = 160,
}: QuantileDotplotProps) {
  const isMobile = useMediaQuery("(max-width: 767px)");
  // Reduce dot count on mobile to keep chart readable at smaller widths
  const effectiveNDots = isMobile ? Math.min(nDots, 50) : nDots;
  const std = predStd ?? 0.08;

  const { quantiles, nDemWins } = useMemo(() => {
    const vals: number[] = [];
    for (let i = 1; i <= effectiveNDots; i++) {
      const p = i / (effectiveNDots + 1);
      vals.push(predDemShare + std * normalQuantile(p));
    }
    const nDem = vals.filter((v) => v > 0.5).length;
    return { quantiles: vals, nDemWins: nDem };
  }, [predDemShare, std, effectiveNDots]);

  // Layout: COLS columns, stacking rows from bottom up
  const rows = Math.ceil(effectiveNDots / COLS);
  const padding = { top: 12, right: 12, bottom: 12, left: 12 };
  const dotSpacingX = (width - padding.left - padding.right) / COLS;
  const dotSpacingY = (height - padding.top - padding.bottom) / rows;

  // Sort quantiles low-to-high so we can fill left-to-right, bottom-to-top
  const sorted = [...quantiles].sort((a, b) => a - b);

  const xScale = scaleLinear({
    domain: [0, COLS],
    range: [padding.left, width - padding.right],
  });

  return (
    <div>
      <svg
        width={width}
        height={height}
        aria-label={`Quantile dotplot: Democrat wins in ${nDemWins} of ${effectiveNDots} scenarios`}
        role="img"
      >
        <Group>
          {sorted.map((val, i) => {
            const col = i % COLS;
            const row = Math.floor(i / COLS);
            // Bottom-to-top: row 0 is at the bottom
            const cx = xScale(col + 0.5);
            const cy = height - padding.bottom - row * dotSpacingY - dotSpacingY / 2;
            const isDem = val > 0.5;
            return (
              <circle
                key={i}
                cx={cx}
                cy={cy}
                r={DOT_RADIUS}
                fill={isDem ? PALETTE.DEM_SECONDARY : PALETTE.GOP_SECONDARY}
                opacity={0.85}
              />
            );
          })}
        </Group>
      </svg>

      {/* Caption */}
      <p className="text-sm text-muted-foreground mt-2">
        In{" "}
        <strong style={{ color: PALETTE.DEM_SECONDARY }}>{nDemWins}</strong> of {effectiveNDots} scenarios,
        the Democrat wins. In{" "}
        <strong style={{ color: PALETTE.GOP_SECONDARY }}>{effectiveNDots - nDemWins}</strong>, the Republican wins.
      </p>

      {/* Visually-hidden table for screen reader users.
         The caption is rendered as a <caption> element inside sr-only — but browsers
         give <caption> display:table-caption which can escape the clip rect and float
         visibly on the page.  Using aria-label on the table is sufficient for screen
         readers and avoids the visual leak. */}
      <table className="sr-only" aria-label={`Model outcome distribution — ${effectiveNDots} simulated scenarios`}>
        <thead>
          <tr>
            <th scope="col">Outcome</th>
            <th scope="col">Scenarios</th>
            <th scope="col">Percentage</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Democrat wins</td>
            <td>{nDemWins}</td>
            <td>{((nDemWins / effectiveNDots) * 100).toFixed(0)}%</td>
          </tr>
          <tr>
            <td>Republican wins</td>
            <td>{effectiveNDots - nDemWins}</td>
            <td>{(((effectiveNDots - nDemWins) / effectiveNDots) * 100).toFixed(0)}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
