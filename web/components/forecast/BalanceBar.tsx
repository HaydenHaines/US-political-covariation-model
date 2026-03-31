"use client";

import { useRouter } from "next/navigation";
import { RATING_COLORS, RATING_LABELS, DUSTY_INK } from "@/lib/config/palette";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { SenateRaceData } from "@/lib/api";
import { cn } from "@/lib/utils";

interface BalanceBarProps {
  races: SenateRaceData[];
  /**
   * Projected Dem seat total (safe seats + favored contested seats, tossups excluded).
   * Displayed above the balance bar and used to position the majority marker.
   */
  demSeats: number;
  /**
   * Projected GOP seat total (safe seats + favored contested seats, tossups excluded).
   */
  gopSeats: number;
}

/** Rating sort order: most D first → tossup → most R last */
const RATING_ORDER: Record<string, number> = {
  safe_d: 0,
  likely_d: 1,
  lean_d: 2,
  tossup: 3,
  lean_r: 4,
  likely_r: 5,
  safe_r: 6,
};

/**
 * Format a signed Dem margin (positive = Dem advantage) as a partisan string.
 */
function formatRaceMargin(margin: number): string {
  if (Math.abs(margin) < 0.005) return "EVEN";
  const pct = (Math.abs(margin) * 100).toFixed(1);
  return margin > 0 ? `D+${pct}` : `R+${pct}`;
}

/**
 * Senate balance bar — shows all 100 Senate seats as thin segments.
 *
 * Layout (left → right):
 *   [demSeats safe D seats] [contested races, sorted safe_d → safe_r] [gopSeats safe R seats]
 *
 * The "51 needed for control" marker is positioned at segment 51 from the left.
 * Non-contested safe seats are shorter (24px); contested seats are taller (32px).
 * Competitive races retain click-to-navigate and hover tooltips.
 */
export function BalanceBar({ races, demSeats, gopSeats }: BalanceBarProps) {
  const router = useRouter();

  const sorted = [...races].sort(
    (a, b) => (RATING_ORDER[a.rating] ?? 3) - (RATING_ORDER[b.rating] ?? 3),
  );

  // Safe-seat colors: muted, clearly "not a race" variants from the palette
  const DEM_SAFE_COLOR = DUSTY_INK.safeD;   // "#2d4a6f"
  const GOP_SAFE_COLOR = DUSTY_INK.safeR;   // "#6e3535"

  // Height constants
  const SAFE_HEIGHT = 24;
  const CONTESTED_HEIGHT = 32;

  // Total segments must equal 100; guard against data inconsistency
  const totalContested = sorted.length;
  const safeDCount = Math.max(0, Math.min(demSeats, 100 - totalContested - gopSeats));
  const safeRCount = Math.max(0, Math.min(gopSeats, 100 - totalContested - safeDCount));

  // Position of the "51 needed" marker: after demSeats safe segments
  // The marker sits between segment 50 and 51 (zero-indexed), i.e. at 50% of total 100
  // We express as a percentage of the full bar width.
  const markerPct = (demSeats / 100) * 100;

  // Mobile summary
  const ratingGroups = sorted.reduce<Record<string, SenateRaceData[]>>((acc, race) => {
    const key = race.rating;
    if (!acc[key]) acc[key] = [];
    acc[key].push(race);
    return acc;
  }, {});

  const tossups = ratingGroups["tossup"] ?? [];
  const leanD = [...(ratingGroups["lean_d"] ?? []), ...(ratingGroups["likely_d"] ?? [])];
  const leanR = [...(ratingGroups["lean_r"] ?? []), ...(ratingGroups["likely_r"] ?? [])];

  return (
    <TooltipProvider delay={100}>
      <div className="mb-6">
        {/* Seat counts above bar */}
        <div className="flex justify-between mb-2 text-sm font-semibold">
          <span style={{ color: RATING_COLORS.safe_d }}>{demSeats}D</span>
          <span className="text-muted-foreground">51 needed for control</span>
          <span style={{ color: RATING_COLORS.safe_r }}>{gopSeats}R</span>
        </div>

        {/* Mobile: text summary (<768px) */}
        <div className="md:hidden space-y-1 text-sm py-2 px-3 rounded-md border border-[rgb(var(--color-border))] bg-[var(--color-surface)]">
          {tossups.length > 0 && (
            <p style={{ color: RATING_COLORS.tossup }}>
              <span className="font-semibold">Tossup:</span>{" "}
              {tossups.map((r) => r.state).join(", ")}
            </p>
          )}
          {leanD.length > 0 && (
            <p style={{ color: RATING_COLORS.lean_d }}>
              <span className="font-semibold">Lean/Likely D:</span>{" "}
              {leanD.map((r) => r.state).join(", ")}
            </p>
          )}
          {leanR.length > 0 && (
            <p style={{ color: RATING_COLORS.lean_r }}>
              <span className="font-semibold">Lean/Likely R:</span>{" "}
              {leanR.map((r) => r.state).join(", ")}
            </p>
          )}
          {tossups.length === 0 && leanD.length === 0 && leanR.length === 0 && (
            <p className="text-muted-foreground italic">No competitive races.</p>
          )}
        </div>

        {/* Desktop: 100-segment bar (≥768px) */}
        <div
          className="hidden md:block relative"
          style={{ height: CONTESTED_HEIGHT }}
        >
          {/* All 100 segments laid out as a flex row, vertically centered */}
          <div
            className="flex items-end rounded-md overflow-hidden border border-[rgb(var(--color-border))]"
            style={{ height: CONTESTED_HEIGHT }}
          >
            {/* Left block: safe Dem seats */}
            {Array.from({ length: safeDCount }, (_, i) => (
              <div
                key={`safe-d-${i}`}
                style={{
                  flex: 1,
                  height: SAFE_HEIGHT,
                  backgroundColor: DEM_SAFE_COLOR,
                  opacity: 0.75,
                }}
              />
            ))}

            {/* Middle: contested races with tooltips */}
            {sorted.map((race) => (
              <Tooltip key={race.slug}>
                <TooltipTrigger
                  render={
                    <button
                      className="transition-opacity hover:opacity-80 focus:outline-none focus:ring-2 focus:ring-offset-1 min-w-[6px]"
                      style={{
                        flex: 1,
                        height: CONTESTED_HEIGHT,
                        backgroundColor:
                          RATING_COLORS[race.rating as keyof typeof RATING_COLORS] ??
                          RATING_COLORS.tossup,
                      }}
                      onClick={() => router.push(`/forecast/${race.slug}`)}
                      aria-label={`${race.state}: ${formatRaceMargin(race.margin)}`}
                    />
                  }
                />
                <TooltipContent>
                  <p className="font-semibold">{race.state}</p>
                  <p>
                    {formatRaceMargin(race.margin)} ·{" "}
                    {RATING_LABELS[race.rating as keyof typeof RATING_LABELS] ?? race.rating}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {race.n_polls} poll{race.n_polls === 1 ? "" : "s"}
                  </p>
                </TooltipContent>
              </Tooltip>
            ))}

            {/* Right block: safe GOP seats */}
            {Array.from({ length: safeRCount }, (_, i) => (
              <div
                key={`safe-r-${i}`}
                style={{
                  flex: 1,
                  height: SAFE_HEIGHT,
                  backgroundColor: GOP_SAFE_COLOR,
                  opacity: 0.75,
                }}
              />
            ))}
          </div>

          {/* "51 needed" marker — positioned at the 51st seat from left */}
          <div
            className="absolute top-0 h-full w-px bg-foreground opacity-40 pointer-events-none"
            style={{ left: `${markerPct}%` }}
            aria-hidden="true"
          />
        </div>
      </div>
    </TooltipProvider>
  );
}
