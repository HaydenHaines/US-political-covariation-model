"use client";

import { useChamberProbability } from "@/lib/hooks/use-chamber-probability";
import { Skeleton } from "@/components/ui/skeleton";
import { PALETTE } from "@/lib/config/palette";

/**
 * ChamberProbabilityBanner — the "One Big Number" anchor above the balance bar.
 *
 * Shows the leading party's chamber control probability in large text, with a
 * secondary line noting the opposing party's probability.  When the race is
 * very close (within 5pp of 50/50), both probabilities are shown side-by-side.
 *
 * Data comes from the Monte Carlo simulation endpoint which runs N=10,000
 * simulations using the model's per-race predictions and uncertainty.
 */
export function ChamberProbabilityBanner() {
  const { data, isLoading } = useChamberProbability();

  if (isLoading || !data) {
    return (
      <div className="mb-4 space-y-2">
        <Skeleton className="h-10 w-80" />
        <Skeleton className="h-4 w-48" />
      </div>
    );
  }

  // Choose which party leads and what probability to headline.
  // We use dem_majority_pct (≥51 seats) as the primary metric since in a GOP
  // presidency there is no VP tiebreaker for Democrats.
  const demPct = data.dem_majority_pct;
  const repPct = data.rep_control_pct;
  const isClose = Math.abs(demPct - 50) < 5;

  const leadingParty = demPct >= repPct ? "Democrats" : "Republicans";
  const leadingPct = demPct >= repPct ? demPct : repPct;
  const leadingColor = demPct >= repPct ? PALETTE.DEM_PRIMARY : PALETTE.GOP_PRIMARY;
  const trailingParty = demPct >= repPct ? "Republicans" : "Democrats";
  const trailingPct = demPct >= repPct ? repPct : demPct;
  const trailingColor = demPct >= repPct ? PALETTE.GOP_PRIMARY : PALETTE.DEM_PRIMARY;

  return (
    <div className="mb-5" aria-label="Chamber control probability">
      {isClose ? (
        /* Near-tossup: show both probabilities side-by-side */
        <div className="flex items-baseline gap-4 flex-wrap">
          <span className="font-serif text-3xl font-bold" style={{ color: PALETTE.DEM_PRIMARY }}>
            Dems {data.dem_majority_pct.toFixed(0)}%
          </span>
          <span className="text-muted-foreground font-medium">vs.</span>
          <span className="font-serif text-3xl font-bold" style={{ color: PALETTE.GOP_PRIMARY }}>
            GOP {data.rep_control_pct.toFixed(0)}%
          </span>
          <span className="text-sm text-muted-foreground self-center">
            chance of Senate majority
          </span>
        </div>
      ) : (
        /* Clear leader: show the headline party prominently */
        <div>
          <p className="font-serif text-4xl font-bold leading-tight" style={{ color: leadingColor }}>
            {leadingParty}:{" "}
            <span className="tabular-nums">{leadingPct.toFixed(0)}%</span>{" "}
            <span className="text-2xl font-semibold">chance of Senate majority</span>
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            <span style={{ color: trailingColor }}>{trailingParty}</span>:{" "}
            {trailingPct.toFixed(0)}% &middot; based on {data.n_simulations.toLocaleString()} simulations &middot;{" "}
            {data.median_dem_seats}D / {data.median_rep_seats}R projected median
          </p>
        </div>
      )}
    </div>
  );
}
