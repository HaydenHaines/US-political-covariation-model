"use client";

import { useGovernorOverview } from "@/lib/hooks/use-governor-overview";
import { useGovernorSimulation } from "@/lib/hooks/use-governor-simulation";
import { FundamentalsCard } from "@/components/forecast/FundamentalsCard";
import { GovernorPollingCard } from "@/components/forecast/GovernorPollingCard";
import {
  GovernorSeatRiskCard,
  GovernorSeatRiskCardSkeleton,
} from "@/components/forecast/GovernorSeatRiskCard";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { ELECTION_YEAR, GOVERNOR_RACES_COUNT } from "@/lib/config/election";
import { DUSTY_INK } from "@/lib/config/palette";
import type { SenateRaceData } from "@/lib/api";

/**
 * Governor overview page.
 *
 * Displays structural model forecasts for all 36 gubernatorial races.
 * Unlike the Senate page there is no chamber control concept — governors
 * are independent executives — so no balance bar, seat totals, or blend
 * controls are shown.  Race cards link to the shared /forecast/[slug]
 * detail page.
 */
export default function GovernorPage() {
  const { data, error, isLoading, mutate } = useGovernorOverview();
  const { data: simData } = useGovernorSimulation();

  if (error) {
    return <ErrorAlert title="Failed to load Governor forecast" retry={() => mutate()} />;
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96" />
        <GovernorSeatRiskCardSkeleton />
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  // Count open seats from API data (term-limited, resigned, or vacated).
  const openSeatCount = data.races.filter((r) => r.is_open_seat).length;

  // Separate races into D-leaning / competitive / R-leaning groups for display.
  // "Competitive" = tossup, lean_d, or lean_r.
  const dLeaningRatings = new Set(["safe_d", "likely_d"]);
  const rLeaningRatings = new Set(["safe_r", "likely_r"]);
  const competitiveRatings = new Set(["tossup", "lean_d", "lean_r"]);

  // GovernorRaceData is a superset of SenateRaceData — cast is safe.
  // Slug is remapped to state abbreviation lowercase (e.g. "wa") so detail
  // page URLs are /forecast/governor/wa rather than /forecast/governor/2026-wa-governor.
  const allRaces = data.races.map((r) => ({
    ...r,
    slug: r.state.toLowerCase(),
  })) as unknown as SenateRaceData[];

  const competitiveRaces = allRaces.filter((r) => competitiveRatings.has(r.rating));
  const dLeaningRaces = allRaces.filter((r) => dLeaningRatings.has(r.rating));
  const rLeaningRaces = allRaces.filter((r) => rLeaningRatings.has(r.rating));

  // Compute median + 10th/90th percentile D seat counts from simulation buckets.
  let simStats: { median: number; lo80: number; hi80: number } | null = null;
  if (simData?.buckets && simData.buckets.length > 0) {
    const sorted = [...simData.buckets].sort((a, b) => a.d_seats - b.d_seats);
    let cum = 0;
    let lo80: number | null = null;
    let median: number | null = null;
    let hi80: number | null = null;
    for (const b of sorted) {
      cum += b.probability;
      if (lo80 === null && cum >= 0.1) lo80 = b.d_seats;
      if (median === null && cum >= 0.5) median = b.d_seats;
      if (hi80 === null && cum >= 0.9) { hi80 = b.d_seats; break; }
    }
    if (median !== null && lo80 !== null && hi80 !== null) {
      simStats = { median, lo80, hi80 };
    }
  }

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-2">
        {ELECTION_YEAR} Governor Races
      </h1>
      <p className="text-sm mb-6" style={{ color: "var(--color-text-muted)" }}>
        {GOVERNOR_RACES_COUNT} governors on the ballot in {ELECTION_YEAR},
        including {openSeatCount} open seats.
        {data.updated_at && (
          <> Polls updated {data.updated_at}.</>
        )}
      </p>

      {/* National environment — structural forecast applies to all race types */}
      <FundamentalsCard />

      {/* Polling coverage summary — total polls, races polled, coverage quality */}
      <GovernorPollingCard />

      {/* Simulation-based seat distribution summary */}
      {simStats && (
        <section
          className="mb-8 rounded-md p-4 text-sm"
          aria-label="Simulation Summary"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
          }}
        >
          <div className="flex flex-wrap items-baseline justify-between gap-3 mb-1">
            <h2
              className="font-serif text-lg"
              style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
            >
              Simulation Outlook
            </h2>
            <span className="font-mono text-xs" style={{ color: "var(--color-text-muted)" }}>
              Monte Carlo
            </span>
          </div>
          <p className="mb-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
            Expected governor seat counts across all races — median result with 80% range
          </p>
          <dl className="space-y-2">
            <div className="flex items-center justify-between gap-4">
              <dt style={{ color: "var(--color-text-muted)" }}>Democratic governors (median)</dt>
              <dd className="font-mono font-semibold" style={{ color: DUSTY_INK.safeD }}>
                {simStats.median}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt style={{ color: "var(--color-text-muted)" }}>Republican governors (median)</dt>
              <dd className="font-mono font-semibold" style={{ color: DUSTY_INK.safeR }}>
                {36 - simStats.median}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt style={{ color: "var(--color-text-muted)" }}>D range (80% interval)</dt>
              <dd className="font-mono font-semibold" style={{ color: "var(--color-text-muted)" }}>
                {simStats.lo80}–{simStats.hi80}
              </dd>
            </div>
          </dl>
        </section>
      )}

      {/* Partisan balance and competitive seat exposure */}
      <GovernorSeatRiskCard races={data.races} />

      {/* Competitive races first — these are what readers care most about */}
      {competitiveRaces.length > 0 && (
        <RaceCardGrid races={competitiveRaces} title="Competitive Races" basePath="/forecast/governor" />
      )}

      {/* D-leaning races */}
      {dLeaningRaces.length > 0 && (
        <RaceCardGrid races={dLeaningRaces} title="Likely and Safe Democratic" basePath="/forecast/governor" />
      )}

      {/* R-leaning races */}
      {rLeaningRaces.length > 0 && (
        <RaceCardGrid races={rLeaningRaces} title="Likely and Safe Republican" basePath="/forecast/governor" />
      )}
    </div>
  );
}
