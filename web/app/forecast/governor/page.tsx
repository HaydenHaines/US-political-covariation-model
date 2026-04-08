"use client";

import { useGovernorOverview } from "@/lib/hooks/use-governor-overview";
import { FundamentalsCard } from "@/components/forecast/FundamentalsCard";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { ELECTION_YEAR, GOVERNOR_RACES_COUNT } from "@/lib/config/election";
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

  if (error) {
    return <ErrorAlert title="Failed to load Governor forecast" retry={() => mutate()} />;
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96" />
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

  // GovernorRaceData is a superset of SenateRaceData — cast is safe
  const allRaces = data.races as unknown as SenateRaceData[];

  const competitiveRaces = allRaces.filter((r) => competitiveRatings.has(r.rating));
  const dLeaningRaces = allRaces.filter((r) => dLeaningRatings.has(r.rating));
  const rLeaningRaces = allRaces.filter((r) => rLeaningRatings.has(r.rating));

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

      {/* Competitive races first — these are what readers care most about */}
      {competitiveRaces.length > 0 && (
        <RaceCardGrid races={competitiveRaces} title="Competitive Races" />
      )}

      {/* D-leaning races */}
      {dLeaningRaces.length > 0 && (
        <RaceCardGrid races={dLeaningRaces} title="Likely and Safe Democratic" />
      )}

      {/* R-leaning races */}
      {rLeaningRaces.length > 0 && (
        <RaceCardGrid races={rLeaningRaces} title="Likely and Safe Republican" />
      )}
    </div>
  );
}
