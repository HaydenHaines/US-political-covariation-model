"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { useGovernorOverview } from "@/lib/hooks/use-governor-overview";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { NationalProjectionStrip } from "@/components/forecast/NationalProjectionStrip";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { ELECTION_YEAR } from "@/lib/config/election";
import type { SenateRaceData } from "@/lib/api";

const COMPETITIVE_RATINGS = new Set(["tossup", "lean_d", "lean_r"]);

export default function CompetitiveRacesPage() {
  const { data: senateData, error: senateError, isLoading: senateLoading, mutate: senateMutate } = useSenateOverview();
  const { data: govData, error: govError, isLoading: govLoading, mutate: govMutate } = useGovernorOverview();

  if (senateError || govError) {
    return (
      <ErrorAlert
        title="Failed to load competitive races"
        retry={() => { senateMutate(); govMutate(); }}
      />
    );
  }

  if (senateLoading || govLoading || !senateData || !govData) {
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

  const competitiveSenateRaces = senateData.races.filter((r) =>
    COMPETITIVE_RATINGS.has(r.rating),
  );

  // GovernorRaceData is a superset of SenateRaceData — cast is safe
  const competitiveGovRaces = (
    govData.races as unknown as SenateRaceData[]
  ).filter((r) => COMPETITIVE_RATINGS.has(r.rating));

  const totalCount = competitiveSenateRaces.length + competitiveGovRaces.length;

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-2">
        {ELECTION_YEAR} Competitive Races
      </h1>
      <p className="text-sm mb-6" style={{ color: "var(--color-text-muted)" }}>
        {totalCount} tossup or lean race{totalCount !== 1 ? "s" : ""} across Senate and Governor contests.
      </p>
      <NationalProjectionStrip />

      {competitiveSenateRaces.length > 0 && (
        <RaceCardGrid
          races={competitiveSenateRaces}
          title="Competitive Senate Races"
          basePath="/forecast/senate"
        />
      )}

      {competitiveGovRaces.length > 0 && (
        <RaceCardGrid
          races={competitiveGovRaces}
          title="Competitive Governor Races"
          basePath="/forecast/governor"
        />
      )}

      {totalCount === 0 && (
        <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
          No competitive races at this time.
        </p>
      )}
    </div>
  );
}
