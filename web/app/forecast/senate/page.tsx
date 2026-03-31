"use client";

import { useState } from "react";
import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { BalanceBar } from "@/components/forecast/BalanceBar";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { ELECTION_YEAR } from "@/lib/config/election";
import type { SenateRaceData } from "@/lib/api";

/** Ratings that are genuine tossups — the most competitive races. */
const TOSSUP_RATINGS = new Set<string>(["tossup"]);

/** Lean ratings — model has a directional view but still competitive. */
const LEAN_RATINGS = new Set<string>(["lean_d", "lean_r"]);

/** Likely ratings — meaningful model advantage, but not settled. */
const LIKELY_RATINGS = new Set<string>(["likely_d", "likely_r"]);

/** Safe ratings — model treats these as decided. */
const SAFE_RATINGS = new Set<string>(["safe_d", "safe_r"]);

function filterRaces(races: SenateRaceData[], ratingSet: Set<string>): SenateRaceData[] {
  return races.filter((r) => ratingSet.has(r.rating));
}

export default function SenatePage() {
  const { data, error, isLoading, mutate } = useSenateOverview();
  // Safe races are collapsed by default — they're decided and not interesting
  const [safeExpanded, setSafeExpanded] = useState(false);

  if (error) {
    return <ErrorAlert title="Failed to load Senate forecast" retry={() => mutate()} />;
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-8 w-full" />
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  const tossupRaces = filterRaces(data.races, TOSSUP_RATINGS);
  const leanRaces = filterRaces(data.races, LEAN_RATINGS);
  const likelyRaces = filterRaces(data.races, LIKELY_RATINGS);
  const safeRaces = filterRaces(data.races, SAFE_RATINGS);

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">{ELECTION_YEAR} United States Senate</h1>

      <BalanceBar
        races={data.races}
        demSeats={data.dem_projected}
        gopSeats={data.gop_projected}
      />

      {tossupRaces.length > 0 && (
        <RaceCardGrid races={tossupRaces} title="Key Races" />
      )}

      {leanRaces.length > 0 && (
        <RaceCardGrid races={leanRaces} title="Leaning" />
      )}

      {likelyRaces.length > 0 && (
        <RaceCardGrid races={likelyRaces} title="Likely" />
      )}

      {safeRaces.length > 0 && (
        <section className="mb-8">
          <button
            className="flex items-center gap-2 font-serif text-lg font-semibold mb-3 hover:opacity-75 transition-opacity"
            onClick={() => setSafeExpanded((v) => !v)}
            aria-expanded={safeExpanded}
          >
            <span>Safe ({safeRaces.length})</span>
            <span className="text-sm font-normal text-muted-foreground" aria-hidden="true">
              {safeExpanded ? "▲ collapse" : "▼ expand"}
            </span>
          </button>
          {safeExpanded && <RaceCardGrid races={safeRaces} title="" />}
        </section>
      )}
    </div>
  );
}
