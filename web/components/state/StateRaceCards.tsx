"use client";

import { RaceCard } from "@/components/forecast/RaceCard";
import { useRaceHistory } from "@/lib/hooks/use-race-history";
import type { SenateRaceData } from "@/lib/api";

interface StateRaceCardsProps {
  races: SenateRaceData[];
  basePath?: string;
}

export function StateRaceCards({ races, basePath }: StateRaceCardsProps) {
  const { historyBySlug } = useRaceHistory();

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {races.map((race) => (
        <RaceCard
          key={race.slug}
          race={race}
          sparklineHistory={historyBySlug?.get(race.slug)}
          basePath={basePath}
        />
      ))}
    </div>
  );
}
