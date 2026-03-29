"use client";

import Link from "next/link";
import type { SenateRaceData } from "@/lib/api";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { Skeleton } from "@/components/ui/skeleton";

/** Maximum number of competitive races to display. */
const MAX_TICKER_RACES = 8;

interface RaceTickerProps {
  races: SenateRaceData[] | undefined;
  isLoading: boolean;
}

/**
 * Horizontal scrollable strip of the most competitive races.
 * Sorted by absolute margin (closest races first), capped at MAX_TICKER_RACES.
 */
export function RaceTicker({ races, isLoading }: RaceTickerProps) {
  if (isLoading || !races) {
    return (
      <section className="py-4">
        <h2
          className="mb-3 text-center text-sm font-semibold uppercase tracking-wider"
          style={{ color: "var(--color-text-muted)" }}
        >
          Closest Races
        </h2>
        <div className="flex flex-wrap justify-center gap-3 px-4 pb-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-32 shrink-0 rounded-lg" />
          ))}
        </div>
      </section>
    );
  }

  // margin is centered at 0 (positive = Dem). Sort by closeness to 0.
  const sorted = [...races]
    .sort((a, b) => Math.abs(a.margin) - Math.abs(b.margin))
    .slice(0, MAX_TICKER_RACES);

  if (sorted.length === 0) return null;

  return (
    <section className="py-4">
      <h2
        className="mb-3 text-center text-sm font-semibold uppercase tracking-wider"
        style={{ color: "var(--color-text-muted)" }}
      >
        Closest Races
      </h2>

      <div className="flex flex-wrap justify-center gap-3 px-4 pb-2">
        {sorted.map((race) => (
          <Link
            key={race.slug}
            href={`/forecast/${race.slug}`}
            className="flex shrink-0 flex-col items-center gap-1.5 rounded-lg border border-[var(--color-border)] px-4 py-3 no-underline transition-colors hover:bg-[var(--color-surface-raised)]"
            style={{ background: "var(--color-surface)" }}
          >
            <span
              className="text-lg font-bold"
              style={{ color: "var(--color-text)" }}
            >
              {race.state}
            </span>
            <MarginDisplay demShare={race.margin + 0.5} size="sm" />
            <RatingBadge rating={race.rating} />
          </Link>
        ))}
      </div>
    </section>
  );
}
