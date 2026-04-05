import { RaceCard } from "./RaceCard";
import type { RaceMarginPoint, SenateRaceData } from "@/lib/api";

interface RaceCardGridProps {
  races: SenateRaceData[];
  title: string;
  /**
   * Optional slug to history lookup for sparkline rendering.
   * When provided, each card receives its race's margin history.
   * When absent, cards render without sparklines.
   */
  historyBySlug?: Map<string, RaceMarginPoint[]>;
}

/**
 * Grid of race cards, sorted by competitiveness (smallest absolute margin first).
 *
 * Mobile (<768px): horizontal snap-scroll carousel -- cards are fixed-width and
 * users swipe through them. Desktop (>=768px): responsive grid layout.
 */
export function RaceCardGrid({ races, title, historyBySlug }: RaceCardGridProps) {
  const sorted = [...races].sort(
    (a, b) => Math.abs(a.margin) - Math.abs(b.margin),
  );

  return (
    <section className="mb-8">
      <h2 className="font-serif text-lg font-semibold mb-3">{title}</h2>

      {/* Mobile: horizontal snap-scroll carousel */}
      <div className="flex md:hidden gap-3 overflow-x-auto snap-x snap-mandatory pb-2 -mx-4 px-4 scrollbar-none">
        {sorted.map((race) => (
          <div key={race.slug} className="snap-start shrink-0 w-52">
            <RaceCard
              race={race}
              sparklineHistory={historyBySlug?.get(race.slug)}
            />
          </div>
        ))}
      </div>

      {/* Desktop: responsive grid */}
      <div className="hidden md:grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3 gap-3">
        {sorted.map((race) => (
          <RaceCard
            key={race.slug}
            race={race}
            sparklineHistory={historyBySlug?.get(race.slug)}
          />
        ))}
      </div>
    </section>
  );
}
