import { RaceCard } from "./RaceCard";
import type { SenateRaceData } from "@/lib/api";

interface RaceCardGridProps {
  races: SenateRaceData[];
  title: string;
}

/**
 * Grid of race cards, sorted by competitiveness (smallest absolute margin first).
 *
 * Mobile (<768px): horizontal snap-scroll carousel — cards are fixed-width and
 * users swipe through them. Desktop (≥768px): responsive grid layout.
 */
export function RaceCardGrid({ races, title }: RaceCardGridProps) {
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
            <RaceCard race={race} />
          </div>
        ))}
      </div>

      {/* Desktop: responsive grid */}
      <div className="hidden md:grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3 gap-3">
        {sorted.map((race) => (
          <RaceCard key={race.slug} race={race} />
        ))}
      </div>
    </section>
  );
}
