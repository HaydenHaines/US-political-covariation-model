import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { FreshnessStamp } from "@/components/shared/FreshnessStamp";
import { SparklineChart } from "./SparklineChart";
import type { RaceMarginPoint, SenateRaceData } from "@/lib/api";

interface RaceCardProps {
  race: SenateRaceData;
  /**
   * Optional sparkline data for this race.
   * When provided, renders a small margin-trend sparkline inside the card.
   * When absent (loading or not yet available), the card renders without it.
   */
  sparklineHistory?: RaceMarginPoint[];
}

/**
 * Individual race card — always a link to the race detail page.
 *
 * Note on margin conversion: The API's `margin` field is centered at 0
 * (positive = Dem advantage). MarginDisplay expects a dem share in [0,1],
 * so we pass `race.margin + 0.5` to convert.
 *
 * When `sparklineHistory` is provided, an inline margin-trend sparkline is
 * rendered next to the current margin, giving readers a quick visual sense
 * of how the race has moved over recent weeks.
 */
export function RaceCard({ race, sparklineHistory }: RaceCardProps) {
  return (
    <Link href={`/forecast/${race.slug}`} className="block h-full">
      <Card className="h-full hover:border-foreground/30 transition-colors cursor-pointer">
        <CardContent className="p-4">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-3">
              <div>
                <div className="text-xs text-muted-foreground mb-1">{race.state}</div>
                <MarginDisplay demShare={race.margin + 0.5} size="lg" />
              </div>
              {sparklineHistory && sparklineHistory.length >= 1 && (
                <SparklineChart
                  history={sparklineHistory}
                  width={80}
                  height={28}
                  ariaLabel={`${race.state} margin trend`}
                />
              )}
            </div>
            <RatingBadge rating={race.rating} className="shrink-0" />
          </div>
          <FreshnessStamp pollCount={race.n_polls} />
        </CardContent>
      </Card>
    </Link>
  );
}
