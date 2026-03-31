"use client";

import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { marginToRating } from "@/lib/config/palette";
import { formatMargin } from "@/lib/format";

interface RaceHeroProps {
  stateName: string;
  raceType: string;
  year: number;
  /** 0-1 dem share (API `prediction` field) */
  prediction: number | null;
  nCounties: number;
  /** Optional 90% confidence interval bounds (0-1 dem share) */
  lo90?: number | null;
  hi90?: number | null;
  /** Number of polls incorporated into this forecast */
  nPolls?: number;
}

/**
 * Hero section for a race detail page.
 *
 * Displays the large margin number, rating badge, and a 90% confidence
 * interval text line when bounds are available.
 */
export function RaceHero({
  stateName,
  raceType,
  year,
  prediction,
  nCounties,
  lo90,
  hi90,
  nPolls = 0,
}: RaceHeroProps) {
  const rating = prediction !== null ? marginToRating(prediction) : "tossup";

  const hasBounds =
    lo90 !== null && lo90 !== undefined && hi90 !== null && hi90 !== undefined;

  return (
    <div className="mb-8">
      {/* Race title */}
      <h1
        className="font-serif text-3xl lg:text-4xl mb-1"
        style={{ fontFamily: "var(--font-serif)", lineHeight: 1.2 }}
      >
        {year} {stateName} {raceType}
      </h1>
      <p className="text-sm text-muted-foreground mb-4">
        {nCounties} {nCounties === 1 ? "county" : "counties"} in model
      </p>

      {/* Margin + badge row */}
      <div className="flex items-center gap-4 flex-wrap mb-3">
        <MarginDisplay demShare={prediction} size="xl" />
        <RatingBadge rating={rating} />
        {nPolls > 0 ? (
          <span
            className="text-xs px-2 py-0.5 rounded-full font-medium"
            style={{
              background: "color-mix(in srgb, var(--color-dem) 12%, transparent)",
              color: "var(--color-dem)",
              border: "1px solid color-mix(in srgb, var(--color-dem) 30%, transparent)",
            }}
          >
            Poll-informed
          </span>
        ) : (
          <span
            className="text-xs"
            style={{ color: "var(--color-text-muted)" }}
          >
            Model prior only
          </span>
        )}
      </div>

      {/* 90% confidence interval */}
      {hasBounds && (
        <p className="text-sm text-muted-foreground">
          90% interval:{" "}
          <span className="font-mono">
            {formatMargin(lo90 as number)}
          </span>{" "}
          to{" "}
          <span className="font-mono">
            {formatMargin(hi90 as number)}
          </span>
        </p>
      )}

      {/* Sub-label when no prediction */}
      {prediction === null && (
        <p className="text-sm text-muted-foreground italic">
          No model prediction available for this race yet.
        </p>
      )}

    </div>
  );
}
