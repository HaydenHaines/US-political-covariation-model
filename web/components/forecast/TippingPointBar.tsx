"use client";

import { RATING_COLORS } from "@/lib/config/palette";
import type { SenateRaceData } from "@/lib/api";

/**
 * Senate tipping point bar — shows all 33 Class II contested seats sorted by
 * Dem vote share (strongest D on the left → strongest R on the right).
 *
 * The 18th seat from the left is the "tipping point": Democrats hold 33
 * non-contested holdover seats and need 18 of the 33 Class II races to reach
 * 51 seats and retake the majority. A thick black border marks this seat.
 *
 * Flanked by labels for the holdover seats on each side:
 *   Left:  "33 Dem Seats not on the ballot in 2026"
 *   Right: "34 Republican Seats not on the ballot in 2026"
 *
 * Color: each segment uses the Dusty Ink palette rating color so the visual
 * language is consistent with the balance bar and race cards.
 */
interface TippingPointBarProps {
  /** All 33 Class II senate races from the overview endpoint. */
  races: SenateRaceData[];
}

/**
 * Number of Dem and GOP holdover seats (not up in 2026).
 * Class I + Class III holdovers: 33D + 34R = 67 total.
 * These are structural constants that match the API's DEM_SAFE_SEATS /
 * GOP_SAFE_SEATS minus the Class II seats up this cycle.
 */
const DEM_HOLDOVER_SEATS = 33;
const GOP_HOLDOVER_SEATS = 34;

/**
 * How many Class II seats Dems must win to reach 51 majority seats:
 *   51 target - 33 holdover = 18 wins needed.
 * The 18th seat (1-indexed from the left in the sorted bar) is the tipping
 * point — flipping this seat is what tips majority control.
 */
const DEM_WINS_NEEDED = 51 - DEM_HOLDOVER_SEATS;

/** Height of each segment in the tipping point bar (px). */
const BAR_HEIGHT = 32;

/** Height of the state label area below the bar (px). */
const LABEL_AREA_HEIGHT = 20;

export function TippingPointBar({ races }: TippingPointBarProps) {
  // Sort highest D margin (most positive) → highest R margin (most negative),
  // left to right.  This puts the most D-favorable seats on the left and the
  // most R-favorable on the right, so the tipping point is in the middle.
  const sorted = [...races].sort((a, b) => b.margin - a.margin);

  return (
    <div className="mb-8">
      {/* Section heading */}
      <h2
        className="font-serif text-lg font-semibold mb-1"
        style={{ color: "var(--color-text)" }}
      >
        Tipping Point
      </h2>
      <p
        className="text-sm mb-4"
        style={{ color: "var(--color-text-muted)" }}
      >
        Democrats need{" "}
        <strong>{DEM_WINS_NEEDED} of {races.length}</strong> contested seats to
        reach a majority. The {DEM_WINS_NEEDED}th seat from the left (outlined
        in black) is the tipping point.
      </p>

      {/* Outer wrapper: labels + bar in one aligned block */}
      <div className="w-full">
        {/* Holdover-seat labels above the bar */}
        <div className="flex justify-between mb-1 text-xs font-medium">
          <span style={{ color: "var(--color-text-muted)" }}>
            {DEM_HOLDOVER_SEATS} Dem Seats not on the ballot in 2026
          </span>
          <span style={{ color: "var(--color-text-muted)" }}>
            {GOP_HOLDOVER_SEATS} Republican Seats not on the ballot in 2026
          </span>
        </div>

        {/* Bar + state labels */}
        <div
          className="relative w-full"
          style={{ height: BAR_HEIGHT + LABEL_AREA_HEIGHT }}
        >
          <div
            className="flex w-full rounded-md overflow-visible border border-[rgb(var(--color-border))]"
            style={{ height: BAR_HEIGHT }}
          >
            {sorted.map((race, idx) => {
              const isTippingPoint = idx + 1 === DEM_WINS_NEEDED;
              const color =
                RATING_COLORS[race.rating as keyof typeof RATING_COLORS] ??
                RATING_COLORS.tossup;

              return (
                <div
                  key={race.slug}
                  className="relative flex items-center justify-center"
                  style={{
                    flex: 1,
                    height: BAR_HEIGHT,
                    backgroundColor: color,
                    // Tipping point seat: thick black right-edge border to
                    // delineate "18th seat" from "19th seat".
                    ...(isTippingPoint
                      ? { borderRight: "3px solid #111111", zIndex: 1 }
                      : {}),
                  }}
                  title={`${race.state}: ${race.margin >= 0 ? "D+" : "R+"}${(Math.abs(race.margin) * 100).toFixed(1)} (${race.rating})`}
                  aria-label={`${race.state}: ${race.rating}`}
                >
                  {/* State abbreviation inside the segment */}
                  <span
                    className="pointer-events-none select-none"
                    style={{
                      fontSize: "9px",
                      fontWeight: 600,
                      color: "rgba(255,255,255,0.85)",
                      lineHeight: 1,
                      letterSpacing: "0.02em",
                    }}
                    aria-hidden="true"
                  >
                    {race.state}
                  </span>

                  {/* Tipping point marker: small label above the right edge */}
                  {isTippingPoint && (
                    <span
                      className="absolute pointer-events-none select-none text-center"
                      style={{
                        bottom: BAR_HEIGHT + 2,
                        right: -24,
                        fontSize: "9px",
                        fontWeight: 700,
                        color: "#111111",
                        whiteSpace: "nowrap",
                        lineHeight: 1,
                        zIndex: 2,
                      }}
                      aria-hidden="true"
                    >
                      Maj. →
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Legend */}
        <p
          className="text-xs mt-1 text-right"
          style={{ color: "var(--color-text-subtle)" }}
        >
          Seats sorted by Dem vote share · Left = strongest D, Right = strongest R
        </p>
      </div>
    </div>
  );
}
