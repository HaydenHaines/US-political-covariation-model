"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { SectionWeightSliders, SectionWeights } from "./SectionWeightSliders";
import { BalanceBar } from "./BalanceBar";
import { RaceCardGrid } from "./RaceCardGrid";
import { SeatBalanceTimeline } from "./SeatBalanceTimeline";
import { useRaceHistory } from "@/lib/hooks/use-race-history";
import type { SenateRaceData } from "@/lib/api";

// How long to wait after the last slider move before firing the API call.
const DEBOUNCE_MS = 400;

// Default blend weights (60% model prior, 30% state polls, 10% national polls).
const DEFAULT_WEIGHTS: SectionWeights = {
  model_prior: 60,
  state_polls: 30,
  national_polls: 10,
};

/** Per-race summary returned by POST /forecast/overview/blend. */
interface OverviewRaceSummary {
  slug: string;
  prediction: number | null;
  pred_std: number | null;
  rating_label: string;
}

/** Full response shape from POST /forecast/overview/blend. */
interface OverviewBlendResult {
  dem_seats: number;
  rep_seats: number;
  races: OverviewRaceSummary[];
}

/** Rating categories that are competitive enough to show in the expanded grids. */
const TOSSUP_RATINGS = new Set<string>(["tossup"]);
const LEAN_RATINGS = new Set<string>(["lean_d", "lean_r"]);
const LIKELY_RATINGS = new Set<string>(["likely_d", "likely_r"]);
const SAFE_RATINGS = new Set<string>(["safe_d", "safe_r"]);

interface OverviewBlendControlsProps {
  /** All 33 Class II senate races from the initial SSR/SWR fetch. */
  initialRaces: SenateRaceData[];
  /** Projected Dem seats from the initial fetch. */
  initialDemSeats: number;
  /** Projected GOP seats from the initial fetch. */
  initialGopSeats: number;
  /** API base URL (from lib/api.ts -- "/api/v1" or env-configured). */
  apiBase: string;
}

/**
 * Client component that owns the national blend-slider state for the senate
 * overview page.
 *
 * Renders in order:
 *   1. BalanceBar -- live seat counts + 100-segment bar
 *   2. "Adjust Forecast Blend" collapsible panel (default collapsed)
 *   3. Race card grids (Key Races / Leaning / Likely / Safe)
 *
 * When the user adjusts the sliders, a debounced POST /forecast/overview/blend
 * call recalculates all 33 races simultaneously and updates:
 *   - BalanceBar projected seat totals
 *   - Every race card's margin and rating badge
 *
 * Loading state: subtle opacity fade on the entire section while a call is
 * in-flight.  On error, the previous good values are retained silently.
 */
export function OverviewBlendControls({
  initialRaces,
  initialDemSeats,
  initialGopSeats,
  apiBase,
}: OverviewBlendControlsProps) {
  // Live race data -- starts as the SSR-provided list; overwritten by blend responses
  const [races, setRaces] = useState<SenateRaceData[]>(initialRaces);
  const [demSeats, setDemSeats] = useState(initialDemSeats);
  const [gopSeats, setGopSeats] = useState(initialGopSeats);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  // Race history for sparklines -- fetched once, not affected by blend slider changes.
  const { historyBySlug } = useRaceHistory();

  // Keep the previous good state so we can fall back on API error
  const prevRaces = useRef<SenateRaceData[]>(races);
  const prevDem = useRef(demSeats);
  const prevGop = useRef(gopSeats);
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync initial props into state when the underlying SWR data revalidates
  // (e.g. the 5-minute refresh fires after the user has been on the page).
  // This resets any custom blend back to the model default -- intentional,
  // since the data itself has changed.
  useEffect(() => {
    setRaces(initialRaces);
    setDemSeats(initialDemSeats);
    setGopSeats(initialGopSeats);
    prevRaces.current = initialRaces;
    prevDem.current = initialDemSeats;
    prevGop.current = initialGopSeats;
  }, [initialRaces, initialDemSeats, initialGopSeats]);

  const handleWeightsChange = useCallback(
    (weights: SectionWeights) => {
      if (debounceTimer.current !== null) {
        clearTimeout(debounceTimer.current);
      }

      debounceTimer.current = setTimeout(async () => {
        setIsLoading(true);
        try {
          const res = await fetch(`${apiBase}/forecast/overview/blend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(weights),
          });

          if (!res.ok) {
            // Non-2xx -- retain previous values
            return;
          }

          const data: OverviewBlendResult = await res.json();

          // Merge the blend results back onto the original race list so that
          // fields the overview endpoint doesn't return (state, n_polls, etc.)
          // are preserved.  We key the lookup by slug.
          const blendBySlug = new Map<string, OverviewRaceSummary>(
            data.races.map((r) => [r.slug, r]),
          );

          const updatedRaces: SenateRaceData[] = races.map((race) => {
            const blended = blendBySlug.get(race.slug);
            if (!blended) return race;
            return {
              ...race,
              // Convert prediction (0-1 dem share) back to margin (centered at 0)
              margin: blended.prediction !== null
                ? blended.prediction - 0.5
                : race.margin,
              rating: blended.rating_label,
            };
          });

          setRaces(updatedRaces);
          setDemSeats(data.dem_seats);
          setGopSeats(data.rep_seats);
          prevRaces.current = updatedRaces;
          prevDem.current = data.dem_seats;
          prevGop.current = data.rep_seats;
        } catch {
          // Network error -- retain previous values silently
          setRaces(prevRaces.current);
          setDemSeats(prevDem.current);
          setGopSeats(prevGop.current);
        } finally {
          setIsLoading(false);
        }
      }, DEBOUNCE_MS);
    },
    [races, apiBase],
  );

  // Clean up debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimer.current !== null) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, []);

  const tossupRaces = races.filter((r) => TOSSUP_RATINGS.has(r.rating));
  const leanRaces = races.filter((r) => LEAN_RATINGS.has(r.rating));
  const likelyRaces = races.filter((r) => LIKELY_RATINGS.has(r.rating));
  const safeRaces = races.filter((r) => SAFE_RATINGS.has(r.rating));

  return (
    <>
      {/* Balance bar -- fades slightly during recalculation */}
      <div
        style={{ transition: "opacity 150ms ease", opacity: isLoading ? 0.5 : 1 }}
        aria-busy={isLoading}
      >
        <BalanceBar races={races} demSeats={demSeats} gopSeats={gopSeats} />
      </div>

      {/* Seat balance timeline -- shows projected seat counts changing over time */}
      <div className="mb-6">
        <SeatBalanceTimeline />
      </div>

      {/* Collapsible blend controls */}
      <div className="mb-6">
        <button
          className="flex items-center gap-2 text-sm font-medium hover:opacity-75 transition-opacity w-full text-left"
          onClick={() => setIsExpanded((v) => !v)}
          aria-expanded={isExpanded}
          aria-controls="overview-blend-panel"
        >
          <span style={{ color: "var(--color-text-muted)" }}>
            {isExpanded ? "---" : "---"}
          </span>
          <span>Adjust Forecast Blend</span>
          {isLoading && (
            <span
              className="ml-2 text-xs"
              style={{ color: "var(--color-text-muted)" }}
            >
              Recalculating
            </span>
          )}
        </button>

        {isExpanded && (
          <div id="overview-blend-panel" className="mt-3">
            <p
              className="text-sm mb-3"
              style={{ color: "var(--color-text-muted)" }}
            >
              Adjust how much weight the model gives to the structural prior
              versus available polling. Changes update all race projections
              simultaneously.
            </p>
            <SectionWeightSliders
              initial={DEFAULT_WEIGHTS}
              onChange={handleWeightsChange}
            />
          </div>
        )}
      </div>

      {/* Race card grids -- also faded during recalculation */}
      <div
        style={{ transition: "opacity 150ms ease", opacity: isLoading ? 0.5 : 1 }}
        aria-busy={isLoading}
      >
        {tossupRaces.length > 0 && (
          <RaceCardGrid races={tossupRaces} title="Key Races" historyBySlug={historyBySlug} />
        )}
        {leanRaces.length > 0 && (
          <RaceCardGrid races={leanRaces} title="Leaning" historyBySlug={historyBySlug} />
        )}
        {likelyRaces.length > 0 && (
          <RaceCardGrid races={likelyRaces} title="Likely" historyBySlug={historyBySlug} />
        )}
        {safeRaces.length > 0 && (
          <SafeRacesSection races={safeRaces} historyBySlug={historyBySlug} />
        )}
      </div>
    </>
  );
}

/**
 * The "Safe" section keeps its own expand/collapse state independently of the
 * blend controls so the two don't interfere with each other.
 */
function SafeRacesSection({
  races,
  historyBySlug,
}: {
  races: SenateRaceData[];
  historyBySlug?: ReturnType<typeof useRaceHistory>["historyBySlug"];
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <section className="mb-8">
      <button
        className="flex items-center gap-2 font-serif text-lg font-semibold mb-3 hover:opacity-75 transition-opacity"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <span>Safe ({races.length})</span>
        <span className="text-sm font-normal text-muted-foreground" aria-hidden="true">
          {expanded ? "collapse" : "expand"}
        </span>
      </button>
      {expanded && <RaceCardGrid races={races} title="" historyBySlug={historyBySlug} />}
    </section>
  );
}
