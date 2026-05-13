"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { useGovernorOverview } from "@/lib/hooks/use-governor-overview";
import { FundamentalsCard } from "@/components/forecast/FundamentalsCard";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { ELECTION_YEAR, GOVERNOR_RACES_COUNT } from "@/lib/config/election";
import type { GovernorRaceData } from "@/lib/api";

const STATE_FILTER_URL_DEBOUNCE_MS = 200;

/**
 * Governor overview page.
 *
 * Displays structural model forecasts for all 36 gubernatorial races.
 * Unlike the Senate page there is no chamber control concept — governors
 * are independent executives — so no balance bar, seat totals, or blend
 * controls are shown.  Race cards link to the shared /forecast/[slug]
 * detail page.
 */
export default function GovernorPage() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const searchParamsString = searchParams.toString();

  const { data, error, isLoading, mutate } = useGovernorOverview();
  const [stateFilter, setStateFilter] = useState(
    () => searchParams.get("state") ?? "",
  );
  const [openSeatsOnly, setOpenSeatsOnly] = useState(
    () => searchParams.get("openSeat") === "1",
  );

  const stateFilterUrlTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync filter state from URL when Next.js search params update.
  useEffect(() => {
    const params = new URLSearchParams(searchParamsString);
    setStateFilter(params.get("state") ?? "");
    setOpenSeatsOnly(params.get("openSeat") === "1");
  }, [searchParamsString]);

  // Restore filter state on browser back/forward.
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      setStateFilter(params.get("state") ?? "");
      setOpenSeatsOnly(params.get("openSeat") === "1");
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Clean up debounce timer on unmount.
  useEffect(() => {
    return () => {
      if (stateFilterUrlTimer.current !== null) {
        clearTimeout(stateFilterUrlTimer.current);
      }
    };
  }, []);

  const writeStateToUrl = useCallback(
    (value: string) => {
      const params = new URLSearchParams(window.location.search);
      if (value) {
        params.set("state", value);
      } else {
        params.delete("state");
      }
      const query = params.toString();
      const nextUrl = query ? `${pathname}?${query}` : pathname;
      const currentUrl = `${window.location.pathname}${window.location.search}`;
      if (nextUrl !== currentUrl) {
        window.history.pushState(null, "", nextUrl);
      }
    },
    [pathname],
  );

  const writeOpenSeatToUrl = useCallback(
    (value: boolean) => {
      const params = new URLSearchParams(window.location.search);
      if (value) {
        params.set("openSeat", "1");
      } else {
        params.delete("openSeat");
      }
      const query = params.toString();
      const nextUrl = query ? `${pathname}?${query}` : pathname;
      const currentUrl = `${window.location.pathname}${window.location.search}`;
      if (nextUrl !== currentUrl) {
        window.history.pushState(null, "", nextUrl);
      }
    },
    [pathname],
  );

  const handleStateFilterChange = useCallback(
    (value: string) => {
      setStateFilter(value);
      if (stateFilterUrlTimer.current !== null) {
        clearTimeout(stateFilterUrlTimer.current);
      }
      stateFilterUrlTimer.current = setTimeout(() => {
        writeStateToUrl(value);
      }, STATE_FILTER_URL_DEBOUNCE_MS);
    },
    [writeStateToUrl],
  );

  const handleOpenSeatChange = useCallback(
    (value: boolean) => {
      setOpenSeatsOnly(value);
      writeOpenSeatToUrl(value);
    },
    [writeOpenSeatToUrl],
  );

  if (error) {
    return <ErrorAlert title="Failed to load Governor forecast" retry={() => mutate()} />;
  }

  if (isLoading || !data) {
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

  // Count open seats from API data (term-limited, resigned, or vacated).
  const openSeatCount = data.races.filter((r) => r.is_open_seat).length;

  // Separate races into D-leaning / competitive / R-leaning groups for display.
  // "Competitive" = tossup, lean_d, or lean_r.
  const dLeaningRatings = new Set(["safe_d", "likely_d"]);
  const rLeaningRatings = new Set(["safe_r", "likely_r"]);
  const competitiveRatings = new Set(["tossup", "lean_d", "lean_r"]);

  const stateNeedle = stateFilter.trim().toLowerCase();
  const filteredAll = data.races.filter((race) => {
    const matchesState =
      !stateNeedle || race.state.toLowerCase().includes(stateNeedle);
    const matchesOpenSeat = !openSeatsOnly || race.is_open_seat;

    return matchesState && matchesOpenSeat;
  });

  const competitiveRaces = filteredAll.filter((r) => competitiveRatings.has(r.rating));
  const dLeaningRaces = filteredAll.filter((r) => dLeaningRatings.has(r.rating));
  const rLeaningRaces = filteredAll.filter((r) => rLeaningRatings.has(r.rating));

  const noResults =
    (stateNeedle || openSeatsOnly) &&
    competitiveRaces.length + dLeaningRaces.length + rLeaningRaces.length === 0;

  const raceGroupStateList = (races: GovernorRaceData[]) =>
    races.map((race) => race.state).join(" ");
  const raceGroupOpenSeatStateList = (races: GovernorRaceData[]) =>
    races.filter((race) => race.is_open_seat).map((race) => race.state).join(" ");

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-2">
        {ELECTION_YEAR} Governor Races
      </h1>
      <p className="text-sm mb-6" style={{ color: "var(--color-text-muted)" }}>
        {GOVERNOR_RACES_COUNT} governors on the ballot in {ELECTION_YEAR},
        including {openSeatCount} open seats.
        {data.updated_at && (
          <> Polls updated {data.updated_at}.</>
        )}
      </p>

      <div className="flex flex-wrap gap-3 mb-4 items-end">
        <label className="flex flex-col gap-1">
          <span
            className="text-xs font-medium"
            style={{ color: "var(--color-text-muted)" }}
          >
            Filter by state
          </span>
          <input
            type="text"
            placeholder="Filter by state..."
            value={stateFilter}
            onChange={(e) => handleStateFilterChange(e.target.value)}
            data-testid="state-filter"
            style={{
              padding: "6px 10px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid var(--color-border)",
              background: "var(--color-surface)",
              color: "var(--color-dusty-ink, var(--color-text))",
            }}
          />
        </label>
        <label
          className="inline-flex items-center gap-2 text-sm"
          style={{ color: "var(--color-dusty-ink, var(--color-text))" }}
        >
          <input
            type="checkbox"
            checked={openSeatsOnly}
            onChange={(e) => handleOpenSeatChange(e.target.checked)}
            data-testid="open-seat-filter"
            style={{ accentColor: "var(--color-accent)" }}
          />
          Open seats only
        </label>
      </div>

      {/* National environment — structural forecast applies to all race types */}
      <FundamentalsCard />

      {noResults ? (
        <p
          className="text-sm mt-4"
          data-testid="governor-filter-empty-state"
          style={{ color: "var(--color-text-muted)" }}
        >
          No governor races match the selected filters.
        </p>
      ) : (
        <>
          {/* Competitive races first — these are what readers care most about */}
          {competitiveRaces.length > 0 && (
            <div
              aria-label="Competitive governor races"
              data-testid="governor-race-group"
              data-group="competitive"
              data-states={raceGroupStateList(competitiveRaces)}
              data-open-seat-states={raceGroupOpenSeatStateList(competitiveRaces)}
            >
              <RaceCardGrid races={competitiveRaces} title="Competitive Races" />
            </div>
          )}

          {/* D-leaning races */}
          {dLeaningRaces.length > 0 && (
            <div
              aria-label="Likely and safe Democratic governor races"
              data-testid="governor-race-group"
              data-group="d-leaning"
              data-states={raceGroupStateList(dLeaningRaces)}
              data-open-seat-states={raceGroupOpenSeatStateList(dLeaningRaces)}
            >
              <RaceCardGrid races={dLeaningRaces} title="Likely and Safe Democratic" />
            </div>
          )}

          {/* R-leaning races */}
          {rLeaningRaces.length > 0 && (
            <div
              aria-label="Likely and safe Republican governor races"
              data-testid="governor-race-group"
              data-group="r-leaning"
              data-states={raceGroupStateList(rLeaningRaces)}
              data-open-seat-states={raceGroupOpenSeatStateList(rLeaningRaces)}
            >
              <RaceCardGrid races={rLeaningRaces} title="Likely and Safe Republican" />
            </div>
          )}
        </>
      )}
    </div>
  );
}
