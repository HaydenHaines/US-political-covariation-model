"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { ChamberProbabilityBanner } from "@/components/forecast/ChamberProbabilityBanner";
import { FundamentalsCard } from "@/components/forecast/FundamentalsCard";
import { PollingCard } from "@/components/forecast/PollingCard";
import { NationalScenarioSliders } from "@/components/forecast/NationalScenarioSliders";
import { OverviewBlendControls } from "@/components/forecast/OverviewBlendControls";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { ELECTION_YEAR } from "@/lib/config/election";

// The API base is baked at build time via the env var; fall back to the
// relative path so Next.js rewrites handle it in production.
const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

export default function SenatePage() {
  const { data, error, isLoading, mutate } = useSenateOverview();

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

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">
        {ELECTION_YEAR} United States Senate
      </h1>

      {/* Chamber control probability — "One Big Number" anchor above the balance bar */}
      <ChamberProbabilityBanner />

      {/* National Environment — fundamentals model: approval + economy → structural shift */}
      <FundamentalsCard />

      {/* Polling coverage summary — total polls, races polled, coverage quality */}
      <PollingCard />

      {/* Scenario explorer — what-if sliders for national environment + turnout */}
      <NationalScenarioSliders />

      {/*
       * OverviewBlendControls owns the blend sliders and race card grids.
       * It starts with SSR/SWR data and updates all sections simultaneously
       * when the user adjusts the blend sliders.
       */}
      <OverviewBlendControls
        initialRaces={data.races}
        initialDemSeats={data.dem_projected}
        initialGopSeats={data.gop_projected}
        apiBase={API_BASE}
      />
    </div>
  );
}
