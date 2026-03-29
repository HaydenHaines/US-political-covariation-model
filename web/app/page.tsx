"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { HeroSection } from "@/components/landing/HeroSection";
import { RaceTicker } from "@/components/landing/RaceTicker";
import { EntryPoints } from "@/components/landing/EntryPoints";
import { FreshnessStamp } from "@/components/shared/FreshnessStamp";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { BalanceBar } from "@/components/forecast/BalanceBar";

export default function LandingPage() {
  const { data, error, isLoading, mutate } = useSenateOverview();

  const totalPolls = data?.races.reduce((sum, r) => sum + r.n_polls, 0);

  return (
    <div className="mx-auto max-w-5xl">
      {error && (
        <div className="px-4 pt-6">
          <ErrorAlert
            title="Failed to load forecast"
            message={error.message}
            retry={() => mutate()}
          />
        </div>
      )}

      <HeroSection data={data} isLoading={isLoading} />

      {data && (
        <div className="max-w-4xl mx-auto px-4">
          <BalanceBar
            races={data.races}
            demSeats={data.dem_seats_safe}
            gopSeats={data.gop_seats_safe}
          />
        </div>
      )}

      <RaceTicker races={data?.races} isLoading={isLoading} />
      <EntryPoints />

      {data && (
        <div className="flex justify-center pb-8">
          <FreshnessStamp
            pollCount={totalPolls}
            updatedAt={data.updated_at ?? undefined}
          />
        </div>
      )}
    </div>
  );
}
