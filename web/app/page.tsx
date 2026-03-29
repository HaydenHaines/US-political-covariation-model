"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { HeroSection } from "@/components/landing/HeroSection";
import { RaceTicker } from "@/components/landing/RaceTicker";
import { EntryPoints } from "@/components/landing/EntryPoints";
import { FreshnessStamp } from "@/components/shared/FreshnessStamp";
import { ErrorAlert } from "@/components/shared/ErrorAlert";

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
      <RaceTicker races={data?.races} isLoading={isLoading} />
      <EntryPoints />

      {data && (
        <div className="flex justify-center pb-8">
          <FreshnessStamp pollCount={totalPolls} />
        </div>
      )}
    </div>
  );
}
