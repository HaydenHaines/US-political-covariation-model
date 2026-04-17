"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { useSenateScrollyContext } from "@/lib/hooks/use-senate-scrolly-context";
import { useModelVersion } from "@/lib/hooks/use-model-version";
import { HeroSection } from "@/components/landing/HeroSection";
import { FreshnessStamp } from "@/components/shared/FreshnessStamp";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { MiniMap } from "@/components/landing/MiniMap";
import { ScrollyNarrative } from "@/components/home/ScrollyNarrative";
import { Skeleton } from "@/components/ui/skeleton";

export default function LandingPage() {
  const { data, error, isLoading, mutate } = useSenateOverview();
  const { data: scrollyData, isLoading: scrollyLoading } = useSenateScrollyContext();
  const { communityCount } = useModelVersion();

  const totalPolls = data?.races.reduce((sum, r) => sum + r.n_polls, 0);

  return (
    <div>
      {error && (
        <div className="px-4 pt-6 max-w-5xl mx-auto">
          <ErrorAlert
            title="Failed to load forecast"
            message={error.message}
            retry={() => mutate()}
          />
        </div>
      )}

      {/* Hero — constrained to max-w-5xl */}
      <div className="mx-auto max-w-5xl">
        <HeroSection data={data} isLoading={isLoading} communityCount={communityCount} />
      </div>

      {/* Mini map — wider column so it spans more of the screen */}
      {data?.state_colors && (
        <div className="w-full max-w-[1200px] mx-auto px-4 mb-2">
          <MiniMap stateColors={data.state_colors} />
        </div>
      )}

      {/* Map caption + scroll invitation — back in max-w-5xl for alignment with hero text */}
      <div className="mx-auto max-w-5xl">
        {data?.state_colors && (
          <p
            className="text-xs px-4 mb-6 text-center"
            style={{ color: "var(--color-text-subtle)" }}
          >
            Map area reflects geography, not population or electoral weight.
          </p>
        )}

        {/* Scroll invitation */}
        <div className="flex flex-col items-center py-8 gap-2">
          <p
            className="text-sm font-medium tracking-wide"
            style={{ color: "var(--color-text-muted)" }}
          >
            Scroll to explore the Senate breakdown
          </p>
          {/* Animated down arrow */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="animate-bounce"
            style={{ color: "var(--color-text-subtle)" }}
            aria-hidden="true"
          >
            <path d="M12 5v14" />
            <path d="m19 12-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Divider before scrollytelling */}
      <div
        className="border-t"
        style={{ borderColor: "var(--color-border)" }}
        aria-hidden="true"
      />

      {/* Scrollytelling narrative */}
      {scrollyLoading || !scrollyData || !data ? (
        <div className="py-20 max-w-5xl mx-auto px-4">
          <div className="space-y-4 max-w-2xl">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-3/4" />
            <Skeleton className="h-4 w-48" />
          </div>
        </div>
      ) : (
        <ScrollyNarrative scrollyData={scrollyData} overviewData={data} />
      )}

      {/* Footer freshness stamp */}
      {data && (
        <div className="flex justify-center pb-8 max-w-5xl mx-auto">
          <FreshnessStamp
            pollCount={totalPolls}
            updatedAt={data.updated_at ?? undefined}
          />
        </div>
      )}
    </div>
  );
}
