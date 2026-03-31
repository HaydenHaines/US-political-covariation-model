import type { SenateOverviewData } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

interface HeroSectionProps {
  data: SenateOverviewData | undefined;
  isLoading: boolean;
  /** Number of electoral communities in the current model (from /model/version). */
  communityCount: number;
}

export function HeroSection({ data, isLoading, communityCount }: HeroSectionProps) {
  if (isLoading || !data) {
    return (
      <section className="flex flex-col items-center gap-4 py-12 text-center">
        <Skeleton className="h-8 w-80" />
        <div className="flex items-center gap-4">
          <Skeleton className="h-20 w-24" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-20 w-24" />
        </div>
        <Skeleton className="h-5 w-64" />
      </section>
    );
  }

  return (
    <section className="flex flex-col items-center gap-4 py-12 text-center">
      <h1
        className="text-2xl font-bold sm:text-3xl"
        style={{ color: "var(--color-text)" }}
      >
        {data.headline}
      </h1>

      <div className="flex items-baseline gap-3">
        <span
          className="font-mono text-6xl font-bold tracking-tight sm:text-7xl"
          style={{ color: "var(--forecast-safe-d)" }}
        >
          {data.dem_projected}
          <span className="ml-1 text-2xl font-semibold sm:text-3xl">D</span>
        </span>

        <span
          className="text-2xl font-light sm:text-3xl"
          style={{ color: "var(--color-text-muted)" }}
        >
          &ndash;
        </span>

        <span
          className="font-mono text-6xl font-bold tracking-tight sm:text-7xl"
          style={{ color: "var(--forecast-safe-r)" }}
        >
          {data.gop_projected}
          <span className="ml-1 text-2xl font-semibold sm:text-3xl">R</span>
        </span>
      </div>

      <p
        className="max-w-md text-base"
        style={{ color: "var(--color-text-muted)" }}
      >
        {data.subtitle}
      </p>

      <p
        className="max-w-lg text-sm"
        style={{ color: "var(--color-text-subtle, var(--color-text-muted))", opacity: 0.75 }}
      >
        Based on {communityCount} electoral communities discovered from how places move together — not polls alone.
      </p>
    </section>
  );
}
