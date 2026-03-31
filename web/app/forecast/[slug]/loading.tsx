import { Skeleton } from "@/components/ui/skeleton";

/**
 * Skeleton for race detail page — shown during navigation before the
 * async server component resolves its data fetches.
 *
 * Mirrors the rough layout: breadcrumb + hero + dotplot + polls + type breakdown.
 */
export default function RaceDetailLoading() {
  return (
    <div className="max-w-2xl mx-auto py-8 px-4 pb-20">
      {/* Breadcrumb */}
      <Skeleton className="w-48 h-4 mb-6" />

      {/* Hero: headline + CI bar + dotplot */}
      <div className="space-y-4 mb-10">
        <Skeleton className="w-3/4 h-12" />
        <Skeleton className="w-1/2 h-6" />
        <Skeleton className="w-full h-[160px] mt-6" />
      </div>

      {/* Forecast blend sliders */}
      <div className="space-y-3 mb-10">
        <Skeleton className="w-32 h-5" />
        <Skeleton className="w-full h-10" />
        <Skeleton className="w-full h-10" />
      </div>

      {/* Polls section */}
      <div className="mb-10">
        <Skeleton className="w-36 h-7 mb-4" />
        {/* Poll trend chart */}
        <Skeleton className="w-full h-[220px] mb-6" />
        {/* Poll table rows */}
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="w-full h-10" />
          ))}
        </div>
      </div>

      {/* Electoral types breakdown */}
      <div className="mb-10">
        <Skeleton className="w-56 h-7 mb-4" />
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="w-full h-12" />
          ))}
        </div>
      </div>
    </div>
  );
}
