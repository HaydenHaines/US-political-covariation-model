import { Skeleton } from "@/components/ui/skeleton";

/**
 * Skeleton for county detail page — shown during navigation before the
 * async server component resolves its data fetches.
 *
 * Mirrors the rough layout: header + type badges + narrative + demographics +
 * election history chart + similar counties.
 */
export default function CountyDetailLoading() {
  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "40px 24px 80px" }}>
      {/* Breadcrumb */}
      <Skeleton className="w-56 h-4 mb-6" />

      {/* Title */}
      <Skeleton className="w-2/3 h-10 mb-4" />

      {/* Type badge + super-type badge + rating + lean */}
      <div className="flex flex-wrap gap-3 mb-8">
        <Skeleton className="w-36 h-7 rounded" />
        <Skeleton className="w-28 h-7 rounded" />
        <Skeleton className="w-20 h-7 rounded" />
        <Skeleton className="w-16 h-7 rounded" />
      </div>

      {/* Narrative block */}
      <Skeleton className="w-full h-16 mb-10" />

      {/* Demographics section */}
      <Skeleton className="w-36 h-7 mb-4" />
      <div className="space-y-2 mb-10">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="w-full h-8" />
        ))}
      </div>

      {/* Election history chart */}
      <Skeleton className="w-40 h-7 mb-4" />
      <Skeleton className="w-full h-[220px] mb-10" />

      {/* Similar counties */}
      <Skeleton className="w-40 h-7 mb-4" />
      <div className="grid grid-cols-2 gap-2">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="w-full h-10" />
        ))}
      </div>
    </div>
  );
}
