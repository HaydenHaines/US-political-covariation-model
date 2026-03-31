import { Skeleton } from "@/components/ui/skeleton";

/**
 * Skeleton for type detail page — shown during navigation before the
 * async server component resolves its parallel data fetches.
 *
 * Mirrors the rough layout: header + badges + narrative + demographics +
 * shift chart + similar types + member geography + county list.
 */
export default function TypeDetailLoading() {
  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "40px 24px 80px" }}>
      {/* Breadcrumb */}
      <Skeleton className="w-40 h-4 mb-6" />

      {/* Title + subtitle */}
      <Skeleton className="w-3/4 h-10 mb-2" />
      <Skeleton className="w-48 h-4 mb-6" />

      {/* Super-type badge + rating badge + lean */}
      <div className="flex gap-3 mb-8">
        <Skeleton className="w-28 h-7 rounded" />
        <Skeleton className="w-20 h-7 rounded" />
        <Skeleton className="w-16 h-7 rounded" />
      </div>

      {/* Narrative block */}
      <Skeleton className="w-full h-20 mb-10" />

      {/* Demographics section */}
      <Skeleton className="w-36 h-7 mb-2" />
      <Skeleton className="w-64 h-4 mb-4" />
      <div className="space-y-2 mb-10">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="w-full h-8" />
        ))}
      </div>

      {/* Shift history chart */}
      <Skeleton className="w-48 h-7 mb-4" />
      <Skeleton className="w-full h-[220px] mb-10" />

      {/* Similar types */}
      <Skeleton className="w-36 h-7 mb-2" />
      <Skeleton className="w-72 h-4 mb-4" />
      <div className="grid grid-cols-2 gap-3 mb-10">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="w-full h-16" />
        ))}
      </div>

      {/* Member geography map */}
      <Skeleton className="w-40 h-7 mb-2" />
      <Skeleton className="w-full h-[280px] mb-10" />

      {/* Member counties list */}
      <Skeleton className="w-44 h-7 mb-2" />
      <Skeleton className="w-56 h-4 mb-4" />
      <div className="grid grid-cols-2 gap-2">
        {Array.from({ length: 12 }).map((_, i) => (
          <Skeleton key={i} className="w-full h-8" />
        ))}
      </div>
    </div>
  );
}
