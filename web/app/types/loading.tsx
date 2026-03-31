import { Skeleton } from "@/components/ui/skeleton";

/**
 * Skeleton for the types directory page — shown during navigation before the
 * async server component resolves its data fetches.
 *
 * Mirrors the rough layout: header + stat cards + scatter plot +
 * comparison table placeholder + type card grid.
 */
export default function TypesLoading() {
  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "40px 24px 80px" }}>
      {/* Breadcrumb */}
      <Skeleton className="w-32 h-4 mb-6" />

      {/* Sub-nav tabs */}
      <div className="flex gap-4 mb-6">
        <Skeleton className="w-12 h-5" />
        <Skeleton className="w-10 h-5" />
        <Skeleton className="w-14 h-5" />
      </div>

      {/* Page title + blurb */}
      <Skeleton className="w-64 h-10 mb-4" />
      <Skeleton className="w-full h-20 mb-10" />

      {/* Summary stat cards */}
      <div className="grid grid-cols-3 gap-3 mb-10">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="w-full h-16 rounded" />
        ))}
      </div>

      {/* Jump nav bar */}
      <Skeleton className="w-full h-10 mb-8 rounded" />

      {/* Scatter plot section */}
      <Skeleton className="w-48 h-7 mb-2" />
      <Skeleton className="w-full h-4 mb-4" />
      <Skeleton className="w-full h-[420px] mb-14 rounded" />

      {/* Comparison table section */}
      <Skeleton className="w-36 h-7 mb-2" />
      <Skeleton className="w-full h-4 mb-4" />
      <Skeleton className="w-full h-12 mb-2 rounded" />
      <Skeleton className="w-full h-40 mb-14 rounded" />

      {/* Type cards grid — one super-type section */}
      <Skeleton className="w-48 h-7 mb-4" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: 12,
          marginBottom: 48,
        }}
      >
        {Array.from({ length: 12 }).map((_, i) => (
          <Skeleton key={i} className="w-full h-20 rounded" />
        ))}
      </div>
    </div>
  );
}
