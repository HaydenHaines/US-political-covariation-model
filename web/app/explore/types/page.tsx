import type { Metadata } from "next";
import { Suspense } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { TypeGrid } from "@/components/explore/TypeGrid";
import { ComparisonTable } from "@/components/explore/ComparisonTable";
import { Skeleton } from "@/components/ui/skeleton";

// visx is a large bundle — load it dynamically to keep the initial JS payload small
const ScatterPlot = dynamic(
  () => import("@/components/explore/ScatterPlot").then((m) => m.ScatterPlot),
  {
    ssr: false,
    loading: () => <Skeleton className="w-full h-[420px]" />,
  },
);

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

export const metadata: Metadata = {
  title: "Explore Electoral Types | WetherVane",
  description:
    "Browse all 100 electoral types discovered by WetherVane — behavioral families of counties that move together politically. Filter, compare, and explore via scatter plot.",
  openGraph: {
    title: "Explore Electoral Types | WetherVane",
    description:
      "100 electoral types discovered from county-level shift patterns. Browse by super-type family, explore via interactive scatter plot.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary",
    title: "Explore Electoral Types | WetherVane",
    description:
      "Browse WetherVane's 100 electoral types. Filter by name, explore demographics via scatter plot.",
  },
};

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/**
 * Types directory page — /explore/types
 *
 * Client components (TypeGrid, ScatterPlot) handle SWR data fetching.
 * This server component provides layout, metadata, and the static shell.
 */
export default function ExploreTypesPage() {
  return (
    <article
      style={{
        maxWidth: 1080,
        margin: "0 auto",
        padding: "40px 24px 80px",
      }}
    >
      {/* Breadcrumb */}
      <nav
        style={{
          fontSize: 13,
          color: "var(--color-text-muted)",
          marginBottom: 24,
        }}
      >
        <Link
          href="/explore/map"
          style={{ color: "var(--color-dem)", textDecoration: "none" }}
        >
          Map
        </Link>
        {" / "}
        <span>Explore Types</span>
      </nav>

      {/* Page header */}
      <div style={{ marginBottom: 40 }}>
        <p
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: 13,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-text-muted)",
            margin: "0 0 12px",
          }}
        >
          WetherVane
        </p>
        <h1
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 38,
            fontWeight: 700,
            margin: "0 0 16px",
            lineHeight: 1.15,
          }}
        >
          Electoral Types
        </h1>
        <p
          style={{
            fontSize: 17,
            lineHeight: 1.7,
            color: "var(--color-text)",
            maxWidth: 640,
            margin: 0,
            borderLeft: "3px solid var(--color-border)",
            paddingLeft: 16,
          }}
        >
          WetherVane discovers 100 electoral types by clustering counties on
          their shift patterns across elections. Types are grouped into
          super-type families. Each type page shows its demographic profile,
          2026 forecast lean, and every county that belongs to it.
        </p>
      </div>

      {/* Scatter plot section */}
      <section style={{ marginBottom: 56 }}>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            fontWeight: 700,
            margin: "0 0 16px",
          }}
        >
          Explore by Demographics
        </h2>
        <p
          style={{
            fontSize: 14,
            color: "var(--color-text-muted)",
            margin: "0 0 20px",
            lineHeight: 1.6,
          }}
        >
          Each dot is one electoral type. Select axes to compare demographic
          characteristics across types. Hover a dot for details.
        </p>
        <div
          style={{
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            padding: "24px",
            background: "var(--color-surface)",
            overflowX: "auto",
          }}
        >
          <ScatterPlot width={700} height={420} />
        </div>
      </section>

      {/* Type directory grid */}
      <section>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            fontWeight: 700,
            margin: "0 0 16px",
          }}
        >
          All Types
        </h2>
        <TypeGrid />
      </section>

      {/* Comparison table section */}
      <section style={{ marginTop: 56 }}>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            fontWeight: 700,
            margin: "0 0 8px",
          }}
        >
          Compare Types
        </h2>
        <p
          style={{
            fontSize: 14,
            color: "var(--color-text-muted)",
            margin: "0 0 20px",
            lineHeight: 1.6,
          }}
        >
          Select up to 4 types to compare their demographic and political
          profiles side-by-side. The comparison URL is shareable.
        </p>
        <Suspense fallback={null}>
          <ComparisonTable />
        </Suspense>
      </section>

      {/* Footer nav */}
      <div
        style={{
          paddingTop: 24,
          marginTop: 48,
          borderTop: "1px solid var(--color-border)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <Link
          href="/explore/map"
          style={{
            color: "var(--color-dem)",
            textDecoration: "none",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          ← Back to map
        </Link>
        <Link
          href="/methodology"
          style={{
            color: "var(--color-dem)",
            textDecoration: "none",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          How types are discovered →
        </Link>
      </div>
    </article>
  );
}
