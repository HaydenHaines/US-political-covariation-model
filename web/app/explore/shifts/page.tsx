import type { Metadata } from "next";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

// visx is a large bundle — load dynamically to reduce initial page JS
const ShiftSmallMultiples = dynamic(
  () =>
    import("@/components/explore/ShiftSmallMultiples").then(
      (m) => m.ShiftSmallMultiples,
    ),
  {
    ssr: false,
    loading: () => <Skeleton className="w-full h-[400px]" />,
  },
);

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

export const metadata: Metadata = {
  title: "Historical Shifts by Super-Type | WetherVane",
  description:
    "How each electoral super-type shifted politically across presidential cycles from 2000 to 2024. Discover realignment patterns, stable coalitions, and diverging communities.",
  openGraph: {
    title: "Historical Shifts by Super-Type | WetherVane",
    description:
      "Presidential Dem shift across 2000–2024 for each of WetherVane's electoral super-type families. See which communities realigned and which stayed put.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary",
    title: "Historical Shifts by Super-Type | WetherVane",
    description:
      "WetherVane electoral super-type shift trajectories across 2000–2024 presidential cycles.",
  },
};

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/**
 * Historical Shifts page — /explore/shifts
 *
 * Displays a small multiples grid with one mini chart per super-type.
 * Each chart shows mean presidential Dem shift across cycles 2000→2024.
 * All charts share the same Y scale so magnitudes are directly comparable.
 *
 * The ShiftSmallMultiples client component aggregates type-level scatter data
 * to super-type means; no dedicated API endpoint is required.
 */
export default function ExploreShiftsPage() {
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
        aria-label="Breadcrumb"
        style={{
          fontSize: 13,
          color: "var(--color-text-muted)",
          marginBottom: 24,
        }}
      >
        <Link
          href="/explore"
          style={{ color: "var(--color-dem)", textDecoration: "none" }}
        >
          Explore
        </Link>
        {" / "}
        <Link
          href="/explore/map"
          style={{ color: "var(--color-dem)", textDecoration: "none" }}
        >
          Map
        </Link>
        {" / "}
        <Link
          href="/explore/types"
          style={{ color: "var(--color-dem)", textDecoration: "none" }}
        >
          Explore Types
        </Link>
        {" / "}
        <span>Historical Shifts</span>
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
          Historical Shifts
        </h1>
        <p
          style={{
            fontSize: 17,
            lineHeight: 1.7,
            color: "var(--color-text)",
            maxWidth: 680,
            margin: 0,
            borderLeft: "3px solid var(--color-border)",
            paddingLeft: 16,
          }}
        >
          How did each electoral community shift across the last six presidential
          cycles? Each panel below tracks the mean Dem presidential shift within
          one super-type family from 2000 to 2024. Panels share the same vertical
          scale, so swings are directly comparable across communities.
        </p>
      </div>

      {/* Reading guide */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 20,
          marginBottom: 32,
          fontSize: 13,
          color: "var(--color-text-muted)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              display: "inline-block",
              width: 32,
              height: 3,
              background: "var(--color-dem)",
              borderRadius: 2,
            }}
          />
          <span>Dem shift (positive = Dem gain)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              display: "inline-block",
              width: 32,
              height: 3,
              background: "#c4707a",
              borderRadius: 2,
            }}
          />
          <span>Rep shift (negative = Rep gain)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              display: "inline-block",
              width: 32,
              height: 1,
              borderTop: "1px dashed #888",
            }}
          />
          <span>Zero baseline</span>
        </div>
      </div>

      {/* Small multiples grid */}
      <section>
        <ShiftSmallMultiples />
      </section>

      {/* Footer nav */}
      <div
        style={{
          paddingTop: 24,
          marginTop: 56,
          borderTop: "1px solid var(--color-border)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <Link
          href="/explore/types"
          style={{
            color: "var(--color-dem)",
            textDecoration: "none",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          ← All types
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
