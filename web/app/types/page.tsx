import type { Metadata } from "next";
import { Suspense } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
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

// ── Types ─────────────────────────────────────────────────────────────────

interface TypeSummary {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  median_hh_income: number | null;
  pct_bachelors_plus: number | null;
  pct_white_nh: number | null;
  log_pop_density: number | null;
}

interface SuperTypeSummary {
  super_type_id: number;
  display_name: string;
  member_type_ids: number[];
  n_counties: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

async function fetchTypes(): Promise<TypeSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/types`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function fetchSuperTypes(): Promise<SuperTypeSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/super-types`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

function formatLean(demShare: number | null): { text: string; color: string } {
  if (demShare === null) return { text: "—", color: "var(--color-text-muted)" };
  const margin = Math.abs(demShare - 0.5) * 100;
  if (demShare > 0.5) {
    return { text: `D+${margin.toFixed(1)}`, color: "var(--color-dem)" };
  }
  return { text: `R+${margin.toFixed(1)}`, color: "var(--color-rep)" };
}

function formatIncome(val: number | null): string {
  if (val === null) return "—";
  return `$${Math.round(val / 1000)}K`;
}

function formatPct(val: number | null): string {
  if (val === null) return "—";
  return `${(val * 100).toFixed(0)}%`;
}

// ── Metadata ──────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "Electoral Types Index | WetherVane",
  description:
    "Browse all 100 electoral types discovered by WetherVane — behavioral families of counties that move together politically. Each type links to its demographic profile, member counties, and 2026 forecast lean.",
  openGraph: {
    title: "Electoral Types Index | WetherVane",
    description:
      "100 electoral types discovered from county-level shift patterns. Browse by super-type family, see political lean, demographics, and member counties.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary",
    title: "Electoral Types Index | WetherVane",
    description:
      "Browse all 100 electoral types discovered by WetherVane. Each type links to demographics, member counties, and 2026 forecast lean.",
  },
};

// ── Page Component ────────────────────────────────────────────────────────

export default async function TypesPage() {
  const [types, superTypes] = await Promise.all([fetchTypes(), fetchSuperTypes()]);

  // Build super-type lookup
  const superTypeMap = new Map<number, SuperTypeSummary>(
    superTypes.map((st) => [st.super_type_id, st])
  );

  // Group types by super-type
  const bySuperType = new Map<number, TypeSummary[]>();
  for (const t of types) {
    const group = bySuperType.get(t.super_type_id) ?? [];
    group.push(t);
    bySuperType.set(t.super_type_id, group);
  }

  // Sorted super-type IDs
  const superTypeIds = Array.from(bySuperType.keys()).sort((a, b) => a - b);

  // Fallback: if API unavailable, still render something useful
  const isEmpty = types.length === 0;

  return (
    <article
      style={{
        maxWidth: 960,
        margin: "0 auto",
        padding: "40px 24px 80px",
      }}
    >
      {/* Breadcrumb */}
      <nav style={{ fontSize: 13, color: "var(--color-text-muted)", marginBottom: 24 }}>
        <Link href="/explore/map" style={{ color: "var(--color-dem)", textDecoration: "none" }}>
          Map
        </Link>
        {" / "}
        <span>Types</span>
      </nav>

      {/* Explore sub-navigation */}
      <div className="flex gap-4 text-sm mb-6">
        <span className="font-semibold text-[var(--color-text)]">Types</span>
        <a href="/explore/map" className="text-[var(--color-text-subtle)] hover:text-[var(--color-text)]">Map</a>
        <a href="/explore/shifts" className="text-[var(--color-text-subtle)] hover:text-[var(--color-text)]">Shifts</a>
      </div>

      {/* Header */}
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
          their shift patterns across elections. Types are grouped into 5
          super-type families. Each type page shows its demographic profile,
          2026 forecast lean, and every county that belongs to it.
        </p>
      </div>

      {/* Summary stats bar */}
      {!isEmpty && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
            gap: 12,
            marginBottom: 40,
          }}
        >
          {[
            { label: "Fine types", value: String(types.length) },
            { label: "Super-types", value: String(superTypes.length || superTypeIds.length) },
            { label: "Counties covered", value: types.reduce((s, t) => s + t.n_counties, 0).toLocaleString("en-US") },
          ].map(({ label, value }) => (
            <div
              key={label}
              style={{
                padding: "14px 16px",
                border: "1px solid var(--color-border)",
                borderRadius: 6,
                background: "var(--color-surface)",
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: "var(--color-text-muted)",
                  marginBottom: 4,
                }}
              >
                {label}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-serif)",
                  fontSize: 24,
                  fontWeight: 700,
                  lineHeight: 1.1,
                }}
              >
                {value}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* API unavailable fallback */}
      {isEmpty && (
        <div
          style={{
            padding: "32px 24px",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            background: "var(--color-surface)",
            textAlign: "center",
            marginBottom: 40,
          }}
        >
          <p style={{ color: "var(--color-text-muted)", margin: 0 }}>
            Type data is loading. If this persists, the model API may be
            temporarily unavailable.
          </p>
        </div>
      )}

      {/* Scatter plot — explore types by demographic axes */}
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

      {/* Comparison table — side-by-side type comparison */}
      <section style={{ marginBottom: 56 }}>
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

      {/* Types grouped by super-type */}
      {superTypeIds.map((superTypeId) => {
        const st = superTypeMap.get(superTypeId);
        const members = bySuperType.get(superTypeId) ?? [];
        const stName = st?.display_name ?? `Super-Type ${superTypeId}`;

        return (
          <section key={superTypeId} style={{ marginBottom: 48 }}>
            {/* Super-type header */}
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: 12,
                marginBottom: 16,
                paddingBottom: 10,
                borderBottom: "2px solid var(--color-text)",
              }}
            >
              <h2
                style={{
                  fontFamily: "var(--font-serif)",
                  fontSize: 22,
                  fontWeight: 700,
                  margin: 0,
                }}
              >
                {stName}
              </h2>
              <span style={{ fontSize: 14, color: "var(--color-text-muted)" }}>
                {members.length} {members.length === 1 ? "type" : "types"} ·{" "}
                {members.reduce((s, t) => s + t.n_counties, 0).toLocaleString("en-US")} counties
              </span>
            </div>

            {/* Type cards grid */}
            <div
              className="types-grid"
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                gap: 12,
              }}
            >
              {members.map((t) => {
                const lean = formatLean(t.mean_pred_dem_share);
                return (
                  <Link
                    key={t.type_id}
                    href={`/type/${t.type_id}`}
                    style={{ textDecoration: "none", display: "block" }}
                  >
                    <div
                      style={{
                        padding: "14px 16px",
                        border: "1px solid var(--color-border)",
                        borderRadius: 6,
                        background: "var(--color-surface)",
                        transition: "border-color 0.15s",
                        cursor: "pointer",
                      }}
                    >
                      {/* Type name + lean */}
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "flex-start",
                          gap: 8,
                          marginBottom: 8,
                        }}
                      >
                        <div>
                          <span
                            style={{
                              fontSize: 11,
                              color: "var(--color-text-muted)",
                              fontFamily: "var(--font-sans)",
                              display: "block",
                              marginBottom: 2,
                            }}
                          >
                            Type {t.type_id}
                          </span>
                          <span
                            style={{
                              fontFamily: "var(--font-serif)",
                              fontSize: 15,
                              fontWeight: 700,
                              color: "var(--color-text)",
                              lineHeight: 1.3,
                            }}
                          >
                            {t.display_name}
                          </span>
                        </div>
                        <span
                          style={{
                            fontSize: 13,
                            fontWeight: 700,
                            color: lean.color,
                            whiteSpace: "nowrap",
                            flexShrink: 0,
                            paddingTop: 2,
                          }}
                        >
                          {lean.text}
                        </span>
                      </div>

                      {/* Key stats row */}
                      <div
                        style={{
                          display: "flex",
                          gap: 16,
                          fontSize: 12,
                          color: "var(--color-text-muted)",
                        }}
                      >
                        <span>{t.n_counties} counties</span>
                        {t.median_hh_income !== null && (
                          <span>{formatIncome(t.median_hh_income)} income</span>
                        )}
                        {t.pct_bachelors_plus !== null && (
                          <span>{formatPct(t.pct_bachelors_plus)} college</span>
                        )}
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </section>
        );
      })}

      {/* Footer nav */}
      <div
        style={{
          paddingTop: 24,
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
