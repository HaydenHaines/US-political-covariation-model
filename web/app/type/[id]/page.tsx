import type { Metadata } from "next";
import Link from "next/link";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { DemographicsPanel } from "@/components/detail/DemographicsPanel";
import { ShiftHistoryChart } from "@/components/detail/ShiftHistoryChart";
import { CorrelatedTypes } from "@/components/detail/CorrelatedTypes";
import { MemberGeography } from "@/components/detail/MemberGeography";
import { Breadcrumbs } from "@/components/nav/Breadcrumbs";
import { marginToRating, getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import { formatMargin } from "@/lib/format";

// ── Types ─────────────────────────────────────────────────────────────────

interface TypeCounty {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
}

interface TypeDetail {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  demographics: Record<string, number>;
  shift_profile: Record<string, number> | null;
  narrative: string | null;
  counties: TypeCounty[];
}

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

const SITE_URL = "https://wethervane.hhaines.duckdns.org";

async function fetchType(id: string): Promise<TypeDetail | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/types/${id}`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
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

async function fetchAllTypes(): Promise<TypeSummary[]> {
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

function stripStateSuffix(name: string | null): string {
  if (!name) return "Unknown County";
  return name.replace(/,\s*[A-Z]{2}$/, "");
}

// ── Metadata ──────────────────────────────────────────────────────────────

type PageProps = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { id } = await params;
  const data = await fetchType(id);
  if (!data) {
    return { title: "Type Not Found — WetherVane" };
  }

  const lean = formatMargin(data.mean_pred_dem_share);
  const countyWord = data.n_counties === 1 ? "county" : "counties";
  const description = data.narrative
    ? `${data.narrative} Includes ${data.n_counties} ${countyWord}. Political lean: ${lean}.`
    : `Electoral type ${data.type_id}: ${data.display_name}. Includes ${data.n_counties} ${countyWord}. Political lean: ${lean}. Explore demographics and member counties on WetherVane.`;

  return {
    title: `Type ${data.type_id}: ${data.display_name} | WetherVane`,
    description,
    openGraph: {
      title: `Type ${data.type_id}: ${data.display_name} | WetherVane`,
      description,
      type: "article",
      siteName: "WetherVane",
      images: [
        {
          url: `/type/${id}/opengraph-image`,
          width: 1200,
          height: 630,
          alt: `${data.display_name} — WetherVane electoral type`,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: `Type ${data.type_id}: ${data.display_name} | WetherVane`,
      description,
    },
  };
}

// ── Static params ─────────────────────────────────────────────────────────

export function generateStaticParams() {
  // Pre-generate all 100 type pages (IDs 0-99)
  return Array.from({ length: 100 }, (_, i) => ({ id: String(i) }));
}

// ── Page Component ────────────────────────────────────────────────────────

export default async function TypePage({ params }: PageProps) {
  const { id } = await params;
  const [data, superTypes, allTypes] = await Promise.all([
    fetchType(id),
    fetchSuperTypes(),
    fetchAllTypes(),
  ]);

  if (!data) {
    return (
      <div style={{ padding: "60px 24px", textAlign: "center" }}>
        <h1 style={{ fontFamily: "var(--font-serif)" }}>Type Not Found</h1>
        <p style={{ color: "var(--color-text-muted)" }}>
          No data available for type {id}.
        </p>
        <Link href="/explore/map" style={{ color: "var(--color-dem)" }}>
          Back to map
        </Link>
      </div>
    );
  }

  const superType = superTypes.find((st) => st.super_type_id === data.super_type_id);
  const superTypeName = superType?.display_name ?? `Super-Type ${data.super_type_id}`;
  const superTypeHex = rgbToHex(getSuperTypeColor(data.super_type_id));
  const countyWord = data.n_counties === 1 ? "county" : "counties";
  const lean = formatMargin(data.mean_pred_dem_share);

  // Group counties by state
  const countiesByState: Record<string, TypeCounty[]> = {};
  for (const county of data.counties) {
    if (!countiesByState[county.state_abbr]) {
      countiesByState[county.state_abbr] = [];
    }
    countiesByState[county.state_abbr].push(county);
  }
  const sortedStates = Object.keys(countiesByState).sort();

  const typeDescription = data.narrative
    ? `${data.narrative} Includes ${data.n_counties} ${countyWord}. Political lean: ${lean}.`
    : `Electoral type ${data.type_id}: ${data.display_name}. Includes ${data.n_counties} ${countyWord}. Political lean: ${lean}.`;

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Dataset",
    name: `${data.display_name} — WetherVane Electoral Type ${data.type_id}`,
    description: typeDescription,
    url: `${SITE_URL}/type/${id}`,
    creator: { "@type": "Organization", name: "WetherVane", url: SITE_URL },
    variableMeasured: [
      { "@type": "PropertyValue", name: "Number of Counties", value: data.n_counties },
      ...(data.mean_pred_dem_share !== null
        ? [
            {
              "@type": "PropertyValue",
              name: "Mean Predicted Democratic Vote Share",
              value: data.mean_pred_dem_share,
            },
            { "@type": "PropertyValue", name: "Political Lean", value: lean },
          ]
        : []),
    ],
    keywords: [
      "electoral type",
      "political forecast",
      "county clustering",
      data.display_name,
      "2026 midterms",
    ],
  };

  return (
    <article
      style={{ maxWidth: 800, margin: "0 auto", padding: "40px 24px 80px" }}
    >
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <Breadcrumbs currentPage={`Type ${data.type_id}`} />

      {/* Header */}
      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: 32,
          margin: "0 0 4px",
          lineHeight: 1.2,
        }}
      >
        {data.display_name}
      </h1>
      <p style={{ fontSize: 14, color: "var(--color-text-muted)", margin: "0 0 20px" }}>
        Type {data.type_id} — {data.n_counties} {countyWord}
      </p>

      {/* Super-type badge + lean */}
      <div
        style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24, alignItems: "center" }}
      >
        <span
          style={{
            display: "inline-block",
            padding: "4px 12px",
            borderRadius: 4,
            fontSize: 13,
            fontWeight: 600,
            background: superTypeHex + "22",
            border: `1px solid ${superTypeHex}`,
            color: superTypeHex,
          }}
        >
          {superTypeName}
        </span>
        {data.mean_pred_dem_share !== null && (
          <RatingBadge rating={marginToRating(data.mean_pred_dem_share)} />
        )}
        <MarginDisplay demShare={data.mean_pred_dem_share} size="lg" />
      </div>

      {/* Narrative */}
      {data.narrative && (
        <p
          style={{
            fontSize: 16,
            lineHeight: 1.7,
            color: "var(--color-text)",
            marginBottom: 32,
            borderLeft: "3px solid var(--color-border)",
            paddingLeft: 16,
          }}
        >
          {data.narrative}
        </p>
      )}

      {/* Demographics */}
      {Object.keys(data.demographics).length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 22,
              marginBottom: 16,
            }}
          >
            Demographics
          </h2>
          <DemographicsPanel demographics={data.demographics} />
        </section>
      )}

      {/* Shift History */}
      {data.shift_profile && Object.keys(data.shift_profile).length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 22,
              marginBottom: 16,
            }}
          >
            Electoral Shift History
          </h2>
          <ShiftHistoryChart shiftProfile={data.shift_profile} />
        </section>
      )}

      {/* Similar Types */}
      <section style={{ marginBottom: 40 }}>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            marginBottom: 8,
          }}
        >
          Similar Types
        </h2>
        <p
          style={{
            fontSize: 14,
            color: "var(--color-text-muted)",
            marginBottom: 16,
          }}
        >
          Other types in the <strong>{superTypeName}</strong> super-type that
          move together structurally.
        </p>
        <CorrelatedTypes
          allTypes={allTypes}
          superTypes={superTypes}
          currentTypeId={data.type_id}
          superTypeId={data.super_type_id}
        />
      </section>

      {/* Member Geography */}
      {data.counties.length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 22,
              marginBottom: 8,
            }}
          >
            Member Geography
          </h2>
          <p
            style={{
              fontSize: 14,
              color: "var(--color-text-muted)",
              marginBottom: 16,
            }}
          >
            Counties classified as <strong>{data.display_name}</strong>{" "}
            highlighted in{" "}
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: superTypeHex,
                verticalAlign: "middle",
              }}
            />{" "}
            {superTypeName} color.
          </p>
          <MemberGeography
            typeId={data.type_id}
            superTypeId={data.super_type_id}
            counties={data.counties}
          />
        </section>
      )}

      {/* Member Counties */}
      {data.counties.length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 22,
              marginBottom: 8,
            }}
          >
            Member Counties
          </h2>
          <p
            style={{
              fontSize: 14,
              color: "var(--color-text-muted)",
              marginBottom: 16,
            }}
          >
            {data.n_counties} {countyWord} classified as{" "}
            <strong>{data.display_name}</strong>
          </p>

          {sortedStates.map((stateAbbr) => (
            <div key={stateAbbr} style={{ marginBottom: 20 }}>
              <h3
                style={{
                  fontFamily: "var(--font-serif)",
                  fontSize: 13,
                  fontWeight: 600,
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                  marginBottom: 8,
                }}
              >
                {stateAbbr}
              </h3>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "4px 16px",
                }}
              >
                {countiesByState[stateAbbr].map((county) => (
                  <Link
                    key={county.county_fips}
                    href={`/county/${county.county_fips}`}
                    style={{
                      padding: "6px 0",
                      fontSize: 14,
                      color: "var(--color-dem)",
                      textDecoration: "none",
                      borderBottom: "1px solid var(--color-bg)",
                    }}
                  >
                    {stripStateSuffix(county.county_name)}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </section>
      )}

      {/* Footer nav */}
      <div
        style={{
          paddingTop: 24,
          borderTop: "1px solid var(--color-border)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Link
          href="/types"
          style={{ color: "var(--color-dem)", textDecoration: "none", fontSize: 14 }}
        >
          ← All Types
        </Link>
        <Link
          href={`/explore/map?type=${data.type_id}`}
          style={{
            color: "var(--color-dem)",
            textDecoration: "none",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          View on Map
        </Link>
      </div>
    </article>
  );
}
