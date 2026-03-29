import type { Metadata } from "next";
import Link from "next/link";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { DemographicsPanel } from "@/components/detail/DemographicsPanel";
import { SimilarCounties } from "@/components/detail/SimilarCounties";
import { Breadcrumbs } from "@/components/nav/Breadcrumbs";
import { marginToRating, getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import { formatMargin } from "@/lib/format";

// ── Types ─────────────────────────────────────────────────────────────────

interface SiblingCounty {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
}

interface CountyDetail {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
  dominant_type: number;
  super_type: number;
  type_display_name: string;
  super_type_display_name: string;
  narrative: string | null;
  pred_dem_share: number | null;
  demographics: Record<string, number>;
  sibling_counties: SiblingCounty[];
}

interface CountySummary {
  county_fips: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

const SITE_URL = "https://wethervane.hhaines.duckdns.org";

const STATE_NAMES: Record<string, string> = {
  AL: "Alabama", AK: "Alaska", AZ: "Arizona", AR: "Arkansas", CA: "California",
  CO: "Colorado", CT: "Connecticut", DE: "Delaware", FL: "Florida", GA: "Georgia",
  HI: "Hawaii", ID: "Idaho", IL: "Illinois", IN: "Indiana", IA: "Iowa",
  KS: "Kansas", KY: "Kentucky", LA: "Louisiana", ME: "Maine", MD: "Maryland",
  MA: "Massachusetts", MI: "Michigan", MN: "Minnesota", MS: "Mississippi",
  MO: "Missouri", MT: "Montana", NE: "Nebraska", NV: "Nevada", NH: "New Hampshire",
  NJ: "New Jersey", NM: "New Mexico", NY: "New York", NC: "North Carolina",
  ND: "North Dakota", OH: "Ohio", OK: "Oklahoma", OR: "Oregon", PA: "Pennsylvania",
  RI: "Rhode Island", SC: "South Carolina", SD: "South Dakota", TN: "Tennessee",
  TX: "Texas", UT: "Utah", VT: "Vermont", VA: "Virginia", WA: "Washington",
  WV: "West Virginia", WI: "Wisconsin", WY: "Wyoming", DC: "District of Columbia",
};

async function fetchCounty(fips: string): Promise<CountyDetail | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/counties/${fips}`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function fetchAllFips(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/counties`, {
      next: { revalidate: 86400 },
    });
    if (!res.ok) return [];
    const data: CountySummary[] = await res.json();
    return data.map((c) => c.county_fips);
  } catch {
    return [];
  }
}

function stripStateSuffix(name: string | null): string {
  if (!name) return "Unknown County";
  return name.replace(/,\s*[A-Z]{2}$/, "");
}

// ── Metadata ──────────────────────────────────────────────────────────────

type PageProps = { params: Promise<{ fips: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { fips } = await params;
  const data = await fetchCounty(fips);
  if (!data) {
    return { title: "County Not Found — WetherVane" };
  }

  const name = stripStateSuffix(data.county_name);
  const state = STATE_NAMES[data.state_abbr] || data.state_abbr;
  const lean = formatMargin(data.pred_dem_share);
  const title = `${name}, ${state} — WetherVane Electoral Profile`;
  const description = `${name} is a ${data.type_display_name} county in ${state}. Political lean: ${lean}. See demographics and 2026 forecast.`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      type: "article",
      siteName: "WetherVane",
      images: [
        {
          url: `/county/${fips}/opengraph-image`,
          width: 1200,
          height: 630,
          alt: `${name}, ${state} electoral profile`,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description: `${name} is a ${data.type_display_name} county in ${state}. Political lean: ${lean}.`,
    },
  };
}

// ── Static params ─────────────────────────────────────────────────────────

export async function generateStaticParams() {
  const fipsList = await fetchAllFips();
  return fipsList.map((fips) => ({ fips }));
}

// ── Page Component ────────────────────────────────────────────────────────

export default async function CountyPage({ params }: PageProps) {
  const { fips } = await params;
  const data = await fetchCounty(fips);

  if (!data) {
    return (
      <div style={{ padding: "60px 24px", textAlign: "center" }}>
        <h1 style={{ fontFamily: "var(--font-serif)" }}>County Not Found</h1>
        <p style={{ color: "var(--color-text-muted)" }}>
          No data available for FIPS code {fips}.
        </p>
        <Link href="/explore/map" style={{ color: "var(--color-dem)" }}>
          Back to map
        </Link>
      </div>
    );
  }

  const name = stripStateSuffix(data.county_name);
  const state = STATE_NAMES[data.state_abbr] || data.state_abbr;
  const lean = formatMargin(data.pred_dem_share);
  const superTypeHex = rgbToHex(getSuperTypeColor(data.super_type));

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Place",
    name: `${name}, ${state}`,
    description: `${name} is a ${data.type_display_name} county in ${state}. Political lean: ${lean}.`,
    url: `${SITE_URL}/county/${fips}`,
    containedInPlace: {
      "@type": "AdministrativeArea",
      name: state,
    },
    additionalProperty: [
      {
        "@type": "PropertyValue",
        name: "Electoral Type",
        value: data.type_display_name,
      },
      {
        "@type": "PropertyValue",
        name: "Super-Type",
        value: data.super_type_display_name,
      },
      ...(data.pred_dem_share !== null
        ? [
            {
              "@type": "PropertyValue",
              name: "Predicted Democratic Two-Party Vote Share",
              value: data.pred_dem_share,
            },
            {
              "@type": "PropertyValue",
              name: "Political Lean",
              value: lean,
            },
          ]
        : []),
    ],
  };

  return (
    <article
      id="main-content"
      style={{ maxWidth: 800, margin: "0 auto", padding: "40px 24px 80px" }}
    >
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <Breadcrumbs
        currentPage={name}
        extraParents={[{ label: state, href: `/explore/map?state=${data.state_abbr}` }]}
      />

      {/* Header */}
      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: 32,
          margin: "0 0 8px",
          lineHeight: 1.2,
        }}
      >
        {name}, {state}
      </h1>

      {/* Type + super-type + lean row */}
      <div
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          marginBottom: 24,
          alignItems: "center",
        }}
      >
        <Link
          href={`/type/${data.dominant_type}`}
          style={{
            display: "inline-block",
            padding: "4px 12px",
            borderRadius: 4,
            fontSize: 13,
            fontWeight: 600,
            background: "var(--color-bg)",
            border: "1px solid var(--color-border)",
            color: "var(--color-dem)",
            textDecoration: "none",
          }}
        >
          {data.type_display_name}
        </Link>
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
          {data.super_type_display_name}
        </span>
        {data.pred_dem_share !== null && (
          <RatingBadge rating={marginToRating(data.pred_dem_share)} />
        )}
        <MarginDisplay demShare={data.pred_dem_share} size="lg" />
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

      {/* Election History */}
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
        <p style={{ fontSize: 14, color: "var(--color-text-muted)", margin: 0 }}>
          County-level shift history coming soon. See the{" "}
          <a
            href={`/type/${data.dominant_type}`}
            style={{ color: "var(--color-dem)", textDecoration: "none" }}
          >
            {data.type_display_name}
          </a>{" "}
          type page for the shift profile of this county&apos;s electoral community.
        </p>
      </section>

      {/* Similar Counties */}
      {data.sibling_counties.length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 22,
              marginBottom: 16,
            }}
          >
            Similar Counties
          </h2>
          <SimilarCounties
            siblings={data.sibling_counties}
            typeName={data.type_display_name}
          />
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
          href={`/type/${data.dominant_type}`}
          style={{ color: "var(--color-dem)", textDecoration: "none", fontSize: 14 }}
        >
          ← Type: {data.type_display_name}
        </Link>
        <Link
          href={`/explore/map?focus=${fips}`}
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
