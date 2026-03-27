import type { Metadata } from "next";
import Link from "next/link";

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

// ── Helpers ───────────────────────────────────────────────────────────────

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

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

function stripCountySuffix(name: string | null): string {
  if (!name) return "Unknown County";
  return name.replace(/,\s*[A-Z]{2}$/, "");
}

function formatLean(demShare: number | null): { text: string; color: string } {
  if (demShare === null) return { text: "N/A", color: "var(--color-text-muted)" };
  const margin = Math.abs(demShare - 0.5) * 100;
  if (demShare > 0.5) {
    return { text: `D+${margin.toFixed(1)}`, color: "var(--color-dem)" };
  }
  return { text: `R+${margin.toFixed(1)}`, color: "var(--color-rep)" };
}

function formatPct(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}

function formatIncome(val: number): string {
  return `$${Math.round(val).toLocaleString("en-US")}`;
}

function formatPopulation(val: number): string {
  return Math.round(val).toLocaleString("en-US");
}

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

// ── Metadata ──────────────────────────────────────────────────────────────

type PageProps = { params: Promise<{ fips: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { fips } = await params;
  const data = await fetchCounty(fips);
  if (!data) {
    return { title: "County Not Found — WetherVane" };
  }

  const name = stripCountySuffix(data.county_name);
  const state = STATE_NAMES[data.state_abbr] || data.state_abbr;
  const lean = formatLean(data.pred_dem_share);

  return {
    title: `${name}, ${state} — WetherVane Electoral Profile`,
    description: `${name} is a ${data.type_display_name} county in ${state}. Political lean: ${lean.text}. See demographics and 2026 forecast.`,
    openGraph: {
      title: `${name}, ${state} — WetherVane Electoral Profile`,
      description: `${name} is a ${data.type_display_name} county in ${state}. Political lean: ${lean.text}. See demographics and 2026 forecast.`,
      type: "article",
      siteName: "WetherVane",
      images: [{
        url: `/county/${fips}/opengraph-image`,
        width: 1200,
        height: 630,
        alt: `${name}, ${state} electoral profile`,
      }],
    },
    twitter: {
      card: "summary_large_image",
      title: `${name}, ${state} — WetherVane Electoral Profile`,
      description: `${name} is a ${data.type_display_name} county in ${state}. Political lean: ${lean.text}.`,
    },
  };
}

// ── Demographics display config ───────────────────────────────────────────

interface DemoField {
  key: string;
  label: string;
  format: (v: number) => string;
  section: "income" | "education" | "race" | "religion" | "other";
}

const DEMO_FIELDS: DemoField[] = [
  { key: "pop_total", label: "Population", format: formatPopulation, section: "other" },
  { key: "median_hh_income", label: "Median Household Income", format: formatIncome, section: "income" },
  { key: "pct_bachelors_plus", label: "Bachelor's Degree+", format: formatPct, section: "education" },
  { key: "pct_graduate", label: "Graduate Degree", format: formatPct, section: "education" },
  { key: "pct_white_nh", label: "White (Non-Hispanic)", format: formatPct, section: "race" },
  { key: "pct_black", label: "Black", format: formatPct, section: "race" },
  { key: "pct_hispanic", label: "Hispanic", format: formatPct, section: "race" },
  { key: "pct_asian", label: "Asian", format: formatPct, section: "race" },
  { key: "pct_owner_occupied", label: "Owner-Occupied Housing", format: formatPct, section: "other" },
  { key: "pct_wfh", label: "Work From Home", format: formatPct, section: "other" },
  { key: "pct_management", label: "Management/Professional", format: formatPct, section: "other" },
  { key: "median_age", label: "Median Age", format: (v) => v.toFixed(1), section: "other" },
];

const SECTION_LABELS: Record<string, string> = {
  income: "Income",
  education: "Education",
  race: "Race & Ethnicity",
  religion: "Religion",
  other: "Other",
};

const SECTION_ORDER = ["income", "education", "race", "other"];

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
        <Link href="/forecast" style={{ color: "var(--color-dem)" }}>
          Back to map
        </Link>
      </div>
    );
  }

  const name = stripCountySuffix(data.county_name);
  const state = STATE_NAMES[data.state_abbr] || data.state_abbr;
  const lean = formatLean(data.pred_dem_share);

  // Group demographics by section
  const demoBySection: Record<string, { label: string; value: string }[]> = {};
  for (const section of SECTION_ORDER) {
    demoBySection[section] = [];
  }
  for (const field of DEMO_FIELDS) {
    const val = data.demographics[field.key];
    if (val !== undefined) {
      if (!demoBySection[field.section]) demoBySection[field.section] = [];
      demoBySection[field.section].push({
        label: field.label,
        value: field.format(val),
      });
    }
  }

  const siteUrl = "https://wethervane.hhaines.duckdns.org";
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Place",
    name: `${name}, ${state}`,
    description: `${name} is a ${data.type_display_name} county in ${state}. Political lean: ${lean.text}.`,
    url: `${siteUrl}/county/${fips}`,
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
              value: lean.text,
            },
          ]
        : []),
    ],
  };

  return (
    <article className="county-detail-article" style={{
      maxWidth: 800,
      margin: "0 auto",
      padding: "40px 24px 80px",
    }}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {/* Breadcrumb */}
      <nav style={{
        fontSize: 13,
        color: "var(--color-text-muted)",
        marginBottom: 24,
      }}>
        <Link href="/forecast" style={{ color: "var(--color-dem)", textDecoration: "none" }}>
          Map
        </Link>
        {" / "}
        <span>{data.state_abbr}</span>
        {" / "}
        <span>{name}</span>
      </nav>

      {/* Header */}
      <h1 style={{
        fontFamily: "var(--font-serif)",
        fontSize: 32,
        margin: "0 0 8px",
        lineHeight: 1.2,
      }}>
        {name}, {state}
      </h1>

      {/* Type + Lean badges */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
        <span style={{
          display: "inline-block",
          padding: "4px 12px",
          borderRadius: 4,
          fontSize: 13,
          fontWeight: 600,
          background: "var(--color-bg)",
          border: "1px solid var(--color-border)",
          color: "var(--color-text)",
        }}>
          {data.type_display_name}
        </span>
        <span style={{
          display: "inline-block",
          padding: "4px 12px",
          borderRadius: 4,
          fontSize: 13,
          background: "var(--color-bg)",
          border: "1px solid var(--color-border)",
          color: "var(--color-text-muted)",
        }}>
          {data.super_type_display_name}
        </span>
        <span style={{
          display: "inline-block",
          padding: "4px 12px",
          borderRadius: 4,
          fontSize: 14,
          fontWeight: 700,
          color: lean.color,
          background: "var(--color-surface)",
          border: `1px solid ${lean.color}`,
        }}>
          {lean.text}
        </span>
      </div>

      {/* Narrative */}
      {data.narrative && (
        <p style={{
          fontSize: 16,
          lineHeight: 1.7,
          color: "var(--color-text)",
          marginBottom: 32,
          borderLeft: "3px solid var(--color-border)",
          paddingLeft: 16,
        }}>
          {data.narrative}
        </p>
      )}

      {/* Demographics */}
      <section style={{ marginBottom: 40 }}>
        <h2 style={{
          fontFamily: "var(--font-serif)",
          fontSize: 22,
          marginBottom: 16,
        }}>
          Demographics
        </h2>

        {SECTION_ORDER.map((section) => {
          const items = demoBySection[section];
          if (!items || items.length === 0) return null;
          return (
            <div key={section} style={{ marginBottom: 20 }}>
              <h3 style={{
                fontFamily: "var(--font-serif)",
                fontSize: 15,
                color: "var(--color-text-muted)",
                marginBottom: 8,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}>
                {SECTION_LABELS[section]}
              </h3>
              <div className="county-demo-grid" style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "6px 24px",
              }}>
                {items.map((item) => (
                  <div key={item.label} style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "6px 0",
                    borderBottom: "1px solid var(--color-bg)",
                    fontSize: 14,
                  }}>
                    <span style={{ color: "var(--color-text-muted)" }}>{item.label}</span>
                    <span style={{ fontWeight: 600 }}>{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </section>

      {/* Similar Counties */}
      {data.sibling_counties.length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2 style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            marginBottom: 16,
          }}>
            Similar Counties
          </h2>
          <p style={{
            fontSize: 14,
            color: "var(--color-text-muted)",
            marginBottom: 12,
          }}>
            Other counties classified as <strong>{data.type_display_name}</strong>:
          </p>
          <div className="county-siblings-grid" style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "4px 16px",
          }}>
            {data.sibling_counties.map((s) => (
              <Link
                key={s.county_fips}
                href={`/county/${s.county_fips}`}
                style={{
                  padding: "6px 0",
                  fontSize: 14,
                  color: "var(--color-dem)",
                  textDecoration: "none",
                  borderBottom: "1px solid var(--color-bg)",
                }}
              >
                {stripCountySuffix(s.county_name)}, {s.state_abbr}
              </Link>
            ))}
          </div>
        </section>
      )}

      {/* Back to map */}
      <div style={{
        paddingTop: 24,
        borderTop: "1px solid var(--color-border)",
        textAlign: "center",
      }}>
        <Link href="/forecast" style={{
          color: "var(--color-dem)",
          textDecoration: "none",
          fontSize: 14,
          fontWeight: 600,
        }}>
          View on Map
        </Link>
      </div>
    </article>
  );
}
