import type { Metadata } from "next";
import Link from "next/link";

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
  counties: TypeCounty[];
  demographics: Record<string, number>;
  shift_profile: Record<string, number> | null;
  narrative: string | null;
  // Extended fields from super-types lookup
  super_type_display_name?: string;
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

function stripCountySuffix(name: string | null): string {
  if (!name) return "Unknown County";
  return name.replace(/,\s*[A-Z]{2}$/, "");
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
  { key: "evangelical_share", label: "Evangelical", format: formatPct, section: "religion" },
  { key: "catholic_share", label: "Catholic", format: formatPct, section: "religion" },
  { key: "mainline_share", label: "Mainline Protestant", format: formatPct, section: "religion" },
  { key: "religious_adherence_rate", label: "Religious Adherence", format: formatPct, section: "religion" },
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

const SECTION_ORDER = ["income", "education", "race", "religion", "other"];

// ── Metadata ──────────────────────────────────────────────────────────────

type PageProps = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { id } = await params;
  const data = await fetchType(id);
  if (!data) {
    return { title: "Type Not Found — WetherVane" };
  }

  const lean = formatLean(data.mean_pred_dem_share);
  const countyWord = data.n_counties === 1 ? "county" : "counties";
  const description = data.narrative
    ? `${data.narrative} Includes ${data.n_counties} ${countyWord}. Political lean: ${lean.text}.`
    : `Electoral type ${data.type_id}: ${data.display_name}. Includes ${data.n_counties} ${countyWord}. Political lean: ${lean.text}. Explore demographics and member counties on WetherVane.`;

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
  const [data, superTypes] = await Promise.all([
    fetchType(id),
    fetchSuperTypes(),
  ]);

  if (!data) {
    return (
      <div style={{ padding: "60px 24px", textAlign: "center" }}>
        <h1 style={{ fontFamily: "var(--font-serif)" }}>Type Not Found</h1>
        <p style={{ color: "var(--color-text-muted)" }}>
          No data available for type {id}.
        </p>
        <Link href="/forecast" style={{ color: "var(--color-dem)" }}>
          Back to map
        </Link>
      </div>
    );
  }

  const lean = formatLean(data.mean_pred_dem_share);

  const siteUrl = "https://wethervane.hhaines.duckdns.org";
  const countyWord = data.n_counties === 1 ? "county" : "counties";
  const typeDescription = data.narrative
    ? `${data.narrative} Includes ${data.n_counties} ${countyWord}. Political lean: ${lean.text}.`
    : `Electoral type ${data.type_id}: ${data.display_name}. Includes ${data.n_counties} ${countyWord}. Political lean: ${lean.text}.`;
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Dataset",
    name: `${data.display_name} — WetherVane Electoral Type ${data.type_id}`,
    description: typeDescription,
    url: `${siteUrl}/type/${id}`,
    creator: {
      "@type": "Organization",
      name: "WetherVane",
      url: siteUrl,
    },
    variableMeasured: [
      {
        "@type": "PropertyValue",
        name: "Number of Counties",
        value: data.n_counties,
      },
      ...(data.mean_pred_dem_share !== null
        ? [
            {
              "@type": "PropertyValue",
              name: "Mean Predicted Democratic Vote Share",
              value: data.mean_pred_dem_share,
            },
            {
              "@type": "PropertyValue",
              name: "Political Lean",
              value: lean.text,
            },
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

  const superType = superTypes.find((st) => st.super_type_id === data.super_type_id);
  const superTypeName = superType?.display_name ?? `Super-Type ${data.super_type_id}`;

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

  // Group counties by state for display
  const countiesByState: Record<string, TypeCounty[]> = {};
  for (const county of data.counties) {
    if (!countiesByState[county.state_abbr]) {
      countiesByState[county.state_abbr] = [];
    }
    countiesByState[county.state_abbr].push(county);
  }
  const sortedStates = Object.keys(countiesByState).sort();

  return (
    <article style={{
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
        <Link href="/types" style={{ color: "var(--color-dem)", textDecoration: "none" }}>
          Types
        </Link>
        {" / "}
        <span>Type {data.type_id}</span>
      </nav>

      {/* Header */}
      <h1 style={{
        fontFamily: "var(--font-serif)",
        fontSize: 32,
        margin: "0 0 4px",
        lineHeight: 1.2,
      }}>
        {data.display_name}
      </h1>
      <p style={{
        fontSize: 14,
        color: "var(--color-text-muted)",
        margin: "0 0 20px",
      }}>
        Type {data.type_id} — {data.n_counties} {data.n_counties === 1 ? "county" : "counties"}
      </p>

      {/* Super-type + Lean badges */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
        <span style={{
          display: "inline-block",
          padding: "4px 12px",
          borderRadius: 4,
          fontSize: 13,
          background: "var(--color-bg)",
          border: "1px solid var(--color-border)",
          color: "var(--color-text-muted)",
        }}>
          {superTypeName}
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
              <div style={{
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

      {/* Member Counties */}
      {data.counties.length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2 style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            marginBottom: 8,
          }}>
            Member Counties
          </h2>
          <p style={{
            fontSize: 14,
            color: "var(--color-text-muted)",
            marginBottom: 16,
          }}>
            {data.n_counties} {data.n_counties === 1 ? "county" : "counties"} classified as <strong>{data.display_name}</strong>
          </p>

          {sortedStates.map((stateAbbr) => (
            <div key={stateAbbr} style={{ marginBottom: 20 }}>
              <h3 style={{
                fontFamily: "var(--font-serif)",
                fontSize: 15,
                color: "var(--color-text-muted)",
                marginBottom: 8,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}>
                {stateAbbr}
              </h3>
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "4px 16px",
              }}>
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
                    {stripCountySuffix(county.county_name)}
                  </Link>
                ))}
              </div>
            </div>
          ))}
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
