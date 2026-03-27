import type { Metadata } from "next";
import Link from "next/link";
import { marginLabel, formatMargin } from "@/lib/typeDisplay";

// ── Types ─────────────────────────────────────────────────────────────────

interface RacePoll {
  date: string | null;
  pollster: string | null;
  dem_share: number;
  n_sample: number | null;
}

interface TypeBreakdown {
  type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
}

interface RaceDetail {
  race: string;
  slug: string;
  state_abbr: string;
  race_type: string;
  year: number;
  prediction: number | null;
  n_counties: number;
  polls: RacePoll[];
  type_breakdown: TypeBreakdown[];
}

// ── Helpers ───────────────────────────────────────────────────────────────

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

// All 50 states + DC
const STATE_NAMES: Record<string, string> = {
  AL: "Alabama",
  AK: "Alaska",
  AZ: "Arizona",
  AR: "Arkansas",
  CA: "California",
  CO: "Colorado",
  CT: "Connecticut",
  DE: "Delaware",
  DC: "District of Columbia",
  FL: "Florida",
  GA: "Georgia",
  HI: "Hawaii",
  ID: "Idaho",
  IL: "Illinois",
  IN: "Indiana",
  IA: "Iowa",
  KS: "Kansas",
  KY: "Kentucky",
  LA: "Louisiana",
  ME: "Maine",
  MD: "Maryland",
  MA: "Massachusetts",
  MI: "Michigan",
  MN: "Minnesota",
  MS: "Mississippi",
  MO: "Missouri",
  MT: "Montana",
  NE: "Nebraska",
  NV: "Nevada",
  NH: "New Hampshire",
  NJ: "New Jersey",
  NM: "New Mexico",
  NY: "New York",
  NC: "North Carolina",
  ND: "North Dakota",
  OH: "Ohio",
  OK: "Oklahoma",
  OR: "Oregon",
  PA: "Pennsylvania",
  RI: "Rhode Island",
  SC: "South Carolina",
  SD: "South Dakota",
  TN: "Tennessee",
  TX: "Texas",
  UT: "Utah",
  VT: "Vermont",
  VA: "Virginia",
  WA: "Washington",
  WV: "West Virginia",
  WI: "Wisconsin",
  WY: "Wyoming",
};

async function fetchRaceDetail(slug: string): Promise<RaceDetail | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race/${slug}`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function fetchRaceSlugs(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race-slugs`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

function raceMarginLabel(demShare: number | null): { text: string; color: string } {
  return marginLabel(demShare, 1, "No prediction yet");
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  try {
    return new Date(dateStr).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return dateStr;
  }
}

// ── Metadata ──────────────────────────────────────────────────────────────

type PageProps = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const data = await fetchRaceDetail(slug);
  if (!data) {
    return { title: "Race Not Found — WetherVane" };
  }

  const stateName = STATE_NAMES[data.state_abbr] ?? data.state_abbr;
  const lean = raceMarginLabel(data.prediction);
  const title = `${data.year} ${stateName} ${data.race_type} | WetherVane`;
  const description = data.prediction !== null
    ? `WetherVane forecasts the ${data.year} ${stateName} ${data.race_type} race at ${lean.text}. Based on ${data.n_counties} counties and electoral type modeling.`
    : `WetherVane's forecast for the ${data.year} ${stateName} ${data.race_type} race. Explore county-level predictions and polling data.`;

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
          url: `/forecast/${slug}/opengraph-image`,
          width: 1200,
          height: 630,
          alt: `${data.year} ${stateName} ${data.race_type} forecast — WetherVane`,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
    },
  };
}

// ── Static params ─────────────────────────────────────────────────────────

export async function generateStaticParams() {
  const slugs = await fetchRaceSlugs();
  return slugs.map((slug) => ({ slug }));
}

// ── Page Component ────────────────────────────────────────────────────────

export default async function RaceDetailPage({ params }: PageProps) {
  const { slug } = await params;
  const data = await fetchRaceDetail(slug);

  if (!data) {
    return (
      <div style={{ padding: "60px 24px", textAlign: "center" }}>
        <h1 style={{ fontFamily: "var(--font-serif)" }}>Race Not Found</h1>
        <p style={{ color: "var(--color-text-muted)" }}>
          No data available for this race.
        </p>
        <Link href="/forecast" style={{ color: "var(--color-dem)" }}>
          Back to forecast
        </Link>
      </div>
    );
  }

  const stateName = STATE_NAMES[data.state_abbr] ?? data.state_abbr;
  const lean = raceMarginLabel(data.prediction);

  const siteUrl = "https://wethervane.hhaines.duckdns.org";
  const raceDescription =
    data.prediction !== null
      ? `WetherVane forecasts the ${data.year} ${stateName} ${data.race_type} race at ${lean.text}. Based on ${data.n_counties} counties and electoral type modeling.`
      : `WetherVane's forecast for the ${data.year} ${stateName} ${data.race_type} race. Explore county-level predictions and polling data.`;
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Event",
    name: `${data.year} ${stateName} ${data.race_type}`,
    description: raceDescription,
    url: `${siteUrl}/forecast/${slug}`,
    startDate: `${data.year}-11-03`,
    eventStatus: "https://schema.org/EventScheduled",
    location: {
      "@type": "AdministrativeArea",
      name: stateName,
      address: {
        "@type": "PostalAddress",
        addressCountry: "US",
        addressRegion: data.state_abbr,
      },
    },
    organizer: {
      "@type": "Organization",
      name: "WetherVane",
      url: siteUrl,
    },
    additionalProperty: [
      {
        "@type": "PropertyValue",
        name: "Race Type",
        value: data.race_type,
      },
      {
        "@type": "PropertyValue",
        name: "Counties in Model",
        value: data.n_counties,
      },
      ...(data.prediction !== null
        ? [
            {
              "@type": "PropertyValue",
              name: "Model Predicted Democratic Two-Party Vote Share",
              value: data.prediction,
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
    <article id="main-content" style={{
      maxWidth: 800,
      margin: "0 auto",
      padding: "40px 24px 80px",
    }}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {/* Breadcrumb */}
      <nav aria-label="breadcrumb" style={{
        fontSize: 13,
        color: "var(--color-text-muted)",
        marginBottom: 24,
      }}>
        <ol style={{
          listStyle: "none",
          margin: 0,
          padding: 0,
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: "0 4px",
        }}>
          <li>
            <Link href="/" style={{ color: "var(--color-dem)", textDecoration: "none" }}>
              Home
            </Link>
          </li>
          <li aria-hidden="true" style={{ userSelect: "none" }}>/</li>
          <li>
            <Link href="/forecast" style={{ color: "var(--color-dem)", textDecoration: "none" }}>
              Forecast
            </Link>
          </li>
          <li aria-hidden="true" style={{ userSelect: "none" }}>/</li>
          <li aria-current="page">
            {data.year} {stateName} {data.race_type}
          </li>
        </ol>
      </nav>

      {/* Header */}
      <h1 style={{
        fontFamily: "var(--font-serif)",
        fontSize: 32,
        margin: "0 0 4px",
        lineHeight: 1.2,
      }}>
        {data.year} {stateName} {data.race_type}
      </h1>
      <p style={{
        fontSize: 14,
        color: "var(--color-text-muted)",
        margin: "0 0 20px",
      }}>
        {data.n_counties} {data.n_counties === 1 ? "county" : "counties"} in model
      </p>

      {/* State badge + predicted outcome badge */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 32 }}>
        <span style={{
          display: "inline-block",
          padding: "4px 12px",
          borderRadius: 4,
          fontSize: 13,
          background: "var(--color-bg)",
          border: "1px solid var(--color-border)",
          color: "var(--color-text-muted)",
          fontWeight: 600,
        }}>
          {data.state_abbr}
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

      {/* Prediction summary */}
      <section style={{ marginBottom: 40 }}>
        <h2 style={{
          fontFamily: "var(--font-serif)",
          fontSize: 22,
          marginBottom: 16,
        }}>
          Model Prediction
        </h2>
        {data.prediction !== null ? (
          <div style={{
            padding: "20px 24px",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 6,
          }}>
            <p style={{ fontSize: 16, margin: "0 0 8px", color: "var(--color-text)" }}>
              Our model predicts a{" "}
              <strong style={{ color: lean.color }}>{lean.text}</strong>{" "}
              margin in the {data.year} {stateName} {data.race_type} race.
            </p>
            <p style={{ fontSize: 14, margin: 0, color: "var(--color-text-muted)" }}>
              Based on electoral type modeling across {data.n_counties} counties.
              This is a structural forecast — not a polling average.
            </p>
          </div>
        ) : (
          <p style={{ color: "var(--color-text-muted)", fontSize: 15 }}>
            No county predictions available for this race yet.
          </p>
        )}
      </section>

      {/* Recent polls table */}
      <section style={{ marginBottom: 40 }}>
        <h2 style={{
          fontFamily: "var(--font-serif)",
          fontSize: 22,
          marginBottom: 16,
        }}>
          Recent Polls
        </h2>
        {data.polls.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: 14,
            }}>
              <thead>
                <tr style={{ borderBottom: "2px solid var(--color-border)" }}>
                  <th style={{ textAlign: "left", padding: "8px 12px 8px 0", color: "var(--color-text-muted)", fontWeight: 600 }}>Date</th>
                  <th style={{ textAlign: "left", padding: "8px 12px", color: "var(--color-text-muted)", fontWeight: 600 }}>Pollster</th>
                  <th style={{ textAlign: "right", padding: "8px 12px", color: "var(--color-text-muted)", fontWeight: 600 }}>Margin</th>
                  <th style={{ textAlign: "right", padding: "8px 0 8px 12px", color: "var(--color-text-muted)", fontWeight: 600 }}>Sample</th>
                </tr>
              </thead>
              <tbody>
                {data.polls.map((poll, i) => {
                  const pollMargin = marginLabel(poll.dem_share);
                  return (
                    <tr key={i} style={{ borderBottom: "1px solid var(--color-bg)" }}>
                      <td style={{ padding: "8px 12px 8px 0", color: "var(--color-text-muted)" }}>
                        {formatDate(poll.date)}
                      </td>
                      <td style={{ padding: "8px 12px" }}>
                        {poll.pollster ?? "Unknown"}
                      </td>
                      <td style={{ textAlign: "right", padding: "8px 12px", color: pollMargin.color, fontWeight: 600 }}>
                        {pollMargin.text}
                      </td>
                      <td style={{ textAlign: "right", padding: "8px 0 8px 12px", color: "var(--color-text-muted)" }}>
                        {poll.n_sample !== null ? poll.n_sample.toLocaleString("en-US") : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: "var(--color-text-muted)", fontSize: 15 }}>
            No polls available yet for this race.
          </p>
        )}
      </section>

      {/* Electoral types breakdown */}
      {data.type_breakdown.length > 0 && (
        <section style={{ marginBottom: 40 }}>
          <h2 style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            marginBottom: 8,
          }}>
            Electoral Types in {stateName}
          </h2>
          <p style={{
            fontSize: 14,
            color: "var(--color-text-muted)",
            marginBottom: 16,
          }}>
            The most common electoral types in this state, by county count.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {data.type_breakdown.map((t) => {
              const typeLean = marginLabel(t.mean_pred_dem_share);
              return (
                <div key={t.type_id} style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "10px 16px",
                  background: "var(--color-surface)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 4,
                  gap: 16,
                }}>
                  <div>
                    <Link
                      href={`/type/${t.type_id}`}
                      style={{
                        fontSize: 15,
                        fontWeight: 600,
                        color: "var(--color-dem)",
                        textDecoration: "none",
                      }}
                    >
                      {t.display_name}
                    </Link>
                    <span style={{
                      fontSize: 13,
                      color: "var(--color-text-muted)",
                      marginLeft: 10,
                    }}>
                      {t.n_counties} {t.n_counties === 1 ? "county" : "counties"}
                    </span>
                  </div>
                  <span style={{
                    fontSize: 14,
                    fontWeight: 700,
                    color: typeLean.color,
                    flexShrink: 0,
                  }}>
                    {typeLean.text}
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Back to forecast */}
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
          Back to Forecast
        </Link>
      </div>
    </article>
  );
}
