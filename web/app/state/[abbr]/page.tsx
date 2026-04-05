import type { Metadata } from "next";
import Link from "next/link";
import { RaceCard } from "@/components/forecast/RaceCard";
import { StateCountyTable } from "@/components/state/StateCountyTable";
import type { CountyTableRow } from "@/components/state/StateCountyTable";
import { StateTypeDistribution } from "@/components/state/StateTypeDistribution";
import type { StateTypeItem } from "@/components/state/StateTypeDistribution";
import { marginToRating } from "@/lib/config/palette";
import { STATE_NAMES } from "@/lib/config/states";
import type { SenateRaceData } from "@/lib/api";

// ── Constants ──────────────────────────────────────────────────────────────

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

const SITE_URL = "https://wethervane.hhaines.duckdns.org";

// All 50 states + DC (must match STATE_NAMES keys)
const ALL_STATE_ABBRS = Object.keys(STATE_NAMES) as string[];

// Revalidate every hour — model predictions change infrequently
export const revalidate = 3600;

// ── API Types ──────────────────────────────────────────────────────────────

interface RaceMetadata {
  race_id: string;
  slug: string;
  race_type: string;
  state: string;
  year: number;
  has_predictions: boolean;
  n_polls: number;
}

interface CountyRow {
  county_fips: string;
  state_abbr: string;
  community_id: number | null;
  dominant_type: number | null;
  super_type: number | null;
  pred_dem_share: number | null;
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

interface TypeDetail {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  counties: Array<{ county_fips: string; county_name: string | null; state_abbr: string }>;
  demographics: Record<string, number>;
  shift_profile: Record<string, number> | null;
  narrative: string | null;
}

interface CountyDetail {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
  dominant_type: number;
  super_type: number;
  type_display_name: string;
  pred_dem_share: number | null;
}

// ── Data fetchers ──────────────────────────────────────────────────────────

async function fetchRaceMetadata(): Promise<RaceMetadata[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race-metadata`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function fetchRaceDetail(slug: string): Promise<{
  prediction: number | null;
  n_polls: number;
  rating: string;
  state: string;
  race_type: string;
} | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race/${slug}`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    const data = await res.json();
    return {
      prediction: data.prediction,
      n_polls: data.n_polls ?? data.polls?.length ?? 0,
      rating: data.prediction !== null ? marginToRating(data.prediction) : "tossup",
      state: STATE_NAMES[data.state_abbr] ?? data.state_abbr,
      race_type: data.race_type,
    };
  } catch {
    return null;
  }
}

async function fetchAllCounties(): Promise<CountyRow[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/counties`, {
      next: { revalidate: 86400 },
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
      next: { revalidate: 86400 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function fetchCountyDetail(fips: string): Promise<CountyDetail | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/counties/${fips}`, {
      next: { revalidate: 86400 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ── Composite state data ───────────────────────────────────────────────────

interface StateData {
  abbr: string;
  name: string;
  races: SenateRaceData[];
  counties: CountyTableRow[];
  typeDistribution: StateTypeItem[];
  totalCounties: number;
}

async function fetchStateData(abbr: string): Promise<StateData | null> {
  const upperAbbr = abbr.toUpperCase();
  const stateName = STATE_NAMES[upperAbbr];
  if (!stateName) return null;

  // Fetch all data in parallel
  const [raceMetadata, allCounties, allTypes] = await Promise.all([
    fetchRaceMetadata(),
    fetchAllCounties(),
    fetchAllTypes(),
  ]);

  // Filter counties to this state
  const stateCountyRows = allCounties.filter((c) => c.state_abbr === upperAbbr);

  // Build races for this state
  const stateRaceMeta = raceMetadata.filter((r) => r.state === upperAbbr);
  const raceDetails = await Promise.all(
    stateRaceMeta.map(async (r) => {
      if (!r.has_predictions) return null;
      const detail = await fetchRaceDetail(r.slug);
      if (!detail) return null;
      const margin = detail.prediction !== null ? detail.prediction - 0.5 : 0;
      const card: SenateRaceData = {
        state: stateName,
        race: `${r.year} ${stateName} ${r.race_type}`,
        slug: r.slug,
        rating: detail.rating,
        margin,
        n_polls: detail.n_polls,
      };
      return card;
    }),
  );
  const races = raceDetails.filter((r): r is SenateRaceData => r !== null);

  // Build type index for the state
  // Map from type_id → TypeSummary for quick lookup
  const typeIndex = new Map<number, TypeSummary>(allTypes.map((t) => [t.type_id, t]));

  // Count counties per type in this state
  const typeCountyCounts = new Map<number, number>();
  for (const c of stateCountyRows) {
    if (c.dominant_type != null) {
      typeCountyCounts.set(
        c.dominant_type,
        (typeCountyCounts.get(c.dominant_type) ?? 0) + 1,
      );
    }
  }

  // Build county table rows — batch-fetch county details for name + type name
  // Only fetch details for counties where we don't already have names.
  // We use county detail endpoint for each county to get county_name + type info.
  // Limit parallel fetches to avoid hammering the API.
  const BATCH_SIZE = 20;
  const countyTableRows: CountyTableRow[] = [];

  for (let i = 0; i < stateCountyRows.length; i += BATCH_SIZE) {
    const batch = stateCountyRows.slice(i, i + BATCH_SIZE);
    const details = await Promise.all(batch.map((c) => fetchCountyDetail(c.county_fips)));
    for (let j = 0; j < batch.length; j++) {
      const base = batch[j];
      const detail = details[j];
      const typeSummary = base.dominant_type != null ? typeIndex.get(base.dominant_type) : null;
      countyTableRows.push({
        county_fips: base.county_fips,
        county_name: detail?.county_name ?? null,
        state_abbr: base.state_abbr,
        dominant_type: base.dominant_type,
        super_type: base.super_type,
        pred_dem_share: base.pred_dem_share,
        type_display_name: detail?.type_display_name ?? typeSummary?.display_name,
      });
    }
  }

  // Build type distribution — only types present in this state
  const typeDistribution: StateTypeItem[] = Array.from(typeCountyCounts).map(
    ([typeId, count]) => {
      const ts = typeIndex.get(typeId);
      if (!ts) return null;
      return {
        type_id: ts.type_id,
        super_type_id: ts.super_type_id,
        display_name: ts.display_name,
        n_counties: count,
        mean_pred_dem_share: ts.mean_pred_dem_share,
      };
    },
  ).filter((t): t is StateTypeItem => t !== null);

  return {
    abbr: upperAbbr,
    name: stateName,
    races,
    counties: countyTableRows,
    typeDistribution,
    totalCounties: stateCountyRows.length,
  };
}

// ── generateStaticParams ───────────────────────────────────────────────────

export async function generateStaticParams() {
  return ALL_STATE_ABBRS.map((abbr) => ({ abbr: abbr.toLowerCase() }));
}

// ── generateMetadata ───────────────────────────────────────────────────────

type PageProps = { params: Promise<{ abbr: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { abbr } = await params;
  const upperAbbr = abbr.toUpperCase();
  const stateName = STATE_NAMES[upperAbbr];
  if (!stateName) {
    return { title: "State Not Found — WetherVane" };
  }

  const title = `${stateName} 2026 Election Forecast | WetherVane`;
  const description = `WetherVane's 2026 election forecast for ${stateName}. See county-level predictions, electoral community types, and race forecasts for Senate and Governor.`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      type: "website",
      siteName: "WetherVane",
    },
    twitter: { card: "summary", title, description },
    alternates: {
      canonical: `${SITE_URL}/state/${upperAbbr}`,
    },
  };
}

// ── Page ───────────────────────────────────────────────────────────────────

export default async function StateHubPage({ params }: PageProps) {
  const { abbr } = await params;
  const upperAbbr = abbr.toUpperCase();
  const stateName = STATE_NAMES[upperAbbr];

  if (!stateName) {
    return (
      <div className="text-center py-16 px-6">
        <h1
          className="font-serif text-2xl mb-3"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          State Not Found
        </h1>
        <p className="text-muted-foreground mb-6">
          No data available for &ldquo;{abbr.toUpperCase()}&rdquo;.
        </p>
        <Link
          href="/"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)" }}
        >
          Back to Home
        </Link>
      </div>
    );
  }

  const data = await fetchStateData(abbr);

  // JSON-LD structured data
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: `${stateName} 2026 Election Forecast`,
    url: `${SITE_URL}/state/${upperAbbr}`,
    description: `2026 election forecast and electoral community analysis for ${stateName}.`,
    isPartOf: { "@type": "WebSite", name: "WetherVane", url: SITE_URL },
    about: {
      "@type": "AdministrativeArea",
      name: stateName,
      address: {
        "@type": "PostalAddress",
        addressCountry: "US",
        addressRegion: upperAbbr,
      },
    },
  };

  const totalCounties = data?.totalCounties ?? 0;
  const raceCount = data?.races.length ?? 0;

  // BreadcrumbList schema — enables rich breadcrumb display in Google SERPs
  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Home", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "Forecast", item: `${SITE_URL}/forecast` },
      { "@type": "ListItem", position: 3, name: stateName, item: `${SITE_URL}/state/${upperAbbr}` },
    ],
  };

  return (
    <article
      id="main-content"
      className="max-w-2xl mx-auto py-8 px-4 pb-20"
    >
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />

      {/* Breadcrumb */}
      <nav
        aria-label="breadcrumb"
        className="text-xs mb-6"
        style={{ color: "var(--color-text-muted)" }}
      >
        <ol className="flex flex-wrap items-center gap-x-1 list-none p-0 m-0">
          <li>
            <Link
              href="/"
              style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
            >
              Home
            </Link>
          </li>
          <li aria-hidden="true">/</li>
          <li>
            <Link
              href="/forecast"
              style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
            >
              Forecast
            </Link>
          </li>
          <li aria-hidden="true">/</li>
          <li aria-current="page">{stateName}</li>
        </ol>
      </nav>

      {/* Hero */}
      <header className="mb-10">
        <div className="flex items-baseline gap-3 mb-2">
          <span
            className="text-xs font-semibold uppercase tracking-wider px-2 py-1 rounded"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            {upperAbbr}
          </span>
        </div>
        <h1
          className="font-serif text-3xl font-bold mb-2"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          {stateName} 2026 Election Forecast
        </h1>
        <p className="text-base" style={{ color: "var(--color-text-muted)" }}>
          {raceCount > 0
            ? `${raceCount} tracked ${raceCount === 1 ? "race" : "races"} · `
            : ""}
          {totalCounties > 0
            ? `${totalCounties} ${totalCounties === 1 ? "county" : "counties"} in model`
            : "No county data available"}
        </p>
      </header>

      {/* Race cards */}
      {data && data.races.length > 0 && (
        <section className="mb-10">
          <h2
            className="font-serif text-xl mb-4"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            2026 Races
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {data.races.map((race) => (
              <RaceCard key={race.slug} race={race} />
            ))}
          </div>
        </section>
      )}

      {data && data.races.length === 0 && (
        <section className="mb-10">
          <div
            className="rounded-md px-4 py-3 text-sm"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            No tracked 2026 races in {stateName} yet. Senate seats in {stateName} are
            not up in 2026.
          </div>
        </section>
      )}

      {/* Type distribution */}
      {data && data.typeDistribution.length > 0 && (
        <section className="mb-10">
          <h2
            className="font-serif text-xl mb-1"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Electoral Community Types
          </h2>
          <p
            className="text-sm mb-4"
            style={{ color: "var(--color-text-muted)" }}
          >
            Electoral communities found in {stateName}, by county count.{" "}
            <Link
              href="/types"
              style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
            >
              Learn about types →
            </Link>
          </p>
          <StateTypeDistribution
            types={data.typeDistribution}
            totalCounties={data.totalCounties}
          />
        </section>
      )}

      {/* County table */}
      {data && data.counties.length > 0 && (
        <section className="mb-10">
          <h2
            className="font-serif text-xl mb-4"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Counties
          </h2>
          <StateCountyTable counties={data.counties} />
        </section>
      )}

      {/* Footer nav */}
      <div
        className="pt-6 border-t text-center"
        style={{ borderColor: "var(--color-border)" }}
      >
        <Link
          href="/forecast"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
        >
          ← Back to Forecast
        </Link>
      </div>
    </article>
  );
}
