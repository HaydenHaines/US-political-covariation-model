import type { Metadata } from "next";
import Link from "next/link";
import dynamic from "next/dynamic";
import { RaceHero } from "@/components/forecast/RaceHero";
import { PollTable } from "@/components/forecast/PollTable";
import { TypesBreakdown } from "@/components/forecast/TypesBreakdown";
import { marginToRating } from "@/lib/config/palette";
import { STATE_NAMES } from "@/lib/config/states";
import { Skeleton } from "@/components/ui/skeleton";

// visx dotplot — heavy bundle, below the hero; load dynamically
const QuantileDotplot = dynamic(
  () =>
    import("@/components/forecast/QuantileDotplot").then(
      (m) => m.QuantileDotplot,
    ),
  {
    ssr: false,
    loading: () => <Skeleton className="w-full h-[160px]" />,
  },
);

// Weight sliders — client-only interactivity, no SSR needed
const SectionWeightSliders = dynamic(
  () =>
    import("@/components/forecast/SectionWeightSliders").then(
      (m) => m.SectionWeightSliders,
    ),
  { ssr: false },
);

// Poll trend chart — visx + SWR, client-only
const PollTrendChart = dynamic(
  () =>
    import("@/components/forecast/PollTrendChart").then(
      (m) => m.PollTrendChart,
    ),
  {
    ssr: false,
    loading: () => <Skeleton className="w-full h-[220px]" />,
  },
);

// Revalidate every 5 minutes — polls and forecast data update periodically
export const revalidate = 300;

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
  forecast_mode?: string;
  state_pred_national?: number | null;
  state_pred_local?: number | null;
  candidate_effect_margin?: number | null;
  n_polls?: number;
  pred_std?: number | null;
  pred_lo90?: number | null;
  pred_hi90?: number | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

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

/** Fallback state-level std when API doesn't provide one. */
const FALLBACK_STD = 0.065;

// ── Metadata ──────────────────────────────────────────────────────────────

type PageProps = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const data = await fetchRaceDetail(slug);
  if (!data) {
    return { title: "Race Not Found — WetherVane" };
  }

  const stateName = STATE_NAMES[data.state_abbr] ?? data.state_abbr;
  const rating = data.prediction !== null ? marginToRating(data.prediction) : null;
  const title = `${data.year} ${stateName} ${data.race_type} | WetherVane`;
  const description =
    data.prediction !== null && rating !== null
      ? `WetherVane forecasts the ${data.year} ${stateName} ${data.race_type} race as ${rating.replace("_", " ")}. Based on ${data.n_counties} ${data.n_counties === 1 ? "county" : "counties"} and electoral type modeling.`
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
    twitter: { card: "summary_large_image", title, description },
  };
}

// ── Static params ─────────────────────────────────────────────────────────

export async function generateStaticParams() {
  const slugs = await fetchRaceSlugs();
  return slugs.map((slug) => ({ slug }));
}

// ── Page ──────────────────────────────────────────────────────────────────

export default async function RaceDetailPage({ params }: PageProps) {
  const { slug } = await params;
  const data = await fetchRaceDetail(slug);

  if (!data) {
    return (
      <div className="text-center py-16 px-6">
        <h1 className="font-serif text-2xl mb-3" style={{ fontFamily: "var(--font-serif)" }}>
          Race Not Found
        </h1>
        <p className="text-muted-foreground mb-6">No data available for this race.</p>
        <Link href="/forecast" className="text-sm font-semibold" style={{ color: "var(--forecast-safe-d)" }}>
          Back to forecast
        </Link>
      </div>
    );
  }

  const stateName = STATE_NAMES[data.state_abbr] ?? data.state_abbr;
  const siteUrl = "https://wethervane.hhaines.duckdns.org";
  const rating = data.prediction !== null ? marginToRating(data.prediction) : null;

  // JSON-LD structured data for SEO
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Event",
    name: `${data.year} ${stateName} ${data.race_type}`,
    url: `${siteUrl}/forecast/${slug}`,
    startDate: `${data.year}-11-03`,
    eventStatus: "https://schema.org/EventScheduled",
    location: {
      "@type": "AdministrativeArea",
      name: stateName,
      address: { "@type": "PostalAddress", addressCountry: "US", addressRegion: data.state_abbr },
    },
    organizer: { "@type": "Organization", name: "WetherVane", url: siteUrl },
    additionalProperty: [
      { "@type": "PropertyValue", name: "Race Type", value: data.race_type },
      { "@type": "PropertyValue", name: "Counties in Model", value: data.n_counties },
      ...(data.prediction !== null && rating !== null
        ? [
            { "@type": "PropertyValue", name: "Model Predicted Democratic Vote Share", value: data.prediction },
            { "@type": "PropertyValue", name: "Political Lean", value: rating },
          ]
        : []),
    ],
  };

  const predStd = data.pred_std ?? FALLBACK_STD;
  const lo90 = data.pred_lo90 ?? (data.prediction !== null ? data.prediction - 1.645 * predStd : null);
  const hi90 = data.pred_hi90 ?? (data.prediction !== null ? data.prediction + 1.645 * predStd : null);
  const nPolls = data.n_polls ?? data.polls.length;

  return (
    <article id="main-content" className="max-w-2xl mx-auto py-8 px-4 pb-20">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      {/* Breadcrumb */}
      <nav aria-label="breadcrumb" className="text-xs mb-6" style={{ color: "var(--color-text-muted)" }}>
        <ol className="flex flex-wrap items-center gap-x-1 list-none p-0 m-0">
          <li>
            <Link href="/" style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}>
              Home
            </Link>
          </li>
          <li aria-hidden="true">/</li>
          <li>
            <Link href="/forecast" style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}>
              Forecast
            </Link>
          </li>
          <li aria-hidden="true">/</li>
          <li aria-current="page">
            {data.year} {stateName} {data.race_type}
          </li>
        </ol>
      </nav>

      {/* Hero — large margin, rating badge, CI */}
      <RaceHero
        raceName={data.race}
        stateName={stateName}
        raceType={data.race_type}
        year={data.year}
        prediction={data.prediction}
        nCounties={data.n_counties}
        lo90={lo90}
        hi90={hi90}
      />

      {/* Outcome distribution dotplot */}
      {data.prediction !== null && (
        <section className="mb-10">
          <h2 className="font-serif text-xl mb-4" style={{ fontFamily: "var(--font-serif)" }}>
            Outcome Distribution
          </h2>
          <p className="text-sm mb-3" style={{ color: "var(--color-text-muted)" }}>
            Each dot represents one possible scenario. The distribution is derived
            from the model&apos;s prediction and estimated uncertainty (±{(predStd * 100).toFixed(0)}pp std).
          </p>
          <QuantileDotplot
            predDemShare={data.prediction}
            predStd={predStd}
            nDots={100}
            width={480}
            height={160}
          />
        </section>
      )}

      {/* Polls section */}
      <section className="mb-10">
        <h2 className="font-serif text-xl mb-4" style={{ fontFamily: "var(--font-serif)" }}>
          Recent Polls
          {nPolls > 0 && (
            <span className="text-sm font-normal ml-2" style={{ color: "var(--color-text-muted)" }}>
              ({nPolls} poll{nPolls !== 1 ? "s" : ""})
            </span>
          )}
        </h2>
        {nPolls === 0 && (
          <p
            className="text-sm mb-4 rounded-md px-4 py-3"
            style={{
              color: "var(--color-text-muted)",
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            This forecast is based on the structural model prior — no race-specific polls have been incorporated yet.
          </p>
        )}
        {/* Poll trend chart — shows movement over time; no-polls state handled inside */}
        {nPolls > 0 && (
          <div className="mb-6">
            <PollTrendChart slug={slug} width={480} />
          </div>
        )}
        <PollTable polls={data.polls} />
      </section>

      {/* Electoral types breakdown */}
      {data.type_breakdown.length > 0 && (
        <section className="mb-10">
          <h2 className="font-serif text-xl mb-4" style={{ fontFamily: "var(--font-serif)" }}>
            Electoral Types in {stateName}
          </h2>
          <TypesBreakdown types={data.type_breakdown} stateName={stateName} />
        </section>
      )}

      {/* Forecast blend controls */}
      <section className="mb-10">
        <h2 className="font-serif text-xl mb-4" style={{ fontFamily: "var(--font-serif)" }}>
          Forecast Blend
        </h2>
        <p className="text-sm mb-3" style={{ color: "var(--color-text-muted)" }}>
          Adjust how the forecast weights the structural model prior against available polling data.
          {data.state_pred_local !== null && data.state_pred_local !== undefined && (
            <> State-level model prior: <span className="font-mono">{((data.state_pred_local) * 100).toFixed(1)}% D</span>.</>
          )}
          {data.state_pred_national !== null && data.state_pred_national !== undefined && (
            <> National-adjusted: <span className="font-mono">{((data.state_pred_national) * 100).toFixed(1)}% D</span>.</>
          )}
        </p>
        <SectionWeightSliders
          initial={{
            model_prior: nPolls === 0 ? 80 : 60,
            state_polls: nPolls === 0 ? 15 : 30,
            national_polls: nPolls === 0 ? 5 : 10,
          }}
        />
      </section>

      {/* Model notes */}
      <section
        className="mb-10 rounded-md p-4 text-sm"
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        <h3 className="font-semibold mb-2 text-sm" style={{ color: "var(--color-text)" }}>
          About this forecast
        </h3>
        <ul className="space-y-1 list-disc list-inside">
          <li>
            Structural forecast based on electoral type modeling across {data.n_counties} {data.n_counties === 1 ? "county" : "counties"}.
          </li>
          <li>
            Types are discovered from historical shift patterns — not from polls or demographics alone.
          </li>
          {nPolls > 0 && (
            <li>
              {nPolls} race-specific poll{nPolls !== 1 ? "s" : ""} have been incorporated
              via Bayesian update through the type covariance structure.
            </li>
          )}
          <li>
            Forecast mode:{" "}
            <span className="font-mono">{data.forecast_mode ?? "local"}</span>.
          </li>
        </ul>
      </section>

      {/* Back link */}
      <div className="pt-6 border-t text-center" style={{ borderColor: "var(--color-border)" }}>
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
