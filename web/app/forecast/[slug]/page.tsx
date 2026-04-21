import type { Metadata } from "next";
import Link from "next/link";
import dynamic from "next/dynamic";
import { PollTable } from "@/components/forecast/PollTable";
import { TypesBreakdown } from "@/components/forecast/TypesBreakdown";
import { HistoricalContextCard } from "@/components/forecast/HistoricalContextCard";
import type { HistoricalContext } from "@/components/forecast/HistoricalContextCard";
import { FundamentalsCard } from "@/components/forecast/FundamentalsCard";
import { PollConfidenceBadge } from "@/components/forecast/PollConfidenceBadge";
import type { PollConfidence } from "@/components/forecast/PollConfidenceBadge";
import { marginToRating } from "@/lib/config/palette";
import { STATE_NAMES } from "@/lib/config/states";
import { Skeleton } from "@/components/ui/skeleton";

// RaceBlendControls owns the hero, dotplot, and blend sliders.
// It is "use client" because it manages slider state and fires API calls.
// The rest of this page remains a Server Component (SSR).
const RaceBlendControls = dynamic(
  () =>
    import("@/components/forecast/RaceBlendControls").then(
      (m) => m.RaceBlendControls,
    ),
  {
    ssr: false,
    // Skeleton sized to match the hero + dotplot area during hydration
    loading: () => (
      <div className="space-y-4 mb-10">
        <Skeleton className="w-3/4 h-12" />
        <Skeleton className="w-1/2 h-6" />
        <Skeleton className="w-full h-[160px] mt-6" />
      </div>
    ),
  },
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

// Candidate sabermetric badges — SWR + tooltips, client-only
const CandidateBadges = dynamic(
  () =>
    import("@/components/forecast/CandidateBadges").then(
      (m) => m.CandidateBadges,
    ),
  { ssr: false },
);

// Moneyball Fit Scores — SWR + party toggle, client-only
const FitScoreSection = dynamic(
  () =>
    import("@/components/forecast/FitScoreSection").then(
      (m) => m.FitScoreSection,
    ),
  {
    ssr: false,
    loading: () => <Skeleton className="w-full h-[180px]" />,
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
  grade: string | null;
}

interface TypeBreakdown {
  type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  // Total 2024 votes across counties of this type in the state (from API).
  // Types are pre-sorted by this value descending so urban types appear first.
  total_votes: number | null;
}

interface CandidateIncumbent {
  name: string;
  party: string;
}

interface CandidateInfo {
  incumbent: CandidateIncumbent;
  status: string;
  status_detail?: string | null;
  rating?: string | null;
  candidates: Record<string, string[]>;
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
  historical_context?: HistoricalContext | null;
  poll_confidence?: PollConfidence | null;
  candidate_info?: CandidateInfo | null;
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

// ── Candidate section ────────────────────────────────────────────────────

/** Party abbreviation to display color CSS variable. */
const PARTY_COLORS: Record<string, string> = {
  D: "var(--color-dem)",
  R: "var(--color-rep)",
  I: "var(--color-text-muted)",
};

/**
 * Displays candidate names, incumbent status, and race rating
 * in the hero area above the forecast controls.
 *
 * Rendered server-side from static candidate data — no client
 * interactivity needed.
 */
function CandidateSection({ info }: { info: CandidateInfo }) {
  const isOpen = info.status === "open" || info.status === "special";
  const partyColor = PARTY_COLORS[info.incumbent.party] ?? "var(--color-text-muted)";

  // Collect all declared candidates across parties for the matchup line
  const allCandidatesByParty = Object.entries(info.candidates)
    .filter(([, names]) => names.length > 0)
    .sort(([a], [b]) => {
      // Show incumbent's party first
      if (a === info.incumbent.party) return -1;
      if (b === info.incumbent.party) return 1;
      return a.localeCompare(b);
    });

  return (
    <div
      className="mb-6 rounded-md px-4 py-3 text-sm"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      {/* Incumbent / open seat status */}
      <p className="mb-1">
        {isOpen ? (
          <>
            <span className="font-semibold" style={{ color: "var(--color-text)" }}>
              Open Seat
            </span>
            {info.status_detail && (
              <span style={{ color: "var(--color-text-muted)" }}>
                {" "}&mdash; {info.status_detail}
              </span>
            )}
          </>
        ) : (
          <>
            <span style={{ color: "var(--color-text-muted)" }}>Incumbent: </span>
            <span className="font-semibold" style={{ color: partyColor }}>
              {info.incumbent.name} ({info.incumbent.party})
            </span>
          </>
        )}
      </p>

      {/* Race rating badge */}
      {info.rating && (
        <p className="mb-1">
          <span style={{ color: "var(--color-text-muted)" }}>Rating: </span>
          <span className="font-medium" style={{ color: "var(--color-text)" }}>
            {info.rating}
          </span>
        </p>
      )}

      {/* Declared candidates by party */}
      {allCandidatesByParty.length > 0 && (
        <div className="mt-2">
          <span style={{ color: "var(--color-text-muted)" }}>Candidates: </span>
          {allCandidatesByParty.map(([party, names], idx) => (
            <span key={party}>
              {idx > 0 && <span style={{ color: "var(--color-text-muted)" }}> vs </span>}
              <span style={{ color: PARTY_COLORS[party] ?? "var(--color-text-muted)" }}>
                {names.join(", ")} ({party})
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
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
  const rating = data.prediction !== null ? marginToRating(data.prediction) : null;

  // Build SEO title with candidate names when available.
  // "2026 Georgia Senate Forecast -- Ossoff vs Carter | WetherVane"
  // For open/special seats: "2026 Georgia Senate Forecast -- Open Seat | WetherVane"
  const ci = data.candidate_info;
  let titleSuffix = "";
  if (ci) {
    const isOpen = ci.status === "open" || ci.status === "special";
    if (isOpen) {
      titleSuffix = " -- Open Seat";
    } else {
      // Find the top challenger from the opposing party
      const incumbentParty = ci.incumbent.party;
      const opposingParty = incumbentParty === "D" ? "R" : "D";
      const challengers = ci.candidates[opposingParty] ?? [];
      const incumbentLast = ci.incumbent.name.split(" ").pop() ?? ci.incumbent.name;
      if (challengers.length > 0) {
        const challengerLast = challengers[0].split(" ").pop() ?? challengers[0];
        titleSuffix = ` -- ${incumbentLast} vs ${challengerLast}`;
      } else {
        titleSuffix = ` -- ${incumbentLast}`;
      }
    }
  }
  const title = `${data.year} ${stateName} ${data.race_type} Forecast${titleSuffix} | WetherVane`;

  // Build SEO description including candidate context
  let candidateContext = "";
  if (ci) {
    const isOpen = ci.status === "open" || ci.status === "special";
    if (isOpen && ci.status_detail) {
      candidateContext = ` ${ci.status_detail}.`;
    } else if (!isOpen) {
      candidateContext = ` Incumbent: ${ci.incumbent.name} (${ci.incumbent.party}).`;
    }
    if (ci.rating) {
      candidateContext += ` Rated ${ci.rating}.`;
    }
  }
  const description =
    data.prediction !== null && rating !== null
      ? `WetherVane forecasts the ${data.year} ${stateName} ${data.race_type} race as ${rating.replace("_", " ")}.${candidateContext} Based on ${data.n_counties} ${data.n_counties === 1 ? "county" : "counties"} and electoral type modeling.`
      : `WetherVane's forecast for the ${data.year} ${stateName} ${data.race_type} race.${candidateContext} Explore county-level predictions and polling data.`;

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

  // BreadcrumbList schema — enables rich breadcrumb display in Google SERPs
  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Home", item: siteUrl },
      { "@type": "ListItem", position: 2, name: "Forecast", item: `${siteUrl}/forecast` },
      {
        "@type": "ListItem",
        position: 3,
        name: `${data.year} ${stateName} ${data.race_type}`,
        item: `${siteUrl}/forecast/${slug}`,
      },
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
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
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

      {/* Candidate info — SSR, static data from candidates_2026.json via API */}
      {data.candidate_info && (
        <CandidateSection info={data.candidate_info} />
      )}

      {/* Sabermetric badge pills — client component, fetches via SWR */}
      <CandidateBadges raceKey={`${data.year} ${data.state_abbr} ${data.race_type}`} />

      {/*
        RaceBlendControls is a client component that owns:
          - RaceHero (margin, rating badge, CI bounds) — live-updated by blend
          - Outcome Distribution dotplot — live-updated by blend
          - Forecast Blend section with SectionWeightSliders

        It hydrates with SSR values and recalculates via
        POST /api/v1/forecast/race/{slug}/blend on slider changes (400ms debounce).
      */}
      <RaceBlendControls
        slug={slug}
        apiBase={API_BASE}
        initialPrediction={data.prediction}
        initialPredStd={predStd}
        initialLo90={lo90}
        initialHi90={hi90}
        hasPolls={nPolls > 0}
        statePredLocal={data.state_pred_local}
        statePredNational={data.state_pred_national}
        stateName={stateName}
        raceType={data.race_type}
        year={data.year}
        nCounties={data.n_counties}
        nPolls={nPolls}
      />

      {/* Polls section */}
      <section className="mb-10">
        <h2 className="font-serif text-xl mb-4 flex flex-wrap items-center gap-2" style={{ fontFamily: "var(--font-serif)" }}>
          Recent Polls
          {nPolls > 0 && (
            <span className="text-sm font-normal" style={{ color: "var(--color-text-muted)" }}>
              ({nPolls} poll{nPolls !== 1 ? "s" : ""})
            </span>
          )}
          {data.poll_confidence && (
            <PollConfidenceBadge confidence={data.poll_confidence} />
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

      {/* Historical context — only rendered for the 15 tracked competitive races */}
      {data.historical_context && (
        <HistoricalContextCard
          context={data.historical_context}
          stateName={stateName}
          stateAbbr={data.state_abbr}
        />
      )}

      {/* Moneyball Fit Scores — which historical candidates match this district */}
      <FitScoreSection
        raceKey={`${data.year} ${data.state_abbr} ${data.race_type}`}
        defaultParty={
          data.candidate_info?.incumbent?.party === "R" ? "R" : "D"
        }
      />

      {/* National Environment — structural fundamentals signal common to all races */}
      <FundamentalsCard />

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
        </ul>
        {/* Freshness timestamp — shows most recent poll date when polls exist */}
        <p className="mt-3 text-xs" style={{ color: "var(--color-text-muted)", opacity: 0.75 }}>
          {nPolls > 0 && data.polls.length > 0 && data.polls[0].date
            ? <>Last updated: {new Date(data.polls[0].date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}</>
            : <>No polls yet — forecast reflects structural model prior only.</>
          }
        </p>
      </section>

      {/* Back link + compare link */}
      <div
        className="pt-6 border-t flex flex-wrap items-center justify-center gap-6"
        style={{ borderColor: "var(--color-border)" }}
      >
        <Link
          href="/forecast"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
        >
          ← Back to Forecast
        </Link>
        <Link
          href={`/compare/races?races=${slug},`}
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-tossup)", textDecoration: "none" }}
        >
          Compare with another race →
        </Link>
      </div>
    </article>
  );
}
