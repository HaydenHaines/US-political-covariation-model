"use client";

import Link from "next/link";
import { useGovernorOverview } from "@/lib/hooks/use-governor-overview";
import { useGovernorSimulation } from "@/lib/hooks/use-governor-simulation";
import { usePolls } from "@/lib/hooks/use-polls";
import { useRaceHistory } from "@/lib/hooks/use-race-history";
import { PollTrendChart } from "@/components/forecast/PollTrendChart";
import { PollTable } from "@/components/forecast/PollTable";
import { SparklineChart } from "@/components/forecast/SparklineChart";
import { FundamentalsCard } from "@/components/forecast/FundamentalsCard";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { formatMargin } from "@/lib/format";
import { ratingLabel } from "@/lib/colors";
import { STATE_NAMES } from "@/lib/config/states";
import type { Rating } from "@/lib/colors";
import type { RaceDetailPoll } from "@/lib/api";

// Abramowitz & Stegun normal CDF approximation (max error < 7.5e-8)
function normalCDF(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const poly =
    t *
    (0.31938153 +
      t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const phi = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const p = 1 - phi * poly;
  return z >= 0 ? p : 1 - p;
}

function winProbabilities(margin: number, predStd: number): { dem: number; rep: number } {
  const z = margin / predStd;
  const dem = normalCDF(z);
  return { dem, rep: 1 - dem };
}

const FALLBACK_STD = 0.065;

type PageProps = { params: { slug: string } };

export default function GovernorRaceDetailPage({ params }: PageProps) {
  const { slug } = params;

  const { data: overviewData, error: overviewError, isLoading: overviewLoading, mutate } = useGovernorOverview();
  const { data: simData } = useGovernorSimulation();
  const { historyBySlug } = useRaceHistory();
  const raceHistory = historyBySlug.get(slug) ?? [];

  const race = overviewData?.races.find((r) => r.state.toLowerCase() === slug);

  // Derive the race string needed for the polls filter once we have overview data
  const raceName = race?.race ?? null;
  const { data: polls, isLoading: pollsLoading } = usePolls(
    raceName ? { race: raceName } : {},
  );

  if (overviewError) {
    return (
      <ErrorAlert
        title="Failed to load governor race data"
        retry={() => mutate()}
      />
    );
  }

  if (overviewLoading || !overviewData) {
    return (
      <div className="max-w-2xl mx-auto py-8 px-4 space-y-6">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-8 w-3/4" />
        <Skeleton className="h-6 w-48" />
        <Skeleton className="h-[220px] w-full rounded-md" />
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (!race) {
    return (
      <div className="text-center py-16 px-6">
        <h1 className="font-serif text-2xl mb-3" style={{ color: "var(--color-text)" }}>
          Race Not Found
        </h1>
        <p className="text-sm mb-6" style={{ color: "var(--color-text-muted)" }}>
          No governor race found for this slug.
        </p>
        <Link
          href="/forecast/governor"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
        >
          ← Governor Races
        </Link>
      </div>
    );
  }

  const stateName = STATE_NAMES[race.state] ?? race.state;
  // margin = pred_dem_share - 0.5; positive = Dem-favored
  const winProb = winProbabilities(race.margin, FALLBACK_STD);
  const { text: marginText, party: marginParty } = race.margin !== 0
    ? { text: formatMargin(race.margin + 0.5), party: race.margin > 0 ? "dem" : "gop" as const }
    : { text: "EVEN", party: "even" as const };
  const marginColor =
    marginParty === "dem"
      ? "var(--forecast-safe-d)"
      : marginParty === "gop"
      ? "var(--forecast-safe-r)"
      : "var(--forecast-tossup)";

  // Compute median Dem governor count from simulation buckets for context
  let medianDemGovs: number | null = null;
  if (simData?.buckets && simData.buckets.length > 0) {
    let cumProb = 0;
    const sorted = [...simData.buckets].sort((a, b) => a.d_seats - b.d_seats);
    for (const bucket of sorted) {
      cumProb += bucket.probability;
      if (cumProb >= 0.5) {
        medianDemGovs = bucket.d_seats;
        break;
      }
    }
  }

  // Map PollRow → RaceDetailPoll for PollTable (grade not available from polls API)
  const pollTableData: RaceDetailPoll[] = (polls ?? []).map((p) => ({
    date: p.date,
    pollster: p.pollster,
    dem_share: p.dem_share,
    n_sample: p.n_sample,
    grade: null,
  }));

  const nPolls = race.n_polls;

  return (
    <article className="max-w-2xl mx-auto py-8 px-4 pb-20">
      {/* Back link */}
      <nav className="text-xs mb-6" style={{ color: "var(--color-text-muted)" }}>
        <Link
          href="/forecast/governor"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
        >
          ← Governor Races
        </Link>
      </nav>

      {/* Race headline */}
      <header className="mb-6">
        <div className="flex flex-wrap items-center gap-3 mb-2">
          <h1
            className="font-serif text-2xl font-bold"
            style={{ color: "var(--color-text)" }}
          >
            {stateName} Governor
          </h1>
          <RatingBadge rating={race.rating} />
        </div>

        {/* Win probability + margin */}
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm" style={{ color: "var(--color-text-muted)" }}>
            <span style={{ color: "var(--forecast-safe-d)", fontWeight: 600 }}>
              D {Math.round(winProb.dem * 100)}%
            </span>
            {" / "}
            <span style={{ color: "var(--forecast-safe-r)", fontWeight: 600 }}>
              R {Math.round(winProb.rep * 100)}%
            </span>
          </span>
          <span
            className="font-mono font-bold text-sm"
            style={{ color: marginColor }}
          >
            {marginText}
          </span>
        </div>

        {/* Race fundamentals: incumbent, open seat, rating label */}
        <div
          className="mt-4 rounded-md px-4 py-3 text-sm space-y-1"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
          }}
        >
          <p>
            <span style={{ color: "var(--color-text-muted)" }}>Rating: </span>
            <span className="font-medium" style={{ color: "var(--color-text)" }}>
              {ratingLabel(race.rating as Rating)}
            </span>
          </p>
          <p>
            <span style={{ color: "var(--color-text-muted)" }}>
              {race.is_open_seat ? "Open seat" : "Incumbent party"}:{" "}
            </span>
            <span
              className="font-semibold"
              style={{
                color:
                  race.incumbent_party === "D"
                    ? "var(--color-dem)"
                    : race.incumbent_party === "R"
                    ? "var(--color-rep)"
                    : "var(--color-text-muted)",
              }}
            >
              {race.is_open_seat
                ? `${race.incumbent_party} (no incumbent running)`
                : race.incumbent_party}
            </span>
          </p>
          <p>
            <span style={{ color: "var(--color-text-muted)" }}>Model lean: </span>
            <span className="font-mono font-semibold" style={{ color: marginColor }}>
              {marginText}
            </span>
          </p>
          {medianDemGovs !== null && (
            <p>
              <span style={{ color: "var(--color-text-muted)" }}>Simulation (all gov races): </span>
              <span style={{ color: "var(--color-text)" }}>
                D {medianDemGovs} / R {36 - medianDemGovs} (median)
              </span>
            </p>
          )}
        </div>
      </header>

      {/* Forecast history sparkline */}
      {raceHistory.length > 0 && (
        <section className="mb-10">
          <h2
            className="font-serif text-xl mb-4"
            style={{ color: "var(--color-text)" }}
          >
            Forecast History
          </h2>
          <div
            className="rounded-md px-4 py-4"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            <SparklineChart
              history={raceHistory}
              width={480}
              height={60}
              ariaLabel={`Forecast margin history for ${stateName} Governor`}
            />
          </div>
        </section>
      )}

      {/* Polls section */}
      <section className="mb-10">
        <h2
          className="font-serif text-xl mb-4 flex flex-wrap items-center gap-2"
          style={{ color: "var(--color-text)" }}
        >
          Recent Polls
          {nPolls > 0 && (
            <span className="text-sm font-normal" style={{ color: "var(--color-text-muted)" }}>
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
            No race-specific polls yet — forecast reflects the structural model prior only.
          </p>
        )}

        {nPolls > 0 && (
          <div className="mb-6">
            <PollTrendChart slug={slug} width={480} />
          </div>
        )}

        {pollsLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        ) : (
          <PollTable polls={pollTableData} />
        )}
      </section>

      {/* National environment fundamentals */}
      <FundamentalsCard />

      {/* Footer back link */}
      <div
        className="pt-6 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        <Link
          href="/forecast/governor"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
        >
          ← Back to Governor Races
        </Link>
      </div>
    </article>
  );
}
