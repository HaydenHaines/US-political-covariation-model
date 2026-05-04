"use client";

import Link from "next/link";
import { useRaceDetail } from "@/lib/hooks/use-race-detail";
import { CandidateBadges } from "@/components/forecast/CandidateBadges";
import { PollTrendChart } from "@/components/forecast/PollTrendChart";
import { PollTable } from "@/components/forecast/PollTable";
import { HistoricalContextCard } from "@/components/forecast/HistoricalContextCard";
import { PollConfidenceBadge } from "@/components/forecast/PollConfidenceBadge";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { formatMargin } from "@/lib/format";
import { STATE_NAMES } from "@/lib/config/states";

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

function winProbabilities(
  prediction: number,
  predStd: number,
): { dem: number; rep: number } {
  const z = (prediction - 0.5) / predStd;
  const dem = normalCDF(z);
  return { dem, rep: 1 - dem };
}

const FALLBACK_STD = 0.065;

const PARTY_COLORS: Record<string, string> = {
  D: "var(--color-dem)",
  R: "var(--color-rep)",
  I: "var(--color-text-muted)",
};

type PageProps = { params: { slug: string } };

export default function SenateRaceDetailPage({ params }: PageProps) {
  const { slug } = params;
  const { data, error, isLoading, mutate } = useRaceDetail(slug);

  if (error) {
    return (
      <ErrorAlert
        title="Failed to load race data"
        retry={() => mutate()}
      />
    );
  }

  if (isLoading || !data) {
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

  const stateName = STATE_NAMES[data.state_abbr] ?? data.state_abbr;
  const raceKey = `${data.year} ${data.state_abbr} ${data.race_type}`;
  const predStd = data.pred_std ?? FALLBACK_STD;
  const nPolls = data.n_polls ?? data.polls.length;

  const winProb =
    data.prediction !== null
      ? winProbabilities(data.prediction, predStd)
      : null;

  const { text: marginText, party: marginParty } = data.prediction !== null
    ? { text: formatMargin(data.prediction), party: data.prediction > 0.5 ? "dem" : data.prediction < 0.5 ? "gop" : "even" as const }
    : { text: "—", party: "even" as const };

  const marginColor =
    marginParty === "dem"
      ? "var(--forecast-safe-d)"
      : marginParty === "gop"
      ? "var(--forecast-safe-r)"
      : "var(--forecast-tossup)";

  return (
    <article className="max-w-2xl mx-auto py-8 px-4 pb-20">
      {/* Back link */}
      <nav className="text-xs mb-6" style={{ color: "var(--color-text-muted)" }}>
        <Link
          href="/forecast/senate"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
        >
          ← Senate Races
        </Link>
      </nav>

      {/* Race headline */}
      <header className="mb-6">
        <h1
          className="font-serif text-2xl font-bold mb-2"
          style={{ color: "var(--color-text)" }}
        >
          {data.year} {stateName} {data.race_type}
        </h1>

        {/* Win probability + margin */}
        <div className="flex flex-wrap items-center gap-3">
          {winProb !== null && (
            <span className="text-sm" style={{ color: "var(--color-text-muted)" }}>
              <span style={{ color: "var(--forecast-safe-d)", fontWeight: 600 }}>
                D {Math.round(winProb.dem * 100)}%
              </span>
              {" / "}
              <span style={{ color: "var(--forecast-safe-r)", fontWeight: 600 }}>
                R {Math.round(winProb.rep * 100)}%
              </span>
            </span>
          )}
          {data.prediction !== null && (
            <span
              className="font-mono font-bold text-sm"
              style={{ color: marginColor }}
            >
              {marginText}
            </span>
          )}
          {data.poll_confidence && (
            <PollConfidenceBadge confidence={data.poll_confidence} />
          )}
        </div>

        {/* Candidate info */}
        {data.candidate_info && (
          <div
            className="mt-4 rounded-md px-4 py-3 text-sm"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            {data.candidate_info.status === "open" ||
            data.candidate_info.status === "special" ? (
              <p>
                <span className="font-semibold" style={{ color: "var(--color-text)" }}>
                  Open Seat
                </span>
                {data.candidate_info.status_detail && (
                  <span style={{ color: "var(--color-text-muted)" }}>
                    {" "}— {data.candidate_info.status_detail}
                  </span>
                )}
              </p>
            ) : (
              <p>
                <span style={{ color: "var(--color-text-muted)" }}>Incumbent: </span>
                <span
                  className="font-semibold"
                  style={{
                    color:
                      PARTY_COLORS[data.candidate_info.incumbent.party] ??
                      "var(--color-text-muted)",
                  }}
                >
                  {data.candidate_info.incumbent.name} (
                  {data.candidate_info.incumbent.party})
                </span>
              </p>
            )}
            {data.candidate_info.rating && (
              <p className="mt-1">
                <span style={{ color: "var(--color-text-muted)" }}>Rating: </span>
                <span className="font-medium" style={{ color: "var(--color-text)" }}>
                  {data.candidate_info.rating}
                </span>
              </p>
            )}
          </div>
        )}
      </header>

      {/* Candidate performance badges — SWR, renders when badge data exists */}
      <CandidateBadges raceKey={raceKey} />

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

        <PollTable polls={data.polls} />
      </section>

      {/* Historical context — only for tracked competitive races */}
      {data.historical_context && (
        <HistoricalContextCard
          context={data.historical_context}
          stateName={stateName}
          stateAbbr={data.state_abbr}
        />
      )}

      {/* Footer back link */}
      <div
        className="pt-6 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        <Link
          href="/forecast/senate"
          className="text-sm font-semibold"
          style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
        >
          ← Back to Senate Races
        </Link>
      </div>
    </article>
  );
}
