"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";

// ── Helpers ───────────────────────────────────────────────────────────────

/** Bucket poll counts into a coverage quality tier. */
function coverageLabel(totalPolls: number, racesWithPolls: number, totalRaces: number): {
  label: string;
  color: string;
  narrative: string;
} {
  const coveragePct = totalRaces > 0 ? racesWithPolls / totalRaces : 0;

  if (totalPolls === 0) {
    return {
      label: "No Polls",
      color: "var(--color-text-muted)",
      narrative: "No state-level Senate polling has been collected yet this cycle.",
    };
  }
  if (coveragePct < 0.3) {
    return {
      label: "Sparse",
      color: "var(--forecast-lean-r)",
      narrative:
        "Polling coverage is thin — fewer than a third of contested races have been polled. " +
        "The forecast relies heavily on the structural model prior in unpolled states.",
    };
  }
  if (coveragePct < 0.6) {
    return {
      label: "Moderate",
      color: "var(--forecast-tossup)",
      narrative:
        "About half of contested races have polling. The forecast blends structural priors " +
        "with available polling, but several states remain data-poor.",
    };
  }
  return {
    label: "Good",
    color: "var(--forecast-lean-d)",
    narrative:
      "Most contested races have at least one poll. The forecast can meaningfully " +
      "blend polling signal with the structural prior across the map.",
  };
}

// ── Loading skeleton ──────────────────────────────────────────────────────

function PollingCardSkeleton() {
  return (
    <section
      className="mb-8 rounded-md p-4 text-sm animate-pulse"
      aria-label="Polling Overview loading"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <div
        className="h-5 w-36 rounded mb-3"
        style={{ background: "var(--color-border)" }}
      />
      <div
        className="h-4 w-56 rounded mb-4"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex justify-between">
            <div
              className="h-3 w-32 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
            <div
              className="h-3 w-16 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Main component ────────────────────────────────────────────────────────

/**
 * PollingCard — summary of current polling coverage for the Senate forecast.
 *
 * Shows total polls collected, how many races have polling data, average
 * polls per polled race, and a narrative about coverage quality.
 * Follows the FundamentalsCard pattern: surface background, Dusty Ink vars.
 */
export function PollingCard() {
  const { data, error, isLoading } = useSenateOverview();

  if (isLoading) {
    return <PollingCardSkeleton />;
  }

  // Silently hide on error — same pattern as FundamentalsCard
  if (error || !data) {
    return null;
  }

  const totalPolls = data.races.reduce((sum, r) => sum + r.n_polls, 0);
  const racesWithPolls = data.races.filter((r) => r.n_polls > 0).length;
  const totalRaces = data.races.length;
  const avgPollsPerPolledRace =
    racesWithPolls > 0 ? (totalPolls / racesWithPolls).toFixed(1) : "0";

  const { label: qualityLabel, color: qualityColor, narrative } = coverageLabel(
    totalPolls,
    racesWithPolls,
    totalRaces,
  );

  // Find the most-polled and least-polled contested races for context
  const polledRaces = data.races.filter((r) => r.n_polls > 0);
  const mostPolled = polledRaces.length > 0
    ? [...polledRaces].sort((a, b) => b.n_polls - a.n_polls)[0]
    : null;

  const updatedLabel = data.updated_at
    ? new Date(data.updated_at).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : null;

  return (
    <section
      className="mb-8 rounded-md p-4 text-sm"
      aria-label="Polling Overview"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      {/* Header row */}
      <div className="flex flex-wrap items-baseline justify-between gap-3 mb-1">
        <h2
          className="font-serif text-lg"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          Polling Overview
        </h2>
        <span
          className="font-mono font-bold text-xl"
          style={{ color: qualityColor }}
          aria-label={`Coverage quality: ${qualityLabel}`}
        >
          {qualityLabel}
        </span>
      </div>

      {/* Subheader */}
      <p className="mb-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
        State-level Senate polls collected for the current cycle
        {updatedLabel && (
          <> &middot; last updated {updatedLabel}</>
        )}
      </p>

      {/* Stats breakdown */}
      <dl className="space-y-2 mb-4">
        <div className="flex items-center justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>Total Polls</dt>
          <dd className="font-mono font-semibold" style={{ color: "var(--color-text)" }}>
            {totalPolls}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>Races Polled</dt>
          <dd className="font-mono font-semibold" style={{ color: "var(--color-text)" }}>
            {racesWithPolls} / {totalRaces}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>Avg. Polls per Race</dt>
          <dd className="font-mono font-semibold" style={{ color: "var(--color-text)" }}>
            {avgPollsPerPolledRace}
          </dd>
        </div>
        {mostPolled && (
          <div className="flex items-center justify-between gap-4">
            <dt style={{ color: "var(--color-text-muted)" }}>Most Polled</dt>
            <dd className="font-mono font-semibold" style={{ color: "var(--color-text)" }}>
              {mostPolled.state} ({mostPolled.n_polls})
            </dd>
          </div>
        )}
      </dl>

      {/* Narrative */}
      <div
        className="pt-3 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        <p style={{ color: "var(--color-text-muted)" }}>{narrative}</p>
      </div>
    </section>
  );
}
