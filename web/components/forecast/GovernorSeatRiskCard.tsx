"use client";

import { DUSTY_INK, RATING_COLORS } from "@/lib/config/palette";
import type { GovernorRaceData } from "@/lib/api";

const COMPETITIVE = new Set(["tossup", "lean_d", "lean_r"]);

interface GovernorSeatRiskCardProps {
  races: GovernorRaceData[];
}

// ── Loading skeleton ──────────────────────────────────────────────────────

export function GovernorSeatRiskCardSkeleton() {
  return (
    <section
      className="mb-8 rounded-md p-4 text-sm animate-pulse"
      aria-label="Seat Risk loading"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <div
        className="h-5 w-28 rounded mb-3"
        style={{ background: "var(--color-border)" }}
      />
      <div
        className="h-4 w-64 rounded mb-4"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div
        className="h-4 rounded mb-4"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div className="space-y-2">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="flex justify-between">
            <div
              className="h-3 w-36 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
            <div
              className="h-3 w-8 rounded"
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
 * GovernorSeatRiskCard — partisan balance and competitive exposure summary.
 *
 * Shows D-held vs R-held governor totals as a proportional bar, plus
 * counts of competitive seats at risk per party.
 */
export function GovernorSeatRiskCard({ races }: GovernorSeatRiskCardProps) {
  const total = races.length;
  const dHeld = races.filter((r) => r.incumbent_party === "D").length;
  const rHeld = races.filter((r) => r.incumbent_party === "R").length;
  const dAtRisk = races.filter(
    (r) => r.incumbent_party === "D" && COMPETITIVE.has(r.rating),
  ).length;
  const rAtRisk = races.filter(
    (r) => r.incumbent_party === "R" && COMPETITIVE.has(r.rating),
  ).length;

  const dPct = total > 0 ? (dHeld / total) * 100 : 50;
  const rPct = 100 - dPct;

  const competitiveTotal = dAtRisk + rAtRisk;
  const narrativeParts = [
    dAtRisk > 0 ? `${dAtRisk} Democratic-held` : null,
    rAtRisk > 0 ? `${rAtRisk} Republican-held` : null,
  ].filter(Boolean);

  return (
    <section
      className="mb-8 rounded-md p-4 text-sm"
      aria-label="Seat Risk"
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
          Seat Risk
        </h2>
        <span className="font-mono text-xs" style={{ color: "var(--color-text-muted)" }}>
          {total} races on the ballot
        </span>
      </div>

      {/* Subheader */}
      <p className="mb-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
        Current partisan control and competitive exposure across all governor races
      </p>

      {/* Partisan balance bar */}
      <div
        className="mb-4"
        aria-label={`${dHeld} Democratic-held, ${rHeld} Republican-held governors`}
      >
        <div className="flex justify-between mb-1 text-xs font-semibold">
          <span style={{ color: DUSTY_INK.safeD }}>{dHeld}D</span>
          <span style={{ color: DUSTY_INK.safeR }}>{rHeld}R</span>
        </div>
        <div
          className="flex rounded overflow-hidden"
          style={{ height: 16, border: "1px solid var(--color-border)" }}
          role="img"
          aria-hidden="true"
        >
          <div
            style={{
              width: `${dPct}%`,
              backgroundColor: DUSTY_INK.safeD,
              opacity: 0.85,
            }}
          />
          <div
            style={{
              width: `${rPct}%`,
              backgroundColor: DUSTY_INK.safeR,
              opacity: 0.85,
            }}
          />
        </div>
      </div>

      {/* Risk breakdown */}
      <dl className="space-y-2 mb-4">
        <div className="flex items-center justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>D-held governors</dt>
          <dd
            className="font-mono font-semibold"
            style={{ color: DUSTY_INK.safeD }}
          >
            {dHeld}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>D seats competitive</dt>
          <dd
            className="font-mono font-semibold"
            style={{
              color:
                dAtRisk > 0
                  ? RATING_COLORS.lean_d
                  : "var(--color-text-muted)",
            }}
          >
            {dAtRisk}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>R-held governors</dt>
          <dd
            className="font-mono font-semibold"
            style={{ color: DUSTY_INK.safeR }}
          >
            {rHeld}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>R seats competitive</dt>
          <dd
            className="font-mono font-semibold"
            style={{
              color:
                rAtRisk > 0
                  ? RATING_COLORS.lean_r
                  : "var(--color-text-muted)",
            }}
          >
            {rAtRisk}
          </dd>
        </div>
      </dl>

      {/* Narrative */}
      <div
        className="pt-3 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        <p style={{ color: "var(--color-text-muted)" }}>
          {competitiveTotal === 0
            ? "No governor seats are rated competitive at this time."
            : `${competitiveTotal} seat${competitiveTotal === 1 ? "" : "s"} rated competitive — ${narrativeParts.join(" and ")}.`}
        </p>
      </div>
    </section>
  );
}
