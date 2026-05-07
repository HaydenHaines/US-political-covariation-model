"use client";

import type { GovernorRaceData } from "@/lib/api";

const COMPETITIVE = new Set(["tossup", "lean_d", "lean_r"]);

function pctColor(val: number): string {
  if (val > 0) return "var(--forecast-lean-d)";
  if (val < 0) return "var(--forecast-lean-r)";
  return "var(--color-text-muted)";
}

function fmtPct(val: number): string {
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(1)}%`;
}

export function GovernorEconomicCardSkeleton() {
  return (
    <section
      className="mb-8 rounded-md p-4 text-sm animate-pulse"
      aria-label="State Economic Context loading"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <div
        className="h-5 w-48 rounded mb-3"
        style={{ background: "var(--color-border)" }}
      />
      <div
        className="h-4 w-72 rounded mb-4"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex justify-between">
            <div
              className="h-3 w-20 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
            <div
              className="h-3 w-32 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

interface GovernorEconomicCardProps {
  races: GovernorRaceData[];
}

export function GovernorEconomicCard({ races }: GovernorEconomicCardProps) {
  const competitiveWithEcon = races.filter(
    (r) => COMPETITIVE.has(r.rating) && r.econ !== null,
  );

  if (competitiveWithEcon.length === 0) return null;

  return (
    <section
      className="mb-8 rounded-md p-4 text-sm"
      aria-label="State Economic Context"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-3 mb-1">
        <h2
          className="font-serif text-lg"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          State Economic Context
        </h2>
        <span
          className="font-mono text-xs"
          style={{ color: "var(--color-text-muted)" }}
        >
          QCEW 2020–2023
        </span>
      </div>

      <p className="mb-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
        Wage growth and employment change for competitive states — aggregate county data
      </p>

      <div
        className="flex items-center justify-between gap-2 mb-2 pb-1 border-b text-xs"
        style={{
          borderColor: "var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        <span className="w-8">State</span>
        <div className="flex gap-6 font-mono">
          <span className="w-28 text-right">Wage Growth</span>
          <span className="w-28 text-right">Employment Δ</span>
        </div>
      </div>

      <dl className="space-y-1">
        {competitiveWithEcon.map((r) => {
          const econ = r.econ!;
          return (
            <div
              key={r.state}
              className="flex items-center justify-between gap-2"
            >
              <dt
                className="font-mono font-semibold w-8"
                style={{ color: "var(--color-text)" }}
              >
                {r.state}
              </dt>
              <div className="flex gap-6 font-mono font-semibold">
                <dd
                  className="w-28 text-right"
                  style={{ color: pctColor(econ.wage_growth_pct) }}
                >
                  {fmtPct(econ.wage_growth_pct)}
                </dd>
                <dd
                  className="w-28 text-right"
                  style={{ color: pctColor(econ.employment_change_pct) }}
                >
                  {fmtPct(econ.employment_change_pct)}
                </dd>
              </div>
            </div>
          );
        })}
      </dl>

      <div
        className="pt-3 mt-2 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        <p style={{ color: "var(--color-text-muted)" }}>
          Wage growth reflects average annual pay change (total wages ÷ employment). Employment
          change reflects total employment headcount shift. Both computed from BLS QCEW county data.
        </p>
      </div>
    </section>
  );
}
