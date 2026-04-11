"use client";

import { useState, useMemo } from "react";
import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { useFundamentals } from "@/lib/hooks/use-fundamentals";
import { PALETTE } from "@/lib/config/palette";

// ── Constants ────────────────────────────────────────────────────────────

/** Slider range: D+10 to R+10 expressed as signed percentage points. */
const ENV_MIN = -10;
const ENV_MAX = 10;
const ENV_STEP = 0.5;

/** Turnout levels and their modeled effect on margins. */
const TURNOUT_LEVELS = [
  {
    label: "Low",
    /** Low turnout historically favors GOP by ~1.5pp in midterms. */
    marginShift: -0.015,
  },
  {
    label: "Medium",
    marginShift: 0,
  },
  {
    label: "High",
    /** High turnout historically favors Dems by ~1.5pp. */
    marginShift: 0.015,
  },
] as const;

// ── Rating thresholds (mirror palette.ts marginToRating) ─────────────────

function scenarioRating(margin: number): string {
  const abs = Math.abs(margin);
  if (abs < 0.03) return "tossup";
  if (margin > 0) {
    if (abs >= 0.15) return "safe_d";
    if (abs >= 0.08) return "likely_d";
    return "lean_d";
  }
  if (abs >= 0.15) return "safe_r";
  if (abs >= 0.08) return "likely_r";
  return "lean_r";
}

// ── Helpers ───────────────────────────────────────────────────────────────

function formatEnvLabel(pp: number): string {
  if (Math.abs(pp) < 0.05) return "EVEN";
  const abs = Math.abs(pp).toFixed(1);
  return pp > 0 ? `D+${abs}` : `R+${abs}`;
}

function envColor(pp: number): string {
  if (Math.abs(pp) < 0.05) return "var(--forecast-tossup)";
  return pp > 0 ? PALETTE.DEM_PRIMARY : PALETTE.GOP_PRIMARY;
}

// ── Loading skeleton ──────────────────────────────────────────────────────

function ScenarioSlidersSkeleton() {
  return (
    <section
      className="mb-8 rounded-md p-4 text-sm animate-pulse"
      aria-label="Scenario Explorer loading"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <div
        className="h-5 w-44 rounded mb-3"
        style={{ background: "var(--color-border)" }}
      />
      <div
        className="h-4 w-64 rounded mb-4"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div
        className="h-8 w-full rounded mb-3"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div
        className="h-8 w-full rounded"
        style={{ background: "var(--color-border-subtle)" }}
      />
    </section>
  );
}

// ── Main component ────────────────────────────────────────────────────────

/**
 * NationalScenarioSliders — exploratory "what if" tool for Senate scenarios.
 *
 * Lets users adjust a national environment slider (D+10 to R+10) and a
 * turnout toggle (Low/Medium/High) to see how projected seat totals change.
 * All calculations are client-side from existing race data — this does NOT
 * update the header tipping point bar or the main forecast.
 *
 * The environment shift applies a uniform margin adjustment to every race.
 * The turnout toggle applies a smaller, historically-motivated shift.
 */
export function NationalScenarioSliders() {
  const { data: overview, error: overviewError, isLoading: overviewLoading } = useSenateOverview();
  const { data: fundamentals } = useFundamentals();

  // Default environment: current combined shift from fundamentals, or 0
  const defaultEnv = fundamentals?.combined_shift_pp ?? 0;

  const [envShiftPp, setEnvShiftPp] = useState<number | null>(null);
  const [turnoutIdx, setTurnoutIdx] = useState(1); // Medium

  // Resolve the effective environment value — use default until user touches slider
  const effectiveEnvPp = envShiftPp ?? defaultEnv;

  // Recalculate scenario whenever inputs change
  const scenario = useMemo(() => {
    if (!overview) return null;

    const turnout = TURNOUT_LEVELS[turnoutIdx];
    // Convert environment from pp to fraction (margins are in 0-1 scale centered at 0)
    const envShiftFrac = effectiveEnvPp / 100;
    const turnoutShiftFrac = turnout.marginShift;
    const totalShift = envShiftFrac + turnoutShiftFrac;

    // Adjust each race margin and re-rate
    let demWins = 0;
    let gopWins = 0;
    const adjustedRaces = overview.races.map((race) => {
      const newMargin = race.margin + totalShift;
      const newRating = scenarioRating(newMargin);

      // Count seats: positive margin = Dem win, negative = GOP win
      // Tossups split evenly in expectation
      if (newMargin > 0) {
        demWins++;
      } else if (newMargin < 0) {
        gopWins++;
      }
      // Exactly 0 counted for neither — rare edge case

      return { ...race, margin: newMargin, rating: newRating };
    });

    // Add safe holdover seats
    const totalDem = overview.dem_seats_safe + demWins;
    const totalGop = overview.gop_seats_safe + gopWins;

    return { adjustedRaces, totalDem, totalGop };
  }, [overview, effectiveEnvPp, turnoutIdx]);

  if (overviewLoading) {
    return <ScenarioSlidersSkeleton />;
  }

  if (overviewError || !overview || !scenario) {
    return null;
  }

  const { totalDem, totalGop } = scenario;

  // Bar widths as percentages of 100 seats
  const demPct = totalDem;
  const gopPct = totalGop;

  return (
    <section
      className="mb-8 rounded-md p-4 text-sm"
      aria-label="Scenario Explorer"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      {/* Header */}
      <h2
        className="font-serif text-lg mb-1"
        style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
      >
        Scenario Explorer
      </h2>
      <p className="mb-5 text-xs" style={{ color: "var(--color-text-muted)" }}>
        Adjust the national environment and turnout to see how Senate projections
        shift. This is an exploratory tool — it does not change the main forecast.
      </p>

      {/* National environment slider */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-2">
          <label
            htmlFor="env-slider"
            style={{ color: "var(--color-text-muted)" }}
          >
            National Environment
          </label>
          <span
            className="font-mono font-bold"
            style={{ color: envColor(effectiveEnvPp) }}
          >
            {formatEnvLabel(effectiveEnvPp)}
          </span>
        </div>
        <input
          id="env-slider"
          type="range"
          min={ENV_MIN}
          max={ENV_MAX}
          step={ENV_STEP}
          value={effectiveEnvPp}
          onChange={(e) => setEnvShiftPp(parseFloat(e.target.value))}
          className="w-full scenario-slider"
          aria-label={`National environment: ${formatEnvLabel(effectiveEnvPp)}`}
        />
        <div
          className="flex justify-between text-xs mt-1"
          style={{ color: "var(--color-text-muted)" }}
        >
          <span>R+10</span>
          <span>EVEN</span>
          <span>D+10</span>
        </div>
      </div>

      {/* Turnout toggle */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-2">
          <span style={{ color: "var(--color-text-muted)" }}>Turnout</span>
          <span
            className="font-mono font-semibold"
            style={{ color: "var(--color-text)" }}
          >
            {TURNOUT_LEVELS[turnoutIdx].label}
          </span>
        </div>
        <div className="flex gap-2">
          {TURNOUT_LEVELS.map((level, idx) => (
            <button
              key={level.label}
              onClick={() => setTurnoutIdx(idx)}
              className="flex-1 py-1.5 px-3 rounded text-xs font-medium transition-colors"
              style={{
                background:
                  idx === turnoutIdx
                    ? "var(--color-text)"
                    : "var(--color-border)",
                color:
                  idx === turnoutIdx
                    ? "var(--color-surface)"
                    : "var(--color-text-muted)",
              }}
              aria-pressed={idx === turnoutIdx}
            >
              {level.label}
            </button>
          ))}
        </div>
      </div>

      {/* Scenario result: simple seat bar */}
      <div
        className="pt-4 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        {/* Seat count labels */}
        <div className="flex justify-between mb-2 font-semibold">
          <span style={{ color: PALETTE.DEM_PRIMARY }}>
            {totalDem}D
          </span>
          <span
            className="text-xs font-normal"
            style={{ color: "var(--color-text-muted)" }}
          >
            51 for control
          </span>
          <span style={{ color: PALETTE.GOP_PRIMARY }}>
            {totalGop}R
          </span>
        </div>

        {/* Horizontal seat bar */}
        <div
          className="relative flex w-full rounded-md overflow-hidden"
          style={{ height: 28, border: "1px solid var(--color-border)" }}
        >
          {/* Dem segment */}
          <div
            style={{
              width: `${demPct}%`,
              backgroundColor: PALETTE.DEM_PRIMARY,
              transition: "width 200ms ease",
            }}
          />
          {/* GOP segment */}
          <div
            style={{
              width: `${gopPct}%`,
              backgroundColor: PALETTE.GOP_PRIMARY,
              transition: "width 200ms ease",
            }}
          />

          {/* 51-seat majority marker */}
          <div
            className="absolute top-0 pointer-events-none"
            style={{
              left: "51%",
              height: "100%",
              width: 2,
              backgroundColor: "var(--color-text)",
              transform: "translateX(-50%)",
            }}
            aria-hidden="true"
          />
        </div>

        {/* Delta from current forecast */}
        <p
          className="mt-2 text-xs text-center"
          style={{ color: "var(--color-text-muted)" }}
        >
          {totalDem === overview.dem_projected && totalGop === overview.gop_projected
            ? "Matches current forecast"
            : `${totalDem > overview.dem_projected ? "+" : ""}${totalDem - overview.dem_projected}D / ${totalGop > overview.gop_projected ? "+" : ""}${totalGop - overview.gop_projected}R vs. current forecast`}
        </p>
      </div>
    </section>
  );
}
