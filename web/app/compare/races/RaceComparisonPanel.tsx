"use client";

import Link from "next/link";
import type { RaceComparisonData } from "./RaceComparisonClient";
import { formatMargin, parseMargin, absoluteDate } from "@/lib/format";
import { marginToRating, RATING_LABELS, RATING_COLORS, getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import { STATE_NAMES } from "@/lib/config/states";

// ── Helpers ────────────────────────────────────────────────────────────────

/** Format a confidence interval as a partisan range string, e.g. "D+1.0 – D+9.3". */
function formatCI(lo: number | null, hi: number | null): string {
  if (lo === null || hi === null) return "—";
  return `${formatMargin(lo)} – ${formatMargin(hi)}`;
}

/** Format a party margin as "+X.Xpp" with sign and party prefix. */
function formatLastMargin(margin: number): string {
  const abs = Math.abs(margin).toFixed(1);
  if (Math.abs(margin) < 0.05) return "EVEN";
  return margin > 0 ? `D+${abs}` : `R+${abs}`;
}

/** Format a forecast shift, e.g. "+2.3pp (Dem)" or "−1.1pp (Rep)". */
function formatShift(shift: number | null): string {
  if (shift === null) return "—";
  const abs = Math.abs(shift).toFixed(1);
  if (Math.abs(shift) < 0.05) return "No change";
  return shift > 0 ? `+${abs}pp (Dem)` : `−${abs}pp (Rep)`;
}

// ── Section header ────────────────────────────────────────────────────────

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h3
      className="text-xs font-semibold uppercase tracking-wide mb-3"
      style={{ color: "var(--color-text-muted)" }}
    >
      {children}
    </h3>
  );
}

// ── Stat row ──────────────────────────────────────────────────────────────

function StatRow({
  label,
  value,
  valueColor,
}: {
  label: string;
  value: React.ReactNode;
  valueColor?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-2 text-sm py-1">
      <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span
        className="font-mono font-semibold text-right"
        style={{ color: valueColor ?? "var(--color-text)" }}
      >
        {value}
      </span>
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────

interface RaceComparisonPanelProps {
  data: RaceComparisonData;
}

/**
 * A single column in the side-by-side race comparison.
 *
 * Renders four sections:
 * 1. Race header with margin, rating badge, and CI
 * 2. Polls — count, latest poll date/pollster, confidence label
 * 3. Electoral type composition — top types by vote weight
 * 4. Historical context — last result, 2024 presidential result, forecast shift
 */
export function RaceComparisonPanel({ data }: RaceComparisonPanelProps) {
  const stateName = STATE_NAMES[data.state_abbr] ?? data.state_abbr;
  const raceTitle = `${data.year} ${stateName} ${data.race_type}`;

  const { text: marginText, party } = parseMargin(data.prediction);
  const marginColor =
    party === "dem"
      ? "var(--forecast-safe-d)"
      : party === "gop"
      ? "var(--forecast-safe-r)"
      : "var(--forecast-tossup)";

  const rating = data.prediction !== null ? marginToRating(data.prediction) : null;
  const ratingLabel = rating ? RATING_LABELS[rating] : "Unknown";
  const ratingColor = rating ? RATING_COLORS[rating] : "var(--color-text-muted)";

  return (
    <div
      className="rounded-md overflow-hidden"
      style={{
        border: "1px solid var(--color-border)",
        background: "var(--color-surface)",
      }}
    >
      {/* ── Race header ──────────────────────────────────────────────── */}
      <div
        className="px-4 pt-4 pb-3"
        style={{ borderBottom: "1px solid var(--color-border)" }}
      >
        <div className="flex items-start justify-between gap-2 mb-1">
          <Link
            href={`/forecast/${data.slug}`}
            className="text-base font-bold leading-tight hover:underline"
            style={{
              fontFamily: "var(--font-serif)",
              color: "var(--color-text)",
              textDecoration: "none",
            }}
          >
            {raceTitle}
          </Link>
          {rating && (
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded flex-shrink-0"
              style={{
                background: ratingColor,
                color: "#fff",
              }}
            >
              {ratingLabel}
            </span>
          )}
        </div>

        {/* Prediction margin */}
        <div className="flex items-baseline gap-3 mt-2">
          <span
            className="text-2xl font-bold font-mono"
            style={{ color: marginColor }}
          >
            {marginText}
          </span>
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            predicted margin
          </span>
        </div>

        {/* 90% confidence interval */}
        <div className="text-xs mt-1" style={{ color: "var(--color-text-muted)" }}>
          90% CI: {formatCI(data.pred_lo90, data.pred_hi90)}
        </div>

        {/* Link to full race detail */}
        <div className="mt-2">
          <Link
            href={`/forecast/${data.slug}`}
            className="text-xs"
            style={{ color: "var(--forecast-safe-d)" }}
          >
            View full forecast →
          </Link>
        </div>
      </div>

      {/* ── Polls section ─────────────────────────────────────────────── */}
      <div
        className="px-4 py-3"
        style={{ borderBottom: "1px solid var(--color-border)" }}
      >
        <SectionHeader>Polls</SectionHeader>
        <StatRow label="Poll count" value={data.n_polls} />
        {data.poll_confidence && (
          <StatRow
            label="Poll confidence"
            value={data.poll_confidence.label}
            valueColor={
              data.poll_confidence.label === "High"
                ? "var(--forecast-safe-d)"
                : data.poll_confidence.label === "Medium"
                ? "var(--forecast-tossup)"
                : "var(--forecast-lean-r)"
            }
          />
        )}
        {data.poll_confidence && (
          <div
            className="text-xs mt-1"
            style={{ color: "var(--color-text-muted)", opacity: 0.8 }}
          >
            {data.poll_confidence.tooltip}
          </div>
        )}
        {data.latest_poll && (
          <>
            <div
              className="mt-2 pt-2 text-xs font-semibold uppercase tracking-wide"
              style={{ color: "var(--color-text-muted)", borderTop: "1px solid var(--color-border-subtle)" }}
            >
              Latest poll
            </div>
            <StatRow
              label={data.latest_poll.pollster ?? "Unknown pollster"}
              value={absoluteDate(data.latest_poll.date)}
            />
            <StatRow
              label="Dem share"
              value={`${(data.latest_poll.dem_share * 100).toFixed(1)}%`}
              valueColor={
                data.latest_poll.dem_share > 0.505
                  ? "var(--forecast-safe-d)"
                  : data.latest_poll.dem_share < 0.495
                  ? "var(--forecast-safe-r)"
                  : "var(--forecast-tossup)"
              }
            />
            {data.latest_poll.grade && (
              <StatRow label="Pollster grade" value={data.latest_poll.grade} />
            )}
          </>
        )}
        {data.n_polls === 0 && (
          <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            No polls available — forecast is based on structural model prior only.
          </p>
        )}
      </div>

      {/* ── Type composition section ───────────────────────────────────── */}
      <div
        className="px-4 py-3"
        style={{ borderBottom: "1px solid var(--color-border)" }}
      >
        <SectionHeader>Electoral Type Composition</SectionHeader>
        {data.type_breakdown.length === 0 ? (
          <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            No type data available.
          </p>
        ) : (
          <div className="flex flex-col gap-1.5">
            {data.type_breakdown.map((t) => {
              const color = rgbToHex(getSuperTypeColor(t.type_id % 10));
              const { text: typeMarg, party: typeParty } = parseMargin(t.mean_pred_dem_share);
              const typeMarginColor =
                typeParty === "dem"
                  ? "var(--forecast-safe-d)"
                  : typeParty === "gop"
                  ? "var(--forecast-safe-r)"
                  : "var(--forecast-tossup)";

              return (
                <div
                  key={t.type_id}
                  className="flex items-center justify-between gap-2 rounded px-3 py-1.5 text-xs"
                  style={{
                    background: "var(--color-bg)",
                    borderLeft: `3px solid ${color}`,
                    border: `1px solid var(--color-border)`,
                    borderLeftColor: color,
                    borderLeftWidth: "3px",
                  }}
                >
                  <Link
                    href={`/type/${t.type_id}`}
                    className="truncate font-medium"
                    style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
                  >
                    {t.display_name}
                  </Link>
                  <span
                    className="font-mono font-bold flex-shrink-0"
                    style={{ color: typeMarginColor }}
                  >
                    {typeMarg}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Historical context section ─────────────────────────────────── */}
      <div className="px-4 py-3">
        <SectionHeader>Historical Context</SectionHeader>
        {data.historical_context ? (
          <>
            {/* Last race result */}
            <div className="text-xs font-semibold mb-1" style={{ color: "var(--color-text-muted)" }}>
              Last race ({data.historical_context.last_race.year})
            </div>
            <StatRow
              label={data.historical_context.last_race.winner}
              value={formatLastMargin(data.historical_context.last_race.margin)}
              valueColor={
                data.historical_context.last_race.party === "D"
                  ? "var(--forecast-safe-d)"
                  : "var(--forecast-safe-r)"
              }
            />
            {data.historical_context.last_race.note && (
              <div
                className="text-xs mb-2 italic"
                style={{ color: "var(--color-text-muted)", opacity: 0.75 }}
              >
                {data.historical_context.last_race.note}
              </div>
            )}

            {/* 2024 presidential */}
            <div
              className="text-xs font-semibold mb-1 mt-2 pt-2"
              style={{
                color: "var(--color-text-muted)",
                borderTop: "1px solid var(--color-border-subtle)",
              }}
            >
              2024 Presidential
            </div>
            <StatRow
              label={data.historical_context.presidential_2024.winner}
              value={formatLastMargin(data.historical_context.presidential_2024.margin)}
              valueColor={
                data.historical_context.presidential_2024.party === "D"
                  ? "var(--forecast-safe-d)"
                  : "var(--forecast-safe-r)"
              }
            />

            {/* Forecast shift */}
            <div
              className="text-xs font-semibold mb-1 mt-2 pt-2"
              style={{
                color: "var(--color-text-muted)",
                borderTop: "1px solid var(--color-border-subtle)",
              }}
            >
              Forecast vs. last race
            </div>
            <StatRow
              label="Shift"
              value={formatShift(data.historical_context.forecast_shift)}
              valueColor={
                data.historical_context.forecast_shift === null
                  ? undefined
                  : data.historical_context.forecast_shift > 0.05
                  ? "var(--forecast-safe-d)"
                  : data.historical_context.forecast_shift < -0.05
                  ? "var(--forecast-safe-r)"
                  : "var(--color-text-muted)"
              }
            />
          </>
        ) : (
          <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            Historical context is only available for tracked competitive races.
          </p>
        )}
      </div>
    </div>
  );
}
