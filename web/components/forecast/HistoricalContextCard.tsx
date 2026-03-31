/**
 * HistoricalContextCard — displays the last election result and 2024
 * presidential lean for a tracked race, plus the model's projected shift.
 *
 * This is a pure Server Component: it receives pre-fetched data as props
 * and renders static HTML.  No client-side state needed.
 */

interface LastRaceResult {
  year: number;
  winner: string;
  party: string;
  /** Percentage-point margin. Positive = Dem advantage, negative = Rep. */
  margin: number;
  note?: string | null;
}

interface PresidentialResult {
  winner: string;
  party: string;
  /** Percentage-point margin. Positive = Dem advantage, negative = Rep. */
  margin: number;
  note?: string | null;
}

export interface HistoricalContext {
  last_race: LastRaceResult;
  presidential_2024: PresidentialResult;
  /** Model forecast margin minus last_race margin (pp). Positive = Dem shift. */
  forecast_shift: number | null;
}

interface Props {
  context: HistoricalContext;
  stateName: string;
  stateAbbr: string;
}

/** Format a raw percentage-point margin (not dem share) as "D+X.X" or "R+X.X". */
function formatPpMargin(marginPp: number, decimals = 1): string {
  const abs = Math.abs(marginPp);
  if (abs < 0.05) return "EVEN";
  const formatted = abs.toFixed(decimals);
  return marginPp > 0 ? `D+${formatted}` : `R+${formatted}`;
}

function partyColor(party: string): string {
  if (party === "D") return "var(--forecast-safe-d)";
  if (party === "R") return "var(--forecast-safe-r)";
  return "var(--forecast-tossup)";
}

/** Format a shift value: positive = "D+X" shift, negative = "R+X" shift. */
function formatShift(shiftPp: number): { label: string; color: string } {
  const abs = Math.abs(shiftPp);
  if (abs < 0.05) return { label: "No shift from last result", color: "var(--forecast-tossup)" };
  const formatted = abs.toFixed(1);
  if (shiftPp > 0) {
    return { label: `D+${formatted} shift vs. last result`, color: "var(--forecast-safe-d)" };
  }
  return { label: `R+${formatted} shift vs. last result`, color: "var(--forecast-safe-r)" };
}

export function HistoricalContextCard({ context, stateName, stateAbbr }: Props) {
  const { last_race, presidential_2024, forecast_shift } = context;

  const lastMarginLabel = formatPpMargin(last_race.margin);
  const lastPartyColor = partyColor(last_race.party);

  const presMarginLabel = formatPpMargin(presidential_2024.margin);
  const presPartyColor = partyColor(presidential_2024.party);

  const shiftInfo = forecast_shift !== null ? formatShift(forecast_shift) : null;

  return (
    <section
      className="mb-10 rounded-md p-4 text-sm"
      aria-label="Historical electoral context"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <h2
        className="font-serif text-lg mb-4"
        style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
      >
        Historical Context
      </h2>

      <dl className="space-y-3">
        {/* Last election for this seat */}
        <div className="flex items-start justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>
            {last_race.year} result{last_race.note ? ` (${last_race.note})` : ""}
          </dt>
          <dd className="text-right">
            <span style={{ color: "var(--color-text)" }}>
              {last_race.winner} ({last_race.party})
            </span>{" "}
            <span className="font-mono font-bold" style={{ color: lastPartyColor }}>
              {lastMarginLabel}
            </span>
          </dd>
        </div>

        {/* 2024 presidential lean */}
        <div className="flex items-start justify-between gap-4">
          <dt style={{ color: "var(--color-text-muted)" }}>
            2024 presidential ({stateAbbr})
          </dt>
          <dd className="text-right">
            <span style={{ color: "var(--color-text)" }}>
              {presidential_2024.winner} ({presidential_2024.party})
            </span>{" "}
            <span className="font-mono font-bold" style={{ color: presPartyColor }}>
              {presMarginLabel}
            </span>
          </dd>
        </div>

        {/* Model-projected shift vs. last result */}
        {shiftInfo !== null && (
          <div
            className="pt-3 mt-1 border-t flex items-start justify-between gap-4"
            style={{ borderColor: "var(--color-border)" }}
          >
            <dt style={{ color: "var(--color-text-muted)" }}>
              Model vs. last result
            </dt>
            <dd
              className="text-right font-mono font-bold"
              style={{ color: shiftInfo.color }}
            >
              {shiftInfo.label}
            </dd>
          </div>
        )}
      </dl>
    </section>
  );
}
