/**
 * ShiftHistoryChart — displays electoral shift data from the API's shift_profile.
 *
 * Renders shift values as a sorted table with partisan-colored bars for
 * visual context. Focuses on presidential shifts (pres_d_shift_*) by default
 * since those carry cross-state signal and are most meaningful to users.
 *
 * The shift values are signed fractions (positive = Dem shift, negative = R shift).
 */

import { formatMargin } from "@/lib/format";
import { DUSTY_INK } from "@/lib/config/palette";

interface ShiftHistoryChartProps {
  shiftProfile: Record<string, number>;
}

interface ParsedShift {
  key: string;
  label: string;
  value: number;
  /** Cycle pair string like "08→12" for sorting */
  sortKey: string;
}

/** Parse a shift field key into a display label. */
function parseShiftKey(key: string): { label: string; sortKey: string } | null {
  // Match pres_d_shift_XX_YY
  const match = /^pres_(d|r|turnout)_shift_(\d{2})_(\d{2})$/.exec(key);
  if (!match) return null;
  const [, type, from, to] = match;
  const typeLabel =
    type === "d" ? "Dem" : type === "r" ? "Rep" : "Turnout";
  return {
    label: `${typeLabel} shift '${from}→'${to}`,
    sortKey: `${from}_${to}_${type}`,
  };
}

/** Clamp a shift value to [-0.15, 0.15] for bar sizing. */
function barWidth(value: number, maxMagnitude = 0.15): number {
  return Math.min(Math.abs(value) / maxMagnitude, 1) * 100;
}

export function ShiftHistoryChart({ shiftProfile }: ShiftHistoryChartProps) {
  const shifts: ParsedShift[] = [];

  for (const [key, value] of Object.entries(shiftProfile)) {
    const parsed = parseShiftKey(key);
    if (!parsed) continue;
    // Only show Dem presidential shifts for cleaner display
    if (!key.startsWith("pres_d_shift_")) continue;
    shifts.push({ key, label: parsed.label, value, sortKey: parsed.sortKey });
  }

  // Sort chronologically
  shifts.sort((a, b) => a.sortKey.localeCompare(b.sortKey));

  if (shifts.length === 0) {
    return (
      <p style={{ color: "var(--color-text-muted)", fontSize: 14 }}>
        No shift data available.
      </p>
    );
  }

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "140px 1fr 72px",
          gap: "2px 12px",
          alignItems: "center",
          fontSize: 13,
        }}
      >
        {/* Header */}
        <span style={{ color: "var(--color-text-muted)", fontWeight: 600 }}>Cycle</span>
        <span style={{ color: "var(--color-text-muted)", fontWeight: 600 }}>Shift</span>
        <span
          style={{
            color: "var(--color-text-muted)",
            fontWeight: 600,
            textAlign: "right",
          }}
        >
          Value
        </span>

        {/* Data rows */}
        {shifts.map((s) => {
          const isDem = s.value >= 0;
          const barColor = isDem ? DUSTY_INK.leanD : DUSTY_INK.leanR;
          const width = barWidth(s.value);

          return (
            <>
              <span
                key={`label-${s.key}`}
                style={{ color: "var(--color-text)", paddingTop: 4 }}
              >
                {s.label}
              </span>
              <div
                key={`bar-${s.key}`}
                style={{
                  height: 14,
                  background: "var(--color-bg)",
                  borderRadius: 2,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                {isDem ? (
                  <div
                    style={{
                      position: "absolute",
                      left: "50%",
                      top: 0,
                      height: "100%",
                      width: `${width / 2}%`,
                      background: barColor,
                      borderRadius: "0 2px 2px 0",
                    }}
                  />
                ) : (
                  <div
                    style={{
                      position: "absolute",
                      right: "50%",
                      top: 0,
                      height: "100%",
                      width: `${width / 2}%`,
                      background: barColor,
                      borderRadius: "2px 0 0 2px",
                    }}
                  />
                )}
                {/* Center line */}
                <div
                  style={{
                    position: "absolute",
                    left: "50%",
                    top: 0,
                    width: 1,
                    height: "100%",
                    background: "var(--color-border)",
                  }}
                />
              </div>
              <span
                key={`val-${s.key}`}
                style={{
                  textAlign: "right",
                  fontVariantNumeric: "tabular-nums",
                  fontWeight: 600,
                  color: isDem ? DUSTY_INK.leanD : DUSTY_INK.leanR,
                  fontSize: 13,
                }}
              >
                {formatMargin(0.5 + s.value)}
              </span>
            </>
          );
        })}
      </div>
      <p
        style={{
          fontSize: 12,
          color: "var(--color-text-subtle, var(--color-text-muted))",
          marginTop: 12,
        }}
      >
        Presidential Dem shift by cycle — mean across member counties. Positive = Dem gain.
      </p>
    </div>
  );
}
