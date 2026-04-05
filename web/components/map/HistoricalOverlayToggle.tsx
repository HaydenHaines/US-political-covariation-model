"use client";

/**
 * HistoricalOverlayToggle — compact year-selector buttons for the historical
 * presidential election overlay on the stained-glass county map.
 *
 * Renders "2020 · 2016 · 2012 · None" as pill buttons using the Dusty Ink
 * design system.  The selected year gets a filled background; unselected
 * buttons are outlined.  "None" clears the overlay.
 *
 * Positioning: absolute, top-right corner of the map, below the title bar.
 * The parent MapShell controls positioning in context; this component is
 * unstyled for layout — the caller passes a `style` prop if needed.
 */

import { DUSTY_INK } from "@/lib/config/palette";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Available presidential years for the historical overlay. */
export const HISTORICAL_YEARS = [2020, 2016, 2012] as const;
export type HistoricalYear = (typeof HISTORICAL_YEARS)[number];

interface HistoricalOverlayToggleProps {
  /** Currently selected year, or null when no overlay is active. */
  selectedYear: HistoricalYear | null;
  /** Called when the user picks a year or clears the overlay. */
  onYearChange: (year: HistoricalYear | null) => void;
  /** Whether the overlay data is currently loading. */
  isLoading?: boolean;
}

// ---------------------------------------------------------------------------
// Shared styles (using Dusty Ink design system values)
// ---------------------------------------------------------------------------

const CONTAINER_STYLE: React.CSSProperties = {
  position: "absolute",
  top: 12,
  right: 54,  // Clear of the zoom controls at right: 16 (32px wide + 6px gap)
  zIndex: 10,
  display: "flex",
  alignItems: "center",
  gap: 4,
  background: "var(--color-surface, #fafaf8)",
  border: `1px solid var(--color-border, ${DUSTY_INK.border})`,
  borderRadius: 6,
  padding: "3px 6px",
  boxShadow: "0 1px 3px rgba(58,54,50,0.08)",
};

const LABEL_STYLE: React.CSSProperties = {
  fontSize: 10,
  color: `var(--color-text-subtle, ${DUSTY_INK.textSubtle})`,
  fontFamily: "var(--font-sans)",
  marginRight: 2,
  letterSpacing: "0.03em",
  textTransform: "uppercase",
  userSelect: "none",
};

function PillButton({
  active,
  onClick,
  children,
  disabled,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "2px 8px",
        borderRadius: 4,
        border: active
          ? `1px solid var(--color-border, ${DUSTY_INK.border})`
          : `1px solid transparent`,
        background: active
          ? `var(--color-border, ${DUSTY_INK.border})`
          : "transparent",
        color: active
          ? `var(--color-text, ${DUSTY_INK.text})`
          : `var(--color-text-muted, ${DUSTY_INK.textMuted})`,
        fontSize: 11,
        fontFamily: "var(--font-sans)",
        fontWeight: active ? 600 : 400,
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.5 : 1,
        lineHeight: "18px",
        transition: "background 0.12s, color 0.12s",
      }}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function HistoricalOverlayToggle({
  selectedYear,
  onYearChange,
  isLoading = false,
}: HistoricalOverlayToggleProps) {
  return (
    <div style={CONTAINER_STYLE} aria-label="Historical election overlay">
      <span style={LABEL_STYLE}>History</span>

      {HISTORICAL_YEARS.map((year) => (
        <PillButton
          key={year}
          active={selectedYear === year}
          onClick={() => onYearChange(selectedYear === year ? null : year)}
          disabled={isLoading}
        >
          {year}
        </PillButton>
      ))}

      <PillButton
        active={selectedYear === null}
        onClick={() => onYearChange(null)}
        disabled={false}
      >
        None
      </PillButton>
    </div>
  );
}
