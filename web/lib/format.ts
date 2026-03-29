/**
 * Formatting utilities for WetherVane.
 *
 * Pure functions that transform raw API values into display strings.
 * No side effects, no dependencies on React or DOM.
 *
 * CRITICAL BUG HISTORY: formatMargin threshold must be 0.005 (0.5pp),
 * NOT 0.5 (50pp). This was caught in audit. The threshold is in the
 * same units as the margin (fraction, not percentage points).
 */

import { getFieldConfig } from "./config/display";
import type { FormatType } from "./config/display";

// ---------------------------------------------------------------------------
// Core formatters
// ---------------------------------------------------------------------------

/** Format a 0-1 fraction as a percentage string. "0.532" -> "53.2%" */
export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format a dollar value. "75630.18" -> "$75,630" */
export function formatCurrency(value: number): string {
  return `$${Math.round(value).toLocaleString("en-US")}`;
}

/** Format a plain number with locale grouping. "1066710" -> "1,066,710" */
export function formatNumber(value: number, decimals = 1): string {
  // Always use locale grouping (commas) and trim trailing zeros.
  // "259178.3" -> "259,178.3"; "35.0" -> "35"; "1066710" -> "1,066,710"
  return value.toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format a number as a rounded integer with locale grouping.
 * Used for population and other whole-number quantities.
 * "439179.0" -> "439,179"
 */
export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

/**
 * Format a per-1,000 value as a percentage.
 *
 * RCMS religious adherence rate uses "adherents per 1,000 population."
 * Divide by 10 to get percentage, then display with "%" suffix.
 * Do NOT multiply by 100 — that's what formatPercent does for 0-1 fractions.
 *
 * Example: 533.83 -> "53.4%"
 */
export function formatPer1000ToPct(value: number, decimals = 1): string {
  return `${(value / 10).toFixed(decimals)}%`;
}

/**
 * Format a raw value with no transformation. Falls back to locale string
 * for numbers, or string coercion for other types.
 */
export function formatRaw(value: number): string {
  return value.toLocaleString("en-US");
}

// ---------------------------------------------------------------------------
// Margin formatting (partisan lean)
// ---------------------------------------------------------------------------

/**
 * Format a dem share (0-1) as a partisan margin string.
 *
 * Examples:
 *   0.532 -> "D+3.2"
 *   0.468 -> "R+3.2"
 *   0.501 -> "EVEN"   (within 0.5pp)
 *   null  -> "---"
 *
 * CRITICAL: The threshold is 0.005 (0.5 percentage points), NOT 0.5.
 * A margin of 0.004 (D+0.4) should display as EVEN.
 */
export function formatMargin(
  demShare: number | null,
  decimals = 1,
  nullText = "\u2014",
): string {
  if (demShare === null || demShare === undefined) return nullText;
  const margin = demShare - 0.5;
  const abs = Math.abs(margin);
  if (abs < 0.005) return "EVEN";
  const pct = (abs * 100).toFixed(decimals);
  return margin > 0 ? `D+${pct}` : `R+${pct}`;
}

/**
 * Parse a dem share into text + party for colored display.
 *
 * Returns { text, party } where party is "dem" | "gop" | "even".
 * Components use this to apply partisan coloring to margin text.
 */
export function parseMargin(
  demShare: number | null,
): { text: string; party: "dem" | "gop" | "even" } {
  if (demShare === null || demShare === undefined) {
    return { text: "\u2014", party: "even" };
  }
  const margin = demShare - 0.5;
  const abs = Math.abs(margin);
  if (abs < 0.005) return { text: "EVEN", party: "even" };
  const pct = (abs * 100).toFixed(1);
  if (margin > 0) return { text: `D+${pct}`, party: "dem" };
  return { text: `R+${pct}`, party: "gop" };
}

// ---------------------------------------------------------------------------
// Time formatting
// ---------------------------------------------------------------------------

/**
 * Format a date string or Date as a human-readable "time ago" string.
 *
 * Examples:
 *   "2026-03-27" (1 day ago) -> "1d ago"
 *   "2026-03-15" (13 days ago) -> "13d ago"
 *   null -> "---"
 */
export function timeAgo(date: string | Date | null): string {
  if (!date) return "\u2014";
  const d = typeof date === "string" ? new Date(date) : date;
  const now = Date.now();
  const diffMs = now - d.getTime();

  if (diffMs < 0) return "just now";

  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;

  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;

  const years = Math.floor(days / 365);
  return `${years}y ago`;
}

// ---------------------------------------------------------------------------
// Field-aware formatting
// ---------------------------------------------------------------------------

/** Map from FormatType to formatter function. */
const FORMAT_DISPATCH: Record<FormatType, (value: number) => string> = {
  percent:        (v) => formatPercent(v),
  currency:       (v) => formatCurrency(v),
  number:         (v) => formatNumber(v),
  integer:        (v) => formatInteger(v),
  per1000_to_pct: (v) => formatPer1000ToPct(v),
  margin:         (v) => formatMargin(v),
  raw:            (v) => formatRaw(v),
};

/**
 * Format a field value using the display config registry.
 *
 * Looks up the field's format type in display.ts and applies the
 * corresponding formatter. Returns null-safe "---" for null/undefined.
 *
 * This is the primary entry point for components that render API fields
 * generically (e.g., demographics panels, type detail cards).
 */
export function formatField(key: string, value: number | null | undefined): string {
  if (value === null || value === undefined) return "\u2014";
  const config = getFieldConfig(key);
  const formatter = FORMAT_DISPATCH[config.format];
  return formatter(value);
}
