// Shared display constants for type demographics — used by TypePanel and TypeCompareTable.

export const DEMO_DISPLAY: Record<string, { label: string; fmt: "pct" | "dollar" | "num" }> = {
  median_hh_income: { label: "Median income", fmt: "dollar" },
  median_age: { label: "Median age", fmt: "num" },
  pct_white_nh: { label: "White (non-Hispanic)", fmt: "pct" },
  pct_black: { label: "Black", fmt: "pct" },
  pct_hispanic: { label: "Hispanic", fmt: "pct" },
  pct_asian: { label: "Asian", fmt: "pct" },
  pct_bachelors_plus: { label: "Bachelor's+", fmt: "pct" },
  pct_graduate: { label: "Graduate degree", fmt: "pct" },
  pct_management: { label: "Management workers", fmt: "pct" },
  pct_owner_occupied: { label: "Owner-occupied", fmt: "pct" },
  pct_wfh: { label: "Work from home", fmt: "pct" },
  pct_transit: { label: "Transit commuters", fmt: "pct" },
  pct_car: { label: "Car commuters", fmt: "pct" },
  evangelical_share: { label: "Evangelical", fmt: "pct" },
  mainline_share: { label: "Mainline Protestant", fmt: "pct" },
  catholic_share: { label: "Catholic", fmt: "pct" },
  black_protestant_share: { label: "Black Protestant", fmt: "pct" },
  congregations_per_1000: { label: "Congregations/1K", fmt: "num" },
  religious_adherence_rate: { label: "Religious adherence", fmt: "num" },
  pop_per_sq_mi: { label: "Pop. density (per mi²)", fmt: "num" },
  land_area_sq_mi: { label: "Avg. land area (mi²)", fmt: "num" },
  net_migration_rate: { label: "Net migration rate", fmt: "num" },
  inflow_outflow_ratio: { label: "Migration in/out ratio", fmt: "num" },
};

export const DEMO_SKIP = new Set([
  "pop_total", "pop_white_nh", "pop_black", "pop_asian", "pop_hispanic",
  "housing_total", "housing_owner", "educ_total", "educ_bachelors_plus",
  "commute_total", "commute_car", "commute_transit", "commute_wfh",
  "n_counties",
  // Log-transformed / z-score derived columns — raw values are meaningless to users
  "log_median_hh_income", "log_pop_density",
  // Intermediate values that don't display well
  "avg_inflow_income", "migration_diversity",
  // Text fields that leak through demographics dict
  "narrative",
]);

export function formatDemoValue(value: number, fmt: "pct" | "dollar" | "num"): string {
  if (fmt === "dollar") return `$${Math.round(value).toLocaleString()}`;
  if (fmt === "pct") return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(1);
}

export function prettifyKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\bpct\b/g, "%")
    .replace(/\bhh\b/g, "household")
    .replace(/^./, (c) => c.toUpperCase());
}

export function inferFormat(key: string): "pct" | "dollar" | "num" {
  if (key.startsWith("pct_") || key.endsWith("_share")) return "pct";
  if (key.includes("income")) return "dollar";
  return "num";
}

/**
 * Format a dem share (0–1) as a partisan margin string.
 *
 * Examples: 0.532 → "D+3.2", 0.468 → "R+3.2", 0.5 → "EVEN", null → "—"
 *
 * @param demShare - Democratic two-party vote share (0–1 scale)
 * @param decimals - Number of decimal places (default 1)
 * @param nullText - Text to show when demShare is null (default "—")
 */
export function formatMargin(
  demShare: number | null,
  decimals = 1,
  nullText = "—",
): string {
  if (demShare === null) return nullText;
  const margin = Math.abs(demShare - 0.5) * 100;
  if (margin < 0.05) return "EVEN";
  return demShare > 0.5
    ? `D+${margin.toFixed(decimals)}`
    : `R+${margin.toFixed(decimals)}`;
}

/**
 * Format a confidence interval as a partisan margin range.
 *
 * Examples: (0.48, 0.55) → "R+2.0 to D+5.0"
 */
export function formatMarginRange(
  lo: number | null,
  hi: number | null,
  decimals = 1,
): string {
  if (lo === null || hi === null) return "—";
  return `${formatMargin(lo, decimals)} to ${formatMargin(hi, decimals)}`;
}

/**
 * Return { text, color } for a dem share — convenience for components that
 * need both the formatted string and the partisan color.
 */
export function marginLabel(
  demShare: number | null,
  decimals = 1,
  nullText = "—",
): { text: string; color: string } {
  return {
    text: formatMargin(demShare, decimals, nullText),
    color: marginColor(demShare),
  };
}

/** Return the partisan color for a dem share value. */
export function marginColor(demShare: number | null): string {
  if (demShare === null) return "var(--color-text-muted)";
  return demShare >= 0.5 ? "var(--color-dem)" : "var(--color-rep)";
}

// Legacy aliases — kept for backward compatibility with TypeCompareTable
/** @deprecated Use formatMargin instead */
export function formatLean(share: number | null): string {
  return formatMargin(share, 0);
}

/** @deprecated Use marginColor instead */
export function leanColor(share: number | null): string {
  return marginColor(share);
}
