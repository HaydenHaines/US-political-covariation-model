/**
 * Field display configuration: maps API field names to human-readable labels,
 * format types, UI sections, and sort order.
 *
 * This is the core mechanism for config-driven rendering. When the model adds
 * a new demographic field, add one entry here. No component changes needed.
 *
 * Update this file when:
 * - A new demographic or shift field is added to the API
 * - A field's display label or formatting needs to change
 * - Fields should be regrouped into different sections
 */

export type FormatType =
  | "percent"
  | "currency"
  | "number"
  | "integer"
  | "per1000_to_pct"
  | "margin"
  | "raw";

export type Section =
  | "race_ethnicity"
  | "economics"
  | "education"
  | "religion"
  | "geography"
  | "political"
  | "shift"
  | "other";

export interface FieldConfig {
  /** Human-readable label for display */
  label: string;
  /** How to format the value */
  format: FormatType;
  /** UI section for grouping in detail panels */
  section: Section;
  /** Sort order within section (lower = higher) */
  order: number;
}

/**
 * Master field configuration registry.
 *
 * Keys are API field names exactly as returned by the backend.
 * Values control how the field is displayed across all components.
 */
const FIELD_CONFIG: Record<string, FieldConfig> = {
  // -- Race & Ethnicity --
  pct_white_nh:           { label: "White (non-Hispanic)",  format: "percent",  section: "race_ethnicity", order: 0 },
  pct_black:              { label: "Black",                 format: "percent",  section: "race_ethnicity", order: 1 },
  pct_hispanic:           { label: "Hispanic",              format: "percent",  section: "race_ethnicity", order: 2 },
  pct_asian:              { label: "Asian",                 format: "percent",  section: "race_ethnicity", order: 3 },

  // -- Economics --
  median_hh_income:       { label: "Median household income", format: "currency", section: "economics", order: 0 },
  pct_owner_occupied:     { label: "Owner-occupied housing",  format: "percent",  section: "economics", order: 1 },
  pct_wfh:                { label: "Work from home",          format: "percent",  section: "economics", order: 2 },
  pct_management:         { label: "Management workers",      format: "percent",  section: "economics", order: 3 },
  pct_transit:            { label: "Transit commuters",       format: "percent",  section: "economics", order: 4 },
  pct_car:                { label: "Car commuters",           format: "percent",  section: "economics", order: 5 },
  net_migration_rate:     { label: "Net migration rate",      format: "percent",  section: "economics", order: 6 },
  inflow_outflow_ratio:   { label: "Inflow/outflow ratio",    format: "number",   section: "economics", order: 7 },

  // -- Education --
  pct_bachelors_plus:     { label: "Bachelor's degree+",  format: "percent", section: "education", order: 0 },
  pct_graduate:           { label: "Graduate degree+",    format: "percent", section: "education", order: 1 },

  // -- Religion --
  // GOTCHA: religious_adherence_rate is per-1,000 population (RCMS convention),
  // NOT a 0-1 fraction. Use "per1000_to_pct" format which divides by 10 and
  // appends "%". Do NOT use "percent" (which multiplies by 100).
  evangelical_share:        { label: "Evangelical",           format: "percent",        section: "religion", order: 0 },
  mainline_share:           { label: "Mainline Protestant",   format: "percent",        section: "religion", order: 1 },
  catholic_share:           { label: "Catholic",              format: "percent",        section: "religion", order: 2 },
  black_protestant_share:   { label: "Black Protestant",      format: "percent",        section: "religion", order: 3 },
  congregations_per_1000:   { label: "Congregations per 1K",  format: "number",         section: "religion", order: 4 },
  religious_adherence_rate: { label: "Religious adherence",   format: "per1000_to_pct", section: "religion", order: 5 },

  // -- Geography --
  pop_total:              { label: "Total population",  format: "integer", section: "geography", order: 0 },
  median_age:             { label: "Median age",         format: "number",  section: "geography", order: 1 },
  land_area_sq_mi:        { label: "Land area (sq mi)",  format: "number",  section: "geography", order: 2 },
  pop_per_sq_mi:          { label: "Population density", format: "number",  section: "geography", order: 3 },

  // -- Political --
  mean_dem_share:         { label: "Mean Dem share",       format: "margin",  section: "political", order: 0 },
  pred_dem_share:         { label: "Predicted Dem share",  format: "margin",  section: "political", order: 1 },
  mean_pred_dem_share:    { label: "Mean predicted Dem share", format: "margin", section: "political", order: 2 },
};

/**
 * Shift field pattern: pres_d_shift_XX_YY, pres_r_shift_XX_YY, pres_turnout_shift_XX_YY
 * These are generated dynamically rather than enumerated, since the set grows
 * with each election cycle.
 */
const SHIFT_PATTERN = /^pres_(d|r|turnout)_shift_(\d{2})_(\d{2})$/;

const SHIFT_LABELS: Record<string, string> = {
  d: "Dem shift",
  r: "Rep shift",
  turnout: "Turnout shift",
};

function yearLabel(twoDigit: string): string {
  const n = parseInt(twoDigit, 10);
  return n >= 90 ? `'${twoDigit}` : `'${twoDigit}`;
}

function buildShiftConfig(key: string): FieldConfig | null {
  const match = SHIFT_PATTERN.exec(key);
  if (!match) return null;
  const [, type, fromYear, toYear] = match;
  const label = `${SHIFT_LABELS[type]} ${yearLabel(fromYear)}-${yearLabel(toYear)}`;
  // Sort by year pair, then by type (d=0, r=1, turnout=2)
  const typeOrder = type === "d" ? 0 : type === "r" ? 1 : 2;
  const order = parseInt(fromYear, 10) * 10 + typeOrder;
  return { label, format: "margin", section: "shift", order };
}

/**
 * Fields to skip in generic demographic renders — raw counts, intermediate
 * values, and log-transformed columns that aren't meaningful to users.
 */
export const SKIP_FIELDS = new Set([
  "pop_white_nh", "pop_black", "pop_asian", "pop_hispanic",
  "housing_total", "housing_owner", "educ_total", "educ_bachelors_plus",
  "commute_total", "commute_car", "commute_transit", "commute_wfh",
  "n_counties",
  "log_median_hh_income",
  // log-transformed columns are not meaningful to display directly
  "log_pop_density",
  "avg_inflow_income", "migration_diversity",
  "narrative",
]);

/** Section display order and labels. */
export const SECTION_META: Record<Section, { label: string; order: number }> = {
  political:       { label: "Political",          order: 0 },
  race_ethnicity:  { label: "Race & Ethnicity",   order: 1 },
  economics:       { label: "Economics",           order: 2 },
  education:       { label: "Education",           order: 3 },
  religion:        { label: "Religion",            order: 4 },
  geography:       { label: "Geography",           order: 5 },
  shift:           { label: "Electoral Shifts",    order: 6 },
  other:           { label: "Other",               order: 7 },
};

/**
 * Titleize a raw API field key as a fallback label.
 * "pct_white_nh" -> "Pct White Nh"
 */
function titleize(key: string): string {
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/**
 * Get the display configuration for a field. Returns the registered config
 * for known fields, dynamically builds config for shift fields, and returns
 * a sensible fallback for unknown fields.
 */
export function getFieldConfig(key: string): FieldConfig {
  // Known static field
  const known = FIELD_CONFIG[key];
  if (known) return known;

  // Dynamic shift field
  const shiftConfig = buildShiftConfig(key);
  if (shiftConfig) return shiftConfig;

  // Fallback: titleize the key, format as raw, put in "other" section
  return {
    label: titleize(key),
    format: "raw",
    section: "other",
    order: 999,
  };
}

/**
 * Get all field configs for a set of keys, grouped by section and sorted.
 * Useful for rendering a demographics panel from a Record<string, number>.
 */
export function groupFieldsBySection(
  keys: string[],
): { section: Section; label: string; fields: { key: string; config: FieldConfig }[] }[] {
  const groups = new Map<Section, { key: string; config: FieldConfig }[]>();

  for (const key of keys) {
    if (SKIP_FIELDS.has(key)) continue;
    const config = getFieldConfig(key);
    const existing = groups.get(config.section);
    if (existing) {
      existing.push({ key, config });
    } else {
      groups.set(config.section, [{ key, config }]);
    }
  }

  // Sort fields within each section
  const groupEntries = Array.from(groups.entries());
  for (const [, fields] of groupEntries) {
    fields.sort((a, b) => a.config.order - b.config.order);
  }

  // Sort sections by their meta order
  return groupEntries
    .sort(([a], [b]) => SECTION_META[a].order - SECTION_META[b].order)
    .map(([section, fields]) => ({
      section,
      label: SECTION_META[section].label,
      fields,
    }));
}
