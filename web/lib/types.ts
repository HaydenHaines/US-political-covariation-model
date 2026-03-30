/**
 * Shared TypeScript interfaces matching WetherVane API v1 responses.
 *
 * This is the canonical location for all API response types. The older
 * definitions in `api.ts` remain for backward compatibility but should
 * be migrated to import from here.
 *
 * When the API changes, update these interfaces FIRST, then fix callers.
 */

// ---------------------------------------------------------------------------
// Senate
// ---------------------------------------------------------------------------

export interface SenateRace {
  state: string;
  race: string;
  slug: string;
  rating: string;
  margin: number;
  n_polls: number;
}

export interface SenateOverview {
  headline: string;
  subtitle: string;
  dem_seats_safe: number;
  gop_seats_safe: number;
  races: SenateRace[];
  state_colors: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Types (electoral community types)
// ---------------------------------------------------------------------------

export interface TypeSummary {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  median_hh_income: number | null;
  pct_bachelors_plus: number | null;
  pct_white_nh: number | null;
  log_pop_density: number | null;
}

export interface TypeCounty {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
}

export interface TypeDetail {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  median_hh_income: number | null;
  pct_bachelors_plus: number | null;
  pct_white_nh: number | null;
  log_pop_density: number | null;
  counties: TypeCounty[];
  demographics: Record<string, number>;
  shift_profile: Record<string, number> | null;
  narrative: string | null;
}

// ---------------------------------------------------------------------------
// Super-types
// ---------------------------------------------------------------------------

export interface SuperTypeSummary {
  super_type_id: number;
  display_name: string;
  member_type_ids: number[];
  n_counties: number;
}

// ---------------------------------------------------------------------------
// Correlated types (from LW electoral covariance matrix)
// ---------------------------------------------------------------------------

export interface CorrelatedTypeData {
  type_id: number;
  display_name: string;
  super_type_id: number;
  n_counties: number;
  mean_pred_dem_share: number | null;
  /** Ledoit-Wolf regularized electoral correlation coefficient [-1, 1] */
  correlation: number;
}

// ---------------------------------------------------------------------------
// Counties
// ---------------------------------------------------------------------------

export interface CountyDemographics {
  pop_total: number | null;
  pct_white_nh: number | null;
  pct_black: number | null;
  pct_hispanic: number | null;
  pct_asian: number | null;
  median_age: number | null;
  median_hh_income: number | null;
  pct_bachelors_plus: number | null;
  pct_owner_occupied: number | null;
  pct_wfh: number | null;
  pct_management: number | null;
  evangelical_share: number | null;
  mainline_share: number | null;
  catholic_share: number | null;
  black_protestant_share: number | null;
  congregations_per_1000: number | null;
  religious_adherence_rate: number | null;
  [key: string]: number | null | undefined;
}

export interface SiblingCounty {
  county_fips: string;
  county_name: string;
  state_abbr: string;
  pred_dem_share: number | null;
}

export interface CountyDetail {
  county_fips: string;
  county_name: string;
  state_abbr: string;
  dominant_type: number;
  super_type: number;
  type_display_name: string;
  super_type_display_name: string;
  narrative: string | null;
  pred_dem_share: number | null;
  demographics: CountyDemographics;
  sibling_counties: SiblingCounty[];
}

// ---------------------------------------------------------------------------
// Forecast
// ---------------------------------------------------------------------------

export interface ForecastRow {
  county_fips: string;
  county_name: string;
  state_abbr: string;
  race: string;
  pred_dem_share: number | null;
  pred_std: number | null;
  pred_lo90: number | null;
  pred_hi90: number | null;
  state_pred: number | null;
  poll_avg: number | null;
}

// ---------------------------------------------------------------------------
// Polls
// ---------------------------------------------------------------------------

export interface PollRow {
  race: string;
  geography: string;
  geo_level: string;
  dem_share: number;
  n_sample: number;
  date: string | null;
  pollster: string | null;
}

// ---------------------------------------------------------------------------
// Scatter data (types explorer)
// ---------------------------------------------------------------------------

export interface TypeScatterPoint {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  demographics: Record<string, number>;
  shift_profile: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Ratings (partisan lean categories)
// ---------------------------------------------------------------------------

export type Rating =
  | "safe_d"
  | "likely_d"
  | "lean_d"
  | "tossup"
  | "lean_r"
  | "likely_r"
  | "safe_r";

// ---------------------------------------------------------------------------
// Poll trend chart
// ---------------------------------------------------------------------------

export interface PollTrendPoll {
  date: string;
  pollster: string | null;
  dem_share: number;
  rep_share: number | null;
  sample_size: number | null;
}

export interface PollTrend {
  dates: string[];
  dem_trend: number[];
  rep_trend: number[];
}

export interface PollTrendResponse {
  race: string;
  slug: string;
  polls: PollTrendPoll[];
  trend: PollTrend | null;
}
