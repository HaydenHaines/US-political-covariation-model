const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

export interface CommunitySummary {
  community_id: number;
  display_name: string;
  n_counties: number;
  states: string[];
  dominant_type_id: number | null;
  mean_pred_dem_share: number | null;
}

export interface CountyRow {
  county_fips: string;
  state_abbr: string;
  community_id: number;
  dominant_type: number | null;
  super_type: number | null;
  pred_dem_share: number | null;
}

export interface ForecastRow {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
  race: string;
  pred_dem_share: number | null;
  pred_std: number | null;
  pred_lo90: number | null;
  pred_hi90: number | null;
  state_pred: number | null;
  poll_avg: number | null;
}

export interface CommunityDemographics {
  pop_total: number | null;
  pct_white_nh: number | null;
  pct_black: number | null;
  pct_asian: number | null;
  pct_hispanic: number | null;
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
}

export interface CommunityDetail {
  community_id: number;
  display_name: string;
  n_counties: number;
  states: string[];
  dominant_type_id: number | null;
  counties: Array<{
    county_fips: string;
    county_name: string | null;
    state_abbr: string;
    pred_dem_share: number | null;
  }>;
  shift_profile: Record<string, number>;
  demographics: CommunityDemographics | null;
}

export async function fetchCommunities(): Promise<CommunitySummary[]> {
  const res = await fetch(`${API_BASE}/communities`);
  if (!res.ok) throw new Error(`/communities failed: ${res.status}`);
  return res.json();
}

export async function fetchCommunityDetail(id: number): Promise<CommunityDetail> {
  const res = await fetch(`${API_BASE}/communities/${id}`);
  if (!res.ok) throw new Error(`/communities/${id} failed: ${res.status}`);
  return res.json();
}

export async function fetchCounties(): Promise<CountyRow[]> {
  const res = await fetch(`${API_BASE}/counties`);
  if (!res.ok) throw new Error(`/counties failed: ${res.status}`);
  return res.json();
}

export async function fetchCountyDetail(fips: string): Promise<import("@/lib/types").CountyDetail> {
  const res = await fetch(`${API_BASE}/counties/${fips}`);
  if (!res.ok) throw new Error(`/counties/${fips} failed: ${res.status}`);
  return res.json();
}

export async function fetchForecast(race?: string, state?: string): Promise<ForecastRow[]> {
  const params = new URLSearchParams();
  if (race) params.set("race", race);
  if (state) params.set("state", state);
  const res = await fetch(`${API_BASE}/forecast?${params}`);
  if (!res.ok) throw new Error(`/forecast failed: ${res.status}`);
  return res.json();
}

export interface TypeSummary {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  // Key demographics for tooltip display (pre-fetched alongside type names)
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

export interface TypeDetail extends TypeSummary {
  counties: TypeCounty[];
  demographics: Record<string, number>;
  shift_profile: Record<string, number> | null;
  narrative: string | null;
}

export interface SuperTypeSummary {
  super_type_id: number;
  display_name: string;
  member_type_ids: number[];
  n_counties: number;
}

export async function fetchTypes(): Promise<TypeSummary[]> {
  const res = await fetch(`${API_BASE}/types`);
  if (!res.ok) throw new Error(`/types failed: ${res.status}`);
  return res.json();
}

export async function fetchTypeDetail(id: number): Promise<TypeDetail> {
  const res = await fetch(`${API_BASE}/types/${id}`);
  if (!res.ok) throw new Error(`/types/${id} failed: ${res.status}`);
  return res.json();
}

export async function fetchSuperTypes(): Promise<SuperTypeSummary[]> {
  const res = await fetch(`${API_BASE}/super-types`);
  if (!res.ok) throw new Error(`/super-types failed: ${res.status}`);
  return res.json();
}

export async function feedPoll(body: {
  state: string; race: string; dem_share: number; n: number;
}): Promise<ForecastRow[]> {
  const res = await fetch(`${API_BASE}/forecast/poll`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`/forecast/poll failed: ${res.status}`);
  return res.json();
}

export interface MultiPollResponse {
  counties: ForecastRow[];
  polls_used: number;
  date_range: string;
  effective_n_total: number;
}

export interface TypeScatterPoint {
  type_id: number;
  super_type_id: number;
  display_name: string;
  n_counties: number;
  demographics: Record<string, number>;
  shift_profile: Record<string, number>;
}

export async function fetchTypeScatterData(): Promise<TypeScatterPoint[]> {
  const res = await fetch(`${API_BASE}/types/scatter-data`);
  if (!res.ok) throw new Error(`/types/scatter-data failed: ${res.status}`);
  return res.json();
}

export async function feedMultiplePolls(body: {
  cycle: string;
  state: string;
  race?: string;
  half_life_days?: number;
  apply_quality?: boolean;
  section_weights?: {
    model_prior: number;
    state_polls: number;
    national_polls: number;
  };
}): Promise<MultiPollResponse> {
  const res = await fetch(`${API_BASE}/forecast/polls`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`/forecast/polls failed: ${res.status}`);
  return res.json();
}

export async function fetchForecastRaces(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/forecast/races`);
  if (!res.ok) throw new Error(`/forecast/races failed: ${res.status}`);
  return res.json();
}

export interface PollRow {
  race: string;
  geography: string;
  geo_level: string;
  dem_share: number;
  n_sample: number;
  date: string | null;
  pollster: string | null;
}

export interface SenateRaceData {
  state: string;
  race: string;
  slug: string;
  rating: string;
  margin: number;
  n_polls: number;
  zone?: string;
}

export interface SenateOverviewData {
  headline: string;
  subtitle: string;
  /** Safe seats not up in 2026 (baseline, not displayed directly in the hero). */
  dem_seats_safe: number;
  gop_seats_safe: number;
  /**
   * Projected seat totals: safe seats + contested seats the model clearly favors.
   * Tossups are excluded from both sides (standard forecasting convention).
   * These are the numbers displayed in the hero section and balance bar.
   */
  dem_projected: number;
  gop_projected: number;
  races: SenateRaceData[];
  /** Map from state abbreviation to hex color for the mini map. */
  state_colors?: Record<string, string>;
  /** Seat counts per narrative zone (not_up_d, safe_up_d, contested_d, tossup, contested_r, safe_up_r, not_up_r). */
  zone_counts?: Record<string, number>;
  /** ISO date string of the most recently scraped poll, if available. */
  updated_at?: string | null;
}

export async function fetchSenateOverview(): Promise<SenateOverviewData> {
  const res = await fetch(`${API_BASE}/senate/overview`);
  if (!res.ok) throw new Error(`/senate/overview failed: ${res.status}`);
  return res.json();
}

export interface GovernorRaceData {
  state: string;
  race: string;
  slug: string;
  rating: string;
  /** Signed Dem margin: pred_dem_share - 0.5. Positive = Dem-favored. */
  margin: number;
  /** Which party currently holds the governorship ("D" or "R"). */
  incumbent_party: string;
  n_polls: number;
  /** Whether this is an open seat (no incumbent running). */
  is_open_seat: boolean;
}

export interface GovernorOverviewData {
  races: GovernorRaceData[];
  /** ISO date string of the most recently scraped governor poll, if available. */
  updated_at: string | null;
}

export async function fetchGovernorOverview(): Promise<GovernorOverviewData> {
  const res = await fetch(`${API_BASE}/governor/overview`);
  if (!res.ok) throw new Error(`/governor/overview failed: ${res.status}`);
  return res.json();
}

export interface StructuralContext {
  baseline_year: number;
  baseline_label: string;
  baseline_dem_two_party: number;
  dem_wins_at_baseline: number;
  dem_holdover_seats: number;
  total_dem_projected: number;
  seats_needed_for_majority: number;
  structural_gap: number;
}

export interface SenateScrollyContextData {
  zone_counts: Record<string, number>;
  not_up_d_states: string[];
  not_up_r_states: string[];
  structural_context: StructuralContext;
  competitive_races: SenateRaceData[];
}

export async function fetchSenateScrollyContext(): Promise<SenateScrollyContextData> {
  const res = await fetch(`${API_BASE}/senate/scrolly-context`);
  if (!res.ok) throw new Error(`/senate/scrolly-context failed: ${res.status}`);
  return res.json();
}

export interface CorrelatedTypeData {
  type_id: number;
  display_name: string;
  super_type_id: number;
  n_counties: number;
  mean_pred_dem_share: number | null;
  /** Ledoit-Wolf regularized electoral correlation coefficient [-1, 1] */
  correlation: number;
}

export async function fetchCorrelatedTypes(typeId: number, n = 4): Promise<CorrelatedTypeData[]> {
  const res = await fetch(`${API_BASE}/types/${typeId}/correlated?n=${n}`);
  if (!res.ok) throw new Error(`/types/${typeId}/correlated failed: ${res.status}`);
  return res.json();
}

export async function fetchPolls(params: {
  race?: string;
  state?: string;
  cycle?: string;
}): Promise<PollRow[]> {
  const p = new URLSearchParams();
  if (params.race) p.set("race", params.race);
  if (params.state) p.set("state", params.state);
  if (params.cycle) p.set("cycle", params.cycle);
  const res = await fetch(`${API_BASE}/polls?${p}`);
  if (!res.ok) throw new Error(`/polls failed: ${res.status}`);
  return res.json();
}

// ── Forecast changelog ───────────────────────────────────────────────────

export interface ChangelogRaceDiff {
  race: string;
  before: number | null;
  after: number | null;
  delta: number | null;
}

export interface ChangelogEntry {
  date: string;
  note: string | null;
  diffs: ChangelogRaceDiff[];
}

export interface ChangelogResponse {
  entries: ChangelogEntry[];
  current_snapshot_date: string | null;
}

export async function fetchChangelog(): Promise<ChangelogResponse> {
  const res = await fetch(`${API_BASE}/forecast/changelog`);
  if (!res.ok) throw new Error(`/forecast/changelog failed: ${res.status}`);
  return res.json();
}

export async function fetchPollTrend(slug: string): Promise<import("@/lib/types").PollTrendResponse> {
  const res = await fetch(`${API_BASE}/forecast/race/${slug}/poll-trend`);
  if (!res.ok) throw new Error(`/forecast/race/${slug}/poll-trend failed: ${res.status}`);
  return res.json();
}

// ── Chamber probability ──────────────────────────────────────────────────

// ── Seat balance timeline ────────────────────────────────────────────────

/** One entry in the Senate seat balance time series. */
export interface SeatHistoryEntry {
  /** ISO date string, e.g. "2026-04-01". */
  date: string;
  /** Projected total Dem seats (holdover + contested wins). */
  dem_projected: number;
  /** Projected total GOP seats (holdover + contested wins). */
  gop_projected: number;
}

export async function fetchSeatHistory(): Promise<SeatHistoryEntry[]> {
  const res = await fetch(`${API_BASE}/forecast/seat-history`);
  if (!res.ok) throw new Error(`/forecast/seat-history failed: ${res.status}`);
  return res.json();
}

export interface SeatDistributionBucket {
  seats: number;
  probability: number;
}

export interface ChamberProbabilityData {
  /** Fraction of simulations where Dems win >=50 seats (0-100 scale). */
  dem_control_pct: number;
  /** Fraction of simulations where GOP wins (0-100 scale). */
  rep_control_pct: number;
  /** Fraction of simulations where Dems win >=51 seats (outright majority, 0-100 scale). */
  dem_majority_pct: number;
  median_dem_seats: number;
  median_rep_seats: number;
  seat_distribution: SeatDistributionBucket[];
  n_simulations: number;
  n_modeled_races: number;
  n_safe_races: number;
}

export async function fetchChamberProbability(): Promise<ChamberProbabilityData> {
  const res = await fetch(`${API_BASE}/senate/chamber-probability`);
  if (!res.ok) throw new Error(`/senate/chamber-probability failed: ${res.status}`);
  return res.json();
}

// ── Fundamentals model ───────────────────────────────────────────────────

export interface FundamentalsSnapshot {
  cycle: number;
  in_party: string;
  approval_net_oct: number | null;
  gdp_q2_growth_pct: number | null;
  unemployment_oct: number | null;
  cpi_yoy_oct: number | null;
  consumer_sentiment: number | null;
}

/** State-level economic signal from QCEW data. */
export interface StateEconEntry {
  state_fips: string;
  state_abbr: string | null;
  /** Employment growth relative to national (pp). */
  emp_growth_rel_pp: number;
  /** Wage growth relative to national (pp). */
  wage_growth_rel_pp: number;
  /** Manufacturing employment share (percent). */
  mfg_emp_share_pct: number;
  /** Resulting shift adjustment from state econ (pp). */
  shift_adjustment_pp: number;
}

export interface FundamentalsData {
  /** Dem two-party share shift (0-1 scale, positive = Dem gain). */
  shift: number;
  /** Same shift expressed in percentage points. */
  shift_pp: number;
  /** Contribution from presidential approval (pp). */
  approval_contribution_pp: number;
  /** Contribution from GDP growth (pp). */
  gdp_contribution_pp: number;
  /** Contribution from unemployment (pp). */
  unemployment_contribution_pp: number;
  /** Contribution from CPI inflation (pp). */
  cpi_contribution_pp: number;
  /** Leave-one-out RMSE across training cycles (pp). Indicates model uncertainty. */
  loo_rmse_pp: number;
  /** Number of historical midterm cycles used for training. */
  n_training: number;
  /** Blend weight assigned to the fundamentals component (0-1). */
  weight: number;
  /**
   * Combined generic ballot shift (pp) — fundamentals blended with generic
   * ballot polls.  This is the headline number for display.
   */
  combined_shift_pp: number;
  /** Current-cycle economic snapshot values. */
  snapshot: FundamentalsSnapshot;
  /** Whether state-level economic adjustment is active. */
  state_econ_enabled: boolean;
  /** Sensitivity parameter for state economic modulation. */
  state_econ_sensitivity: number;
  /** Per-state economic signals and adjustments (empty if disabled). */
  state_econ: StateEconEntry[];
}

export async function fetchFundamentals(): Promise<FundamentalsData> {
  const res = await fetch(`${API_BASE}/forecast/fundamentals`);
  if (!res.ok) throw new Error(`/forecast/fundamentals failed: ${res.status}`);
  return res.json();
}

// ── Race margin history (sparklines) ─────────────────────────────────────

/** One data point in a race's margin time series. */
export interface RaceMarginPoint {
  /** ISO date string, e.g. "2026-04-01". */
  date: string;
  /**
   * Signed Dem margin: avg_dem_share - 0.5.
   * Positive = Dem-favored, negative = GOP-favored.
   */
  margin: number;
}

/** Per-race margin history entry, keyed by the race's URL slug. */
export interface RaceHistoryEntry {
  /** Race slug, e.g. "2026-fl-senate". */
  slug: string;
  /** Chronologically ordered margin series (oldest first). */
  history: RaceMarginPoint[];
}

export async function fetchRaceHistory(): Promise<RaceHistoryEntry[]> {
  const res = await fetch(`${API_BASE}/forecast/race-history`);
  if (!res.ok) throw new Error(`/forecast/race-history failed: ${res.status}`);
  return res.json();
}

// ── Historical presidential election results ─────────────────────────────────

export interface HistoricalCountyRow {
  county_fips: string;
  dem_share: number;
  total_votes: number | null;
}

export interface HistoricalElectionResponse {
  year: number;
  counties: HistoricalCountyRow[];
}

/**
 * Fetch county-level Dem two-party share for a past presidential election.
 * Available years: 2012, 2016, 2020.
 *
 * Returns a Map<county_fips, dem_share> for O(1) lookup during map rendering.
 */
export async function fetchHistoricalElection(year: number): Promise<Map<string, number>> {
  const res = await fetch(`${API_BASE}/historical/presidential/${year}`);
  if (!res.ok) throw new Error(`/historical/presidential/${year} failed: ${res.status}`);
  const data: HistoricalElectionResponse = await res.json();
  const map = new Map<string, number>();
  for (const row of data.counties) {
    map.set(row.county_fips, row.dem_share);
  }
  return map;
}

// ── Sabermetrics: candidate badges & CTOV ─���───────────────────────────────

export interface CandidateBadge {
  name: string;
  score: number;
  provisional?: boolean;
  kind?: "catalog" | "signature";
  type_id?: number | null;
  fallback_reason?: string | null;
}

export interface RaceCandidateSummary {
  bioguide_id: string;
  name: string;
  party: string;
  badges: CandidateBadge[];
  cec: number;
  badge_scores: Record<string, number>;
}

export interface RaceCandidatesResponse {
  race_key: string;
  candidates: RaceCandidateSummary[];
}

export async function fetchRaceCandidates(raceKey: string): Promise<RaceCandidatesResponse> {
  const res = await fetch(`${API_BASE}/races/${encodeURIComponent(raceKey)}/candidates`);
  if (!res.ok) throw new Error(`/races/${raceKey}/candidates failed: ${res.status}`);
  return res.json();
}
