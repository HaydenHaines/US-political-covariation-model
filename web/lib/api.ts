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
