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
