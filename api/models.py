# api/models.py
"""Pydantic response models for the WetherVane API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    db: str
    contract: str = "ok"


class ModelVersionResponse(BaseModel):
    version_id: str
    k: int | None
    j: int | None
    holdout_r: str | None  # VARCHAR in DB (may be range like "0.93–0.98")
    shift_type: str | None
    created_at: str | None


class CommunitySummary(BaseModel):
    community_id: int
    display_name: str
    n_counties: int
    states: list[str]
    dominant_type_id: int | None
    mean_pred_dem_share: float | None


class CountyInCommunity(BaseModel):
    county_fips: str
    county_name: str | None
    state_abbr: str
    pred_dem_share: float | None


class CommunityDemographics(BaseModel):
    """Population-weighted demographic profile for a community."""
    pop_total: float | None = None
    pct_white_nh: float | None = None
    pct_black: float | None = None
    pct_asian: float | None = None
    pct_hispanic: float | None = None
    median_age: float | None = None
    median_hh_income: float | None = None
    pct_bachelors_plus: float | None = None
    pct_owner_occupied: float | None = None
    pct_wfh: float | None = None
    pct_management: float | None = None
    evangelical_share: float | None = None
    mainline_share: float | None = None
    catholic_share: float | None = None
    black_protestant_share: float | None = None
    congregations_per_1000: float | None = None
    religious_adherence_rate: float | None = None


class CommunityDetail(BaseModel):
    community_id: int
    display_name: str
    n_counties: int
    states: list[str]
    dominant_type_id: int | None
    counties: list[CountyInCommunity]
    shift_profile: dict[str, float]
    demographics: CommunityDemographics | None = None


class CountyRow(BaseModel):
    county_fips: str
    state_abbr: str
    community_id: int | None = None
    dominant_type: int | None = None
    super_type: int | None = None
    pred_dem_share: float | None = None


class ForecastRow(BaseModel):
    county_fips: str
    county_name: str | None
    state_abbr: str
    race: str
    pred_dem_share: float | None
    pred_std: float | None
    pred_lo90: float | None
    pred_hi90: float | None
    state_pred: float | None
    poll_avg: float | None


class PollInput(BaseModel):
    state: str          # e.g. "FL"
    race: str           # e.g. "FL_Senate"
    dem_share: float = Field(..., ge=0.0, le=1.0)
    n: int = 600        # poll sample size


class MultiPollInput(BaseModel):
    cycle: str              # e.g. "2020", "2022"
    state: str              # e.g. "FL"
    race: str | None = None  # optional filter (e.g. "President", "Senate")
    half_life_days: float = 30.0
    apply_quality: bool = True


class MultiPollResponse(BaseModel):
    counties: list[ForecastRow]
    polls_used: int
    date_range: str         # "2020-01-15 to 2020-11-02"
    effective_n_total: int  # sum of adjusted sample sizes


class PollRow(BaseModel):
    race: str
    geography: str          # state abbreviation for state-level polls (e.g. "FL")
    geo_level: str          # "state" | "county" | "district"
    dem_share: float
    n_sample: int
    date: str | None
    pollster: str | None


# ── Type-primary models ─────────────────────────────────────────────────────

class TypeSummary(BaseModel):
    type_id: int
    super_type_id: int
    display_name: str
    n_counties: int
    mean_pred_dem_share: float | None = None
    # Key demographics for tooltip display (pre-fetched, no per-hover API calls)
    median_hh_income: float | None = None
    pct_bachelors_plus: float | None = None
    pct_white_nh: float | None = None
    log_pop_density: float | None = None


class TypeCounty(BaseModel):
    county_fips: str
    county_name: str | None
    state_abbr: str


class TypeDetail(TypeSummary):
    counties: list[TypeCounty]  # resolved county names
    demographics: dict[str, float]  # demographic profile
    shift_profile: dict[str, float] | None = None
    narrative: str | None = None  # auto-generated description


class SuperTypeSummary(BaseModel):
    super_type_id: int
    display_name: str
    member_type_ids: list[int]
    n_counties: int


class TypeScatterPoint(BaseModel):
    """One data point per type for the Shift Explorer scatter plot."""
    type_id: int
    super_type_id: int
    display_name: str
    n_counties: int
    demographics: dict[str, float]
    shift_profile: dict[str, float]


# ── County detail (SEO page) ──────────────────────────────────────────────

class SiblingCounty(BaseModel):
    county_fips: str
    county_name: str | None
    state_abbr: str


class CountyDetail(BaseModel):
    county_fips: str
    county_name: str | None
    state_abbr: str
    dominant_type: int
    super_type: int
    type_display_name: str
    super_type_display_name: str
    narrative: str | None = None
    pred_dem_share: float | None = None
    demographics: dict[str, float]
    sibling_counties: list[SiblingCounty]
