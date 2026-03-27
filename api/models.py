# api/models.py
"""Pydantic response models for the WetherVane API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    db: str
    contract: str = "ok"


# ── Model accuracy (static, changes only on retrain) ────────────────────────

class OverallAccuracy(BaseModel):
    loo_r: float
    holdout_r: float
    coherence: float
    rmse: float
    covariance_val_r: float
    n_counties: int
    n_types: int
    n_super_types: int


class CrossElectionResult(BaseModel):
    cycle: str
    loo_r: float
    label: str


class MethodComparison(BaseModel):
    method: str
    loo_r: float


class AccuracyResponse(BaseModel):
    overall: OverallAccuracy
    cross_election: list[CrossElectionResult]
    method_comparison: list[MethodComparison]


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
    generic_ballot_shift: float | None = None
    """Optional explicit generic ballot shift override (Dem share units, e.g. 0.016).
    When None, the shift is auto-calculated from generic ballot polls in the CSV.
    When 0.0, no generic ballot adjustment is applied."""


class SectionWeights(BaseModel):
    model_prior: float = Field(1.0, ge=0.0, le=2.0)
    state_polls: float = Field(1.0, ge=0.0, le=2.0)
    national_polls: float = Field(1.0, ge=0.0, le=2.0)


class GenericBallotInfo(BaseModel):
    """National environment adjustment derived from generic ballot polling.

    The ``shift`` field is the key value: it represents the difference between
    the current congressional generic ballot average and the 2024 presidential
    national Dem share (0.4841).  A positive shift means Democrats are running
    ahead of their 2024 presidential baseline nationally.

    This shift is applied as a flat additive correction to all county priors
    before race-specific Bayesian updates, anchoring the midterm forecast to
    the current national environment rather than 2024 presidential results.
    """

    gb_avg: float
    """Weighted average of generic ballot polls (Dem two-party share, 0–1)."""

    pres_baseline: float
    """2024 presidential national Dem share used as reference (0.4841)."""

    shift: float
    """gb_avg - pres_baseline.  Positive = Dems ahead of 2024 pres baseline."""

    shift_pp: float
    """shift expressed in percentage points (shift × 100)."""

    n_polls: int
    """Number of generic ballot polls used to compute gb_avg."""

    source: str
    """'auto' (computed from polls CSV) or 'manual' (caller-provided)."""


class MultiPollInput(BaseModel):
    cycle: str              # e.g. "2020", "2022"
    state: str              # e.g. "FL"
    race: str | None = None  # optional filter (e.g. "President", "Senate")
    half_life_days: float = 30.0
    apply_quality: bool = True
    section_weights: SectionWeights = Field(default_factory=SectionWeights)
    generic_ballot_shift: float | None = None
    """Optional explicit generic ballot shift override (Dem share units, e.g. 0.016).
    When None, the shift is auto-calculated from generic ballot polls in the CSV.
    When 0.0, no generic ballot adjustment is applied."""


class MultiPollResponse(BaseModel):
    counties: list[ForecastRow]
    polls_used: int
    date_range: str         # "2020-01-15 to 2020-11-02"
    effective_n_total: int  # sum of adjusted sample sizes
    generic_ballot: GenericBallotInfo | None = None
    """National environment adjustment applied before the Bayesian update.
    None when generic ballot adjustment is disabled (shift=0.0)."""


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


# ── Race detail (SEO page) ───────────────────────────────────────────────

class RacePoll(BaseModel):
    date: str | None
    pollster: str | None
    dem_share: float
    n_sample: int | None


class TypeBreakdown(BaseModel):
    type_id: int
    display_name: str
    n_counties: int
    mean_pred_dem_share: float | None


class RaceDetail(BaseModel):
    race: str
    slug: str
    state_abbr: str
    race_type: str
    year: int
    prediction: float | None
    n_counties: int
    polls: list[RacePoll]
    type_breakdown: list[TypeBreakdown]


# ── Embed widget ─────────────────────────────────────────────────────────

class EmbedResponse(BaseModel):
    """Metadata for the embed widget.  Includes a ready-to-paste iframe snippet."""
    slug: str
    race_title: str          # e.g. "2026 FL Senate"
    lean_label: str          # e.g. "D+4.2" or "R+11.7"
    lean_color: str          # hex color for the lean label
    dem_pct: float | None    # 0–1 two-party Dem share
    rep_pct: float | None    # 0–1 two-party Rep share
    n_counties: int
    iframe_snippet: str      # ready-to-paste HTML


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
