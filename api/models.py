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
    dominant_type: int | None = None


class PollInput(BaseModel):
    state: str  # e.g. "FL"
    race: str  # e.g. "FL_Senate"
    dem_share: float = Field(..., ge=0.0, le=1.0)
    n: int = 600  # poll sample size
    generic_ballot_shift: float | None = None
    """Optional explicit generic ballot shift override (Dem share units, e.g. 0.016).
    When None, the shift is auto-calculated from generic ballot polls in the CSV.
    When 0.0, no generic ballot adjustment is applied."""


class SectionWeights(BaseModel):
    model_prior: float = Field(1.0, ge=0.0, le=2.0)
    state_polls: float = Field(1.0, ge=0.0, le=2.0)
    national_polls: float = Field(1.0, ge=0.0, le=2.0)


class BlendWeights(BaseModel):
    """Section weights expressed as 0–100 percentages (must sum to 100).

    The race detail page uses this scale so sliders map directly to
    percentages.  The backend normalises to [0, 1] multipliers before
    passing into the prediction engine.
    """

    model_prior: float = Field(60.0, ge=0.0, le=100.0)
    state_polls: float = Field(30.0, ge=0.0, le=100.0)
    national_polls: float = Field(10.0, ge=0.0, le=100.0)


class BlendResult(BaseModel):
    """Slim forecast result returned by the blend recalculation endpoint.

    Contains only the values displayed in the hero and dotplot so the
    response payload is small and the client can update incrementally
    without re-fetching the full race detail.
    """

    prediction: float | None
    pred_std: float | None
    pred_lo90: float | None
    pred_hi90: float | None


class OverviewBlendRaceSummary(BaseModel):
    """Per-race result from the overview blend recalculation endpoint."""

    slug: str
    prediction: float | None
    pred_std: float | None
    rating_label: str


class OverviewBlendResult(BaseModel):
    """Full response from POST /forecast/overview/blend.

    dem_seats / rep_seats are projected totals including safe non-competitive
    seats (DEM_SAFE_SEATS=47, GOP_SAFE_SEATS=53) so the BalanceBar can be
    updated without the frontend knowing those constants.
    """

    dem_seats: int
    rep_seats: int
    races: list[OverviewBlendRaceSummary]


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
    """Number of RCP CSV-sourced generic ballot polls used to compute gb_avg."""

    n_yougov_polls: int = 0
    """Number of YouGov weekly crosstab issues used (after deduplication with CSV)."""

    source: str
    """'auto' (computed from polls CSV + YouGov) or 'manual' (caller-provided)."""

    baseline_year: int = 2024
    """Election year used as the structural prior baseline (always 2024 for now)."""

    baseline_label: str = ""
    """Human-readable label for the presidential baseline, e.g. 'R+3.2' or 'D+0.5'.
    Derived from pres_baseline: shift = pres_baseline - 0.5; negative → 'R+X', positive → 'D+X'.
    """


class StateEconEntry(BaseModel):
    """Economic signal for a single state from QCEW data."""

    state_fips: str
    """2-digit state FIPS code."""

    state_abbr: str | None = None
    """State abbreviation (e.g. 'CA'), resolved from FIPS."""

    emp_growth_rel_pp: float
    """Employment growth relative to national (percentage points)."""

    wage_growth_rel_pp: float
    """Average wage growth relative to national (percentage points)."""

    mfg_emp_share_pct: float
    """Manufacturing employment share (percent)."""

    shift_adjustment_pp: float
    """Resulting shift adjustment from state econ (percentage points)."""


class FundamentalsResponse(BaseModel):
    """Structural fundamentals model output.

    The fundamentals model predicts a national Dem share shift based on
    presidential approval, GDP growth, unemployment, and CPI inflation.
    It is blended with the generic ballot shift at a configurable weight.
    Optionally includes state-level economic adjustments from QCEW data.
    """

    shift: float
    """Fundamentals-predicted Dem share shift (fraction, e.g. 0.011 = +1.1pp)."""

    shift_pp: float
    """shift expressed in percentage points."""

    approval_contribution_pp: float
    """Approval rating component (pp)."""

    gdp_contribution_pp: float
    """GDP growth component (pp)."""

    unemployment_contribution_pp: float
    """Unemployment rate component (pp)."""

    cpi_contribution_pp: float
    """CPI inflation component (pp)."""

    loo_rmse_pp: float
    """Leave-one-out RMSE (pp) — the uncertainty estimate."""

    n_training: int
    """Number of historical midterm cycles used for calibration."""

    weight: float
    """Blending weight (0–1) used to combine with generic ballot."""

    combined_shift_pp: float
    """The final blended shift: weight * fundamentals + (1-weight) * generic_ballot (pp)."""

    snapshot: dict
    """Current-cycle input values used to compute the shift."""

    state_econ_enabled: bool = False
    """Whether state-level economic adjustment is active."""

    state_econ_sensitivity: float = 0.0
    """Sensitivity parameter for state economic modulation."""

    state_econ: list[StateEconEntry] = []
    """Per-state economic signals and adjustments (empty if disabled)."""


class MultiPollInput(BaseModel):
    cycle: str  # e.g. "2020", "2022"
    state: str  # e.g. "FL"
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
    date_range: str  # "2020-01-15 to 2020-11-02"
    effective_n_total: int  # sum of adjusted sample sizes
    generic_ballot: GenericBallotInfo | None = None
    """National environment adjustment applied before the Bayesian update.
    None when generic ballot adjustment is disabled (shift=0.0)."""


class PollRow(BaseModel):
    race: str
    geography: str  # state abbreviation for state-level polls (e.g. "FL")
    geo_level: str  # "state" | "county" | "district"
    dem_share: float
    n_sample: int
    date: str | None
    pollster: str | None
    grade: str | None = None  # Silver Bulletin letter grade (e.g. "A+", "B-")


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


class CorrelatedType(BaseModel):
    """A type most electorally correlated with a given type, from the LW covariance matrix."""

    type_id: int
    display_name: str
    super_type_id: int
    n_counties: int
    mean_pred_dem_share: float | None
    correlation: float


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
    grade: str | None = None  # Silver Bulletin letter grade (e.g. "A+", "B-")


class TypeBreakdown(BaseModel):
    type_id: int
    display_name: str
    n_counties: int
    mean_pred_dem_share: float | None
    # Total 2024 votes across counties of this type in the state.
    # Used to sort types by electoral weight rather than raw county count —
    # without this, states like MI show only rural types because they have
    # more small counties than urban ones (see GitHub #21).
    total_votes: int | None = None


# ── Historical context (static per-race, from api/data/historical_results.json) ──


class LastRaceResult(BaseModel):
    """Most recent election result for this specific Senate seat or Governor office."""

    year: int
    winner: str
    party: str
    # Positive = Dem advantage (D+X), negative = Rep advantage (R+X), in percentage points.
    margin: float
    note: str | None = None


class PresidentialResult(BaseModel):
    """2024 presidential two-party result for this state."""

    winner: str
    party: str
    # Positive = Dem advantage, negative = Rep advantage, in percentage points.
    margin: float
    note: str | None = None


class HistoricalContext(BaseModel):
    """Historical electoral context for a single tracked race."""

    last_race: LastRaceResult
    presidential_2024: PresidentialResult
    # Model forecast minus last_race margin (pp). Positive = Dem shift vs. last result.
    forecast_shift: float | None = None


class PollConfidence(BaseModel):
    """Diversity and coverage metrics for polls available for a race.

    Confidence label derivation:
    - "High":   3+ distinct pollsters AND 2+ distinct methodologies
    - "Medium": 2+ distinct pollsters OR 2+ distinct methodologies (not both for High)
    - "Low":    fewer than 2 pollsters or no polls at all

    Methodology is inferred from the notes field using the LV/RV convention
    used by all three poll sources (Wikipedia, 270toWin, RealClearPolling).
    Polls with neither LV nor RV in their notes are labeled "Unknown".
    """

    n_polls: int
    """Total number of polls for this race."""
    n_pollsters: int
    """Number of distinct pollster names."""
    n_methodologies: int
    """Number of distinct methodologies detected (LV, RV, Unknown)."""
    label: str
    """'High', 'Medium', or 'Low' based on diversity thresholds."""
    tooltip: str
    """Human-readable summary, e.g. '3 pollsters · 2 methods · 8 polls'."""


class CandidateIncumbent(BaseModel):
    """Current office holder for a race."""

    name: str
    party: str


class CandidateInfo(BaseModel):
    """Candidate field data for a race, sourced from candidates_2026.json.

    status values: 'incumbent_running', 'open', 'special'
    """

    incumbent: CandidateIncumbent
    status: str
    status_detail: str | None = None
    rating: str | None = None
    candidates: dict[str, list[str]]
    """Candidates by party (e.g. {"R": ["Name"], "D": ["Name"]})."""


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
    forecast_mode: str = "local"
    state_pred_national: float | None = None
    state_pred_local: float | None = None
    candidate_effect_margin: float | None = None
    n_polls: int = 0
    pred_std: float | None = None
    pred_lo90: float | None = None
    pred_hi90: float | None = None
    historical_context: HistoricalContext | None = None
    poll_confidence: PollConfidence | None = None
    candidate_info: CandidateInfo | None = None


# ── Embed widget ─────────────────────────────────────────────────────────


class EmbedResponse(BaseModel):
    """Metadata for the embed widget.  Includes a ready-to-paste iframe snippet."""

    slug: str
    race_title: str  # e.g. "2026 FL Senate"
    lean_label: str  # e.g. "D+4.2" or "R+11.7"
    lean_color: str  # hex color for the lean label
    dem_pct: float | None  # 0–1 two-party Dem share
    rep_pct: float | None  # 0–1 two-party Rep share
    n_counties: int
    iframe_snippet: str  # ready-to-paste HTML


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


class ElectionHistoryPoint(BaseModel):
    year: int
    election_type: str  # "president" | "senate" | "governor"
    dem_share: float
    total_votes: int | None = None


# ── Chamber probability ──────────────────────────────────────────────────


class SeatDistributionBucket(BaseModel):
    """Probability mass for a specific Dem seat total from Monte Carlo simulation."""

    seats: int
    probability: float


class ChamberProbabilityResponse(BaseModel):
    """Monte Carlo chamber control probability for the 2026 Senate.

    Derived by simulating N independent race outcomes drawn from each race's
    predicted Dem share and uncertainty (Normal(pred, std)).  Safe/unmodeled
    races use a fixed high-confidence draw for the incumbent party.
    """

    dem_control_pct: float
    """Fraction of simulations where Democrats win >=50 seats (0-100 scale)."""

    rep_control_pct: float
    """Fraction of simulations where Republicans win (0-100 scale)."""

    dem_majority_pct: float
    """Fraction of simulations where Democrats win >=51 seats (outright majority, 0-100 scale)."""

    median_dem_seats: int
    """Median Democratic seat total across all simulations."""

    median_rep_seats: int
    """Median Republican seat total across all simulations."""

    seat_distribution: list[SeatDistributionBucket]
    """Probability mass function over possible Dem seat totals."""

    n_simulations: int
    """Number of Monte Carlo draws used."""

    n_modeled_races: int
    """Number of contested races with model predictions (Normal draws)."""

    n_safe_races: int
    """Number of races treated as safe (high-confidence incumbent win)."""


# ── Forecast changelog ───────────────────────────────────────────────────


class ChangelogRaceDiff(BaseModel):
    race: str
    before: float | None
    after: float | None
    delta: float | None


class ChangelogEntry(BaseModel):
    date: str
    note: str | None = None
    diffs: list[ChangelogRaceDiff]


class ChangelogResponse(BaseModel):
    entries: list[ChangelogEntry]
    current_snapshot_date: str | None = None


# ── Poll trend chart ─────────────────────────────────────────────────────


class PollTrendPoll(BaseModel):
    """A single poll data point for the trend chart."""

    date: str
    pollster: str | None
    dem_share: float
    rep_share: float | None
    sample_size: int | None


class PollTrend(BaseModel):
    """Smoothed trend line — dates and weighted moving-average shares.

    The CI fields hold 95% confidence bounds derived from the 2022 backtest RMSE
    by race type (Senate: ±3.7pp, Governor: ±5.5pp).  They represent the
    empirical uncertainty around the structural model, not the poll noise itself.
    """

    dates: list[str]
    dem_trend: list[float]
    rep_trend: list[float]
    # 95% CI = point estimate ± 2 * backtest_rmse, clamped to [0, 1].
    # Parallel arrays matching ``dates`` length.
    dem_ci_lower: list[float]
    dem_ci_upper: list[float]
    rep_ci_lower: list[float]
    rep_ci_upper: list[float]


class PollTrendResponse(BaseModel):
    race: str
    slug: str
    polls: list[PollTrendPoll]
    trend: PollTrend | None


# ── Pollster accuracy ────────────────────────────────────────────────────────


class PollsterAccuracyEntry(BaseModel):
    """Accuracy metrics for a single pollster, computed against 2022 backtest actuals."""

    pollster: str
    """Canonical pollster name."""

    rank: int
    """Rank by RMSE, 1 = most accurate."""

    n_polls: int
    """Number of polls included in this analysis."""

    n_races: int
    """Number of distinct races covered by this pollster."""

    rmse_pp: float
    """Root mean squared error in percentage points (lower = more accurate)."""

    mean_error_pp: float
    """Mean signed error in percentage points.
    Positive = pollster systematically over-predicted Democratic share.
    Negative = pollster systematically over-predicted Republican share.
    """


class PollsterAccuracyResponse(BaseModel):
    """Ranked list of pollster accuracy metrics from the 2022 backtest."""

    description: str
    """Human-readable description of methodology."""

    n_pollsters: int
    """Total number of pollsters in the analysis."""

    pollsters: list[PollsterAccuracyEntry]
    """Pollsters sorted by rank (ascending RMSE -- best first)."""


# ── Sabermetrics: candidate badges & CTOV ───────────────────────────────────


class CandidateBadge(BaseModel):
    """A single performance badge earned by a candidate."""

    name: str
    """Badge display name, e.g. 'Hispanic Appeal' or 'Low Rural Populist'."""

    score: float
    """Signed magnitude of the badge dimension (positive = strength, negative = weakness)."""

    provisional: bool = False
    """True when the candidate ran in only 1 race — consistency could not be measured."""

    kind: str = "catalog"
    """Badge source:
    - 'catalog': preset demographic badge from the hardcoded BADGE_CATALOG
    - 'signature': auto-discovered community-type fingerprint (specific type ID)
    - 'discovered': PCA-derived axis from data-driven badge discovery
    """

    type_id: int | None = None
    """Set for signature badges — the community type ID driving the signal."""

    fallback_reason: str | None = None
    """'small_pool' if the party pool was too small for within-party thresholding."""

    pc_index: int | None = None
    """Set for discovered badges — the zero-based PCA component index."""

    top_demographics: list[tuple[str, float]] | None = None
    """Set for discovered badges — (demographic_column, correlation) pairs
    that drove the axis name.  Ordered by absolute correlation, descending."""

    explained_variance_ratio: float | None = None
    """Set for discovered badges — fraction of total CTOV variance this axis explains."""


class RaceResult(BaseModel):
    """A single race entry from the candidate registry.

    Each entry corresponds to one election cycle the candidate ran in.
    ``actual_dem_share_2party`` is the two-party Democratic vote share (0–1).
    ``result`` is 'win' or 'loss' from the candidate's perspective.
    """

    year: int
    state: str
    office: str
    special: bool = False
    party: str
    result: str
    actual_dem_share_2party: float | None = None


class CampaignStats(BaseModel):
    """FEC-derived campaign finance statistics for the most recent available cycle.

    SDR (Small-Dollar Ratio): fraction of individual contributions under $200.
    FER (Fundraising Efficiency Ratio): total_receipts / total_disbursements; >1 means
      more was raised than spent, <1 means the campaign ran a deficit.
    burn_rate: total_disbursements / total_receipts — the complement of FER.
      >1 means spending outpaced fundraising.
    """

    sdr: float | None
    """Small-Dollar Ratio: % of individual donations under $200 (0–1 fraction)."""

    fer: float | None
    """Fundraising Efficiency Ratio: receipts / disbursements."""

    burn_rate: float | None
    """Burn rate: disbursements / receipts (the campaign spending tempo)."""

    cycle: int | None
    """Election cycle year this data is drawn from (e.g. 2022 or 2024)."""

    has_fec_record: bool
    """False when the candidate has no FEC filing data in the pipeline."""


class LegislativeStats(BaseModel):
    """DW-NOMINATE ideology scores and Legislative Effectiveness scores.

    DW-NOMINATE dim1 runs from −1 (most liberal) to +1 (most conservative).
    LES is the Volden-Wiseman Legislative Effectiveness Score (unnormalized,
    median ≈ 1.0 for the full Congress).
    """

    nominate_dim1: float | None
    """Career-average DW-NOMINATE first dimension (ideology, −1 to +1)."""

    les_score: float | None
    """Career-average Legislative Effectiveness Score."""

    les2_score: float | None
    """Career-average LES2 (benchmark-adjusted effectiveness, Volden–Wiseman)."""

    congresses_served: int | None
    """Number of Congresses the member has served in total."""

    has_legislative_record: bool
    """False when the candidate has no VOTEVIEW or LES data in the pipeline."""


class CandidateBadgesResponse(BaseModel):
    """Full badge profile for a single candidate (by bioguide ID)."""

    bioguide_id: str
    name: str
    party: str
    n_races: int
    badges: list[CandidateBadge]
    badge_scores: dict[str, float]
    """All badge dimension scores (not just earned badges)."""
    cec: float
    """Candidate Effect Consistency — how stable the candidate's effect is across races (0–1)."""
    races: list[RaceResult] = []
    """All race entries from the candidate registry, sorted by year descending."""
    campaign_stats: CampaignStats | None = None
    """FEC campaign finance stats for the most recent cycle. None if data not loaded."""
    legislative_stats: LegislativeStats | None = None
    """DW-NOMINATE and LES legislative profile. None if data not loaded."""


class CTOVEntry(BaseModel):
    """One community type's CTOV value for a candidate."""

    type_id: int
    display_name: str
    ctov: float
    """Over/underperformance in this community type (positive = Dem overperformance)."""


class CTOVResponse(BaseModel):
    """Top CTOV (Candidate Type-Over Vote) entries for a candidate-race."""

    bioguide_id: str
    name: str
    party: str
    year: int
    state: str
    office: str
    entries: list[CTOVEntry]
    """Top 10 types by absolute CTOV value."""


class RaceCandidateSummary(BaseModel):
    """Badge summary for one candidate in a race."""

    bioguide_id: str
    name: str
    party: str
    badges: list[CandidateBadge]
    cec: float
    badge_scores: dict[str, float]


class RaceCandidatesResponse(BaseModel):
    """All candidates with badges for a given race."""

    race_key: str
    candidates: list[RaceCandidateSummary]


class CandidateListItem(BaseModel):
    """Summary row for the candidates directory listing.

    Includes enough info to render a candidate card with name, party,
    office/state context, badge count, and CEC.  Full badge data is
    only fetched when the user navigates to the profile page.
    """

    bioguide_id: str
    name: str
    party: str
    n_races: int
    cec: float
    badges: list[str]
    """Badge names only — no scores — for compact card display."""
    states: list[str]
    """Unique states the candidate has run in, derived from registry races."""
    offices: list[str]
    """Unique office types ('Senate', 'Governor'), derived from registry races."""
    years: list[int]
    """Election years the candidate has appeared in, ascending order."""


class CandidateListResponse(BaseModel):
    """Paginated list of candidates matching query parameters."""

    candidates: list[CandidateListItem]
    total: int
    """Total matching count before pagination."""


class PredecessorInfo(BaseModel):
    """The most recent predecessor in the same state/office/party slot.

    Used for single-race candidates who lack a cross-cycle consistency signal.
    This is a low-trust comparison — labeled clearly in the UI.
    """

    bioguide_id: str
    name: str
    year: int


class FitScoreEntry(BaseModel):
    """One candidate's fit score for a target district."""

    rank: int
    bioguide_id: str
    name: str
    party: str
    n_races: int
    fit_score: float
    # Type IDs (0-indexed) that contribute most to the fit score.
    top_type_ids: list[int]
    # Human-readable display names for the top contributing types.
    top_type_names: list[str]


class FitScoreResponse(BaseModel):
    """Ranked candidate-district fit scores for a target race."""

    race_key: str
    state: str
    party_filter: str | None
    candidates: list[FitScoreEntry]
