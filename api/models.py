# api/models.py
"""Pydantic response models for the Bedrock API."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: str


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


class CommunityDetail(BaseModel):
    community_id: int
    display_name: str
    n_counties: int
    states: list[str]
    dominant_type_id: int | None
    counties: list[CountyInCommunity]
    shift_profile: dict[str, float]


class CountyRow(BaseModel):
    county_fips: str
    state_abbr: str
    community_id: int


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
    dem_share: float    # 0.0–1.0
    n: int = 600        # poll sample size
