"""Blend endpoints: per-race and overview blend recalculation."""
from __future__ import annotations

import duckdb
import numpy as np
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.models import (
    BlendResult,
    BlendWeights,
    OverviewBlendRaceSummary,
    OverviewBlendResult,
    SectionWeights,
)

from ._helpers import (
    _DEFAULT_DEM_SHARE_PRIOR,
    _SLIDER_NORM,
    _STATE_STD_FALLBACK,
    _VOTE_WEIGHTED_STATE_PRED_SQL,
    _Z90,
    _apply_behavior_if_needed,
    _compute_state_std,
    _get_std_floor,
    slug_to_race,
)

router = APIRouter(tags=["forecast"])


@router.post("/forecast/race/{slug}/blend", response_model=BlendResult)
def recalculate_blend(
    slug: str,
    body: BlendWeights,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> BlendResult:
    """Re-run the forecast for a race with custom section weights.

    The frontend race detail page calls this endpoint whenever the user
    adjusts the model-prior / state-polls / national-polls sliders.  It
    returns only the fields needed to update the hero: prediction,
    pred_std, pred_lo90, pred_hi90.

    Section weights are expressed as 0-100 percentages on the wire (to
    match the slider scale) and normalised to [0, 2] multipliers here
    before being forwarded to the prediction engine.  The default 60/30/10
    split becomes model_prior=1.2, state_polls=0.6, national_polls=0.2
    after normalisation.

    Returns the structural prior unchanged when no polls are available or
    when the type-score data is not loaded into app state.
    """
    version_id = request.app.state.version_id
    race = slug_to_race(slug)

    # Resolve race metadata for state lookup
    try:
        race_meta = db.execute(
            "SELECT race_type, state, year FROM races WHERE race_id = ?",
            [race],
        ).fetchone()
    except duckdb.CatalogException:
        race_meta = None

    if race_meta:
        race_type = race_meta[0]  # e.g. "senate" or "governor" from races table
        state_abbr = race_meta[1]
        year = race_meta[2]
    else:
        parts = slug.split("-")
        race_type = parts[2] if len(parts) >= 3 else ""
        state_abbr = parts[1].upper() if len(parts) >= 2 else "??"
        year = int(parts[0]) if parts[0].isdigit() else 2026

    # Empirical error floor for this race type (calibrated from 2022 backtest)
    std_floor = _get_std_floor(race_type)

    # Fetch state polls for this race
    polls_df = db.execute(
        """
        SELECT dem_share, n_sample, geography, geo_level, race, date, pollster
        FROM polls
        WHERE cycle = ? AND LOWER(race) = LOWER(?) AND geo_level = 'state'
        ORDER BY date
        """,
        [str(year), race],
    ).fetchdf()

    # If no polls, return the structural prior prediction unchanged.
    # Blend controls have no effect when there is nothing to blend.
    if polls_df.empty:
        pred_row = db.execute(
            f"""
            SELECT
                {_VOTE_WEIGHTED_STATE_PRED_SQL} AS state_pred,
                AVG(p.pred_std) AS mean_std
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
            """,
            [version_id, race, state_abbr],
        ).fetchone()
        if pred_row is None or pred_row[0] is None:
            return BlendResult(prediction=None, pred_std=None, pred_lo90=None, pred_hi90=None)
        pred = float(pred_row[0])
        # Use the larger of the generic fallback and the empirical floor so we
        # don't report false precision when no county variance data is available.
        std = float(pred_row[1]) if pred_row[1] else _STATE_STD_FALLBACK
        std = max(std, std_floor)
        return BlendResult(
            prediction=pred,
            pred_std=std,
            pred_lo90=pred - _Z90 * std,
            pred_hi90=pred + _Z90 * std,
        )

    # Normalise 0-100 percentage weights to [0, 2] multipliers.
    sw = SectionWeights(
        model_prior=body.model_prior / _SLIDER_NORM,
        state_polls=body.state_polls / _SLIDER_NORM,
        national_polls=body.national_polls / _SLIDER_NORM,
    )

    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    type_county_fips = getattr(request.app.state, "type_county_fips", None)

    # If type model isn't loaded, fall back to the stored prediction
    if not all(x is not None for x in [type_scores, type_covariance, type_priors, type_county_fips]):
        pred_row = db.execute(
            f"""
            SELECT
                {_VOTE_WEIGHTED_STATE_PRED_SQL} AS state_pred
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
            """,
            [version_id, race, state_abbr],
        ).fetchone()
        if pred_row is None or pred_row[0] is None:
            return BlendResult(prediction=None, pred_std=None, pred_lo90=None, pred_hi90=None)
        pred = float(pred_row[0])
        # Use the larger of fallback and empirical floor for race-type awareness.
        fallback_std = max(_STATE_STD_FALLBACK, std_floor)
        return BlendResult(
            prediction=pred,
            pred_std=fallback_std,
            pred_lo90=pred - _Z90 * fallback_std,
            pred_hi90=pred + _Z90 * fallback_std,
        )

    # Build poll list — use section weight to scale effective N
    race_polls = [
        (
            float(row["dem_share"]),
            max(1, int(row["n_sample"] * sw.state_polls)),
            row["geography"],
        )
        for _, row in polls_df.iterrows()
    ]

    from src.prediction.forecast_runner import predict_race

    tract_states = getattr(request.app.state, "tract_states", {})
    fips_list = type_county_fips
    states_list = [tract_states.get(f, f[:2]) for f in fips_list]
    names_list = [f"Tract {f}" for f in fips_list]

    ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
    county_priors = (
        np.array([ridge_priors.get(f, _DEFAULT_DEM_SHARE_PRIOR) for f in fips_list])
        if ridge_priors
        else None
    )

    county_priors = _apply_behavior_if_needed(request, county_priors, race)

    result_df = predict_race(
        race=race,
        polls=race_polls,
        type_scores=type_scores,
        type_covariance=type_covariance,
        type_priors=type_priors,
        county_fips=fips_list,
        states=states_list,
        county_names=names_list,
        county_priors=county_priors,
        prior_weight=sw.model_prior,
    )

    # Aggregate to state-level vote-weighted prediction
    state_rows = result_df[result_df["state"] == state_abbr]
    if state_rows.empty:
        return BlendResult(prediction=None, pred_std=None, pred_lo90=None, pred_hi90=None)

    tract_votes: dict[str, float] = getattr(request.app.state, "tract_votes", {})
    vote_weights = (
        state_rows["county_fips"].map(lambda f: tract_votes.get(f, 0)).values.astype(float)
    )
    total_w = vote_weights.sum()
    if total_w > 0:
        pred = float((state_rows["pred_dem_share"].values * vote_weights).sum() / total_w)
    else:
        pred = float(state_rows["pred_dem_share"].mean())

    # Derive state-level std from county-level CI data, using the race-type-
    # specific empirical floor from 2022 backtest.
    preds = state_rows["pred_dem_share"].values.astype(float)
    state_std = _compute_state_std(preds, vote_weights, pred, race_type=race_type)

    return BlendResult(
        prediction=pred,
        pred_std=state_std,
        pred_lo90=pred - _Z90 * state_std,
        pred_hi90=pred + _Z90 * state_std,
    )


@router.post("/forecast/overview/blend", response_model=OverviewBlendResult)
def recalculate_overview_blend(
    body: BlendWeights,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> OverviewBlendResult:
    """Recalculate all 33 Class II Senate races with custom blend weights.

    Accepts section weights as 0-100 percentages (model_prior, state_polls,
    national_polls). Iterates over all tracked states, calls the per-race
    recalculate_blend() function for each, and aggregates projected seat totals.

    Returns dem_seats and rep_seats as projected chamber totals (holdover seats
    not up in 2026 + favored contested seats), so the BalanceBar can be
    updated without the frontend knowing those constants.
    """
    # Local import to avoid circular dependency (senate.py never imports forecast.py)
    from api.routers.senate import (
        SENATE_2026_STATES,
        _DEM_HOLDOVER_SEATS,
        _GOP_HOLDOVER_SEATS,
        _CLASS_II_INCUMBENT,
        _DEFAULT_SAFE_MARGIN,
        _margin_to_rating,
        _TOSSUP_MAX,
    )

    race_summaries: list[OverviewBlendRaceSummary] = []
    dem_favored = 0
    gop_favored = 0

    for st in sorted(SENATE_2026_STATES):
        race_str = f"2026 {st} Senate"
        slug = race_str.lower().replace(" ", "-")

        # Reuse per-race blend logic directly
        blend_result = recalculate_blend(slug=slug, body=body, request=request, db=db)

        if blend_result.prediction is not None:
            margin = blend_result.prediction - 0.5
        else:
            # Race has no model data — treat as safe for incumbent party
            incumbent_party = _CLASS_II_INCUMBENT.get(st, "R")
            margin = _DEFAULT_SAFE_MARGIN if incumbent_party == "D" else -_DEFAULT_SAFE_MARGIN

        rating = _margin_to_rating(margin)

        # Seat counts: only count races that aren't tossups
        if margin > _TOSSUP_MAX:
            dem_favored += 1
        elif margin < -_TOSSUP_MAX:
            gop_favored += 1

        race_summaries.append(OverviewBlendRaceSummary(
            slug=slug,
            prediction=blend_result.prediction,
            pred_std=blend_result.pred_std,
            rating_label=rating,
        ))

    return OverviewBlendResult(
        dem_seats=_DEM_HOLDOVER_SEATS + dem_favored,
        rep_seats=_GOP_HOLDOVER_SEATS + gop_favored,
        races=race_summaries,
    )
