from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import get_db
from api.models import ForecastRow, MultiPollInput, MultiPollResponse, PollInput, PollRow

router = APIRouter(tags=["forecast"])


@router.get("/forecast", response_model=list[ForecastRow])
def get_forecast(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    race: str | None = Query(None, description="Filter by race (e.g. FL_Senate)"),
    state: str | None = Query(None, description="Filter by state abbreviation (e.g. FL)"),
):
    version_id = request.app.state.version_id

    conditions = ["p.version_id = ?"]
    params: list = [version_id]

    if race:
        conditions.append("p.race = ?")
        params.append(race)
    if state:
        conditions.append("c.state_abbr = ?")
        params.append(state)

    where = " AND ".join(conditions)

    rows = db.execute(
        f"""
        SELECT
            p.county_fips,
            c.county_name,
            c.state_abbr,
            p.race,
            p.pred_dem_share,
            p.pred_std,
            p.pred_lo90,
            p.pred_hi90,
            p.state_pred,
            p.poll_avg
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE {where}
        ORDER BY p.race, c.state_abbr, p.county_fips
        """,
        params,
    ).fetchdf()

    if rows.empty:
        return []

    return [
        ForecastRow(
            county_fips=row["county_fips"],
            county_name=row["county_name"] if row["county_name"] else None,
            state_abbr=row["state_abbr"],
            race=row["race"],
            pred_dem_share=None if pd.isna(row["pred_dem_share"]) else float(row["pred_dem_share"]),
            pred_std=None if pd.isna(row["pred_std"]) else float(row["pred_std"]),
            pred_lo90=None if pd.isna(row["pred_lo90"]) else float(row["pred_lo90"]),
            pred_hi90=None if pd.isna(row["pred_hi90"]) else float(row["pred_hi90"]),
            state_pred=None if pd.isna(row["state_pred"]) else float(row["state_pred"]),
            poll_avg=None if pd.isna(row["poll_avg"]) else float(row["poll_avg"]),
        )
        for _, row in rows.iterrows()
    ]


@router.get("/forecast/races", response_model=list[str])
def get_forecast_races(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Return distinct race labels available in the predictions table, sorted."""
    version_id = request.app.state.version_id
    rows = db.execute(
        "SELECT DISTINCT race FROM predictions WHERE version_id = ? ORDER BY race",
        [version_id],
    ).fetchall()
    return [r[0] for r in rows]


@router.get("/polls", response_model=list[PollRow])
def get_polls(
    race: str | None = Query(None, description="Filter by race label (e.g. '2026 FL Senate')"),
    state: str | None = Query(None, description="Filter by state abbreviation (e.g. 'FL')"),
    cycle: str = Query("2026", description="Poll cycle year"),
):
    """Return available polls from the poll CSV for a given race/state."""
    from src.propagation.poll_weighting import load_polls_with_notes

    try:
        polls, _ = load_polls_with_notes(cycle, race=race, geography=state)
    except FileNotFoundError:
        return []

    return [
        PollRow(
            race=p.race,
            geography=p.geography,
            geo_level=p.geo_level,
            dem_share=p.dem_share,
            n_sample=p.n_sample,
            date=p.date or None,
            pollster=p.pollster or None,
        )
        for p in polls
    ]


@router.post("/forecast/poll", response_model=list[ForecastRow])
def update_forecast_with_poll(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Run a Bayesian update: given a new poll, return updated county predictions.

    Uses type-based pipeline if type data is loaded, otherwise falls back to HAC.
    """
    # ── Try type-based prediction first ──────────────────────────────────
    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    type_county_fips = getattr(request.app.state, "type_county_fips", None)

    if (
        type_scores is not None
        and type_covariance is not None
        and type_priors is not None
        and type_county_fips is not None
    ):
        return _forecast_poll_types(
            poll, request, db, type_scores, type_covariance, type_priors, type_county_fips
        )

    # ── Fallback: HAC community-based pipeline ──────────────────────────
    return _forecast_poll_hac(poll, request, db)


def _forecast_poll_types(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection,
    type_scores: np.ndarray,
    type_covariance: np.ndarray,
    type_priors: np.ndarray,
    type_county_fips: list[str],
) -> list[ForecastRow]:
    """Type-based Bayesian update using predict_race from predict_2026_types."""
    from src.prediction.predict_2026_types import predict_race

    # Get county names from DB
    county_info = db.execute(
        """SELECT c.county_fips, c.county_name, c.state_abbr
           FROM counties c ORDER BY c.county_fips""",
        [],
    ).fetchdf()
    name_map = dict(zip(county_info["county_fips"], county_info["county_name"]))
    state_map = dict(zip(county_info["county_fips"], county_info["state_abbr"]))

    states = [state_map.get(f, f[:2]) for f in type_county_fips]
    county_names = [name_map.get(f, "") for f in type_county_fips]

    # Build county-level priors from Ridge predictions (if loaded at startup)
    ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
    if ridge_priors:
        county_priors = np.array(
            [ridge_priors.get(f, 0.45) for f in type_county_fips]
        )
    else:
        county_priors = None  # falls back to type-mean inside predict_race

    # poll.state is the authoritative source for which state was polled;
    # passing it explicitly avoids the fragile race-string scan.
    result_df = predict_race(
        race=poll.race,
        polls=[(poll.dem_share, poll.n, poll.state)],
        type_scores=type_scores,
        type_covariance=type_covariance,
        type_priors=type_priors,
        county_fips=type_county_fips,
        states=states,
        county_names=county_names,
        state_filter=None,
        county_priors=county_priors,
    )

    # Compute state-level prediction as the mean of polled-state county
    # predictions — consistent with how the HAC pipeline populates state_pred.
    state_rows = result_df[result_df["state"] == poll.state]
    state_pred_val = float(state_rows["pred_dem_share"].mean()) if not state_rows.empty else None

    results = []
    for _, row in result_df.iterrows():
        results.append(
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=poll.race,
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * 1.645),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=poll.dem_share,
            )
        )
    return results


def _forecast_poll_hac(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection,
) -> list[ForecastRow]:
    """Original HAC community-based Bayesian update."""
    sigma = request.app.state.sigma
    K = request.app.state.K
    mu_prior = request.app.state.mu_prior
    state_weights = request.app.state.state_weights
    county_weights = request.app.state.county_weights

    if state_weights.empty or county_weights.empty:
        raise HTTPException(status_code=503, detail="Weight matrices not loaded")

    # Build W row for this state
    weight_cols = sorted([c for c in state_weights.columns if c.startswith("community_")])
    state_row = state_weights[state_weights["state_abbr"] == poll.state]
    if state_row.empty:
        raise HTTPException(status_code=404, detail=f"State {poll.state} not found in weights")

    W = state_row[weight_cols].values  # shape (1, K)
    y = np.array([poll.dem_share])
    sigma_poll = np.array([np.sqrt(poll.dem_share * (1 - poll.dem_share) / poll.n)])

    # Bayesian update
    R = np.diag(sigma_poll ** 2)
    sigma_inv = np.linalg.inv(sigma + np.eye(K) * 1e-8)
    sigma_post_inv = sigma_inv + W.T @ np.linalg.inv(R) @ W
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (sigma_inv @ mu_prior + W.T @ np.linalg.solve(R, y))

    # Map community posteriors → county predictions via hard assignment
    version_id = request.app.state.version_id
    county_info = db.execute(
        """SELECT c.county_fips, c.county_name, c.state_abbr, ca.community_id
           FROM counties c
           JOIN community_assignments ca ON c.county_fips = ca.county_fips AND ca.version_id = ?
           ORDER BY c.county_fips""",
        [version_id],
    ).fetchdf()

    state_pred_val = float((W @ mu_post).item())
    community_stds = np.sqrt(np.diag(sigma_post))

    results = []
    for _, row in county_info.iterrows():
        cid = int(row["community_id"])
        if 0 <= cid < K:
            pred = float(mu_post[cid])
            std = float(community_stds[cid])
        else:
            pred = float(np.mean(mu_post))
            std = 0.05
        results.append(
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state_abbr"],
                race=poll.race,
                pred_dem_share=pred,
                pred_std=std,
                pred_lo90=pred - 1.645 * std,
                pred_hi90=pred + 1.645 * std,
                state_pred=state_pred_val,
                poll_avg=poll.dem_share,
            )
        )

    return results


@router.post("/forecast/polls", response_model=MultiPollResponse)
def update_forecast_with_multi_polls(
    body: MultiPollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Feed multiple polls from a historical cycle through time-decayed,
    quality-weighted Bayesian update. Returns county-level predictions.
    """
    from src.propagation.poll_weighting import (
        aggregate_polls,
        apply_all_weights,
        election_day_for_cycle,
        load_polls_with_notes,
    )

    # Load polls + notes from CSV
    try:
        polls, notes = load_polls_with_notes(
            body.cycle, race=body.race, geography=body.state
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No poll data found for cycle {body.cycle}",
        )

    if not polls:
        raise HTTPException(
            status_code=404,
            detail=f"No matching polls for cycle={body.cycle}, state={body.state}, race={body.race}",
        )

    # Apply weighting
    ref_date = election_day_for_cycle(body.cycle)
    weighted = apply_all_weights(
        polls,
        reference_date=ref_date,
        half_life_days=body.half_life_days,
        poll_notes=notes if body.apply_quality else None,
        apply_quality=body.apply_quality,
    )

    # Build metadata
    dates = sorted(p.date for p in polls if p.date)
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "unknown"
    effective_n_total = sum(p.n_sample for p in weighted)

    # Build poll list for stacked Bayesian update — do not collapse.
    # Collapsing loses geographic information when polls cover different
    # states and conflates structurally distinct signals into one scalar.
    race_polls = [
        (p.dem_share, p.n_sample, p.geography)
        for p in weighted
        if p.geo_level == "state"
    ]

    if not race_polls:
        raise HTTPException(status_code=400, detail="No state-level polls after filtering")

    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    type_county_fips = getattr(request.app.state, "type_county_fips", None)

    if all(x is not None for x in [type_scores, type_covariance, type_priors, type_county_fips]):
        from src.prediction.predict_2026_types import predict_race

        county_info = db.execute(
            "SELECT county_fips, county_name, state_abbr FROM counties ORDER BY county_fips"
        ).fetchdf()
        name_map = dict(zip(county_info["county_fips"], county_info["county_name"]))
        state_map = dict(zip(county_info["county_fips"], county_info["state_abbr"]))

        fips_list = type_county_fips
        states_list = [state_map.get(f, f[:2]) for f in fips_list]
        names_list = [name_map.get(f, "") for f in fips_list]

        ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
        county_priors = (
            np.array([ridge_priors.get(f, 0.45) for f in fips_list])
            if ridge_priors else None
        )

        result_df = predict_race(
            race=body.race or race_polls[0][2],
            polls=race_polls,
            type_scores=type_scores,
            type_covariance=type_covariance,
            type_priors=type_priors,
            county_fips=fips_list,
            states=states_list,
            county_names=names_list,
            county_priors=county_priors,
        )

        state_pred_val = None
        if body.state:
            state_rows = result_df[result_df["state"] == body.state]
            if not state_rows.empty:
                state_pred_val = float(state_rows["pred_dem_share"].mean())

        county_results = [
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=body.race or race_polls[0][2],
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * 1.645),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=float(np.mean([p[0] for p in race_polls])),
            )
            for _, row in result_df.iterrows()
        ]
    else:
        # HAC fallback: multi-poll stacking not yet supported; collapse to one
        # effective poll as an approximation until HAC gains stacking support.
        combined_share, combined_n = aggregate_polls(weighted)
        race_label = body.race or polls[0].race
        single_poll = PollInput(
            state=body.state, race=race_label, dem_share=combined_share, n=combined_n
        )
        county_results = _forecast_poll_hac(single_poll, request, db)

    return MultiPollResponse(
        counties=county_results,
        polls_used=len(polls),
        date_range=date_range,
        effective_n_total=effective_n_total,
    )
