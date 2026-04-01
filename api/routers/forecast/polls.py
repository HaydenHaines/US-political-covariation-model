"""Poll endpoints: poll listing, single-poll update, multi-poll update."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import get_db
from api.models import (
    ForecastRow,
    MultiPollInput,
    MultiPollResponse,
    PollInput,
    PollRow,
)

from ._helpers import (
    _DEFAULT_DEM_SHARE_PRIOR,
    _MATRIX_JITTER,
    _Z90,
    _apply_behavior_if_needed,
    _lookup_pollster_grade,
)

router = APIRouter(tags=["forecast"])


@router.get("/polls", response_model=list[PollRow])
def get_polls(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    race: str | None = Query(None, description="Filter by race label (e.g. '2026 FL Senate')"),
    state: str | None = Query(None, description="Filter by state abbreviation (e.g. 'FL')"),
    cycle: str = Query("2026", description="Poll cycle year"),
):
    """Return available polls from DuckDB polls table."""
    conditions = ["cycle = ?"]
    params: list = [cycle]
    if race:
        conditions.append("LOWER(race) LIKE ?")
        params.append(f"%{race.lower()}%")
    if state:
        conditions.append("geography = ?")
        params.append(state)
    where = " AND ".join(conditions)
    rows = db.execute(f"SELECT * FROM polls WHERE {where} ORDER BY date", params).fetchdf()

    return [
        PollRow(
            race=row["race"],
            geography=row["geography"],
            geo_level=row["geo_level"],
            dem_share=float(row["dem_share"]),
            n_sample=int(row["n_sample"]),
            date=row["date"] if row["date"] else None,
            pollster=row["pollster"] if row["pollster"] else None,
            grade=_lookup_pollster_grade(
                request, row["pollster"] if row["pollster"] else None
            ),
        )
        for _, row in rows.iterrows()
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
    """Type-based Bayesian update using predict_race from forecast_runner."""
    from src.prediction.forecast_runner import predict_race

    # Tract-level lookups from app.state (no DuckDB query needed)
    tract_states = getattr(request.app.state, "tract_states", {})
    states = [tract_states.get(f, f[:2]) for f in type_county_fips]
    county_names = [f"Tract {f}" for f in type_county_fips]

    # Build county-level priors from Ridge predictions (if loaded at startup)
    ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
    if ridge_priors:
        county_priors = np.array(
            [ridge_priors.get(f, _DEFAULT_DEM_SHARE_PRIOR) for f in type_county_fips]
        )
    else:
        county_priors = None  # falls back to type-mean inside predict_race

    county_priors = _apply_behavior_if_needed(request, county_priors, poll.race)

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

    # Vote-weighted state-level prediction: SUM(pred * votes) / SUM(votes).
    state_rows = result_df[result_df["state"] == poll.state]
    state_pred_val = None
    if not state_rows.empty:
        tract_votes = getattr(request.app.state, "tract_votes", {})
        weights = state_rows["county_fips"].map(lambda f: tract_votes.get(f, 0)).values.astype(float)
        total_w = weights.sum()
        if total_w > 0:
            state_pred_val = float(
                (state_rows["pred_dem_share"].values * weights).sum() / total_w
            )
        else:
            state_pred_val = float(state_rows["pred_dem_share"].mean())

    results = []
    for _, row in result_df.iterrows():
        results.append(
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=poll.race,
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * _Z90),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=poll.dem_share,
                dominant_type=int(row["dominant_type"]) if "dominant_type" in row.index else None,
            )
        )
    return results


def _run_bayesian_update(
    mu_prior: np.ndarray,
    sigma: np.ndarray,
    w_matrix: np.ndarray,
    poll_means: np.ndarray,
    poll_stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Gaussian Bayesian update for type/community means given polls.

    This is the shared math kernel used by both the type-based and HAC community-based
    forecast paths.  The update follows standard Gaussian conditioning:

        posterior precision = prior precision + W^T R^{-1} W
        posterior mean      = Sigma_post (Sigma_prior^{-1} mu_prior + W^T R^{-1} y)

    where R = diag(sigma_poll^2) is the measurement noise covariance.

    Args:
        mu_prior:   Prior mean vector, shape (K,).
        sigma:      Prior covariance matrix, shape (K, K).
        w_matrix:   Weight matrix, shape (n_polls, K).  Each row is the geographic
                    type/community composition of one polled area (W vector).
        poll_means: Observed Democratic share for each poll, shape (n_polls,).
        poll_stds:  Measurement uncertainty (std) for each poll, shape (n_polls,).
                    Computed from binomial variance: sqrt(p*(1-p)/n).

    Returns:
        Tuple of (mu_post, sigma_post):
            mu_post    — posterior mean, shape (K,).
            sigma_post — posterior covariance, shape (K, K).  Callers that need
                         per-community standard deviations can use sqrt(diag(sigma_post)).
    """
    K = len(mu_prior)
    R = np.diag(poll_stds ** 2)
    # Tikhonov regularization keeps the prior covariance positive definite under
    # floating-point noise.  _MATRIX_JITTER is tiny relative to true eigenvalues.
    sigma_inv = np.linalg.inv(sigma + np.eye(K) * _MATRIX_JITTER)
    sigma_post_inv = sigma_inv + w_matrix.T @ np.linalg.inv(R) @ w_matrix
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (sigma_inv @ mu_prior + w_matrix.T @ np.linalg.solve(R, poll_means))
    return mu_post, sigma_post


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
    poll_means = np.array([poll.dem_share])
    poll_stds = np.array([np.sqrt(poll.dem_share * (1 - poll.dem_share) / poll.n)])

    mu_post, sigma_post = _run_bayesian_update(mu_prior, sigma, W, poll_means, poll_stds)

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
                pred_lo90=pred - _Z90 * std,
                pred_hi90=pred + _Z90 * std,
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
    # Load polls from DuckDB instead of CSV
    rows_df = db.execute(
        "SELECT * FROM polls WHERE cycle=? ORDER BY date",
        [body.cycle],
    ).fetchdf()

    if body.state:
        rows_df = rows_df[rows_df["geography"] == body.state]
    if body.race:
        rows_df = rows_df[rows_df["race"].str.contains(body.race, case=False, na=False)]

    if rows_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No matching polls for cycle={body.cycle}, state={body.state}, race={body.race}",
        )

    from src.propagation.poll_weighting import (
        aggregate_polls,
        apply_all_weights,
        election_day_for_cycle,
    )
    from src.propagation.propagate_polls import PollObservation
    polls = [
        PollObservation(
            geography=row["geography"],
            dem_share=float(row["dem_share"]),
            n_sample=int(row["n_sample"]),
            race=row["race"],
            date=row["date"] or "",
            pollster=row["pollster"] or "",
            geo_level=row["geo_level"],
        )
        for _, row in rows_df.iterrows()
    ]
    notes = list(rows_df["notes"].fillna(""))

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
    # Apply section weight to poll effective N (Option A from spec).
    sw = body.section_weights
    race_polls = [
        (p.dem_share, max(1, int(p.n_sample * sw.state_polls)), p.geography)
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
        from src.prediction.forecast_runner import predict_race

        # Tract-level lookups from app.state (no DuckDB query needed)
        tract_states = getattr(request.app.state, "tract_states", {})
        fips_list = type_county_fips
        states_list = [tract_states.get(f, f[:2]) for f in fips_list]
        names_list = [f"Tract {f}" for f in fips_list]

        ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
        county_priors = (
            np.array([ridge_priors.get(f, _DEFAULT_DEM_SHARE_PRIOR) for f in fips_list])
            if ridge_priors else None
        )

        county_priors = _apply_behavior_if_needed(request, county_priors, body.race)

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
            prior_weight=sw.model_prior,
        )

        state_pred_val = None
        if body.state:
            state_rows = result_df[result_df["state"] == body.state]
            if not state_rows.empty:
                # Vote-weighted average: SUM(pred * votes) / SUM(votes).
                tract_votes = getattr(request.app.state, "tract_votes", {})
                weights = state_rows["county_fips"].map(lambda f: tract_votes.get(f, 0)).values.astype(float)
                total_w = weights.sum()
                if total_w > 0:
                    state_pred_val = float(
                        (state_rows["pred_dem_share"].values * weights).sum() / total_w
                    )
                else:
                    state_pred_val = float(state_rows["pred_dem_share"].mean())

        county_results = [
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=body.race or race_polls[0][2],
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * _Z90),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=float(np.mean([p[0] for p in race_polls])),
                dominant_type=int(row["dominant_type"]) if "dominant_type" in row.index else None,
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
