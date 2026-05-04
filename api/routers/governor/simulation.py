"""GET /governor/simulation — Monte Carlo governor seat simulation."""
from __future__ import annotations

import logging

import duckdb
import numpy as np
from fastapi import APIRouter, Depends, Query, Request

from api.db import get_db
from api.models import GovernorSimulationBucket, GovernorSimulationResponse
from api.routers.governor._helpers import (
    GOVERNOR_2026_STATES,
    _GOVERNOR_INCUMBENT,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["governor"])

_STATE_STD_FLOOR = 0.035
_STATE_STD_CAP = 0.15
_STATE_STD_FALLBACK = 0.065
_SAFE_SEAT_STD = 0.05
_N_SIMULATIONS = 10_000

_N_GOVERNOR_RACES = len(GOVERNOR_2026_STATES)  # 36


def _compute_state_prediction(county_rows) -> tuple[float, float]:
    preds_arr = county_rows["pred_dem_share"].values.astype(float)
    votes_arr = county_rows["votes"].values.astype(float)
    total_votes = votes_arr.sum()

    if total_votes > 0:
        state_pred = float(np.dot(preds_arr, votes_arr) / total_votes)
        weights = votes_arr / total_votes
        county_var = float(np.sum(weights * (preds_arr - state_pred) ** 2))
        n_eff = max(1.0, 1.0 / np.sum(weights ** 2))
        raw_std = float(np.sqrt(county_var / n_eff))
    else:
        state_pred = float(np.mean(preds_arr))
        raw_std = _STATE_STD_FALLBACK

    std = max(raw_std, _STATE_STD_FLOOR)
    std = min(std, _STATE_STD_CAP)
    return state_pred, std


def _collect_race_data(
    db: duckdb.DuckDBPyConnection,
    version_id: str,
    mode_filter: str,
) -> tuple[list[tuple[float, float]], int, int]:
    """Collect per-state prediction data for the Monte Carlo simulation.

    Returns (modeled_races, safe_dem_wins, safe_gop_wins).
    """
    modeled_races: list[tuple[float, float]] = []
    safe_dem_wins = 0
    safe_gop_wins = 0

    for st in sorted(GOVERNOR_2026_STATES):
        race = f"2026 {st} Governor"

        county_rows = db.execute(
            f"""
            SELECT
                p.pred_dem_share,
                p.pred_std,
                COALESCE(c.total_votes_2024, 0) AS votes
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ?
              AND p.race = ?
              AND c.state_abbr = ?
              AND p.pred_dem_share IS NOT NULL
              {mode_filter}
            """,
            [version_id, race, st],
        ).fetchdf()

        if county_rows.empty:
            incumbent = _GOVERNOR_INCUMBENT.get(st, "R")
            if incumbent == "D":
                safe_dem_wins += 1
            else:
                safe_gop_wins += 1
            continue

        state_pred, std = _compute_state_prediction(county_rows)
        modeled_races.append((state_pred, std))

    return modeled_races, safe_dem_wins, safe_gop_wins


def _simulate_governor_seats(
    modeled_races: list[tuple[float, float]],
    safe_dem_wins: int,
    safe_gop_wins: int,
    n_sims: int,
    rng_seed: int | None = 42,
) -> GovernorSimulationResponse:
    """Run Monte Carlo simulation for all 36 governor races.

    Each race is drawn from Normal(pred, std), clipped to [0, 1].
    Dem wins where draw > 0.5.  Safe Dem seats use pred=0.75, safe GOP pred=0.25.
    Returns the distribution over (d_seats, r_seats) pairs.
    """
    rng = np.random.default_rng(rng_seed)

    if len(modeled_races) > 0:
        preds = np.array([r[0] for r in modeled_races])
        stds = np.array([r[1] for r in modeled_races])
    else:
        preds = np.empty(0)
        stds = np.empty(0)

    safe_dem_preds = np.full(safe_dem_wins, 0.75)
    safe_dem_stds = np.full(safe_dem_wins, _SAFE_SEAT_STD)
    safe_gop_preds = np.full(safe_gop_wins, 0.25)
    safe_gop_stds = np.full(safe_gop_wins, _SAFE_SEAT_STD)

    all_preds = np.concatenate([preds, safe_dem_preds, safe_gop_preds])
    all_stds = np.concatenate([stds, safe_dem_stds, safe_gop_stds])

    draws = rng.normal(loc=all_preds, scale=all_stds, size=(n_sims, len(all_preds)))
    draws = np.clip(draws, 0.0, 1.0)

    dem_wins_per_sim = np.sum(draws > 0.5, axis=1)

    buckets = []
    for d in range(_N_GOVERNOR_RACES + 1):
        prob = float(np.mean(dem_wins_per_sim == d))
        if prob > 0.0001:
            buckets.append(
                GovernorSimulationBucket(
                    d_seats=d,
                    r_seats=_N_GOVERNOR_RACES - d,
                    probability=round(prob, 4),
                )
            )

    return GovernorSimulationResponse(buckets=buckets)


@router.get("/governor/simulation", response_model=GovernorSimulationResponse)
def get_governor_simulation(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    n_simulations: int = Query(
        _N_SIMULATIONS,
        ge=1000,
        le=100_000,
        description="Number of Monte Carlo simulations (default 10,000)",
    ),
    date: str | None = Query(
        None,
        description="Forecast date filter (reserved for future use)",
    ),
) -> GovernorSimulationResponse:
    """Monte Carlo simulation of 2026 governor seat totals.

    Runs N independent simulations across all 36 governor races.  For each
    race with model predictions, draws from Normal(pred, std).  Unmodeled
    races use the incumbent party as a strong favorite (pred=0.75/0.25,
    std=0.05).

    Returns the distribution over (d_seats, r_seats) pairs where d+r=36.
    """
    version_id = getattr(request.app.state, "version_id", None)
    if not version_id:
        safe_dem = sum(1 for p in _GOVERNOR_INCUMBENT.values() if p == "D")
        safe_gop = sum(1 for p in _GOVERNOR_INCUMBENT.values() if p == "R")
        return _simulate_governor_seats([], safe_dem, safe_gop, n_simulations)

    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    modeled_races, safe_dem_wins, safe_gop_wins = _collect_race_data(
        db, version_id, _mode_filter,
    )

    return _simulate_governor_seats(
        modeled_races=modeled_races,
        safe_dem_wins=safe_dem_wins,
        safe_gop_wins=safe_gop_wins,
        n_sims=n_simulations,
    )
