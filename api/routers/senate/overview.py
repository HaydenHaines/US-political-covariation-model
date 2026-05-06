"""GET /senate/overview — national Senate forecast summary for the landing page."""
from __future__ import annotations

import logging

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.routers.senate._helpers import (
    DEM_SAFE_SEATS,
    GOP_SAFE_SEATS,
    SENATE_2026_STATES,
    _build_headline,
    _rating_sort_key,
    build_state_colors,
    build_zone_counts,
    classify_race,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["senate"])


def _fetch_predictions(
    db: duckdb.DuckDBPyConnection,
    version_id: str,
    mode_filter: str,
) -> dict[str, tuple[str, float]]:
    """Fetch vote-weighted state prediction per senate race.

    Returns a dict mapping race name -> (state_abbr, state_pred).
    """
    pred_by_race: dict[str, tuple[str, float]] = {}
    for st in sorted(SENATE_2026_STATES):
        race = f"2026 {st} Senate"
        row = db.execute(
            f"""
            SELECT
                CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                          / SUM(COALESCE(c.total_votes_2024, 0))
                     ELSE AVG(p.pred_dem_share)
                END AS state_pred
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ?
              AND p.race = ?
              AND c.state_abbr = ?
              {mode_filter}
            """,
            [version_id, race, st],
        ).fetchone()
        if row and row[0] is not None:
            pred_by_race[race] = (st, float(row[0]))
    return pred_by_race


def _fetch_poll_counts(db: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Fetch poll counts per senate race. Returns empty dict on DB errors."""
    try:
        polls_df = db.execute(
            """
            SELECT race, COUNT(*) AS n_polls
            FROM polls
            WHERE LOWER(race) LIKE '%senate%'
            GROUP BY race
            """
        ).fetchdf()
    except duckdb.Error:
        return {}

    poll_counts: dict[str, int] = {}
    for _, row in polls_df.iterrows():
        poll_counts[str(row["race"])] = int(row["n_polls"])
    return poll_counts


def _fetch_latest_poll_date(db: duckdb.DuckDBPyConnection) -> str | None:
    """Return the most recent poll date scraped, or None."""
    try:
        row = db.execute(
            "SELECT MAX(date) AS max_date FROM polls WHERE date IS NOT NULL"
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except duckdb.Error:
        pass
    return None


@router.get("/senate/overview")
def get_senate_overview(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict:
    """Return the national Senate forecast summary for the landing page.

    Reads vote-weighted state predictions from the predictions table — the
    same source used by the race detail endpoint in forecast.py — so that
    overview and detail pages always agree.
    """
    version_id = getattr(request.app.state, "version_id", None)
    if not version_id:
        return {
            "headline": "No Senate Forecasts Available",
            "subtitle": "predictions not yet loaded",
            "dem_seats_safe": DEM_SAFE_SEATS,
            "gop_seats_safe": GOP_SAFE_SEATS,
            "dem_current": DEM_SAFE_SEATS,
            "gop_current": GOP_SAFE_SEATS,
            "dem_projected": DEM_SAFE_SEATS,
            "gop_projected": GOP_SAFE_SEATS,
            "races": [],
        }

    # Check if forecast_mode column exists (backward compat)
    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    pred_by_race = _fetch_predictions(db, version_id, _mode_filter)
    poll_counts = _fetch_poll_counts(db)

    # Build classified race list with poll counts
    races = []
    for st in sorted(SENATE_2026_STATES):
        race_info = classify_race(st, pred_by_race)
        race_info["n_polls"] = poll_counts.get(race_info["race"], 0)
        races.append(race_info)

    # Sort: tossups first, then lean, likely, safe; break ties alphabetically
    races.sort(key=lambda r: (_rating_sort_key(r["rating"]), r["state"]))

    headline, subtitle, dem_projected, gop_projected = _build_headline(races)

    return {
        "headline": headline,
        "subtitle": subtitle,
        "dem_seats_safe": DEM_SAFE_SEATS,
        "gop_seats_safe": GOP_SAFE_SEATS,
        "dem_current": DEM_SAFE_SEATS,
        "gop_current": GOP_SAFE_SEATS,
        "dem_projected": dem_projected,
        "gop_projected": gop_projected,
        "races": races,
        "zone_counts": build_zone_counts(races),
        "state_colors": build_state_colors(races),
        "updated_at": _fetch_latest_poll_date(db),
    }
