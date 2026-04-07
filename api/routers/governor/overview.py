"""GET /governor/overview — national Governor forecast summary page."""
from __future__ import annotations

import logging

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.routers.governor._helpers import (
    GOVERNOR_2026_STATES,
    classify_governor_race,
    rating_sort_key,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["governor"])


def _fetch_governor_predictions(
    db: duckdb.DuckDBPyConnection,
    version_id: str,
    mode_filter: str,
) -> dict[str, tuple[str, float]]:
    """Fetch vote-weighted state prediction per governor race.

    Returns a dict mapping race name -> (state_abbr, state_pred).
    Vote-weighting uses total_votes_2024 from the counties table so that
    dense urban counties don't get the same weight as sparse rural counties.
    """
    pred_by_race: dict[str, tuple[str, float]] = {}
    for st in sorted(GOVERNOR_2026_STATES):
        race = f"2026 {st} Governor"
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


def _fetch_governor_poll_counts(db: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Fetch poll counts per governor race. Returns empty dict on DB errors."""
    try:
        polls_df = db.execute(
            """
            SELECT race, COUNT(*) AS n_polls
            FROM polls
            WHERE LOWER(race) LIKE '%governor%'
            GROUP BY race
            """
        ).fetchdf()
    except duckdb.Error:
        return {}

    return {str(row["race"]): int(row["n_polls"]) for _, row in polls_df.iterrows()}


def _fetch_latest_poll_date(db: duckdb.DuckDBPyConnection) -> str | None:
    """Return the most recent governor poll date scraped, or None."""
    try:
        row = db.execute(
            """
            SELECT MAX(date) AS max_date
            FROM polls
            WHERE date IS NOT NULL AND LOWER(race) LIKE '%governor%'
            """
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except duckdb.Error:
        pass
    return None


@router.get("/governor/overview")
def get_governor_overview(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict:
    """Return the national Governor forecast summary.

    Fetches vote-weighted state predictions from the predictions table —
    the same source used by race detail pages — so overview and detail
    always agree.  Governor races are independent executives: there is no
    chamber control concept, so no balance bar or seat totals are returned.

    Response shape::

        {
          "races": [
            {
              "state": "OH",
              "race": "2026 OH Governor",
              "slug": "2026-oh-governor",
              "rating": "lean_r",
              "margin": -0.054,
              "incumbent_party": "R",
              "n_polls": 16
            },
            ...
          ],
          "updated_at": "2026-04-01"
        }
    """
    version_id = getattr(request.app.state, "version_id", None)
    if not version_id:
        # No model loaded — return structural fallbacks using incumbent parties
        races = [classify_governor_race(st) for st in sorted(GOVERNOR_2026_STATES)]
        races.sort(key=lambda r: (rating_sort_key(r["rating"]), r["state"]))
        return {"races": races, "updated_at": None}

    # Check if forecast_mode column exists (backward compatibility with older DB versions)
    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    pred_by_race = _fetch_governor_predictions(db, version_id, _mode_filter)
    poll_counts = _fetch_governor_poll_counts(db)

    races = []
    for st in sorted(GOVERNOR_2026_STATES):
        race_info = classify_governor_race(st, pred_by_race)
        race_info["n_polls"] = poll_counts.get(race_info["race"], 0)
        races.append(race_info)

    # Sort: tossups and lean races first (most competitive), safe seats last.
    # Break ties alphabetically by state.
    races.sort(key=lambda r: (rating_sort_key(r["rating"]), r["state"]))

    return {
        "races": races,
        "updated_at": _fetch_latest_poll_date(db),
    }
