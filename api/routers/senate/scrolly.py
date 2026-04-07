"""GET /senate/scrolly-context — narrative context for the scrollytelling homepage."""
from __future__ import annotations

import logging

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.routers.senate._helpers import (
    _DEM_HOLDOVER_SEATS,
    SENATE_2026_STATES,
    SENATE_DELEGATION,
    _compute_baseline_label,
    build_zone_counts,
    classify_race,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["senate"])


def _fetch_scrolly_predictions(
    db: duckdb.DuckDBPyConnection,
    version_id: str | None,
    mode_filter: str,
) -> dict[str, tuple[str, float]]:
    """Fetch vote-weighted predictions for scrolly context.

    Returns dict mapping race name -> (state_abbr, state_pred).
    Returns empty dict if version_id is None.
    """
    if not version_id:
        return {}

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


def _fetch_scrolly_poll_counts(db: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Fetch poll counts for scrolly competitive races."""
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
    for _, poll_row in polls_df.iterrows():
        poll_counts[str(poll_row["race"])] = int(poll_row["n_polls"])
    return poll_counts


def _build_structural_context(races: list[dict]) -> dict:
    """Compute structural context for the scrollytelling narrative.

    Counts how many Class II races Democrats currently win at the model's
    predicted margins, and computes the gap to majority.
    """

    dem_wins_at_baseline = sum(1 for r in races if r["margin"] > 0)
    total_dem_projected = _DEM_HOLDOVER_SEATS + dem_wins_at_baseline
    seats_needed_for_majority = 51
    structural_gap = seats_needed_for_majority - total_dem_projected

    # The structural argument uses 2020 as the reference environment: Class II
    # senators were last on the ballot in 2020 (presidential year). Democrats
    # won approximately 52.4% of the aggregate two-party vote in Class II races
    # that cycle, capturing most competitive seats (Ossoff, Hickenlooper, etc.).
    # Source: Class II aggregate two-party results, MIT Election Data Lab.
    _CLASS_II_2020_DEM_TWO_PARTY: float = 0.524
    baseline_label = _compute_baseline_label(_CLASS_II_2020_DEM_TWO_PARTY)

    return {
        "baseline_year": 2020,
        "baseline_label": baseline_label,
        "baseline_dem_two_party": _CLASS_II_2020_DEM_TWO_PARTY,
        "dem_wins_at_baseline": dem_wins_at_baseline,
        "dem_holdover_seats": _DEM_HOLDOVER_SEATS,
        "total_dem_projected": total_dem_projected,
        "seats_needed_for_majority": seats_needed_for_majority,
        "structural_gap": structural_gap,
    }


def _build_not_up_states() -> tuple[list[str], list[str]]:
    """Identify states not up for election in 2026, split by party."""
    not_up_d = sorted(
        st for st, party in SENATE_DELEGATION.items()
        if st not in SENATE_2026_STATES and party in ("D", "I")
    )
    not_up_r = sorted(
        st for st, party in SENATE_DELEGATION.items()
        if st not in SENATE_2026_STATES and party == "R"
    )
    return not_up_d, not_up_r


@router.get("/senate/scrolly-context")
def get_scrolly_context(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict:
    """Return narrative context data for the scrollytelling homepage.

    Provides the structural framing needed to tell the story of the 2026
    Senate map:
    - zone_counts: how many seats fall in each of the 7 narrative buckets
    - not_up states: which states are sitting out the election cycle
    - structural_context: what Democrats would need to win at the baseline

    This endpoint shares prediction data with /senate/overview (same DB
    queries) so both views stay in sync automatically.
    """
    version_id = getattr(request.app.state, "version_id", None)

    _has_mode = False
    if version_id:
        try:
            _has_mode = "forecast_mode" in [
                row[0] for row in db.execute("DESCRIBE predictions").fetchall()
            ]
        except duckdb.Error:
            pass
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    pred_by_race = _fetch_scrolly_predictions(db, version_id, _mode_filter)

    # Build classified race list
    races = [classify_race(st, pred_by_race) for st in sorted(SENATE_2026_STATES)]

    # Attach poll counts
    poll_counts = _fetch_scrolly_poll_counts(db)
    for r in races:
        r["n_polls"] = poll_counts.get(r["race"], 0)

    not_up_d_states, not_up_r_states = _build_not_up_states()

    competitive_ratings = {"lean_d", "tossup", "lean_r"}
    competitive_races = [r for r in races if r["rating"] in competitive_ratings]

    return {
        "zone_counts": build_zone_counts(races),
        "not_up_d_states": not_up_d_states,
        "not_up_r_states": not_up_r_states,
        "structural_context": _build_structural_context(races),
        "competitive_races": competitive_races,
    }
