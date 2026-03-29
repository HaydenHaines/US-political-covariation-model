from __future__ import annotations

import logging

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, Request

from api.db import get_db

log = logging.getLogger(__name__)

router = APIRouter(tags=["senate"])

# Seats not up for election in 2026 (Class I + Class III senators already decided)
DEM_SAFE_SEATS = 47
GOP_SAFE_SEATS = 53

# Current Senate composition: which party holds each state's seats.
# "D" = Democrat, "R" = Republican, "I" = Independent (caucuses with D).
# For map coloring, non-contested states show their overall delegation color.
SENATE_DELEGATION = {
    "AL": "R", "AK": "R", "AZ": "D", "AR": "R", "CA": "D", "CO": "D",
    "CT": "D", "DE": "D", "FL": "R", "GA": "D", "HI": "D", "ID": "R",
    "IL": "D", "IN": "R", "IA": "R", "KS": "R", "KY": "R", "LA": "R",
    "ME": "split", "MD": "D", "MA": "D", "MI": "D", "MN": "D", "MS": "R",
    "MO": "R", "MT": "R", "NE": "R", "NV": "D", "NH": "D", "NJ": "D",
    "NM": "D", "NY": "D", "NC": "R", "ND": "R", "OH": "R", "OK": "R",
    "OR": "D", "PA": "split", "RI": "D", "SC": "R", "SD": "R", "TN": "R",
    "TX": "R", "UT": "R", "VT": "D", "VA": "D", "WA": "D", "WV": "R",
    "WI": "split", "WY": "R", "DC": "D",
}

SENATE_2026_STATES = {
    "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS",
    "KY", "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE",
    "NH", "NJ", "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
    "VA", "WV", "WY",
}

# Rating margin thresholds (margin = dem_share - 0.5)
_TOSSUP_MAX = 0.03
_LEAN_MAX = 0.08
_LIKELY_MAX = 0.15


def _margin_to_rating(margin: float) -> str:
    """Convert signed Dem margin to a rating label.

    margin = state_pred - 0.5 (positive = Dem-favored, negative = GOP-favored)
    """
    abs_m = abs(margin)
    if abs_m < _TOSSUP_MAX:
        return "tossup"
    direction = "_d" if margin > 0 else "_r"
    if abs_m < _LEAN_MAX:
        return f"lean{direction}"
    if abs_m < _LIKELY_MAX:
        return f"likely{direction}"
    return f"safe{direction}"


def _rating_sort_key(rating: str) -> int:
    """Sort races: safe D first, through tossup, to safe R last."""
    return {
        "safe_d": 0, "likely_d": 1, "lean_d": 2,
        "tossup": 3,
        "lean_r": 4, "likely_r": 5, "safe_r": 6,
    }.get(rating, 3)


def _build_headline(races: list[dict]) -> tuple[str, str]:
    """Derive a headline + subtitle from the current race ratings.

    Returns (headline, subtitle).
    """
    # Count races where the model favors each party (tossups split as contested)
    dem_favored = sum(1 for r in races if r["margin"] > _TOSSUP_MAX)
    gop_favored = sum(1 for r in races if r["margin"] < -_TOSSUP_MAX)
    n_tossup = sum(1 for r in races if r["rating"] == "tossup")
    competitive = [r for r in races if r["rating"] in ("tossup", "lean_d", "lean_r")]
    n_competitive = len(competitive)

    # Seat projections: safe seats plus clearly-favored contested seats
    dem_projected = DEM_SAFE_SEATS + dem_favored
    gop_projected = GOP_SAFE_SEATS + gop_favored

    seat_diff = dem_projected - gop_projected

    if abs(seat_diff) <= 2:
        subtitle_parts = [f"{n_tossup} tossup" if n_tossup == 1 else f"{n_tossup} tossups"]
        if n_competitive > n_tossup:
            subtitle_parts.append(f"{n_competitive - n_tossup} more lean races")
        return (
            "Senate Control on a Knife's Edge",
            f"{' · '.join(subtitle_parts)} in play",
        )
    if gop_projected > dem_projected:
        if gop_projected >= 55:
            subtitle = f"GOP projected {gop_projected} seats · {n_competitive} competitive races"
            return "Republicans Strongly Favored to Hold the Senate", subtitle
        subtitle = f"GOP projected {gop_projected} seats · {n_competitive} competitive races"
        return "Republicans Favored to Hold the Senate", subtitle
    if gop_projected >= 55:
        subtitle = f"Dems projected {dem_projected} seats · {n_competitive} competitive races"
        return "Democrats Strongly Favored to Flip the Senate", subtitle
    subtitle = f"Dems projected {dem_projected} seats · {n_competitive} competitive races"
    return "Democrats Favored to Flip the Senate", subtitle


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
            "races": [],
        }

    # Check if forecast_mode column exists (backward compat)
    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'national'" if _has_mode else ""

    # Vote-weighted state prediction for all senate races at once,
    # matching the same SQL used in forecast.py race detail endpoint.
    senate_preds = db.execute(
        f"""
        SELECT
            p.race,
            c.state_abbr,
            CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                 THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                      / SUM(COALESCE(c.total_votes_2024, 0))
                 ELSE AVG(p.pred_dem_share)
            END AS state_pred,
            COUNT(*) AS n_counties
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE p.version_id = ?
          AND LOWER(p.race) LIKE '%senate%'
          {_mode_filter}
        GROUP BY p.race, c.state_abbr
        """,
        [version_id],
    ).fetchall()

    # Build lookup: race -> (state_abbr, state_pred)
    pred_by_race: dict[str, tuple[str, float]] = {}
    for race_str, state_abbr, state_pred, _n in senate_preds:
        pred_by_race[race_str] = (state_abbr, float(state_pred))

    # Fetch poll counts per senate race
    try:
        polls_df = db.execute(
            """
            SELECT race, COUNT(*) AS n_polls
            FROM polls
            WHERE LOWER(race) LIKE '%senate%'
            GROUP BY race
            """
        ).fetchdf()
    except Exception:
        polls_df = pd.DataFrame()

    poll_counts: dict[str, int] = {}
    for _, row in polls_df.iterrows():
        poll_counts[str(row["race"])] = int(row["n_polls"])

    races = []
    for st in sorted(SENATE_2026_STATES):
        race = f"2026 {st} Senate"
        if race not in pred_by_race:
            continue

        state_abbr, state_pred = pred_by_race[race]
        margin = state_pred - 0.5
        rating = _margin_to_rating(margin)
        slug = race.lower().replace(" ", "-")
        n_polls = poll_counts.get(race, 0)

        races.append({
            "state": state_abbr,
            "race": race,
            "slug": slug,
            "rating": rating,
            "margin": round(margin, 4),
            "n_polls": n_polls,
        })

    # Sort: tossups first, then lean, likely, safe; break ties alphabetically by state
    races.sort(key=lambda r: (_rating_sort_key(r["rating"]), r["state"]))

    headline, subtitle = _build_headline(races)

    # Build state_colors map: every state gets a hex color for the map.
    # Contested states → rating-based color. Non-contested → delegation party color.
    # Dusty Ink palette
    _RATING_COLORS = {
        "safe_d": "#2d4a6f", "likely_d": "#4b6d90", "lean_d": "#7e9ab5",
        "tossup": "#8a6b8a",
        "lean_r": "#c4907a", "likely_r": "#9e5e4e", "safe_r": "#6e3535",
    }
    _PARTY_COLORS = {
        "D": "#3a5f8a",    # Muted dark blue — clearly "Dem-held, no race"
        "R": "#7a4a4a",    # Muted dark red — clearly "GOP-held, no race"
        "split": "#5a5a5a",
    }

    race_by_state = {r["state"]: r for r in races}
    state_colors = {}
    for st, delegation in SENATE_DELEGATION.items():
        if st in race_by_state:
            # Contested: use race rating color
            state_colors[st] = _RATING_COLORS.get(race_by_state[st]["rating"], "#b5a995")
        else:
            # Not contested in 2026: use delegation color (lighter shade)
            state_colors[st] = _PARTY_COLORS.get(delegation, "#eae7e2")

    # Freshness: report the most recent poll date scraped, if available
    updated_at: str | None = None
    try:
        row = db.execute(
            "SELECT MAX(date) AS max_date FROM polls WHERE date IS NOT NULL"
        ).fetchone()
        if row and row[0]:
            updated_at = str(row[0])
    except Exception:
        pass

    return {
        "headline": headline,
        "subtitle": subtitle,
        "dem_seats_safe": DEM_SAFE_SEATS,
        "gop_seats_safe": GOP_SAFE_SEATS,
        "races": races,
        "state_colors": state_colors,
        "updated_at": updated_at,
    }
