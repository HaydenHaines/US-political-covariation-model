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

# Which party currently holds each Class II Senate seat (up in 2026).
# "D" includes independents who caucus with Democrats (VT, ME).
# Used to assign ratings/margins to seats that have no model prediction yet
# (e.g. uncontested races or states with insufficient county data).
_CLASS_II_INCUMBENT: dict[str, str] = {
    "AL": "R",  # Richard Shelby seat → Tommy Tuberville
    "AK": "R",  # Lisa Murkowski
    "AR": "R",  # Tom Cotton
    "CO": "D",  # John Hickenlooper
    "DE": "D",  # Chris Coons
    "GA": "D",  # Jon Ossoff
    "IA": "R",  # Chuck Grassley
    "ID": "R",  # Jim Risch
    "IL": "D",  # Dick Durbin
    "KS": "R",  # Roger Marshall
    "KY": "R",  # Mitch McConnell
    "LA": "R",  # Bill Cassidy
    "MA": "D",  # Ed Markey
    "ME": "D",  # Angus King (I, caucuses D)
    "MI": "D",  # Gary Peters
    "MN": "D",  # Tina Smith
    "MS": "R",  # Roger Wicker
    "MT": "R",  # Steve Daines
    "NC": "R",  # Thom Tillis
    "NE": "R",  # Deb Fischer
    "NH": "D",  # Jeanne Shaheen
    "NJ": "D",  # Andy Kim
    "NM": "D",  # Martin Heinrich
    "OK": "R",  # James Lankford
    "OR": "D",  # Jeff Merkley
    "RI": "D",  # Jack Reed
    "SC": "R",  # Lindsey Graham
    "SD": "R",  # Mike Rounds
    "TN": "R",  # Marsha Blackburn
    "TX": "R",  # John Cornyn
    "VA": "D",  # Mark Warner
    "WV": "R",  # Shelley Moore Capito (flipped R after Manchin retirement)
    "WY": "R",  # John Barrasso
}

# Default margin magnitude for seats with no model prediction.
# Safe means the model hasn't bothered predicting them — treat as solidly held.
# Positive = safe D, negative = safe R.
_DEFAULT_SAFE_MARGIN = 0.25

# Rating margin thresholds (margin = dem_share - 0.5)
_TOSSUP_MAX = 0.03
_LEAN_MAX = 0.08
_LIKELY_MAX = 0.15

# Map colors — Dusty Ink palette.
# Contested states → rating-based color; non-contested → delegation party color.
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
_UNCONTESTED_FALLBACK_COLOR = "#b5a995"  # neutral beige — uncontested seats with no delegation data
_PARTY_UNKNOWN_COLOR = "#eae7e2"  # off-white — unknown delegation party


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


def _build_headline(races: list[dict]) -> tuple[str, str, int, int]:
    """Derive a headline + subtitle and projected seat totals from current race ratings.

    Seat projections count safe seats (not up in 2026) plus contested seats
    that the model clearly favors for each party. Tossups are excluded from
    both totals — the standard forecasting convention, matching how outlets
    like 538 and Cook Report present seat projections.

    Returns (headline, subtitle, dem_projected, gop_projected).
    """
    # Count races where the model clearly favors each party (tossups excluded)
    dem_favored = sum(1 for r in races if r["margin"] > _TOSSUP_MAX)
    gop_favored = sum(1 for r in races if r["margin"] < -_TOSSUP_MAX)
    n_tossup = sum(1 for r in races if r["rating"] == "tossup")
    competitive = [r for r in races if r["rating"] in ("tossup", "lean_d", "lean_r")]
    n_competitive = len(competitive)

    # Projected totals: off-cycle safe seats + contested seats the model favors.
    # Tossups intentionally excluded from both sides — their outcome is uncertain.
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
            dem_projected,
            gop_projected,
        )
    if gop_projected > dem_projected:
        if gop_projected >= 55:
            subtitle = f"GOP projected {gop_projected} seats · {n_competitive} competitive races"
            return "Republicans Strongly Favored to Hold the Senate", subtitle, dem_projected, gop_projected
        subtitle = f"GOP projected {gop_projected} seats · {n_competitive} competitive races"
        return "Republicans Favored to Hold the Senate", subtitle, dem_projected, gop_projected
    if dem_projected >= 55:
        subtitle = f"Dems projected {dem_projected} seats · {n_competitive} competitive races"
        return "Democrats Strongly Favored to Flip the Senate", subtitle, dem_projected, gop_projected
    subtitle = f"Dems projected {dem_projected} seats · {n_competitive} competitive races"
    return "Democrats Favored to Flip the Senate", subtitle, dem_projected, gop_projected


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
            "dem_projected": DEM_SAFE_SEATS,
            "gop_projected": GOP_SAFE_SEATS,
            "races": [],
        }

    # Check if forecast_mode column exists (backward compat)
    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    # Vote-weighted state prediction per senate race.
    # Each race is filtered to only include counties IN that state,
    # matching the same SQL approach used in forecast.py race detail.
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
              {_mode_filter}
            """,
            [version_id, race, st],
        ).fetchone()
        if row and row[0] is not None:
            pred_by_race[race] = (st, float(row[0]))

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
        n_polls = poll_counts.get(race, 0)
        slug = race.lower().replace(" ", "-")

        if race in pred_by_race:
            # Model has a prediction for this race — use it.
            state_abbr, state_pred = pred_by_race[race]
            margin = state_pred - 0.5
            rating = _margin_to_rating(margin)
        else:
            # No model prediction yet (seat not in training data, insufficient
            # county coverage, etc.). Fall back to the incumbent's party and
            # treat it as safely held — the frontend needs *all* 33 races for
            # the balance bar and seat-count summary to be meaningful.
            state_abbr = st
            incumbent_party = _CLASS_II_INCUMBENT.get(st, "R")
            margin = _DEFAULT_SAFE_MARGIN if incumbent_party == "D" else -_DEFAULT_SAFE_MARGIN
            rating = "safe_d" if incumbent_party == "D" else "safe_r"

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

    headline, subtitle, dem_projected, gop_projected = _build_headline(races)

    # Build state_colors map: every state gets a hex color for the map.
    # Contested states → rating-based color. Non-contested → delegation party color.
    race_by_state = {r["state"]: r for r in races}
    state_colors = {}
    for st, delegation in SENATE_DELEGATION.items():
        if st in race_by_state:
            # Contested: use race rating color
            state_colors[st] = _RATING_COLORS.get(race_by_state[st]["rating"], _UNCONTESTED_FALLBACK_COLOR)
        else:
            # Not contested in 2026: use delegation color (lighter shade)
            state_colors[st] = _PARTY_COLORS.get(delegation, _PARTY_UNKNOWN_COLOR)

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
        # Projected totals include model-favored contested seats; tossups excluded.
        # These are the numbers the hero section and balance bar should display.
        "dem_projected": dem_projected,
        "gop_projected": gop_projected,
        "races": races,
        "state_colors": state_colors,
        "updated_at": updated_at,
    }
