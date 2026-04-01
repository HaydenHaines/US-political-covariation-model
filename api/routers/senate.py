from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from api.db import get_db
from api.models import ChamberProbabilityResponse, SeatDistributionBucket

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

# Derived seat counts — computed from _CLASS_II_INCUMBENT so they stay in sync
# automatically if the incumbent map is ever updated.
#
# Class II (up in 2026): 33 seats.
# Holdovers (not up in 2026): 100 - 33 = 67 seats.
#   Dem holdovers: DEM_SAFE_SEATS(47) − Dem Class II seats up(14)  = 33
#   GOP holdovers: GOP_SAFE_SEATS(53) − GOP Class II seats up(19)  = 34
#
# These are the correct baselines for projected-seat arithmetic.  Using
# DEM_SAFE_SEATS/GOP_SAFE_SEATS directly double-counts the Class II seats.
_DEM_CLASS_II_COUNT = sum(1 for p in _CLASS_II_INCUMBENT.values() if p == "D")
_GOP_CLASS_II_COUNT = sum(1 for p in _CLASS_II_INCUMBENT.values() if p == "R")
_DEM_HOLDOVER_SEATS = DEM_SAFE_SEATS - _DEM_CLASS_II_COUNT   # seats not up in 2026
_GOP_HOLDOVER_SEATS = GOP_SAFE_SEATS - _GOP_CLASS_II_COUNT   # seats not up in 2026


def _rating_to_zone(rating: str, incumbent: str) -> str:
    """Map a race rating + incumbent party to a scrollytelling zone label.

    Zone categories group seats into seven buckets for narrative display:
      - safe_up_d   D incumbent, model rates safe/likely D
      - contested_d D incumbent, model rates lean D (close but Dem-favored)
      - tossup       tossup regardless of incumbent
      - contested_r  R incumbent, model rates lean R (close but R-favored)
      - safe_up_r   R incumbent, model rates safe/likely R

    Edge cases where the rating side disagrees with the incumbent
    (e.g. D-held seat rated lean_r) are resolved by the **rating**, not
    the incumbent.  The model's view of the race determines the zone.
    """
    if rating == "tossup":
        return "tossup"
    # Determine direction from rating label
    if rating in ("safe_d", "likely_d"):
        return "safe_up_d"
    if rating == "lean_d":
        return "contested_d"
    if rating == "lean_r":
        return "contested_r"
    # safe_r or likely_r
    return "safe_up_r"


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

    # Projected totals: holdover seats (not up in 2026) + Class II seats the
    # model clearly favors.  Tossups excluded from both sides — their outcome
    # is uncertain.
    #
    # IMPORTANT: use _DEM_HOLDOVER_SEATS / _GOP_HOLDOVER_SEATS, NOT
    # DEM_SAFE_SEATS / GOP_SAFE_SEATS.  The latter counts ALL current Dem/GOP
    # seats including Class II seats that are up in 2026; adding dem_favored /
    # gop_favored on top would double-count those seats and produce totals like
    # 55D + 71R = 126 instead of the correct 100.
    dem_projected = _DEM_HOLDOVER_SEATS + dem_favored
    gop_projected = _GOP_HOLDOVER_SEATS + gop_favored

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
    # Subtitle shows the competitive breakdown, not a projected total. Using
    # "GOP projected 71 seats" contradicts the balance bar (which shows only
    # the non-contested safe-seat count, 53R). Both numbers are valid but count
    # different things; putting them on the same screen without explanation
    # confuses readers. The competitive breakdown is self-contained and matches
    # what the balance bar conveys visually.
    tossup_label = f"{n_tossup} tossup" if n_tossup == 1 else f"{n_tossup} tossups"
    competitive_subtitle = f"{n_competitive} competitive races · {tossup_label}"

    if gop_projected > dem_projected:
        if gop_projected >= 55:
            return "Republicans Strongly Favored to Hold the Senate", competitive_subtitle, dem_projected, gop_projected
        return "Republicans Favored to Hold the Senate", competitive_subtitle, dem_projected, gop_projected
    if dem_projected >= 55:
        return "Democrats Strongly Favored to Flip the Senate", competitive_subtitle, dem_projected, gop_projected
    return "Democrats Favored to Flip the Senate", competitive_subtitle, dem_projected, gop_projected


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

        # Zone derives from model rating; incumbent_party is always from _CLASS_II_INCUMBENT
        # even when the model has a prediction (so edge cases resolve via rating side).
        incumbent_party = _CLASS_II_INCUMBENT.get(state_abbr, "R")
        zone = _rating_to_zone(rating, incumbent_party)

        races.append({
            "state": state_abbr,
            "race": race,
            "slug": slug,
            "rating": rating,
            "margin": round(margin, 4),
            "n_polls": n_polls,
            "zone": zone,
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

    # Compute zone_counts: seats in each of the 7 narrative buckets.
    # The Class II contested seats come from the races list (all 33).
    # Holdover (not-up) seats come from SENATE_DELEGATION minus the 33 Class II states.
    # Note: SENATE_DELEGATION has 51 entries (50 states + DC); each entry represents
    # the *overall* delegation, which has 2 seats.  But we want seats not up in 2026,
    # which is 100 - 33 = 67 seats total (34D + 33R computed from _CLASS_II_INCUMBENT).
    # We track those via _DEM_HOLDOVER_SEATS and _GOP_HOLDOVER_SEATS.
    zone_counts: dict[str, int] = {
        "not_up_d": _DEM_HOLDOVER_SEATS,
        "safe_up_d": sum(1 for r in races if r["zone"] == "safe_up_d"),
        "contested_d": sum(1 for r in races if r["zone"] == "contested_d"),
        "tossup": sum(1 for r in races if r["zone"] == "tossup"),
        "contested_r": sum(1 for r in races if r["zone"] == "contested_r"),
        "safe_up_r": sum(1 for r in races if r["zone"] == "safe_up_r"),
        "not_up_r": _GOP_HOLDOVER_SEATS,
    }

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
        "zone_counts": zone_counts,
        "state_colors": state_colors,
        "updated_at": updated_at,
    }


# ── Senate composition assumptions for chamber probability ───────────────────
#
# Class II (up in 2026): 33 seats.  Democrats defend 12, Republicans defend 21.
# Class I + III (not up in 2026): 67 seats.
# Current holdover split: 35D/I + 32R (of the 67 not up in 2026).
#
# We calculate starting Dem holdover seats from first principles:
#   Total current Senate: 47D/I + 53R = 100
#   Class II up in 2026: 13D + 20R = 33  (from _CLASS_II_INCUMBENT)
#   Holdover (not up):  47-13 = 34D + 53-20 = 33R = 67 seats
#
# So Democrats start with 34 holdover seats.  To reach majority they need
# >=17 wins from the 33 races up (to hit >=51).  However, because 2026 falls
# under a Republican presidency (no VP tiebreaker for Dems), the effective
# majority threshold is 51 outright.  We report both >=50 and >=51 metrics.

# State-level uncertainty parameters -- mirrors the values in forecast.py.
# Keeping them here avoids a cross-router import while ensuring both routers
# use the same calibration.  See forecast.py docs for calibration notes.
_STATE_STD_FLOOR = 0.035      # minimum state-level std
_STATE_STD_CAP = 0.15         # hard cap -- beyond this, the race is essentially a coin flip
_STATE_STD_FALLBACK = 0.065   # used when vote-weighted std is unavailable

# (_DEM_CLASS_II_COUNT, _GOP_CLASS_II_COUNT, _DEM_HOLDOVER_SEATS, _GOP_HOLDOVER_SEATS
# are defined near the top of this module, right after the color constants.)

# Uncertainty for "safe" seats: the model hasn't predicted these, but we treat
# the incumbent as a strong favorite.  A 5pp std (sigma=0.05) centered at +/-0.25
# from 50% gives roughly a 95% win probability -- clearly safe without being
# a certainty (a small upset tail keeps the simulation honest).
_SAFE_SEAT_STD = 0.05

# Seats in the distribution output: filter to a window around the median
# to keep the payload small while covering >99% of the probability mass.
_DISTRIBUTION_SEAT_RANGE = range(35, 66)  # 35-65 seats covers all realistic outcomes

# Number of Monte Carlo draws.  10,000 gives ~0.5pp standard error on a
# 50/50 probability estimate -- more than enough for display purposes.
_N_SIMULATIONS = 10_000


def _simulate_chamber_probability(
    modeled_races: list[tuple[float, float]],
    safe_dem_wins: int,
    safe_gop_wins: int,
    dem_holdover: int,
    n_sims: int,
    rng_seed: int | None = 42,
) -> ChamberProbabilityResponse:
    """Monte Carlo chamber control simulation.

    For each simulation:
    1. Draw each modeled race from Normal(pred, std) -- clip to [0, 1].
    2. Count Dem wins where draw > 0.5.
    3. Add safe Dem wins (high-confidence).
    4. Add holdover (not up in 2026) Dem seats.
    5. Total Dem seats = holdover + safe wins + modeled wins.

    Args:
        modeled_races: List of (dem_share_pred, pred_std) for contested races
            where the model has a prediction.
        safe_dem_wins: Number of D-held seats with no model prediction (treated
            as safe with _SAFE_SEAT_STD uncertainty).
        safe_gop_wins: Number of R-held seats with no model prediction.
        dem_holdover: Dem seats not up in 2026.
        n_sims: Number of Monte Carlo simulations.
        rng_seed: Optional fixed seed for reproducibility in tests.
    """
    rng = np.random.default_rng(rng_seed)

    n_modeled = len(modeled_races)
    n_safe = safe_dem_wins + safe_gop_wins

    # Build prediction and std arrays for all contested seats.
    if n_modeled > 0:
        preds = np.array([r[0] for r in modeled_races])
        stds = np.array([r[1] for r in modeled_races])
    else:
        preds = np.empty(0)
        stds = np.empty(0)

    # Safe Dem seats: center at 0.75 (strong favorite), small std
    safe_dem_preds = np.full(safe_dem_wins, 0.75)
    safe_dem_stds = np.full(safe_dem_wins, _SAFE_SEAT_STD)

    # Safe GOP seats: center at 0.25 (strong underdog), small std
    safe_gop_preds = np.full(safe_gop_wins, 0.25)
    safe_gop_stds = np.full(safe_gop_wins, _SAFE_SEAT_STD)

    all_preds = np.concatenate([preds, safe_dem_preds, safe_gop_preds])
    all_stds = np.concatenate([stds, safe_dem_stds, safe_gop_stds])

    n_total_races = len(all_preds)

    if n_total_races == 0:
        # No races to simulate -- return deterministic result from holdover alone
        total_dem = dem_holdover
        return ChamberProbabilityResponse(
            dem_control_pct=100.0 if total_dem >= 50 else 0.0,
            rep_control_pct=100.0 if total_dem < 51 else 0.0,
            dem_majority_pct=100.0 if total_dem >= 51 else 0.0,
            median_dem_seats=total_dem,
            median_rep_seats=100 - total_dem,
            seat_distribution=[SeatDistributionBucket(seats=total_dem, probability=1.0)],
            n_simulations=n_sims,
            n_modeled_races=0,
            n_safe_races=0,
        )

    # Draw all simulations at once: shape (n_sims, n_total_races).
    # np.clip ensures draws stay in [0, 1].
    draws = rng.normal(loc=all_preds, scale=all_stds, size=(n_sims, n_total_races))
    draws = np.clip(draws, 0.0, 1.0)

    # Dem wins per simulation: each race where draw > 0.5
    dem_wins_per_sim = np.sum(draws > 0.5, axis=1)

    # Total Dem seats = holdover (already decided) + wins from contested races
    total_dem_per_sim = dem_holdover + dem_wins_per_sim

    # Probability estimates
    dem_control_pct = float(np.mean(total_dem_per_sim >= 50) * 100)
    dem_majority_pct = float(np.mean(total_dem_per_sim >= 51) * 100)
    rep_control_pct = float(np.mean(total_dem_per_sim < 50) * 100)

    median_dem = int(np.median(total_dem_per_sim))
    median_rep = 100 - median_dem

    seat_distribution = []
    for seats in _DISTRIBUTION_SEAT_RANGE:
        prob = float(np.mean(total_dem_per_sim == seats))
        if prob > 0.0001:
            seat_distribution.append(SeatDistributionBucket(seats=seats, probability=round(prob, 4)))

    return ChamberProbabilityResponse(
        dem_control_pct=round(dem_control_pct, 1),
        rep_control_pct=round(rep_control_pct, 1),
        dem_majority_pct=round(dem_majority_pct, 1),
        median_dem_seats=median_dem,
        median_rep_seats=median_rep,
        seat_distribution=seat_distribution,
        n_simulations=n_sims,
        n_modeled_races=n_modeled,
        n_safe_races=n_safe,
    )


@router.get("/senate/chamber-probability", response_model=ChamberProbabilityResponse)
def get_chamber_probability(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    n_simulations: int = Query(
        _N_SIMULATIONS,
        ge=1000,
        le=100_000,
        description="Number of Monte Carlo simulations (default 10,000)",
    ),
) -> ChamberProbabilityResponse:
    """Compute Monte Carlo chamber control probability for the 2026 Senate.

    Loads vote-weighted state predictions for all 33 Class II seats.  For races
    with model predictions, draws each simulation from Normal(pred, std).  For
    races without predictions (safe seats), treats the incumbent as a strong
    favorite (pred=0.75/0.25, std=0.05).

    The Dem total in each simulation = holdover seats (not up in 2026) + wins
    from the 33 contested races.  Control probability is fraction of sims where
    Dems reach >=50 (with VP) or >=51 (outright majority).
    """
    version_id = getattr(request.app.state, "version_id", None)
    if not version_id:
        total_dem = DEM_SAFE_SEATS
        return ChamberProbabilityResponse(
            dem_control_pct=0.0,
            rep_control_pct=100.0,
            dem_majority_pct=0.0,
            median_dem_seats=total_dem,
            median_rep_seats=100 - total_dem,
            seat_distribution=[SeatDistributionBucket(seats=total_dem, probability=1.0)],
            n_simulations=n_simulations,
            n_modeled_races=0,
            n_safe_races=0,
        )

    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    modeled_races: list[tuple[float, float]] = []
    safe_dem_wins = 0
    safe_gop_wins = 0

    for st in sorted(SENATE_2026_STATES):
        race = f"2026 {st} Senate"

        # Fetch county-level predictions and vote weights.
        # State pred and std are computed in Python because DuckDB does not
        # allow window function calls inside aggregate expressions.
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
              {_mode_filter}
            """,
            [version_id, race, st],
        ).fetchdf()

        if county_rows.empty:
            incumbent = _CLASS_II_INCUMBENT.get(st, "R")
            if incumbent == "D":
                safe_dem_wins += 1
            else:
                safe_gop_wins += 1
            continue

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
        modeled_races.append((state_pred, std))

    return _simulate_chamber_probability(
        modeled_races=modeled_races,
        safe_dem_wins=safe_dem_wins,
        safe_gop_wins=safe_gop_wins,
        dem_holdover=_DEM_HOLDOVER_SEATS,
        n_sims=n_simulations,
    )


# ── Scrolly-context endpoint ─────────────────────────────────────────────────


def _compute_baseline_label(pres_dem_share: float) -> str:
    """Format the 2024 presidential Dem share as a party-margin label.

    Measures how far the national Dem two-party share deviates from 50/50:
      shift = pres_dem_share - 0.5
      negative shift → Republican advantage → "R+X.X"
      positive shift → Democrat advantage → "D+X.X"

    Example: 0.4841 → shift=-0.0159 → "R+1.6"
    """
    shift = pres_dem_share - 0.5
    magnitude = round(abs(shift) * 100, 1)
    if shift < 0:
        return f"R+{magnitude}"
    return f"D+{magnitude}"


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
    from src.prediction.generic_ballot import PRES_DEM_SHARE_2024_NATIONAL

    # ── Re-use the same prediction + fallback logic as /senate/overview ──────
    version_id = getattr(request.app.state, "version_id", None)

    _has_mode = False
    if version_id:
        try:
            _has_mode = "forecast_mode" in [
                row[0] for row in db.execute("DESCRIBE predictions").fetchall()
            ]
        except Exception:
            pass
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    races: list[dict] = []
    for st in sorted(SENATE_2026_STATES):
        race = f"2026 {st} Senate"
        slug = race.lower().replace(" ", "-")
        incumbent_party = _CLASS_II_INCUMBENT.get(st, "R")

        if version_id:
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
        else:
            row = None

        if row and row[0] is not None:
            margin = float(row[0]) - 0.5
            rating = _margin_to_rating(margin)
        else:
            margin = _DEFAULT_SAFE_MARGIN if incumbent_party == "D" else -_DEFAULT_SAFE_MARGIN
            rating = "safe_d" if incumbent_party == "D" else "safe_r"

        zone = _rating_to_zone(rating, incumbent_party)
        races.append({
            "state": st,
            "race": race,
            "slug": slug,
            "rating": rating,
            "margin": round(margin, 4),
            "n_polls": 0,  # filled in below after poll_counts lookup
            "zone": zone,
        })

    # ── Poll counts for competitive races ────────────────────────────────────
    # The battleground race cards in the scrollytelling section display n_polls.
    # Fetch the same counts the overview endpoint uses so both views are in sync.
    try:
        scrolly_polls_df = db.execute(
            """
            SELECT race, COUNT(*) AS n_polls
            FROM polls
            WHERE LOWER(race) LIKE '%senate%'
            GROUP BY race
            """
        ).fetchdf()
    except Exception:
        scrolly_polls_df = pd.DataFrame()

    scrolly_poll_counts: dict[str, int] = {}
    for _, poll_row in scrolly_polls_df.iterrows():
        scrolly_poll_counts[str(poll_row["race"])] = int(poll_row["n_polls"])

    for r in races:
        r["n_polls"] = scrolly_poll_counts.get(r["race"], 0)

    # ── Zone counts ──────────────────────────────────────────────────────────
    zone_counts: dict[str, int] = {
        "not_up_d": _DEM_HOLDOVER_SEATS,
        "safe_up_d": sum(1 for r in races if r["zone"] == "safe_up_d"),
        "contested_d": sum(1 for r in races if r["zone"] == "contested_d"),
        "tossup": sum(1 for r in races if r["zone"] == "tossup"),
        "contested_r": sum(1 for r in races if r["zone"] == "contested_r"),
        "safe_up_r": sum(1 for r in races if r["zone"] == "safe_up_r"),
        "not_up_r": _GOP_HOLDOVER_SEATS,
    }

    # ── Not-up states ────────────────────────────────────────────────────────
    # States in SENATE_DELEGATION that have no Class II seat up in 2026.
    # "D" includes independents who caucus with Democrats.
    not_up_d_states = sorted(
        st for st, party in SENATE_DELEGATION.items()
        if st not in SENATE_2026_STATES and party in ("D", "I")
    )
    not_up_r_states = sorted(
        st for st, party in SENATE_DELEGATION.items()
        if st not in SENATE_2026_STATES and party == "R"
    )

    # ── Structural context ───────────────────────────────────────────────────
    # Counts how many Class II races Democrats currently win at the model's
    # predicted margins (margin > 0 means Dem-favored; tossups excluded).
    dem_wins_at_baseline = sum(1 for r in races if r["margin"] > 0)
    total_dem_projected = _DEM_HOLDOVER_SEATS + dem_wins_at_baseline
    seats_needed_for_majority = 51
    structural_gap = seats_needed_for_majority - total_dem_projected

    # The structural argument uses 2018 as the reference environment: the last
    # midterm before 2026, and the strongest recent D midterm performance.
    # Democrats won 53.4% of the national two-party House vote in 2018 (D+6.8).
    # Source: MIT Election Data and Science Lab House Popular Vote Totals.
    # We ask: even if Democrats repeat 2018's strong environment, do they win?
    _MIDTERM_2018_DEM_TWO_PARTY: float = 0.534
    baseline_label = _compute_baseline_label(_MIDTERM_2018_DEM_TWO_PARTY)

    structural_context = {
        "baseline_year": 2018,
        "baseline_label": baseline_label,
        "baseline_dem_two_party": _MIDTERM_2018_DEM_TWO_PARTY,
        "dem_wins_at_baseline": dem_wins_at_baseline,
        "dem_holdover_seats": _DEM_HOLDOVER_SEATS,
        "total_dem_projected": total_dem_projected,
        "seats_needed_for_majority": seats_needed_for_majority,
        "structural_gap": structural_gap,
    }

    # ── Competitive races ────────────────────────────────────────────────────
    competitive_ratings = {"lean_d", "tossup", "lean_r"}
    competitive_races = [r for r in races if r["rating"] in competitive_ratings]

    return {
        "zone_counts": zone_counts,
        "not_up_d_states": not_up_d_states,
        "not_up_r_states": not_up_r_states,
        "structural_context": structural_context,
        "competitive_races": competitive_races,
    }
