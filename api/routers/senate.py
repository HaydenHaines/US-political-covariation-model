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

# Derived from _CLASS_II_INCUMBENT above -- count at import time so the
# constant stays in sync with the incumbent map automatically.
_DEM_CLASS_II_COUNT = sum(1 for p in _CLASS_II_INCUMBENT.values() if p == "D")
_GOP_CLASS_II_COUNT = sum(1 for p in _CLASS_II_INCUMBENT.values() if p == "R")
_DEM_HOLDOVER_SEATS = DEM_SAFE_SEATS - _DEM_CLASS_II_COUNT  # seats not up in 2026

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
