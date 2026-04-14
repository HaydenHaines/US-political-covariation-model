"""Shared helpers and constants for Senate forecast endpoints.

Contains rating/zone classification, headline generation, and all
constant data (delegations, incumbents, colors, thresholds).
"""
from __future__ import annotations

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
    # Class II regular elections
    "AL", "AK", "AR", "CO", "DE", "GA", "IA", "ID", "IL", "KS",
    "KY", "LA", "MA", "ME", "MI", "MN", "MS", "MT", "NC", "NE",
    "NH", "NJ", "NM", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
    "VA", "WV", "WY",
    # Special elections (Rubio → Moody in FL, Vance → Husted in OH)
    "FL", "OH",
}

# Which party currently holds each Class II Senate seat (up in 2026).
# "D" includes independents who caucus with Democrats (VT, ME).
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
    # Special elections
    "FL": "R",  # Marco Rubio seat → Ashley Moody (appointed)
    "OH": "R",  # JD Vance seat → Jon Husted (appointed)
}

# Default margin magnitude for seats with no model prediction.
# Safe means the model hasn't bothered predicting them — treat as solidly held.
# Positive = safe D, negative = safe R.
_DEFAULT_SAFE_MARGIN = 0.25

# Rating margin thresholds — imported from the shared ratings module.
# Re-exported here so existing imports (e.g. from senate.__init__) continue to work.
from api.ratings import TOSSUP_MAX as _TOSSUP_MAX  # noqa: E402, F401

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

    margin = state_pred - 0.5 (positive = Dem-favored, negative = GOP-favored).
    Delegates to the shared ``api.ratings.margin_to_rating`` implementation.
    """
    from api.ratings import margin_to_rating
    return margin_to_rating(margin)


def _rating_sort_key(rating: str) -> int:
    """Sort races: safe D first, through tossup, to safe R last."""
    return {
        "safe_d": 0, "likely_d": 1, "lean_d": 2,
        "tossup": 3,
        "lean_r": 4, "likely_r": 5, "safe_r": 6,
    }.get(rating, 3)


def _build_headline(races: list[dict]) -> tuple[str, str, int, int]:
    """Derive a headline + subtitle and projected seat totals from current race ratings.

    Seat projections assign ALL 33 contested seats to D or R based on margin
    sign (positive = D-favored, negative = R-favored). Tossups are included by
    their margin direction so the headline totals always sum to 100 seats. The
    tossup count is surfaced in the subtitle for context, since tossup seats
    genuinely could go either way.

    This convention shows projected outcomes assuming each race goes to the
    currently-leading party, making the headline numbers more actionable than
    the traditional "tossups excluded" approach.

    Returns (headline, subtitle, dem_projected, gop_projected).
    """
    # Every contested seat is assigned: positive margin → D, negative → R.
    # Zero-margin ties are assigned to R (conservative default).
    dem_wins = sum(1 for r in races if r["margin"] > 0)
    gop_wins = sum(1 for r in races if r["margin"] <= 0)

    n_tossup = sum(1 for r in races if r["rating"] == "tossup")
    competitive = [r for r in races if r["rating"] in ("tossup", "lean_d", "lean_r")]
    n_competitive = len(competitive)

    # Projected totals: holdover seats (not up in 2026) + all Class II seats
    # assigned by margin direction (tossups included via their margin sign).
    dem_projected = _DEM_HOLDOVER_SEATS + dem_wins
    gop_projected = _GOP_HOLDOVER_SEATS + gop_wins

    headline, subtitle = _format_headline_text(
        dem_projected, gop_projected, n_tossup, n_competitive,
    )
    return headline, subtitle, dem_projected, gop_projected


def _format_headline_text(
    dem_projected: int,
    gop_projected: int,
    n_tossup: int,
    n_competitive: int,
) -> tuple[str, str]:
    """Build headline and subtitle strings from projected seat counts.

    Separated from _build_headline to keep each function under 30 lines.
    """
    seat_diff = dem_projected - gop_projected

    if abs(seat_diff) <= 2:
        subtitle_parts = [f"{n_tossup} tossup" if n_tossup == 1 else f"{n_tossup} tossups"]
        if n_competitive > n_tossup:
            subtitle_parts.append(f"{n_competitive - n_tossup} more lean races")
        return (
            "Senate Control on a Knife's Edge",
            f"{' · '.join(subtitle_parts)} in play",
        )

    # Subtitle shows the competitive breakdown, not a projected total.
    tossup_label = f"{n_tossup} tossup" if n_tossup == 1 else f"{n_tossup} tossups"
    competitive_subtitle = f"{n_competitive} competitive races · {tossup_label}"

    if gop_projected > dem_projected:
        if gop_projected >= 55:
            return "Republicans Strongly Favored to Hold the Senate", competitive_subtitle
        return "Republicans Favored to Hold the Senate", competitive_subtitle
    if dem_projected >= 55:
        return "Democrats Strongly Favored to Flip the Senate", competitive_subtitle
    return "Democrats Favored to Flip the Senate", competitive_subtitle


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


def build_zone_counts(races: list[dict]) -> dict[str, int]:
    """Compute seats in each of the 7 narrative buckets.

    The Class II contested seats come from the races list (all 33).
    Holdover (not-up) seats come from _DEM_HOLDOVER_SEATS / _GOP_HOLDOVER_SEATS.
    """
    return {
        "not_up_d": _DEM_HOLDOVER_SEATS,
        "safe_up_d": sum(1 for r in races if r["zone"] == "safe_up_d"),
        "contested_d": sum(1 for r in races if r["zone"] == "contested_d"),
        "tossup": sum(1 for r in races if r["zone"] == "tossup"),
        "contested_r": sum(1 for r in races if r["zone"] == "contested_r"),
        "safe_up_r": sum(1 for r in races if r["zone"] == "safe_up_r"),
        "not_up_r": _GOP_HOLDOVER_SEATS,
    }


def classify_race(
    st: str,
    pred_by_race: dict[str, tuple[str, float]] | None = None,
) -> dict:
    """Classify a single Senate race into rating/zone/margin.

    If pred_by_race is provided and contains the race, uses the model prediction.
    Otherwise falls back to incumbent party as a safe hold.

    Returns a dict with state, race, slug, rating, margin, n_polls, zone.
    """
    race = f"2026 {st} Senate"
    slug = race.lower().replace(" ", "-")
    incumbent_party = _CLASS_II_INCUMBENT.get(st, "R")

    if pred_by_race and race in pred_by_race:
        _, state_pred = pred_by_race[race]
        margin = state_pred - 0.5
        rating = _margin_to_rating(margin)
    else:
        margin = _DEFAULT_SAFE_MARGIN if incumbent_party == "D" else -_DEFAULT_SAFE_MARGIN
        rating = "safe_d" if incumbent_party == "D" else "safe_r"

    zone = _rating_to_zone(rating, incumbent_party)
    return {
        "state": st,
        "race": race,
        "slug": slug,
        "rating": rating,
        "margin": round(margin, 4),
        "n_polls": 0,
        "zone": zone,
    }


def build_state_colors(races: list[dict]) -> dict[str, str]:
    """Build state_colors map: every state gets a hex color for the map.

    Contested states use rating-based color. Non-contested use delegation party color.
    """
    race_by_state = {r["state"]: r for r in races}
    state_colors = {}
    for st, delegation in SENATE_DELEGATION.items():
        if st in race_by_state:
            state_colors[st] = _RATING_COLORS.get(
                race_by_state[st]["rating"], _UNCONTESTED_FALLBACK_COLOR,
            )
        else:
            state_colors[st] = _PARTY_COLORS.get(delegation, _PARTY_UNKNOWN_COLOR)
    return state_colors
