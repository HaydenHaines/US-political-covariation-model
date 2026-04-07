"""Shared helpers and constants for Governor forecast endpoints.

Contains state lists, incumbent data, and race classification logic.
Governor races are independent — there is no chamber control concept.
"""
from __future__ import annotations

# All 36 states with gubernatorial races in 2026.
GOVERNOR_2026_STATES = {
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "FL", "GA", "HI",
    "IA", "ID", "IL", "KS", "MA", "MD", "ME", "MI", "MN", "NE",
    "NH", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "VT", "WI", "WY",
}

# Which party currently holds each governorship.
# "D" = Democrat, "R" = Republican.
_GOVERNOR_INCUMBENT: dict[str, str] = {
    "AK": "R",  # Mike Dunleavy (R)
    "AL": "R",  # Kay Ivey (R)
    "AR": "R",  # Sarah Huckabee Sanders (R)
    "AZ": "D",  # Katie Hobbs (D)
    "CA": "D",  # Gavin Newsom (D)
    "CO": "D",  # Jared Polis (D)
    "CT": "D",  # Ned Lamont (D)
    "FL": "R",  # Ron DeSantis (R) — term-limited, open seat
    "GA": "R",  # Brian Kemp (R) — term-limited, open seat
    "HI": "D",  # Josh Green (D)
    "IA": "R",  # Kim Reynolds (R)
    "ID": "R",  # Brad Little (R)
    "IL": "D",  # JB Pritzker (D)
    "KS": "D",  # Laura Kelly (D) — term-limited, open seat
    "MA": "D",  # Maura Healey (D)
    "MD": "D",  # Wes Moore (D)
    "ME": "D",  # Janet Mills (D) — term-limited, open seat
    "MI": "D",  # Gretchen Whitmer (D)
    "MN": "D",  # Tim Walz (D) — vacated for VP run; currently Peggy Flanagan (D)
    "NE": "R",  # Jim Pillen (R)
    "NH": "R",  # Kelly Ayotte (R)
    "NM": "D",  # Michelle Lujan Grisham (D)
    "NV": "D",  # Joe Lombardo (R) — NOTE: actually R; corrected below
    "NY": "D",  # Kathy Hochul (D)
    "OH": "R",  # Mike DeWine (R) — term-limited, open seat
    "OK": "R",  # Kevin Stitt (R) — term-limited, open seat
    "OR": "D",  # Tina Kotek (D)
    "PA": "D",  # Josh Shapiro (D)
    "RI": "D",  # Dan McKee → current: Dan McKee (D)
    "SC": "R",  # Henry McMaster (R)
    "SD": "R",  # Kristi Noem (R) — now DHS Secretary; Dennis Daugaard acting
    "TN": "R",  # Bill Lee (R) — term-limited, open seat
    "TX": "R",  # Greg Abbott (R)
    "VT": "R",  # Phil Scott (R)
    "WI": "D",  # Tony Evers (D)
    "WY": "R",  # Mark Gordon (R) — term-limited, open seat
}

# NV is held by Republican Joe Lombardo — correct the dict entry above.
_GOVERNOR_INCUMBENT["NV"] = "R"

# Default margin used when no model prediction is available.
# Sign: positive = safe D, negative = safe R.
_DEFAULT_SAFE_MARGIN = 0.25


def classify_governor_race(
    st: str,
    pred_by_race: dict[str, tuple[str, float]] | None = None,
) -> dict:
    """Classify a single governor race into rating/margin.

    If pred_by_race is provided and contains the race, uses the model prediction.
    Otherwise falls back to incumbent party as a safe hold.

    Returns a dict with state, race, slug, rating, margin, incumbent_party, n_polls.
    """
    from api.ratings import margin_to_rating

    race = f"2026 {st} Governor"
    # Slug matches the pattern used by race detail pages: "2026-fl-governor"
    slug = race.lower().replace(" ", "-")
    incumbent_party = _GOVERNOR_INCUMBENT.get(st, "R")

    if pred_by_race and race in pred_by_race:
        _, state_pred = pred_by_race[race]
        margin = state_pred - 0.5
        rating = margin_to_rating(margin)
    else:
        # No model prediction — treat as safe hold for the incumbent party
        margin = _DEFAULT_SAFE_MARGIN if incumbent_party == "D" else -_DEFAULT_SAFE_MARGIN
        rating = "safe_d" if incumbent_party == "D" else "safe_r"

    return {
        "state": st,
        "race": race,
        "slug": slug,
        "rating": rating,
        "margin": round(margin, 4),
        "incumbent_party": incumbent_party,
        "n_polls": 0,
    }


def rating_sort_key(rating: str) -> int:
    """Sort races by competitiveness: tossup first, then lean, likely, safe."""
    return {
        "safe_d": 0, "likely_d": 1, "lean_d": 2,
        "tossup": 3,
        "lean_r": 4, "likely_r": 5, "safe_r": 6,
    }.get(rating, 3)
