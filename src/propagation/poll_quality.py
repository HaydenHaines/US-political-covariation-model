"""Pollster quality scoring: Silver Bulletin ratings and 538 grade fallback.

Assigns a quality multiplier to each poll based on pollster reputation.
Silver Bulletin ratings (XLSX) are the primary source; 538 numeric grades
embedded in poll notes are the fallback when the XLSX is unavailable.

Quality multipliers are applied multiplicatively to effective sample size,
so a 1.2x multiplier makes a poll count as 20% larger and a 0.3x multiplier
makes it count as 70% smaller.
"""

from __future__ import annotations

import logging
from copy import copy

from src.propagation.propagate_polls import PollObservation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silver Bulletin integration
# ---------------------------------------------------------------------------

# Multiplier range for Silver Bulletin quality scores [0, 1] -> [min, max]
# Score 0.0 (banned) -> _SB_MIN_MULTIPLIER; score 1.0 (A+) -> _SB_MAX_MULTIPLIER
_SB_MIN_MULTIPLIER: float = 0.3
_SB_MAX_MULTIPLIER: float = 1.2

# Cached flag: None = not yet checked; True = SB available; False = not available
_SB_AVAILABLE: bool | None = None


def _sb_score_to_multiplier(score: float) -> float:
    """Linearly rescale a Silver Bulletin quality score [0, 1] to a multiplier."""
    return _SB_MIN_MULTIPLIER + score * (_SB_MAX_MULTIPLIER - _SB_MIN_MULTIPLIER)


def _get_sb_quality(pollster_name: str) -> float | None:
    """Return Silver Bulletin quality multiplier for *pollster_name*, or None if unavailable.

    Returns None when the Silver Bulletin XLSX is not present (so caller
    can fall back to 538 grades).  The first successful load caches a
    ``_SB_AVAILABLE=True`` flag; the first failure caches ``False`` to
    avoid repeated FileNotFoundError checks.
    """
    global _SB_AVAILABLE
    if _SB_AVAILABLE is False:
        return None

    try:
        from src.assembly.silver_bulletin_ratings import get_pollster_quality
        score = get_pollster_quality(pollster_name)
        _SB_AVAILABLE = True
        return _sb_score_to_multiplier(score)
    except FileNotFoundError:
        log.debug(
            "Silver Bulletin XLSX not found; falling back to 538 grade-based weighting"
        )
        _SB_AVAILABLE = False
        return None
    except Exception as exc:  # pragma: no cover
        log.warning("Silver Bulletin lookup failed (%s); using 538 grades", exc)
        _SB_AVAILABLE = False
        return None


def reset_sb_cache() -> None:
    """Reset all Silver Bulletin and house effect caches (for testing).

    Resets the availability flag AND the house effect caches so that tests
    calling this function start from a completely clean state — no stale
    house effect data from a previous test run.
    """
    global _SB_AVAILABLE
    _SB_AVAILABLE = None
    # Import here to avoid circular imports at module level — house_effects
    # is a sibling module that manages its own cache.
    from src.propagation.house_effects import reset_house_effect_cache
    reset_house_effect_cache()


# ---------------------------------------------------------------------------
# 538-grade fallback tables
# ---------------------------------------------------------------------------

# 538 numeric_grade -> quality multiplier
# Higher numeric grade = better pollster
# Scale: 3.0 = A+, ~2.5 = A, ~2.0 = A/B, ~1.5 = B, ~1.0 = B/C, ~0.5 = C, <0.5 = D
_DEFAULT_GRADE_MULTIPLIERS: dict[str, float] = {
    "A+": 1.2,
    "A": 1.1,
    "A/B": 1.0,
    "B": 0.9,
    "B/C": 0.8,
    "C": 0.7,
    "C/D": 0.5,
    "D": 0.3,
}

# No grade -> default multiplier
_NO_GRADE_MULTIPLIER = 0.8


def _numeric_grade_to_letter(grade_val: float) -> str:
    """Convert 538 numeric grade (0-3 scale) to letter grade."""
    if grade_val >= 2.8:
        return "A+"
    elif grade_val >= 2.4:
        return "A"
    elif grade_val >= 2.0:
        return "A/B"
    elif grade_val >= 1.5:
        return "B"
    elif grade_val >= 1.0:
        return "B/C"
    elif grade_val >= 0.5:
        return "C"
    elif grade_val >= 0.3:
        return "C/D"
    else:
        return "D"


def extract_grade_from_notes(notes: str) -> str | None:
    """Extract grade value from notes field (format: '...;grade=2.5;...').

    Returns letter grade string (e.g. 'A', 'B/C') or None if not found.
    """
    if not notes:
        return None
    for part in notes.split(";"):
        part = part.strip()
        if part.startswith("grade="):
            try:
                val = float(part[6:])
                return _numeric_grade_to_letter(val)
            except (ValueError, IndexError):
                return None
    return None


def grade_to_multiplier(
    grade: str | None,
    grade_multipliers: dict[str, float] | None = None,
) -> float:
    """Convert a letter grade to a quality multiplier."""
    if grade is None:
        return _NO_GRADE_MULTIPLIER
    table = grade_multipliers or _DEFAULT_GRADE_MULTIPLIERS
    return table.get(grade, _NO_GRADE_MULTIPLIER)


def apply_pollster_quality(
    polls: list[PollObservation],
    poll_notes: list[str] | None = None,
    grade_multipliers: dict[str, float] | None = None,
    use_silver_bulletin: bool = True,
) -> list[PollObservation]:
    """Adjust effective sample sizes by pollster quality.

    Quality source priority:
      1. Silver Bulletin ratings (when XLSX is present and ``use_silver_bulletin``
         is True): ``get_pollster_quality(poll.pollster)`` returns a [0, 1]
         score, which is linearly rescaled to [0.3, 1.2].
      2. 538 numeric grade from poll_notes (format: "...;grade=2.5;...").
         Applied when Silver Bulletin is unavailable or disabled.

    If poll_notes is None and Silver Bulletin is unavailable, all polls get
    the no-grade default (0.8x).

    Returns new PollObservation copies with adjusted n_sample.
    """
    result: list[PollObservation] = []

    for i, poll in enumerate(polls):
        multiplier: float | None = None

        # --- Priority 1: Silver Bulletin ---
        if use_silver_bulletin and poll.pollster:
            multiplier = _get_sb_quality(poll.pollster)

        # --- Priority 2: 538 grade from notes ---
        if multiplier is None:
            notes = poll_notes[i] if poll_notes and i < len(poll_notes) else ""
            grade = extract_grade_from_notes(notes)
            multiplier = grade_to_multiplier(grade, grade_multipliers)

        n_effective = int(max(1, round(poll.n_sample * multiplier)))
        new_poll = copy(poll)
        new_poll.n_sample = n_effective
        result.append(new_poll)

    return result
