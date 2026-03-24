"""Poll weighting: time decay, pollster quality, and multi-poll aggregation.

Adjusts effective sample sizes of PollObservation objects based on:
  - Recency (exponential time decay)
  - Pollster quality (grade-based multiplier)

Then aggregates multiple weighted polls into a single effective poll via
inverse-variance weighting for downstream Bayesian update.

Pollster quality source priority:
  1. Silver Bulletin ratings (``get_pollster_quality``) when the XLSX is present.
     Returns a score in [0, 1] which is rescaled to a multiplier range of
     [_SB_MIN_MULTIPLIER, _SB_MAX_MULTIPLIER] (default 0.3–1.2).
  2. Fall back to 538 numeric grade embedded in poll notes (``grade=2.5`` key)
     when Silver Bulletin XLSX is not downloaded.

Usage:
  from src.propagation.poll_weighting import apply_all_weights, aggregate_polls

  weighted = apply_all_weights(polls, notes, reference_date="2020-11-03")
  combined_share, combined_n = aggregate_polls(weighted)
"""

from __future__ import annotations

import csv
import logging
import math
from copy import copy
from datetime import date, timedelta
from pathlib import Path

from src.propagation.propagate_polls import PollObservation

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]

# ---------------------------------------------------------------------------
# Silver Bulletin integration
# ---------------------------------------------------------------------------

# Multiplier range for Silver Bulletin quality scores [0, 1] -> [min, max]
# Score 0.0 (banned) → _SB_MIN_MULTIPLIER; score 1.0 (A+) → _SB_MAX_MULTIPLIER
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
    """Reset the Silver Bulletin availability flag (useful in tests)."""
    global _SB_AVAILABLE
    _SB_AVAILABLE = None


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


def _parse_date(s: str) -> date:
    """Parse YYYY-MM-DD date string."""
    parts = s.strip().split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


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


# ---------------------------------------------------------------------------
# Core weighting functions
# ---------------------------------------------------------------------------


def apply_time_decay(
    polls: list[PollObservation],
    reference_date: str,
    half_life_days: float = 30.0,
) -> list[PollObservation]:
    """Adjust effective sample sizes by exponential time decay.

    decay = 2^(-age_days / half_life_days)
    n_effective = int(max(1, round(poll.n_sample * decay)))

    Returns new PollObservation copies with adjusted n_sample.
    reference_date is typically election day or "today".
    """
    ref = _parse_date(reference_date)
    result: list[PollObservation] = []

    for poll in polls:
        if not poll.date:
            # No date -> no decay, keep as-is
            result.append(copy(poll))
            continue

        poll_date = _parse_date(poll.date)
        age_days = (ref - poll_date).days
        if age_days < 0:
            age_days = 0  # Future polls get no decay

        decay = 2.0 ** (-age_days / half_life_days)
        n_effective = int(max(1, round(poll.n_sample * decay)))

        new_poll = copy(poll)
        new_poll.n_sample = n_effective
        result.append(new_poll)

    return result


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


def apply_all_weights(
    polls: list[PollObservation],
    reference_date: str,
    half_life_days: float = 30.0,
    poll_notes: list[str] | None = None,
    apply_quality: bool = True,
    use_silver_bulletin: bool = True,
) -> list[PollObservation]:
    """Apply both time decay and pollster quality weighting.

    Time decay is always applied. Pollster quality is applied only if
    apply_quality is True.

    When Silver Bulletin XLSX is present and ``use_silver_bulletin`` is True,
    pollster quality uses Silver Bulletin ratings (priority 1).  Otherwise
    falls back to 538 grade embedded in poll_notes (priority 2).
    """
    weighted = apply_time_decay(polls, reference_date, half_life_days)
    if apply_quality:
        weighted = apply_pollster_quality(
            weighted, poll_notes, use_silver_bulletin=use_silver_bulletin
        )
    return weighted


# ---------------------------------------------------------------------------
# Multi-poll aggregation
# ---------------------------------------------------------------------------


def aggregate_polls(polls: list[PollObservation]) -> tuple[float, int]:
    """Combine multiple polls into a single effective poll via inverse-variance weighting.

    Each poll's variance is p*(1-p)/n. Inverse-variance weighting gives
    the minimum-variance unbiased estimate of the underlying share.

    Returns (combined_dem_share, combined_effective_n).

    Raises ValueError if polls is empty.
    """
    if not polls:
        raise ValueError("No polls to aggregate")

    # Guard against edge cases where dem_share is exactly 0 or 1
    variances = []
    for p in polls:
        ds = max(0.001, min(0.999, p.dem_share))
        variances.append(ds * (1 - ds) / p.n_sample)

    inv_vars = [1.0 / v for v in variances]
    total_inv_var = sum(inv_vars)

    combined_share = sum(iv * p.dem_share for iv, p in zip(inv_vars, polls)) / total_inv_var
    combined_var = 1.0 / total_inv_var
    # Back out effective N from combined variance: var = p*(1-p)/n => n = p*(1-p)/var
    cs = max(0.001, min(0.999, combined_share))
    combined_n = int(max(1, round(cs * (1 - cs) / combined_var)))

    return combined_share, combined_n


# ---------------------------------------------------------------------------
# CSV notes loader (parallel to load_polls)
# ---------------------------------------------------------------------------


def load_poll_notes(cycle: str) -> list[str]:
    """Load the notes column from polls_{cycle}.csv.

    Returns a list of notes strings in the same order as the CSV rows
    (after header). This parallels the output of load_polls() when called
    without filters.
    """
    path = PROJECT_ROOT / "data" / "polls" / f"polls_{cycle}.csv"
    if not path.exists():
        return []

    notes: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes.append(row.get("notes", ""))
    return notes


def load_polls_with_notes(
    cycle: str,
    race: str | None = None,
    geography: str | None = None,
) -> tuple[list[PollObservation], list[str]]:
    """Load polls and their notes in parallel, applying the same filters.

    Returns (polls, notes) lists of the same length.
    """
    path = PROJECT_ROOT / "data" / "polls" / f"polls_{cycle}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Poll CSV not found: {path}")

    polls: list[PollObservation] = []
    notes_list: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_dem = row.get("dem_share", "").strip()
            raw_n = row.get("n_sample", "").strip()
            if not raw_dem or not raw_n:
                continue
            try:
                dem_share = float(raw_dem)
                n_sample = int(float(raw_n))
            except ValueError:
                continue
            if not (0.0 < dem_share < 1.0) or n_sample <= 0:
                continue

            row_race = row.get("race", "").strip()
            row_geo = row.get("geography", "").strip()
            geo_level = row.get("geo_level", "state").strip() or "state"
            row_date = row.get("date", "").strip()
            pollster = row.get("pollster", "").strip()
            row_notes = row.get("notes", "").strip()

            # Apply filters
            if race is not None and race.lower() not in row_race.lower():
                continue
            if geography is not None and row_geo != geography:
                continue

            polls.append(PollObservation(
                geography=row_geo,
                dem_share=dem_share,
                n_sample=n_sample,
                race=row_race,
                date=row_date,
                pollster=pollster,
                geo_level=geo_level,
            ))
            notes_list.append(row_notes)

    # Sort by date (keep notes aligned)
    if polls:
        pairs = sorted(zip(polls, notes_list), key=lambda x: x[0].date)
        polls = [p for p, _ in pairs]
        notes_list = [n for _, n in pairs]

    return polls, notes_list


# ---------------------------------------------------------------------------
# Election day lookup
# ---------------------------------------------------------------------------

_ELECTION_DAYS: dict[str, str] = {
    "2016": "2016-11-08",
    "2018": "2018-11-06",
    "2020": "2020-11-03",
    "2022": "2022-11-08",
    "2024": "2024-11-05",
    "2026": "2026-11-03",
}


def election_day_for_cycle(cycle: str) -> str:
    """Return the election day date string for a given cycle."""
    return _ELECTION_DAYS.get(cycle, f"{cycle}-11-03")
