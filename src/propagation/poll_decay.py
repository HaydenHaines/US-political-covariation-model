"""Time decay and pre-primary discount for poll weighting.

Adjusts effective sample sizes based on:
  - Recency: exponential decay with configurable half-life
  - Pre/post-primary status: polls before the primary are discounted
    because the candidate matchup may not be finalized

Both adjustments are multiplicative on n_sample.
"""

from __future__ import annotations

import csv
import logging
from copy import copy
from datetime import date
from pathlib import Path

from src.propagation.propagate_polls import PollObservation

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]


def _parse_date(s: str) -> date:
    """Parse YYYY-MM-DD date string."""
    parts = s.strip().split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


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


# ---------------------------------------------------------------------------
# Pre/post-primary discount
# ---------------------------------------------------------------------------

# Default weight applied multiplicatively to polls conducted before the primary.
# Post-primary polls are unaffected (factor = 1.0).
_PRE_PRIMARY_DISCOUNT: float = 0.5

# Expected race suffix patterns in the poll `race` field, keyed by race_type.
# A poll's race field contains values like "2026 FL Senate" or "2026 OH Governor".
_RACE_TYPE_ALIASES: dict[str, str] = {
    "Senate": "Senate",
    "Governor": "Governor",
}


def _parse_primary_calendar(path: Path) -> dict[tuple[str, str], date]:
    """Load primary_calendar CSV into a lookup dict keyed by (state, race_type).

    CSV columns: state, race_type, primary_date (YYYY-MM-DD).
    Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        log.debug("Primary calendar not found at %s; no pre-primary discounting", path)
        return {}

    calendar: dict[tuple[str, str], date] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row.get("state", "").strip()
            race_type = row.get("race_type", "").strip()
            raw_date = row.get("primary_date", "").strip()
            if state and race_type and raw_date:
                calendar[(state, race_type)] = _parse_date(raw_date)
    return calendar


def _extract_state_and_race_type(race: str) -> tuple[str | None, str | None]:
    """Extract (state_abbr, race_type) from a race label like "2026 FL Senate".

    Returns (None, None) if the race string cannot be parsed.
    """
    if not race:
        return None, None
    parts = race.strip().split()
    # Expected pattern: "<year> <STATE> <RaceType>" e.g. "2026 FL Senate"
    if len(parts) < 3:
        return None, None
    state = parts[1].upper()
    race_type = parts[2].capitalize()
    if race_type not in _RACE_TYPE_ALIASES:
        return None, None
    return state, _RACE_TYPE_ALIASES[race_type]


def apply_primary_discount(
    polls: list[PollObservation],
    primary_calendar_path: Path | str | None = None,
    discount_factor: float = _PRE_PRIMARY_DISCOUNT,
) -> list[PollObservation]:
    """Apply a multiplicative discount to polls conducted before the primary.

    Pre-primary polls are less informative about general election outcomes
    because the candidate matchup may not be finalized and respondent attention
    to the general race is lower.  This function scales the effective sample
    size of pre-primary polls by ``discount_factor`` (default 0.5), leaving
    post-primary polls unchanged.

    Parameters
    ----------
    polls:
        List of PollObservation objects.  Each must have ``race`` (e.g.
        "2026 FL Senate") and ``date`` (YYYY-MM-DD) populated for the
        discount to apply.
    primary_calendar_path:
        Path to the primary calendar CSV.  Defaults to
        ``data/polls/primary_calendar_2026.csv`` relative to the project root.
        Polls without a matching calendar entry are unchanged.
    discount_factor:
        Multiplicative factor applied to ``n_sample`` for pre-primary polls.
        Must be in (0, 1].  Default 0.5.

    Returns
    -------
    list[PollObservation]
        New PollObservation copies with pre-primary polls' n_sample discounted.
    """
    if not 0 < discount_factor <= 1.0:
        raise ValueError(f"discount_factor must be in (0, 1]; got {discount_factor}")

    if primary_calendar_path is None:
        primary_calendar_path = PROJECT_ROOT / "data" / "polls" / "primary_calendar_2026.csv"
    calendar = _parse_primary_calendar(Path(primary_calendar_path))

    result: list[PollObservation] = []
    for poll in polls:
        new_poll = copy(poll)

        if not poll.date or not poll.race:
            result.append(new_poll)
            continue

        state, race_type = _extract_state_and_race_type(poll.race)
        if state is None or race_type is None:
            result.append(new_poll)
            continue

        primary_date = calendar.get((state, race_type))
        if primary_date is None:
            result.append(new_poll)
            continue

        poll_date = _parse_date(poll.date)
        if poll_date < primary_date:
            discounted_n = int(max(1, round(poll.n_sample * discount_factor)))
            log.debug(
                "Pre-primary discount: %s poll on %s -> n_sample %d -> %d (factor=%.2f)",
                poll.race,
                poll.date,
                poll.n_sample,
                discounted_n,
                discount_factor,
            )
            new_poll.n_sample = discounted_n

        result.append(new_poll)

    return result


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
