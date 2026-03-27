"""Poll weighting: time decay, pollster quality, house effect correction, and aggregation.

Adjusts effective sample sizes of PollObservation objects based on:
  - Recency (exponential time decay)
  - Pollster quality (grade-based multiplier)

Applies house effect (partisan bias) correction to dem_share before weighting:
  - Corrects for systematic over/under-estimation of Democrats by each pollster
  - Silver Bulletin house effects are the primary source; 538 bias_ppm is fallback

Then aggregates multiple weighted polls into a single effective poll via
inverse-variance weighting for downstream Bayesian update.

Pollster quality source priority:
  1. Silver Bulletin ratings (``get_pollster_quality``) when the XLSX is present.
     Returns a score in [0, 1] which is rescaled to a multiplier range of
     [_SB_MIN_MULTIPLIER, _SB_MAX_MULTIPLIER] (default 0.3–1.2).
  2. Fall back to 538 numeric grade embedded in poll notes (``grade=2.5`` key)
     when Silver Bulletin XLSX is not downloaded.

House effect correction source priority:
  1. Silver Bulletin "House Effect" column (mean-reverted, percentage points).
  2. 538 ``bias_ppm`` column (same sign convention).
  3. 0.0 (no correction) for unknown pollsters.

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
# House effect correction
# ---------------------------------------------------------------------------

# Clamp corrected dem_share to this range so it stays a valid proportion.
_HE_DEM_SHARE_MIN: float = 0.01
_HE_DEM_SHARE_MAX: float = 0.99

# Cached house effect lookups: None = not yet loaded.
_SB_HOUSE_EFFECTS: dict[str, float] | None = None
_538_BIAS: dict[str, float] | None = None


def _ensure_house_effect_caches() -> None:
    """Populate module-level house effect caches on first call."""
    global _SB_HOUSE_EFFECTS, _538_BIAS

    if _SB_HOUSE_EFFECTS is None:
        try:
            from src.assembly.silver_bulletin_ratings import load_pollster_house_effects
            _SB_HOUSE_EFFECTS = load_pollster_house_effects()
            log.debug("Loaded %d Silver Bulletin house effects", len(_SB_HOUSE_EFFECTS))
        except FileNotFoundError:
            log.debug("Silver Bulletin XLSX not found; SB house effects unavailable")
            _SB_HOUSE_EFFECTS = {}
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load Silver Bulletin house effects: %s", exc)
            _SB_HOUSE_EFFECTS = {}

    if _538_BIAS is None:
        try:
            from src.assembly.silver_bulletin_ratings import load_538_bias
            _538_BIAS = load_538_bias()
            log.debug("Loaded %d 538 bias estimates", len(_538_BIAS))
        except FileNotFoundError:
            log.debug("538 pollster-ratings CSV not found; 538 bias unavailable")
            _538_BIAS = {}
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load 538 bias estimates: %s", exc)
            _538_BIAS = {}


def _lookup_house_effect(
    pollster_name: str,
    sb_house_effects: dict[str, float],
    bias_538: dict[str, float],
) -> tuple[float, str]:
    """Return (house_effect_pp, source) for a pollster using priority order.

    Priority: Silver Bulletin exact → Silver Bulletin fuzzy → 538 exact → 0.0.
    Returns the source string for logging ('sb_exact', 'sb_fuzzy', '538', 'none').
    """
    # 1. Silver Bulletin — delegate fuzzy matching to the silver_bulletin module
    if sb_house_effects:
        try:
            from src.assembly.silver_bulletin_ratings import _normalize, _name_similarity
        except ImportError:  # pragma: no cover
            pass
        else:
            # Exact match
            if pollster_name in sb_house_effects:
                return sb_house_effects[pollster_name], "sb_exact"

            # Normalised / fuzzy match
            norm_query = _normalize(pollster_name)
            norm_sb = {_normalize(k): v for k, v in sb_house_effects.items()}
            if norm_query in norm_sb:
                return norm_sb[norm_query], "sb_exact"

            # Best Jaccard fuzzy match (threshold 0.4 matches get_pollster_quality)
            _FUZZY_THRESHOLD = 0.4
            best_val: float | None = None
            best_sim = 0.0
            for norm_key, val in norm_sb.items():
                sim = _name_similarity(norm_query, norm_key)
                if sim > best_sim:
                    best_sim = sim
                    best_val = val
            if best_sim >= _FUZZY_THRESHOLD and best_val is not None:
                return best_val, "sb_fuzzy"

    # 2. 538 bias — exact name match only (no fuzzy; 538 names differ more)
    if bias_538 and pollster_name in bias_538:
        return bias_538[pollster_name], "538"

    return 0.0, "none"


def apply_house_effect_correction(
    polls: list[PollObservation],
    sb_house_effects: dict[str, float] | None = None,
    bias_538: dict[str, float] | None = None,
) -> list[PollObservation]:
    """Correct each poll's dem_share for pollster partisan bias (house effects).

    For each poll, looks up the pollster's house effect using priority order:
      1. Silver Bulletin "House Effect" (mean-reverted, pp)
      2. 538 ``bias_ppm``
      3. 0.0 (no correction)

    Adjustment: corrected_dem_share = dem_share - house_effect / 100
      Positive house_effect means pollster overestimates Democrats, so subtracting
      moves the share toward the true value.

    Result is clamped to [_HE_DEM_SHARE_MIN, _HE_DEM_SHARE_MAX] to keep it
    a valid proportion.  The original ``dem_share`` is preserved on the returned
    PollObservation; ``dem_share`` is updated to the corrected value.

    Parameters
    ----------
    polls:
        List of PollObservation objects to correct.
    sb_house_effects:
        Optional pre-loaded Silver Bulletin house effects dict.  If None,
        loads from the module-level cache (populated from XLSX on first call).
    bias_538:
        Optional pre-loaded 538 bias dict.  If None, loads from the module-level
        cache (populated from CSV on first call).

    Returns
    -------
    list[PollObservation]
        New PollObservation copies with dem_share corrected for house effects.
        The uncorrected value is accessible via poll.raw_dem_share if needed
        (the field is not currently on PollObservation; original logged here).
    """
    _ensure_house_effect_caches()
    he_dict = sb_house_effects if sb_house_effects is not None else (_SB_HOUSE_EFFECTS or {})
    b538_dict = bias_538 if bias_538 is not None else (_538_BIAS or {})

    result: list[PollObservation] = []
    for poll in polls:
        new_poll = copy(poll)
        if not poll.pollster:
            result.append(new_poll)
            continue

        house_effect_pp, source = _lookup_house_effect(poll.pollster, he_dict, b538_dict)

        if source == "none":
            log.debug("House effect: pollster %r not found, no correction applied", poll.pollster)
        else:
            log.debug(
                "House effect: pollster %r → %.3f pp (%s); dem_share %.4f → %.4f",
                poll.pollster,
                house_effect_pp,
                source,
                poll.dem_share,
                poll.dem_share - house_effect_pp / 100.0,
            )

        correction = house_effect_pp / 100.0
        corrected = poll.dem_share - correction
        corrected = max(_HE_DEM_SHARE_MIN, min(_HE_DEM_SHARE_MAX, corrected))
        new_poll.dem_share = corrected
        result.append(new_poll)

    return result


def reset_house_effect_cache() -> None:
    """Reset house effect caches (useful in tests that inject custom data)."""
    global _SB_HOUSE_EFFECTS, _538_BIAS
    _SB_HOUSE_EFFECTS = None
    _538_BIAS = None


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
                "Pre-primary discount: %s poll on %s → n_sample %d → %d (factor=%.2f)",
                poll.race,
                poll.date,
                poll.n_sample,
                discounted_n,
                discount_factor,
            )
            new_poll.n_sample = discounted_n

        result.append(new_poll)

    return result


def apply_all_weights(
    polls: list[PollObservation],
    reference_date: str,
    half_life_days: float = 30.0,
    poll_notes: list[str] | None = None,
    apply_quality: bool = True,
    use_silver_bulletin: bool = True,
    apply_house_effects: bool = True,
    use_primary_discount: bool = True,
    primary_calendar_path: Path | str | None = None,
    primary_discount_factor: float = _PRE_PRIMARY_DISCOUNT,
) -> list[PollObservation]:
    """Apply all weighting steps to a list of polls.

    Processing order (matches the intended inference pipeline):
      1. House effect correction (adjusts dem_share for partisan bias) — applied first
         so that downstream weighting operates on bias-corrected shares.
      2. Pre/post-primary discount (scales n_sample for pre-primary polls).
      3. Time decay (reduces effective N for older polls).
      4. Pollster quality (rescales effective N by quality grade).

    House effect correction is skipped when ``apply_house_effects`` is False.
    Pre-primary discounting is skipped when ``use_primary_discount`` is False.
    Pollster quality is skipped when ``apply_quality`` is False.

    When Silver Bulletin XLSX is present and ``use_silver_bulletin`` is True,
    pollster quality uses Silver Bulletin ratings (priority 1).  Otherwise
    falls back to 538 grade embedded in poll_notes (priority 2).
    """
    working = list(polls)
    if apply_house_effects:
        working = apply_house_effect_correction(working)
    if use_primary_discount:
        working = apply_primary_discount(
            working,
            primary_calendar_path=primary_calendar_path,
            discount_factor=primary_discount_factor,
        )
    working = apply_time_decay(working, reference_date, half_life_days)
    if apply_quality:
        working = apply_pollster_quality(
            working, poll_notes, use_silver_bulletin=use_silver_bulletin
        )
    return working


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
