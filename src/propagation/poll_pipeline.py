"""Poll weighting orchestration: combined pipeline, aggregation, and CSV loading.

Ties together the individual weighting steps (house effects, time decay,
pollster quality, primary discount) into a single ``apply_all_weights``
pipeline, and provides inverse-variance aggregation and CSV poll loading.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.propagation.house_effects import apply_house_effect_correction
from src.propagation.poll_decay import (
    _PRE_PRIMARY_DISCOUNT,
    apply_primary_discount,
    apply_time_decay,
)
from src.propagation.poll_quality import apply_pollster_quality
from src.propagation.propagate_polls import PollObservation

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]


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
