"""Generic ballot adjustment for midterm forecasts.

The county-level priors (ridge_county_priors.parquet) are trained on 2024
presidential Dem share.  In a midterm year the national environment typically
differs from the prior presidential baseline — especially when the in-party has
performed unusually well or poorly.

This module computes a ``national_gb_shift``:

    national_gb_shift = generic_ballot_avg - PRES_DEM_SHARE_2024_NATIONAL

Applying this flat shift to all county priors before the race-specific Bayesian
update moves the entire baseline toward the current national environment without
distorting the relative differences between counties.

The adjustment is **additive** and **applied to county priors only** — it does
not affect the Bayesian update machinery (type covariance, poll weighting, etc.).
After the shift the priors are clipped to [0.01, 0.99] to prevent unphysical values.

Typical usage (called from predict_race or the forecast API):

    from src.prediction.generic_ballot import compute_gb_shift, apply_gb_shift

    gb_info = compute_gb_shift(polls_path)
    shifted_priors = apply_gb_shift(county_priors, gb_info.shift)
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2024 national presidential Dem two-party share (Dem / (Dem + Rep) votes).
# Source: Associated Press final certified results; 74,223,975 Dem / 155,480,149 total.
PRES_DEM_SHARE_2024_NATIONAL: float = 0.4841

# Race label and geo_level used to identify generic ballot rows in the polls CSV.
_GB_RACE_LABEL: str = "2026 Generic Ballot"
_GB_GEO_LEVEL: str = "national"

# Clamp adjusted priors to this range so they remain valid probabilities.
_PRIOR_MIN: float = 0.01
_PRIOR_MAX: float = 0.99

# Default path to the YouGov weekly generic ballot crosstab JSON file.
_YOUGOV_GB_JSON_PATH: Path = PROJECT_ROOT / "data" / "polls" / "yougov_generic_ballot_2026.json"


@dataclass(frozen=True)
class GenericBallotInfo:
    """Result of a generic ballot calculation.

    Attributes
    ----------
    gb_avg:
        Weighted average of generic ballot polls (Dem two-party share).
    pres_baseline:
        2024 presidential national Dem share used as the reference point.
    shift:
        gb_avg - pres_baseline.  Positive = Dems doing better than 2024 pres.
    n_polls:
        Number of CSV-sourced generic ballot polls used.
    n_yougov_polls:
        Number of YouGov weekly crosstab issues used (after deduplication).
    source:
        Human-readable description for API/display ("auto" or "manual").
    """

    gb_avg: float
    pres_baseline: float
    shift: float
    n_polls: int
    n_yougov_polls: int
    source: str


def load_generic_ballot_polls(
    polls_path: Path | str | None = None,
) -> list[tuple[float, int]]:
    """Load generic ballot polls from the cycle CSV.

    Returns a list of (dem_share, n_sample) tuples for all rows whose
    race label starts with ``_GB_RACE_LABEL`` and geo_level == ``_GB_GEO_LEVEL``.
    Returns an empty list if no matching rows are found or the file does not exist.

    Parameters
    ----------
    polls_path:
        Path to polls CSV.  Defaults to ``data/polls/polls_2026.csv`` relative
        to the project root.
    """
    if polls_path is None:
        polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    polls_path = Path(polls_path)

    if not polls_path.exists():
        log.debug("Polls file not found at %s; no generic ballot polls", polls_path)
        return []

    result: list[tuple[float, int]] = []
    with polls_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            race = row.get("race", "").strip()
            geo_level = row.get("geo_level", "").strip()
            if not race.startswith(_GB_RACE_LABEL):
                continue
            if geo_level != _GB_GEO_LEVEL:
                continue
            raw_dem = row.get("dem_share", "").strip()
            raw_n = row.get("n_sample", "").strip()
            try:
                dem_share = float(raw_dem)
                n_sample = int(float(raw_n))
            except (ValueError, TypeError):
                log.warning("Skipping malformed generic ballot row: %r", row)
                continue
            if not (0.0 < dem_share < 1.0) or n_sample <= 0:
                continue
            result.append((dem_share, n_sample))

    log.debug("Loaded %d generic ballot polls from %s", len(result), polls_path)
    return result


def load_yougov_generic_ballot_polls(
    json_path: Path | str | None = None,
) -> list[tuple[float, int, str, str]]:
    """Load YouGov weekly generic ballot issues from the crosstab JSON file.

    YouGov publishes weekly issues with demographic crosstabs.  Each issue has a
    ``gb_topline`` (Dem two-party share) and ``n_total_ballot`` (number of
    respondents who answered the ballot question).  We use both for a
    sample-size-weighted average that mirrors the CSV poll format.

    Returns a list of ``(dem_share, n_sample, date_start, date_end)`` tuples.
    The date fields enable deduplication against the CSV when the same YouGov
    poll appears in both sources (e.g., as an Economist/YouGov entry in RCP).
    Returns an empty list if the file is missing or contains no valid entries.

    Parameters
    ----------
    json_path:
        Path to the YouGov crosstab JSON.  Defaults to
        ``data/polls/yougov_generic_ballot_2026.json`` relative to the project root.
    """
    if json_path is None:
        json_path = _YOUGOV_GB_JSON_PATH
    json_path = Path(json_path)

    if not json_path.exists():
        log.debug("YouGov GB JSON not found at %s; skipping", json_path)
        return []

    try:
        with json_path.open(encoding="utf-8") as f:
            issues = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load YouGov GB JSON at %s: %s", json_path, exc)
        return []

    result: list[tuple[float, int, str, str]] = []
    for entry in issues:
        try:
            gb_topline = float(entry["gb_topline"])
            n_ballot = int(entry["n_total_ballot"])
            date_start = str(entry.get("date_start", ""))
            date_end = str(entry.get("date_end", ""))
        except (KeyError, ValueError, TypeError):
            log.warning("Skipping malformed YouGov GB entry: %r", entry)
            continue

        if not (0.0 < gb_topline < 1.0) or n_ballot <= 0:
            log.warning("Skipping out-of-range YouGov GB entry: topline=%s n=%s", gb_topline, n_ballot)
            continue

        result.append((gb_topline, n_ballot, date_start, date_end))

    log.debug("Loaded %d YouGov generic ballot issues from %s", len(result), json_path)
    return result


def _deduplicate_yougov_polls(
    csv_polls_path: Path,
    yougov_polls: list[tuple[float, int, str, str]],
) -> list[tuple[float, int]]:
    """Remove YouGov entries whose date range overlaps any row already in the CSV.

    The CSV may contain an Economist/YouGov entry for a date that also appears in
    the YouGov JSON.  We check each YouGov issue's (date_start, date_end) window
    against the dates in the CSV's generic ballot rows; if any CSV date falls
    within the window, we drop the YouGov issue to avoid double-counting.

    Parameters
    ----------
    csv_polls_path:
        Path to the polls CSV (may not exist; handled gracefully).
    yougov_polls:
        Output of ``load_yougov_generic_ballot_polls()``.

    Returns
    -------
    list[tuple[float, int]]
        YouGov polls with no date-range overlap against CSV rows, as
        ``(dem_share, n_sample)`` tuples ready for ``compute_gb_average()``.
    """
    from datetime import date

    # Collect all date strings from generic ballot CSV rows for fast lookup.
    csv_dates: set[str] = set()
    if csv_polls_path.exists():
        with csv_polls_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                race = row.get("race", "").strip()
                geo_level = row.get("geo_level", "").strip()
                if race.startswith(_GB_RACE_LABEL) and geo_level == _GB_GEO_LEVEL:
                    raw_date = row.get("date", "").strip()
                    if raw_date:
                        csv_dates.add(raw_date)

    result: list[tuple[float, int]] = []
    for dem_share, n_sample, date_start, date_end in yougov_polls:
        try:
            # Parse the YouGov window boundaries; default to open-ended if missing.
            start = date.fromisoformat(date_start) if date_start else None
            end = date.fromisoformat(date_end) if date_end else None
        except ValueError:
            # Malformed date — keep the entry; can't check overlap.
            result.append((dem_share, n_sample))
            continue

        overlaps = False
        for csv_date_str in csv_dates:
            try:
                csv_dt = date.fromisoformat(csv_date_str)
            except ValueError:
                continue
            # CSV date falls within [date_start, date_end] → overlap.
            if start is not None and end is not None and start <= csv_dt <= end:
                overlaps = True
                break

        if overlaps:
            log.debug(
                "Dropping YouGov GB issue %s–%s: date overlaps CSV entry",
                date_start,
                date_end,
            )
        else:
            result.append((dem_share, n_sample))

    return result


def compute_gb_average(polls: list[tuple[float, int]]) -> float:
    """Compute sample-size-weighted average of generic ballot polls.

    Parameters
    ----------
    polls:
        List of (dem_share, n_sample) tuples.

    Returns
    -------
    float
        Weighted average Dem share.  Returns ``PRES_DEM_SHARE_2024_NATIONAL``
        when the list is empty (i.e., shift = 0, no adjustment applied).
    """
    if not polls:
        return PRES_DEM_SHARE_2024_NATIONAL
    total_n = sum(n for _, n in polls)
    if total_n == 0:
        return PRES_DEM_SHARE_2024_NATIONAL
    return sum(dem * n for dem, n in polls) / total_n


def compute_gb_shift(
    polls_path: Path | str | None = None,
    manual_shift: float | None = None,
    yougov_json_path: Path | str | None = None,
    extra_gb_polls: list[tuple[float, int]] | None = None,
) -> GenericBallotInfo:
    """Compute the national environment shift relative to the 2024 presidential baseline.

    Combines polls from the RCP-sourced CSV (``polls_path``) and the YouGov
    weekly crosstab JSON (``yougov_json_path``) into a single sample-size-weighted
    average.  YouGov entries whose date window overlaps a CSV row are dropped to
    prevent double-counting the same poll from two sources.

    Parameters
    ----------
    polls_path:
        Path to the polls CSV.  Defaults to ``data/polls/polls_2026.csv``.
        Ignored when ``manual_shift`` is provided.
    manual_shift:
        When provided, use this value as the shift directly (skips poll loading).
        Useful for API callers that pass an explicit override.
    yougov_json_path:
        Path to the YouGov weekly GB crosstab JSON.  Defaults to
        ``data/polls/yougov_generic_ballot_2026.json``.  Pass an explicit path
        in tests to avoid depending on the real data file.
        Ignored when ``manual_shift`` is provided.
    extra_gb_polls:
        Additional (dem_share, n_sample) tuples injected directly into the
        weighted average.  Used by the early results pathway to incorporate
        off-year election results (e.g., VA/NJ 2025 governor) as Generic
        Ballot signal without putting them in the main polls CSV.
        These are appended after CSV and YouGov polls; no deduplication is
        applied because early results come from a separate file with
        different pollster labels.
        Ignored when ``manual_shift`` is provided.

    Returns
    -------
    GenericBallotInfo
        Struct with gb_avg, pres_baseline, shift, n_polls, n_yougov_polls, source.
    """
    if manual_shift is not None:
        # Caller provided an explicit override — use it directly.
        gb_avg = PRES_DEM_SHARE_2024_NATIONAL + manual_shift
        return GenericBallotInfo(
            gb_avg=gb_avg,
            pres_baseline=PRES_DEM_SHARE_2024_NATIONAL,
            shift=manual_shift,
            n_polls=0,
            n_yougov_polls=0,
            source="manual",
        )

    resolved_polls_path = Path(polls_path) if polls_path is not None else (
        PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    )

    # Load CSV polls and YouGov weekly issues separately so we can report counts.
    csv_polls = load_generic_ballot_polls(resolved_polls_path)

    yougov_raw = load_yougov_generic_ballot_polls(yougov_json_path)
    # Deduplicate: drop YouGov entries whose date window overlaps a CSV entry.
    yougov_polls = _deduplicate_yougov_polls(resolved_polls_path, yougov_raw)

    # Append early-cycle election results (e.g., VA/NJ governor) as additional
    # Generic Ballot signal.  These are pre-validated by load_early_results() and
    # already have n_sample capped at max_effective_n, so no further adjustment needed.
    injected_polls = extra_gb_polls or []

    all_polls = csv_polls + yougov_polls + injected_polls
    gb_avg = compute_gb_average(all_polls)
    shift = gb_avg - PRES_DEM_SHARE_2024_NATIONAL

    log.info(
        "Generic ballot: avg=%.4f, pres_baseline=%.4f, shift=%.4f pp "
        "(%d CSV polls + %d YouGov issues + %d early results = %d total)",
        gb_avg,
        PRES_DEM_SHARE_2024_NATIONAL,
        shift * 100,
        len(csv_polls),
        len(yougov_polls),
        len(injected_polls),
        len(all_polls),
    )

    return GenericBallotInfo(
        gb_avg=gb_avg,
        pres_baseline=PRES_DEM_SHARE_2024_NATIONAL,
        shift=shift,
        n_polls=len(csv_polls),
        n_yougov_polls=len(yougov_polls),
        source="auto",
    )


def apply_gb_shift(
    county_priors: np.ndarray,
    shift: float,
) -> np.ndarray:
    """Apply a flat national environment shift to county priors.

    Each county's prior is shifted by the same amount (``shift``), then clipped
    to [``_PRIOR_MIN``, ``_PRIOR_MAX``] to keep values in a valid probability range.

    Parameters
    ----------
    county_priors:
        ndarray of shape (N,), county-level prior Dem shares.
    shift:
        Flat shift in Dem share units (e.g. +0.016 for +1.6pp D improvement).

    Returns
    -------
    ndarray of shape (N,)
        Adjusted county priors.
    """
    adjusted = county_priors.astype(float) + shift
    return np.clip(adjusted, _PRIOR_MIN, _PRIOR_MAX)
