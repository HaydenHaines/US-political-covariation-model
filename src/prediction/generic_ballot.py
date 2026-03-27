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
import logging
from dataclasses import dataclass
from pathlib import Path

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
        Number of generic ballot polls used.
    source:
        Human-readable description for API/display ("auto" or "manual").
    """

    gb_avg: float
    pres_baseline: float
    shift: float
    n_polls: int
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
) -> GenericBallotInfo:
    """Compute the national environment shift relative to the 2024 presidential baseline.

    Parameters
    ----------
    polls_path:
        Path to the polls CSV.  Defaults to ``data/polls/polls_2026.csv``.
        Ignored when ``manual_shift`` is provided.
    manual_shift:
        When provided, use this value as the shift directly (skips poll loading).
        Useful for API callers that pass an explicit override.

    Returns
    -------
    GenericBallotInfo
        Struct with gb_avg, pres_baseline, shift, n_polls, source.
    """
    if manual_shift is not None:
        # Caller provided an explicit override — use it directly.
        gb_avg = PRES_DEM_SHARE_2024_NATIONAL + manual_shift
        return GenericBallotInfo(
            gb_avg=gb_avg,
            pres_baseline=PRES_DEM_SHARE_2024_NATIONAL,
            shift=manual_shift,
            n_polls=0,
            source="manual",
        )

    polls = load_generic_ballot_polls(polls_path)
    gb_avg = compute_gb_average(polls)
    shift = gb_avg - PRES_DEM_SHARE_2024_NATIONAL

    log.info(
        "Generic ballot: avg=%.4f, pres_baseline=%.4f, shift=%.4f pp (%d polls)",
        gb_avg, PRES_DEM_SHARE_2024_NATIONAL, shift * 100, len(polls),
    )

    return GenericBallotInfo(
        gb_avg=gb_avg,
        pres_baseline=PRES_DEM_SHARE_2024_NATIONAL,
        shift=shift,
        n_polls=len(polls),
        source="auto",
    )


def apply_gb_shift(
    county_priors: "import numpy as np; np.ndarray",  # type: ignore[valid-type]
    shift: float,
) -> "import numpy as np; np.ndarray":  # type: ignore[valid-type]
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
    import numpy as np
    adjusted = county_priors.astype(float) + shift
    return np.clip(adjusted, _PRIOR_MIN, _PRIOR_MAX)
