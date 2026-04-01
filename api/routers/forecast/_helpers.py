"""Shared helpers for the forecast router package.

Contains slug conversion, historical results loading, pollster grade lookup,
margin-to-rating conversion, baseline label formatting, and common forecast
computation helpers (behavior adjustment, vote-weighted state std).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import Request

from api.ratings import dem_share_to_rating, margin_to_rating  # noqa: F401

# Path to the static historical results data file (lives alongside the api/ package)
_HISTORICAL_RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "historical_results.json"


def _load_historical_results() -> dict:
    """Load and return the historical results dict from disk.

    Returns an empty dict when the file is missing (graceful degradation).
    Strips comment keys (those starting with '_') used for documentation.
    """
    if not _HISTORICAL_RESULTS_PATH.exists():
        return {}
    with _HISTORICAL_RESULTS_PATH.open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# Loaded once at import time - this file changes only when race data is manually updated
_HISTORICAL_RESULTS: dict = _load_historical_results()

# Uncertainty model parameters — see docs/ARCHITECTURE.md for calibration notes
_STATE_STD_FLOOR = 0.035      # minimum state-level std; prevents over-confidence when counties agree
_STATE_STD_CAP = 0.15         # hard cap; beyond this, the race is essentially a coin flip
_STATE_STD_FALLBACK = 0.065   # used when poll-derived std is unavailable
_MATRIX_JITTER = 1e-8         # Tikhonov regularization keeps covariance PD during matrix inversion
_Z90 = 1.645                  # z-score for 90% confidence interval


# SQL fragment: vote-weighted state-level aggregation of predicted Dem share.
# Falls back to simple AVG when total_votes_2024 is NULL.
# Usage: embed in a SELECT that JOINs predictions p with counties c.
_VOTE_WEIGHTED_STATE_PRED_SQL = """\
CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
          / SUM(COALESCE(c.total_votes_2024, 0))
     ELSE AVG(p.pred_dem_share)
END"""


def race_to_slug(race: str) -> str:
    """Convert race label to URL slug. "2026 FL Governor" → "2026-fl-governor"."""
    return race.lower().replace(" ", "-")


def slug_to_race(slug: str) -> str:
    """Convert URL slug back to race label. "2026-fl-governor" → "2026 FL Governor"."""
    parts = slug.split("-")
    if len(parts) < 3:
        return slug
    year = parts[0]
    state = parts[1].upper()
    race_type = " ".join(p.capitalize() for p in parts[2:])
    return f"{year} {state} {race_type}"


def _lookup_pollster_grade(request: Request, pollster_name: str | None) -> str | None:
    """Look up Silver Bulletin letter grade for a pollster, with fuzzy matching."""
    if not pollster_name:
        return None
    grades = getattr(request.app.state, "pollster_grades", {})
    norm_grades = getattr(request.app.state, "pollster_grades_normalized", {})
    if not grades:
        return None
    # Exact match
    if pollster_name in grades:
        return grades[pollster_name]
    # Normalized match
    from src.assembly.silver_bulletin_ratings import _normalize, _name_similarity
    norm = _normalize(pollster_name)
    if norm in norm_grades:
        return norm_grades[norm]
    # Fuzzy match (Jaccard > 0.4)
    best_grade, best_sim = None, 0.0
    for nk, grade in norm_grades.items():
        sim = _name_similarity(norm, nk)
        if sim > best_sim:
            best_sim = sim
            best_grade = grade
    return best_grade if best_sim >= 0.4 else None


def _format_baseline_label(pres_baseline: float) -> str:
    """Format the presidential baseline as a party-margin label, e.g. 'R+3.2' or 'D+0.5'.

    The label measures how far the 2024 presidential Dem share deviates from 50/50.
    shift = pres_baseline - 0.5; negative shift → Republican advantage → 'R+X'.
    """
    shift = pres_baseline - 0.5
    magnitude = round(abs(shift) * 100, 1)
    if shift < 0:
        return f"R+{magnitude}"
    return f"D+{magnitude}"


def marginToRating(dem_share: float) -> str:
    """Python equivalent of the frontend marginToRating for API use.

    DEPRECATED: Use ``dem_share_to_rating`` from ``api.ratings`` instead.
    Kept as a thin wrapper for backward compatibility with existing imports.
    """
    return dem_share_to_rating(dem_share)


# ── Named constants for magic numbers ────────────────────────────────────────

# Default prior for Dem two-party share when a county/tract has no Ridge
# prediction.  Slightly below 0.5 reflects the structural R lean of
# geographically-distributed units (many small rural tracts vs few large
# urban ones).
_DEFAULT_DEM_SHARE_PRIOR = 0.45

# Fallback median poll sample size when no sample sizes are available.
# 600 is a reasonable median for US political polls (Pew/Gallup typical
# range 500-1,500; state polls skew smaller).
_DEFAULT_SAMPLE_SIZE = 600

# Minimum absolute change in predicted Dem share (fraction) between
# snapshots to consider a change "meaningful" in the changelog.
# 0.002 = 0.2 percentage points.
_MIN_CHANGELOG_DELTA = 0.002

# The presidential election year used as the structural baseline for
# the generic ballot adjustment.
_BASELINE_YEAR = 2024

# Divisor for normalizing 0-100 slider percentages to [0, 2] multipliers.
# 100 / 50 = 2.0 keeps the scale symmetric around the default of 1.0.
_SLIDER_NORM = 50.0


# ── Shared computation helpers ───────────────────────────────────────────────

def _apply_behavior_if_needed(
    request: Request,
    county_priors: "np.ndarray | None",
    race: str,
) -> "np.ndarray | None":
    """Apply voter behavior adjustment for off-cycle races, if data is loaded.

    Checks app.state for behavior_tau and behavior_delta, determines whether
    the race is off-cycle (non-presidential), and applies the turnout/choice
    shift adjustment from the behavior layer.

    Returns the adjusted priors array (or the original if no adjustment was
    needed).
    """
    if county_priors is None:
        return county_priors

    behavior_tau = getattr(request.app.state, "behavior_tau", None)
    behavior_delta = getattr(request.app.state, "behavior_delta", None)
    type_scores = getattr(request.app.state, "type_scores", None)

    race_str = (race or "").lower()
    is_offcycle = not any(kw in race_str for kw in ["president", "pres"])

    if (
        behavior_tau is not None
        and behavior_delta is not None
        and is_offcycle
        and type_scores is not None
        and type_scores.shape[1] == len(behavior_tau)
    ):
        from src.behavior.voter_behavior import apply_behavior_adjustment

        county_priors = apply_behavior_adjustment(
            county_priors, type_scores, behavior_tau, behavior_delta, is_offcycle=True
        )

    return county_priors


def _compute_state_std(
    county_predictions: "np.ndarray",
    county_votes: "np.ndarray",
    state_pred: float,
) -> float:
    """Compute vote-weighted state-level standard deviation from county predictions.

    Uses the effective sample size (N_eff = 1 / sum(w_i^2)) to scale the
    weighted variance of county predictions around the state mean.  Result
    is clamped to [_STATE_STD_FLOOR, _STATE_STD_CAP].

    Falls back to _STATE_STD_FALLBACK when there are fewer than 2 counties
    or total vote weight is zero.
    """
    total_w = county_votes.sum()
    if total_w <= 0 or len(county_predictions) < 2:
        return _STATE_STD_FALLBACK

    weights_norm = county_votes / total_w
    county_var = float(np.sum(weights_norm * (county_predictions - state_pred) ** 2))
    n_eff = max(1.0, 1.0 / np.sum(weights_norm ** 2))
    state_std = float(np.sqrt(county_var / n_eff))
    state_std = max(state_std, _STATE_STD_FLOOR)
    state_std = min(state_std, _STATE_STD_CAP)
    return state_std
