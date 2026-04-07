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

# Path to the candidate data file (lives in the data/config/ directory at project root)
_CANDIDATES_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "config" / "candidates_2026.json"
)


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


def _load_candidates() -> dict[str, dict]:
    """Load candidate data from candidates_2026.json, keyed by race_id.

    The JSON file organizes candidates by race type (senate, governor).
    This function flattens them into a single dict keyed by race_id
    (e.g. "2026 GA Senate") for O(1) lookup during request handling.

    Returns an empty dict when the file is missing (graceful degradation).
    """
    if not _CANDIDATES_PATH.exists():
        return {}
    with _CANDIDATES_PATH.open() as f:
        data = json.load(f)
    # Flatten: merge senate and governor dicts into one lookup
    flat: dict[str, dict] = {}
    for race_type_key in ("senate", "governor"):
        if race_type_key in data:
            flat.update(data[race_type_key])
    return flat


# Loaded once at import time - these files change only when race data is manually updated
_HISTORICAL_RESULTS: dict = _load_historical_results()
_CANDIDATES: dict[str, dict] = _load_candidates()

# Uncertainty model parameters — see docs/ARCHITECTURE.md for calibration notes
_STATE_STD_CAP = 0.15         # hard cap; beyond this, the race is essentially a coin flip
_STATE_STD_FALLBACK = 0.065   # used when poll-derived std is unavailable (generic fallback)
_MATRIX_JITTER = 1e-8         # Tikhonov regularization keeps covariance PD during matrix inversion
_Z90 = 1.645                  # z-score for 90% confidence interval

# Empirical error floors by race type, derived from 2022 backtest
# (data/experiments/backtest_2022_results.json, raw_prior state_metrics).
#
# Senate: RMSE across 28 contested states = 3.7pp.  Small because Senate
# candidates have high name recognition and most states are non-competitive.
#
# Governor: RMSE across competitive states (error < 15pp, N=29) = 5.5pp.
# Uncompetitive/landslide states excluded — they distort the floor upward
# and are not useful for uncertainty calibration in competitive races.
#
# These constants are the *minimum* std the model will report.  They
# prevent false precision: even a model with county-level agreement
# should never claim tighter confidence than historical backtest errors show.
_SENATE_STD_FLOOR = 0.037     # 3.7pp — 2022 backtest RMSE, Senate, 28 states
_GOVERNOR_STD_FLOOR = 0.055   # 5.5pp — 2022 backtest RMSE, Governor, competitive states
_GENERIC_STD_FLOOR = 0.035    # generic floor for unrecognized race types


def _get_std_floor(race_type: str) -> float:
    """Return the empirical error floor for a race type.

    The floor is the minimum std the model will report for a state-level
    prediction.  It is calibrated from 2022 backtest RMSE by race type.

    Parameters
    ----------
    race_type:
        Normalized race type string from the races table (e.g. 'senate',
        'governor').  Case-insensitive.
    """
    normalized = race_type.lower().strip()
    if "senate" in normalized:
        return _SENATE_STD_FLOOR
    if "governor" in normalized or "gov" in normalized:
        return _GOVERNOR_STD_FLOOR
    return _GENERIC_STD_FLOOR


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
    race_type: str = "",
) -> float:
    """Compute vote-weighted state-level standard deviation from county predictions.

    Uses the effective sample size (N_eff = 1 / sum(w_i^2)) to scale the
    weighted variance of county predictions around the state mean.  The result
    is clamped to [empirical_floor, _STATE_STD_CAP].

    The empirical floor is calibrated per race type from the 2022 backtest
    (data/experiments/backtest_2022_results.json):
    - Senate:   3.7pp (28-state RMSE)
    - Governor: 5.5pp (competitive-state RMSE, excl. landslides)
    - Other:    3.5pp (generic fallback)

    This prevents false precision: even when all counties agree on a prediction,
    the model should not claim tighter confidence than historical errors allow.

    Falls back to _STATE_STD_FALLBACK when there are fewer than 2 counties
    or total vote weight is zero.

    Parameters
    ----------
    county_predictions:
        Array of county-level predicted Dem shares (0-1).
    county_votes:
        Array of vote counts per county (used as weights).
    state_pred:
        Vote-weighted state mean prediction (used to center variance).
    race_type:
        Race type string (e.g. 'senate', 'governor').  Used to select the
        empirical error floor.  Empty string falls back to generic floor.
    """
    std_floor = _get_std_floor(race_type)

    total_w = county_votes.sum()
    if total_w <= 0 or len(county_predictions) < 2:
        # Can't compute variance; fall back to the larger of the generic fallback
        # and the race-type-specific floor so we don't report false precision.
        return max(_STATE_STD_FALLBACK, std_floor)

    weights_norm = county_votes / total_w
    county_var = float(np.sum(weights_norm * (county_predictions - state_pred) ** 2))
    n_eff = max(1.0, 1.0 / np.sum(weights_norm ** 2))
    state_std = float(np.sqrt(county_var / n_eff))
    state_std = max(state_std, std_floor)
    state_std = min(state_std, _STATE_STD_CAP)
    return state_std
