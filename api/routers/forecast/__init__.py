"""Forecast router package -- combines domain-focused sub-routers into one router.

main.py imports ``from api.routers import forecast`` and uses ``forecast.router``,
so this module re-exports a combined ``router`` that includes all sub-routers.
"""
from fastapi import APIRouter

from . import blend, changelog, comparisons, overview, polls, race_detail, race_history, seat_history

# Re-export helpers that tests and other modules import directly
from ._helpers import (  # noqa: F401
    _BASELINE_YEAR,
    _DEFAULT_DEM_SHARE_PRIOR,
    _DEFAULT_SAMPLE_SIZE,
    _HISTORICAL_RESULTS,
    _HISTORICAL_RESULTS_PATH,
    _MATRIX_JITTER,
    _MIN_CHANGELOG_DELTA,
    _SLIDER_NORM,
    _VOTE_WEIGHTED_STATE_PRED_SQL,
    _STATE_STD_CAP,
    _STATE_STD_FALLBACK,
    _STATE_STD_FLOOR,
    _Z90,
    _apply_behavior_if_needed,
    _compute_state_std,
    _format_baseline_label,
    _load_historical_results,
    _lookup_pollster_grade,
    marginToRating,
    race_to_slug,
    slug_to_race,
)

# Re-export changelog constants used by tests
from .changelog import SNAPSHOTS_DIR, TRACKED_RACES  # noqa: F401

# Re-export the changelog endpoint function used by tests
from .changelog import get_forecast_changelog  # noqa: F401

router = APIRouter(tags=["forecast"])
router.include_router(overview.router)
router.include_router(race_detail.router)
router.include_router(blend.router)
router.include_router(polls.router)
router.include_router(changelog.router)
router.include_router(comparisons.router)
router.include_router(race_history.router)
router.include_router(seat_history.router)
