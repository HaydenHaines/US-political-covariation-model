# api/routers/pollsters.py
"""Pollster accuracy endpoint.

Serves pre-computed accuracy metrics from the 2022 backtest analysis.
Generate the data file by running:
    uv run python scripts/pollster_accuracy.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.models import (
    PollCoverageReportResponse,
    PollsterAccuracyEntry,
    PollsterAccuracyResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["pollsters"])

# Resolved relative to this file so it works regardless of working directory.
_ACCURACY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "experiments"
    / "pollster_accuracy.json"
)
_COVERAGE_REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "diagnostics"
    / "poll_coverage_report.json"
)


def _load_accuracy_data() -> dict:
    """Load pollster accuracy JSON from disk.

    Raises HTTPException(503) if the file hasn't been generated yet so that
    callers get a clear, actionable error message rather than a 500.
    """
    if not _ACCURACY_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Pollster accuracy data not yet generated. "
                "Run: uv run python scripts/pollster_accuracy.py"
            ),
        )
    with open(_ACCURACY_PATH) as f:
        return json.load(f)


def _load_coverage_report() -> dict:
    """Load poll coverage diagnostics JSON from disk.

    Raises HTTPException(503) if the report has not been generated yet.
    """
    if not _COVERAGE_REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Poll coverage diagnostics report not yet generated. "
                "Run: uv run python scripts/analyze_poll_coverage.py"
            ),
        )
    with open(_COVERAGE_REPORT_PATH) as f:
        return json.load(f)


@router.get("/pollsters/accuracy", response_model=PollsterAccuracyResponse)
def get_pollster_accuracy() -> PollsterAccuracyResponse:
    """Return ranked pollster accuracy metrics from the 2022 backtest.

    Pollsters are ranked by RMSE (percentage points) against actual 2022
    election outcomes. Rank 1 is the most accurate pollster.

    The data is generated offline by ``scripts/pollster_accuracy.py``,
    which cross-references polls from ``data/polls/polls_2022.csv`` against
    state-level actuals in ``data/experiments/backtest_2022_results.json``.
    """
    raw = _load_accuracy_data()

    pollsters = [
        PollsterAccuracyEntry(
            pollster=entry["pollster"],
            rank=entry["rank"],
            n_polls=entry["n_polls"],
            n_races=entry["n_races"],
            rmse_pp=entry["rmse_pp"],
            mean_error_pp=entry["mean_error_pp"],
        )
        for entry in raw.get("pollsters", [])
    ]

    return PollsterAccuracyResponse(
        description=raw.get("description", ""),
        n_pollsters=raw.get("n_pollsters", len(pollsters)),
        pollsters=pollsters,
    )


@router.get("/pollsters/coverage", response_model=PollCoverageReportResponse)
def get_poll_coverage_report() -> PollCoverageReportResponse:
    """Return the generated poll coverage diagnostics report.

    The data is generated offline by ``scripts/analyze_poll_coverage.py`` and
    identifies demographic coverage gaps in polls with xt_ composition data.
    """
    return PollCoverageReportResponse.model_validate(_load_coverage_report())
