"""GET /forecast/race-history -- per-race margin movement over time.

Reads all forecast snapshots from ``data/forecast_snapshots/`` and returns
a time series of dem margin (= dem_share - 0.5) for every race that
appears in the snapshots.

Margin sign convention:
  > 0  ->  Democrat-leaning
  < 0  ->  Republican-leaning
  = 0  ->  exactly tied

This powers the inline sparklines on the senate overview page.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter(tags=["forecast"])

# Snapshots directory -- same path changelog.py and seat_history.py use.
SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "forecast_snapshots"


def _load_race_history_from_snapshots(
    snapshots_dir: Path,
) -> list[dict]:
    """Load per-race margin history from all snapshot JSON files.

    Each snapshot contains a ``predictions`` dict mapping race names
    (e.g. "2026 FL Senate") to avg_dem_share floats in [0, 1].
    We convert each to a signed margin (dem_share - 0.5) and group
    all dates for each race into a time series.

    Returns a list of dicts:
        [{"slug": "2026-fl-senate", "history": [{"date": "...", "margin": 0.04}, ...]}, ...]

    Races with only one snapshot still get returned -- the sparkline will
    render a dot or flat line for single-point series.
    """
    if not snapshots_dir.exists():
        return []

    snapshot_files = sorted(snapshots_dir.glob("*.json"))
    if not snapshot_files:
        return []

    # Accumulate: race_name -> list of {date, margin}
    race_to_history: dict[str, list[dict]] = {}

    for f in snapshot_files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("race_history: could not read snapshot %s -- skipping", f.name)
            continue

        date = data.get("date")
        predictions = data.get("predictions", {})

        if not date:
            log.warning("race_history: snapshot %s has no 'date' field -- skipping", f.name)
            continue

        for race, dem_share in predictions.items():
            if not isinstance(dem_share, (int, float)):
                continue
            margin = round(float(dem_share) - 0.5, 6)
            if race not in race_to_history:
                race_to_history[race] = []
            race_to_history[race].append({"date": date, "margin": margin})

    # Convert to the response list, sorted by race name for stable output.
    # History within each race is already chronological because snapshot_files
    # is sorted alphabetically (filenames are YYYY-MM-DD.json).
    return [
        {"slug": _race_to_slug(race), "history": history}
        for race, history in sorted(race_to_history.items())
    ]


def _race_to_slug(race: str) -> str:
    """Convert "2026 FL Senate" -> "2026-fl-senate".

    Mirrors the convention used by race_to_slug() in _helpers.py.
    Kept local to avoid a circular import through _helpers' heavyweight deps.
    """
    return race.lower().replace(" ", "-")


@router.get("/forecast/race-history")
def get_race_history() -> list[dict]:
    """Return per-race margin history from forecast snapshots.

    Each entry represents one race:
    ``{"slug": "2026-fl-senate", "history": [{"date": "2026-03-29", "margin": 0.04}, ...]}``.

    Margin is defined as ``avg_dem_share - 0.5`` -- positive values are
    Dem-favored, negative are GOP-favored.  History is sorted chronologically
    (oldest first).  Races with only one snapshot are included; the frontend
    sparkline handles single-point gracefully.

    If no snapshots exist, an empty list is returned.
    """
    return _load_race_history_from_snapshots(SNAPSHOTS_DIR)
