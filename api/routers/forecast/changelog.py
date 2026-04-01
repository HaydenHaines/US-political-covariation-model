"""Forecast changelog endpoint: track prediction changes between snapshots."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from api.models import ChangelogEntry, ChangelogRaceDiff, ChangelogResponse

from ._helpers import _MIN_CHANGELOG_DELTA

router = APIRouter(tags=["forecast"])

SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "forecast_snapshots"

# Races with real poll-adjusted predictions (not just baseline copies).
# Only show changes for these to avoid cluttering with 60+ identical baseline races.
#
# UPDATE WHEN: a new race gets poll-adjusted predictions added to the model.
# A race should appear here once it has at least one poll ingested and produces
# predictions distinct from the structural baseline.
TRACKED_RACES: set[str] = {
    "2026 FL Senate", "2026 FL Governor", "2026 GA Senate", "2026 GA Governor",
    "2026 IA Senate", "2026 ME Senate", "2026 MI Senate", "2026 MI Governor",
    "2026 MN Senate", "2026 MN Governor", "2026 NC Senate", "2026 NH Senate",
    "2026 NH Governor", "2026 OH Governor", "2026 OR Senate",
    "2026 PA Governor", "2026 TX Senate", "2026 TX Governor",
    "2026 WI Governor", "2026 AL Senate", "2026 AL Governor",
}


@router.get("/forecast/changelog", response_model=ChangelogResponse)
def get_forecast_changelog() -> ChangelogResponse:
    """Return a changelog of forecast prediction changes between weekly snapshots.

    Snapshots are stored as JSON files in ``data/forecast_snapshots/``.
    Each file contains ``{date, predictions: {race: avg_dem_share}, note?}``.
    Entries are returned newest-first.
    """

    if not SNAPSHOTS_DIR.exists():
        return ChangelogResponse(entries=[], current_snapshot_date=None)

    snapshot_files = sorted(SNAPSHOTS_DIR.glob("*.json"))
    if not snapshot_files:
        return ChangelogResponse(entries=[], current_snapshot_date=None)

    # Load all snapshots ordered by date
    snapshots: list[dict] = []
    for f in snapshot_files:
        try:
            data = json.loads(f.read_text())
            snapshots.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    if not snapshots:
        return ChangelogResponse(entries=[], current_snapshot_date=None)

    entries: list[ChangelogEntry] = []

    # First snapshot = baseline entry (no diffs, just the initial state)
    first = snapshots[0]
    first_preds = first.get("predictions", {})
    initial_diffs = [
        ChangelogRaceDiff(race=race, before=None, after=val, delta=None)
        for race, val in sorted(first_preds.items())
        if race in TRACKED_RACES
    ]
    entries.append(ChangelogEntry(
        date=first.get("date", "unknown"),
        note=first.get("note", "Initial baseline"),
        diffs=initial_diffs,
    ))

    # Subsequent snapshots = diffs against previous
    for i in range(1, len(snapshots)):
        prev_preds = snapshots[i - 1].get("predictions", {})
        curr_preds = snapshots[i].get("predictions", {})
        curr_date = snapshots[i].get("date", "unknown")
        curr_note = snapshots[i].get("note")

        diffs: list[ChangelogRaceDiff] = []
        all_races = sorted((set(prev_preds) | set(curr_preds)) & TRACKED_RACES)

        for race in all_races:
            b = prev_preds.get(race)
            a = curr_preds.get(race)

            if b is not None and a is not None:
                delta = a - b
                if abs(delta) < _MIN_CHANGELOG_DELTA:
                    continue
            else:
                delta = None

            diffs.append(ChangelogRaceDiff(
                race=race, before=b, after=a, delta=delta,
            ))

        if diffs:
            entries.append(ChangelogEntry(
                date=curr_date, note=curr_note, diffs=diffs,
            ))

    # Return newest-first
    entries.reverse()

    return ChangelogResponse(
        entries=entries,
        current_snapshot_date=snapshots[-1].get("date"),
    )
