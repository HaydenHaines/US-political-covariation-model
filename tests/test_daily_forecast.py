"""Tests for scripts/run_daily_forecast.py.

Covers:
  1. write_snapshot creates a dated JSON file with all expected per-race keys.
  2. A second run (new probs + prior snapshot) produces correct delta_pp_vs_yesterday.
  3. find_previous_snapshot skips today's file when looking for a prior.
  4. load_snapshot_win_probs round-trips the win_prob values correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_daily_forecast import (
    build_race_records,
    compute_deltas,
    find_previous_snapshot,
    load_snapshot_win_probs,
    write_snapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RACES = {
    "2026 AZ Senate": 0.45,
    "2026 GA Senate": -0.30,
}

_XT_COUNTS = {
    "2026 AZ Senate": 3,
    "2026 GA Senate": 0,
}


def _write_day(forecast_dir: Path, date_str: str, probs: dict[str, float]) -> Path:
    deltas = compute_deltas(probs, None)
    records = build_race_records(probs, {}, deltas)
    return write_snapshot(records, date_str, forecast_dir)


# ---------------------------------------------------------------------------
# 1. Output file creation
# ---------------------------------------------------------------------------


def test_write_snapshot_creates_expected_structure(tmp_path):
    """write_snapshot creates a dated JSON file with all expected per-race keys."""
    deltas = compute_deltas(_RACES, None)
    records = build_race_records(_RACES, _XT_COUNTS, deltas)
    out_path = write_snapshot(records, "2026-05-03", tmp_path)

    assert out_path == tmp_path / "2026-05-03.json"
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert data["date"] == "2026-05-03"
    assert isinstance(data["races"], list)
    assert len(data["races"]) == len(_RACES)

    expected_keys = {"race", "win_prob", "enriched_poll_count", "delta_pp_vs_yesterday"}
    for entry in data["races"]:
        assert set(entry.keys()) == expected_keys

    race_map = {e["race"]: e for e in data["races"]}
    assert race_map["2026 AZ Senate"]["win_prob"] == pytest.approx(0.45, abs=1e-4)
    assert race_map["2026 AZ Senate"]["enriched_poll_count"] == 3
    assert race_map["2026 AZ Senate"]["delta_pp_vs_yesterday"] is None  # first run


def test_write_snapshot_creates_forecast_dir_when_absent(tmp_path):
    """write_snapshot creates the output directory if it does not exist."""
    nested = tmp_path / "data" / "forecasts"
    records = build_race_records(_RACES, {}, compute_deltas(_RACES, None))
    out_path = write_snapshot(records, "2026-05-03", nested)

    assert nested.exists()
    assert out_path.exists()


# ---------------------------------------------------------------------------
# 2. Delta computation (second run)
# ---------------------------------------------------------------------------


def test_second_run_produces_delta_vs_yesterday(tmp_path):
    """compute_deltas returns correct per-race deltas when a prior snapshot exists."""
    day1_probs = {"2026 AZ Senate": 0.45, "2026 GA Senate": -0.30}
    day2_probs = {"2026 AZ Senate": 0.60, "2026 GA Senate": -0.20}

    # Write day 1
    day1_path = _write_day(tmp_path, "2026-05-03", day1_probs)

    # Compute day 2 deltas using the loaded day-1 snapshot
    prev_probs = load_snapshot_win_probs(day1_path)
    deltas = compute_deltas(day2_probs, prev_probs)

    assert deltas["2026 AZ Senate"] == pytest.approx(0.15, abs=1e-4)
    assert deltas["2026 GA Senate"] == pytest.approx(0.10, abs=1e-4)


def test_first_run_deltas_are_none(tmp_path):
    """compute_deltas returns None for all races when no prior snapshot is available."""
    deltas = compute_deltas(_RACES, None)
    assert all(v is None for v in deltas.values())


def test_race_absent_from_prior_gets_none_delta(tmp_path):
    """compute_deltas returns None for a race that did not appear in the prior snapshot."""
    prior = {"2026 AZ Senate": 0.45}
    today = {"2026 AZ Senate": 0.50, "2026 GA Senate": -0.30}

    deltas = compute_deltas(today, prior)
    assert deltas["2026 AZ Senate"] == pytest.approx(0.05, abs=1e-4)
    assert deltas["2026 GA Senate"] is None


# ---------------------------------------------------------------------------
# 3. find_previous_snapshot
# ---------------------------------------------------------------------------


def test_find_previous_snapshot_skips_today(tmp_path):
    """find_previous_snapshot returns the most recent snapshot that is NOT today."""
    for date_str in ["2026-05-01", "2026-05-02", "2026-05-03"]:
        (tmp_path / f"{date_str}.json").write_text("{}")

    result = find_previous_snapshot(tmp_path, "2026-05-03")
    assert result is not None
    assert result.name == "2026-05-02.json"


def test_find_previous_snapshot_returns_none_when_no_prior(tmp_path):
    """find_previous_snapshot returns None when the only snapshot is today's."""
    (tmp_path / "2026-05-03.json").write_text("{}")
    result = find_previous_snapshot(tmp_path, "2026-05-03")
    assert result is None


def test_find_previous_snapshot_returns_none_on_missing_dir(tmp_path):
    """find_previous_snapshot returns None when the directory doesn't exist."""
    result = find_previous_snapshot(tmp_path / "nonexistent", "2026-05-03")
    assert result is None


# ---------------------------------------------------------------------------
# 4. load_snapshot_win_probs
# ---------------------------------------------------------------------------


def test_load_snapshot_win_probs_roundtrip(tmp_path):
    """load_snapshot_win_probs recovers win_prob values written by write_snapshot."""
    records = build_race_records(_RACES, _XT_COUNTS, compute_deltas(_RACES, None))
    path = write_snapshot(records, "2026-05-03", tmp_path)

    probs = load_snapshot_win_probs(path)
    assert probs is not None
    assert probs["2026 AZ Senate"] == pytest.approx(0.45, abs=1e-4)
    assert probs["2026 GA Senate"] == pytest.approx(-0.30, abs=1e-4)


def test_load_snapshot_win_probs_returns_none_on_missing_file(tmp_path):
    result = load_snapshot_win_probs(tmp_path / "nonexistent.json")
    assert result is None


def test_load_snapshot_win_probs_returns_none_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    result = load_snapshot_win_probs(bad)
    assert result is None
