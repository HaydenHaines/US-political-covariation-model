"""Tests for src.reporting.forecast_diff."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import duckdb
import pytest

from src.reporting.forecast_diff import (
    RaceDiff,
    compute_diff,
    format_summary,
    snapshot_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(rows: list[tuple]) -> str:
    """Build a DuckDB file with the predictions table and return its path.

    NamedTemporaryFile creates an empty file; DuckDB refuses to open a
    pre-existing non-DuckDB file, so we delete it before connecting.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
    tmp.close()
    Path(tmp.name).unlink()  # let DuckDB create the file fresh
    con = duckdb.connect(tmp.name)
    con.execute("""
        CREATE TABLE predictions (
            county_fips    VARCHAR NOT NULL,
            race           VARCHAR NOT NULL,
            version_id     VARCHAR NOT NULL,
            pred_dem_share DOUBLE,
            pred_std       DOUBLE,
            pred_lo90      DOUBLE,
            pred_hi90      DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    if rows:
        con.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, NULL, NULL, NULL)",
            rows,
        )
    con.close()
    return tmp.name


# ---------------------------------------------------------------------------
# snapshot_predictions tests
# ---------------------------------------------------------------------------

class TestSnapshotPredictions:
    def test_returns_empty_for_missing_db(self, tmp_path):
        result = snapshot_predictions(tmp_path / "nonexistent.duckdb")
        assert result == {}

    def test_returns_empty_for_empty_predictions(self):
        db = _make_db([])
        result = snapshot_predictions(db)
        assert result == {}
        Path(db).unlink()

    def test_averages_across_counties(self):
        rows = [
            ("01001", "Senate-AL", "v1", 0.40),
            ("01003", "Senate-AL", "v1", 0.50),
            ("01005", "Senate-AL", "v1", 0.60),
        ]
        db = _make_db(rows)
        result = snapshot_predictions(db)
        Path(db).unlink()
        assert "Senate-AL" in result
        assert abs(result["Senate-AL"] - 0.50) < 1e-9

    def test_multiple_races(self):
        rows = [
            ("01001", "Senate-AL", "v1", 0.40),
            ("01001", "Governor-AL", "v1", 0.55),
        ]
        db = _make_db(rows)
        result = snapshot_predictions(db)
        Path(db).unlink()
        assert set(result.keys()) == {"Senate-AL", "Governor-AL"}
        assert abs(result["Senate-AL"] - 0.40) < 1e-9
        assert abs(result["Governor-AL"] - 0.55) < 1e-9

    def test_uses_latest_version_id(self):
        # v2 should win over v1 (MAX(version_id) is lexicographically later)
        rows = [
            ("01001", "Senate-AL", "v1", 0.30),
            ("01001", "Senate-AL", "v2", 0.70),
        ]
        db = _make_db(rows)
        result = snapshot_predictions(db)
        Path(db).unlink()
        assert abs(result["Senate-AL"] - 0.70) < 1e-9

    def test_ignores_null_pred_dem_share(self):
        rows = [
            ("01001", "Senate-AL", "v1", 0.40),
            ("01003", "Senate-AL", "v1", None),  # NULL — should be excluded
        ]
        db = _make_db(rows)
        result = snapshot_predictions(db)
        Path(db).unlink()
        # Average of only the non-NULL row
        assert abs(result["Senate-AL"] - 0.40) < 1e-9


# ---------------------------------------------------------------------------
# compute_diff tests
# ---------------------------------------------------------------------------

class TestComputeDiff:
    def test_no_change_returns_empty(self):
        snap = {"Senate-AL": 0.45, "Governor-GA": 0.52}
        result = compute_diff(snap, snap, threshold=0.005)
        assert result == []

    def test_small_change_below_threshold_excluded(self):
        before = {"Senate-AL": 0.450}
        after = {"Senate-AL": 0.453}  # delta = 0.003 < 0.005
        result = compute_diff(before, after, threshold=0.005)
        assert result == []

    def test_large_change_above_threshold_included(self):
        before = {"Senate-AL": 0.45}
        after = {"Senate-AL": 0.46}  # delta = 0.01 >= 0.005
        result = compute_diff(before, after, threshold=0.005)
        assert len(result) == 1
        diff = result[0]
        assert diff["race"] == "Senate-AL"
        assert abs(diff["before"] - 0.45) < 1e-9
        assert abs(diff["after"] - 0.46) < 1e-9
        assert abs(diff["delta"] - 0.01) < 1e-9

    def test_multiple_races_changing(self):
        before = {
            "Senate-AL": 0.45,
            "Governor-GA": 0.52,
            "Senate-NC": 0.48,
        }
        after = {
            "Senate-AL": 0.47,   # +0.02 — above threshold
            "Governor-GA": 0.52,  # no change
            "Senate-NC": 0.44,   # -0.04 — above threshold
        }
        result = compute_diff(before, after, threshold=0.005)
        races = {d["race"] for d in result}
        assert races == {"Senate-AL", "Senate-NC"}
        # Sorted by descending absolute delta — Senate-NC delta=0.04 first
        assert result[0]["race"] == "Senate-NC"
        assert result[1]["race"] == "Senate-AL"

    def test_new_race_always_included(self):
        before = {"Senate-AL": 0.45}
        after = {"Senate-AL": 0.45, "Governor-GA": 0.52}  # GA is new
        result = compute_diff(before, after, threshold=0.005)
        races = {d["race"] for d in result}
        assert "Governor-GA" in races
        ga = next(d for d in result if d["race"] == "Governor-GA")
        assert math.isnan(ga["before"])
        assert abs(ga["after"] - 0.52) < 1e-9

    def test_removed_race_always_included(self):
        before = {"Senate-AL": 0.45, "Governor-GA": 0.52}
        after = {"Senate-AL": 0.45}  # GA removed
        result = compute_diff(before, after, threshold=0.005)
        races = {d["race"] for d in result}
        assert "Governor-GA" in races
        ga = next(d for d in result if d["race"] == "Governor-GA")
        assert abs(ga["before"] - 0.52) < 1e-9
        assert math.isnan(ga["after"])

    def test_exact_threshold_boundary_included(self):
        before = {"Senate-AL": 0.45}
        after = {"Senate-AL": 0.455}  # delta == threshold
        result = compute_diff(before, after, threshold=0.005)
        assert len(result) == 1

    def test_custom_threshold(self):
        before = {"Senate-AL": 0.45}
        after = {"Senate-AL": 0.46}  # delta=0.01
        # With high threshold, no change
        result = compute_diff(before, after, threshold=0.05)
        assert result == []
        # With low threshold, change detected
        result = compute_diff(before, after, threshold=0.005)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# format_summary tests
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_empty_diff_returns_no_changes(self):
        out = format_summary([])
        assert "No meaningful" in out

    def test_single_change_shown(self):
        diffs: list[RaceDiff] = [
            RaceDiff(race="Senate-AL", before=0.45, after=0.47, delta=0.02)
        ]
        out = format_summary(diffs)
        assert "Senate-AL" in out
        assert "45.0%" in out
        assert "47.0%" in out

    def test_positive_delta_shown(self):
        diffs: list[RaceDiff] = [
            RaceDiff(race="Senate-AL", before=0.45, after=0.47, delta=0.02)
        ]
        out = format_summary(diffs)
        # Positive delta — should show + prefix somewhere
        assert "+" in out

    def test_negative_delta_shown(self):
        diffs: list[RaceDiff] = [
            RaceDiff(race="Senate-NC", before=0.48, after=0.44, delta=-0.04)
        ]
        out = format_summary(diffs)
        assert "Senate-NC" in out
        assert "44.0%" in out

    def test_new_race_labeled(self):
        diffs: list[RaceDiff] = [
            RaceDiff(race="Governor-GA", before=float("nan"), after=0.52, delta=float("nan"))
        ]
        out = format_summary(diffs)
        assert "NEW RACE" in out
        assert "Governor-GA" in out

    def test_removed_race_labeled(self):
        diffs: list[RaceDiff] = [
            RaceDiff(race="Governor-GA", before=0.52, after=float("nan"), delta=float("nan"))
        ]
        out = format_summary(diffs)
        assert "REMOVED" in out
        assert "Governor-GA" in out

    def test_multiple_races_in_summary(self):
        diffs: list[RaceDiff] = [
            RaceDiff(race="Senate-AL", before=0.45, after=0.47, delta=0.02),
            RaceDiff(race="Senate-NC", before=0.48, after=0.44, delta=-0.04),
        ]
        out = format_summary(diffs)
        assert "2 race(s)" in out
        assert "Senate-AL" in out
        assert "Senate-NC" in out


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_diff_mode(self, tmp_path):
        """CLI --before / --after mode produces a summary."""
        before_file = tmp_path / "before.json"
        after_file = tmp_path / "after.json"
        before_file.write_text(json.dumps({"Senate-AL": 0.45}))
        after_file.write_text(json.dumps({"Senate-AL": 0.47}))

        from src.reporting.forecast_diff import main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--before", str(before_file), "--after", str(after_file)])

        output = buf.getvalue()
        assert "Senate-AL" in output

    def test_cli_snapshot_mode(self):
        """CLI --snapshot writes JSON file."""
        rows = [("01001", "Senate-AL", "v1", 0.45)]
        db = _make_db(rows)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name

        from src.reporting.forecast_diff import main
        main(["--db", db, "--snapshot", "--out", out_path])

        data = json.loads(Path(out_path).read_text())
        assert "Senate-AL" in data
        assert abs(data["Senate-AL"] - 0.45) < 1e-9

        Path(db).unlink()
        Path(out_path).unlink()
