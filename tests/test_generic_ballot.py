"""Tests for src/prediction/generic_ballot.py.

Covers: load, compute, apply functions and integration with predict_race.
"""
from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.prediction.generic_ballot import (
    PRES_DEM_SHARE_2024_NATIONAL,
    GenericBallotInfo,
    _deduplicate_yougov_polls,
    apply_gb_shift,
    compute_gb_average,
    compute_gb_shift,
    load_generic_ballot_polls,
    load_yougov_generic_ballot_polls,
)
from src.prediction.forecast_runner import predict_race


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_polls_csv(rows: list[dict], tmp_path: Path) -> Path:
    """Write rows to a temporary polls CSV and return the path."""
    path = tmp_path / "polls_test.csv"
    fieldnames = ["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _make_synthetic_inputs(N: int = 20, J: int = 4):
    """Create minimal synthetic inputs for predict_race."""
    rng = np.random.RandomState(0)
    type_scores = rng.randn(N, J)
    # Make each county dominated by one type for clear predictions
    for i in range(N):
        type_scores[i, i % J] += 2.0

    A = rng.randn(J, J) * 0.02
    type_covariance = A @ A.T + np.eye(J) * 0.001

    type_priors = np.array([0.35, 0.55, 0.48, 0.42])
    county_fips = [f"{i:05d}" for i in range(N)]
    county_priors = np.full(N, 0.45)
    states = ["FL"] * N

    return type_scores, type_covariance, type_priors, county_fips, county_priors, states


# ---------------------------------------------------------------------------
# Unit tests: load_generic_ballot_polls
# ---------------------------------------------------------------------------

class TestLoadGenericBallotPolls:
    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_generic_ballot_polls(tmp_path / "nonexistent.csv")
        assert result == []

    def test_loads_matching_rows(self, tmp_path):
        rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-01-01", "pollster": "Ipsos", "notes": ""},
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.51", "n_sample": "800", "date": "2026-02-01", "pollster": "MorningConsult", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path)
        polls = load_generic_ballot_polls(path)
        assert len(polls) == 2
        assert polls[0] == (0.52, 1000)
        assert polls[1] == (0.51, 800)

    def test_skips_race_specific_rows(self, tmp_path):
        """Rows with state geo_level or non-GB race should be ignored."""
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.47", "n_sample": "600", "date": "2026-01-01", "pollster": "Siena", "notes": ""},
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-01-01", "pollster": "Ipsos", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path)
        polls = load_generic_ballot_polls(path)
        assert len(polls) == 1
        assert polls[0][0] == pytest.approx(0.52)

    def test_skips_invalid_rows(self, tmp_path):
        """Rows with invalid dem_share or n_sample should be silently skipped."""
        rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "bad", "n_sample": "1000", "date": "2026-01-01", "pollster": "X", "notes": ""},
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-01-01", "pollster": "Good", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path)
        polls = load_generic_ballot_polls(path)
        assert len(polls) == 1


# ---------------------------------------------------------------------------
# Unit tests: compute_gb_average
# ---------------------------------------------------------------------------

class TestComputeGbAverage:
    def test_empty_returns_pres_baseline(self):
        result = compute_gb_average([])
        assert result == pytest.approx(PRES_DEM_SHARE_2024_NATIONAL)

    def test_single_poll(self):
        result = compute_gb_average([(0.52, 1000)])
        assert result == pytest.approx(0.52)

    def test_sample_size_weighted(self):
        # Two polls: one at 0.50 (n=1000) and one at 0.60 (n=0 would be degenerate).
        # 1000*0.50 + 1000*0.60 = 1100, total = 2000, avg = 0.55
        result = compute_gb_average([(0.50, 1000), (0.60, 1000)])
        assert result == pytest.approx(0.55)

    def test_larger_n_dominates(self):
        # Large-n poll should dominate
        result = compute_gb_average([(0.48, 100), (0.55, 10000)])
        assert result > 0.54  # Should be much closer to 0.55


# ---------------------------------------------------------------------------
# Unit tests: compute_gb_shift
# ---------------------------------------------------------------------------

class TestComputeGbShift:
    def test_manual_shift_used_directly(self):
        gb = compute_gb_shift(manual_shift=0.025)
        assert gb.shift == pytest.approx(0.025)
        assert gb.source == "manual"
        assert gb.n_polls == 0

    def test_zero_manual_shift(self):
        gb = compute_gb_shift(manual_shift=0.0)
        assert gb.shift == pytest.approx(0.0)
        assert gb.source == "manual"

    def test_auto_from_polls_csv(self, tmp_path):
        rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-01-01", "pollster": "Ipsos", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path)
        # Pass a nonexistent YouGov path so this test exercises CSV-only behavior.
        gb = compute_gb_shift(polls_path=path, yougov_json_path=tmp_path / "no_yougov.json")
        assert gb.source == "auto"
        assert gb.n_polls == 1
        assert gb.n_yougov_polls == 0
        assert gb.gb_avg == pytest.approx(0.52)
        assert gb.shift == pytest.approx(0.52 - PRES_DEM_SHARE_2024_NATIONAL)

    def test_no_polls_gives_zero_shift(self, tmp_path):
        """When no generic ballot polls exist (CSV or YouGov), shift should be 0.0."""
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.47", "n_sample": "600", "date": "2026-01-01", "pollster": "Siena", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path)
        # Pass a nonexistent YouGov path so neither source has GB data.
        gb = compute_gb_shift(polls_path=path, yougov_json_path=tmp_path / "no_yougov.json")
        assert gb.shift == pytest.approx(0.0)
        assert gb.n_polls == 0
        assert gb.n_yougov_polls == 0

    def test_returns_generic_ballot_info_dataclass(self):
        gb = compute_gb_shift(manual_shift=0.016)
        assert isinstance(gb, GenericBallotInfo)
        assert gb.pres_baseline == pytest.approx(PRES_DEM_SHARE_2024_NATIONAL)


# ---------------------------------------------------------------------------
# Unit tests: apply_gb_shift
# ---------------------------------------------------------------------------

class TestApplyGbShift:
    def test_positive_shift_increases_priors(self):
        priors = np.array([0.40, 0.50, 0.60])
        shifted = apply_gb_shift(priors, 0.02)
        assert shifted == pytest.approx([0.42, 0.52, 0.62])

    def test_negative_shift_decreases_priors(self):
        priors = np.array([0.40, 0.50, 0.60])
        shifted = apply_gb_shift(priors, -0.03)
        assert shifted == pytest.approx([0.37, 0.47, 0.57])

    def test_zero_shift_unchanged(self):
        priors = np.array([0.40, 0.50, 0.60])
        shifted = apply_gb_shift(priors, 0.0)
        assert shifted == pytest.approx(priors)

    def test_clipped_to_valid_range(self):
        """Priors near 0 or 1 should be clipped to [0.01, 0.99]."""
        priors = np.array([0.005, 0.995])
        shifted_up = apply_gb_shift(priors, 0.1)
        assert shifted_up[1] <= 0.99
        shifted_down = apply_gb_shift(priors, -0.1)
        assert shifted_down[0] >= 0.01

    def test_does_not_modify_original(self):
        """apply_gb_shift should return a new array, not modify in place."""
        priors = np.array([0.40, 0.50, 0.60])
        original_copy = priors.copy()
        apply_gb_shift(priors, 0.05)
        assert priors == pytest.approx(original_copy)


# ---------------------------------------------------------------------------
# Integration tests: predict_race with generic_ballot_shift
# ---------------------------------------------------------------------------

class TestPredictRaceWithGenericBallot:
    def test_positive_shift_increases_predictions(self):
        """A positive generic ballot shift should increase all county predictions."""
        ts, tc, tp, fips, cp, states = _make_synthetic_inputs()
        result_no_shift = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=cp, generic_ballot_shift=0.0,
        )
        result_shifted = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=cp, generic_ballot_shift=0.03,
        )
        preds_no_shift = result_no_shift["pred_dem_share"].values
        preds_shifted = result_shifted["pred_dem_share"].values
        assert np.all(preds_shifted >= preds_no_shift - 1e-9)
        assert np.mean(preds_shifted) > np.mean(preds_no_shift)

    def test_negative_shift_decreases_predictions(self):
        ts, tc, tp, fips, cp, states = _make_synthetic_inputs()
        result_no_shift = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=cp, generic_ballot_shift=0.0,
        )
        result_shifted = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=cp, generic_ballot_shift=-0.03,
        )
        preds_no_shift = result_no_shift["pred_dem_share"].values
        preds_shifted = result_shifted["pred_dem_share"].values
        assert np.mean(preds_shifted) < np.mean(preds_no_shift)

    def test_zero_shift_gives_same_predictions(self):
        ts, tc, tp, fips, cp, states = _make_synthetic_inputs()
        result_a = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=cp, generic_ballot_shift=0.0,
        )
        result_b = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=cp,  # default is 0.0
        )
        np.testing.assert_allclose(
            result_a["pred_dem_share"].values,
            result_b["pred_dem_share"].values,
            atol=1e-12,
        )

    def test_shift_not_applied_without_county_priors(self):
        """Generic ballot shift is a no-op when county_priors is None (legacy path)."""
        ts, tc, tp, fips, cp, states = _make_synthetic_inputs()
        result_a = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=None, generic_ballot_shift=0.05,
        )
        result_b = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=None, generic_ballot_shift=0.0,
        )
        # Both should be identical since county_priors=None means shift cannot apply
        np.testing.assert_allclose(
            result_a["pred_dem_share"].values,
            result_b["pred_dem_share"].values,
            atol=1e-12,
        )

    def test_predictions_remain_in_valid_range(self):
        ts, tc, tp, fips, cp, states = _make_synthetic_inputs()
        # Extreme shift to test clipping
        result = predict_race(
            race="test", type_scores=ts, type_covariance=tc,
            type_priors=tp, county_fips=fips, states=states,
            county_priors=cp, generic_ballot_shift=0.5,
        )
        preds = result["pred_dem_share"].values
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)


# ---------------------------------------------------------------------------
# Helpers for YouGov tests
# ---------------------------------------------------------------------------

def _write_yougov_json(entries: list[dict], tmp_path: Path) -> Path:
    """Write a list of YouGov issue dicts to a temp JSON file and return the path."""
    path = tmp_path / "yougov_test.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _make_yougov_entry(
    gb_topline: float = 0.52,
    n_total_ballot: int = 1500,
    date_start: str = "2026-01-02",
    date_end: str = "2026-01-05",
) -> dict:
    """Create a minimal well-formed YouGov issue dict."""
    return {
        "date_start": date_start,
        "date_end": date_end,
        "gb_topline": gb_topline,
        "n_total_ballot": n_total_ballot,
        "n_total": n_total_ballot + 10,
    }


# ---------------------------------------------------------------------------
# Unit tests: load_yougov_generic_ballot_polls
# ---------------------------------------------------------------------------

class TestLoadYougovGenericBallotPolls:
    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_yougov_generic_ballot_polls(tmp_path / "nonexistent.json")
        assert result == []

    def test_loads_valid_entries(self, tmp_path):
        entries = [
            _make_yougov_entry(0.5493, 1547, "2026-01-02", "2026-01-05"),
            _make_yougov_entry(0.5417, 1597, "2026-01-09", "2026-01-12"),
        ]
        path = _write_yougov_json(entries, tmp_path)
        result = load_yougov_generic_ballot_polls(path)
        assert len(result) == 2
        assert result[0] == pytest.approx((0.5493, 1547, "2026-01-02", "2026-01-05"))
        assert result[1] == pytest.approx((0.5417, 1597, "2026-01-09", "2026-01-12"))

    def test_skips_out_of_range_topline(self, tmp_path):
        """gb_topline of 0.0 or 1.0 is invalid and should be skipped."""
        entries = [
            _make_yougov_entry(0.0, 1000, "2026-01-02", "2026-01-05"),
            _make_yougov_entry(1.0, 1000, "2026-01-09", "2026-01-12"),
            _make_yougov_entry(0.52, 1500, "2026-01-16", "2026-01-19"),
        ]
        path = _write_yougov_json(entries, tmp_path)
        result = load_yougov_generic_ballot_polls(path)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(0.52)

    def test_skips_zero_or_negative_n_ballot(self, tmp_path):
        """Entries with n_total_ballot <= 0 should be dropped."""
        entries = [
            _make_yougov_entry(0.52, 0),
            _make_yougov_entry(0.53, 1500),
        ]
        path = _write_yougov_json(entries, tmp_path)
        result = load_yougov_generic_ballot_polls(path)
        assert len(result) == 1

    def test_skips_missing_required_fields(self, tmp_path):
        """Entries missing gb_topline or n_total_ballot should be silently skipped."""
        entries = [
            {"date_start": "2026-01-02", "date_end": "2026-01-05", "n_total_ballot": 1500},  # no gb_topline
            _make_yougov_entry(0.53, 1500),
        ]
        path = _write_yougov_json(entries, tmp_path)
        result = load_yougov_generic_ballot_polls(path)
        assert len(result) == 1

    def test_handles_malformed_json(self, tmp_path):
        """A file with invalid JSON should return an empty list, not raise."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        result = load_yougov_generic_ballot_polls(path)
        assert result == []

    def test_date_fields_preserved(self, tmp_path):
        """date_start and date_end should be returned as third and fourth tuple elements."""
        entry = _make_yougov_entry(0.52, 1500, "2026-02-06", "2026-02-09")
        path = _write_yougov_json([entry], tmp_path)
        result = load_yougov_generic_ballot_polls(path)
        assert result[0][2] == "2026-02-06"
        assert result[0][3] == "2026-02-09"


# ---------------------------------------------------------------------------
# Unit tests: _deduplicate_yougov_polls
# ---------------------------------------------------------------------------

class TestDeduplicateYougovPolls:
    def test_keeps_all_entries_when_no_overlap(self, tmp_path):
        """YouGov dates before any CSV entry should all be kept."""
        csv_rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-03-30",
             "pollster": "Economist/YouGov", "notes": ""},
        ]
        path = _write_polls_csv(csv_rows, tmp_path)
        # YouGov entries end on Jan 26 at latest — no overlap with Mar 30 CSV date.
        yougov = [
            (0.5493, 1547, "2026-01-02", "2026-01-05"),
            (0.5417, 1597, "2026-01-09", "2026-01-12"),
        ]
        result = _deduplicate_yougov_polls(path, yougov)
        assert len(result) == 2

    def test_drops_entry_whose_window_contains_csv_date(self, tmp_path):
        """A YouGov entry whose date window brackets a CSV date should be dropped."""
        csv_rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-01-04",
             "pollster": "Economist/YouGov", "notes": ""},
        ]
        path = _write_polls_csv(csv_rows, tmp_path)
        # This YouGov entry spans Jan 2–5, which contains the CSV date Jan 4.
        yougov = [(0.5493, 1547, "2026-01-02", "2026-01-05")]
        result = _deduplicate_yougov_polls(path, yougov)
        assert len(result) == 0

    def test_keeps_non_overlapping_when_some_overlap(self, tmp_path):
        """Only the overlapping entry is dropped; others are kept."""
        csv_rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-01-04",
             "pollster": "Economist/YouGov", "notes": ""},
        ]
        path = _write_polls_csv(csv_rows, tmp_path)
        yougov = [
            (0.5493, 1547, "2026-01-02", "2026-01-05"),  # overlaps Jan 4
            (0.5417, 1597, "2026-01-09", "2026-01-12"),  # no overlap
        ]
        result = _deduplicate_yougov_polls(path, yougov)
        assert len(result) == 1
        assert result[0] == (0.5417, 1597)

    def test_handles_missing_csv_file(self, tmp_path):
        """A missing CSV should not raise; all YouGov entries should be kept."""
        yougov = [(0.5493, 1547, "2026-01-02", "2026-01-05")]
        result = _deduplicate_yougov_polls(tmp_path / "no_such.csv", yougov)
        assert len(result) == 1

    def test_dedup_returns_two_tuple_format(self, tmp_path):
        """Output should be (dem_share, n_sample) 2-tuples, not 4-tuples."""
        path = _write_polls_csv([], tmp_path)
        yougov = [(0.52, 1500, "2026-01-02", "2026-01-05")]
        result = _deduplicate_yougov_polls(path, yougov)
        assert len(result) == 1
        assert result[0] == (0.52, 1500)


# ---------------------------------------------------------------------------
# Unit tests: compute_gb_shift with YouGov integration
# ---------------------------------------------------------------------------

class TestComputeGbShiftWithYougov:
    def test_combines_csv_and_yougov_polls(self, tmp_path):
        """compute_gb_shift should incorporate both CSV and YouGov data."""
        csv_rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-03-30",
             "pollster": "Economist/YouGov", "notes": ""},
        ]
        csv_path = _write_polls_csv(csv_rows, tmp_path)
        yougov_entries = [
            _make_yougov_entry(0.54, 2000, "2026-01-02", "2026-01-05"),
        ]
        yougov_path = _write_yougov_json(yougov_entries, tmp_path)

        gb = compute_gb_shift(polls_path=csv_path, yougov_json_path=yougov_path)

        assert gb.n_polls == 1
        assert gb.n_yougov_polls == 1
        # Combined: (0.52*1000 + 0.54*2000) / 3000 = 1600/3000 ≈ 0.5333
        expected_avg = (0.52 * 1000 + 0.54 * 2000) / 3000
        assert gb.gb_avg == pytest.approx(expected_avg, abs=1e-6)
        assert gb.source == "auto"

    def test_yougov_only_no_csv_polls(self, tmp_path):
        """Works when CSV has no GB rows but YouGov JSON has entries."""
        # CSV with only a state race — no GB rows.
        csv_rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.47", "n_sample": "600", "date": "2026-01-01",
             "pollster": "Siena", "notes": ""},
        ]
        csv_path = _write_polls_csv(csv_rows, tmp_path)
        yougov_entries = [_make_yougov_entry(0.54, 1500, "2026-01-02", "2026-01-05")]
        yougov_path = _write_yougov_json(yougov_entries, tmp_path)

        gb = compute_gb_shift(polls_path=csv_path, yougov_json_path=yougov_path)

        assert gb.n_polls == 0
        assert gb.n_yougov_polls == 1
        assert gb.gb_avg == pytest.approx(0.54)

    def test_missing_yougov_file_handled_gracefully(self, tmp_path):
        """A missing YouGov JSON should not cause an error — just zero YouGov polls."""
        csv_rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-03-30",
             "pollster": "Ipsos", "notes": ""},
        ]
        csv_path = _write_polls_csv(csv_rows, tmp_path)
        missing_json = tmp_path / "no_yougov.json"

        gb = compute_gb_shift(polls_path=csv_path, yougov_json_path=missing_json)

        assert gb.n_polls == 1
        assert gb.n_yougov_polls == 0
        assert gb.gb_avg == pytest.approx(0.52)

    def test_dedup_prevents_double_counting(self, tmp_path):
        """YouGov entry whose window contains a CSV date is excluded from average."""
        # CSV has an entry on Jan 4 (inside the YouGov Jan 2-5 window).
        csv_rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-01-04",
             "pollster": "Economist/YouGov", "notes": ""},
        ]
        csv_path = _write_polls_csv(csv_rows, tmp_path)
        # Same YouGov entry with a very different topline to detect if it leaks in.
        yougov_entries = [_make_yougov_entry(0.99, 1500, "2026-01-02", "2026-01-05")]
        yougov_path = _write_yougov_json(yougov_entries, tmp_path)

        gb = compute_gb_shift(polls_path=csv_path, yougov_json_path=yougov_path)

        # The YouGov entry should be dropped — only the CSV entry should count.
        assert gb.n_yougov_polls == 0
        assert gb.gb_avg == pytest.approx(0.52)

    def test_n_yougov_polls_zero_for_manual_shift(self):
        """Manual shift path should report n_yougov_polls=0."""
        gb = compute_gb_shift(manual_shift=0.025)
        assert gb.n_yougov_polls == 0

    def test_result_has_correct_shift(self, tmp_path):
        """shift should equal gb_avg - PRES_DEM_SHARE_2024_NATIONAL."""
        csv_rows = [
            {"race": "2026 Generic Ballot", "geography": "national", "geo_level": "national",
             "dem_share": "0.52", "n_sample": "1000", "date": "2026-03-30",
             "pollster": "Ipsos", "notes": ""},
        ]
        csv_path = _write_polls_csv(csv_rows, tmp_path)
        yougov_entries = [_make_yougov_entry(0.54, 1000, "2026-01-02", "2026-01-05")]
        yougov_path = _write_yougov_json(yougov_entries, tmp_path)

        gb = compute_gb_shift(polls_path=csv_path, yougov_json_path=yougov_path)

        assert gb.shift == pytest.approx(gb.gb_avg - PRES_DEM_SHARE_2024_NATIONAL)
