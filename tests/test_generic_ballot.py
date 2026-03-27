"""Tests for src/prediction/generic_ballot.py.

Covers: load, compute, apply functions and integration with predict_race.
"""
from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.prediction.generic_ballot import (
    PRES_DEM_SHARE_2024_NATIONAL,
    GenericBallotInfo,
    apply_gb_shift,
    compute_gb_average,
    compute_gb_shift,
    load_generic_ballot_polls,
)
from src.prediction.predict_2026_types import predict_race


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
        gb = compute_gb_shift(polls_path=path)
        assert gb.source == "auto"
        assert gb.n_polls == 1
        assert gb.gb_avg == pytest.approx(0.52)
        assert gb.shift == pytest.approx(0.52 - PRES_DEM_SHARE_2024_NATIONAL)

    def test_no_polls_gives_zero_shift(self, tmp_path):
        """When no generic ballot polls exist, shift should be 0.0."""
        rows = [
            {"race": "2026 FL Senate", "geography": "FL", "geo_level": "state",
             "dem_share": "0.47", "n_sample": "600", "date": "2026-01-01", "pollster": "Siena", "notes": ""},
        ]
        path = _write_polls_csv(rows, tmp_path)
        gb = compute_gb_shift(polls_path=path)
        assert gb.shift == pytest.approx(0.0)
        assert gb.n_polls == 0

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
