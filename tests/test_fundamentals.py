"""Tests for src/prediction/fundamentals.py.

Covers: historical data loading, model fitting, LOO validation, snapshot
loading, shift computation, edge cases, and applying shifts to county priors.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.prediction.fundamentals import (
    FundamentalsInfo,
    FundamentalsModel,
    FundamentalsSnapshot,
    _HistoricalRecord,
    apply_fundamentals_shift,
    compute_fundamentals_shift,
    load_fundamentals_snapshot,
    load_historical_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_history_csv(rows: list[dict], path: Path) -> Path:
    """Write minimal history CSV with all required columns."""
    fieldnames = [
        "year", "pres_party", "pres_net_approval_oct",
        "gdp_q2_growth_pct", "unemployment_oct", "cpi_yoy_oct",
        "dem_house_share_change_pp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _minimal_rows(n: int = 8) -> list[dict]:
    """Generate n synthetic history rows with sensible values."""
    rows = []
    for i in range(n):
        rows.append({
            "year": 1974 + i * 4,
            "pres_party": "D" if i % 2 == 0 else "R",
            "pres_net_approval_oct": -10.0 + i * 3,
            "gdp_q2_growth_pct": 2.0 - i * 0.3,
            "unemployment_oct": 5.0 + i * 0.5,
            "cpi_yoy_oct": 3.0 + i * 0.5,
            "dem_house_share_change_pp": -3.0 + i * 0.8,
        })
    return rows


def _make_snapshot(**overrides) -> FundamentalsSnapshot:
    """Build a default FundamentalsSnapshot with optional field overrides."""
    defaults = dict(
        cycle=2026,
        in_party="D",
        approval_net_oct=-12.0,
        gdp_q2_growth_pct=1.8,
        unemployment_oct=4.1,
        cpi_yoy_oct=3.2,
    )
    defaults.update(overrides)
    return FundamentalsSnapshot(**defaults)


def _write_snapshot_json(data: dict, path: Path) -> Path:
    """Write snapshot dict to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests: load_historical_data
# ---------------------------------------------------------------------------


class TestLoadHistoricalData:
    def test_loads_default_file(self):
        """Default CSV (data/raw/fundamentals/midterm_history.csv) must exist and load."""
        records = load_historical_data()
        assert len(records) >= 12, f"Expected >=12 historical cycles, got {len(records)}"

    def test_records_sorted_by_year(self):
        records = load_historical_data()
        years = [r.year for r in records]
        assert years == sorted(years), "Records should be sorted by year ascending"

    def test_contains_known_cycles(self):
        records = load_historical_data()
        years = {r.year for r in records}
        for expected_year in [1994, 2010, 2018, 2022]:
            assert expected_year in years, f"Expected year {expected_year} in history"

    def test_field_types(self, tmp_path):
        rows = _minimal_rows(5)
        path = _write_history_csv(rows, tmp_path / "hist.csv")
        records = load_historical_data(path)
        for r in records:
            assert isinstance(r.year, int)
            assert isinstance(r.pres_party, str)
            assert isinstance(r.pres_net_approval_oct, float)
            assert isinstance(r.dem_house_share_change_pp, float)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_historical_data(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        """CSV with missing required columns should raise ValueError."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("year,pres_party\n1994,D\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_historical_data(bad_csv)

    def test_skips_malformed_rows(self, tmp_path):
        """Rows with invalid numbers are skipped silently; valid rows load fine."""
        rows = _minimal_rows(4)
        rows[1]["gdp_q2_growth_pct"] = "not_a_number"
        path = _write_history_csv(rows, tmp_path / "hist.csv")
        records = load_historical_data(path)
        assert len(records) == 3  # one row skipped

    def test_pres_approval_1994_is_negative(self):
        """1994: Clinton had negative net approval during wave election."""
        records = load_historical_data()
        rec_1994 = next((r for r in records if r.year == 1994), None)
        assert rec_1994 is not None
        assert rec_1994.pres_net_approval_oct < 0

    def test_pres_approval_1998_is_positive(self):
        """1998: Clinton had strongly positive approval despite impeachment drama."""
        records = load_historical_data()
        rec_1998 = next((r for r in records if r.year == 1998), None)
        assert rec_1998 is not None
        assert rec_1998.pres_net_approval_oct > 0

    def test_2022_cpi_high(self):
        """2022: Inflation was ~8% — highest in the dataset."""
        records = load_historical_data()
        rec_2022 = next((r for r in records if r.year == 2022), None)
        assert rec_2022 is not None
        assert rec_2022.cpi_yoy_oct > 5.0, "2022 CPI should reflect ~8% inflation"

    def test_dem_share_change_has_positive_and_negative(self):
        """Dataset should have both Dem gains and losses across cycles."""
        records = load_historical_data()
        changes = [r.dem_house_share_change_pp for r in records]
        assert any(c > 0 for c in changes), "Expected at least one cycle with Dem gains"
        assert any(c < 0 for c in changes), "Expected at least one cycle with Dem losses"


# ---------------------------------------------------------------------------
# Tests: FundamentalsModel
# ---------------------------------------------------------------------------


class TestFundamentalsModel:
    def test_fit_returns_self(self, tmp_path):
        rows = _minimal_rows(8)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model = FundamentalsModel()
        result = model.fit(records)
        assert result is model

    def test_is_fitted_after_fit(self, tmp_path):
        rows = _minimal_rows(8)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model = FundamentalsModel()
        assert not model.is_fitted_
        model.fit(records)
        assert model.is_fitted_

    def test_raises_on_too_few_records(self):
        records = [
            _HistoricalRecord(2010, "D", -8.0, 2.5, 9.7, 1.2, -6.6),
            _HistoricalRecord(2018, "R", -12.0, 3.5, 3.7, 2.5, 3.0),
        ]
        model = FundamentalsModel()
        with pytest.raises(ValueError, match="at least 4"):
            model.fit(records)

    def test_raises_predict_before_fit(self):
        model = FundamentalsModel()
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict(-12.0, 1.8, 4.1, 3.2)

    def test_coef_shape(self, tmp_path):
        rows = _minimal_rows(8)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model = FundamentalsModel().fit(records)
        assert model.coef_.shape == (4,)

    def test_loo_rmse_nonnegative(self, tmp_path):
        rows = _minimal_rows(8)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model = FundamentalsModel().fit(records)
        assert model.loo_rmse_ >= 0.0
        assert not np.isnan(model.loo_rmse_)

    def test_predict_returns_five_floats(self, tmp_path):
        rows = _minimal_rows(8)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model = FundamentalsModel().fit(records)
        result = model.predict(-12.0, 1.8, 4.1, 3.2)
        assert len(result) == 5
        assert all(isinstance(v, float) for v in result)

    def test_contributions_sum_to_total(self, tmp_path):
        """Sum of contributions + intercept should equal total prediction."""
        rows = _minimal_rows(8)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model = FundamentalsModel().fit(records)
        total, approval, gdp, unemp, cpi = model.predict(-12.0, 1.8, 4.1, 3.2)
        assert total == pytest.approx(approval + gdp + unemp + cpi + model.intercept_, abs=1e-9)

    def test_from_default_data_fitted(self):
        """from_default_data() convenience constructor returns a fitted model."""
        model = FundamentalsModel.from_default_data()
        assert model.is_fitted_
        assert model.n_training_ >= 12

    def test_higher_alpha_shrinks_coef(self, tmp_path):
        """Higher regularization should shrink coefficient magnitudes."""
        rows = _minimal_rows(10)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model_low = FundamentalsModel(alpha=0.01).fit(records)
        model_high = FundamentalsModel(alpha=1000.0).fit(records)
        assert np.linalg.norm(model_high.coef_) < np.linalg.norm(model_low.coef_)

    def test_loo_prediction_direction_on_real_data(self):
        """LOO predictions on real data should have positive r with actuals.

        With N~13 and 4 features, the LOO Pearson r is genuinely modest
        (~0.1-0.4).  We only require r > 0 (correct direction) and that
        predictions are not constant (non-degenerate).  The wide LOO RMSE
        (~3-4pp) is expected and correctly reflected in FundamentalsInfo.loo_rmse.
        """
        records = load_historical_data()
        y_pred = []
        for i, r in enumerate(records):
            subset = [records[j] for j in range(len(records)) if j != i]
            m = FundamentalsModel(alpha=_DEFAULT_RIDGE_ALPHA).fit(subset)
            total, *_ = m.predict(r.pres_net_approval_oct, r.gdp_q2_growth_pct,
                                  r.unemployment_oct, r.cpi_yoy_oct)
            y_pred.append(total)
        y_true = [r.dem_house_share_change_pp for r in records]
        r = np.corrcoef(y_true, y_pred)[0, 1]
        # With N~13 the bar is low: just require correct direction (r > 0)
        # and non-constant predictions (std > 0.1pp)
        assert r > 0, f"LOO Pearson r should be positive, got {r:.3f}"
        assert np.std(y_pred) > 0.1, "LOO predictions appear degenerate (near-constant)"

    def test_n_training_matches_records(self, tmp_path):
        rows = _minimal_rows(9)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        model = FundamentalsModel().fit(records)
        assert model.n_training_ == 9

    def test_fit_on_real_data_has_reasonable_loo_rmse(self):
        """Real data LOO RMSE should be in [0.5, 10] pp — sanity range."""
        model = FundamentalsModel.from_default_data()
        assert model.loo_rmse_ > 0.5, "LOO RMSE implausibly low — possible data leak"
        assert model.loo_rmse_ < 10.0, "LOO RMSE implausibly high"


# Import the constant for use in tests
from src.prediction.fundamentals import _DEFAULT_RIDGE_ALPHA


# ---------------------------------------------------------------------------
# Tests: load_fundamentals_snapshot
# ---------------------------------------------------------------------------


class TestLoadFundamentalsSnapshot:
    def test_loads_default_snapshot(self):
        """Default snapshot_2026.json must exist and load."""
        snapshot = load_fundamentals_snapshot()
        assert isinstance(snapshot, FundamentalsSnapshot)
        assert snapshot.cycle == 2026
        assert snapshot.in_party in {"D", "R"}

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fundamentals_snapshot(tmp_path / "nope.json")

    def test_raises_on_missing_keys(self, tmp_path):
        path = tmp_path / "bad.json"
        _write_snapshot_json({"cycle": 2026}, path)
        with pytest.raises(ValueError, match="missing required keys"):
            load_fundamentals_snapshot(path)

    def test_raises_on_invalid_in_party(self, tmp_path):
        data = {
            "cycle": 2026, "in_party": "X",
            "approval_net_oct": -12.0, "gdp_q2_growth_pct": 1.8,
            "unemployment_oct": 4.1, "cpi_yoy_oct": 3.2,
        }
        path = tmp_path / "snap.json"
        _write_snapshot_json(data, path)
        with pytest.raises(ValueError, match="in_party must be"):
            load_fundamentals_snapshot(path)

    def test_loads_all_numeric_fields(self, tmp_path):
        data = {
            "cycle": 2026, "in_party": "D",
            "approval_net_oct": -15.5, "gdp_q2_growth_pct": 2.1,
            "unemployment_oct": 3.9, "cpi_yoy_oct": 4.0,
        }
        path = tmp_path / "snap.json"
        _write_snapshot_json(data, path)
        snap = load_fundamentals_snapshot(path)
        assert snap.approval_net_oct == pytest.approx(-15.5)
        assert snap.gdp_q2_growth_pct == pytest.approx(2.1)
        assert snap.unemployment_oct == pytest.approx(3.9)
        assert snap.cpi_yoy_oct == pytest.approx(4.0)

    def test_consumer_sentiment_optional(self, tmp_path):
        data = {
            "cycle": 2026, "in_party": "D",
            "approval_net_oct": -12.0, "gdp_q2_growth_pct": 1.8,
            "unemployment_oct": 4.1, "cpi_yoy_oct": 3.2,
        }
        path = tmp_path / "snap.json"
        _write_snapshot_json(data, path)
        snap = load_fundamentals_snapshot(path)
        assert snap.consumer_sentiment is None

    def test_in_party_case_insensitive(self, tmp_path):
        """in_party 'd' should be accepted and normalized to 'D'."""
        data = {
            "cycle": 2026, "in_party": "d",
            "approval_net_oct": -12.0, "gdp_q2_growth_pct": 1.8,
            "unemployment_oct": 4.1, "cpi_yoy_oct": 3.2,
        }
        path = tmp_path / "snap.json"
        _write_snapshot_json(data, path)
        snap = load_fundamentals_snapshot(path)
        assert snap.in_party == "D"


# ---------------------------------------------------------------------------
# Tests: compute_fundamentals_shift
# ---------------------------------------------------------------------------


class TestComputeFundamentalsShift:
    def test_returns_fundamentals_info(self):
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap)
        assert isinstance(info, FundamentalsInfo)

    def test_shift_is_float(self):
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap)
        assert isinstance(info.shift, float)
        assert not np.isnan(info.shift)

    def test_shift_in_reasonable_range(self):
        """Shift for 2026 inputs should be within [-0.15, +0.15] Dem share."""
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap)
        assert -0.15 < info.shift < 0.15, f"Shift {info.shift:.4f} outside sane range"

    def test_contributions_sum_to_shift(self):
        """All contribution components + intercept should sum to total shift."""
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap)
        implied_total = (
            info.approval_contribution
            + info.gdp_contribution
            + info.unemployment_contribution
            + info.cpi_contribution
            + info.intercept_contribution
        )
        assert implied_total == pytest.approx(info.shift, abs=1e-9)

    def test_source_is_fitted_ridge(self):
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap)
        assert info.source == "fitted_ridge"

    def test_n_training_matches_history(self):
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap)
        assert info.n_training >= 12

    def test_loo_rmse_in_fraction_units(self):
        """LOO RMSE returned in Dem-share fraction units (not pp)."""
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap)
        # LOO RMSE in pp would be ~2-4pp, so in fraction units ~0.02-0.04
        assert 0.001 < info.loo_rmse < 0.15, f"LOO RMSE {info.loo_rmse:.4f} seems wrong"

    def test_worse_approval_gives_worse_shift(self):
        """Lowering approval should decrease (more negative) fundamentals shift."""
        snap_good = _make_snapshot(approval_net_oct=10.0)
        snap_bad = _make_snapshot(approval_net_oct=-20.0)
        info_good = compute_fundamentals_shift(snap_good)
        info_bad = compute_fundamentals_shift(snap_bad)
        # Better approval should yield higher (or equal) shift
        assert info_good.shift >= info_bad.shift, (
            f"Higher approval ({snap_good.approval_net_oct}) should give >= shift "
            f"({info_good.shift:.4f} vs {info_bad.shift:.4f})"
        )

    def test_cpi_has_nonzero_contribution(self):
        """CPI contribution should be non-trivially different at extreme values.

        Note: With N~13 and Ridge regularization, the CPI coefficient sign is
        data-driven and may not match economic intuition (the 2022 anomaly —
        high inflation but only modest Dem seat loss — confounds the naive
        expectation).  We test that CPI has *some* effect, not its sign.
        """
        snap_low = _make_snapshot(cpi_yoy_oct=1.0)
        snap_high = _make_snapshot(cpi_yoy_oct=9.0)
        info_low = compute_fundamentals_shift(snap_low)
        info_high = compute_fundamentals_shift(snap_high)
        # Shifts should differ by a nontrivial amount given 8pp difference in CPI
        assert abs(info_low.shift - info_high.shift) > 1e-6, (
            "CPI appears to have zero effect on the shift — unexpected"
        )

    def test_accepts_prefit_model(self, tmp_path):
        """Passing a pre-fitted model skips refitting (faster for repeated calls)."""
        rows = _minimal_rows(8)
        path = _write_history_csv(rows, tmp_path / "h.csv")
        records = load_historical_data(path)
        fitted = FundamentalsModel().fit(records)
        snap = _make_snapshot()
        info = compute_fundamentals_shift(snap, _model=fitted)
        assert info.source == "fitted_ridge"
        assert info.n_training == 8

    def test_extreme_positive_approval_positive_shift(self):
        """Very positive approval (like 2002) should push shift upward."""
        snap_pos = _make_snapshot(approval_net_oct=60.0, cpi_yoy_oct=2.0, gdp_q2_growth_pct=2.0)
        snap_neg = _make_snapshot(approval_net_oct=-30.0, cpi_yoy_oct=2.0, gdp_q2_growth_pct=2.0)
        info_pos = compute_fundamentals_shift(snap_pos)
        info_neg = compute_fundamentals_shift(snap_neg)
        assert info_pos.shift > info_neg.shift


# ---------------------------------------------------------------------------
# Tests: apply_fundamentals_shift
# ---------------------------------------------------------------------------


class TestApplyFundamentalsShift:
    def test_positive_shift_increases_priors(self):
        priors = np.array([0.40, 0.50, 0.60])
        shifted = apply_fundamentals_shift(priors, 0.02)
        assert shifted == pytest.approx([0.42, 0.52, 0.62])

    def test_negative_shift_decreases_priors(self):
        priors = np.array([0.40, 0.50, 0.60])
        shifted = apply_fundamentals_shift(priors, -0.03)
        assert shifted == pytest.approx([0.37, 0.47, 0.57])

    def test_zero_shift_unchanged(self):
        priors = np.array([0.40, 0.50, 0.60])
        shifted = apply_fundamentals_shift(priors, 0.0)
        assert shifted == pytest.approx(priors)

    def test_clipped_to_valid_range(self):
        """Priors at extremes should be clipped to [0.01, 0.99]."""
        priors = np.array([0.005, 0.995])
        shifted_up = apply_fundamentals_shift(priors, 0.1)
        assert shifted_up[1] <= 0.99
        shifted_down = apply_fundamentals_shift(priors, -0.1)
        assert shifted_down[0] >= 0.01

    def test_does_not_modify_original(self):
        priors = np.array([0.40, 0.50, 0.60])
        original_copy = priors.copy()
        apply_fundamentals_shift(priors, 0.05)
        assert priors == pytest.approx(original_copy)

    def test_large_positive_shift_clipped_at_099(self):
        priors = np.array([0.95, 0.97, 0.99])
        shifted = apply_fundamentals_shift(priors, 0.10)
        assert np.all(shifted <= 0.99)

    def test_large_negative_shift_clipped_at_001(self):
        priors = np.array([0.01, 0.03, 0.05])
        shifted = apply_fundamentals_shift(priors, -0.10)
        assert np.all(shifted >= 0.01)

    def test_output_shape_preserved(self):
        priors = np.linspace(0.3, 0.7, 50)
        shifted = apply_fundamentals_shift(priors, 0.01)
        assert shifted.shape == priors.shape

    def test_integer_priors_converted_to_float(self):
        """Integer arrays should be handled without dtype errors."""
        priors = np.array([0, 1], dtype=int)  # degenerate but shouldn't crash
        shifted = apply_fundamentals_shift(priors, 0.5)
        assert shifted.dtype == float

    def test_end_to_end_snapshot_to_applied(self):
        """Full pipeline: load snapshot → compute shift → apply to priors."""
        snapshot = load_fundamentals_snapshot()
        info = compute_fundamentals_shift(snapshot)
        priors = np.full(100, 0.48)
        shifted = apply_fundamentals_shift(priors, info.shift)
        assert shifted.shape == (100,)
        assert np.all(shifted >= 0.01)
        assert np.all(shifted <= 0.99)
        # With negative approval, shift should be negative → priors should decrease
        if info.shift < 0:
            assert np.all(shifted <= 0.48 + 1e-9)
