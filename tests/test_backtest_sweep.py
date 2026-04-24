"""Tests for src/validation/backtest_sweep.py.

Tests cover:
  - Year-adaptive prior loading (correct year selection, fallback, coverage)
  - Parameterized backtest runner (params injection, output shape)
  - Parameter sweep (grid generation, output DataFrame shape)
  - Prior comparison (Ridge vs adaptive, output structure)
  - Edge cases (missing data, unknown years)
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.validation.backtest_sweep import (
    _ALL_RACE_CONFIGS,
    _FALLBACK_DEM_SHARE,
    _PRESIDENTIAL_ACTUALS_YEARS,
    _find_prior_presidential_year,
    _format_params,
    _quick_grid,
    build_blended_priors,
    build_year_adaptive_priors,
    compare_priors,
    run_backtest_with_params,
    sweep_parameters,
)
from tests.conftest import skip_if_missing

# Files in ``data/raw/`` and ``data/assembled/`` are gitignored, so CI never
# has them. The backtest integration tests below need actual historical
# election data to run; when absent they skip instead of erroring.
_PRES_CHECKING = "data/raw/fivethirtyeight/checking-our-work-data/presidential_elections.csv"
_SEN_CHECKING = "data/raw/fivethirtyeight/checking-our-work-data/us_senate_elections.csv"
_GOV_CHECKING = "data/raw/fivethirtyeight/checking-our-work-data/governors_elections.csv"
_ACTUALS_PRES_2020 = "data/assembled/medsl_county_presidential_2020.parquet"
_ACTUALS_PRES_2016 = "data/assembled/medsl_county_presidential_2016.parquet"
_ACTUALS_PRES_2008 = "data/assembled/medsl_county_presidential_2008.parquet"
_ACTUALS_SEN_2022 = "data/assembled/medsl_county_senate_2022.parquet"
_ACTUALS_GOV_2018 = "data/assembled/algara_county_governor_2018.parquet"
_COMMUNITIES_TYPES = "data/communities/type_assignments.parquet"


# ---------------------------------------------------------------------------
# Year selection logic
# ---------------------------------------------------------------------------

class TestFindPriorPresidentialYear:
    def test_2008_uses_2004(self):
        assert _find_prior_presidential_year(2008) == 2004

    def test_2012_uses_2008(self):
        assert _find_prior_presidential_year(2012) == 2008

    def test_2016_uses_2012(self):
        assert _find_prior_presidential_year(2016) == 2012

    def test_2020_uses_2016(self):
        assert _find_prior_presidential_year(2020) == 2016

    def test_2010_senate_uses_2008(self):
        """Senate 2010 should use the most recent pres election: 2008."""
        assert _find_prior_presidential_year(2010) == 2008

    def test_2018_governor_uses_2016(self):
        """Governor 2018 should use pres 2016."""
        assert _find_prior_presidential_year(2018) == 2016

    def test_2022_uses_2020(self):
        assert _find_prior_presidential_year(2022) == 2020

    def test_year_before_all_data_returns_none(self):
        """If target_year is before all available data, return None."""
        assert _find_prior_presidential_year(1999) is None

    def test_year_2000_returns_none(self):
        """2000 is in our data but there's nothing before it."""
        assert _find_prior_presidential_year(2000) is None


# ---------------------------------------------------------------------------
# Year-adaptive prior loading
# ---------------------------------------------------------------------------

class TestBuildYearAdaptivePriors:
    """Tests that actually load data from disk (integration tests)."""

    def test_2008_priors_from_2004_returns_correct_shape(self):
        """Loading 2004 actuals for 2008 backtest should produce a float array."""
        # Use a small subset of FIPS codes we know exist in 2004 data.
        test_fips = ["01001", "06037", "17031", "36061", "48201"]
        priors = build_year_adaptive_priors(test_fips, 2008)
        assert priors.shape == (5,)
        assert priors.dtype == np.float64 or priors.dtype == np.float32

    def test_priors_are_valid_dem_shares(self):
        """All loaded priors should be in (0, 1) for real counties."""
        test_fips = ["01001", "06037", "17031", "36061", "48201"]
        priors = build_year_adaptive_priors(test_fips, 2012)
        # All should be valid probabilities (or fallback, which is 0.45).
        for i, p in enumerate(priors):
            assert 0.0 < p < 1.0, f"Prior {p} for {test_fips[i]} out of range"

    def test_missing_fips_gets_fallback(self):
        """A fake FIPS code should get the fallback prior."""
        priors = build_year_adaptive_priors(["99999"], 2016)
        assert abs(priors[0] - _FALLBACK_DEM_SHARE) < 1e-9

    def test_very_early_year_gets_all_fallback(self):
        """Year before all available data should produce all-fallback array."""
        test_fips = ["01001", "06037"]
        priors = build_year_adaptive_priors(test_fips, 1999)
        np.testing.assert_allclose(priors, _FALLBACK_DEM_SHARE)

    @skip_if_missing(_ACTUALS_PRES_2016)
    def test_prior_year_matches_expected(self):
        """Verify that 2020 backtest loads 2016 data (not 2020 itself)."""
        # Load both the 2016 actuals directly and the year-adaptive priors for 2020.
        # They should be identical for the same FIPS.
        test_fips = ["06037"]  # LA County
        assembled_dir = Path(__file__).resolve().parents[1] / "data" / "assembled"

        priors = build_year_adaptive_priors(test_fips, 2020, assembled_dir=assembled_dir)

        df_2016 = pd.read_parquet(assembled_dir / "medsl_county_presidential_2016.parquet")
        df_2016["county_fips"] = df_2016["county_fips"].astype(str).str.zfill(5)
        la_row = df_2016[df_2016["county_fips"] == "06037"]
        expected = la_row["pres_dem_share_2016"].values[0]

        assert abs(priors[0] - expected) < 1e-9, (
            f"Year-adaptive prior {priors[0]} != 2016 actual {expected}"
        )


# ---------------------------------------------------------------------------
# Parameterized backtest runner
# ---------------------------------------------------------------------------

class TestRunBacktestWithParams:
    """Integration tests that run actual backtests with different params."""

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_default_params_produce_valid_metrics(self):
        """Run with defaults and check output structure."""
        result = run_backtest_with_params(2020, "president", {})
        assert "r" in result
        assert "rmse" in result
        assert "bias" in result
        assert "direction_accuracy" in result
        assert "n_counties" in result
        assert result["n_counties"] > 0
        assert 0.5 < result["r"] < 1.0, f"Unexpected r={result['r']}"

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2008, _COMMUNITIES_TYPES)
    def test_year_adaptive_priors_flag(self):
        """Year-adaptive priors should produce different (generally better) results."""
        ridge = run_backtest_with_params(2008, "president", {"use_year_adaptive_priors": False})
        adaptive = run_backtest_with_params(2008, "president", {"use_year_adaptive_priors": True})

        # Both should produce valid metrics.
        assert not math.isnan(ridge["r"])
        assert not math.isnan(adaptive["r"])
        # They should be different (different priors → different forecasts).
        assert ridge["r"] != adaptive["r"], "Ridge and adaptive priors gave identical r"

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_custom_lam_changes_output(self):
        """Different lam values should produce different results."""
        low_lam = run_backtest_with_params(2020, "president", {"lam": 0.1})
        high_lam = run_backtest_with_params(2020, "president", {"lam": 10.0})
        assert low_lam["r"] != high_lam["r"], "Different lam values gave identical r"

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_poll_blend_scale_changes_output(self):
        """Different poll_blend_scale should affect county predictions."""
        low_k = run_backtest_with_params(2020, "president", {"poll_blend_scale": 1.0})
        high_k = run_backtest_with_params(2020, "president", {"poll_blend_scale": 50.0})
        assert low_k["rmse"] != high_k["rmse"], "Different poll_blend_scale gave identical RMSE"

    @skip_if_missing(_SEN_CHECKING, _ACTUALS_SEN_2022, _COMMUNITIES_TYPES)
    def test_senate_backtest_works(self):
        """Senate race backtest should produce valid output."""
        result = run_backtest_with_params(2022, "senate", {"use_year_adaptive_priors": True})
        assert result["n_counties"] > 0
        assert not math.isnan(result["r"])

    @skip_if_missing(_GOV_CHECKING, _ACTUALS_GOV_2018, _COMMUNITIES_TYPES)
    def test_governor_backtest_works(self):
        """Governor race backtest should produce valid output."""
        result = run_backtest_with_params(2018, "governor", {})
        assert result["n_counties"] > 0
        assert not math.isnan(result["r"])

    def test_invalid_race_raises(self):
        """Invalid race type should raise."""
        with pytest.raises(ValueError):
            run_backtest_with_params(2020, "dogcatcher", {})

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_params_are_recorded_in_output(self):
        """The result dict should include the params that were used."""
        params = {"lam": 2.5, "mu": 0.5, "poll_blend_scale": 7.0}
        result = run_backtest_with_params(2020, "president", params)
        assert result["params"]["lam"] == 2.5
        assert result["params"]["mu"] == 0.5
        assert result["params"]["poll_blend_scale"] == 7.0


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

@skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _SEN_CHECKING, _ACTUALS_SEN_2022, _COMMUNITIES_TYPES)
class TestSweepParameters:
    def test_small_sweep_output_shape(self):
        """A 2x2 grid over 2 elections should produce 2*2*2 = 8 rows."""
        grid = {"lam": [0.5, 2.0], "mu": [0.5, 2.0]}
        configs = [(2020, "president"), (2022, "senate")]
        df = sweep_parameters(grid, configs)

        assert len(df) == 4 * 2  # 4 param combos × 2 elections
        assert "lam" in df.columns
        assert "mu" in df.columns
        assert "year" in df.columns
        assert "r" in df.columns
        assert "rmse" in df.columns

    def test_sweep_includes_all_metric_columns(self):
        """Output should have r, rmse, bias, direction_accuracy."""
        grid = {"lam": [1.0]}
        configs = [(2020, "president")]
        df = sweep_parameters(grid, configs)

        for col in ["r", "rmse", "bias", "direction_accuracy"]:
            assert col in df.columns, f"Missing column: {col}"
            assert not df[col].isna().all(), f"Column {col} is all NaN"


# ---------------------------------------------------------------------------
# Prior comparison
# ---------------------------------------------------------------------------

class TestComparePriors:
    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_comparison_output_structure(self):
        """compare_priors should return a DataFrame with both Ridge and adaptive columns."""
        configs = [(2020, "president")]
        df = compare_priors(configs)

        assert len(df) == 1
        for col in ["ridge_r", "adaptive_r", "r_delta", "ridge_rmse", "adaptive_rmse", "rmse_delta"]:
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestFormatParams:
    def test_float_params(self):
        s = _format_params({"lam": 1.5, "mu": 0.3})
        assert "lam=1.50" in s
        assert "mu=0.30" in s

    def test_bool_params(self):
        s = _format_params({"use_year_adaptive_priors": True})
        assert "Y" in s

    def test_empty_params(self):
        assert _format_params({}) == ""


class TestQuickGrid:
    def test_quick_grid_has_expected_params(self):
        grid, configs = _quick_grid()
        assert "lam" in grid
        assert "mu" in grid
        assert "poll_blend_scale" in grid
        assert "use_year_adaptive_priors" in grid
        assert "prior_decay" in grid
        assert "half_life_days" in grid
        assert len(configs) >= 3, "Quick grid should have at least 3 elections"

    def test_quick_grid_prior_decay_values(self):
        """Quick grid should include decay=0 (baseline) and at least one positive value."""
        grid, _ = _quick_grid()
        assert 0.0 in grid["prior_decay"]
        assert any(v > 0 for v in grid["prior_decay"])

    def test_quick_grid_half_life_days_values(self):
        """Quick grid should include the default 30.0 and at least one other value."""
        grid, _ = _quick_grid()
        assert 30.0 in grid["half_life_days"]
        assert len(grid["half_life_days"]) >= 2


class TestAllRaceConfigs:
    def test_all_configs_count(self):
        """Should have 4 pres + 8 senate + 3 governor = 15 elections."""
        assert len(_ALL_RACE_CONFIGS) == 15

    def test_all_configs_contain_expected_types(self):
        types = {rt for _, rt in _ALL_RACE_CONFIGS}
        assert types == {"president", "senate", "governor"}


# ---------------------------------------------------------------------------
# Blended prior tests
# ---------------------------------------------------------------------------

class TestBuildBlendedPriors:
    """Tests for the temporal prior blending function."""

    TEST_FIPS = ["01001", "06037", "17031", "36061", "48201"]

    def test_decay_zero_matches_single_year(self):
        """decay=0.0 should produce identical results to build_year_adaptive_priors()."""
        fips = self.TEST_FIPS
        blended = build_blended_priors(fips, 2020, prior_decay=0.0)
        single = build_year_adaptive_priors(fips, 2020)
        np.testing.assert_allclose(
            blended, single,
            err_msg="decay=0 blended != single-year adaptive priors",
        )

    def test_decay_zero_matches_single_year_2008(self):
        """decay=0.0 should replicate single-year priors for any target year."""
        fips = self.TEST_FIPS
        blended = build_blended_priors(fips, 2008, prior_decay=0.0)
        single = build_year_adaptive_priors(fips, 2008)
        np.testing.assert_allclose(blended, single)

    @skip_if_missing(_ACTUALS_PRES_2016)
    def test_equal_weight_decay_one(self):
        """decay=1.0 should average all prior elections equally.

        We verify this by manually loading each prior election and computing
        the unweighted mean, then comparing to build_blended_priors output.
        """
        fips = ["06037"]  # LA County — present in all elections
        assembled_dir = Path(__file__).resolve().parents[1] / "data" / "assembled"
        target_year = 2020
        prior_years = [y for y in _PRESIDENTIAL_ACTUALS_YEARS if y < target_year]

        shares = []
        for year in prior_years:
            path = assembled_dir / f"medsl_county_presidential_{year}.parquet"
            df = pd.read_parquet(path)
            df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
            share_col = f"pres_dem_share_{year}"
            row = df[df["county_fips"] == "06037"]
            shares.append(float(row[share_col].values[0]))

        expected_mean = float(np.mean(shares))
        blended = build_blended_priors(fips, target_year, prior_decay=1.0, assembled_dir=assembled_dir)

        assert abs(blended[0] - expected_mean) < 1e-6, (
            f"Equal-weight blended={blended[0]:.6f} != manual mean={expected_mean:.6f}"
        )

    @skip_if_missing(_ACTUALS_PRES_2016)
    def test_weight_normalization(self):
        """For any decay value, the blended output should be a valid weighted average.

        Property: the result must lie between the min and max of the individual
        election values.  This holds iff weights are non-negative and sum to 1.
        We test the property rather than internal weights directly.
        """
        fips = ["06037"]
        assembled_dir = Path(__file__).resolve().parents[1] / "data" / "assembled"
        target_year = 2020
        prior_years = [y for y in _PRESIDENTIAL_ACTUALS_YEARS if y < target_year]

        # Gather all individual per-year shares for LA County.
        shares = []
        for year in prior_years:
            path = assembled_dir / f"medsl_county_presidential_{year}.parquet"
            df = pd.read_parquet(path)
            df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
            share_col = f"pres_dem_share_{year}"
            row = df[df["county_fips"] == "06037"]
            shares.append(float(row[share_col].values[0]))

        lo, hi = min(shares), max(shares)

        for decay in [0.1, 0.3, 0.5, 0.7, 1.0]:
            blended = build_blended_priors(fips, target_year, prior_decay=decay, assembled_dir=assembled_dir)
            val = blended[0]
            assert lo - 1e-9 <= val <= hi + 1e-9, (
                f"decay={decay}: blended={val:.4f} is outside [{lo:.4f}, {hi:.4f}] "
                "— weights are not normalized or not non-negative"
            )

    def test_missing_county_falls_back(self):
        """A fake FIPS code should get the fallback share regardless of decay."""
        for decay in [0.0, 0.5, 1.0]:
            priors = build_blended_priors(["99999"], 2020, prior_decay=decay)
            assert abs(priors[0] - _FALLBACK_DEM_SHARE) < 1e-6, (
                f"decay={decay}: fake FIPS got {priors[0]}, expected fallback {_FALLBACK_DEM_SHARE}"
            )

    def test_no_prior_years_returns_fallback(self):
        """Target year before all available data should return fallback for all counties."""
        priors = build_blended_priors(self.TEST_FIPS, 1999, prior_decay=0.5)
        np.testing.assert_allclose(priors, _FALLBACK_DEM_SHARE)

    @skip_if_missing(_ACTUALS_PRES_2016)
    def test_decay_changes_output(self):
        """Different decay values should produce different results (for multi-election windows)."""
        fips = self.TEST_FIPS
        # 2020 has 5 prior elections: 2000, 2004, 2008, 2012, 2016.
        # decay=0 → only 2016; decay=1 → equal average; these must differ.
        out_0 = build_blended_priors(fips, 2020, prior_decay=0.0)
        out_1 = build_blended_priors(fips, 2020, prior_decay=1.0)
        # They should differ in at least one county.
        assert not np.allclose(out_0, out_1), (
            "decay=0.0 and decay=1.0 produced identical priors — blending is not working"
        )

    def test_returns_correct_shape(self):
        """Output shape must match input county list length."""
        fips = self.TEST_FIPS
        priors = build_blended_priors(fips, 2016, prior_decay=0.5)
        assert priors.shape == (len(fips),)

    def test_values_are_valid_dem_shares(self):
        """All blended priors must be in (0, 1)."""
        fips = self.TEST_FIPS
        priors = build_blended_priors(fips, 2020, prior_decay=0.5)
        for i, p in enumerate(priors):
            assert 0.0 < p < 1.0, f"Prior {p} for {fips[i]} is out of [0, 1]"


# ---------------------------------------------------------------------------
# half_life_days pass-through test
# ---------------------------------------------------------------------------

class TestHalfLifeDaysPassThrough:
    """Verify that half_life_days is accepted and influences forecast output."""

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_half_life_days_param_accepted(self):
        """run_backtest_with_params should not raise when half_life_days is supplied."""
        result = run_backtest_with_params(
            2020, "president", {"half_life_days": 30.0}
        )
        assert "r" in result
        assert not math.isnan(result["r"])

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_half_life_days_passed_to_run_forecast(self):
        """half_life_days should be forwarded to run_forecast(), not silently dropped.

        We mock run_forecast() to capture the kwargs it was called with, then
        verify half_life_days appears at the correct value.  This is a white-box
        test but it's the right boundary to mock: run_forecast is an external
        module we don't own in this test context.
        """
        from unittest.mock import MagicMock, patch
        from src.prediction.forecast_engine import ForecastResult

        # Build a minimal ForecastResult-like object so the loop doesn't crash.
        mock_fr = MagicMock(spec=ForecastResult)
        mock_fr.county_preds_local = np.full(3154, 0.5)

        captured_kwargs: dict = {}

        def fake_run_forecast(**kwargs):
            captured_kwargs.update(kwargs)
            return {}  # Empty dict → run produces "no_matched_counties" result, which is fine.

        with patch("src.validation.backtest_sweep.run_forecast", side_effect=fake_run_forecast):
            run_backtest_with_params(2020, "president", {"half_life_days": 14.0})

        assert "half_life_days" in captured_kwargs, (
            "run_forecast was not called with half_life_days"
        )
        assert captured_kwargs["half_life_days"] == 14.0, (
            f"half_life_days forwarded as {captured_kwargs['half_life_days']}, expected 14.0"
        )


# ---------------------------------------------------------------------------
# prior_decay integration with run_backtest_with_params
# ---------------------------------------------------------------------------

class TestRunBacktestWithPriorDecay:
    """Verify prior_decay is wired through to the actual backtest."""

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2008, _COMMUNITIES_TYPES)
    def test_prior_decay_zero_same_as_adaptive_default(self):
        """prior_decay=0 should match use_year_adaptive_priors=True (single-year) results."""
        default_adaptive = run_backtest_with_params(
            2008, "president", {"use_year_adaptive_priors": True}
        )
        decay_zero = run_backtest_with_params(
            2008, "president", {"use_year_adaptive_priors": True, "prior_decay": 0.0}
        )
        # Same priors → same forecast → same metrics.
        assert default_adaptive["r"] == decay_zero["r"]
        assert default_adaptive["rmse"] == decay_zero["rmse"]

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_prior_decay_positive_produces_valid_metrics(self):
        """prior_decay=0.5 should produce valid (non-NaN) metrics."""
        result = run_backtest_with_params(
            2020, "president",
            {"use_year_adaptive_priors": True, "prior_decay": 0.5},
        )
        assert not math.isnan(result["r"]), "prior_decay=0.5 produced NaN r"
        assert result["n_counties"] > 0

    @skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
    def test_prior_decay_ignored_without_adaptive_priors(self):
        """prior_decay should have no effect when use_year_adaptive_priors=False."""
        ridge_no_decay = run_backtest_with_params(
            2020, "president",
            {"use_year_adaptive_priors": False, "prior_decay": 0.0},
        )
        ridge_with_decay = run_backtest_with_params(
            2020, "president",
            {"use_year_adaptive_priors": False, "prior_decay": 0.9},
        )
        # Both use Ridge priors → identical results.
        assert ridge_no_decay["r"] == ridge_with_decay["r"]
        assert ridge_no_decay["rmse"] == ridge_with_decay["rmse"]
