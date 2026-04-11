"""Tests for the governor-specific Ridge model and prior routing.

Covers:
  - train_and_save() produces valid output with correct shape and range
  - Governor priors differ from presidential priors (they're trained differently)
  - load_county_priors_with_ridge_governor() falls back correctly
  - predict_2026_types.run() routes priors by race type
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.county_priors import (
    load_county_priors_with_ridge,
    load_county_priors_with_ridge_governor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_type_assignments(n_counties: int = 20, j: int = 5) -> pd.DataFrame:
    """Minimal type_assignments.parquet content for testing."""
    rng = np.random.default_rng(42)
    scores = rng.random((n_counties, j))
    scores = scores / scores.sum(axis=1, keepdims=True)  # row-normalize
    df = pd.DataFrame(
        scores,
        columns=[f"type_{i}_score" for i in range(j)],
    )
    fips = [str(i).zfill(5) for i in range(10001, 10001 + n_counties)]
    df.insert(0, "county_fips", fips)
    return df


def _make_demographics(county_fips: list[str]) -> pd.DataFrame:
    """Minimal county_features_national.parquet content for testing."""
    rng = np.random.default_rng(99)
    n = len(county_fips)
    df = pd.DataFrame({
        "county_fips": county_fips,
        "median_age": rng.uniform(25, 55, n),
        "pct_white": rng.uniform(0.2, 0.95, n),
        "pct_college": rng.uniform(0.1, 0.6, n),
    })
    return df


def _make_algara_governor(county_fips: list[str], year: int) -> pd.DataFrame:
    """Minimal Algara governor parquet content for testing."""
    rng = np.random.default_rng(year)
    n = len(county_fips)
    share = rng.uniform(0.30, 0.65, n)
    df = pd.DataFrame({
        "county_fips": county_fips,
        "state_abbr": ["TS"] * n,
        f"gov_dem_{year}": rng.uniform(1000, 10000, n),
        f"gov_rep_{year}": rng.uniform(1000, 10000, n),
        f"gov_total_{year}": rng.uniform(2000, 20000, n),
        f"gov_dem_share_{year}": share,
    })
    return df


def _make_medsl_2022_governor(county_fips: list[str]) -> pd.DataFrame:
    """Minimal MEDSL 2022 governor parquet for testing."""
    rng = np.random.default_rng(2022)
    n = len(county_fips)
    share = rng.uniform(0.30, 0.65, n)
    df = pd.DataFrame({
        "county_fips": county_fips,
        "state_abbr": ["TS"] * n,
        "gov_dem_2022": rng.uniform(1000, 10000, n),
        "gov_rep_2022": rng.uniform(1000, 10000, n),
        "gov_total_2022": rng.uniform(2000, 20000, n),
        "gov_dem_share_2022": share,
    })
    return df


# ---------------------------------------------------------------------------
# Tests for compute_county_historical_gov_mean
# ---------------------------------------------------------------------------

class TestComputeCountyHistoricalGovMean:
    def test_returns_mean_for_counties_with_data(self, tmp_path):
        """Counties with governor history should have their mean returned."""
        from src.prediction.train_ridge_model_governor import (
            compute_county_historical_gov_mean,
        )

        fips = ["10001", "10002", "10003"]
        for year in [2010, 2014, 2018]:
            df = _make_algara_governor(fips, year)
            df.to_parquet(tmp_path / f"algara_county_governor_{year}.parquet", index=False)

        means = compute_county_historical_gov_mean(fips, tmp_path, years=[2010, 2014, 2018])
        assert means.shape == (3,)
        # All counties have data in all three years, so mean should be != 0.45 fallback
        assert not np.all(means == 0.45)

    def test_fallback_for_counties_without_data(self, tmp_path):
        """Counties not in any governor file should fall back to 0.45."""
        from src.prediction.train_ridge_model_governor import (
            compute_county_historical_gov_mean,
        )

        # Write governor data for different FIPS than queried
        df = _make_algara_governor(["99999"], 2018)
        df.to_parquet(tmp_path / "algara_county_governor_2018.parquet", index=False)

        fips = ["10001", "10002"]
        means = compute_county_historical_gov_mean(fips, tmp_path, years=[2018])
        np.testing.assert_array_equal(means, [0.45, 0.45])

    def test_missing_year_file_skipped(self, tmp_path):
        """Missing parquet files are silently skipped (only available years used)."""
        from src.prediction.train_ridge_model_governor import (
            compute_county_historical_gov_mean,
        )

        fips = ["10001", "10002"]
        # Only write 2014, not 2010 or 2018
        df = _make_algara_governor(fips, 2014)
        df.to_parquet(tmp_path / "algara_county_governor_2014.parquet", index=False)

        means = compute_county_historical_gov_mean(fips, tmp_path, years=[2010, 2014, 2018])
        # Should still have values from 2014 (not fallback 0.45)
        assert means.shape == (2,)
        assert not np.all(means == 0.45)


# ---------------------------------------------------------------------------
# Tests for load_governor_target
# ---------------------------------------------------------------------------

class TestLoadGovernorTarget:
    def test_returns_nan_for_missing_counties(self, tmp_path):
        """Counties without 2022 governor data should return NaN."""
        from src.prediction.train_ridge_model_governor import load_governor_target

        # Write 2022 data for only some counties
        df = _make_medsl_2022_governor(["10001"])
        df.to_parquet(tmp_path / "medsl_county_2022_governor.parquet", index=False)

        y = load_governor_target(["10001", "99999"], tmp_path)
        assert not np.isnan(y[0])
        assert np.isnan(y[1])

    def test_values_in_valid_range(self, tmp_path):
        """Target values should be in [0, 1] range."""
        from src.prediction.train_ridge_model_governor import load_governor_target

        fips = [str(i).zfill(5) for i in range(10001, 10021)]
        df = _make_medsl_2022_governor(fips)
        df.to_parquet(tmp_path / "medsl_county_2022_governor.parquet", index=False)

        y = load_governor_target(fips, tmp_path)
        valid = y[~np.isnan(y)]
        assert np.all((valid >= 0.0) & (valid <= 1.0))


# ---------------------------------------------------------------------------
# Tests for train_and_save
# ---------------------------------------------------------------------------

class TestTrainAndSave:
    def _setup_training_data(self, tmp_dir: Path) -> tuple[Path, Path, Path, Path]:
        """Write minimal training fixtures and return path tuple."""
        assembled_dir = tmp_dir / "assembled"
        assembled_dir.mkdir()
        communities_dir = tmp_dir / "communities"
        communities_dir.mkdir()
        demo_dir = tmp_dir
        output_dir = tmp_dir / "output"

        n = 30
        fips = [str(i).zfill(5) for i in range(10001, 10001 + n)]

        # Type assignments
        ta_df = _make_type_assignments(n_counties=n, j=5)
        ta_df["county_fips"] = fips
        ta_path = communities_dir / "type_assignments.parquet"
        ta_df.to_parquet(ta_path, index=False)

        # Demographics
        demo_df = _make_demographics(fips)
        demo_path = demo_dir / "county_features_national.parquet"
        demo_df.to_parquet(demo_path, index=False)

        # Historical governor (required by compute_county_historical_gov_mean)
        for year in [2010, 2014, 2018]:
            gov_df = _make_algara_governor(fips, year)
            gov_df.to_parquet(assembled_dir / f"algara_county_governor_{year}.parquet", index=False)

        # 2022 governor target
        gov_2022 = _make_medsl_2022_governor(fips)
        gov_2022.to_parquet(assembled_dir / "medsl_county_2022_governor.parquet", index=False)

        return ta_path, demo_path, assembled_dir, output_dir

    def test_produces_parquet_and_json(self, tmp_path):
        """train_and_save() should produce ridge_county_priors_governor.parquet + ridge_meta.json."""
        from src.prediction.train_ridge_model_governor import train_and_save

        ta_path, demo_path, assembled_dir, output_dir = self._setup_training_data(tmp_path)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        assert (output_dir / "ridge_county_priors_governor.parquet").exists()
        assert (output_dir / "ridge_meta.json").exists()
        assert result["r2"] >= 0.0
        assert result["n_counties"] > 0

    def test_predictions_in_valid_range(self, tmp_path):
        """All predicted values should be clipped to [0, 1]."""
        from src.prediction.train_ridge_model_governor import train_and_save

        ta_path, demo_path, assembled_dir, output_dir = self._setup_training_data(tmp_path)
        train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        df = pd.read_parquet(output_dir / "ridge_county_priors_governor.parquet")
        assert "county_fips" in df.columns
        assert "ridge_pred_dem_share" in df.columns
        assert df["ridge_pred_dem_share"].between(0.0, 1.0).all()

    def test_meta_json_has_required_fields(self, tmp_path):
        """Metadata JSON should include key training parameters."""
        from src.prediction.train_ridge_model_governor import train_and_save

        ta_path, demo_path, assembled_dir, output_dir = self._setup_training_data(tmp_path)
        train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        meta = json.loads((output_dir / "ridge_meta.json").read_text())
        for key in ("alpha", "r2_train", "n_counties", "n_training_samples", "target"):
            assert key in meta, f"Missing key '{key}' in ridge_meta.json"
        assert meta["target"] == "gov_dem_share_2022"


# ---------------------------------------------------------------------------
# Tests for load_county_priors_with_ridge_governor (fallback logic)
# ---------------------------------------------------------------------------

class TestLoadCountyPriorsWithRidgeGovernor:
    def test_falls_back_to_presidential_when_governor_missing(self, tmp_path):
        """When the governor model file doesn't exist, should use presidential priors."""
        fips = ["10001", "10002", "10003"]
        pres_path = tmp_path / "ridge_county_priors.parquet"
        pres_df = pd.DataFrame({
            "county_fips": fips,
            "ridge_pred_dem_share": [0.55, 0.45, 0.60],
        })
        pres_df.to_parquet(pres_path, index=False)

        gov_path = tmp_path / "ridge_county_priors_governor.parquet"
        # Governor file does NOT exist

        # We need historical data for the fallback inside load_county_priors_with_ridge
        # Just patch compute_county_priors to return simple values
        with patch(
            "src.prediction.county_priors.compute_county_priors",
            return_value=np.array([0.40, 0.41, 0.42]),
        ):
            result = load_county_priors_with_ridge_governor(
                fips,
                governor_priors_path=gov_path,
                presidential_priors_path=pres_path,
            )

        # Presidential Ridge priors should be used since governor file missing
        np.testing.assert_allclose(result, [0.55, 0.45, 0.60])

    def test_governor_priors_blend_with_presidential(self, tmp_path):
        """Counties in the governor model get a blended prior (w=0.7 gov + 0.3 pres)."""
        fips = ["10001", "10002", "10003"]

        pres_path = tmp_path / "ridge_county_priors.parquet"
        pres_vals = [0.55, 0.45, 0.60]
        pd.DataFrame({
            "county_fips": fips,
            "ridge_pred_dem_share": pres_vals,
        }).to_parquet(pres_path, index=False)

        gov_path = tmp_path / "ridge_county_priors_governor.parquet"
        gov_vals = [0.48, 0.52, 0.44]
        pd.DataFrame({
            "county_fips": fips,
            "ridge_pred_dem_share": gov_vals,
        }).to_parquet(gov_path, index=False)

        with patch(
            "src.prediction.county_priors.compute_county_priors",
            return_value=np.array([0.40, 0.41, 0.42]),
        ):
            result = load_county_priors_with_ridge_governor(
                fips,
                governor_priors_path=gov_path,
                presidential_priors_path=pres_path,
            )

        w = 0.7
        expected = [w * g + (1 - w) * p for g, p in zip(gov_vals, pres_vals)]
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_partial_coverage_falls_back_per_county(self, tmp_path):
        """Counties missing from the governor model fall back to presidential."""
        fips = ["10001", "10002", "10003"]

        pres_path = tmp_path / "ridge_county_priors.parquet"
        pd.DataFrame({
            "county_fips": fips,
            "ridge_pred_dem_share": [0.55, 0.45, 0.60],
        }).to_parquet(pres_path, index=False)

        gov_path = tmp_path / "ridge_county_priors_governor.parquet"
        # Governor model only covers the first two counties
        pd.DataFrame({
            "county_fips": ["10001", "10002"],
            "ridge_pred_dem_share": [0.48, 0.52],
        }).to_parquet(gov_path, index=False)

        with patch(
            "src.prediction.county_priors.compute_county_priors",
            return_value=np.array([0.40, 0.41, 0.42]),
        ):
            result = load_county_priors_with_ridge_governor(
                fips,
                governor_priors_path=gov_path,
                presidential_priors_path=pres_path,
            )

        # 10001 and 10002 are blended (w=0.7 governor + 0.3 presidential)
        # 10003 missing from governor → pure presidential
        w = 0.7
        np.testing.assert_allclose(result[0], w * 0.48 + (1 - w) * 0.55, atol=1e-6)
        np.testing.assert_allclose(result[1], w * 0.52 + (1 - w) * 0.45, atol=1e-6)
        np.testing.assert_allclose(result[2], 0.60)


# ---------------------------------------------------------------------------
# Tests that governor priors differ from presidential priors
# ---------------------------------------------------------------------------

class TestGovernorVsPresidentialPriorsDiffer:
    def test_governor_and_presidential_priors_are_different(self, tmp_path):
        """Governor and presidential priors should differ when models produce different values.

        This verifies that the governor fallback chain actually uses governor values
        when they're available, rather than silently ignoring them.
        """
        fips = ["10001", "10002", "10003"]

        # Presidential priors at 0.55 / 0.45 / 0.60
        pres_path = tmp_path / "presidential.parquet"
        pd.DataFrame({
            "county_fips": fips,
            "ridge_pred_dem_share": [0.55, 0.45, 0.60],
        }).to_parquet(pres_path, index=False)

        # Governor priors at clearly different values: 0.50 / 0.40 / 0.55
        gov_path = tmp_path / "governor.parquet"
        pd.DataFrame({
            "county_fips": fips,
            "ridge_pred_dem_share": [0.50, 0.40, 0.55],
        }).to_parquet(gov_path, index=False)

        # Use side_effect to return a fresh array on each call, preventing
        # in-place mutation aliasing (load_county_priors_with_ridge mutates
        # the returned array in-place, so returning the same object twice
        # would cause result1 to be overwritten by the governor function's
        # subsequent modification).
        with patch(
            "src.prediction.county_priors.compute_county_priors",
            side_effect=lambda *a, **kw: np.array([0.40, 0.41, 0.42]),
        ):
            pres_priors = load_county_priors_with_ridge(
                fips, ridge_priors_path=pres_path
            )
            gov_priors = load_county_priors_with_ridge_governor(
                fips,
                governor_priors_path=gov_path,
                presidential_priors_path=pres_path,
            )

        # Governor priors should be blended (w=0.7 governor + 0.3 presidential)
        w = 0.7
        expected_blended = np.array([
            w * 0.50 + (1 - w) * 0.55,
            w * 0.40 + (1 - w) * 0.45,
            w * 0.55 + (1 - w) * 0.60,
        ])
        np.testing.assert_allclose(gov_priors, expected_blended, atol=1e-6)
        # Presidential priors should match the presidential parquet (0.55/0.45/0.60)
        np.testing.assert_allclose(pres_priors, [0.55, 0.45, 0.60])
        # The two sets should still differ
        assert not np.allclose(pres_priors, gov_priors)


# ---------------------------------------------------------------------------
# Tests for race-type prior routing in predict_2026_types
# ---------------------------------------------------------------------------

class TestPredictRaceTypePriorRouting:
    """Test that run() routes priors by race type without running the full pipeline."""

    def test_governor_races_receive_governor_priors(self):
        """Governor races should call run_forecast with county_prior_values_gov."""
        from unittest.mock import MagicMock, call

        # Build minimal mock inputs
        n_counties = 5
        j = 3
        type_scores = np.ones((n_counties, j)) / j
        pres_priors = np.full(n_counties, 0.50)
        gov_priors = np.full(n_counties, 0.45)  # Distinct from presidential

        from src.prediction.forecast_engine import ForecastResult

        j = 3
        mock_fr = ForecastResult(
            theta_prior=np.full(j, 0.45),
            theta_national=np.full(j, 0.45),
            delta_race=np.zeros(j),
            county_preds_national=pres_priors.copy(),
            county_preds_local=pres_priors.copy(),
            n_polls=0,
        )

        with (
            patch("src.prediction.predict_2026_types.load_type_data") as mock_load,
            patch(
                "src.prediction.predict_2026_types.load_county_priors_with_ridge",
                return_value=pres_priors,
            ),
            patch(
                "src.prediction.predict_2026_types.load_county_priors_with_ridge_governor",
                return_value=gov_priors,
            ),
            patch("src.prediction.predict_2026_types.load_county_metadata") as mock_meta,
            patch("src.prediction.predict_2026_types.load_county_votes") as mock_votes,
            patch("src.prediction.predict_2026_types.load_polls") as mock_polls,
            patch("src.prediction.predict_2026_types.compute_gb_shift") as mock_gb,
            patch("src.prediction.predict_2026_types.run_forecast") as mock_run_forecast,
        ):
            from src.assembly.define_races import Race

            mock_load.return_value = (
                [str(i).zfill(5) for i in range(10001, 10001 + n_counties)],
                type_scores,
                np.eye(j),
                np.full(j, 0.45),
            )
            mock_meta.return_value = (["OH"] * n_counties, [""] * n_counties)
            mock_votes.return_value = np.ones(n_counties)
            mock_polls.return_value = ({}, {})

            gb_info = MagicMock()
            gb_info.shift = 0.0
            gb_info.n_polls = 0
            gb_info.source = "test"
            mock_gb.return_value = gb_info

            # run_forecast returns {race_id: ForecastResult}
            mock_run_forecast.return_value = {"2026 OH Governor": mock_fr}

            with (
                patch(
                    "src.assembly.define_races.load_races",
                    return_value=[
                        Race("2026 OH Governor", "governor", "OH", 2026),
                        Race("2026 OH Senate", "senate", "OH", 2026),
                    ],
                ),
                patch("pandas.DataFrame.to_parquet"),
            ):
                from src.prediction.predict_2026_types import run

                # The test should not raise, and run_forecast should be called
                # with the governor priors for governor races.
                try:
                    run()
                except Exception:
                    pass  # Output write may fail in test; we check mock calls below

            # Verify that run_forecast was called at least once with governor priors
            calls = mock_run_forecast.call_args_list
            governor_call_used_gov_priors = any(
                "county_priors" in c.kwargs
                and np.array_equal(c.kwargs["county_priors"], gov_priors)
                for c in calls
            )
            assert governor_call_used_gov_priors, (
                "run_forecast was not called with governor priors for governor races. "
                f"Actual calls: {calls}"
            )
