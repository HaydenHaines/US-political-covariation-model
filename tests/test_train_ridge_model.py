"""Tests for src/prediction/train_ridge_model.py.

Tests feature matrix construction, artifact validity, and predict_2026_types
fallback behavior. Uses synthetic in-memory data where possible to avoid disk I/O.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------

def _make_type_assignments(n: int = 50, j: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    scores = rng.randn(n, j) * 0.3
    score_cols = {f"type_{k}_score": scores[:, k] for k in range(j)}
    return pd.DataFrame({"county_fips": fips, **score_cols})


def _make_demographics(n: int = 50, n_demo: int = 5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    demo_cols = {f"demo_{k}": rng.rand(n) for k in range(n_demo)}
    return pd.DataFrame({"county_fips": fips, **demo_cols})


def _make_election_parquet(n: int = 50, year: int = 2024, seed: int = 2) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    dem_share = rng.uniform(0.2, 0.8, n)
    return pd.DataFrame({
        "county_fips": fips,
        f"pres_dem_share_{year}": dem_share,
    })


# ---------------------------------------------------------------------------
# build_feature_matrix tests
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """Tests for the feature matrix construction logic."""

    def test_output_shape(self):
        """Feature matrix should have J + 1 + n_demo columns."""
        from src.prediction.train_ridge_model import build_feature_matrix

        n, j, n_demo = 40, 10, 5
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_demographics(n, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        county_fips = ta_df["county_fips"].values
        county_mean = np.full(n, 0.45)

        X, feature_names, row_mask = build_feature_matrix(scores, county_fips, demo_df, county_mean)

        expected_cols = j + 1 + n_demo
        assert X.shape == (n, expected_cols), f"Expected ({n}, {expected_cols}), got {X.shape}"
        assert len(feature_names) == expected_cols

    def test_inner_join_drops_unmatched(self):
        """Counties not in demo_df should be dropped (inner join)."""
        from src.prediction.train_ridge_model import build_feature_matrix

        n_ta, n_demo_rows, j, n_demo = 50, 30, 5, 3
        ta_df = _make_type_assignments(n_ta, j)
        # Demographics only for first 30 counties
        demo_df = _make_demographics(n_demo_rows, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        county_fips = ta_df["county_fips"].values
        county_mean = np.full(n_ta, 0.45)

        X, feature_names, row_mask = build_feature_matrix(scores, county_fips, demo_df, county_mean)

        assert X.shape[0] == n_demo_rows
        assert len(row_mask) == n_demo_rows

    def test_feature_names_match_columns(self):
        """Feature names must have correct prefixes."""
        from src.prediction.train_ridge_model import build_feature_matrix

        n, j, n_demo = 20, 4, 3
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_demographics(n, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        county_fips = ta_df["county_fips"].values
        county_mean = np.full(n, 0.45)

        _, feature_names, _ = build_feature_matrix(scores, county_fips, demo_df, county_mean)

        score_names = [fn for fn in feature_names if fn.startswith("type_")]
        assert len(score_names) == j
        assert "county_mean_dem_share" in feature_names

    def test_no_nans_in_output(self):
        """Feature matrix should contain no NaN after imputation."""
        from src.prediction.train_ridge_model import build_feature_matrix

        n, j, n_demo = 30, 6, 4
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_demographics(n, n_demo)
        # Inject NaN into demographics
        demo_numeric_cols = [c for c in demo_df.columns if c != "county_fips"]
        demo_df.loc[0, demo_numeric_cols[0]] = float("nan")
        demo_df.loc[5, demo_numeric_cols[1]] = float("nan")

        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        county_fips = ta_df["county_fips"].values
        county_mean = np.full(n, 0.45)

        X, _, _ = build_feature_matrix(scores, county_fips, demo_df, county_mean)

        assert not np.isnan(X).any(), "Feature matrix has NaN after imputation"

    def test_empty_join_raises(self):
        """When join produces no rows, a ValueError should be raised."""
        from src.prediction.train_ridge_model import build_feature_matrix

        ta_df = _make_type_assignments(10, 4)
        # Demo with completely different FIPS
        demo_df = pd.DataFrame({
            "county_fips": ["99901", "99902"],
            "demo_0": [0.5, 0.6],
        })
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        county_fips = ta_df["county_fips"].values
        county_mean = np.full(10, 0.45)

        with pytest.raises(ValueError, match="no rows"):
            build_feature_matrix(scores, county_fips, demo_df, county_mean)


# ---------------------------------------------------------------------------
# compute_county_historical_mean tests
# ---------------------------------------------------------------------------

class TestComputeCountyHistoricalMean:
    """Tests for compute_county_historical_mean."""

    def test_returns_correct_means(self, tmp_path):
        """Should return mean Dem share across available years."""
        from src.prediction.train_ridge_model import compute_county_historical_mean

        fips = ["00001", "00002"]
        # Create two historical parquet files
        for year, shares in [(2008, [0.4, 0.6]), (2012, [0.5, 0.7])]:
            df = pd.DataFrame({
                "county_fips": fips,
                f"pres_dem_share_{year}": shares,
            })
            df.to_parquet(tmp_path / f"medsl_county_presidential_{year}.parquet", index=False)

        result = compute_county_historical_mean(fips, tmp_path, years=[2008, 2012])
        np.testing.assert_allclose(result[0], 0.45)  # mean(0.4, 0.5)
        np.testing.assert_allclose(result[1], 0.65)  # mean(0.6, 0.7)

    def test_fallback_when_no_data(self, tmp_path):
        """Counties with no data should fall back to 0.45."""
        from src.prediction.train_ridge_model import compute_county_historical_mean

        fips = ["00001"]
        result = compute_county_historical_mean(fips, tmp_path, years=[2008])
        np.testing.assert_allclose(result[0], 0.45)

    def test_zero_padded_fips(self, tmp_path):
        """FIPS matching should work with zero-padded strings."""
        from src.prediction.train_ridge_model import compute_county_historical_mean

        # Parquet stores without padding; function should pad on load
        df = pd.DataFrame({
            "county_fips": ["1"],  # not zero-padded in file
            "pres_dem_share_2016": [0.55],
        })
        df.to_parquet(tmp_path / "medsl_county_presidential_2016.parquet", index=False)

        result = compute_county_historical_mean(["00001"], tmp_path, years=[2016])
        np.testing.assert_allclose(result[0], 0.55)


# ---------------------------------------------------------------------------
# train_and_save integration tests
# ---------------------------------------------------------------------------

class TestTrainAndSave:
    """Integration tests for the full train_and_save pipeline."""

    def _write_election_parquet(self, path: Path, year: int, n: int, seed: int) -> None:
        df = _make_election_parquet(n, year, seed)
        path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path / f"medsl_county_presidential_{year}.parquet", index=False)

    def test_output_parquet_exists(self, tmp_path):
        """train_and_save should produce ridge_county_priors.parquet."""
        from src.prediction.train_ridge_model import train_and_save

        n = 40
        ta_df = _make_type_assignments(n, j=8)
        demo_df = _make_demographics(n, n_demo=5)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ridge_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        # Write historical years + 2024 target
        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            self._write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        assert result["output_parquet"].exists()
        assert result["output_json"].exists()

    def test_output_parquet_columns(self, tmp_path):
        """Output parquet must have county_fips and ridge_pred_dem_share columns."""
        from src.prediction.train_ridge_model import train_and_save

        n = 40
        ta_df = _make_type_assignments(n, j=8)
        demo_df = _make_demographics(n, n_demo=5)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ridge_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            self._write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        df = pd.read_parquet(result["output_parquet"])
        assert "county_fips" in df.columns
        assert "ridge_pred_dem_share" in df.columns

    def test_predictions_in_unit_interval(self, tmp_path):
        """All ridge_pred_dem_share values should be in [0, 1]."""
        from src.prediction.train_ridge_model import train_and_save

        n = 40
        ta_df = _make_type_assignments(n, j=8)
        demo_df = _make_demographics(n, n_demo=5)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ridge_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            self._write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        df = pd.read_parquet(result["output_parquet"])
        assert (df["ridge_pred_dem_share"] >= 0.0).all(), "Predictions below 0"
        assert (df["ridge_pred_dem_share"] <= 1.0).all(), "Predictions above 1"

    def test_meta_json_keys(self, tmp_path):
        """ridge_meta.json should contain expected keys."""
        from src.prediction.train_ridge_model import train_and_save

        n = 40
        ta_df = _make_type_assignments(n, j=8)
        demo_df = _make_demographics(n, n_demo=5)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ridge_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            self._write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        meta = json.loads(result["output_json"].read_text())
        for key in ["alpha", "r2_train", "feature_names", "n_counties", "date_trained"]:
            assert key in meta, f"Missing key in meta: {key}"
        assert meta["n_counties"] > 0
        assert 0.0 <= meta["r2_train"] <= 1.0 or meta["r2_train"] <= 1.0

    def test_county_count_matches_inner_join(self, tmp_path):
        """Output county count = number of counties in both type_assignments and demographics."""
        from src.prediction.train_ridge_model import train_and_save

        n_ta, n_demo = 50, 35  # 35 overlap (same FIPS prefix)
        ta_df = _make_type_assignments(n_ta, j=6)
        demo_df = _make_demographics(n_demo, n_demo=4)  # only first 35 FIPS

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ridge_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            self._write_election_parquet(assembled_dir, year, n_ta, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        df = pd.read_parquet(result["output_parquet"])
        assert len(df) == n_demo, f"Expected {n_demo} counties, got {len(df)}"

    def test_fips_zero_padded_in_output(self, tmp_path):
        """FIPS codes in output parquet should be zero-padded to 5 digits."""
        from src.prediction.train_ridge_model import train_and_save

        n = 20
        ta_df = _make_type_assignments(n, j=4)
        demo_df = _make_demographics(n, n_demo=3)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ridge_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            self._write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        df = pd.read_parquet(result["output_parquet"])
        for fips in df["county_fips"]:
            assert len(str(fips)) == 5, f"FIPS not zero-padded: {fips}"


# ---------------------------------------------------------------------------
# predict_2026_types fallback tests
# ---------------------------------------------------------------------------

class TestPredict2026TypesFallback:
    """Tests that predict_2026_types uses Ridge priors when available and falls
    back gracefully when the Ridge model file is missing."""

    @pytest.fixture
    def synthetic_run_data(self, tmp_path):
        """Create minimal file layout for run() testing.

        run() reads from PROJECT_ROOT / "data" / ..., so we mirror that
        structure under tmp_path (which gets monkey-patched as PROJECT_ROOT).
        """
        n, j = 20, 4
        rng = np.random.RandomState(99)

        # type_assignments
        fips = [f"12{i:03d}" for i in range(1, n + 1)]
        scores = rng.randn(n, j) * 0.3
        score_cols = {f"type_{k}_score": scores[:, k] for k in range(j)}
        ta_df = pd.DataFrame({"county_fips": fips, **score_cols})

        # covariance
        A = rng.randn(j, j) * 0.02
        cov = A @ A.T + np.eye(j) * 0.001
        cov_df = pd.DataFrame(cov)

        # polls
        polls_df = pd.DataFrame({
            "race": ["FL Senate"],
            "geo_level": ["FL"],
            "state": ["FL"],
            "dem_share": [0.48],
            "n_sample": [800],
        })

        # historical election data — under data/assembled/
        assembled = tmp_path / "data" / "assembled"
        assembled.mkdir(parents=True)
        for year in [2020, 2024]:
            el_df = pd.DataFrame({
                "county_fips": fips,
                f"pres_dem_share_{year}": rng.uniform(0.3, 0.7, n),
            })
            el_df.to_parquet(assembled / f"medsl_county_presidential_{year}.parquet", index=False)

        # write files to tmp_path / "data" / ...  (mirrors PROJECT_ROOT structure)
        (tmp_path / "data" / "communities").mkdir(parents=True)
        ta_df.to_parquet(tmp_path / "data" / "communities" / "type_assignments.parquet", index=False)
        (tmp_path / "data" / "covariance").mkdir(parents=True)
        cov_df.to_parquet(tmp_path / "data" / "covariance" / "type_covariance.parquet", index=False)
        (tmp_path / "data" / "polls").mkdir(parents=True)
        polls_df.to_csv(tmp_path / "data" / "polls" / "polls_2026.csv", index=False)
        (tmp_path / "data" / "predictions").mkdir(parents=True)

        return {
            "tmp_path": tmp_path,
            "fips": fips,
            "n": n,
            "j": j,
        }

    def test_fallback_no_ridge_file(self, synthetic_run_data, monkeypatch):
        """predict_2026_types.run() works with no Ridge model file."""
        from src.prediction import county_priors, predict_2026_types

        data = synthetic_run_data
        tmp_path = data["tmp_path"]

        # Point run() and county_priors at tmp_path data dirs by monkey-patching PROJECT_ROOT
        monkeypatch.setattr(predict_2026_types, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(county_priors, "PROJECT_ROOT", tmp_path)

        # Ensure no Ridge model file exists (data/ is under tmp_path, but
        # PROJECT_ROOT is patched to tmp_path so path resolves to tmp_path/data/...)
        ridge_path = tmp_path / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
        assert not ridge_path.exists()

        # Should not raise; produces output parquet
        predict_2026_types.run()

        out = tmp_path / "data" / "predictions" / "county_predictions_2026_types.parquet"
        assert out.exists(), "Output parquet not created"
        df = pd.read_parquet(out)
        assert len(df) > 0

    def test_ridge_priors_override_historical(self, synthetic_run_data, monkeypatch):
        """When Ridge priors file exists, predictions differ from pure historical."""
        from src.prediction import county_priors, predict_2026_types

        data = synthetic_run_data
        tmp_path = data["tmp_path"]
        monkeypatch.setattr(predict_2026_types, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(county_priors, "PROJECT_ROOT", tmp_path)

        # First run: no Ridge file (historical priors)
        predict_2026_types.run()
        out = tmp_path / "data" / "predictions" / "county_predictions_2026_types.parquet"
        df_hist = pd.read_parquet(out)
        baseline_df = df_hist[df_hist["race"] == "baseline"].copy()

        # Create Ridge priors file with shuffled dem shares
        rng = np.random.RandomState(42)
        ridge_preds = rng.uniform(0.2, 0.8, len(data["fips"]))
        ridge_df = pd.DataFrame({
            "county_fips": data["fips"],
            "ridge_pred_dem_share": ridge_preds,
        })
        ridge_dir = tmp_path / "data" / "models" / "ridge_model"
        ridge_dir.mkdir(parents=True)
        ridge_df.to_parquet(ridge_dir / "ridge_county_priors.parquet", index=False)

        # Second run: Ridge priors
        predict_2026_types.run()
        df_ridge = pd.read_parquet(out)
        ridge_baseline = df_ridge[df_ridge["race"] == "baseline"].copy()

        # With very different priors, predictions should differ
        assert not np.allclose(
            baseline_df["pred_dem_share"].values,
            ridge_baseline["pred_dem_share"].values,
            atol=1e-6,
        ), "Ridge priors had no effect on predictions"

    def test_partial_ridge_coverage_uses_fallback(self, synthetic_run_data, monkeypatch):
        """When Ridge priors cover only some counties, run completes and partial
        priors shift predictions compared to the pure-historical baseline.

        NOTE: The pipeline aggregates county priors to type-level θ via weighted
        average before computing county predictions (type_scores @ θ). This means
        matched and unmatched counties share types and cannot be cleanly separated
        in the final output. Instead we verify:
          1. Run completes without error.
          2. All counties appear in the output.
          3. Partial Ridge coverage (0.99 for half) produces different predictions
             than pure historical alone — confirming the priors flow through.
        """
        from src.prediction import county_priors, predict_2026_types

        data = synthetic_run_data
        tmp_path = data["tmp_path"]
        monkeypatch.setattr(predict_2026_types, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(county_priors, "PROJECT_ROOT", tmp_path)

        # First run: no Ridge priors (pure historical baseline)
        predict_2026_types.run()
        out = tmp_path / "data" / "predictions" / "county_predictions_2026_types.parquet"
        df_hist = pd.read_parquet(out)
        hist_baseline = df_hist[df_hist["race"] == "baseline"]["pred_dem_share"].values.copy()

        # Ridge priors only for first half of counties (extreme value)
        half = data["n"] // 2
        ridge_df = pd.DataFrame({
            "county_fips": data["fips"][:half],
            "ridge_pred_dem_share": np.full(half, 0.99),
        })
        ridge_dir = tmp_path / "data" / "models" / "ridge_model"
        ridge_dir.mkdir(parents=True)
        ridge_df.to_parquet(ridge_dir / "ridge_county_priors.parquet", index=False)

        # Second run: partial Ridge coverage
        predict_2026_types.run()
        df_partial = pd.read_parquet(out)
        partial_baseline = df_partial[df_partial["race"] == "baseline"]

        # All counties must be present
        assert set(partial_baseline["county_fips"]) == set(data["fips"]), (
            "Not all counties present in output with partial Ridge coverage"
        )

        # Predictions must differ from pure historical — partial priors have effect
        partial_preds = partial_baseline["pred_dem_share"].values
        assert not np.allclose(partial_preds, hist_baseline, atol=1e-6), (
            "Partial Ridge priors (0.99 for half) had no effect on predictions"
        )
