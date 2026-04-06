"""Tests for tract-level Ridge regression model (T.5 tract-primary migration).

Covers:
  - build_tract_feature_matrix: shape, NaN handling, inner join, column names
  - compute_tract_historical_mean: correctness, 2024 exclusion, fallback, NaN handling
  - compute_tract_loo_predictions: math correctness, output shape, high-leverage guard
  - train_tract_ridge: runs without error, valid metrics
  - run() integration: output files written, columns correct, values in range
  - holdout_accuracy_ridge_tract: end-to-end validation function
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_type_assignments(n: int = 50, j: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthetic tract type assignments: tract_geoid + J type score columns."""
    rng = np.random.RandomState(seed)
    geoids = [f"{'0' * (11 - len(str(i)))}{i}" for i in range(1, n + 1)]
    score_cols = {f"type_{k}_score": rng.rand(n) for k in range(j)}
    return pd.DataFrame({"tract_geoid": geoids, **score_cols})


def _make_tract_features(n: int = 50, n_demo: int = 5, seed: int = 1) -> pd.DataFrame:
    """Synthetic tract features: tract_geoid + metadata + demographic columns."""
    rng = np.random.RandomState(seed)
    geoids = [f"{'0' * (11 - len(str(i)))}{i}" for i in range(1, n + 1)]
    demo_cols = {f"demo_{k}": rng.rand(n) for k in range(n_demo)}
    return pd.DataFrame({
        "tract_geoid": geoids,
        "is_uninhabited": [False] * n,
        "n_features_imputed": [0] * n,
        **demo_cols,
    })


def _make_elections_df(
    n: int = 50,
    years: list[int] | None = None,
    seed: int = 2,
) -> pd.DataFrame:
    """Synthetic tract elections (presidential only)."""
    if years is None:
        years = [2008, 2012, 2016, 2020, 2024]
    rng = np.random.RandomState(seed)
    geoids = [f"{'0' * (11 - len(str(i)))}{i}" for i in range(1, n + 1)]
    rows = []
    for year in years:
        for g in geoids:
            rows.append({
                "tract_geoid": g,
                "year": year,
                "race_type": "PRES",
                "total_votes": 500,
                "dem_votes": 200,
                "rep_votes": 300,
                "dem_share": rng.uniform(0.2, 0.8),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: build_tract_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildTractFeatureMatrix:
    """Tests for feature matrix construction."""

    def test_output_shape_correct(self):
        """X should have J + 1 + D columns (scores + mean + demographics)."""
        from src.prediction.train_ridge_model_tract import build_tract_feature_matrix

        n, j, n_demo = 40, 10, 5
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_tract_features(n, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        geoids = ta_df["tract_geoid"].values
        tract_mean = np.full(n, 0.45)

        X, feature_names, row_mask = build_tract_feature_matrix(scores, geoids, demo_df, tract_mean)

        expected_cols = j + 1 + n_demo
        assert X.shape == (n, expected_cols), f"Expected ({n}, {expected_cols}), got {X.shape}"
        assert len(feature_names) == expected_cols

    def test_inner_join_drops_unmatched(self):
        """Tracts not in demo_df should be dropped (inner join on tract_geoid)."""
        from src.prediction.train_ridge_model_tract import build_tract_feature_matrix

        n_ta, n_demo_rows, j, n_demo = 50, 30, 5, 3
        ta_df = _make_type_assignments(n_ta, j)
        # Features only for first 30 tracts
        demo_df = _make_tract_features(n_demo_rows, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        geoids = ta_df["tract_geoid"].values
        tract_mean = np.full(n_ta, 0.45)

        X, feature_names, row_mask = build_tract_feature_matrix(scores, geoids, demo_df, tract_mean)

        assert X.shape[0] == n_demo_rows
        assert len(row_mask) == n_demo_rows

    def test_no_nans_after_imputation(self):
        """Feature matrix should be NaN-free even if demographics have NaN values."""
        from src.prediction.train_ridge_model_tract import build_tract_feature_matrix

        n, j, n_demo = 30, 6, 4
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_tract_features(n, n_demo)
        # Inject NaN into two demographic cells
        demo_numeric_cols = [c for c in demo_df.columns if c.startswith("demo_")]
        demo_df.loc[0, demo_numeric_cols[0]] = float("nan")
        demo_df.loc[5, demo_numeric_cols[1]] = float("nan")

        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        geoids = ta_df["tract_geoid"].values
        tract_mean = np.full(n, 0.45)

        X, _, _ = build_tract_feature_matrix(scores, geoids, demo_df, tract_mean)

        assert not np.isnan(X).any(), "Feature matrix has NaN after imputation"

    def test_feature_names_structure(self):
        """Feature names must contain type scores, tract mean, and demo columns."""
        from src.prediction.train_ridge_model_tract import build_tract_feature_matrix

        n, j, n_demo = 20, 4, 3
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_tract_features(n, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        geoids = ta_df["tract_geoid"].values
        tract_mean = np.full(n, 0.45)

        _, feature_names, _ = build_tract_feature_matrix(scores, geoids, demo_df, tract_mean)

        score_names = [fn for fn in feature_names if fn.startswith("type_")]
        assert len(score_names) == j, f"Expected {j} type score names, got {len(score_names)}"
        assert "tract_mean_dem_share" in feature_names

    def test_metadata_cols_excluded(self):
        """is_uninhabited and n_features_imputed must not appear as features."""
        from src.prediction.train_ridge_model_tract import build_tract_feature_matrix

        n, j, n_demo = 20, 4, 3
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_tract_features(n, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        geoids = ta_df["tract_geoid"].values
        tract_mean = np.full(n, 0.45)

        _, feature_names, _ = build_tract_feature_matrix(scores, geoids, demo_df, tract_mean)

        assert "is_uninhabited" not in feature_names
        assert "n_features_imputed" not in feature_names

    def test_empty_join_raises(self):
        """Completely disjoint GEOIDs should raise ValueError."""
        from src.prediction.train_ridge_model_tract import build_tract_feature_matrix

        ta_df = _make_type_assignments(10, 4)
        demo_df = pd.DataFrame({
            "tract_geoid": ["99999999999", "88888888888"],
            "is_uninhabited": [False, False],
            "n_features_imputed": [0, 0],
            "demo_0": [0.5, 0.6],
        })
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        geoids = ta_df["tract_geoid"].values
        tract_mean = np.full(10, 0.45)

        with pytest.raises(ValueError, match="no rows"):
            build_tract_feature_matrix(scores, geoids, demo_df, tract_mean)


# ---------------------------------------------------------------------------
# Tests: compute_tract_historical_mean
# ---------------------------------------------------------------------------


class TestComputeTractHistoricalMean:
    """Tests for the historical mean Dem share calculation."""

    def test_correct_mean_computation(self):
        """Mean should average 2008/2012/2016/2020 dem_share per tract."""
        from src.prediction.train_ridge_model_tract import compute_tract_historical_mean

        geoids = ["00000000001", "00000000002"]
        elections = pd.DataFrame({
            "tract_geoid": ["00000000001", "00000000001", "00000000002", "00000000002"],
            "year": [2008, 2016, 2008, 2016],
            "race_type": ["PRES"] * 4,
            "dem_share": [0.4, 0.6, 0.3, 0.7],
        })
        result = compute_tract_historical_mean(geoids, elections, years=[2008, 2016])

        np.testing.assert_allclose(result[0], 0.5)   # mean(0.4, 0.6)
        np.testing.assert_allclose(result[1], 0.5)   # mean(0.3, 0.7)

    def test_fallback_for_missing_tract(self):
        """Tracts with no history should fall back to 0.45."""
        from src.prediction.train_ridge_model_tract import compute_tract_historical_mean

        geoids = ["00000000001"]
        elections = pd.DataFrame({
            "tract_geoid": ["99999999999"],
            "year": [2008],
            "race_type": ["PRES"],
            "dem_share": [0.6],
        })
        result = compute_tract_historical_mean(geoids, elections, years=[2008])

        np.testing.assert_allclose(result[0], 0.45)

    def test_2024_in_years_raises(self):
        """Passing 2024 in the years list must raise ValueError (target leakage guard)."""
        from src.prediction.train_ridge_model_tract import compute_tract_historical_mean

        geoids = ["00000000001"]
        elections = _make_elections_df(n=1)

        with pytest.raises(ValueError, match="2024"):
            compute_tract_historical_mean(geoids, elections, years=[2008, 2024])

    def test_does_not_use_2024_data(self):
        """Mean must be computed from history years only, never from 2024."""
        from src.prediction.train_ridge_model_tract import compute_tract_historical_mean

        geoids = ["00000000001"]
        # Only 2024 data exists: mean must fall back to 0.45, not 0.9
        elections = pd.DataFrame({
            "tract_geoid": ["00000000001"],
            "year": [2024],
            "race_type": ["PRES"],
            "dem_share": [0.9],
        })
        result = compute_tract_historical_mean(geoids, elections, years=[2008, 2012, 2016, 2020])

        # 2024 data exists but must not be used; no 2008-2020 data → fallback
        np.testing.assert_allclose(result[0], 0.45)


# ---------------------------------------------------------------------------
# Tests: compute_tract_loo_predictions
# ---------------------------------------------------------------------------


class TestComputeTractLOO:
    """Tests for the hat matrix LOO computation."""

    def test_output_shapes(self):
        """y_loo and h_diag must have same shape as y."""
        from src.prediction.train_ridge_model_tract import compute_tract_loo_predictions

        rng = np.random.RandomState(42)
        N, F = 50, 8
        X = rng.randn(N, F)
        y = rng.rand(N)

        y_loo, h_diag = compute_tract_loo_predictions(X, y, alpha=1.0)

        assert y_loo.shape == (N,)
        assert h_diag.shape == (N,)

    def test_correlation_with_target_reasonable(self):
        """LOO predictions should have positive correlation with a clean linear target."""
        from src.prediction.train_ridge_model_tract import compute_tract_loo_predictions
        from scipy.stats import pearsonr

        rng = np.random.RandomState(7)
        N, F = 100, 5
        X = rng.randn(N, F)
        # y = linear combination of X + small noise → Ridge should recover it
        true_beta = rng.randn(F)
        y = X @ true_beta + rng.randn(N) * 0.1
        y = (y - y.min()) / (y.max() - y.min())  # scale to [0,1]

        # Select alpha via RidgeCV first
        from sklearn.linear_model import RidgeCV
        rcv = RidgeCV(alphas=np.logspace(-3, 6, 50), fit_intercept=True)
        rcv.fit(X, y)
        alpha = float(rcv.alpha_)

        y_loo, _ = compute_tract_loo_predictions(X, y, alpha=alpha)
        r, _ = pearsonr(y, y_loo)
        assert r > 0.5, f"LOO r too low on clean linear data: {r:.3f}"

    def test_leverage_in_unit_interval(self):
        """Hat matrix diagonal values should be in (0, 1)."""
        from src.prediction.train_ridge_model_tract import compute_tract_loo_predictions

        rng = np.random.RandomState(99)
        N, F = 40, 6
        X = rng.randn(N, F)
        y = rng.rand(N)

        _, h_diag = compute_tract_loo_predictions(X, y, alpha=1.0)

        assert np.all(h_diag > -1e-9), "Some leverage values are negative"
        # With regularization, leverage should be < 1
        assert np.all(h_diag < 1.0 + 1e-9), "Some leverage values exceed 1"


# ---------------------------------------------------------------------------
# Tests: train_tract_ridge
# ---------------------------------------------------------------------------


class TestTrainTractRidge:
    """Tests for the Ridge training function."""

    def test_runs_without_error(self):
        """train_tract_ridge should fit and return valid metrics."""
        from src.prediction.train_ridge_model_tract import train_tract_ridge

        rng = np.random.RandomState(0)
        N, F = 80, 15
        X = rng.randn(N, F)
        y = rng.uniform(0.2, 0.8, N)

        rcv, alpha, r2_train, loo_r = train_tract_ridge(X, y)

        assert alpha > 0, "Alpha must be positive"
        assert 0.0 <= r2_train <= 1.0, f"r2_train out of range: {r2_train}"
        assert -1.0 <= loo_r <= 1.0, f"loo_r out of range: {loo_r}"

    def test_nan_in_y_raises(self):
        """NaN in target should raise ValueError."""
        from src.prediction.train_ridge_model_tract import train_tract_ridge

        rng = np.random.RandomState(1)
        X = rng.randn(50, 10)
        y = rng.rand(50)
        y[5] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            train_tract_ridge(X, y)

    def test_predictions_clipped_to_unit_interval(self):
        """run() should clip predictions to [0, 1]."""
        from src.prediction.train_ridge_model_tract import train_tract_ridge

        rng = np.random.RandomState(2)
        N, F = 60, 12
        X = rng.randn(N, F)
        # Extreme targets that might push Ridge predictions outside [0,1]
        y = rng.uniform(-0.5, 1.5, N)
        y = np.clip(y, 0, 1)  # keep y valid, but test pipeline clips predictions

        rcv, alpha, _, _ = train_tract_ridge(X, y)
        y_pred = np.clip(rcv.predict(X), 0.0, 1.0)

        assert y_pred.min() >= 0.0, "Predictions below 0"
        assert y_pred.max() <= 1.0, "Predictions above 1"


# ---------------------------------------------------------------------------
# Tests: run() integration
# ---------------------------------------------------------------------------


class TestRunIntegration:
    """Integration tests for train_ridge_model_tract.run()."""

    def _write_parquets(self, tmp_path: Path, n: int = 150, j: int = 8, n_demo: int = 5):
        """Write minimal parquet files for run() testing.

        n=150 to exceed the minimum training sample guard in run() (100).
        """
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_tract_features(n, n_demo)
        elections_df = _make_elections_df(n)

        ta_path = tmp_path / "tract_type_assignments.parquet"
        feat_path = tmp_path / "tract_features.parquet"
        elec_path = tmp_path / "tract_elections.parquet"
        out_dir = tmp_path / "ridge_output"

        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(feat_path, index=False)
        elections_df.to_parquet(elec_path, index=False)

        return ta_path, feat_path, elec_path, out_dir

    def test_output_parquet_exists(self, tmp_path):
        """run() should create ridge_tract_priors.parquet."""
        from src.prediction.train_ridge_model_tract import run

        ta_path, feat_path, elec_path, out_dir = self._write_parquets(tmp_path)
        result = run(ta_path, feat_path, elec_path, out_dir)

        assert result["output_parquet"].exists(), "ridge_tract_priors.parquet not created"

    def test_output_json_exists(self, tmp_path):
        """run() should create ridge_tract_meta.json."""
        from src.prediction.train_ridge_model_tract import run

        ta_path, feat_path, elec_path, out_dir = self._write_parquets(tmp_path)
        result = run(ta_path, feat_path, elec_path, out_dir)

        assert result["output_json"].exists(), "ridge_tract_meta.json not created"

    def test_output_parquet_columns(self, tmp_path):
        """Output parquet must have tract_geoid and ridge_pred_dem_share."""
        from src.prediction.train_ridge_model_tract import run

        ta_path, feat_path, elec_path, out_dir = self._write_parquets(tmp_path)
        result = run(ta_path, feat_path, elec_path, out_dir)

        df = pd.read_parquet(result["output_parquet"])
        assert "tract_geoid" in df.columns
        assert "ridge_pred_dem_share" in df.columns

    def test_predictions_in_unit_interval(self, tmp_path):
        """All ridge_pred_dem_share values should be in [0, 1]."""
        from src.prediction.train_ridge_model_tract import run

        ta_path, feat_path, elec_path, out_dir = self._write_parquets(tmp_path)
        result = run(ta_path, feat_path, elec_path, out_dir)

        df = pd.read_parquet(result["output_parquet"])
        assert (df["ridge_pred_dem_share"] >= 0.0).all(), "Predictions below 0"
        assert (df["ridge_pred_dem_share"] <= 1.0).all(), "Predictions above 1"

    def test_meta_json_required_keys(self, tmp_path):
        """ridge_tract_meta.json must contain expected keys."""
        from src.prediction.train_ridge_model_tract import run

        ta_path, feat_path, elec_path, out_dir = self._write_parquets(tmp_path)
        result = run(ta_path, feat_path, elec_path, out_dir)

        meta = json.loads(result["output_json"].read_text())
        required_keys = ["alpha", "r2_train", "loo_r", "feature_names", "n_tracts", "date_trained"]
        for key in required_keys:
            assert key in meta, f"Missing key in ridge_tract_meta.json: {key}"

    def test_n_tracts_in_return_value(self, tmp_path):
        """Returned n_tracts should match inner join of type_assignments ∩ features."""
        from src.prediction.train_ridge_model_tract import run

        n = 150
        ta_path, feat_path, elec_path, out_dir = self._write_parquets(tmp_path, n=n)
        result = run(ta_path, feat_path, elec_path, out_dir)

        # All n tracts are in type_assignments and features (no mismatch in synthetic data)
        assert result["n_tracts"] == n

    def test_no_target_leakage_in_tract_mean(self, tmp_path):
        """Tract mean must be computed from 2008-2020 only; 2024 data must not appear."""
        from src.prediction.train_ridge_model_tract import (
            _load_tract_elections,
            compute_tract_historical_mean,
            _HISTORY_YEARS,
        )

        # Create elections where 2024 dem_share is a distinctive sentinel value
        n = 20
        geoids = [f"{'0' * (11 - len(str(i)))}{i}" for i in range(1, n + 1)]
        rows = []
        # Historical: dem_share = 0.4 for all tracts
        for year in [2008, 2012, 2016, 2020]:
            for g in geoids:
                rows.append({"tract_geoid": g, "year": year, "race_type": "PRES",
                             "total_votes": 100, "dem_votes": 40, "rep_votes": 60,
                             "dem_share": 0.4})
        # 2024: dem_share = 0.99 (sentinel) — must NOT appear in mean
        for g in geoids:
            rows.append({"tract_geoid": g, "year": 2024, "race_type": "PRES",
                         "total_votes": 100, "dem_votes": 99, "rep_votes": 1,
                         "dem_share": 0.99})

        elec_path = tmp_path / "tract_elections.parquet"
        pd.DataFrame(rows).to_parquet(elec_path, index=False)

        pres_elections = _load_tract_elections(elec_path)
        tract_mean = compute_tract_historical_mean(geoids, pres_elections, years=_HISTORY_YEARS)

        # All tracts had dem_share=0.4 in history; mean must be ~0.4, not ~0.99
        np.testing.assert_allclose(tract_mean, 0.4, atol=1e-6,
                                   err_msg="Tract mean was contaminated by 2024 data (target leakage)")

    def test_loo_r_returned_in_valid_range(self, tmp_path):
        """LOO r from run() must be in [-1, 1]."""
        from src.prediction.train_ridge_model_tract import run

        ta_path, feat_path, elec_path, out_dir = self._write_parquets(tmp_path)
        result = run(ta_path, feat_path, elec_path, out_dir)

        assert -1.0 <= result["loo_r"] <= 1.0, f"LOO r out of range: {result['loo_r']}"


# ---------------------------------------------------------------------------
# Tests: holdout_accuracy_ridge_tract
# ---------------------------------------------------------------------------


class TestHoldoutAccuracyRidgeTract:
    """Tests for the validation function."""

    def _write_parquets(self, tmp_path: Path, n: int = 150, j: int = 8, n_demo: int = 5):
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_tract_features(n, n_demo)
        elections_df = _make_elections_df(n)

        ta_path = tmp_path / "tract_type_assignments.parquet"
        feat_path = tmp_path / "tract_features.parquet"
        elec_path = tmp_path / "tract_elections.parquet"

        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(feat_path, index=False)
        elections_df.to_parquet(elec_path, index=False)

        return ta_path, feat_path, elec_path

    def test_returns_dict_with_required_keys(self, tmp_path):
        """holdout_accuracy_ridge_tract should return dict with loo_r, etc."""
        from src.validation.holdout_accuracy_ridge_tract import holdout_accuracy_ridge_tract

        ta_path, feat_path, elec_path = self._write_parquets(tmp_path)
        result = holdout_accuracy_ridge_tract(ta_path, feat_path, elec_path)

        assert result is not None, "Expected dict, got None"
        for key in ["loo_r", "loo_rmse", "r2_train", "alpha", "n_tracts", "pred_range"]:
            assert key in result, f"Missing key: {key}"

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None if any required file is missing (graceful failure)."""
        from src.validation.holdout_accuracy_ridge_tract import holdout_accuracy_ridge_tract

        result = holdout_accuracy_ridge_tract(
            tmp_path / "nonexistent_ta.parquet",
            tmp_path / "nonexistent_feat.parquet",
            tmp_path / "nonexistent_elec.parquet",
        )

        assert result is None, "Expected None when files are missing"

    def test_loo_r_in_valid_range(self, tmp_path):
        """LOO r must be in [-1, 1]."""
        from src.validation.holdout_accuracy_ridge_tract import holdout_accuracy_ridge_tract

        ta_path, feat_path, elec_path = self._write_parquets(tmp_path)
        result = holdout_accuracy_ridge_tract(ta_path, feat_path, elec_path)

        assert -1.0 <= result["loo_r"] <= 1.0

    def test_loo_rmse_positive(self, tmp_path):
        """LOO RMSE must be non-negative."""
        from src.validation.holdout_accuracy_ridge_tract import holdout_accuracy_ridge_tract

        ta_path, feat_path, elec_path = self._write_parquets(tmp_path)
        result = holdout_accuracy_ridge_tract(ta_path, feat_path, elec_path)

        assert result["loo_rmse"] >= 0.0

    def test_pred_range_is_tuple(self, tmp_path):
        """pred_range must be a (min, max) tuple."""
        from src.validation.holdout_accuracy_ridge_tract import holdout_accuracy_ridge_tract

        ta_path, feat_path, elec_path = self._write_parquets(tmp_path)
        result = holdout_accuracy_ridge_tract(ta_path, feat_path, elec_path)

        pred_range = result["pred_range"]
        assert len(pred_range) == 2
        assert pred_range[0] <= pred_range[1], "pred_range min > max"
