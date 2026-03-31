"""Tests for src/prediction/train_ensemble_model.py.

Tests module import, feature matrix construction (via shared train_ridge_model
helpers), ensemble prediction value range, and metadata completeness.
Uses synthetic in-memory data to avoid disk I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — synthetic data factories (mirrors test_train_ridge_model.py)
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


def _write_election_parquet(path: Path, year: int, n: int, seed: int) -> None:
    df = _make_election_parquet(n, year, seed)
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / f"medsl_county_presidential_{year}.parquet", index=False)


# ---------------------------------------------------------------------------
# Test: module import
# ---------------------------------------------------------------------------

class TestModuleImport:
    """Module and its public API must be importable."""

    def test_import_train_ensemble_model(self):
        """Module imports without errors."""
        import src.prediction.train_ensemble_model  # noqa: F401

    def test_train_and_save_callable(self):
        """train_and_save function is importable and callable."""
        from src.prediction.train_ensemble_model import train_and_save
        assert callable(train_and_save)

    def test_hgb_params_defined(self):
        """HGB_PARAMS constant is defined and has expected keys."""
        from src.prediction.train_ensemble_model import HGB_PARAMS
        for key in ("max_iter", "learning_rate", "max_depth", "min_samples_leaf",
                    "l2_regularization", "random_state"):
            assert key in HGB_PARAMS, f"HGB_PARAMS missing key: {key}"


# ---------------------------------------------------------------------------
# Test: build_feature_matrix (imported from train_ridge_model)
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """Ensemble model reuses build_feature_matrix from train_ridge_model."""

    def test_feature_matrix_expected_shape(self):
        """build_feature_matrix produces (n_matched, J + 1 + n_demo) matrix."""
        from src.prediction.train_ridge_model import build_feature_matrix

        n, j, n_demo = 40, 8, 5
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_demographics(n, n_demo)
        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        county_fips = ta_df["county_fips"].values
        county_mean = np.full(n, 0.45)

        X, feature_names, row_mask = build_feature_matrix(scores, county_fips, demo_df, county_mean)

        expected_cols = j + 1 + n_demo
        assert X.shape == (n, expected_cols), f"Expected ({n}, {expected_cols}), got {X.shape}"
        assert len(feature_names) == expected_cols

    def test_feature_matrix_no_nans(self):
        """Feature matrix has no NaN after imputation."""
        from src.prediction.train_ridge_model import build_feature_matrix

        n, j, n_demo = 30, 6, 4
        ta_df = _make_type_assignments(n, j)
        demo_df = _make_demographics(n, n_demo)
        # Inject NaN
        demo_numeric_cols = [c for c in demo_df.columns if c != "county_fips"]
        demo_df.loc[0, demo_numeric_cols[0]] = float("nan")

        scores = ta_df[[c for c in ta_df.columns if c.endswith("_score")]].values.astype(float)
        county_fips = ta_df["county_fips"].values
        county_mean = np.full(n, 0.45)

        X, _, _ = build_feature_matrix(scores, county_fips, demo_df, county_mean)
        assert not np.isnan(X).any()


# ---------------------------------------------------------------------------
# Test: ensemble prediction values in [0, 1]
# ---------------------------------------------------------------------------

class TestEnsemblePredictions:
    """Ensemble predictions must stay in [0, 1] after clipping."""

    def test_predictions_in_unit_interval(self, tmp_path):
        """All ridge_pred_dem_share values must be in [0, 1]."""
        from src.prediction.train_ensemble_model import train_and_save

        n = 50
        ta_df = _make_type_assignments(n, j=8)
        demo_df = _make_demographics(n, n_demo=5)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ensemble_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            _write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        df = pd.read_parquet(result["output_parquet"])
        assert (df["ridge_pred_dem_share"] >= 0.0).all(), "Ensemble prediction below 0"
        assert (df["ridge_pred_dem_share"] <= 1.0).all(), "Ensemble prediction above 1"

    def test_output_columns_backward_compatible(self, tmp_path):
        """Output parquet must have county_fips and ridge_pred_dem_share columns."""
        from src.prediction.train_ensemble_model import train_and_save

        n = 40
        ta_df = _make_type_assignments(n, j=6)
        demo_df = _make_demographics(n, n_demo=4)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ensemble_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            _write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        df = pd.read_parquet(result["output_parquet"])
        assert "county_fips" in df.columns, "Missing county_fips column"
        assert "ridge_pred_dem_share" in df.columns, "Missing ridge_pred_dem_share column"

    def test_ensemble_blends_differently_than_ridge_alone(self, tmp_path):
        """Ensemble predictions differ from Ridge-only predictions."""
        from src.prediction.train_ensemble_model import train_and_save as ensemble_train
        from src.prediction.train_ridge_model import train_and_save as ridge_train

        n = 50
        ta_df = _make_type_assignments(n, j=8, seed=7)
        demo_df = _make_demographics(n, n_demo=5, seed=8)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            _write_election_parquet(assembled_dir, year, n, seed)

        ridge_out = tmp_path / "ridge_output"
        ensemble_out = tmp_path / "ensemble_output"

        ridge_result = ridge_train(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=ridge_out,
        )
        ensemble_result = ensemble_train(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=ensemble_out,
        )

        ridge_df = pd.read_parquet(ridge_result["output_parquet"])
        ensemble_df = pd.read_parquet(ensemble_result["output_parquet"])

        merged = ridge_df.merge(ensemble_df, on="county_fips", suffixes=("_ridge", "_ensemble"))
        diffs = np.abs(
            merged["ridge_pred_dem_share_ridge"].values
            - merged["ridge_pred_dem_share_ensemble"].values
        )
        assert diffs.max() > 1e-6, "Ensemble predictions are identical to Ridge-only (unexpected)"


# ---------------------------------------------------------------------------
# Test: metadata includes ensemble fields
# ---------------------------------------------------------------------------

class TestEnsembleMetadata:
    """ridge_meta.json must include all ensemble-specific fields."""

    def test_metadata_ensemble_keys(self, tmp_path):
        """meta.json must contain ensemble-specific keys."""
        from src.prediction.train_ensemble_model import train_and_save

        n = 40
        ta_df = _make_type_assignments(n, j=6)
        demo_df = _make_demographics(n, n_demo=4)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ensemble_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            _write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        meta = json.loads(result["output_json"].read_text())

        # Ensemble-specific fields
        for key in ("r2_ridge", "r2_hgb", "ensemble_method",
                    "ensemble_ridge_weight", "ensemble_hgb_weight", "hgb_params"):
            assert key in meta, f"Missing ensemble key in meta: {key}"

        # Backward-compat fields
        for key in ("alpha", "r2_train", "feature_names", "n_counties", "date_trained"):
            assert key in meta, f"Missing backward-compat key in meta: {key}"

        # Values sanity
        assert meta["ensemble_ridge_weight"] == pytest.approx(0.5)
        assert meta["ensemble_hgb_weight"] == pytest.approx(0.5)
        assert meta["n_counties"] > 0
        assert 0.0 <= meta["r2_ridge"] <= 1.0

    def test_metadata_hgb_params_match_constants(self, tmp_path):
        """hgb_params in meta must match HGB_PARAMS constant."""
        from src.prediction.train_ensemble_model import HGB_PARAMS, train_and_save

        n = 40
        ta_df = _make_type_assignments(n, j=6)
        demo_df = _make_demographics(n, n_demo=4)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ensemble_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            _write_election_parquet(assembled_dir, year, n, seed)

        result = train_and_save(
            type_assignments_path=ta_path,
            demographics_path=demo_path,
            assembled_dir=assembled_dir,
            output_dir=output_dir,
        )

        meta = json.loads(result["output_json"].read_text())
        assert meta["hgb_params"]["max_iter"] == HGB_PARAMS["max_iter"]
        assert meta["hgb_params"]["learning_rate"] == pytest.approx(HGB_PARAMS["learning_rate"])
        assert meta["hgb_params"]["max_depth"] == HGB_PARAMS["max_depth"]
        assert meta["hgb_params"]["min_samples_leaf"] == HGB_PARAMS["min_samples_leaf"]

    def test_training_metrics_json_written(self, tmp_path):
        """train_and_save must write data/model/training_metrics.json."""
        from src.prediction.train_ensemble_model import train_and_save
        import src.prediction.train_ensemble_model as mod

        n = 40
        ta_df = _make_type_assignments(n, j=6)
        demo_df = _make_demographics(n, n_demo=4)

        ta_path = tmp_path / "type_assignments.parquet"
        demo_path = tmp_path / "county_features_national.parquet"
        assembled_dir = tmp_path / "assembled"
        output_dir = tmp_path / "ensemble_output"
        ta_df.to_parquet(ta_path, index=False)
        demo_df.to_parquet(demo_path, index=False)

        for year, seed in zip([2008, 2012, 2016, 2020, 2024], range(5)):
            _write_election_parquet(assembled_dir, year, n, seed)

        # Point training_metrics path to tmp_path so we don't pollute real data
        metrics_path = tmp_path / "training_metrics.json"
        original_path = mod._TRAINING_METRICS_PATH
        mod._TRAINING_METRICS_PATH = metrics_path
        try:
            train_and_save(
                type_assignments_path=ta_path,
                demographics_path=demo_path,
                assembled_dir=assembled_dir,
                output_dir=output_dir,
            )
        finally:
            mod._TRAINING_METRICS_PATH = original_path

        assert metrics_path.exists(), "training_metrics.json not written"
        tm = json.loads(metrics_path.read_text())
        assert "date_trained" in tm
        assert "ridge" in tm
        assert "hgb" in tm
        assert "ensemble" in tm
        assert tm["ridge"]["train_r2"] >= 0.0
        assert tm["ensemble"]["pred_min"] >= 0.0
        assert tm["ensemble"]["pred_max"] <= 1.0


# ---------------------------------------------------------------------------
# Test: _read_loo_metric
# ---------------------------------------------------------------------------

class TestReadLooMetric:
    """_read_loo_metric must safely read from accuracy_metrics.json."""

    def test_reads_existing_metric(self, tmp_path):
        """Returns float when method exists in accuracy_metrics.json."""
        from src.prediction.train_ensemble_model import _read_loo_metric
        import src.prediction.train_ensemble_model as mod

        metrics = {
            "method_comparison": [
                {"method": "Ridge (scores only)", "loo_r": 0.533},
                {"method": "Ridge+HGB ensemble", "loo_r": 0.711},
            ]
        }
        metrics_path = tmp_path / "accuracy_metrics.json"
        metrics_path.write_text(json.dumps(metrics))

        original = mod._ACCURACY_METRICS_PATH
        mod._ACCURACY_METRICS_PATH = metrics_path
        try:
            result = _read_loo_metric("Ridge+HGB ensemble")
            assert result == pytest.approx(0.711)
        finally:
            mod._ACCURACY_METRICS_PATH = original

    def test_returns_none_for_missing_method(self, tmp_path):
        """Returns None when the requested method is not in the file."""
        from src.prediction.train_ensemble_model import _read_loo_metric
        import src.prediction.train_ensemble_model as mod

        metrics = {"method_comparison": [{"method": "Something else", "loo_r": 0.5}]}
        metrics_path = tmp_path / "accuracy_metrics.json"
        metrics_path.write_text(json.dumps(metrics))

        original = mod._ACCURACY_METRICS_PATH
        mod._ACCURACY_METRICS_PATH = metrics_path
        try:
            assert _read_loo_metric("Nonexistent") is None
        finally:
            mod._ACCURACY_METRICS_PATH = original

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None when accuracy_metrics.json doesn't exist."""
        from src.prediction.train_ensemble_model import _read_loo_metric
        import src.prediction.train_ensemble_model as mod

        original = mod._ACCURACY_METRICS_PATH
        mod._ACCURACY_METRICS_PATH = tmp_path / "nonexistent.json"
        try:
            assert _read_loo_metric("Ridge+HGB ensemble") is None
        finally:
            mod._ACCURACY_METRICS_PATH = original
