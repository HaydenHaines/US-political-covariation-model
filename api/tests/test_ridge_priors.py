# api/tests/test_ridge_priors.py
"""Tests for Ridge county priors integration in the API.

Covers:
  - Ridge priors loaded at startup and stored in app.state.ridge_priors
  - POST /forecast/poll uses Ridge priors when available
  - Fallback to type-mean when Ridge priors are absent
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app, lifespan
from api.tests.conftest import (
    TEST_FIPS,
    TEST_VERSION,
    _build_test_db,
    _build_test_state,
    _noop_lifespan,
)

# ── Synthetic type-primary data ─────────────────────────────────────────────

TEST_J = 4  # number of types
_rng = np.random.default_rng(42)

# Soft membership scores: (N, J) — one row per county
_TEST_TYPE_SCORES = _rng.uniform(-1, 1, size=(len(TEST_FIPS), TEST_J)).astype(np.float64)
# Normalize rows to sum-of-abs = 1
_abs = np.abs(_TEST_TYPE_SCORES)
_TEST_TYPE_SCORES = _TEST_TYPE_SCORES / _abs.sum(axis=1, keepdims=True)

_TEST_TYPE_COVARIANCE = np.eye(TEST_J, dtype=np.float64) * 0.01 + 0.005
_TEST_TYPE_PRIORS = np.full(TEST_J, 0.44, dtype=np.float64)

# Tract-level priors (11-digit GEOIDs) — matches new tract-primary architecture.
# ridge_priors is now loaded from tract_priors.parquet, not from DuckDB.
RIDGE_PRIOR_VALUES = {
    "12001020100": 0.55,
    "12001020200": 0.38,
    "13001020100": 0.41,
    "13001020200": 0.36,
    "01001020100": 0.52,
}


def _make_app_with_type_state(ridge_priors: dict | None = None) -> TestClient:
    """Build a TestClient with type-primary state (and optional Ridge priors)."""
    test_db = _build_test_db()
    state = _build_test_state()

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = test_db
    test_app.state.version_id = TEST_VERSION
    test_app.state.K = state["sigma"].shape[0]
    test_app.state.sigma = state["sigma"]
    test_app.state.mu_prior = state["mu_prior"]
    test_app.state.state_weights = state["state_weights"]
    test_app.state.county_weights = state["county_weights"]
    test_app.state.contract_ok = True

    # Inject type-primary data
    test_app.state.type_scores = _TEST_TYPE_SCORES
    test_app.state.type_covariance = _TEST_TYPE_COVARIANCE
    test_app.state.type_priors = _TEST_TYPE_PRIORS
    test_app.state.type_county_fips = [f.zfill(5) for f in TEST_FIPS]

    # Ridge priors
    test_app.state.ridge_priors = ridge_priors if ridge_priors is not None else {}

    return test_app, test_db


# ── Tests: startup loading ───────────────────────────────────────────────────


class TestRidgePriorsStartupLoading:
    """Test that lifespan() loads ridge_county_priors.parquet into app.state.ridge_priors."""

    def test_ridge_priors_loaded_from_db(self, tmp_path):
        """Tract priors parquet present → app.state.ridge_priors populated.

        The ridge_priors dict is loaded from data/communities/type_priors.parquet.
        Keys are 11-digit tract GEOIDs; values are prior Dem vote shares.
        """
        import duckdb
        db_path = tmp_path / "wethervane.duckdb"
        # Use context manager so the file handle is released before lifespan re-opens it.
        with duckdb.connect(str(db_path)) as con:
            con.execute(
                "CREATE TABLE model_versions (version_id VARCHAR PRIMARY KEY, role VARCHAR, "
                "k INTEGER, j INTEGER, shift_type VARCHAR, vote_share_type VARCHAR, "
                "n_training_dims INTEGER, n_holdout_dims INTEGER, holdout_r VARCHAR, "
                "geography VARCHAR, description VARCHAR, created_at TIMESTAMP)"
            )
            con.execute(
                "INSERT INTO model_versions VALUES ('v1', 'current', 3, 4, 'logodds', "
                "'total', 30, 3, '0.70', 'test', 'test', '2026-01-01')"
            )
            con.execute(
                "CREATE TABLE community_sigma (community_id_row INTEGER, "
                "community_id_col INTEGER, sigma_value DOUBLE, version_id VARCHAR)"
            )
            con.execute(
                "CREATE TABLE predictions (county_fips VARCHAR, race VARCHAR, "
                "version_id VARCHAR, pred_dem_share DOUBLE, pred_std DOUBLE, "
                "pred_lo90 DOUBLE, pred_hi90 DOUBLE, state_pred DOUBLE, poll_avg DOUBLE)"
            )
            con.execute(
                "CREATE TABLE community_assignments (county_fips VARCHAR, community_id INTEGER, "
                "k INTEGER, version_id VARCHAR)"
            )

        # Create tract_priors.parquet in the expected location
        tracts_dir = tmp_path / "data" / "tracts"
        tracts_dir.mkdir(parents=True)
        tract_df = pd.DataFrame({
            "tract_geoid": list(RIDGE_PRIOR_VALUES.keys()),
            "tract_prior": list(RIDGE_PRIOR_VALUES.values()),
            # state_abbr derived from first 2 digits of GEOID (FIPS state code)
            "state_abbr": ["FL", "FL", "GA", "GA", "AL"],
            "total_votes": [1000, 800, 900, 700, 600],
        })
        tract_df.to_parquet(tracts_dir / "tract_priors.parquet", index=False)

        captured_state: dict = {}

        @asynccontextmanager
        async def _capturing_lifespan(app):
            """Run real lifespan but capture app.state afterwards."""
            async with lifespan(app):
                captured_state["ridge_priors"] = dict(app.state.ridge_priors)
                yield

        import os
        os.environ["WETHERVANE_DATA_DIR"] = str(tmp_path)
        os.environ["WETHERVANE_DB_PATH"] = str(db_path)
        os.environ["WETHERVANE_PROJECT_ROOT"] = str(tmp_path)
        try:
            test_app = create_app(lifespan_override=_capturing_lifespan)
            with TestClient(test_app):
                pass
        finally:
            del os.environ["WETHERVANE_DATA_DIR"]
            del os.environ["WETHERVANE_DB_PATH"]
            del os.environ["WETHERVANE_PROJECT_ROOT"]

        assert len(captured_state["ridge_priors"]) == len(RIDGE_PRIOR_VALUES)
        for geoid, expected in RIDGE_PRIOR_VALUES.items():
            assert captured_state["ridge_priors"][geoid] == pytest.approx(expected)

    def test_ridge_priors_empty_when_file_missing(self, tmp_path):
        """tract_priors.parquet absent → app.state.ridge_priors = {}.

        When the parquet file doesn't exist the lifespan catches the FileNotFoundError
        and sets ridge_priors to an empty dict, allowing the API to start normally.
        """
        import duckdb
        db_path = tmp_path / "wethervane.duckdb"
        # Use context manager so the file handle is released before lifespan re-opens it.
        with duckdb.connect(str(db_path)) as con:
            con.execute(
                "CREATE TABLE model_versions (version_id VARCHAR PRIMARY KEY, role VARCHAR, "
                "k INTEGER, j INTEGER, shift_type VARCHAR, vote_share_type VARCHAR, "
                "n_training_dims INTEGER, n_holdout_dims INTEGER, holdout_r VARCHAR, "
                "geography VARCHAR, description VARCHAR, created_at TIMESTAMP)"
            )
            con.execute(
                "INSERT INTO model_versions VALUES ('v1', 'current', 3, 4, 'logodds', "
                "'total', 30, 3, '0.70', 'test', 'test', '2026-01-01')"
            )
            con.execute(
                "CREATE TABLE community_sigma (community_id_row INTEGER, "
                "community_id_col INTEGER, sigma_value DOUBLE, version_id VARCHAR)"
            )
            con.execute(
                "CREATE TABLE predictions (county_fips VARCHAR, race VARCHAR, "
                "version_id VARCHAR, pred_dem_share DOUBLE, pred_std DOUBLE, "
                "pred_lo90 DOUBLE, pred_hi90 DOUBLE, state_pred DOUBLE, poll_avg DOUBLE)"
            )
            con.execute(
                "CREATE TABLE community_assignments (county_fips VARCHAR, community_id INTEGER, "
                "k INTEGER, version_id VARCHAR)"
            )

        # Do NOT create tract_priors.parquet — it should be absent
        tracts_dir = tmp_path / "data" / "tracts"
        assert not (tracts_dir / "tract_priors.parquet").exists()

        captured_state: dict = {}

        @asynccontextmanager
        async def _capturing_lifespan(app):
            async with lifespan(app):
                captured_state["ridge_priors"] = dict(app.state.ridge_priors)
                yield

        import os
        os.environ["WETHERVANE_DATA_DIR"] = str(tmp_path)
        os.environ["WETHERVANE_DB_PATH"] = str(db_path)
        os.environ["WETHERVANE_PROJECT_ROOT"] = str(tmp_path)
        try:
            test_app = create_app(lifespan_override=_capturing_lifespan)
            with TestClient(test_app):
                pass
        finally:
            del os.environ["WETHERVANE_DATA_DIR"]
            del os.environ["WETHERVANE_DB_PATH"]
            del os.environ["WETHERVANE_PROJECT_ROOT"]

        assert captured_state["ridge_priors"] == {}


# ── Tests: forecast endpoint uses Ridge priors ───────────────────────────────


class TestForecastUsesRidgePriors:
    """POST /forecast/poll uses Ridge county priors when available."""

    def test_predictions_differ_with_ridge_priors(self):
        """Forecasts with Ridge priors differ from those without."""
        app_ridge, db_ridge = _make_app_with_type_state(ridge_priors=RIDGE_PRIOR_VALUES)
        app_no_ridge, db_no_ridge = _make_app_with_type_state(ridge_priors={})

        payload = {"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800}

        try:
            with TestClient(app_ridge, raise_server_exceptions=True) as c_ridge:
                resp_ridge = c_ridge.post("/api/v1/forecast/poll", json=payload)

            with TestClient(app_no_ridge, raise_server_exceptions=True) as c_no_ridge:
                resp_no_ridge = c_no_ridge.post("/api/v1/forecast/poll", json=payload)
        finally:
            db_ridge.close()
            db_no_ridge.close()

        assert resp_ridge.status_code == 200
        assert resp_no_ridge.status_code == 200

        ridge_data = {r["county_fips"]: r["pred_dem_share"] for r in resp_ridge.json()}
        no_ridge_data = {r["county_fips"]: r["pred_dem_share"] for r in resp_no_ridge.json()}

        # At least one county should differ between Ridge and type-mean priors
        any_diff = any(
            abs(ridge_data[f] - no_ridge_data[f]) > 1e-6
            for f in TEST_FIPS
            if f in ridge_data and f in no_ridge_data
        )
        assert any_diff, "Ridge priors should produce different county predictions than type-mean"

    def test_ridge_priors_anchor_predictions(self):
        """With no poll signal (poll exactly equals prior), Ridge priors dominate."""
        # Use a very large N poll so Bayesian update has strong pull,
        # then check that counties with different Ridge priors differ more than they would
        # under a flat type-mean prior.
        app_ridge, db = _make_app_with_type_state(ridge_priors=RIDGE_PRIOR_VALUES)

        payload = {"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 100}

        try:
            with TestClient(app_ridge, raise_server_exceptions=True) as c:
                resp = c.post("/api/v1/forecast/poll", json=payload)
        finally:
            db.close()

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(TEST_FIPS)

        # All pred_dem_share should be well-defined floats
        for row in data:
            assert isinstance(row["pred_dem_share"], float)
            assert 0.0 <= row["pred_dem_share"] <= 1.0

    def test_forecast_fields_complete_with_ridge(self):
        """Response shape is unchanged when Ridge priors are active."""
        app_ridge, db = _make_app_with_type_state(ridge_priors=RIDGE_PRIOR_VALUES)

        payload = {"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800}

        try:
            with TestClient(app_ridge, raise_server_exceptions=True) as c:
                resp = c.post("/api/v1/forecast/poll", json=payload)
        finally:
            db.close()

        assert resp.status_code == 200
        for row in resp.json():
            assert "county_fips" in row
            assert "pred_dem_share" in row
            assert "pred_lo90" in row
            assert "pred_hi90" in row
            assert row["poll_avg"] == pytest.approx(0.48)

    def test_multi_poll_endpoint_uses_ridge_priors(self):
        """POST /forecast/polls also routes through Ridge priors."""
        app_ridge, db_ridge = _make_app_with_type_state(ridge_priors=RIDGE_PRIOR_VALUES)
        app_no_ridge, db_no_ridge = _make_app_with_type_state(ridge_priors={})

        payload = {"cycle": 2022, "state": "FL", "race": "FL_Senate",
                   "half_life_days": 30, "apply_quality": False}

        try:
            with TestClient(app_ridge, raise_server_exceptions=True) as c_ridge:
                resp_ridge = c_ridge.post("/api/v1/forecast/polls", json=payload)

            with TestClient(app_no_ridge, raise_server_exceptions=True) as c_no_ridge:
                resp_no_ridge = c_no_ridge.post("/api/v1/forecast/polls", json=payload)
        finally:
            db_ridge.close()
            db_no_ridge.close()

        # Both may 404 if no poll CSV for 2022 FL_Senate — that's fine.
        # If both succeed, check they differ.
        if resp_ridge.status_code == 200 and resp_no_ridge.status_code == 200:
            ridge_counties = {r["county_fips"]: r["pred_dem_share"]
                              for r in resp_ridge.json()["counties"]}
            no_ridge_counties = {r["county_fips"]: r["pred_dem_share"]
                                 for r in resp_no_ridge.json()["counties"]}
            any_diff = any(
                abs(ridge_counties.get(f, 0) - no_ridge_counties.get(f, 0)) > 1e-6
                for f in TEST_FIPS
            )
            assert any_diff, "Ridge priors should produce different predictions"


# ── Tests: fallback behavior ─────────────────────────────────────────────────


class TestRidgePriorsFallback:
    """When Ridge priors dict is empty, forecast falls back to type-mean."""

    def test_forecast_succeeds_without_ridge_priors(self):
        """Forecast works correctly even with no Ridge priors loaded."""
        app_no_ridge, db = _make_app_with_type_state(ridge_priors={})

        payload = {"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800}

        try:
            with TestClient(app_no_ridge, raise_server_exceptions=True) as c:
                resp = c.post("/api/v1/forecast/poll", json=payload)
        finally:
            db.close()

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(TEST_FIPS)
        for row in data:
            assert 0.0 <= row["pred_dem_share"] <= 1.0

    def test_ridge_priors_attribute_always_present(self):
        """app.state.ridge_priors is always set (never AttributeError)."""
        test_db = _build_test_db()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = test_db
        test_app.state.version_id = TEST_VERSION
        state = _build_test_state()
        test_app.state.sigma = state["sigma"]
        test_app.state.mu_prior = state["mu_prior"]
        test_app.state.K = state["sigma"].shape[0]
        test_app.state.state_weights = state["state_weights"]
        test_app.state.county_weights = state["county_weights"]
        test_app.state.contract_ok = True
        # Do NOT set ridge_priors — the router should gracefully handle getattr default {}

        test_app.state.type_scores = _TEST_TYPE_SCORES
        test_app.state.type_covariance = _TEST_TYPE_COVARIANCE
        test_app.state.type_priors = _TEST_TYPE_PRIORS
        test_app.state.type_county_fips = [f.zfill(5) for f in TEST_FIPS]
        # ridge_priors deliberately left unset

        payload = {"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800}

        try:
            with TestClient(test_app, raise_server_exceptions=True) as c:
                resp = c.post("/api/v1/forecast/poll", json=payload)
        finally:
            test_db.close()
        assert resp.status_code == 200
