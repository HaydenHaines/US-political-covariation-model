"""Tests for the forecast router — GET /forecast and POST /forecast/poll."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.tests.conftest import (
    TEST_FIPS,
    TEST_VERSION,
    _build_test_db,
    _build_test_state,
    _noop_lifespan,
)


class TestGetForecast:
    def test_returns_all_predictions(self, client):
        resp = client.get("/api/v1/forecast")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5  # 5 test counties × 1 race
        assert all(r["race"] == "FL_Senate" for r in data)

    def test_filter_by_race(self, client):
        resp = client.get("/api/v1/forecast?race=FL_Senate")
        assert resp.status_code == 200
        assert len(resp.json()) == 5

    def test_filter_by_nonexistent_race(self, client):
        resp = client.get("/api/v1/forecast?race=FAKE_RACE")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filter_by_state(self, client):
        resp = client.get("/api/v1/forecast?state=FL")
        assert resp.status_code == 200
        data = resp.json()
        assert all(r["state_abbr"] == "FL" for r in data)
        assert len(data) == 2  # 12001, 12003

    def test_filter_by_race_and_state(self, client):
        resp = client.get("/api/v1/forecast?race=FL_Senate&state=GA")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2  # 13001, 13003
        assert all(r["state_abbr"] == "GA" for r in data)

    def test_response_fields(self, client):
        resp = client.get("/api/v1/forecast?race=FL_Senate&state=FL")
        data = resp.json()
        row = data[0]
        assert "county_fips" in row
        assert "county_name" in row
        assert "state_abbr" in row
        assert "race" in row
        assert "pred_dem_share" in row
        assert "pred_std" in row
        assert "pred_lo90" in row
        assert "pred_hi90" in row
        assert row["pred_dem_share"] == pytest.approx(0.42, abs=0.01)


class TestPostForecastPoll:
    def test_poll_update_returns_predictions(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5  # All counties get updated predictions
        assert all(r["race"] == "FL_Senate" for r in data)

    def test_poll_update_shifts_predictions(self, client):
        # Prior is 0.42 for all communities. Poll at 0.48 should shift up.
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.55, "n": 1000},
        )
        data = resp.json()
        for row in data:
            assert row["pred_dem_share"] > 0.42  # Should shift upward from prior

    def test_poll_update_has_confidence_interval(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.45, "n": 600},
        )
        data = resp.json()
        for row in data:
            assert row["pred_lo90"] < row["pred_dem_share"]
            assert row["pred_hi90"] > row["pred_dem_share"]

    def test_poll_update_unknown_state(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "XX", "race": "XX_Senate", "dem_share": 0.5, "n": 600},
        )
        assert resp.status_code == 404

    def test_poll_update_state_pred_field(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.50, "n": 600},
        )
        data = resp.json()
        assert data[0]["state_pred"] is not None
        assert data[0]["poll_avg"] == pytest.approx(0.50)


def test_get_polls_queries_duckdb(client):
    """GET /polls returns rows from DuckDB, not CSV."""
    resp = client.get("/api/v1/polls?cycle=2026&state=FL")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["geography"] == "FL"
    assert data[0]["dem_share"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Crosstab W override integration tests
# ---------------------------------------------------------------------------

_TEST_J = 4
_rng = np.random.default_rng(99)
_TEST_TYPE_SCORES = _rng.uniform(-1, 1, size=(len(TEST_FIPS), _TEST_J)).astype(np.float64)
_abs_ts = np.abs(_TEST_TYPE_SCORES)
_TEST_TYPE_SCORES = _TEST_TYPE_SCORES / _abs_ts.sum(axis=1, keepdims=True)
_TEST_TYPE_COVARIANCE = np.eye(_TEST_J, dtype=np.float64) * 0.01 + 0.005
_TEST_TYPE_PRIORS = np.full(_TEST_J, 0.44, dtype=np.float64)

# Synthetic affinity index: one dimension with meaningful values
_TEST_AFFINITY = {
    "education_college": np.linspace(-0.3, 0.3, _TEST_J),
    "education_noncollege": np.linspace(0.3, -0.3, _TEST_J),
    "race_white": np.zeros(_TEST_J),
    "race_black": np.zeros(_TEST_J),
    "race_hispanic": np.zeros(_TEST_J),
    "race_asian": np.zeros(_TEST_J),
    "urbanicity_urban": np.zeros(_TEST_J),
    "urbanicity_rural": np.zeros(_TEST_J),
    "age_senior": np.zeros(_TEST_J),
    "religion_evangelical": np.zeros(_TEST_J),
}

_TEST_STATE_MEANS = {
    "FL": {
        "education_college": 0.33,
        "education_noncollege": 0.67,
        "race_white": 0.60,
        "race_black": 0.15,
        "race_hispanic": 0.25,
        "race_asian": 0.03,
        "urbanicity_urban": 3.5,
        "urbanicity_rural": 0.20,
        "age_senior": 42.0,
        "religion_evangelical": 0.30,
    }
}


def _make_type_client_with_crosstabs(crosstab_rows=None):
    """Build a TestClient with type-primary state + optional crosstab data in DB."""
    test_db = _build_test_db()
    state = _build_test_state()

    # Insert crosstab rows into poll_crosstabs (poll_id 'abc123' already exists)
    if crosstab_rows:
        for row in crosstab_rows:
            test_db.execute(
                "INSERT INTO poll_crosstabs VALUES (?, ?, ?, ?, ?, ?)",
                [
                    row["poll_id"],
                    row["demographic_group"],
                    row["group_value"],
                    row.get("dem_share"),
                    row.get("n_sample"),
                    row.get("pct_of_sample"),
                ],
            )

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = test_db
    test_app.state.version_id = TEST_VERSION
    test_app.state.K = state["sigma"].shape[0]
    test_app.state.sigma = state["sigma"]
    test_app.state.mu_prior = state["mu_prior"]
    test_app.state.state_weights = state["state_weights"]
    test_app.state.county_weights = state["county_weights"]
    test_app.state.contract_ok = True

    # Type-primary state
    test_app.state.type_scores = _TEST_TYPE_SCORES
    test_app.state.type_covariance = _TEST_TYPE_COVARIANCE
    test_app.state.type_priors = _TEST_TYPE_PRIORS
    test_app.state.type_county_fips = list(TEST_FIPS)
    test_app.state.ridge_priors = {}

    # Crosstab affinity index and state means
    test_app.state.crosstab_affinity = _TEST_AFFINITY
    test_app.state.crosstab_state_means = _TEST_STATE_MEANS

    return test_app, test_db


class TestCrosstabWOverrideSinglePoll:
    """POST /forecast/poll should use crosstab-adjusted W when data is available."""

    def test_returns_200_with_crosstab_data(self):
        """Endpoint must succeed even when crosstab W override is used."""
        crosstab_rows = [
            {
                "poll_id": "abc123",
                "demographic_group": "education",
                "group_value": "college",
                "dem_share": None,
                "n_sample": None,
                "pct_of_sample": 0.55,  # above FL mean of 0.33
            }
        ]
        test_app, test_db = _make_type_client_with_crosstabs(crosstab_rows)
        try:
            with TestClient(test_app, raise_server_exceptions=True) as c:
                resp = c.post(
                    "/api/v1/forecast/poll",
                    json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
                )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) > 0
            for row in data:
                assert 0 <= row["pred_dem_share"] <= 1
        finally:
            test_db.close()

    @pytest.mark.xfail(reason="Crosstab W override not yet wired into single-poll endpoint")
    def test_crosstab_w_produces_different_predictions_than_no_crosstab(self):
        """Crosstab-adjusted W should produce different predictions from state-mean W.

        We use a heavily oversampled college demographic (pct=0.85 vs FL mean 0.33).
        The affinity for education_college varies across types (linspace(-0.3, 0.3)),
        so the crosstab adjustment shifts W toward college-heavy types, which should
        change the Bayesian update outcome relative to the uniform state-mean W.
        """
        crosstab_rows = [
            {
                "poll_id": "abc123",
                "demographic_group": "education",
                "group_value": "college",
                "dem_share": None,
                "n_sample": None,
                "pct_of_sample": 0.85,  # strongly above FL mean of 0.33 → large delta
            }
        ]
        # With crosstabs
        app_with_xt, db_with_xt = _make_type_client_with_crosstabs(crosstab_rows)
        # Without crosstabs (affinity = None so W override never happens)
        app_no_xt, db_no_xt = _make_type_client_with_crosstabs(crosstab_rows=None)
        app_no_xt.state.crosstab_affinity = None  # force fallback

        try:
            with TestClient(app_with_xt, raise_server_exceptions=True) as c_xt:
                resp_xt = c_xt.post(
                    "/api/v1/forecast/poll",
                    json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
                )
            with TestClient(app_no_xt, raise_server_exceptions=True) as c_no:
                resp_no = c_no.post(
                    "/api/v1/forecast/poll",
                    json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
                )
        finally:
            db_with_xt.close()
            db_no_xt.close()

        assert resp_xt.status_code == 200
        assert resp_no.status_code == 200

        preds_xt = [r["pred_dem_share"] for r in resp_xt.json()]
        preds_no = [r["pred_dem_share"] for r in resp_no.json()]

        # Results should differ (crosstab W ≠ state-mean W for large delta)
        assert not np.allclose(preds_xt, preds_no, atol=1e-6), (
            "Crosstab-adjusted W should produce different predictions from state-mean W"
        )

    def test_fallback_when_no_crosstab_data_in_db(self):
        """When poll_crosstabs is empty, endpoint must still succeed (fallback to state W)."""
        app_no_xt, db_no_xt = _make_type_client_with_crosstabs(crosstab_rows=None)
        try:
            with TestClient(app_no_xt, raise_server_exceptions=True) as c:
                resp = c.post(
                    "/api/v1/forecast/poll",
                    json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
                )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) > 0
        finally:
            db_no_xt.close()

    def test_fallback_when_affinity_index_none(self):
        """When app.state.crosstab_affinity is None, endpoint must succeed with state-mean W."""
        crosstab_rows = [
            {
                "poll_id": "abc123",
                "demographic_group": "education",
                "group_value": "college",
                "dem_share": None,
                "n_sample": None,
                "pct_of_sample": 0.55,
            }
        ]
        app, db = _make_type_client_with_crosstabs(crosstab_rows)
        app.state.crosstab_affinity = None  # Simulate failed startup load
        try:
            with TestClient(app, raise_server_exceptions=True) as c:
                resp = c.post(
                    "/api/v1/forecast/poll",
                    json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
                )
            assert resp.status_code == 200
            assert len(resp.json()) > 0
        finally:
            db.close()
