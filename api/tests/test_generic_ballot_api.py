"""Tests for the GET /forecast/generic-ballot endpoint.

Exercises the endpoint's response schema and the manual_shift override.
"""
from __future__ import annotations

import pytest

from api.tests.conftest import (
    _build_test_db,
    _build_test_state,
    _noop_lifespan,
    TEST_VERSION,
    TEST_FIPS,
    TEST_K,
)
from api.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def basic_client():
    """Minimal TestClient (no type-primary state) for generic-ballot endpoint."""
    test_db = _build_test_db()
    state = _build_test_state()
    app = create_app(lifespan_override=_noop_lifespan)
    app.state.db = test_db
    app.state.version_id = TEST_VERSION
    app.state.K = TEST_K
    app.state.sigma = state["sigma"]
    app.state.mu_prior = state["mu_prior"]
    app.state.state_weights = state["state_weights"]
    app.state.county_weights = state["county_weights"]
    app.state.contract_ok = True
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    test_db.close()


class TestGetGenericBallot:
    def test_returns_200(self, basic_client):
        resp = basic_client.get("/api/v1/forecast/generic-ballot")
        assert resp.status_code == 200

    def test_response_has_required_fields(self, basic_client):
        resp = basic_client.get("/api/v1/forecast/generic-ballot")
        data = resp.json()
        assert "gb_avg" in data
        assert "pres_baseline" in data
        assert "shift" in data
        assert "shift_pp" in data
        assert "n_polls" in data
        assert "source" in data

    def test_pres_baseline_is_correct(self, basic_client):
        """pres_baseline must always be 0.4841 (2024 presidential national dem share)."""
        resp = basic_client.get("/api/v1/forecast/generic-ballot")
        data = resp.json()
        assert data["pres_baseline"] == pytest.approx(0.4841, abs=0.0001)

    def test_shift_pp_is_shift_times_100(self, basic_client):
        """shift_pp should equal shift × 100."""
        resp = basic_client.get("/api/v1/forecast/generic-ballot")
        data = resp.json()
        assert data["shift_pp"] == pytest.approx(data["shift"] * 100, abs=0.001)

    def test_manual_shift_override(self, basic_client):
        """When manual_shift is provided, source should be 'manual' and shift should match."""
        resp = basic_client.get("/api/v1/forecast/generic-ballot?manual_shift=0.025")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "manual"
        assert data["shift"] == pytest.approx(0.025)
        assert data["n_polls"] == 0

    def test_zero_manual_shift(self, basic_client):
        """Passing manual_shift=0.0 should give a zero shift with source='manual'."""
        resp = basic_client.get("/api/v1/forecast/generic-ballot?manual_shift=0.0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "manual"
        assert data["shift"] == pytest.approx(0.0)

    def test_auto_source_without_override(self, basic_client):
        """Without manual_shift, source should be 'auto'."""
        resp = basic_client.get("/api/v1/forecast/generic-ballot")
        data = resp.json()
        assert data["source"] == "auto"

    def test_shift_consistent_with_gb_avg_and_baseline(self, basic_client):
        """shift must equal gb_avg - pres_baseline."""
        resp = basic_client.get("/api/v1/forecast/generic-ballot?manual_shift=0.018")
        data = resp.json()
        expected_shift = data["gb_avg"] - data["pres_baseline"]
        assert data["shift"] == pytest.approx(expected_shift, abs=1e-6)
