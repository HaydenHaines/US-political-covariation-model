"""Tests for GET /api/v1/governor/simulation."""
from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routers.governor._helpers import GOVERNOR_2026_STATES, _GOVERNOR_INCUMBENT
from api.routers.governor.simulation import _simulate_governor_seats
from api.tests.conftest import _noop_lifespan


# ── Unit tests ──────────────────────────────────────────────────────────────


class TestSimulateGovernorSeats:
    def test_buckets_probability_sums_to_one(self):
        """Probability mass across all buckets should sum close to 1.0."""
        safe_dem = sum(1 for p in _GOVERNOR_INCUMBENT.values() if p == "D")
        safe_gop = sum(1 for p in _GOVERNOR_INCUMBENT.values() if p == "R")
        result = _simulate_governor_seats(
            modeled_races=[],
            safe_dem_wins=safe_dem,
            safe_gop_wins=safe_gop,
            n_sims=5000,
            rng_seed=42,
        )
        total_prob = sum(b.probability for b in result.buckets)
        assert 0.95 < total_prob <= 1.01

    def test_d_plus_r_equals_36(self):
        """Every bucket must have d_seats + r_seats == 36."""
        result = _simulate_governor_seats(
            modeled_races=[(0.5, 0.05)] * 10,
            safe_dem_wins=5,
            safe_gop_wins=21,
            n_sims=2000,
            rng_seed=7,
        )
        for bucket in result.buckets:
            assert bucket.d_seats + bucket.r_seats == len(GOVERNOR_2026_STATES)

    def test_strong_dem_races_yield_high_d_seats(self):
        """All races strongly Dem → median d_seats should be near 36."""
        result = _simulate_governor_seats(
            modeled_races=[(0.9, 0.01)] * 36,
            safe_dem_wins=0,
            safe_gop_wins=0,
            n_sims=1000,
            rng_seed=0,
        )
        high_d = sum(b.probability for b in result.buckets if b.d_seats >= 34)
        assert high_d > 0.9

    def test_strong_gop_races_yield_low_d_seats(self):
        """All races strongly GOP → d_seats should be near 0."""
        result = _simulate_governor_seats(
            modeled_races=[(0.1, 0.01)] * 36,
            safe_dem_wins=0,
            safe_gop_wins=0,
            n_sims=1000,
            rng_seed=0,
        )
        low_d = sum(b.probability for b in result.buckets if b.d_seats <= 2)
        assert low_d > 0.9


# ── Endpoint integration tests ──────────────────────────────────────────────


def _build_governor_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with two governor races modeled (AZ=lean D, TX=safe R)."""
    con = duckdb.connect(":memory:")

    con.execute("""
        CREATE TABLE counties (
            county_fips      VARCHAR PRIMARY KEY,
            state_abbr       VARCHAR NOT NULL,
            state_fips       VARCHAR NOT NULL,
            county_name      VARCHAR,
            total_votes_2024 INTEGER
        )
    """)
    counties = [
        ("04001", "AZ", "04", "Apache County, AZ", 50000),
        ("04003", "AZ", "04", "Cochise County, AZ", 30000),
        ("48001", "TX", "48", "Anderson County, TX", 100000),
    ]
    for row in counties:
        con.execute("INSERT INTO counties VALUES (?, ?, ?, ?, ?)", list(row))

    con.execute("""
        CREATE TABLE model_versions (
            version_id VARCHAR PRIMARY KEY,
            role VARCHAR, k INTEGER, j INTEGER,
            shift_type VARCHAR, vote_share_type VARCHAR,
            n_training_dims INTEGER, n_holdout_dims INTEGER,
            holdout_r VARCHAR, geography VARCHAR,
            description VARCHAR, created_at TIMESTAMP
        )
    """)
    con.execute(
        "INSERT INTO model_versions VALUES ('test_v1', 'current', 3, 7, "
        "'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')"
    )

    con.execute("""
        CREATE TABLE predictions (
            county_fips VARCHAR NOT NULL,
            race        VARCHAR NOT NULL,
            version_id  VARCHAR NOT NULL,
            pred_dem_share DOUBLE,
            pred_std       DOUBLE,
            pred_lo90      DOUBLE,
            pred_hi90      DOUBLE,
            state_pred     DOUBLE,
            poll_avg       DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    # AZ Governor: lean D (~0.53)
    for fips, share in [("04001", 0.535), ("04003", 0.525)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 AZ Governor', 'test_v1', ?, 0.04, ?, ?, 0.53, 0.53)",
            [fips, share, share - 0.07, share + 0.07],
        )
    # TX Governor: safe R (~0.38)
    con.execute(
        "INSERT INTO predictions VALUES ('48001', '2026 TX Governor', 'test_v1', 0.38, 0.04, 0.31, 0.45, 0.38, 0.38)"
    )

    return con


@pytest.fixture
def governor_client():
    con = _build_governor_db()
    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = con
    test_app.state.version_id = "test_v1"
    test_app.state.contract_ok = True
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c
    con.close()


class TestGovernorSimulationEndpoint:
    def test_status_200(self, governor_client):
        resp = governor_client.get("/api/v1/governor/simulation")
        assert resp.status_code == 200

    def test_response_has_buckets(self, governor_client):
        data = governor_client.get("/api/v1/governor/simulation").json()
        assert "buckets" in data
        assert isinstance(data["buckets"], list)
        assert len(data["buckets"]) > 0

    def test_bucket_fields_present(self, governor_client):
        data = governor_client.get("/api/v1/governor/simulation").json()
        for bucket in data["buckets"]:
            assert "d_seats" in bucket
            assert "r_seats" in bucket
            assert "probability" in bucket

    def test_d_plus_r_always_36(self, governor_client):
        data = governor_client.get("/api/v1/governor/simulation").json()
        for bucket in data["buckets"]:
            assert bucket["d_seats"] + bucket["r_seats"] == 36

    def test_probabilities_sum_near_one(self, governor_client):
        data = governor_client.get("/api/v1/governor/simulation").json()
        total = sum(b["probability"] for b in data["buckets"])
        assert 0.95 < total <= 1.01

    def test_custom_n_simulations(self, governor_client):
        resp = governor_client.get("/api/v1/governor/simulation?n_simulations=2000")
        assert resp.status_code == 200

    def test_n_simulations_below_minimum_rejected(self, governor_client):
        resp = governor_client.get("/api/v1/governor/simulation?n_simulations=500")
        assert resp.status_code == 422

    def test_date_param_accepted(self, governor_client):
        resp = governor_client.get("/api/v1/governor/simulation?date=2026-04-01")
        assert resp.status_code == 200

    def test_no_version_id_returns_incumbent_fallback(self):
        """When no model is loaded, the endpoint uses incumbent-party fallbacks."""
        con = duckdb.connect(":memory:")
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = None
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            resp = c.get("/api/v1/governor/simulation")
        assert resp.status_code == 200
        data = resp.json()
        assert "buckets" in data
        total = sum(b["probability"] for b in data["buckets"])
        assert 0.95 < total <= 1.01
