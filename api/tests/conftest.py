# api/tests/conftest.py
"""Shared test fixtures for the Bedrock API tests.

Pattern: create a fresh app instance per test using create_app(lifespan_override=_noop_lifespan)
to skip the real DB startup, then set app.state fields directly before yielding the TestClient.
This avoids requiring data/bedrock.duckdb or any real parquet files during testing.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@asynccontextmanager
async def _noop_lifespan(app):
    """Test lifespan: skip all DB/file loading, just yield."""
    yield

# ── Synthetic data constants ────────────────────────────────────────────────
TEST_VERSION = "test_v1"
TEST_FIPS = ["12001", "12003", "13001", "13003", "01001"]
TEST_K = 3
TEST_COMMUNITIES = {
    "12001": 0, "12003": 0, "13001": 1, "13003": 2, "01001": 2
}
TEST_RACES = ["FL_Senate"]


def _build_test_db() -> duckdb.DuckDBPyConnection:
    """Build an in-memory DuckDB with synthetic data matching the real schema."""
    con = duckdb.connect(":memory:")

    con.execute("""
        CREATE TABLE counties (
            county_fips VARCHAR PRIMARY KEY,
            state_abbr  VARCHAR NOT NULL,
            state_fips  VARCHAR NOT NULL,
            county_name VARCHAR
        )
    """)
    counties_data = [
        ("12001", "FL", "12", "Alachua County, FL"),
        ("12003", "FL", "12", "Baker County, FL"),
        ("13001", "GA", "13", "Appling County, GA"),
        ("13003", "GA", "13", "Atkinson County, GA"),
        ("01001", "AL", "01", "Autauga County, AL"),
    ]
    for row in counties_data:
        con.execute("INSERT INTO counties VALUES (?, ?, ?, ?)", list(row))

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
        "INSERT INTO model_versions VALUES (?, 'current', ?, 7, 'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')",
        [TEST_VERSION, TEST_K],
    )

    con.execute("""
        CREATE TABLE community_assignments (
            county_fips VARCHAR NOT NULL,
            community_id INTEGER NOT NULL,
            k INTEGER NOT NULL,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (county_fips, k, version_id)
        )
    """)
    for fips, cid in TEST_COMMUNITIES.items():
        con.execute(
            "INSERT INTO community_assignments VALUES (?, ?, ?, ?)",
            [fips, cid, TEST_K, TEST_VERSION],
        )

    con.execute("""
        CREATE TABLE type_assignments (
            community_id INTEGER NOT NULL,
            k INTEGER NOT NULL,
            dominant_type_id INTEGER,
            j INTEGER,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (community_id, k, version_id)
        )
    """)
    for cid in range(TEST_K):
        con.execute(
            "INSERT INTO type_assignments VALUES (?, ?, ?, 7, ?)",
            [cid, TEST_K, cid % 7, TEST_VERSION],
        )

    con.execute("""
        CREATE TABLE predictions (
            county_fips VARCHAR NOT NULL,
            race VARCHAR NOT NULL,
            version_id VARCHAR NOT NULL,
            pred_dem_share DOUBLE,
            pred_std DOUBLE,
            pred_lo90 DOUBLE,
            pred_hi90 DOUBLE,
            state_pred DOUBLE,
            poll_avg DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    for fips in TEST_FIPS:
        con.execute(
            "INSERT INTO predictions VALUES (?, 'FL_Senate', ?, 0.42, 0.03, 0.37, 0.47, 0.44, 0.46)",
            [fips, TEST_VERSION],
        )

    # Minimal schema: only one shift column. Extend when shift_profile endpoints need it.
    con.execute("""
        CREATE TABLE county_shifts (
            county_fips VARCHAR NOT NULL,
            version_id  VARCHAR NOT NULL,
            pres_d_shift_00_04 DOUBLE,
            PRIMARY KEY (county_fips, version_id)
        )
    """)
    for fips in TEST_FIPS:
        con.execute("INSERT INTO county_shifts VALUES (?, ?, 0.01)", [fips, TEST_VERSION])

    con.execute("""
        CREATE TABLE community_sigma (
            community_id_row INTEGER NOT NULL,
            community_id_col INTEGER NOT NULL,
            sigma_value DOUBLE,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (community_id_row, community_id_col, version_id)
        )
    """)
    sigma = np.eye(TEST_K) * 0.01 + np.ones((TEST_K, TEST_K)) * 0.005
    for i in range(TEST_K):
        for j in range(TEST_K):
            con.execute(
                "INSERT INTO community_sigma VALUES (?, ?, ?, ?)",
                [i, j, float(sigma[i, j]), TEST_VERSION],
            )

    # ── Type-primary tables ────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE types (
            type_id INTEGER NOT NULL,
            super_type_id INTEGER NOT NULL,
            display_name VARCHAR NOT NULL,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (type_id, version_id)
        )
    """)
    # 4 types, 2 super-types
    type_data = [
        (0, 0, "Rural Conservative", TEST_VERSION),
        (1, 0, "Small Town Traditional", TEST_VERSION),
        (2, 1, "Suburban Moderate", TEST_VERSION),
        (3, 1, "Urban Progressive", TEST_VERSION),
    ]
    for td in type_data:
        con.execute("INSERT INTO types VALUES (?, ?, ?, ?)", list(td))

    con.execute("""
        CREATE TABLE county_type_assignments (
            county_fips VARCHAR NOT NULL,
            dominant_type INTEGER NOT NULL,
            super_type INTEGER NOT NULL,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (county_fips, version_id)
        )
    """)
    county_type_map = {
        "12001": (3, 1), "12003": (0, 0), "13001": (1, 0),
        "13003": (0, 0), "01001": (2, 1),
    }
    for fips, (dt, st) in county_type_map.items():
        con.execute(
            "INSERT INTO county_type_assignments VALUES (?, ?, ?, ?)",
            [fips, dt, st, TEST_VERSION],
        )

    return con


def _build_test_state(K: int = TEST_K) -> dict:
    """Build synthetic in-memory state (sigma, mu_prior, weights)."""
    sigma = np.eye(K) * 0.01 + np.ones((K, K)) * 0.005
    mu_prior = np.full(K, 0.42)

    # State weights: K communities per state
    state_weights = pd.DataFrame([
        {"state_abbr": "FL", "state_fips": "12", **{f"community_{i}": 1/K for i in range(K)}},
        {"state_abbr": "GA", "state_fips": "13", **{f"community_{i}": 1/K for i in range(K)}},
        {"state_abbr": "AL", "state_fips": "01", **{f"community_{i}": 1/K for i in range(K)}},
    ])

    county_weights = pd.DataFrame([
        {"county_fips": fips, "community_id": TEST_COMMUNITIES[fips]}
        for fips in TEST_FIPS
    ])

    return {
        "sigma": sigma,
        "mu_prior": mu_prior,
        "state_weights": state_weights,
        "county_weights": county_weights,
    }


@pytest.fixture
def client():
    """TestClient with synthetic DuckDB and in-memory state injected.

    Creates a fresh app instance with _noop_lifespan so no real files are touched.
    app.state fields are set directly before the TestClient context manager runs.
    """
    test_db = _build_test_db()
    state = _build_test_state()

    # Fresh app instance with no-op lifespan (skips real DuckDB/parquet loading)
    test_app = create_app(lifespan_override=_noop_lifespan)

    # Set state fields that routers expect
    test_app.state.db = test_db
    test_app.state.version_id = TEST_VERSION
    test_app.state.K = TEST_K
    test_app.state.sigma = state["sigma"]
    test_app.state.mu_prior = state["mu_prior"]
    test_app.state.state_weights = state["state_weights"]
    test_app.state.county_weights = state["county_weights"]

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c

    test_db.close()
