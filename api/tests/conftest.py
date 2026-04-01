# api/tests/conftest.py
"""Shared test fixtures for the WetherVane API tests.

Pattern: create a fresh app instance per test using create_app(lifespan_override=_noop_lifespan)
to skip the real DB startup, then set app.state fields directly before yielding the TestClient.
This avoids requiring data/wethervane.duckdb or any real parquet files during testing.
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

# Covariance matrix shape for test sigma: diagonal dominance with mild shared structure.
# These values are deliberately small so no community overwhelms the Bayesian update.
_SIGMA_DIAGONAL_VARIANCE = 0.01   # per-community variance in test sigma matrix
_SIGMA_OFF_DIAGONAL_COV = 0.005   # shared covariance between communities in test matrix


def _add_core_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate the base county and model_version tables."""
    con.execute("""
        CREATE TABLE counties (
            county_fips      VARCHAR PRIMARY KEY,
            state_abbr       VARCHAR NOT NULL,
            state_fips       VARCHAR NOT NULL,
            county_name      VARCHAR,
            total_votes_2024 INTEGER
        )
    """)
    counties_data = [
        ("12001", "FL", "12", "Alachua County, FL", 100000),
        ("12003", "FL", "12", "Baker County, FL", 15000),
        ("13001", "GA", "13", "Appling County, GA", 80000),
        ("13003", "GA", "13", "Atkinson County, GA", 8000),
        ("01001", "AL", "01", "Autauga County, AL", 28000),
    ]
    for row in counties_data:
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
        "INSERT INTO model_versions VALUES (?, 'current', ?, 7, 'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')",
        [TEST_VERSION, TEST_K],
    )


def _add_community_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate community_assignments and type_assignments tables."""
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


def _add_prediction_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate predictions and races tables."""
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

    # Races table: maps race_id → state, required by the state-filtered comparisons query.
    # The comparisons and snapshot endpoints join predictions through races to restrict
    # aggregation to in-state counties, avoiding cross-state baseline dilution (issue #16).
    con.execute("""
        CREATE TABLE races (
            race_id   VARCHAR PRIMARY KEY,
            race_type VARCHAR NOT NULL,
            state     VARCHAR NOT NULL,
            year      INTEGER NOT NULL,
            district  INTEGER
        )
    """)
    con.execute("INSERT INTO races VALUES ('FL_Senate', 'senate', 'FL', 2026, NULL)")


def _add_shift_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate county_shifts table with a minimal schema."""
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


def _add_covariance_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate community_sigma with a diagonal-dominant test covariance matrix."""
    con.execute("""
        CREATE TABLE community_sigma (
            community_id_row INTEGER NOT NULL,
            community_id_col INTEGER NOT NULL,
            sigma_value DOUBLE,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (community_id_row, community_id_col, version_id)
        )
    """)
    sigma = np.eye(TEST_K) * _SIGMA_DIAGONAL_VARIANCE + np.ones((TEST_K, TEST_K)) * _SIGMA_OFF_DIAGONAL_COV
    for i in range(TEST_K):
        for j in range(TEST_K):
            con.execute(
                "INSERT INTO community_sigma VALUES (?, ?, ?, ?)",
                [i, j, float(sigma[i, j]), TEST_VERSION],
            )


def _add_type_primary_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate types, super_types, and county_type_assignments tables."""
    con.execute("""
        CREATE TABLE types (
            type_id INTEGER NOT NULL,
            super_type_id INTEGER NOT NULL,
            display_name VARCHAR NOT NULL,
            median_hh_income DOUBLE,
            pct_bachelors_plus DOUBLE,
            pct_white_nh DOUBLE,
            log_pop_density DOUBLE,
            narrative VARCHAR,
            version_id VARCHAR NOT NULL,
            PRIMARY KEY (type_id, version_id)
        )
    """)
    # 4 types, 2 super-types
    type_data = [
        (0, 0, "Rural Conservative", 45000.0, 0.15, 0.85, 1.5, "A rural type.", TEST_VERSION),
        (1, 0, "Small Town Traditional", 50000.0, 0.20, 0.80, 2.0, "A small town type.", TEST_VERSION),
        (2, 1, "Suburban Moderate", 75000.0, 0.35, 0.70, 3.5, "A suburban type.", TEST_VERSION),
        (3, 1, "Urban Progressive", 65000.0, 0.45, 0.40, 4.5, "An urban type.", TEST_VERSION),
    ]
    for td in type_data:
        con.execute("INSERT INTO types VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", list(td))

    con.execute("""
        CREATE TABLE super_types (
            super_type_id INTEGER PRIMARY KEY,
            display_name VARCHAR
        )
    """)
    con.execute("INSERT INTO super_types VALUES (0, 'Rural & Conservative')")
    con.execute("INSERT INTO super_types VALUES (1, 'Suburban & Moderate')")

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


def _add_demographics_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate county_demographics with synthetic demographic data."""
    con.execute("""
        CREATE TABLE county_demographics (
            county_fips VARCHAR PRIMARY KEY,
            pop_total BIGINT,
            pct_white_nh DOUBLE,
            pct_black DOUBLE,
            pct_asian DOUBLE,
            pct_hispanic DOUBLE,
            median_age DOUBLE,
            median_hh_income BIGINT,
            log_median_hh_income DOUBLE,
            pct_bachelors_plus DOUBLE,
            pct_graduate DOUBLE,
            pct_owner_occupied DOUBLE,
            pct_wfh DOUBLE,
            pct_transit DOUBLE,
            pct_management DOUBLE
        )
    """)
    demo_data = [
        ("12001", 280000, 0.55, 0.20, 0.06, 0.12, 32.0, 48000, 4.68, 0.42, 0.20, 0.45, 0.12, 0.03, 0.38),
        ("12003", 28000, 0.80, 0.15, 0.01, 0.03, 41.0, 42000, 4.62, 0.12, 0.05, 0.75, 0.03, 0.00, 0.22),
        ("13001", 18500, 0.60, 0.35, 0.01, 0.03, 40.0, 35000, 4.54, 0.14, 0.06, 0.68, 0.02, 0.00, 0.20),
        ("13003", 8300, 0.55, 0.38, 0.01, 0.05, 38.0, 30000, 4.48, 0.10, 0.04, 0.60, 0.01, 0.00, 0.18),
        ("01001", 58000, 0.73, 0.19, 0.01, 0.03, 39.0, 68000, 4.83, 0.30, 0.13, 0.75, 0.05, 0.01, 0.36),
    ]
    for d in demo_data:
        con.execute(
            "INSERT INTO county_demographics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            list(d),
        )


def _add_polling_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate polls, poll_crosstabs, and poll_notes tables."""
    con.execute("""
        CREATE TABLE polls (
            poll_id   VARCHAR NOT NULL,
            race      VARCHAR NOT NULL,
            geography VARCHAR NOT NULL,
            geo_level VARCHAR NOT NULL,
            dem_share FLOAT   NOT NULL,
            n_sample  INTEGER NOT NULL,
            date      VARCHAR,
            pollster  VARCHAR,
            notes     VARCHAR,
            cycle     VARCHAR NOT NULL,
            PRIMARY KEY (poll_id)
        )
    """)
    con.execute("""
        INSERT INTO polls VALUES
        ('abc123', 'FL_Senate', 'FL', 'state', 0.45, 600, '2026-01-15', 'Siena', 'grade=A', '2026')
    """)
    con.execute("CREATE TABLE poll_crosstabs (poll_id VARCHAR, demographic_group VARCHAR, group_value VARCHAR, dem_share FLOAT, n_sample INTEGER, pct_of_sample FLOAT)")
    con.execute("CREATE TABLE poll_notes (poll_id VARCHAR, note_type VARCHAR, note_value VARCHAR)")
    con.execute("INSERT INTO poll_notes VALUES ('abc123', 'grade', 'A')")


def _build_test_db() -> duckdb.DuckDBPyConnection:
    """Build an in-memory DuckDB with synthetic data matching the real schema."""
    con = duckdb.connect(":memory:")
    _add_core_tables(con)
    _add_community_tables(con)
    _add_prediction_tables(con)
    _add_shift_table(con)
    _add_covariance_table(con)
    _add_type_primary_tables(con)
    _add_demographics_table(con)
    _add_polling_tables(con)
    return con


def _build_test_state(K: int = TEST_K) -> dict:
    """Build synthetic in-memory state (sigma, mu_prior, weights)."""
    sigma = np.eye(K) * _SIGMA_DIAGONAL_VARIANCE + np.ones((K, K)) * _SIGMA_OFF_DIAGONAL_COV
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
    test_app.state.contract_ok = True
    # 4×4 synthetic correlation matrix for the 4 test types
    rng = np.random.default_rng(42)
    _raw = rng.random((4, 4))
    _sym = (_raw + _raw.T) / 2
    np.fill_diagonal(_sym, 1.0)
    test_app.state.type_correlation = _sym

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c

    test_db.close()


@pytest.fixture
def client_no_types():
    """TestClient with a DB that has no type-primary tables."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE counties (county_fips VARCHAR PRIMARY KEY, state_abbr VARCHAR, state_fips VARCHAR, county_name VARCHAR, total_votes_2024 INTEGER)")
    con.execute("INSERT INTO counties VALUES ('12001', 'FL', '12', 'Alachua', 100000)")
    con.execute("CREATE TABLE model_versions (version_id VARCHAR PRIMARY KEY, role VARCHAR, k INTEGER, j INTEGER, shift_type VARCHAR, vote_share_type VARCHAR, n_training_dims INTEGER, n_holdout_dims INTEGER, holdout_r VARCHAR, geography VARCHAR, description VARCHAR, created_at TIMESTAMP)")
    con.execute("INSERT INTO model_versions VALUES ('test_v1', 'current', 3, 7, 'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')")
    con.execute("CREATE TABLE community_assignments (county_fips VARCHAR, community_id INTEGER, k INTEGER, version_id VARCHAR, PRIMARY KEY(county_fips, k, version_id))")
    con.execute("INSERT INTO community_assignments VALUES ('12001', 0, 3, 'test_v1')")
    con.execute("CREATE TABLE community_sigma (community_id_row INTEGER, community_id_col INTEGER, sigma_value DOUBLE, version_id VARCHAR)")
    con.execute("INSERT INTO community_sigma VALUES (0, 0, 0.01, 'test_v1')")

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = con
    test_app.state.version_id = "test_v1"
    test_app.state.K = 3
    test_app.state.sigma = np.eye(3) * 0.01
    test_app.state.mu_prior = np.full(3, 0.42)
    test_app.state.state_weights = pd.DataFrame()
    test_app.state.county_weights = pd.DataFrame()
    test_app.state.contract_ok = False

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c
    con.close()
