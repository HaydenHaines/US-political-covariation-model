"""Tests for GET /api/v1/governor/overview.

Regression tests for the counties-table-empty bug (2026-04-29):
  When the counties table is empty the predictions JOIN returns NULL for every
  race, causing classify_governor_race() to fall back to _DEFAULT_SAFE_MARGIN
  (±0.25).  The frontend then displays D+25pp or R+25pp for every race.

These tests guard against that regression by verifying:
  1. When counties are populated, margins reflect model predictions (not a
     uniform fallback constant).
  2. When version_id is None, the endpoint returns structural fallbacks without
     crashing.
  3. When the counties table is empty (simulating the bug), the endpoint still
     returns 200 but each race hits the structural fallback.
"""
from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routers.governor._helpers import _DEFAULT_SAFE_MARGIN
from api.tests.conftest import _noop_lifespan

_AZ_LEAN_D_PRED = 0.565   # ≈ AZ Governor real model value (~D+13pp after incumbent bonus)
_TX_SAFE_R_PRED = 0.432   # TX Governor real model value (~R+14pp)


def _build_governor_overview_db(include_counties: bool = True) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with AZ (lean D) and TX (safe R) governor races."""
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
    if include_counties:
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
        "INSERT INTO model_versions VALUES "
        "('test_v1', 'current', 3, 7, 'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')"
    )

    con.execute("""
        CREATE TABLE predictions (
            county_fips    VARCHAR NOT NULL,
            race           VARCHAR NOT NULL,
            version_id     VARCHAR NOT NULL,
            pred_dem_share DOUBLE,
            pred_std       DOUBLE,
            pred_lo90      DOUBLE,
            pred_hi90      DOUBLE,
            state_pred     DOUBLE,
            poll_avg       DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    # AZ Governor: lean D (both counties ~D+13pp)
    for fips, share in [("04001", _AZ_LEAN_D_PRED), ("04003", _AZ_LEAN_D_PRED - 0.01)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 AZ Governor', 'test_v1', ?, 0.04, ?, ?, ?, ?)",
            [fips, share, share - 0.07, share + 0.07, share, share],
        )
    # TX Governor: safe R (~R+14pp)
    con.execute(
        "INSERT INTO predictions VALUES ('48001', '2026 TX Governor', 'test_v1', ?, 0.04, ?, ?, ?, ?)",
        [_TX_SAFE_R_PRED, _TX_SAFE_R_PRED - 0.07, _TX_SAFE_R_PRED + 0.07,
         _TX_SAFE_R_PRED, _TX_SAFE_R_PRED],
    )

    return con


@pytest.fixture
def overview_client():
    con = _build_governor_overview_db(include_counties=True)
    app = create_app(lifespan_override=_noop_lifespan)
    app.state.db = con
    app.state.version_id = "test_v1"
    app.state.contract_ok = True
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    con.close()


@pytest.fixture
def empty_counties_client():
    """Client whose counties table is empty — simulates the 2026-04-29 bug."""
    con = _build_governor_overview_db(include_counties=False)
    app = create_app(lifespan_override=_noop_lifespan)
    app.state.db = con
    app.state.version_id = "test_v1"
    app.state.contract_ok = True
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    con.close()


@pytest.fixture
def no_version_client():
    """Client with no version_id set — no model loaded."""
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE counties (county_fips VARCHAR PRIMARY KEY, state_abbr VARCHAR NOT NULL,"
        " state_fips VARCHAR NOT NULL, county_name VARCHAR, total_votes_2024 INTEGER)"
    )
    app = create_app(lifespan_override=_noop_lifespan)
    app.state.db = con
    app.state.version_id = None
    app.state.contract_ok = True
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    con.close()


class TestGovernorOverviewWithModel:
    """Happy-path tests: counties populated, model loaded."""

    def test_returns_200(self, overview_client):
        resp = overview_client.get("/api/v1/governor/overview")
        assert resp.status_code == 200

    def test_races_field_present(self, overview_client):
        data = overview_client.get("/api/v1/governor/overview").json()
        assert "races" in data
        assert isinstance(data["races"], list)

    def test_az_uses_model_prediction_not_fallback(self, overview_client):
        """AZ margin must come from the predictions table, not _DEFAULT_SAFE_MARGIN.

        Regression: when counties was empty, every race returned ±_DEFAULT_SAFE_MARGIN
        because the JOIN produced NULL and classify_governor_race() used the fallback.
        """
        data = overview_client.get("/api/v1/governor/overview").json()
        az = next((r for r in data["races"] if r["state"] == "AZ"), None)
        assert az is not None, "AZ race missing from response"
        # Model prediction puts AZ at D+13pp (margin ≈ 0.105 after incumbency bonus).
        # The fallback would put it at exactly +0.25. Ensure we are NOT seeing the fallback.
        assert abs(az["margin"] - _DEFAULT_SAFE_MARGIN) > 0.05, (
            f"AZ margin {az['margin']} looks like the hardcoded fallback "
            f"(_DEFAULT_SAFE_MARGIN={_DEFAULT_SAFE_MARGIN}). "
            "Did the counties table fail to JOIN?"
        )

    def test_tx_uses_model_prediction_not_fallback(self, overview_client):
        """TX (R-held) margin must come from model, not -_DEFAULT_SAFE_MARGIN."""
        data = overview_client.get("/api/v1/governor/overview").json()
        tx = next((r for r in data["races"] if r["state"] == "TX"), None)
        assert tx is not None, "TX race missing from response"
        assert abs(tx["margin"] - (-_DEFAULT_SAFE_MARGIN)) > 0.05, (
            f"TX margin {tx['margin']} looks like the hardcoded fallback. "
            "Did the counties table fail to JOIN?"
        )

    def test_modeled_races_have_varying_margins(self, overview_client):
        """AZ and TX margins must differ — a constant value signals fallback mode."""
        data = overview_client.get("/api/v1/governor/overview").json()
        az = next((r for r in data["races"] if r["state"] == "AZ"), None)
        tx = next((r for r in data["races"] if r["state"] == "TX"), None)
        if az and tx:
            assert az["margin"] != tx["margin"], (
                "AZ and TX have identical margins — this indicates the fallback "
                "constant is being returned instead of model predictions."
            )

    def test_race_fields_present(self, overview_client):
        data = overview_client.get("/api/v1/governor/overview").json()
        for race in data["races"]:
            assert "state" in race
            assert "race" in race
            assert "slug" in race
            assert "rating" in race
            assert "margin" in race
            assert "incumbent_party" in race
            assert "is_open_seat" in race
            assert "n_polls" in race

    def test_az_rated_as_dem_lean_or_likely(self, overview_client):
        """AZ (lean D) should not be rated as safe_r or tossup given model prediction."""
        data = overview_client.get("/api/v1/governor/overview").json()
        az = next((r for r in data["races"] if r["state"] == "AZ"), None)
        assert az is not None
        assert az["rating"] in {"lean_d", "likely_d", "safe_d"}, (
            f"AZ rated '{az['rating']}' but model predicts D+13pp"
        )

    def test_tx_rated_as_rep_leaning(self, overview_client):
        """TX (safe R) should be rated R-leaning given model prediction."""
        data = overview_client.get("/api/v1/governor/overview").json()
        tx = next((r for r in data["races"] if r["state"] == "TX"), None)
        assert tx is not None
        assert tx["rating"] in {"lean_r", "likely_r", "safe_r"}, (
            f"TX rated '{tx['rating']}' but model predicts R+14pp"
        )


class TestGovernorOverviewEmptyCounties:
    """Regression tests for the 2026-04-29 bug: counties table empty."""

    def test_still_returns_200_when_counties_empty(self, empty_counties_client):
        """Empty counties must not crash the endpoint."""
        resp = empty_counties_client.get("/api/v1/governor/overview")
        assert resp.status_code == 200

    def test_az_falls_back_to_safe_d_when_counties_empty(self, empty_counties_client):
        """When counties is empty the JOIN fails; AZ falls back to +_DEFAULT_SAFE_MARGIN."""
        data = empty_counties_client.get("/api/v1/governor/overview").json()
        az = next((r for r in data["races"] if r["state"] == "AZ"), None)
        assert az is not None
        # With empty counties: margin == _DEFAULT_SAFE_MARGIN for D incumbents
        assert az["margin"] == _DEFAULT_SAFE_MARGIN, (
            f"Expected fallback margin {_DEFAULT_SAFE_MARGIN}, got {az['margin']}"
        )

    def test_fallback_rating_is_safe_d_for_d_incumbents(self, empty_counties_client):
        data = empty_counties_client.get("/api/v1/governor/overview").json()
        az = next((r for r in data["races"] if r["state"] == "AZ"), None)
        assert az is not None
        assert az["rating"] == "safe_d"


class TestGovernorOverviewNoModel:
    """When no model is loaded (version_id=None), structural fallbacks are returned."""

    def test_returns_200(self, no_version_client):
        resp = no_version_client.get("/api/v1/governor/overview")
        assert resp.status_code == 200

    def test_races_list_present(self, no_version_client):
        data = no_version_client.get("/api/v1/governor/overview").json()
        assert "races" in data
        assert isinstance(data["races"], list)
        assert len(data["races"]) > 0

    def test_all_margins_are_fallback_constants(self, no_version_client):
        """With no model every race should be at ±_DEFAULT_SAFE_MARGIN."""
        data = no_version_client.get("/api/v1/governor/overview").json()
        for race in data["races"]:
            assert abs(abs(race["margin"]) - _DEFAULT_SAFE_MARGIN) < 1e-9, (
                f"{race['state']} margin {race['margin']} is not ±{_DEFAULT_SAFE_MARGIN}"
            )

    def test_updated_at_is_none_when_no_model(self, no_version_client):
        data = no_version_client.get("/api/v1/governor/overview").json()
        assert data.get("updated_at") is None
