"""Tests for candidate_info field on the race detail response.

Validates that the API correctly reads candidates_2026.json and includes
candidate data (incumbent, status, rating, challengers) in the race detail
response. Uses the real candidates_2026.json file on disk — no mocking —
because the data is static and checked into version control.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.tests.conftest import _build_test_state, _noop_lifespan

# Re-use the race detail DB builder and test fixture pattern from test_race_detail.
# We need GA Senate predictions in the test DB so the race detail endpoint can find them.

TEST_VERSION = "test_v1"


def _build_candidate_test_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a GA Senate race for candidate info testing.

    GA Senate is chosen because it has rich candidate data in candidates_2026.json:
    incumbent Jon Ossoff (D), multiple R challengers, rated Toss-up.
    """
    con = duckdb.connect(":memory:")

    con.execute("""
        CREATE TABLE counties (
            county_fips VARCHAR PRIMARY KEY, state_abbr VARCHAR NOT NULL,
            state_fips VARCHAR NOT NULL, county_name VARCHAR, total_votes_2024 INTEGER
        )
    """)
    con.execute("INSERT INTO counties VALUES ('13001', 'GA', '13', 'Appling County, GA', 80000)")
    con.execute("INSERT INTO counties VALUES ('13003', 'GA', '13', 'Atkinson County, GA', 8000)")

    con.execute("""
        CREATE TABLE model_versions (
            version_id VARCHAR PRIMARY KEY, role VARCHAR, k INTEGER, j INTEGER,
            shift_type VARCHAR, vote_share_type VARCHAR, n_training_dims INTEGER,
            n_holdout_dims INTEGER, holdout_r VARCHAR, geography VARCHAR,
            description VARCHAR, created_at TIMESTAMP
        )
    """)
    con.execute(
        "INSERT INTO model_versions VALUES "
        "(?, 'current', 3, 4, 'logodds', 'total', 30, 3, "
        "'0.70', 'test', 'test', '2026-01-01')",
        [TEST_VERSION],
    )

    con.execute("""
        CREATE TABLE predictions (
            county_fips VARCHAR NOT NULL, race VARCHAR NOT NULL, version_id VARCHAR NOT NULL,
            pred_dem_share DOUBLE, pred_std DOUBLE, pred_lo90 DOUBLE, pred_hi90 DOUBLE,
            state_pred DOUBLE, poll_avg DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    for fips in ["13001", "13003"]:
        con.execute(
            "INSERT INTO predictions VALUES "
            "(?, '2026 GA Senate', ?, 0.49, 0.03, "
            "0.44, 0.54, 0.49, NULL)",
            [fips, TEST_VERSION],
        )

    con.execute("""
        CREATE TABLE races (
            race_id VARCHAR PRIMARY KEY, race_type VARCHAR NOT NULL,
            state VARCHAR NOT NULL, year INTEGER NOT NULL, district INTEGER
        )
    """)
    con.execute("INSERT INTO races VALUES ('2026 GA Senate', 'senate', 'GA', 2026, NULL)")

    con.execute("""
        CREATE TABLE types (
            type_id INTEGER NOT NULL, super_type_id INTEGER NOT NULL, display_name VARCHAR NOT NULL,
            median_hh_income DOUBLE, pct_bachelors_plus DOUBLE, pct_white_nh DOUBLE,
            log_pop_density DOUBLE, narrative VARCHAR, version_id VARCHAR NOT NULL,
            PRIMARY KEY (type_id, version_id)
        )
    """)
    con.execute(
        "INSERT INTO types VALUES "
        "(0, 0, 'Rural Conservative', 45000, 0.15, 0.85, "
        "1.5, NULL, ?)",
        [TEST_VERSION],
    )

    con.execute(
        "CREATE TABLE super_types "
        "(super_type_id INTEGER PRIMARY KEY, display_name VARCHAR)"
    )
    con.execute("INSERT INTO super_types VALUES (0, 'Rural')")

    con.execute("""
        CREATE TABLE county_type_assignments (
            county_fips VARCHAR NOT NULL, dominant_type INTEGER NOT NULL,
            super_type INTEGER NOT NULL, version_id VARCHAR NOT NULL,
            PRIMARY KEY (county_fips, version_id)
        )
    """)
    con.execute("INSERT INTO county_type_assignments VALUES ('13001', 0, 0, ?)", [TEST_VERSION])
    con.execute("INSERT INTO county_type_assignments VALUES ('13003', 0, 0, ?)", [TEST_VERSION])

    # Polls table (empty for this race — candidate info doesn't depend on polls)
    con.execute("""
        CREATE TABLE polls (
            poll_id VARCHAR NOT NULL, race VARCHAR NOT NULL, geography VARCHAR NOT NULL,
            geo_level VARCHAR NOT NULL, dem_share FLOAT NOT NULL, n_sample INTEGER NOT NULL,
            date VARCHAR, pollster VARCHAR, notes VARCHAR, cycle VARCHAR NOT NULL,
            PRIMARY KEY (poll_id)
        )
    """)
    con.execute(
        "CREATE TABLE poll_crosstabs (poll_id VARCHAR, "
        "demographic_group VARCHAR, group_value VARCHAR, "
        "dem_share FLOAT, n_sample INTEGER)"
    )
    con.execute(
        "CREATE TABLE poll_notes (poll_id VARCHAR, "
        "note_type VARCHAR, note_value VARCHAR)"
    )
    con.execute(
        "CREATE TABLE county_shifts (county_fips VARCHAR, "
        "version_id VARCHAR, pres_d_shift_00_04 DOUBLE)"
    )
    con.execute(
        "CREATE TABLE county_demographics ("
        "county_fips VARCHAR PRIMARY KEY, pop_total BIGINT, "
        "pct_white_nh DOUBLE, pct_black DOUBLE, "
        "pct_asian DOUBLE, pct_hispanic DOUBLE, "
        "median_age DOUBLE, median_hh_income BIGINT, "
        "log_median_hh_income DOUBLE, pct_bachelors_plus DOUBLE, "
        "pct_graduate DOUBLE, pct_owner_occupied DOUBLE, "
        "pct_wfh DOUBLE, pct_transit DOUBLE, "
        "pct_management DOUBLE)"
    )

    return con


@pytest.fixture
def candidate_client():
    """TestClient with GA Senate race data for candidate info tests."""
    db = _build_candidate_test_db()
    state = _build_test_state()

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = db
    test_app.state.version_id = TEST_VERSION
    test_app.state.K = 3
    test_app.state.sigma = np.eye(3) * 0.01
    test_app.state.mu_prior = np.full(3, 0.42)
    test_app.state.state_weights = state["state_weights"]
    test_app.state.county_weights = state["county_weights"]
    test_app.state.contract_ok = True

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c

    db.close()


class TestCandidateInfo:
    """Tests for candidate_info field on the race detail response."""

    def test_race_detail_includes_candidate_info_field(self, candidate_client):
        """Race detail response includes candidate_info key."""
        resp = candidate_client.get("/api/v1/forecast/race/2026-ga-senate")
        assert resp.status_code == 200
        data = resp.json()
        assert "candidate_info" in data

    def test_ga_senate_has_candidate_data(self, candidate_client):
        """GA Senate is in candidates_2026.json and should return candidate data."""
        resp = candidate_client.get("/api/v1/forecast/race/2026-ga-senate")
        data = resp.json()
        ci = data["candidate_info"]
        assert ci is not None, "GA Senate should have candidate_info from candidates_2026.json"

    def test_incumbent_structure(self, candidate_client):
        """Candidate info includes incumbent name and party."""
        resp = candidate_client.get("/api/v1/forecast/race/2026-ga-senate")
        ci = resp.json()["candidate_info"]
        assert ci["incumbent"]["name"] == "Jon Ossoff"
        assert ci["incumbent"]["party"] == "D"

    def test_status_field(self, candidate_client):
        """GA Senate incumbent is running (status = 'incumbent_running')."""
        resp = candidate_client.get("/api/v1/forecast/race/2026-ga-senate")
        ci = resp.json()["candidate_info"]
        assert ci["status"] == "incumbent_running"

    def test_rating_field(self, candidate_client):
        """GA Senate is rated Toss-up in the candidate data."""
        resp = candidate_client.get("/api/v1/forecast/race/2026-ga-senate")
        ci = resp.json()["candidate_info"]
        assert ci["rating"] == "Toss-up"

    def test_candidates_by_party(self, candidate_client):
        """Candidates dict has party keys with lists of candidate names."""
        resp = candidate_client.get("/api/v1/forecast/race/2026-ga-senate")
        ci = resp.json()["candidate_info"]
        assert "R" in ci["candidates"]
        assert "D" in ci["candidates"]
        # Ossoff is the D candidate
        assert "Jon Ossoff" in ci["candidates"]["D"]
        # At least one R challenger
        assert len(ci["candidates"]["R"]) > 0

    def test_candidate_info_none_for_unknown_race(self, candidate_client):
        """Races not in candidates_2026.json return candidate_info=None.

        The test DB only has GA Senate predictions; a made-up race
        would 404 from the predictions check. But we can check via
        the _build_candidate_info helper directly.
        """
        from api.routers.forecast.race_detail import _build_candidate_info

        result = _build_candidate_info("2099 ZZ Fake Race")
        assert result is None


class TestCandidateInfoUnit:
    """Unit tests for _build_candidate_info helper."""

    def test_returns_none_for_missing_race(self):
        from api.routers.forecast.race_detail import _build_candidate_info

        assert _build_candidate_info("9999 XX Nonexistent") is None

    def test_returns_candidate_info_for_known_race(self):
        from api.routers.forecast.race_detail import _build_candidate_info

        result = _build_candidate_info("2026 GA Senate")
        assert result is not None
        assert result.incumbent.name == "Jon Ossoff"
        assert result.incumbent.party == "D"
        assert result.status == "incumbent_running"
        assert result.rating == "Toss-up"

    def test_open_seat_has_status_detail(self):
        """Open seats include status_detail explaining why the seat is open."""
        from api.routers.forecast.race_detail import _build_candidate_info

        result = _build_candidate_info("2026 MI Senate")
        assert result is not None
        assert result.status == "open"
        assert result.status_detail is not None
        assert "Peters" in result.status_detail

    def test_special_election_status(self):
        """Special elections have status='special'."""
        from api.routers.forecast.race_detail import _build_candidate_info

        result = _build_candidate_info("2026 OH Senate")
        assert result is not None
        assert result.status == "special"
        assert "Vance" in result.status_detail
