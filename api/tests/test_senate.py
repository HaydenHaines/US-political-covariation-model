"""Tests for GET /api/v1/senate/overview."""
from __future__ import annotations

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routers.senate import (
    DEM_SAFE_SEATS,
    GOP_SAFE_SEATS,
    SENATE_2026_STATES,
    _build_headline,
    _margin_to_rating,
    _rating_sort_key,
    _CLASS_II_INCUMBENT,
)
from api.tests.conftest import _noop_lifespan


# ── Unit tests for helper functions ────────────────────────────────────────


class TestMarginToRating:
    def test_tossup_within_3pp(self):
        assert _margin_to_rating(0.02) == "tossup"
        assert _margin_to_rating(-0.02) == "tossup"
        assert _margin_to_rating(0.0) == "tossup"

    def test_lean_between_3_and_8pp(self):
        assert _margin_to_rating(0.05) == "lean_d"
        assert _margin_to_rating(-0.05) == "lean_r"
        assert _margin_to_rating(0.03) == "lean_d"

    def test_likely_between_8_and_15pp(self):
        assert _margin_to_rating(0.10) == "likely_d"
        assert _margin_to_rating(-0.10) == "likely_r"
        assert _margin_to_rating(0.08) == "likely_d"

    def test_safe_above_15pp(self):
        assert _margin_to_rating(0.20) == "safe_d"
        assert _margin_to_rating(-0.20) == "safe_r"
        assert _margin_to_rating(0.15) == "safe_d"

    def test_boundary_at_3pp_is_lean(self):
        # abs(margin) == 0.03 is NOT < 0.03, so it's lean
        assert _margin_to_rating(0.03) == "lean_d"

    def test_boundary_at_8pp_is_likely(self):
        assert _margin_to_rating(0.08) == "likely_d"

    def test_boundary_at_15pp_is_safe(self):
        assert _margin_to_rating(0.15) == "safe_d"


class TestRatingSortKey:
    def test_tossup_in_middle(self):
        assert _rating_sort_key("lean_d") < _rating_sort_key("tossup")
        assert _rating_sort_key("tossup") < _rating_sort_key("lean_r")

    def test_safe_r_sorts_last(self):
        assert _rating_sort_key("safe_r") > _rating_sort_key("likely_r")

    def test_full_order(self):
        order = ["safe_d", "likely_d", "lean_d", "tossup", "lean_r", "likely_r", "safe_r"]
        keys = [_rating_sort_key(r) for r in order]
        assert keys == sorted(keys)


class TestBuildHeadline:
    def test_knife_edge(self):
        """When projected seats are within 2, headline says 'Knife's Edge'."""
        # DEM_SAFE=47, GOP_SAFE=53. Need dem_favored - gop_favored to bring
        # the gap to within 2. 6 Dem-leaning races → 53 vs 53 → diff=0.
        races = [{"rating": "lean_d", "margin": 0.05}] * 6
        headline, subtitle, dem_proj, gop_proj = _build_headline(races)
        assert "Knife" in headline

    def test_gop_favored(self):
        """With no competitive races going Dem, GOP holds its safe-seat lead."""
        races = [
            {"rating": "tossup", "margin": 0.01},
            {"rating": "tossup", "margin": -0.01},
            {"rating": "tossup", "margin": 0.02},
        ]
        headline, subtitle, dem_proj, gop_proj = _build_headline(races)
        assert "Republican" in headline

    def test_dem_favored(self):
        """Enough Dem-favored races to overcome the safe-seat deficit."""
        # Need 9+ Dem-leaning to get dem_projected=56 vs gop_projected=53, diff=3
        races = [{"rating": "lean_d", "margin": 0.05}] * 9
        headline, subtitle, dem_proj, gop_proj = _build_headline(races)
        assert "Democrat" in headline

    def test_projected_counts_returned(self):
        """_build_headline returns accurate projected seat counts."""
        # 6 lean_d races → dem_favored=6, gop_favored=0
        # dem_projected = 47 + 6 = 53, gop_projected = 53 + 0 = 53
        races = [{"rating": "lean_d", "margin": 0.05}] * 6
        _, _, dem_proj, gop_proj = _build_headline(races)
        assert dem_proj == DEM_SAFE_SEATS + 6
        assert gop_proj == GOP_SAFE_SEATS

    def test_tossups_excluded_from_projections(self):
        """Tossups do not count toward either party's projected total."""
        races = [{"rating": "tossup", "margin": 0.02}] * 5
        _, _, dem_proj, gop_proj = _build_headline(races)
        # Tossups add 0 to both sides
        assert dem_proj == DEM_SAFE_SEATS
        assert gop_proj == GOP_SAFE_SEATS


# ── Fixtures ────────────────────────────────────────────────────────────────


def _build_senate_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with synthetic Senate race data.

    Uses TX and GA — both are in SENATE_2026_STATES.
    """
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
        ("48001", "TX", "48", "Anderson County, TX", 100000),
        ("48003", "TX", "48", "Andrews County, TX",    15000),
        ("13001", "GA", "13", "Appling County, GA",  80000),
        ("13003", "GA", "13", "Atkinson County, GA",  8000),
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
    # TX Senate: predictions (not directly used by new code, kept for schema)
    for fips, share in [("48001", 0.502), ("48003", 0.498)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 TX Senate', 'test_v1', ?, 0.03, ?, ?, 0.50, 0.50)",
            [fips, share, share - 0.05, share + 0.05],
        )
    # GA Senate
    for fips, share in [("13001", 0.462), ("13003", 0.458)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 GA Senate', 'test_v1', ?, 0.03, ?, ?, 0.46, 0.46)",
            [fips, share, share - 0.05, share + 0.05],
        )

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
    con.execute(
        "INSERT INTO polls VALUES ('p1', '2026 TX Senate', 'TX', 'state', 0.50, 600, "
        "'2026-01-15', 'Siena', NULL, '2026')"
    )
    con.execute(
        "INSERT INTO polls VALUES ('p2', '2026 TX Senate', 'TX', 'state', 0.49, 700, "
        "'2026-02-01', 'Quinnipiac', NULL, '2026')"
    )
    # No polls for GA Senate

    return con


def _build_type_model_state(app) -> None:
    """Set up minimal type model data on app.state for senate overview.

    Creates 4 synthetic tracts (2 TX, 2 GA) with J=3 types.
    TX tracts have priors ~0.50 (tossup), GA tracts ~0.46 (lean R).
    """
    J = 3
    tract_fips = ["48001", "48003", "13001", "13003"]
    # Soft membership scores (4 tracts x 3 types)
    app.state.type_scores = np.array([
        [0.6, 0.3, 0.1],
        [0.5, 0.4, 0.1],
        [0.2, 0.5, 0.3],
        [0.3, 0.4, 0.3],
    ])
    app.state.type_county_fips = tract_fips
    app.state.type_covariance = np.eye(J) * 0.01
    app.state.type_priors = np.array([0.52, 0.48, 0.45])
    app.state.ridge_priors = {
        "48001": 0.502, "48003": 0.498,  # TX: ~tossup
        "13001": 0.462, "13003": 0.458,  # GA: lean R
    }
    app.state.tract_states = {
        "48001": "TX", "48003": "TX",
        "13001": "GA", "13003": "GA",
    }
    app.state.tract_votes = {
        "48001": 100000, "48003": 15000,
        "13001": 80000, "13003": 8000,
    }
    app.state.behavior_tau = None
    app.state.behavior_delta = None


@pytest.fixture
def senate_client():
    con = _build_senate_db()
    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = con
    test_app.state.version_id = "test_v1"
    test_app.state.contract_ok = True
    _build_type_model_state(test_app)
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c
    con.close()


# ── Endpoint tests ──────────────────────────────────────────────────────────


class TestSenateOverview:
    def test_status_200(self, senate_client):
        resp = senate_client.get("/api/v1/senate/overview")
        assert resp.status_code == 200

    def test_response_shape(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        assert "headline" in data
        assert "subtitle" in data
        assert "dem_seats_safe" in data
        assert "gop_seats_safe" in data
        assert "dem_projected" in data
        assert "gop_projected" in data
        assert "races" in data
        assert isinstance(data["races"], list)

    def test_safe_seat_counts(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        assert data["dem_seats_safe"] == DEM_SAFE_SEATS
        assert data["gop_seats_safe"] == GOP_SAFE_SEATS

    def test_projected_seat_counts_present(self, senate_client):
        """Projected totals are included and are at least as large as safe totals."""
        data = senate_client.get("/api/v1/senate/overview").json()
        assert data["dem_projected"] >= DEM_SAFE_SEATS
        assert data["gop_projected"] >= GOP_SAFE_SEATS

    def test_only_senate_races_returned(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert "senate" in race["race"].lower() or "Senate" in race["race"]

    def test_tx_senate_has_polls(self, senate_client):
        """TX has 2 polls; predict_race runs and produces a real prediction."""
        data = senate_client.get("/api/v1/senate/overview").json()
        tx = next(r for r in data["races"] if r["state"] == "TX")
        assert tx["n_polls"] == 2
        # With polls at ~0.50, prediction should be near tossup
        assert abs(tx["margin"]) < 0.10

    def test_ga_senate_no_polls_uses_prior(self, senate_client):
        """GA has no polls; uses behavior-adjusted prior (~0.46 = lean R)."""
        data = senate_client.get("/api/v1/senate/overview").json()
        ga = next(r for r in data["races"] if r["state"] == "GA")
        assert ga["n_polls"] == 0
        assert ga["margin"] < 0  # GOP-favored from prior

    def test_slug_format(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert " " not in race["slug"]
            assert race["slug"] == race["race"].lower().replace(" ", "-")

    def test_poll_counts(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        tx = next(r for r in data["races"] if r["state"] == "TX")
        ga = next(r for r in data["races"] if r["state"] == "GA")
        assert tx["n_polls"] == 2
        assert ga["n_polls"] == 0

    def test_tossup_sorts_before_lean(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        ratings = [r["rating"] for r in data["races"]]
        sort_order = [_rating_sort_key(r) for r in ratings]
        assert sort_order == sorted(sort_order)

    def test_headline_is_string(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        assert isinstance(data["headline"], str)
        assert len(data["headline"]) > 0

    def test_no_governor_race_in_output(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert "governor" not in race["race"].lower()

    def test_margin_is_rounded_float(self, senate_client):
        data = senate_client.get("/api/v1/senate/overview").json()
        for race in data["races"]:
            assert isinstance(race["margin"], float)


class TestSenateOverviewEmptyDB:
    """Endpoint with no predictions — returns all 33 races using incumbent fallbacks."""

    def test_returns_all_races_via_fallback(self):
        with duckdb.connect(":memory:") as con:
            con.execute(
                "CREATE TABLE model_versions (version_id VARCHAR PRIMARY KEY, role VARCHAR, "
                "k INTEGER, j INTEGER, shift_type VARCHAR, vote_share_type VARCHAR, "
                "n_training_dims INTEGER, n_holdout_dims INTEGER, holdout_r VARCHAR, "
                "geography VARCHAR, description VARCHAR, created_at TIMESTAMP)"
            )
            con.execute(
                "INSERT INTO model_versions VALUES ('v1', 'current', 3, 7, 'logodds', "
                "'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')"
            )
            con.execute(
                "CREATE TABLE counties (county_fips VARCHAR PRIMARY KEY, state_abbr VARCHAR, "
                "state_fips VARCHAR, county_name VARCHAR, total_votes_2024 INTEGER)"
            )
            con.execute(
                "CREATE TABLE predictions (county_fips VARCHAR, race VARCHAR, version_id VARCHAR, "
                "pred_dem_share DOUBLE, pred_std DOUBLE, pred_lo90 DOUBLE, pred_hi90 DOUBLE, "
                "state_pred DOUBLE, poll_avg DOUBLE)"
            )
            con.execute(
                "CREATE TABLE polls (poll_id VARCHAR PRIMARY KEY, race VARCHAR, geography VARCHAR, "
                "geo_level VARCHAR, dem_share FLOAT, n_sample INTEGER, date VARCHAR, "
                "pollster VARCHAR, notes VARCHAR, cycle VARCHAR)"
            )

            test_app = create_app(lifespan_override=_noop_lifespan)
            test_app.state.db = con
            test_app.state.version_id = "v1"
            test_app.state.contract_ok = True

            with TestClient(test_app, raise_server_exceptions=True) as c:
                resp = c.get("/api/v1/senate/overview")

        assert resp.status_code == 200
        data = resp.json()
        # All 33 Class II races should appear — using incumbent-party fallbacks
        # when no model predictions are available.
        assert len(data["races"]) == len(SENATE_2026_STATES)
        assert data["dem_seats_safe"] == DEM_SAFE_SEATS
        assert data["gop_seats_safe"] == GOP_SAFE_SEATS
        assert "dem_projected" in data
        assert "gop_projected" in data
        # Without predictions every race uses safe defaults — all are safe_d or safe_r
        ratings = {r["rating"] for r in data["races"]}
        assert ratings <= {"safe_d", "safe_r"}
        # Verify incumbent-party fallback for a known state
        tx_race = next(r for r in data["races"] if r["state"] == "TX")
        assert tx_race["rating"] == "safe_r"  # TX is R-held (John Cornyn)
        assert tx_race["n_polls"] == 0
        il_race = next(r for r in data["races"] if r["state"] == "IL")
        assert il_race["rating"] == "safe_d"  # IL is D-held (Dick Durbin)
