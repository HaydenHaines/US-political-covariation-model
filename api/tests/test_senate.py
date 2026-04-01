"""Tests for GET /api/v1/senate/overview and GET /api/v1/senate/chamber-probability."""
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
    _simulate_chamber_probability,
    _CLASS_II_INCUMBENT,
    _DEM_CLASS_II_COUNT,
    _GOP_CLASS_II_COUNT,
    _DEM_HOLDOVER_SEATS,
    _GOP_HOLDOVER_SEATS,
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
        """When projected seats are within 2, headline says 'Knife\'s Edge'."""
        # Holdovers: _DEM_HOLDOVER_SEATS=33, _GOP_HOLDOVER_SEATS=34 -> diff=-1 -> knife edge.
        # 1 lean_d race -> 34D vs 34R -> diff=0 -> still knife edge.
        races = [{"rating": "lean_d", "margin": 0.05}] * 1
        headline, subtitle, dem_proj, gop_proj = _build_headline(races)
        assert "Knife" in headline

    def test_gop_favored(self):
        """GOP-favored headline requires holdover+wins gap to exceed 2 seats."""
        # _GOP_HOLDOVER_SEATS(34) - _DEM_HOLDOVER_SEATS(33) = 1 -> within knife-edge range.
        # Add 2 lean_r races to push GOP to 36, Dem stays 33 -> diff = -3.
        races = [{"rating": "lean_r", "margin": -0.05}] * 2
        headline, subtitle, dem_proj, gop_proj = _build_headline(races)
        assert "Republican" in headline

    def test_dem_favored(self):
        """Enough Dem-favored races to overcome the holdover deficit."""
        # _DEM_HOLDOVER_SEATS=33, _GOP_HOLDOVER_SEATS=34.
        # 4 lean_d -> dem=37 vs gop=34, diff=+3 -> Democrats Favored.
        races = [{"rating": "lean_d", "margin": 0.05}] * 4
        headline, subtitle, dem_proj, gop_proj = _build_headline(races)
        assert "Democrat" in headline

    def test_projected_counts_returned(self):
        """_build_headline returns accurate projected seat counts.

        Projected totals start from HOLDOVER seats (not up in 2026), not
        the full current chamber. Using DEM_SAFE_SEATS(47)/GOP_SAFE_SEATS(53)
        as the base double-counts Class II seats (producing 126 instead of 100).
        """
        # 6 lean_d races -> dem_favored=6, gop_favored=0
        races = [{"rating": "lean_d", "margin": 0.05}] * 6
        _, _, dem_proj, gop_proj = _build_headline(races)
        assert dem_proj == _DEM_HOLDOVER_SEATS + 6
        assert gop_proj == _GOP_HOLDOVER_SEATS

    def test_tossups_excluded_from_projections(self):
        """Tossups do not count toward either party's projected total."""
        races = [{"rating": "tossup", "margin": 0.02}] * 5
        _, _, dem_proj, gop_proj = _build_headline(races)
        # Tossups add 0 to both sides; baseline is holdover seats not up in 2026
        assert dem_proj == _DEM_HOLDOVER_SEATS
        assert gop_proj == _GOP_HOLDOVER_SEATS


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
        """Projected totals start from holdover seats, not current totals.

        Tossups are excluded from both sides, so D+R <= 100.
        """
        data = senate_client.get("/api/v1/senate/overview").json()
        assert data["dem_projected"] >= _DEM_HOLDOVER_SEATS
        assert data["gop_projected"] >= _GOP_HOLDOVER_SEATS
        assert data["dem_projected"] + data["gop_projected"] <= 100

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

# ── Chamber probability unit tests ──────────────────────────────────────────


class TestIncumbentCounts:
    """Derived incumbent counts stay in sync with the _CLASS_II_INCUMBENT map."""

    def test_dem_count_matches_map(self):
        expected = sum(1 for p in _CLASS_II_INCUMBENT.values() if p == "D")
        assert _DEM_CLASS_II_COUNT == expected

    def test_gop_count_matches_map(self):
        expected = sum(1 for p in _CLASS_II_INCUMBENT.values() if p == "R")
        assert _GOP_CLASS_II_COUNT == expected

    def test_counts_sum_to_33(self):
        # There are 33 Class II seats up in 2026
        assert _DEM_CLASS_II_COUNT + _GOP_CLASS_II_COUNT == len(SENATE_2026_STATES)

    def test_holdover_seats_correct(self):
        # Holdover = safe seats not up in 2026 on the Dem side
        assert _DEM_HOLDOVER_SEATS == DEM_SAFE_SEATS - _DEM_CLASS_II_COUNT


class TestSimulateChamberProbability:
    """Unit tests for _simulate_chamber_probability helper."""

    def test_certain_dem_control(self):
        """When all races are strongly Dem (pred=0.9, tiny std), Dems win every sim."""
        modeled = [(0.9, 0.01)] * 33
        result = _simulate_chamber_probability(
            modeled_races=modeled,
            safe_dem_wins=0,
            safe_gop_wins=0,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=1000,
            rng_seed=0,
        )
        assert result.dem_majority_pct > 99.0
        assert result.rep_control_pct < 1.0

    def test_certain_gop_control(self):
        """When all races are strongly GOP (pred=0.1, tiny std), GOP wins every sim."""
        modeled = [(0.1, 0.01)] * 33
        result = _simulate_chamber_probability(
            modeled_races=modeled,
            safe_dem_wins=0,
            safe_gop_wins=0,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=1000,
            rng_seed=0,
        )
        assert result.rep_control_pct > 99.0
        assert result.dem_majority_pct < 1.0

    def test_tossup_near_fifty_fifty(self):
        """All races at dead 50/50 -- neither party should dominate."""
        modeled = [(0.5, 0.03)] * 33
        result = _simulate_chamber_probability(
            modeled_races=modeled,
            safe_dem_wins=0,
            safe_gop_wins=0,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=5000,
            rng_seed=42,
        )
        assert 10.0 < result.dem_majority_pct < 90.0
        assert 10.0 < result.rep_control_pct < 90.0

    def test_probabilities_sum_near_100(self):
        """rep_control_pct + dem_control_pct should cover all outcomes."""
        modeled = [(0.5, 0.05)] * 10
        result = _simulate_chamber_probability(
            modeled_races=modeled,
            safe_dem_wins=5,
            safe_gop_wins=18,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=2000,
            rng_seed=7,
        )
        total = result.dem_control_pct + result.rep_control_pct
        assert abs(total - 100.0) < 0.5

    def test_seat_distribution_sums_to_one(self):
        """Probability mass in seat_distribution should sum close to 1.0."""
        modeled = [(0.5, 0.04)] * 15
        result = _simulate_chamber_probability(
            modeled_races=modeled,
            safe_dem_wins=5,
            safe_gop_wins=13,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=2000,
            rng_seed=99,
        )
        total_prob = sum(b.probability for b in result.seat_distribution)
        assert 0.95 < total_prob <= 1.01

    def test_metadata_fields_correct(self):
        """n_modeled_races and n_safe_races reflect the inputs."""
        modeled = [(0.52, 0.03)] * 8
        result = _simulate_chamber_probability(
            modeled_races=modeled,
            safe_dem_wins=10,
            safe_gop_wins=15,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=500,
            rng_seed=1,
        )
        assert result.n_modeled_races == 8
        assert result.n_safe_races == 25
        assert result.n_simulations == 500

    def test_median_dem_plus_rep_equals_100(self):
        """Median Dem + Rep seats should always add to 100."""
        modeled = [(0.48, 0.05)] * 20
        result = _simulate_chamber_probability(
            modeled_races=modeled,
            safe_dem_wins=3,
            safe_gop_wins=10,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=1000,
            rng_seed=5,
        )
        assert result.median_dem_seats + result.median_rep_seats == 100

    def test_no_modeled_races(self):
        """When there are no contested races, safe seats determine the outcome."""
        result = _simulate_chamber_probability(
            modeled_races=[],
            safe_dem_wins=_DEM_CLASS_II_COUNT,
            safe_gop_wins=_GOP_CLASS_II_COUNT,
            dem_holdover=_DEM_HOLDOVER_SEATS,
            n_sims=1000,
            rng_seed=0,
        )
        assert result.rep_control_pct > 90.0


# ── Chamber probability endpoint tests ──────────────────────────────────────


class TestChamberProbabilityEndpoint:
    """Integration tests for GET /api/v1/senate/chamber-probability."""

    def test_status_200(self, senate_client):
        resp = senate_client.get("/api/v1/senate/chamber-probability")
        assert resp.status_code == 200

    def test_response_shape(self, senate_client):
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        assert "dem_control_pct" in data
        assert "rep_control_pct" in data
        assert "dem_majority_pct" in data
        assert "median_dem_seats" in data
        assert "median_rep_seats" in data
        assert "seat_distribution" in data
        assert "n_simulations" in data
        assert "n_modeled_races" in data
        assert "n_safe_races" in data

    def test_probabilities_are_0_to_100(self, senate_client):
        """All probability values are in the 0-100 range."""
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        for key in ("dem_control_pct", "rep_control_pct", "dem_majority_pct"):
            assert 0.0 <= data[key] <= 100.0, f"{key} = {data[key]} out of range"

    def test_dem_and_rep_control_sum_to_100(self, senate_client):
        """dem_control_pct + rep_control_pct should sum to 100 (complementary events)."""
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        total = data["dem_control_pct"] + data["rep_control_pct"]
        assert abs(total - 100.0) < 1.0

    def test_median_seats_sum_to_100(self, senate_client):
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        assert data["median_dem_seats"] + data["median_rep_seats"] == 100

    def test_seat_distribution_is_list(self, senate_client):
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        assert isinstance(data["seat_distribution"], list)
        assert len(data["seat_distribution"]) > 0

    def test_seat_distribution_buckets_have_seats_and_probability(self, senate_client):
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        for bucket in data["seat_distribution"]:
            assert "seats" in bucket
            assert "probability" in bucket
            assert 0 <= bucket["seats"] <= 100
            assert 0.0 <= bucket["probability"] <= 1.0

    def test_n_simulations_matches_default(self, senate_client):
        """Default n_simulations should be 10,000."""
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        assert data["n_simulations"] == 10_000

    def test_custom_n_simulations(self, senate_client):
        """?n_simulations query param overrides the default."""
        data = senate_client.get("/api/v1/senate/chamber-probability?n_simulations=2000").json()
        assert data["n_simulations"] == 2000

    def test_n_simulations_below_minimum_rejected(self, senate_client):
        """n_simulations < 1000 is rejected with 422."""
        resp = senate_client.get("/api/v1/senate/chamber-probability?n_simulations=500")
        assert resp.status_code == 422

    def test_gop_favored_with_test_data(self, senate_client):
        """Test DB has TX and GA both leaning R; GOP should be favored."""
        data = senate_client.get("/api/v1/senate/chamber-probability").json()
        assert data["rep_control_pct"] > 50.0



# ── Overview Blend endpoint tests ──────────────────────────────────────────


class TestOverviewBlend:
    """Tests for POST /api/v1/forecast/overview/blend — GitHub issue #14."""

    def test_status_200(self, senate_client):
        resp = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        )
        assert resp.status_code == 200

    def test_response_shape(self, senate_client):
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        ).json()
        assert "dem_seats" in data
        assert "rep_seats" in data
        assert "races" in data
        assert isinstance(data["races"], list)

    def test_returns_all_33_races(self, senate_client):
        """The endpoint covers every 2026 Senate race, not just contested ones."""
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        ).json()
        assert len(data["races"]) == len(SENATE_2026_STATES)

    def test_race_summary_fields(self, senate_client):
        """Each race summary has slug, prediction, pred_std, and rating_label."""
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        ).json()
        for race in data["races"]:
            for field in ("slug", "prediction", "pred_std", "rating_label"):
                assert field in race, f"Missing field '{field}' in race {race}"

    def test_seat_totals_are_integers(self, senate_client):
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        ).json()
        assert isinstance(data["dem_seats"], int)
        assert isinstance(data["rep_seats"], int)

    def test_seat_totals_at_least_holdover_counts(self, senate_client):
        """Projected totals are always >= the holdover baseline (seats not up in 2026)."""
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        ).json()
        assert data["dem_seats"] >= _DEM_HOLDOVER_SEATS
        assert data["rep_seats"] >= _GOP_HOLDOVER_SEATS

    def test_different_weights_accepted(self, senate_client):
        """Any valid weight combination must return 200."""
        for weights in [
            {"model_prior": 100, "state_polls": 0, "national_polls": 0},
            {"model_prior": 0, "state_polls": 100, "national_polls": 0},
            {"model_prior": 33, "state_polls": 33, "national_polls": 34},
        ]:
            resp = senate_client.post(
                "/api/v1/forecast/overview/blend",
                json=weights,
            )
            assert resp.status_code == 200, f"Failed for weights {weights}: {resp.text}"

    def test_slugs_match_expected_format(self, senate_client):
        """All slugs are lowercase hyphenated senate race slugs."""
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        ).json()
        for race in data["races"]:
            slug = race["slug"]
            assert slug == slug.lower(), f"Slug not lowercase: {slug}"
            assert " " not in slug, f"Slug contains space: {slug}"
            assert "senate" in slug, f"Slug missing 'senate': {slug}"

    def test_rating_labels_are_valid(self, senate_client):
        """All rating_label values must be one of the 7 canonical ratings."""
        valid_ratings = {"safe_d", "likely_d", "lean_d", "tossup", "lean_r", "likely_r", "safe_r"}
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 60, "state_polls": 30, "national_polls": 10},
        ).json()
        for race in data["races"]:
            assert race["rating_label"] in valid_ratings, (
                f"Unknown rating '{race['rating_label']}' for slug '{race['slug']}'"
            )

    def test_model_prior_only_preserves_prediction(self, senate_client):
        """With model_prior=100 and state_polls=0, poll influence is zeroed out.

        TX has polls in the fixture.  Even with polls zeroed, the structural
        prior should still yield a valid float prediction.
        """
        data = senate_client.post(
            "/api/v1/forecast/overview/blend",
            json={"model_prior": 100, "state_polls": 0, "national_polls": 0},
        ).json()
        tx_race = next(r for r in data["races"] if "tx" in r["slug"])
        assert tx_race["prediction"] is not None
        assert isinstance(tx_race["prediction"], float)

    def test_not_500_on_edge_weights(self, senate_client):
        """No weight combination should produce a 5xx response."""
        for weights in [
            {"model_prior": 0, "state_polls": 0, "national_polls": 100},
            {"model_prior": 50, "state_polls": 50, "national_polls": 0},
        ]:
            resp = senate_client.post(
                "/api/v1/forecast/overview/blend",
                json=weights,
            )
            assert resp.status_code < 500, (
                f"Got {resp.status_code} for weights {weights}: {resp.text}"
            )
