"""Tests for the scrollytelling backend feature (GitHub issue #15).

Covers:
  - Zone assignment logic: correct zone for various rating/incumbent combos
  - zone_counts sums to 100 (34D holdover + 33 Class II + 33R holdover = 100)
  - /senate/overview now includes zone on each race and zone_counts
  - /senate/scrolly-context endpoint returns expected schema
  - structural_context arithmetic (gap = 51 - holdover - wins)
  - baseline_label formatting (R+X or D+X from pres_dem_share)
  - GenericBallotInfo now includes baseline_year and baseline_label
"""
from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routers.senate import (
    DEM_SAFE_SEATS,
    GOP_SAFE_SEATS,
    SENATE_2026_STATES,
    _CLASS_II_INCUMBENT,
    _DEM_CLASS_II_COUNT,
    _DEM_HOLDOVER_SEATS,
    _GOP_CLASS_II_COUNT,
    _compute_baseline_label,
    _rating_to_zone,
)
from api.tests.conftest import _noop_lifespan


# ── Unit tests: _rating_to_zone ─────────────────────────────────────────────


class TestRatingToZone:
    """Zone assignment must be driven by the rating, with incumbent as tiebreaker."""

    def test_safe_d_gives_safe_up_d(self):
        assert _rating_to_zone("safe_d", "D") == "safe_up_d"

    def test_likely_d_gives_safe_up_d(self):
        assert _rating_to_zone("likely_d", "D") == "safe_up_d"

    def test_lean_d_gives_contested_d(self):
        assert _rating_to_zone("lean_d", "D") == "contested_d"

    def test_tossup_gives_tossup_regardless_of_incumbent(self):
        assert _rating_to_zone("tossup", "D") == "tossup"
        assert _rating_to_zone("tossup", "R") == "tossup"

    def test_lean_r_gives_contested_r(self):
        assert _rating_to_zone("lean_r", "R") == "contested_r"

    def test_likely_r_gives_safe_up_r(self):
        assert _rating_to_zone("likely_r", "R") == "safe_up_r"

    def test_safe_r_gives_safe_up_r(self):
        assert _rating_to_zone("safe_r", "R") == "safe_up_r"

    def test_edge_case_lean_r_with_d_incumbent(self):
        """A D-held seat rated lean_r → contested_r, not contested_d.
        The rating side (lean_r) wins, not the incumbent party.
        """
        assert _rating_to_zone("lean_r", "D") == "contested_r"

    def test_edge_case_lean_d_with_r_incumbent(self):
        """An R-held seat rated lean_d → contested_d, not contested_r."""
        assert _rating_to_zone("lean_d", "R") == "contested_d"

    def test_edge_case_safe_r_with_d_incumbent(self):
        """A D-held seat rated safe_r → safe_up_r (a major upset scenario)."""
        assert _rating_to_zone("safe_r", "D") == "safe_up_r"

    def test_edge_case_safe_d_with_r_incumbent(self):
        """An R-held seat rated safe_d → safe_up_d."""
        assert _rating_to_zone("safe_d", "R") == "safe_up_d"


# ── Unit tests: _compute_baseline_label ─────────────────────────────────────


class TestComputeBaselineLabel:
    def test_republican_advantage(self):
        """pres_baseline < 0.5 → R+X label."""
        # 0.4841 → shift = -0.0159 → R+1.6
        label = _compute_baseline_label(0.4841)
        assert label.startswith("R+")

    def test_correct_magnitude_for_2024_baseline(self):
        """2024 presidential baseline 0.4841 → R+1.6."""
        label = _compute_baseline_label(0.4841)
        assert label == "R+1.6"

    def test_democrat_advantage(self):
        """pres_baseline > 0.5 → D+X label."""
        label = _compute_baseline_label(0.53)
        assert label.startswith("D+")

    def test_democrat_advantage_magnitude(self):
        label = _compute_baseline_label(0.53)
        assert label == "D+3.0"

    def test_exact_fifty_fifty_is_d_zero(self):
        """pres_baseline == 0.5 → shift == 0 → D+0.0."""
        label = _compute_baseline_label(0.5)
        assert label == "D+0.0"

    def test_r_plus_ten(self):
        label = _compute_baseline_label(0.40)
        assert label == "R+10.0"


# ── Integration: zone_counts sums to 100 ────────────────────────────────────


class TestZoneCountsTotal:
    """zone_counts across all 7 buckets must always sum to 100."""

    def test_zone_counts_sum_to_100_with_fallback(self):
        """With no predictions, all races fall back to safe_d/safe_r.
        Zone counts across 7 buckets must still total 100.
        """
        # 33 Class II races + 67 holdover = 100 seats total
        # _DEM_HOLDOVER_SEATS + _GOP_CLASS_II_COUNT (safe_up_r) +
        #   _DEM_CLASS_II_COUNT (safe_up_d) + (GOP_SAFE_SEATS - _GOP_CLASS_II_COUNT)
        gop_holdover = GOP_SAFE_SEATS - _GOP_CLASS_II_COUNT
        expected_total = _DEM_HOLDOVER_SEATS + _DEM_CLASS_II_COUNT + _GOP_CLASS_II_COUNT + gop_holdover
        assert expected_total == 100, (
            f"Sanity check: holdover+class2 seats = {expected_total}, should be 100"
        )

    def test_zone_counts_from_overview_endpoint_sum_to_100(self):
        """zone_counts from the live endpoint always sums to 100."""
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/overview").json()
        con.close()

        zc = data["zone_counts"]
        total = sum(zc.values())
        assert total == 100, f"zone_counts sum = {total}, expected 100. Counts: {zc}"

    def test_zone_counts_from_scrolly_context_sum_to_100(self):
        """zone_counts in scrolly-context also sums to 100."""
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        zc = data["zone_counts"]
        total = sum(zc.values())
        assert total == 100, f"zone_counts sum = {total}, expected 100. Counts: {zc}"


# ── Integration: /senate/overview zone field ─────────────────────────────────


class TestOverviewZoneField:
    """Every race in /senate/overview must have a valid zone field."""

    _VALID_ZONES = frozenset(
        {"safe_up_d", "contested_d", "tossup", "contested_r", "safe_up_r"}
    )

    def test_every_race_has_zone(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/overview").json()
        con.close()

        for race in data["races"]:
            assert "zone" in race, f"Missing 'zone' on race {race['state']}"
            assert race["zone"] in self._VALID_ZONES, (
                f"Unknown zone '{race['zone']}' for {race['state']}"
            )

    def test_overview_has_zone_counts(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/overview").json()
        con.close()

        assert "zone_counts" in data
        zc = data["zone_counts"]
        expected_keys = {"not_up_d", "safe_up_d", "contested_d", "tossup", "contested_r", "safe_up_r", "not_up_r"}
        assert set(zc.keys()) == expected_keys

    def test_zone_counts_are_non_negative_integers(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/overview").json()
        con.close()

        for key, val in data["zone_counts"].items():
            assert isinstance(val, int), f"zone_counts[{key}] is not int: {val}"
            assert val >= 0, f"zone_counts[{key}] is negative: {val}"

    def test_tossup_zone_with_tossup_prediction(self):
        """A predicted margin near zero should produce zone='tossup'."""
        con = _build_db_with_tossup_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/overview").json()
        con.close()

        # TX is predicted at exactly 0.50 → margin=0.0 → tossup
        tx_race = next(r for r in data["races"] if r["state"] == "TX")
        assert tx_race["zone"] == "tossup"


# ── Integration: /senate/scrolly-context schema ──────────────────────────────


class TestScrollyContextEndpoint:
    """Endpoint returns the expected shape and valid values."""

    def test_returns_200(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            resp = c.get("/api/v1/senate/scrolly-context")
        con.close()
        assert resp.status_code == 200

    def test_top_level_keys_present(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        assert "zone_counts" in data
        assert "not_up_d_states" in data
        assert "not_up_r_states" in data
        assert "structural_context" in data
        assert "competitive_races" in data

    def test_structural_context_keys_present(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        sc = data["structural_context"]
        expected_keys = {
            "baseline_year", "baseline_label", "baseline_dem_two_party",
            "dem_wins_at_baseline", "dem_holdover_seats", "total_dem_projected",
            "seats_needed_for_majority", "structural_gap",
        }
        assert expected_keys <= set(sc.keys()), (
            f"Missing keys: {expected_keys - set(sc.keys())}"
        )

    def test_baseline_year_is_2018(self):
        """Structural context references 2018 midterm as the baseline environment."""
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()
        assert data["structural_context"]["baseline_year"] == 2018

    def test_baseline_label_is_d_plus(self):
        """2018 national Dem share 0.534 > 0.5 → baseline_label starts with D+."""
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()
        assert data["structural_context"]["baseline_label"].startswith("D+")

    def test_not_up_states_are_sorted(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        d_states = data["not_up_d_states"]
        r_states = data["not_up_r_states"]
        assert d_states == sorted(d_states), "not_up_d_states not sorted"
        assert r_states == sorted(r_states), "not_up_r_states not sorted"

    def test_not_up_states_exclude_2026_states(self):
        """States in not_up_d/r_states must not be in SENATE_2026_STATES."""
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        for st in data["not_up_d_states"]:
            assert st not in SENATE_2026_STATES, f"{st} is in 2026 states but also in not_up_d_states"
        for st in data["not_up_r_states"]:
            assert st not in SENATE_2026_STATES, f"{st} is in 2026 states but also in not_up_r_states"

    def test_competitive_races_only_lean_and_tossup(self):
        """competitive_races must only include lean_d, tossup, lean_r ratings."""
        con = _build_db_with_varied_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        valid = {"lean_d", "tossup", "lean_r"}
        for race in data["competitive_races"]:
            assert race["rating"] in valid, (
                f"Non-competitive rating '{race['rating']}' in competitive_races"
            )

    def test_works_with_no_version_id(self):
        """When version_id is None (no DB), the endpoint returns 200 with fallbacks."""
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = None  # no predictions loaded
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            resp = c.get("/api/v1/senate/scrolly-context")
        con.close()
        assert resp.status_code == 200
        data = resp.json()
        assert sum(data["zone_counts"].values()) == 100


# ── Integration: structural_context arithmetic ───────────────────────────────


class TestStructuralContextArithmetic:
    """structural_context values must be internally consistent."""

    def test_gap_equals_51_minus_total_projected(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        sc = data["structural_context"]
        expected_gap = sc["seats_needed_for_majority"] - sc["total_dem_projected"]
        assert sc["structural_gap"] == expected_gap, (
            f"structural_gap {sc['structural_gap']} != {sc['seats_needed_for_majority']} - {sc['total_dem_projected']}"
        )

    def test_total_projected_equals_holdover_plus_wins(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        sc = data["structural_context"]
        expected_total = sc["dem_holdover_seats"] + sc["dem_wins_at_baseline"]
        assert sc["total_dem_projected"] == expected_total

    def test_dem_holdover_matches_constant(self):
        """dem_holdover_seats must match the module-level constant."""
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        assert data["structural_context"]["dem_holdover_seats"] == _DEM_HOLDOVER_SEATS

    def test_seats_needed_for_majority_is_51(self):
        con = _build_minimal_db_with_predictions()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = con
        test_app.state.version_id = "test_v1"
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/senate/scrolly-context").json()
        con.close()

        assert data["structural_context"]["seats_needed_for_majority"] == 51


# ── Integration: GenericBallotInfo baseline fields ───────────────────────────


class TestGenericBallotBaselineFields:
    """The /forecast/generic-ballot endpoint now includes baseline_year and baseline_label."""

    def test_baseline_year_present(self):
        """Response must include baseline_year."""
        from api.tests.conftest import _build_test_db, _build_test_state, TEST_K, TEST_VERSION

        test_db = _build_test_db()
        state = _build_test_state()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = test_db
        test_app.state.version_id = TEST_VERSION
        test_app.state.K = TEST_K
        test_app.state.sigma = state["sigma"]
        test_app.state.mu_prior = state["mu_prior"]
        test_app.state.state_weights = state["state_weights"]
        test_app.state.county_weights = state["county_weights"]
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/forecast/generic-ballot").json()
        test_db.close()

        assert "baseline_year" in data
        assert data["baseline_year"] == 2024

    def test_baseline_label_present_and_r_plus(self):
        """baseline_label must be present and reflect R+ (2024 Dem share < 0.5)."""
        from api.tests.conftest import _build_test_db, _build_test_state, TEST_K, TEST_VERSION

        test_db = _build_test_db()
        state = _build_test_state()
        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = test_db
        test_app.state.version_id = TEST_VERSION
        test_app.state.K = TEST_K
        test_app.state.sigma = state["sigma"]
        test_app.state.mu_prior = state["mu_prior"]
        test_app.state.state_weights = state["state_weights"]
        test_app.state.county_weights = state["county_weights"]
        test_app.state.contract_ok = True
        with TestClient(test_app, raise_server_exceptions=True) as c:
            data = c.get("/api/v1/forecast/generic-ballot").json()
        test_db.close()

        assert "baseline_label" in data
        assert data["baseline_label"].startswith("R+"), (
            f"Expected R+ label, got: {data['baseline_label']}"
        )


# ── DB fixtures for tests ────────────────────────────────────────────────────


def _build_minimal_db_with_predictions() -> duckdb.DuckDBPyConnection:
    """Minimal in-memory DuckDB with TX and GA predictions (same as existing senate tests)."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE counties (
            county_fips VARCHAR PRIMARY KEY, state_abbr VARCHAR,
            state_fips VARCHAR, county_name VARCHAR, total_votes_2024 INTEGER
        )
    """)
    counties = [
        ("48001", "TX", "48", "Anderson County, TX", 100000),
        ("48003", "TX", "48", "Andrews County, TX",   15000),
        ("13001", "GA", "13", "Appling County, GA",   80000),
        ("13003", "GA", "13", "Atkinson County, GA",   8000),
    ]
    for row in counties:
        con.execute("INSERT INTO counties VALUES (?, ?, ?, ?, ?)", list(row))

    con.execute("""
        CREATE TABLE model_versions (
            version_id VARCHAR PRIMARY KEY, role VARCHAR, k INTEGER, j INTEGER,
            shift_type VARCHAR, vote_share_type VARCHAR, n_training_dims INTEGER,
            n_holdout_dims INTEGER, holdout_r VARCHAR, geography VARCHAR,
            description VARCHAR, created_at TIMESTAMP
        )
    """)
    con.execute(
        "INSERT INTO model_versions VALUES ('test_v1', 'current', 3, 7, "
        "'logodds', 'total', 30, 3, '0.90', 'test', 'test', '2026-01-01')"
    )

    con.execute("""
        CREATE TABLE predictions (
            county_fips VARCHAR NOT NULL, race VARCHAR NOT NULL,
            version_id VARCHAR NOT NULL, pred_dem_share DOUBLE,
            pred_std DOUBLE, pred_lo90 DOUBLE, pred_hi90 DOUBLE,
            state_pred DOUBLE, poll_avg DOUBLE,
            PRIMARY KEY (county_fips, race, version_id)
        )
    """)
    # TX: strong R (margin = 0.45 - 0.5 = -0.05 → lean_r)
    for fips, share in [("48001", 0.45), ("48003", 0.44)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 TX Senate', 'test_v1', ?, 0.03, ?, ?, 0.45, 0.45)",
            [fips, share, share - 0.05, share + 0.05],
        )
    # GA: lean D (margin = 0.55 - 0.5 = +0.05)
    for fips, share in [("13001", 0.55), ("13003", 0.54)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 GA Senate', 'test_v1', ?, 0.03, ?, ?, 0.55, 0.55)",
            [fips, share, share - 0.05, share + 0.05],
        )

    con.execute("""
        CREATE TABLE polls (
            poll_id VARCHAR PRIMARY KEY, race VARCHAR, geography VARCHAR,
            geo_level VARCHAR, dem_share FLOAT, n_sample INTEGER,
            date VARCHAR, pollster VARCHAR, notes VARCHAR, cycle VARCHAR
        )
    """)

    return con


def _build_db_with_tossup_predictions() -> duckdb.DuckDBPyConnection:
    """DB with TX predicted at exactly 0.50 → tossup zone."""
    con = _build_minimal_db_with_predictions()
    # Override TX predictions to 0.50 (dead tossup)
    con.execute("DELETE FROM predictions WHERE race = '2026 TX Senate'")
    for fips, share in [("48001", 0.50), ("48003", 0.50)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 TX Senate', 'test_v1', ?, 0.03, ?, ?, 0.50, 0.50)",
            [fips, share, share - 0.05, share + 0.05],
        )
    return con


def _build_db_with_varied_predictions() -> duckdb.DuckDBPyConnection:
    """DB with one lean_d and one tossup race — for competitive_races filtering test."""
    con = _build_minimal_db_with_predictions()
    # TX: tossup (0.502 → margin=0.002 → tossup)
    con.execute("DELETE FROM predictions WHERE race = '2026 TX Senate'")
    for fips, share in [("48001", 0.502), ("48003", 0.501)]:
        con.execute(
            "INSERT INTO predictions VALUES (?, '2026 TX Senate', 'test_v1', ?, 0.03, ?, ?, 0.502, 0.50)",
            [fips, share, share - 0.05, share + 0.05],
        )
    # GA: lean D (0.55 → margin=0.05 → lean_d) — already set up
    return con
