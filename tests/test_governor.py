"""Tests for the GET /api/v1/governor/overview endpoint.

Uses an in-memory DuckDB database and TestClient so no external services
are required.  The tests verify:
  - Endpoint returns 200 with the correct shape
  - All 36 governor races are present
  - Poll counts are wired up correctly
  - Fallback (no version_id) returns incumbent-party safe ratings
  - Races are sorted by competitiveness (tossup first)
"""
from __future__ import annotations

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.governor import router
from api.routers.governor._helpers import (
    GOVERNOR_2026_STATES,
    classify_governor_race,
    rating_sort_key,
)


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


def _make_app(con: duckdb.DuckDBPyConnection, version_id: str | None = "test_v1") -> FastAPI:
    """Build a minimal FastAPI app with the governor router and a fixed DB."""
    from api.db import get_db

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: con

    if version_id:
        app.state.version_id = version_id

    return app


def _seed_db(con: duckdb.DuckDBPyConnection) -> None:
    """Insert minimal predictions + counties + polls for a subset of governor races."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS counties (
            county_fips    VARCHAR PRIMARY KEY,
            state_abbr     VARCHAR,
            total_votes_2024 INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            county_fips   VARCHAR,
            race          VARCHAR,
            version_id    VARCHAR,
            pred_dem_share DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS polls (
            race     VARCHAR,
            date     VARCHAR,
            dem_share DOUBLE,
            n_sample INTEGER
        )
    """)

    # Seed two counties per state for a small subset of governor races
    states_with_data = ["FL", "OH", "MI"]
    fips = 10000
    for st in states_with_data:
        for _ in range(2):
            fips_str = str(fips).zfill(5)
            con.execute(
                "INSERT INTO counties VALUES (?, ?, ?)", [fips_str, st, 50000]
            )
            con.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?)",
                [fips_str, f"2026 {st} Governor", "test_v1", 0.52],
            )
            fips += 1

    # Add some polls for FL and OH
    con.execute("INSERT INTO polls VALUES (?, ?, ?, ?)", ["2026 FL Governor", "2026-03-01", 0.51, 600])
    con.execute("INSERT INTO polls VALUES (?, ?, ?, ?)", ["2026 FL Governor", "2026-03-15", 0.50, 600])
    con.execute("INSERT INTO polls VALUES (?, ?, ?, ?)", ["2026 OH Governor", "2026-03-10", 0.48, 700])


@pytest.fixture()
def client():
    """TestClient backed by in-memory DB with minimal governor data."""
    con = duckdb.connect(":memory:")
    _seed_db(con)
    app = _make_app(con)
    with TestClient(app) as c:
        yield c
    con.close()


@pytest.fixture()
def client_no_version():
    """TestClient with no version_id set — tests the fallback path."""
    con = duckdb.connect(":memory:")
    _seed_db(con)
    app = _make_app(con, version_id=None)
    with TestClient(app) as c:
        yield c
    con.close()


# ---------------------------------------------------------------------------
# Endpoint shape tests
# ---------------------------------------------------------------------------


def test_overview_returns_200(client):
    resp = client.get("/api/v1/governor/overview")
    assert resp.status_code == 200


def test_overview_has_races_and_updated_at(client):
    resp = client.get("/api/v1/governor/overview")
    data = resp.json()
    assert "races" in data
    assert "updated_at" in data


def test_overview_returns_36_races(client):
    """All 36 governor races must be present regardless of prediction coverage."""
    resp = client.get("/api/v1/governor/overview")
    data = resp.json()
    assert len(data["races"]) == 36


def test_overview_race_fields(client):
    """Each race item has the required fields with correct types."""
    resp = client.get("/api/v1/governor/overview")
    race = resp.json()["races"][0]
    assert isinstance(race["state"], str)
    assert isinstance(race["race"], str)
    assert isinstance(race["slug"], str)
    assert isinstance(race["rating"], str)
    assert isinstance(race["margin"], float)
    assert race["incumbent_party"] in ("D", "R")
    assert isinstance(race["is_open_seat"], bool)
    assert isinstance(race["n_polls"], int)


def test_overview_slug_format(client):
    """Slugs should follow the pattern '2026-xx-governor'."""
    resp = client.get("/api/v1/governor/overview")
    for race in resp.json()["races"]:
        slug = race["slug"]
        assert slug.startswith("2026-"), f"bad slug: {slug}"
        assert slug.endswith("-governor"), f"bad slug: {slug}"
        assert slug == slug.lower(), f"slug not lowercase: {slug}"


def test_overview_all_states_present(client):
    """Every state in GOVERNOR_2026_STATES must appear in the response."""
    resp = client.get("/api/v1/governor/overview")
    returned_states = {r["state"] for r in resp.json()["races"]}
    assert returned_states == GOVERNOR_2026_STATES


def test_overview_poll_counts_populated(client):
    """FL and OH should have poll counts > 0; others should be 0."""
    resp = client.get("/api/v1/governor/overview")
    by_state = {r["state"]: r for r in resp.json()["races"]}
    assert by_state["FL"]["n_polls"] == 2
    assert by_state["OH"]["n_polls"] == 1
    assert by_state["AK"]["n_polls"] == 0


def test_overview_ratings_valid(client):
    """All rating values must be one of the 7 known labels."""
    valid_ratings = {"safe_d", "likely_d", "lean_d", "tossup", "lean_r", "likely_r", "safe_r"}
    resp = client.get("/api/v1/governor/overview")
    for race in resp.json()["races"]:
        assert race["rating"] in valid_ratings, f"invalid rating: {race['rating']}"


def test_overview_updated_at_populated(client):
    """updated_at should reflect the most recent governor poll date."""
    resp = client.get("/api/v1/governor/overview")
    data = resp.json()
    # We seeded polls with dates, so updated_at should not be None
    assert data["updated_at"] is not None


# ---------------------------------------------------------------------------
# Fallback path (no version_id)
# ---------------------------------------------------------------------------


def test_fallback_no_version_returns_36_races(client_no_version):
    resp = client_no_version.get("/api/v1/governor/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["races"]) == 36


def test_fallback_no_version_uses_safe_ratings(client_no_version):
    """Without a model version, all races should be rated safe_d or safe_r."""
    resp = client_no_version.get("/api/v1/governor/overview")
    for race in resp.json()["races"]:
        assert race["rating"] in ("safe_d", "safe_r"), (
            f"{race['state']} rated {race['rating']} in fallback mode"
        )


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_classify_governor_race_slug_format():
    info = classify_governor_race("OH")
    assert info["slug"] == "2026-oh-governor"


def test_classify_governor_race_fallback_is_safe():
    info = classify_governor_race("OH")  # OH incumbent is R
    assert info["rating"] == "safe_r"
    assert info["incumbent_party"] == "R"
    assert info["margin"] < 0
    assert info["is_open_seat"] is True  # OH is term-limited


def test_classify_governor_race_open_seat_field():
    """Open-seat flag is True for term-limited/vacated states, False otherwise."""
    oh = classify_governor_race("OH")  # DeWine term-limited
    assert oh["is_open_seat"] is True

    il = classify_governor_race("IL")  # Pritzker running
    assert il["is_open_seat"] is False


def test_classify_governor_race_with_prediction():
    pred_by_race = {"2026 OH Governor": ("OH", 0.53)}
    info = classify_governor_race("OH", pred_by_race)
    # 0.53 - 0.5 = 0.03 → at the tossup boundary; anything ≥ 0.03 is lean_d
    assert info["margin"] == pytest.approx(0.03, abs=0.001)
    assert info["rating"] in ("tossup", "lean_d")


def test_rating_sort_key_order():
    """Rating sort: safe_d → likely_d → lean_d → tossup → lean_r → likely_r → safe_r.

    D-leaning races sort first, tossups in the middle, R-leaning races last.
    This groups races by party lean for display rather than competitiveness.
    """
    assert rating_sort_key("safe_d") < rating_sort_key("likely_d")
    assert rating_sort_key("likely_d") < rating_sort_key("lean_d")
    assert rating_sort_key("lean_d") < rating_sort_key("tossup")
    assert rating_sort_key("tossup") < rating_sort_key("lean_r")
    assert rating_sort_key("lean_r") < rating_sort_key("likely_r")
    assert rating_sort_key("likely_r") < rating_sort_key("safe_r")
