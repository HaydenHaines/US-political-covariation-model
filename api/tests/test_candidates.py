# api/tests/test_candidates.py
"""Tests for the sabermetrics candidate endpoints (badges, CTOV, race lookup)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.tests.conftest import _build_test_db, _noop_lifespan

# ── Test data fixtures ──────────────────────────────────────────────────────

_SAMPLE_BADGES = {
    "T000001": {
        "name": "Test Senator",
        "party": "D",
        "n_races": 3,
        "badges": ["Hispanic Appeal", "Turnout Monster"],
        "badge_scores": {
            "Hispanic Appeal": 0.045,
            "Black Community Strength": -0.012,
            "Senior Whisperer": 0.003,
            "Suburban Professional": 0.021,
            "Rural Populist": -0.008,
            "Faith Coalition": 0.015,
            "Turnout Monster": 0.038,
        },
        "cec": 0.72,
    },
    "T000002": {
        "name": "Test Governor",
        "party": "R",
        "n_races": 2,
        "badges": ["Rural Populist"],
        "badge_scores": {
            "Hispanic Appeal": -0.015,
            "Black Community Strength": -0.005,
            "Senior Whisperer": 0.010,
            "Suburban Professional": -0.020,
            "Rural Populist": 0.042,
            "Faith Coalition": 0.025,
            "Turnout Monster": -0.011,
        },
        "cec": 0.58,
    },
}

_SAMPLE_REGISTRY = {
    "persons": {
        "T000001": {
            "name": "Test Senator",
            "party": "D",
            "bioguide_id": "T000001",
            "races": [
                {"year": 2016, "state": "GA", "office": "Senate"},
                {"year": 2022, "state": "GA", "office": "Senate"},
            ],
        },
        "T000002": {
            "name": "Test Governor",
            "party": "R",
            "bioguide_id": "T000002",
            "races": [
                {"year": 2022, "state": "FL", "office": "Governor"},
            ],
        },
    }
}

_SAMPLE_CANDIDATES_2026 = {
    "_meta": {"description": "test"},
    "senate": {
        "2026 GA Senate": {
            "state": "Georgia",
            "incumbent": {"name": "Test Senator", "party": "D"},
            "status": "incumbent_running",
            "candidates": {
                "D": ["Test Senator"],
                "R": ["Challenger One"],
            },
        },
    },
    "governor": {},
}

# CTOV data: 2 rows (one per race-year for T000001)
_CTOV_COLS = ["person_id", "name", "party", "year", "state", "office",
              "actual_dem_share", "pred_dem_share", "mvd"] + \
             [f"ctov_type_{i}" for i in range(100)]


def _make_ctov_df() -> pd.DataFrame:
    """Create a small CTOV DataFrame for testing."""
    rows = []
    for year in (2016, 2022):
        row = {
            "person_id": "T000001",
            "name": "Test Senator",
            "party": "D",
            "year": year,
            "state": "GA",
            "office": "Senate",
            "actual_dem_share": 0.48,
            "pred_dem_share": 0.45,
            "mvd": 0.03,
        }
        # Set CTOV values: type 5 has highest, type 42 second highest
        for i in range(100):
            if i == 5:
                row[f"ctov_type_{i}"] = 0.08
            elif i == 42:
                row[f"ctov_type_{i}"] = -0.065
            elif i == 10:
                row[f"ctov_type_{i}"] = 0.04
            else:
                row[f"ctov_type_{i}"] = 0.001 * (i % 10 - 5)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def candidates_client():
    """TestClient with patched sabermetrics data."""
    from api.main import create_app

    test_db = _build_test_db()

    # Patch the module-level data in the candidates router
    with patch("api.routers.candidates._BADGES", _SAMPLE_BADGES), \
         patch("api.routers.candidates._REGISTRY", _SAMPLE_REGISTRY["persons"]), \
         patch("api.routers.candidates._CTOV", _make_ctov_df()), \
         patch("api.routers.candidates._CANDIDATES_2026", {
             "2026 GA Senate": _SAMPLE_CANDIDATES_2026["senate"]["2026 GA Senate"],
         }), \
         patch("api.routers.candidates._NAME_TO_BIOGUIDE", {
             "Test Senator": "T000001",
             "Test Governor": "T000002",
         }), \
         patch("api.routers.candidates._SUPER_TYPE_NAMES", None):

        test_app = create_app(lifespan_override=_noop_lifespan)
        test_app.state.db = test_db
        test_app.state.version_id = "test_v1"
        test_app.state.K = 3
        test_app.state.sigma = np.eye(3) * 0.01
        test_app.state.mu_prior = np.full(3, 0.42)
        test_app.state.state_weights = pd.DataFrame()
        test_app.state.county_weights = pd.DataFrame()
        test_app.state.contract_ok = True

        with TestClient(test_app, raise_server_exceptions=True) as c:
            yield c

        test_db.close()


# ── Tests: GET /candidates/{bioguide_id} ────────────────────────────────────


class TestGetCandidate:
    def test_valid_candidate(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bioguide_id"] == "T000001"
        assert data["name"] == "Test Senator"
        assert data["party"] == "D"
        assert data["n_races"] == 3
        assert data["cec"] == pytest.approx(0.72)

    def test_badges_present(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000001")
        data = resp.json()
        badge_names = [b["name"] for b in data["badges"]]
        assert "Hispanic Appeal" in badge_names
        assert "Turnout Monster" in badge_names
        assert len(data["badges"]) == 2

    def test_badge_scores_all_dimensions(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000001")
        data = resp.json()
        # badge_scores should include all dimensions, not just earned badges
        assert len(data["badge_scores"]) == 7
        assert "Rural Populist" in data["badge_scores"]

    def test_404_unknown_candidate(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/UNKNOWN999")
        assert resp.status_code == 404

    def test_second_candidate(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000002")
        assert resp.status_code == 200
        data = resp.json()
        assert data["party"] == "R"
        assert len(data["badges"]) == 1
        assert data["badges"][0]["name"] == "Rural Populist"


# ── Tests: GET /candidates/{bioguide_id}/ctov ───────────────────────────────


class TestGetCandidateCTOV:
    def test_valid_ctov(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000001/ctov")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bioguide_id"] == "T000001"
        assert data["name"] == "Test Senator"
        # Most recent year should be returned by default
        assert data["year"] == 2022

    def test_ctov_entries_sorted_by_absolute_value(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000001/ctov")
        data = resp.json()
        entries = data["entries"]
        assert len(entries) == 10
        # First entry should be the highest absolute CTOV
        assert entries[0]["type_id"] == 5
        assert entries[0]["ctov"] == pytest.approx(0.08)
        # Second should be type 42 (negative, but high absolute)
        assert entries[1]["type_id"] == 42
        assert entries[1]["ctov"] == pytest.approx(-0.065)

    def test_ctov_with_year_filter(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000001/ctov?year=2016")
        assert resp.status_code == 200
        data = resp.json()
        assert data["year"] == 2016

    def test_ctov_404_unknown_candidate(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/UNKNOWN999/ctov")
        assert resp.status_code == 404

    def test_ctov_404_wrong_year(self, candidates_client):
        resp = candidates_client.get("/api/v1/candidates/T000001/ctov?year=2000")
        assert resp.status_code == 404


# ── Tests: GET /races/{race_key}/candidates ─────────────────────────────────


class TestGetRaceCandidates:
    def test_valid_race(self, candidates_client):
        resp = candidates_client.get("/api/v1/races/2026 GA Senate/candidates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["race_key"] == "2026 GA Senate"
        # Only Test Senator should match (Challenger One not in registry)
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["bioguide_id"] == "T000001"

    def test_race_candidate_badges(self, candidates_client):
        resp = candidates_client.get("/api/v1/races/2026 GA Senate/candidates")
        data = resp.json()
        candidate = data["candidates"][0]
        badge_names = [b["name"] for b in candidate["badges"]]
        assert "Hispanic Appeal" in badge_names
        assert candidate["cec"] == pytest.approx(0.72)

    def test_404_unknown_race(self, candidates_client):
        resp = candidates_client.get("/api/v1/races/2026 ZZ Senate/candidates")
        assert resp.status_code == 404

    def test_dash_format_race_key(self, candidates_client):
        """Race keys with dashes should be converted to spaces."""
        resp = candidates_client.get("/api/v1/races/2026-GA-Senate/candidates")
        assert resp.status_code == 200

    def test_case_insensitive_race_key(self, candidates_client):
        resp = candidates_client.get("/api/v1/races/2026 ga senate/candidates")
        assert resp.status_code == 200
