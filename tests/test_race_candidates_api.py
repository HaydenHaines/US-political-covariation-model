"""Tests for the GET /races/{race_key}/candidates endpoint.

Verifies that badge_scores is included in the response and has the correct
structure.  Uses monkeypatching to inject controlled test data without
requiring real badge data files on disk.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_lifespan(app):
    """Test lifespan: skip DB/parquet loading so data/wethervane.duckdb isn't required.

    Mirrors the pattern from api/tests/conftest.py (introduced by the #247 CI fix).
    """
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal badge data matching what candidate_badges.json actually contains.
_MOCK_BADGES = {
    "A000001": {
        "name": "Jane Doe",
        "party": "D",
        "n_races": 3,
        "badges": ["Senior Whisperer", "Faith Coalition"],
        "badge_scores": {
            "Senior Whisperer": 0.495,
            "Black Community Strength": 0.036,
            "Hispanic Appeal": -0.041,
            "Rural Populist": 0.030,
            "Faith Coalition": 0.045,
            "Turnout Monster": 0.033,
        },
        "badge_details": {},
        "cec": 0.72,
    },
    "B000002": {
        "name": "John Smith",
        "party": "R",
        "n_races": 1,
        "badges": [],
        "badge_scores": {},
        "badge_details": {},
        "cec": 0.0,
    },
}

# Minimal candidates_2026 data for one race containing our two test candidates.
_MOCK_CANDIDATES_2026 = {
    "2026 TEST Senate": {
        "candidates": {
            "D": ["Jane Doe"],
            "R": ["John Smith"],
        },
        "incumbent": {},
    }
}

# Minimal name→bioguide crosswalk (mirrors _NAME_TO_BIOGUIDE in the candidates router).
_MOCK_NAME_TO_BIOGUIDE = {
    "Jane Doe": "A000001",
    "John Smith": "B000002",
}


@pytest.fixture(scope="module")
def test_client():
    """TestClient with mocked badge/candidate data."""
    import api.routers.candidates as candidates_module
    from api.main import create_app

    app = create_app(lifespan_override=_noop_lifespan)

    with (
        patch.object(candidates_module, "_BADGES", _MOCK_BADGES),
        patch.object(candidates_module, "_CANDIDATES_2026", _MOCK_CANDIDATES_2026),
        patch.object(candidates_module, "_NAME_TO_BIOGUIDE", _MOCK_NAME_TO_BIOGUIDE),
    ):
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRaceCandidatesResponseShape:
    """Validate the structure of the race-candidates API response."""

    def test_returns_200_for_known_race(self, test_client):
        resp = test_client.get("/api/v1/races/2026 TEST Senate/candidates")
        assert resp.status_code == 200

    def test_response_has_race_key(self, test_client):
        resp = test_client.get("/api/v1/races/2026 TEST Senate/candidates")
        body = resp.json()
        assert "race_key" in body
        assert body["race_key"] == "2026 TEST Senate"

    def test_response_has_candidates_list(self, test_client):
        resp = test_client.get("/api/v1/races/2026 TEST Senate/candidates")
        body = resp.json()
        assert "candidates" in body
        assert isinstance(body["candidates"], list)

    def test_candidates_include_badge_scores_field(self, test_client):
        """Every candidate object in the response must have a badge_scores field."""
        resp = test_client.get("/api/v1/races/2026 TEST Senate/candidates")
        candidates = resp.json()["candidates"]
        assert len(candidates) > 0, "Expected at least one candidate with badge data"
        for candidate in candidates:
            assert "badge_scores" in candidate, (
                f"badge_scores missing from candidate {candidate.get('name')}"
            )

    def test_badge_scores_is_dict_of_floats(self, test_client):
        """badge_scores must be a dict mapping dimension names to float scores."""
        resp = test_client.get("/api/v1/races/2026 TEST Senate/candidates")
        for candidate in resp.json()["candidates"]:
            scores = candidate["badge_scores"]
            assert isinstance(scores, dict), (
                f"badge_scores should be a dict, got {type(scores)}"
            )
            for key, val in scores.items():
                assert isinstance(key, str), f"badge_scores key should be str, got {type(key)}"
                assert isinstance(val, (int, float)), (
                    f"badge_scores value for '{key}' should be numeric, got {type(val)}"
                )

    def test_candidate_with_scores_has_correct_values(self, test_client):
        """Jane Doe's badge_scores match the mock data."""
        resp = test_client.get("/api/v1/races/2026 TEST Senate/candidates")
        candidates = resp.json()["candidates"]
        jane = next((c for c in candidates if c["name"] == "Jane Doe"), None)
        assert jane is not None, "Jane Doe not found in response"
        scores = jane["badge_scores"]
        assert abs(scores["Senior Whisperer"] - 0.495) < 1e-6
        assert abs(scores["Hispanic Appeal"] - (-0.041)) < 1e-6

    def test_candidate_with_no_scores_returns_empty_dict(self, test_client):
        """Candidates with no badge_scores return an empty dict, not null."""
        resp = test_client.get("/api/v1/races/2026 TEST Senate/candidates")
        candidates = resp.json()["candidates"]
        john = next((c for c in candidates if c["name"] == "John Smith"), None)
        assert john is not None, "John Smith not found in response"
        assert john["badge_scores"] == {}

    def test_returns_404_for_unknown_race(self, test_client):
        resp = test_client.get("/api/v1/races/2026 ZZ Senate/candidates")
        assert resp.status_code == 404


class TestRaceCandidatesSlugNormalization:
    """Race key normalization: dashes should be treated as spaces."""

    def test_dash_separated_key_resolves(self, test_client):
        """'2026-TEST-Senate' should resolve the same as '2026 TEST Senate'."""
        resp = test_client.get("/api/v1/races/2026-TEST-Senate/candidates")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["candidates"]) > 0
