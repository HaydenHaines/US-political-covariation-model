"""Tests for the forecast router — GET /forecast and POST /forecast/poll."""
from __future__ import annotations

import pytest


class TestGetForecast:
    def test_returns_all_predictions(self, client):
        resp = client.get("/api/v1/forecast")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5  # 5 test counties × 1 race
        assert all(r["race"] == "FL_Senate" for r in data)

    def test_filter_by_race(self, client):
        resp = client.get("/api/v1/forecast?race=FL_Senate")
        assert resp.status_code == 200
        assert len(resp.json()) == 5

    def test_filter_by_nonexistent_race(self, client):
        resp = client.get("/api/v1/forecast?race=FAKE_RACE")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filter_by_state(self, client):
        resp = client.get("/api/v1/forecast?state=FL")
        assert resp.status_code == 200
        data = resp.json()
        assert all(r["state_abbr"] == "FL" for r in data)
        assert len(data) == 2  # 12001, 12003

    def test_filter_by_race_and_state(self, client):
        resp = client.get("/api/v1/forecast?race=FL_Senate&state=GA")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2  # 13001, 13003
        assert all(r["state_abbr"] == "GA" for r in data)

    def test_response_fields(self, client):
        resp = client.get("/api/v1/forecast?race=FL_Senate&state=FL")
        data = resp.json()
        row = data[0]
        assert "county_fips" in row
        assert "county_name" in row
        assert "state_abbr" in row
        assert "race" in row
        assert "pred_dem_share" in row
        assert "pred_std" in row
        assert "pred_lo90" in row
        assert "pred_hi90" in row
        assert row["pred_dem_share"] == pytest.approx(0.42, abs=0.01)


class TestPostForecastPoll:
    def test_poll_update_returns_predictions(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5  # All counties get updated predictions
        assert all(r["race"] == "FL_Senate" for r in data)

    def test_poll_update_shifts_predictions(self, client):
        # Prior is 0.42 for all communities. Poll at 0.48 should shift up.
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.55, "n": 1000},
        )
        data = resp.json()
        for row in data:
            assert row["pred_dem_share"] > 0.42  # Should shift upward from prior

    def test_poll_update_has_confidence_interval(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.45, "n": 600},
        )
        data = resp.json()
        for row in data:
            assert row["pred_lo90"] < row["pred_dem_share"]
            assert row["pred_hi90"] > row["pred_dem_share"]

    def test_poll_update_unknown_state(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "XX", "race": "XX_Senate", "dem_share": 0.5, "n": 600},
        )
        assert resp.status_code == 404

    def test_poll_update_state_pred_field(self, client):
        resp = client.post(
            "/api/v1/forecast/poll",
            json={"state": "FL", "race": "FL_Senate", "dem_share": 0.50, "n": 600},
        )
        data = resp.json()
        assert data[0]["state_pred"] is not None
        assert data[0]["poll_avg"] == pytest.approx(0.50)


def test_get_polls_queries_duckdb(client):
    """GET /polls returns rows from DuckDB, not CSV."""
    resp = client.get("/api/v1/polls?cycle=2026&state=FL")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["geography"] == "FL"
    assert data[0]["dem_share"] == pytest.approx(0.45)
