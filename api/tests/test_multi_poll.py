"""Tests for the POST /forecast/polls multi-poll endpoint."""
from __future__ import annotations

import pytest


class TestMultiPollEndpoint:
    def test_missing_cycle_returns_404(self, client):
        """Cycle with no polls in DuckDB should return 404."""
        resp = client.post(
            "/api/v1/forecast/polls",
            json={"cycle": "9999", "state": "FL"},
        )
        assert resp.status_code == 404

    def test_returns_expected_shape(self, client):
        """Valid request should return MultiPollResponse shape."""
        # The test DB fixture inserts one poll: abc123, FL_Senate, FL, state, 0.45, cycle=2026
        resp = client.post(
            "/api/v1/forecast/polls",
            json={"cycle": "2026", "state": "FL", "race": "FL_Senate"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "counties" in data
        assert "polls_used" in data
        assert "date_range" in data
        assert "effective_n_total" in data
        assert data["polls_used"] == 1
        assert data["effective_n_total"] > 0
        assert len(data["counties"]) > 0

    def test_no_matching_polls_returns_404(self, client):
        """No polls matching filters should return 404."""
        # Filter for a state not in the test DB
        resp = client.post(
            "/api/v1/forecast/polls",
            json={"cycle": "2026", "state": "TX"},
        )
        assert resp.status_code == 404
