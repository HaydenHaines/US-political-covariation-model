"""Tests for the POST /forecast/polls multi-poll endpoint."""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestMultiPollEndpoint:
    def test_missing_cycle_returns_404(self, client):
        """Missing poll CSV should return 404."""
        resp = client.post(
            "/api/v1/forecast/polls",
            json={"cycle": "9999", "state": "FL"},
        )
        assert resp.status_code == 404

    def test_returns_expected_shape(self, client, tmp_path):
        """Valid request should return MultiPollResponse shape."""
        # Write a temporary poll CSV
        csv_path = tmp_path / "polls_2020.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"])
            w.writerow(["2020 FL President", "FL", "state", "0.48", "800", "2020-10-01", "TestPoll", "grade=2.5"])
            w.writerow(["2020 FL President", "FL", "state", "0.50", "600", "2020-10-15", "TestPoll2", "grade=1.5"])

        with patch(
            "src.propagation.poll_weighting.PROJECT_ROOT",
            tmp_path,
        ):
            # Create the expected directory structure
            polls_dir = tmp_path / "data" / "polls"
            polls_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(csv_path, polls_dir / "polls_2020.csv")

            resp = client.post(
                "/api/v1/forecast/polls",
                json={"cycle": "2020", "state": "FL", "race": "President"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "counties" in data
        assert "polls_used" in data
        assert "date_range" in data
        assert "effective_n_total" in data
        assert data["polls_used"] == 2
        assert data["effective_n_total"] > 0
        assert "2020-10-01" in data["date_range"]
        assert len(data["counties"]) > 0

    def test_no_matching_polls_returns_404(self, client, tmp_path):
        """No polls matching filters should return 404."""
        polls_dir = tmp_path / "data" / "polls"
        polls_dir.mkdir(parents=True, exist_ok=True)
        csv_path = polls_dir / "polls_2020.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"])
            w.writerow(["2020 GA President", "GA", "state", "0.50", "800", "2020-10-01", "TestPoll", ""])

        with patch(
            "src.propagation.poll_weighting.PROJECT_ROOT",
            tmp_path,
        ):
            resp = client.post(
                "/api/v1/forecast/polls",
                json={"cycle": "2020", "state": "FL"},
            )

        assert resp.status_code == 404
