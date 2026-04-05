# api/tests/test_historical.py
"""Tests for the historical presidential election endpoint.

The endpoint reads parquet files from data/assembled/ — in tests we patch
the data directory to point at a temp directory with synthetic parquet files.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.routers import historical as hist_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_pres_parquet(tmp_dir: Path, year: int) -> None:
    """Write a minimal synthetic presidential parquet file for the given year."""
    assembled = tmp_dir / "assembled"
    assembled.mkdir(parents=True, exist_ok=True)
    data = {
        "county_fips": ["12001", "12003", "13001", "00000"],  # 00000 is the synthetic aggregate row
        f"pres_dem_share_{year}": [0.55, 0.38, 0.61, 0.50],
        f"pres_total_{year}": [100000, 15000, 80000, None],
    }
    df = pd.DataFrame(data)
    df.to_parquet(assembled / f"medsl_county_presidential_{year}.parquet", index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHistoricalPresidential:
    """Tests for GET /api/v1/historical/presidential/{year}."""

    @pytest.fixture(autouse=True)
    def patch_data_dir(self, tmp_path):
        """Point the historical router at a temp directory with synthetic data."""
        _write_synthetic_pres_parquet(tmp_path, 2020)
        _write_synthetic_pres_parquet(tmp_path, 2016)
        _write_synthetic_pres_parquet(tmp_path, 2012)

        # Patch the module-level _ASSEMBLED path used by _load_presidential_year
        with patch.object(hist_module, "_ASSEMBLED", tmp_path / "assembled"):
            yield

    def test_valid_year_returns_200(self, client):
        resp = client.get("/api/v1/historical/presidential/2020")
        assert resp.status_code == 200

    def test_response_structure(self, client):
        resp = client.get("/api/v1/historical/presidential/2020")
        data = resp.json()
        assert data["year"] == 2020
        assert "counties" in data
        assert isinstance(data["counties"], list)

    def test_excludes_synthetic_aggregate_row(self, client):
        """The '00000' synthetic aggregate county should be excluded."""
        resp = client.get("/api/v1/historical/presidential/2020")
        fips_list = [c["county_fips"] for c in resp.json()["counties"]]
        assert "00000" not in fips_list

    def test_county_row_fields(self, client):
        resp = client.get("/api/v1/historical/presidential/2020")
        counties = resp.json()["counties"]
        row = next(c for c in counties if c["county_fips"] == "12001")
        assert abs(row["dem_share"] - 0.55) < 0.001
        assert row["total_votes"] == 100000

    def test_null_total_votes_allowed(self, client):
        """Rows with null total_votes (e.g. the 13001 entry) are still returned."""
        resp = client.get("/api/v1/historical/presidential/2020")
        counties = resp.json()["counties"]
        # 13001 has null total_votes in our synthetic data
        row = next(c for c in counties if c["county_fips"] == "13001")
        assert row["county_fips"] == "13001"

    def test_all_available_years_work(self, client):
        for year in (2012, 2016, 2020):
            resp = client.get(f"/api/v1/historical/presidential/{year}")
            assert resp.status_code == 200, f"year {year} failed: {resp.text}"

    def test_unavailable_year_returns_404(self, client):
        resp = client.get("/api/v1/historical/presidential/2000")
        assert resp.status_code == 404

    def test_404_includes_available_years(self, client):
        resp = client.get("/api/v1/historical/presidential/1984")
        assert "2012" in resp.json()["detail"]

    def test_county_count_excludes_aggregate(self, client):
        """Should return 3 counties (12001, 12003, 13001) — not the 00000 row."""
        resp = client.get("/api/v1/historical/presidential/2020")
        assert len(resp.json()["counties"]) == 3
