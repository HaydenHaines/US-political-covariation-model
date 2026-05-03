"""Tests for GET /api/v1/forecast/xt-impact endpoint.

Uses monkeypatching to avoid hitting real data files.  The underlying
make_xt_impact_report() is stubbed out so tests run in CI without gitignored data.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared fixture: a stub report returned by make_xt_impact_report
# ---------------------------------------------------------------------------

_STUB_REPORT = {
    "enriched_deltas": {
        f"2026 Race{i}": float(i) * 0.1
        for i in range(1, 26)  # 25 races so we can test limit
    },
    "mean_delta": 0.13,
    "max_delta": 2.4,
    "races_with_xt": 12,
    "report_date": "2026-05-03",
}


@pytest.fixture
def client():
    """TestClient with make_xt_impact_report stubbed out."""
    from api.routers.forecast import xt_impact as xt_module

    # Clear any TTL cache between tests
    xt_module._cache.clear()

    with patch(
        "src.prediction.forecast_engine.make_xt_impact_report",
        return_value=_STUB_REPORT,
    ):
        # Import app after patching so the stub is in place when the router loads
        from api.main import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


class TestXtImpactEndpoint:
    def test_returns_200_with_top_movers(self, client):
        resp = client.get("/api/v1/forecast/xt-impact")
        assert resp.status_code == 200
        data = resp.json()
        assert "top_movers" in data

    def test_response_has_all_required_fields(self, client):
        resp = client.get("/api/v1/forecast/xt-impact")
        data = resp.json()
        for field in ("top_movers", "mean_delta", "max_delta", "races_with_xt", "report_date"):
            assert field in data, f"Missing field: {field}"

    def test_top_movers_sorted_by_abs_delta_descending(self, client):
        resp = client.get("/api/v1/forecast/xt-impact")
        movers = resp.json()["top_movers"]
        assert len(movers) > 1
        deltas = [abs(m["delta_pp"]) for m in movers]
        assert deltas == sorted(deltas, reverse=True), "top_movers not sorted by |delta_pp| desc"

    def test_each_mover_has_race_id_and_delta_pp(self, client):
        resp = client.get("/api/v1/forecast/xt-impact")
        for mover in resp.json()["top_movers"]:
            assert "race_id" in mover
            assert "delta_pp" in mover

    def test_limit_param_returns_exact_count(self, client):
        resp = client.get("/api/v1/forecast/xt-impact?limit=5")
        assert resp.status_code == 200
        movers = resp.json()["top_movers"]
        assert len(movers) == 5

    def test_limit_1_returns_single_item(self, client):
        resp = client.get("/api/v1/forecast/xt-impact?limit=1")
        assert len(resp.json()["top_movers"]) == 1

    def test_default_limit_is_20(self, client):
        # stub has 25 races; default limit should cap at 20
        resp = client.get("/api/v1/forecast/xt-impact")
        assert len(resp.json()["top_movers"]) == 20

    def test_scalar_fields_match_stub(self, client):
        data = client.get("/api/v1/forecast/xt-impact").json()
        assert data["mean_delta"] == pytest.approx(0.13)
        assert data["max_delta"] == pytest.approx(2.4)
        assert data["races_with_xt"] == 12
        assert data["report_date"] == "2026-05-03"


# ---------------------------------------------------------------------------
# race_type filter tests
# ---------------------------------------------------------------------------

_STUB_REPORT_TYPED = {
    "enriched_deltas": {
        "2026-tx-senate": 1.5,
        "2026-fl-senate": 1.2,
        "2026-pa-senate": 0.5,
        "2026-nc-governor": 0.9,
        "2026-wi-governor": 0.7,
    },
    "xt_race_counts": {
        "2026-tx-senate": 3,
        "2026-fl-senate": 2,
        "2026-pa-senate": 1,
        "2026-nc-governor": 1,
        "2026-wi-governor": 2,
    },
    "mean_delta": 0.96,
    "max_delta": 1.5,
    "races_with_xt": 5,
    "report_date": "2026-05-03",
}


@pytest.fixture
def client_typed():
    from api.routers.forecast import xt_impact as xt_module

    xt_module._cache.clear()

    with patch(
        "src.prediction.forecast_engine.make_xt_impact_report",
        return_value=_STUB_REPORT_TYPED,
    ):
        from api.main import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestXtImpactRaceTypeFilter:
    def test_governor_returns_only_governor_races(self, client_typed):
        movers = client_typed.get("/api/v1/forecast/xt-impact?race_type=governor").json()[
            "top_movers"
        ]
        assert len(movers) == 2
        assert all("governor" in m["race_id"] for m in movers)

    def test_senate_returns_only_senate_races(self, client_typed):
        movers = client_typed.get("/api/v1/forecast/xt-impact?race_type=senate").json()[
            "top_movers"
        ]
        assert len(movers) == 3
        assert all("senate" in m["race_id"] for m in movers)

    def test_race_type_filter_is_case_insensitive(self, client_typed):
        movers = client_typed.get("/api/v1/forecast/xt-impact?race_type=GOVERNOR").json()[
            "top_movers"
        ]
        assert len(movers) == 2

    def test_omitted_race_type_returns_all(self, client_typed):
        # Use limit=100 to avoid the module-level forecast_cache populated by TestXtImpactEndpoint
        movers = client_typed.get("/api/v1/forecast/xt-impact?limit=100").json()["top_movers"]
        assert len(movers) == 5

    def test_no_match_returns_empty_list(self, client_typed):
        movers = client_typed.get("/api/v1/forecast/xt-impact?race_type=president").json()[
            "top_movers"
        ]
        assert movers == []

    def test_filtered_movers_still_sorted_by_abs_delta(self, client_typed):
        movers = client_typed.get("/api/v1/forecast/xt-impact?race_type=senate").json()[
            "top_movers"
        ]
        deltas = [abs(m["delta_pp"]) for m in movers]
        assert deltas == sorted(deltas, reverse=True)
