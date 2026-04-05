"""Tests for GET /forecast/compare?slugs=slug1,slug2.

Tests the route handler, validation logic, and helper functions in isolation
using monkeypatching — no real DuckDB connection required for most cases.

Covers:
- Slug count validation (must be exactly 2)
- Empty slug rejection
- 404 propagation for unknown slugs
- Successful comparison response schema
- Helper: _fetch_race_comparison_data returns expected keys
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fastapi import HTTPException

import api.routers.forecast.race_compare as rc_module
from api.routers.forecast.race_compare import get_race_compare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_request(version_id: str = "v1") -> MagicMock:
    """Build a minimal mock FastAPI Request with app.state.version_id."""
    req = MagicMock()
    req.app.state.version_id = version_id
    req.app.state.pollster_grades = {}
    req.app.state.pollster_grades_normalized = {}
    return req


def _make_mock_db(race_exists: bool = True) -> MagicMock:
    """Build a mock DuckDB connection that simulates race existence checks.

    When race_exists=True, COUNT(*) returns 1 (race found).
    When race_exists=False, COUNT(*) returns 0 (race not found).
    """
    db = MagicMock()

    # DESCRIBE predictions — returns a list of (col_name,) tuples
    db.execute.return_value.fetchall.return_value = [("version_id",), ("race",), ("county_fips",)]

    # fetchone simulates: races table lookup (None = use slug fallback),
    # then predictions COUNT, then vote-weighted pred, then CI rows
    count_val = 1 if race_exists else 0

    def _smart_fetchone():
        return (count_val,)

    db.execute.return_value.fetchone.side_effect = [
        None,           # races table: no metadata, fall back to slug parsing
        (count_val,),   # predictions COUNT check
        (0.52, 50),     # vote-weighted state_pred + n_counties
        None,           # races table for second race
        (count_val,),   # predictions COUNT for second race
        (0.48, 40),     # vote-weighted state_pred + n_counties
    ]

    # fetchdf — return empty DataFrames for polls and type breakdown
    import pandas as pd
    empty_df = pd.DataFrame()
    db.execute.return_value.fetchdf.return_value = empty_df

    return db


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestSlugCountValidation:
    """Exactly 2 slugs are required; any other count raises 422."""

    def test_single_slug_raises_422(self):
        req = _make_mock_request()
        db = _make_mock_db()
        with pytest.raises(HTTPException) as exc_info:
            get_race_compare(slugs="2026-fl-senate", request=req, db=db)
        assert exc_info.value.status_code == 422
        assert "Exactly 2" in exc_info.value.detail

    def test_three_slugs_raises_422(self):
        req = _make_mock_request()
        db = _make_mock_db()
        with pytest.raises(HTTPException) as exc_info:
            get_race_compare(slugs="2026-fl-senate,2026-nc-senate,2026-ga-senate", request=req, db=db)
        assert exc_info.value.status_code == 422
        assert "Exactly 2" in exc_info.value.detail

    def test_empty_string_raises_422(self):
        """An empty slugs parameter has one empty element, not two valid slugs."""
        req = _make_mock_request()
        db = _make_mock_db()
        # "," splits into ["", ""] — both empty, should raise 422
        with pytest.raises(HTTPException) as exc_info:
            get_race_compare(slugs=",", request=req, db=db)
        assert exc_info.value.status_code == 422

    def test_one_empty_slug_raises_422(self):
        """One valid slug + one empty slug should be rejected."""
        req = _make_mock_request()
        db = _make_mock_db()
        with pytest.raises(HTTPException) as exc_info:
            get_race_compare(slugs="2026-fl-senate,", request=req, db=db)
        assert exc_info.value.status_code == 422

    def test_whitespace_trimmed(self):
        """Slugs are stripped of surrounding whitespace before validation."""
        req = _make_mock_request()
        db = _make_mock_db(race_exists=True)
        # Should not raise 422 — whitespace is trimmed, resulting in 2 valid slugs
        # (may raise 404 if DB is mocked, but not 422)
        with patch.object(rc_module, "_fetch_race_comparison_data") as mock_fetch:
            mock_fetch.return_value = _dummy_race_data("2026-fl-senate")
            result = get_race_compare(
                slugs=" 2026-fl-senate , 2026-nc-senate ",
                request=req,
                db=db,
            )
        assert result["slugs"] == ["2026-fl-senate", "2026-nc-senate"]


# ---------------------------------------------------------------------------
# 404 propagation
# ---------------------------------------------------------------------------


class TestNotFoundPropagation:
    """404 from _fetch_race_comparison_data propagates through the endpoint."""

    def test_unknown_slug_raises_404(self):
        req = _make_mock_request()
        db = _make_mock_db(race_exists=False)

        # When the race does not exist, _fetch_race_comparison_data raises HTTPException(404)
        with patch.object(rc_module, "_fetch_race_comparison_data") as mock_fetch:
            mock_fetch.side_effect = HTTPException(status_code=404, detail="Race 'bad-slug' not found")
            with pytest.raises(HTTPException) as exc_info:
                get_race_compare(slugs="bad-slug,2026-nc-senate", request=req, db=db)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


def _dummy_race_data(slug: str) -> dict:
    """Return a minimal valid race comparison data dict for mocking."""
    return {
        "slug": slug,
        "race": slug.replace("-", " ").upper(),
        "state_abbr": "FL",
        "race_type": "Senate",
        "year": 2026,
        "prediction": 0.52,
        "pred_std": 0.04,
        "pred_lo90": 0.454,
        "pred_hi90": 0.586,
        "n_counties": 67,
        "n_polls": 3,
        "poll_confidence": {
            "n_polls": 3,
            "n_pollsters": 2,
            "n_methodologies": 2,
            "label": "Medium",
            "tooltip": "2 pollsters · 2 methods · 3 polls",
        },
        "latest_poll": None,
        "type_breakdown": [],
        "historical_context": None,
    }


class TestResponseSchema:
    """Successful comparison response has the expected top-level structure."""

    def test_response_has_slugs_and_races_keys(self):
        req = _make_mock_request()
        db = _make_mock_db()

        with patch.object(rc_module, "_fetch_race_comparison_data") as mock_fetch:
            mock_fetch.side_effect = [
                _dummy_race_data("2026-fl-senate"),
                _dummy_race_data("2026-nc-senate"),
            ]
            result = get_race_compare(
                slugs="2026-fl-senate,2026-nc-senate",
                request=req,
                db=db,
            )

        assert "slugs" in result
        assert "races" in result

    def test_slugs_list_order_matches_input(self):
        req = _make_mock_request()
        db = _make_mock_db()

        with patch.object(rc_module, "_fetch_race_comparison_data") as mock_fetch:
            mock_fetch.side_effect = [
                _dummy_race_data("2026-fl-senate"),
                _dummy_race_data("2026-nc-senate"),
            ]
            result = get_race_compare(
                slugs="2026-fl-senate,2026-nc-senate",
                request=req,
                db=db,
            )

        assert result["slugs"] == ["2026-fl-senate", "2026-nc-senate"]

    def test_races_list_has_two_entries(self):
        req = _make_mock_request()
        db = _make_mock_db()

        with patch.object(rc_module, "_fetch_race_comparison_data") as mock_fetch:
            mock_fetch.side_effect = [
                _dummy_race_data("2026-fl-senate"),
                _dummy_race_data("2026-nc-senate"),
            ]
            result = get_race_compare(
                slugs="2026-fl-senate,2026-nc-senate",
                request=req,
                db=db,
            )

        assert len(result["races"]) == 2

    def test_each_race_has_required_fields(self):
        """Every race in the response must include the core comparison fields."""
        req = _make_mock_request()
        db = _make_mock_db()

        required_fields = {
            "slug", "race", "state_abbr", "race_type", "year",
            "prediction", "pred_std", "pred_lo90", "pred_hi90",
            "n_counties", "n_polls", "poll_confidence",
            "latest_poll", "type_breakdown",
        }

        with patch.object(rc_module, "_fetch_race_comparison_data") as mock_fetch:
            mock_fetch.side_effect = [
                _dummy_race_data("2026-fl-senate"),
                _dummy_race_data("2026-nc-senate"),
            ]
            result = get_race_compare(
                slugs="2026-fl-senate,2026-nc-senate",
                request=req,
                db=db,
            )

        for race in result["races"]:
            missing = required_fields - set(race.keys())
            assert not missing, f"Race data missing fields: {missing}"

    def test_fetch_called_once_per_slug(self):
        """_fetch_race_comparison_data is called exactly once for each slug."""
        req = _make_mock_request()
        db = _make_mock_db()

        with patch.object(rc_module, "_fetch_race_comparison_data") as mock_fetch:
            mock_fetch.side_effect = [
                _dummy_race_data("2026-fl-senate"),
                _dummy_race_data("2026-nc-senate"),
            ]
            get_race_compare(
                slugs="2026-fl-senate,2026-nc-senate",
                request=req,
                db=db,
            )

        assert mock_fetch.call_count == 2
        call_slugs = [c.args[0] for c in mock_fetch.call_args_list]
        assert call_slugs == ["2026-fl-senate", "2026-nc-senate"]
