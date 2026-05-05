"""Tests for validate_predictions() — Tier 1 build-time sanity checks.

Verifies that the prediction validator catches systematic model bugs
like type-compression (#139) while passing valid predictions.
"""
from __future__ import annotations

import duckdb
import pytest


def _setup_db_with_predictions(
    con: duckdb.DuckDBPyConnection,
    state_preds: dict[str, float],
    n_counties_per_state: int = 10,
) -> None:
    """Create minimal DuckDB tables with given state-level predictions.

    Distributes predictions across synthetic counties within each state,
    with uniform vote weights so vote-weighted average = the given value.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS counties (
            county_fips VARCHAR PRIMARY KEY,
            state_abbr VARCHAR,
            county_name VARCHAR,
            total_votes_2024 INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            county_fips VARCHAR,
            race VARCHAR,
            version_id VARCHAR,
            forecast_mode VARCHAR,
            pred_dem_share DOUBLE
        )
    """)

    fips_counter = 1000
    for state, pred_value in state_preds.items():
        for i in range(n_counties_per_state):
            fips = str(fips_counter).zfill(5)
            fips_counter += 1
            con.execute(
                "INSERT INTO counties VALUES (?, ?, ?, ?)",
                [fips, state, f"{state} County {i}", 10000],
            )
            con.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                [fips, f"Senate {state}", "v_types_test", "local", pred_value],
            )


@pytest.fixture
def fresh_con():
    """In-memory DuckDB connection for each test."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()


class TestValidatePredictionsPass:
    """Realistic predictions should pass all checks."""

    def test_valid_predictions_pass(self, fresh_con):
        from src.db.validate import validate_predictions

        # Realistic spread of predictions
        preds = {
            "MA": 0.63, "RI": 0.60, "DE": 0.58, "IL": 0.57,
            "NJ": 0.55, "NH": 0.52, "GA": 0.49, "NC": 0.48,
            "TX": 0.43, "SC": 0.42, "KY": 0.37, "AL": 0.36,
            "OK": 0.34, "ID": 0.32, "WV": 0.30, "WY": 0.27,
        }
        _setup_db_with_predictions(fresh_con, preds)
        errors = validate_predictions(fresh_con)
        assert errors == [], f"Valid predictions should pass: {errors}"


class TestValidatePredictionsCatchBugs:
    """Each check should catch its target bug."""

    def test_catches_type_compression(self, fresh_con):
        """Type-compression: all predictions near 0.49."""
        from src.db.validate import validate_predictions

        # All states predict ~0.49 (the bug #139 produced)
        preds = {
            "MA": 0.491, "NJ": 0.489, "IL": 0.492,
            "WY": 0.488, "WV": 0.490, "GA": 0.491,
            "TX": 0.489, "SC": 0.490, "OK": 0.491,
        }
        _setup_db_with_predictions(fresh_con, preds)
        errors = validate_predictions(fresh_con)
        assert len(errors) > 0, "Should catch compressed predictions"
        # Should trigger spread check AND safe-seat checks
        assert any("SPREAD" in e for e in errors)

    def test_catches_safe_d_violation(self, fresh_con):
        """Safe D state predicting R should be caught."""
        from src.db.validate import validate_predictions

        preds = {
            "MA": 0.40,  # MA predicting R — clearly wrong
            "NJ": 0.55, "IL": 0.57, "GA": 0.49,
            "TX": 0.43, "WY": 0.27, "WV": 0.30, "OK": 0.34, "ID": 0.32,
        }
        _setup_db_with_predictions(fresh_con, preds)
        errors = validate_predictions(fresh_con)
        assert any("SAFE D STATE" in e and "MA" in e for e in errors)

    def test_catches_safe_r_violation(self, fresh_con):
        """Safe R state predicting D should be caught."""
        from src.db.validate import validate_predictions

        preds = {
            "WY": 0.55,  # WY predicting D — clearly wrong
            "MA": 0.63, "NJ": 0.55, "IL": 0.57, "GA": 0.49,
            "TX": 0.43, "WV": 0.30, "OK": 0.34, "ID": 0.32,
        }
        _setup_db_with_predictions(fresh_con, preds)
        errors = validate_predictions(fresh_con)
        assert any("SAFE R STATE" in e and "WY" in e for e in errors)

    def test_catches_nj_canary(self, fresh_con):
        """NJ predicting R is a structural bug indicator."""
        from src.db.validate import validate_predictions

        preds = {
            "NJ": 0.48,  # NJ R+2 — the original bug
            "MA": 0.63, "IL": 0.57, "GA": 0.49,
            "TX": 0.43, "WY": 0.27, "WV": 0.30, "OK": 0.34, "ID": 0.32,
        }
        _setup_db_with_predictions(fresh_con, preds)
        errors = validate_predictions(fresh_con)
        assert any("NJ CANARY" in e for e in errors)

    def test_catches_one_sided_predictions(self, fresh_con):
        """All predictions leaning one direction should be caught."""
        from src.db.validate import validate_predictions

        # Everything leans D — no R predictions
        preds = {
            "MA": 0.63, "NJ": 0.58, "IL": 0.57, "GA": 0.56,
            "TX": 0.55, "WY": 0.52, "WV": 0.51, "OK": 0.50, "ID": 0.49,
        }
        _setup_db_with_predictions(fresh_con, preds)
        errors = validate_predictions(fresh_con)
        assert any("ONE-SIDED" in e for e in errors)

    def test_catches_extreme_margins(self, fresh_con):
        """Predictions outside [0.05, 0.95] should be caught."""
        from src.db.validate import validate_predictions

        preds = {
            "MA": 0.97,  # Extreme — no statewide race hits 97%
            "NJ": 0.55, "IL": 0.57, "GA": 0.49,
            "TX": 0.43, "WY": 0.27, "WV": 0.30, "OK": 0.34, "ID": 0.32,
        }
        _setup_db_with_predictions(fresh_con, preds)
        errors = validate_predictions(fresh_con)
        assert any("EXTREME MARGINS" in e for e in errors)


class TestValidatePredictionsCatchesEmptyCounties:
    """Regression test for 2026-04-29: empty counties table causes D+25 fallback."""

    def test_catches_empty_counties_when_predictions_present(self, fresh_con):
        """validate_predictions must error when counties is empty but predictions exist.

        Root cause: an empty counties table makes every JOIN return NULL, causing
        all overview endpoints to return ±_DEFAULT_SAFE_MARGIN (0.25) for every
        race — shown as D+25pp or R+25pp in the frontend.
        """
        from src.db.validate import validate_predictions

        # Predictions exist but counties is empty
        fresh_con.execute("""
            CREATE TABLE counties (
                county_fips VARCHAR PRIMARY KEY,
                state_abbr VARCHAR,
                county_name VARCHAR,
                total_votes_2024 INTEGER
            )
        """)
        fresh_con.execute("""
            CREATE TABLE predictions (
                county_fips VARCHAR, race VARCHAR,
                version_id VARCHAR, forecast_mode VARCHAR,
                pred_dem_share DOUBLE
            )
        """)
        for fips, share in [("04001", 0.54), ("04003", 0.55), ("48001", 0.40)]:
            fresh_con.execute(
                "INSERT INTO predictions VALUES (?, '2026 AZ Governor', 'v1', 'local', ?)",
                [fips, share],
            )

        errors = validate_predictions(fresh_con)
        assert any("COUNTIES TABLE EMPTY" in e for e in errors), (
            f"Expected COUNTIES TABLE EMPTY error, got: {errors}"
        )


class TestValidatePredictionsEdgeCases:
    """Edge cases that should not crash."""

    def test_no_predictions_table(self, fresh_con):
        """Missing predictions table should warn, not crash."""
        from src.db.validate import validate_predictions

        errors = validate_predictions(fresh_con)
        assert errors == []

    def test_empty_predictions(self, fresh_con):
        """Empty predictions table should warn, not crash."""
        from src.db.validate import validate_predictions

        fresh_con.execute("""
            CREATE TABLE predictions (
                county_fips VARCHAR, race VARCHAR,
                version_id VARCHAR, forecast_mode VARCHAR,
                pred_dem_share DOUBLE
            )
        """)
        fresh_con.execute("""
            CREATE TABLE counties (
                county_fips VARCHAR, state_abbr VARCHAR,
                county_name VARCHAR, total_votes_2024 INTEGER
            )
        """)
        errors = validate_predictions(fresh_con)
        assert errors == []
