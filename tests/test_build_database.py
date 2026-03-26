"""Tests for DuckDB contract validation."""
import duckdb
import pytest

from src.db.build_database import validate_contract


def _make_valid_db() -> duckdb.DuckDBPyConnection:
    """Build a minimal in-memory DuckDB that passes contract validation."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE counties (county_fips VARCHAR, state_abbr VARCHAR, county_name VARCHAR)")
    con.execute("INSERT INTO counties VALUES ('12001', 'FL', 'Alachua')")
    con.execute("CREATE TABLE super_types (super_type_id INTEGER, display_name VARCHAR)")
    con.execute("INSERT INTO super_types VALUES (0, 'Test Super Type')")
    con.execute("CREATE TABLE types (type_id INTEGER, super_type_id INTEGER, display_name VARCHAR)")
    con.execute("INSERT INTO types VALUES (0, 0, 'Test Type')")
    con.execute("CREATE TABLE county_type_assignments (county_fips VARCHAR, dominant_type INTEGER, super_type INTEGER)")
    con.execute("INSERT INTO county_type_assignments VALUES ('12001', 0, 0)")
    # Domain tables (required by updated contract)
    con.execute("CREATE TABLE type_scores (county_fips VARCHAR, type_id INTEGER, score FLOAT, version_id VARCHAR)")
    con.execute("CREATE TABLE type_priors (type_id INTEGER, mean_dem_share FLOAT, version_id VARCHAR)")
    con.execute("CREATE TABLE polls (poll_id VARCHAR, race VARCHAR, geography VARCHAR, geo_level VARCHAR, dem_share FLOAT, n_sample INTEGER, cycle VARCHAR)")
    return con


def test_valid_db_passes():
    con = _make_valid_db()
    errors = validate_contract(con)
    con.close()
    assert errors == []


def test_missing_table_detected():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE counties (county_fips VARCHAR, state_abbr VARCHAR, county_name VARCHAR)")
    errors = validate_contract(con)
    con.close()
    assert any("MISSING TABLE: super_types" in e for e in errors)


def test_missing_column_detected():
    con = _make_valid_db()
    con.execute("DROP TABLE super_types")
    con.execute("CREATE TABLE super_types (super_type_id INTEGER)")  # missing display_name
    errors = validate_contract(con)
    con.close()
    assert any("MISSING COLUMN: super_types.display_name" in e for e in errors)


def test_orphan_super_type_detected():
    con = _make_valid_db()
    con.execute("DELETE FROM county_type_assignments")
    con.execute("INSERT INTO county_type_assignments VALUES ('12001', 0, 99)")
    errors = validate_contract(con)
    con.close()
    assert any("ORPHAN super_type" in e for e in errors)


def test_orphan_dominant_type_detected():
    con = _make_valid_db()
    con.execute("DELETE FROM county_type_assignments")
    con.execute("INSERT INTO county_type_assignments VALUES ('12001', 99, 0)")
    errors = validate_contract(con)
    con.close()
    assert any("ORPHAN dominant_type" in e for e in errors)


def test_optional_predictions_not_required():
    con = _make_valid_db()
    errors = validate_contract(con)
    con.close()
    assert not any("predictions" in e for e in errors)


def test_optional_predictions_validated_if_present():
    con = _make_valid_db()
    con.execute("CREATE TABLE predictions (county_fips VARCHAR)")  # missing race, pred_dem_share
    errors = validate_contract(con)
    con.close()
    assert any("MISSING COLUMN: predictions.race" in e for e in errors)
