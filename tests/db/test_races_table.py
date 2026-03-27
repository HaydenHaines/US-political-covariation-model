"""Tests for the races table in DuckDB."""
import pytest
import duckdb
from pathlib import Path


DB_PATH = Path("data/wethervane.duckdb")


@pytest.fixture
def db():
    if not DB_PATH.exists():
        pytest.skip("DuckDB not built")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    del con  # Use del instead of close() — see DuckDB heap corruption gotcha


def test_races_table_exists(db):
    tables = [row[0] for row in db.execute("SHOW TABLES").fetchall()]
    assert "races" in tables


def test_races_table_has_all_registered_races(db):
    from src.assembly.define_races import load_races
    registry = load_races(2026)
    db_races = db.execute("SELECT race_id FROM races").fetchall()
    db_ids = {row[0] for row in db_races}
    registry_ids = {r.race_id for r in registry}
    assert registry_ids == db_ids


def test_races_table_schema(db):
    cols = db.execute("DESCRIBE races").fetchall()
    col_names = {row[0] for row in cols}
    assert {"race_id", "race_type", "state", "year"} <= col_names
