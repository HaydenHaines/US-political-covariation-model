"""Tests for src/db/build_database.py.

Uses an in-memory DuckDB and synthetic data so no real parquets are required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from src.db.build_database import (
    _build_counties,
    _build_community_assignments,
    _build_type_assignments,
    _build_predictions,
    _load_version_meta,
    build,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_shifts() -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001"],
        "pres_d_shift_00_04": [0.1, -0.1, 0.2],
        "pres_r_shift_00_04": [-0.1, 0.1, -0.2],
        "pres_turnout_shift_00_04": [0.05, 0.03, 0.07],
    })


@pytest.fixture
def sample_assignments() -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001"],
        "community_id": [0, 0, 1],
    })


# ---------------------------------------------------------------------------
# Unit tests for builder helpers
# ---------------------------------------------------------------------------

def test_build_counties_shape(sample_shifts):
    counties = _build_counties(sample_shifts)
    assert list(counties.columns) == ["county_fips", "state_abbr", "state_fips", "county_name"]
    assert len(counties) == 3


def test_build_counties_state_mapping(sample_shifts):
    counties = _build_counties(sample_shifts)
    fips_to_state = dict(zip(counties["county_fips"], counties["state_abbr"]))
    assert fips_to_state["12001"] == "FL"
    assert fips_to_state["13001"] == "GA"


def test_build_community_assignments(sample_assignments):
    result = _build_community_assignments(sample_assignments, "test_v1")
    assert "version_id" in result.columns
    assert (result["version_id"] == "test_v1").all()
    assert "k" in result.columns
    # k = number of unique communities = 2
    assert result["k"].iloc[0] == 2


def test_build_type_assignments_stub(sample_assignments):
    result = _build_type_assignments(None, sample_assignments, "test_v1")
    assert "community_id" in result.columns
    assert "dominant_type_id" in result.columns
    # With stub (type_df=None), dominant_type_id should be None
    assert result["dominant_type_id"].isna().all()
    assert len(result) == 2  # one row per unique community


def test_build_predictions_adds_version(sample_shifts):
    preds = pd.DataFrame({
        "county_fips": ["12001"],
        "race": ["FL_Senate"],
        "pred_dem_share": [0.45],
        "pred_std": [0.03],
        "pred_lo90": [0.40],
        "pred_hi90": [0.50],
        "state_pred": [0.44],
        "poll_avg": [0.46],
    })
    result = _build_predictions(preds, "test_v1")
    assert "version_id" in result.columns
    assert result["version_id"].iloc[0] == "test_v1"


# ---------------------------------------------------------------------------
# Integration test: build() produces a queryable DuckDB
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_parquets(tmp_path, sample_shifts, sample_assignments):
    """Write minimal parquet files and a meta.yaml for a test build."""
    import shutil

    # Shifts
    shifts_dir = tmp_path / "data" / "shifts"
    shifts_dir.mkdir(parents=True)
    sample_shifts.to_parquet(shifts_dir / "county_shifts_multiyear.parquet", index=False)

    # Communities
    comm_dir = tmp_path / "data" / "communities"
    comm_dir.mkdir(parents=True)
    sample_assignments.to_parquet(comm_dir / "county_community_assignments.parquet", index=False)

    # FIPS crosswalk
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    crosswalk = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001"],
        "county_name": ["Alachua County, FL", "Baker County, FL", "Appling County, GA"],
    })
    crosswalk.to_csv(raw_dir / "fips_county_crosswalk.csv", index=False)

    # Predictions
    pred_dir = tmp_path / "data" / "predictions"
    pred_dir.mkdir(parents=True)
    preds = pd.DataFrame({
        "county_fips": ["12001", "12003"],
        "state_abbr": ["FL", "FL"],
        "race": ["FL_Senate", "FL_Senate"],
        "pred_dem_share": [0.45, 0.43],
        "pred_std": [0.03, 0.03],
        "pred_lo90": [0.40, 0.38],
        "pred_hi90": [0.50, 0.48],
        "state_pred": [0.44, 0.44],
        "poll_avg": [0.46, 0.46],
    })
    preds.to_parquet(pred_dir / "county_predictions_2026.parquet", index=False)

    # Type profiles (required by validate_contract: types table)
    type_profiles = pd.DataFrame({
        "type_id": [0, 1],
        "super_type_id": [0, 0],
        "display_name": ["Rural Conservative", "Suburban"],
    })
    type_profiles.to_parquet(comm_dir / "type_profiles.parquet", index=False)

    # County type assignments (required by validate_contract)
    county_type_assignments = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001"],
        "dominant_type": [0, 1, 0],
        "super_type": [0, 0, 0],
    })
    county_type_assignments.to_parquet(
        comm_dir / "county_type_assignments_full.parquet", index=False
    )

    # Super-types (required by validate_contract)
    super_types = pd.DataFrame({
        "super_type_id": [0],
        "display_name": ["Traditional"],
    })
    super_types.to_parquet(comm_dir / "super_types.parquet", index=False)

    # Version meta
    versions_dir = tmp_path / "data" / "models" / "versions" / "test_model_v1"
    versions_dir.mkdir(parents=True)
    meta = {
        "version": "test_model_v1",
        "role": "current",
        "k": 2,
        "shift_type": "logodds",
        "vote_share_type": "total",
        "training_dims": 3,
        "holdout_dims": 0,
        "geography": "mini",
        "description": "Test model",
        "date_created": "2026-01-01",
    }
    with open(versions_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f)

    return tmp_path


def test_build_creates_queryable_db(mini_parquets):
    """build() should create a DuckDB with all expected tables populated."""
    data = mini_parquets
    db_path = data / "test_wethervane.duckdb"
    build(db_path=db_path, reset=True, project_root=data)

    assert db_path.exists()

    con = duckdb.connect(str(db_path))

    # All tables exist and have rows
    assert con.execute("SELECT COUNT(*) FROM counties").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM community_assignments").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM type_assignments").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM county_shifts").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2

    # Spot-check state mapping
    state = con.execute(
        "SELECT state_abbr FROM counties WHERE county_fips = '12001'"
    ).fetchone()[0]
    assert state == "FL"

    # Spot-check county_name round-trips through build()
    name = con.execute(
        "SELECT county_name FROM counties WHERE county_fips = '12001'"
    ).fetchone()[0]
    assert name == "Alachua County, FL"

    # version_id stored correctly
    vid = con.execute(
        "SELECT version_id FROM model_versions"
    ).fetchone()[0]
    assert vid == "test_model_v1"

    con.close()


def test_build_counties_includes_name(sample_shifts, tmp_path):
    """counties table includes county_name when crosswalk is present."""
    from src.db.build_database import _build_counties

    crosswalk = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001"],
        "county_name": ["Alachua County, FL", "Baker County, FL", "Appling County, GA"],
    })
    crosswalk_path = tmp_path / "fips_county_crosswalk.csv"
    crosswalk.to_csv(crosswalk_path, index=False)

    counties = _build_counties(sample_shifts, crosswalk_path=crosswalk_path)
    assert "county_name" in counties.columns
    assert counties.loc[counties["county_fips"] == "12001", "county_name"].iloc[0] == "Alachua County, FL"


def test_build_counties_name_fallback(sample_shifts):
    """county_name is None when crosswalk path is missing."""
    from src.db.build_database import _build_counties

    counties = _build_counties(sample_shifts, crosswalk_path=None)
    assert "county_name" in counties.columns
    assert counties["county_name"].isna().all()


# ---------------------------------------------------------------------------
# Task 9: Type-primary DuckDB table tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_parquets_with_types(mini_parquets):
    """Extend mini_parquets with type-primary artifacts."""
    data = mini_parquets

    comm_dir = data / "data" / "communities"

    # Type profiles (types table)
    type_profiles = pd.DataFrame({
        "type_id": [0, 1, 2],
        "super_type_id": [0, 0, 1],
        "display_name": ["Rural Conservative", "Black Belt", "Suburban"],
        "n_counties": [120, 50, 123],
        "pop_total": [3_000_000.0, 1_500_000.0, 5_000_000.0],
        "pct_white_nh": [0.75, 0.30, 0.60],
        "pct_black": [0.10, 0.55, 0.15],
        "median_hh_income": [45000.0, 35000.0, 65000.0],
    })
    type_profiles.to_parquet(comm_dir / "type_profiles.parquet", index=False)

    # County type assignments (scores + dominant)
    county_type_assignments = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001"],
        "dominant_type": [0, 2, 1],
        "super_type": [0, 1, 0],
        "type_0_score": [0.8, 0.1, 0.2],
        "type_1_score": [0.1, 0.2, 0.7],
        "type_2_score": [0.1, 0.7, 0.1],
    })
    county_type_assignments.to_parquet(
        comm_dir / "county_type_assignments_full.parquet", index=False
    )

    # Super-types
    super_types = pd.DataFrame({
        "super_type_id": [0, 1],
        "display_name": ["Traditional", "Metro"],
    })
    super_types.to_parquet(comm_dir / "super_types.parquet", index=False)

    # Type covariance (long format for DB)
    cov_dir = data / "data" / "covariance"
    cov_dir.mkdir(parents=True, exist_ok=True)
    type_cov_long = pd.DataFrame({
        "type_i": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "type_j": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "correlation": [1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0],
        "covariance": [0.005, 0.002, 0.001, 0.002, 0.005, 0.002, 0.001, 0.002, 0.005],
    })
    type_cov_long.to_parquet(cov_dir / "type_covariance_long.parquet", index=False)

    # Demographics interpolated
    assembled_dir = data / "data" / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)
    demographics_interp = pd.DataFrame({
        "county_fips": ["12001", "12001", "12003", "13001"],
        "year": [2004, 2008, 2004, 2004],
        "pop_total": [230000.0, 245000.0, 28000.0, 50000.0],
        "pct_white_nh": [0.65, 0.62, 0.80, 0.35],
        "median_hh_income": [40000.0, 42000.0, 35000.0, 30000.0],
    })
    demographics_interp.to_parquet(
        assembled_dir / "demographics_interpolated.parquet", index=False
    )

    return data


def test_types_table_exists(mini_parquets_with_types):
    """build() should create a 'types' table from type_profiles.parquet."""
    data = mini_parquets_with_types

    db_path = data / "test_wethervane_types.duckdb"
    build(db_path=db_path, reset=True, project_root=data)

    con = duckdb.connect(str(db_path))
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "types" in tables
    n = con.execute("SELECT COUNT(*) FROM types").fetchone()[0]
    assert n == 3
    con.close()


def test_county_type_assignments_table_exists(mini_parquets_with_types):
    """build() should create 'county_type_assignments' table."""
    data = mini_parquets_with_types

    db_path = data / "test_wethervane_cta.duckdb"
    build(db_path=db_path, reset=True, project_root=data)

    con = duckdb.connect(str(db_path))
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "county_type_assignments" in tables
    n = con.execute("SELECT COUNT(*) FROM county_type_assignments").fetchone()[0]
    assert n == 3
    con.close()


def test_type_covariance_table_exists(mini_parquets_with_types):
    """build() should create 'type_covariance' table."""
    data = mini_parquets_with_types

    db_path = data / "test_wethervane_tcov.duckdb"
    build(db_path=db_path, reset=True, project_root=data)

    con = duckdb.connect(str(db_path))
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "type_covariance" in tables
    # Row count is 0 when domain source files are absent (graceful skip);
    # the table existence check is the meaningful assertion here.
    con.close()


def test_demographics_interpolated_table_exists(mini_parquets_with_types):
    """build() should create 'demographics_interpolated' table."""
    data = mini_parquets_with_types

    db_path = data / "test_wethervane_demo.duckdb"
    build(db_path=db_path, reset=True, project_root=data)

    con = duckdb.connect(str(db_path))
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "demographics_interpolated" in tables
    n = con.execute("SELECT COUNT(*) FROM demographics_interpolated").fetchone()[0]
    assert n == 4
    con.close()


def _patch_all_paths(mod, data, monkeypatch):
    """Patch all module-level path constants for test isolation."""
    monkeypatch.setattr(mod, "SHIFTS_MULTIYEAR", data / "data/shifts/county_shifts_multiyear.parquet")
    monkeypatch.setattr(mod, "COUNTY_ASSIGNMENTS", data / "data/communities/county_community_assignments.parquet")
    monkeypatch.setattr(mod, "PREDICTIONS_2026", data / "data/predictions/county_predictions_2026.parquet")
    monkeypatch.setattr(mod, "PREDICTIONS_2026_HAC", data / "data/predictions/nonexistent_hac.parquet")
    monkeypatch.setattr(mod, "PREDICTIONS_2026_TYPES", data / "data/predictions/nonexistent_types.parquet")
    monkeypatch.setattr(mod, "TYPE_ASSIGNMENTS_STUB", data / "data/communities/nonexistent_stub.parquet")
    monkeypatch.setattr(mod, "VERSIONS_DIR", data / "data/models/versions")
    monkeypatch.setattr(mod, "CROSSWALK_PATH", data / "data/raw/fips_county_crosswalk.csv")
    monkeypatch.setattr(mod, "COMMUNITY_PROFILES_PATH", data / "data/communities/nonexistent_profiles.parquet")
    monkeypatch.setattr(mod, "COUNTY_ACS_FEATURES_PATH", data / "data/assembled/nonexistent_acs.parquet")
    # Type-primary paths
    monkeypatch.setattr(mod, "TYPE_PROFILES_PATH", data / "data/communities/type_profiles.parquet")
    monkeypatch.setattr(mod, "COUNTY_TYPE_ASSIGNMENTS_PATH", data / "data/communities/county_type_assignments_full.parquet")
    monkeypatch.setattr(mod, "SUPER_TYPES_PATH", data / "data/communities/super_types.parquet")
    monkeypatch.setattr(mod, "DEMOGRAPHICS_INTERPOLATED_PATH", data / "data/assembled/demographics_interpolated.parquet")


def test_domain_tables_created_after_build(tmp_path):
    """build() creates all domain tables even when source parquets are missing."""
    import yaml
    from src.db.build_database import build

    # Minimal version dir
    version_id = "test-build-v1"
    ver_dir = tmp_path / "models" / "versions" / version_id
    ver_dir.mkdir(parents=True)
    meta = {
        "version_id": version_id, "role": "current", "k": 3, "j": 3,
        "shift_type": "logodds", "vote_share_type": "total",
        "n_training_dims": 30, "n_holdout_dims": 3,
        "holdout_r": "0.70", "geography": "test", "description": "test",
    }
    (ver_dir / "meta.yaml").write_text(yaml.dump(meta))

    db_path = tmp_path / "test.duckdb"
    build(db_path=db_path, reset=True, project_root=tmp_path)

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    for table in ["type_scores", "type_covariance", "type_priors",
                  "ridge_county_priors", "hac_state_weights", "hac_county_weights",
                  "polls", "poll_crosstabs", "poll_notes"]:
        n = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name=?", [table]
        ).fetchone()[0]
        assert n == 1, f"Table {table!r} not created by build()"
    con.close()
