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
    assert list(counties.columns) == ["county_fips", "state_abbr", "state_fips"]
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


def test_build_creates_queryable_db(mini_parquets, monkeypatch):
    """build() should create a DuckDB with all expected tables populated."""
    import src.db.build_database as mod

    # Patch module-level path constants to use tmp_path
    data = mini_parquets
    monkeypatch.setattr(mod, "SHIFTS_MULTIYEAR", data / "data/shifts/county_shifts_multiyear.parquet")
    monkeypatch.setattr(mod, "COUNTY_ASSIGNMENTS", data / "data/communities/county_community_assignments.parquet")
    monkeypatch.setattr(mod, "PREDICTIONS_2026", data / "data/predictions/county_predictions_2026.parquet")
    monkeypatch.setattr(mod, "PREDICTIONS_2026_HAC", data / "data/predictions/nonexistent_hac.parquet")
    monkeypatch.setattr(mod, "TYPE_ASSIGNMENTS_STUB", data / "data/communities/nonexistent_stub.parquet")
    monkeypatch.setattr(mod, "VERSIONS_DIR", data / "data/models/versions")

    db_path = data / "test_bedrock.duckdb"
    build(db_path=db_path, reset=True)

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

    # version_id stored correctly
    vid = con.execute(
        "SELECT version_id FROM model_versions"
    ).fetchone()[0]
    assert vid == "test_model_v1"

    con.close()
