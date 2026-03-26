"""Tests for model domain ingest: type_scores, type_covariance, type_priors,
ridge_county_priors, hac_state_weights, hac_county_weights."""
from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.db.domains import DomainIngestionError
from src.db.domains.model import ingest, _cross_compliance, COVARIANCE_SYMMETRY_TOL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_FIPS = ["12001", "12003", "13001"]
TEST_J = 3  # number of types


def _base_db() -> duckdb.DuckDBPyConnection:
    """In-memory DB with counties + model_versions tables."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE counties (
            county_fips VARCHAR PRIMARY KEY,
            state_abbr VARCHAR, state_fips VARCHAR, county_name VARCHAR
        )
    """)
    for fips in TEST_FIPS:
        state = {"12": "FL", "13": "GA"}[fips[:2]]
        con.execute("INSERT INTO counties VALUES (?, ?, ?, ?)", [fips, state, fips[:2], f"County {fips}"])
    con.execute("""
        CREATE TABLE model_versions (
            version_id VARCHAR PRIMARY KEY, role VARCHAR, k INTEGER, j INTEGER,
            shift_type VARCHAR, vote_share_type VARCHAR, n_training_dims INTEGER,
            n_holdout_dims INTEGER, holdout_r VARCHAR, geography VARCHAR,
            description VARCHAR, created_at TIMESTAMP
        )
    """)
    con.execute("INSERT INTO model_versions VALUES ('test_v1','current',3,3,'logodds','total',30,3,'0.90','test','test','2026-01-01')")
    return con


def _write_type_assignments(tmp: Path) -> None:
    """Write a valid wide-format type_assignments.parquet."""
    df = pd.DataFrame({
        "county_fips": TEST_FIPS,
        "type_0_score": [0.5, 0.3, 0.2],
        "type_1_score": [0.3, 0.5, 0.4],
        "type_2_score": [0.2, 0.2, 0.4],
    })
    (tmp / "data" / "communities").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp / "data" / "communities" / "type_assignments.parquet", index=False)


def _write_type_covariance(tmp: Path) -> None:
    """Write a valid J×J square covariance matrix."""
    cov = np.eye(TEST_J) * 0.01 + np.ones((TEST_J, TEST_J)) * 0.002
    (tmp / "data" / "covariance").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov).to_parquet(tmp / "data" / "covariance" / "type_covariance.parquet")


def _write_type_profiles(tmp: Path) -> None:
    """Write a valid type_profiles.parquet with mean_dem_share."""
    df = pd.DataFrame({"type_id": range(TEST_J), "mean_dem_share": [0.40, 0.50, 0.60]})
    df.to_parquet(tmp / "data" / "communities" / "type_profiles.parquet", index=False)


def _write_ridge_priors(tmp: Path) -> None:
    """Write a valid ridge_county_priors.parquet."""
    df = pd.DataFrame({"county_fips": TEST_FIPS, "ridge_pred_dem_share": [0.42, 0.38, 0.55]})
    (tmp / "data" / "models" / "ridge_model").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet", index=False)


@pytest.fixture
def tmp_data(tmp_path):
    """Write all four parquets to a temp directory tree."""
    _write_type_assignments(tmp_path)
    _write_type_covariance(tmp_path)
    _write_type_profiles(tmp_path)
    _write_ridge_priors(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

def test_ingest_creates_type_scores(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    n = con.execute("SELECT COUNT(*) FROM type_scores WHERE version_id = 'test_v1'").fetchone()[0]
    assert n == len(TEST_FIPS) * TEST_J  # N counties × J types


def test_type_scores_values_sum_to_one_per_county(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    df = con.execute("SELECT county_fips, SUM(score) as total FROM type_scores WHERE version_id='test_v1' GROUP BY county_fips").fetchdf()
    assert (df["total"] - 1.0).abs().max() < 0.01


def test_ingest_creates_type_covariance_symmetric(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    n = con.execute("SELECT COUNT(*) FROM type_covariance WHERE version_id='test_v1'").fetchone()[0]
    assert n == TEST_J ** 2
    # Symmetry: value at (i,j) == value at (j,i)
    asym = con.execute(f"""
        SELECT COUNT(*) FROM type_covariance a
        JOIN type_covariance b ON a.type_i=b.type_j AND a.type_j=b.type_i AND a.version_id=b.version_id
        WHERE ABS(a.value - b.value) > {COVARIANCE_SYMMETRY_TOL} AND a.version_id='test_v1'
    """).fetchone()[0]
    assert asym == 0


def test_ingest_creates_type_priors(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    df = con.execute("SELECT * FROM type_priors WHERE version_id='test_v1' ORDER BY type_id").fetchdf()
    assert len(df) == TEST_J
    assert list(df["mean_dem_share"]) == pytest.approx([0.40, 0.50, 0.60])


def test_ingest_creates_ridge_priors(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    n = con.execute("SELECT COUNT(*) FROM ridge_county_priors WHERE version_id='test_v1'").fetchone()[0]
    assert n == len(TEST_FIPS)


def test_type_ids_zero_indexed_contiguous(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    ids = sorted(con.execute("SELECT DISTINCT type_id FROM type_scores WHERE version_id='test_v1'").df()["type_id"].tolist())
    assert ids == list(range(TEST_J))


# ---------------------------------------------------------------------------
# Tests: validation failures abort ingest
# ---------------------------------------------------------------------------

def test_asymmetric_covariance_raises(tmp_data):
    # Overwrite with asymmetric matrix
    cov = np.eye(TEST_J) * 0.01
    cov[0, 1] = 0.5  # asymmetric
    pd.DataFrame(cov).to_parquet(tmp_data / "data" / "covariance" / "type_covariance.parquet")
    con = _base_db()
    with pytest.raises(DomainIngestionError, match="not symmetric"):
        ingest(con, "test_v1", tmp_data)


def test_unknown_county_fips_raises(tmp_data):
    # Add a row with unknown FIPS to type_assignments
    df = pd.read_parquet(tmp_data / "data" / "communities" / "type_assignments.parquet")
    df.loc[len(df)] = {"county_fips": "99999", "type_0_score": 0.5, "type_1_score": 0.3, "type_2_score": 0.2}
    df.to_parquet(tmp_data / "data" / "communities" / "type_assignments.parquet", index=False)
    con = _base_db()
    with pytest.raises(DomainIngestionError, match="county_fips"):
        ingest(con, "test_v1", tmp_data)


def test_score_out_of_range_raises(tmp_data):
    df = pd.read_parquet(tmp_data / "data" / "communities" / "type_assignments.parquet")
    df["type_0_score"] = 1.5  # out of [0,1]
    df.to_parquet(tmp_data / "data" / "communities" / "type_assignments.parquet", index=False)
    con = _base_db()
    with pytest.raises(DomainIngestionError):
        ingest(con, "test_v1", tmp_data)


# ---------------------------------------------------------------------------
# Tests: HAC weight ingestion
# ---------------------------------------------------------------------------

def _write_hac_state_weights(tmp: Path) -> None:
    """Write a valid wide-format community_weights_state_hac.parquet."""
    df = pd.DataFrame({
        "state_abbr": ["FL", "GA"],
        "community_0": [0.6, 0.4],
        "community_1": [0.3, 0.4],
        "community_2": [0.1, 0.2],
    })
    (tmp / "data" / "propagation").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp / "data" / "propagation" / "community_weights_state_hac.parquet", index=False)


def _write_hac_county_weights(tmp: Path) -> None:
    """Write a valid wide-format community_weights_county_hac.parquet."""
    df = pd.DataFrame({
        "county_fips": TEST_FIPS,
        "community_0": [0.5, 0.2, 0.3],
        "community_1": [0.3, 0.5, 0.4],
        "community_2": [0.2, 0.3, 0.3],
    })
    df.to_parquet(tmp / "data" / "propagation" / "community_weights_county_hac.parquet", index=False)


@pytest.fixture
def tmp_data_with_hac(tmp_data):
    """tmp_data plus HAC weight parquets."""
    _write_hac_state_weights(tmp_data)
    _write_hac_county_weights(tmp_data)
    return tmp_data


def test_ingest_creates_hac_state_weights(tmp_data_with_hac):
    con = _base_db()
    ingest(con, "test_v1", tmp_data_with_hac)
    n = con.execute("SELECT COUNT(*) FROM hac_state_weights WHERE version_id='test_v1'").fetchone()[0]
    # 2 states × 3 communities = 6
    assert n == 2 * 3


def test_ingest_creates_hac_county_weights(tmp_data_with_hac):
    con = _base_db()
    ingest(con, "test_v1", tmp_data_with_hac)
    n = con.execute("SELECT COUNT(*) FROM hac_county_weights WHERE version_id='test_v1'").fetchone()[0]
    # 3 counties × 3 communities = 9
    assert n == len(TEST_FIPS) * 3


def test_hac_weights_missing_gracefully_skipped(tmp_data):
    """HAC parquets absent → ingest succeeds (tables are empty, not error)."""
    con = _base_db()
    ingest(con, "test_v1", tmp_data)  # no HAC files in tmp_data
    n_sw = con.execute("SELECT COUNT(*) FROM hac_state_weights").fetchone()[0]
    n_cw = con.execute("SELECT COUNT(*) FROM hac_county_weights").fetchone()[0]
    assert n_sw == 0
    assert n_cw == 0
