"""Model domain: type scores, covariance, priors, ridge priors, HAC weights.

Replaces the six pd.read_parquet() calls in api/main.py at startup.
All tables are version-linked (version_id FK → model_versions).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.db.domains import DomainIngestionError, DomainSpec

log = logging.getLogger(__name__)

# Default type-level dem_share prior used when type_profiles.parquet
# lacks a mean_dem_share column (the column is computed in a later
# pipeline stage not yet run).
DEFAULT_TYPE_PRIOR = 0.45

# Tolerance for covariance matrix symmetry check: |A[i,j] - A[j,i]| ≤ tol.
COVARIANCE_SYMMETRY_TOL = 1e-6

DOMAIN_SPEC = DomainSpec(
    name="model",
    tables=[
        "type_scores", "type_covariance", "type_priors",
        "ridge_county_priors", "hac_state_weights", "hac_county_weights",
    ],
    description="KMeans type scores, covariance matrix, priors, ridge predictions, HAC fallback weights",
    version_key="version_id",
)

# ---------------------------------------------------------------------------
# Pydantic schemas — validate before writing to DuckDB
# ---------------------------------------------------------------------------

class TypeScoreRow(BaseModel):
    county_fips: str
    type_id: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)


class TypeCovarianceRow(BaseModel):
    type_i: int = Field(ge=0)
    type_j: int = Field(ge=0)
    value: float


class TypePriorRow(BaseModel):
    type_id: int = Field(ge=0)
    mean_dem_share: float = Field(ge=0.0, le=1.0)


class RidgeCountyPriorRow(BaseModel):
    county_fips: str
    ridge_pred_dem_share: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS type_scores (
    county_fips VARCHAR NOT NULL,
    type_id     INTEGER NOT NULL,
    score       FLOAT   NOT NULL,
    version_id  VARCHAR NOT NULL,
    PRIMARY KEY (county_fips, type_id, version_id)
);
CREATE TABLE IF NOT EXISTS type_covariance (
    type_i      INTEGER NOT NULL,
    type_j      INTEGER NOT NULL,
    value       FLOAT   NOT NULL,
    version_id  VARCHAR NOT NULL,
    PRIMARY KEY (type_i, type_j, version_id)
);
CREATE TABLE IF NOT EXISTS type_priors (
    type_id        INTEGER NOT NULL,
    mean_dem_share FLOAT   NOT NULL,
    version_id     VARCHAR NOT NULL,
    PRIMARY KEY (type_id, version_id)
);
CREATE TABLE IF NOT EXISTS ridge_county_priors (
    county_fips          VARCHAR NOT NULL,
    ridge_pred_dem_share FLOAT   NOT NULL,
    version_id           VARCHAR NOT NULL,
    PRIMARY KEY (county_fips, version_id)
);
CREATE TABLE IF NOT EXISTS hac_state_weights (
    state_abbr   VARCHAR NOT NULL,
    community_id INTEGER NOT NULL,
    weight       FLOAT   NOT NULL,
    version_id   VARCHAR NOT NULL,
    PRIMARY KEY (state_abbr, community_id, version_id)
);
CREATE TABLE IF NOT EXISTS hac_county_weights (
    county_fips  VARCHAR NOT NULL,
    community_id INTEGER NOT NULL,
    weight       FLOAT   NOT NULL,
    version_id   VARCHAR NOT NULL,
    PRIMARY KEY (county_fips, community_id, version_id)
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_rows(schema_class, rows: list[dict], source: str) -> None:
    for i, row in enumerate(rows):
        try:
            schema_class(**row)
        except Exception as exc:
            raise DomainIngestionError("model", source, f"row {i}: {exc}") from exc


def _cross_compliance(con: duckdb.DuckDBPyConnection, version_id: str) -> None:
    # type_scores.county_fips must exist in counties
    orphans = con.execute("""
        SELECT DISTINCT ts.county_fips
        FROM type_scores ts
        LEFT JOIN counties c ON ts.county_fips = c.county_fips
        WHERE c.county_fips IS NULL AND ts.version_id = ?
    """, [version_id]).fetchdf()
    if not orphans.empty:
        raise DomainIngestionError(
            "model", "type_scores",
            f"unknown county_fips (first 5): {orphans['county_fips'].tolist()[:5]}"
        )

    # type_ids consistent across tables
    max_ts = con.execute("SELECT MAX(type_id) FROM type_scores WHERE version_id=?", [version_id]).fetchone()[0]
    max_tc = con.execute("SELECT MAX(type_i) FROM type_covariance WHERE version_id=?", [version_id]).fetchone()[0]
    max_tp = con.execute("SELECT MAX(type_id) FROM type_priors WHERE version_id=?", [version_id]).fetchone()[0]
    if not (max_ts == max_tc == max_tp):
        raise DomainIngestionError(
            "model", "type tables",
            f"type_id max inconsistent: scores={max_ts}, covariance={max_tc}, priors={max_tp}"
        )

    # type_ids must be zero-indexed contiguous
    actual = sorted(
        con.execute("SELECT DISTINCT type_id FROM type_scores WHERE version_id=?", [version_id])
        .df()["type_id"].tolist()
    )
    expected = list(range(max_ts + 1))
    if actual != expected:
        raise DomainIngestionError(
            "model", "type_scores",
            f"type_ids not zero-indexed contiguous (first 5 actual: {actual[:5]})"
        )

    # ridge_county_priors.county_fips must exist in counties
    orphan_r = con.execute("""
        SELECT DISTINCT r.county_fips
        FROM ridge_county_priors r
        LEFT JOIN counties c ON r.county_fips = c.county_fips
        WHERE c.county_fips IS NULL AND r.version_id = ?
    """, [version_id]).fetchdf()
    if not orphan_r.empty:
        raise DomainIngestionError(
            "model", "ridge_county_priors",
            f"unknown county_fips (first 5): {orphan_r['county_fips'].tolist()[:5]}"
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create all model domain tables (idempotent)."""
    con.execute(_DDL)


def ingest(
    con: duckdb.DuckDBPyConnection,
    version_id: str,
    project_root: Path,
) -> None:
    """Validate and ingest all model domain parquets into DuckDB.

    Gracefully skips individual files that are missing (type pipeline
    artifacts are generated during model runs — they may not exist yet
    during schema-only builds). Cross-compliance runs only when type_scores
    was populated.
    """
    create_tables(con)

    # Clear existing rows for this version
    for table in DOMAIN_SPEC.tables:
        con.execute(f"DELETE FROM {table} WHERE version_id = ?", [version_id])

    data = project_root / "data"
    _ingest_type_scores(con, version_id, data / "communities" / "type_assignments.parquet")
    _ingest_type_covariance(con, version_id, data / "covariance" / "type_covariance.parquet")
    _ingest_type_priors(con, version_id, data / "communities" / "type_profiles.parquet")
    _ingest_ridge_priors(con, version_id, data / "models" / "ridge_model" / "ridge_county_priors.parquet")
    _ingest_hac_state_weights(con, version_id, data / "propagation" / "community_weights_state_hac.parquet")
    _ingest_hac_county_weights(con, version_id, data / "propagation" / "community_weights_county_hac.parquet")

    # Cross-compliance only if type_scores were ingested
    has_scores = con.execute("SELECT COUNT(*) FROM type_scores WHERE version_id=?", [version_id]).fetchone()[0]
    if has_scores > 0:
        _cross_compliance(con, version_id)
        log.info("Model domain cross-compliance passed")


def _ingest_type_scores(con, version_id, path):
    if not path.exists():
        log.warning("type_assignments.parquet not found; skipping type_scores")
        return
    ta = pd.read_parquet(path)
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)
    score_cols = sorted([c for c in ta.columns if c.endswith("_score")])
    if not score_cols:
        raise DomainIngestionError("model", str(path), "no *_score columns found")

    rows = []
    for col in score_cols:
        type_id = int(col.split("_")[1])  # "type_7_score" → 7
        for fips, score in zip(ta["county_fips"], ta[col]):
            rows.append({"county_fips": fips, "type_id": type_id, "score": float(score)})

    _validate_rows(TypeScoreRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO type_scores SELECT * FROM df")
    log.info("type_scores: %d rows (%d counties × %d types)", len(rows), len(ta), len(score_cols))


def _ingest_type_covariance(con, version_id, path):
    if not path.exists():
        log.warning("type_covariance.parquet not found; skipping type_covariance")
        return
    cov_df = pd.read_parquet(path)
    J = cov_df.shape[0]
    mat = cov_df.values[:J, :J].astype(float)
    if not np.allclose(mat, mat.T, atol=COVARIANCE_SYMMETRY_TOL):
        raise DomainIngestionError("model", str(path), "covariance matrix is not symmetric")

    rows = [{"type_i": i, "type_j": j, "value": float(mat[i, j])} for i in range(J) for j in range(J)]
    _validate_rows(TypeCovarianceRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO type_covariance SELECT * FROM df")
    log.info("type_covariance: %d×%d (%d rows)", J, J, len(rows))


def _ingest_type_priors(con, version_id, path):
    """Ingest type-level dem_share priors.

    `type_profiles.parquet` is written by describe_types.py with demographic
    columns only — it does NOT include mean_dem_share yet. When the column is
    absent, fall back to 0.45 for all types (matching api/main.py behavior).
    The fallback reads J from the already-ingested type_scores table.
    """
    if not path.exists():
        log.warning("type_profiles.parquet not found; skipping type_priors")
        return
    profiles = pd.read_parquet(path)
    if "mean_dem_share" in profiles.columns:
        # Use explicit type_id column, not positional index
        rows = [
            {"type_id": int(row["type_id"]), "mean_dem_share": float(row["mean_dem_share"])}
            for _, row in profiles.iterrows()
        ]
    else:
        # Graceful fallback: use 0.45 for every type discovered in type_scores
        log.warning(
            "type_profiles.parquet has no mean_dem_share column — using 0.45 default for all types"
        )
        type_ids = (
            con.execute(
                "SELECT DISTINCT type_id FROM type_scores WHERE version_id=? ORDER BY type_id",
                [version_id],
            )
            .df()["type_id"]
            .tolist()
        )
        if not type_ids:
            log.warning("type_scores also empty; skipping type_priors ingest")
            return
        rows = [{"type_id": int(tid), "mean_dem_share": DEFAULT_TYPE_PRIOR} for tid in type_ids]
    _validate_rows(TypePriorRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO type_priors SELECT * FROM df")
    log.info("type_priors: %d rows", len(rows))


def _ingest_ridge_priors(con, version_id, path):
    if not path.exists():
        log.warning("ridge_county_priors.parquet not found; skipping")
        return
    rf = pd.read_parquet(path)
    rf["county_fips"] = rf["county_fips"].astype(str).str.zfill(5)
    rows = [
        {"county_fips": row["county_fips"], "ridge_pred_dem_share": float(row["ridge_pred_dem_share"])}
        for _, row in rf.iterrows()
    ]
    _validate_rows(RidgeCountyPriorRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO ridge_county_priors SELECT * FROM df")
    log.info("ridge_county_priors: %d rows", len(rows))


def _ingest_hac_state_weights(con, version_id, path):
    if not path.exists():
        log.warning("community_weights_state_hac.parquet not found; skipping hac_state_weights")
        return
    sw = pd.read_parquet(path)
    comm_cols = sorted([c for c in sw.columns if c.startswith("community_")])
    rows = [
        {"state_abbr": row["state_abbr"], "community_id": int(col.split("_")[1]), "weight": float(row[col])}
        for _, row in sw.iterrows()
        for col in comm_cols
    ]
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO hac_state_weights SELECT * FROM df")
    log.info("hac_state_weights: %d rows", len(rows))


def _ingest_hac_county_weights(con, version_id, path):
    if not path.exists():
        log.warning("community_weights_county_hac.parquet not found; skipping hac_county_weights")
        return
    cw = pd.read_parquet(path)
    comm_cols = sorted([c for c in cw.columns if c.startswith("community_")])
    rows = [
        {"county_fips": str(row["county_fips"]).zfill(5), "community_id": int(col.split("_")[1]), "weight": float(row[col])}
        for _, row in cw.iterrows()
        for col in comm_cols
    ]
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO hac_county_weights SELECT * FROM df")
    log.info("hac_county_weights: %d rows", len(rows))
