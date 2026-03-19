"""Build data/bedrock.duckdb — the central query layer for the Bedrock pipeline.

Reads parquet artifacts from data/shifts/, data/communities/, data/predictions/,
and model version metadata from data/models/versions/*/meta.yaml, then ingests
everything into a single DuckDB file.

Schema
------
counties               one row per county (FIPS, state)
model_versions         version registry with metadata, K, shift_type, holdout_r
community_assignments  Layer 1: county → community_id, keyed by (county_fips, k, version_id)
type_assignments       Layer 2: community → dominant_type_id [stub], keyed by (community_id, k, version_id)
county_shifts          one row per county per version, all shift dims as columns
predictions            2026 forward predictions per (county_fips, race, version_id)

Usage
-----
    python src/db/build_database.py                  # builds from defaults
    python src/db/build_database.py --reset          # drop and rebuild
    python src/db/build_database.py --db path/to/other.duckdb
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "bedrock.duckdb"

# Parquet source paths
SHIFTS_MULTIYEAR = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
COUNTY_ASSIGNMENTS = PROJECT_ROOT / "data" / "communities" / "county_community_assignments.parquet"
PREDICTIONS_2026 = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026.parquet"
TYPE_ASSIGNMENTS_STUB = PROJECT_ROOT / "data" / "communities" / "county_type_assignments_stub.parquet"
VERSIONS_DIR = PROJECT_ROOT / "data" / "models" / "versions"
PREDICTIONS_2026_HAC = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_hac.parquet"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS counties (
    county_fips  VARCHAR PRIMARY KEY,
    state_abbr   VARCHAR NOT NULL,
    state_fips   VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id        VARCHAR PRIMARY KEY,
    role              VARCHAR,          -- 'current', 'previous', 'county_baseline', etc.
    k                 INTEGER,          -- community count K (NULL = not yet determined)
    j                 INTEGER,          -- type count J (NULL = not yet determined)
    shift_type        VARCHAR,          -- 'logodds' or 'raw'
    vote_share_type   VARCHAR,          -- 'total' or 'twoparty'
    n_training_dims   INTEGER,
    n_holdout_dims    INTEGER,
    holdout_r         VARCHAR,          -- holdout Pearson r or range (NULL if not yet validated)
    geography         VARCHAR,          -- e.g. 'FL+GA+AL (293 counties)'
    description       VARCHAR,
    created_at        TIMESTAMP
);

CREATE TABLE IF NOT EXISTS community_assignments (
    county_fips   VARCHAR  NOT NULL,
    community_id  INTEGER  NOT NULL,
    k             INTEGER  NOT NULL,    -- total communities in this model run
    version_id    VARCHAR  NOT NULL,
    PRIMARY KEY (county_fips, k, version_id)
);

CREATE TABLE IF NOT EXISTS type_assignments (
    community_id      INTEGER  NOT NULL,
    k                 INTEGER  NOT NULL,
    dominant_type_id  INTEGER,          -- NULL if stub
    j                 INTEGER,          -- total types
    version_id        VARCHAR  NOT NULL,
    PRIMARY KEY (community_id, k, version_id)
);

CREATE TABLE IF NOT EXISTS county_shifts (
    county_fips  VARCHAR  NOT NULL,
    version_id   VARCHAR  NOT NULL,
    PRIMARY KEY (county_fips, version_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    county_fips    VARCHAR  NOT NULL,
    race           VARCHAR  NOT NULL,
    version_id     VARCHAR  NOT NULL,
    pred_dem_share DOUBLE,
    pred_std       DOUBLE,
    pred_lo90      DOUBLE,
    pred_hi90      DOUBLE,
    state_pred     DOUBLE,
    poll_avg       DOUBLE,
    PRIMARY KEY (county_fips, race, version_id)
);

CREATE TABLE IF NOT EXISTS community_sigma (
    community_id_row  INTEGER NOT NULL,
    community_id_col  INTEGER NOT NULL,
    sigma_value       DOUBLE,
    version_id        VARCHAR NOT NULL,
    PRIMARY KEY (community_id_row, community_id_col, version_id)
);
"""

# State FIPS → abbreviation mapping (hardcoded for FL/GA/AL)
_STATE_FIPS_TO_ABBR = {"12": "FL", "13": "GA", "01": "AL"}


def _load_version_meta(versions_dir: Path) -> list[dict]:
    """Load all meta.yaml files from versioned model directories."""
    meta_list = []
    if not versions_dir.exists():
        log.warning("Versions dir not found: %s", versions_dir)
        return meta_list
    for version_dir in sorted(versions_dir.iterdir()):
        meta_path = version_dir / "meta.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                m = yaml.safe_load(f)
            meta_list.append(m)
            vid = m.get("version_id") or m.get("version") or version_dir.name
            log.info("Loaded version meta: %s (%s)", vid, m.get("role"))
    return meta_list


def _build_counties(shifts: pd.DataFrame) -> pd.DataFrame:
    """Derive the counties table from shift FIPS column."""
    fips = shifts["county_fips"].unique()
    df = pd.DataFrame({"county_fips": sorted(fips)})
    df["state_fips"] = df["county_fips"].str[:2]
    df["state_abbr"] = df["state_fips"].map(_STATE_FIPS_TO_ABBR).fillna("??")
    return df[["county_fips", "state_abbr", "state_fips"]]


def _build_county_shifts(shifts: pd.DataFrame, version_id: str) -> pd.DataFrame:
    """Add version_id column to the wide shifts DataFrame."""
    df = shifts.copy()
    df["version_id"] = version_id
    return df


def _build_community_assignments(
    assignments: pd.DataFrame, version_id: str
) -> pd.DataFrame:
    """Normalize community assignments for DuckDB ingestion."""
    df = assignments[["county_fips", "community_id"]].copy()
    k = int(df["community_id"].nunique())
    df["k"] = k
    df["version_id"] = version_id
    return df[["county_fips", "community_id", "k", "version_id"]]


def _build_type_assignments(
    type_df: pd.DataFrame | None, assignments: pd.DataFrame, version_id: str
) -> pd.DataFrame:
    """Build type_assignments rows from stub or empty DataFrame."""
    k = int(assignments["community_id"].nunique())
    unique_communities = sorted(assignments["community_id"].unique())

    if type_df is not None and "dominant_type_id" in type_df.columns:
        df = type_df[["community_id", "dominant_type_id"]].copy()
        j = int(type_df.get("j", [None])[0]) if "j" in type_df.columns else None
    else:
        # Stub: one row per community with NULL dominant_type_id
        df = pd.DataFrame({"community_id": unique_communities, "dominant_type_id": None})
        j = None

    df["k"] = k
    df["j"] = j
    df["version_id"] = version_id
    return df[["community_id", "k", "dominant_type_id", "j", "version_id"]]


def _build_predictions(preds: pd.DataFrame, version_id: str) -> pd.DataFrame:
    df = preds.copy()
    df["version_id"] = version_id
    cols = [
        "county_fips", "race", "version_id",
        "pred_dem_share", "pred_std", "pred_lo90", "pred_hi90",
        "state_pred", "poll_avg",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def build(db_path: Path, reset: bool = False) -> None:
    """Build or update bedrock.duckdb from current pipeline artifacts."""

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if reset and db_path.exists():
        db_path.unlink()
        log.info("Dropped existing database: %s", db_path)

    con = duckdb.connect(str(db_path))

    # ── Create schema ──────────────────────────────────────────────────────────
    con.executemany("", [])
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
    log.info("Schema created/verified")

    # ── Load source data ───────────────────────────────────────────────────────
    log.info("Loading shifts from %s", SHIFTS_MULTIYEAR)
    shifts = pd.read_parquet(SHIFTS_MULTIYEAR)
    shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)

    log.info("Loading community assignments from %s", COUNTY_ASSIGNMENTS)
    assignments = pd.read_parquet(COUNTY_ASSIGNMENTS)
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
    # Prefer community_id column; fall back to 'community' if needed
    if "community_id" not in assignments.columns and "community" in assignments.columns:
        assignments = assignments.rename(columns={"community": "community_id"})

    type_df = None
    if TYPE_ASSIGNMENTS_STUB.exists():
        log.info("Loading type assignments stub from %s", TYPE_ASSIGNMENTS_STUB)
        type_df = pd.read_parquet(TYPE_ASSIGNMENTS_STUB)

    predictions = None
    if PREDICTIONS_2026.exists():
        log.info("Loading predictions from %s", PREDICTIONS_2026)
        predictions = pd.read_parquet(PREDICTIONS_2026)
        predictions["county_fips"] = predictions["county_fips"].astype(str).str.zfill(5)

    # ── Load model version metadata ────────────────────────────────────────────
    version_metas = _load_version_meta(VERSIONS_DIR)

    # ── Ingest counties ────────────────────────────────────────────────────────
    counties_df = _build_counties(shifts)
    con.execute("DELETE FROM counties")
    con.execute("INSERT INTO counties SELECT * FROM counties_df")
    log.info("Ingested %d counties", len(counties_df))

    # ── Ingest model versions ──────────────────────────────────────────────────
    con.execute("DELETE FROM model_versions")
    for m in version_metas:
        # meta.yaml uses "version" key; "date_created" for creation date
        version_id = m.get("version_id") or m.get("version")
        created_raw = m.get("created_at") or m.get("date_created") or datetime.now(timezone.utc).isoformat()
        con.execute(
            """
            INSERT INTO model_versions
                (version_id, role, k, j, shift_type, vote_share_type,
                 n_training_dims, n_holdout_dims, holdout_r, geography, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                version_id,
                m.get("role"),
                m.get("k"),
                m.get("j"),
                m.get("shift_type"),
                m.get("vote_share_type"),
                m.get("n_training_dims") or m.get("training_dims"),
                m.get("n_holdout_dims") or m.get("holdout_dims"),
                str(m.get("holdout_r")) if m.get("holdout_r") is not None else None,
                m.get("geography"),
                m.get("description"),
                created_raw,
            ],
        )
    log.info("Ingested %d model versions", len(version_metas))

    # ── Ingest county shifts (current version) ─────────────────────────────────
    # Identify current version: prefer role='current', fallback to latest alphabetically
    current_meta = next(
        (m for m in version_metas if m.get("role") == "current"),
        version_metas[-1] if version_metas else {"version": "unknown"},
    )
    current_version_id = current_meta.get("version_id") or current_meta.get("version", "unknown")
    log.info("Using version_id '%s' for shift/assignment ingestion", current_version_id)

    shift_rows = _build_county_shifts(shifts, current_version_id)
    shift_cols = [c for c in shifts.columns if c != "county_fips"]

    # Drop existing shifts for this version and rebuild
    con.execute(f"DELETE FROM county_shifts WHERE version_id = '{current_version_id}'")

    # DuckDB can't ingest a wide parquet with dynamic columns via the fixed schema.
    # Instead we use duckdb's native parquet reader + CREATE OR REPLACE TABLE pattern
    # to handle the dynamic shift columns alongside the fixed ones.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shift_rows.to_parquet(tmp_path, index=False)
        # Use REPLACE because the base CREATE TABLE only has county_fips + version_id columns;
        # we store shifts in a dedicated wide table re-created from parquet.
        con.execute("DROP TABLE IF EXISTS county_shifts")
        con.execute(f"CREATE TABLE county_shifts AS SELECT * FROM read_parquet('{tmp_path}')")
        log.info("Ingested county_shifts: %d rows × %d cols", len(shift_rows), len(shift_rows.columns))
    finally:
        os.unlink(tmp_path)

    # ── Ingest community assignments ───────────────────────────────────────────
    ca_rows = _build_community_assignments(assignments, current_version_id)
    con.execute(f"DELETE FROM community_assignments WHERE version_id = '{current_version_id}'")
    con.execute("INSERT INTO community_assignments SELECT * FROM ca_rows")
    log.info(
        "Ingested community_assignments: %d rows, k=%d", len(ca_rows), ca_rows["k"].iloc[0]
    )

    # ── Ingest type assignments ────────────────────────────────────────────────
    ta_rows = _build_type_assignments(type_df, assignments, current_version_id)
    con.execute(f"DELETE FROM type_assignments WHERE version_id = '{current_version_id}'")
    con.execute("INSERT INTO type_assignments SELECT * FROM ta_rows")
    log.info("Ingested type_assignments: %d rows", len(ta_rows))

    # ── Ingest predictions ─────────────────────────────────────────────────────
    if predictions is not None:
        pred_rows = _build_predictions(predictions, current_version_id)
        con.execute(f"DELETE FROM predictions WHERE version_id = '{current_version_id}'")
        con.execute("INSERT INTO predictions SELECT * FROM pred_rows")
        log.info("Ingested predictions: %d rows", len(pred_rows))
    else:
        log.warning("No predictions file found; skipping predictions table")

    # ── Ingest HAC predictions ──────────────────────────────────────────────────
    if PREDICTIONS_2026_HAC.exists():
        pred_hac = pd.read_parquet(PREDICTIONS_2026_HAC)
        pred_hac["county_fips"] = pred_hac["county_fips"].astype(str).str.zfill(5)
        pred_hac_rows = _build_predictions(pred_hac, current_version_id)
        existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
        new_rows = pred_hac_rows[~pred_hac_rows["race"].isin(existing_races)]
        if len(new_rows):
            con.execute("INSERT INTO predictions SELECT * FROM new_rows")
            log.info("Ingested HAC predictions: %d rows", len(new_rows))
    else:
        log.info("No HAC predictions file found; skipping")

    # ── Ingest community sigma ─────────────────────────────────────────────────
    sigma_path = PROJECT_ROOT / "data" / "covariance" / "county_community_sigma.parquet"
    if sigma_path.exists():
        sigma_df = pd.read_parquet(sigma_path)
        sigma_long_rows = []
        for row_id in sigma_df.index:
            for col_id in sigma_df.columns:
                sigma_long_rows.append({
                    "community_id_row": int(row_id),
                    "community_id_col": int(col_id),
                    "sigma_value": float(sigma_df.loc[row_id, col_id]),
                    "version_id": current_version_id,
                })
        sigma_long = pd.DataFrame(sigma_long_rows)
        con.execute(f"DELETE FROM community_sigma WHERE version_id = '{current_version_id}'")
        con.execute("INSERT INTO community_sigma SELECT * FROM sigma_long")
        log.info("Ingested community_sigma: %d cells", len(sigma_long))
    else:
        log.info("No county_community_sigma.parquet found; skipping sigma table")

    # ── Summary query ──────────────────────────────────────────────────────────
    log.info("Database build complete: %s", db_path)
    print("\n=== bedrock.duckdb summary ===")
    for table in ["counties", "model_versions", "community_assignments", "type_assignments", "county_shifts", "predictions", "community_sigma"]:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n:,} rows")
        except Exception as e:
            print(f"  {table}: ERROR — {e}")

    # Show model versions
    print("\n=== Model Versions ===")
    rows = con.execute(
        "SELECT version_id, role, k, shift_type, vote_share_type, holdout_r FROM model_versions ORDER BY version_id"
    ).fetchall()
    print(f"  {'version_id':<45}  {'role':<20}  {'k':>4}  {'shift_type':<10}  {'holdout_r'}")
    for row in rows:
        vid, role, k, st, vst, hr = row
        print(f"  {str(vid):<45}  {str(role):<20}  {str(k):>4}  {str(st):<10}  {str(hr)}")

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bedrock.duckdb from pipeline artifacts")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Output DuckDB path")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the database")
    args = parser.parse_args()
    build(db_path=args.db, reset=args.reset)


if __name__ == "__main__":
    main()
