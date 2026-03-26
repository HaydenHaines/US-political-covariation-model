"""Build data/wethervane.duckdb — the central query layer for the WetherVane pipeline.

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

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DB = PROJECT_ROOT / "data" / "wethervane.duckdb"

# Parquet source paths
SHIFTS_MULTIYEAR = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
COUNTY_ASSIGNMENTS = PROJECT_ROOT / "data" / "communities" / "county_community_assignments.parquet"
PREDICTIONS_2026 = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026.parquet"
TYPE_ASSIGNMENTS_STUB = PROJECT_ROOT / "data" / "communities" / "county_type_assignments_stub.parquet"
VERSIONS_DIR = PROJECT_ROOT / "data" / "models" / "versions"
PREDICTIONS_2026_HAC = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_hac.parquet"
PREDICTIONS_2026_TYPES = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"
COMMUNITY_PROFILES_PATH = PROJECT_ROOT / "data" / "communities" / "community_profiles.parquet"
COUNTY_ACS_FEATURES_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"

# Type-primary pipeline paths
TYPE_PROFILES_PATH = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
# Narrative generator (template-based, no LLM)
from src.description.generate_narratives import generate_all_narratives  # noqa: E402
COUNTY_TYPE_ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "communities" / "county_type_assignments_full.parquet"
SUPER_TYPES_PATH = PROJECT_ROOT / "data" / "communities" / "super_types.parquet"
DEMOGRAPHICS_INTERPOLATED_PATH = PROJECT_ROOT / "data" / "assembled" / "demographics_interpolated.parquet"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS counties (
    county_fips  VARCHAR PRIMARY KEY,
    state_abbr   VARCHAR NOT NULL,
    state_fips   VARCHAR NOT NULL,
    county_name  VARCHAR
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

CREATE TABLE IF NOT EXISTS community_profiles (
    community_id          INTEGER PRIMARY KEY,
    n_counties            INTEGER,
    pop_total             DOUBLE,
    pct_white_nh          DOUBLE,
    pct_black             DOUBLE,
    pct_asian             DOUBLE,
    pct_hispanic          DOUBLE,
    median_age            DOUBLE,
    median_hh_income      DOUBLE,
    pct_bachelors_plus    DOUBLE,
    pct_owner_occupied    DOUBLE,
    pct_wfh               DOUBLE,
    pct_management        DOUBLE,
    evangelical_share     DOUBLE,
    mainline_share        DOUBLE,
    catholic_share        DOUBLE,
    black_protestant_share DOUBLE,
    congregations_per_1000 DOUBLE,
    religious_adherence_rate DOUBLE
);

CREATE TABLE IF NOT EXISTS county_demographics (
    county_fips           VARCHAR PRIMARY KEY,
    pop_total             DOUBLE,
    pct_white_nh          DOUBLE,
    pct_black             DOUBLE,
    pct_asian             DOUBLE,
    pct_hispanic          DOUBLE,
    median_age            DOUBLE,
    median_hh_income      DOUBLE,
    pct_bachelors_plus    DOUBLE,
    pct_owner_occupied    DOUBLE,
    pct_wfh               DOUBLE,
    pct_management        DOUBLE
);

CREATE TABLE IF NOT EXISTS super_types (
    super_type_id  INTEGER PRIMARY KEY,
    display_name   VARCHAR
);

CREATE TABLE IF NOT EXISTS types (
    type_id        INTEGER PRIMARY KEY,
    super_type_id  INTEGER,
    display_name   VARCHAR
);

CREATE TABLE IF NOT EXISTS county_type_assignments (
    county_fips    VARCHAR NOT NULL,
    dominant_type  INTEGER,
    super_type     INTEGER
);
"""

# State FIPS → abbreviation mapping: sourced from config/model.yaml (all 50+DC).
# Crosswalk CSV is the primary source for county names; this dict handles state_abbr
# for any county whose state is in scope.
from src.core import config as _cfg  # noqa: E402
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR  # fips_prefix → abbr


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


_DEFAULT_CROSSWALK = object()  # sentinel: use module-level CROSSWALK_PATH


def _build_counties(
    shifts: pd.DataFrame,
    crosswalk_path: Path | None = _DEFAULT_CROSSWALK,  # type: ignore[assignment]
) -> pd.DataFrame:
    """Derive the counties table from shift FIPS column, optionally joining county names.

    Args:
        shifts: DataFrame with a county_fips column.
        crosswalk_path: Path to fips_county_crosswalk.csv.  Pass ``None`` to
            skip name lookup (county_name will be all-NULL).  Omit (or pass the
            sentinel ``_DEFAULT_CROSSWALK``) to use the module-level
            ``CROSSWALK_PATH`` constant.
    """
    if crosswalk_path is _DEFAULT_CROSSWALK:
        crosswalk_path = CROSSWALK_PATH

    fips = shifts["county_fips"].unique()
    df = pd.DataFrame({"county_fips": sorted(fips)})
    df["state_fips"] = df["county_fips"].str[:2]
    df["state_abbr"] = df["state_fips"].map(_STATE_FIPS_TO_ABBR).fillna("??")

    if crosswalk_path is not None and Path(crosswalk_path).exists():
        xwalk = pd.read_csv(crosswalk_path, dtype=str)
        xwalk["county_fips"] = xwalk["county_fips"].str.zfill(5)
        df = df.merge(xwalk[["county_fips", "county_name"]], on="county_fips", how="left")
    else:
        df["county_name"] = None

    return df[["county_fips", "state_abbr", "state_fips", "county_name"]]


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


def validate_contract(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Validate DuckDB matches the API-frontend contract.

    Returns a list of violation strings. Empty list = pass.
    See docs/superpowers/specs/2026-03-21-api-frontend-contract-design.md
    """
    errors: list[str] = []

    required = {
        "super_types": ["super_type_id", "display_name"],
        "types": ["type_id", "super_type_id", "display_name"],
        "county_type_assignments": ["county_fips", "dominant_type", "super_type"],
        "counties": ["county_fips", "state_abbr", "county_name"],
        "type_scores": ["county_fips", "type_id", "score"],
        "type_priors": ["type_id", "mean_dem_share"],
        "polls": ["poll_id", "race", "geography", "dem_share"],
    }

    optional = {
        "predictions": ["county_fips", "race", "pred_dem_share"],
    }

    def _check_table(table: str, columns: list[str], is_required: bool) -> None:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            if is_required:
                errors.append(f"MISSING TABLE: {table}")
            return
        actual_cols = set(con.execute(f'SELECT * FROM "{table}" LIMIT 0').fetchdf().columns)
        for col in columns:
            if col not in actual_cols:
                errors.append(f"MISSING COLUMN: {table}.{col}")

    for table, columns in required.items():
        _check_table(table, columns, is_required=True)
    for table, columns in optional.items():
        _check_table(table, columns, is_required=False)

    # Referential integrity (only if required tables exist)
    if not any("MISSING TABLE" in e for e in errors):
        orphans = con.execute("""
            SELECT DISTINCT cta.super_type
            FROM county_type_assignments cta
            LEFT JOIN super_types st ON cta.super_type = st.super_type_id
            WHERE st.super_type_id IS NULL AND cta.super_type IS NOT NULL
        """).fetchdf()
        if not orphans.empty:
            ids = orphans["super_type"].tolist()
            errors.append(f"ORPHAN super_type values in county_type_assignments: {ids}")

        orphan_types = con.execute("""
            SELECT DISTINCT cta.dominant_type
            FROM county_type_assignments cta
            LEFT JOIN types t ON cta.dominant_type = t.type_id
            WHERE t.type_id IS NULL AND cta.dominant_type IS NOT NULL
        """).fetchdf()
        if not orphan_types.empty:
            ids = orphan_types["dominant_type"].tolist()
            errors.append(f"ORPHAN dominant_type values in county_type_assignments: {ids}")

    return errors


def build(db_path: Path, reset: bool = False, project_root: Path | None = None) -> None:
    """Build or update wethervane.duckdb from current pipeline artifacts."""

    _project_root = project_root if project_root is not None else PROJECT_ROOT

    # Derive data paths from the resolved project root so callers can pass
    # a custom root (e.g. tmp_path in tests) without monkeypatching.
    _data = _project_root / "data"
    _shifts_path = _data / "shifts" / "county_shifts_multiyear.parquet"
    _assignments_path = _data / "communities" / "county_community_assignments.parquet"
    _stub_path = _data / "communities" / "county_type_assignments_stub.parquet"
    _predictions_path = _data / "predictions" / "county_predictions_2026.parquet"
    _predictions_hac_path = _data / "predictions" / "county_predictions_2026_hac.parquet"
    _predictions_types_path = _data / "predictions" / "county_predictions_2026_types.parquet"
    _versions_dir = _data / "models" / "versions"
    _crosswalk_path = _data / "raw" / "fips_county_crosswalk.csv"
    _sigma_path = _data / "covariance" / "county_community_sigma.parquet"
    _community_profiles_path = _data / "communities" / "community_profiles.parquet"
    _county_acs_path = _data / "assembled" / "county_acs_features.parquet"
    _type_profiles_path = _data / "communities" / "type_profiles.parquet"
    _county_type_assignments_path = _data / "communities" / "county_type_assignments_full.parquet"
    _super_types_path = _data / "communities" / "super_types.parquet"
    _demographics_interpolated_path = _data / "assembled" / "demographics_interpolated.parquet"

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
    shifts = None
    if _shifts_path.exists():
        log.info("Loading shifts from %s", _shifts_path)
        shifts = pd.read_parquet(_shifts_path)
        shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)
    else:
        log.warning("Shifts file not found: %s; skipping core ingestion", _shifts_path)

    assignments = None
    if _assignments_path.exists():
        log.info("Loading community assignments from %s", _assignments_path)
        assignments = pd.read_parquet(_assignments_path)
        assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
        # Prefer community_id column; fall back to 'community' if needed
        if "community_id" not in assignments.columns and "community" in assignments.columns:
            assignments = assignments.rename(columns={"community": "community_id"})
    else:
        log.warning("Assignments file not found: %s; skipping community ingestion", _assignments_path)

    type_df = None
    if _stub_path.exists():
        log.info("Loading type assignments stub from %s", _stub_path)
        type_df = pd.read_parquet(_stub_path)

    predictions = None
    if _predictions_path.exists():
        log.info("Loading predictions from %s", _predictions_path)
        predictions = pd.read_parquet(_predictions_path)
        predictions["county_fips"] = predictions["county_fips"].astype(str).str.zfill(5)

    # ── Load model version metadata ────────────────────────────────────────────
    version_metas = _load_version_meta(_versions_dir)

    # ── Ingest counties ────────────────────────────────────────────────────────
    if shifts is not None:
        counties_df = _build_counties(shifts, crosswalk_path=_crosswalk_path)
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

    if shifts is not None:
        shift_rows = _build_county_shifts(shifts, current_version_id)

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
    if assignments is not None:
        ca_rows = _build_community_assignments(assignments, current_version_id)
        con.execute(f"DELETE FROM community_assignments WHERE version_id = '{current_version_id}'")
        con.execute("INSERT INTO community_assignments SELECT * FROM ca_rows")
        log.info(
            "Ingested community_assignments: %d rows, k=%d", len(ca_rows), ca_rows["k"].iloc[0]
        )

        # ── Ingest type assignments ────────────────────────────────────────────
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
    if _predictions_hac_path.exists():
        pred_hac = pd.read_parquet(_predictions_hac_path)
        pred_hac["county_fips"] = pred_hac["county_fips"].astype(str).str.zfill(5)
        pred_hac_rows = _build_predictions(pred_hac, current_version_id)
        existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
        new_rows = pred_hac_rows[~pred_hac_rows["race"].isin(existing_races)]
        if len(new_rows):
            con.execute("INSERT INTO predictions SELECT * FROM new_rows")
            log.info("Ingested HAC predictions: %d rows", len(new_rows))
    else:
        log.info("No HAC predictions file found; skipping")

    # ── Ingest types predictions ───────────────────────────────────────────────
    if _predictions_types_path.exists():
        pred_types = pd.read_parquet(_predictions_types_path)
        pred_types["county_fips"] = pred_types["county_fips"].astype(str).str.zfill(5)
        pred_types_rows = _build_predictions(pred_types, current_version_id)
        existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
        # Types predictions take precedence: remove any overlapping (county, race) pairs first
        new_races = set(pred_types_rows["race"].unique()) - existing_races
        overlap_races = set(pred_types_rows["race"].unique()) & existing_races
        if overlap_races:
            for r in overlap_races:
                overlap_fips = pred_types_rows.loc[pred_types_rows["race"] == r, "county_fips"].tolist()
                placeholders = ", ".join(["?"] * len(overlap_fips))
                con.execute(
                    f"DELETE FROM predictions WHERE race = ? AND county_fips IN ({placeholders}) AND version_id = ?",
                    [r] + overlap_fips + [current_version_id],
                )
        con.execute("INSERT INTO predictions SELECT * FROM pred_types_rows")
        log.info("Ingested types predictions: %d rows", len(pred_types_rows))
    else:
        log.info("No types predictions file found; skipping")

    # ── Ingest community sigma ─────────────────────────────────────────────────
    if _sigma_path.exists():
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

    # ── Ingest community profiles (demographics overlay) ───────────────────────
    if _community_profiles_path.exists():
        cp_df = pd.read_parquet(_community_profiles_path)
        # Only load the fixed demographic columns into the structured table
        profile_cols = [
            "community_id", "n_counties", "pop_total",
            "pct_white_nh", "pct_black", "pct_asian", "pct_hispanic",
            "median_age", "median_hh_income", "pct_bachelors_plus",
            "pct_owner_occupied", "pct_wfh", "pct_management",
            "evangelical_share", "mainline_share", "catholic_share",
            "black_protestant_share", "congregations_per_1000",
            "religious_adherence_rate",
        ]
        cp_insert = cp_df[[c for c in profile_cols if c in cp_df.columns]].copy()
        con.execute("DELETE FROM community_profiles")
        con.execute("INSERT INTO community_profiles SELECT * FROM cp_insert")
        log.info("Ingested community_profiles: %d rows", len(cp_insert))
    else:
        log.info("No community_profiles.parquet found; skipping")

    # ── Ingest county demographics ──────────────────────────────────────────────
    if _county_acs_path.exists():
        cd_df = pd.read_parquet(_county_acs_path)
        cd_df["county_fips"] = cd_df["county_fips"].astype(str).str.zfill(5)
        con.execute("DROP TABLE IF EXISTS county_demographics")
        con.execute("CREATE TABLE county_demographics AS SELECT * FROM cd_df")
        log.info("Ingested county_demographics: %d rows (%d columns)", len(cd_df), len(cd_df.columns))
    else:
        log.info("No county_acs_features.parquet found; skipping")

    # ── Ingest type profiles (types table) ────────────────────────────────────
    if _type_profiles_path.exists():
        tp_df = pd.read_parquet(_type_profiles_path)
        # Add super_type_id from county_type_assignments if available
        if _county_type_assignments_path.exists() and "super_type_id" not in tp_df.columns:
            cta_tmp = pd.read_parquet(_county_type_assignments_path)
            if "super_type" in cta_tmp.columns and "dominant_type" in cta_tmp.columns:
                type_to_super = cta_tmp.groupby("dominant_type")["super_type"].first()
                tp_df["super_type_id"] = tp_df["type_id"].map(type_to_super)
                log.info("Added super_type_id to types table from county_type_assignments")
        # Add display_name if missing (generic names until proper naming pipeline)
        if "display_name" not in tp_df.columns:
            tp_df["display_name"] = tp_df["type_id"].apply(lambda x: f"Type {x}")
        # ── Generate and attach narratives ────────────────────────────────────
        log.info("Generating type narratives from demographic z-scores")
        try:
            narratives = generate_all_narratives(str(_type_profiles_path))
            tp_df["narrative"] = tp_df["type_id"].map(narratives)
            log.info("Attached narratives to %d types", tp_df["narrative"].notna().sum())
        except Exception as exc:
            log.warning("Narrative generation failed (%s); types table will lack narrative column", exc)
        con.execute("DROP TABLE IF EXISTS types")
        con.execute("CREATE TABLE types AS SELECT * FROM tp_df")
        log.info("Ingested types: %d rows", len(tp_df))
    else:
        log.info("No type_profiles.parquet found; skipping types table")

    # ── Ingest county type assignments ─────────────────────────────────────────
    if _county_type_assignments_path.exists():
        cta_df = pd.read_parquet(_county_type_assignments_path)
        cta_df["county_fips"] = cta_df["county_fips"].astype(str).str.zfill(5)
        con.execute("DROP TABLE IF EXISTS county_type_assignments")
        con.execute("CREATE TABLE county_type_assignments AS SELECT * FROM cta_df")
        log.info("Ingested county_type_assignments: %d rows", len(cta_df))
    else:
        log.info("No county_type_assignments_full.parquet found; skipping")

    # ── Ingest super-types ─────────────────────────────────────────────────────
    if _super_types_path.exists():
        st_df = pd.read_parquet(_super_types_path)
        con.execute("DELETE FROM super_types")
        con.execute("INSERT INTO super_types SELECT * FROM st_df")
        log.info("Ingested super_types: %d rows", len(st_df))
    else:
        log.info("No super_types.parquet found; skipping")

    # ── Domain ingest ────────────────────────────────────────────────────────
    from src.db.domains.model import ingest as ingest_model, create_tables as model_ddl
    from src.db.domains.polling import ingest as ingest_polling, create_tables as polling_ddl

    model_ddl(con)
    polling_ddl(con)

    ingest_model(con, current_version_id, _project_root)
    ingest_polling(con, "2026", _project_root)

    # ── Ingest demographics interpolated ───────────────────────────────────────
    if _demographics_interpolated_path.exists():
        di_df = pd.read_parquet(_demographics_interpolated_path)
        di_df["county_fips"] = di_df["county_fips"].astype(str).str.zfill(5)
        con.execute("DROP TABLE IF EXISTS demographics_interpolated")
        con.execute("CREATE TABLE demographics_interpolated AS SELECT * FROM di_df")
        log.info("Ingested demographics_interpolated: %d rows", len(di_df))
    else:
        log.info("No demographics_interpolated.parquet found; skipping")

    # ── Summary query ──────────────────────────────────────────────────────────
    log.info("Database build complete: %s", db_path)
    print("\n=== wethervane.duckdb summary ===")
    for table in ["counties", "model_versions", "community_assignments", "type_assignments", "county_shifts", "predictions", "community_sigma", "community_profiles", "county_demographics", "types", "county_type_assignments", "super_types", "type_covariance", "demographics_interpolated"]:
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

    # ── Contract validation ────────────────────────────────────────────────────
    errors = validate_contract(con)
    if errors:
        for e in errors:
            log.error("CONTRACT VIOLATION: %s", e)
        con.close()
        import sys
        sys.exit(1)
    log.info("Contract validation passed")

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wethervane.duckdb from pipeline artifacts")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Output DuckDB path")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the database")
    args = parser.parse_args()
    build(db_path=args.db, reset=args.reset)


if __name__ == "__main__":
    main()
