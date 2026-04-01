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
import gc
import logging
import sys
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DB = PROJECT_ROOT / "data" / "wethervane.duckdb"

# ── Data source paths ──────────────────────────────────────────────────────────
# All paths are relative to PROJECT_ROOT. These change when the data layout changes.
# If you rename/move a file, update the corresponding constant here.
# Note: build() re-derives these from a caller-supplied project_root to support
# tests that pass a custom root (e.g. tmp_path). The module-level constants
# exist for reference and any external callers.

DATA_DIR = PROJECT_ROOT / "data"

# ── Shift data ─────────────────────────────────────────────────────────────────
SHIFTS_MULTIYEAR = DATA_DIR / "shifts" / "county_shifts_multiyear.parquet"

# ── Community / type assignment data ──────────────────────────────────────────
COMMUNITIES_DIR = DATA_DIR / "communities"
COUNTY_ASSIGNMENTS = COMMUNITIES_DIR / "county_community_assignments.parquet"
TYPE_ASSIGNMENTS_STUB = COMMUNITIES_DIR / "county_type_assignments_stub.parquet"
COMMUNITY_PROFILES_PATH = COMMUNITIES_DIR / "community_profiles.parquet"
TYPE_PROFILES_PATH = COMMUNITIES_DIR / "type_profiles.parquet"
COUNTY_TYPE_ASSIGNMENTS_PATH = COMMUNITIES_DIR / "county_type_assignments_full.parquet"
SUPER_TYPES_PATH = COMMUNITIES_DIR / "super_types.parquet"

# ── Predictions ────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_2026 = PREDICTIONS_DIR / "county_predictions_2026.parquet"
PREDICTIONS_2026_HAC = PREDICTIONS_DIR / "county_predictions_2026_hac.parquet"
PREDICTIONS_2026_TYPES = PREDICTIONS_DIR / "county_predictions_2026_types.parquet"

# ── Covariance ─────────────────────────────────────────────────────────────────
COVARIANCE_DIR = DATA_DIR / "covariance"
SIGMA_PATH = COVARIANCE_DIR / "county_community_sigma.parquet"

# ── Tract assignments ──────────────────────────────────────────────────────────
TRACTS_DIR = DATA_DIR / "tracts"
TRACT_TYPE_ASSIGNMENTS_PATH = TRACTS_DIR / "national_tract_assignments.parquet"

# ── Model version metadata ─────────────────────────────────────────────────────
VERSIONS_DIR = DATA_DIR / "models" / "versions"

# ── Assembled / raw reference data ────────────────────────────────────────────
CROSSWALK_PATH = DATA_DIR / "raw" / "fips_county_crosswalk.csv"
PRES_2024_PATH = DATA_DIR / "assembled" / "medsl_county_presidential_2024.parquet"
COUNTY_ACS_FEATURES_PATH = DATA_DIR / "assembled" / "county_acs_features.parquet"
DEMOGRAPHICS_INTERPOLATED_PATH = DATA_DIR / "assembled" / "demographics_interpolated.parquet"

# ── DRY helpers re-exported for backward compat ──────────────────────────────
from src.db._utils import normalize_fips, cycle_connection  # noqa: E402, F401

# ── Sub-module imports ────────────────────────────────────────────────────────
from src.db.schema import create_schema as _create_schema  # noqa: E402
from src.db.ingest import ingest_all as _ingest_data  # noqa: E402
from src.db.validate import (  # noqa: E402
    validate_contract,
    report_summary as _report_summary,
    validate_integrity as _validate_integrity,
)
from src.db.transforms import (  # noqa: E402
    load_version_meta as _load_version_meta,
    build_counties as _build_counties,
    build_county_shifts as _build_county_shifts,
    build_community_assignments as _build_community_assignments,
    build_type_assignments as _build_type_assignments,
    build_predictions as _build_predictions,
)
from src.db.ingest import insert_via_parquet as _insert_via_parquet  # noqa: E402


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(project_root: Path) -> dict[str, Path]:
    """Derive all data source paths from a project root.

    Centralizes path construction so callers can pass a custom root
    (e.g. tmp_path in tests) without monkeypatching module-level constants.
    Returns a flat dict keyed by a short logical name.
    """
    data = project_root / "data"
    communities = data / "communities"
    predictions = data / "predictions"
    return {
        "shifts": data / "shifts" / "county_shifts_multiyear.parquet",
        "assignments": communities / "county_community_assignments.parquet",
        "stub": communities / "county_type_assignments_stub.parquet",
        "predictions": predictions / "county_predictions_2026.parquet",
        "predictions_hac": predictions / "county_predictions_2026_hac.parquet",
        "predictions_types": predictions / "county_predictions_2026_types.parquet",
        "versions_dir": data / "models" / "versions",
        "crosswalk": data / "raw" / "fips_county_crosswalk.csv",
        "sigma": data / "covariance" / "county_community_sigma.parquet",
        "community_profiles": communities / "community_profiles.parquet",
        "county_acs": data / "assembled" / "county_acs_features.parquet",
        "type_profiles": communities / "type_profiles.parquet",
        "county_type_assignments": communities / "county_type_assignments_full.parquet",
        "tract_type_assignments": data / "tracts" / "national_tract_assignments.parquet",
        "super_types": communities / "super_types.parquet",
        "demographics_interpolated": data / "assembled" / "demographics_interpolated.parquet",
    }


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------

def _reset_database(db_path: Path) -> None:
    """Drop the existing database file so the next connect starts fresh."""
    if db_path.exists():
        db_path.unlink()
        log.info("Dropped existing database: %s", db_path)


def build(db_path: Path, reset: bool = False, project_root: Path | None = None) -> None:
    """Build or update wethervane.duckdb from current pipeline artifacts.

    Orchestrates five stages in sequence:
    1. Optional reset (drop existing DB file)
    2. Schema creation (CREATE TABLE IF NOT EXISTS for all tables)
    3. Data ingestion (load parquets, insert rows, cycle connection to avoid DuckDB heap bug)
    4. Summary report (row counts + model versions)
    5. Contract validation (API-frontend contract check; exits 1 on violation)
    """
    _project_root = project_root if project_root is not None else PROJECT_ROOT
    paths = _resolve_paths(_project_root)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if reset:
        _reset_database(db_path)

    con = duckdb.connect(str(db_path))
    _create_schema(con)
    # _ingest_data manages its own connection cycling internally.
    con = cycle_connection(con, db_path, "post-schema")

    _ingest_data(con, db_path, paths, _project_root)

    con = duckdb.connect(str(db_path))
    _report_summary(con, db_path)
    _validate_integrity(con)
    con = cycle_connection(con, db_path, "post-validate")


def main() -> None:
    """Entry point for build_database.py.

    DuckDB 1.5.x corrupts glibc's malloc heap after many large DataFrame
    inserts, causing a SIGABRT crash mid-build. The fix is to use jemalloc
    as the memory allocator (LD_PRELOAD). This function auto-detects jemalloc
    and re-execs itself with LD_PRELOAD set if it's not already active.
    """
    import os
    import subprocess

    parser = argparse.ArgumentParser(description="Build wethervane.duckdb from pipeline artifacts")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Output DuckDB path")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the database")
    args = parser.parse_args()

    # ── jemalloc auto-detection ────────────────────────────────────────────────
    # DuckDB 1.5.x corrupts glibc's malloc heap after many large DataFrame
    # inserts. We must verify jemalloc is *actually loaded* (not just that the
    # env var is set) because `uv run` may not propagate LD_PRELOAD to the
    # Python subprocess. A sentinel var prevents infinite re-exec loops.
    SENTINEL = "_BUILD_DB_JEMALLOC_REEXEC"
    jemalloc_paths = [
        "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
        "/usr/lib/libjemalloc.so.2",
        "/usr/local/lib/libjemalloc.so.2",
    ]

    def _jemalloc_actually_loaded() -> bool:
        """Check /proc/self/maps to verify jemalloc is in the process address space."""
        try:
            maps = Path("/proc/self/maps").read_text()
            return "libjemalloc" in maps
        except OSError:
            return False

    already_reexeced = os.environ.get(SENTINEL) == "1"

    if not _jemalloc_actually_loaded() and not already_reexeced:
        jemalloc_lib = next((p for p in jemalloc_paths if os.path.exists(p)), None)
        if jemalloc_lib:
            log.info(
                "DuckDB 1.5.x malloc bug: re-executing with LD_PRELOAD=%s to avoid SIGABRT",
                jemalloc_lib,
            )
            env = os.environ.copy()
            env["LD_PRELOAD"] = jemalloc_lib
            env[SENTINEL] = "1"
            result = subprocess.run([sys.executable] + sys.argv, env=env)
            sys.exit(result.returncode)
        else:
            log.warning(
                "jemalloc not found — DuckDB 1.5.x may crash with SIGABRT on large builds. "
                "Install libjemalloc2 (apt install libjemalloc2) to fix this."
            )
    elif _jemalloc_actually_loaded():
        log.info("jemalloc verified loaded in process address space")
    elif already_reexeced:
        log.warning(
            "jemalloc re-exec attempted but jemalloc still not loaded — "
            "proceeding without jemalloc (crash risk)"
        )

    build(db_path=args.db, reset=args.reset)

    # DuckDB 1.5.x corrupts the glibc heap during large DataFrame inserts.
    # Even with `del con; gc.collect()`, Python's finalization hits the
    # corrupted heap and crashes with `free(): corrupted unsorted chunks`.
    # Since all data is written and contract-validated, skip finalization.
    os._exit(0)


if __name__ == "__main__":
    main()
