"""Migration: add super_type_id column to types table + seed super_types.

The types table was built before super_type_id was wired into the ingest
pipeline. This migration:
  1. Adds the super_type_id column to types (if missing)
  2. Seeds super_types with 5 canonical names for the J=100 model
  3. Assigns each of the 100 types to a super_type via Ward HAC on
     demographic profiles (matching the approach from commit 4c69e50)

Super-type names from J=100 clustering run (commit 4c69e50):
  Cluster by highest-weight feature:
    pct_black dominant → Black-Belt Urban
    pct_hispanic dominant → Hispanic Exurban
    high evangelical + low density → Rural Evangelical
    high education + high income → Affluent College
    remainder (young, lower density) → Rural Young

Usage:
    uv run python scripts/migrate_add_super_type_id.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "wethervane.duckdb"
TYPE_PROFILES_PATH = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"

# Canonical super-type IDs and placeholder names (assigned after clustering)
N_SUPER = 5


def _assign_super_type_names(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign the 5 canonical names to clusters by matching dominant features.

    Uses a greedy best-match: each cluster is scored for each name, then
    names are assigned uniquely to avoid collisions.
    """
    canonical = {
        "Black-Belt Urban": lambda r: r.get("pct_black", 0),
        "Hispanic Exurban": lambda r: r.get("pct_hispanic", 0),
        "Rural Evangelical": lambda r: r.get("evangelical_share", 0) * (1 / (1 + r.get("log_pop_density", 0))),
        "Affluent College": lambda r: r.get("pct_bachelors_plus", 0) + r.get("log_median_hh_income", 0) / 10,
        "Rural Young": lambda r: 1 / (1 + r.get("log_pop_density", 0)),
    }
    names = list(canonical.keys())
    # Build score matrix: clusters x names
    cids = sorted(centroids.index.tolist())
    scores = np.zeros((len(cids), len(names)))
    for i, cid in enumerate(cids):
        row = centroids.loc[cid]
        for j, (name, fn) in enumerate(canonical.items()):
            scores[i, j] = fn(row)

    # Greedy assignment: pick highest score, mark row+col used
    assigned: dict[int, str] = {}
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = sorted(
        [(scores[i, j], i, j) for i in range(len(cids)) for j in range(len(names))],
        reverse=True,
    )
    for _, i, j in flat:
        if i in used_rows or j in used_cols:
            continue
        assigned[cids[i]] = names[j]
        used_rows.add(i)
        used_cols.add(j)
        if len(assigned) == len(cids):
            break

    return assigned


def _compute_super_type_assignments(profiles_path: Path) -> tuple[dict[int, int], list[tuple[int, str]]]:
    """Cluster 100 types into 5 super-types via Ward HAC on demographic profiles.

    Returns:
        type_to_super: mapping of type_id → super_type_id
        super_type_defs: list of (super_type_id, display_name)
    """
    tp = pd.read_parquet(profiles_path)
    feature_cols = [
        "pct_bachelors_plus", "pct_white_nh", "pct_black", "pct_hispanic",
        "pct_asian", "evangelical_share", "median_age", "log_pop_density",
        "pct_owner_occupied", "log_median_hh_income",
    ]
    available = [c for c in feature_cols if c in tp.columns]
    X = tp[available].fillna(tp[available].median())
    X_scaled = StandardScaler().fit_transform(X)

    clustering = AgglomerativeClustering(n_clusters=N_SUPER, linkage="ward")
    labels = clustering.fit_predict(X_scaled)

    tp = tp.copy()
    tp["_cluster"] = labels

    # Build centroid per cluster in original (unscaled) units for naming
    centroids = tp.groupby("_cluster")[available].mean()
    cluster_names = _assign_super_type_names(centroids)

    # Map cluster IDs (0..4) to super_type_ids (0..4)
    type_to_super = dict(zip(tp["type_id"].tolist(), labels.tolist()))
    super_type_defs = [(int(cid), cluster_names[cid]) for cid in sorted(cluster_names)]

    return type_to_super, super_type_defs


def migrate(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    con = duckdb.connect(str(db_path))

    # 1. Add super_type_id to types table if missing
    existing_cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'types'"
        ).fetchall()
    }
    if "super_type_id" not in existing_cols:
        con.execute("ALTER TABLE types ADD COLUMN super_type_id INTEGER")
        log.info("Added super_type_id column to types table")
    else:
        log.info("super_type_id column already exists in types table")

    # 2. Compute assignments from type profiles (if available)
    if TYPE_PROFILES_PATH.exists():
        type_to_super, super_type_defs = _compute_super_type_assignments(TYPE_PROFILES_PATH)
        log.info(
            "Computed super_type assignments for %d types → %d super_types",
            len(type_to_super),
            len(super_type_defs),
        )

        # 3. Populate super_types table
        con.execute("DELETE FROM super_types")
        for st_id, name in super_type_defs:
            con.execute(
                "INSERT INTO super_types (super_type_id, display_name) VALUES (?, ?)",
                [st_id, name],
            )
        log.info("Seeded super_types: %s", {st_id: name for st_id, name in super_type_defs})

        # 4. Update types.super_type_id
        for type_id, super_id in type_to_super.items():
            con.execute(
                "UPDATE types SET super_type_id = ? WHERE type_id = ?",
                [int(super_id), int(type_id)],
            )

        assigned = con.execute(
            "SELECT COUNT(*) FROM types WHERE super_type_id IS NOT NULL"
        ).fetchone()[0]
        log.info("Updated super_type_id for %d types", assigned)
    else:
        # Fallback: seed placeholder names only, leave super_type_id NULL
        log.warning(
            "type_profiles.parquet not found at %s — seeding placeholder super_types, "
            "types.super_type_id will remain NULL",
            TYPE_PROFILES_PATH,
        )
        placeholder = [
            (0, "Rural Young"),
            (1, "Black-Belt Urban"),
            (2, "Hispanic Exurban"),
            (3, "Rural Evangelical"),
            (4, "Affluent College"),
        ]
        con.execute("DELETE FROM super_types")
        for st_id, name in placeholder:
            con.execute(
                "INSERT INTO super_types (super_type_id, display_name) VALUES (?, ?)",
                [st_id, name],
            )

    # 5. Verify
    n_types = con.execute("SELECT COUNT(*) FROM types").fetchone()[0]
    n_super = con.execute("SELECT COUNT(*) FROM super_types").fetchone()[0]
    null_super = con.execute(
        "SELECT COUNT(*) FROM types WHERE super_type_id IS NULL"
    ).fetchone()[0]
    log.info(
        "types: %d rows (%d with super_type_id=NULL); super_types: %d rows",
        n_types,
        null_super,
        n_super,
    )

    con.close()
    log.info("Migration complete: %s", db_path)


if __name__ == "__main__":
    migrate()
