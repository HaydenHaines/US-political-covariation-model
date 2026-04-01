"""Data ingestion pipeline for wethervane.duckdb.

Each ``ingest_*`` function handles one logical domain. ``ingest_all()``
orchestrates them in sequence, cycling the DuckDB connection between
stages to avoid DuckDB 1.5.x heap corruption.
"""
from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.db._utils import normalize_fips as _normalize_fips, cycle_connection as _cycle_connection
from src.db.transforms import (
    build_community_assignments,
    build_counties,
    build_county_shifts,
    build_predictions,
    build_type_assignments,
    load_version_meta,
)
from src.description.generate_narratives import generate_all_narratives
from src.db.domains.model import ingest as ingest_model, create_tables as model_ddl
from src.db.domains.polling import ingest as ingest_polling, create_tables as polling_ddl

log = logging.getLogger(__name__)

POLL_INGEST_CYCLE = "2026"  # Election cycle to ingest polling data for

# Fixed demographic columns for the community_profiles table
_PROFILE_COLS = [
    "community_id", "n_counties", "pop_total",
    "pct_white_nh", "pct_black", "pct_asian", "pct_hispanic",
    "median_age", "median_hh_income", "pct_bachelors_plus",
    "pct_owner_occupied", "pct_wfh", "pct_management",
    "evangelical_share", "mainline_share", "catholic_share",
    "black_protestant_share", "congregations_per_1000",
    "religious_adherence_rate",
]


def insert_via_parquet(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    *,
    mode: str = "insert",
) -> None:
    """Insert a DataFrame into DuckDB via register/unregister to avoid heap corruption.

    ``mode='insert'`` appends to an existing table; ``mode='create'`` drops and
    recreates. Uses the view bridge instead of the implicit Python-DataFrame
    bridge that triggers DuckDB 1.5.x heap corruption on large transfers.
    """
    _VIEW = "_tmp_insert_view"
    con.register(_VIEW, df)
    try:
        if mode == "create":
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM {_VIEW}")
        else:
            con.execute(f"INSERT INTO {table} SELECT * FROM {_VIEW}")
    finally:
        con.unregister(_VIEW)


# _cycle_connection imported from src.db._utils



def _ingest_model_versions(
    con: duckdb.DuckDBPyConnection,
    version_metas: list[dict],
) -> None:
    """Ingest model version metadata from meta.yaml files."""
    con.execute("DELETE FROM model_versions")
    for m in version_metas:
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


def _ingest_shifts_and_assignments(
    con: duckdb.DuckDBPyConnection,
    shifts: pd.DataFrame | None,
    assignments: pd.DataFrame | None,
    type_df: pd.DataFrame | None,
    current_version_id: str,
) -> None:
    """Ingest county shifts, community assignments, and type assignments."""
    if shifts is not None:
        shift_rows = build_county_shifts(shifts, current_version_id)
        con.execute(f"DELETE FROM county_shifts WHERE version_id = '{current_version_id}'")
        # county_shifts has dynamic columns (one per election shift dim) so the
        # fixed CREATE TABLE schema doesn't match. Recreate the table from the
        # DataFrame via register/unregister to handle arbitrary column sets.
        insert_via_parquet(con, "county_shifts", shift_rows, mode="create")
        log.info("Ingested county_shifts: %d rows x %d cols", len(shift_rows), len(shift_rows.columns))

    if assignments is not None:
        ca_rows = build_community_assignments(assignments, current_version_id)
        con.execute(f"DELETE FROM community_assignments WHERE version_id = '{current_version_id}'")
        insert_via_parquet(con, "community_assignments", ca_rows)
        log.info(
            "Ingested community_assignments: %d rows, k=%d", len(ca_rows), ca_rows["k"].iloc[0]
        )

        ta_rows = build_type_assignments(type_df, assignments, current_version_id)
        con.execute(f"DELETE FROM type_assignments WHERE version_id = '{current_version_id}'")
        insert_via_parquet(con, "type_assignments", ta_rows)
        log.info("Ingested type_assignments: %d rows", len(ta_rows))


def _check_predictions_types_staleness(parquet_path: Path) -> bool:
    """Check if the types predictions parquet is from the current pipeline.

    The current pipeline (post-2026-03-27) writes a 'forecast_mode' column
    with values 'national' and 'local'.  The old pipeline wrote a single-mode
    output without that column, producing stale predictions where Atlanta metro
    polls were over-weighted (GA Senate showed D+20.8 vs D+5.8 from live engine).

    Returns True when the parquet is valid (has forecast_mode), False when stale.

    See: scripts/audit_ga_forecast.py, Issue #94 sub-task 4.
    """
    if not parquet_path.exists():
        return True  # Non-existent → skip, not stale

    # Read only the schema (no row data) to check for the required column.
    cols = pd.read_parquet(parquet_path, columns=[]).columns.tolist()

    if "forecast_mode" not in cols:
        log.warning(
            "STALE PREDICTIONS DETECTED: %s lacks 'forecast_mode' column. "
            "This parquet was generated by the pre-forecast-engine pipeline. "
            "Regenerate with: uv run python -m src.prediction.predict_2026_types",
            parquet_path,
        )
        return False
    return True


def _ingest_predictions(
    con: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame | None,
    paths: dict[str, Path],
    current_version_id: str,
) -> None:
    """Ingest all prediction variants: base, HAC, and types."""
    if predictions is not None:
        pred_rows = build_predictions(predictions, current_version_id)
        con.execute(f"DELETE FROM predictions WHERE version_id = '{current_version_id}'")
        insert_via_parquet(con, "predictions", pred_rows)
        log.info("Ingested predictions: %d rows", len(pred_rows))
    else:
        log.warning("No predictions file found; skipping predictions table")

    # HAC predictions — fill in races not covered by the primary predictions
    if paths["predictions_hac"].exists():
        pred_hac = pd.read_parquet(paths["predictions_hac"])
        pred_hac["county_fips"] = pred_hac["county_fips"].pipe(_normalize_fips)
        pred_hac_rows = build_predictions(pred_hac, current_version_id)
        existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
        new_rows = pred_hac_rows[~pred_hac_rows["race"].isin(existing_races)]
        if len(new_rows):
            insert_via_parquet(con, "predictions", new_rows)
            log.info("Ingested HAC predictions: %d rows", len(new_rows))
    else:
        log.info("No HAC predictions file found; skipping")

    # Types predictions take precedence over base predictions for overlapping races.
    # Staleness check: warn if parquet lacks forecast_mode (pre-forecast-engine artifact).
    if paths["predictions_types"].exists():
        _check_predictions_types_staleness(paths["predictions_types"])
        pred_types = pd.read_parquet(paths["predictions_types"])
        pred_types["county_fips"] = pred_types["county_fips"].pipe(_normalize_fips)
        pred_types_rows = build_predictions(pred_types, current_version_id)
        existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
        overlap_races = set(pred_types_rows["race"].unique()) & existing_races
        if overlap_races:
            for r in overlap_races:
                overlap_fips = pred_types_rows.loc[pred_types_rows["race"] == r, "county_fips"].tolist()
                placeholders = ", ".join(["?"] * len(overlap_fips))
                con.execute(
                    f"DELETE FROM predictions WHERE race = ? AND county_fips IN ({placeholders}) AND version_id = ?",
                    [r] + overlap_fips + [current_version_id],
                )
        insert_via_parquet(con, "predictions", pred_types_rows)
        log.info("Ingested types predictions: %d rows", len(pred_types_rows))
    else:
        log.info("No types predictions file found; skipping")



def _ingest_types_and_assignments(
    con: duckdb.DuckDBPyConnection,
    paths: dict[str, Path],
) -> None:
    """Ingest type profiles, county/tract type assignments, and super-types."""
    # Type profiles → types table
    if paths["type_profiles"].exists():
        tp_df = pd.read_parquet(paths["type_profiles"])
        # Add super_type_id from county_type_assignments if available
        if paths["county_type_assignments"].exists() and "super_type_id" not in tp_df.columns:
            cta_tmp = pd.read_parquet(paths["county_type_assignments"])
            if "super_type" in cta_tmp.columns and "dominant_type" in cta_tmp.columns:
                type_to_super = cta_tmp.groupby("dominant_type")["super_type"].first()
                tp_df["super_type_id"] = tp_df["type_id"].map(type_to_super)
                log.info("Added super_type_id to types table from county_type_assignments")
        if "display_name" not in tp_df.columns:
            tp_df["display_name"] = tp_df["type_id"].apply(lambda x: f"Type {x}")
        # Generate and attach narratives from demographic z-scores
        log.info("Generating type narratives from demographic z-scores")
        try:
            narratives = generate_all_narratives(str(paths["type_profiles"]))
            tp_df["narrative"] = tp_df["type_id"].map(narratives)
            log.info("Attached narratives to %d types", tp_df["narrative"].notna().sum())
        except Exception as exc:
            log.warning("Narrative generation failed (%s); types table will lack narrative column", exc)
        insert_via_parquet(con, "types", tp_df, mode="create")
        log.info("Ingested types: %d rows", len(tp_df))
    else:
        log.info("No type_profiles.parquet found; skipping types table")

    # County type assignments — use DuckDB's native parquet reader directly
    # to avoid the Python bridge entirely.
    if paths["county_type_assignments"].exists():
        p = str(paths["county_type_assignments"])
        con.execute("DROP TABLE IF EXISTS county_type_assignments")
        con.execute(f"CREATE TABLE county_type_assignments AS SELECT * FROM read_parquet('{p}')")
        n_cta = con.execute("SELECT COUNT(*) FROM county_type_assignments").fetchone()[0]
        log.info("Ingested county_type_assignments: %d rows", n_cta)
    else:
        log.info("No county_type_assignments_full.parquet found; skipping")

    # Tract type assignments — dedup on GEOID because the source parquet has
    # 112,257 rows but only 81,129 unique GEOIDs (some tracts appear in
    # multiple state runs).
    if paths["tract_type_assignments"].exists():
        tta_df = pd.read_parquet(
            paths["tract_type_assignments"],
            columns=["GEOID", "dominant_type", "super_type"],
        )
        tta_df = tta_df.drop_duplicates(subset="GEOID")
        tta_df = tta_df.rename(columns={"GEOID": "tract_geoid"})
        tta_df["dominant_type"] = tta_df["dominant_type"].astype("int32")
        tta_df["super_type"] = tta_df["super_type"].astype("int32")
        con.execute("DROP TABLE IF EXISTS tract_type_assignments")
        con.execute(
            "CREATE TABLE tract_type_assignments "
            "(tract_geoid VARCHAR PRIMARY KEY, dominant_type INTEGER, super_type INTEGER)"
        )
        con.register("_tta_view", tta_df)
        con.execute("INSERT INTO tract_type_assignments SELECT * FROM _tta_view")
        con.unregister("_tta_view")
        n_tta = con.execute("SELECT COUNT(*) FROM tract_type_assignments").fetchone()[0]
        log.info("Ingested tract_type_assignments: %d rows (from %d source rows)", n_tta, len(tta_df) + (112257 - 81129))
    else:
        log.info("No national_tract_assignments.parquet found; skipping tract_type_assignments")

    # Super-types
    if paths["super_types"].exists():
        st_df = pd.read_parquet(paths["super_types"])
        con.execute("DELETE FROM super_types")
        insert_via_parquet(con, "super_types", st_df)
        log.info("Ingested super_types: %d rows", len(st_df))
    else:
        log.info("No super_types.parquet found; skipping")


def _load_source_data(paths: dict[str, Path]) -> dict:
    """Load parquet source files into DataFrames, returning a dict of results."""
    result: dict = {"shifts": None, "assignments": None, "type_df": None, "predictions": None}
    if paths["shifts"].exists():
        log.info("Loading shifts from %s", paths["shifts"])
        result["shifts"] = pd.read_parquet(paths["shifts"])
        result["shifts"]["county_fips"] = result["shifts"]["county_fips"].pipe(_normalize_fips)
    else:
        log.warning("Shifts file not found: %s; skipping core ingestion", paths["shifts"])
    if paths["assignments"].exists():
        log.info("Loading community assignments from %s", paths["assignments"])
        assignments = pd.read_parquet(paths["assignments"])
        assignments["county_fips"] = assignments["county_fips"].pipe(_normalize_fips)
        if "community_id" not in assignments.columns and "community" in assignments.columns:
            assignments = assignments.rename(columns={"community": "community_id"})
        result["assignments"] = assignments
    else:
        log.warning("Assignments file not found: %s; skipping community ingestion", paths["assignments"])
    if paths["stub"].exists():
        log.info("Loading type assignments stub from %s", paths["stub"])
        result["type_df"] = pd.read_parquet(paths["stub"])
    if paths["predictions"].exists():
        log.info("Loading predictions from %s", paths["predictions"])
        result["predictions"] = pd.read_parquet(paths["predictions"])
        result["predictions"]["county_fips"] = result["predictions"]["county_fips"].pipe(_normalize_fips)
    result["version_metas"] = load_version_meta(paths["versions_dir"])
    return result


def ingest_all(
    con: duckdb.DuckDBPyConnection,
    db_path: Path,
    paths: dict[str, Path],
    project_root: Path,
) -> None:
    """Load all parquet artifacts and insert them into DuckDB.

    Connection is cycled at several checkpoints to avoid DuckDB 1.5.x heap corruption.
    """
    data = _load_source_data(paths)
    shifts, assignments, type_df = data["shifts"], data["assignments"], data["type_df"]
    predictions, version_metas = data["predictions"], data["version_metas"]

    # Counties, races, versions, shifts, assignments
    if shifts is not None:
        counties_df = build_counties(shifts, crosswalk_path=paths["crosswalk"])
        con.execute("DELETE FROM counties")
        insert_via_parquet(con, "counties", counties_df)
        log.info("Ingested %d counties", len(counties_df))

    from src.assembly.define_races import load_races
    con.execute("DELETE FROM races")
    races = load_races(2026)
    for r in races:
        con.execute("INSERT INTO races VALUES (?, ?, ?, ?, ?)",
                     [r.race_id, r.race_type, r.state, r.year, r.district])
    log.info("Ingested races: %d rows", len(races))
    _ingest_model_versions(con, version_metas)

    # Identify current version for shift/assignment/prediction ingestion
    current_meta = next(
        (m for m in version_metas if m.get("role") == "current"),
        version_metas[-1] if version_metas else {"version": "unknown"},
    )
    current_version_id = current_meta.get("version_id") or current_meta.get("version", "unknown")
    log.info("Using version_id '%s' for shift/assignment ingestion", current_version_id)
    _ingest_shifts_and_assignments(con, shifts, assignments, type_df, current_version_id)
    con = _cycle_connection(con, db_path, "post-shifts")

    _ingest_predictions(con, predictions, paths, current_version_id)
    con = _cycle_connection(con, db_path, "post-predictions")
    # Sigma + profiles + demographics
    if paths["sigma"].exists():
        sigma_df = pd.read_parquet(paths["sigma"])
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
        insert_via_parquet(con, "community_sigma", sigma_long)
        log.info("Ingested community_sigma: %d cells", len(sigma_long))
    else:
        log.info("No county_community_sigma.parquet found; skipping sigma table")
    con = _cycle_connection(con, db_path, "mid-build")

    # Community profiles + county demographics
    if paths["community_profiles"].exists():
        cp_df = pd.read_parquet(paths["community_profiles"])
        cp_insert = cp_df[[c for c in _PROFILE_COLS if c in cp_df.columns]].copy()
        con.execute("DELETE FROM community_profiles")
        insert_via_parquet(con, "community_profiles", cp_insert)
        log.info("Ingested community_profiles: %d rows", len(cp_insert))
    else:
        log.info("No community_profiles.parquet found; skipping")
    if paths["county_acs"].exists():
        cd_df = pd.read_parquet(paths["county_acs"])
        cd_df["county_fips"] = cd_df["county_fips"].pipe(_normalize_fips)
        insert_via_parquet(con, "county_demographics", cd_df, mode="create")
        log.info("Ingested county_demographics: %d rows (%d columns)", len(cd_df), len(cd_df.columns))
    else:
        log.info("No county_acs_features.parquet found; skipping")
    con = _cycle_connection(con, db_path, "post-demographics")
    _ingest_types_and_assignments(con, paths)
    con = _cycle_connection(con, db_path, "pre-domain")

    # Domain modules (model scores, polling)
    model_ddl(con)
    polling_ddl(con)
    ingest_model(con, current_version_id, project_root)
    ingest_polling(con, POLL_INGEST_CYCLE, project_root)
    con = _cycle_connection(con, db_path, "post-domain")
    # Demographics interpolated
    if paths["demographics_interpolated"].exists():
        di_df = pd.read_parquet(paths["demographics_interpolated"])
        di_df["county_fips"] = di_df["county_fips"].pipe(_normalize_fips)
        insert_via_parquet(con, "demographics_interpolated", di_df, mode="create")
        log.info("Ingested demographics_interpolated: %d rows", len(di_df))
    else:
        log.info("No demographics_interpolated.parquet found; skipping")
    del con  # use del (not close()) to avoid crash on corrupted heap
    gc.collect()
