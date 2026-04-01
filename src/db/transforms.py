"""Data transform functions for building DuckDB tables.

Each function takes raw DataFrames (loaded from parquet) and returns a
DataFrame shaped for DuckDB insertion. No I/O happens here — just
column selection, renaming, and enrichment.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.core import config as _cfg
from src.db._utils import normalize_fips as _normalize_fips

log = logging.getLogger(__name__)

# State FIPS → abbreviation mapping: sourced from config/model.yaml (all 50+DC).
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR


def load_version_meta(versions_dir: Path) -> list[dict]:
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


_DEFAULT_CROSSWALK = object()  # sentinel: use caller-supplied crosswalk


def build_counties(
    shifts: pd.DataFrame,
    crosswalk_path: Path | None = _DEFAULT_CROSSWALK,  # type: ignore[assignment]
    pres_2024_path: Path | None = None,
) -> pd.DataFrame:
    """Derive the counties table from shift FIPS column, optionally joining county names
    and 2024 presidential vote totals (used for population-weighted state aggregation).

    Args:
        shifts: DataFrame with a county_fips column.
        crosswalk_path: Path to fips_county_crosswalk.csv.  Pass ``None`` to
            skip name lookup (county_name will be all-NULL).  Omit (or pass the
            sentinel ``_DEFAULT_CROSSWALK``) to use the module-level constant.
        pres_2024_path: Path to medsl_county_presidential_2024.parquet.  When
            provided (and the file exists), ``total_votes_2024`` is populated
            from ``pres_total_2024``.  Falls back to NULL when not available.
    """
    # Lazy import to resolve module-level path constants. These are only needed
    # when callers omit crosswalk_path / pres_2024_path (the build pipeline
    # always passes explicit paths; tests pass None or explicit paths).
    if crosswalk_path is _DEFAULT_CROSSWALK:
        from src.db.build_database import CROSSWALK_PATH
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

    # Join 2024 presidential vote totals for population-weighted state aggregation.
    # total_votes_2024 is NULL when data is unavailable; the API falls back to
    # uniform weighting when the column is missing or all-NULL.
    if pres_2024_path is None:
        from src.db.build_database import PRES_2024_PATH
        pres_2024_path = PRES_2024_PATH
    _pres_path = pres_2024_path
    if _pres_path is not None and Path(_pres_path).exists():
        pres_df = pd.read_parquet(_pres_path)
        pres_df["county_fips"] = pres_df["county_fips"].pipe(_normalize_fips)
        df = df.merge(
            pres_df[["county_fips", "pres_total_2024"]].rename(
                columns={"pres_total_2024": "total_votes_2024"}
            ),
            on="county_fips",
            how="left",
        )
        # Cast to nullable int (some counties may be missing from the parquet)
        df["total_votes_2024"] = pd.to_numeric(df["total_votes_2024"], errors="coerce").astype(
            "Int64"
        )
        n_matched = df["total_votes_2024"].notna().sum()
        log.info("Joined total_votes_2024 for %d / %d counties", n_matched, len(df))
    else:
        df["total_votes_2024"] = None
        log.warning(
            "2024 presidential parquet not found at %s; total_votes_2024 will be NULL",
            _pres_path,
        )

    return df[["county_fips", "state_abbr", "state_fips", "county_name", "total_votes_2024"]]


def build_county_shifts(shifts: pd.DataFrame, version_id: str) -> pd.DataFrame:
    """Add version_id column to the wide shifts DataFrame."""
    df = shifts.copy()
    df["version_id"] = version_id
    return df


def build_community_assignments(
    assignments: pd.DataFrame, version_id: str
) -> pd.DataFrame:
    """Normalize community assignments for DuckDB ingestion."""
    df = assignments[["county_fips", "community_id"]].copy()
    k = int(df["community_id"].nunique())
    df["k"] = k
    df["version_id"] = version_id
    return df[["county_fips", "community_id", "k", "version_id"]]


def build_type_assignments(
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


def build_predictions(preds: pd.DataFrame, version_id: str) -> pd.DataFrame:
    """Shape predictions DataFrame for DuckDB insertion."""
    df = preds.copy()
    df["version_id"] = version_id
    cols = [
        "county_fips", "race", "version_id", "forecast_mode",
        "pred_dem_share", "pred_std", "pred_lo90", "pred_hi90",
        "state_pred", "poll_avg",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "local" if c == "forecast_mode" else None
    return df[cols]
