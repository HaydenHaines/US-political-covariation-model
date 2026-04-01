"""Shared utilities for the db sub-modules.

Keeps DRY helpers in one place so ingest.py, transforms.py, and
build_database.py can all import without circular dependencies.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


def normalize_fips(series: pd.Series) -> pd.Series:
    """Normalize a FIPS column to a zero-padded 5-character string.

    Census FIPS codes are sometimes stored as integers (e.g. 12001) or short
    strings (e.g. "1001"). This ensures consistent "01001" format before any
    join or insert.
    """
    return series.astype(str).str.zfill(5)


def cycle_connection(
    con: duckdb.DuckDBPyConnection, db_path: Path, label: str
) -> duckdb.DuckDBPyConnection:
    """Close the current DuckDB connection and open a fresh one.

    DuckDB 1.x corrupts the glibc malloc heap after many large DataFrame
    inserts. Using ``del con`` (not ``con.close()``) avoids crashing on the
    corrupted heap during teardown.

    Returns a new open connection to the same database file.
    """
    del con
    gc.collect()
    new_con = duckdb.connect(str(db_path))
    log.info("Connection cycled (%s)", label)
    return new_con
