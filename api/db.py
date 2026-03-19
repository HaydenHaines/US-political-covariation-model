# api/db.py
"""DuckDB connection dependency for FastAPI."""
from __future__ import annotations

import duckdb
from fastapi import Request


def get_db(request: Request) -> duckdb.DuckDBPyConnection:
    """FastAPI dependency: returns the shared read-only DuckDB connection."""
    return request.app.state.db
