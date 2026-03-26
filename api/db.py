# api/db.py
"""DuckDB connection dependency for FastAPI."""
from __future__ import annotations

from typing import Generator

import duckdb
from fastapi import Request


def get_db(request: Request) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """FastAPI dependency: opens a fresh read-only DuckDB connection per request.

    DuckDB does not support concurrent queries on a single shared connection.
    Opening one connection per request avoids InvalidInputException races when
    the browser fires multiple API calls in parallel (e.g. via Promise.all).
    Read-only mode allows unlimited concurrent readers on the same file.

    In tests, app.state.db is set directly (in-memory DuckDB); that connection
    is returned as-is without closing (the test fixture owns its lifecycle).
    """
    # Test mode: fixture sets app.state.db directly with an in-memory connection
    if hasattr(request.app.state, "db"):
        yield request.app.state.db
        return

    # Production mode: open a fresh read-only connection per request
    db_path: str = request.app.state.db_path
    conn = duckdb.connect(db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()
