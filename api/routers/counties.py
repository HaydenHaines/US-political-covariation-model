from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.models import CountyRow

router = APIRouter(tags=["counties"])


def _has_table(db: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return result is not None and result[0] > 0


@router.get("/counties", response_model=list[CountyRow])
def list_counties(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    version_id = request.app.state.version_id

    has_types = _has_table(db, "county_type_assignments")

    if has_types:
        rows = db.execute(
            """
            SELECT c.county_fips, c.state_abbr, ca.community_id,
                   cta.dominant_type, cta.super_type
            FROM counties c
            JOIN community_assignments ca
                ON c.county_fips = ca.county_fips
                AND ca.version_id = ?
            LEFT JOIN county_type_assignments cta
                ON c.county_fips = cta.county_fips
            ORDER BY c.county_fips
            """,
            [version_id],
        ).fetchdf()
    else:
        rows = db.execute(
            """
            SELECT c.county_fips, c.state_abbr, ca.community_id
            FROM counties c
            JOIN community_assignments ca
                ON c.county_fips = ca.county_fips
                AND ca.version_id = ?
            ORDER BY c.county_fips
            """,
            [version_id],
        ).fetchdf()

    results = []
    for _, row in rows.iterrows():
        dominant_type = None
        super_type = None
        if has_types and "dominant_type" in row.index:
            import pandas as pd

            dominant_type = None if pd.isna(row["dominant_type"]) else int(row["dominant_type"])
            super_type = None if pd.isna(row["super_type"]) else int(row["super_type"])
        results.append(
            CountyRow(
                county_fips=row["county_fips"],
                state_abbr=row["state_abbr"],
                community_id=int(row["community_id"]),
                dominant_type=dominant_type,
                super_type=super_type,
            )
        )
    return results
