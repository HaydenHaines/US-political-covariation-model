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
    has_communities = _has_table(db, "community_assignments")

    if has_types:
        rows = db.execute(
            """
            SELECT c.county_fips, c.state_abbr,
                   ca.community_id,
                   cta.dominant_type, cta.super_type
            FROM counties c
            JOIN county_type_assignments cta
                ON c.county_fips = cta.county_fips
            LEFT JOIN community_assignments ca
                ON c.county_fips = ca.county_fips
                AND ca.version_id = ?
            ORDER BY c.county_fips
            """,
            [version_id],
        ).fetchdf()
    elif has_communities:
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
    else:
        rows = db.execute(
            """
            SELECT county_fips, state_abbr
            FROM counties
            ORDER BY county_fips
            """,
        ).fetchdf()

    import pandas as pd

    results = []
    for _, row in rows.iterrows():
        community_id = None
        dominant_type = None
        super_type = None
        if "community_id" in row.index and not pd.isna(row["community_id"]):
            community_id = int(row["community_id"])
        if "dominant_type" in row.index and not pd.isna(row["dominant_type"]):
            dominant_type = int(row["dominant_type"])
        if "super_type" in row.index and not pd.isna(row["super_type"]):
            super_type = int(row["super_type"])
        results.append(
            CountyRow(
                county_fips=row["county_fips"],
                state_abbr=row["state_abbr"],
                community_id=community_id,
                dominant_type=dominant_type,
                super_type=super_type,
            )
        )
    return results
