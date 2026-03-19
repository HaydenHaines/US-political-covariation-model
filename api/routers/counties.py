from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.models import CountyRow

router = APIRouter(tags=["counties"])


@router.get("/counties", response_model=list[CountyRow])
def list_counties(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    version_id = request.app.state.version_id

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

    return [
        CountyRow(
            county_fips=row["county_fips"],
            state_abbr=row["state_abbr"],
            community_id=int(row["community_id"]),
        )
        for _, row in rows.iterrows()
    ]
