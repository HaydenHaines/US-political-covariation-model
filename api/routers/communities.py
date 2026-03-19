from __future__ import annotations

import pandas as pd
import duckdb
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import CommunityDetail, CommunitySummary, CountyInCommunity

router = APIRouter(tags=["communities"])


def _make_display_name(community_id: int, states: list[str]) -> str:
    return f"Community {community_id} ({'/'.join(sorted(states))})"


@router.get("/communities", response_model=list[CommunitySummary])
def list_communities(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    version_id = request.app.state.version_id

    rows = db.execute(
        """
        SELECT
            ca.community_id,
            COUNT(DISTINCT ca.county_fips) AS n_counties,
            ARRAY_AGG(DISTINCT c.state_abbr) AS states,
            ta.dominant_type_id,
            AVG(p.pred_dem_share) AS mean_pred_dem_share
        FROM community_assignments ca
        JOIN counties c ON ca.county_fips = c.county_fips
        LEFT JOIN type_assignments ta
            ON ca.community_id = ta.community_id
            AND ca.k = ta.k
            AND ca.version_id = ta.version_id
        LEFT JOIN predictions p
            ON ca.county_fips = p.county_fips
            AND p.version_id = ca.version_id
        WHERE ca.version_id = ?
        GROUP BY ca.community_id, ta.dominant_type_id
        ORDER BY ca.community_id
        """,
        [version_id],
    ).fetchdf()

    results = []
    for _, row in rows.iterrows():
        raw_states = row["states"]
        states = sorted(set(raw_states)) if raw_states is not None else []
        type_id = None if pd.isna(row["dominant_type_id"]) else int(row["dominant_type_id"])
        mean_share = None if pd.isna(row["mean_pred_dem_share"]) else float(row["mean_pred_dem_share"])
        results.append(
            CommunitySummary(
                community_id=int(row["community_id"]),
                display_name=_make_display_name(int(row["community_id"]), states),
                n_counties=int(row["n_counties"]),
                states=states,
                dominant_type_id=type_id,
                mean_pred_dem_share=mean_share,
            )
        )
    return results


@router.get("/communities/{community_id}", response_model=CommunityDetail)
def get_community(
    community_id: int,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    version_id = request.app.state.version_id

    # Check exists
    exists = db.execute(
        "SELECT 1 FROM community_assignments WHERE community_id = ? AND version_id = ? LIMIT 1",
        [community_id, version_id],
    ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"Community {community_id} not found")

    # Counties in this community with predictions
    county_rows = db.execute(
        """
        SELECT
            ca.county_fips,
            c.county_name,
            c.state_abbr,
            p.pred_dem_share
        FROM community_assignments ca
        JOIN counties c ON ca.county_fips = c.county_fips
        LEFT JOIN predictions p
            ON ca.county_fips = p.county_fips
            AND p.version_id = ca.version_id
        WHERE ca.community_id = ? AND ca.version_id = ?
        ORDER BY c.state_abbr, ca.county_fips
        """,
        [community_id, version_id],
    ).fetchdf()

    states = sorted(county_rows["state_abbr"].unique().tolist())
    counties = [
        CountyInCommunity(
            county_fips=row["county_fips"],
            county_name=row["county_name"] if row["county_name"] else None,
            state_abbr=row["state_abbr"],
            pred_dem_share=None if pd.isna(row["pred_dem_share"]) else float(row["pred_dem_share"]),
        )
        for _, row in county_rows.iterrows()
    ]

    # Shift profile: mean of each shift column across member counties
    shift_cols_row = db.execute(
        "SELECT * FROM county_shifts LIMIT 0"
    ).fetchdf()
    shift_col_names = [
        c for c in shift_cols_row.columns
        if c not in ("county_fips", "version_id")
    ]

    shift_profile: dict[str, float] = {}
    if shift_col_names:
        member_fips = county_rows["county_fips"].tolist()
        placeholders = ", ".join("?" * len(member_fips))
        shift_rows = db.execute(
            f"SELECT * FROM county_shifts WHERE county_fips IN ({placeholders}) AND version_id = ?",
            member_fips + [version_id],
        ).fetchdf()
        for col in shift_col_names:
            if col in shift_rows.columns:
                val = shift_rows[col].mean()
                shift_profile[col] = float(val) if not pd.isna(val) else 0.0

    # Dominant type
    type_row = db.execute(
        "SELECT dominant_type_id FROM type_assignments WHERE community_id = ? AND version_id = ? LIMIT 1",
        [community_id, version_id],
    ).fetchone()
    dominant_type_id = int(type_row[0]) if type_row and type_row[0] is not None else None

    return CommunityDetail(
        community_id=community_id,
        display_name=_make_display_name(community_id, states),
        n_counties=len(counties),
        states=states,
        dominant_type_id=dominant_type_id,
        counties=counties,
        shift_profile=shift_profile,
    )
