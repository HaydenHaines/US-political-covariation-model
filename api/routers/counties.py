from __future__ import annotations

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import CountyDetail, CountyRow, SiblingCounty

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

    has_predictions = _has_table(db, "predictions")

    if has_types:
        if has_predictions:
            rows = db.execute(
                """
                SELECT c.county_fips, c.state_abbr,
                       ca.community_id,
                       cta.dominant_type, cta.super_type,
                       AVG(p.pred_dem_share) AS pred_dem_share
                FROM counties c
                JOIN county_type_assignments cta
                    ON c.county_fips = cta.county_fips
                LEFT JOIN community_assignments ca
                    ON c.county_fips = ca.county_fips
                    AND ca.version_id = ?
                LEFT JOIN predictions p
                    ON c.county_fips = p.county_fips
                GROUP BY c.county_fips, c.state_abbr, ca.community_id,
                         cta.dominant_type, cta.super_type
                ORDER BY c.county_fips
                """,
                [version_id],
            ).fetchdf()
        else:
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

    results = []
    for _, row in rows.iterrows():
        community_id = None
        dominant_type = None
        super_type = None
        pred_dem_share = None
        if "community_id" in row.index and not pd.isna(row["community_id"]):
            community_id = int(row["community_id"])
        if "dominant_type" in row.index and not pd.isna(row["dominant_type"]):
            dominant_type = int(row["dominant_type"])
        if "super_type" in row.index and not pd.isna(row["super_type"]):
            super_type = int(row["super_type"])
        if "pred_dem_share" in row.index and not pd.isna(row["pred_dem_share"]):
            pred_dem_share = float(row["pred_dem_share"])
        results.append(
            CountyRow(
                county_fips=row["county_fips"],
                state_abbr=row["state_abbr"],
                community_id=community_id,
                dominant_type=dominant_type,
                super_type=super_type,
                pred_dem_share=pred_dem_share,
            )
        )
    return results


@router.get("/counties/{fips}", response_model=CountyDetail)
def get_county_detail(
    fips: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Return detailed profile for a single county (SEO county page)."""
    # ── Core county + type info ───────────────────────────────────────────
    row = db.execute(
        """
        SELECT c.county_fips, c.county_name, c.state_abbr,
               cta.dominant_type, cta.super_type,
               t.display_name  AS type_display_name,
               st.display_name AS super_type_display_name,
               t.narrative
        FROM counties c
        JOIN county_type_assignments cta ON c.county_fips = cta.county_fips
        JOIN types t ON cta.dominant_type = t.type_id
        JOIN super_types st ON t.super_type_id = st.super_type_id
        WHERE c.county_fips = ?
        LIMIT 1
        """,
        [fips],
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"County {fips} not found")

    (
        county_fips, county_name, state_abbr,
        dominant_type, super_type,
        type_display_name, super_type_display_name,
        narrative,
    ) = row

    # ── Baseline prediction (AVG across races) ────────────────────────────
    pred_row = db.execute(
        "SELECT AVG(pred_dem_share) FROM predictions WHERE county_fips = ?",
        [fips],
    ).fetchone()
    pred_dem_share = float(pred_row[0]) if pred_row and pred_row[0] is not None else None

    # ── Demographics ──────────────────────────────────────────────────────
    demo_row = db.execute(
        "SELECT * FROM county_demographics WHERE county_fips = ?",
        [fips],
    ).fetchone()
    demographics: dict[str, float] = {}
    if demo_row is not None:
        demo_cols = [
            desc[0]
            for desc in db.execute("DESCRIBE county_demographics").fetchall()
        ]
        for col, val in zip(demo_cols, demo_row):
            if col == "county_fips":
                continue
            if val is not None:
                demographics[col] = float(val)

    # ── Sibling counties (same dominant_type, limit 20) ───────────────────
    siblings = db.execute(
        """
        SELECT c.county_fips, c.county_name, c.state_abbr
        FROM county_type_assignments cta
        JOIN counties c ON cta.county_fips = c.county_fips
        WHERE cta.dominant_type = ? AND cta.county_fips != ?
        ORDER BY c.state_abbr, c.county_name
        LIMIT 20
        """,
        [dominant_type, fips],
    ).fetchall()
    sibling_counties = [
        SiblingCounty(county_fips=s[0], county_name=s[1], state_abbr=s[2])
        for s in siblings
    ]

    return CountyDetail(
        county_fips=county_fips,
        county_name=county_name,
        state_abbr=state_abbr,
        dominant_type=int(dominant_type),
        super_type=int(super_type),
        type_display_name=type_display_name,
        super_type_display_name=super_type_display_name,
        narrative=narrative,
        pred_dem_share=pred_dem_share,
        demographics=demographics,
        sibling_counties=sibling_counties,
    )
