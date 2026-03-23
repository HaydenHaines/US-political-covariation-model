from __future__ import annotations

import pandas as pd
import duckdb
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import (
    CommunityDemographics,
    CommunityDetail,
    CommunitySummary,
    CountyInCommunity,
    SuperTypeSummary,
    TypeCounty,
    TypeDetail,
    TypeSummary,
)

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
            f"SELECT * FROM county_shifts WHERE county_fips IN ({placeholders})",
            member_fips,
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

    # Demographics profile from community_profiles table
    demographics: CommunityDemographics | None = None
    try:
        demo_row = db.execute(
            """
            SELECT
                pop_total, pct_white_nh, pct_black, pct_asian, pct_hispanic,
                median_age, median_hh_income, pct_bachelors_plus,
                pct_owner_occupied, pct_wfh, pct_management,
                evangelical_share, mainline_share, catholic_share,
                black_protestant_share, congregations_per_1000,
                religious_adherence_rate
            FROM community_profiles
            WHERE community_id = ?
            LIMIT 1
            """,
            [community_id],
        ).fetchdf()
        if not demo_row.empty:
            r = demo_row.iloc[0]

            def _f(v) -> float | None:
                return None if pd.isna(v) else float(v)

            demographics = CommunityDemographics(
                pop_total=_f(r["pop_total"]),
                pct_white_nh=_f(r["pct_white_nh"]),
                pct_black=_f(r["pct_black"]),
                pct_asian=_f(r["pct_asian"]),
                pct_hispanic=_f(r["pct_hispanic"]),
                median_age=_f(r["median_age"]),
                median_hh_income=_f(r["median_hh_income"]),
                pct_bachelors_plus=_f(r["pct_bachelors_plus"]),
                pct_owner_occupied=_f(r["pct_owner_occupied"]),
                pct_wfh=_f(r["pct_wfh"]),
                pct_management=_f(r["pct_management"]),
                evangelical_share=_f(r["evangelical_share"]),
                mainline_share=_f(r["mainline_share"]),
                catholic_share=_f(r["catholic_share"]),
                black_protestant_share=_f(r["black_protestant_share"]),
                congregations_per_1000=_f(r["congregations_per_1000"]),
                religious_adherence_rate=_f(r["religious_adherence_rate"]),
            )
    except Exception:
        # community_profiles table may not exist in test DBs — graceful fallback
        pass

    return CommunityDetail(
        community_id=community_id,
        display_name=_make_display_name(community_id, states),
        n_counties=len(counties),
        states=states,
        dominant_type_id=dominant_type_id,
        counties=counties,
        shift_profile=shift_profile,
        demographics=demographics,
    )


# ── Type endpoints ───────────────────────────────────────────────────────────

def _has_table(db: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Check if a table exists in DuckDB."""
    result = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return result is not None and result[0] > 0


@router.get("/types", response_model=list[TypeSummary])
def list_types(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    """List all electoral types with summary info."""
    if not _has_table(db, "types"):
        return []

    rows = db.execute(
        """
        SELECT
            t.type_id,
            t.super_type_id,
            t.display_name,
            COUNT(DISTINCT cta.county_fips) AS n_counties,
            NULL AS mean_pred_dem_share
        FROM types t
        LEFT JOIN county_type_assignments cta
            ON t.type_id = cta.dominant_type
        GROUP BY t.type_id, t.super_type_id, t.display_name
        ORDER BY t.type_id
        """,
    ).fetchdf()

    results = []
    for _, row in rows.iterrows():
        mean_share = None if pd.isna(row["mean_pred_dem_share"]) else float(row["mean_pred_dem_share"])
        results.append(
            TypeSummary(
                type_id=int(row["type_id"]),
                super_type_id=int(row["super_type_id"]),
                display_name=str(row["display_name"]),
                n_counties=int(row["n_counties"]),
                mean_pred_dem_share=mean_share,
            )
        )
    return results


@router.get("/types/{type_id}", response_model=TypeDetail)
def get_type(
    type_id: int,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Get detail for a single electoral type."""
    if not _has_table(db, "types"):
        raise HTTPException(status_code=404, detail="Type data not available")

    # Get type metadata
    type_row = db.execute(
        "SELECT type_id, super_type_id, display_name FROM types WHERE type_id = ? LIMIT 1",
        [type_id],
    ).fetchone()
    if not type_row:
        raise HTTPException(status_code=404, detail=f"Type {type_id} not found")

    tid, super_type_id, display_name = type_row

    # Counties assigned to this type (dominant), joined with names
    county_rows = db.execute(
        """
        SELECT cta.county_fips, c.county_name, c.state_abbr
        FROM county_type_assignments cta
        LEFT JOIN counties c ON cta.county_fips = c.county_fips
        WHERE cta.dominant_type = ?
        ORDER BY c.state_abbr, cta.county_fips
        """,
        [type_id],
    ).fetchdf()

    counties: list[TypeCounty] = [
        TypeCounty(
            county_fips=row["county_fips"],
            county_name=row["county_name"] if row["county_name"] else None,
            state_abbr=row["state_abbr"],
        )
        for _, row in county_rows.iterrows()
    ] if not county_rows.empty else []

    # Keep bare FIPS list for shift profile lookup
    county_fips_list = [c.county_fips for c in counties]

    # Demographic profile from the types table itself (which contains all profile data)
    demographics: dict[str, float] = {}
    try:
        demo_row = db.execute(
            "SELECT * FROM types WHERE type_id = ? LIMIT 1",
            [type_id],
        ).fetchdf()
        if not demo_row.empty:
            r = demo_row.iloc[0]
            skip_cols = {"type_id", "super_type_id", "display_name", "n_counties"}
            for col in demo_row.columns:
                if col not in skip_cols:
                    val = r[col]
                    if not pd.isna(val):
                        demographics[col] = float(val)
    except Exception:
        pass

    # Shift profile: mean shifts across member counties
    shift_profile: dict[str, float] | None = None
    if county_fips_list:
        try:
            shift_cols_row = db.execute("SELECT * FROM county_shifts LIMIT 0").fetchdf()
            shift_col_names = [
                c for c in shift_cols_row.columns if c not in ("county_fips", "version_id")
            ]
            if shift_col_names:
                placeholders = ", ".join("?" * len(county_fips_list))
                shift_rows = db.execute(
                    f"SELECT * FROM county_shifts WHERE county_fips IN ({placeholders})",
                    county_fips_list,
                ).fetchdf()
                shift_profile = {}
                for col in shift_col_names:
                    if col in shift_rows.columns:
                        val = shift_rows[col].mean()
                        shift_profile[col] = float(val) if not pd.isna(val) else 0.0
        except Exception:
            pass

    return TypeDetail(
        type_id=int(tid),
        super_type_id=int(super_type_id),
        display_name=str(display_name),
        n_counties=len(counties),
        counties=counties,
        demographics=demographics,
        shift_profile=shift_profile,
    )



@router.get("/super-types", response_model=list[SuperTypeSummary])
def list_super_types(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    """List super-types with member type IDs."""
    if not _has_table(db, "types"):
        return []

    # Super-type names come from the super_types table
    if _has_table(db, "super_types"):
        rows = db.execute(
            """
            SELECT
                st.super_type_id,
                st.display_name,
                ARRAY_AGG(DISTINCT t.type_id ORDER BY t.type_id) AS member_type_ids,
                COUNT(DISTINCT cta.county_fips) AS n_counties
            FROM super_types st
            LEFT JOIN types t ON t.super_type_id = st.super_type_id
            LEFT JOIN county_type_assignments cta ON t.type_id = cta.dominant_type
            GROUP BY st.super_type_id, st.display_name
            ORDER BY st.super_type_id
            """,
        ).fetchdf()
    else:
        rows = db.execute(
            """
            SELECT
                t.super_type_id,
                MIN(t.display_name) AS display_name,
                ARRAY_AGG(DISTINCT t.type_id ORDER BY t.type_id) AS member_type_ids,
                COUNT(DISTINCT cta.county_fips) AS n_counties
            FROM types t
            LEFT JOIN county_type_assignments cta ON t.type_id = cta.dominant_type
            GROUP BY t.super_type_id
            ORDER BY t.super_type_id
            """,
        ).fetchdf()

    results = []
    for _, row in rows.iterrows():
        member_ids = row["member_type_ids"]
        if member_ids is None:
            member_ids = []
        results.append(
            SuperTypeSummary(
                super_type_id=int(row["super_type_id"]),
                display_name=str(row["display_name"]),
                member_type_ids=[int(x) for x in member_ids],
                n_counties=int(row["n_counties"]),
            )
        )
    return results
