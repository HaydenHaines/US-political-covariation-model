"""Electoral type endpoints: list, detail, scatter, correlated, super-types."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import (
    CorrelatedType,
    SuperTypeSummary,
    TypeCounty,
    TypeDetail,
    TypeScatterPoint,
    TypeSummary,
)

router = APIRouter(tags=["communities"])


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
            AVG(p.pred_dem_share) AS mean_pred_dem_share,
            t.median_hh_income,
            t.pct_bachelors_plus,
            t.pct_white_nh,
            t.log_pop_density
        FROM types t
        LEFT JOIN county_type_assignments cta
            ON t.type_id = cta.dominant_type
        LEFT JOIN predictions p
            ON cta.county_fips = p.county_fips
        GROUP BY t.type_id, t.super_type_id, t.display_name,
                 t.median_hh_income, t.pct_bachelors_plus,
                 t.pct_white_nh, t.log_pop_density
        ORDER BY t.type_id
        """,
    ).fetchdf()

    def _f(v) -> float | None:
        return None if pd.isna(v) else float(v)

    results = []
    for _, row in rows.iterrows():
        results.append(
            TypeSummary(
                type_id=int(row["type_id"]),
                super_type_id=None if pd.isna(row["super_type_id"]) else int(row["super_type_id"]),
                display_name=str(row["display_name"]),
                n_counties=int(row["n_counties"]),
                mean_pred_dem_share=_f(row["mean_pred_dem_share"]),
                median_hh_income=_f(row["median_hh_income"]),
                pct_bachelors_plus=_f(row["pct_bachelors_plus"]),
                pct_white_nh=_f(row["pct_white_nh"]),
                log_pop_density=_f(row["log_pop_density"]),
            )
        )
    return results


@router.get("/types/scatter-data", response_model=list[TypeScatterPoint])
def get_type_scatter_data(db: duckdb.DuckDBPyConnection = Depends(get_db)):
    """Return all types with demographics and shift profiles for scatter plot."""
    if not _has_table(db, "types"):
        return []

    # Get all types with county counts
    type_rows = db.execute(
        """
        SELECT
            t.type_id,
            t.super_type_id,
            t.display_name,
            COUNT(DISTINCT cta.county_fips) AS n_counties
        FROM types t
        LEFT JOIN county_type_assignments cta ON t.type_id = cta.dominant_type
        GROUP BY t.type_id, t.super_type_id, t.display_name
        ORDER BY t.type_id
        """,
    ).fetchdf()

    if type_rows.empty:
        return []

    # Get all type demographics in one query
    demo_rows = db.execute("SELECT * FROM types ORDER BY type_id").fetchdf()
    skip_cols = {"type_id", "super_type_id", "display_name", "n_counties", "narrative"}
    demo_cols = [c for c in demo_rows.columns if c not in skip_cols]

    # Get shift column names
    shift_col_names: list[str] = []
    try:
        shift_cols_row = db.execute("SELECT * FROM county_shifts LIMIT 0").fetchdf()
        shift_col_names = [
            c for c in shift_cols_row.columns if c not in ("county_fips", "version_id")
        ]
    except duckdb.Error:
        pass

    # Get all county-type assignments and shifts in one join
    shift_by_type: dict[int, dict[str, float]] = {}
    if shift_col_names:
        try:
            # Average shifts grouped by dominant_type
            agg_cols = ", ".join(f"AVG(cs.{c}) AS {c}" for c in shift_col_names)
            shift_df = db.execute(
                f"""
                SELECT cta.dominant_type AS type_id, {agg_cols}
                FROM county_type_assignments cta
                JOIN county_shifts cs ON cta.county_fips = cs.county_fips
                GROUP BY cta.dominant_type
                """,
            ).fetchdf()
            for _, row in shift_df.iterrows():
                tid = int(row["type_id"])
                profile: dict[str, float] = {}
                for col in shift_col_names:
                    if col in row and not pd.isna(row[col]):
                        profile[col] = float(row[col])
                shift_by_type[tid] = profile
        except duckdb.Error:
            pass

    def _f(v) -> float | None:
        return None if pd.isna(v) else float(v)

    results = []
    for _, trow in type_rows.iterrows():
        tid = int(trow["type_id"])

        # Demographics from types table
        demographics: dict[str, float] = {}
        demo_match = demo_rows[demo_rows["type_id"] == tid]
        if not demo_match.empty:
            r = demo_match.iloc[0]
            for col in demo_cols:
                if col in demo_match.columns:
                    val = _f(r[col])
                    if val is not None:
                        demographics[col] = val

        results.append(TypeScatterPoint(
            type_id=tid,
            super_type_id=int(trow["super_type_id"]),
            display_name=str(trow["display_name"]),
            n_counties=int(trow["n_counties"]),
            demographics=demographics,
            shift_profile=shift_by_type.get(tid, {}),
        ))

    return results


@router.get("/types/{type_id}/correlated", response_model=list[CorrelatedType])
def get_correlated_types(
    type_id: int,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    n: int = 5,
):
    """Return the N types most correlated with the given type, using the
    Ledoit-Wolf regularized observed electoral correlation matrix."""
    correlation = request.app.state.type_correlation
    if correlation is None:
        raise HTTPException(503, "Correlation matrix not loaded")

    J = correlation.shape[0]
    if type_id < 0 or type_id >= J:
        raise HTTPException(404, f"Type {type_id} not found (valid: 0-{J-1})")

    n = min(n, J - 1)  # can't return more than J-1 (excluding self)

    row = correlation[type_id]
    # Sort by correlation descending, excluding self
    sorted_indices = np.argsort(-row)
    top_indices = [int(i) for i in sorted_indices if i != type_id][:n]

    results = []
    for idx in top_indices:
        type_row = db.execute(
            """SELECT t.type_id, t.display_name, t.super_type_id,
                      COUNT(DISTINCT cta.county_fips) AS n_counties,
                      AVG(p.pred_dem_share) AS mean_pred_dem_share
               FROM types t
               LEFT JOIN county_type_assignments cta ON t.type_id = cta.dominant_type
               LEFT JOIN predictions p ON cta.county_fips = p.county_fips
               WHERE t.type_id = ?
               GROUP BY t.type_id, t.display_name, t.super_type_id""",
            [idx],
        ).fetchone()
        if type_row:
            mean_share = None if type_row[4] is None or pd.isna(type_row[4]) else float(type_row[4])
            results.append(
                CorrelatedType(
                    type_id=type_row[0],
                    display_name=type_row[1],
                    super_type_id=type_row[2],
                    n_counties=type_row[3],
                    mean_pred_dem_share=mean_share,
                    correlation=round(float(row[idx]), 4),
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

    # Detect whether the narrative column exists (may be absent in test DBs)
    _has_narrative = bool(
        db.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name='types' AND column_name='narrative'",
        ).fetchone()[0]
    )

    # Get type metadata
    if _has_narrative:
        type_row = db.execute(
            "SELECT type_id, super_type_id, display_name, narrative FROM types WHERE type_id = ? LIMIT 1",
            [type_id],
        ).fetchone()
        if not type_row:
            raise HTTPException(status_code=404, detail=f"Type {type_id} not found")
        tid, super_type_id, display_name, narrative = type_row
    else:
        type_row = db.execute(
            "SELECT type_id, super_type_id, display_name FROM types WHERE type_id = ? LIMIT 1",
            [type_id],
        ).fetchone()
        if not type_row:
            raise HTTPException(status_code=404, detail=f"Type {type_id} not found")
        tid, super_type_id, display_name = type_row
        narrative = None

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
            skip_cols = {"type_id", "super_type_id", "display_name", "n_counties", "narrative"}
            for col in demo_row.columns:
                if col not in skip_cols:
                    val = r[col]
                    if not pd.isna(val):
                        try:
                            demographics[col] = float(val)
                        except (ValueError, TypeError):
                            pass
    except duckdb.Error:
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
        except duckdb.Error:
            pass

    # Compute mean prediction across member counties
    mean_pred: float | None = None
    if county_fips_list:
        try:
            placeholders = ", ".join("?" * len(county_fips_list))
            pred_row = db.execute(
                f"SELECT AVG(pred_dem_share) FROM predictions WHERE county_fips IN ({placeholders})",
                county_fips_list,
            ).fetchone()
            if pred_row and pred_row[0] is not None:
                mean_pred = float(pred_row[0])
        except duckdb.Error:
            pass

    return TypeDetail(
        type_id=int(tid),
        super_type_id=int(super_type_id),
        display_name=str(display_name),
        n_counties=len(counties),
        mean_pred_dem_share=mean_pred,
        counties=counties,
        demographics=demographics,
        shift_profile=shift_profile,
        narrative=str(narrative) if narrative is not None else None,
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
                ARRAY_AGG(DISTINCT t.type_id ORDER BY t.type_id) FILTER (WHERE t.type_id IS NOT NULL) AS member_type_ids,
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
        if member_ids is None or not hasattr(member_ids, "__iter__"):
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
