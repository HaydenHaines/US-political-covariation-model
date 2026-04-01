from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import CountyDetail, CountyRow, ElectionHistoryPoint, SiblingCounty

# Parquet files live under $WETHERVANE_DATA_DIR/assembled/ (default: data/assembled/)
_DATA_DIR = Path(os.environ.get("WETHERVANE_DATA_DIR", "data"))
_ASSEMBLED = _DATA_DIR / "assembled"

# Presidential election years available as parquet files
_PRES_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
# Senate election years available
_SEN_YEARS = [2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022]
# Governor election years available
_GOV_YEARS_ALGARA = [1994, 1998, 2002, 2006, 2010, 2014, 2018]
_GOV_YEARS_MEDSL = [2022]

router = APIRouter(tags=["counties"])


def _has_table(db: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return result is not None and result[0] > 0


# ── list_counties helpers ─────────────────────────────────────────────────────

def _query_counties_with_types_and_predictions(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> pd.DataFrame:
    """Query counties with type assignments, community assignments, and predictions."""
    return db.execute(
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


def _query_counties_with_types(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> pd.DataFrame:
    """Query counties with type assignments and community assignments (no predictions)."""
    return db.execute(
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


def _query_counties_with_communities(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> pd.DataFrame:
    """Query counties with community assignments only (no type info)."""
    return db.execute(
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


def _query_counties_bare(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query counties with no type or community data."""
    return db.execute(
        """
        SELECT county_fips, state_abbr
        FROM counties
        ORDER BY county_fips
        """,
    ).fetchdf()


def _select_counties_query(
    db: duckdb.DuckDBPyConnection,
    version_id: str,
    has_types: bool,
    has_predictions: bool,
    has_communities: bool,
) -> pd.DataFrame:
    """Select the appropriate counties query based on available tables."""
    if has_types:
        if has_predictions:
            return _query_counties_with_types_and_predictions(db, version_id)
        return _query_counties_with_types(db, version_id)
    if has_communities:
        return _query_counties_with_communities(db, version_id)
    return _query_counties_bare(db)


def _row_to_county_row(row: pd.Series) -> CountyRow:
    """Convert a DataFrame row to a CountyRow model, coercing nullable fields."""
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
    return CountyRow(
        county_fips=row["county_fips"],
        state_abbr=row["state_abbr"],
        community_id=community_id,
        dominant_type=dominant_type,
        super_type=super_type,
        pred_dem_share=pred_dem_share,
    )


@router.get("/counties", response_model=list[CountyRow])
def list_counties(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    version_id = request.app.state.version_id

    has_types = _has_table(db, "county_type_assignments")
    has_communities = _has_table(db, "community_assignments")
    has_predictions = _has_table(db, "predictions")

    rows = _select_counties_query(db, version_id, has_types, has_predictions, has_communities)
    return [_row_to_county_row(row) for _, row in rows.iterrows()]


# ── get_county_detail helpers ─────────────────────────────────────────────────

def _fetch_county_core(
    db: duckdb.DuckDBPyConnection, fips: str
) -> tuple:
    """Fetch core county info: fips, name, state, type assignments, display names, narrative."""
    return db.execute(
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


def _fetch_county_pred_dem_share(
    db: duckdb.DuckDBPyConnection, fips: str
) -> float | None:
    """Fetch average predicted Dem share across all races for a county."""
    pred_row = db.execute(
        "SELECT AVG(pred_dem_share) FROM predictions WHERE county_fips = ?",
        [fips],
    ).fetchone()
    return float(pred_row[0]) if pred_row and pred_row[0] is not None else None


def _fetch_county_demographics(
    db: duckdb.DuckDBPyConnection, fips: str
) -> dict[str, float]:
    """Fetch all demographic columns for a county as a float dict."""
    demo_row = db.execute(
        "SELECT * FROM county_demographics WHERE county_fips = ?",
        [fips],
    ).fetchone()
    if demo_row is None:
        return {}
    demo_cols = [
        desc[0]
        for desc in db.execute("DESCRIBE county_demographics").fetchall()
    ]
    return {
        col: float(val)
        for col, val in zip(demo_cols, demo_row)
        if col != "county_fips" and val is not None
    }


def _fetch_sibling_counties(
    db: duckdb.DuckDBPyConnection, dominant_type: int, fips: str
) -> list[SiblingCounty]:
    """Fetch up to 20 counties with the same dominant type, excluding the given fips."""
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
    return [
        SiblingCounty(county_fips=s[0], county_name=s[1], state_abbr=s[2])
        for s in siblings
    ]


@router.get("/counties/{fips}", response_model=CountyDetail)
def get_county_detail(
    fips: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Return detailed profile for a single county (SEO county page)."""
    row = _fetch_county_core(db, fips)
    if row is None:
        raise HTTPException(status_code=404, detail=f"County {fips} not found")

    (
        county_fips, county_name, state_abbr,
        dominant_type, super_type,
        type_display_name, super_type_display_name,
        narrative,
    ) = row

    pred_dem_share = _fetch_county_pred_dem_share(db, fips)
    demographics = _fetch_county_demographics(db, fips)
    sibling_counties = _fetch_sibling_counties(db, dominant_type, fips)

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


def _read_share(path: Path, fips: str, col: str, total_col: str) -> tuple[float, int | None] | None:
    """Read a single county's dem_share and total_votes from a parquet file.

    Returns (dem_share, total_votes) or None if county not found or value is NaN.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["county_fips", col, total_col])
    except (KeyError, ValueError):
        # Column may not exist (e.g. old schema)
        try:
            df = pd.read_parquet(path, columns=["county_fips", col])
            df[total_col] = None
        except (KeyError, ValueError):
            return None
    row = df[df["county_fips"] == fips]
    if row.empty:
        return None
    val = row[col].values[0]
    if pd.isna(val):
        return None
    total = row[total_col].values[0] if total_col in row.columns else None
    total_int: int | None = None if (total is None or pd.isna(total)) else int(total)
    return float(val), total_int


def _collect_history_points(
    fips: str,
    years: list[int],
    election_type: str,
    path_template: str,
    col_template: str,
    total_template: str,
) -> list[ElectionHistoryPoint]:
    """Collect ElectionHistoryPoint entries for a given election type and set of years.

    Args:
        fips: County FIPS code.
        years: List of election years to check.
        election_type: Label for the election type (e.g. "president", "senate", "governor").
        path_template: Format string for the parquet path, receives `year` as a keyword arg.
        col_template: Format string for the dem_share column name, receives `year`.
        total_template: Format string for the total_votes column name, receives `year`.
    """
    points = []
    for year in years:
        path = _ASSEMBLED / path_template.format(year=year)
        result = _read_share(path, fips, col_template.format(year=year), total_template.format(year=year))
        if result is not None:
            dem_share, total_votes = result
            points.append(ElectionHistoryPoint(
                year=year,
                election_type=election_type,
                dem_share=dem_share,
                total_votes=total_votes,
            ))
    return points


@router.get("/counties/{fips}/history", response_model=list[ElectionHistoryPoint])
def get_county_history(fips: str):
    """Return raw election results (Dem two-party share) for a county across all available cycles.

    Covers presidential (2000-2024), Senate (2002-2022), and gubernatorial (1994-2022) elections
    where county-level parquet data is available.  Years with no data for this county are omitted.
    Results are sorted by year then election_type.
    """
    points: list[ElectionHistoryPoint] = []

    points += _collect_history_points(
        fips, _PRES_YEARS, "president",
        "medsl_county_presidential_{year}.parquet",
        "pres_dem_share_{year}", "pres_total_{year}",
    )
    points += _collect_history_points(
        fips, _SEN_YEARS, "senate",
        "medsl_county_senate_{year}.parquet",
        "senate_dem_share_{year}", "senate_total_{year}",
    )
    points += _collect_history_points(
        fips, _GOV_YEARS_ALGARA, "governor",
        "algara_county_governor_{year}.parquet",
        "gov_dem_share_{year}", "gov_total_{year}",
    )
    # MEDSL 2022 governor uses a different naming convention for the parquet file
    # (single combined file for all counties rather than per-year files)
    points += _collect_history_points(
        fips, _GOV_YEARS_MEDSL, "governor",
        "medsl_county_2022_governor.parquet",
        "gov_dem_share_{year}", "gov_total_{year}",
    )

    # Sort by year, then election_type for consistent ordering
    points.sort(key=lambda p: (p.year, p.election_type))
    return points
