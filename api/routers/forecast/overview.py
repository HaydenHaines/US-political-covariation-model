"""Forecast overview endpoints: forecast table, race list, slugs, metadata, generic ballot."""
from __future__ import annotations

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from api.db import get_db
from api.models import ForecastRow, FundamentalsResponse, GenericBallotInfo

from ._helpers import _BASELINE_YEAR, _format_baseline_label, race_to_slug

router = APIRouter(tags=["forecast"])


@router.get("/forecast", response_model=list[ForecastRow])
def get_forecast(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    race: str | None = Query(None, description="Filter by race (e.g. FL_Senate)"),
    state: str | None = Query(None, description="Filter by state abbreviation (e.g. FL)"),
):
    version_id = request.app.state.version_id

    conditions = ["p.version_id = ?"]
    params: list = [version_id]

    if race:
        conditions.append("p.race = ?")
        params.append(race)
    if state:
        conditions.append("c.state_abbr = ?")
        params.append(state)

    where = " AND ".join(conditions)

    rows = db.execute(
        f"""
        SELECT
            p.county_fips,
            c.county_name,
            c.state_abbr,
            p.race,
            p.pred_dem_share,
            p.pred_std,
            p.pred_lo90,
            p.pred_hi90,
            p.state_pred,
            p.poll_avg
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE {where}
        ORDER BY p.race, c.state_abbr, p.county_fips
        """,
        params,
    ).fetchdf()

    if rows.empty:
        return []

    return [
        ForecastRow(
            county_fips=row["county_fips"],
            county_name=row["county_name"] if row["county_name"] else None,
            state_abbr=row["state_abbr"],
            race=row["race"],
            pred_dem_share=None if pd.isna(row["pred_dem_share"]) else float(row["pred_dem_share"]),
            pred_std=None if pd.isna(row["pred_std"]) else float(row["pred_std"]),
            pred_lo90=None if pd.isna(row["pred_lo90"]) else float(row["pred_lo90"]),
            pred_hi90=None if pd.isna(row["pred_hi90"]) else float(row["pred_hi90"]),
            state_pred=None if pd.isna(row["state_pred"]) else float(row["state_pred"]),
            poll_avg=None if pd.isna(row["poll_avg"]) else float(row["poll_avg"]),
        )
        for _, row in rows.iterrows()
    ]


@router.get("/forecast/races", response_model=list[str])
def get_forecast_races(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Return distinct race labels available in the predictions table, sorted."""
    version_id = request.app.state.version_id
    rows = db.execute(
        "SELECT DISTINCT race FROM predictions WHERE version_id = ? ORDER BY race",
        [version_id],
    ).fetchall()
    return [r[0] for r in rows]


@router.get("/forecast/race-slugs", response_model=list[str])
def get_race_slugs(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[str]:
    """Return URL slugs for all non-baseline races (for sitemap / generateStaticParams)."""
    version_id = request.app.state.version_id
    rows = db.execute(
        "SELECT DISTINCT race FROM predictions WHERE version_id = ? AND race != 'baseline' ORDER BY race",
        [version_id],
    ).fetchall()
    return [race_to_slug(r[0]) for r in rows]


@router.get("/forecast/race-metadata")
def get_race_metadata(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """Return metadata for all defined races from the registry.

    Includes whether predictions exist and how many polls are available,
    enabling the frontend to render a complete races index page even for
    races that haven't been predicted yet.
    """
    version_id = request.app.state.version_id
    try:
        rows = db.execute(
            """
            SELECT r.race_id, r.race_type, r.state, r.year,
                   COALESCE((SELECT COUNT(*) FROM predictions p
                             WHERE p.race = r.race_id AND p.version_id = ?), 0) AS n_predictions,
                   COALESCE((SELECT COUNT(*) FROM polls pl
                             WHERE pl.race = r.race_id), 0) AS n_polls
            FROM races r
            ORDER BY r.state, r.race_type
            """,
            [version_id],
        ).fetchall()
    except duckdb.CatalogException:
        # races table may not exist in older databases
        return []

    return [
        {
            "race_id": row[0],
            "slug": race_to_slug(row[0]),
            "race_type": row[1],
            "state": row[2],
            "year": row[3],
            "has_predictions": row[4] > 0,
            "n_polls": row[5],
        }
        for row in rows
    ]


@router.get("/forecast/generic-ballot", response_model=GenericBallotInfo)
def get_generic_ballot(
    manual_shift: float | None = Query(
        None, description="Override shift value (Dem share units)"
    ),
) -> GenericBallotInfo:
    """Return the national environment adjustment from generic ballot polling."""
    from src.prediction.generic_ballot import PRES_DEM_SHARE_2024_NATIONAL
    from src.prediction.generic_ballot import compute_gb_shift as _compute

    info = _compute(manual_shift=manual_shift)
    return GenericBallotInfo(
        gb_avg=info.gb_avg,
        pres_baseline=info.pres_baseline,
        shift=info.shift,
        shift_pp=info.shift * 100,
        n_polls=info.n_polls,
        n_yougov_polls=info.n_yougov_polls,
        source=info.source,
        baseline_year=_BASELINE_YEAR,
        baseline_label=_format_baseline_label(PRES_DEM_SHARE_2024_NATIONAL),
    )


@router.get("/forecast/fundamentals", response_model=FundamentalsResponse)
def get_fundamentals() -> FundamentalsResponse:
    """Return the structural fundamentals model prediction for the current cycle.

    The fundamentals model uses presidential approval, GDP growth, unemployment,
    and CPI inflation to estimate a structural Dem share shift — independent of
    any polling data.  It is blended with the generic ballot shift at a
    configurable weight (default 30%) to produce the combined environment prior.
    """
    import json as _json
    from pathlib import Path

    from src.prediction.fundamentals import (
        compute_fundamentals_shift as _compute_fund,
        load_fundamentals_snapshot as _load_snap,
    )
    from src.prediction.generic_ballot import compute_gb_shift as _compute_gb

    # Load config for the blending weight
    params_path = Path(__file__).resolve().parents[3] / "data" / "config" / "prediction_params.json"
    params = _json.loads(params_path.read_text()) if params_path.exists() else {}
    fund_params = params.get("fundamentals", {})
    weight = float(fund_params.get("fundamentals_weight", 0.3))
    enabled = bool(fund_params.get("enabled", False))

    snapshot = _load_snap()
    fund_info = _compute_fund(snapshot)
    gb_info = _compute_gb()

    combined_pp = weight * fund_info.shift * 100 + (1 - weight) * gb_info.shift * 100

    return FundamentalsResponse(
        shift=fund_info.shift,
        shift_pp=round(fund_info.shift * 100, 2),
        approval_contribution_pp=round(fund_info.approval_contribution * 100, 2),
        gdp_contribution_pp=round(fund_info.gdp_contribution * 100, 2),
        unemployment_contribution_pp=round(fund_info.unemployment_contribution * 100, 2),
        cpi_contribution_pp=round(fund_info.cpi_contribution * 100, 2),
        loo_rmse_pp=round(fund_info.loo_rmse * 100, 2),
        n_training=fund_info.n_training,
        weight=weight if enabled else 0.0,
        combined_shift_pp=round(combined_pp, 2) if enabled else round(gb_info.shift * 100, 2),
        snapshot={
            "cycle": snapshot.cycle,
            "in_party": snapshot.in_party,
            "approval_net_oct": snapshot.approval_net_oct,
            "gdp_q2_growth_pct": snapshot.gdp_q2_growth_pct,
            "unemployment_oct": snapshot.unemployment_oct,
            "cpi_yoy_oct": snapshot.cpi_yoy_oct,
            "consumer_sentiment": snapshot.consumer_sentiment,
        },
    )
