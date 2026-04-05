"""Race comparison endpoint: GET /forecast/compare?slugs=slug1,slug2.

Returns side-by-side comparison data for exactly two races.  Reuses the
same DB queries and helpers as race_detail.py — no new data pipelines.

Comparison data includes:
- Prediction (dem share, margin, 90% CI)
- Poll count and confidence summary
- Top 5 electoral types by vote contribution
- Historical context (last race result, presidential baseline)

Validation rules:
- Exactly 2 slugs required (not 1, not 3+)
- Both slugs must exist in the predictions table
"""
from __future__ import annotations

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import get_db
from api.models import (
    HistoricalContext,
    PollConfidence,
    TypeBreakdown,
)

from ._helpers import (
    _VOTE_WEIGHTED_STATE_PRED_SQL,
    _Z90,
    _compute_state_std,
    _get_std_floor,
    _lookup_pollster_grade,
    slug_to_race,
)
from .race_detail import (
    _build_historical_context,
    _compute_poll_confidence,
    _infer_methodology,  # noqa: F401  (imported for test visibility)
)

router = APIRouter(tags=["forecast"])

# Maximum number of types returned per race in the comparison view.
# Five provides enough signal without overwhelming the narrow comparison columns.
_MAX_TYPES = 5


def _fetch_race_comparison_data(
    slug: str,
    request: Request,
    db: duckdb.DuckDBPyConnection,
    version_id: str,
) -> dict:
    """Fetch all comparison data for a single race slug.

    Returns a dict matching the RaceComparison schema.  Raises HTTPException
    with 404 if the slug has no predictions in the database.

    This is intentionally a standalone function (not a method) so it can be
    called twice cleanly for the two-slug comparison endpoint.
    """
    race = slug_to_race(slug)

    # Look up race metadata from the races table
    try:
        race_meta = db.execute(
            "SELECT race_type, state, year FROM races WHERE race_id = ?",
            [race],
        ).fetchone()
    except duckdb.CatalogException:
        race_meta = None

    if race_meta:
        race_type = race_meta[0].capitalize()
        state_abbr = race_meta[1]
        year = race_meta[2]
    else:
        # Fallback: parse from slug for backward compat
        parts = slug.split("-")
        state_abbr = parts[1].upper() if len(parts) >= 2 else "??"
        race_type = " ".join(p.capitalize() for p in parts[2:]) if len(parts) >= 3 else "Unknown"
        year = int(parts[0]) if parts[0].isdigit() else 2026

    # Validate that predictions exist for this race
    exists = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE version_id = ? AND race = ?",
        [version_id, race],
    ).fetchone()
    if not exists or exists[0] == 0:
        raise HTTPException(status_code=404, detail=f"Race '{slug}' not found")

    # Check for forecast_mode column (backward compat)
    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = 'local'" if _has_mode else ""

    # State-level prediction: vote-weighted average
    pred_row = db.execute(
        f"""
        SELECT
            {_VOTE_WEIGHTED_STATE_PRED_SQL} AS state_pred,
            COUNT(*) AS n_counties
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ? {_mode_filter}
        """,
        [version_id, race, state_abbr],
    ).fetchone()

    prediction = None if (pred_row is None or pred_row[0] is None) else float(pred_row[0])
    n_counties = int(pred_row[1]) if pred_row else 0

    # State-level uncertainty
    pred_std = None
    pred_lo90 = None
    pred_hi90 = None
    if prediction is not None:
        ci_rows = db.execute(
            f"""
            SELECT p.pred_dem_share, p.pred_std,
                   COALESCE(c.total_votes_2024, 1) AS votes
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
              AND p.pred_dem_share IS NOT NULL {_mode_filter}
            """,
            [version_id, race, state_abbr],
        ).fetchdf()

        if not ci_rows.empty and len(ci_rows) > 1:
            votes = ci_rows["votes"].values.astype(float)
            preds = ci_rows["pred_dem_share"].values.astype(float)
            pred_std = _compute_state_std(preds, votes, prediction, race_type=race_type)
        else:
            import numpy as np  # local import — only needed in the fallback branch
            _ = np  # suppress unused warning
            from api.routers.forecast._helpers import _STATE_STD_FALLBACK
            pred_std = max(_STATE_STD_FALLBACK, _get_std_floor(race_type))

        pred_lo90 = prediction - _Z90 * pred_std
        pred_hi90 = prediction + _Z90 * pred_std

    # Polls
    polls_df = db.execute(
        """
        SELECT date, pollster, dem_share, n_sample, notes
        FROM polls
        WHERE cycle = ? AND LOWER(race) = LOWER(?)
        ORDER BY date
        """,
        [str(year), race],
    ).fetchdf()

    # Latest poll summary (date + dem_share of most recent poll)
    latest_poll = None
    if not polls_df.empty:
        last_row = polls_df.iloc[-1]
        latest_poll = {
            "date": str(last_row["date"]) if last_row["date"] else None,
            "pollster": last_row["pollster"] if last_row["pollster"] else None,
            "dem_share": float(last_row["dem_share"]),
            "grade": _lookup_pollster_grade(
                request, last_row["pollster"] if last_row["pollster"] else None
            ),
        }

    poll_confidence = _compute_poll_confidence(polls_df)

    # Type breakdown: top N types by total vote contribution
    breakdown_df = db.execute(
        f"""
        SELECT
            cta.dominant_type AS type_id,
            t.display_name,
            COUNT(*) AS n_counties,
            AVG(p.pred_dem_share) AS mean_pred_dem_share,
            SUM(COALESCE(c.total_votes_2024, 0)) AS total_votes
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        JOIN county_type_assignments cta ON p.county_fips = cta.county_fips
        JOIN types t ON cta.dominant_type = t.type_id
        WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ? {_mode_filter}
        GROUP BY cta.dominant_type, t.display_name
        ORDER BY SUM(COALESCE(c.total_votes_2024, 0)) DESC, COUNT(*) DESC
        LIMIT ?
        """,
        [version_id, race, state_abbr, _MAX_TYPES],
    ).fetchdf()

    type_breakdown = [
        TypeBreakdown(
            type_id=int(row["type_id"]),
            display_name=row["display_name"],
            n_counties=int(row["n_counties"]),
            mean_pred_dem_share=None if pd.isna(row["mean_pred_dem_share"]) else float(row["mean_pred_dem_share"]),
            total_votes=int(row["total_votes"]) if row["total_votes"] else None,
        )
        for _, row in breakdown_df.iterrows()
    ]

    historical_context = _build_historical_context(slug, prediction)

    return {
        "slug": slug,
        "race": race,
        "state_abbr": state_abbr,
        "race_type": race_type,
        "year": year,
        "prediction": prediction,
        "pred_std": pred_std,
        "pred_lo90": pred_lo90,
        "pred_hi90": pred_hi90,
        "n_counties": n_counties,
        "n_polls": len(polls_df),
        "poll_confidence": poll_confidence.model_dump() if poll_confidence else None,
        "latest_poll": latest_poll,
        "type_breakdown": [t.model_dump() for t in type_breakdown],
        "historical_context": historical_context.model_dump() if historical_context else None,
    }


@router.get("/forecast/compare")
def get_race_compare(
    slugs: str = Query(
        ...,
        description="Comma-separated list of exactly 2 race slugs, e.g. 2026-fl-senate,2026-nc-senate",
    ),
    request: Request = None,  # type: ignore[assignment]  # injected by FastAPI
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict:
    """Return side-by-side comparison data for exactly 2 races.

    Query parameter: ``slugs=slug1,slug2``

    Returns a dict with ``races`` (list of 2 race data dicts) and ``slugs``
    (the two slug strings for client reference).

    Errors:
    - 422: not exactly 2 slugs provided
    - 404: either slug does not exist in the predictions table
    """
    slug_list = [s.strip() for s in slugs.split(",")]

    # Enforce exactly 2 slugs — not 1 and not 3+
    if len(slug_list) != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly 2 race slugs required; got {len(slug_list)}. "
                   "Pass ?slugs=slug1,slug2",
        )

    slug_a, slug_b = slug_list

    # Validate that neither slug is empty
    if not slug_a or not slug_b:
        raise HTTPException(
            status_code=422,
            detail="Both slugs must be non-empty strings.",
        )

    version_id = request.app.state.version_id

    race_a = _fetch_race_comparison_data(slug_a, request, db, version_id)
    race_b = _fetch_race_comparison_data(slug_b, request, db, version_id)

    return {
        "slugs": [slug_a, slug_b],
        "races": [race_a, race_b],
    }
