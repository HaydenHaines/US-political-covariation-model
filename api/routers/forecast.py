from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import get_db
from api.models import (
    BlendResult,
    BlendWeights,
    OverviewBlendRaceSummary,
    OverviewBlendResult,
    ChangelogEntry,
    ChangelogRaceDiff,
    ChangelogResponse,
    ForecastRow,
    GenericBallotInfo,
    HistoricalContext,
    LastRaceResult,
    MultiPollInput,
    MultiPollResponse,
    PollInput,
    PollRow,
    PollTrendPoll,
    PollTrendResponse,
    PollTrend,
    PresidentialResult,
    RaceDetail,
    RacePoll,
    SectionWeights,
    TypeBreakdown,
)

# Path to the static historical results data file (lives alongside the api/ package)
_HISTORICAL_RESULTS_PATH = Path(__file__).parent.parent / "data" / "historical_results.json"


def _load_historical_results() -> dict:
    """Load and return the historical results dict from disk.

    Returns an empty dict when the file is missing (graceful degradation).
    Strips comment keys (those starting with '_') used for documentation.
    """
    if not _HISTORICAL_RESULTS_PATH.exists():
        return {}
    with _HISTORICAL_RESULTS_PATH.open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# Loaded once at import time - this file changes only when race data is manually updated
_HISTORICAL_RESULTS: dict = _load_historical_results()

# Uncertainty model parameters — see docs/ARCHITECTURE.md for calibration notes
_STATE_STD_FLOOR = 0.035      # minimum state-level std; prevents over-confidence when counties agree
_STATE_STD_CAP = 0.15         # hard cap; beyond this, the race is essentially a coin flip
_STATE_STD_FALLBACK = 0.065   # used when poll-derived std is unavailable
_MATRIX_JITTER = 1e-8         # Tikhonov regularization keeps covariance PD during matrix inversion
_Z90 = 1.645                  # z-score for 90% confidence interval

router = APIRouter(tags=["forecast"])


def _lookup_pollster_grade(request: Request, pollster_name: str | None) -> str | None:
    """Look up Silver Bulletin letter grade for a pollster, with fuzzy matching."""
    if not pollster_name:
        return None
    grades = getattr(request.app.state, "pollster_grades", {})
    norm_grades = getattr(request.app.state, "pollster_grades_normalized", {})
    if not grades:
        return None
    # Exact match
    if pollster_name in grades:
        return grades[pollster_name]
    # Normalized match
    from src.assembly.silver_bulletin_ratings import _normalize, _name_similarity
    norm = _normalize(pollster_name)
    if norm in norm_grades:
        return norm_grades[norm]
    # Fuzzy match (Jaccard > 0.4)
    best_grade, best_sim = None, 0.0
    for nk, grade in norm_grades.items():
        sim = _name_similarity(norm, nk)
        if sim > best_sim:
            best_sim = sim
            best_grade = grade
    return best_grade if best_sim >= 0.4 else None


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


def race_to_slug(race: str) -> str:
    """Convert race label to URL slug. "2026 FL Governor" → "2026-fl-governor"."""
    return race.lower().replace(" ", "-")


def slug_to_race(slug: str) -> str:
    """Convert URL slug back to race label. "2026-fl-governor" → "2026 FL Governor"."""
    parts = slug.split("-")
    if len(parts) < 3:
        return slug
    year = parts[0]
    state = parts[1].upper()
    race_type = " ".join(p.capitalize() for p in parts[2:])
    return f"{year} {state} {race_type}"


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


@router.get("/forecast/race/{slug}", response_model=RaceDetail)
def get_race_detail(
    slug: str,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    mode: str = Query("local", description="Forecast mode: 'national' or 'local'"),
) -> RaceDetail:
    """Return state-level prediction, polls, and type breakdown for a single race."""
    version_id = request.app.state.version_id
    race = slug_to_race(slug)
    forecast_mode = mode if mode in ("national", "local") else "local"

    # Look up race metadata from the races table (preferred) or fall back to
    # slug parsing for backward compatibility with pre-registry predictions.
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
        # Fallback: parse from slug (backward compat with old predictions)
        parts = slug.split("-")
        state_abbr = parts[1].upper() if len(parts) >= 2 else "??"
        race_type = " ".join(p.capitalize() for p in parts[2:]) if len(parts) >= 3 else "Unknown"
        year = int(parts[0]) if parts[0].isdigit() else 2026

    # Validate the race has predictions
    exists = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE version_id = ? AND race = ?",
        [version_id, race],
    ).fetchone()
    if not exists or exists[0] == 0:
        raise HTTPException(status_code=404, detail=f"Race '{race}' not found")

    # Check if forecast_mode column exists (backward compat)
    _has_mode = "forecast_mode" in [
        row[0] for row in db.execute("DESCRIBE predictions").fetchall()
    ]
    _mode_filter = "AND p.forecast_mode = ?" if _has_mode else ""
    _mode_params = [forecast_mode] if _has_mode else []

    # State-level prediction: vote-weighted average SUM(pred * votes) / SUM(votes).
    # Falls back to simple AVG when total_votes_2024 is NULL.
    pred_row = db.execute(
        f"""
        SELECT
            CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                 THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                      / SUM(COALESCE(c.total_votes_2024, 0))
                 ELSE AVG(p.pred_dem_share)
            END AS state_pred,
            COUNT(*) AS n_counties
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ? {_mode_filter}
        """,
        [version_id, race, state_abbr] + _mode_params,
    ).fetchone()

    prediction = None if (pred_row is None or pred_row[0] is None) else float(pred_row[0])
    n_counties = int(pred_row[1]) if pred_row else 0

    # State-level uncertainty from county-level CI data.
    # Uses vote-weighted std of county predictions around the state mean,
    # then shrinks toward the model's LOO RMSE (0.059) as a floor.
    state_std = None
    state_lo90 = None
    state_hi90 = None
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
            [version_id, race, state_abbr] + _mode_params,
        ).fetchdf()

        if not ci_rows.empty and len(ci_rows) > 1:
            votes = ci_rows["votes"].values.astype(float)
            preds = ci_rows["pred_dem_share"].values.astype(float)
            total_votes = votes.sum()

            if total_votes > 0:
                weights = votes / total_votes
                # Weighted variance of county predictions around the state mean
                county_var = float(np.sum(weights * (preds - prediction) ** 2))
                # Scale by sqrt(1/N_eff) — effective number of independent
                # observations, bounded by number of types represented
                n_eff = max(1.0, 1.0 / np.sum(weights ** 2))
                state_std = float(np.sqrt(county_var / n_eff))
                # Floor: model LOO RMSE of 0.059 ensures we don't understate
                # uncertainty even when all counties agree
                state_std = max(state_std, _STATE_STD_FLOOR)
                # Cap: never wider than 15pp (uninformative)
                state_std = min(state_std, _STATE_STD_CAP)
            else:
                state_std = _STATE_STD_FALLBACK  # fallback to historical estimate
        else:
            state_std = _STATE_STD_FALLBACK

        state_lo90 = prediction - _Z90 * state_std
        state_hi90 = prediction + _Z90 * state_std

    # Also get the other mode's prediction for comparison
    other_mode = "national" if forecast_mode == "local" else "local"
    other_pred_row = None
    if _has_mode:
        other_pred_row = db.execute(
            """
            SELECT
                CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                          / SUM(COALESCE(c.total_votes_2024, 0))
                     ELSE AVG(p.pred_dem_share)
                END AS state_pred
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
              AND p.forecast_mode = ?
            """,
            [version_id, race, state_abbr, other_mode],
        ).fetchone()
    state_pred_national = None
    state_pred_local = None
    if forecast_mode == "local":
        state_pred_local = prediction
        state_pred_national = float(other_pred_row[0]) if other_pred_row and other_pred_row[0] else None
    else:
        state_pred_national = prediction
        state_pred_local = float(other_pred_row[0]) if other_pred_row and other_pred_row[0] else None

    # Polls for this race
    polls_df = db.execute(
        """
        SELECT date, pollster, dem_share, n_sample
        FROM polls
        WHERE cycle = ? AND LOWER(race) = LOWER(?)
        ORDER BY date
        """,
        [str(year), race],
    ).fetchdf()

    polls: list[RacePoll] = [
        RacePoll(
            date=row["date"] if row["date"] else None,
            pollster=row["pollster"] if row["pollster"] else None,
            dem_share=float(row["dem_share"]),
            n_sample=int(row["n_sample"]) if not pd.isna(row["n_sample"]) else None,
            grade=_lookup_pollster_grade(
                request, row["pollster"] if row["pollster"] else None
            ),
        )
        for _, row in polls_df.iterrows()
    ]

    # Type breakdown: top 5 types by total vote contribution in this state.
    # Sorted by SUM(total_votes_2024) DESC so urban/high-population types
    # appear first even when they cover fewer counties. Without this,
    # states like MI show only rural types because small rural counties
    # outnumber large urban ones. Falls back to COUNT(*) when votes are NULL.
    # (GitHub issue #21)
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
        LIMIT 5
        """,
        [version_id, race, state_abbr] + _mode_params,
    ).fetchdf()

    type_breakdown: list[TypeBreakdown] = [
        TypeBreakdown(
            type_id=int(row["type_id"]),
            display_name=row["display_name"],
            n_counties=int(row["n_counties"]),
            mean_pred_dem_share=None if pd.isna(row["mean_pred_dem_share"]) else float(row["mean_pred_dem_share"]),
            total_votes=int(row["total_votes"]) if row["total_votes"] else None,
        )
        for _, row in breakdown_df.iterrows()
    ]

    candidate_effect = None
    if state_pred_local is not None and state_pred_national is not None:
        candidate_effect = state_pred_local - state_pred_national

    return RaceDetail(
        race=race,
        slug=slug,
        state_abbr=state_abbr,
        race_type=race_type,
        year=year,
        prediction=prediction,
        n_counties=n_counties,
        polls=polls,
        type_breakdown=type_breakdown,
        forecast_mode=forecast_mode,
        state_pred_national=state_pred_national,
        state_pred_local=state_pred_local,
        candidate_effect_margin=candidate_effect,
        n_polls=len(polls),
        pred_std=state_std,
        pred_lo90=state_lo90,
        pred_hi90=state_hi90,
        historical_context=_build_historical_context(slug, prediction),
    )


def _build_historical_context(slug: str, prediction: float | None) -> HistoricalContext | None:
    """Build a HistoricalContext for a race slug from the static data file.

    Returns None when the slug has no entry in historical_results.json
    (i.e. it is not one of the tracked competitive races).

    forecast_shift is calculated as: current model prediction margin minus
    the last race margin.  Both are in percentage-point margin space where
    the model prediction (Dem two-party share) is converted via
    (pred - 0.5) * 200.  Positive shift = more Dem than last result.
    """
    entry = _HISTORICAL_RESULTS.get(slug)
    if entry is None:
        return None

    last = entry["last_race"]
    pres = entry["presidential_2024"]

    last_result = LastRaceResult(
        year=last["year"],
        winner=last["winner"],
        party=last["party"],
        margin=last["margin"],
        note=last.get("note"),
    )
    pres_result = PresidentialResult(
        winner=pres["winner"],
        party=pres["party"],
        margin=pres["margin"],
        note=pres.get("note"),
    )

    forecast_shift = None
    if prediction is not None:
        # Convert Dem two-party share to margin pp: (dem_share - 0.5) * 200
        # e.g. 0.52 -> D+4.0;  0.48 -> R+4.0 (represented as -4.0)
        model_margin_pp = (prediction - 0.5) * 200.0
        forecast_shift = round(model_margin_pp - last_result.margin, 1)

    return HistoricalContext(
        last_race=last_result,
        presidential_2024=pres_result,
        forecast_shift=forecast_shift,
    )


@router.get("/forecast/race/{slug}/history", response_model=HistoricalContext)
def get_race_history(slug: str) -> HistoricalContext:
    """Return historical electoral context for a tracked race.

    Reads from the static api/data/historical_results.json file which
    covers the competitive races WetherVane tracks closely.  Returns
    404 for races not in that file (safe/uncompetitive races).

    Note: forecast_shift is not populated here (no prediction context);
    fetch /forecast/race/{slug} to get the full context with forecast_shift.
    """
    entry = _HISTORICAL_RESULTS.get(slug)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical data for race \'{slug}\'. "
                "Only tracked competitive races have history."
            ),
        )

    last = entry["last_race"]
    pres = entry["presidential_2024"]

    return HistoricalContext(
        last_race=LastRaceResult(
            year=last["year"],
            winner=last["winner"],
            party=last["party"],
            margin=last["margin"],
            note=last.get("note"),
        ),
        presidential_2024=PresidentialResult(
            winner=pres["winner"],
            party=pres["party"],
            margin=pres["margin"],
            note=pres.get("note"),
        ),
        forecast_shift=None,  # no prediction available in this standalone endpoint
    )


@router.post("/forecast/race/{slug}/blend", response_model=BlendResult)
def recalculate_blend(
    slug: str,
    body: BlendWeights,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> BlendResult:
    """Re-run the forecast for a race with custom section weights.

    The frontend race detail page calls this endpoint whenever the user
    adjusts the model-prior / state-polls / national-polls sliders.  It
    returns only the fields needed to update the hero: prediction,
    pred_std, pred_lo90, pred_hi90.

    Section weights are expressed as 0-100 percentages on the wire (to
    match the slider scale) and normalised to [0, 2] multipliers here
    before being forwarded to the prediction engine.  The default 60/30/10
    split becomes model_prior=1.2, state_polls=0.6, national_polls=0.2
    after normalisation.

    Returns the structural prior unchanged when no polls are available or
    when the type-score data is not loaded into app state.
    """
    version_id = request.app.state.version_id
    race = slug_to_race(slug)

    # Resolve race metadata for state lookup
    try:
        race_meta = db.execute(
            "SELECT race_type, state, year FROM races WHERE race_id = ?",
            [race],
        ).fetchone()
    except duckdb.CatalogException:
        race_meta = None

    if race_meta:
        state_abbr = race_meta[1]
        year = race_meta[2]
    else:
        parts = slug.split("-")
        state_abbr = parts[1].upper() if len(parts) >= 2 else "??"
        year = int(parts[0]) if parts[0].isdigit() else 2026

    # Fetch state polls for this race
    polls_df = db.execute(
        """
        SELECT dem_share, n_sample, geography, geo_level, race, date, pollster
        FROM polls
        WHERE cycle = ? AND LOWER(race) = LOWER(?) AND geo_level = 'state'
        ORDER BY date
        """,
        [str(year), race],
    ).fetchdf()

    # If no polls, return the structural prior prediction unchanged.
    # Blend controls have no effect when there is nothing to blend.
    if polls_df.empty:
        pred_row = db.execute(
            """
            SELECT
                CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                          / SUM(COALESCE(c.total_votes_2024, 0))
                     ELSE AVG(p.pred_dem_share)
                END AS state_pred,
                AVG(p.pred_std) AS mean_std
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
            """,
            [version_id, race, state_abbr],
        ).fetchone()
        if pred_row is None or pred_row[0] is None:
            return BlendResult(prediction=None, pred_std=None, pred_lo90=None, pred_hi90=None)
        pred = float(pred_row[0])
        std = float(pred_row[1]) if pred_row[1] else _STATE_STD_FALLBACK
        std = max(std, _STATE_STD_FLOOR)
        return BlendResult(
            prediction=pred,
            pred_std=std,
            pred_lo90=pred - _Z90 * std,
            pred_hi90=pred + _Z90 * std,
        )

    # Normalise 0-100 percentage weights to [0, 2] multipliers.
    # 100 / 50 = 2.0 keeps the scale symmetric around the default of 1.0.
    norm = 50.0
    sw = SectionWeights(
        model_prior=body.model_prior / norm,
        state_polls=body.state_polls / norm,
        national_polls=body.national_polls / norm,
    )

    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    type_county_fips = getattr(request.app.state, "type_county_fips", None)

    # If type model isn't loaded, fall back to the stored prediction
    if not all(x is not None for x in [type_scores, type_covariance, type_priors, type_county_fips]):
        pred_row = db.execute(
            """
            SELECT
                CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                          / SUM(COALESCE(c.total_votes_2024, 0))
                     ELSE AVG(p.pred_dem_share)
                END AS state_pred
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
            """,
            [version_id, race, state_abbr],
        ).fetchone()
        if pred_row is None or pred_row[0] is None:
            return BlendResult(prediction=None, pred_std=None, pred_lo90=None, pred_hi90=None)
        pred = float(pred_row[0])
        return BlendResult(
            prediction=pred,
            pred_std=_STATE_STD_FALLBACK,
            pred_lo90=pred - _Z90 * _STATE_STD_FALLBACK,
            pred_hi90=pred + _Z90 * _STATE_STD_FALLBACK,
        )

    # Build poll list — use section weight to scale effective N
    race_polls = [
        (
            float(row["dem_share"]),
            max(1, int(row["n_sample"] * sw.state_polls)),
            row["geography"],
        )
        for _, row in polls_df.iterrows()
    ]

    from src.prediction.forecast_runner import predict_race

    tract_states = getattr(request.app.state, "tract_states", {})
    fips_list = type_county_fips
    states_list = [tract_states.get(f, f[:2]) for f in fips_list]
    names_list = [f"Tract {f}" for f in fips_list]

    ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
    county_priors = (
        np.array([ridge_priors.get(f, 0.45) for f in fips_list])
        if ridge_priors
        else None
    )

    # Apply behavior adjustment for off-cycle races
    behavior_tau = getattr(request.app.state, "behavior_tau", None)
    behavior_delta = getattr(request.app.state, "behavior_delta", None)
    race_str = race.lower()
    is_offcycle = not any(kw in race_str for kw in ["president", "pres"])

    if (
        behavior_tau is not None
        and behavior_delta is not None
        and county_priors is not None
        and is_offcycle
        and type_scores.shape[1] == len(behavior_tau)
    ):
        from src.behavior.voter_behavior import apply_behavior_adjustment
        county_priors = apply_behavior_adjustment(
            county_priors, type_scores, behavior_tau, behavior_delta, is_offcycle=True
        )

    result_df = predict_race(
        race=race,
        polls=race_polls,
        type_scores=type_scores,
        type_covariance=type_covariance,
        type_priors=type_priors,
        county_fips=fips_list,
        states=states_list,
        county_names=names_list,
        county_priors=county_priors,
        prior_weight=sw.model_prior,
    )

    # Aggregate to state-level vote-weighted prediction
    state_rows = result_df[result_df["state"] == state_abbr]
    if state_rows.empty:
        return BlendResult(prediction=None, pred_std=None, pred_lo90=None, pred_hi90=None)

    tract_votes: dict[str, float] = getattr(request.app.state, "tract_votes", {})
    vote_weights = (
        state_rows["county_fips"].map(lambda f: tract_votes.get(f, 0)).values.astype(float)
    )
    total_w = vote_weights.sum()
    if total_w > 0:
        pred = float((state_rows["pred_dem_share"].values * vote_weights).sum() / total_w)
    else:
        pred = float(state_rows["pred_dem_share"].mean())

    # Derive state-level std from county-level CI data
    preds = state_rows["pred_dem_share"].values.astype(float)
    if total_w > 0 and len(preds) > 1:
        weights_norm = vote_weights / total_w
        county_var = float(np.sum(weights_norm * (preds - pred) ** 2))
        n_eff = max(1.0, 1.0 / np.sum(weights_norm ** 2))
        state_std = float(np.sqrt(county_var / n_eff))
        state_std = max(state_std, _STATE_STD_FLOOR)
        state_std = min(state_std, _STATE_STD_CAP)
    else:
        state_std = _STATE_STD_FALLBACK

    return BlendResult(
        prediction=pred,
        pred_std=state_std,
        pred_lo90=pred - _Z90 * state_std,
        pred_hi90=pred + _Z90 * state_std,
    )



@router.post("/forecast/overview/blend", response_model=OverviewBlendResult)
def recalculate_overview_blend(
    body: BlendWeights,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> OverviewBlendResult:
    """Recalculate all 33 Class II Senate races with custom blend weights.

    Accepts section weights as 0-100 percentages (model_prior, state_polls,
    national_polls). Iterates over all tracked states, calls the per-race
    recalculate_blend() function for each, and aggregates projected seat totals.

    Returns dem_seats and rep_seats as projected chamber totals (holdover seats
    not up in 2026 + favored contested seats), so the BalanceBar can be
    updated without the frontend knowing those constants.
    """
    # Local import to avoid circular dependency (senate.py never imports forecast.py)
    from api.routers.senate import (
        SENATE_2026_STATES,
        _DEM_HOLDOVER_SEATS,
        _GOP_HOLDOVER_SEATS,
        _CLASS_II_INCUMBENT,
        _DEFAULT_SAFE_MARGIN,
        _margin_to_rating,
        _TOSSUP_MAX,
    )

    race_summaries: list[OverviewBlendRaceSummary] = []
    dem_favored = 0
    gop_favored = 0

    for st in sorted(SENATE_2026_STATES):
        race_str = f"2026 {st} Senate"
        slug = race_str.lower().replace(" ", "-")

        # Reuse per-race blend logic directly
        blend_result = recalculate_blend(slug=slug, body=body, request=request, db=db)

        if blend_result.prediction is not None:
            margin = blend_result.prediction - 0.5
        else:
            # Race has no model data — treat as safe for incumbent party
            incumbent_party = _CLASS_II_INCUMBENT.get(st, "R")
            margin = _DEFAULT_SAFE_MARGIN if incumbent_party == "D" else -_DEFAULT_SAFE_MARGIN

        rating = _margin_to_rating(margin)

        # Seat counts: only count races that aren't tossups
        if margin > _TOSSUP_MAX:
            dem_favored += 1
        elif margin < -_TOSSUP_MAX:
            gop_favored += 1

        race_summaries.append(OverviewBlendRaceSummary(
            slug=slug,
            prediction=blend_result.prediction,
            pred_std=blend_result.pred_std,
            rating_label=rating,
        ))

    return OverviewBlendResult(
        dem_seats=_DEM_HOLDOVER_SEATS + dem_favored,
        rep_seats=_GOP_HOLDOVER_SEATS + gop_favored,
        races=race_summaries,
    )


@router.get("/forecast/race/{slug}/poll-trend", response_model=PollTrendResponse)
def get_poll_trend(
    slug: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> PollTrendResponse:
    """Return poll data and a weighted moving-average trend line for a single race.

    Polls are returned in chronological order.  The trend is a 30-day rolling
    weighted average — each poll contributes proportionally to its sample size.
    Two-party Republican share is inferred as ``1 - dem_share`` since the polls
    table stores only the Democratic share.

    Returns an empty polls list and ``trend=None`` when no polls are available.
    """
    race = slug_to_race(slug)

    polls_df = db.execute(
        """
        SELECT date, pollster, dem_share, n_sample
        FROM polls
        WHERE LOWER(race) = LOWER(?)
          AND date IS NOT NULL
        ORDER BY date
        """,
        [race],
    ).fetchdf()

    if polls_df.empty:
        return PollTrendResponse(
            race=race,
            slug=slug,
            polls=[],
            trend=None,
        )

    poll_points: list[PollTrendPoll] = [
        PollTrendPoll(
            date=str(row["date"]),
            pollster=row["pollster"] if row["pollster"] else None,
            dem_share=float(row["dem_share"]),
            rep_share=float(1.0 - row["dem_share"]),
            sample_size=int(row["n_sample"]) if not pd.isna(row["n_sample"]) else None,
        )
        for _, row in polls_df.iterrows()
    ]

    trend = _compute_poll_trend(polls_df)
    return PollTrendResponse(race=race, slug=slug, polls=poll_points, trend=trend)


def _compute_poll_trend(polls_df: "pd.DataFrame") -> PollTrend | None:
    """Compute a 30-day rolling weighted moving average over poll data.

    Weights each poll by its sample size.  For each day in the trend output
    we average all polls within the preceding 30 days.  Output dates are
    the unique poll dates (no interpolation) — this keeps the response small
    and avoids inventing data between polls.

    Returns ``None`` if there are fewer than 2 polls (no trend to fit).
    """
    if len(polls_df) < 2:
        return None

    # Parse dates; drop rows with unparseable dates
    polls_df = polls_df.copy()
    polls_df["_dt"] = pd.to_datetime(polls_df["date"], errors="coerce")
    polls_df = polls_df.dropna(subset=["_dt"]).sort_values("_dt")

    if len(polls_df) < 2:
        return None

    # Replace missing sample sizes with the median
    median_n = int(polls_df["n_sample"].median()) if polls_df["n_sample"].notna().any() else 600
    polls_df["_n"] = polls_df["n_sample"].fillna(median_n).astype(float)

    window_days = pd.Timedelta(days=30)

    trend_dates: list[str] = []
    trend_dem: list[float] = []
    trend_rep: list[float] = []

    for _, anchor in polls_df.iterrows():
        anchor_dt = anchor["_dt"]
        # All polls within 30 days before (and including) this poll
        mask = (polls_df["_dt"] >= anchor_dt - window_days) & (polls_df["_dt"] <= anchor_dt)
        window = polls_df[mask]

        total_weight = float(window["_n"].sum())
        if total_weight == 0:
            continue

        dem_avg = float((window["dem_share"] * window["_n"]).sum() / total_weight)
        trend_dates.append(str(anchor["date"]))
        trend_dem.append(round(dem_avg, 4))
        trend_rep.append(round(1.0 - dem_avg, 4))

    if not trend_dates:
        return None

    return PollTrend(dates=trend_dates, dem_trend=trend_dem, rep_trend=trend_rep)


def _format_baseline_label(pres_baseline: float) -> str:
    """Format the presidential baseline as a party-margin label, e.g. 'R+3.2' or 'D+0.5'.

    The label measures how far the 2024 presidential Dem share deviates from 50/50.
    shift = pres_baseline - 0.5; negative shift → Republican advantage → 'R+X'.
    """
    shift = pres_baseline - 0.5
    magnitude = round(abs(shift) * 100, 1)
    if shift < 0:
        return f"R+{magnitude}"
    return f"D+{magnitude}"


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
        source=info.source,
        baseline_year=2024,
        baseline_label=_format_baseline_label(PRES_DEM_SHARE_2024_NATIONAL),
    )


@router.get("/polls", response_model=list[PollRow])
def get_polls(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    race: str | None = Query(None, description="Filter by race label (e.g. '2026 FL Senate')"),
    state: str | None = Query(None, description="Filter by state abbreviation (e.g. 'FL')"),
    cycle: str = Query("2026", description="Poll cycle year"),
):
    """Return available polls from DuckDB polls table."""
    conditions = ["cycle = ?"]
    params: list = [cycle]
    if race:
        conditions.append("LOWER(race) LIKE ?")
        params.append(f"%{race.lower()}%")
    if state:
        conditions.append("geography = ?")
        params.append(state)
    where = " AND ".join(conditions)
    rows = db.execute(f"SELECT * FROM polls WHERE {where} ORDER BY date", params).fetchdf()

    return [
        PollRow(
            race=row["race"],
            geography=row["geography"],
            geo_level=row["geo_level"],
            dem_share=float(row["dem_share"]),
            n_sample=int(row["n_sample"]),
            date=row["date"] if row["date"] else None,
            pollster=row["pollster"] if row["pollster"] else None,
            grade=_lookup_pollster_grade(
                request, row["pollster"] if row["pollster"] else None
            ),
        )
        for _, row in rows.iterrows()
    ]


@router.post("/forecast/poll", response_model=list[ForecastRow])
def update_forecast_with_poll(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Run a Bayesian update: given a new poll, return updated county predictions.

    Uses type-based pipeline if type data is loaded, otherwise falls back to HAC.
    """
    # ── Try type-based prediction first ──────────────────────────────────
    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    type_county_fips = getattr(request.app.state, "type_county_fips", None)

    if (
        type_scores is not None
        and type_covariance is not None
        and type_priors is not None
        and type_county_fips is not None
    ):
        return _forecast_poll_types(
            poll, request, db, type_scores, type_covariance, type_priors, type_county_fips
        )

    # ── Fallback: HAC community-based pipeline ──────────────────────────
    return _forecast_poll_hac(poll, request, db)


def _forecast_poll_types(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection,
    type_scores: np.ndarray,
    type_covariance: np.ndarray,
    type_priors: np.ndarray,
    type_county_fips: list[str],
) -> list[ForecastRow]:
    """Type-based Bayesian update using predict_race from forecast_runner."""
    from src.prediction.forecast_runner import predict_race

    # Tract-level lookups from app.state (no DuckDB query needed)
    tract_states = getattr(request.app.state, "tract_states", {})
    states = [tract_states.get(f, f[:2]) for f in type_county_fips]
    county_names = [f"Tract {f}" for f in type_county_fips]

    # Build county-level priors from Ridge predictions (if loaded at startup)
    ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
    if ridge_priors:
        county_priors = np.array(
            [ridge_priors.get(f, 0.45) for f in type_county_fips]
        )
    else:
        county_priors = None  # falls back to type-mean inside predict_race

    # Apply behavior adjustment for off-cycle races
    behavior_tau = getattr(request.app.state, "behavior_tau", None)
    behavior_delta = getattr(request.app.state, "behavior_delta", None)
    race_str = (poll.race or "").lower()
    is_offcycle = not any(kw in race_str for kw in ["president", "pres"])

    if (behavior_tau is not None and behavior_delta is not None
            and county_priors is not None and is_offcycle
            and type_scores.shape[1] == len(behavior_tau)):
        from src.behavior.voter_behavior import apply_behavior_adjustment
        county_priors = apply_behavior_adjustment(
            county_priors, type_scores, behavior_tau, behavior_delta, is_offcycle=True
        )

    # poll.state is the authoritative source for which state was polled;
    # passing it explicitly avoids the fragile race-string scan.
    result_df = predict_race(
        race=poll.race,
        polls=[(poll.dem_share, poll.n, poll.state)],
        type_scores=type_scores,
        type_covariance=type_covariance,
        type_priors=type_priors,
        county_fips=type_county_fips,
        states=states,
        county_names=county_names,
        state_filter=None,
        county_priors=county_priors,
    )

    # Vote-weighted state-level prediction: SUM(pred * votes) / SUM(votes).
    state_rows = result_df[result_df["state"] == poll.state]
    state_pred_val = None
    if not state_rows.empty:
        tract_votes = getattr(request.app.state, "tract_votes", {})
        weights = state_rows["county_fips"].map(lambda f: tract_votes.get(f, 0)).values.astype(float)
        total_w = weights.sum()
        if total_w > 0:
            state_pred_val = float(
                (state_rows["pred_dem_share"].values * weights).sum() / total_w
            )
        else:
            state_pred_val = float(state_rows["pred_dem_share"].mean())

    results = []
    for _, row in result_df.iterrows():
        results.append(
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=poll.race,
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * _Z90),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=poll.dem_share,
                dominant_type=int(row["dominant_type"]) if "dominant_type" in row.index else None,
            )
        )
    return results


def _run_bayesian_update(
    mu_prior: np.ndarray,
    sigma: np.ndarray,
    w_matrix: np.ndarray,
    poll_means: np.ndarray,
    poll_stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Gaussian Bayesian update for type/community means given polls.

    This is the shared math kernel used by both the type-based and HAC community-based
    forecast paths.  The update follows standard Gaussian conditioning:

        posterior precision = prior precision + W^T R^{-1} W
        posterior mean      = Sigma_post (Sigma_prior^{-1} mu_prior + W^T R^{-1} y)

    where R = diag(sigma_poll^2) is the measurement noise covariance.

    Args:
        mu_prior:   Prior mean vector, shape (K,).
        sigma:      Prior covariance matrix, shape (K, K).
        w_matrix:   Weight matrix, shape (n_polls, K).  Each row is the geographic
                    type/community composition of one polled area (W vector).
        poll_means: Observed Democratic share for each poll, shape (n_polls,).
        poll_stds:  Measurement uncertainty (std) for each poll, shape (n_polls,).
                    Computed from binomial variance: sqrt(p*(1-p)/n).

    Returns:
        Tuple of (mu_post, sigma_post):
            mu_post    — posterior mean, shape (K,).
            sigma_post — posterior covariance, shape (K, K).  Callers that need
                         per-community standard deviations can use sqrt(diag(sigma_post)).
    """
    K = len(mu_prior)
    R = np.diag(poll_stds ** 2)
    # Tikhonov regularization keeps the prior covariance positive definite under
    # floating-point noise.  _MATRIX_JITTER is tiny relative to true eigenvalues.
    sigma_inv = np.linalg.inv(sigma + np.eye(K) * _MATRIX_JITTER)
    sigma_post_inv = sigma_inv + w_matrix.T @ np.linalg.inv(R) @ w_matrix
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (sigma_inv @ mu_prior + w_matrix.T @ np.linalg.solve(R, poll_means))
    return mu_post, sigma_post


def _forecast_poll_hac(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection,
) -> list[ForecastRow]:
    """Original HAC community-based Bayesian update."""
    sigma = request.app.state.sigma
    K = request.app.state.K
    mu_prior = request.app.state.mu_prior
    state_weights = request.app.state.state_weights
    county_weights = request.app.state.county_weights

    if state_weights.empty or county_weights.empty:
        raise HTTPException(status_code=503, detail="Weight matrices not loaded")

    # Build W row for this state
    weight_cols = sorted([c for c in state_weights.columns if c.startswith("community_")])
    state_row = state_weights[state_weights["state_abbr"] == poll.state]
    if state_row.empty:
        raise HTTPException(status_code=404, detail=f"State {poll.state} not found in weights")

    W = state_row[weight_cols].values  # shape (1, K)
    poll_means = np.array([poll.dem_share])
    poll_stds = np.array([np.sqrt(poll.dem_share * (1 - poll.dem_share) / poll.n)])

    mu_post, sigma_post = _run_bayesian_update(mu_prior, sigma, W, poll_means, poll_stds)

    # Map community posteriors → county predictions via hard assignment
    version_id = request.app.state.version_id
    county_info = db.execute(
        """SELECT c.county_fips, c.county_name, c.state_abbr, ca.community_id
           FROM counties c
           JOIN community_assignments ca ON c.county_fips = ca.county_fips AND ca.version_id = ?
           ORDER BY c.county_fips""",
        [version_id],
    ).fetchdf()

    state_pred_val = float((W @ mu_post).item())
    community_stds = np.sqrt(np.diag(sigma_post))

    results = []
    for _, row in county_info.iterrows():
        cid = int(row["community_id"])
        if 0 <= cid < K:
            pred = float(mu_post[cid])
            std = float(community_stds[cid])
        else:
            pred = float(np.mean(mu_post))
            std = 0.05
        results.append(
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state_abbr"],
                race=poll.race,
                pred_dem_share=pred,
                pred_std=std,
                pred_lo90=pred - _Z90 * std,
                pred_hi90=pred + _Z90 * std,
                state_pred=state_pred_val,
                poll_avg=poll.dem_share,
            )
        )

    return results


@router.post("/forecast/polls", response_model=MultiPollResponse)
def update_forecast_with_multi_polls(
    body: MultiPollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Feed multiple polls from a historical cycle through time-decayed,
    quality-weighted Bayesian update. Returns county-level predictions.
    """
    # Load polls from DuckDB instead of CSV
    rows_df = db.execute(
        "SELECT * FROM polls WHERE cycle=? ORDER BY date",
        [body.cycle],
    ).fetchdf()

    if body.state:
        rows_df = rows_df[rows_df["geography"] == body.state]
    if body.race:
        rows_df = rows_df[rows_df["race"].str.contains(body.race, case=False, na=False)]

    if rows_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No matching polls for cycle={body.cycle}, state={body.state}, race={body.race}",
        )

    from src.propagation.poll_weighting import (
        aggregate_polls,
        apply_all_weights,
        election_day_for_cycle,
    )
    from src.propagation.propagate_polls import PollObservation
    polls = [
        PollObservation(
            geography=row["geography"],
            dem_share=float(row["dem_share"]),
            n_sample=int(row["n_sample"]),
            race=row["race"],
            date=row["date"] or "",
            pollster=row["pollster"] or "",
            geo_level=row["geo_level"],
        )
        for _, row in rows_df.iterrows()
    ]
    notes = list(rows_df["notes"].fillna(""))

    # Apply weighting
    ref_date = election_day_for_cycle(body.cycle)
    weighted = apply_all_weights(
        polls,
        reference_date=ref_date,
        half_life_days=body.half_life_days,
        poll_notes=notes if body.apply_quality else None,
        apply_quality=body.apply_quality,
    )

    # Build metadata
    dates = sorted(p.date for p in polls if p.date)
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "unknown"
    effective_n_total = sum(p.n_sample for p in weighted)

    # Build poll list for stacked Bayesian update — do not collapse.
    # Collapsing loses geographic information when polls cover different
    # states and conflates structurally distinct signals into one scalar.
    # Apply section weight to poll effective N (Option A from spec).
    sw = body.section_weights
    race_polls = [
        (p.dem_share, max(1, int(p.n_sample * sw.state_polls)), p.geography)
        for p in weighted
        if p.geo_level == "state"
    ]

    if not race_polls:
        raise HTTPException(status_code=400, detail="No state-level polls after filtering")

    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    type_county_fips = getattr(request.app.state, "type_county_fips", None)

    if all(x is not None for x in [type_scores, type_covariance, type_priors, type_county_fips]):
        from src.prediction.forecast_runner import predict_race

        # Tract-level lookups from app.state (no DuckDB query needed)
        tract_states = getattr(request.app.state, "tract_states", {})
        fips_list = type_county_fips
        states_list = [tract_states.get(f, f[:2]) for f in fips_list]
        names_list = [f"Tract {f}" for f in fips_list]

        ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
        county_priors = (
            np.array([ridge_priors.get(f, 0.45) for f in fips_list])
            if ridge_priors else None
        )

        # Apply behavior adjustment for off-cycle races
        behavior_tau = getattr(request.app.state, "behavior_tau", None)
        behavior_delta = getattr(request.app.state, "behavior_delta", None)
        race_str = (body.race or "").lower()
        is_offcycle = not any(kw in race_str for kw in ["president", "pres"])

        if (behavior_tau is not None and behavior_delta is not None
                and county_priors is not None and is_offcycle
                and type_scores.shape[1] == len(behavior_tau)):
            from src.behavior.voter_behavior import apply_behavior_adjustment
            county_priors = apply_behavior_adjustment(
                county_priors, type_scores, behavior_tau, behavior_delta, is_offcycle=True
            )

        result_df = predict_race(
            race=body.race or race_polls[0][2],
            polls=race_polls,
            type_scores=type_scores,
            type_covariance=type_covariance,
            type_priors=type_priors,
            county_fips=fips_list,
            states=states_list,
            county_names=names_list,
            county_priors=county_priors,
            prior_weight=sw.model_prior,
        )

        state_pred_val = None
        if body.state:
            state_rows = result_df[result_df["state"] == body.state]
            if not state_rows.empty:
                # Vote-weighted average: SUM(pred * votes) / SUM(votes).
                tract_votes = getattr(request.app.state, "tract_votes", {})
                weights = state_rows["county_fips"].map(lambda f: tract_votes.get(f, 0)).values.astype(float)
                total_w = weights.sum()
                if total_w > 0:
                    state_pred_val = float(
                        (state_rows["pred_dem_share"].values * weights).sum() / total_w
                    )
                else:
                    state_pred_val = float(state_rows["pred_dem_share"].mean())

        county_results = [
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=body.race or race_polls[0][2],
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * _Z90),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=float(np.mean([p[0] for p in race_polls])),
                dominant_type=int(row["dominant_type"]) if "dominant_type" in row.index else None,
            )
            for _, row in result_df.iterrows()
        ]
    else:
        # HAC fallback: multi-poll stacking not yet supported; collapse to one
        # effective poll as an approximation until HAC gains stacking support.
        combined_share, combined_n = aggregate_polls(weighted)
        race_label = body.race or polls[0].race
        single_poll = PollInput(
            state=body.state, race=race_label, dem_share=combined_share, n=combined_n
        )
        county_results = _forecast_poll_hac(single_poll, request, db)

    return MultiPollResponse(
        counties=county_results,
        polls_used=len(polls),
        date_range=date_range,
        effective_n_total=effective_n_total,
    )


# ---------------------------------------------------------------------------
# Forecast changelog
# ---------------------------------------------------------------------------

SNAPSHOTS_DIR = Path(__file__).resolve().parents[2] / "data" / "forecast_snapshots"
COMPARISONS_FILE = Path(__file__).resolve().parents[2] / "data" / "comparisons" / "ratings_2026.json"

# Races that have real poll-adjusted predictions (not just baseline copies).
# Only show changes for these to avoid cluttering with 60+ identical baseline races.
TRACKED_RACES = {
    "2026 FL Senate", "2026 FL Governor", "2026 GA Senate", "2026 GA Governor",
    "2026 IA Senate", "2026 ME Senate", "2026 MI Senate", "2026 MI Governor",
    "2026 MN Senate", "2026 MN Governor", "2026 NC Senate", "2026 NH Senate",
    "2026 NH Governor", "2026 OH Governor", "2026 OR Senate",
    "2026 PA Governor", "2026 TX Senate", "2026 TX Governor",
    "2026 WI Governor", "2026 AL Senate", "2026 AL Governor",
}


@router.get("/forecast/changelog", response_model=ChangelogResponse)
def get_forecast_changelog() -> ChangelogResponse:
    """Return a changelog of forecast prediction changes between weekly snapshots.

    Snapshots are stored as JSON files in ``data/forecast_snapshots/``.
    Each file contains ``{date, predictions: {race: avg_dem_share}, note?}``.
    Entries are returned newest-first.
    """
    import json
    import math

    if not SNAPSHOTS_DIR.exists():
        return ChangelogResponse(entries=[], current_snapshot_date=None)

    snapshot_files = sorted(SNAPSHOTS_DIR.glob("*.json"))
    if not snapshot_files:
        return ChangelogResponse(entries=[], current_snapshot_date=None)

    # Load all snapshots ordered by date
    snapshots: list[dict] = []
    for f in snapshot_files:
        try:
            data = json.loads(f.read_text())
            snapshots.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    if not snapshots:
        return ChangelogResponse(entries=[], current_snapshot_date=None)

    entries: list[ChangelogEntry] = []

    # First snapshot = baseline entry (no diffs, just the initial state)
    first = snapshots[0]
    first_preds = first.get("predictions", {})
    initial_diffs = [
        ChangelogRaceDiff(race=race, before=None, after=val, delta=None)
        for race, val in sorted(first_preds.items())
        if race in TRACKED_RACES
    ]
    entries.append(ChangelogEntry(
        date=first.get("date", "unknown"),
        note=first.get("note", "Initial baseline"),
        diffs=initial_diffs,
    ))

    # Subsequent snapshots = diffs against previous
    for i in range(1, len(snapshots)):
        prev_preds = snapshots[i - 1].get("predictions", {})
        curr_preds = snapshots[i].get("predictions", {})
        curr_date = snapshots[i].get("date", "unknown")
        curr_note = snapshots[i].get("note")

        diffs: list[ChangelogRaceDiff] = []
        all_races = sorted((set(prev_preds) | set(curr_preds)) & TRACKED_RACES)

        for race in all_races:
            b = prev_preds.get(race)
            a = curr_preds.get(race)

            if b is not None and a is not None:
                delta = a - b
                if abs(delta) < 0.002:  # < 0.2pp — not meaningful
                    continue
            else:
                delta = None

            diffs.append(ChangelogRaceDiff(
                race=race, before=b, after=a, delta=delta,
            ))

        if diffs:
            entries.append(ChangelogEntry(
                date=curr_date, note=curr_note, diffs=diffs,
            ))

    # Return newest-first
    entries.reverse()

    return ChangelogResponse(
        entries=entries,
        current_snapshot_date=snapshots[-1].get("date"),
    )


# ---------------------------------------------------------------------------
# Comparisons endpoint
# ---------------------------------------------------------------------------

@router.get("/forecast/comparisons")
def get_forecast_comparisons(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> dict:
    """Return WetherVane predictions alongside ratings from major forecasters.

    Reads manual ratings from data/comparisons/ratings_2026.json and merges
    with live state-level predictions from the database. Races with no manual
    ratings entry still appear with WetherVane predictions only.
    """
    import json as _json

    version_id = request.app.state.version_id

    # Load manual ratings file
    comparisons_data: dict = {}
    if COMPARISONS_FILE.exists():
        try:
            comparisons_data = _json.loads(COMPARISONS_FILE.read_text())
        except (_json.JSONDecodeError, OSError):
            comparisons_data = {}

    sources = comparisons_data.get("sources", [])
    last_updated = comparisons_data.get("last_updated")
    manual_ratings: dict[str, dict] = comparisons_data.get("ratings", {})

    # Fetch vote-weighted state-level predictions for all races.
    #
    # IMPORTANT: filter counties to the race's own state via the `races` table.
    # The predictions table stores county rows for ALL 3,154 US counties for every
    # race, with non-state counties carrying the national baseline value (~0.318).
    # Without the state filter, the vote-weighted average is diluted by ~3,100
    # out-of-state baseline rows and converges to ~48.3% D (R+1.7pp) for almost
    # every race — the "48.3% D · R+1.7" bug reported in issues #16 and #17.
    # Joining through races.state ensures only in-state counties contribute.
    try:
        rows = db.execute(
            """
            SELECT
                p.race,
                CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                          / SUM(COALESCE(c.total_votes_2024, 0))
                     ELSE AVG(p.pred_dem_share)
                END AS state_pred,
                AVG(p.pred_std) AS avg_std,
                COUNT(*) AS n_counties
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            JOIN races r ON p.race = r.race_id
            WHERE p.version_id = ?
              AND p.race != 'baseline'
              AND c.state_abbr = r.state
            GROUP BY p.race
            ORDER BY p.race
            """,
            [version_id],
        ).fetchall()
    except Exception:
        rows = []

    # Build per-race predictions dict
    wv_predictions: dict[str, dict] = {}
    for row in rows:
        race_id, state_pred, avg_std, n_counties = row
        pred_val = None if state_pred is None else float(state_pred)
        rating = marginToRating(pred_val) if pred_val is not None else None
        wv_predictions[race_id] = {
            "pred_dem_share": pred_val,
            "pred_std": None if avg_std is None else float(avg_std),
            "rating": rating,
            "n_counties": int(n_counties),
            "slug": race_to_slug(race_id),
        }

    # Merge: all races that appear in either predictions or manual ratings
    all_races = sorted(set(wv_predictions.keys()) | set(manual_ratings.keys()))

    race_rows = []
    for race_id in all_races:
        wv = wv_predictions.get(race_id, {})
        manual = manual_ratings.get(race_id, {})

        # Parse race metadata from ID (e.g. "2026 FL Senate")
        parts = race_id.split()
        year = int(parts[0]) if parts and parts[0].isdigit() else 2026
        state_abbr = parts[1] if len(parts) > 1 else ""
        race_type = " ".join(parts[2:]) if len(parts) > 2 else ""

        race_rows.append({
            "race_id": race_id,
            "slug": wv.get("slug") or race_to_slug(race_id),
            "year": year,
            "state_abbr": state_abbr,
            "race_type": race_type,
            "wethervane": {
                "pred_dem_share": wv.get("pred_dem_share"),
                "pred_std": wv.get("pred_std"),
                "rating": wv.get("rating"),
                "n_counties": wv.get("n_counties"),
            },
            "cook": manual.get("cook"),
            "sabato": manual.get("sabato"),
            "inside": manual.get("inside"),
        })

    return {
        "last_updated": last_updated,
        "sources": sources,
        "races": race_rows,
    }


def marginToRating(dem_share: float) -> str:
    """Python equivalent of the frontend marginToRating for API use."""
    margin = dem_share - 0.5
    abs_margin = abs(margin)
    if abs_margin < 0.03:
        return "tossup"
    if margin > 0:
        if abs_margin >= 0.15:
            return "safe_d"
        if abs_margin >= 0.08:
            return "likely_d"
        return "lean_d"
    if abs_margin >= 0.15:
        return "safe_r"
    if abs_margin >= 0.08:
        return "likely_r"
    return "lean_r"
