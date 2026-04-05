"""Race detail endpoints: single race view, history, poll trend."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import get_db
from api.models import (
    HistoricalContext,
    LastRaceResult,
    PollConfidence,
    PollTrend,
    PollTrendPoll,
    PollTrendResponse,
    PresidentialResult,
    RaceDetail,
    RacePoll,
    TypeBreakdown,
)

from ._helpers import (
    _DEFAULT_SAMPLE_SIZE,
    _HISTORICAL_RESULTS,
    _STATE_STD_FALLBACK,
    _VOTE_WEIGHTED_STATE_PRED_SQL,
    _Z90,
    _compute_state_std,
    _get_std_floor,
    _lookup_pollster_grade,
    slug_to_race,
)

router = APIRouter(tags=["forecast"])

# Methodology keywords in notes field, in priority order.
# All three poll sources (Wikipedia, 270toWin, RealClearPolling) use LV/RV labels.
_METHOD_PATTERNS: list[str] = ["LV", "RV"]


def _infer_methodology(notes: str | None) -> str:
    """Return the first matching methodology keyword found in poll notes.

    Returns 'Unknown' when neither LV nor RV appears in the notes string,
    or when notes is None.  This handles polls from sources that do not
    record the voter screen in their notes.
    """
    if not notes:
        return "Unknown"
    for pattern in _METHOD_PATTERNS:
        if pattern in notes:
            return pattern
    return "Unknown"


def _compute_poll_confidence(polls_df: "pd.DataFrame") -> PollConfidence:
    """Derive poll source diversity metrics and a confidence label from a polls DataFrame.

    The DataFrame is expected to have 'pollster' and 'notes' columns, matching
    what the polls table provides.  Empty or missing values are handled gracefully.

    Confidence thresholds:
    - "High":   3+ distinct pollsters AND 2+ distinct methodologies
    - "Medium": 2+ distinct pollsters OR 2+ distinct methodologies
    - "Low":    fewer than 2 pollsters (includes the zero-polls case)
    """
    if polls_df.empty:
        return PollConfidence(
            n_polls=0,
            n_pollsters=0,
            n_methodologies=0,
            label="Low",
            tooltip="No polls",
        )

    n_polls = len(polls_df)

    # Count distinct non-null pollster names
    n_pollsters = int(polls_df["pollster"].dropna().nunique())

    # Infer methodology from notes and count distinct values
    methodologies = polls_df["notes"].apply(_infer_methodology)
    n_methodologies = int(methodologies.nunique())

    # Derive label from diversity thresholds
    if n_pollsters >= 3 and n_methodologies >= 2:
        label = "High"
    elif n_pollsters >= 2 or n_methodologies >= 2:
        label = "Medium"
    else:
        label = "Low"

    pollster_word = "pollster" if n_pollsters == 1 else "pollsters"
    method_word = "method" if n_methodologies == 1 else "methods"
    poll_word = "poll" if n_polls == 1 else "polls"
    tooltip = f"{n_pollsters} {pollster_word} · {n_methodologies} {method_word} · {n_polls} {poll_word}"

    return PollConfidence(
        n_polls=n_polls,
        n_pollsters=n_pollsters,
        n_methodologies=n_methodologies,
        label=label,
        tooltip=tooltip,
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
            {_VOTE_WEIGHTED_STATE_PRED_SQL} AS state_pred,
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
    # then applies an empirical floor from the 2022 backtest by race type:
    #   Senate:   3.7pp floor (28-state RMSE from backtest)
    #   Governor: 5.5pp floor (competitive-state RMSE from backtest)
    # This prevents false precision when all counties happen to agree.
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
            state_std = _compute_state_std(preds, votes, prediction, race_type=race_type)
        else:
            # Single-county or no data: use the empirical floor, not a generic fallback.
            # We take max of fallback and floor to avoid reporting less uncertainty
            # than the race type's backtest errors warrant.
            state_std = max(_STATE_STD_FALLBACK, _get_std_floor(race_type))

        state_lo90 = prediction - _Z90 * state_std
        state_hi90 = prediction + _Z90 * state_std

    # Also get the other mode's prediction for comparison
    other_mode = "national" if forecast_mode == "local" else "local"
    other_pred_row = None
    if _has_mode:
        other_pred_row = db.execute(
            f"""
            SELECT
                {_VOTE_WEIGHTED_STATE_PRED_SQL} AS state_pred
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

    # Polls for this race — fetch notes so we can derive methodology for confidence scoring
    polls_df = db.execute(
        """
        SELECT date, pollster, dem_share, n_sample, notes
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

    poll_confidence = _compute_poll_confidence(polls_df)

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
        poll_confidence=poll_confidence,
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
                f"No historical data for race '{slug}'. "
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
    median_n = int(polls_df["n_sample"].median()) if polls_df["n_sample"].notna().any() else _DEFAULT_SAMPLE_SIZE
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
