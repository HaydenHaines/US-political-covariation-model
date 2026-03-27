from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.db import get_db
from api.models import (
    EmbedResponse,
    ForecastRow,
    MultiPollInput,
    MultiPollResponse,
    PollInput,
    PollRow,
    RaceDetail,
    RacePoll,
    TypeBreakdown,
)

log = logging.getLogger(__name__)

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


@router.get("/forecast/race/{slug}", response_model=RaceDetail)
def get_race_detail(
    slug: str,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> RaceDetail:
    """Return state-level prediction, polls, and type breakdown for a single race."""
    version_id = request.app.state.version_id
    race = slug_to_race(slug)

    # Validate the race exists
    exists = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE version_id = ? AND race = ?",
        [version_id, race],
    ).fetchone()
    if not exists or exists[0] == 0:
        raise HTTPException(status_code=404, detail=f"Race '{race}' not found")

    # Parse state_abbr and race_type from slug parts
    parts = slug.split("-")
    state_abbr = parts[1].upper() if len(parts) >= 2 else "??"
    race_type = " ".join(p.capitalize() for p in parts[2:]) if len(parts) >= 3 else "Unknown"
    year = int(parts[0]) if parts[0].isdigit() else 2026

    # State-level prediction: mean of county pred_dem_share for this race+state
    pred_row = db.execute(
        """
        SELECT AVG(p.pred_dem_share) AS state_pred, COUNT(*) AS n_counties
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
        """,
        [version_id, race, state_abbr],
    ).fetchone()

    prediction = None if (pred_row is None or pred_row[0] is None) else float(pred_row[0])
    n_counties = int(pred_row[1]) if pred_row else 0

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
        )
        for _, row in polls_df.iterrows()
    ]

    # Type breakdown: top 5 types by county count in this state for this race
    breakdown_df = db.execute(
        """
        SELECT
            cta.dominant_type AS type_id,
            t.display_name,
            COUNT(*) AS n_counties,
            AVG(p.pred_dem_share) AS mean_pred_dem_share
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        JOIN county_type_assignments cta ON p.county_fips = cta.county_fips
        JOIN types t ON cta.dominant_type = t.type_id
        WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
        GROUP BY cta.dominant_type, t.display_name
        ORDER BY COUNT(*) DESC
        LIMIT 5
        """,
        [version_id, race, state_abbr],
    ).fetchdf()

    type_breakdown: list[TypeBreakdown] = [
        TypeBreakdown(
            type_id=int(row["type_id"]),
            display_name=row["display_name"],
            n_counties=int(row["n_counties"]),
            mean_pred_dem_share=None if pd.isna(row["mean_pred_dem_share"]) else float(row["mean_pred_dem_share"]),
        )
        for _, row in breakdown_df.iterrows()
    ]

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
    )


@router.get("/embed/{slug}", response_model=EmbedResponse, tags=["embed"])
def get_embed_metadata(
    slug: str,
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> EmbedResponse:
    """Return embed metadata for a single race, including a ready-to-paste iframe snippet.

    Callers that only need the raw prediction data should use
    ``GET /forecast/race/{slug}`` instead.  This endpoint is purpose-built for
    CMS integrations that want a fully-formed iframe snippet without
    constructing the URL themselves.
    """
    site_url = "https://wethervane.hhaines.duckdns.org"
    version_id = request.app.state.version_id
    race = slug_to_race(slug)

    exists = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE version_id = ? AND race = ?",
        [version_id, race],
    ).fetchone()
    if not exists or exists[0] == 0:
        raise HTTPException(status_code=404, detail=f"Race '{race}' not found")

    parts = slug.split("-")
    state_abbr = parts[1].upper() if len(parts) >= 2 else "??"
    race_type = " ".join(p.capitalize() for p in parts[2:]) if len(parts) >= 3 else "Unknown"
    year = int(parts[0]) if parts[0].isdigit() else 2026

    pred_row = db.execute(
        """
        SELECT AVG(p.pred_dem_share) AS state_pred, COUNT(*) AS n_counties
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE p.version_id = ? AND p.race = ? AND c.state_abbr = ?
        """,
        [version_id, race, state_abbr],
    ).fetchone()

    dem_pct = None if (pred_row is None or pred_row[0] is None) else float(pred_row[0])
    n_counties = int(pred_row[1]) if pred_row else 0

    # Partisan lean label and color
    if dem_pct is not None:
        margin = abs(dem_pct - 0.5) * 100
        if dem_pct > 0.5:
            lean_label = f"D+{margin:.1f}"
            lean_color = "#2166ac"
        else:
            lean_label = f"R+{margin:.1f}"
            lean_color = "#d73027"
    else:
        lean_label = "No prediction"
        lean_color = "#666666"

    race_title = f"{year} {state_abbr} {race_type}"
    embed_url = f"{site_url}/embed/{slug}"
    iframe_snippet = (
        f'<iframe src="{embed_url}" '
        f'width="400" height="200" frameborder="0" '
        f'style="border:none;max-width:100%;" '
        f'title="{race_title} forecast — WetherVane">'
        f"</iframe>"
    )

    return EmbedResponse(
        slug=slug,
        race_title=race_title,
        lean_label=lean_label,
        lean_color=lean_color,
        dem_pct=dem_pct,
        rep_pct=(1.0 - dem_pct) if dem_pct is not None else None,
        n_counties=n_counties,
        iframe_snippet=iframe_snippet,
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


def _get_crosstab_w_override(
    poll_id: str,
    state_abbr: str,
    db: duckdb.DuckDBPyConnection,
    request: Request,
    type_scores: np.ndarray,
    county_fips_list: list[str],
    states_list: list[str],
) -> np.ndarray | None:
    """Build a crosstab-adjusted W override for a poll if crosstab data is available.

    Returns a normalised ndarray of shape (J,) if crosstab data exists and the
    affinity index is loaded, otherwise returns None (caller uses state-mean W).

    This is intentionally defensive: any failure returns None so the caller
    falls back to the state-mean W rather than crashing the request.
    """
    from src.propagation.crosstab_w_builder import compute_state_baseline_w, construct_w_row

    affinity = getattr(request.app.state, "crosstab_affinity", None)
    if affinity is None:
        return None  # Affinity index not loaded at startup

    state_means_all: dict[str, dict[str, float]] = getattr(request.app.state, "crosstab_state_means", {})
    state_means = state_means_all.get(state_abbr)
    if not state_means:
        return None  # No demographic baseline for this state

    try:
        crosstab_rows_df = db.execute(
            """SELECT demographic_group, group_value, pct_of_sample
               FROM poll_crosstabs
               WHERE poll_id = ? AND pct_of_sample IS NOT NULL""",
            [poll_id],
        ).fetchdf()
    except Exception as exc:
        log.debug("poll_crosstabs query failed for poll_id=%s: %s", poll_id, exc)
        return None

    if crosstab_rows_df.empty:
        return None  # No crosstab data for this poll

    crosstab_dicts = crosstab_rows_df.to_dict("records")

    # Population data for baseline W — use pop_total from county_features if available.
    # Fall back to uniform weights if not present.
    try:
        pop_df = db.execute(
            "SELECT county_fips, pop_total FROM county_demographics ORDER BY county_fips"
        ).fetchdf()
        pop_map = dict(zip(pop_df["county_fips"], pop_df["pop_total"].astype(float)))
        county_pops = np.array([max(pop_map.get(f, 1.0), 1.0) for f in county_fips_list])
    except Exception:
        county_pops = np.ones(len(county_fips_list))

    state_mask = np.array([s == state_abbr for s in states_list])
    if not state_mask.any():
        return None

    try:
        w_base = compute_state_baseline_w(
            np.abs(type_scores),  # abs scores for population weighting (same as predict_race)
            county_pops,
            state_mask,
        )
        w_override = construct_w_row(
            poll_crosstabs=crosstab_dicts,
            state_baseline_w=w_base,
            affinity_index=affinity,
            state_demographic_means=state_means,
        )
    except Exception as exc:
        log.warning(
            "construct_w_row failed for poll_id=%s state=%s: %s",
            poll_id, state_abbr, exc,
        )
        return None

    return w_override


def _forecast_poll_types(
    poll: PollInput,
    request: Request,
    db: duckdb.DuckDBPyConnection,
    type_scores: np.ndarray,
    type_covariance: np.ndarray,
    type_priors: np.ndarray,
    type_county_fips: list[str],
) -> list[ForecastRow]:
    """Type-based Bayesian update using predict_race from predict_2026_types."""
    from src.prediction.predict_2026_types import predict_race

    # Get county names from DB
    county_info = db.execute(
        """SELECT c.county_fips, c.county_name, c.state_abbr
           FROM counties c ORDER BY c.county_fips""",
        [],
    ).fetchdf()
    name_map = dict(zip(county_info["county_fips"], county_info["county_name"]))
    state_map = dict(zip(county_info["county_fips"], county_info["state_abbr"]))

    states = [state_map.get(f, f[:2]) for f in type_county_fips]
    county_names = [name_map.get(f, "") for f in type_county_fips]

    # Build county-level priors from Ridge predictions (if loaded at startup)
    ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
    if ridge_priors:
        county_priors = np.array(
            [ridge_priors.get(f, 0.45) for f in type_county_fips]
        )
    else:
        county_priors = None  # falls back to type-mean inside predict_race

    # Look up this poll's ID so we can query poll_crosstabs.
    # poll_id is not part of PollInput (ad-hoc polls don't have one), so we
    # attempt a best-effort lookup by matching on race+geography+dem_share.
    # If no match, w_override is None and we fall back to state-mean W.
    poll_id_row = db.execute(
        """SELECT poll_id FROM polls
           WHERE LOWER(race) = LOWER(?) AND geography = ?
           ORDER BY date DESC LIMIT 1""",
        [poll.race, poll.state],
    ).fetchone()
    poll_id = poll_id_row[0] if poll_id_row else None

    w_override = None
    if poll_id is not None:
        w_override = _get_crosstab_w_override(
            poll_id=poll_id,
            state_abbr=poll.state,
            db=db,
            request=request,
            type_scores=type_scores,
            county_fips_list=type_county_fips,
            states_list=states,
        )
        if w_override is not None:
            log.info(
                "Using crosstab-adjusted W for poll_id=%s race=%s state=%s",
                poll_id, poll.race, poll.state,
            )

    # poll.state is the authoritative source for which state was polled;
    # passing it explicitly avoids the fragile race-string scan.
    result_df = predict_race(
        race=poll.race,
        polls=[(poll.dem_share, poll.n, poll.state, w_override)],
        type_scores=type_scores,
        type_covariance=type_covariance,
        type_priors=type_priors,
        county_fips=type_county_fips,
        states=states,
        county_names=county_names,
        state_filter=None,
        county_priors=county_priors,
    )

    # Compute state-level prediction as the mean of polled-state county
    # predictions — consistent with how the HAC pipeline populates state_pred.
    state_rows = result_df[result_df["state"] == poll.state]
    state_pred_val = float(state_rows["pred_dem_share"].mean()) if not state_rows.empty else None

    results = []
    for _, row in result_df.iterrows():
        results.append(
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=poll.race,
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * 1.645),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=poll.dem_share,
            )
        )
    return results


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
    y = np.array([poll.dem_share])
    sigma_poll = np.array([np.sqrt(poll.dem_share * (1 - poll.dem_share) / poll.n)])

    # Bayesian update
    R = np.diag(sigma_poll ** 2)
    sigma_inv = np.linalg.inv(sigma + np.eye(K) * 1e-8)
    sigma_post_inv = sigma_inv + W.T @ np.linalg.inv(R) @ W
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (sigma_inv @ mu_prior + W.T @ np.linalg.solve(R, y))

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
                pred_lo90=pred - 1.645 * std,
                pred_hi90=pred + 1.645 * std,
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
    # Build the weighted poll list including geography, retaining poll_id for crosstab lookup.
    # rows_df is indexed by original CSV row; we need to align poll_id with the weighted list.
    # The weighted list preserves the ordering of polls (PollObservation order = rows_df order).
    # Build a poll_id list parallel to `polls` (which mirrors rows_df ordering).
    poll_ids_by_index = list(rows_df["poll_id"]) if "poll_id" in rows_df.columns else [None] * len(polls)

    weighted_state_indices = [
        i for i, p in enumerate(weighted) if p.geo_level == "state"
    ]
    race_polls_base = [
        (weighted[i].dem_share, max(1, int(weighted[i].n_sample * sw.state_polls)), weighted[i].geography)
        for i in weighted_state_indices
    ]

    if not race_polls_base:
        raise HTTPException(status_code=400, detail="No state-level polls after filtering")

    type_scores = getattr(request.app.state, "type_scores", None)
    type_covariance = getattr(request.app.state, "type_covariance", None)
    type_priors = getattr(request.app.state, "type_priors", None)
    type_county_fips = getattr(request.app.state, "type_county_fips", None)

    if all(x is not None for x in [type_scores, type_covariance, type_priors, type_county_fips]):
        from src.prediction.predict_2026_types import predict_race

        county_info = db.execute(
            "SELECT county_fips, county_name, state_abbr FROM counties ORDER BY county_fips"
        ).fetchdf()
        name_map = dict(zip(county_info["county_fips"], county_info["county_name"]))
        state_map = dict(zip(county_info["county_fips"], county_info["state_abbr"]))

        fips_list = type_county_fips
        states_list = [state_map.get(f, f[:2]) for f in fips_list]
        names_list = [name_map.get(f, "") for f in fips_list]

        ridge_priors: dict[str, float] = getattr(request.app.state, "ridge_priors", {})
        county_priors = (
            np.array([ridge_priors.get(f, 0.45) for f in fips_list])
            if ridge_priors else None
        )

        # Augment each poll tuple with a crosstab W override when available.
        # Falls back to None (state-mean W) when no crosstab data exists.
        race_polls: list[tuple] = []
        for idx, (dem_share, n, state_abbr) in zip(weighted_state_indices, race_polls_base):
            pid = poll_ids_by_index[idx] if idx < len(poll_ids_by_index) else None
            w_override = None
            if pid is not None:
                w_override = _get_crosstab_w_override(
                    poll_id=pid,
                    state_abbr=state_abbr,
                    db=db,
                    request=request,
                    type_scores=type_scores,
                    county_fips_list=fips_list,
                    states_list=states_list,
                )
                if w_override is not None:
                    log.info(
                        "Multi-poll: using crosstab-adjusted W for poll_id=%s state=%s",
                        pid, state_abbr,
                    )
            race_polls.append((dem_share, n, state_abbr, w_override))

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
                state_pred_val = float(state_rows["pred_dem_share"].mean())

        county_results = [
            ForecastRow(
                county_fips=row["county_fips"],
                county_name=row["county_name"] if row["county_name"] else None,
                state_abbr=row["state"],
                race=body.race or race_polls[0][2],
                pred_dem_share=float(row["pred_dem_share"]),
                pred_std=float(row["ci_upper"] - row["ci_lower"]) / (2 * 1.645),
                pred_lo90=float(row["ci_lower"]),
                pred_hi90=float(row["ci_upper"]),
                state_pred=state_pred_val,
                poll_avg=float(np.mean([p[0] for p in race_polls])),
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
