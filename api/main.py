# api/main.py
"""FastAPI application factory for the WetherVane API."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.cache import forecast_cache, make_cache_key
from api.routers import cache as cache_router
from api.routers import (
    candidates,
    communities,
    counties,
    forecast,
    governor,
    historical,
    meta,
    pollsters,
    senate,
    tracts,
)
from src.propagation.crosstab_w_builder import CROSSTAB_DIMENSION_MAP, build_affinity_index

log = logging.getLogger(__name__)

_FLAT_DEM_SHARE_PRIOR = 0.45  # uninformative prior when no historical community data is available


def _get_current_version_id(db: duckdb.DuckDBPyConnection) -> str:
    row = db.execute(
        "SELECT version_id FROM model_versions WHERE role = 'current' LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    # Fallback: latest alphabetically
    row = db.execute(
        "SELECT version_id FROM model_versions ORDER BY version_id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else "unknown"


def _load_sigma(db: duckdb.DuckDBPyConnection, version_id: str) -> tuple[np.ndarray, int]:
    """Load K×K sigma matrix from community_sigma table. Returns (sigma, K)."""
    rows = db.execute(
        """SELECT community_id_row, community_id_col, sigma_value
           FROM community_sigma WHERE version_id = ?
           ORDER BY community_id_row, community_id_col""",
        [version_id],
    ).fetchdf()
    if rows.empty:
        log.warning("community_sigma table is empty for version %s", version_id)
        return np.eye(1), 1
    K = int(max(rows["community_id_row"].max(), rows["community_id_col"].max())) + 1
    sigma = np.zeros((K, K))
    for _, r in rows.iterrows():
        sigma[int(r["community_id_row"]), int(r["community_id_col"])] = float(r["sigma_value"])
    return sigma, K


def _load_mu_prior(db: duckdb.DuckDBPyConnection, version_id: str, K: int) -> np.ndarray:
    """Compute community-level prior from predictions table (mean pred_dem_share per community)."""
    rows = db.execute(
        """SELECT ca.community_id, AVG(p.pred_dem_share) as mean_share
           FROM predictions p
           JOIN community_assignments ca
             ON p.county_fips = ca.county_fips AND p.version_id = ca.version_id
           WHERE p.version_id = ?
           GROUP BY ca.community_id
           ORDER BY ca.community_id""",
        [version_id],
    ).fetchdf()
    mu = np.full(K, _FLAT_DEM_SHARE_PRIOR)
    for _, r in rows.iterrows():
        idx = int(r["community_id"])
        if 0 <= idx < K:
            mu[idx] = float(r["mean_share"])
    return mu


def _load_tract_type_data(
    project_root: Path,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Load tract-level type data from the current J=100 model.

    Returns (type_scores, tract_fips, type_covariance, type_priors).
    type_scores: (N_tracts, J) numpy array of soft membership scores.
    tract_fips: list of N_tracts GEOID strings.
    type_covariance: (J, J) Ledoit-Wolf regularized covariance.
    type_priors: (J,) mean Dem share per type.
    """
    communities_dir = project_root / "data" / "communities"
    covariance_dir = project_root / "data" / "covariance"

    # Assignments: deduplicated tract GEOIDs with J=100 type scores.
    # Source: data/communities/tract_type_assignments.parquet (T.3 migration output).
    # NOTE: Stale J=130 artifacts moved to data/tracts/_deprecated_j130/ (S508).
    assignments = pd.read_parquet(communities_dir / "tract_type_assignments.parquet")
    assignments = assignments.drop_duplicates(subset="tract_geoid")
    score_cols = sorted(
        [c for c in assignments.columns if c.startswith("type_") and c.endswith("_score")],
        key=lambda x: int(x.split("_")[1]),
    )
    type_scores = assignments[score_cols].values.astype(float)  # (N, J)
    tract_fips = assignments["tract_geoid"].astype(str).str.zfill(11).tolist()

    # Covariance: (J, J) Ledoit-Wolf regularized, from parquet.
    type_covariance = pd.read_parquet(covariance_dir / "type_covariance.parquet").values.astype(float)

    # Priors: per-type mean Dem share.
    priors_df = pd.read_parquet(communities_dir / "type_priors.parquet")
    type_priors = priors_df.sort_values("type_id")["prior_dem_share"].values.astype(float)

    return type_scores, tract_fips, type_covariance, type_priors


def _load_ridge_priors_from_db(db: duckdb.DuckDBPyConnection, version_id: str) -> dict[str, float]:
    """Load ridge county priors as a dict from DuckDB."""
    df = db.execute(
        "SELECT county_fips, ridge_pred_dem_share FROM ridge_county_priors WHERE version_id=?",
        [version_id],
    ).fetchdf()
    if df.empty:
        return {}
    return dict(zip(df["county_fips"], df["ridge_pred_dem_share"]))


def _load_hac_weights_from_db(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct state_weights and county_weights DataFrames from DuckDB.

    Returns DataFrames matching the shape that _forecast_poll_hac expects:
    state_weights has columns [state_abbr, community_0, community_1, ...]
    county_weights has columns [county_fips, community_id]
    """
    sw_df = db.execute(
        "SELECT state_abbr, community_id, weight FROM hac_state_weights WHERE version_id=?",
        [version_id],
    ).fetchdf()
    if sw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    state_weights = sw_df.pivot(index="state_abbr", columns="community_id", values="weight").reset_index()
    # Sort by community_id to ensure stable column ordering
    comm_ids = sorted(c for c in state_weights.columns if c != "state_abbr")
    state_weights = state_weights[["state_abbr"] + comm_ids]
    state_weights.columns = ["state_abbr"] + [f"community_{i}" for i in comm_ids]

    # `_forecast_poll_hac` only reads county_fips → community_id mapping.
    # Other columns that may exist in the source parquet (state_fips, etc.)
    # are intentionally excluded here.
    cw_df = db.execute(
        "SELECT county_fips, community_id FROM hac_county_weights WHERE version_id=?",
        [version_id],
    ).fetchdf()
    return state_weights, cw_df


def _load_tract_model(app, project_root: Path) -> None:
    """Load type-primary model data (tract-level scores, covariance, priors)."""
    try:
        (
            app.state.type_scores,
            app.state.type_county_fips,
            app.state.type_covariance,
            app.state.type_priors,
        ) = _load_tract_type_data(project_root)
        J = app.state.type_scores.shape[1]
        log.info("Loaded tract type data: %d tracts × %d types",
                 app.state.type_scores.shape[0], J)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.warning("Could not load tract type data: %s", exc)
        app.state.type_scores = None
        app.state.type_county_fips = None
        app.state.type_covariance = None
        app.state.type_priors = None


def _load_tract_priors(app, tracts_dir: Path) -> None:
    """Load per-tract Ridge priors, state mappings, and vote totals."""
    try:
        df = pd.read_parquet(tracts_dir / "tract_priors.parquet")
        app.state.ridge_priors = dict(zip(df["tract_geoid"], df["tract_prior"]))
        app.state.tract_states = dict(zip(df["tract_geoid"], df["state_abbr"]))
        app.state.tract_votes = dict(zip(df["tract_geoid"], df["total_votes"]))
        log.info("Loaded tract priors: %d tracts", len(app.state.ridge_priors))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.warning("Could not load tract priors: %s", exc)
        app.state.ridge_priors = {}
        app.state.tract_states = {}
        app.state.tract_votes = {}


def _load_crosstab_affinity(app, data_dir: Path, startup_db) -> None:
    """Load crosstab affinity index and per-state demographic means."""
    try:
        type_profiles_df = pd.read_parquet(data_dir / "communities" / "type_profiles.parquet")
        county_demographics_df = pd.read_parquet(data_dir / "assembled" / "county_features_national.parquet")
        app.state.crosstab_affinity = build_affinity_index(type_profiles_df, county_demographics_df)
        log.info("Loaded crosstab affinity index (%d dimensions)", len(app.state.crosstab_affinity))

        app.state.crosstab_state_means = _compute_state_demographic_means(
            county_demographics_df, startup_db
        )
        log.info("Computed per-state crosstab demographic means for %d states",
                 len(app.state.crosstab_state_means))
    except (FileNotFoundError, KeyError, ValueError, duckdb.Error) as exc:
        log.warning("Could not build crosstab affinity index: %s", exc)
        app.state.crosstab_affinity = None
        app.state.crosstab_state_means = {}


def _compute_state_demographic_means(
    county_demographics_df: pd.DataFrame,
    db: duckdb.DuckDBPyConnection,
) -> dict[str, dict[str, float]]:
    """Population-weighted demographic means per state for crosstab dimensions."""
    real_features = {
        key: col
        for key, col in CROSSTAB_DIMENSION_MAP.items()
        if col is not None and col in county_demographics_df.columns
    }
    counties_df = db.execute("SELECT county_fips, state_abbr FROM counties").fetchdf()
    merged = county_demographics_df.merge(counties_df, on="county_fips", how="left")

    state_means: dict[str, dict[str, float]] = {}
    if "state_abbr" not in merged.columns or "pop_total" not in merged.columns:
        return state_means

    for state_abbr, grp in merged.groupby("state_abbr"):
        pops = grp["pop_total"].to_numpy(dtype=float)
        if pops.sum() <= 0:
            continue
        dim_means: dict[str, float] = {}
        for dim_key, col in real_features.items():
            if col in grp.columns:
                dim_means[dim_key] = float(np.average(grp[col].to_numpy(dtype=float), weights=pops))
        # Inverted dimension means (delta is negated in construct_w_row)
        if "education_college" in dim_means:
            dim_means["education_noncollege"] = dim_means["education_college"]
        if "urbanicity_urban" in dim_means:
            dim_means["urbanicity_rural"] = dim_means["urbanicity_urban"]
        state_means[str(state_abbr)] = dim_means
    return state_means


def _load_correlation_matrix(app, project_root: Path) -> None:
    """Load type correlation matrix for similar-type queries."""
    corr_path = project_root / "data" / "covariance" / "type_correlation.parquet"
    if corr_path.exists():
        corr_df = pd.read_parquet(corr_path)
        app.state.type_correlation = corr_df.values
        log.info("Loaded type correlation matrix: %s", corr_df.shape)
    else:
        app.state.type_correlation = None
        log.warning("Type correlation matrix not found at %s", corr_path)


def _load_behavior_layer(app, data_dir: Path) -> None:
    """Load τ (turnout ratio) and δ (residual choice shift) arrays."""
    tau_path = data_dir / "behavior" / "tau.npy"
    delta_path = data_dir / "behavior" / "delta.npy"
    if tau_path.exists() and delta_path.exists():
        app.state.behavior_tau = np.load(tau_path)
        app.state.behavior_delta = np.load(delta_path)
        log.info("Loaded behavior layer: τ shape=%s, δ shape=%s",
                 app.state.behavior_tau.shape, app.state.behavior_delta.shape)
    else:
        app.state.behavior_tau = None
        app.state.behavior_delta = None
        log.warning("Behavior layer not found — predictions will use presidential baseline")


def _load_pollster_grades(app) -> None:
    """Load Silver Bulletin pollster grades for quality weighting."""
    try:
        from src.assembly.silver_bulletin_ratings import _normalize, load_pollster_grades
        grades = load_pollster_grades()
        app.state.pollster_grades = grades
        app.state.pollster_grades_normalized = {_normalize(n): g for n, g in grades.items()}
        log.info("Loaded pollster grades: %d pollsters", len(grades))
    except (ImportError, FileNotFoundError, KeyError, ValueError) as e:
        app.state.pollster_grades = {}
        app.state.pollster_grades_normalized = {}
        log.warning("Failed to load pollster grades: %s", e)


_REQUIRED_TABLES = ["super_types", "types", "county_type_assignments"]


def _validate_contract(app, db: duckdb.DuckDBPyConnection) -> None:
    """Check that required DuckDB tables exist for the frontend contract."""
    contract_ok = True
    for table_name in _REQUIRED_TABLES:
        try:
            result = db.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            if not result or result[0] == 0:
                log.warning("CONTRACT: missing table %s — frontend will show degraded state", table_name)
                contract_ok = False
        except duckdb.Error:
            contract_ok = False
    app.state.contract_ok = contract_ok
    log.info("Contract status: %s", "ok" if contract_ok else "degraded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open DB and load in-memory data at startup; close at shutdown."""
    data_dir = Path(os.environ.get("WETHERVANE_DATA_DIR", "data"))
    db_path = Path(os.environ.get("WETHERVANE_DB_PATH", str(data_dir / "wethervane.duckdb")))
    project_root = Path(os.environ.get("WETHERVANE_PROJECT_ROOT", "."))

    log.info("Opening DuckDB at %s (read_only=True)", db_path)
    app.state.db_path = str(db_path)
    startup_db = duckdb.connect(str(db_path), read_only=True)

    version_id = _get_current_version_id(startup_db)
    app.state.version_id = version_id
    log.info("Using model version: %s", version_id)

    sigma, K = _load_sigma(startup_db, version_id)
    app.state.sigma = sigma
    app.state.K = K
    log.info("Loaded sigma matrix (%d×%d)", K, K)

    app.state.mu_prior = _load_mu_prior(startup_db, version_id, K)
    log.info("Loaded mu_prior: %s", app.state.mu_prior.round(3))

    _load_tract_model(app, project_root)
    _load_tract_priors(app, project_root / "data" / "tracts")
    try:
        app.state.state_weights, app.state.county_weights = _load_hac_weights_from_db(startup_db, version_id)
        log.info("Loaded HAC weights: %d states", len(app.state.state_weights))
    except (duckdb.Error, KeyError, ValueError) as exc:
        log.warning("Could not load HAC weights from DB: %s", exc)
        app.state.state_weights = pd.DataFrame()
        app.state.county_weights = pd.DataFrame()
    _load_crosstab_affinity(app, data_dir, startup_db)
    _load_correlation_matrix(app, project_root)
    _load_behavior_layer(app, data_dir)
    _load_pollster_grades(app)
    _validate_contract(app, startup_db)

    startup_db.close()
    log.info("Startup DuckDB connection closed")

    # Attach the module-level cache singleton to app.state so routers and
    # middleware can reach it via request.app.state.cache.
    app.state.cache = forecast_cache
    log.info("Forecast cache attached (TTL=%ds)", forecast_cache.ttl_seconds)

    yield

    log.info("Cache stats at shutdown: %s", forecast_cache.stats())


# Paths whose GET responses are safe to cache.
# POST endpoints (poll updates) compute on-the-fly and must never be cached.
# /health and /model/* are cheap and reflect live state — exclude them too.
_CACHEABLE_PREFIXES = (
    "/api/v1/forecast",
    "/api/v1/polls",
)


def create_app(lifespan_override=None) -> FastAPI:
    """Factory function. Pass lifespan_override=_noop_lifespan in tests to skip DB startup."""
    app = FastAPI(
        title="WetherVane API",
        description="Electoral community model data for WetherVane",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan_override if lifespan_override is not None else lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten after launch
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def forecast_cache_middleware(request: Request, call_next):
        """Cache GET responses for forecast/poll endpoints.

        Only caches successful (HTTP 200) JSON responses.  POST requests and
        non-forecast paths are passed through without touching the cache.
        Cache is keyed by path + sorted query params so different filters
        produce independent cache entries.
        """
        # Only cache GET requests to designated cacheable paths.
        is_cacheable = (
            request.method == "GET"
            and any(request.url.path.startswith(prefix) for prefix in _CACHEABLE_PREFIXES)
        )
        if not is_cacheable:
            return await call_next(request)

        cache = getattr(request.app.state, "cache", None)
        if cache is None:
            return await call_next(request)

        key = make_cache_key(request.url.path, dict(request.query_params))
        hit, cached_body = cache.get(key)
        if hit:
            log.debug("Cache hit: %s", key)
            return Response(
                content=cached_body,
                media_type="application/json",
                headers={"X-Cache": "HIT"},
            )

        log.debug("Cache miss: %s", key)
        response = await call_next(request)

        # Only cache successful JSON responses to avoid caching errors.
        if response.status_code == 200:
            # Consume the streaming response body so we can store and replay it.
            body = b"".join([chunk async for chunk in response.body_iterator])
            cache.set(key, body)
            return Response(
                content=body,
                status_code=response.status_code,
                media_type=response.media_type or "application/json",
                headers={**dict(response.headers), "X-Cache": "MISS"},
            )

        return response

    app.include_router(meta.router, prefix="/api/v1")
    app.include_router(communities.router, prefix="/api/v1")
    app.include_router(counties.router, prefix="/api/v1")
    app.include_router(forecast.router, prefix="/api/v1")
    app.include_router(historical.router, prefix="/api/v1")
    app.include_router(senate.router, prefix="/api/v1")
    app.include_router(governor.router, prefix="/api/v1")
    app.include_router(pollsters.router, prefix="/api/v1")
    app.include_router(tracts.router, prefix="/api/v1")
    app.include_router(candidates.router, prefix="/api/v1")
    app.include_router(cache_router.router, prefix="/api/v1")

    return app


app = create_app()  # Production app: uses real lifespan


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
