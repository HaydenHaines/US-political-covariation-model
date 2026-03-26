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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import communities, counties, forecast, meta

log = logging.getLogger(__name__)


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
    mu = np.full(K, 0.45)
    for _, r in rows.iterrows():
        idx = int(r["community_id"])
        if 0 <= idx < K:
            mu[idx] = float(r["mean_share"])
    return mu


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open DB and load in-memory data at startup; close at shutdown."""
    data_dir = Path(os.environ.get("WETHERVANE_DATA_DIR", "data"))
    db_path = Path(os.environ.get("WETHERVANE_DB_PATH", str(data_dir / "wethervane.duckdb")))

    log.info("Opening DuckDB at %s (read_only=True)", db_path)
    # Store the path so each request can open its own connection (avoids DuckDB
    # concurrent-query cancellation when Promise.all fires multiple requests).
    app.state.db_path = str(db_path)

    # Use a short-lived startup connection only for loading in-memory data.
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

    # Load weight matrices from parquet (not in DuckDB)
    state_w_path = data_dir / "propagation" / "community_weights_state_hac.parquet"
    county_w_path = data_dir / "propagation" / "community_weights_county_hac.parquet"
    if state_w_path.exists():
        app.state.state_weights = pd.read_parquet(state_w_path)
        log.info("Loaded state_weights: %d rows", len(app.state.state_weights))
    else:
        log.warning("state_weights not found at %s — POST /forecast/poll will fail", state_w_path)
        app.state.state_weights = pd.DataFrame()

    if county_w_path.exists():
        app.state.county_weights = pd.read_parquet(county_w_path)
        log.info("Loaded county_weights: %d rows", len(app.state.county_weights))
    else:
        log.warning("county_weights not found at %s", county_w_path)
        app.state.county_weights = pd.DataFrame()

    # ── Type-primary data (optional — graceful fallback if not present) ──
    app.state.type_scores = None
    app.state.type_covariance = None
    app.state.type_priors = None
    app.state.type_county_fips = None

    type_scores_path = data_dir / "communities" / "type_assignments.parquet"
    type_cov_path = data_dir / "covariance" / "type_covariance.parquet"
    type_profiles_path = data_dir / "communities" / "type_profiles.parquet"

    if type_scores_path.exists() and type_cov_path.exists():
        try:
            ta_df = pd.read_parquet(type_scores_path)
            score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
            if score_cols:
                app.state.type_scores = ta_df[score_cols].values
                app.state.type_county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
                J = app.state.type_scores.shape[1]
                log.info("Loaded type_scores: %d counties x %d types", *app.state.type_scores.shape)

                cov_df = pd.read_parquet(type_cov_path)
                app.state.type_covariance = cov_df.values[:J, :J]
                log.info("Loaded type_covariance: %d x %d", J, J)

                # Type priors
                app.state.type_priors = np.full(J, 0.45)
                if type_profiles_path.exists():
                    profiles = pd.read_parquet(type_profiles_path)
                    if "mean_dem_share" in profiles.columns:
                        app.state.type_priors = profiles["mean_dem_share"].values[:J]
                log.info("Type priors loaded: %s", np.round(app.state.type_priors, 3))
        except Exception:
            log.warning("Failed to load type data — falling back to HAC pipeline", exc_info=True)
    else:
        log.info("Type data not found — will use HAC pipeline for forecasts")

    # ── Contract check ─────────────────────────────────────────────────────────
    contract_ok = True
    for table_name in ["super_types", "types", "county_type_assignments"]:
        try:
            result = startup_db.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            if not result or result[0] == 0:
                log.warning("CONTRACT: missing table %s — frontend will show degraded state", table_name)
                contract_ok = False
        except Exception:
            contract_ok = False
    app.state.contract_ok = contract_ok
    log.info("Contract status: %s", "ok" if contract_ok else "degraded")

    startup_db.close()
    log.info("Startup DuckDB connection closed")

    yield


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

    app.include_router(meta.router, prefix="/api/v1")
    app.include_router(communities.router, prefix="/api/v1")
    app.include_router(counties.router, prefix="/api/v1")
    app.include_router(forecast.router, prefix="/api/v1")

    return app


app = create_app()  # Production app: uses real lifespan


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
