# api/routers/meta.py
from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.models import (
    AccuracyResponse,
    CrossElectionResult,
    HealthResponse,
    MethodComparison,
    ModelVersionResponse,
    OverallAccuracy,
)

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "error"
    contract = "ok" if getattr(request.app.state, "contract_ok", True) else "degraded"
    return HealthResponse(status="ok", db=db_status, contract=contract)


@router.get("/model/version", response_model=ModelVersionResponse)
def model_version(db: duckdb.DuckDBPyConnection = Depends(get_db)):
    row = db.execute(
        """SELECT version_id, k, j, holdout_r, shift_type, created_at
           FROM model_versions WHERE role = 'current' LIMIT 1"""
    ).fetchone()
    if row is None:
        row = db.execute(
            """SELECT version_id, k, j, holdout_r, shift_type, created_at
               FROM model_versions ORDER BY version_id DESC LIMIT 1"""
        ).fetchone()
    if row is None:
        return ModelVersionResponse(
            version_id="unknown", k=None, j=None,
            holdout_r=None, shift_type=None, created_at=None,
        )
    version_id, k, j, holdout_r, shift_type, created_at = row
    return ModelVersionResponse(
        version_id=str(version_id),
        k=int(k) if k is not None else None,
        j=int(j) if j is not None else None,
        holdout_r=str(holdout_r) if holdout_r is not None else None,
        shift_type=str(shift_type) if shift_type is not None else None,
        created_at=str(created_at) if created_at is not None else None,
    )


@router.get("/model/accuracy", response_model=AccuracyResponse)
def model_accuracy() -> AccuracyResponse:
    """Return model backtesting and validation accuracy metrics.

    These metrics are hardcoded because they are stable model metadata that
    only changes on retrain. They reflect the type-primary-v1.0 model state
    as documented in MEMORY.md and CLAUDE.md baselines.
    """
    return AccuracyResponse(
        overall=OverallAccuracy(
            loo_r=0.711,
            holdout_r=0.698,
            coherence=0.783,
            rmse=0.073,
            covariance_val_r=0.915,
            n_counties=3154,
            n_types=100,
            n_super_types=5,
        ),
        cross_election=[
            CrossElectionResult(cycle="2008→2012", loo_r=0.45, label="Obama→Obama"),
            CrossElectionResult(cycle="2012→2016", loo_r=0.64, label="Obama→Trump"),
            CrossElectionResult(cycle="2016→2020", loo_r=0.42, label="Trump→Biden"),
            CrossElectionResult(cycle="2020→2024", loo_r=0.40, label="Biden→Trump"),
        ],
        method_comparison=[
            MethodComparison(method="Type-mean baseline", loo_r=0.448),
            MethodComparison(method="Ridge (scores only)", loo_r=0.533),
            MethodComparison(method="Ridge (all features)", loo_r=0.671),
            MethodComparison(method="Ridge+HGB ensemble", loo_r=0.711),
        ],
    )
