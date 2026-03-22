# api/routers/meta.py
from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db
from api.models import HealthResponse, ModelVersionResponse

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
