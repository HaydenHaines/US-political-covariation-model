# api/routers/candidates.py
"""Sabermetrics candidate endpoints: badges, CTOV, and race-level candidate lookups.

Data is loaded once at import time from static JSON/Parquet files produced by
the sabermetrics pipeline (Phases 1-2).  These files are small (<1MB total)
and change only when the pipeline is re-run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import (
    CandidateBadge,
    CandidateBadgesResponse,
    CTOVEntry,
    CTOVResponse,
    RaceCandidatesResponse,
    RaceCandidateSummary,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["candidates"])

# ── Static data loaded at import time ────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_SABERMETRICS_DIR = _DATA_DIR / "sabermetrics"
_CANDIDATES_2026_PATH = _DATA_DIR / "config" / "candidates_2026.json"

# Badge data: keyed by bioguide ID
_BADGES: dict[str, dict] = {}
_BADGES_PATH = _SABERMETRICS_DIR / "candidate_badges.json"
if _BADGES_PATH.exists():
    with _BADGES_PATH.open() as f:
        _BADGES = json.load(f)
    log.info("Loaded %d candidate badge profiles", len(_BADGES))

# CTOV data: DataFrame with per-candidate, per-race, per-type overperformance
_CTOV: pd.DataFrame = pd.DataFrame()
_CTOV_PATH = _SABERMETRICS_DIR / "candidate_ctov.parquet"
if _CTOV_PATH.exists():
    _CTOV = pd.read_parquet(_CTOV_PATH)
    log.info("Loaded CTOV data: %d rows x %d cols", *_CTOV.shape)

# Candidate registry: maps bioguide IDs to names, parties, races
_REGISTRY: dict[str, dict] = {}
_REGISTRY_PATH = _SABERMETRICS_DIR / "candidate_registry.json"
if _REGISTRY_PATH.exists():
    with _REGISTRY_PATH.open() as f:
        raw = json.load(f)
    _REGISTRY = raw.get("persons", raw) if isinstance(raw, dict) else {}
    log.info("Loaded candidate registry: %d persons", len(_REGISTRY))

# Candidates 2026: race-level candidate data for linking to sabermetrics
_CANDIDATES_2026: dict[str, dict] = {}
if _CANDIDATES_2026_PATH.exists():
    with _CANDIDATES_2026_PATH.open() as f:
        raw_2026 = json.load(f)
    for race_type_key in ("senate", "governor"):
        if race_type_key in raw_2026:
            _CANDIDATES_2026.update(raw_2026[race_type_key])
    log.info("Loaded %d race entries from candidates_2026.json", len(_CANDIDATES_2026))

# ── Name-to-bioguide reverse index ──────────────────────────────────────────
# candidates_2026.json lacks bioguide IDs, so we match by exact name against
# the registry.  This is built once at import time.

_NAME_TO_BIOGUIDE: dict[str, str] = {}
for _pid, _pdata in _REGISTRY.items():
    _name = _pdata.get("name", "")
    if _name:
        _NAME_TO_BIOGUIDE[_name] = _pid


def _resolve_bioguide(candidate_name: str) -> str | None:
    """Resolve a candidate display name to a bioguide ID via the registry.

    Handles the common case where candidates_2026.json includes parenthetical
    annotations like 'Katie Britt (appointed interim)' by stripping them.
    """
    clean = candidate_name.split("(")[0].strip()
    return _NAME_TO_BIOGUIDE.get(clean)


# ── Super-type display names (cached) ───────────────────────────────────────

_TYPE_DISPLAY_NAMES: dict[int, str] | None = None


def _get_type_display_names(db: duckdb.DuckDBPyConnection) -> dict[int, str]:
    """Fetch per-type display names from DuckDB ``types`` table, cached after first call.

    Uses the individual type display names (J=100 types), not super-type names.
    The cache persists for the process lifetime since type names only change on
    retrain (at which point the server is restarted).
    """
    global _TYPE_DISPLAY_NAMES  # noqa: PLW0603
    if _TYPE_DISPLAY_NAMES is not None:
        return _TYPE_DISPLAY_NAMES

    try:
        rows = db.execute("SELECT type_id, display_name FROM types").fetchall()
        _TYPE_DISPLAY_NAMES = {int(r[0]): str(r[1]) for r in rows}
    except Exception:
        log.warning("Could not load type display names from DuckDB")
        _TYPE_DISPLAY_NAMES = {}
    return _TYPE_DISPLAY_NAMES


def _build_badges(badge_data: dict) -> list[CandidateBadge]:
    """Convert raw badge data into a list of CandidateBadge models.

    Reads from badge_details when available (includes provisional, kind, type_id,
    fallback_reason).  Falls back to the legacy badges list + badge_scores for
    artifacts produced before those fields existed.
    """
    details = badge_data.get("badge_details")
    if details:
        scores = badge_data.get("badge_scores", {})
        return [
            CandidateBadge(
                name=d["name"],
                score=d.get("score", scores.get(d["name"], 0.0)),
                provisional=d.get("provisional", False),
                kind=d.get("kind", "catalog"),
                type_id=d.get("type_id"),
                fallback_reason=d.get("fallback_reason"),
            )
            for d in details
        ]
    # Legacy fallback: artifact pre-dates badge_details
    badge_names = badge_data.get("badges", [])
    scores = badge_data.get("badge_scores", {})
    result = []
    for name in badge_names:
        score = scores.get(name, None)
        if score is None and name.startswith("Low "):
            score = scores.get(name[4:], 0.0)
        result.append(CandidateBadge(name=name, score=score or 0.0))
    return result


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get(
    "/candidates/{bioguide_id}",
    response_model=CandidateBadgesResponse,
    summary="Full candidate sabermetrics profile",
)
def get_candidate(bioguide_id: str):
    """Return badges, badge scores, and CEC for a candidate by bioguide ID."""
    if bioguide_id not in _BADGES:
        raise HTTPException(status_code=404, detail=f"Candidate {bioguide_id} not found")

    data = _BADGES[bioguide_id]
    return CandidateBadgesResponse(
        bioguide_id=bioguide_id,
        name=data["name"],
        party=data["party"],
        n_races=data["n_races"],
        badges=_build_badges(data),
        badge_scores=data.get("badge_scores", {}),
        cec=data.get("cec", 0.0),
    )


@router.get(
    "/candidates/{bioguide_id}/ctov",
    response_model=CTOVResponse,
    summary="CTOV vector for a candidate (top 10 types by absolute value)",
)
def get_candidate_ctov(
    request: Request,
    bioguide_id: str,
    year: int | None = None,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Return the top 10 community types by absolute CTOV for a candidate.

    If year is specified, returns CTOV for that specific race.
    Otherwise returns the most recent race available.
    """
    if _CTOV.empty:
        raise HTTPException(status_code=404, detail="CTOV data not available")

    # Filter to this candidate
    mask = _CTOV["person_id"] == bioguide_id
    if mask.sum() == 0:
        raise HTTPException(status_code=404, detail=f"No CTOV data for {bioguide_id}")

    candidate_rows = _CTOV[mask]
    if year is not None:
        candidate_rows = candidate_rows[candidate_rows["year"] == year]
        if candidate_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No CTOV data for {bioguide_id} in {year}",
            )

    # Take the most recent race
    row = candidate_rows.sort_values("year", ascending=False).iloc[0]

    # Extract CTOV columns (ctov_type_0 .. ctov_type_99)
    ctov_cols = [c for c in _CTOV.columns if c.startswith("ctov_type_")]
    ctov_values: dict[int, float] = {}
    for col in ctov_cols:
        type_id = int(col.replace("ctov_type_", ""))
        val = float(row[col])
        if pd.notna(val):
            ctov_values[type_id] = val

    # Sort by absolute value and take top 10
    sorted_types = sorted(ctov_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    # Resolve display names from DuckDB super_types table
    type_names = _get_type_display_names(db)

    entries = [
        CTOVEntry(
            type_id=tid,
            display_name=type_names.get(tid, f"Type {tid}"),
            ctov=val,
        )
        for tid, val in sorted_types
    ]

    return CTOVResponse(
        bioguide_id=bioguide_id,
        name=str(row["name"]),
        party=str(row["party"]),
        year=int(row["year"]),
        state=str(row["state"]),
        office=str(row["office"]),
        entries=entries,
    )


@router.get(
    "/races/{race_key:path}/candidates",
    response_model=RaceCandidatesResponse,
    summary="All candidates with badges for a race",
)
def get_race_candidates(race_key: str):
    """Return badge summaries for all candidates in a given race.

    race_key uses the same format as candidates_2026.json keys,
    e.g. '2026 GA Senate'.  The endpoint accepts URL-encoded spaces
    or dashes (converted to spaces).
    """
    # Normalize: convert dashes to spaces for slug-style keys
    normalized_key = race_key.replace("-", " ")

    # Try exact match first, then case-insensitive
    race_data = _CANDIDATES_2026.get(normalized_key)
    if race_data is None:
        # Try case-insensitive
        for k, v in _CANDIDATES_2026.items():
            if k.lower() == normalized_key.lower():
                race_data = v
                normalized_key = k
                break

    if race_data is None:
        raise HTTPException(status_code=404, detail=f"Race '{race_key}' not found")

    candidates_out: list[RaceCandidateSummary] = []

    # Collect all candidate names from the race
    all_candidates: list[tuple[str, str]] = []  # (name, party)
    for party, names in race_data.get("candidates", {}).items():
        for name in names:
            all_candidates.append((name, party))

    # Also check the incumbent (they might not be listed under candidates)
    incumbent = race_data.get("incumbent", {})
    if isinstance(incumbent, dict) and incumbent.get("name"):
        inc_name = incumbent["name"]
        inc_party = incumbent.get("party", "")
        # Avoid duplicates
        already_listed = any(
            inc_name.split("(")[0].strip() == n.split("(")[0].strip()
            for n, _ in all_candidates
        )
        if not already_listed:
            all_candidates.append((inc_name, inc_party))

    for name, party in all_candidates:
        bioguide = _resolve_bioguide(name)
        if bioguide and bioguide in _BADGES:
            badge_data = _BADGES[bioguide]
            candidates_out.append(
                RaceCandidateSummary(
                    bioguide_id=bioguide,
                    name=badge_data["name"],
                    party=badge_data["party"],
                    badges=_build_badges(badge_data),
                    cec=badge_data.get("cec", 0.0),
                    badge_scores=badge_data.get("badge_scores", {}),
                )
            )

    return RaceCandidatesResponse(
        race_key=normalized_key,
        candidates=candidates_out,
    )
