"""Forecast comparisons endpoint: WetherVane vs major forecasters."""
from __future__ import annotations

import json as _json
from pathlib import Path

import duckdb
from fastapi import APIRouter, Depends, Request

from api.db import get_db

from ._helpers import marginToRating, race_to_slug

router = APIRouter(tags=["forecast"])

COMPARISONS_FILE = Path(__file__).resolve().parents[3] / "data" / "comparisons" / "ratings_2026.json"


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
    except duckdb.Error:
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
