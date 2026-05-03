"""GET /forecast/xt-impact — cross-type poll impact scores for top race movers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(tags=["forecast"])

# Module-level 1-hour TTL cache: {"data": ..., "expires": datetime}
_cache: dict[str, Any] = {}
_CACHE_TTL = timedelta(hours=1)


def _get_cached_report(races: list[str] | None, race_type: str | None) -> dict | None:
    races_key = ",".join(sorted(races)) if races else "__all__"
    key = f"{races_key}|rt={race_type or ''}"
    entry = _cache.get(key)
    if entry and datetime.now(tz=timezone.utc) < entry["expires"]:
        return entry["data"]
    return None


def _set_cached_report(races: list[str] | None, race_type: str | None, data: dict) -> None:
    races_key = ",".join(sorted(races)) if races else "__all__"
    key = f"{races_key}|rt={race_type or ''}"
    _cache[key] = {"data": data, "expires": datetime.now(tz=timezone.utc) + _CACHE_TTL}


@router.get("/forecast/xt-impact")
def get_xt_impact(
    races: str | None = Query(
        default=None,
        description="Comma-separated race IDs to include, e.g. 'ga-senate,pa-senate'. "
        "Omit to include all races.",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        description="Maximum number of top movers to return (sorted by |delta_pp| descending).",
    ),
    race_type: str | None = Query(
        default=None,
        description="Filter top movers to races whose race_id contains this substring "
        "(case-insensitive). E.g. 'governor' or 'senate'.",
    ),
) -> dict:
    """Return cross-type poll impact scores showing how xt_ crosstab data shifts each race forecast.

    Calls make_xt_impact_report() and formats the result as top movers sorted by
    absolute delta. Results are cached for 1 hour per (races, race_type) combination.
    """
    race_list: list[str] | None = None
    if races:
        race_list = [r.strip() for r in races.split(",") if r.strip()]

    cached = _get_cached_report(race_list, race_type)
    if cached is None:
        from src.prediction.forecast_engine import make_xt_impact_report

        cached = make_xt_impact_report(races=race_list)
        _set_cached_report(race_list, race_type, cached)

    enriched_deltas: dict[str, float] = cached.get("enriched_deltas", {})
    xt_race_counts: dict[str, int] = cached.get("xt_race_counts", {})
    movers = [
        {
            "race_id": race_id,
            "delta_pp": delta,
            "n_xt_polls": xt_race_counts.get(race_id, 0),
        }
        for race_id, delta in enriched_deltas.items()
    ]
    if race_type:
        rt_lower = race_type.lower()
        movers = [m for m in movers if rt_lower in m["race_id"].lower()]
    top_movers = sorted(movers, key=lambda x: abs(x["delta_pp"]), reverse=True)[:limit]

    return {
        "top_movers": top_movers,
        "mean_delta": cached.get("mean_delta", 0.0),
        "max_delta": cached.get("max_delta", 0.0),
        "races_with_xt": cached.get("races_with_xt", 0),
        "report_date": cached.get("report_date", ""),
    }
