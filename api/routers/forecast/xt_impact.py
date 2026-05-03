"""GET /forecast/xt-impact — cross-type poll impact scores for top race movers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(tags=["forecast"])

# Module-level 1-hour TTL cache: {"data": ..., "expires": datetime}
_cache: dict[str, Any] = {}
_CACHE_TTL = timedelta(hours=1)


def _get_cached_report(races: list[str] | None) -> dict | None:
    key = ",".join(sorted(races)) if races else "__all__"
    entry = _cache.get(key)
    if entry and datetime.now(tz=timezone.utc) < entry["expires"]:
        return entry["data"]
    return None


def _set_cached_report(races: list[str] | None, data: dict) -> None:
    key = ",".join(sorted(races)) if races else "__all__"
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
) -> dict:
    """Return cross-type poll impact scores showing how xt_ crosstab data shifts each race forecast.

    Calls make_xt_impact_report() and formats the result as top movers sorted by
    absolute delta. Results are cached for 1 hour per (races, limit) combination.
    """
    race_list: list[str] | None = None
    if races:
        race_list = [r.strip() for r in races.split(",") if r.strip()]

    cached = _get_cached_report(race_list)
    if cached is None:
        from src.prediction.forecast_engine import make_xt_impact_report

        cached = make_xt_impact_report(races=race_list)
        _set_cached_report(race_list, cached)

    enriched_deltas: dict[str, float] = cached.get("enriched_deltas", {})
    top_movers = sorted(
        [{"race_id": race_id, "delta_pp": delta} for race_id, delta in enriched_deltas.items()],
        key=lambda x: abs(x["delta_pp"]),
        reverse=True,
    )[:limit]

    return {
        "top_movers": top_movers,
        "mean_delta": cached.get("mean_delta", 0.0),
        "max_delta": cached.get("max_delta", 0.0),
        "races_with_xt": cached.get("races_with_xt", 0),
        "report_date": cached.get("report_date", ""),
    }
