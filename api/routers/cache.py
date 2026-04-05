# api/routers/cache.py
"""Cache management endpoints.

Exposes a single POST endpoint to invalidate the in-memory forecast cache.
Intended for use by the poll-scrape cron job after DuckDB is rebuilt so that
the next request fetches fresh predictions rather than serving stale data.

Usage (from wethervane-poll-scrape.sh after scrape completes):
    curl -s -X POST http://localhost:8002/api/v1/cache/invalidate
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

log = logging.getLogger(__name__)

router = APIRouter(tags=["cache"])


@router.post("/cache/invalidate")
def post_cache_invalidate(request: Request) -> dict:
    """Evict all cached forecast responses.

    Call this after a poll scrape or DuckDB rebuild so the next request
    fetches fresh predictions from the database.  Returns the number of
    cache entries that were cleared.
    """
    cache = getattr(request.app.state, "cache", None)
    count = cache.clear() if cache is not None else 0
    log.info("Cache invalidated via API: %d entries cleared", count)
    return {"cleared": count, "status": "ok"}


@router.get("/cache/stats")
def get_cache_stats(request: Request) -> dict:
    """Return current cache statistics (live entries, hits, misses).

    Useful for monitoring and debugging.  Does not modify the cache.
    """
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        return {"entries": 0, "hits": 0, "misses": 0, "ttl_seconds": 0}
    return {**cache.stats(), "ttl_seconds": cache.ttl_seconds}
