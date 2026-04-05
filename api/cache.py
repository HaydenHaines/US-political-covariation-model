# api/cache.py
"""In-memory TTL cache for forecast API responses.

Predictions change at most 2x/week (after poll scrapes), so caching
responses for 1 hour is safe and significantly reduces DuckDB query load.

Design notes:
- Pure stdlib, no external dependencies (time + dict).
- Thread-safe for read-heavy workloads: Python's GIL protects dict reads/writes.
- Cache key = endpoint path + sorted query params, so
  /forecast?race=FL_Senate and /forecast?state=FL are separate entries.
- Invalidation: call ``cache.clear()`` or ``invalidate_cache()`` after a
  poll scrape or DuckDB rebuild.  The /api/v1/cache/invalidate endpoint
  exposes this to the poll-scrape cron job.
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# Default TTL: 1 hour.  Predictions change at most 2x/week so this is very safe.
DEFAULT_TTL_SECONDS = 3600


class TTLCache:
    """Simple in-memory cache with per-entry TTL expiry.

    Each entry stores (value, expiry_timestamp).  Entries are evicted lazily
    on the next read — no background thread is needed for this access pattern.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        """Return (hit, value).  hit=False means the key is absent or expired."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return False, None
        value, expiry = entry
        if time.monotonic() > expiry:
            # Lazy eviction: remove the stale entry so memory is reclaimed.
            del self._store[key]
            self._misses += 1
            return False, None
        self._hits += 1
        return True, value

    def set(self, key: str, value: Any) -> None:
        """Store value with TTL expiry."""
        self._store[key] = (value, time.monotonic() + self._ttl)

    def clear(self) -> int:
        """Evict all entries.  Returns the number of entries removed."""
        count = len(self._store)
        self._store.clear()
        log.info("Cache cleared: %d entries removed", count)
        return count

    def stats(self) -> dict[str, int]:
        """Return current cache statistics."""
        # Count non-expired entries for an accurate size report.
        now = time.monotonic()
        live = sum(1 for _, (_, exp) in self._store.items() if exp > now)
        return {
            "entries": live,
            "hits": self._hits,
            "misses": self._misses,
        }

    @property
    def ttl_seconds(self) -> int:
        return self._ttl


# Module-level singleton shared across the FastAPI application.
# Instantiated once here; main.py attaches it to app.state.cache at startup
# so routers can access it via request.app.state.cache.
forecast_cache = TTLCache(ttl_seconds=DEFAULT_TTL_SECONDS)


def make_cache_key(path: str, query_params: dict[str, str]) -> str:
    """Build a stable cache key from the URL path and sorted query parameters.

    Sorting query params ensures that ?race=FL&state=FL and ?state=FL&race=FL
    map to the same cache entry.
    """
    if query_params:
        qs = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
        return f"{path}?{qs}"
    return path


def invalidate_cache() -> int:
    """Clear the global forecast cache.  Returns number of entries removed.

    Called from:
    - POST /api/v1/cache/invalidate (manual / cron trigger)
    - After DuckDB rebuild (future hook)
    """
    return forecast_cache.clear()
