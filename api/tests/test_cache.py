# api/tests/test_cache.py
"""Tests for the TTL cache and forecast cache middleware.

Tests cover:
1. TTLCache unit behaviour (get/set/expiry/clear/stats)
2. make_cache_key normalisation
3. Middleware: GET /forecast responses are cached (X-Cache: MISS then HIT)
4. Middleware: POST requests bypass the cache
5. Middleware: non-forecast paths bypass the cache
6. Middleware: non-200 responses are not cached
7. Cache invalidation via POST /cache/invalidate
8. Cache stats via GET /cache/stats
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from api.cache import TTLCache, make_cache_key
from api.tests.conftest import (
    _build_test_db,
    _build_test_state,
    _noop_lifespan,
)
from api.main import create_app


# ── TTLCache unit tests ───────────────────────────────────────────────────────

class TestTTLCache:
    def test_miss_on_empty(self):
        cache = TTLCache(ttl_seconds=60)
        hit, value = cache.get("no-such-key")
        assert not hit
        assert value is None

    def test_hit_after_set(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", b'{"result": 1}')
        hit, value = cache.get("k")
        assert hit
        assert value == b'{"result": 1}'

    def test_miss_after_expiry(self):
        # Use a tiny TTL then mock monotonic to advance past it.
        cache = TTLCache(ttl_seconds=1)
        cache.set("k", b"data")

        # Advance time by 2 seconds beyond the entry's expiry.
        with patch("api.cache.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 3600
            hit, _ = cache.get("k")
        assert not hit

    def test_set_overwrites_existing_entry(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", b"first")
        cache.set("k", b"second")
        _, val = cache.get("k")
        assert val == b"second"

    def test_clear_removes_all_entries(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("a", b"1")
        cache.set("b", b"2")
        removed = cache.clear()
        assert removed == 2
        hit, _ = cache.get("a")
        assert not hit

    def test_stats_tracks_hits_and_misses(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", b"v")

        cache.get("k")           # hit
        cache.get("k")           # hit
        cache.get("no-key")      # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_stats_entries_excludes_expired(self):
        cache = TTLCache(ttl_seconds=1)
        cache.set("k", b"v")

        with patch("api.cache.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 3600
            stats = cache.stats()
        assert stats["entries"] == 0

    def test_clear_returns_zero_on_empty_cache(self):
        cache = TTLCache(ttl_seconds=60)
        assert cache.clear() == 0


# ── make_cache_key tests ──────────────────────────────────────────────────────

class TestMakeCacheKey:
    def test_path_only_when_no_params(self):
        key = make_cache_key("/api/v1/forecast", {})
        assert key == "/api/v1/forecast"

    def test_params_appended_sorted(self):
        key = make_cache_key("/api/v1/forecast", {"state": "FL", "race": "FL_Senate"})
        # Sorted alphabetically: race before state
        assert key == "/api/v1/forecast?race=FL_Senate&state=FL"

    def test_param_order_does_not_matter(self):
        key1 = make_cache_key("/p", {"b": "2", "a": "1"})
        key2 = make_cache_key("/p", {"a": "1", "b": "2"})
        assert key1 == key2

    def test_different_paths_produce_different_keys(self):
        k1 = make_cache_key("/api/v1/forecast", {"race": "FL"})
        k2 = make_cache_key("/api/v1/polls", {"race": "FL"})
        assert k1 != k2


# ── Middleware + cache endpoint integration tests ─────────────────────────────

def _make_cached_client():
    """Build a TestClient with the full middleware stack and a fresh cache."""
    test_db = _build_test_db()
    state = _build_test_state()

    test_app = create_app(lifespan_override=_noop_lifespan)
    test_app.state.db = test_db
    test_app.state.version_id = "test_v1"
    test_app.state.K = state["sigma"].shape[0]
    test_app.state.sigma = state["sigma"]
    test_app.state.mu_prior = state["mu_prior"]
    test_app.state.state_weights = state["state_weights"]
    test_app.state.county_weights = state["county_weights"]
    test_app.state.contract_ok = True

    # Attach a fresh cache instance so tests don't share state.
    from api.cache import TTLCache
    test_app.state.cache = TTLCache(ttl_seconds=3600)

    from fastapi.testclient import TestClient
    return TestClient(test_app, raise_server_exceptions=True), test_db


class TestCacheMiddleware:
    def test_first_request_is_cache_miss(self):
        client, db = _make_cached_client()
        with client:
            resp = client.get("/api/v1/forecast?race=FL_Senate")
        assert resp.status_code == 200
        assert resp.headers.get("x-cache") == "MISS"
        db.close()

    def test_second_request_is_cache_hit(self):
        client, db = _make_cached_client()
        with client:
            client.get("/api/v1/forecast?race=FL_Senate")
            resp2 = client.get("/api/v1/forecast?race=FL_Senate")
        assert resp2.status_code == 200
        assert resp2.headers.get("x-cache") == "HIT"
        db.close()

    def test_cached_body_matches_original(self):
        client, db = _make_cached_client()
        with client:
            r1 = client.get("/api/v1/forecast?race=FL_Senate")
            r2 = client.get("/api/v1/forecast?race=FL_Senate")
        assert r1.json() == r2.json()
        db.close()

    def test_different_query_params_are_separate_entries(self):
        client, db = _make_cached_client()
        with client:
            r_fl = client.get("/api/v1/forecast?state=FL")
            r_ga = client.get("/api/v1/forecast?state=GA")
            # Both should be MISS (first access each)
            assert r_fl.headers.get("x-cache") == "MISS"
            assert r_ga.headers.get("x-cache") == "MISS"
            # Second access of FL should now be HIT
            r_fl2 = client.get("/api/v1/forecast?state=FL")
            assert r_fl2.headers.get("x-cache") == "HIT"
        db.close()

    def test_post_requests_bypass_cache(self):
        """POST /forecast/poll must never be cached — it computes on-the-fly."""
        client, db = _make_cached_client()
        with client:
            resp = client.post(
                "/api/v1/forecast/poll",
                json={"state": "FL", "race": "FL_Senate", "dem_share": 0.48, "n": 800},
            )
        # POST responses should not carry an X-Cache header (not going through cache logic).
        assert "x-cache" not in resp.headers
        db.close()

    def test_non_forecast_paths_bypass_cache(self):
        """Health endpoint must not be cached (it reflects live DB state)."""
        client, db = _make_cached_client()
        with client:
            resp = client.get("/api/v1/health")
        assert "x-cache" not in resp.headers
        db.close()

    def test_polls_endpoint_is_cached(self):
        """GET /polls is a read-only forecast path and should be cached."""
        client, db = _make_cached_client()
        with client:
            r1 = client.get("/api/v1/polls?cycle=2026")
            r2 = client.get("/api/v1/polls?cycle=2026")
        assert r1.headers.get("x-cache") == "MISS"
        assert r2.headers.get("x-cache") == "HIT"
        db.close()


class TestCacheEndpoints:
    def test_invalidate_clears_cache(self):
        client, db = _make_cached_client()
        with client:
            # Populate cache
            client.get("/api/v1/forecast?race=FL_Senate")
            client.get("/api/v1/forecast?race=FL_Senate")  # now cached

            # Invalidate
            inv = client.post("/api/v1/cache/invalidate")
            assert inv.status_code == 200
            assert inv.json()["status"] == "ok"
            assert inv.json()["cleared"] >= 0  # at least 0

            # Next request should be a MISS again
            r3 = client.get("/api/v1/forecast?race=FL_Senate")
            assert r3.headers.get("x-cache") == "MISS"
        db.close()

    def test_stats_endpoint_returns_expected_shape(self):
        client, db = _make_cached_client()
        with client:
            resp = client.get("/api/v1/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "hits" in data
        assert "misses" in data
        assert "ttl_seconds" in data
        assert data["ttl_seconds"] == 3600
        db.close()

    def test_stats_reflect_cache_activity(self):
        client, db = _make_cached_client()
        with client:
            client.get("/api/v1/forecast?race=FL_Senate")  # miss
            client.get("/api/v1/forecast?race=FL_Senate")  # hit
            stats = client.get("/api/v1/cache/stats").json()
        # At least 1 hit and 1 miss after the above pattern
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        db.close()
