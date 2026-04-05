"""Benchmark API endpoint latency.

Hits key forecast/poll endpoints N times and records p50/p95 response times.
Results saved to data/benchmarks/YYYY-MM-DD.json for regression tracking.

Usage:
    uv run python scripts/benchmark_api_latency.py [--runs 50] [--base-url http://localhost:8002]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"

# Endpoints to benchmark (path, label)
ENDPOINTS = [
    ("/api/v1/health", "health"),
    ("/api/v1/forecast/races", "forecast_races"),
    ("/api/v1/forecast/race/2026-ga-senate", "race_detail"),
    ("/api/v1/polls", "polls"),
    ("/api/v1/types", "types_list"),
    ("/api/v1/types/0", "type_detail"),
    ("/api/v1/counties?state=GA", "counties_ga"),
    ("/api/v1/senate/overview", "senate_overview"),
    ("/api/v1/cache/stats", "cache_stats"),
]


def benchmark_endpoint(
    client: httpx.Client, url: str, runs: int
) -> dict[str, float]:
    """Hit an endpoint N times and return latency statistics in ms."""
    latencies: list[float] = []
    errors = 0

    for _ in range(runs):
        start = time.perf_counter()
        try:
            resp = client.get(url, timeout=10.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors += 1
        except httpx.RequestError:
            errors += 1

    if not latencies:
        return {"p50": -1, "p95": -1, "mean": -1, "errors": errors, "samples": 0}

    latencies.sort()
    p50_idx = int(len(latencies) * 0.50)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)

    return {
        "p50": round(latencies[p50_idx], 1),
        "p95": round(latencies[p95_idx], 1),
        "mean": round(statistics.mean(latencies), 1),
        "min": round(latencies[0], 1),
        "max": round(latencies[-1], 1),
        "errors": errors,
        "samples": len(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark WetherVane API latency")
    parser.add_argument("--runs", type=int, default=50, help="Requests per endpoint")
    parser.add_argument("--base-url", default="http://localhost:8002", help="API base URL")
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    results: dict[str, dict] = {}

    with httpx.Client(base_url=args.base_url) as client:
        # Warm up cache
        for path, _ in ENDPOINTS:
            try:
                client.get(path, timeout=10.0)
            except httpx.RequestError:
                pass

        for path, label in ENDPOINTS:
            stats = benchmark_endpoint(client, path, args.runs)
            results[label] = {**stats, "path": path}
            status = f"p50={stats['p50']}ms p95={stats['p95']}ms"
            if stats["errors"]:
                status += f" ({stats['errors']} errors)"
            print(f"  {label:25s} {status}")

    output = {
        "date": today,
        "base_url": args.base_url,
        "runs_per_endpoint": args.runs,
        "endpoints": results,
    }

    out_path = BENCHMARK_DIR / f"{today}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
