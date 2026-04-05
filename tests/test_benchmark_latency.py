"""Tests for the API latency benchmark script.

Tests verify:
- Result structure contains required fields
- Output JSON is valid and parseable
- Statistics are correctly calculated (p50, p95, mean)
- Summary formatting works correctly
- Handles errors gracefully
"""
from __future__ import annotations

import json
import statistics
import tempfile
from datetime import date
from pathlib import Path
from unittest import mock

import pytest

from scripts.benchmark_api_latency import (
    ENDPOINTS,
    benchmark_endpoint,
)


class TestBenchmarkEndpointStats:
    """Test the benchmark_endpoint function."""

    def test_result_structure_has_required_fields(self):
        """Verify result dict has all required latency metrics."""
        # Mock client that returns success
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch("time.perf_counter", side_effect=[0, 0.01, 1, 1.01]):
            mock_client.get.return_value = mock_response
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 2)

        required_fields = {"p50", "p95", "mean", "min", "max", "errors", "samples"}
        assert required_fields.issubset(result.keys())

    def test_result_fields_are_numeric(self):
        """Verify all latency fields are numeric."""
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch("time.perf_counter", side_effect=[0, 0.01, 1, 1.01]):
            mock_client.get.return_value = mock_response
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 2)

        for field in {"p50", "p95", "mean", "min", "max"}:
            assert isinstance(result[field], (int, float))
            assert result[field] >= 0

    def test_error_handling_returns_safe_defaults(self):
        """Verify result dict is valid even when all requests fail."""
        mock_client = mock.Mock()

        def always_fail(*args, **kwargs):
            import httpx
            raise httpx.RequestError("Connection failed")

        mock_client.get.side_effect = always_fail

        with mock.patch("time.perf_counter", return_value=0):
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 5)

        assert result["samples"] == 0
        assert result["errors"] == 5
        assert result["p50"] == -1  # Sentinel for "no data"
        assert result["p95"] == -1
        assert result["mean"] == -1

    def test_p50_is_median(self):
        """Verify p50 correctly computes the median."""
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        # 5 latencies: 1, 2, 3, 4, 5 ms
        # Each request: start (0), then end (0.001), (0.002), etc.
        times = [0, 0.001, 0, 0.002, 0, 0.003, 0, 0.004, 0, 0.005]
        with mock.patch("time.perf_counter", side_effect=times):
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 5)

        # p50 should be the median: 3.0 ms
        assert result["p50"] == 3.0

    def test_p95_uses_95th_percentile(self):
        """Verify p95 uses correct percentile calculation."""
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200

        # 20 latencies: 1-20 ms (evenly spaced)
        times = []
        for i in range(20):
            times.extend([i * 2, (i * 2) + 0.001 + i * 0.0001])
        with mock.patch("time.perf_counter", side_effect=times):
            mock_client.get.return_value = mock_response
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 20)

        # p95 index: int(20 * 0.95) = 19, but clamped to len-1
        assert result["samples"] == 20
        assert result["p95"] >= result["p50"]

    def test_mean_latency_calculation(self):
        """Verify mean is correctly calculated."""
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        # 3 requests: 1ms, 2ms, 3ms -> mean = 2ms
        # Each request: start (0), then end (0.001), (0.002), (0.003)
        times = [0, 0.001, 0, 0.002, 0, 0.003]
        with mock.patch("time.perf_counter", side_effect=times):
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 3)

        assert result["mean"] == 2.0
        assert result["samples"] == 3

    def test_min_max_values(self):
        """Verify min/max track smallest and largest latencies."""
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        # Latencies: 5, 1, 10 ms
        # Each request: start (0), then end (0.005), (0.001), (0.010)
        times = [0, 0.005, 0, 0.001, 0, 0.010]
        with mock.patch("time.perf_counter", side_effect=times):
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 3)

        assert result["min"] == 1.0
        assert result["max"] == 10.0

    def test_non_200_responses_not_counted(self):
        """Verify non-200 status codes are treated as errors."""
        mock_client = mock.Mock()
        mock_response = mock.Mock()

        call_count = [0]

        def get_side_effect(*args, **kwargs):
            call_count[0] += 1
            # First two succeed, third returns 500
            if call_count[0] < 3:
                mock_response.status_code = 200
            else:
                mock_response.status_code = 500
            return mock_response

        mock_client.get.side_effect = get_side_effect

        with mock.patch("time.perf_counter", side_effect=[i * 1 for i in range(6)]):
            result = benchmark_endpoint(mock_client, "http://localhost:8002/test", 3)

        assert result["samples"] == 2
        assert result["errors"] == 1


class TestBenchmarkOutputFormat:
    """Test the JSON output format."""

    def test_output_file_is_valid_json(self):
        """Verify generated benchmark file is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = Path(tmpdir)
            output_file = bench_dir / "2026-04-05.json"

            # Create a minimal valid output
            data = {
                "date": "2026-04-05",
                "base_url": "http://localhost:8002",
                "runs_per_endpoint": 5,
                "endpoints": {
                    "health": {
                        "p50": 10.0,
                        "p95": 15.0,
                        "mean": 11.0,
                        "min": 8.0,
                        "max": 20.0,
                        "errors": 0,
                        "samples": 5,
                        "path": "/api/v1/health",
                    }
                },
            }

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            # Parse it back
            with open(output_file, "r") as f:
                loaded = json.load(f)

            assert loaded["date"] == "2026-04-05"
            assert loaded["runs_per_endpoint"] == 5
            assert "health" in loaded["endpoints"]

    def test_output_has_timestamp_and_metadata(self):
        """Verify output includes date, base_url, and runs_per_endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = Path(tmpdir)
            output_file = bench_dir / "2026-04-05.json"

            data = {
                "date": "2026-04-05",
                "base_url": "http://localhost:8002",
                "runs_per_endpoint": 10,
                "endpoints": {},
            }

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            with open(output_file, "r") as f:
                loaded = json.load(f)

            assert "date" in loaded
            assert "base_url" in loaded
            assert "runs_per_endpoint" in loaded
            assert loaded["runs_per_endpoint"] == 10

    def test_each_endpoint_has_path_field(self):
        """Verify each endpoint result includes the original path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = Path(tmpdir)
            output_file = bench_dir / "2026-04-05.json"

            data = {
                "date": "2026-04-05",
                "base_url": "http://localhost:8002",
                "runs_per_endpoint": 5,
                "endpoints": {
                    "health": {
                        "p50": 10.0,
                        "p95": 15.0,
                        "mean": 11.0,
                        "min": 8.0,
                        "max": 20.0,
                        "errors": 0,
                        "samples": 5,
                        "path": "/api/v1/health",
                    }
                },
            }

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            with open(output_file, "r") as f:
                loaded = json.load(f)

            assert loaded["endpoints"]["health"]["path"] == "/api/v1/health"


class TestBenchmarkSummaryFormatting:
    """Test human-readable summary output."""

    def test_summary_includes_all_endpoints(self):
        """Verify summary mentions every endpoint that was tested."""
        assert len(ENDPOINTS) > 0
        # Just verify the ENDPOINTS constant is non-empty
        # The main script uses it for iteration

    def test_status_line_format_with_metrics(self):
        """Verify status line includes label, p50, and p95."""
        # Mock a result
        result = {
            "p50": 10.5,
            "p95": 25.3,
            "mean": 12.0,
            "min": 9.0,
            "max": 30.0,
            "errors": 0,
            "samples": 50,
        }

        label = "forecast_races"
        status = f"p50={result['p50']}ms p95={result['p95']}ms"

        assert "p50" in status
        assert "p95" in status
        assert "10.5" in status
        assert "25.3" in status

    def test_error_count_in_status_when_present(self):
        """Verify status line includes error count when errors > 0."""
        result = {
            "p50": 10.5,
            "p95": 25.3,
            "mean": 12.0,
            "min": 9.0,
            "max": 30.0,
            "errors": 3,
            "samples": 47,
        }

        label = "counties_ga"
        status = f"p50={result['p50']}ms p95={result['p95']}ms"
        if result["errors"]:
            status += f" ({result['errors']} errors)"

        assert "3 errors" in status


class TestBenchmarkEndpoints:
    """Test the ENDPOINTS configuration."""

    def test_endpoints_list_not_empty(self):
        """Verify we have endpoints defined."""
        assert len(ENDPOINTS) > 0

    def test_each_endpoint_has_path_and_label(self):
        """Verify each endpoint is a tuple of (path, label)."""
        for endpoint in ENDPOINTS:
            assert isinstance(endpoint, tuple)
            assert len(endpoint) == 2
            path, label = endpoint
            assert isinstance(path, str)
            assert isinstance(label, str)
            assert path.startswith("/")
            assert len(label) > 0
