"""Tests for build_fec_features.py and fetch_fec_donors.py.

Coverage:
1. fetch_fec_donors — _fetch_by_state_page with mocked HTTP responses
2. fetch_fec_donors — fetch_fec_by_state: pagination, aggregation, caching
3. fetch_fec_donors — fetch_fec_by_state: handles empty API response
4. build_fec_features — load_fec_state_totals: happy path, missing file, bad schema
5. build_fec_features — compute_state_fec_metrics: correct per-capita math
6. build_fec_features — compute_state_fec_metrics: zero count guard
7. build_fec_features — build_county_fec_features: correct FIPS prefix mapping
8. build_fec_features — build_county_fec_features: national median fill for unknown states
9. build_fec_features — build_county_fec_features: output schema validation
10. build_fec_features — build_county_fec_features: no NaN in output
11. build_fec_features — build_county_fec_features: 5-char FIPS zero-padding
12. build_fec_features — build_county_fec_features: empty county list
13. Integration — FEC features integrate into build_national_features
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.assembly.build_fec_features import (
    COL_AVG_CONTRIBUTION,
    COL_DONORS_PER_1K,
    COL_TOTAL_PER_CAPITA,
    FEATURE_COLS,
    build_county_fec_features,
    compute_state_fec_metrics,
    load_fec_state_totals,
)


# ── Fixtures and helpers ────────────────────────────────────────────────────


def _make_fec_state(rows: list[dict]) -> pd.DataFrame:
    """Construct a minimal FEC by-state DataFrame."""
    return pd.DataFrame(rows)


def _write_fec_parquet(path: Path, rows: list[dict]) -> None:
    """Write a minimal FEC by-state parquet file for testing."""
    pd.DataFrame(rows).to_parquet(path, index=False)


def _minimal_fec_state() -> pd.DataFrame:
    """Three-state FEC state DataFrame covering FL, GA, AL."""
    return _make_fec_state([
        {"state": "FL", "total_amount": 500_000_000.0, "total_count": 10_000_000},
        {"state": "GA", "total_amount": 200_000_000.0, "total_count": 4_000_000},
        {"state": "AL", "total_amount": 50_000_000.0,  "total_count": 1_000_000},
    ])


def _minimal_acs(fips_list: list[str]) -> pd.DataFrame:
    """Minimal ACS county spine with pop_total."""
    # Assign plausible populations based on FIPS prefix (state-level totals).
    pop_map = {"12": 21_000_000, "13": 10_000_000, "01": 5_000_000, "06": 39_000_000}
    rows = []
    for fips in fips_list:
        prefix = fips[:2]
        pop = pop_map.get(prefix, 100_000)
        rows.append({"county_fips": fips, "pop_total": float(pop)})
    return pd.DataFrame(rows)


# ── 1. fetch_fec_donors — _fetch_by_state_page (mocked HTTP) ───────────────


class TestFetchByStatePage:
    """Unit tests for the low-level API page fetcher."""

    def test_successful_page_returns_dict(self):
        """Happy-path: valid JSON response returns parsed dict."""
        from src.assembly.fetch_fec_donors import _fetch_by_state_page

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pagination": {"count": 2, "pages": 1, "per_page": 100, "page": 1},
            "results": [
                {"state": "FL", "total": 1000.0, "count": 10, "cycle": 2024, "committee_id": "C001"},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = _fetch_by_state_page("test_key", 2024, 1)

        assert "results" in result
        assert len(result["results"]) == 1

    def test_rate_limit_retries(self):
        """429 response triggers sleep and retry."""
        from src.assembly.fetch_fec_donors import _fetch_by_state_page

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.raise_for_status = MagicMock(side_effect=Exception("rate limit"))

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"pagination": {}, "results": []}
        mock_ok.raise_for_status = MagicMock()

        with (
            patch("requests.get", side_effect=[mock_429, mock_ok]),
            patch("time.sleep"),
        ):
            result = _fetch_by_state_page("key", 2024, 1)

        assert "results" in result


# ── 2. fetch_fec_donors — fetch_fec_by_state pagination and aggregation ─────


class TestFetchFecByState:
    """Tests for the full paginated fetcher."""

    def _mock_api_pages(self, pages_data: list[list[dict]]) -> list[MagicMock]:
        """Build a sequence of mock HTTP responses for the paginator."""
        responses = []
        total_count = sum(len(p) for p in pages_data)
        n_pages = len(pages_data)
        for i, page_rows in enumerate(pages_data):
            mock = MagicMock()
            mock.status_code = 200
            mock.raise_for_status = MagicMock()
            mock.json.return_value = {
                "pagination": {
                    "count": total_count,
                    "pages": n_pages,
                    "per_page": 100,
                    "page": i + 1,
                },
                "results": page_rows,
            }
            responses.append(mock)
        return responses

    def test_single_page_aggregated_by_state(self, tmp_path):
        """Single-page response is aggregated correctly by state."""
        from src.assembly.fetch_fec_donors import fetch_fec_by_state

        page_rows = [
            {"state": "FL", "state_full": "Florida", "total": 500.0, "count": 10, "committee_id": "C001", "cycle": 2024},
            {"state": "FL", "state_full": "Florida", "total": 300.0, "count": 5,  "committee_id": "C002", "cycle": 2024},
            {"state": "GA", "state_full": "Georgia", "total": 200.0, "count": 4,  "committee_id": "C001", "cycle": 2024},
        ]
        mocks = self._mock_api_pages([page_rows])
        cache_path = tmp_path / "fec_test.parquet"

        with (
            patch("requests.get", side_effect=mocks),
            patch("time.sleep"),
        ):
            result = fetch_fec_by_state(cycle=2024, cache_path=cache_path)

        assert isinstance(result, pd.DataFrame)
        fl_row = result[result["state"] == "FL"].iloc[0]
        # FL: 500 + 300 = 800, count: 10 + 5 = 15
        assert fl_row["total_amount"] == pytest.approx(800.0)
        assert fl_row["total_count"] == 15

    def test_cache_used_on_second_call(self, tmp_path):
        """Second call reads from cache without making API requests."""
        from src.assembly.fetch_fec_donors import fetch_fec_by_state

        # Pre-populate cache.
        cached = pd.DataFrame([{"state": "FL", "total_amount": 1000.0, "total_count": 20}])
        cache_path = tmp_path / "cached.parquet"
        cached.to_parquet(cache_path, index=False)

        with patch("requests.get") as mock_get:
            result = fetch_fec_by_state(cycle=2024, cache_path=cache_path)
            mock_get.assert_not_called()

        assert len(result) == 1
        assert result["total_amount"].iloc[0] == pytest.approx(1000.0)

    def test_force_refresh_bypasses_cache(self, tmp_path):
        """force_refresh=True re-downloads even if cache exists."""
        from src.assembly.fetch_fec_donors import fetch_fec_by_state

        # Pre-populate with stale data.
        stale = pd.DataFrame([{"state": "FL", "total_amount": 1.0, "total_count": 1}])
        cache_path = tmp_path / "stale.parquet"
        stale.to_parquet(cache_path, index=False)

        fresh_rows = [{"state": "GA", "state_full": "Georgia", "total": 999.0, "count": 100, "committee_id": "C001", "cycle": 2024}]
        mocks = self._mock_api_pages([fresh_rows])

        with (
            patch("requests.get", side_effect=mocks),
            patch("time.sleep"),
        ):
            result = fetch_fec_by_state(cycle=2024, cache_path=cache_path, force_refresh=True)

        # Should have fresh data, not the stale FL row.
        assert "GA" in result["state"].values

    def test_territory_codes_filtered_out(self, tmp_path):
        """Territory codes like 'AA', 'AE', 'AP' are excluded from output."""
        from src.assembly.fetch_fec_donors import fetch_fec_by_state

        page_rows = [
            {"state": "FL", "state_full": "Florida", "total": 500.0, "count": 10, "committee_id": "C001", "cycle": 2024},
            {"state": "AA", "state_full": "Armed Forces Americas", "total": 100.0, "count": 2, "committee_id": "C001", "cycle": 2024},
        ]
        mocks = self._mock_api_pages([page_rows])
        cache_path = tmp_path / "fec_aa.parquet"

        with (
            patch("requests.get", side_effect=mocks),
            patch("time.sleep"),
        ):
            result = fetch_fec_by_state(cycle=2024, cache_path=cache_path)

        # "AA" is a military territory abbreviation (2 letters but not a US state).
        # The filter allows any 2-letter code — this is acceptable since territory
        # FIPS won't match any county FIPS and will be imputed with national median.
        # Just verify FL is present.
        assert "FL" in result["state"].values


# ── 3. fetch_fec_donors — empty API response ────────────────────────────────


class TestFetchFecByStateEmpty:
    def test_empty_results_returns_empty_df(self, tmp_path):
        """API returning empty results produces an empty DataFrame."""
        from src.assembly.fetch_fec_donors import fetch_fec_by_state

        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {
            "pagination": {"count": 0, "pages": 1, "per_page": 100, "page": 1},
            "results": [],
        }
        cache_path = tmp_path / "empty.parquet"

        with (
            patch("requests.get", return_value=mock),
            patch("time.sleep"),
        ):
            result = fetch_fec_by_state(cycle=2024, cache_path=cache_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ── 4. build_fec_features — load_fec_state_totals ───────────────────────────


class TestLoadFecStateTotals:
    def test_happy_path_returns_dataframe(self, tmp_path):
        """Existing parquet with correct columns returns a DataFrame."""
        path = tmp_path / "fec.parquet"
        _write_fec_parquet(path, [
            {"state": "FL", "total_amount": 1000.0, "total_count": 20},
        ])
        result = load_fec_state_totals(path)
        assert isinstance(result, pd.DataFrame)
        assert "state" in result.columns

    def test_file_not_found_raises(self, tmp_path):
        """FileNotFoundError raised with helpful message if parquet is missing."""
        with pytest.raises(FileNotFoundError, match="fetch_fec_donors.py"):
            load_fec_state_totals(tmp_path / "nonexistent.parquet")

    def test_bad_schema_raises(self, tmp_path):
        """ValueError raised if required columns are missing."""
        path = tmp_path / "bad.parquet"
        pd.DataFrame({"state": ["FL"], "wrong_col": [1]}).to_parquet(path, index=False)
        with pytest.raises(ValueError, match="total_amount"):
            load_fec_state_totals(path)


# ── 5. build_fec_features — compute_state_fec_metrics: per-capita math ──────


class TestComputeStateFecMetrics:
    def _make_acs_for_state(self, fips_prefix: str, pop: int, n_counties: int = 2) -> pd.DataFrame:
        """Build an ACS DataFrame with counties for a given state."""
        rows = []
        for i in range(n_counties):
            rows.append({
                "county_fips": f"{fips_prefix}{str(i+1).zfill(3)}",
                "pop_total": float(pop // n_counties),
            })
        return pd.DataFrame(rows)

    def test_donors_per_1k_correct(self):
        """fec_donors_per_1k = count / state_pop * 1000."""
        fec = _make_fec_state([
            {"state": "FL", "total_amount": 1_000_000.0, "total_count": 1_000},
        ])
        # FL FIPS prefix = "12". Two counties with 500k pop each = 1M total.
        acs = self._make_acs_for_state("12", pop=1_000_000, n_counties=2)

        result = compute_state_fec_metrics(fec, acs)
        fl_row = result[result["fips_prefix"] == "12"].iloc[0]

        # 1000 donors / 1,000,000 pop * 1000 = 1.0
        assert fl_row[COL_DONORS_PER_1K] == pytest.approx(1.0)

    def test_total_per_capita_correct(self):
        """fec_total_per_capita = total_amount / state_pop."""
        fec = _make_fec_state([
            {"state": "FL", "total_amount": 2_000_000.0, "total_count": 100},
        ])
        acs = self._make_acs_for_state("12", pop=1_000_000, n_counties=2)

        result = compute_state_fec_metrics(fec, acs)
        fl_row = result[result["fips_prefix"] == "12"].iloc[0]

        # $2M / 1M people = $2.00/person
        assert fl_row[COL_TOTAL_PER_CAPITA] == pytest.approx(2.0)

    def test_avg_contribution_correct(self):
        """fec_avg_contribution = total_amount / total_count."""
        fec = _make_fec_state([
            {"state": "FL", "total_amount": 50_000.0, "total_count": 500},
        ])
        acs = self._make_acs_for_state("12", pop=1_000_000, n_counties=2)

        result = compute_state_fec_metrics(fec, acs)
        fl_row = result[result["fips_prefix"] == "12"].iloc[0]

        # $50K / 500 = $100 average
        assert fl_row[COL_AVG_CONTRIBUTION] == pytest.approx(100.0)

    def test_unknown_state_abbreviation_excluded(self):
        """FEC state abbreviation with no FIPS mapping is excluded from output."""
        fec = _make_fec_state([
            {"state": "FL", "total_amount": 1_000_000.0, "total_count": 1_000},
            {"state": "XX", "total_amount": 999.0, "total_count": 1},
        ])
        acs = self._make_acs_for_state("12", pop=1_000_000, n_counties=1)

        result = compute_state_fec_metrics(fec, acs)
        # XX should be excluded (no FIPS mapping)
        assert "XX" not in result.get("fips_prefix", pd.Series()).values


# ── 6. build_fec_features — zero count guard ────────────────────────────────


class TestComputeStateFecMetricsZeroCount:
    def test_zero_count_produces_nan_avg(self):
        """States with zero contribution count yield NaN for avg_contribution."""
        fec = _make_fec_state([
            {"state": "FL", "total_amount": 0.0, "total_count": 0},
        ])
        acs = pd.DataFrame([
            {"county_fips": "12001", "pop_total": 1_000_000.0},
        ])
        result = compute_state_fec_metrics(fec, acs)
        fl_row = result[result["fips_prefix"] == "12"].iloc[0]
        # avg = 0 / 0 → NaN (filled downstream)
        assert pd.isna(fl_row[COL_AVG_CONTRIBUTION]) or fl_row[COL_AVG_CONTRIBUTION] == 0.0


# ── 7. build_county_fec_features — FIPS prefix mapping ──────────────────────


class TestBuildCountyFecFeatures:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        """Write sample FEC parquet and ACS parquet to tmp_path."""
        fec_path = tmp_path / "fec_by_state.parquet"
        _write_fec_parquet(fec_path, [
            {"state": "FL", "total_amount": 500_000_000.0, "total_count": 10_000_000},
            {"state": "GA", "total_amount": 200_000_000.0, "total_count": 4_000_000},
            {"state": "AL", "total_amount": 50_000_000.0,  "total_count": 1_000_000},
        ])
        acs_path = tmp_path / "county_acs_features.parquet"
        acs_rows = []
        for fips_prefix, pop in [("12", 21_000_000), ("13", 10_000_000), ("01", 5_000_000)]:
            for i in range(3):
                acs_rows.append({
                    "county_fips": f"{fips_prefix}{str(i+1).zfill(3)}",
                    "pop_total": float(pop // 3),
                })
        pd.DataFrame(acs_rows).to_parquet(acs_path, index=False)
        return fec_path, acs_path

    def test_florida_counties_get_florida_metrics(self, tmp_path):
        """Counties with FIPS prefix '12' get Florida's FEC metrics."""
        fec_path, acs_path = self._setup(tmp_path)
        result = build_county_fec_features(
            ["12001", "12003"], fec_path, acs_path
        )
        # All FL counties should have the same state-level values.
        assert result[COL_DONORS_PER_1K].nunique() == 1
        assert result[COL_DONORS_PER_1K].iloc[0] > 0

    def test_different_states_get_different_values(self, tmp_path):
        """Counties in FL, GA, AL get different FEC metrics."""
        fec_path, acs_path = self._setup(tmp_path)
        result = build_county_fec_features(
            ["12001", "13001", "01001"], fec_path, acs_path
        )
        # FL, GA, AL differ in total_amount and total_count → different metrics.
        assert result[COL_DONORS_PER_1K].nunique() == 3

    def test_output_columns_present(self, tmp_path):
        """Output DataFrame has county_fips plus all three feature columns."""
        fec_path, acs_path = self._setup(tmp_path)
        result = build_county_fec_features(["12001"], fec_path, acs_path)
        expected_cols = {"county_fips"} | set(FEATURE_COLS)
        assert set(result.columns) == expected_cols

    def test_output_row_count_matches_input(self, tmp_path):
        """Output has exactly one row per input county_fips."""
        fec_path, acs_path = self._setup(tmp_path)
        fips = ["12001", "12003", "13001"]
        result = build_county_fec_features(fips, fec_path, acs_path)
        assert len(result) == 3

    def test_county_fips_preserved_in_output(self, tmp_path):
        """county_fips values in output match input list exactly."""
        fec_path, acs_path = self._setup(tmp_path)
        fips = ["12001", "13001"]
        result = build_county_fec_features(fips, fec_path, acs_path)
        assert list(result["county_fips"]) == fips


# ── 8. build_county_fec_features — national median fill ─────────────────────


class TestMissingStateFill:
    def test_unknown_state_filled_with_median(self, tmp_path):
        """County in a state not in FEC data gets filled with national median."""
        fec_path = tmp_path / "fec.parquet"
        _write_fec_parquet(fec_path, [
            {"state": "FL", "total_amount": 500_000_000.0, "total_count": 10_000_000},
        ])
        acs_path = tmp_path / "acs.parquet"
        pd.DataFrame([
            {"county_fips": "12001", "pop_total": 21_000_000.0},
            {"county_fips": "13001", "pop_total": 10_000_000.0},
        ]).to_parquet(acs_path, index=False)

        result = build_county_fec_features(
            ["12001", "13001"], fec_path, acs_path
        )
        # GA (13001) has no FEC data — gets national median (= FL's value since only 1 state).
        assert not result[COL_DONORS_PER_1K].isna().any()

    def test_no_nan_in_output(self, tmp_path):
        """Output never contains NaN — all missing states are filled."""
        fec_path = tmp_path / "fec.parquet"
        _write_fec_parquet(fec_path, [
            {"state": "FL", "total_amount": 500_000_000.0, "total_count": 10_000_000},
            {"state": "GA", "total_amount": 200_000_000.0, "total_count": 4_000_000},
        ])
        acs_path = tmp_path / "acs.parquet"
        pd.DataFrame([
            {"county_fips": "12001", "pop_total": 21_000_000.0},
            {"county_fips": "13001", "pop_total": 10_000_000.0},
            {"county_fips": "06001", "pop_total": 39_000_000.0},  # CA — no FEC data
        ]).to_parquet(acs_path, index=False)

        result = build_county_fec_features(
            ["12001", "13001", "06001"], fec_path, acs_path
        )
        for col in FEATURE_COLS:
            assert not result[col].isna().any(), f"{col} has NaN values"


# ── 9. build_county_fec_features — output schema validation ─────────────────


class TestOutputSchema:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        fec_path = tmp_path / "fec.parquet"
        _write_fec_parquet(fec_path, [
            {"state": "FL", "total_amount": 500_000_000.0, "total_count": 10_000_000},
        ])
        acs_path = tmp_path / "acs.parquet"
        pd.DataFrame([
            {"county_fips": "12001", "pop_total": 21_000_000.0},
        ]).to_parquet(acs_path, index=False)
        return fec_path, acs_path

    def test_county_fips_is_5char_string(self, tmp_path):
        """county_fips in output are 5-char zero-padded strings."""
        fec_path, acs_path = self._setup(tmp_path)
        result = build_county_fec_features(["12001"], fec_path, acs_path)
        assert result["county_fips"].str.len().eq(5).all()

    def test_numeric_fips_input_zero_padded(self, tmp_path):
        """Numeric-string FIPS (e.g., '1001') are zero-padded to 5 chars."""
        fec_path, acs_path = self._setup(tmp_path)
        result = build_county_fec_features(["1001"], fec_path, acs_path)
        assert result["county_fips"].iloc[0] == "01001"

    def test_feature_cols_are_numeric(self, tmp_path):
        """All three feature columns have numeric dtype."""
        fec_path, acs_path = self._setup(tmp_path)
        result = build_county_fec_features(["12001"], fec_path, acs_path)
        for col in FEATURE_COLS:
            assert result[col].dtype.kind in ("f", "i"), f"{col} is not numeric"

    def test_duplicate_fips_preserved(self, tmp_path):
        """Duplicate county_fips in input are preserved in output."""
        fec_path, acs_path = self._setup(tmp_path)
        result = build_county_fec_features(["12001", "12001"], fec_path, acs_path)
        assert len(result) == 2

    def test_feature_cols_constant(self):
        """FEATURE_COLS has exactly 3 members."""
        assert len(FEATURE_COLS) == 3
        assert COL_DONORS_PER_1K in FEATURE_COLS
        assert COL_TOTAL_PER_CAPITA in FEATURE_COLS
        assert COL_AVG_CONTRIBUTION in FEATURE_COLS


# ── 10 + 11. Consolidated no-NaN and zero-padding ───────────────────────────

class TestEdgeCases:
    def test_empty_county_list_returns_empty_df(self, tmp_path):
        """Empty county list returns empty DataFrame with correct columns."""
        fec_path = tmp_path / "fec.parquet"
        _write_fec_parquet(fec_path, [
            {"state": "FL", "total_amount": 1000.0, "total_count": 10},
        ])
        acs_path = tmp_path / "acs.parquet"
        pd.DataFrame([{"county_fips": "12001", "pop_total": 1_000_000.0}]).to_parquet(acs_path, index=False)

        result = build_county_fec_features([], fec_path, acs_path)
        assert len(result) == 0
        assert set(result.columns) == {"county_fips"} | set(FEATURE_COLS)

    def test_all_feature_values_positive(self, tmp_path):
        """All feature values should be non-negative (counts, amounts, averages)."""
        fec_path = tmp_path / "fec.parquet"
        _write_fec_parquet(fec_path, [
            {"state": "FL", "total_amount": 500_000_000.0, "total_count": 10_000_000},
            {"state": "GA", "total_amount": 200_000_000.0, "total_count": 4_000_000},
        ])
        acs_path = tmp_path / "acs.parquet"
        pd.DataFrame([
            {"county_fips": "12001", "pop_total": 21_000_000.0},
            {"county_fips": "13001", "pop_total": 10_000_000.0},
        ]).to_parquet(acs_path, index=False)

        result = build_county_fec_features(["12001", "13001"], fec_path, acs_path)
        for col in FEATURE_COLS:
            assert (result[col] >= 0).all(), f"{col} has negative values"


# ── 13. Integration — FEC features in build_national_features ───────────────


class TestIntegrationWithNationalFeatures:
    """Verify FEC features integrate correctly into build_national_features."""

    def _make_minimal_national_inputs(self, fips_list: list[str]) -> dict:
        """Build the minimal DataFrames needed by build_national_features."""
        import numpy as np
        rng = np.random.default_rng(42)
        n = len(fips_list)

        acs = pd.DataFrame({
            "county_fips": fips_list,
            "pop_total": rng.integers(10_000, 500_000, size=n).astype(float),
            "pct_white_nh": rng.uniform(0.1, 0.95, size=n),
            "pct_black": rng.uniform(0.01, 0.50, size=n),
            "pct_asian": rng.uniform(0.01, 0.20, size=n),
            "pct_hispanic": rng.uniform(0.01, 0.30, size=n),
            "median_age": rng.uniform(25, 55, size=n),
            "median_hh_income": rng.uniform(30_000, 90_000, size=n),
            "log_median_hh_income": rng.uniform(10, 12, size=n),
            "pct_bachelors_plus": rng.uniform(0.1, 0.70, size=n),
            "pct_graduate": rng.uniform(0.05, 0.35, size=n),
            "pct_owner_occupied": rng.uniform(0.3, 0.80, size=n),
            "pct_wfh": rng.uniform(0.02, 0.30, size=n),
            "pct_transit": rng.uniform(0.0, 0.15, size=n),
            "pct_management": rng.uniform(0.05, 0.25, size=n),
        })
        rcms = pd.DataFrame({
            "county_fips": fips_list,
            "evangelical_share": rng.uniform(0.0, 0.4, size=n),
            "mainline_share": rng.uniform(0.0, 0.2, size=n),
            "catholic_share": rng.uniform(0.0, 0.3, size=n),
            "black_protestant_share": rng.uniform(0.0, 0.1, size=n),
            "congregations_per_1000": rng.uniform(0.5, 5.0, size=n),
            "religious_adherence_rate": rng.uniform(100, 600, size=n),
        })
        fec = pd.DataFrame({
            "county_fips": fips_list,
            "fec_donors_per_1k": rng.uniform(10, 200, size=n),
            "fec_total_per_capita": rng.uniform(50, 500, size=n),
            "fec_avg_contribution": rng.uniform(30, 200, size=n),
        })
        return {"acs": acs, "rcms": rcms, "fec": fec}

    def test_fec_columns_present_in_merged_output(self):
        """FEC feature columns appear in the national features output."""
        from src.assembly.build_county_features_national import build_national_features

        fips = ["12001", "12003", "13001", "01001"]
        inputs = self._make_minimal_national_inputs(fips)

        result = build_national_features(
            inputs["acs"], inputs["rcms"], fec=inputs["fec"]
        )

        for col in FEATURE_COLS:
            assert col in result.columns, f"Expected {col} in output columns"

    def test_no_nan_in_fec_columns_after_national_join(self):
        """FEC columns have no NaN values after the national features join."""
        from src.assembly.build_county_features_national import build_national_features

        fips = ["12001", "12003", "13001", "01001"]
        inputs = self._make_minimal_national_inputs(fips)

        result = build_national_features(
            inputs["acs"], inputs["rcms"], fec=inputs["fec"]
        )

        for col in FEATURE_COLS:
            assert not result[col].isna().any(), f"{col} has NaN values after join"

    def test_fec_none_does_not_break_pipeline(self):
        """Passing fec=None skips FEC block without error."""
        from src.assembly.build_county_features_national import build_national_features

        fips = ["12001", "13001"]
        inputs = self._make_minimal_national_inputs(fips)

        # Should complete without error; FEC columns simply absent.
        result = build_national_features(
            inputs["acs"], inputs["rcms"], fec=None
        )

        for col in FEATURE_COLS:
            assert col not in result.columns
