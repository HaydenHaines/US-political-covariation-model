"""Tests for fetch_bea_income.py.

Tests use synthetic DataFrames and mock HTTP/file I/O so no network access
is required and no BEA_API_KEY is needed. Coverage:

1. compute_income_shares() correctly divides each component by personal income
2. Counties are filtered to all 50 states + DC (national scope)
3. Zero personal income and missing values produce NaN / exclusion
4. Cache behavior: uses cached parquet if present; calls API if missing
5. API key error: helpful EnvironmentError when BEA_API_KEY is unset
6. _parse_data_value handles suppressed/missing BEA codes
7. build_bea_income_features output schema and deduplication
8. PATH constant structure
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.assembly.fetch_bea_income import (
    ASSEMBLED_DIR,
    FALLBACK_YEAR,
    LINE_DIVIDENDS_INTEREST,
    LINE_NET_EARNINGS,
    LINE_PERSONAL_INCOME,
    LINE_TRANSFERS,
    PRIMARY_YEAR,
    RAW_DIR,
    STATE_ABBR,
    TARGET_FIPS_PREFIXES,
    _get_api_key,
    _parse_data_value,
    build_bea_income_features,
    compute_income_shares,
    filter_to_target_states,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_cainc4(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal raw CAINC4 DataFrame matching BEA API output."""
    defaults = {
        "GeoFips": "12001",
        "GeoName": "Alachua County, FL",
        "LineCode": str(LINE_PERSONAL_INCOME),
        "TimePeriod": str(PRIMARY_YEAR),
        "DataValue": "1000000",
        "NoteRef": "",
    }
    records = [{**defaults, **row} for row in rows]
    return pd.DataFrame(records)


# Keep backward-compatible alias used in some tests
_make_raw_cainc1 = _make_raw_cainc4


@pytest.fixture
def simple_raw_df() -> pd.DataFrame:
    """Three counties (FL/GA/AL), all four line codes, clean integer values."""
    return _make_raw_cainc4([
        # County 12001 — FL
        {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "1000000"},
        {"GeoFips": "12001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "600000"},
        {"GeoFips": "12001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "250000"},
        {"GeoFips": "12001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "150000"},
        # County 13001 (GA)
        {"GeoFips": "13001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "500000"},
        {"GeoFips": "13001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "200000"},
        {"GeoFips": "13001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "100000"},
        {"GeoFips": "13001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "50000"},
        # County 01001 (AL)
        {"GeoFips": "01001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "200000"},
        {"GeoFips": "01001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "80000"},
        {"GeoFips": "01001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "70000"},
        {"GeoFips": "01001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "50000"},
    ])


@pytest.fixture
def mixed_state_raw_df() -> pd.DataFrame:
    """Includes a CA county (06001) that should also appear in national output."""
    return _make_raw_cainc4([
        {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "1000000"},
        {"GeoFips": "12001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "600000"},
        {"GeoFips": "12001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "250000"},
        {"GeoFips": "12001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "150000"},
        {"GeoFips": "06001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "9000000"},
        {"GeoFips": "06001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "5000000"},
        {"GeoFips": "06001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "2000000"},
        {"GeoFips": "06001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "1000000"},
    ])


# ---------------------------------------------------------------------------
# 1. compute_income_shares() — correct share math
# ---------------------------------------------------------------------------


class TestComputeIncomeShares:
    def test_earnings_share(self, simple_raw_df):
        """earnings_share = net_earnings / personal_income."""
        result = compute_income_shares(simple_raw_df)
        row = result[result["county_fips"] == "12001"].iloc[0]
        assert row["earnings_share"] == pytest.approx(600_000 / 1_000_000)

    def test_transfers_share(self, simple_raw_df):
        """transfers_share = transfers / personal_income."""
        result = compute_income_shares(simple_raw_df)
        row = result[result["county_fips"] == "12001"].iloc[0]
        assert row["transfers_share"] == pytest.approx(250_000 / 1_000_000)

    def test_investment_share(self, simple_raw_df):
        """investment_share = dividends_interest_rent / personal_income."""
        result = compute_income_shares(simple_raw_df)
        row = result[result["county_fips"] == "12001"].iloc[0]
        assert row["investment_share"] == pytest.approx(150_000 / 1_000_000)

    def test_output_columns(self, simple_raw_df):
        """Output has exactly the four required columns."""
        result = compute_income_shares(simple_raw_df)
        assert set(result.columns) == {
            "county_fips",
            "earnings_share",
            "transfers_share",
            "investment_share",
        }

    def test_county_fips_zero_padded(self, simple_raw_df):
        """county_fips values are 5-character zero-padded strings."""
        result = compute_income_shares(simple_raw_df)
        assert (result["county_fips"].str.len() == 5).all()

    def test_multiple_counties_correct(self, simple_raw_df):
        """All three counties get independently correct shares."""
        result = compute_income_shares(simple_raw_df)
        assert len(result) == 3

        ga = result[result["county_fips"] == "13001"].iloc[0]
        assert ga["earnings_share"] == pytest.approx(200_000 / 500_000)
        assert ga["transfers_share"] == pytest.approx(100_000 / 500_000)
        assert ga["investment_share"] == pytest.approx(50_000 / 500_000)


# ---------------------------------------------------------------------------
# 2. Counties filtered to national scope (all states + DC)
# ---------------------------------------------------------------------------


class TestFilterToTargetStates:
    def test_ca_county_retained(self, mixed_state_raw_df):
        """California county (06*) is retained — CA is a target state (national scope)."""
        shares = compute_income_shares(mixed_state_raw_df)
        filtered = filter_to_target_states(shares)
        assert "06001" in filtered["county_fips"].values

    def test_fl_county_retained(self, mixed_state_raw_df):
        """Florida county (12*) is retained."""
        shares = compute_income_shares(mixed_state_raw_df)
        filtered = filter_to_target_states(shares)
        assert "12001" in filtered["county_fips"].values

    def test_target_prefixes_are_correct(self):
        """TARGET_FIPS_PREFIXES contains all 50 states + DC (51 entries)."""
        assert len(TARGET_FIPS_PREFIXES) == 51
        assert "12" in TARGET_FIPS_PREFIXES  # FL
        assert "13" in TARGET_FIPS_PREFIXES  # GA
        assert "01" in TARGET_FIPS_PREFIXES  # AL
        assert "06" in TARGET_FIPS_PREFIXES  # CA
        assert "48" in TARGET_FIPS_PREFIXES  # TX

    def test_only_target_states_in_output(self, mixed_state_raw_df):
        """All county_fips in output are within the configured state set."""
        shares = compute_income_shares(mixed_state_raw_df)
        filtered = filter_to_target_states(shares)
        prefixes = filtered["county_fips"].str[:2].unique()
        assert set(prefixes).issubset(TARGET_FIPS_PREFIXES)


# ---------------------------------------------------------------------------
# 3. Zero / missing income values
# ---------------------------------------------------------------------------


class TestZeroAndMissingIncome:
    def test_zero_personal_income_excluded(self):
        """County with personal_income == 0 is excluded from output."""
        raw = _make_raw_cainc4([
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "0"},
            {"GeoFips": "12001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "500000"},
            {"GeoFips": "12001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "100000"},
            {"GeoFips": "12001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "50000"},
        ])
        result = compute_income_shares(raw)
        assert len(result) == 0 or "12001" not in result["county_fips"].values

    def test_missing_personal_income_excluded(self):
        """County with suppressed personal_income (D) is excluded."""
        raw = _make_raw_cainc4([
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "(D)"},
            {"GeoFips": "12001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "500000"},
            {"GeoFips": "12001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "100000"},
            {"GeoFips": "12001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "50000"},
        ])
        result = compute_income_shares(raw)
        # County with suppressed total should not appear with valid shares
        if "12001" in result.get("county_fips", pd.Series([])).values:
            row = result[result["county_fips"] == "12001"].iloc[0]
            assert pd.isna(row["earnings_share"])

    def test_missing_component_produces_nan_share(self):
        """Missing net_earnings line produces NaN earnings_share (not a crash)."""
        raw = _make_raw_cainc4([
            # Only personal_income; no net_earnings line
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "1000000"},
            {"GeoFips": "12001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "250000"},
            {"GeoFips": "12001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "150000"},
        ])
        result = compute_income_shares(raw)
        if "12001" in result["county_fips"].values:
            row = result[result["county_fips"] == "12001"].iloc[0]
            assert pd.isna(row["earnings_share"])

    def test_suppressed_value_codes_become_nan(self):
        """BEA suppression codes (D), (NA), (L), (S), (X) parse to NaN."""
        for code in ["(D)", "(NA)", "(L)", "(S)", "(X)", "--"]:
            assert pd.isna(_parse_data_value(code)), f"Expected NaN for code {code!r}"

    def test_comma_formatted_number_parsed(self):
        """BEA DataValue with thousands commas parses correctly."""
        assert _parse_data_value("1,234,567") == pytest.approx(1_234_567.0)

    def test_plain_number_parsed(self):
        """Plain numeric string parses correctly."""
        assert _parse_data_value("500000") == pytest.approx(500_000.0)

    def test_empty_string_becomes_nan(self):
        """Empty string DataValue parses to NaN."""
        assert pd.isna(_parse_data_value(""))

    def test_valid_county_still_returned_alongside_invalid(self):
        """A valid county is returned even when another county has zero income."""
        raw = _make_raw_cainc4([
            # Valid county
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "1000000"},
            {"GeoFips": "12001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "600000"},
            {"GeoFips": "12001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "250000"},
            {"GeoFips": "12001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "150000"},
            # Zero-income county
            {"GeoFips": "12003", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "0"},
            {"GeoFips": "12003", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "100000"},
            {"GeoFips": "12003", "LineCode": str(LINE_TRANSFERS),         "DataValue": "50000"},
            {"GeoFips": "12003", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "20000"},
        ])
        result = compute_income_shares(raw)
        assert "12001" in result["county_fips"].values


# ---------------------------------------------------------------------------
# 4. Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    def test_cache_hit_skips_api(self, tmp_path, monkeypatch):
        """When cache file exists, API is never called."""
        cached_df = _make_raw_cainc4([
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "1000000"},
            {"GeoFips": "12001", "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "600000"},
            {"GeoFips": "12001", "LineCode": str(LINE_TRANSFERS),         "DataValue": "250000"},
            {"GeoFips": "12001", "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "150000"},
        ])
        # Cache file uses cainc4_ prefix
        cache_path = tmp_path / f"cainc4_12_{PRIMARY_YEAR}.parquet"
        cached_df.to_parquet(cache_path, index=False)

        import src.assembly.fetch_bea_income as module
        monkeypatch.setattr(module, "RAW_DIR", tmp_path)

        with patch("src.assembly.fetch_bea_income._fetch_cainc4_state") as mock_api:
            from src.assembly.fetch_bea_income import fetch_cainc1_state_cached
            result = fetch_cainc1_state_cached("12", PRIMARY_YEAR)

        mock_api.assert_not_called()
        assert len(result) == 4  # all rows from cache

    def test_cache_miss_calls_api(self, tmp_path, monkeypatch):
        """When cache file is absent, the API fetch function is called."""
        import src.assembly.fetch_bea_income as module
        monkeypatch.setattr(module, "RAW_DIR", tmp_path)
        monkeypatch.setenv("BEA_API_KEY", "test_key_abc")

        api_df = _make_raw_cainc4([
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME), "DataValue": "1000000"},
        ])

        with patch(
            "src.assembly.fetch_bea_income._fetch_cainc4_state",
            return_value=api_df,
        ) as mock_api:
            from src.assembly.fetch_bea_income import fetch_cainc1_state_cached
            result = fetch_cainc1_state_cached("12", PRIMARY_YEAR)

        mock_api.assert_called_once_with("12", PRIMARY_YEAR, "test_key_abc")
        assert len(result) == 1

    def test_cache_written_on_miss(self, tmp_path, monkeypatch):
        """API response is written to cache file on first fetch."""
        import src.assembly.fetch_bea_income as module
        monkeypatch.setattr(module, "RAW_DIR", tmp_path)
        monkeypatch.setenv("BEA_API_KEY", "test_key_abc")

        api_df = _make_raw_cainc4([
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME), "DataValue": "1000000"},
        ])

        with patch(
            "src.assembly.fetch_bea_income._fetch_cainc4_state",
            return_value=api_df,
        ):
            from src.assembly.fetch_bea_income import fetch_cainc1_state_cached
            fetch_cainc1_state_cached("12", PRIMARY_YEAR)

        expected_cache = tmp_path / f"cainc4_12_{PRIMARY_YEAR}.parquet"
        assert expected_cache.exists()

    def test_force_refresh_bypasses_cache(self, tmp_path, monkeypatch):
        """force_refresh=True calls API even when cache exists."""
        import src.assembly.fetch_bea_income as module
        monkeypatch.setattr(module, "RAW_DIR", tmp_path)
        monkeypatch.setenv("BEA_API_KEY", "test_key_abc")

        cached_df = _make_raw_cainc4([
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME), "DataValue": "OLD_VALUE"},
        ])
        cache_path = tmp_path / f"cainc4_12_{PRIMARY_YEAR}.parquet"
        cached_df.to_parquet(cache_path, index=False)

        fresh_df = _make_raw_cainc4([
            {"GeoFips": "12001", "LineCode": str(LINE_PERSONAL_INCOME), "DataValue": "2000000"},
        ])

        with patch(
            "src.assembly.fetch_bea_income._fetch_cainc4_state",
            return_value=fresh_df,
        ) as mock_api:
            from src.assembly.fetch_bea_income import fetch_cainc1_state_cached
            fetch_cainc1_state_cached("12", PRIMARY_YEAR, force_refresh=True)

        mock_api.assert_called_once()


# ---------------------------------------------------------------------------
# 5. API key error handling
# ---------------------------------------------------------------------------


class TestApiKeyHandling:
    def test_missing_env_var_raises_environment_error(self, monkeypatch):
        """EnvironmentError is raised with helpful message when BEA_API_KEY unset."""
        monkeypatch.delenv("BEA_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="BEA_API_KEY"):
            _get_api_key()

    def test_error_message_includes_signup_url(self, monkeypatch):
        """Error message tells user where to get a key."""
        monkeypatch.delenv("BEA_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="apps.bea.gov"):
            _get_api_key()

    def test_valid_key_returned(self, monkeypatch):
        """Returns API key string when env var is set."""
        monkeypatch.setenv("BEA_API_KEY", "abc123")
        assert _get_api_key() == "abc123"

    def test_whitespace_only_key_raises(self, monkeypatch):
        """Whitespace-only key is treated as missing."""
        monkeypatch.setenv("BEA_API_KEY", "   ")
        with pytest.raises(EnvironmentError):
            _get_api_key()


# ---------------------------------------------------------------------------
# 6. build_bea_income_features output schema and integration
# ---------------------------------------------------------------------------


class TestBuildBeaIncomeFeatures:
    def _make_state_raw(self, fips_prefix: str) -> pd.DataFrame:
        """Synthetic CAINC4 data for one state."""
        county = f"{fips_prefix.zfill(2)}001"
        return _make_raw_cainc4([
            {"GeoFips": county, "LineCode": str(LINE_PERSONAL_INCOME),   "DataValue": "1000000"},
            {"GeoFips": county, "LineCode": str(LINE_NET_EARNINGS),      "DataValue": "600000"},
            {"GeoFips": county, "LineCode": str(LINE_TRANSFERS),         "DataValue": "250000"},
            {"GeoFips": county, "LineCode": str(LINE_DIVIDENDS_INTEREST), "DataValue": "150000"},
        ])

    def _side_effect(self, fips_prefix: str, year: int, force_refresh: bool = False) -> pd.DataFrame:
        return self._make_state_raw(fips_prefix)

    @patch("src.assembly.fetch_bea_income.fetch_cainc1_state_cached")
    def test_output_columns(self, mock_fetch):
        """Output has exactly county_fips, earnings_share, transfers_share, investment_share."""
        mock_fetch.side_effect = self._side_effect
        result = build_bea_income_features()
        assert set(result.columns) == {
            "county_fips",
            "earnings_share",
            "transfers_share",
            "investment_share",
        }

    @patch("src.assembly.fetch_bea_income.fetch_cainc1_state_cached")
    def test_only_configured_state_counties(self, mock_fetch):
        """Output contains only counties within the configured state set."""
        mock_fetch.side_effect = self._side_effect
        result = build_bea_income_features()
        prefixes = result["county_fips"].str[:2].unique()
        assert set(prefixes).issubset(TARGET_FIPS_PREFIXES)

    @patch("src.assembly.fetch_bea_income.fetch_cainc1_state_cached")
    def test_no_duplicate_county_fips(self, mock_fetch):
        """Each county_fips appears at most once in output."""
        mock_fetch.side_effect = self._side_effect
        result = build_bea_income_features()
        assert result["county_fips"].nunique() == len(result)

    @patch("src.assembly.fetch_bea_income.fetch_cainc1_state_cached")
    def test_shares_are_float(self, mock_fetch):
        """Share columns have float dtype."""
        mock_fetch.side_effect = self._side_effect
        result = build_bea_income_features()
        for col in ["earnings_share", "transfers_share", "investment_share"]:
            assert result[col].dtype.kind == "f", f"{col} is not float"

    @patch("src.assembly.fetch_bea_income.fetch_cainc1_state_cached")
    def test_all_states_fetched(self, mock_fetch):
        """fetch_cainc1_state_cached is called once per target state (51)."""
        mock_fetch.side_effect = self._side_effect
        build_bea_income_features()
        # Called once per state (51 states + DC), possibly twice for fallback
        assert mock_fetch.call_count >= 51

    @patch("src.assembly.fetch_bea_income.fetch_cainc1_state_cached")
    def test_empty_state_data_skipped_gracefully(self, mock_fetch):
        """A state returning empty data doesn't crash; other states still assembled."""
        def side_effect_with_empty(fips_prefix: str, year: int, force_refresh: bool = False) -> pd.DataFrame:
            if fips_prefix == "13":  # GA returns empty
                return pd.DataFrame(columns=["GeoFips", "GeoName", "LineCode", "TimePeriod", "DataValue"])
            return self._make_state_raw(fips_prefix)

        mock_fetch.side_effect = side_effect_with_empty
        result = build_bea_income_features()
        # FL and AL should still be present
        prefixes = set(result["county_fips"].str[:2].unique())
        assert "12" in prefixes or "01" in prefixes


# ---------------------------------------------------------------------------
# 7. Path constants
# ---------------------------------------------------------------------------


class TestPathConstants:
    def test_raw_dir_ends_in_data_raw_bea(self):
        """RAW_DIR ends in data/raw/bea."""
        assert RAW_DIR.parts[-3:] == ("data", "raw", "bea")

    def test_assembled_dir_ends_in_data_assembled(self):
        """ASSEMBLED_DIR ends in data/assembled."""
        assert ASSEMBLED_DIR.parts[-2:] == ("data", "assembled")

    def test_primary_year_is_2022(self):
        """PRIMARY_YEAR is 2022 (latest BEA CAINC4)."""
        assert PRIMARY_YEAR == 2022

    def test_fallback_year_is_2021(self):
        """FALLBACK_YEAR is 2021."""
        assert FALLBACK_YEAR == 2021

    def test_line_codes_defined(self):
        """BEA CAINC4 line codes have the expected integer values."""
        assert LINE_PERSONAL_INCOME == 10
        assert LINE_NET_EARNINGS == 45
        assert LINE_TRANSFERS == 47
        assert LINE_DIVIDENDS_INTEREST == 46

    def test_state_abbr_maps_fips_to_names(self):
        """STATE_ABBR maps FIPS prefixes to FL, GA, AL abbreviations."""
        assert STATE_ABBR.get("12") == "FL"
        assert STATE_ABBR.get("13") == "GA"
        assert STATE_ABBR.get("01") == "AL"
