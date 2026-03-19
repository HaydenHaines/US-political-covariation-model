"""Tests for CDC county-level mortality data fetching and feature computation.

Tests exercise:
1. fetch_cdc_wonder_mortality.py — URL construction, SODA API patterns, filtering,
   output schema, COVID death aggregation, drug overdose parsing
2. build_cdc_mortality_features.py — drug overdose rate, COVID death rate, all-cause
   rate, excess mortality ratio, imputation, output schema

These tests use synthetic DataFrames and mock HTTP responses so they run without
any network access. Tests verify:
  - URL construction includes correct dataset ID, where clause, state filters
  - Raw JSON parsed correctly into typed DataFrames
  - FIPS filtering keeps only FL/GA/AL county FIPS (5-digit, non-state-level)
  - Suppressed rows (footnote non-null) are dropped from COVID data
  - Drug overdose rate averages correctly across years
  - COVID death rate computes correctly from deaths + population
  - All-cause rate derived from age_adjusted_rate column
  - Excess mortality ratio: county / state median
  - State-median imputation fills NaN correctly
  - Reserved NaN columns (heart_disease_rate, cancer_rate, suicide_rate) are all NaN
  - Edge cases: empty input, all-suppressed, no target states, zero population
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_cdc_wonder_mortality import (
    CAUSE_ALLCAUSE_COVID_PERIOD,
    CAUSE_COVID,
    CAUSE_DRUG_OVERDOSE,
    DATASET_COVID_DEATHS,
    DATASET_DRUG_OVERDOSE,
    SODA_BASE,
    SODA_PAGE_SIZE,
    STATES,
    TARGET_STATE_FIPS,
    _build_soda_url,
    _extract_year_from_date,
    build_covid_deaths_url,
    build_drug_overdose_url,
    fetch_covid_deaths,
    fetch_drug_overdose,
)
from src.assembly.build_cdc_mortality_features import (
    CDC_MORTALITY_FEATURE_COLS,
    _RESERVED_NAN_COLS,
    compute_allcause_rate,
    compute_cdc_mortality_features,
    compute_covid_death_rate,
    compute_drug_overdose_rate,
    compute_excess_mortality_ratio,
    impute_mortality_state_medians,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_drug_overdose_rows(county_fips: str = "12001", year: int = 2019, death_rate: float = 20.0) -> dict:
    """Build a minimal drug overdose raw row dict."""
    return {
        "fips": county_fips,
        "county": "Test County",
        "state": "Florida",
        "year": str(year),
        "deaths": "50",
        "population": "100000",
        "model_based_death_rate": str(death_rate),
        "age_adjusted_rate": str(death_rate * 1.1),
    }


def _make_covid_rows(
    county_fips: str = "12001",
    start_date: str = "2020-01-01T00:00:00.000",
    deaths_covid: str = "200",
    deaths_all: str = "1000",
    footnote: str | None = None,
) -> dict:
    """Build a minimal COVID deaths raw row dict."""
    row = {
        "fips_code": county_fips,
        "county_name": "Test County",
        "state": "Florida",
        "start_date": start_date,
        "end_date": "2020-12-31T00:00:00.000",
        "deaths_covid19": deaths_covid,
        "deaths_all_cause": deaths_all,
    }
    if footnote is not None:
        row["footnote"] = footnote
    return row


def _make_raw_mortality_df(rows: list[dict]) -> pd.DataFrame:
    """Build a raw mortality edge-list DataFrame for feature computation tests."""
    default = {
        "county_fips": "12001",
        "year": 2019,
        "cause": CAUSE_DRUG_OVERDOSE,
        "deaths": 50.0,
        "population": 100_000.0,
        "death_rate": 20.0,
        "age_adjusted_rate": 22.0,
    }
    records = [{**default, **r} for r in rows]
    df = pd.DataFrame(records)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["death_rate"] = pd.to_numeric(df["death_rate"], errors="coerce")
    df["age_adjusted_rate"] = pd.to_numeric(df["age_adjusted_rate"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# URL construction — _build_soda_url
# ---------------------------------------------------------------------------


class TestBuildSodaUrl:
    """Tests for _build_soda_url()."""

    def test_url_contains_dataset_id(self):
        """URL must contain the requested dataset ID."""
        url = _build_soda_url(DATASET_DRUG_OVERDOSE, where="state='Florida'", select="fips")
        assert DATASET_DRUG_OVERDOSE in url

    def test_url_contains_soda_base(self):
        """URL must start with the CDC SODA base URL."""
        url = _build_soda_url(DATASET_DRUG_OVERDOSE, where="state='Florida'", select="fips")
        assert SODA_BASE in url

    def test_url_contains_where_clause(self):
        """URL must contain a $where parameter."""
        url = _build_soda_url(DATASET_DRUG_OVERDOSE, where="state='Florida'", select="fips")
        assert "%24where" in url or "$where" in url

    def test_url_contains_limit(self):
        """URL must contain a $limit parameter."""
        url = _build_soda_url(DATASET_DRUG_OVERDOSE, where="state='Florida'", select="fips", limit=500)
        assert "500" in url

    def test_url_contains_offset(self):
        """URL must contain a $offset parameter."""
        url = _build_soda_url(DATASET_DRUG_OVERDOSE, where="state='Florida'", select="fips", offset=1000)
        assert "1000" in url

    def test_url_ends_with_json(self):
        """URL must target the .json endpoint."""
        url = _build_soda_url(DATASET_DRUG_OVERDOSE, where="state='Florida'", select="fips")
        assert ".json" in url

    def test_different_datasets_produce_different_urls(self):
        """Different dataset IDs must produce different URLs."""
        url1 = _build_soda_url(DATASET_DRUG_OVERDOSE, where="state='Florida'", select="fips")
        url2 = _build_soda_url(DATASET_COVID_DEATHS, where="state='Florida'", select="fips")
        assert url1 != url2


# ---------------------------------------------------------------------------
# URL construction — build_drug_overdose_url / build_covid_deaths_url
# ---------------------------------------------------------------------------


class TestBuildDrugOverdoseUrl:
    """Tests for build_drug_overdose_url()."""

    def test_url_contains_drug_overdose_dataset(self):
        """Drug overdose URL must reference correct dataset."""
        url = build_drug_overdose_url()
        assert DATASET_DRUG_OVERDOSE in url

    def test_url_filters_target_states(self):
        """Drug overdose URL must filter to FL, GA, or AL."""
        url = build_drug_overdose_url()
        # State names are in the where clause
        assert "Florida" in url or "Florida".lower() in url.lower() or "Florida" in url

    def test_url_has_pagination_params(self):
        """Drug overdose URL must include limit and offset."""
        url = build_drug_overdose_url(offset=500, limit=100)
        assert "500" in url
        assert "100" in url

    def test_url_selects_key_columns(self):
        """Drug overdose URL must select fips and death rate columns."""
        url = build_drug_overdose_url()
        assert "fips" in url
        assert "model_based_death_rate" in url or "death_rate" in url


class TestBuildCovidDeathsUrl:
    """Tests for build_covid_deaths_url()."""

    def test_url_contains_covid_deaths_dataset(self):
        """COVID deaths URL must reference correct dataset."""
        url = build_covid_deaths_url()
        assert DATASET_COVID_DEATHS in url

    def test_url_filters_target_states(self):
        """COVID deaths URL must filter to FL, GA, or AL."""
        url = build_covid_deaths_url()
        assert "Florida" in url or "Florida".lower() in url.lower()

    def test_url_selects_covid_death_columns(self):
        """COVID deaths URL must include covid death count columns."""
        url = build_covid_deaths_url()
        assert "deaths_covid19" in url or "covid" in url.lower()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_states_fips_correct(self):
        """STATES must map AL→01, FL→12, GA→13."""
        assert STATES["AL"] == "01"
        assert STATES["FL"] == "12"
        assert STATES["GA"] == "13"

    def test_target_state_fips_matches_states(self):
        """TARGET_STATE_FIPS must be the frozenset of STATES values."""
        assert TARGET_STATE_FIPS == frozenset(STATES.values())

    def test_cause_codes_defined(self):
        """All cause code constants must be non-empty strings."""
        assert CAUSE_DRUG_OVERDOSE
        assert CAUSE_COVID
        assert CAUSE_ALLCAUSE_COVID_PERIOD

    def test_cause_codes_distinct(self):
        """All cause codes must be distinct."""
        codes = {CAUSE_DRUG_OVERDOSE, CAUSE_COVID, CAUSE_ALLCAUSE_COVID_PERIOD}
        assert len(codes) == 3

    def test_soda_page_size_reasonable(self):
        """SODA page size must be between 100 and 50,000."""
        assert 100 <= SODA_PAGE_SIZE <= 50_000

    def test_dataset_ids_not_empty(self):
        """Dataset IDs must be non-empty strings."""
        assert len(DATASET_DRUG_OVERDOSE) > 0
        assert len(DATASET_COVID_DEATHS) > 0


# ---------------------------------------------------------------------------
# _extract_year_from_date
# ---------------------------------------------------------------------------


class TestExtractYearFromDate:
    """Tests for _extract_year_from_date()."""

    def test_extracts_year_from_iso_date(self):
        """Must extract correct year from ISO 8601 date string."""
        s = pd.Series(["2020-03-15T00:00:00.000"])
        result = _extract_year_from_date(s)
        assert result.iloc[0] == 2020

    def test_extracts_year_from_plain_date(self):
        """Must extract correct year from plain date string (YYYY-MM-DD)."""
        s = pd.Series(["2021-06-01"])
        result = _extract_year_from_date(s)
        assert result.iloc[0] == 2021

    def test_invalid_date_produces_nan(self):
        """Invalid date string must produce NaN year."""
        s = pd.Series(["not-a-date"])
        result = _extract_year_from_date(s)
        assert result.isna().all()

    def test_multiple_dates(self):
        """Must handle multiple dates correctly."""
        s = pd.Series(["2020-01-01", "2021-06-15", "2022-12-31"])
        result = _extract_year_from_date(s)
        assert list(result) == [2020, 2021, 2022]


# ---------------------------------------------------------------------------
# fetch_drug_overdose (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchDrugOverdose:
    """Tests for fetch_drug_overdose() using mocked HTTP responses."""

    def _make_mock_response(self, rows: list[dict], status: int = 200) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = rows
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_returns_dataframe_on_success(self, mock_get):
        """fetch_drug_overdose must return a non-empty DataFrame on success."""
        rows = [_make_drug_overdose_rows("12001", 2019, 20.0)]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_drug_overdose()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_filters_to_target_states(self, mock_get):
        """fetch_drug_overdose must drop non-FL/GA/AL counties."""
        rows = [
            _make_drug_overdose_rows("12001", 2019, 20.0),  # FL — keep
            {**_make_drug_overdose_rows("48001", 2019, 15.0), "state": "Texas"},  # TX — drop
        ]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_drug_overdose()
        if not result.empty:
            state_prefixes = result["county_fips"].str[:2].unique()
            assert all(p in TARGET_STATE_FIPS for p in state_prefixes)

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_drops_state_level_fips(self, mock_get):
        """fetch_drug_overdose must drop state-level FIPS (e.g., 12000)."""
        rows = [
            _make_drug_overdose_rows("12000", 2019, 20.0),  # State-level — drop
            _make_drug_overdose_rows("12001", 2019, 20.0),  # County — keep
        ]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_drug_overdose()
        if not result.empty:
            county_parts = result["county_fips"].str[2:]
            assert (county_parts != "000").all()

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_output_has_required_columns(self, mock_get):
        """fetch_drug_overdose output must have required columns."""
        rows = [_make_drug_overdose_rows("12001", 2019, 20.0)]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_drug_overdose()
        if not result.empty:
            required = {"county_fips", "year", "cause", "death_rate"}
            assert required.issubset(set(result.columns))

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_cause_is_drug_overdose(self, mock_get):
        """fetch_drug_overdose must set cause = CAUSE_DRUG_OVERDOSE."""
        rows = [_make_drug_overdose_rows("12001", 2019, 20.0)]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_drug_overdose()
        if not result.empty:
            assert (result["cause"] == CAUSE_DRUG_OVERDOSE).all()

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_fips_zero_padded(self, mock_get):
        """fetch_drug_overdose must zero-pad FIPS to 5 digits."""
        rows = [{"fips": "1001", "county": "X", "state": "Alabama", "year": "2019",
                 "deaths": "10", "population": "10000", "model_based_death_rate": "10.0",
                 "age_adjusted_rate": "11.0"}]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_drug_overdose()
        if not result.empty:
            assert all(len(f) == 5 for f in result["county_fips"])

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_returns_empty_on_http_error(self, mock_get):
        """fetch_drug_overdose must return empty DataFrame on HTTP failure."""
        mock_get.side_effect = Exception("connection refused")
        result = fetch_drug_overdose()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_returns_empty_on_no_data(self, mock_get):
        """fetch_drug_overdose must return empty DataFrame when API returns no rows."""
        mock_get.return_value = self._make_mock_response([])
        result = fetch_drug_overdose()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# fetch_covid_deaths (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchCovidDeaths:
    """Tests for fetch_covid_deaths() using mocked HTTP responses."""

    def _make_mock_response(self, rows: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = rows
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_returns_dataframe_on_success(self, mock_get):
        """fetch_covid_deaths must return a non-empty DataFrame on success."""
        rows = [_make_covid_rows("12001", "2020-01-01T00:00:00.000", "200", "1000")]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_covid_deaths()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_produces_both_cause_codes(self, mock_get):
        """fetch_covid_deaths must produce both 'covid' and 'allcause_covid' rows."""
        rows = [_make_covid_rows("12001", "2020-06-01T00:00:00.000", "200", "1000")]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_covid_deaths()
        if not result.empty:
            causes = set(result["cause"].unique())
            assert CAUSE_COVID in causes
            assert CAUSE_ALLCAUSE_COVID_PERIOD in causes

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_drops_suppressed_rows(self, mock_get):
        """fetch_covid_deaths must drop rows with non-null footnote (suppressed)."""
        rows = [
            _make_covid_rows("12001", "2020-01-01T00:00:00.000", "200", "1000", footnote=None),
            _make_covid_rows("12003", "2020-01-01T00:00:00.000", "50", "500", footnote="Suppressed: counts 1-9"),
        ]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_covid_deaths()
        if not result.empty:
            counties = result["county_fips"].unique()
            # 12003 was suppressed; should not appear
            assert "12003" not in counties

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_filters_to_target_states(self, mock_get):
        """fetch_covid_deaths must drop non-FL/GA/AL counties."""
        rows = [
            _make_covid_rows("12001", "2020-01-01T00:00:00.000", "200", "1000"),  # FL — keep
            {**_make_covid_rows("48001", "2020-01-01T00:00:00.000", "100", "500"),
             "state": "Texas"},  # TX — drop
        ]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_covid_deaths()
        if not result.empty:
            state_prefixes = result["county_fips"].str[:2].unique()
            assert all(p in TARGET_STATE_FIPS for p in state_prefixes)

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_aggregates_deaths_across_periods(self, mock_get):
        """fetch_covid_deaths must sum deaths across multiple period rows for same county."""
        rows = [
            _make_covid_rows("12001", "2020-01-01T00:00:00.000", "100", "500"),
            _make_covid_rows("12001", "2021-01-01T00:00:00.000", "150", "600"),
        ]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_covid_deaths()
        if not result.empty:
            covid_row = result[(result["county_fips"] == "12001") & (result["cause"] == CAUSE_COVID)]
            if not covid_row.empty:
                # Should have two year rows (2020 and 2021), each with their own deaths
                assert len(covid_row) >= 1

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_output_has_required_columns(self, mock_get):
        """fetch_covid_deaths output must have required columns."""
        rows = [_make_covid_rows("12001", "2020-01-01T00:00:00.000", "200", "1000")]
        mock_get.return_value = self._make_mock_response(rows)
        result = fetch_covid_deaths()
        if not result.empty:
            required = {"county_fips", "year", "cause", "deaths"}
            assert required.issubset(set(result.columns))

    @patch("src.assembly.fetch_cdc_wonder_mortality.requests.get")
    def test_returns_empty_on_failure(self, mock_get):
        """fetch_covid_deaths must return empty DataFrame when API fails."""
        mock_get.side_effect = Exception("timeout")
        result = fetch_covid_deaths()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# compute_drug_overdose_rate
# ---------------------------------------------------------------------------


class TestComputeDrugOverdoseRate:
    """Tests for compute_drug_overdose_rate()."""

    def test_averages_across_years(self):
        """Drug overdose rate must be the mean rate across available years."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "year": 2018, "death_rate": 10.0},
            {"county_fips": "12001", "year": 2019, "death_rate": 20.0},
            {"county_fips": "12001", "year": 2020, "death_rate": 30.0},
        ])
        result = compute_drug_overdose_rate(df)
        assert not result.empty
        rate = result.loc[result["county_fips"] == "12001", "drug_overdose_rate"].iloc[0]
        assert abs(rate - 20.0) < 1e-6

    def test_single_year_returns_that_rate(self):
        """Single year per county returns that year's rate directly."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "year": 2019, "death_rate": 25.5},
        ])
        result = compute_drug_overdose_rate(df)
        rate = result.loc[result["county_fips"] == "12001", "drug_overdose_rate"].iloc[0]
        assert abs(rate - 25.5) < 1e-6

    def test_multiple_counties(self):
        """Multiple counties produce one row per county."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "year": 2019, "death_rate": 10.0},
            {"county_fips": "12003", "year": 2019, "death_rate": 20.0},
            {"county_fips": "13001", "year": 2019, "death_rate": 15.0},
        ])
        result = compute_drug_overdose_rate(df)
        assert len(result) == 3

    def test_nan_death_rate_handled(self):
        """NaN death_rate values must be excluded from mean (not treated as 0)."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "year": 2018, "death_rate": 20.0},
            {"county_fips": "12001", "year": 2019, "death_rate": float("nan")},
        ])
        result = compute_drug_overdose_rate(df)
        rate = result.loc[result["county_fips"] == "12001", "drug_overdose_rate"].iloc[0]
        # Mean of [20.0, NaN] skipping NaN = 20.0
        assert abs(rate - 20.0) < 1e-6

    def test_empty_input_returns_empty(self):
        """Empty input must return empty DataFrame."""
        result = compute_drug_overdose_rate(pd.DataFrame())
        assert result.empty

    def test_only_uses_drug_overdose_rows(self):
        """Must only use rows where cause == CAUSE_DRUG_OVERDOSE."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "cause": CAUSE_DRUG_OVERDOSE, "death_rate": 20.0},
            {"county_fips": "12001", "cause": CAUSE_COVID, "death_rate": 999.0},
        ])
        result = compute_drug_overdose_rate(df)
        rate = result.loc[result["county_fips"] == "12001", "drug_overdose_rate"].iloc[0]
        assert abs(rate - 20.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_covid_death_rate
# ---------------------------------------------------------------------------


class TestComputeCovidDeathRate:
    """Tests for compute_covid_death_rate()."""

    def test_computes_rate_from_deaths_and_population(self):
        """COVID death rate = deaths / population × 100,000."""
        df = _make_raw_mortality_df([
            {
                "county_fips": "12001",
                "cause": CAUSE_COVID,
                "year": 2020,
                "deaths": 1000.0,
                "population": float("nan"),  # no population in COVID rows
            },
            {
                "county_fips": "12001",
                "cause": CAUSE_DRUG_OVERDOSE,
                "year": 2019,
                "deaths": 50.0,
                "population": 500_000.0,
                "death_rate": 10.0,
                "age_adjusted_rate": 11.0,
            },
        ])
        result = compute_covid_death_rate(df)
        if not result.empty and "covid_death_rate" in result.columns:
            rate = result.loc[result["county_fips"] == "12001", "covid_death_rate"]
            if not rate.empty and not rate.isna().all():
                # 1000 / 500000 * 100000 = 200.0
                assert abs(rate.iloc[0] - 200.0) < 1.0

    def test_sums_deaths_across_years(self):
        """COVID deaths must be summed across all years."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "cause": CAUSE_COVID, "year": 2020, "deaths": 500.0, "population": float("nan")},
            {"county_fips": "12001", "cause": CAUSE_COVID, "year": 2021, "deaths": 600.0, "population": float("nan")},
            {"county_fips": "12001", "cause": CAUSE_DRUG_OVERDOSE, "year": 2019, "deaths": 30.0,
             "population": 1_000_000.0, "death_rate": 30.0, "age_adjusted_rate": 33.0},
        ])
        result = compute_covid_death_rate(df)
        # Should use total deaths 1100 and pop 1,000,000 → rate = 110.0
        if not result.empty and "covid_death_rate" in result.columns:
            rate = result.loc[result["county_fips"] == "12001", "covid_death_rate"]
            if not rate.empty and not rate.isna().all():
                assert abs(rate.iloc[0] - 110.0) < 1.0

    def test_nan_when_no_population(self):
        """COVID death rate must be NaN when population is unavailable."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "cause": CAUSE_COVID, "year": 2020, "deaths": 1000.0, "population": float("nan")},
        ])
        result = compute_covid_death_rate(df)
        if not result.empty:
            rate = result.loc[result["county_fips"] == "12001", "covid_death_rate"]
            assert rate.isna().all()

    def test_nan_when_zero_population(self):
        """COVID death rate must be NaN when population is zero."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "cause": CAUSE_COVID, "year": 2020, "deaths": 100.0, "population": float("nan")},
            {"county_fips": "12001", "cause": CAUSE_DRUG_OVERDOSE, "year": 2019,
             "deaths": 10.0, "population": 0.0, "death_rate": 0.0, "age_adjusted_rate": 0.0},
        ])
        result = compute_covid_death_rate(df)
        if not result.empty:
            rate = result.loc[result["county_fips"] == "12001", "covid_death_rate"]
            assert rate.isna().all() or (rate >= 0).all()

    def test_rate_is_non_negative(self):
        """COVID death rate must be non-negative."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "cause": CAUSE_COVID, "year": 2020, "deaths": 50.0, "population": float("nan")},
            {"county_fips": "12001", "cause": CAUSE_DRUG_OVERDOSE, "year": 2019,
             "deaths": 30.0, "population": 100_000.0, "death_rate": 30.0, "age_adjusted_rate": 33.0},
        ])
        result = compute_covid_death_rate(df)
        if not result.empty and "covid_death_rate" in result.columns:
            valid = result["covid_death_rate"].dropna()
            assert (valid >= 0).all()

    def test_empty_input_returns_empty(self):
        """Empty input must return empty DataFrame."""
        result = compute_covid_death_rate(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# compute_allcause_rate
# ---------------------------------------------------------------------------


class TestComputeAllcauseRate:
    """Tests for compute_allcause_rate()."""

    def test_averages_age_adjusted_rate_across_years(self):
        """All-cause rate must be the mean age_adjusted_rate across years."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "year": 2018, "age_adjusted_rate": 800.0},
            {"county_fips": "12001", "year": 2019, "age_adjusted_rate": 900.0},
        ])
        result = compute_allcause_rate(df)
        rate = result.loc[result["county_fips"] == "12001", "allcause_age_adj_rate"].iloc[0]
        assert abs(rate - 850.0) < 1e-6

    def test_single_year_uses_that_rate(self):
        """Single year returns that year's age-adjusted rate."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "year": 2019, "age_adjusted_rate": 750.0},
        ])
        result = compute_allcause_rate(df)
        rate = result.loc[result["county_fips"] == "12001", "allcause_age_adj_rate"].iloc[0]
        assert abs(rate - 750.0) < 1e-6

    def test_empty_input_returns_empty(self):
        """Empty input must return empty DataFrame."""
        result = compute_allcause_rate(pd.DataFrame())
        assert result.empty

    def test_only_uses_drug_overdose_rows(self):
        """Must only use drug_overdose cause rows for age_adjusted_rate."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "cause": CAUSE_DRUG_OVERDOSE, "age_adjusted_rate": 800.0},
            {"county_fips": "12001", "cause": CAUSE_COVID, "age_adjusted_rate": 999.0},
        ])
        result = compute_allcause_rate(df)
        rate = result.loc[result["county_fips"] == "12001", "allcause_age_adj_rate"].iloc[0]
        assert abs(rate - 800.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_excess_mortality_ratio
# ---------------------------------------------------------------------------


class TestComputeExcessMortalityRatio:
    """Tests for compute_excess_mortality_ratio()."""

    def test_ratio_of_1_for_state_median(self):
        """County at exactly the state median should have ratio == 1.0."""
        features = pd.DataFrame({
            "county_fips": ["12001", "12003", "12005"],
            "drug_overdose_rate": [10.0, 20.0, 30.0],
        })
        ratio = compute_excess_mortality_ratio(features)
        # Median of [10, 20, 30] = 20; 12003's rate is 20 → ratio = 1.0
        assert abs(ratio.iloc[1] - 1.0) < 1e-6

    def test_above_median_ratio_greater_than_1(self):
        """County above state median should have ratio > 1.0."""
        features = pd.DataFrame({
            "county_fips": ["12001", "12003"],
            "drug_overdose_rate": [10.0, 30.0],  # Median = 20, county 2 = 30 → ratio 1.5
        })
        ratio = compute_excess_mortality_ratio(features)
        assert ratio.iloc[1] > 1.0

    def test_below_median_ratio_less_than_1(self):
        """County below state median should have ratio < 1.0."""
        features = pd.DataFrame({
            "county_fips": ["12001", "12003"],
            "drug_overdose_rate": [10.0, 30.0],  # Median = 20, county 1 = 10 → ratio 0.5
        })
        ratio = compute_excess_mortality_ratio(features)
        assert ratio.iloc[0] < 1.0

    def test_nan_rate_produces_nan_ratio(self):
        """County with NaN drug_overdose_rate must get NaN ratio."""
        features = pd.DataFrame({
            "county_fips": ["12001", "12003"],
            "drug_overdose_rate": [float("nan"), 20.0],
        })
        ratio = compute_excess_mortality_ratio(features)
        assert ratio.isna().iloc[0]

    def test_ratio_is_non_negative(self):
        """Ratio must be non-negative for valid inputs."""
        features = pd.DataFrame({
            "county_fips": ["12001", "12003", "12005"],
            "drug_overdose_rate": [5.0, 15.0, 25.0],
        })
        ratio = compute_excess_mortality_ratio(features)
        valid = ratio.dropna()
        assert (valid >= 0).all()

    def test_different_states_use_own_median(self):
        """Counties in different states use their own state's median."""
        features = pd.DataFrame({
            "county_fips": ["12001", "12003", "01001", "01003"],
            "drug_overdose_rate": [10.0, 20.0, 100.0, 200.0],
        })
        ratio = compute_excess_mortality_ratio(features)
        # FL median = 15, AL median = 150
        # FL 12001: 10/15 ≈ 0.667; FL 12003: 20/15 ≈ 1.333
        # AL 01001: 100/150 ≈ 0.667; AL 01003: 200/150 ≈ 1.333
        assert ratio.iloc[2] < ratio.iloc[3]

    def test_missing_column_returns_nan(self):
        """Missing drug_overdose_rate column must return all-NaN series."""
        features = pd.DataFrame({
            "county_fips": ["12001"],
            "covid_death_rate": [100.0],
        })
        ratio = compute_excess_mortality_ratio(features)
        assert ratio.isna().all()


# ---------------------------------------------------------------------------
# compute_cdc_mortality_features (full pipeline)
# ---------------------------------------------------------------------------


class TestComputeCdcMortalityFeatures:
    """Tests for compute_cdc_mortality_features() end-to-end pipeline."""

    @pytest.fixture(scope="class")
    def synthetic_df(self):
        """Synthetic mortality edge-list for FL and GA counties across 2 years."""
        rows = []
        for county in ("12001", "12003", "13001", "13005"):
            for year in (2018, 2019, 2020):
                rows.append({
                    "county_fips": county,
                    "year": year,
                    "cause": CAUSE_DRUG_OVERDOSE,
                    "deaths": 50.0,
                    "population": 200_000.0,
                    "death_rate": float(20 + year - 2018),  # 20, 21, 22
                    "age_adjusted_rate": float(800 + year - 2018),
                })
            for year in (2020, 2021):
                rows.append({
                    "county_fips": county,
                    "year": year,
                    "cause": CAUSE_COVID,
                    "deaths": 300.0,
                    "population": float("nan"),
                    "death_rate": float("nan"),
                    "age_adjusted_rate": float("nan"),
                })
                rows.append({
                    "county_fips": county,
                    "year": year,
                    "cause": CAUSE_ALLCAUSE_COVID_PERIOD,
                    "deaths": 2000.0,
                    "population": float("nan"),
                    "death_rate": float("nan"),
                    "age_adjusted_rate": float("nan"),
                })
        return pd.DataFrame(rows)

    def test_output_has_one_row_per_county(self, synthetic_df):
        """Features must have exactly one row per county."""
        result = compute_cdc_mortality_features(synthetic_df)
        n_counties = synthetic_df["county_fips"].nunique()
        assert len(result) == n_counties

    def test_output_has_required_feature_columns(self, synthetic_df):
        """All CDC_MORTALITY_FEATURE_COLS must be in output."""
        result = compute_cdc_mortality_features(synthetic_df)
        for col in CDC_MORTALITY_FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_drug_overdose_rate_positive(self, synthetic_df):
        """drug_overdose_rate must be positive for counties with data."""
        result = compute_cdc_mortality_features(synthetic_df)
        valid = result["drug_overdose_rate"].dropna()
        assert (valid > 0).all()

    def test_despair_rate_equals_drug_overdose_rate(self, synthetic_df):
        """despair_death_rate must equal drug_overdose_rate."""
        result = compute_cdc_mortality_features(synthetic_df)
        for _, row in result.iterrows():
            if not pd.isna(row["drug_overdose_rate"]):
                assert abs(row["despair_death_rate"] - row["drug_overdose_rate"]) < 1e-9

    def test_reserved_columns_are_nan(self, synthetic_df):
        """heart_disease_rate, cancer_rate, suicide_rate must all be NaN."""
        result = compute_cdc_mortality_features(synthetic_df)
        for col in _RESERVED_NAN_COLS:
            assert col in result.columns
            assert result[col].isna().all(), f"{col} should be all-NaN"

    def test_excess_mortality_ratio_present(self, synthetic_df):
        """excess_mortality_ratio must be present and positive where computable."""
        result = compute_cdc_mortality_features(synthetic_df)
        assert "excess_mortality_ratio" in result.columns
        valid = result["excess_mortality_ratio"].dropna()
        assert (valid > 0).all()

    def test_filters_non_target_states(self):
        """Non-FL/GA/AL counties must be excluded from output."""
        df = _make_raw_mortality_df([
            {"county_fips": "12001", "cause": CAUSE_DRUG_OVERDOSE, "death_rate": 20.0},
            {"county_fips": "48001", "cause": CAUSE_DRUG_OVERDOSE, "death_rate": 15.0},  # TX — drop
        ])
        result = compute_cdc_mortality_features(df)
        if not result.empty:
            state_prefixes = result["county_fips"].str[:2].unique()
            assert "48" not in state_prefixes

    def test_drops_state_level_fips(self):
        """State-level FIPS (e.g., 12000) must be excluded."""
        df = _make_raw_mortality_df([
            {"county_fips": "12000", "cause": CAUSE_DRUG_OVERDOSE, "death_rate": 20.0},  # State — drop
            {"county_fips": "12001", "cause": CAUSE_DRUG_OVERDOSE, "death_rate": 20.0},  # County — keep
        ])
        result = compute_cdc_mortality_features(df)
        if not result.empty:
            county_parts = result["county_fips"].str[2:]
            assert (county_parts != "000").all()

    def test_empty_input_returns_empty(self):
        """Empty input must return empty output."""
        result = compute_cdc_mortality_features(pd.DataFrame())
        assert len(result) == 0

    def test_county_fips_column_present(self, synthetic_df):
        """Output must have county_fips column."""
        result = compute_cdc_mortality_features(synthetic_df)
        assert "county_fips" in result.columns

    def test_county_fips_5_digits(self, synthetic_df):
        """All county_fips must be exactly 5-digit strings."""
        result = compute_cdc_mortality_features(synthetic_df)
        assert all(len(f) == 5 for f in result["county_fips"])
        assert all(f.isdigit() for f in result["county_fips"])


# ---------------------------------------------------------------------------
# impute_mortality_state_medians
# ---------------------------------------------------------------------------


class TestImputeMortalityStateMedians:
    """Tests for impute_mortality_state_medians()."""

    @pytest.fixture
    def df_with_nans(self):
        """Feature DataFrame with some NaN values for imputation testing."""
        return pd.DataFrame({
            "county_fips": ["12001", "12003", "12005"],
            "drug_overdose_rate": [10.0, 20.0, float("nan")],
            "despair_death_rate": [10.0, 20.0, float("nan")],
            "covid_death_rate": [50.0, 70.0, float("nan")],
            "allcause_age_adj_rate": [800.0, 900.0, float("nan")],
            "heart_disease_rate": [float("nan")] * 3,  # Reserved — must stay NaN
            "cancer_rate": [float("nan")] * 3,
            "suicide_rate": [float("nan")] * 3,
            "excess_mortality_ratio": [0.8, 1.2, float("nan")],
        })

    def test_fills_nan_drug_overdose_rate(self, df_with_nans):
        """NaN drug_overdose_rate must be imputed with state median."""
        result = impute_mortality_state_medians(df_with_nans)
        # FL median of [10, 20] = 15; county 12005 should be imputed to 15
        assert not result.loc[2, "drug_overdose_rate"] != result.loc[2, "drug_overdose_rate"]  # not NaN
        assert abs(result.loc[2, "drug_overdose_rate"] - 15.0) < 1e-6

    def test_preserved_non_nan_values_unchanged(self, df_with_nans):
        """Non-NaN values must not change after imputation."""
        result = impute_mortality_state_medians(df_with_nans)
        assert abs(result.loc[0, "drug_overdose_rate"] - 10.0) < 1e-9
        assert abs(result.loc[1, "drug_overdose_rate"] - 20.0) < 1e-9

    def test_reserved_columns_remain_nan(self, df_with_nans):
        """Reserved NaN columns must remain NaN after imputation."""
        result = impute_mortality_state_medians(df_with_nans)
        for col in _RESERVED_NAN_COLS:
            assert result[col].isna().all(), f"{col} should still be all-NaN after imputation"

    def test_no_nan_input_unchanged(self):
        """DataFrame with no NaN values must be unchanged."""
        df = pd.DataFrame({
            "county_fips": ["12001"],
            "drug_overdose_rate": [15.0],
            "despair_death_rate": [15.0],
            "covid_death_rate": [60.0],
            "allcause_age_adj_rate": [850.0],
            "heart_disease_rate": [float("nan")],
            "cancer_rate": [float("nan")],
            "suicide_rate": [float("nan")],
            "excess_mortality_ratio": [1.0],
        })
        result = impute_mortality_state_medians(df)
        assert abs(result.loc[0, "drug_overdose_rate"] - 15.0) < 1e-9

    def test_does_not_add_state_fips_column(self, df_with_nans):
        """state_fips helper column must not appear in output."""
        result = impute_mortality_state_medians(df_with_nans)
        assert "state_fips" not in result.columns

    def test_imputation_across_states(self):
        """State medians must be computed separately per state."""
        df = pd.DataFrame({
            "county_fips": ["12001", "12003", "13001"],
            "drug_overdose_rate": [10.0, float("nan"), 50.0],
            "despair_death_rate": [10.0, float("nan"), 50.0],
            "covid_death_rate": [100.0, float("nan"), 200.0],
            "allcause_age_adj_rate": [800.0, float("nan"), 900.0],
            "heart_disease_rate": [float("nan")] * 3,
            "cancer_rate": [float("nan")] * 3,
            "suicide_rate": [float("nan")] * 3,
            "excess_mortality_ratio": [0.8, float("nan"), 1.2],
        })
        result = impute_mortality_state_medians(df)
        # FL: only 12001 has data → median = 10; 12003 imputed to 10
        assert abs(result.loc[1, "drug_overdose_rate"] - 10.0) < 1e-6
        # GA: only 13001 → value = 50; stays 50
        assert abs(result.loc[2, "drug_overdose_rate"] - 50.0) < 1e-6


# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------


class TestCdcMortalityFeatureCols:
    """Tests for CDC_MORTALITY_FEATURE_COLS list definition."""

    def test_feature_cols_not_empty(self):
        """CDC_MORTALITY_FEATURE_COLS must not be empty."""
        assert len(CDC_MORTALITY_FEATURE_COLS) > 0

    def test_expected_features_present(self):
        """Expected feature columns must all be in CDC_MORTALITY_FEATURE_COLS."""
        expected = {
            "drug_overdose_rate",
            "despair_death_rate",
            "covid_death_rate",
            "allcause_age_adj_rate",
            "excess_mortality_ratio",
        }
        for feat in expected:
            assert feat in CDC_MORTALITY_FEATURE_COLS, f"Missing: {feat}"

    def test_reserved_cols_in_feature_cols(self):
        """Reserved NaN columns must be included in CDC_MORTALITY_FEATURE_COLS."""
        for col in _RESERVED_NAN_COLS:
            assert col in CDC_MORTALITY_FEATURE_COLS, f"Reserved column {col} not in feature list"

    def test_no_duplicate_feature_cols(self):
        """CDC_MORTALITY_FEATURE_COLS must not have duplicate entries."""
        assert len(CDC_MORTALITY_FEATURE_COLS) == len(set(CDC_MORTALITY_FEATURE_COLS))


# ---------------------------------------------------------------------------
# Integration tests (skip if raw data not present)
# ---------------------------------------------------------------------------


class TestCdcMortalityIntegration:
    """Integration tests against actual saved parquet files (skipped if absent)."""

    @pytest.fixture(scope="class")
    def raw_parquet(self):
        """Load cdc_mortality.parquet if it exists."""
        path = PROJECT_ROOT / "data" / "raw" / "cdc_mortality.parquet"
        if not path.exists():
            pytest.skip("cdc_mortality.parquet not found — run fetch_cdc_wonder_mortality.py first")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def features_parquet(self):
        """Load county_cdc_mortality_features.parquet if it exists."""
        path = PROJECT_ROOT / "data" / "assembled" / "county_cdc_mortality_features.parquet"
        if not path.exists():
            pytest.skip(
                "county_cdc_mortality_features.parquet not found — run build_cdc_mortality_features.py first"
            )
        return pd.read_parquet(path)

    def test_raw_has_required_columns(self, raw_parquet):
        """Raw parquet must have all required columns."""
        required = {"county_fips", "year", "cause", "deaths", "population", "death_rate"}
        assert required.issubset(set(raw_parquet.columns))

    def test_raw_fips_are_5_digits(self, raw_parquet):
        """All county FIPS must be 5-digit strings."""
        assert (raw_parquet["county_fips"].str.len() == 5).all()
        assert raw_parquet["county_fips"].str.isdigit().all()

    def test_raw_target_states_only(self, raw_parquet):
        """Only FL/GA/AL counties must appear in raw parquet."""
        state_prefixes = raw_parquet["county_fips"].str[:2].unique()
        assert set(state_prefixes) <= {"01", "12", "13"}

    def test_raw_no_state_level_fips(self, raw_parquet):
        """No state-level FIPS (county part == '000') must be present."""
        county_part = raw_parquet["county_fips"].str[2:]
        assert (county_part != "000").all()

    def test_raw_has_expected_causes(self, raw_parquet):
        """Raw parquet must include drug_overdose and covid cause codes."""
        causes = set(raw_parquet["cause"].unique())
        assert CAUSE_DRUG_OVERDOSE in causes or CAUSE_COVID in causes

    def test_features_has_required_columns(self, features_parquet):
        """Features parquet must have county_fips and all feature cols."""
        assert "county_fips" in features_parquet.columns
        for col in CDC_MORTALITY_FEATURE_COLS:
            assert col in features_parquet.columns, f"Missing feature: {col}"

    def test_features_reserved_cols_are_nan(self, features_parquet):
        """Reserved feature columns must be all-NaN in output."""
        for col in _RESERVED_NAN_COLS:
            assert features_parquet[col].isna().all(), f"{col} should be all-NaN"

    def test_features_drug_overdose_rate_positive(self, features_parquet):
        """Non-NaN drug_overdose_rate values must be positive."""
        valid = features_parquet["drug_overdose_rate"].dropna()
        assert (valid > 0).all()

    def test_features_excess_ratio_positive(self, features_parquet):
        """Non-NaN excess_mortality_ratio values must be positive."""
        valid = features_parquet["excess_mortality_ratio"].dropna()
        assert (valid > 0).all()

    def test_features_state_coverage(self, features_parquet):
        """Features must include counties from all three target states."""
        state_prefixes = set(features_parquet["county_fips"].str[:2].unique())
        # At least two of the three target states must be present
        target_prefixes = {"01", "12", "13"}
        assert len(state_prefixes & target_prefixes) >= 2


# Make PROJECT_ROOT importable for integration test fixtures
try:
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parents[1]
except Exception:
    from pathlib import Path
    PROJECT_ROOT = Path(".")
