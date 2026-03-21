"""Tests for decennial census fetching and demographic interpolation.

Tests exercise:
1. fetch_census_decennial.py — URL construction, variable crosswalk, response
   parsing, FIPS zero-padding, housing/education derivation, retry logic
2. interpolate_demographics.py — linear interpolation, CPI adjustment,
   pre-2000 flat extrapolation, post-2020 flat extrapolation, derived ratios

All tests use mocked HTTP responses (unittest.mock.patch on requests.get)
so they run without network access or a Census API key.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[1]

EXPECTED_STANDARDIZED_COLS = {
    "county_fips",
    "year",
    "pop_total",
    "pop_white_nh",
    "pop_black",
    "pop_asian",
    "pop_hispanic",
    "median_age",
    "median_hh_income",
    "housing_total",
    "housing_owner",
    "educ_total",
    "educ_bachelors_plus",
    "commute_total",
    "commute_car",
    "commute_transit",
    "commute_wfh",
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_response(data: list[list], status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response returning *data* as JSON."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


def _sf1_2000_row(state: str = "12", county: str = "001") -> list[list]:
    """Minimal Census 2000 SF1 response for one county."""
    headers = ["P001001", "P004005", "P004006", "P004008", "P004002",
               "P013001", "H001001", "H004002", "H004003", "state", "county"]
    row = ["100000", "60000", "20000", "5000", "15000",
           "38.5", "40000", "20000", "5000", state, county]
    return [headers, row]


def _sf3_2000_row(state: str = "12", county: str = "001") -> list[list]:
    """Minimal Census 2000 SF3 response for one county."""
    headers = ["P053001", "P037001", "P037015", "P037016", "P037017", "P037018",
               "P037032", "P037033", "P037034", "P037035",
               "P030001", "P030003", "P030005", "P030016", "state", "county"]
    row = ["45000", "80000", "5000", "2000", "500", "200",
           "6000", "2500", "600", "300",
           "50000", "35000", "3000", "1000", state, county]
    return [headers, row]


def _sf1_2010_row(state: str = "12", county: str = "001") -> list[list]:
    """Minimal Census 2010 SF1 response."""
    headers = ["P001001", "P005003", "P005004", "P005006", "P005010",
               "P013001", "H001001", "H004002", "H004003", "state", "county"]
    row = ["120000", "65000", "25000", "8000", "18000",
           "40.0", "45000", "22000", "6000", state, county]
    return [headers, row]


def _acs5_2010_row(state: str = "12", county: str = "001") -> list[list]:
    """Minimal ACS5 2010 response."""
    headers = ["B19013_001E", "B15002_001E",
               "B15002_015E", "B15002_016E", "B15002_017E", "B15002_018E",
               "B15002_032E", "B15002_033E", "B15002_034E", "B15002_035E",
               "B08301_001E", "B08301_003E", "B08301_010E", "B08301_021E",
               "state", "county"]
    row = ["55000", "90000",
           "6000", "2500", "600", "250",
           "7000", "3000", "700", "300",
           "60000", "40000", "5000", "2000",
           state, county]
    return [headers, row]


def _dhc_2020_row(state: str = "12", county: str = "001") -> list[list]:
    """Minimal Census 2020 DHC response."""
    headers = ["P1_001N", "P5_003N", "P5_004N", "P5_006N", "P5_010N",
               "P13_001N", "H1_001N", "H10_002N", "state", "county"]
    row = ["140000", "70000", "28000", "10000", "22000",
           "42.0", "50000", "30000", state, county]
    return [headers, row]


def _acs5_2020_row(state: str = "12", county: str = "001") -> list[list]:
    """Minimal ACS5 2020 response (same B-table codes as 2010)."""
    headers = ["B19013_001E", "B15002_001E",
               "B15002_015E", "B15002_016E", "B15002_017E", "B15002_018E",
               "B15002_032E", "B15002_033E", "B15002_034E", "B15002_035E",
               "B08301_001E", "B08301_003E", "B08301_010E", "B08301_021E",
               "state", "county"]
    row = ["65000", "100000",
           "8000", "3500", "900", "400",
           "9000", "4000", "1000", "500",
           "70000", "45000", "6000", "5000",
           state, county]
    return [headers, row]


# ── Multi-county helpers ───────────────────────────────────────────────────────


def _multi_county_response(single_row_fn, counties: list[tuple[str, str]]) -> list[list]:
    """Build a multi-county response from a single-row function.

    *counties* is a list of (state_fips, county_fips) tuples.
    """
    first = single_row_fn(counties[0][0], counties[0][1])
    headers = first[0]
    rows = []
    for st, co in counties:
        data = single_row_fn(st, co)
        rows.append(data[1])
    return [headers] + rows


# ===========================================================================
# SECTION 1: fetch_census_decennial tests
# ===========================================================================


class TestURLConstruction:
    """Verify correct Census API URL construction for each year/endpoint."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2000_sf1_url(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.return_value = _mock_response(_sf1_2000_row())
        # Second call for SF3
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        fetch_year(2000, states={"FL": "12"})
        first_call = mock_get.call_args_list[0]
        assert "2000/dec/sf1" in first_call.args[0]

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2000_sf3_url(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        fetch_year(2000, states={"FL": "12"})
        second_call = mock_get.call_args_list[1]
        assert "2000/dec/sf3" in second_call.args[0]

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2010_sf1_url(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2010_row()),
            _mock_response(_acs5_2010_row()),
        ]
        fetch_year(2010, states={"FL": "12"})
        first_call = mock_get.call_args_list[0]
        assert "2010/dec/sf1" in first_call.args[0]

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2010_acs5_url(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2010_row()),
            _mock_response(_acs5_2010_row()),
        ]
        fetch_year(2010, states={"FL": "12"})
        second_call = mock_get.call_args_list[1]
        assert "2010/acs/acs5" in second_call.args[0]

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2020_dhc_url(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_dhc_2020_row()),
            _mock_response(_acs5_2020_row()),
        ]
        fetch_year(2020, states={"FL": "12"})
        first_call = mock_get.call_args_list[0]
        assert "2020/dec/dhc" in first_call.args[0]

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2020_acs5_url(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_dhc_2020_row()),
            _mock_response(_acs5_2020_row()),
        ]
        fetch_year(2020, states={"FL": "12"})
        second_call = mock_get.call_args_list[1]
        assert "2020/acs/acs5" in second_call.args[0]


class TestResponseParsing:
    """Verify that API JSON responses are parsed into correct DataFrames."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2000_returns_dataframe(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        df = fetch_year(2000, states={"FL": "12"})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2010_returns_dataframe(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2010_row()),
            _mock_response(_acs5_2010_row()),
        ]
        df = fetch_year(2010, states={"FL": "12"})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2020_returns_dataframe(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_dhc_2020_row()),
            _mock_response(_acs5_2020_row()),
        ]
        df = fetch_year(2020, states={"FL": "12"})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_multi_state_returns_all_counties(self, mock_get):
        """Fetching 3 states should concatenate results."""
        from src.assembly.fetch_census_decennial import fetch_year
        counties_12 = [("12", "001"), ("12", "003")]
        counties_13 = [("13", "001")]
        counties_01 = [("01", "001")]
        # 6 calls: 2 per state (sf1 + sf3)
        mock_get.side_effect = [
            _mock_response(_multi_county_response(_sf1_2000_row, counties_01)),
            _mock_response(_multi_county_response(_sf3_2000_row, counties_01)),
            _mock_response(_multi_county_response(_sf1_2000_row, counties_12)),
            _mock_response(_multi_county_response(_sf3_2000_row, counties_12)),
            _mock_response(_multi_county_response(_sf1_2000_row, counties_13)),
            _mock_response(_multi_county_response(_sf3_2000_row, counties_13)),
        ]
        df = fetch_year(2000)
        assert len(df) == 4  # 2 + 1 + 1


class TestVariableCrosswalk:
    """Verify all 7 standardized demographic measures are present for each year."""

    CORE_MEASURES = [
        "pop_total", "pop_white_nh", "pop_black", "pop_asian", "pop_hispanic",
        "median_age", "median_hh_income", "housing_total", "housing_owner",
        "educ_total", "educ_bachelors_plus",
        "commute_total", "commute_car", "commute_transit", "commute_wfh",
    ]

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2000_has_all_measures(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        df = fetch_year(2000, states={"FL": "12"})
        for col in self.CORE_MEASURES:
            assert col in df.columns, f"Missing {col} in 2000 data"

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2010_has_all_measures(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2010_row()),
            _mock_response(_acs5_2010_row()),
        ]
        df = fetch_year(2010, states={"FL": "12"})
        for col in self.CORE_MEASURES:
            assert col in df.columns, f"Missing {col} in 2010 data"

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2020_has_all_measures(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_dhc_2020_row()),
            _mock_response(_acs5_2020_row()),
        ]
        df = fetch_year(2020, states={"FL": "12"})
        for col in self.CORE_MEASURES:
            assert col in df.columns, f"Missing {col} in 2020 data"


class TestFIPSZeroPadding:
    """Verify county_fips is a zero-padded 5-character string."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_fips_is_5_digit_string(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row("01", "003")),
            _mock_response(_sf3_2000_row("01", "003")),
        ]
        df = fetch_year(2000, states={"AL": "01"})
        fips = df["county_fips"].iloc[0]
        assert isinstance(fips, str)
        assert len(fips) == 5
        assert fips == "01003"

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_fips_preserves_leading_zeros(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row("01", "001")),
            _mock_response(_sf3_2000_row("01", "001")),
        ]
        df = fetch_year(2000, states={"AL": "01"})
        fips = df["county_fips"].iloc[0]
        assert fips.startswith("0"), "Leading zero lost in FIPS code"


class TestHousingOwnerComputation:
    """Housing owner computation differs across census years."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2000_housing_owner_is_sum(self, mock_get):
        """2000: housing_owner = H004002 (mortgage) + H004003 (free & clear)."""
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        df = fetch_year(2000, states={"FL": "12"})
        assert df["housing_owner"].iloc[0] == 25000  # 20000 + 5000

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2010_housing_owner_is_sum(self, mock_get):
        """2010: housing_owner = H004002 (mortgage) + H004003 (free & clear)."""
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2010_row()),
            _mock_response(_acs5_2010_row()),
        ]
        df = fetch_year(2010, states={"FL": "12"})
        assert df["housing_owner"].iloc[0] == 28000  # 22000 + 6000

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2020_housing_owner_is_direct(self, mock_get):
        """2020: housing_owner = H10_002N (already total owner-occupied)."""
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_dhc_2020_row()),
            _mock_response(_acs5_2020_row()),
        ]
        df = fetch_year(2020, states={"FL": "12"})
        assert df["housing_owner"].iloc[0] == 30000


class TestEducationComputation:
    """Education BA+ is sum of 8 cells (4 male + 4 female degree levels)."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2000_educ_sum(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        df = fetch_year(2000, states={"FL": "12"})
        # male: 5000+2000+500+200=7700; female: 6000+2500+600+300=9400 → 17100
        assert df["educ_bachelors_plus"].iloc[0] == 17100

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2010_educ_sum(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2010_row()),
            _mock_response(_acs5_2010_row()),
        ]
        df = fetch_year(2010, states={"FL": "12"})
        # male: 6000+2500+600+250=9350; female: 7000+3000+700+300=11000 → 20350
        assert df["educ_bachelors_plus"].iloc[0] == 20350

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_2020_educ_sum(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_dhc_2020_row()),
            _mock_response(_acs5_2020_row()),
        ]
        df = fetch_year(2020, states={"FL": "12"})
        # male: 8000+3500+900+400=12800; female: 9000+4000+1000+500=14500 → 27300
        assert df["educ_bachelors_plus"].iloc[0] == 27300


class TestYearColumn:
    """Each output DataFrame should have a 'year' column."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_year_column_present(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        df = fetch_year(2000, states={"FL": "12"})
        assert "year" in df.columns
        assert df["year"].iloc[0] == 2000


class TestOutputColumnConsistency:
    """All three years produce DataFrames with the same column set."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_columns_identical_across_years(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        dfs = {}
        for year, side_effects in [
            (2000, [_mock_response(_sf1_2000_row()), _mock_response(_sf3_2000_row())]),
            (2010, [_mock_response(_sf1_2010_row()), _mock_response(_acs5_2010_row())]),
            (2020, [_mock_response(_dhc_2020_row()), _mock_response(_acs5_2020_row())]),
        ]:
            mock_get.side_effect = side_effects
            dfs[year] = fetch_year(year, states={"FL": "12"})
        assert set(dfs[2000].columns) == set(dfs[2010].columns) == set(dfs[2020].columns)

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_columns_match_expected(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        df = fetch_year(2000, states={"FL": "12"})
        assert set(df.columns) == EXPECTED_STANDARDIZED_COLS


class TestNumericCasting:
    """All measure columns should be numeric after parsing."""

    @patch("src.assembly.fetch_census_decennial.requests.get")
    def test_values_are_numeric(self, mock_get):
        from src.assembly.fetch_census_decennial import fetch_year
        mock_get.side_effect = [
            _mock_response(_sf1_2000_row()),
            _mock_response(_sf3_2000_row()),
        ]
        df = fetch_year(2000, states={"FL": "12"})
        for col in ["pop_total", "median_age", "median_hh_income"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


# ===========================================================================
# SECTION 2: interpolate_demographics tests
# ===========================================================================


@pytest.fixture
def census_frames() -> dict[int, pd.DataFrame]:
    """Synthetic census DataFrames for 2000, 2010, 2020 for one county."""
    base = {
        "county_fips": ["12001"],
        "pop_total": [100000.0],
        "pop_white_nh": [60000.0],
        "pop_black": [20000.0],
        "pop_asian": [5000.0],
        "pop_hispanic": [15000.0],
        "median_age": [38.0],
        "housing_total": [40000.0],
        "housing_owner": [25000.0],
        "educ_total": [80000.0],
        "educ_bachelors_plus": [17000.0],
        "commute_total": [50000.0],
        "commute_car": [35000.0],
        "commute_transit": [3000.0],
        "commute_wfh": [1000.0],
    }
    frames = {}
    for yr, scale in [(2000, 1.0), (2010, 1.2), (2020, 1.5)]:
        d = {k: v if k == "county_fips" else [x * scale for x in v] for k, v in base.items()}
        d["year"] = [yr]
        # Set income in nominal dollars for that year
        if yr == 2000:
            d["median_hh_income"] = [45000.0]  # 1999 dollars
        elif yr == 2010:
            d["median_hh_income"] = [55000.0]  # 2010 dollars
        else:
            d["median_hh_income"] = [65000.0]  # 2020 dollars
        frames[yr] = pd.DataFrame(d)
    return frames


class TestInterpolationMidpoint:
    """Interpolation at the midpoint of two census years."""

    def test_midpoint_2005(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2005)
        row = result.iloc[0]
        # Midpoint between 2000 (100000) and 2010 (120000) = 110000
        assert row["pop_total"] == pytest.approx(110000.0)

    def test_midpoint_2015(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2015)
        row = result.iloc[0]
        # Midpoint between 2010 (120000) and 2020 (150000) = 135000
        assert row["pop_total"] == pytest.approx(135000.0)


class TestInterpolationWeighted:
    """Non-midpoint weighted interpolation."""

    def test_2004_weighted(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2004)
        row = result.iloc[0]
        # weight = 4/10 = 0.4; pop = 0.6*100000 + 0.4*120000 = 108000
        assert row["pop_total"] == pytest.approx(108000.0)

    def test_2016_weighted(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2016)
        row = result.iloc[0]
        # weight = 6/10 = 0.6; pop = 0.4*120000 + 0.6*150000 = 138000
        assert row["pop_total"] == pytest.approx(138000.0)


class TestInterpolationAtCensusYear:
    """Interpolation at exact census year returns census value."""

    def test_at_2000(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        assert result["pop_total"].iloc[0] == pytest.approx(100000.0)

    def test_at_2010(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2010)
        assert result["pop_total"].iloc[0] == pytest.approx(120000.0)

    def test_at_2020(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2020)
        assert result["pop_total"].iloc[0] == pytest.approx(150000.0)


class TestExtrapolationFlat:
    """Pre-2000 uses 2000 values; post-2020 uses 2020 values."""

    def test_pre_2000_uses_2000(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 1998)
        assert result["pop_total"].iloc[0] == pytest.approx(100000.0)

    def test_post_2020_uses_2020(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2024)
        assert result["pop_total"].iloc[0] == pytest.approx(150000.0)


class TestCPIAdjustment:
    """Income is CPI-adjusted to 2020 dollars before interpolation."""

    def test_2000_income_adjusted(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        # 45000 * (258.8 / 166.6) ≈ 69,879.95
        expected = 45000.0 * (258.8 / 166.6)
        assert result["median_hh_income"].iloc[0] == pytest.approx(expected, rel=1e-3)

    def test_2010_income_adjusted(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2010)
        # 55000 * (258.8 / 218.1) ≈ 65,243.47
        expected = 55000.0 * (258.8 / 218.1)
        assert result["median_hh_income"].iloc[0] == pytest.approx(expected, rel=1e-3)

    def test_2020_income_unadjusted(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2020)
        # 2020 income already in 2020 dollars → no adjustment
        assert result["median_hh_income"].iloc[0] == pytest.approx(65000.0, rel=1e-3)

    def test_interpolated_income_uses_adjusted(self, census_frames):
        """2005 income should interpolate between CPI-adjusted 2000 and 2010."""
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2005)
        adj_2000 = 45000.0 * (258.8 / 166.6)
        adj_2010 = 55000.0 * (258.8 / 218.1)
        expected = (adj_2000 + adj_2010) / 2
        assert result["median_hh_income"].iloc[0] == pytest.approx(expected, rel=1e-3)


class TestDerivedRatios:
    """Derived percentage columns are computed correctly."""

    def test_pct_white_nh(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        expected = 60000.0 / 100000.0
        assert result["pct_white_nh"].iloc[0] == pytest.approx(expected)

    def test_pct_bachelors_plus(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        expected = 17000.0 / 80000.0
        assert result["pct_bachelors_plus"].iloc[0] == pytest.approx(expected)

    def test_pct_owner_occupied(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        expected = 25000.0 / 40000.0
        assert result["pct_owner_occupied"].iloc[0] == pytest.approx(expected)

    def test_pct_wfh(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        expected = 1000.0 / 50000.0
        assert result["pct_wfh"].iloc[0] == pytest.approx(expected)

    def test_pct_transit(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        expected = 3000.0 / 50000.0
        assert result["pct_transit"].iloc[0] == pytest.approx(expected)

    def test_pct_car(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        expected = 35000.0 / 50000.0
        assert result["pct_car"].iloc[0] == pytest.approx(expected)

    def test_all_pct_columns_present(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2000)
        for col in ["pct_white_nh", "pct_black", "pct_asian", "pct_hispanic",
                     "pct_bachelors_plus", "pct_owner_occupied", "pct_wfh",
                     "pct_transit", "pct_car"]:
            assert col in result.columns, f"Missing derived column: {col}"


class TestInterpolationYearColumn:
    """Interpolated result has correct year."""

    def test_year_in_output(self, census_frames):
        from src.assembly.interpolate_demographics import interpolate_for_year
        result = interpolate_for_year(census_frames, 2008)
        assert result["year"].iloc[0] == 2008


class TestMultiCountyInterpolation:
    """Interpolation works correctly across multiple counties."""

    def test_two_counties(self):
        from src.assembly.interpolate_demographics import interpolate_for_year
        frames = {}
        for yr in [2000, 2010, 2020]:
            scale = {2000: 1.0, 2010: 1.2, 2020: 1.5}[yr]
            frames[yr] = pd.DataFrame({
                "county_fips": ["12001", "13001"],
                "year": [yr, yr],
                "pop_total": [100000 * scale, 50000 * scale],
                "pop_white_nh": [60000 * scale, 30000 * scale],
                "pop_black": [20000 * scale, 15000 * scale],
                "pop_asian": [5000 * scale, 2000 * scale],
                "pop_hispanic": [15000 * scale, 3000 * scale],
                "median_age": [38 * scale, 35 * scale],
                "median_hh_income": [45000.0, 35000.0] if yr == 2000
                    else [55000.0, 42000.0] if yr == 2010
                    else [65000.0, 50000.0],
                "housing_total": [40000 * scale, 20000 * scale],
                "housing_owner": [25000 * scale, 12000 * scale],
                "educ_total": [80000 * scale, 40000 * scale],
                "educ_bachelors_plus": [17000 * scale, 8000 * scale],
                "commute_total": [50000 * scale, 25000 * scale],
                "commute_car": [35000 * scale, 18000 * scale],
                "commute_transit": [3000 * scale, 1000 * scale],
                "commute_wfh": [1000 * scale, 500 * scale],
            })
        result = interpolate_for_year(frames, 2005)
        assert len(result) == 2
        assert set(result["county_fips"]) == {"12001", "13001"}
