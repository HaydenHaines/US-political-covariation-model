"""Tests for src/assembly/interpolate_demographics.py — census interpolation.

Tests the interpolation logic, CPI adjustment, and derived ratios without
needing real census data or config files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.interpolate_demographics import (
    CENSUS_YEARS,
    DEFAULT_CPI,
    DERIVED_RATIOS,
    INCOME_REF_YEAR,
    INTERPOLATION_COLS,
    _adjust_income_to_2020,
    interpolate_for_year,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_census_frame(county_fips: list[str], year: int, base_pop: float = 1000.0) -> pd.DataFrame:
    """Create a synthetic census DataFrame for testing."""
    n = len(county_fips)
    data = {
        "county_fips": county_fips,
        "pop_total": [base_pop] * n,
        "pop_white_nh": [base_pop * 0.6] * n,
        "pop_black": [base_pop * 0.13] * n,
        "pop_asian": [base_pop * 0.06] * n,
        "pop_hispanic": [base_pop * 0.18] * n,
        "median_age": [38.0] * n,
        "median_hh_income": [50000.0] * n,
        "housing_total": [400] * n,
        "housing_owner": [260] * n,
        "educ_total": [700] * n,
        "educ_bachelors_plus": [210] * n,
        "commute_total": [500] * n,
        "commute_car": [400] * n,
        "commute_transit": [50] * n,
        "commute_wfh": [30] * n,
    }
    return pd.DataFrame(data)


def _make_census_frames(fips: list[str]) -> dict[int, pd.DataFrame]:
    """Build census frames for 2000, 2010, 2020 with increasing population."""
    return {
        2000: _make_census_frame(fips, 2000, base_pop=1000),
        2010: _make_census_frame(fips, 2010, base_pop=1200),
        2020: _make_census_frame(fips, 2020, base_pop=1500),
    }


FIPS = ["12001", "12003", "13001"]
CPI = {1999: 166.6, 2010: 218.1, 2020: 258.8}


# ---------------------------------------------------------------------------
# CPI adjustment
# ---------------------------------------------------------------------------

class TestCpiAdjustment:
    def test_adjusts_income_to_2020_dollars(self):
        frames = _make_census_frames(FIPS)
        adjusted = _adjust_income_to_2020(frames, cpi=CPI)
        # 2020 income should be unchanged
        assert adjusted[2020]["median_hh_income"].iloc[0] == pytest.approx(50000.0)
        # 2000 income should be inflated (2020 CPI / 1999 CPI)
        factor_2000 = CPI[2020] / CPI[1999]
        assert adjusted[2000]["median_hh_income"].iloc[0] == pytest.approx(50000.0 * factor_2000)
        # 2010 income inflated
        factor_2010 = CPI[2020] / CPI[2010]
        assert adjusted[2010]["median_hh_income"].iloc[0] == pytest.approx(50000.0 * factor_2010)

    def test_does_not_modify_non_income_columns(self):
        frames = _make_census_frames(FIPS)
        original_pop = frames[2000]["pop_total"].iloc[0]
        _adjust_income_to_2020(frames, cpi=CPI)
        assert frames[2000]["pop_total"].iloc[0] == original_pop


# ---------------------------------------------------------------------------
# interpolate_for_year
# ---------------------------------------------------------------------------

class TestInterpolateForYear:
    def test_pre_2000_uses_2000_flat(self):
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 1996, cpi=CPI)
        assert result["year"].iloc[0] == 1996
        # Population should match the 2000 census (flat, no extrapolation)
        assert result["pop_total"].iloc[0] == pytest.approx(1000.0)

    def test_post_2020_uses_2020_flat(self):
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2024, cpi=CPI)
        assert result["year"].iloc[0] == 2024
        assert result["pop_total"].iloc[0] == pytest.approx(1500.0)

    def test_exact_census_year_no_interpolation(self):
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2010, cpi=CPI)
        assert result["pop_total"].iloc[0] == pytest.approx(1200.0)

    def test_midpoint_interpolation(self):
        """2005 should be midpoint between 2000 and 2010."""
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2005, cpi=CPI)
        expected_pop = (1000 + 1200) / 2
        assert result["pop_total"].iloc[0] == pytest.approx(expected_pop)

    def test_2015_interpolation(self):
        """2015 should be midpoint between 2010 and 2020."""
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2015, cpi=CPI)
        expected_pop = (1200 + 1500) / 2
        assert result["pop_total"].iloc[0] == pytest.approx(expected_pop)

    def test_weighted_interpolation(self):
        """2004 should be 40% toward 2010 from 2000."""
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2004, cpi=CPI)
        weight_later = 4 / 10
        expected_pop = (1 - weight_later) * 1000 + weight_later * 1200
        assert result["pop_total"].iloc[0] == pytest.approx(expected_pop)

    def test_derived_ratios_present(self):
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2015, cpi=CPI)
        for name, _, _ in DERIVED_RATIOS:
            assert name in result.columns

    def test_pct_white_nh_correct(self):
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2020, cpi=CPI)
        expected = 1500 * 0.6 / 1500  # 0.6
        assert result["pct_white_nh"].iloc[0] == pytest.approx(expected)

    def test_year_column_set(self):
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2016, cpi=CPI)
        assert (result["year"] == 2016).all()

    def test_county_fips_preserved(self):
        frames = _make_census_frames(FIPS)
        result = interpolate_for_year(frames, 2010, cpi=CPI)
        assert set(result["county_fips"]) == set(FIPS)

    def test_does_not_mutate_input(self):
        frames = _make_census_frames(FIPS)
        original_income = frames[2000]["median_hh_income"].iloc[0]
        interpolate_for_year(frames, 2015, cpi=CPI)
        assert frames[2000]["median_hh_income"].iloc[0] == pytest.approx(original_income)

    def test_missing_county_in_later_census(self):
        """Counties that only exist in one census year should be handled via intersection."""
        frames = _make_census_frames(FIPS)
        # Remove one county from 2020
        frames[2020] = frames[2020][frames[2020]["county_fips"] != "13001"]
        result = interpolate_for_year(frames, 2015, cpi=CPI)
        # Only common counties should appear
        assert "13001" not in result["county_fips"].values


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_census_years(self):
        assert CENSUS_YEARS == [2000, 2010, 2020]

    def test_income_ref_year_mapping(self):
        assert INCOME_REF_YEAR[2000] == 1999
        assert INCOME_REF_YEAR[2010] == 2010
        assert INCOME_REF_YEAR[2020] == 2020

    def test_interpolation_cols_not_empty(self):
        assert len(INTERPOLATION_COLS) > 0

    def test_derived_ratios_are_tuples(self):
        for item in DERIVED_RATIOS:
            assert len(item) == 3
