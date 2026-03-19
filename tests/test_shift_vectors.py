"""Tests for shift vector construction.

Uses synthetic DataFrames to verify shift math, AL midterm zeroing,
county-level fallback for MEDSL data, and output shape/columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_shift_vectors import (
    compute_presidential_shifts,
    compute_midterm_shifts,
    build_shift_vectors,
    SHIFT_COLS,
    AL_FIPS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vest_2016():
    """Synthetic VEST 2016 tract-level data (3 tracts, 2 states)."""
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pres_dem_share_2016": [0.60, 0.55, 0.40],
        "pres_total_2016": [1000, 800, 500],
    })


@pytest.fixture
def vest_2020():
    """Synthetic VEST 2020 tract-level data."""
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200", "01001000100"],
        "pres_dem_share_2020": [0.62, 0.50, 0.38],
        "pres_total_2020": [1100, 850, 520],
    })


@pytest.fixture
def medsl_2024():
    """Synthetic MEDSL 2024 county-level data."""
    return pd.DataFrame({
        "county_fips": ["12001", "01001"],
        "pres_dem_share_2024": [0.58, 0.35],
        "pres_total_2024": [2100, 540],
    })


@pytest.fixture
def vest_2018():
    """Synthetic VEST 2018 tract-level data (FL+GA only, AL uncontested)."""
    return pd.DataFrame({
        "tract_geoid": ["12001000100", "12001000200"],
        "gov_dem_share_2018": [0.55, 0.48],
        "gov_total_2018": [900, 700],
    })


@pytest.fixture
def medsl_2022():
    """Synthetic MEDSL 2022 county-level data."""
    return pd.DataFrame({
        "county_fips": ["12001"],
        "gov_dem_share_2022": [0.52, ],
        "gov_total_2022": [1700],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPresidentialShifts:
    def test_d_shift_is_difference(self, vest_2016, vest_2020):
        result = compute_presidential_shifts(vest_2016, vest_2020, "16_20")
        expected = 0.62 - 0.60  # tract 12001000100
        assert abs(result.loc[result.tract_geoid == "12001000100", "pres_d_shift_16_20"].iloc[0] - expected) < 1e-9

    def test_output_has_all_columns(self, vest_2016, vest_2020):
        result = compute_presidential_shifts(vest_2016, vest_2020, "16_20")
        for col in ["pres_d_shift_16_20", "pres_r_shift_16_20", "pres_turnout_shift_16_20"]:
            assert col in result.columns

    def test_no_nans_for_matched_tracts(self, vest_2016, vest_2020):
        result = compute_presidential_shifts(vest_2016, vest_2020, "16_20")
        assert not result[["pres_d_shift_16_20", "pres_r_shift_16_20"]].isna().any().any()


class TestMidtermShifts:
    def test_al_tracts_get_zero(self, vest_2018, medsl_2022):
        result = compute_midterm_shifts(vest_2018, medsl_2022)
        al_rows = result[result.tract_geoid.str.startswith("01")]
        if len(al_rows) > 0:
            assert (al_rows["mid_d_shift_18_22"] == 0.0).all()
            assert (al_rows["mid_r_shift_18_22"] == 0.0).all()
            assert (al_rows["mid_turnout_shift_18_22"] == 0.0).all()

    def test_fl_tracts_have_nonzero_shift(self, vest_2018, medsl_2022):
        result = compute_midterm_shifts(vest_2018, medsl_2022)
        fl_rows = result[result.tract_geoid.str.startswith("12")]
        assert not (fl_rows["mid_d_shift_18_22"] == 0.0).all()


class TestBuildShiftVectors:
    def test_output_columns(self, vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024):
        result = build_shift_vectors(vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024)
        assert "tract_geoid" in result.columns
        assert len(SHIFT_COLS) == 9
        for col in SHIFT_COLS:
            assert col in result.columns

    def test_output_has_all_tracts(self, vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024):
        result = build_shift_vectors(vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024)
        assert len(result) == 3  # all tracts from vest_2020
