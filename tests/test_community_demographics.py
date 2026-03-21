"""Tests for the county-level community demographics pipeline.

Covers:
  - fetch_acs_county: fetch helpers and transforms
  - build_county_acs_features: derived ratio computation
  - describe_communities: build_county_community_profiles aggregation
  - DuckDB: community_profiles and county_demographics tables loaded
  - API: /communities/{id} returns demographics sub-object
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# fetch_acs_county
# ---------------------------------------------------------------------------

from src.assembly.fetch_acs_county import (
    build_county_fips,
    cast_numeric,
    rename_columns,
    VARIABLES,
)


class TestBuildCountyFips:
    def test_combines_state_and_county(self):
        df = pd.DataFrame({"state": ["12", "13"], "county": ["001", "005"]})
        result = build_county_fips(df)
        assert "county_fips" in result.columns
        assert result["county_fips"].tolist() == ["12001", "13005"]

    def test_drops_geo_cols(self):
        df = pd.DataFrame({"state": ["12"], "county": ["001"], "pop": [100]})
        result = build_county_fips(df)
        assert "state" not in result.columns
        assert "county" not in result.columns

    def test_five_digit_fips(self):
        df = pd.DataFrame({"state": ["01"], "county": ["003"]})
        result = build_county_fips(df)
        assert len(result["county_fips"].iloc[0]) == 5


class TestCastNumeric:
    def test_converts_strings_to_float(self):
        df = pd.DataFrame({"B03002_001E": ["1000", "2000"]})
        result = cast_numeric(df, ["B03002_001E"])
        assert pd.api.types.is_numeric_dtype(result["B03002_001E"])

    def test_replaces_null_sentinel(self):
        df = pd.DataFrame({"B03002_001E": ["-666666666", "5000"]})
        result = cast_numeric(df, ["B03002_001E"])
        assert pd.isna(result["B03002_001E"].iloc[0])
        assert result["B03002_001E"].iloc[1] == 5000.0

    def test_handles_missing_column_gracefully(self):
        df = pd.DataFrame({"other_col": ["1"]})
        # Should not raise even if api_var not in df
        result = cast_numeric(df, ["B03002_001E"])
        assert "other_col" in result.columns


class TestRenameColumns:
    def test_renames_api_codes(self):
        df = pd.DataFrame({"B03002_001E": [100], "county_fips": ["12001"]})
        result = rename_columns(df)
        assert "pop_total" in result.columns
        assert "B03002_001E" not in result.columns

    def test_leaves_non_api_cols_intact(self):
        df = pd.DataFrame({"county_fips": ["12001"], "B03002_001E": [100]})
        result = rename_columns(df)
        assert "county_fips" in result.columns


# ---------------------------------------------------------------------------
# build_county_acs_features
# ---------------------------------------------------------------------------

from src.assembly.build_county_acs_features import build_features, _safe_ratio


class TestSafeRatio:
    def test_normal_division(self):
        num = pd.Series([100.0, 200.0])
        den = pd.Series([1000.0, 400.0])
        result = _safe_ratio(num, den)
        assert abs(result.iloc[0] - 0.1) < 1e-9
        assert abs(result.iloc[1] - 0.5) < 1e-9

    def test_zero_denominator_returns_nan(self):
        num = pd.Series([50.0])
        den = pd.Series([0.0])
        result = _safe_ratio(num, den)
        assert pd.isna(result.iloc[0])

    def test_nan_denominator_returns_nan(self):
        num = pd.Series([50.0])
        den = pd.Series([float("nan")])
        result = _safe_ratio(num, den)
        assert pd.isna(result.iloc[0])


def _make_raw_acs(n: int = 3) -> pd.DataFrame:
    """Build a minimal synthetic raw ACS DataFrame matching fetch_acs_county output."""
    return pd.DataFrame({
        "county_fips": [f"1200{i}" for i in range(n)],
        "pop_total": [10000.0] * n,
        "pop_white_nh": [6000.0] * n,
        "pop_black": [2000.0] * n,
        "pop_asian": [500.0] * n,
        "pop_hispanic": [1500.0] * n,
        "median_age": [38.0] * n,
        "median_hh_income": [55000.0] * n,
        "educ_total": [7000.0] * n,
        "educ_bachelors": [1200.0] * n,
        "educ_masters": [500.0] * n,
        "educ_professional": [100.0] * n,
        "educ_doctorate": [50.0] * n,
        "housing_units": [4000.0] * n,
        "housing_owner": [2800.0] * n,
        "commute_total": [4500.0] * n,
        "commute_car": [3600.0] * n,
        "commute_transit": [200.0] * n,
        "commute_wfh": [450.0] * n,
        "occ_total": [5000.0] * n,
        "occ_mgmt_male": [800.0] * n,
        "occ_mgmt_female": [700.0] * n,
    })


class TestBuildFeatures:
    def test_output_has_expected_columns(self):
        raw = _make_raw_acs()
        result = build_features(raw)
        expected = [
            "county_fips", "pop_total",
            "pct_white_nh", "pct_black", "pct_asian", "pct_hispanic",
            "median_age", "median_hh_income",
            "pct_bachelors_plus", "pct_owner_occupied", "pct_wfh", "pct_management",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_pct_white_nh_correct(self):
        raw = _make_raw_acs(1)
        result = build_features(raw)
        assert abs(result["pct_white_nh"].iloc[0] - 0.6) < 1e-9

    def test_pct_bachelors_plus_sums_all_higher_ed(self):
        raw = _make_raw_acs(1)
        result = build_features(raw)
        expected = (1200 + 500 + 100 + 50) / 7000
        assert abs(result["pct_bachelors_plus"].iloc[0] - expected) < 1e-9

    def test_pct_owner_occupied_correct(self):
        raw = _make_raw_acs(1)
        result = build_features(raw)
        expected = 2800 / 4000
        assert abs(result["pct_owner_occupied"].iloc[0] - expected) < 1e-9

    def test_pct_wfh_correct(self):
        raw = _make_raw_acs(1)
        result = build_features(raw)
        expected = 450 / 4500
        assert abs(result["pct_wfh"].iloc[0] - expected) < 1e-9

    def test_pct_management_sums_male_and_female(self):
        raw = _make_raw_acs(1)
        result = build_features(raw)
        expected = (800 + 700) / 5000
        assert abs(result["pct_management"].iloc[0] - expected) < 1e-9

    def test_row_count_preserved(self):
        raw = _make_raw_acs(5)
        result = build_features(raw)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# build_county_community_profiles
# ---------------------------------------------------------------------------

from src.description.describe_communities import (
    build_county_community_profiles,
    COUNTY_ACS_COLS,
    COUNTY_RCMS_COLS,
)


def _make_county_assignments(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": [f"1200{i}" for i in range(n)],
        "community_id": [0, 0, 1, 1],
    })


def _make_county_acs(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": [f"1200{i}" for i in range(n)],
        "pop_total": [10000.0, 5000.0, 8000.0, 4000.0],
        "pct_white_nh": [0.6, 0.7, 0.4, 0.5],
        "pct_black": [0.2, 0.1, 0.4, 0.3],
        "pct_asian": [0.05, 0.05, 0.05, 0.05],
        "pct_hispanic": [0.15, 0.15, 0.15, 0.15],
        "median_age": [38.0, 42.0, 35.0, 40.0],
        "median_hh_income": [55000.0, 60000.0, 50000.0, 52000.0],
        "pct_bachelors_plus": [0.25, 0.30, 0.20, 0.22],
        "pct_owner_occupied": [0.65, 0.70, 0.60, 0.62],
        "pct_wfh": [0.10, 0.12, 0.08, 0.09],
        "pct_management": [0.25, 0.28, 0.20, 0.22],
    })


def _make_county_rcms(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": [f"1200{i}" for i in range(n)],
        "evangelical_share": [0.3, 0.4, 0.2, 0.25],
        "mainline_share": [0.1, 0.1, 0.1, 0.1],
        "catholic_share": [0.15, 0.12, 0.18, 0.16],
        "black_protestant_share": [0.05, 0.04, 0.08, 0.06],
        "congregations_per_1000": [5.0, 6.0, 4.0, 4.5],
        "religious_adherence_rate": [450.0, 500.0, 400.0, 420.0],
    })


def _make_county_shifts(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": [f"1200{i}" for i in range(n)],
        "pres_d_shift_16_20": [0.02, 0.03, -0.01, -0.02],
    })


class TestBuildCountyCommunityProfiles:
    def test_one_row_per_community(self):
        result = build_county_community_profiles(
            _make_county_assignments(),
            _make_county_acs(),
            _make_county_rcms(),
            _make_county_shifts(),
        )
        assert len(result) == 2  # communities 0 and 1

    def test_has_required_columns(self):
        result = build_county_community_profiles(
            _make_county_assignments(),
            _make_county_acs(),
            _make_county_rcms(),
            _make_county_shifts(),
        )
        for col in ["community_id", "n_counties", "pop_total"]:
            assert col in result.columns
        assert "pct_white_nh" in result.columns
        assert "evangelical_share" in result.columns

    def test_n_counties_correct(self):
        result = build_county_community_profiles(
            _make_county_assignments(),
            _make_county_acs(),
            _make_county_rcms(),
            _make_county_shifts(),
        )
        # Each community has 2 counties
        assert all(result["n_counties"] == 2)

    def test_pop_total_is_sum(self):
        result = build_county_community_profiles(
            _make_county_assignments(),
            _make_county_acs(),
            _make_county_rcms(),
            _make_county_shifts(),
        )
        comm0 = result.loc[result["community_id"] == 0, "pop_total"].iloc[0]
        assert abs(comm0 - 15000.0) < 1e-6  # 10000 + 5000

    def test_population_weighted_mean(self):
        result = build_county_community_profiles(
            _make_county_assignments(),
            _make_county_acs(),
            _make_county_rcms(),
            _make_county_shifts(),
        )
        # Community 0: counties 12000 (pop 10000, pct_white_nh 0.6)
        #              and 12001 (pop 5000, pct_white_nh 0.7)
        # Weighted mean = (0.6*10000 + 0.7*5000) / 15000 = 0.6333...
        expected = (0.6 * 10000 + 0.7 * 5000) / 15000
        comm0 = result.loc[result["community_id"] == 0, "pct_white_nh"].iloc[0]
        assert abs(comm0 - expected) < 1e-9

    def test_normalises_community_col_name(self):
        """Should accept 'community' column in assignments (pipeline convention)."""
        assignments = _make_county_assignments().rename(
            columns={"community_id": "community"}
        )
        result = build_county_community_profiles(
            assignments, _make_county_acs(), _make_county_rcms(), _make_county_shifts()
        )
        assert "community_id" in result.columns
        assert len(result) == 2

    def test_shift_cols_included(self):
        result = build_county_community_profiles(
            _make_county_assignments(),
            _make_county_acs(),
            _make_county_rcms(),
            _make_county_shifts(),
        )
        assert "pres_d_shift_16_20" in result.columns

    def test_empty_rcms_tolerated(self):
        """RCMS DataFrame with no matching cols should not crash."""
        empty_rcms = pd.DataFrame({"county_fips": [f"1200{i}" for i in range(4)]})
        result = build_county_community_profiles(
            _make_county_assignments(),
            _make_county_acs(),
            empty_rcms,
            _make_county_shifts(),
        )
        assert len(result) == 2


# NOTE: API-level demographics tests live in api/tests/test_community_demographics_api.py
# because the `client` fixture is defined in api/tests/conftest.py.
