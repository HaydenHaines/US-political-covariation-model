"""Tests for type demographic description pipeline.

Covers describe_types():
  - output columns match spec
  - population weighting is mathematically correct
  - election_year filter works
  - all types 0..J-1 present in output
  - rcms columns present when rcms_features provided
  - works without rcms (columns absent, no error)
  - n_counties matches count of dominant_type assignments
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.description.describe_types import describe_types


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_type_assignments(n_counties: int = 6, j: int = 3) -> pd.DataFrame:
    """Build a minimal type_assignments DataFrame with n_counties x J score cols."""
    rng = np.random.default_rng(42)
    data: dict = {"county_fips": [f"1200{i}" for i in range(n_counties)]}
    scores = rng.standard_normal((n_counties, j))
    for k in range(j):
        data[f"type_{k}_score"] = scores[:, k]
    # dominant_type: argmax of abs scores per county
    data["dominant_type"] = np.argmax(np.abs(scores), axis=1)
    return pd.DataFrame(data)


def _make_demographics(n_counties: int = 6, years: list[int] | None = None) -> pd.DataFrame:
    """Build a long-format demographics DataFrame (county_fips, year, demo cols)."""
    if years is None:
        years = [2020]
    rows = []
    for yr in years:
        for i in range(n_counties):
            rows.append({
                "county_fips": f"1200{i}",
                "year": yr,
                "pop_total": float(10000 + i * 1000),
                "pct_white_nh": 0.5 + i * 0.01,
                "pct_black": 0.2 - i * 0.005,
                "pct_asian": 0.05,
                "pct_hispanic": 0.15,
                "median_age": 38.0 + i,
                "median_hh_income_2020": 50000.0 + i * 1000,
                "pct_bachelors_plus": 0.25 + i * 0.01,
                "pct_owner_occupied": 0.65,
                "pct_wfh": 0.10,
                "pct_transit": 0.05,
                "pct_car": 0.80,
            })
    return pd.DataFrame(rows)


def _make_rcms(n_counties: int = 6) -> pd.DataFrame:
    """Build a minimal RCMS features DataFrame."""
    return pd.DataFrame({
        "county_fips": [f"1200{i}" for i in range(n_counties)],
        "evangelical_share": [0.3 + i * 0.01 for i in range(n_counties)],
        "mainline_share": [0.1] * n_counties,
        "catholic_share": [0.15] * n_counties,
        "black_protestant_share": [0.05] * n_counties,
        "congregations_per_1000": [5.0] * n_counties,
        "religious_adherence_rate": [450.0] * n_counties,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDescribeTypesOutputColumns:
    def test_has_type_id(self):
        ta = _make_type_assignments()
        demo = _make_demographics()
        result = describe_types(ta, demo)
        assert "type_id" in result.columns

    def test_has_n_counties(self):
        ta = _make_type_assignments()
        demo = _make_demographics()
        result = describe_types(ta, demo)
        assert "n_counties" in result.columns

    def test_has_demographic_columns(self):
        ta = _make_type_assignments()
        demo = _make_demographics()
        result = describe_types(ta, demo)
        for col in ["pct_white_nh", "pct_black", "pct_asian", "pct_hispanic", "median_age"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_has_pop_total(self):
        ta = _make_type_assignments()
        demo = _make_demographics()
        result = describe_types(ta, demo)
        assert "pop_total" in result.columns


class TestDescribeTypesAllTypesPresent:
    def test_all_types_in_output(self):
        """Every type 0..J-1 must appear in the output."""
        j = 4
        ta = _make_type_assignments(n_counties=12, j=j)
        demo = _make_demographics(n_counties=12)
        result = describe_types(ta, demo)
        assert set(result["type_id"].tolist()) == set(range(j))

    def test_output_row_count_equals_j(self):
        j = 3
        ta = _make_type_assignments(n_counties=9, j=j)
        demo = _make_demographics(n_counties=9)
        result = describe_types(ta, demo)
        assert len(result) == j


class TestDescribeTypesWeighted:
    def test_population_weighted_mean_correct(self):
        """Verify weighted mean with controlled, known data.

        All counties contribute to every type's profile, weighted by their
        abs(score) on that type. This is the soft-membership design: even a
        county not dominant in type 0 still has a small weight on type 0.
        """
        # 4 counties, 2 types
        scores_0 = [2.0, 1.5, 0.1, 0.2]  # type_0_score for each county
        pct_white = [0.60, 0.70, 0.40, 0.50]
        ta = pd.DataFrame({
            "county_fips": ["10001", "10002", "10003", "10004"],
            "type_0_score": scores_0,
            "type_1_score": [0.1, 0.2, 1.8, 1.6],
            "dominant_type": [0, 0, 1, 1],
        })
        demo = pd.DataFrame({
            "county_fips": ["10001", "10002", "10003", "10004"],
            "year": [2020, 2020, 2020, 2020],
            "pop_total": [10000.0, 5000.0, 8000.0, 4000.0],
            "pct_white_nh": pct_white,
        })
        result = describe_types(ta, demo)

        # All 4 counties contribute, weighted by abs(type_0_score)
        # weights = [2.0, 1.5, 0.1, 0.2], total = 3.8
        weights = [abs(s) for s in scores_0]
        total_w = sum(weights)
        expected_type0 = sum(w * v for w, v in zip(weights, pct_white)) / total_w
        row0 = result.loc[result["type_id"] == 0, "pct_white_nh"].iloc[0]
        assert abs(row0 - expected_type0) < 1e-9

    def test_weight_is_abs_score_not_population(self):
        """The weight comes from the type score magnitude, not just pop_total."""
        # Two counties with very different populations but same score → equal weight
        ta = pd.DataFrame({
            "county_fips": ["10001", "10002"],
            "type_0_score": [1.0, -1.0],  # equal absolute scores
            "dominant_type": [0, 0],
        })
        demo = pd.DataFrame({
            "county_fips": ["10001", "10002"],
            "year": [2020, 2020],
            "pop_total": [100000.0, 1000.0],  # very different pops
            "pct_white_nh": [0.4, 0.8],
        })
        result = describe_types(ta, demo)
        # Equal abs scores → unweighted mean
        expected = (0.4 + 0.8) / 2
        row = result.loc[result["type_id"] == 0, "pct_white_nh"].iloc[0]
        assert abs(row - expected) < 1e-9


class TestDescribeTypesElectionYear:
    def test_uses_given_election_year(self):
        """When election_year is specified, only that year's demographics are used."""
        ta = _make_type_assignments(n_counties=6, j=3)
        demo = _make_demographics(n_counties=6, years=[2016, 2020])
        # Make 2016 and 2020 values clearly different
        demo.loc[demo["year"] == 2016, "pct_white_nh"] = 0.99
        demo.loc[demo["year"] == 2020, "pct_white_nh"] = 0.01

        result_2016 = describe_types(ta, demo, election_year=2016)
        result_2020 = describe_types(ta, demo, election_year=2020)

        # pct_white_nh should be close to 0.99 for 2016, 0.01 for 2020
        assert result_2016["pct_white_nh"].mean() > 0.5
        assert result_2020["pct_white_nh"].mean() < 0.5

    def test_uses_latest_year_when_none(self):
        """When election_year is None, uses the latest available year."""
        ta = _make_type_assignments(n_counties=6, j=3)
        demo = _make_demographics(n_counties=6, years=[2016, 2020])
        demo.loc[demo["year"] == 2016, "pct_white_nh"] = 0.99
        demo.loc[demo["year"] == 2020, "pct_white_nh"] = 0.01

        result = describe_types(ta, demo, election_year=None)
        # Should use 2020 (latest)
        assert result["pct_white_nh"].mean() < 0.5


class TestDescribeTypesNCounties:
    def test_n_counties_matches_dominant_type_count(self):
        """n_counties for each type_id must equal count of counties whose dominant_type == type_id."""
        ta = pd.DataFrame({
            "county_fips": [f"1000{i}" for i in range(8)],
            "type_0_score": [2.0, 1.8, 1.5, 1.2, -0.1, -0.2, -0.1, -0.2],
            "type_1_score": [0.1, 0.2, 0.1, 0.1, 2.0, 1.9, 1.7, 1.6],
            "dominant_type": [0, 0, 0, 0, 1, 1, 1, 1],
        })
        demo = _make_demographics(n_counties=8)
        demo["county_fips"] = [f"1000{i}" for i in range(8)]
        result = describe_types(ta, demo)

        expected_type0 = (ta["dominant_type"] == 0).sum()
        expected_type1 = (ta["dominant_type"] == 1).sum()

        n0 = result.loc[result["type_id"] == 0, "n_counties"].iloc[0]
        n1 = result.loc[result["type_id"] == 1, "n_counties"].iloc[0]
        assert n0 == expected_type0
        assert n1 == expected_type1


class TestDescribeTypesWithRcms:
    def test_rcms_columns_present_when_provided(self):
        ta = _make_type_assignments()
        demo = _make_demographics()
        rcms = _make_rcms()
        result = describe_types(ta, demo, rcms_features=rcms)
        assert "evangelical_share" in result.columns

    def test_rcms_columns_multiple(self):
        ta = _make_type_assignments()
        demo = _make_demographics()
        rcms = _make_rcms()
        result = describe_types(ta, demo, rcms_features=rcms)
        for col in ["evangelical_share", "mainline_share", "catholic_share"]:
            assert col in result.columns

    def test_rcms_weighted_correctly(self):
        """RCMS features should also use score-weighted means."""
        ta = pd.DataFrame({
            "county_fips": ["10001", "10002"],
            "type_0_score": [3.0, 1.0],
            "dominant_type": [0, 0],
        })
        demo = pd.DataFrame({
            "county_fips": ["10001", "10002"],
            "year": [2020, 2020],
            "pop_total": [1000.0, 1000.0],
            "pct_white_nh": [0.5, 0.5],
        })
        rcms = pd.DataFrame({
            "county_fips": ["10001", "10002"],
            "evangelical_share": [0.6, 0.2],
        })
        result = describe_types(ta, demo, rcms_features=rcms)
        # Weights: 3.0 and 1.0
        expected = (0.6 * 3.0 + 0.2 * 1.0) / (3.0 + 1.0)
        val = result.loc[result["type_id"] == 0, "evangelical_share"].iloc[0]
        assert abs(val - expected) < 1e-9


class TestDescribeTypesWithoutRcms:
    def test_works_without_rcms(self):
        """Calling describe_types without rcms_features should not raise."""
        ta = _make_type_assignments()
        demo = _make_demographics()
        result = describe_types(ta, demo)
        assert len(result) > 0

    def test_no_rcms_columns_when_not_provided(self):
        ta = _make_type_assignments()
        demo = _make_demographics()
        result = describe_types(ta, demo)
        rcms_cols = ["evangelical_share", "mainline_share", "catholic_share",
                     "black_protestant_share", "congregations_per_1000", "religious_adherence_rate"]
        for col in rcms_cols:
            assert col not in result.columns

    def test_rcms_none_explicit(self):
        """Passing rcms_features=None should behave same as omitting it."""
        ta = _make_type_assignments()
        demo = _make_demographics()
        result = describe_types(ta, demo, rcms_features=None)
        assert "evangelical_share" not in result.columns
        assert len(result) > 0
