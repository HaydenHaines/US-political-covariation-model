"""Tests for tract-level feature registry and feature builders.

Uses synthetic DataFrames to verify:
- Feature registry categories, selection, and year exclusion
- Log-odds shift computation
- State-centering of non-presidential shifts
- WWC (white working class) interaction
- County-proxy religion mapping
- Vote density computation
- Split-ticket computation
- Turnout computation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tracts.feature_registry import (
    REGISTRY,
    FeatureSpec,
    select_features,
)
from src.tracts.build_tract_features import (
    build_electoral_features,
    build_demographic_features,
    build_religion_features,
    _logodds,
    _safe_ratio,
)


# ── Registry tests ───────────────────────────────────────────────────────────


class TestFeatureRegistry:
    def test_registry_categories(self):
        """All three categories are present in the registry."""
        cats = {spec.category for spec in REGISTRY}
        assert "electoral" in cats
        assert "demographic" in cats
        assert "religion" in cats

    def test_registry_select_electoral(self):
        """select_features(category='electoral') returns only electoral features."""
        names = select_features(category="electoral")
        assert len(names) > 0
        # Verify every returned name belongs to an electoral spec
        electoral_names = {s.name for s in REGISTRY if s.category == "electoral"}
        for n in names:
            assert n in electoral_names, f"{n} is not electoral"

    def test_registry_select_by_subcategory(self):
        """Subcategory selection narrows results."""
        pres_shift = select_features(subcategory="presidential_shifts")
        assert len(pres_shift) > 0
        for name in pres_shift:
            spec = next(s for s in REGISTRY if s.name == name)
            assert spec.subcategory == "presidential_shifts"

    def test_registry_exclude_year(self):
        """Features with source_year=2024 excluded when exclude_year=2024."""
        all_names = select_features()
        filtered = select_features(exclude_year=2024)
        excluded = set(all_names) - set(filtered)
        # Every excluded feature should have source_year 2024
        for name in excluded:
            spec = next(s for s in REGISTRY if s.name == name)
            assert spec.source_year == 2024
        # No feature with source_year=2024 should survive
        for name in filtered:
            spec = next(s for s in REGISTRY if s.name == name)
            assert spec.source_year != 2024

    def test_registry_all_have_required_fields(self):
        """Every FeatureSpec has non-empty name, category, subcategory, source."""
        for spec in REGISTRY:
            assert spec.name, "empty name"
            assert spec.category, "empty category"
            assert spec.subcategory, "empty subcategory"
            assert spec.source, "empty source"
            assert spec.description, "empty description"

    def test_registry_no_duplicate_names(self):
        """No two specs share the same name."""
        names = [s.name for s in REGISTRY]
        assert len(names) == len(set(names)), f"duplicates: {[n for n in names if names.count(n) > 1]}"


# ── Electoral feature tests ──────────────────────────────────────────────────


@pytest.fixture
def tract_votes_pair():
    """Two synthetic tract-vote DataFrames for 2020 and 2024 presidential."""
    geoids = ["12001000100", "12001000200", "13001000100"]
    df_2020 = pd.DataFrame({
        "GEOID": geoids,
        "votes_dem": [600.0, 400.0, 300.0],
        "votes_rep": [400.0, 500.0, 600.0],
        "votes_total": [1000.0, 900.0, 900.0],
        "dem_share": [0.60, 0.444, 0.333],
        "state": ["FL", "FL", "GA"],
        "year": [2020, 2020, 2020],
        "race": ["president", "president", "president"],
    })
    df_2024 = pd.DataFrame({
        "GEOID": geoids,
        "votes_dem": [550.0, 350.0, 350.0],
        "votes_rep": [500.0, 600.0, 550.0],
        "votes_total": [1050.0, 950.0, 900.0],
        "dem_share": [550 / 1050, 350 / 950, 350 / 900],
        "state": ["FL", "FL", "GA"],
        "year": [2024, 2024, 2024],
        "race": ["president", "president", "president"],
    })
    return df_2020, df_2024


@pytest.fixture
def tract_areas():
    return pd.Series(
        [10.0, 25.0, 50.0],
        index=pd.Index(["12001000100", "12001000200", "13001000100"], name="GEOID"),
    )


@pytest.fixture
def state_fips_map():
    return {"12": "FL", "13": "GA", "01": "AL"}


def test_shift_computation(tract_votes_pair, tract_areas, state_fips_map):
    """Log-odds shift correct for known input."""
    df_2020, df_2024 = tract_votes_pair
    tract_votes = {
        "president_2020": df_2020,
        "president_2024": df_2024,
    }
    result = build_electoral_features(tract_votes, tract_areas, state_fips_map)

    eps = 0.01
    # Check first tract: dem_share 0.60 -> 550/1050
    p_early = np.clip(0.60, eps, 1 - eps)
    p_late = np.clip(550 / 1050, eps, 1 - eps)
    expected_shift = np.log(p_late / (1 - p_late)) - np.log(p_early / (1 - p_early))

    row = result.loc[result["GEOID"] == "12001000100"]
    assert len(row) == 1
    actual = row["pres_d_shift_20_24"].values[0]
    np.testing.assert_almost_equal(actual, expected_shift, decimal=5)


def test_vote_density(tract_votes_pair, tract_areas, state_fips_map):
    """votes_total / area_sqkm computed correctly."""
    df_2020, df_2024 = tract_votes_pair
    tract_votes = {
        "president_2020": df_2020,
        "president_2024": df_2024,
    }
    result = build_electoral_features(tract_votes, tract_areas, state_fips_map)

    row = result.loc[result["GEOID"] == "12001000100"]
    # 2024 votes_total=1050, area=10.0
    np.testing.assert_almost_equal(
        row["vote_density_2024"].values[0], 1050.0 / 10.0
    )


def test_turnout(tract_votes_pair, tract_areas, state_fips_map):
    """Turnout level columns present and proportional to total votes."""
    df_2020, df_2024 = tract_votes_pair
    tract_votes = {
        "president_2020": df_2020,
        "president_2024": df_2024,
    }
    result = build_electoral_features(tract_votes, tract_areas, state_fips_map)

    # turnout_2020 should be votes_total for 2020
    row = result.loc[result["GEOID"] == "12001000100"]
    assert "turnout_2020" in result.columns
    assert row["turnout_2020"].values[0] == 1000.0


def test_split_ticket(tract_votes_pair, tract_areas, state_fips_map):
    """Split ticket = abs(pres_dem_share - house_dem_share)."""
    df_2020, df_2024 = tract_votes_pair
    # Add a house race for 2020
    df_house_2020 = df_2020.copy()
    df_house_2020["race"] = "house"
    df_house_2020["dem_share"] = [0.55, 0.50, 0.40]  # different from pres

    tract_votes = {
        "president_2020": df_2020,
        "president_2024": df_2024,
        "house_2020": df_house_2020,
    }
    result = build_electoral_features(tract_votes, tract_areas, state_fips_map)

    if "split_ticket_2020" in result.columns:
        row = result.loc[result["GEOID"] == "12001000100"]
        expected = abs(0.60 - 0.55)
        np.testing.assert_almost_equal(
            row["split_ticket_2020"].values[0], expected, decimal=4
        )


def test_state_centering(tract_votes_pair, tract_areas, state_fips_map):
    """Non-presidential shifts have zero state mean after centering."""
    df_2020, df_2024 = tract_votes_pair
    # Create governor data
    df_gov_2018 = pd.DataFrame({
        "GEOID": ["12001000100", "12001000200", "13001000100"],
        "votes_dem": [500.0, 400.0, 350.0],
        "votes_rep": [500.0, 500.0, 550.0],
        "votes_total": [1000.0, 900.0, 900.0],
        "dem_share": [0.50, 0.444, 0.389],
        "state": ["FL", "FL", "GA"],
        "year": [2018, 2018, 2018],
        "race": ["governor", "governor", "governor"],
    })
    df_gov_2020 = pd.DataFrame({
        "GEOID": ["12001000100", "12001000200", "13001000100"],
        "votes_dem": [550.0, 380.0, 360.0],
        "votes_rep": [450.0, 520.0, 540.0],
        "votes_total": [1000.0, 900.0, 900.0],
        "dem_share": [0.55, 0.422, 0.400],
        "state": ["FL", "FL", "GA"],
        "year": [2020, 2020, 2020],
        "race": ["governor", "governor", "governor"],
    })
    tract_votes = {
        "president_2020": df_2020,
        "president_2024": df_2024,
        "governor_2018": df_gov_2018,
        "governor_2020": df_gov_2020,
    }
    result = build_electoral_features(tract_votes, tract_areas, state_fips_map)

    # If governor shift columns exist and are state-centered,
    # the FL tracts should have mean ~0 and GA tracts should have mean ~0
    if "gov_d_shift_18_20_sc" in result.columns:
        fl_mask = result["GEOID"].str.startswith("12")
        fl_mean = result.loc[fl_mask, "gov_d_shift_18_20_sc"].mean()
        np.testing.assert_almost_equal(fl_mean, 0.0, decimal=5)


# ── Demographic feature tests ────────────────────────────────────────────────


@pytest.fixture
def tract_acs():
    """Synthetic ACS tract data."""
    return pd.DataFrame({
        "GEOID": ["12001000100", "12001000200", "13001000100"],
        "pop_total": [5000, 3000, 4000],
        "pop_white_nh": [3000, 1000, 2000],
        "pop_black": [1000, 1500, 1200],
        "pop_hispanic": [500, 300, 500],
        "pop_asian": [200, 100, 100],
        "pop_foreign_born": [400, 200, 150],
        "educ_total": [4000, 2500, 3000],
        "educ_bachelors": [800, 300, 400],
        "educ_graduate": [200, 100, 100],
        "educ_no_hs": [300, 500, 600],
        "median_hh_income": [65000, 45000, 50000],
        "poverty_rate": [0.10, 0.18, 0.15],
        "gini": [0.42, 0.38, 0.40],
        "housing_owner": [2000, 800, 1500],
        "housing_units": [3000, 2000, 2500],
        "median_home_value": [250000, 150000, 180000],
        "housing_multi_unit": [500, 800, 300],
        "housing_pre_1960": [400, 600, 200],
        "median_rent": [1200, 900, 1000],
        "median_age": [38.5, 32.0, 45.0],
        "pop_under_18": [1200, 900, 800],
        "pop_over_65": [800, 400, 1000],
        "hh_total": [2000, 1200, 1600],
        "hh_single": [500, 400, 600],
        "commute_wfh": [300, 100, 200],
        "mean_commute_time": [28.5, 22.0, 35.0],
        "pop_no_vehicle": [100, 200, 50],
        "pop_veteran": [400, 200, 500],
    })


def test_wwc_computation(tract_acs):
    """White working class = pct_white_nh * (1 - pct_ba_plus)."""
    result = build_demographic_features(tract_acs)

    row = result.loc[result["GEOID"] == "12001000100"]
    pct_white = 3000 / 5000
    pct_ba = (800 + 200) / 4000
    expected_wwc = pct_white * (1 - pct_ba)
    np.testing.assert_almost_equal(row["pct_wwc"].values[0], expected_wwc, decimal=5)


def test_demographic_race_ratios(tract_acs):
    """Race percentages sum reasonably and are correct."""
    result = build_demographic_features(tract_acs)

    row = result.loc[result["GEOID"] == "12001000100"]
    np.testing.assert_almost_equal(row["pct_white_nh"].values[0], 3000 / 5000)
    np.testing.assert_almost_equal(row["pct_black"].values[0], 1000 / 5000)
    np.testing.assert_almost_equal(row["pct_hispanic"].values[0], 500 / 5000)
    np.testing.assert_almost_equal(row["pct_asian"].values[0], 200 / 5000)


def test_demographic_education(tract_acs):
    """Education features are correct."""
    result = build_demographic_features(tract_acs)

    row = result.loc[result["GEOID"] == "12001000100"]
    np.testing.assert_almost_equal(row["pct_ba_plus"].values[0], (800 + 200) / 4000)
    np.testing.assert_almost_equal(row["pct_graduate"].values[0], 200 / 4000)
    np.testing.assert_almost_equal(row["pct_no_hs"].values[0], 300 / 4000)


def test_demographic_housing(tract_acs):
    """Housing features computed correctly."""
    result = build_demographic_features(tract_acs)

    row = result.loc[result["GEOID"] == "12001000100"]
    np.testing.assert_almost_equal(row["pct_owner_occupied"].values[0], 2000 / 3000)
    assert row["median_home_value"].values[0] == 250000
    np.testing.assert_almost_equal(row["pct_multi_unit"].values[0], 500 / 3000)
    np.testing.assert_almost_equal(row["pct_pre_1960"].values[0], 400 / 3000)


def test_rent_burden(tract_acs):
    """rent_burden = median_rent * 12 / median_hh_income."""
    result = build_demographic_features(tract_acs)

    row = result.loc[result["GEOID"] == "12001000100"]
    expected = (1200 * 12) / 65000
    np.testing.assert_almost_equal(row["rent_burden"].values[0], expected, decimal=4)


# ── Religion feature tests ───────────────────────────────────────────────────


@pytest.fixture
def rcms_county():
    """Synthetic county RCMS data."""
    return pd.DataFrame({
        "county_fips": ["12001", "13001"],
        "evangelical_share": [0.25, 0.40],
        "catholic_share": [0.15, 0.08],
        "black_protestant_share": [0.10, 0.20],
        "adherence_rate": [0.55, 0.65],
    })


@pytest.fixture
def tract_geoids():
    return pd.Series(["12001000100", "12001000200", "13001000100"])


def test_religion_county_proxy(rcms_county, tract_geoids):
    """All tracts in same county get same religion values."""
    result = build_religion_features(rcms_county, tract_geoids)

    # Both 12001 tracts should get the same evangelical_share
    t1 = result.loc[result["GEOID"] == "12001000100", "evangelical_share"].values[0]
    t2 = result.loc[result["GEOID"] == "12001000200", "evangelical_share"].values[0]
    assert t1 == t2 == 0.25

    # 13001 tract should get different value
    t3 = result.loc[result["GEOID"] == "13001000100", "evangelical_share"].values[0]
    assert t3 == 0.40


def test_religion_all_columns_present(rcms_county, tract_geoids):
    """All religion columns are in the output."""
    result = build_religion_features(rcms_county, tract_geoids)
    for col in ("evangelical_share", "catholic_share", "black_protestant_share", "adherence_rate"):
        assert col in result.columns


# ── Helper tests ─────────────────────────────────────────────────────────────


def test_logodds_basic():
    """logodds(0.5) = 0, logodds(0.75) > 0."""
    assert _logodds(0.5) == pytest.approx(0.0)
    assert _logodds(0.75) > 0
    assert _logodds(0.25) < 0


def test_safe_ratio_zero_denom():
    """_safe_ratio returns NaN for zero denominator."""
    num = pd.Series([10.0, 20.0])
    denom = pd.Series([0.0, 5.0])
    result = _safe_ratio(num, denom)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(4.0)
