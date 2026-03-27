"""Tests for src/tracts/feature_registry.py — feature tagging and selection.

Covers: FeatureSpec, REGISTRY, select_features() with all filter combinations.
"""
from __future__ import annotations

import pytest

from src.tracts.feature_registry import (
    REGISTRY,
    FeatureSpec,
    select_features,
)


# ---------------------------------------------------------------------------
# FeatureSpec dataclass
# ---------------------------------------------------------------------------

class TestFeatureSpec:
    def test_frozen(self):
        spec = FeatureSpec("test", "demo", "sub", "src", 2020, "desc")
        with pytest.raises(AttributeError):
            spec.name = "changed"

    def test_fields(self):
        spec = FeatureSpec("pct_white", "demographic", "race", "acs", 2020, "White share")
        assert spec.name == "pct_white"
        assert spec.category == "demographic"
        assert spec.subcategory == "race"
        assert spec.source == "acs"
        assert spec.source_year == 2020
        assert spec.description == "White share"

    def test_none_source_year(self):
        spec = FeatureSpec("median_age", "demographic", "age", "acs", None, "Age")
        assert spec.source_year is None


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_non_empty(self):
        assert len(REGISTRY) > 0

    def test_all_entries_are_featurespec(self):
        for spec in REGISTRY:
            assert isinstance(spec, FeatureSpec)

    def test_no_duplicate_names(self):
        names = [s.name for s in REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_categories_are_known(self):
        known = {"electoral", "demographic", "religion"}
        for spec in REGISTRY:
            assert spec.category in known, f"Unknown category: {spec.category}"

    def test_has_electoral_features(self):
        electoral = [s for s in REGISTRY if s.category == "electoral"]
        assert len(electoral) > 0

    def test_has_demographic_features(self):
        demographic = [s for s in REGISTRY if s.category == "demographic"]
        assert len(demographic) > 0

    def test_has_religion_features(self):
        religion = [s for s in REGISTRY if s.category == "religion"]
        assert len(religion) > 0

    def test_expected_feature_present(self):
        names = {s.name for s in REGISTRY}
        assert "pct_white_nh" in names
        assert "median_hh_income" in names
        assert "evangelical_share" in names
        assert "pres_dem_share_2020" in names


# ---------------------------------------------------------------------------
# select_features()
# ---------------------------------------------------------------------------

class TestSelectFeatures:
    def test_no_filters_returns_all(self):
        result = select_features()
        assert len(result) == len(REGISTRY)

    def test_filter_by_category(self):
        electoral = select_features(category="electoral")
        demographic = select_features(category="demographic")
        religion = select_features(category="religion")
        # Each should be non-empty
        assert len(electoral) > 0
        assert len(demographic) > 0
        assert len(religion) > 0
        # Together they should equal the full registry
        assert len(electoral) + len(demographic) + len(religion) == len(REGISTRY)

    def test_filter_by_subcategory(self):
        shifts = select_features(subcategory="presidential_shifts")
        assert len(shifts) > 0
        # All should be electoral
        for name in shifts:
            spec = next(s for s in REGISTRY if s.name == name)
            assert spec.subcategory == "presidential_shifts"

    def test_filter_by_category_and_subcategory(self):
        result = select_features(category="electoral", subcategory="presidential_lean")
        assert len(result) > 0
        for name in result:
            spec = next(s for s in REGISTRY if s.name == name)
            assert spec.category == "electoral"
            assert spec.subcategory == "presidential_lean"

    def test_exclude_year_2024(self):
        all_features = select_features()
        no_2024 = select_features(exclude_year=2024)
        # Should exclude some features
        assert len(no_2024) < len(all_features)
        # No feature in result should have source_year=2024
        for name in no_2024:
            spec = next(s for s in REGISTRY if s.name == name)
            assert spec.source_year != 2024

    def test_exclude_year_preserves_none_year(self):
        """Features with source_year=None should survive any exclude_year filter."""
        no_2024 = select_features(exclude_year=2024)
        none_year_features = {s.name for s in REGISTRY if s.source_year is None}
        for name in none_year_features:
            assert name in no_2024

    def test_exclude_year_none_does_nothing(self):
        all_features = select_features()
        result = select_features(exclude_year=None)
        assert result == all_features

    def test_unknown_category_returns_empty(self):
        result = select_features(category="nonexistent")
        assert result == []

    def test_unknown_subcategory_returns_empty(self):
        result = select_features(subcategory="fake_subcategory")
        assert result == []

    def test_returns_list_of_strings(self):
        result = select_features(category="electoral")
        assert all(isinstance(x, str) for x in result)

    def test_combined_filters(self):
        """category + subcategory + exclude_year together."""
        result = select_features(
            category="electoral",
            subcategory="presidential_shifts",
            exclude_year=2024,
        )
        for name in result:
            spec = next(s for s in REGISTRY if s.name == name)
            assert spec.category == "electoral"
            assert spec.subcategory == "presidential_shifts"
            assert spec.source_year != 2024
