"""Tests for the district data pipeline.

Tests cover:
- County-district crosswalk builder (build_district_crosswalk)
- District type composition builder (build_district_types)
- House 2026 race definitions (data/races/house_2026.csv)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_district_crosswalk import (
    AT_LARGE_STATES,
    STATE_FIPS,
    build_crosswalk,
    build_state_crosswalk,
)
from src.assembly.build_district_types import (
    N_TYPES,
    build_district_types,
    load_county_types,
)

PROJECT_ROOT = Path(__file__).parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_crosswalk() -> pd.DataFrame:
    """Minimal county-district crosswalk for testing."""
    return pd.DataFrame({
        "county_fips": ["01001", "01001", "01003", "02001"],
        "state_fips": ["01", "01", "01", "02"],
        "district_id": ["01-02", "01-07", "01-01", "02-00"],
        "overlap_votes": [8000, 2000, 15000, 5000],
        "overlap_fraction": [0.8, 0.2, 1.0, 1.0],
    })


@pytest.fixture
def sample_county_types() -> pd.DataFrame:
    """Minimal county type assignments for testing."""
    n_types = N_TYPES
    data = {
        "county_fips": ["01001", "01003", "02001"],
    }
    # Give each county a distinct type profile
    for j in range(n_types):
        if j == 0:
            data[f"type_{j}"] = [0.5, 0.1, 0.3]
        elif j == 1:
            data[f"type_{j}"] = [0.3, 0.7, 0.2]
        elif j == 2:
            data[f"type_{j}"] = [0.2, 0.2, 0.5]
        else:
            data[f"type_{j}"] = [0.0, 0.0, 0.0]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Crosswalk builder tests
# ---------------------------------------------------------------------------


class TestCrosswalkValidation:
    """Test crosswalk structural properties."""

    def test_overlap_fractions_sum_to_one(self, sample_crosswalk: pd.DataFrame) -> None:
        """Each county's overlap fractions must sum to 1.0."""
        sums = sample_crosswalk.groupby("county_fips")["overlap_fraction"].sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)

    def test_overlap_fractions_non_negative(self, sample_crosswalk: pd.DataFrame) -> None:
        assert (sample_crosswalk["overlap_fraction"] >= 0).all()

    def test_district_id_format(self, sample_crosswalk: pd.DataFrame) -> None:
        """District IDs follow SS-DD format."""
        for did in sample_crosswalk["district_id"]:
            parts = did.split("-")
            assert len(parts) == 2
            assert len(parts[0]) == 2
            assert len(parts[1]) == 2
            assert parts[0].isdigit()
            assert parts[1].isdigit()

    def test_county_fips_five_digits(self, sample_crosswalk: pd.DataFrame) -> None:
        for fips in sample_crosswalk["county_fips"]:
            assert len(fips) == 5
            assert fips.isdigit()

    def test_at_large_states_single_district(self, sample_crosswalk: pd.DataFrame) -> None:
        """At-large states should have district ending in 00."""
        ak_rows = sample_crosswalk[sample_crosswalk["state_fips"] == "02"]
        if len(ak_rows) > 0:
            assert all(ak_rows["district_id"].str.endswith("-00"))
            # Each county maps to exactly one district
            assert all(ak_rows.groupby("county_fips").size() == 1)


# ---------------------------------------------------------------------------
# District type composition tests
# ---------------------------------------------------------------------------


class TestDistrictTypes:
    """Test district type composition builder."""

    def test_type_scores_sum_to_one(
        self,
        sample_crosswalk: pd.DataFrame,
        sample_county_types: pd.DataFrame,
    ) -> None:
        """District type scores must sum to 1.0."""
        result = build_district_types(sample_crosswalk, sample_county_types)
        type_cols = [c for c in result.columns if c.startswith("type_")]
        row_sums = result[type_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_type_scores_non_negative(
        self,
        sample_crosswalk: pd.DataFrame,
        sample_county_types: pd.DataFrame,
    ) -> None:
        result = build_district_types(sample_crosswalk, sample_county_types)
        type_cols = [c for c in result.columns if c.startswith("type_")]
        assert (result[type_cols] >= 0).all().all()

    def test_single_county_district_inherits_types(
        self,
        sample_crosswalk: pd.DataFrame,
        sample_county_types: pd.DataFrame,
    ) -> None:
        """A district containing a single whole county inherits its type profile."""
        result = build_district_types(sample_crosswalk, sample_county_types)
        type_cols = [c for c in result.columns if c.startswith("type_")]

        # District 01-01 contains only county 01003 (overlap=1.0)
        d0101 = result[result["district_id"] == "01-01"][type_cols].iloc[0]
        county_01003 = sample_county_types[
            sample_county_types["county_fips"] == "01003"
        ][type_cols].iloc[0]

        np.testing.assert_allclose(d0101.values, county_01003.values, atol=1e-6)

    def test_split_county_weighted_average(
        self,
        sample_crosswalk: pd.DataFrame,
        sample_county_types: pd.DataFrame,
    ) -> None:
        """A district with a partial county gets weighted type profile."""
        result = build_district_types(sample_crosswalk, sample_county_types)
        type_cols = [c for c in result.columns if c.startswith("type_")]

        # District 01-02 gets 80% of county 01001
        d0102 = result[result["district_id"] == "01-02"][type_cols].iloc[0]
        county_01001 = sample_county_types[
            sample_county_types["county_fips"] == "01001"
        ][type_cols].iloc[0]

        # Since 01-02 only has county 01001 (at 80%), the type profile should
        # match 01001 exactly (after renormalization)
        np.testing.assert_allclose(d0102.values, county_01001.values, atol=1e-6)

    def test_correct_district_count(
        self,
        sample_crosswalk: pd.DataFrame,
        sample_county_types: pd.DataFrame,
    ) -> None:
        result = build_district_types(sample_crosswalk, sample_county_types)
        expected_districts = sample_crosswalk["district_id"].nunique()
        assert len(result) == expected_districts

    def test_multi_county_district_blends_types(self) -> None:
        """A district spanning two equal-weight counties blends their types."""
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
            "district_id": ["01-01", "01-01"],
            "overlap_votes": [5000, 5000],
            "overlap_fraction": [0.5, 0.5],
        })
        # County 01001 is pure type 0, county 01003 is pure type 1
        type_data = {"county_fips": ["01001", "01003"]}
        for j in range(N_TYPES):
            if j == 0:
                type_data[f"type_{j}"] = [1.0, 0.0]
            elif j == 1:
                type_data[f"type_{j}"] = [0.0, 1.0]
            else:
                type_data[f"type_{j}"] = [0.0, 0.0]
        county_types = pd.DataFrame(type_data)

        result = build_district_types(crosswalk, county_types)
        assert len(result) == 1
        # District should be 50/50 type 0 and type 1
        assert abs(result["type_0"].iloc[0] - 0.5) < 1e-6
        assert abs(result["type_1"].iloc[0] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# House 2026 race file tests
# ---------------------------------------------------------------------------


class TestHouse2026Races:
    """Test the house_2026.csv race definition file."""

    @pytest.fixture
    def races(self) -> pd.DataFrame:
        path = PROJECT_ROOT / "data" / "races" / "house_2026.csv"
        if not path.exists():
            pytest.skip("house_2026.csv not found (build with generate script)")
        return pd.read_csv(path, dtype={"district_number": int})

    def test_total_districts(self, races: pd.DataFrame) -> None:
        """Must have exactly 435 House districts."""
        assert len(races) == 435

    def test_unique_race_ids(self, races: pd.DataFrame) -> None:
        assert races["race_id"].is_unique

    def test_unique_district_ids(self, races: pd.DataFrame) -> None:
        assert races["district_id"].is_unique

    def test_all_states_present(self, races: pd.DataFrame) -> None:
        """All 50 states must be represented."""
        assert races["state"].nunique() == 50

    def test_incumbent_party_values(self, races: pd.DataFrame) -> None:
        valid_parties = {"DEM", "REP"}
        assert set(races["incumbent_party"].unique()).issubset(valid_parties)

    def test_district_id_format(self, races: pd.DataFrame) -> None:
        for did in races["district_id"]:
            parts = did.split("-")
            assert len(parts) == 2
            assert len(parts[0]) == 2
            assert len(parts[1]) == 2

    def test_at_large_districts_numbered_00(self, races: pd.DataFrame) -> None:
        """At-large states should have district number 0."""
        at_large = races[races["state"].isin(AT_LARGE_STATES)]
        assert all(at_large["district_number"] == 0)
        assert len(at_large) == len(AT_LARGE_STATES)

    def test_known_district_counts(self, races: pd.DataFrame) -> None:
        """Spot-check known state district counts."""
        known = {
            "CA": 52, "TX": 38, "FL": 28, "NY": 26, "IL": 17,
            "PA": 17, "OH": 15, "NC": 14, "GA": 14, "MI": 13,
            "AK": 1, "DE": 1, "MT": 2, "WY": 1,
        }
        for state, expected in known.items():
            actual = len(races[races["state"] == state])
            assert actual == expected, f"{state}: expected {expected}, got {actual}"

    def test_pvi_populated(self, races: pd.DataFrame) -> None:
        """Most districts should have PVI values."""
        # Allow some missing (8 from incomplete 538 data)
        non_empty = races["pvi"].notna() & (races["pvi"] != "")
        assert non_empty.sum() >= 420

    def test_cook_rating_values(self, races: pd.DataFrame) -> None:
        valid_ratings = {
            "", "Toss-up", "Lean D", "Lean R",
            "Likely D", "Likely R", "Safe D", "Safe R",
        }
        for rating in races["cook_rating"].fillna("").unique():
            assert rating in valid_ratings, f"Unexpected rating: {rating}"


# ---------------------------------------------------------------------------
# Integration: crosswalk + types pipeline
# ---------------------------------------------------------------------------


class TestCrosswalkTypesIntegration:
    """Test the full pipeline from crosswalk to district types."""

    def test_pipeline_preserves_all_districts(
        self,
        sample_crosswalk: pd.DataFrame,
        sample_county_types: pd.DataFrame,
    ) -> None:
        """Every district in the crosswalk gets a type composition."""
        result = build_district_types(sample_crosswalk, sample_county_types)
        expected_districts = set(sample_crosswalk["district_id"])
        actual_districts = set(result["district_id"])
        assert expected_districts == actual_districts

    def test_pipeline_n_type_columns(
        self,
        sample_crosswalk: pd.DataFrame,
        sample_county_types: pd.DataFrame,
    ) -> None:
        result = build_district_types(sample_crosswalk, sample_county_types)
        type_cols = [c for c in result.columns if c.startswith("type_")]
        assert len(type_cols) == N_TYPES

    def test_zero_overlap_fraction_excluded(self) -> None:
        """Counties with 0 overlap fraction don't contribute to district types."""
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
            "district_id": ["01-01", "01-01"],
            "overlap_votes": [10000, 0],
            "overlap_fraction": [1.0, 0.0],
        })
        type_data = {"county_fips": ["01001", "01003"]}
        for j in range(N_TYPES):
            if j == 0:
                type_data[f"type_{j}"] = [1.0, 0.0]
            else:
                type_data[f"type_{j}"] = [0.0, 1.0 if j == 1 else 0.0]
        county_types = pd.DataFrame(type_data)

        result = build_district_types(crosswalk, county_types)
        # Should be pure type 0 (county 01003 has 0 overlap)
        assert abs(result["type_0"].iloc[0] - 1.0) < 1e-6

    def test_unequal_overlap_weights_correctly(self) -> None:
        """A district with 75/25 county overlap produces correct weighting."""
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
            "district_id": ["01-01", "01-01"],
            "overlap_votes": [7500, 2500],
            "overlap_fraction": [0.75, 0.25],
        })
        type_data = {"county_fips": ["01001", "01003"]}
        for j in range(N_TYPES):
            if j == 0:
                type_data[f"type_{j}"] = [1.0, 0.0]
            elif j == 1:
                type_data[f"type_{j}"] = [0.0, 1.0]
            else:
                type_data[f"type_{j}"] = [0.0, 0.0]
        county_types = pd.DataFrame(type_data)

        result = build_district_types(crosswalk, county_types)
        # 75% of county_01001 (pure type_0) + 25% of county_01003 (pure type_1)
        assert abs(result["type_0"].iloc[0] - 0.75) < 1e-6
        assert abs(result["type_1"].iloc[0] - 0.25) < 1e-6

    def test_missing_county_in_types_handled(self) -> None:
        """Counties in crosswalk but missing from type assignments are skipped."""
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "99999"],
            "state_fips": ["01", "99"],
            "district_id": ["01-01", "99-01"],
            "overlap_votes": [10000, 5000],
            "overlap_fraction": [1.0, 1.0],
        })
        type_data = {"county_fips": ["01001"]}
        for j in range(N_TYPES):
            type_data[f"type_{j}"] = [1.0 if j == 0 else 0.0]
        county_types = pd.DataFrame(type_data)

        result = build_district_types(crosswalk, county_types)
        # Only district 01-01 should appear (99-01's county has no types)
        assert len(result) == 1
        assert result["district_id"].iloc[0] == "01-01"


class TestStateConstants:
    """Test that state constants are internally consistent."""

    def test_state_fips_bijectivity(self) -> None:
        """FIPS-to-abbr and abbr-to-FIPS must be inverse mappings."""
        from src.assembly.build_district_crosswalk import STATE_FIPS, STATE_ABBR_TO_FIPS
        for fips, abbr in STATE_FIPS.items():
            assert STATE_ABBR_TO_FIPS[abbr] == fips

    def test_fifty_states_plus_dc(self) -> None:
        from src.assembly.build_district_crosswalk import STATE_FIPS
        assert len(STATE_FIPS) == 51  # 50 states + DC

    def test_at_large_states_valid(self) -> None:
        from src.assembly.build_district_crosswalk import AT_LARGE_STATES, STATE_FIPS
        for st in AT_LARGE_STATES:
            assert st in STATE_FIPS.values(), f"{st} not in STATE_FIPS"
