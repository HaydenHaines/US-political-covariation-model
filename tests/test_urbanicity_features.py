"""Tests for src/assembly/build_urbanicity_features.py.

Coverage:
1. log_pop_density computation with known values
2. Filtering to FL/GA/AL only (non-target states excluded)
3. Zero land area handling — should produce NaN, not -inf
4. Output schema: expected columns, correct dtypes, no spurious NaN
5. land_area_sq_mi unit conversion from ALAND (sq meters)
6. pop_per_sq_mi arithmetic
7. Path constants point to expected directories
8. Empty inputs return empty output
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_urbanicity_features import (
    OUTPUT_PATH,
    GAZETTEER_CACHE,
    SQ_M_PER_SQ_MI,
    TARGET_FIPS_PREFIXES,
    build_urbanicity_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_acs() -> pd.DataFrame:
    """Minimal ACS DataFrame covering FL, GA, AL, and one out-of-scope county."""
    return pd.DataFrame(
        {
            "county_fips": ["12001", "13001", "01001", "06037"],
            "pop_total": [100_000, 50_000, 20_000, 10_000_000],
        }
    )


@pytest.fixture
def minimal_gazetteer() -> pd.DataFrame:
    """Minimal gazetteer with plausible ALAND values (sq meters).

    12001 FL Alachua   ~ 873 sq mi  = 2,261,000,000 sq m  (urban-ish)
    13001 GA Appling   ~ 508 sq mi  = 1,315,000,000 sq m  (rural)
    01001 AL Autauga   ~ 604 sq mi  = 1,564,000,000 sq m
    06037 CA Los Angeles — out-of-scope, should be excluded
    """
    return pd.DataFrame(
        {
            "county_fips": ["12001", "13001", "01001", "06037"],
            "aland_sq_m": [
                2_261_000_000.0,
                1_315_000_000.0,
                1_564_000_000.0,
                10_510_000_000.0,
            ],
        }
    )


# ---------------------------------------------------------------------------
# 1. log_pop_density computation with known values
# ---------------------------------------------------------------------------


class TestLogPopDensityComputation:
    def test_known_value(self):
        """Verify log10(pop/area) matches manual calculation."""
        # 100,000 people, 100 sq mi → density = 1,000 → log10 = 3.0
        aland_sq_m = 100 * SQ_M_PER_SQ_MI  # 100 sq mi in sq meters
        acs = pd.DataFrame({"county_fips": ["12001"], "pop_total": [100_000]})
        gaz = pd.DataFrame({"county_fips": ["12001"], "aland_sq_m": [aland_sq_m]})
        result = build_urbanicity_features(acs, gaz)
        assert len(result) == 1
        assert result.iloc[0]["log_pop_density"] == pytest.approx(3.0, abs=1e-6)

    def test_sparse_county(self):
        """Low-density county: 1,000 people, 1,000 sq mi → density=1 → log10=0."""
        aland_sq_m = 1_000 * SQ_M_PER_SQ_MI
        acs = pd.DataFrame({"county_fips": ["12003"], "pop_total": [1_000]})
        gaz = pd.DataFrame({"county_fips": ["12003"], "aland_sq_m": [aland_sq_m]})
        result = build_urbanicity_features(acs, gaz)
        assert result.iloc[0]["log_pop_density"] == pytest.approx(0.0, abs=1e-6)

    def test_dense_urban_county(self):
        """Dense county: 1,000,000 people, 50 sq mi → density=20,000 → log10≈4.301."""
        aland_sq_m = 50 * SQ_M_PER_SQ_MI
        acs = pd.DataFrame({"county_fips": ["12005"], "pop_total": [1_000_000]})
        gaz = pd.DataFrame({"county_fips": ["12005"], "aland_sq_m": [aland_sq_m]})
        result = build_urbanicity_features(acs, gaz)
        expected = math.log10(1_000_000 / 50)
        assert result.iloc[0]["log_pop_density"] == pytest.approx(expected, abs=1e-5)

    def test_ordering_urban_gt_rural(self, minimal_acs, minimal_gazetteer):
        """Urban county (higher pop/area) has higher log_pop_density."""
        # 12001: 100,000 / ~873 sq mi ≈ 115 pop/sq mi
        # 13001:  50,000 / ~508 sq mi ≈  98 pop/sq mi
        result = build_urbanicity_features(minimal_acs, minimal_gazetteer)
        fl_density = result.loc[result["county_fips"] == "12001", "log_pop_density"].iloc[0]
        ga_density = result.loc[result["county_fips"] == "13001", "log_pop_density"].iloc[0]
        assert fl_density > ga_density


# ---------------------------------------------------------------------------
# 2. Filtering to FL/GA/AL only
# ---------------------------------------------------------------------------


class TestStateFiltering:
    def test_out_of_scope_county_excluded(self, minimal_acs, minimal_gazetteer):
        """California county (06037) must not appear in output."""
        result = build_urbanicity_features(minimal_acs, minimal_gazetteer)
        assert "06037" not in result["county_fips"].values

    def test_only_target_states_present(self, minimal_acs, minimal_gazetteer):
        """All output county_fips must have a FL/GA/AL FIPS prefix."""
        result = build_urbanicity_features(minimal_acs, minimal_gazetteer)
        prefixes = result["county_fips"].str[:2].unique()
        assert set(prefixes).issubset(TARGET_FIPS_PREFIXES)

    def test_all_three_states_represented(self, minimal_acs, minimal_gazetteer):
        """FL, GA, and AL counties all appear in output."""
        result = build_urbanicity_features(minimal_acs, minimal_gazetteer)
        prefixes = set(result["county_fips"].str[:2].tolist())
        assert "12" in prefixes  # FL
        assert "13" in prefixes  # GA
        assert "01" in prefixes  # AL

    def test_target_fips_prefixes_constant(self):
        """TARGET_FIPS_PREFIXES covers exactly FL, GA, AL."""
        assert TARGET_FIPS_PREFIXES == {"01", "12", "13"}


# ---------------------------------------------------------------------------
# 3. Zero land area handling
# ---------------------------------------------------------------------------


class TestZeroLandArea:
    def test_zero_aland_gives_nan_not_neg_inf(self):
        """A county with ALAND=0 should produce NaN log_pop_density, not -inf."""
        acs = pd.DataFrame({"county_fips": ["12001"], "pop_total": [50_000]})
        gaz = pd.DataFrame({"county_fips": ["12001"], "aland_sq_m": [0.0]})
        result = build_urbanicity_features(acs, gaz)
        val = result.iloc[0]["log_pop_density"]
        assert math.isnan(val), f"Expected NaN for zero land area, got {val}"

    def test_zero_aland_land_area_is_nan(self):
        """land_area_sq_mi should also be NaN when ALAND=0."""
        acs = pd.DataFrame({"county_fips": ["12001"], "pop_total": [50_000]})
        gaz = pd.DataFrame({"county_fips": ["12001"], "aland_sq_m": [0.0]})
        result = build_urbanicity_features(acs, gaz)
        assert math.isnan(result.iloc[0]["land_area_sq_mi"])

    def test_zero_aland_does_not_affect_valid_rows(self):
        """When one county has zero area, other counties are still computed correctly."""
        acs = pd.DataFrame(
            {"county_fips": ["12001", "12003"], "pop_total": [50_000, 100_000]}
        )
        area_sq_m = 100 * SQ_M_PER_SQ_MI
        gaz = pd.DataFrame(
            {"county_fips": ["12001", "12003"], "aland_sq_m": [0.0, area_sq_m]}
        )
        result = build_urbanicity_features(acs, gaz)
        valid_row = result[result["county_fips"] == "12003"].iloc[0]
        assert not math.isnan(valid_row["log_pop_density"])
        assert valid_row["log_pop_density"] == pytest.approx(math.log10(1000.0), abs=1e-5)


# ---------------------------------------------------------------------------
# 4. Output schema: expected columns, dtypes, no NaN for normal data
# ---------------------------------------------------------------------------


class TestOutputSchema:
    EXPECTED_COLS = {"county_fips", "log_pop_density", "land_area_sq_mi", "pop_per_sq_mi"}

    def test_expected_columns_present(self, minimal_acs, minimal_gazetteer):
        """Output DataFrame has exactly the expected columns."""
        result = build_urbanicity_features(minimal_acs, minimal_gazetteer)
        assert set(result.columns) == self.EXPECTED_COLS

    def test_county_fips_is_string(self, minimal_acs, minimal_gazetteer):
        """county_fips column is string dtype."""
        result = build_urbanicity_features(minimal_acs, minimal_gazetteer)
        assert result["county_fips"].dtype == object

    def test_numeric_columns_are_float(self, minimal_acs, minimal_gazetteer):
        """log_pop_density, land_area_sq_mi, pop_per_sq_mi are float dtype."""
        result = build_urbanicity_features(minimal_acs, minimal_gazetteer)
        for col in ["log_pop_density", "land_area_sq_mi", "pop_per_sq_mi"]:
            assert result[col].dtype.kind == "f", f"{col} should be float"

    def test_no_nan_for_valid_input(self):
        """With all valid (nonzero) land areas and pop totals, no NaN in output."""
        acs = pd.DataFrame(
            {
                "county_fips": ["12001", "13001", "01001"],
                "pop_total": [100_000, 50_000, 20_000],
            }
        )
        gaz = pd.DataFrame(
            {
                "county_fips": ["12001", "13001", "01001"],
                "aland_sq_m": [
                    500 * SQ_M_PER_SQ_MI,
                    800 * SQ_M_PER_SQ_MI,
                    600 * SQ_M_PER_SQ_MI,
                ],
            }
        )
        result = build_urbanicity_features(acs, gaz)
        assert result.isnull().sum().sum() == 0, "Expected zero NaN in output"

    def test_county_fips_zero_padded(self):
        """county_fips values are 5-character zero-padded strings."""
        acs = pd.DataFrame({"county_fips": ["01001"], "pop_total": [20_000]})
        gaz = pd.DataFrame({"county_fips": ["01001"], "aland_sq_m": [500 * SQ_M_PER_SQ_MI]})
        result = build_urbanicity_features(acs, gaz)
        assert (result["county_fips"].str.len() == 5).all()


# ---------------------------------------------------------------------------
# 5. land_area_sq_mi unit conversion
# ---------------------------------------------------------------------------


class TestLandAreaConversion:
    def test_one_sq_mi_in_sq_meters(self):
        """ALAND = SQ_M_PER_SQ_MI should yield land_area_sq_mi == 1.0."""
        acs = pd.DataFrame({"county_fips": ["12001"], "pop_total": [1_000]})
        gaz = pd.DataFrame({"county_fips": ["12001"], "aland_sq_m": [SQ_M_PER_SQ_MI]})
        result = build_urbanicity_features(acs, gaz)
        assert result.iloc[0]["land_area_sq_mi"] == pytest.approx(1.0, rel=1e-6)

    def test_conversion_constant(self):
        """SQ_M_PER_SQ_MI is the standard 1 sq mi = 2,589,988.11 sq m."""
        assert SQ_M_PER_SQ_MI == pytest.approx(2_589_988.11, rel=1e-4)

    def test_known_county_area(self):
        """500 sq mi county: land_area_sq_mi == 500.0."""
        aland = 500 * SQ_M_PER_SQ_MI
        acs = pd.DataFrame({"county_fips": ["13001"], "pop_total": [50_000]})
        gaz = pd.DataFrame({"county_fips": ["13001"], "aland_sq_m": [aland]})
        result = build_urbanicity_features(acs, gaz)
        assert result.iloc[0]["land_area_sq_mi"] == pytest.approx(500.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 6. pop_per_sq_mi arithmetic
# ---------------------------------------------------------------------------


class TestPopPerSqMi:
    def test_pop_per_sq_mi_calculation(self):
        """pop_per_sq_mi = pop_total / land_area_sq_mi."""
        aland = 200 * SQ_M_PER_SQ_MI
        acs = pd.DataFrame({"county_fips": ["12001"], "pop_total": [400_000]})
        gaz = pd.DataFrame({"county_fips": ["12001"], "aland_sq_m": [aland]})
        result = build_urbanicity_features(acs, gaz)
        # 400,000 / 200 = 2,000
        assert result.iloc[0]["pop_per_sq_mi"] == pytest.approx(2000.0, rel=1e-5)

    def test_log_pop_density_consistent_with_pop_per_sq_mi(self):
        """log_pop_density == log10(pop_per_sq_mi)."""
        aland = 300 * SQ_M_PER_SQ_MI
        acs = pd.DataFrame({"county_fips": ["01001"], "pop_total": [30_000]})
        gaz = pd.DataFrame({"county_fips": ["01001"], "aland_sq_m": [aland]})
        result = build_urbanicity_features(acs, gaz)
        row = result.iloc[0]
        expected_log = math.log10(row["pop_per_sq_mi"])
        assert row["log_pop_density"] == pytest.approx(expected_log, abs=1e-10)


# ---------------------------------------------------------------------------
# 7. Path constants
# ---------------------------------------------------------------------------


class TestPathConstants:
    def test_output_path_in_assembled(self):
        """OUTPUT_PATH ends in data/assembled/county_urbanicity_features.parquet."""
        assert OUTPUT_PATH.parts[-3:] == (
            "data",
            "assembled",
            "county_urbanicity_features.parquet",
        )

    def test_gazetteer_cache_in_raw(self):
        """GAZETTEER_CACHE ends in data/raw/census_gazetteer_counties.txt."""
        assert GAZETTEER_CACHE.parts[-3:] == (
            "data",
            "raw",
            "census_gazetteer_counties.txt",
        )


# ---------------------------------------------------------------------------
# 8. Empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_acs_returns_empty(self, minimal_gazetteer):
        """Empty ACS DataFrame produces empty output."""
        empty_acs = pd.DataFrame(columns=["county_fips", "pop_total"])
        result = build_urbanicity_features(empty_acs, minimal_gazetteer)
        assert len(result) == 0
        assert set(result.columns) == {
            "county_fips",
            "log_pop_density",
            "land_area_sq_mi",
            "pop_per_sq_mi",
        }

    def test_empty_gazetteer_returns_empty(self, minimal_acs):
        """Empty gazetteer produces empty output (no ALAND data to join)."""
        empty_gaz = pd.DataFrame(columns=["county_fips", "aland_sq_m"])
        result = build_urbanicity_features(minimal_acs, empty_gaz)
        assert len(result) == 0

    def test_no_fl_ga_al_in_acs_returns_empty(self, minimal_gazetteer):
        """ACS with only out-of-scope counties produces empty output."""
        acs = pd.DataFrame(
            {"county_fips": ["06037", "48201"], "pop_total": [10_000_000, 4_000_000]}
        )
        result = build_urbanicity_features(acs, minimal_gazetteer)
        assert len(result) == 0
