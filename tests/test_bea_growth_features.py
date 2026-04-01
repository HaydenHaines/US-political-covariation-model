"""Tests for fetch_bea_state_gdp_growth.py.

All tests use synthetic in-memory DataFrames — no network access, no BEA_API_KEY,
no real disk parquet files required. The module's public API is injected via
_gdp_df / _income_df parameters on build_county_bea_growth_features().

Coverage:
1. _parse_data_value()        — BEA suppression codes → NaN, numeric strings → float
2. compute_growth_rates()     — 1yr and 2yr growth from synthetic series
3. compute_growth_rates()     — state-name → FIPS prefix mapping
4. compute_growth_rates()     — unmapped names (e.g., "United States") are skipped
5. compute_growth_rates()     — latest-usable-year selection skips sparse years
6. compute_growth_rates()     — NaN in denominator → NaN growth (not a divide-by-zero)
7. build_county_bea_growth_features() — FIPS prefix matching maps state → counties
8. build_county_bea_growth_features() — missing state filled with national median
9. build_county_bea_growth_features() — no NaN in output when median fill applies
10. build_county_bea_growth_features() — output schema (columns, row count, dtypes)
11. build_county_bea_growth_features() — FIPS zero-padding (5-char strings)
12. build_county_bea_growth_features() — duplicate FIPS preserved
13. build_county_bea_growth_features() — empty county list returns empty DataFrame
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.assembly.fetch_bea_state_gdp_growth import (
    COL_GDP_GROWTH_1YR,
    COL_GDP_GROWTH_2YR,
    COL_INCOME_GROWTH_1YR,
    FEATURE_COLS,
    _STATE_NAME_TO_FIPS_PREFIX,
    _find_latest_usable_year,
    _parse_data_value,
    build_county_bea_growth_features,
    compute_growth_rates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_state_series(
    name_to_values: dict[str, dict[int, float | None]],
    geo_fips_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic BEA tidy DataFrame for testing.

    Parameters
    ----------
    name_to_values:
        {state_name: {year: value_or_None}}
    geo_fips_map:
        Optional {state_name: geo_fips}.  Not used by compute_growth_rates
        (which uses GeoName→FIPS), but included for completeness.

    Returns a tidy DataFrame: GeoFips, GeoName, TimePeriod (int), DataValue (float)
    """
    rows = []
    for name, year_values in name_to_values.items():
        fips = (geo_fips_map or {}).get(name, "XX000")
        for year, val in year_values.items():
            rows.append({
                "GeoFips": fips,
                "GeoName": name,
                "TimePeriod": year,
                "DataValue": float("nan") if val is None else float(val),
            })
    return pd.DataFrame(rows, columns=["GeoFips", "GeoName", "TimePeriod", "DataValue"])


@pytest.fixture
def three_state_gdp():
    """Synthetic GDP for Florida, Georgia, Alabama over 4 years.

    FL: 1000 → 1050 → 1100 → 1155   (+5% each year)
    GA:  500 →  510 →  520 →  530    (+1.92% last year)
    AL:  200 →  210 →  225 →  234    (+4% last year, distinct from FL's 5%)

    Chosen so that all three 1-year growth rates are distinct:
      FL: (1155-1100)/1100 ≈ 5.00%
      GA: (530-520)/520    ≈ 1.92%
      AL: (234-225)/225    ≈ 4.00%
    """
    return _make_state_series(
        {
            "Florida": {2019: 1000.0, 2020: 1050.0, 2021: 1100.0, 2022: 1155.0},
            "Georgia": {2019: 500.0,  2020: 510.0,  2021: 520.0,  2022: 530.0},
            "Alabama": {2019: 200.0,  2020: 210.0,  2021: 225.0,  2022: 234.0},
        }
    )


@pytest.fixture
def three_state_income():
    """Synthetic per-capita income for FL, GA, AL over 4 years.

    FL: 60000 → 62000 → 64000 → 66000
    GA: 55000 → 56000 → 57000 → 58000
    AL: 48000 → 49000 → 50000 → 51000
    """
    return _make_state_series(
        {
            "Florida": {2019: 60000.0, 2020: 62000.0, 2021: 64000.0, 2022: 66000.0},
            "Georgia": {2019: 55000.0, 2020: 56000.0, 2021: 57000.0, 2022: 58000.0},
            "Alabama": {2019: 48000.0, 2020: 49000.0, 2021: 50000.0, 2022: 51000.0},
        }
    )


# ── 1. _parse_data_value ─────────────────────────────────────────────────────


class TestParseDataValue:
    def test_numeric_string_returns_float(self):
        """Numeric string '1234.5' parses to 1234.5."""
        assert _parse_data_value("1234.5") == pytest.approx(1234.5)

    def test_comma_formatted_number(self):
        """Comma-formatted numbers like '1,234,567' parse correctly."""
        assert _parse_data_value("1,234,567") == pytest.approx(1234567.0)

    def test_suppression_code_D(self):
        """(D) suppression code returns NaN."""
        assert math.isnan(_parse_data_value("(D)"))

    def test_suppression_code_NA(self):
        """(NA) suppression code returns NaN."""
        assert math.isnan(_parse_data_value("(NA)"))

    def test_suppression_code_L(self):
        """(L) suppression code returns NaN."""
        assert math.isnan(_parse_data_value("(L)"))

    def test_suppression_code_dash(self):
        """Double-dash suppression code returns NaN."""
        assert math.isnan(_parse_data_value("--"))

    def test_empty_string_returns_nan(self):
        """Empty string returns NaN."""
        assert math.isnan(_parse_data_value(""))

    def test_pandas_nan_returns_nan(self):
        """pandas NaN passthrough returns NaN."""
        assert math.isnan(_parse_data_value(float("nan")))

    def test_integer_string(self):
        """Integer string '42' parses to 42.0."""
        assert _parse_data_value("42") == pytest.approx(42.0)

    def test_negative_value(self):
        """Negative value '-500.0' parses to -500.0."""
        assert _parse_data_value("-500.0") == pytest.approx(-500.0)


# ── 2. compute_growth_rates — known inputs ───────────────────────────────────


class TestComputeGrowthRates:
    def test_1yr_growth_correct_formula(self, three_state_gdp):
        """1-year growth = (GDP_t - GDP_{t-1}) / GDP_{t-1}."""
        # Florida: 2022=1155, 2021=1100 → growth = (1155-1100)/1100 = 0.05
        result = compute_growth_rates(three_state_gdp)
        fl = result[result["fips_prefix"] == "12"]
        assert len(fl) == 1
        assert fl["growth_1yr"].iloc[0] == pytest.approx(0.05, rel=1e-4)

    def test_2yr_growth_correct_formula(self, three_state_gdp):
        """2-year CAGR = (GDP_t / GDP_{t-2})^0.5 - 1."""
        # Florida: 2022=1155, 2020=1050 → CAGR = (1155/1050)^0.5 - 1 ≈ 0.04988
        result = compute_growth_rates(three_state_gdp)
        fl = result[result["fips_prefix"] == "12"]
        expected = (1155.0 / 1050.0) ** 0.5 - 1.0
        assert fl["growth_2yr"].iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_georgia_1yr_growth(self, three_state_gdp):
        """Georgia 1-year growth: (530-520)/520 ≈ 0.01923."""
        result = compute_growth_rates(three_state_gdp)
        ga = result[result["fips_prefix"] == "13"]
        expected = (530.0 - 520.0) / 520.0
        assert ga["growth_1yr"].iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_alabama_present_in_results(self, three_state_gdp):
        """Alabama (FIPS prefix '01') has a row in the result."""
        result = compute_growth_rates(three_state_gdp)
        assert "01" in result["fips_prefix"].values

    def test_negative_growth_handled(self):
        """Negative GDP change produces a negative growth rate."""
        df = _make_state_series({
            "Florida": {2020: 1000.0, 2021: 900.0, 2022: 950.0},
        })
        result = compute_growth_rates(df)
        fl = result[result["fips_prefix"] == "12"]
        # 1yr: (950-900)/900 ≈ +0.0556 (recovered from dip)
        # 2yr: (950/1000)^0.5 - 1 ≈ -0.0253
        assert fl["growth_1yr"].iloc[0] == pytest.approx((950 - 900) / 900, rel=1e-4)
        assert fl["growth_2yr"].iloc[0] == pytest.approx((950 / 1000) ** 0.5 - 1, rel=1e-4)

    def test_zero_denominator_produces_nan(self):
        """A state with GDP=0 in t-1 produces NaN 1-year growth (no ZeroDivisionError)."""
        df = _make_state_series({
            "Florida": {2020: 0.0, 2021: 0.0, 2022: 100.0},
        })
        result = compute_growth_rates(df)
        fl = result[result["fips_prefix"] == "12"]
        # t=2022, t-1=2021 (value=0) → growth_1yr is NaN
        assert math.isnan(fl["growth_1yr"].iloc[0])

    def test_nan_denominator_produces_nan(self):
        """A state with NaN value in t-1 produces NaN growth (not an error).

        We provide 4 years so the MIN_YEARS_REQUIRED=3 check passes after the NaN
        row (2021) is dropped by dropna. The pivot still has 3 columns: 2020, 2022,
        2023. Latest usable year = 2023, t-1 = 2022, t-2 = 2021 (absent → NaN col).
        So growth_1yr = (1150-1100)/1100 = 0.0455, growth_2yr = NaN.
        """
        df = _make_state_series({
            "Florida": {2020: 1000.0, 2021: None, 2022: 1100.0, 2023: 1150.0},
        })
        result = compute_growth_rates(df)
        fl = result[result["fips_prefix"] == "12"]
        # t=2023, t-1=2022, t-2=2021 (dropped — absent from pivot)
        # growth_1yr: (1150-1100)/1100 ≈ 0.0455 — computable
        # growth_2yr: needs t-2=2021, which is absent → NaN
        assert not math.isnan(fl["growth_1yr"].iloc[0])
        assert math.isnan(fl["growth_2yr"].iloc[0])


# ── 3. State-name to FIPS prefix mapping ─────────────────────────────────────


class TestStateFIPSMapping:
    def test_florida_maps_to_12(self, three_state_gdp):
        """GeoName 'Florida' produces fips_prefix '12'."""
        result = compute_growth_rates(three_state_gdp)
        assert "12" in result["fips_prefix"].values

    def test_georgia_maps_to_13(self, three_state_gdp):
        """GeoName 'Georgia' produces fips_prefix '13'."""
        result = compute_growth_rates(three_state_gdp)
        assert "13" in result["fips_prefix"].values

    def test_district_of_columbia_maps_to_11(self):
        """'District of Columbia' maps to FIPS prefix '11'."""
        df = _make_state_series({
            "District of Columbia": {2020: 100.0, 2021: 110.0, 2022: 115.0},
        })
        result = compute_growth_rates(df)
        assert "11" in result["fips_prefix"].values

    def test_custom_mapping_override(self):
        """state_name_to_fips override is respected."""
        df = _make_state_series({
            "TestState": {2020: 100.0, 2021: 105.0, 2022: 110.0},
        })
        custom_map = {"TestState": "99"}
        result = compute_growth_rates(df, state_name_to_fips=custom_map)
        assert "99" in result["fips_prefix"].values

    def test_mapping_covers_all_51_states(self):
        """The built-in mapping covers all 50 states + DC = 51 entries."""
        assert len(_STATE_NAME_TO_FIPS_PREFIX) == 51


# ── 4. Unmapped names are silently skipped ───────────────────────────────────


class TestUnmappedNamesSkipped:
    def test_united_states_aggregate_excluded(self):
        """'United States' aggregate row is skipped — not mapped to any county FIPS."""
        df = _make_state_series({
            "United States": {2020: 20000.0, 2021: 21000.0, 2022: 22000.0},
            "Florida": {2020: 1000.0, 2021: 1050.0, 2022: 1100.0},
        })
        result = compute_growth_rates(df)
        # Only Florida should appear in the result
        assert len(result) == 1
        assert "12" in result["fips_prefix"].values

    def test_territory_excluded(self):
        """Non-state territories not in the FIPS map are excluded without error."""
        df = _make_state_series({
            "Puerto Rico": {2020: 500.0, 2021: 520.0, 2022: 540.0},
            "Georgia": {2020: 500.0,  2021: 510.0,  2022: 520.0},
        })
        result = compute_growth_rates(df)
        assert "13" in result["fips_prefix"].values
        # Puerto Rico is not in the standard mapping — should be absent
        assert len(result) == 1


# ── 5. Latest-usable-year selection ─────────────────────────────────────────


class TestFindLatestUsableYear:
    def test_fully_populated_returns_latest(self):
        """When all states have data in the latest year, it's returned."""
        pivot = pd.DataFrame(
            {2020: [100.0] * 51, 2021: [110.0] * 51, 2022: [120.0] * 51},
            index=[str(i).zfill(2) for i in range(1, 52)],
        )
        assert _find_latest_usable_year(pivot, [2020, 2021, 2022]) == 2022

    def test_sparse_latest_year_skipped(self):
        """If the latest year has <50% state coverage, it's skipped."""
        # 51 states but only 5 have data in 2023
        data_2022 = [100.0] * 51
        data_2023 = [110.0] * 5 + [float("nan")] * 46
        pivot = pd.DataFrame(
            {2022: data_2022, 2023: data_2023},
            index=[str(i).zfill(2) for i in range(1, 52)],
        )
        result = _find_latest_usable_year(pivot, [2022, 2023])
        assert result == 2022

    def test_all_years_sparse_returns_something(self):
        """Even with sparse data everywhere, a year is returned without crashing."""
        data = [100.0] * 10 + [float("nan")] * 41
        pivot = pd.DataFrame(
            {2020: data, 2021: data},
            index=[str(i).zfill(2) for i in range(1, 52)],
        )
        result = _find_latest_usable_year(pivot, [2020, 2021])
        assert result in (2020, 2021)


# ── 6. build_county_bea_growth_features — FIPS prefix mapping ────────────────


class TestBuildCountyBeaGrowthFeatures:
    def test_florida_counties_get_florida_gdp_growth(
        self, three_state_gdp, three_state_income
    ):
        """Counties with FIPS prefix '12' inherit Florida's GDP growth rate."""
        result = build_county_bea_growth_features(
            ["12001", "12003"],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        fl_growth = result[result["county_fips"] == "12001"][COL_GDP_GROWTH_1YR].iloc[0]
        expected = (1155.0 - 1100.0) / 1100.0
        assert fl_growth == pytest.approx(expected, rel=1e-4)

    def test_georgia_counties_get_georgia_growth(
        self, three_state_gdp, three_state_income
    ):
        """Counties with FIPS prefix '13' inherit Georgia's growth rate."""
        result = build_county_bea_growth_features(
            ["13001"],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        ga_growth = result.iloc[0][COL_GDP_GROWTH_1YR]
        expected = (530.0 - 520.0) / 520.0
        assert ga_growth == pytest.approx(expected, rel=1e-4)

    def test_different_states_get_different_values(
        self, three_state_gdp, three_state_income
    ):
        """Counties in different states receive different growth values."""
        result = build_county_bea_growth_features(
            ["12001", "13001", "01001"],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        assert result[COL_GDP_GROWTH_1YR].nunique() == 3

    def test_income_growth_mapped_correctly(
        self, three_state_gdp, three_state_income
    ):
        """Income growth (bea_income_growth_1yr) maps correctly to counties."""
        result = build_county_bea_growth_features(
            ["12001"],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        # Florida income: 2022=66000, 2021=64000 → (66000-64000)/64000 = 0.03125
        expected = (66000.0 - 64000.0) / 64000.0
        assert result.iloc[0][COL_INCOME_GROWTH_1YR] == pytest.approx(expected, rel=1e-4)


# ── 7. Missing state median fill ─────────────────────────────────────────────


class TestMissingStateFill:
    def test_unknown_state_filled_with_median(
        self, three_state_gdp, three_state_income
    ):
        """County in a state not present in BEA data is filled with national median."""
        # three_state_gdp has FL, GA, AL — but not New York (prefix 36).
        # The median of the 3 present states is used to fill NY counties.
        result = build_county_bea_growth_features(
            ["12001", "36001"],  # FL + NY (NY not in fixture)
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        fl_val = result[result["county_fips"] == "12001"][COL_GDP_GROWTH_1YR].iloc[0]
        ny_val = result[result["county_fips"] == "36001"][COL_GDP_GROWTH_1YR].iloc[0]

        # NY should not be NaN — it gets the median fill
        assert not math.isnan(ny_val)
        # NY gets a real number, but it won't equal FL's rate
        # (median of 3 states is the middle one, which isn't FL unless FL is median)
        assert isinstance(ny_val, float)

    def test_no_nan_in_output_with_unknown_state(
        self, three_state_gdp, three_state_income
    ):
        """Output is NaN-free even when an unknown state FIPS is included."""
        result = build_county_bea_growth_features(
            ["12001", "99001"],  # 99 is not a valid state FIPS
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        for col in FEATURE_COLS:
            assert not result[col].isna().any(), f"Found NaN in {col}"

    def test_median_fill_uses_present_state_values(
        self, three_state_income
    ):
        """Median fill is computed from the present states, not a hardcoded value."""
        # Build GDP with just two states so median is their average
        gdp_df = _make_state_series({
            "Florida": {2020: 1000.0, 2021: 1100.0, 2022: 1200.0},  # +9.09% last year
            "Georgia": {2020: 500.0,  2021: 500.0,  2022: 600.0},   # +20% last year
        })
        # fl_growth_1yr = (1200-1100)/1100 ≈ 0.0909
        # ga_growth_1yr = (600-500)/500 = 0.20
        # median of [0.0909, 0.20] = 0.1455 (mean of the two since n=2)
        result = build_county_bea_growth_features(
            ["12001", "13001", "01001"],  # AL not in gdp_df
            _gdp_df=gdp_df,
            _income_df=three_state_income,
        )
        fl_val = result[result["county_fips"] == "12001"][COL_GDP_GROWTH_1YR].iloc[0]
        ga_val = result[result["county_fips"] == "13001"][COL_GDP_GROWTH_1YR].iloc[0]
        al_val = result[result["county_fips"] == "01001"][COL_GDP_GROWTH_1YR].iloc[0]

        expected_median = (fl_val + ga_val) / 2  # n=2: median = mean
        assert al_val == pytest.approx(expected_median, rel=1e-4)


# ── 8. Output schema ──────────────────────────────────────────────────────────


class TestOutputSchema:
    def test_output_columns_exact(self, three_state_gdp, three_state_income):
        """Output has exactly county_fips + three feature columns."""
        result = build_county_bea_growth_features(
            ["12001"],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        assert set(result.columns) == {"county_fips"} | set(FEATURE_COLS)

    def test_feature_cols_constant_includes_all_three(self):
        """FEATURE_COLS contains exactly the three documented column names."""
        assert COL_GDP_GROWTH_1YR in FEATURE_COLS
        assert COL_GDP_GROWTH_2YR in FEATURE_COLS
        assert COL_INCOME_GROWTH_1YR in FEATURE_COLS
        assert len(FEATURE_COLS) == 3

    def test_row_count_matches_input(self, three_state_gdp, three_state_income):
        """Output has exactly one row per input FIPS entry."""
        fips = ["12001", "12003", "13001", "01001"]
        result = build_county_bea_growth_features(
            fips,
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        assert len(result) == len(fips)

    def test_county_fips_preserved_in_output(self, three_state_gdp, three_state_income):
        """county_fips values in output match input order."""
        fips = ["12001", "13001", "01001"]
        result = build_county_bea_growth_features(
            fips,
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        assert list(result["county_fips"]) == fips

    def test_growth_columns_are_float(self, three_state_gdp, three_state_income):
        """All growth feature columns have float dtype."""
        result = build_county_bea_growth_features(
            ["12001"],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        for col in FEATURE_COLS:
            assert result[col].dtype.kind == "f", f"{col} is not float dtype"

    def test_fips_zero_padded_to_five_chars(self, three_state_gdp, three_state_income):
        """county_fips in output are always 5-char zero-padded strings."""
        result = build_county_bea_growth_features(
            ["1001"],  # Alabama without leading zero
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        assert result["county_fips"].iloc[0] == "01001"
        assert result["county_fips"].str.len().eq(5).all()

    def test_duplicate_fips_preserved(self, three_state_gdp, three_state_income):
        """Duplicate county FIPS in input are preserved in output."""
        result = build_county_bea_growth_features(
            ["12001", "12001"],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        assert len(result) == 2

    def test_empty_county_list(self, three_state_gdp, three_state_income):
        """Empty county list returns empty DataFrame with correct columns."""
        result = build_county_bea_growth_features(
            [],
            _gdp_df=three_state_gdp,
            _income_df=three_state_income,
        )
        assert len(result) == 0
        assert set(result.columns) == {"county_fips"} | set(FEATURE_COLS)
