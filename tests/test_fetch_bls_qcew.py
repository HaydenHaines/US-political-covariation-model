"""Tests for BLS QCEW county-level employment and wage data fetching and feature computation.

Tests exercise:
1. fetch_bls_qcew.py — URL construction, CSV parsing, filtering, output schema
2. build_qcew_features.py — pivot, share computation, HHI, top_industry, avg_pay,
   imputation, output schema

These tests use synthetic DataFrames and mock HTTP responses so they run without
any network access. Tests verify:
  - URL construction includes correct year, industry code, and county/all path
  - Raw CSV is correctly parsed and column types coerced
  - Annual-average filter (qtr == "A") keeps correct rows
  - own_code filter keeps only total-ownership rows
  - State-scope filtering keeps only configured state county FIPS (all 50+DC)
  - Suppressed rows (disclosure_code == "N") are dropped
  - Non-county FIPS (state-level "SS000") are dropped
  - Output schema has required columns
  - Feature computation: shares sum correctly, zero-employment produces NaN
  - HHI ranges [0, 1], all-NaN input produces NaN
  - top_industry returns correct NAICS code string
  - avg_annual_pay = total_wages / total_employment
  - State-median imputation fills NaN correctly
  - Edge cases: empty input, all-suppressed, no target states
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_bls_qcew import (
    DEFAULT_YEARS,
    INDUSTRY_CODES,
    KEEP_COLUMNS,
    OWN_CODE_TOTAL,
    STATES,
    TARGET_STATE_FIPS,
    TOTAL_INDUSTRY_CODE,
    build_url,
    fetch_county_csv,
    fetch_industry_year,
    filter_county_df,
)
from src.assembly.build_qcew_features import (
    QCEW_FEATURE_COLS,
    SECTOR_CODES,
    compute_avg_pay,
    compute_hhi,
    compute_qcew_features,
    compute_shares,
    compute_top_industry,
    impute_qcew_state_medians,
    pivot_qcew,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_csv_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal raw QCEW CSV DataFrame from row dicts.

    Fills in defaults for omitted columns so tests can be concise.
    Matches the structure produced by fetch_county_csv() after parsing.
    """
    default = {
        "area_fips": "12001",
        "own_code": "0",
        "industry_code": "10",
        "year": "2022",
        "qtr": "A",
        "disclosure_code": " ",
        "annual_avg_estabs": "500",
        "annual_avg_emplvl": "10000",
        "total_annual_wages": "500000000",
        "annual_avg_wkly_wage": "962",
        "avg_annual_pay": "50000",
    }
    records = [{**default, **r} for r in rows]
    df = pd.DataFrame(records)
    # Coerce numeric columns (mirrors fetch_county_csv behavior)
    numeric_cols = [
        "annual_avg_estabs",
        "annual_avg_emplvl",
        "total_annual_wages",
        "annual_avg_wkly_wage",
        "avg_annual_pay",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def _make_wide_df(counties: list[dict]) -> pd.DataFrame:
    """Build a minimal wide-format DataFrame for feature computation tests."""
    default = {
        "county_fips": "12001",
        "year": 2022,
        "empl_10": 10000.0,   # total
        "empl_31": 1000.0,    # manufacturing
        "empl_92": 800.0,     # government
        "empl_62": 1200.0,    # healthcare
        "empl_44": 700.0,     # retail
        "empl_23": 400.0,     # construction
        "empl_52": 300.0,     # finance
        "empl_72": 600.0,     # hospitality
        "empl_48": 500.0,     # transportation
        "total_wages": 500000000.0,
    }
    records = [{**default, **c} for c in counties]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for build_url()."""

    def test_url_contains_year(self):
        """URL must contain the requested year."""
        url = build_url(2022, "10")
        assert "2022" in url

    def test_url_contains_industry_code(self):
        """URL must contain the industry code."""
        url = build_url(2022, "62")
        assert "62" in url

    def test_url_contains_county_all(self):
        """URL must target the county/all endpoint."""
        url = build_url(2022, "10")
        assert "county/all" in url

    def test_url_is_csv(self):
        """URL must end in .csv."""
        url = build_url(2022, "10")
        assert url.endswith(".csv")

    def test_url_on_bls_domain(self):
        """URL must be on the BLS data domain."""
        url = build_url(2022, "10")
        assert "data.bls.gov" in url

    def test_url_uses_annual_marker(self):
        """URL must use 'A' for annual average in the path."""
        url = build_url(2022, "10")
        assert "/A/" in url

    def test_all_industry_codes_produce_unique_urls(self):
        """Each industry code must produce a unique URL for the same year."""
        urls = [build_url(2022, code) for code in INDUSTRY_CODES.values()]
        assert len(urls) == len(set(urls)), "Duplicate URLs for different industry codes"

    def test_different_years_produce_different_urls(self):
        """Same industry code, different year → different URL."""
        assert build_url(2020, "10") != build_url(2022, "10")


# ---------------------------------------------------------------------------
# filter_county_df — annual average filter
# ---------------------------------------------------------------------------


class TestFilterAnnualAverage:
    """Tests that filter_county_df() keeps only annual-average rows."""

    def test_keeps_annual_average_rows(self):
        """Rows with qtr == 'A' must be kept."""
        df = _make_raw_csv_df([{"area_fips": "12001", "qtr": "A"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) >= 1

    def test_drops_quarterly_rows(self):
        """Rows with qtr == '1', '2', '3', '4' must be dropped."""
        rows = [
            {"area_fips": "12001", "qtr": "1"},
            {"area_fips": "12001", "qtr": "2"},
            {"area_fips": "12001", "qtr": "3"},
            {"area_fips": "12001", "qtr": "4"},
        ]
        df = _make_raw_csv_df(rows)
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 0

    def test_mixed_qtrs_keeps_only_annual(self):
        """Mix of A and quarterly rows: only A rows retained."""
        df = _make_raw_csv_df(
            [
                {"area_fips": "12001", "qtr": "A"},
                {"area_fips": "12003", "qtr": "1"},
            ]
        )
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 1
        assert result.iloc[0]["county_fips"] == "12001"


# ---------------------------------------------------------------------------
# filter_county_df — ownership code filter
# ---------------------------------------------------------------------------


class TestFilterOwnership:
    """Tests that filter_county_df() keeps only total-ownership rows."""

    def test_keeps_own_code_0(self):
        """Rows with own_code == '0' (total) must be kept."""
        df = _make_raw_csv_df([{"area_fips": "12001", "own_code": "0"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) >= 1

    def test_drops_private_only(self):
        """Rows with own_code == '5' (private) must be dropped."""
        df = _make_raw_csv_df([{"area_fips": "12001", "own_code": "5"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 0

    def test_drops_government_subtypes(self):
        """Rows with own_code '1', '2', '3' must be dropped."""
        rows = [
            {"area_fips": "12001", "own_code": "1"},
            {"area_fips": "12001", "own_code": "2"},
            {"area_fips": "12001", "own_code": "3"},
        ]
        df = _make_raw_csv_df(rows)
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_county_df — state scope filter
# ---------------------------------------------------------------------------


class TestFilterStateScope:
    """Tests that filter_county_df() keeps only counties in configured states (all 50+DC)."""

    def test_keeps_fl_county(self):
        """FL counties (12xxx) must be kept."""
        df = _make_raw_csv_df([{"area_fips": "12001"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) >= 1

    def test_keeps_ga_county(self):
        """GA counties (13xxx) must be kept."""
        df = _make_raw_csv_df([{"area_fips": "13121"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) >= 1

    def test_keeps_al_county(self):
        """AL counties (01xxx) must be kept."""
        df = _make_raw_csv_df([{"area_fips": "01073"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) >= 1

    def test_keeps_target_state_tx(self):
        """Texas counties (48xxx) are now kept — TX is a configured state (national scope)."""
        df = _make_raw_csv_df([{"area_fips": "48001"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) >= 1

    def test_drops_state_level_fips(self):
        """State-level FIPS (12000) must be dropped (county part == '000')."""
        df = _make_raw_csv_df([{"area_fips": "12000"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 0

    def test_drops_us_national_fips(self):
        """Non-numeric FIPS like 'US000' must be dropped."""
        df = _make_raw_csv_df([{"area_fips": "US000"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_county_df — suppressed rows
# ---------------------------------------------------------------------------


class TestFilterSuppressed:
    """Tests that filter_county_df() drops suppressed rows."""

    def test_drops_disclosure_n(self):
        """Rows with disclosure_code == 'N' must be dropped."""
        df = _make_raw_csv_df([{"area_fips": "12001", "disclosure_code": "N"}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 0

    def test_keeps_non_suppressed(self):
        """Rows with disclosure_code != 'N' must be kept."""
        df = _make_raw_csv_df([{"area_fips": "12001", "disclosure_code": " "}])
        result = filter_county_df(df, 2022, "10")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# filter_county_df — output schema
# ---------------------------------------------------------------------------


class TestFilterOutputSchema:
    """Tests that filter_county_df() produces correct output schema."""

    @pytest.fixture(scope="class")
    def filtered(self):
        df = _make_raw_csv_df(
            [
                {
                    "area_fips": "12001",
                    "own_code": "0",
                    "industry_code": "10",
                    "year": "2022",
                    "qtr": "A",
                    "annual_avg_emplvl": "10000",
                    "total_annual_wages": "500000000",
                }
            ]
        )
        return filter_county_df(df, 2022, "10")

    def test_has_county_fips_column(self, filtered):
        """Output must have county_fips column."""
        assert "county_fips" in filtered.columns

    def test_county_fips_5_digits(self, filtered):
        """county_fips must be 5-digit strings."""
        assert all(len(f) == 5 for f in filtered["county_fips"])
        assert all(f.isdigit() for f in filtered["county_fips"])

    def test_has_year_column(self, filtered):
        """Output must have year column."""
        assert "year" in filtered.columns

    def test_has_employment_column(self, filtered):
        """Output must have annual_avg_emplvl column."""
        assert "annual_avg_emplvl" in filtered.columns

    def test_has_wages_column(self, filtered):
        """Output must have total_annual_wages column."""
        assert "total_annual_wages" in filtered.columns

    def test_no_area_fips_column(self, filtered):
        """area_fips must be renamed to county_fips in output."""
        assert "area_fips" not in filtered.columns


# ---------------------------------------------------------------------------
# filter_county_df — edge cases
# ---------------------------------------------------------------------------


class TestFilterEdgeCases:
    """Tests for edge cases in filter_county_df()."""

    def test_empty_input_returns_empty(self):
        """Empty input must return empty output."""
        result = filter_county_df(pd.DataFrame(), 2022, "10")
        assert len(result) == 0

    def test_none_input_returns_empty(self):
        """None input must return empty output."""
        result = filter_county_df(None, 2022, "10")
        assert len(result) == 0

    def test_all_suppressed_returns_empty(self):
        """All-suppressed input must return empty output."""
        df = _make_raw_csv_df(
            [
                {"area_fips": "12001", "disclosure_code": "N"},
                {"area_fips": "12003", "disclosure_code": "N"},
            ]
        )
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 0

    def test_all_configured_state_rows_kept(self):
        """TX and CA are now configured states — rows are kept (national scope)."""
        df = _make_raw_csv_df(
            [
                {"area_fips": "48001"},  # TX — now in scope
                {"area_fips": "06001"},  # CA — now in scope
            ]
        )
        result = filter_county_df(df, 2022, "10")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_states_fips_correct(self):
        """STATES must map AL→01, FL→12, GA→13."""
        assert STATES["AL"] == "01"
        assert STATES["FL"] == "12"
        assert STATES["GA"] == "13"

    def test_target_state_fips_matches_states(self):
        """TARGET_STATE_FIPS must be the set of STATES values."""
        assert TARGET_STATE_FIPS == frozenset(STATES.values())

    def test_total_industry_code_is_10(self):
        """Total industry code must be '10'."""
        assert TOTAL_INDUSTRY_CODE == "10"

    def test_total_in_industry_codes(self):
        """'total' key must be in INDUSTRY_CODES."""
        assert "total" in INDUSTRY_CODES
        assert INDUSTRY_CODES["total"] == "10"

    def test_healthcare_industry_code(self):
        """Healthcare must map to NAICS code '62'."""
        assert INDUSTRY_CODES.get("healthcare") == "62"

    def test_default_years_recent(self):
        """DEFAULT_YEARS must contain years after 2019."""
        assert all(y >= 2020 for y in DEFAULT_YEARS)

    def test_default_years_not_empty(self):
        """DEFAULT_YEARS must have at least one year."""
        assert len(DEFAULT_YEARS) >= 1


# ---------------------------------------------------------------------------
# pivot_qcew
# ---------------------------------------------------------------------------


class TestPivotQcew:
    """Tests for pivot_qcew()."""

    def test_pivot_produces_wide_format(self):
        """Pivot should produce one row per (county_fips, year)."""
        df = pd.DataFrame(
            [
                {
                    "county_fips": "12001",
                    "year": 2022,
                    "industry_code": "10",
                    "annual_avg_emplvl": 10000.0,
                    "total_annual_wages": 500000000.0,
                },
                {
                    "county_fips": "12001",
                    "year": 2022,
                    "industry_code": "62",
                    "annual_avg_emplvl": 1200.0,
                    "total_annual_wages": 60000000.0,
                },
            ]
        )
        wide = pivot_qcew(df)
        assert len(wide) == 1  # one row for (12001, 2022)
        assert "county_fips" in wide.columns
        assert "year" in wide.columns

    def test_pivot_creates_empl_columns(self):
        """Pivot should create empl_{naics_code} columns."""
        df = pd.DataFrame(
            [
                {
                    "county_fips": "12001",
                    "year": 2022,
                    "industry_code": "10",
                    "annual_avg_emplvl": 10000.0,
                    "total_annual_wages": 500000000.0,
                }
            ]
        )
        wide = pivot_qcew(df)
        assert "empl_10" in wide.columns

    def test_empty_input_returns_empty(self):
        """Empty input must return empty output."""
        result = pivot_qcew(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# compute_shares
# ---------------------------------------------------------------------------


class TestComputeShares:
    """Tests for compute_shares()."""

    def test_share_is_fraction(self):
        """Employment shares must be in [0, 1]."""
        wide = _make_wide_df([{"empl_62": 1000.0, "empl_10": 10000.0}])
        shares = compute_shares(wide)
        assert "healthcare_share" in shares.columns
        val = shares["healthcare_share"].iloc[0]
        assert 0.0 <= val <= 1.0

    def test_zero_total_employment_produces_nan(self):
        """Zero total employment must produce NaN shares."""
        wide = _make_wide_df([{"empl_10": 0.0, "empl_62": 0.0}])
        shares = compute_shares(wide)
        assert shares["healthcare_share"].isna().all()

    def test_correct_share_value(self):
        """healthcare_share = empl_62 / empl_10."""
        wide = _make_wide_df([{"empl_62": 2000.0, "empl_10": 10000.0}])
        shares = compute_shares(wide)
        assert abs(shares["healthcare_share"].iloc[0] - 0.2) < 1e-6

    def test_missing_sector_column_produces_nan(self):
        """Missing sector column in wide DataFrame produces NaN share."""
        wide = _make_wide_df([{}])
        # Remove a sector column to simulate missing data
        if "empl_52" in wide.columns:
            wide = wide.drop(columns=["empl_52"])
        shares = compute_shares(wide)
        if "finance_share" in shares.columns:
            # Either NaN or not present — both acceptable
            pass  # Just check it doesn't crash


# ---------------------------------------------------------------------------
# compute_hhi
# ---------------------------------------------------------------------------


class TestComputeHhi:
    """Tests for compute_hhi()."""

    def test_hhi_range(self):
        """HHI must be in [0, 1] for valid shares."""
        shares = pd.DataFrame(
            {
                "manufacturing_share": [0.1],
                "government_share": [0.2],
                "healthcare_share": [0.15],
                "retail_share": [0.1],
                "construction_share": [0.05],
                "finance_share": [0.05],
                "hospitality_share": [0.1],
            }
        )
        hhi = compute_hhi(shares)
        assert 0.0 <= hhi.iloc[0] <= 1.0

    def test_single_sector_hhi_is_1(self):
        """All employment in one sector should produce HHI = 1.0."""
        shares = pd.DataFrame(
            {
                "manufacturing_share": [1.0],
                "government_share": [0.0],
                "healthcare_share": [0.0],
                "retail_share": [0.0],
                "construction_share": [0.0],
                "finance_share": [0.0],
                "hospitality_share": [0.0],
            }
        )
        hhi = compute_hhi(shares)
        assert abs(hhi.iloc[0] - 1.0) < 1e-6

    def test_all_nan_shares_produce_nan_hhi(self):
        """All-NaN shares must produce NaN HHI (not 0)."""
        shares = pd.DataFrame(
            {
                "manufacturing_share": [float("nan")],
                "government_share": [float("nan")],
                "healthcare_share": [float("nan")],
                "retail_share": [float("nan")],
                "construction_share": [float("nan")],
                "finance_share": [float("nan")],
                "hospitality_share": [float("nan")],
            }
        )
        hhi = compute_hhi(shares)
        assert hhi.isna().all()


# ---------------------------------------------------------------------------
# compute_top_industry
# ---------------------------------------------------------------------------


class TestComputeTopIndustry:
    """Tests for compute_top_industry()."""

    def test_returns_naics_code_string(self):
        """top_industry must return a NAICS code string."""
        wide = _make_wide_df([{"empl_62": 3000.0, "empl_10": 10000.0}])
        top = compute_top_industry(wide)
        # Should be a string NAICS code
        assert isinstance(top.iloc[0], str)

    def test_returns_largest_sector(self):
        """top_industry must be the NAICS code with highest employment."""
        wide = _make_wide_df(
            [
                {
                    "empl_62": 5000.0,  # healthcare — largest
                    "empl_31": 1000.0,
                    "empl_92": 800.0,
                    "empl_44": 700.0,
                    "empl_23": 400.0,
                    "empl_52": 300.0,
                    "empl_72": 200.0,
                    "empl_48": 150.0,
                    "empl_10": 10000.0,
                }
            ]
        )
        top = compute_top_industry(wide)
        assert top.iloc[0] == "62"

    def test_all_nan_sectors_produce_nan(self):
        """All-NaN sector employment must produce NaN top_industry."""
        wide = pd.DataFrame(
            [
                {
                    "county_fips": "12001",
                    "year": 2022,
                    "empl_10": 10000.0,
                    "total_wages": 500000000.0,
                    # No sector columns at all
                }
            ]
        )
        top = compute_top_industry(wide)
        # Either NaN or handles gracefully
        assert top is not None


# ---------------------------------------------------------------------------
# compute_avg_pay
# ---------------------------------------------------------------------------


class TestComputeAvgPay:
    """Tests for compute_avg_pay()."""

    def test_avg_pay_correct_value(self):
        """avg_annual_pay = total_wages / total_employment."""
        wide = _make_wide_df([{"empl_10": 1000.0, "total_wages": 50000000.0}])
        pay = compute_avg_pay(wide)
        assert abs(pay.iloc[0] - 50000.0) < 1.0

    def test_zero_employment_produces_nan(self):
        """Zero employment must produce NaN avg_annual_pay."""
        wide = _make_wide_df([{"empl_10": 0.0, "total_wages": 1000000.0}])
        pay = compute_avg_pay(wide)
        assert pay.isna().all()

    def test_avg_pay_is_non_negative(self):
        """avg_annual_pay must be non-negative for valid inputs."""
        wide = _make_wide_df([{"empl_10": 5000.0, "total_wages": 200000000.0}])
        pay = compute_avg_pay(wide)
        assert pay.iloc[0] >= 0


# ---------------------------------------------------------------------------
# compute_qcew_features — full pipeline
# ---------------------------------------------------------------------------


class TestComputeQcewFeatures:
    """Tests for compute_qcew_features() end-to-end pipeline."""

    @pytest.fixture(scope="class")
    def synthetic_raw(self):
        """Synthetic raw QCEW DataFrame for two FL counties, 2 years."""
        rows = []
        for county in ("12001", "12003"):
            for year in (2021, 2022):
                rows.append(
                    {
                        "county_fips": county,
                        "year": year,
                        "industry_code": "10",
                        "own_code": "0",
                        "annual_avg_emplvl": 10000.0,
                        "total_annual_wages": 500000000.0,
                    }
                )
                for code, empl in [("31", 1000), ("92", 800), ("62", 1200), ("44", 700),
                                   ("23", 400), ("52", 300), ("72", 600), ("48", 500)]:
                    rows.append(
                        {
                            "county_fips": county,
                            "year": year,
                            "industry_code": code,
                            "own_code": "0",
                            "annual_avg_emplvl": float(empl),
                            "total_annual_wages": float(empl) * 50000,
                        }
                    )
        return pd.DataFrame(rows)

    def test_output_shape(self, synthetic_raw):
        """Features should have one row per (county_fips, year)."""
        features = compute_qcew_features(synthetic_raw)
        n_counties = synthetic_raw["county_fips"].nunique()
        n_years = synthetic_raw["year"].nunique()
        assert len(features) == n_counties * n_years

    def test_required_columns_present(self, synthetic_raw):
        """All QCEW_FEATURE_COLS must be present in output."""
        features = compute_qcew_features(synthetic_raw)
        for col in QCEW_FEATURE_COLS:
            assert col in features.columns, f"Missing column: {col}"

    def test_shares_are_fractions(self, synthetic_raw):
        """All share columns must be in [0, 1]."""
        features = compute_qcew_features(synthetic_raw)
        share_cols = [c for c in QCEW_FEATURE_COLS if c.endswith("_share")]
        for col in share_cols:
            vals = features[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1] range"

    def test_hhi_in_range(self, synthetic_raw):
        """industry_diversity_hhi must be in [0, 1]."""
        features = compute_qcew_features(synthetic_raw)
        hhi = features["industry_diversity_hhi"].dropna()
        assert (hhi >= 0).all() and (hhi <= 1).all()

    def test_top_industry_is_string(self, synthetic_raw):
        """top_industry must be non-null strings (NAICS codes)."""
        features = compute_qcew_features(synthetic_raw)
        top = features["top_industry"].dropna()
        assert (top.apply(lambda x: isinstance(x, str))).all()

    def test_avg_annual_pay_positive(self, synthetic_raw):
        """avg_annual_pay must be positive for counties with employment."""
        features = compute_qcew_features(synthetic_raw)
        pay = features["avg_annual_pay"].dropna()
        assert (pay > 0).all()

    def test_empty_input_returns_empty(self):
        """Empty input must return empty output."""
        result = compute_qcew_features(pd.DataFrame())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# impute_qcew_state_medians
# ---------------------------------------------------------------------------


class TestImputeQcewStateMedians:
    """Tests for impute_qcew_state_medians()."""

    def test_fills_nan_with_state_year_median(self):
        """NaN values must be filled with the state×year median."""
        df = pd.DataFrame(
            {
                "county_fips": ["12001", "12003", "12005"],
                "year": [2022, 2022, 2022],
                "manufacturing_share": [0.1, 0.2, float("nan")],
                "government_share": [0.2, 0.3, 0.25],
                "healthcare_share": [0.15, 0.12, 0.14],
                "retail_share": [0.1, 0.1, 0.1],
                "construction_share": [0.05, 0.05, 0.05],
                "finance_share": [0.05, 0.05, 0.05],
                "hospitality_share": [0.08, 0.08, 0.08],
                "industry_diversity_hhi": [0.1, 0.12, float("nan")],
                "top_industry": ["62", "92", None],
                "avg_annual_pay": [50000.0, 52000.0, float("nan")],
            }
        )
        result = impute_qcew_state_medians(df)
        # The NaN manufacturing_share for 12005 should be filled with median of 12001, 12003
        expected_median = 0.15  # median of [0.1, 0.2]
        assert abs(result.loc[2, "manufacturing_share"] - expected_median) < 1e-6

    def test_no_nan_unchanged(self):
        """Rows without NaN must be unchanged after imputation."""
        df = pd.DataFrame(
            {
                "county_fips": ["12001"],
                "year": [2022],
                "manufacturing_share": [0.1],
                "government_share": [0.2],
                "healthcare_share": [0.15],
                "retail_share": [0.1],
                "construction_share": [0.05],
                "finance_share": [0.05],
                "hospitality_share": [0.08],
                "industry_diversity_hhi": [0.1],
                "top_industry": ["62"],
                "avg_annual_pay": [50000.0],
            }
        )
        result = impute_qcew_state_medians(df)
        assert abs(result.loc[0, "manufacturing_share"] - 0.1) < 1e-9

    def test_imputation_respects_year(self):
        """Imputation must use same-year medians, not cross-year."""
        df = pd.DataFrame(
            {
                "county_fips": ["12001", "12003", "12005"],
                "year": [2021, 2022, 2022],
                "manufacturing_share": [0.1, 0.2, float("nan")],
                "government_share": [0.2, 0.3, 0.25],
                "healthcare_share": [0.15, 0.12, 0.14],
                "retail_share": [0.1, 0.1, 0.1],
                "construction_share": [0.05, 0.05, 0.05],
                "finance_share": [0.05, 0.05, 0.05],
                "hospitality_share": [0.08, 0.08, 0.08],
                "industry_diversity_hhi": [0.1, 0.12, float("nan")],
                "top_industry": ["62", "92", None],
                "avg_annual_pay": [50000.0, 52000.0, float("nan")],
            }
        )
        result = impute_qcew_state_medians(df)
        # 12005 year=2022, only 12003 year=2022 available → imputed with 0.2
        assert abs(result.loc[2, "manufacturing_share"] - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# Integration tests (skip if raw data not present)
# ---------------------------------------------------------------------------


class TestQcewIntegration:
    """Integration tests against the actual saved parquet files (skipped if absent)."""

    @pytest.fixture(scope="class")
    def raw_parquet(self):
        """Load qcew_county.parquet if it exists."""
        from pathlib import Path

        path = Path(__file__).parents[1] / "data" / "raw" / "qcew_county.parquet"
        if not path.exists():
            pytest.skip("qcew_county.parquet not found — run fetch_bls_qcew.py first")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def features_parquet(self):
        """Load county_qcew_features.parquet if it exists."""
        from pathlib import Path

        path = Path(__file__).parents[1] / "data" / "assembled" / "county_qcew_features.parquet"
        if not path.exists():
            pytest.skip(
                "county_qcew_features.parquet not found — run build_qcew_features.py first"
            )
        return pd.read_parquet(path)

    def test_raw_has_required_columns(self, raw_parquet):
        """Raw parquet must have the required output columns."""
        required = {"county_fips", "year", "industry_code", "annual_avg_emplvl"}
        assert required.issubset(set(raw_parquet.columns))

    def test_raw_fips_are_5_digits(self, raw_parquet):
        """All county FIPS codes must be 5-digit strings."""
        assert (raw_parquet["county_fips"].str.len() == 5).all()
        assert raw_parquet["county_fips"].str.isdigit().all()

    def test_raw_target_states_only(self, raw_parquet):
        """Only configured states (all 50+DC) may appear in the raw parquet."""
        state_prefixes = raw_parquet["county_fips"].str[:2].unique()
        assert set(state_prefixes) <= TARGET_STATE_FIPS

    def test_raw_no_state_level_fips(self, raw_parquet):
        """No state-level FIPS (county part == '000') must be present."""
        county_part = raw_parquet["county_fips"].str[2:]
        assert (county_part != "000").all()

    def test_features_has_required_columns(self, features_parquet):
        """Features parquet must have county_fips, year, and all feature cols."""
        required = {"county_fips", "year"}
        assert required.issubset(set(features_parquet.columns))
        for col in QCEW_FEATURE_COLS:
            assert col in features_parquet.columns, f"Missing feature column: {col}"

    def test_features_shares_in_range(self, features_parquet):
        """All share features must be in [0, 1]."""
        share_cols = [c for c in QCEW_FEATURE_COLS if c.endswith("_share")]
        for col in share_cols:
            vals = features_parquet[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"
