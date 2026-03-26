"""Tests for build_county_features_national.py.

Verifies:
- ACS + RCMS join produces correct row count and columns
- FIPS format validated (5-char zero-padded strings)
- Missing RCMS counties imputed with state-level medians (fallback to national)
- Connecticut 2022 planning region FIPS (09xxx) handled via national median fallback
- All RCMS features bounded in reasonable ranges after imputation
- Merge does not lose ACS counties (left join preserves all)
- No spurious duplicate rows
- QCEW industry features present and imputed when missing
- CHR health features present (excl. ACS-overlap cols) and imputed when missing
- Migration features present and imputed when missing
- Urbanicity features present and imputed when missing
- SCI social connectedness features present and imputed when missing
- Total column count correct (ACS + RCMS + QCEW + CHR + Migration + Urbanicity + SCI)
- No duplicate columns after joining all sources
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_county_features_national import (
    CHR_FEATURE_COLS,
    MIGRATION_FEATURE_COLS,
    QCEW_FEATURE_COLS,
    RCMS_FEATURE_COLS,
    SCI_FEATURE_COLS,
    URBANICITY_FEATURE_COLS,
    build_national_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_acs(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic ACS county features DataFrame."""
    rng = np.random.default_rng(0)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "pop_total": rng.integers(5000, 500000, size=n),
        "pct_white_nh": rng.uniform(0.1, 0.95, size=n),
        "pct_black": rng.uniform(0.01, 0.50, size=n),
        "pct_asian": rng.uniform(0.01, 0.20, size=n),
        "pct_hispanic": rng.uniform(0.01, 0.40, size=n),
        "median_age": rng.uniform(30, 55, size=n),
        "median_hh_income": rng.uniform(40000, 120000, size=n),
        "log_median_hh_income": rng.uniform(4.5, 5.2, size=n),
        "pct_bachelors_plus": rng.uniform(0.10, 0.60, size=n),
        "pct_graduate": rng.uniform(0.05, 0.30, size=n),
        "pct_owner_occupied": rng.uniform(0.30, 0.85, size=n),
        "pct_wfh": rng.uniform(0.01, 0.25, size=n),
        "pct_transit": rng.uniform(0.00, 0.15, size=n),
        "pct_management": rng.uniform(0.05, 0.35, size=n),
    })


def _make_rcms(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic RCMS county features DataFrame."""
    rng = np.random.default_rng(1)
    n = len(fips_list)
    fracs = rng.dirichlet(np.array([5.0, 1.5, 2.0, 1.0, 2.0]), size=n)
    return pd.DataFrame({
        "county_fips": fips_list,
        "state_abbr": ["AL"] * n,  # simplified for testing
        "evangelical_share": fracs[:, 0],
        "mainline_share": fracs[:, 1],
        "catholic_share": fracs[:, 2],
        "black_protestant_share": fracs[:, 3],
        "congregations_per_1000": rng.uniform(1.0, 10.0, size=n),
        "religious_adherence_rate": rng.uniform(200, 700, size=n),
    })


# ---------------------------------------------------------------------------
# Tests: basic join
# ---------------------------------------------------------------------------


class TestBuildNationalFeatures:
    def test_output_row_count_equals_acs(self):
        """Left join preserves all ACS counties."""
        acs_fips = ["01001", "01003", "01005", "12001", "12003"]
        rcms_fips = ["01001", "01003", "01005", "12001", "12003"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        assert len(result) == len(acs_fips)

    def test_output_columns_include_acs_and_rcms(self):
        """Output must have both ACS and RCMS feature columns."""
        acs_fips = ["01001", "01003"]
        rcms_fips = ["01001", "01003"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert col in result.columns, f"Missing RCMS column: {col}"
        assert "pct_white_nh" in result.columns
        assert "pct_bachelors_plus" in result.columns
        assert "county_fips" in result.columns

    def test_fips_preserved(self):
        """county_fips values from ACS are preserved unchanged."""
        acs_fips = ["01001", "01003", "12001"]
        rcms_fips = ["01001", "01003", "12001"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        assert set(result["county_fips"].tolist()) == set(acs_fips)

    def test_no_duplicates(self):
        """No duplicate FIPS rows in output."""
        acs_fips = ["01001", "01003", "12001"]
        rcms_fips = ["01001", "01003", "12001"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        assert result["county_fips"].nunique() == len(result)


# ---------------------------------------------------------------------------
# Tests: missing RCMS handling
# ---------------------------------------------------------------------------


class TestMissingRcmsImputation:
    def test_missing_rcms_county_imputed(self):
        """ACS county with no RCMS match gets imputed RCMS features (not NaN)."""
        acs_fips = ["01001", "01003", "01005"]  # 3 ACS counties
        rcms_fips = ["01001", "01003"]           # RCMS missing 01005
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} has NaN after imputation"

    def test_all_state_missing_uses_national_median(self):
        """Counties whose state has no RCMS data at all get national median fallback."""
        # AL counties have RCMS; CT new planning regions (09xxx) do not
        acs_fips = ["01001", "01003", "09110", "09120"]  # AL has RCMS; CT does not
        rcms_fips = ["01001", "01003"]                    # No CT entries
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} still NaN after national fallback"

    def test_imputed_value_equals_state_median(self):
        """Missing RCMS county gets the state-level median of the other counties."""
        # Two AL counties with known evangelical_share: 0.3 and 0.5 → median = 0.4
        acs_fips = ["01001", "01003", "01005"]
        rcms_fips = ["01001", "01003"]
        acs = _make_acs(acs_fips)

        rcms = pd.DataFrame({
            "county_fips": rcms_fips,
            "state_abbr": ["AL", "AL"],
            "evangelical_share": [0.3, 0.5],
            "mainline_share": [0.1, 0.2],
            "catholic_share": [0.2, 0.3],
            "black_protestant_share": [0.05, 0.10],
            "congregations_per_1000": [3.0, 5.0],
            "religious_adherence_rate": [400.0, 600.0],
        })

        result = build_national_features(acs, rcms)
        missing_row = result[result["county_fips"] == "01005"]
        assert len(missing_row) == 1
        # State median of [0.3, 0.5] = 0.4
        assert abs(missing_row["evangelical_share"].iloc[0] - 0.4) < 1e-6

    def test_rcms_features_nonnegative(self):
        """All RCMS feature values are non-negative after imputation."""
        acs_fips = ["01001", "01003", "01005", "12001"]
        rcms_fips = ["01001", "01003"]
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(rcms_fips)
        result = build_national_features(acs, rcms)
        for col in RCMS_FEATURE_COLS:
            assert (result[col] >= 0).all(), f"{col} has negative values"


# ---------------------------------------------------------------------------
# Tests: FIPS validation
# ---------------------------------------------------------------------------


class TestFipsValidation:
    def test_bad_acs_fips_raises(self):
        """Malformed ACS county_fips raises ValueError."""
        bad_acs = _make_acs(["1001", "01003"])  # '1001' is 4 chars, not 5
        rcms = _make_rcms(["01001", "01003"])
        with pytest.raises(ValueError, match="5-char"):
            build_national_features(bad_acs, rcms)

    def test_bad_rcms_fips_raises(self):
        """Malformed RCMS county_fips raises ValueError."""
        acs = _make_acs(["01001", "01003"])
        bad_rcms = _make_rcms(["1001", "01003"])  # '1001' is 4 chars
        with pytest.raises(ValueError, match="5-char"):
            build_national_features(acs, bad_rcms)


# ---------------------------------------------------------------------------
# Helpers for QCEW and CHR
# ---------------------------------------------------------------------------


def _make_qcew(fips_list: list[str], year: int = 2023) -> pd.DataFrame:
    """Minimal synthetic QCEW county × year features DataFrame."""
    rng = np.random.default_rng(10)
    n = len(fips_list)
    shares = rng.dirichlet(np.ones(7), size=n)  # 7 industry shares sum to 1
    return pd.DataFrame({
        "county_fips": fips_list,
        "year": year,
        "manufacturing_share": shares[:, 0],
        "government_share": shares[:, 1],
        "healthcare_share": shares[:, 2],
        "retail_share": shares[:, 3],
        "construction_share": shares[:, 4],
        "finance_share": shares[:, 5],
        "hospitality_share": shares[:, 6],
        "industry_diversity_hhi": rng.uniform(0.03, 0.15, size=n),
        "top_industry": ["manufacturing"] * n,      # should be dropped
        "avg_annual_pay": rng.uniform(35000, 80000, size=n),  # should be dropped
    })


def _make_chr(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic CHR county features DataFrame (all expected cols)."""
    rng = np.random.default_rng(11)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "state_abbr": ["AL"] * n,           # metadata — should be dropped
        "data_year": [2023] * n,            # metadata — should be dropped
        "premature_death_rate": rng.uniform(5000, 15000, size=n),
        "adult_smoking_pct": rng.uniform(0.10, 0.30, size=n),
        "adult_obesity_pct": rng.uniform(0.20, 0.45, size=n),
        "excessive_drinking_pct": rng.uniform(0.10, 0.25, size=n),
        "uninsured_pct": rng.uniform(0.05, 0.25, size=n),
        "primary_care_physicians_rate": rng.uniform(0, 0.005, size=n),
        "mental_health_providers_rate": rng.uniform(0, 0.003, size=n),
        "median_household_income": rng.uniform(40000, 90000, size=n),  # should be dropped
        "children_in_poverty_pct": rng.uniform(0.10, 0.35, size=n),
        "insufficient_sleep_pct": rng.uniform(0.30, 0.45, size=n),
        "physical_inactivity_pct": rng.uniform(0.15, 0.40, size=n),
        "severe_housing_problems_pct": rng.uniform(2.0, 10.0, size=n),
        "drive_alone_pct": rng.uniform(2.0, 8.0, size=n),
        "high_school_completion_pct": rng.uniform(0.75, 0.95, size=n),  # should be dropped
        "some_college_pct": rng.uniform(0.30, 0.70, size=n),           # should be dropped
        "life_expectancy": rng.uniform(0.06, 0.15, size=n),
        "diabetes_prevalence_pct": rng.uniform(0.05, 0.18, size=n),
        "poor_mental_health_days": rng.uniform(2.5, 5.5, size=n),
    })


def _make_migration(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic migration county features DataFrame."""
    rng = np.random.default_rng(20)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "net_migration_rate": rng.uniform(-0.05, 0.10, size=n),
        "avg_inflow_income": rng.uniform(40000, 100000, size=n),
        "migration_diversity": rng.uniform(0.0, 1.0, size=n),
        "inflow_outflow_ratio": rng.uniform(0.5, 2.5, size=n),
    })


def _make_urbanicity(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic urbanicity county features DataFrame."""
    rng = np.random.default_rng(21)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "log_pop_density": rng.uniform(0.0, 4.5, size=n),
        "land_area_sq_mi": rng.uniform(50.0, 5000.0, size=n),
        "pop_per_sq_mi": rng.uniform(1.0, 20000.0, size=n),
    })


# ---------------------------------------------------------------------------
# Tests: QCEW features
# ---------------------------------------------------------------------------


class TestQcewIntegration:
    """QCEW industry features are merged correctly and imputed when missing."""

    def test_qcew_columns_present(self):
        """All QCEW_FEATURE_COLS appear in output when qcew is provided."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        result = build_national_features(acs, rcms, qcew=qcew)
        for col in QCEW_FEATURE_COLS:
            assert col in result.columns, f"Missing QCEW column: {col}"

    def test_qcew_dropped_cols_absent(self):
        """top_industry and avg_annual_pay must NOT appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        result = build_national_features(acs, rcms, qcew=qcew)
        assert "top_industry" not in result.columns
        assert "avg_annual_pay" not in result.columns

    def test_qcew_missing_county_imputed(self):
        """County absent from QCEW 2023 gets state-median imputation (no NaN)."""
        acs_fips = ["01001", "01003", "01005"]
        qcew_fips = ["01001", "01003"]   # 01005 missing
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(acs_fips)
        qcew = _make_qcew(qcew_fips)
        result = build_national_features(acs, rcms, qcew=qcew)
        for col in QCEW_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} has NaN after QCEW imputation"

    def test_qcew_latest_year_used(self):
        """Only 2023 rows are used when qcew has multiple years."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        # Build multi-year QCEW: 2020 with distinct values, 2023 with zeros
        qcew_old = _make_qcew(fips, year=2020)
        qcew_new = _make_qcew(fips, year=2023)
        qcew_new["manufacturing_share"] = 0.999  # sentinel
        qcew = pd.concat([qcew_old, qcew_new], ignore_index=True)
        result = build_national_features(acs, rcms, qcew=qcew)
        # If 2023 was used, manufacturing_share should be 0.999
        assert (result["manufacturing_share"] == 0.999).all()

    def test_qcew_none_no_extra_cols(self):
        """When qcew=None, no QCEW columns appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        result = build_national_features(acs, rcms, qcew=None)
        for col in QCEW_FEATURE_COLS:
            assert col not in result.columns, f"Unexpected QCEW column: {col}"


# ---------------------------------------------------------------------------
# Tests: CHR features
# ---------------------------------------------------------------------------


class TestChrIntegration:
    """CHR health features are merged correctly and ACS overlaps are excluded."""

    def test_chr_columns_present(self):
        """All CHR_FEATURE_COLS appear in output when chr_df is provided."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        chr_df = _make_chr(fips)
        result = build_national_features(acs, rcms, chr_df=chr_df)
        for col in CHR_FEATURE_COLS:
            assert col in result.columns, f"Missing CHR column: {col}"

    def test_chr_acs_overlap_cols_absent(self):
        """median_household_income, high_school_completion_pct, some_college_pct absent."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        chr_df = _make_chr(fips)
        result = build_national_features(acs, rcms, chr_df=chr_df)
        for overlap_col in ("median_household_income", "high_school_completion_pct", "some_college_pct"):
            assert overlap_col not in result.columns, f"ACS-overlap column should be absent: {overlap_col}"

    def test_chr_metadata_cols_absent(self):
        """state_abbr and data_year (metadata) must not appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        chr_df = _make_chr(fips)
        result = build_national_features(acs, rcms, chr_df=chr_df)
        assert "state_abbr" not in result.columns
        assert "data_year" not in result.columns

    def test_chr_missing_county_imputed(self):
        """County absent from CHR gets state-median imputation (no NaN)."""
        acs_fips = ["01001", "01003", "01005"]
        chr_fips = ["01001", "01003"]   # 01005 missing
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(acs_fips)
        chr_df = _make_chr(chr_fips)
        result = build_national_features(acs, rcms, chr_df=chr_df)
        for col in CHR_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} has NaN after CHR imputation"

    def test_chr_none_no_extra_cols(self):
        """When chr_df=None, no CHR columns appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        result = build_national_features(acs, rcms, chr_df=None)
        for col in CHR_FEATURE_COLS:
            assert col not in result.columns, f"Unexpected CHR column: {col}"


# ---------------------------------------------------------------------------
# Tests: Migration features
# ---------------------------------------------------------------------------


class TestMigrationIntegration:
    """Migration features are merged correctly and imputed when missing."""

    def test_migration_columns_present(self):
        """All MIGRATION_FEATURE_COLS appear in output when migration is provided."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        migration = _make_migration(fips)
        result = build_national_features(acs, rcms, migration=migration)
        for col in MIGRATION_FEATURE_COLS:
            assert col in result.columns, f"Missing migration column: {col}"

    def test_migration_missing_county_imputed(self):
        """County absent from migration data gets state-median imputation (no NaN)."""
        acs_fips = ["01001", "01003", "01005"]
        mig_fips = ["01001", "01003"]   # 01005 missing
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(acs_fips)
        migration = _make_migration(mig_fips)
        result = build_national_features(acs, rcms, migration=migration)
        for col in MIGRATION_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} has NaN after migration imputation"

    def test_migration_none_no_extra_cols(self):
        """When migration=None, no migration columns appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        result = build_national_features(acs, rcms, migration=None)
        for col in MIGRATION_FEATURE_COLS:
            assert col not in result.columns, f"Unexpected migration column: {col}"


# ---------------------------------------------------------------------------
# Tests: Urbanicity features
# ---------------------------------------------------------------------------


class TestUrbanicityIntegration:
    """Urbanicity features are merged correctly and imputed when missing."""

    def test_urbanicity_columns_present(self):
        """All URBANICITY_FEATURE_COLS appear in output when urbanicity is provided."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        urbanicity = _make_urbanicity(fips)
        result = build_national_features(acs, rcms, urbanicity=urbanicity)
        for col in URBANICITY_FEATURE_COLS:
            assert col in result.columns, f"Missing urbanicity column: {col}"

    def test_urbanicity_missing_county_imputed(self):
        """County absent from urbanicity data gets state-median imputation (no NaN)."""
        acs_fips = ["01001", "01003", "01005"]
        urb_fips = ["01001", "01003"]   # 01005 missing
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(acs_fips)
        urbanicity = _make_urbanicity(urb_fips)
        result = build_national_features(acs, rcms, urbanicity=urbanicity)
        for col in URBANICITY_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} has NaN after urbanicity imputation"

    def test_urbanicity_none_no_extra_cols(self):
        """When urbanicity=None, no urbanicity columns appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        result = build_national_features(acs, rcms, urbanicity=None)
        for col in URBANICITY_FEATURE_COLS:
            assert col not in result.columns, f"Unexpected urbanicity column: {col}"


# ---------------------------------------------------------------------------
# Tests: no duplicate columns when all sources joined
# ---------------------------------------------------------------------------


class TestNoDuplicateColumns:
    """Joining all sources must not produce duplicate column names."""

    def test_no_duplicate_cols_all_sources(self):
        """Full join (ACS + RCMS + QCEW + CHR) produces no duplicate column names."""
        fips = ["01001", "01003", "01005", "12001"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        result = build_national_features(acs, rcms, qcew=qcew, chr_df=chr_df)
        assert result.columns.nunique() == len(result.columns), (
            f"Duplicate columns: {[c for c in result.columns if list(result.columns).count(c) > 1]}"
        )

    def test_row_count_preserved_all_sources(self):
        """Row count equals ACS input when all sources are joined."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        result = build_national_features(acs, rcms, qcew=qcew, chr_df=chr_df)
        assert len(result) == len(fips)

    def test_total_feature_count_all_sources(self):
        """Output column count equals ACS + RCMS + QCEW + CHR feature counts + county_fips."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        result = build_national_features(acs, rcms, qcew=qcew, chr_df=chr_df)
        # ACS cols (including county_fips) + RCMS + QCEW + CHR
        acs_cols = len(acs.columns)  # includes county_fips
        expected = acs_cols + len(RCMS_FEATURE_COLS) + len(QCEW_FEATURE_COLS) + len(CHR_FEATURE_COLS)
        assert len(result.columns) == expected, (
            f"Expected {expected} cols, got {len(result.columns)}: {list(result.columns)}"
        )

    def test_no_duplicate_cols_all_six_sources(self):
        """Full join (ACS + RCMS + QCEW + CHR + Migration + Urbanicity) produces no duplicate column names."""
        fips = ["01001", "01003", "01005", "12001"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        migration = _make_migration(fips)
        urbanicity = _make_urbanicity(fips)
        result = build_national_features(acs, rcms, qcew=qcew, chr_df=chr_df, migration=migration, urbanicity=urbanicity)
        assert result.columns.nunique() == len(result.columns), (
            f"Duplicate columns: {[c for c in result.columns if list(result.columns).count(c) > 1]}"
        )

    def test_total_feature_count_six_sources(self):
        """Output column count equals ACS + RCMS + QCEW + CHR + Migration + Urbanicity + county_fips."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        migration = _make_migration(fips)
        urbanicity = _make_urbanicity(fips)
        result = build_national_features(
            acs, rcms, qcew=qcew, chr_df=chr_df, migration=migration, urbanicity=urbanicity
        )
        acs_cols = len(acs.columns)  # includes county_fips
        expected = (
            acs_cols
            + len(RCMS_FEATURE_COLS)
            + len(QCEW_FEATURE_COLS)
            + len(CHR_FEATURE_COLS)
            + len(MIGRATION_FEATURE_COLS)
            + len(URBANICITY_FEATURE_COLS)
        )
        assert len(result.columns) == expected, (
            f"Expected {expected} cols, got {len(result.columns)}: {list(result.columns)}"
        )


# ---------------------------------------------------------------------------
# Helper for SCI
# ---------------------------------------------------------------------------


def _make_sci(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic SCI county features DataFrame."""
    rng = np.random.default_rng(30)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "network_diversity": rng.uniform(0.5, 1.0, size=n),
        "pct_sci_instate": rng.uniform(0.0, 1.0, size=n),
        "sci_top5_mean_dem_share": rng.uniform(0.2, 0.8, size=n),
        "sci_geographic_reach": rng.integers(1, 15, size=n).astype(float),
    })


# ---------------------------------------------------------------------------
# Tests: SCI features
# ---------------------------------------------------------------------------


class TestSciIntegration:
    """SCI social connectedness features are merged correctly and imputed when missing."""

    def test_sci_columns_present(self):
        """All SCI_FEATURE_COLS appear in output when sci is provided."""
        fips = ["01001", "01003", "01005"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        sci = _make_sci(fips)
        result = build_national_features(acs, rcms, sci=sci)
        for col in SCI_FEATURE_COLS:
            assert col in result.columns, f"Missing SCI column: {col}"

    def test_sci_missing_county_imputed(self):
        """County absent from SCI data gets state-median imputation (no NaN)."""
        acs_fips = ["01001", "01003", "01005"]
        sci_fips = ["01001", "01003"]   # 01005 missing
        acs = _make_acs(acs_fips)
        rcms = _make_rcms(acs_fips)
        sci = _make_sci(sci_fips)
        result = build_national_features(acs, rcms, sci=sci)
        for col in SCI_FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"{col} has NaN after SCI imputation"

    def test_sci_none_no_extra_cols(self):
        """When sci=None, no SCI columns appear in output."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        result = build_national_features(acs, rcms, sci=None)
        for col in SCI_FEATURE_COLS:
            assert col not in result.columns, f"Unexpected SCI column: {col}"

    def test_sci_fips_validation_raises(self):
        """Malformed SCI county_fips raises ValueError."""
        acs = _make_acs(["01001", "01003"])
        rcms = _make_rcms(["01001", "01003"])
        bad_sci = _make_sci(["1001", "01003"])  # '1001' is 4 chars
        with pytest.raises(ValueError, match="5-char"):
            build_national_features(acs, rcms, sci=bad_sci)

    def test_no_duplicate_cols_all_seven_sources(self):
        """Full join (ACS + RCMS + QCEW + CHR + Migration + Urbanicity + SCI) has no duplicate columns."""
        fips = ["01001", "01003", "01005", "12001"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        migration = _make_migration(fips)
        urbanicity = _make_urbanicity(fips)
        sci = _make_sci(fips)
        result = build_national_features(
            acs, rcms, qcew=qcew, chr_df=chr_df,
            migration=migration, urbanicity=urbanicity, sci=sci,
        )
        assert result.columns.nunique() == len(result.columns), (
            f"Duplicate columns: {[c for c in result.columns if list(result.columns).count(c) > 1]}"
        )

    def test_total_feature_count_all_seven_sources(self):
        """Output column count equals ACS + RCMS + QCEW + CHR + Migration + Urbanicity + SCI."""
        fips = ["01001", "01003"]
        acs = _make_acs(fips)
        rcms = _make_rcms(fips)
        qcew = _make_qcew(fips)
        chr_df = _make_chr(fips)
        migration = _make_migration(fips)
        urbanicity = _make_urbanicity(fips)
        sci = _make_sci(fips)
        result = build_national_features(
            acs, rcms, qcew=qcew, chr_df=chr_df,
            migration=migration, urbanicity=urbanicity, sci=sci,
        )
        acs_cols = len(acs.columns)  # includes county_fips
        expected = (
            acs_cols
            + len(RCMS_FEATURE_COLS)
            + len(QCEW_FEATURE_COLS)
            + len(CHR_FEATURE_COLS)
            + len(MIGRATION_FEATURE_COLS)
            + len(URBANICITY_FEATURE_COLS)
            + len(SCI_FEATURE_COLS)
        )
        assert len(result.columns) == expected, (
            f"Expected {expected} cols, got {len(result.columns)}: {list(result.columns)}"
        )
