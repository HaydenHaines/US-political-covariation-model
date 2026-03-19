"""Tests for RCMS data fetching and feature integration.

Tests exercise:
1. fetch_rcms.py — URL construction, HTML parsing, missing county handling
2. build_features.py — RCMS feature computation logic, NaN handling, share invariants

These tests use synthetic data and mock HTTP responses so they run without
network access or the real RCMS parquet file. Tests verify:
  - URL parameters are constructed correctly for each group/variable
  - HTML parsing extracts the correct county FIPS and values
  - Missing counties (NaN) are handled gracefully
  - Share features (evangelical, mainline, etc.) sum to ≤ 1.0 per county
  - Share features are bounded [0, 1]
  - Imputation fills NaN values with state-level medians
  - All 293 FL+GA+AL counties are covered in the feature output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_rcms import (
    GROUPS,
    STATES,
    VARIABLES,
    _make_m1,
    _parse_map_data,
)
from src.assembly.build_features import (
    RCMS_FEATURE_COLS,
    compute_rcms_features,
    impute_rcms_state_medians,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected county counts: AL=67, FL=67, GA=159
EXPECTED_COUNTY_COUNTS = {"AL": 67, "FL": 67, "GA": 159}
EXPECTED_TOTAL_COUNTIES = sum(EXPECTED_COUNTY_COUNTS.values())  # 293


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_rcms() -> pd.DataFrame:
    """Synthetic RCMS data for 10 counties: 3 AL, 3 FL, 4 GA.

    Designed to test share computation, NaN handling, and imputation.
    Includes two counties with missing group data (catholic NaN, black_prot NaN).
    """
    rng = np.random.default_rng(42)
    n = 10
    fips_list = ["01001", "01003", "01005", "12001", "12003", "12005", "13001", "13003", "13005", "13007"]
    state_list = ["AL", "AL", "AL", "FL", "FL", "FL", "GA", "GA", "GA", "GA"]

    total_adherents = rng.integers(5000, 200000, size=n).astype(float)

    # Draw fractions that sum to ≤ 1 (real RCMS constraint: each tradition ≤ total)
    # Use Dirichlet to ensure properly normalized fractions for 5 "buckets"
    # (evangelical, mainline, catholic, black_protestant, other)
    alpha = np.array([5.0, 1.0, 1.5, 2.0, 2.5])  # biased toward evangelical (Deep South)
    fracs = rng.dirichlet(alpha, size=n)  # shape: (n, 5), rows sum to 1
    evang_frac = fracs[:, 0]
    mainline_frac = fracs[:, 1]
    catholic_frac = fracs[:, 2]
    black_prot_frac = fracs[:, 3]
    # fracs[:, 4] is "other" — not tracked individually

    df = pd.DataFrame({
        "county_fips": fips_list,
        "state_abbr": state_list,
        "adherents_total": total_adherents,
        "adherents_evangelical": total_adherents * evang_frac,
        "adherents_mainline": total_adherents * mainline_frac,
        "adherents_catholic": total_adherents * catholic_frac,
        "adherents_black_protestant": total_adherents * black_prot_frac,
        "congregations_total": rng.integers(50, 500, size=n).astype(float),
        "adherence_rate_total": rng.uniform(300, 700, size=n),
    })

    # Introduce NaN to simulate small counties with no RCMS data for that group
    df.loc[2, "adherents_catholic"] = np.nan        # AL county: no Catholic presence
    df.loc[7, "adherents_black_protestant"] = np.nan  # GA county: no Black Protestant data

    return df


@pytest.fixture(scope="module")
def synthetic_rcms_with_missing_county() -> pd.DataFrame:
    """Synthetic RCMS with one county that has no data at all (total adherents NaN)."""
    df = pd.DataFrame({
        "county_fips": ["01001", "01003", "01005"],
        "state_abbr": ["AL", "AL", "AL"],
        "adherents_total": [50000.0, np.nan, 75000.0],  # 01003 entirely missing
        "adherents_evangelical": [30000.0, np.nan, 45000.0],
        "adherents_mainline": [5000.0, np.nan, 8000.0],
        "adherents_catholic": [2000.0, np.nan, 3000.0],
        "adherents_black_protestant": [4000.0, np.nan, 6000.0],
        "congregations_total": [150.0, np.nan, 200.0],
        "adherence_rate_total": [450.0, np.nan, 520.0],
    })
    return df


# ---------------------------------------------------------------------------
# fetch_rcms.py unit tests
# ---------------------------------------------------------------------------


class TestMakeM1:
    """Tests for the m1 parameter encoding function."""

    def test_adherents_all_groups(self):
        """Adherent count for all groups should produce correct m1."""
        assert _make_m1("1y2020", "9999") == "1_2_9999_2020"

    def test_congregations_all_groups(self):
        """Congregation count for all groups should produce correct m1."""
        assert _make_m1("0y2020", "9999") == "0_2_9999_2020"

    def test_adherence_rate_evangelical(self):
        """Adherence rate for Evangelical should produce correct m1."""
        assert _make_m1("2y2020", "1") == "2_2_1_2020"

    def test_adherents_catholic(self):
        """Adherent count for Catholic should produce correct m1."""
        assert _make_m1("1y2020", "3") == "1_2_3_2020"

    def test_year_is_encoded_correctly(self):
        """Year extracted from t_value must appear in m1."""
        result = _make_m1("1y2020", "9999")
        assert "2020" in result


class TestParseMapData:
    """Tests for the HTML map data extraction function."""

    def test_basic_parsing(self):
        """Should extract county FIPS and float values from ARDA HTML."""
        html = """
        county_map_data = [
            { id: "01073", value: 539405 } ,
            { id: "01097", value: 302706.5 } ,
        ];
        """
        result = _parse_map_data(html)
        assert result == {"01073": 539405.0, "01097": 302706.5}

    def test_returns_empty_on_no_data(self):
        """Should return empty dict when no county data is embedded."""
        result = _parse_map_data("<html>no data here</html>")
        assert result == {}

    def test_fips_are_5_digits(self):
        """All extracted FIPS keys must be 5-digit strings."""
        html = '{ id: "12001", value: 100 } { id: "13001", value: 200 }'
        result = _parse_map_data(html)
        for fips in result:
            assert len(fips) == 5
            assert fips.isdigit()

    def test_values_are_float(self):
        """All extracted values must be floats."""
        html = '{ id: "01001", value: 99999 }'
        result = _parse_map_data(html)
        for val in result.values():
            assert isinstance(val, float)

    def test_ignores_non_5_digit_ids(self):
        """4-digit or 6-digit IDs should not be extracted (regex requires exactly 5)."""
        html = '{ id: "1234", value: 100 } { id: "123456", value: 200 } { id: "12345", value: 300 }'
        result = _parse_map_data(html)
        assert "1234" not in result
        assert "123456" not in result
        assert "12345" in result

    def test_multiple_counties_all_states(self):
        """Should handle a mix of FL, GA, AL county FIPS."""
        html = """
        { id: "01001", value: 1.0 }
        { id: "12001", value: 2.0 }
        { id: "13001", value: 3.0 }
        """
        result = _parse_map_data(html)
        assert len(result) == 3
        assert set(k[:2] for k in result) == {"01", "12", "13"}


class TestFetchRcmsConstants:
    """Tests for module-level constants in fetch_rcms.py."""

    def test_states_are_correct_fips(self):
        """STATES dict must map correct abbreviations to FIPS codes."""
        assert STATES["AL"] == "01"
        assert STATES["FL"] == "12"
        assert STATES["GA"] == "13"

    def test_groups_include_core_traditions(self):
        """GROUPS must include all major religious traditions."""
        assert "evangelical" in GROUPS
        assert "mainline" in GROUPS
        assert "catholic" in GROUPS
        assert "black_protestant" in GROUPS
        assert "all" in GROUPS

    def test_variables_include_key_metrics(self):
        """VARIABLES must include adherents, congregations, and adherence rate."""
        assert "adherents" in VARIABLES
        assert "congregations" in VARIABLES
        assert "adherence_rate" in VARIABLES


# ---------------------------------------------------------------------------
# build_features.py RCMS feature computation tests
# ---------------------------------------------------------------------------


class TestComputeRcmsFeatures:
    """Tests for the compute_rcms_features() function."""

    def test_output_has_required_columns(self, synthetic_rcms):
        """Output must have county_fips, state_abbr, and all RCMS_FEATURE_COLS."""
        result = compute_rcms_features(synthetic_rcms)
        required = {"county_fips", "state_abbr"} | set(RCMS_FEATURE_COLS)
        assert required.issubset(set(result.columns))

    def test_row_count_preserved(self, synthetic_rcms):
        """Output must have the same number of rows as input."""
        result = compute_rcms_features(synthetic_rcms)
        assert len(result) == len(synthetic_rcms)

    def test_shares_are_bounded_zero_to_one(self, synthetic_rcms):
        """All share features must be in [0, 1]."""
        result = compute_rcms_features(synthetic_rcms)
        share_cols = [c for c in RCMS_FEATURE_COLS if c.endswith("_share")]
        for col in share_cols:
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} has values < 0"
            assert (valid <= 1).all(), f"{col} has values > 1"

    def test_shares_sum_leq_one(self, synthetic_rcms):
        """Evangelical + mainline + catholic + black_protestant shares should sum ≤ 1.

        They can be < 1 because "Other" and "Orthodox" traditions are not included
        in the individual group fetches.
        """
        result = compute_rcms_features(synthetic_rcms)
        share_sum = (
            result["evangelical_share"]
            + result["mainline_share"]
            + result["catholic_share"]
            + result["black_protestant_share"]
        )
        # Sum should be ≤ 1 (not all adherents are in these four categories)
        assert (share_sum.dropna() <= 1.0 + 1e-6).all(), (
            f"Share sum exceeds 1.0: max = {share_sum.max():.4f}"
        )

    def test_nan_group_treated_as_zero(self, synthetic_rcms):
        """Counties with NaN for a specific group should have 0 share contribution.

        When adherents_catholic is NaN (county 2 in synthetic data), the
        catholic_share for that county should be 0.0, not NaN.
        """
        result = compute_rcms_features(synthetic_rcms)
        # County at index 2 has NaN for catholic adherents → should be 0 share
        assert result.loc[2, "catholic_share"] == pytest.approx(0.0, abs=1e-9)
        assert not pd.isna(result.loc[2, "catholic_share"])

    def test_congregations_per_1000_is_positive(self, synthetic_rcms):
        """Congregations per 1,000 adherents must be positive for counties with data."""
        result = compute_rcms_features(synthetic_rcms)
        valid = result["congregations_per_1000"].dropna()
        assert (valid > 0).all()

    def test_adherence_rate_preserved(self, synthetic_rcms):
        """religious_adherence_rate must equal the raw ARDA adherence_rate_total."""
        result = compute_rcms_features(synthetic_rcms)
        pd.testing.assert_series_equal(
            result["religious_adherence_rate"].reset_index(drop=True),
            synthetic_rcms["adherence_rate_total"].reset_index(drop=True),
            check_names=False,
        )

    def test_missing_county_produces_nan_shares(self, synthetic_rcms_with_missing_county):
        """Counties with adherents_total=NaN should produce NaN shares (not 0/inf)."""
        result = compute_rcms_features(synthetic_rcms_with_missing_county)
        # County at index 1 (01003) has all NaN input
        assert pd.isna(result.loc[1, "evangelical_share"])
        assert pd.isna(result.loc[1, "catholic_share"])
        assert pd.isna(result.loc[1, "congregations_per_1000"])

    def test_county_fips_passed_through(self, synthetic_rcms):
        """county_fips column must be passed through unchanged."""
        result = compute_rcms_features(synthetic_rcms)
        pd.testing.assert_series_equal(
            result["county_fips"].reset_index(drop=True),
            synthetic_rcms["county_fips"].reset_index(drop=True),
        )


class TestImpute:
    """Tests for the impute_rcms_state_medians() function."""

    def test_imputation_fills_nan_shares(self, synthetic_rcms_with_missing_county):
        """NaN features should be filled after imputation."""
        features = compute_rcms_features(synthetic_rcms_with_missing_county)
        imputed = impute_rcms_state_medians(features)
        # All AL counties → state-level median imputed for county 01003
        assert not pd.isna(imputed.loc[1, "evangelical_share"])

    def test_imputation_uses_state_not_global(self, synthetic_rcms):
        """Imputed values should match state-level median, not global median."""
        # Build a dataset where AL and FL have clearly different distributions
        df = synthetic_rcms.copy()
        # Manually set evangelical_share for a county to NaN to test imputation
        features = compute_rcms_features(df)
        # Set one county to NaN
        features_with_nan = features.copy()
        al_mask = features_with_nan["state_abbr"] == "AL"
        al_indices = features_with_nan[al_mask].index
        test_idx = al_indices[0]
        features_with_nan.loc[test_idx, "evangelical_share"] = np.nan

        imputed = impute_rcms_state_medians(features_with_nan)

        # The imputed value should be the AL median (not FL or GA median)
        al_median = features.loc[al_mask & (features.index != test_idx), "evangelical_share"].median()
        assert imputed.loc[test_idx, "evangelical_share"] == pytest.approx(al_median, rel=1e-3)

    def test_imputation_preserves_non_nan_values(self, synthetic_rcms):
        """Non-NaN values must not be modified by imputation."""
        features = compute_rcms_features(synthetic_rcms)
        features_copy = features.copy()
        imputed = impute_rcms_state_medians(features)

        # Rows without NaN should be identical
        non_nan_mask = features[RCMS_FEATURE_COLS].notna().all(axis=1)
        for col in RCMS_FEATURE_COLS:
            pd.testing.assert_series_equal(
                imputed.loc[non_nan_mask, col],
                features_copy.loc[non_nan_mask, col],
            )

    def test_no_nan_remains_after_imputation(self, synthetic_rcms_with_missing_county):
        """After imputation, no NaN should remain (all counties have state peers)."""
        features = compute_rcms_features(synthetic_rcms_with_missing_county)
        imputed = impute_rcms_state_medians(features)
        # All counties are AL in this fixture, so median from other AL counties is used
        remaining_nan = imputed[RCMS_FEATURE_COLS].isna().sum().sum()
        assert remaining_nan == 0, f"{remaining_nan} NaN values remain after imputation"


class TestRcmsFeatureIntegrity:
    """Integration-style tests that verify the actual saved RCMS parquet file."""

    @pytest.fixture(scope="class")
    def rcms_county_features(self):
        """Load the actual saved county_rcms_features.parquet if it exists."""
        from pathlib import Path
        path = Path(__file__).parents[1] / "data" / "assembled" / "county_rcms_features.parquet"
        if not path.exists():
            pytest.skip("county_rcms_features.parquet not found — run build_features.py first")
        return pd.read_parquet(path)

    def test_covers_all_three_states(self, rcms_county_features):
        """Feature file must contain counties from FL, GA, and AL."""
        states = set(rcms_county_features["state_abbr"])
        assert states == {"FL", "GA", "AL"}

    def test_county_count(self, rcms_county_features):
        """Must have exactly 293 counties (67 AL + 67 FL + 159 GA)."""
        assert len(rcms_county_features) == EXPECTED_TOTAL_COUNTIES, (
            f"Expected {EXPECTED_TOTAL_COUNTIES} counties, got {len(rcms_county_features)}"
        )

    def test_per_state_county_counts(self, rcms_county_features):
        """Each state must have the correct number of counties."""
        counts = rcms_county_features["state_abbr"].value_counts().to_dict()
        for state, expected in EXPECTED_COUNTY_COUNTS.items():
            assert counts.get(state) == expected, (
                f"{state}: expected {expected} counties, got {counts.get(state)}"
            )

    def test_has_all_feature_columns(self, rcms_county_features):
        """All RCMS feature columns must be present."""
        for col in RCMS_FEATURE_COLS:
            assert col in rcms_county_features.columns, f"Missing column: {col}"

    def test_no_nan_in_features(self, rcms_county_features):
        """No NaN values should remain after imputation (file is post-imputation)."""
        nan_counts = rcms_county_features[RCMS_FEATURE_COLS].isna().sum()
        assert nan_counts.sum() == 0, f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}"

    def test_shares_bounded(self, rcms_county_features):
        """All share features must be in [0, 1]."""
        share_cols = [c for c in RCMS_FEATURE_COLS if c.endswith("_share")]
        for col in share_cols:
            assert (rcms_county_features[col] >= 0).all(), f"{col} has values < 0"
            assert (rcms_county_features[col] <= 1).all(), f"{col} has values > 1"

    def test_county_fips_are_five_digits(self, rcms_county_features):
        """County FIPS codes must be 5-digit strings."""
        fips = rcms_county_features["county_fips"]
        assert (fips.str.len() == 5).all(), "Some county FIPS codes are not 5 characters"
        assert fips.str.isdigit().all(), "Some county FIPS codes contain non-digit characters"

    def test_fips_state_prefix_matches_state_abbr(self, rcms_county_features):
        """First 2 digits of county FIPS must match state FIPS codes."""
        fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
        derived = rcms_county_features["county_fips"].str[:2].map(fips_to_abbr)
        mismatches = rcms_county_features["state_abbr"] != derived
        assert not mismatches.any(), (
            f"{mismatches.sum()} rows have mismatched county_fips / state_abbr"
        )

    def test_evangelical_share_is_dominant_in_south(self, rcms_county_features):
        """Median evangelical share should be > 0.5 (typical for Deep South counties)."""
        median_evang = rcms_county_features["evangelical_share"].median()
        assert median_evang > 0.5, (
            f"Median evangelical share = {median_evang:.3f}; expected > 0.5 for Deep South"
        )

    def test_adherence_rate_is_positive(self, rcms_county_features):
        """Adherence rate (adherents per 1,000 residents) must be positive."""
        assert (rcms_county_features["religious_adherence_rate"] > 0).all()

    def test_congregations_per_1000_is_positive(self, rcms_county_features):
        """Congregations per 1,000 adherents must be positive."""
        assert (rcms_county_features["congregations_per_1000"] > 0).all()
