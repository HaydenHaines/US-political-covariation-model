"""Tests for County Health Rankings data fetching and feature computation.

Tests exercise:
1. fetch_county_health_rankings.py — URL construction, CSV parsing, FIPS filtering,
   column extraction, year fallback logic, NaN handling
2. build_county_health_features.py — feature computation, percentage clipping,
   NaN handling, state-median imputation, output schema

These tests use synthetic data and mock HTTP responses so they run without
network access or the real CHR parquet files. Tests verify:
  - CHR URL is constructed with the correct year
  - CSV parsing extracts FIPS codes, county names, and measure values
  - State-scope filtering keeps only FL (12xxx), GA (13xxx), AL (01xxx)
  - State-level summary rows (countycode=000) are excluded
  - Non-5-digit FIPS codes are dropped
  - Measure columns that are missing from the CSV are filled with NaN
  - Percentage features are clipped to [0, 100]
  - State-median imputation fills NaN with correct state peer median
  - Output schema has required columns and types
  - Integration tests check the saved parquet files (skipped if absent)
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_county_health_rankings import (
    CHR_MEASURES,
    CHR_URL_TEMPLATE,
    DEFAULT_YEAR,
    FALLBACK_YEAR,
    OUTPUT_COLUMNS,
    STATES,
    TARGET_STATE_FIPS,
    _find_column,
    build_url,
    fetch_raw_csv,
    parse_chr_csv,
)
from src.assembly.build_county_health_features import (
    CHR_FEATURE_COLS,
    compute_chr_features,
    find_latest_chr_file,
    impute_chr_state_medians,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic CSV generation
# ---------------------------------------------------------------------------


def _make_chr_csv(rows: list[dict], extra_measures: list[str] | None = None) -> str:
    """Build a minimal CHR analytic CSV string for testing.

    The CHR analytic CSV format:
      Row 0: descriptions (skipped by parser using skiprows=1)
      Row 1: column headers (statecode, countycode, county, state, v001_rawvalue, ...)
      Rows 2+: data

    Args:
        rows: List of dicts mapping column name → value.
        extra_measures: Additional measure columns to include beyond CHR_MEASURES.

    Returns:
        CSV string in CHR analytic format (two-header-row style).
    """
    all_measures = list(CHR_MEASURES.keys())
    if extra_measures:
        all_measures += extra_measures

    base_cols = ["statecode", "countycode", "county", "state"] + all_measures

    # Row 0: description row (human-readable; parser skips this)
    desc_row = {col: f"Description of {col}" for col in base_cols}

    # Row 1+: actual data rows
    data_rows = []
    for row in rows:
        full_row = {col: row.get(col, "") for col in base_cols}
        data_rows.append(full_row)

    # Build CSV: description row first, then header+data
    # We output: desc_row as row 0, then a proper CSV with header
    desc_df = pd.DataFrame([desc_row])
    data_df = pd.DataFrame(data_rows)

    buf = StringIO()
    # Write description row (no header)
    desc_df.to_csv(buf, index=False)
    desc_content = buf.getvalue()

    buf2 = StringIO()
    data_df.to_csv(buf2, index=False)
    data_content = buf2.getvalue()

    # CHR format: first row is descriptions, rest is data with header
    # Our parser uses skiprows=1, so row 0 is skipped and row 1 becomes header
    lines_desc = desc_content.strip().split("\n")
    lines_data = data_content.strip().split("\n")

    # First output the description row (without its header), then the data
    combined = lines_desc[1] + "\n" + "\n".join(lines_data)
    return combined


def _make_synthetic_rows() -> list[dict]:
    """Build synthetic CHR data rows for FL, GA, AL counties."""
    return [
        # FL state summary row (should be excluded)
        {
            "statecode": "12", "countycode": "000", "county": "Florida", "state": "FL",
            "v001_rawvalue": "6500", "v009_rawvalue": "0.18", "v011_rawvalue": "0.31",
            "v049_rawvalue": "0.20", "v003_rawvalue": "0.14",
            "v004_rawvalue": "80.0", "v062_rawvalue": "150.0",
            "v063_rawvalue": "55000", "v024_rawvalue": "0.18",
            "v043_rawvalue": "400.0", "v069_rawvalue": "0.35",
            "v070_rawvalue": "0.25", "v042_rawvalue": "0.15",
            "v044_rawvalue": "0.78", "v058_rawvalue": "0.87",
            "v023_rawvalue": "0.62", "v085_rawvalue": "79.0",
            "v060_rawvalue": "0.11", "v036_rawvalue": "4.2",
        },
        # FL county: Alachua
        {
            "statecode": "12", "countycode": "001", "county": "Alachua", "state": "FL",
            "v001_rawvalue": "5800", "v009_rawvalue": "0.16", "v011_rawvalue": "0.28",
            "v049_rawvalue": "0.19", "v003_rawvalue": "0.12",
            "v004_rawvalue": "95.0", "v062_rawvalue": "180.0",
            "v063_rawvalue": "52000", "v024_rawvalue": "0.20",
            "v043_rawvalue": "370.0", "v069_rawvalue": "0.33",
            "v070_rawvalue": "0.22", "v042_rawvalue": "0.13",
            "v044_rawvalue": "0.72", "v058_rawvalue": "0.90",
            "v023_rawvalue": "0.68", "v085_rawvalue": "80.2",
            "v060_rawvalue": "0.09", "v036_rawvalue": "3.9",
        },
        # FL county: Baker (with some NaN measures)
        {
            "statecode": "12", "countycode": "003", "county": "Baker", "state": "FL",
            "v001_rawvalue": "8200", "v009_rawvalue": "0.22", "v011_rawvalue": "0.36",
            "v049_rawvalue": "0.17", "v003_rawvalue": "0.19",
            "v004_rawvalue": "", "v062_rawvalue": "",  # suppressed
            "v063_rawvalue": "44000", "v024_rawvalue": "0.25",
            "v043_rawvalue": "310.0", "v069_rawvalue": "0.40",
            "v070_rawvalue": "0.30", "v042_rawvalue": "0.18",
            "v044_rawvalue": "0.85", "v058_rawvalue": "0.82",
            "v023_rawvalue": "0.50", "v085_rawvalue": "76.5",
            "v060_rawvalue": "0.14", "v036_rawvalue": "4.8",
        },
        # GA county: Appling
        {
            "statecode": "13", "countycode": "001", "county": "Appling", "state": "GA",
            "v001_rawvalue": "9100", "v009_rawvalue": "0.24", "v011_rawvalue": "0.38",
            "v049_rawvalue": "0.15", "v003_rawvalue": "0.21",
            "v004_rawvalue": "45.0", "v062_rawvalue": "90.0",
            "v063_rawvalue": "39000", "v024_rawvalue": "0.28",
            "v043_rawvalue": "450.0", "v069_rawvalue": "0.42",
            "v070_rawvalue": "0.33", "v042_rawvalue": "0.21",
            "v044_rawvalue": "0.80", "v058_rawvalue": "0.80",
            "v023_rawvalue": "0.45", "v085_rawvalue": "74.8",
            "v060_rawvalue": "0.16", "v036_rawvalue": "5.1",
        },
        # GA county: Atkinson
        {
            "statecode": "13", "countycode": "003", "county": "Atkinson", "state": "GA",
            "v001_rawvalue": "10500", "v009_rawvalue": "0.26", "v011_rawvalue": "0.40",
            "v049_rawvalue": "0.13", "v003_rawvalue": "0.25",
            "v004_rawvalue": "30.0", "v062_rawvalue": "60.0",
            "v063_rawvalue": "35000", "v024_rawvalue": "0.32",
            "v043_rawvalue": "520.0", "v069_rawvalue": "0.45",
            "v070_rawvalue": "0.36", "v042_rawvalue": "0.24",
            "v044_rawvalue": "0.78", "v058_rawvalue": "0.76",
            "v023_rawvalue": "0.40", "v085_rawvalue": "73.2",
            "v060_rawvalue": "0.19", "v036_rawvalue": "5.5",
        },
        # AL county: Autauga
        {
            "statecode": "01", "countycode": "001", "county": "Autauga", "state": "AL",
            "v001_rawvalue": "8800", "v009_rawvalue": "0.21", "v011_rawvalue": "0.37",
            "v049_rawvalue": "0.16", "v003_rawvalue": "0.18",
            "v004_rawvalue": "55.0", "v062_rawvalue": "100.0",
            "v063_rawvalue": "48000", "v024_rawvalue": "0.22",
            "v043_rawvalue": "380.0", "v069_rawvalue": "0.38",
            "v070_rawvalue": "0.29", "v042_rawvalue": "0.16",
            "v044_rawvalue": "0.82", "v058_rawvalue": "0.85",
            "v023_rawvalue": "0.56", "v085_rawvalue": "77.1",
            "v060_rawvalue": "0.12", "v036_rawvalue": "4.4",
        },
        # AL county: Baldwin
        {
            "statecode": "01", "countycode": "003", "county": "Baldwin", "state": "AL",
            "v001_rawvalue": "7200", "v009_rawvalue": "0.17", "v011_rawvalue": "0.32",
            "v049_rawvalue": "0.18", "v003_rawvalue": "0.14",
            "v004_rawvalue": "70.0", "v062_rawvalue": "130.0",
            "v063_rawvalue": "56000", "v024_rawvalue": "0.16",
            "v043_rawvalue": "290.0", "v069_rawvalue": "0.34",
            "v070_rawvalue": "0.24", "v042_rawvalue": "0.12",
            "v044_rawvalue": "0.83", "v058_rawvalue": "0.88",
            "v023_rawvalue": "0.60", "v085_rawvalue": "78.5",
            "v060_rawvalue": "0.10", "v036_rawvalue": "4.1",
        },
        # Out-of-scope county: Texas (should be excluded)
        {
            "statecode": "48", "countycode": "201", "county": "Harris", "state": "TX",
            "v001_rawvalue": "7000", "v009_rawvalue": "0.15", "v011_rawvalue": "0.30",
            "v049_rawvalue": "0.18", "v003_rawvalue": "0.20",
            "v004_rawvalue": "90.0", "v062_rawvalue": "160.0",
            "v063_rawvalue": "60000", "v024_rawvalue": "0.18",
            "v043_rawvalue": "500.0", "v069_rawvalue": "0.32",
            "v070_rawvalue": "0.23", "v042_rawvalue": "0.14",
            "v044_rawvalue": "0.79", "v058_rawvalue": "0.86",
            "v023_rawvalue": "0.63", "v085_rawvalue": "79.3",
            "v060_rawvalue": "0.10", "v036_rawvalue": "4.0",
        },
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_csv() -> str:
    """Synthetic CHR CSV string for testing."""
    return _make_chr_csv(_make_synthetic_rows())


@pytest.fixture(scope="module")
def parsed_df(synthetic_csv) -> pd.DataFrame:
    """Parsed DataFrame from synthetic CHR CSV."""
    return parse_chr_csv(synthetic_csv, year=2024)


@pytest.fixture(scope="module")
def features_df(parsed_df) -> pd.DataFrame:
    """Computed CHR features from parsed DataFrame."""
    return compute_chr_features(parsed_df)


@pytest.fixture(scope="module")
def partial_nan_features() -> pd.DataFrame:
    """Synthetic feature DataFrame with known NaN values for imputation tests."""
    rng = np.random.default_rng(99)
    n = 9
    fips = ["12001", "12003", "12005", "13001", "13003", "13005", "01001", "01003", "01005"]
    states = ["FL", "FL", "FL", "GA", "GA", "GA", "AL", "AL", "AL"]

    data = {
        "county_fips": fips,
        "state_abbr": states,
        "data_year": [2024] * n,
    }
    for col in CHR_FEATURE_COLS:
        vals = rng.uniform(0.1, 0.9, size=n).tolist()
        # Introduce NaN at specific positions
        vals[1] = np.nan   # FL county NaN
        vals[4] = np.nan   # GA county NaN
        vals[7] = np.nan   # AL county NaN
        data[col] = vals

    return pd.DataFrame(data)


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
        """TARGET_STATE_FIPS must be the set of FIPS prefixes from STATES."""
        assert TARGET_STATE_FIPS == frozenset(STATES.values())

    def test_chr_measures_count(self):
        """CHR_MEASURES must have at least 15 entries."""
        assert len(CHR_MEASURES) >= 15

    def test_chr_measures_includes_core_health_metrics(self):
        """CHR_MEASURES must include premature death, smoking, obesity, and uninsured."""
        friendly_names = set(CHR_MEASURES.values())
        assert "premature_death_rate" in friendly_names
        assert "adult_smoking_pct" in friendly_names
        assert "adult_obesity_pct" in friendly_names
        assert "uninsured_pct" in friendly_names

    def test_chr_measures_keys_are_rawvalue_format(self):
        """All CHR_MEASURES keys must end in _rawvalue."""
        for key in CHR_MEASURES:
            assert key.endswith("_rawvalue"), f"Key '{key}' does not end in _rawvalue"

    def test_output_columns_includes_required_fields(self):
        """OUTPUT_COLUMNS must include county_fips, state_abbr, county_name, data_year."""
        required = {"county_fips", "state_abbr", "county_name", "data_year"}
        assert required.issubset(set(OUTPUT_COLUMNS))

    def test_chr_feature_cols_count(self):
        """CHR_FEATURE_COLS must have 19 features."""
        assert len(CHR_FEATURE_COLS) == 19

    def test_default_year_and_fallback_differ(self):
        """DEFAULT_YEAR and FALLBACK_YEAR must be different years."""
        assert DEFAULT_YEAR != FALLBACK_YEAR

    def test_fallback_year_is_earlier(self):
        """FALLBACK_YEAR must be earlier than DEFAULT_YEAR."""
        assert FALLBACK_YEAR < DEFAULT_YEAR


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for build_url()."""

    def test_url_contains_year(self):
        """URL must embed the requested year."""
        url = build_url(2024)
        assert "2024" in url

    def test_url_contains_different_year(self):
        """URL with year 2023 must contain '2023'."""
        url = build_url(2023)
        assert "2023" in url

    def test_urls_differ_by_year(self):
        """URLs for different years must differ."""
        assert build_url(2024) != build_url(2023)

    def test_url_uses_template(self):
        """URL must be derived from CHR_URL_TEMPLATE."""
        url = build_url(2024)
        base = CHR_URL_TEMPLATE.split("{year}")[0]
        assert url.startswith(base)

    def test_url_ends_with_csv(self):
        """URL must end with .csv."""
        url = build_url(2024)
        assert url.endswith(".csv")


# ---------------------------------------------------------------------------
# fetch_raw_csv — mocked HTTP
# ---------------------------------------------------------------------------


class TestFetchRawCsv:
    """Tests for fetch_raw_csv() with mocked HTTP responses."""

    def test_returns_string_on_success(self):
        """fetch_raw_csv() must return a string on HTTP 200."""
        mock_response = MagicMock()
        mock_response.text = "col1,col2\nval1,val2"
        mock_response.raise_for_status.return_value = None

        with patch("src.assembly.fetch_county_health_rankings.requests.get", return_value=mock_response):
            result = fetch_raw_csv(2024)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_none_on_http_error(self):
        """fetch_raw_csv() must return None when the request fails."""
        import requests as _requests

        with patch(
            "src.assembly.fetch_county_health_rankings.requests.get",
            side_effect=_requests.RequestException("connection refused"),
        ):
            result = fetch_raw_csv(2024)

        assert result is None

    def test_returns_none_on_404(self):
        """fetch_raw_csv() must return None on 404 (year not available)."""
        import requests as _requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = _requests.HTTPError("404 Not Found")

        with patch("src.assembly.fetch_county_health_rankings.requests.get", return_value=mock_response):
            result = fetch_raw_csv(2025)

        assert result is None


# ---------------------------------------------------------------------------
# parse_chr_csv — CSV parsing
# ---------------------------------------------------------------------------


class TestParseChrCsv:
    """Tests for parse_chr_csv() — raw CSV text → structured DataFrame."""

    def test_output_has_required_columns(self, parsed_df):
        """Output must have county_fips, state_abbr, county_name, data_year."""
        required = {"county_fips", "state_abbr", "county_name", "data_year"}
        assert required.issubset(set(parsed_df.columns))

    def test_county_fips_are_5_digits(self, parsed_df):
        """county_fips must be 5-character digit strings."""
        assert (parsed_df["county_fips"].str.len() == 5).all()
        assert parsed_df["county_fips"].str.isdigit().all()

    def test_state_summary_rows_excluded(self, parsed_df):
        """State-level summary rows (countycode=000) must be excluded."""
        # FL state-level row would have FIPS '12000'
        assert "12000" not in parsed_df["county_fips"].values

    def test_out_of_scope_states_excluded(self, parsed_df):
        """Counties from Texas (48xxx) must be excluded."""
        prefixes = parsed_df["county_fips"].str[:2].unique()
        assert "48" not in prefixes

    def test_fl_ga_al_counties_retained(self, parsed_df):
        """FL (12xxx), GA (13xxx), and AL (01xxx) counties must all be present."""
        prefixes = set(parsed_df["county_fips"].str[:2].unique())
        assert "12" in prefixes
        assert "13" in prefixes
        assert "01" in prefixes

    def test_data_year_column_set(self, parsed_df):
        """data_year column must be set to the year passed to parse_chr_csv."""
        assert (parsed_df["data_year"] == 2024).all()

    def test_measure_values_are_numeric(self, parsed_df):
        """Feature measure columns must be numeric dtype."""
        for col in CHR_FEATURE_COLS:
            assert col in parsed_df.columns, f"Missing feature column: {col}"
            assert pd.api.types.is_float_dtype(parsed_df[col]) or \
                   pd.api.types.is_integer_dtype(parsed_df[col]), \
                f"Column '{col}' is not numeric: dtype={parsed_df[col].dtype}"

    def test_suppressed_values_become_nan(self, parsed_df):
        """Empty-string measure values (suppressed data) must become NaN."""
        # Baker County FL (12003) has empty strings for primary_care_physicians_rate
        baker = parsed_df[parsed_df["county_fips"] == "12003"]
        if len(baker) == 1:
            assert pd.isna(baker.iloc[0]["primary_care_physicians_rate"])
            assert pd.isna(baker.iloc[0]["mental_health_providers_rate"])

    def test_valid_values_parsed_correctly(self, parsed_df):
        """Alachua County FL (12001) values must match the synthetic data."""
        alachua = parsed_df[parsed_df["county_fips"] == "12001"]
        assert len(alachua) == 1
        # premature_death_rate = 5800.0
        assert alachua.iloc[0]["premature_death_rate"] == pytest.approx(5800.0)

    def test_empty_csv_returns_empty_df(self):
        """Empty CSV text must return empty DataFrame with correct schema."""
        result = parse_chr_csv("", year=2024)
        assert len(result) == 0
        assert set(OUTPUT_COLUMNS).issubset(set(result.columns))

    def test_state_abbr_derived_from_fips(self, parsed_df):
        """state_abbr must be derived from county_fips prefix."""
        fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
        derived = parsed_df["county_fips"].str[:2].map(fips_to_abbr)
        pd.testing.assert_series_equal(
            parsed_df["state_abbr"].reset_index(drop=True),
            derived.reset_index(drop=True),
            check_names=False,
        )

    def test_bad_fips_dropped(self):
        """Rows with non-5-digit FIPS must be dropped."""
        rows = [
            {
                "statecode": "1", "countycode": "001", "county": "Weird", "state": "FL",
                **{k: "0.5" for k in CHR_MEASURES},
            },
            {
                "statecode": "12", "countycode": "001", "county": "Alachua", "state": "FL",
                **{k: "0.5" for k in CHR_MEASURES},
            },
        ]
        csv_text = _make_chr_csv(rows)
        result = parse_chr_csv(csv_text, year=2024)
        # Only 5-digit FIPS rows are kept
        assert (result["county_fips"].str.len() == 5).all()


# ---------------------------------------------------------------------------
# FIPS filtering — target states
# ---------------------------------------------------------------------------


class TestFipsFiltering:
    """Tests for FIPS-based state scope filtering in parse_chr_csv."""

    def test_fl_fips_prefix_12_retained(self):
        """FL county FIPS codes starting with '12' must be retained."""
        rows = [{
            "statecode": "12", "countycode": "086", "county": "Miami-Dade", "state": "FL",
            **{k: "0.5" for k in CHR_MEASURES},
        }]
        result = parse_chr_csv(_make_chr_csv(rows), year=2024)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"][:2] == "12"

    def test_ga_fips_prefix_13_retained(self):
        """GA county FIPS codes starting with '13' must be retained."""
        rows = [{
            "statecode": "13", "countycode": "121", "county": "Fulton", "state": "GA",
            **{k: "0.5" for k in CHR_MEASURES},
        }]
        result = parse_chr_csv(_make_chr_csv(rows), year=2024)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"][:2] == "13"

    def test_al_fips_prefix_01_retained(self):
        """AL county FIPS codes starting with '01' must be retained."""
        rows = [{
            "statecode": "01", "countycode": "073", "county": "Jefferson", "state": "AL",
            **{k: "0.5" for k in CHR_MEASURES},
        }]
        result = parse_chr_csv(_make_chr_csv(rows), year=2024)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"][:2] == "01"

    def test_non_target_state_excluded(self):
        """California (06xxx) must be excluded."""
        rows = [{
            "statecode": "06", "countycode": "001", "county": "Alameda", "state": "CA",
            **{k: "0.5" for k in CHR_MEASURES},
        }]
        result = parse_chr_csv(_make_chr_csv(rows), year=2024)
        assert len(result) == 0

    def test_texas_excluded(self):
        """Texas (48xxx) must be excluded."""
        rows = [{
            "statecode": "48", "countycode": "201", "county": "Harris", "state": "TX",
            **{k: "0.5" for k in CHR_MEASURES},
        }]
        result = parse_chr_csv(_make_chr_csv(rows), year=2024)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _find_column — helper function
# ---------------------------------------------------------------------------


class TestFindColumn:
    """Tests for the _find_column() helper."""

    def test_finds_first_match(self):
        """Must return the first candidate found in df.columns."""
        df = pd.DataFrame(columns=["statecode", "countycode", "county"])
        assert _find_column(df, ["statecode", "state_code"]) == "statecode"

    def test_finds_second_candidate(self):
        """Must return the second candidate when first is absent."""
        df = pd.DataFrame(columns=["state_code", "countycode"])
        assert _find_column(df, ["statecode", "state_code"]) == "state_code"

    def test_returns_none_when_none_found(self):
        """Must return None when no candidate is found."""
        df = pd.DataFrame(columns=["col_a", "col_b"])
        assert _find_column(df, ["statecode", "state_code"]) is None

    def test_empty_candidates_returns_none(self):
        """Empty candidates list must return None."""
        df = pd.DataFrame(columns=["statecode"])
        assert _find_column(df, []) is None


# ---------------------------------------------------------------------------
# compute_chr_features — feature computation
# ---------------------------------------------------------------------------


class TestComputeChrFeatures:
    """Tests for compute_chr_features() — DataFrame → feature DataFrame."""

    def test_output_has_required_columns(self, features_df):
        """Output must have county_fips, state_abbr, data_year, and all CHR_FEATURE_COLS."""
        required = {"county_fips", "state_abbr", "data_year"} | set(CHR_FEATURE_COLS)
        assert required.issubset(set(features_df.columns))

    def test_row_count_matches_input(self, parsed_df, features_df):
        """Output must have the same number of rows as input."""
        assert len(features_df) == len(parsed_df)

    def test_county_fips_passed_through(self, parsed_df, features_df):
        """county_fips must be passed through unchanged."""
        pd.testing.assert_series_equal(
            features_df["county_fips"].reset_index(drop=True),
            parsed_df["county_fips"].reset_index(drop=True),
        )

    def test_values_are_numeric(self, features_df):
        """All CHR_FEATURE_COLS must be numeric dtype."""
        for col in CHR_FEATURE_COLS:
            assert pd.api.types.is_float_dtype(features_df[col]) or \
                   pd.api.types.is_integer_dtype(features_df[col]), \
                f"Column '{col}' is not numeric"

    def test_pct_values_clipped_to_100(self):
        """Percentage values exceeding 100 must be clipped to 100."""
        df = pd.DataFrame({
            "county_fips": ["12001"],
            "state_abbr": ["FL"],
            "data_year": [2024],
            "adult_smoking_pct": [120.0],   # impossible but defensive
            "adult_obesity_pct": [0.31],
            "premature_death_rate": [5800.0],
            "excessive_drinking_pct": [0.19],
            "uninsured_pct": [0.12],
            "primary_care_physicians_rate": [95.0],
            "mental_health_providers_rate": [180.0],
            "median_household_income": [52000.0],
            "children_in_poverty_pct": [0.20],
            "violent_crime_rate": [370.0],
            "insufficient_sleep_pct": [0.33],
            "physical_inactivity_pct": [0.22],
            "severe_housing_problems_pct": [0.13],
            "drive_alone_pct": [0.72],
            "high_school_completion_pct": [0.90],
            "some_college_pct": [0.68],
            "life_expectancy": [80.2],
            "diabetes_prevalence_pct": [0.09],
            "poor_mental_health_days": [3.9],
        })
        result = compute_chr_features(df)
        assert result.iloc[0]["adult_smoking_pct"] == pytest.approx(100.0)

    def test_pct_values_clipped_to_zero(self):
        """Negative percentage values must be clipped to 0."""
        df = pd.DataFrame({
            "county_fips": ["12001"],
            "state_abbr": ["FL"],
            "data_year": [2024],
            "adult_smoking_pct": [-5.0],    # impossible but defensive
            "adult_obesity_pct": [0.31],
            "premature_death_rate": [5800.0],
            "excessive_drinking_pct": [0.19],
            "uninsured_pct": [0.12],
            "primary_care_physicians_rate": [95.0],
            "mental_health_providers_rate": [180.0],
            "median_household_income": [52000.0],
            "children_in_poverty_pct": [0.20],
            "violent_crime_rate": [370.0],
            "insufficient_sleep_pct": [0.33],
            "physical_inactivity_pct": [0.22],
            "severe_housing_problems_pct": [0.13],
            "drive_alone_pct": [0.72],
            "high_school_completion_pct": [0.90],
            "some_college_pct": [0.68],
            "life_expectancy": [80.2],
            "diabetes_prevalence_pct": [0.09],
            "poor_mental_health_days": [3.9],
        })
        result = compute_chr_features(df)
        assert result.iloc[0]["adult_smoking_pct"] == pytest.approx(0.0)

    def test_nan_preserved_for_missing_measures(self, features_df):
        """NaN values (suppressed data) must remain NaN in feature output."""
        baker = features_df[features_df["county_fips"] == "12003"]
        if len(baker) == 1:
            assert pd.isna(baker.iloc[0]["primary_care_physicians_rate"])
            assert pd.isna(baker.iloc[0]["mental_health_providers_rate"])

    def test_empty_input_returns_correct_schema(self):
        """Empty input must return empty output with correct schema."""
        result = compute_chr_features(pd.DataFrame())
        required = {"county_fips", "state_abbr", "data_year"} | set(CHR_FEATURE_COLS)
        assert required.issubset(set(result.columns))
        assert len(result) == 0

    def test_missing_feature_column_filled_with_nan(self):
        """If a CHR measure column is absent from input, it must be NaN in output."""
        df = pd.DataFrame({
            "county_fips": ["12001"],
            "state_abbr": ["FL"],
            "data_year": [2024],
            # Deliberately omit all CHR_FEATURE_COLS
        })
        result = compute_chr_features(df)
        for col in CHR_FEATURE_COLS:
            assert col in result.columns
            assert pd.isna(result.iloc[0][col])


# ---------------------------------------------------------------------------
# find_latest_chr_file — file discovery
# ---------------------------------------------------------------------------


class TestFindLatestChrFile:
    """Tests for find_latest_chr_file()."""

    def test_returns_preferred_year_if_exists(self, tmp_path):
        """Must return the preferred year file if it exists."""
        preferred = tmp_path / "chr_2024.parquet"
        fallback = tmp_path / "chr_2023.parquet"
        preferred.touch()
        fallback.touch()
        result = find_latest_chr_file(tmp_path, 2024, 2023)
        assert result == preferred

    def test_falls_back_to_fallback_year(self, tmp_path):
        """Must return the fallback year file when preferred is absent."""
        fallback = tmp_path / "chr_2023.parquet"
        fallback.touch()
        result = find_latest_chr_file(tmp_path, 2024, 2023)
        assert result == fallback

    def test_returns_none_when_no_files(self, tmp_path):
        """Must return None when neither file exists."""
        result = find_latest_chr_file(tmp_path, 2024, 2023)
        assert result is None

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        """Must return None for a directory that doesn't exist."""
        nonexistent = tmp_path / "does_not_exist"
        result = find_latest_chr_file(nonexistent, 2024, 2023)
        assert result is None


# ---------------------------------------------------------------------------
# impute_chr_state_medians — imputation logic
# ---------------------------------------------------------------------------


class TestImputeChrStateMedians:
    """Tests for impute_chr_state_medians()."""

    def test_imputation_fills_nan(self, partial_nan_features):
        """NaN values must be filled after state-median imputation."""
        imputed = impute_chr_state_medians(partial_nan_features)
        # Index 1 is FL county with NaN in all feature cols
        for col in CHR_FEATURE_COLS:
            assert not pd.isna(imputed.loc[1, col]), f"NaN not filled for {col} at index 1"

    def test_imputation_uses_state_median(self, partial_nan_features):
        """Imputed value must match the state median of non-NaN values."""
        imputed = impute_chr_state_medians(partial_nan_features)
        col = "premature_death_rate"
        # FL counties: index 0, 1, 2. Index 1 has NaN. Median of [0, 2] is the expected value.
        fl_mask = partial_nan_features["state_abbr"] == "FL"
        non_nan_mask = partial_nan_features[col].notna()
        fl_median = partial_nan_features.loc[fl_mask & non_nan_mask, col].median()
        assert imputed.loc[1, col] == pytest.approx(fl_median, rel=1e-3)

    def test_imputation_uses_state_not_global(self, partial_nan_features):
        """Imputed FL county must use FL median, not global or GA median."""
        imputed = impute_chr_state_medians(partial_nan_features)
        col = "premature_death_rate"
        fl_mask = partial_nan_features["state_abbr"] == "FL"
        ga_mask = partial_nan_features["state_abbr"] == "GA"
        non_nan = partial_nan_features[col].notna()

        fl_med = partial_nan_features.loc[fl_mask & non_nan, col].median()
        ga_med = partial_nan_features.loc[ga_mask & non_nan, col].median()

        if not np.isclose(fl_med, ga_med):
            assert imputed.loc[1, col] == pytest.approx(fl_med, rel=1e-3)

    def test_imputation_preserves_non_nan_values(self, partial_nan_features):
        """Non-NaN values must not be modified by imputation."""
        original = partial_nan_features.copy()
        imputed = impute_chr_state_medians(partial_nan_features)

        non_nan_mask = partial_nan_features[CHR_FEATURE_COLS[0]].notna()
        for col in CHR_FEATURE_COLS:
            non_nan_mask = partial_nan_features[col].notna()
            pd.testing.assert_series_equal(
                imputed.loc[non_nan_mask, col],
                original.loc[non_nan_mask, col],
            )

    def test_no_nan_after_full_state_imputation(self):
        """After imputation, no NaN remains when all states have non-NaN peers."""
        df = pd.DataFrame({
            "county_fips": ["12001", "12003", "13001", "13003", "01001", "01003"],
            "state_abbr": ["FL", "FL", "GA", "GA", "AL", "AL"],
            "data_year": [2024] * 6,
        })
        rng = np.random.default_rng(0)
        for col in CHR_FEATURE_COLS:
            vals = rng.uniform(0.1, 0.9, size=6)
            # Introduce one NaN per state (one county per state has NaN for this feature)
            vals[1] = np.nan   # FL
            vals[3] = np.nan   # GA
            vals[5] = np.nan   # AL
            df[col] = vals

        imputed = impute_chr_state_medians(df)
        remaining = imputed[CHR_FEATURE_COLS].isna().sum().sum()
        assert remaining == 0, f"{remaining} NaN values remain after imputation"

    def test_state_fips_col_not_in_output(self, partial_nan_features):
        """The temporary state_fips column must be dropped from output."""
        imputed = impute_chr_state_medians(partial_nan_features)
        assert "state_fips" not in imputed.columns


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Tests for the output DataFrame schema."""

    def test_features_df_only_three_states(self, features_df):
        """Features output must contain only FL, GA, and AL counties."""
        prefixes = set(features_df["county_fips"].str[:2].unique())
        assert prefixes <= {"01", "12", "13"}

    def test_fips_state_prefix_matches_state_abbr(self, features_df):
        """FIPS prefix must match state_abbr in features output."""
        fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
        derived = features_df["county_fips"].str[:2].map(fips_to_abbr)
        mismatches = features_df["state_abbr"] != derived.reset_index(drop=True)
        assert not mismatches.any(), f"{mismatches.sum()} FIPS/state_abbr mismatches"

    def test_county_fips_are_five_digit_strings(self, features_df):
        """county_fips must be 5-digit strings throughout the pipeline."""
        assert (features_df["county_fips"].str.len() == 5).all()
        assert features_df["county_fips"].str.isdigit().all()

    def test_no_duplicate_fips_in_features(self, features_df):
        """Each county_fips must appear at most once in the features output."""
        assert features_df["county_fips"].nunique() == len(features_df)


# ---------------------------------------------------------------------------
# Integration tests (skip if data not present)
# ---------------------------------------------------------------------------


class TestChrIntegration:
    """Integration tests that verify the actual saved parquet files (skipped if absent)."""

    @pytest.fixture(scope="class")
    def raw_parquet(self):
        """Load the latest chr_{year}.parquet if it exists."""
        from pathlib import Path
        raw_dir = Path(__file__).parents[1] / "data" / "raw" / "county_health_rankings"
        for year in (2024, 2023):
            path = raw_dir / f"chr_{year}.parquet"
            if path.exists():
                return pd.read_parquet(path)
        pytest.skip("No chr_*.parquet found — run fetch_county_health_rankings.py first")

    @pytest.fixture(scope="class")
    def assembled_parquet(self):
        """Load data/assembled/county_health_features.parquet if it exists."""
        from pathlib import Path
        path = Path(__file__).parents[1] / "data" / "assembled" / "county_health_features.parquet"
        if not path.exists():
            pytest.skip("county_health_features.parquet not found — run build_county_health_features.py first")
        return pd.read_parquet(path)

    @pytest.mark.integration
    def test_raw_has_required_columns(self, raw_parquet):
        """Raw parquet must have county_fips, state_abbr, county_name, data_year."""
        required = {"county_fips", "state_abbr", "county_name", "data_year"}
        assert required.issubset(set(raw_parquet.columns))

    @pytest.mark.integration
    def test_raw_covers_all_three_states(self, raw_parquet):
        """Raw parquet must include counties from FL, GA, and AL."""
        prefixes = raw_parquet["county_fips"].str[:2].unique()
        assert "12" in prefixes, "No FL counties in raw parquet"
        assert "13" in prefixes, "No GA counties in raw parquet"
        assert "01" in prefixes, "No AL counties in raw parquet"

    @pytest.mark.integration
    def test_raw_fips_are_five_digits(self, raw_parquet):
        """All FIPS codes in raw file must be 5-digit strings."""
        assert (raw_parquet["county_fips"].str.len() == 5).all()
        assert raw_parquet["county_fips"].str.isdigit().all()

    @pytest.mark.integration
    def test_raw_no_state_summary_rows(self, raw_parquet):
        """State summary rows (FIPS ending in 000) must not be present."""
        ends_in_000 = raw_parquet["county_fips"].str.endswith("000")
        assert not ends_in_000.any(), "State summary rows found in raw parquet"

    @pytest.mark.integration
    def test_raw_county_count_plausible(self, raw_parquet):
        """FL+GA+AL should have 226 counties total."""
        # FL=67, GA=159, AL=67 = 293 total... but note CHR may not have all
        # Accept anything between 200 and 293
        n = len(raw_parquet)
        assert 200 <= n <= 300, f"Unexpected county count: {n}"

    @pytest.mark.integration
    def test_assembled_has_required_columns(self, assembled_parquet):
        """Assembled parquet must have county_fips, state_abbr, and CHR_FEATURE_COLS."""
        required = {"county_fips", "state_abbr"} | set(CHR_FEATURE_COLS)
        assert required.issubset(set(assembled_parquet.columns))

    @pytest.mark.integration
    def test_assembled_covers_all_three_states(self, assembled_parquet):
        """Assembled features must include counties from all three states."""
        states = set(assembled_parquet["state_abbr"].unique())
        assert states == {"FL", "GA", "AL"}, f"Expected FL/GA/AL, got: {states}"

    @pytest.mark.integration
    def test_assembled_fips_state_prefix_matches_abbr(self, assembled_parquet):
        """FIPS prefixes must match state_abbr values in assembled file."""
        fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
        derived = assembled_parquet["county_fips"].str[:2].map(fips_to_abbr)
        mismatches = assembled_parquet["state_abbr"] != derived
        assert not mismatches.any(), f"{mismatches.sum()} FIPS/state_abbr mismatches"
