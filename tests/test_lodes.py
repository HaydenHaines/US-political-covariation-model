"""Tests for LODES commuting flow data fetching.

Tests exercise:
1. fetch_lodes.py — URL construction, county FIPS extraction, chunk aggregation,
   flow filtering

These tests use synthetic DataFrames and do not make any network calls.
They verify:
  - URL construction for each state/year combination
  - County FIPS extraction from 15-digit block FIPS (with zero-padding)
  - Chunk aggregation groups and sums S000 correctly
  - Flow filtering keeps only target-state flows and removes intra-county flows
  - Edge cases: empty DataFrames, non-target state flows, integer geocodes

Integration tests check the actual saved parquet file (skipped if absent).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembly.fetch_lodes import (
    BASE_URL,
    DEFAULT_YEARS,
    STATES,
    TARGET_STATE_FIPS,
    aggregate_chunk,
    build_url,
    extract_county_fips,
    filter_commuting_flows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_od_chunk(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal OD chunk DataFrame from a list of row dicts.

    Fills in defaults for omitted columns so tests can be concise.
    Mimics the schema produced by pandas.read_csv() on a LODES OD file.
    """
    default = {
        "w_geocode": "120860001001000",  # FL Miami-Dade work block
        "h_geocode": "130890001001000",  # GA Muscogee home block
        "S000": 10,
    }
    records = [{**default, **r} for r in rows]
    return pd.DataFrame(records)


def _make_agg_df(rows: list[dict]) -> pd.DataFrame:
    """Build an aggregated commuting flow DataFrame for filter tests."""
    default = {
        "home_fips": "12086",
        "work_fips": "13089",
        "total_jobs": 100,
    }
    records = [{**default, **r} for r in rows]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for build_url()."""

    def test_fl_2021_url(self):
        """FL 2021 URL must point to the correct census.gov path."""
        url = build_url("fl", 2021)
        assert url == "https://lehd.ces.census.gov/data/lodes/LODES8/fl/od/fl_od_main_JT00_2021.csv.gz"

    def test_ga_2020_url(self):
        """GA 2020 URL must use lowercase state abbreviation."""
        url = build_url("ga", 2020)
        assert url == "https://lehd.ces.census.gov/data/lodes/LODES8/ga/od/ga_od_main_JT00_2020.csv.gz"

    def test_al_2022_url(self):
        """AL 2022 URL must embed state twice (path and filename)."""
        url = build_url("al", 2022)
        assert url == "https://lehd.ces.census.gov/data/lodes/LODES8/al/od/al_od_main_JT00_2022.csv.gz"

    def test_url_contains_base_url(self):
        """All URLs must be on the LEHD census.gov domain."""
        url = build_url("fl", 2021)
        assert BASE_URL in url

    def test_url_ends_with_csv_gz(self):
        """URL must end in .csv.gz (gzipped CSV)."""
        url = build_url("fl", 2021)
        assert url.endswith(".csv.gz")

    def test_url_contains_od_main_jt00(self):
        """URL must reference the OD main all-jobs file (JT00)."""
        url = build_url("fl", 2021)
        assert "od_main_JT00" in url

    def test_url_contains_year(self):
        """Year must appear literally in the URL filename."""
        for year in [2020, 2021, 2022]:
            url = build_url("fl", year)
            assert str(year) in url

    def test_all_states_produce_unique_urls(self):
        """Each state/year combination must produce a unique URL."""
        urls = [
            build_url(state_lower, year)
            for _, (_, state_lower) in STATES.items()
            for year in DEFAULT_YEARS
        ]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    def test_state_appears_in_path_and_filename(self):
        """State abbreviation must appear in both the directory path and filename."""
        url = build_url("fl", 2021)
        # .../LODES8/fl/od/fl_od_main_JT00_2021.csv.gz
        parts = url.split("/")
        # "fl" should appear as a directory segment and in the filename
        assert "fl" in parts  # directory
        assert parts[-1].startswith("fl_")  # filename


# ---------------------------------------------------------------------------
# County FIPS extraction
# ---------------------------------------------------------------------------


class TestExtractCountyFips:
    """Tests for extract_county_fips()."""

    def test_standard_15_digit_string(self):
        """15-digit geocode must return first 5 digits."""
        assert extract_county_fips("120860001001000") == "12086"

    def test_ga_county(self):
        """GA Muscogee county block must return '13089'."""
        assert extract_county_fips("130890001001000") == "13089"

    def test_al_county(self):
        """AL Jefferson county block must return '01073'."""
        assert extract_county_fips("010730001001000") == "01073"

    def test_integer_input_zero_padded(self):
        """Integer geocodes must be zero-padded to 15 digits before extraction."""
        # 1001001001000 is 13 digits — zero-padded to 001001001001000 → county "00100"
        # Use a clearly padded case: leading zeros get added
        result = extract_county_fips(120860001001000)  # int, 15 digits
        assert result == "12086"

    def test_short_integer_gets_zero_padded(self):
        """Short integer geocodes must be zero-padded to 15 digits."""
        # 1073001001000 → str "1073001001000" (13 chars) → zfill(15) → "001073001001000" → "00107"
        result = extract_county_fips(1073001001000)
        assert len(result) == 5
        assert result.isdigit()

    def test_output_is_string(self):
        """extract_county_fips must always return a string."""
        result = extract_county_fips("120860001001000")
        assert isinstance(result, str)

    def test_output_length_is_5(self):
        """Output must always be exactly 5 characters."""
        assert len(extract_county_fips("120860001001000")) == 5
        assert len(extract_county_fips("010730001001000")) == 5
        assert len(extract_county_fips("130890001001000")) == 5

    def test_output_is_digits(self):
        """Output must contain only digit characters."""
        result = extract_county_fips("120860001001000")
        assert result.isdigit()

    def test_leading_zero_preserved(self):
        """FIPS codes with leading zeros (e.g., AL = '01') must preserve them."""
        # AL block: starts with "01"
        result = extract_county_fips("010730001001000")
        assert result.startswith("01")


# ---------------------------------------------------------------------------
# Chunk aggregation
# ---------------------------------------------------------------------------


class TestAggregateChunk:
    """Tests for aggregate_chunk()."""

    def test_basic_aggregation(self):
        """Single row chunk must aggregate to one home/work county pair."""
        chunk = _make_od_chunk([{"w_geocode": "120860001001000", "h_geocode": "130890001001000", "S000": 25}])
        result = aggregate_chunk(chunk)
        assert len(result) == 1
        assert result.iloc[0]["home_fips"] == "13089"
        assert result.iloc[0]["work_fips"] == "12086"
        assert result.iloc[0]["total_jobs"] == 25

    def test_sums_same_county_pair(self):
        """Multiple blocks in the same county pair must be summed."""
        chunk = _make_od_chunk([
            {"w_geocode": "120860001001000", "h_geocode": "130890001001000", "S000": 10},
            {"w_geocode": "120860001002000", "h_geocode": "130890001002000", "S000": 15},
            {"w_geocode": "120860001003000", "h_geocode": "130890001003000", "S000": 5},
        ])
        result = aggregate_chunk(chunk)
        assert len(result) == 1
        assert result.iloc[0]["total_jobs"] == 30

    def test_separate_rows_for_different_county_pairs(self):
        """Different county pairs must produce separate rows."""
        chunk = _make_od_chunk([
            {"w_geocode": "120860001001000", "h_geocode": "130890001001000", "S000": 10},
            {"w_geocode": "010730001001000", "h_geocode": "120860001001000", "S000": 20},
        ])
        result = aggregate_chunk(chunk)
        assert len(result) == 2

    def test_output_columns(self):
        """Output must have exactly home_fips, work_fips, total_jobs."""
        chunk = _make_od_chunk([{}])
        result = aggregate_chunk(chunk)
        assert set(result.columns) == {"home_fips", "work_fips", "total_jobs"}

    def test_s000_renamed_to_total_jobs(self):
        """S000 column must be renamed to total_jobs in output."""
        chunk = _make_od_chunk([{"S000": 42}])
        result = aggregate_chunk(chunk)
        assert "total_jobs" in result.columns
        assert "S000" not in result.columns

    def test_county_fips_are_5_digits(self):
        """home_fips and work_fips must be 5-digit strings."""
        chunk = _make_od_chunk([
            {"w_geocode": "120860001001000", "h_geocode": "010730001001000", "S000": 5},
        ])
        result = aggregate_chunk(chunk)
        assert len(result.iloc[0]["home_fips"]) == 5
        assert len(result.iloc[0]["work_fips"]) == 5

    def test_does_not_modify_original_chunk(self):
        """aggregate_chunk must not mutate the input DataFrame."""
        chunk = _make_od_chunk([{"S000": 10}])
        original_cols = list(chunk.columns)
        _ = aggregate_chunk(chunk)
        assert list(chunk.columns) == original_cols

    def test_empty_chunk_returns_empty(self):
        """Empty input chunk must return an empty DataFrame."""
        chunk = pd.DataFrame(columns=["w_geocode", "h_geocode", "S000"])
        result = aggregate_chunk(chunk)
        assert len(result) == 0

    def test_multiple_home_counties_same_work(self):
        """Workers commuting from different home counties to same work county are separate rows."""
        chunk = _make_od_chunk([
            {"w_geocode": "120860001001000", "h_geocode": "130890001001000", "S000": 10},
            {"w_geocode": "120860001001000", "h_geocode": "010730001001000", "S000": 8},
        ])
        result = aggregate_chunk(chunk)
        assert len(result) == 2
        assert result["total_jobs"].sum() == 18


# ---------------------------------------------------------------------------
# Flow filtering
# ---------------------------------------------------------------------------


class TestFilterCommutingFlows:
    """Tests for filter_commuting_flows()."""

    def test_keeps_flow_into_fl(self):
        """Flows where work county is in FL must be kept."""
        df = _make_agg_df([{"home_fips": "48001", "work_fips": "12086", "total_jobs": 50}])
        result = filter_commuting_flows(df)
        assert len(result) == 1

    def test_keeps_flow_out_of_ga(self):
        """Flows where home county is in GA must be kept."""
        df = _make_agg_df([{"home_fips": "13089", "work_fips": "48001", "total_jobs": 30}])
        result = filter_commuting_flows(df)
        assert len(result) == 1

    def test_keeps_flow_involving_al(self):
        """Flows where either end is in AL must be kept."""
        df = _make_agg_df([{"home_fips": "01073", "work_fips": "48001", "total_jobs": 20}])
        result = filter_commuting_flows(df)
        assert len(result) == 1

    def test_drops_flow_between_non_target_states(self):
        """Flows where neither end is FL/GA/AL must be dropped."""
        df = _make_agg_df([
            {"home_fips": "48001", "work_fips": "06037", "total_jobs": 100},
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 0

    def test_drops_intra_county_flow(self):
        """Flows where home_fips == work_fips must be dropped."""
        df = _make_agg_df([
            {"home_fips": "12086", "work_fips": "12086", "total_jobs": 200},
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 0

    def test_keeps_inter_county_same_state(self):
        """Flows between different counties in the same target state must be kept."""
        df = _make_agg_df([
            {"home_fips": "12086", "work_fips": "12011", "total_jobs": 75},
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 1

    def test_all_three_target_states_accepted(self):
        """Flows involving FL (12), GA (13), or AL (01) are all kept."""
        df = _make_agg_df([
            {"home_fips": "48001", "work_fips": "12086", "total_jobs": 10},  # → FL
            {"home_fips": "48001", "work_fips": "13089", "total_jobs": 10},  # → GA
            {"home_fips": "48001", "work_fips": "01073", "total_jobs": 10},  # → AL
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 3

    def test_mixed_target_and_non_target(self):
        """Only target-state flows kept; non-target dropped."""
        df = _make_agg_df([
            {"home_fips": "12086", "work_fips": "13089", "total_jobs": 50},  # FL→GA: keep
            {"home_fips": "36061", "work_fips": "06037", "total_jobs": 80},  # NY→CA: drop
            {"home_fips": "01073", "work_fips": "48001", "total_jobs": 30},  # AL→TX: keep
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 2

    def test_intra_county_in_target_state_dropped(self):
        """Intra-county flows in target states must also be dropped."""
        df = _make_agg_df([
            {"home_fips": "12086", "work_fips": "12086", "total_jobs": 500},  # FL intra: drop
            {"home_fips": "12086", "work_fips": "13089", "total_jobs": 50},   # FL→GA: keep
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 1
        assert result.iloc[0]["work_fips"] == "13089"

    def test_empty_input_returns_empty(self):
        """Empty input DataFrame must produce empty output."""
        df = pd.DataFrame(columns=["home_fips", "work_fips", "total_jobs"])
        result = filter_commuting_flows(df)
        assert len(result) == 0

    def test_all_non_target_returns_empty(self):
        """Input with no FL/GA/AL flows must produce empty output."""
        df = _make_agg_df([
            {"home_fips": "36061", "work_fips": "06037", "total_jobs": 100},
            {"home_fips": "48001", "work_fips": "17031", "total_jobs": 50},
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 0

    def test_all_intra_county_returns_empty(self):
        """Input containing only intra-county flows must produce empty output."""
        df = _make_agg_df([
            {"home_fips": "12086", "work_fips": "12086", "total_jobs": 100},
            {"home_fips": "13089", "work_fips": "13089", "total_jobs": 200},
        ])
        result = filter_commuting_flows(df)
        assert len(result) == 0

    def test_output_index_reset(self):
        """Output DataFrame index must be reset (0-based sequential)."""
        df = _make_agg_df([
            {"home_fips": "12086", "work_fips": "13089", "total_jobs": 50},
            {"home_fips": "48001", "work_fips": "06037", "total_jobs": 80},  # will be dropped
            {"home_fips": "01073", "work_fips": "13089", "total_jobs": 30},
        ])
        result = filter_commuting_flows(df)
        assert list(result.index) == list(range(len(result)))

    def test_output_columns_preserved(self):
        """filter_commuting_flows must not add or remove columns."""
        df = _make_agg_df([{"home_fips": "12086", "work_fips": "13089", "total_jobs": 50}])
        result = filter_commuting_flows(df)
        assert set(result.columns) == {"home_fips", "work_fips", "total_jobs"}

    def test_extra_column_preserved(self):
        """Extra columns (e.g., year) must be preserved through filtering."""
        df = _make_agg_df([{"home_fips": "12086", "work_fips": "13089", "total_jobs": 50}])
        df["year"] = 2021
        result = filter_commuting_flows(df)
        assert "year" in result.columns
        assert result.iloc[0]["year"] == 2021


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants in fetch_lodes.py."""

    def test_states_contains_fl_ga_al(self):
        """STATES must contain entries for FL, GA, and AL."""
        assert "FL" in STATES
        assert "GA" in STATES
        assert "AL" in STATES

    def test_fl_fips_is_12(self):
        """FL state FIPS prefix must be '12'."""
        assert STATES["FL"][0] == "12"

    def test_ga_fips_is_13(self):
        """GA state FIPS prefix must be '13'."""
        assert STATES["GA"][0] == "13"

    def test_al_fips_is_01(self):
        """AL state FIPS prefix must be '01'."""
        assert STATES["AL"][0] == "01"

    def test_state_lower_values(self):
        """STATES lowercase abbreviations must be 'al', 'fl', 'ga'."""
        lowers = {abbr: lower for abbr, (_, lower) in STATES.items()}
        assert lowers["FL"] == "fl"
        assert lowers["GA"] == "ga"
        assert lowers["AL"] == "al"

    def test_target_state_fips_matches_states(self):
        """TARGET_STATE_FIPS must be the frozenset of FIPS prefixes from STATES."""
        expected = frozenset(fips for fips, _ in STATES.values())
        assert TARGET_STATE_FIPS == expected

    def test_target_state_fips_contains_correct_codes(self):
        """TARGET_STATE_FIPS must contain '01', '12', '13'."""
        assert "01" in TARGET_STATE_FIPS
        assert "12" in TARGET_STATE_FIPS
        assert "13" in TARGET_STATE_FIPS

    def test_default_years_has_three_entries(self):
        """DEFAULT_YEARS must contain exactly 3 years for MVP."""
        assert len(DEFAULT_YEARS) == 3

    def test_default_years_are_recent(self):
        """DEFAULT_YEARS must include 2020, 2021, 2022."""
        assert 2020 in DEFAULT_YEARS
        assert 2021 in DEFAULT_YEARS
        assert 2022 in DEFAULT_YEARS

    def test_base_url_is_census_gov(self):
        """BASE_URL must point to the census.gov LEHD LODES8 endpoint."""
        assert "census.gov" in BASE_URL
        assert "LODES8" in BASE_URL


# ---------------------------------------------------------------------------
# Integration tests (skip if data not present)
# ---------------------------------------------------------------------------


class TestLodesIntegration:
    """Integration tests that verify the actual saved lodes_commuting.parquet."""

    @pytest.fixture(scope="class")
    def lodes_parquet(self):
        """Load the actual saved lodes_commuting.parquet if it exists."""
        from pathlib import Path

        path = Path(__file__).parents[1] / "data" / "raw" / "lodes_commuting.parquet"
        if not path.exists():
            pytest.skip("lodes_commuting.parquet not found — run fetch_lodes.py first")
        return pd.read_parquet(path)

    def test_has_required_columns(self, lodes_parquet):
        """Parquet must have all required output columns."""
        required = {"home_fips", "work_fips", "total_jobs", "year"}
        assert required.issubset(set(lodes_parquet.columns))

    def test_fips_are_5_digits(self, lodes_parquet):
        """All FIPS codes in the saved file must be 5-digit strings."""
        for col in ("home_fips", "work_fips"):
            assert (lodes_parquet[col].str.len() == 5).all(), f"{col} has non-5-digit values"
            assert lodes_parquet[col].str.isdigit().all(), f"{col} has non-digit characters"

    def test_no_intra_county_flows(self, lodes_parquet):
        """No row must have home_fips == work_fips (intra-county filtered out)."""
        self_loops = lodes_parquet["home_fips"] == lodes_parquet["work_fips"]
        assert not self_loops.any(), f"{self_loops.sum()} intra-county flows found"

    def test_target_state_coverage(self, lodes_parquet):
        """Each of FL, GA, AL must appear as home or work county."""
        for fips_prefix in ("01", "12", "13"):
            in_home = lodes_parquet["home_fips"].str.startswith(fips_prefix).any()
            in_work = lodes_parquet["work_fips"].str.startswith(fips_prefix).any()
            assert in_home or in_work, f"No flows found for state FIPS {fips_prefix}"

    def test_total_jobs_positive(self, lodes_parquet):
        """total_jobs must be positive for all rows."""
        assert (lodes_parquet["total_jobs"] > 0).all(), "Some rows have non-positive total_jobs"

    def test_years_present(self, lodes_parquet):
        """Saved file must contain data for all DEFAULT_YEARS."""
        actual_years = set(lodes_parquet["year"].unique())
        assert set(DEFAULT_YEARS) <= actual_years
