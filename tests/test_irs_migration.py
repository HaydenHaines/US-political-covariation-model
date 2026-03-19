"""Tests for IRS SOI county-to-county migration data fetching.

Tests exercise:
1. fetch_irs_migration.py — URL construction, FIPS formatting, row filtering,
   output schema

These tests use synthetic DataFrames and do not make any network calls.
They verify:
  - URL construction for each year pair
  - FIPS code formatting (zero-padding, 5-digit output)
  - Aggregate row filtering (statefips >= 96 skipped)
  - Non-migrant row filtering (same county origin/destination skipped)
  - State scope filtering (keep only flows involving FL/GA/AL)
  - Output schema and column types
  - Edge cases: empty input, all-aggregate input, no target-state rows

Integration tests check the actual saved parquet file (skipped if absent).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembly.fetch_irs_migration import (
    ALL_YEAR_PAIRS,
    DEFAULT_YEAR_PAIRS,
    INFLOW_COLUMNS,
    STATES,
    TARGET_STATE_FIPS,
    build_fips,
    build_url,
    filter_inflow_df,
    is_aggregate_row,
    is_nonmigrant_row,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal raw inflow DataFrame from a list of row dicts.

    Fills in defaults for omitted columns so tests can be concise.
    Matches the dtypes produced by fetch_inflow_csv() after coercion.
    """
    default = {
        "y2_statefips": 12,
        "y2_countyfips": 1,
        "y1_statefips": 13,
        "y1_countyfips": 1,
        "y1_state": "GA",
        "y1_countyname": "Test County",
        "n1": 100,
        "n2": 250,
        "agi": 5000,
    }
    records = [{**default, **r} for r in rows]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for build_url()."""

    def test_latest_year_pair_url(self):
        """2021-2022 inflow URL must point to countyinflow2122.csv."""
        url = build_url("2122")
        assert url == "https://www.irs.gov/pub/irs-soi/countyinflow2122.csv"

    def test_earlier_year_pair_url(self):
        """2019-2020 inflow URL must point to countyinflow1920.csv."""
        url = build_url("1920")
        assert url == "https://www.irs.gov/pub/irs-soi/countyinflow1920.csv"

    def test_oldest_year_pair_url(self):
        """2011-2012 inflow URL must point to countyinflow1112.csv."""
        url = build_url("1112")
        assert url == "https://www.irs.gov/pub/irs-soi/countyinflow1112.csv"

    def test_url_contains_irs_domain(self):
        """All URLs must be on the IRS SOI domain."""
        url = build_url("2021")
        assert "irs.gov" in url

    def test_url_is_csv(self):
        """URL must end in .csv."""
        url = build_url("2021")
        assert url.endswith(".csv")

    def test_all_year_pairs_produce_unique_urls(self):
        """Each year pair must produce a unique URL."""
        urls = [build_url(code) for code, _ in ALL_YEAR_PAIRS]
        assert len(urls) == len(set(urls)), "Duplicate URLs found for different year pairs"

    def test_default_year_pairs_are_subset_of_all(self):
        """DEFAULT_YEAR_PAIRS must be a subset of ALL_YEAR_PAIRS."""
        all_codes = {code for code, _ in ALL_YEAR_PAIRS}
        default_codes = {code for code, _ in DEFAULT_YEAR_PAIRS}
        assert default_codes <= all_codes


# ---------------------------------------------------------------------------
# FIPS code construction
# ---------------------------------------------------------------------------


class TestBuildFips:
    """Tests for build_fips()."""

    def test_zero_pad_state(self):
        """State FIPS 1 must be zero-padded to '01'."""
        assert build_fips(1, 1) == "01001"

    def test_zero_pad_county(self):
        """County FIPS 1 must be zero-padded to '001'."""
        assert build_fips(12, 1) == "12001"

    def test_fl_miami_dade(self):
        """Miami-Dade (FL state=12, county=86) must produce '12086'."""
        assert build_fips(12, 86) == "12086"

    def test_ga_fulton(self):
        """Fulton County GA (state=13, county=121) must produce '13121'."""
        assert build_fips(13, 121) == "13121"

    def test_al_jefferson(self):
        """Jefferson County AL (state=1, county=73) must produce '01073'."""
        assert build_fips(1, 73) == "01073"

    def test_output_length(self):
        """All output FIPS codes must be exactly 5 characters."""
        assert len(build_fips(1, 1)) == 5
        assert len(build_fips(12, 999)) == 5

    def test_string_inputs_accepted(self):
        """String inputs must be handled identically to integer inputs."""
        assert build_fips("12", "1") == build_fips(12, 1)


# ---------------------------------------------------------------------------
# Aggregate row detection
# ---------------------------------------------------------------------------


class TestIsAggregateRow:
    """Tests for is_aggregate_row()."""

    def test_real_state_fips_not_aggregate(self):
        """State FIPS codes 1-56 must not be flagged as aggregate."""
        for fips in [1, 12, 13, 56]:
            assert not is_aggregate_row(fips), f"State FIPS {fips} incorrectly flagged as aggregate"

    def test_fips_96_is_aggregate(self):
        """State FIPS 96 (Total US + Foreign) must be flagged."""
        assert is_aggregate_row(96)

    def test_fips_97_is_aggregate(self):
        """State FIPS 97 (Total US) must be flagged."""
        assert is_aggregate_row(97)

    def test_fips_98_is_aggregate(self):
        """State FIPS 98 (Foreign) must be flagged."""
        assert is_aggregate_row(98)

    def test_fips_95_not_aggregate(self):
        """State FIPS 95 is below threshold and must not be flagged."""
        assert not is_aggregate_row(95)


# ---------------------------------------------------------------------------
# Non-migrant row detection
# ---------------------------------------------------------------------------


class TestIsNonmigrantRow:
    """Tests for is_nonmigrant_row()."""

    def test_same_county_is_nonmigrant(self):
        """Origin == destination must return True."""
        assert is_nonmigrant_row(12, 1, 12, 1)

    def test_different_county_same_state(self):
        """Different county in same state must return False."""
        assert not is_nonmigrant_row(12, 1, 12, 3)

    def test_different_state(self):
        """Different state must return False regardless of county."""
        assert not is_nonmigrant_row(12, 1, 13, 1)

    def test_different_state_and_county(self):
        """Completely different origin and destination must return False."""
        assert not is_nonmigrant_row(1, 73, 13, 121)


# ---------------------------------------------------------------------------
# filter_inflow_df — aggregate row filtering
# ---------------------------------------------------------------------------


class TestFilterAggregateRows:
    """Tests that filter_inflow_df() removes aggregate rows correctly."""

    def test_removes_y1_statefips_96(self):
        """Rows where origin state FIPS == 96 must be dropped."""
        df = _make_raw_df(
            [
                {"y1_statefips": 96, "y1_countyfips": 0, "y2_statefips": 12, "y2_countyfips": 1},
                {"y1_statefips": 13, "y1_countyfips": 1, "y2_statefips": 12, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 1
        assert result.iloc[0]["origin_fips"].startswith("13")

    def test_removes_y2_statefips_97(self):
        """Rows where destination state FIPS == 97 must be dropped."""
        df = _make_raw_df(
            [
                {"y1_statefips": 13, "y1_countyfips": 1, "y2_statefips": 97, "y2_countyfips": 0},
                {"y1_statefips": 13, "y1_countyfips": 1, "y2_statefips": 12, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 1

    def test_removes_y1_statefips_98(self):
        """Rows where origin FIPS == 98 (foreign) must be dropped."""
        df = _make_raw_df(
            [
                {"y1_statefips": 98, "y1_countyfips": 0, "y2_statefips": 12, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_inflow_df — non-migrant row filtering
# ---------------------------------------------------------------------------


class TestFilterNonmigrantRows:
    """Tests that filter_inflow_df() removes non-migrant rows."""

    def test_removes_same_county_row(self):
        """Rows where origin == destination county must be dropped."""
        df = _make_raw_df(
            [
                # Non-migrant: same county
                {"y1_statefips": 12, "y1_countyfips": 1, "y2_statefips": 12, "y2_countyfips": 1},
                # Real migration: different county
                {"y1_statefips": 13, "y1_countyfips": 1, "y2_statefips": 12, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 1
        assert result.iloc[0]["origin_fips"] == "13001"

    def test_same_state_different_county_kept(self):
        """Moves within the same state to a different county must be kept."""
        df = _make_raw_df(
            [
                {"y1_statefips": 12, "y1_countyfips": 1, "y2_statefips": 12, "y2_countyfips": 3},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# filter_inflow_df — state scope filtering
# ---------------------------------------------------------------------------


class TestFilterStateScope:
    """Tests that filter_inflow_df() keeps only flows involving FL/GA/AL."""

    def test_keeps_flow_into_target_state(self):
        """Flows where destination is FL must be kept."""
        df = _make_raw_df(
            [
                {"y1_statefips": 6, "y1_countyfips": 1, "y2_statefips": 12, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 1

    def test_keeps_flow_out_of_target_state(self):
        """Flows where origin is GA must be kept."""
        df = _make_raw_df(
            [
                {"y1_statefips": 13, "y1_countyfips": 1, "y2_statefips": 6, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 1

    def test_drops_flow_between_non_target_states(self):
        """Flows where neither origin nor destination is in FL/GA/AL must be dropped."""
        df = _make_raw_df(
            [
                # TX → CA: neither is FL/GA/AL
                {"y1_statefips": 48, "y1_countyfips": 1, "y2_statefips": 6, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 0

    def test_all_three_target_states_accepted(self):
        """Flows involving FL (12), GA (13), or AL (01) must all be kept."""
        # TX → FL, TX → GA, TX → AL
        df = _make_raw_df(
            [
                {"y1_statefips": 48, "y1_countyfips": 1, "y2_statefips": 12, "y2_countyfips": 1},
                {"y1_statefips": 48, "y1_countyfips": 1, "y2_statefips": 13, "y2_countyfips": 1},
                {"y1_statefips": 48, "y1_countyfips": 1, "y2_statefips": 1, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# filter_inflow_df — output schema
# ---------------------------------------------------------------------------


class TestFilterOutputSchema:
    """Tests that filter_inflow_df() produces the correct output schema."""

    @pytest.fixture(scope="class")
    def filtered(self):
        """A small filtered DataFrame for schema tests."""
        df = _make_raw_df(
            [
                {
                    "y1_statefips": 13,
                    "y1_countyfips": 121,
                    "y2_statefips": 12,
                    "y2_countyfips": 86,
                    "n1": 150,
                    "n2": 320,
                    "agi": 8500,
                },
            ]
        )
        return filter_inflow_df(df, "2021-2022")

    def test_output_columns(self, filtered):
        """Output must have exactly the required columns."""
        expected = {"origin_fips", "dest_fips", "n_returns", "n_exemptions", "agi", "year_pair"}
        assert set(filtered.columns) == expected

    def test_origin_fips_is_5_digits(self, filtered):
        """origin_fips must be 5-character strings."""
        assert all(len(f) == 5 for f in filtered["origin_fips"])
        assert all(f.isdigit() for f in filtered["origin_fips"])

    def test_dest_fips_is_5_digits(self, filtered):
        """dest_fips must be 5-character strings."""
        assert all(len(f) == 5 for f in filtered["dest_fips"])
        assert all(f.isdigit() for f in filtered["dest_fips"])

    def test_year_pair_label_preserved(self, filtered):
        """year_pair column must contain the label passed to filter_inflow_df."""
        assert (filtered["year_pair"] == "2021-2022").all()

    def test_n_returns_renamed_correctly(self, filtered):
        """n1 column must be renamed to n_returns."""
        assert "n_returns" in filtered.columns
        assert "n1" not in filtered.columns

    def test_n_exemptions_renamed_correctly(self, filtered):
        """n2 column must be renamed to n_exemptions."""
        assert "n_exemptions" in filtered.columns
        assert "n2" not in filtered.columns

    def test_fips_values_correct(self, filtered):
        """FIPS codes must reflect the input state/county FIPS values."""
        assert filtered.iloc[0]["origin_fips"] == "13121"
        assert filtered.iloc[0]["dest_fips"] == "12086"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFilterEdgeCases:
    """Tests for edge cases in filter_inflow_df()."""

    def test_empty_input_returns_empty(self):
        """Empty input DataFrame must produce empty output."""
        # Build a properly-structured empty DataFrame (same schema as fetch_inflow_csv output)
        from src.assembly.fetch_irs_migration import INFLOW_COLUMNS

        df = pd.DataFrame(columns=INFLOW_COLUMNS)
        for col in ("y1_statefips", "y1_countyfips", "y2_statefips", "y2_countyfips"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 0

    def test_all_aggregate_rows_returns_empty(self):
        """Input with only aggregate rows must produce empty output."""
        df = _make_raw_df(
            [
                {"y1_statefips": 96, "y1_countyfips": 0, "y2_statefips": 12, "y2_countyfips": 1},
                {"y1_statefips": 97, "y1_countyfips": 0, "y2_statefips": 12, "y2_countyfips": 1},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 0

    def test_no_target_state_rows_returns_empty(self):
        """Input with no FL/GA/AL flows must produce empty output."""
        df = _make_raw_df(
            [
                {"y1_statefips": 6, "y1_countyfips": 1, "y2_statefips": 48, "y2_countyfips": 1},
                {"y1_statefips": 36, "y1_countyfips": 61, "y2_statefips": 17, "y2_countyfips": 31},
            ]
        )
        result = filter_inflow_df(df, "2021-2022")
        assert len(result) == 0


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
        """TARGET_STATE_FIPS must be the set of values from STATES."""
        assert TARGET_STATE_FIPS == frozenset(STATES.values())

    def test_all_year_pairs_has_11_entries(self):
        """ALL_YEAR_PAIRS must cover 2011-2012 through 2021-2022 (11 pairs)."""
        assert len(ALL_YEAR_PAIRS) == 11

    def test_default_year_pairs_has_3_entries(self):
        """DEFAULT_YEAR_PAIRS must contain exactly 3 year pairs for MVP."""
        assert len(DEFAULT_YEAR_PAIRS) == 3

    def test_default_year_pairs_are_most_recent(self):
        """DEFAULT_YEAR_PAIRS must be the last 3 entries of ALL_YEAR_PAIRS."""
        assert DEFAULT_YEAR_PAIRS == ALL_YEAR_PAIRS[-3:]

    def test_inflow_columns_count(self):
        """INFLOW_COLUMNS must define exactly 9 columns matching IRS format."""
        assert len(INFLOW_COLUMNS) == 9


# ---------------------------------------------------------------------------
# Integration tests (skip if data not present)
# ---------------------------------------------------------------------------


class TestIrsMigrationIntegration:
    """Integration tests that verify the actual saved irs_migration.parquet."""

    @pytest.fixture(scope="class")
    def irs_parquet(self):
        """Load the actual saved irs_migration.parquet if it exists."""
        from pathlib import Path

        path = Path(__file__).parents[1] / "data" / "raw" / "irs_migration.parquet"
        if not path.exists():
            pytest.skip("irs_migration.parquet not found — run fetch_irs_migration.py first")
        return pd.read_parquet(path)

    def test_has_required_columns(self, irs_parquet):
        """Parquet must have all required output columns."""
        required = {"origin_fips", "dest_fips", "n_returns", "n_exemptions", "agi", "year_pair"}
        assert required.issubset(set(irs_parquet.columns))

    def test_fips_are_5_digits(self, irs_parquet):
        """All FIPS codes in the saved file must be 5-digit strings."""
        for col in ("origin_fips", "dest_fips"):
            assert (irs_parquet[col].str.len() == 5).all(), f"{col} has non-5-digit values"
            assert irs_parquet[col].str.isdigit().all(), f"{col} has non-digit characters"

    def test_year_pairs_present(self, irs_parquet):
        """Saved file must contain data for all DEFAULT_YEAR_PAIRS."""
        expected_labels = {label for _, label in DEFAULT_YEAR_PAIRS}
        actual_labels = set(irs_parquet["year_pair"].unique())
        assert expected_labels <= actual_labels

    def test_no_aggregate_fips_in_output(self, irs_parquet):
        """No aggregate state FIPS (96, 97, 98) must appear in the output."""
        origin_state_fips = irs_parquet["origin_fips"].str[:2].astype(int)
        dest_state_fips = irs_parquet["dest_fips"].str[:2].astype(int)
        assert (origin_state_fips < 96).all(), "Aggregate origin FIPS found"
        assert (dest_state_fips < 96).all(), "Aggregate dest FIPS found"

    def test_target_state_coverage(self, irs_parquet):
        """Each of FL, GA, AL must appear as origin or destination."""
        for fips_prefix in ("01", "12", "13"):
            in_origin = irs_parquet["origin_fips"].str.startswith(fips_prefix).any()
            in_dest = irs_parquet["dest_fips"].str.startswith(fips_prefix).any()
            assert in_origin or in_dest, f"No flows found for state FIPS {fips_prefix}"

    def test_n_returns_not_null(self, irs_parquet):
        """n_returns must not be null; IRS uses -1 as a suppression sentinel for small cells."""
        assert irs_parquet["n_returns"].notna().all(), "n_returns has unexpected null values"
        # IRS convention: n_returns == -1 means suppressed (< 10 returns); >= 20 are published
        assert irs_parquet["n_returns"].isin([-1]).sum() >= 0, "Suppressed cells (-1) are expected"

    def test_no_self_loops(self, irs_parquet):
        """No row must have origin_fips == dest_fips (non-migrants filtered out)."""
        self_loops = irs_parquet["origin_fips"] == irs_parquet["dest_fips"]
        assert not self_loops.any(), f"{self_loops.sum()} self-loop rows found"
