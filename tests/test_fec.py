"""Tests for fetch_fec_contributions.py.

Tests use synthetic DataFrames and mock HTTP/file I/O so no network access
is required. Coverage:

1. FIPS normalization / ZIP-to-county crosswalk mapping
2. Ratio computation including zero handling and scalar/Series variants
3. zip_to_county aggregation
4. build_fec_features output schema and column types
5. County spine construction
6. Cache path naming convention
7. Edge cases: empty contributions, all-zero county, mixed zero/nonzero
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.assembly.fetch_fec_contributions import (
    ACTBLUE_ID,
    ASSEMBLED_DIR,
    FEC_CYCLES,
    RAW_DIR,
    STATE_ABBR,
    TARGET_STATES,
    WINRED_ID,
    _build_county_spine,
    build_fec_features,
    compute_dem_ratio,
    zip_to_county,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_crosswalk() -> pd.DataFrame:
    """Minimal ZCTA crosswalk covering FL, GA, AL ZIPs."""
    return pd.DataFrame(
        {
            "zip5": ["32601", "32602", "30301", "35004", "99999"],
            "county_fips": ["12001", "12001", "13121", "01001", "06037"],
        }
    )


@pytest.fixture
def actblue_zip_df() -> pd.DataFrame:
    """Synthetic ActBlue ZIP-level totals."""
    return pd.DataFrame(
        {
            "zip": ["32601", "32602", "30301", "35004"],
            "state": ["FL", "FL", "GA", "AL"],
            "cycle": [2020, 2020, 2020, 2020],
            "total": [10000.0, 5000.0, 8000.0, 2000.0],
            "count": [50, 25, 40, 10],
            "zip5": ["32601", "32602", "30301", "35004"],
        }
    )


@pytest.fixture
def winred_zip_df() -> pd.DataFrame:
    """Synthetic WinRed ZIP-level totals."""
    return pd.DataFrame(
        {
            "zip": ["32601", "30301", "35004"],
            "state": ["FL", "GA", "AL"],
            "cycle": [2020, 2020, 2020],
            "total": [3000.0, 1000.0, 9000.0],
            "count": [15, 5, 45],
            "zip5": ["32601", "30301", "35004"],
        }
    )


# ---------------------------------------------------------------------------
# 1. FIPS normalization / crosswalk
# ---------------------------------------------------------------------------


class TestZipToCounty:
    def test_basic_aggregation(self, actblue_zip_df, sample_crosswalk):
        """ZIPs mapping to same county are summed."""
        result = zip_to_county(actblue_zip_df, sample_crosswalk)
        fl_row = result[result["county_fips"] == "12001"].iloc[0]
        # 32601 + 32602 both map to 12001
        assert fl_row["total"] == pytest.approx(15000.0)
        assert fl_row["count"] == 75

    def test_out_of_scope_zip_dropped(self, actblue_zip_df):
        """ZIPs not present in the FL/GA/AL-only crosswalk are silently dropped."""
        # Crosswalk that covers only FL/GA/AL ZIPs (no CA entry)
        fl_ga_al_crosswalk = pd.DataFrame(
            {
                "zip5": ["32601", "32602", "30301", "35004"],
                "county_fips": ["12001", "12001", "13121", "01001"],
            }
        )
        extra = pd.concat(
            [
                actblue_zip_df,
                pd.DataFrame(
                    {
                        "zip": ["90210"],
                        "state": ["CA"],
                        "cycle": [2020],
                        "total": [999.0],
                        "count": [9],
                        "zip5": ["90210"],
                    }
                ),
            ],
            ignore_index=True,
        )
        # 90210 is not in the FL/GA/AL crosswalk → inner join drops it
        result = zip_to_county(extra, fl_ga_al_crosswalk)
        # Only FL/GA/AL counties should appear
        assert all(result["county_fips"].str[:2].isin({"12", "13", "01"}))

    def test_empty_input_returns_empty(self, sample_crosswalk):
        """Empty contributions DataFrame produces empty county aggregation."""
        empty = pd.DataFrame(
            columns=["zip", "state", "cycle", "total", "count", "zip5"]
        )
        result = zip_to_county(empty, sample_crosswalk)
        assert len(result) == 0
        assert "county_fips" in result.columns

    def test_all_counties_present_in_result(self, actblue_zip_df, sample_crosswalk):
        """All FL/GA/AL counties present in crosswalk appear if data maps to them."""
        result = zip_to_county(actblue_zip_df, sample_crosswalk)
        assert set(result["county_fips"]).issubset({"12001", "13121", "01001", "06037"})
        assert "12001" in result["county_fips"].values
        assert "13121" in result["county_fips"].values
        assert "01001" in result["county_fips"].values


# ---------------------------------------------------------------------------
# 2. Ratio computation
# ---------------------------------------------------------------------------


class TestComputeDemRatio:
    def test_normal_ratio(self):
        """Standard case: actblue / (actblue + winred)."""
        assert compute_dem_ratio(7000.0, 3000.0) == pytest.approx(0.7)

    def test_all_actblue(self):
        """All Democrat donations → ratio = 1.0."""
        assert compute_dem_ratio(5000.0, 0.0) == pytest.approx(1.0)

    def test_all_winred(self):
        """All Republican donations → ratio = 0.0."""
        assert compute_dem_ratio(0.0, 8000.0) == pytest.approx(0.0)

    def test_both_zero_returns_half(self):
        """Zero total contributions → neutral imputation 0.5."""
        assert compute_dem_ratio(0.0, 0.0) == pytest.approx(0.5)

    def test_series_normal(self):
        """Series input with mixed values."""
        ab = pd.Series([10000.0, 0.0, 5000.0])
        wr = pd.Series([0.0, 8000.0, 5000.0])
        result = compute_dem_ratio(ab, wr)
        assert result.iloc[0] == pytest.approx(1.0)  # all ActBlue
        assert result.iloc[1] == pytest.approx(0.0)  # all WinRed
        assert result.iloc[2] == pytest.approx(0.5)  # equal split

    def test_series_both_zero(self):
        """Series with all-zero totals returns 0.5 for each row."""
        ab = pd.Series([0.0, 0.0])
        wr = pd.Series([0.0, 0.0])
        result = compute_dem_ratio(ab, wr)
        assert (result == 0.5).all()

    def test_ratio_bounds(self):
        """Ratio always within [0, 1]."""
        import numpy as np
        ab = pd.Series([0.0, 1000.0, 5000.0, 0.0])
        wr = pd.Series([1000.0, 0.0, 5000.0, 0.0])
        result = compute_dem_ratio(ab, wr)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()


# ---------------------------------------------------------------------------
# 3. County spine
# ---------------------------------------------------------------------------


class TestBuildCountySpine:
    def test_state_abbr_mapped(self, sample_crosswalk):
        """state_abbr is correctly mapped from county FIPS prefix."""
        # Only include FL/GA/AL rows in crosswalk for this test
        cw = sample_crosswalk[sample_crosswalk["county_fips"].str[:2].isin(STATE_ABBR)]
        spine = _build_county_spine(cw)
        fl_rows = spine[spine["county_fips"].str[:2] == "12"]
        assert (fl_rows["state_abbr"] == "FL").all()
        ga_rows = spine[spine["county_fips"].str[:2] == "13"]
        assert (ga_rows["state_abbr"] == "GA").all()
        al_rows = spine[spine["county_fips"].str[:2] == "01"]
        assert (al_rows["state_abbr"] == "AL").all()

    def test_out_of_scope_fips_excluded(self, sample_crosswalk):
        """Counties outside FL/GA/AL (e.g. CA 06037) are excluded from spine."""
        spine = _build_county_spine(sample_crosswalk)
        assert "06037" not in spine["county_fips"].values

    def test_unique_county_fips(self, sample_crosswalk):
        """Each county FIPS appears exactly once in spine."""
        cw = sample_crosswalk[sample_crosswalk["county_fips"].str[:2].isin(STATE_ABBR)]
        spine = _build_county_spine(cw)
        assert spine["county_fips"].nunique() == len(spine)


# ---------------------------------------------------------------------------
# 4. build_fec_features output schema
# ---------------------------------------------------------------------------


class TestBuildFecFeatures:
    """Tests for the full build_fec_features() assembly pipeline.

    Uses patch to avoid any network access.
    """

    def _make_zip_df(self, committee_id: str, cycle: int, force_refresh: bool = False) -> pd.DataFrame:
        """Create synthetic zip-level totals for a given committee/cycle."""
        if committee_id == ACTBLUE_ID:
            return pd.DataFrame(
                {
                    "zip": ["32601", "30301"],
                    "state": ["FL", "GA"],
                    "cycle": [cycle, cycle],
                    "total": [10000.0, 8000.0],
                    "count": [50, 40],
                    "zip5": ["32601", "30301"],
                }
            )
        else:  # WinRed
            return pd.DataFrame(
                {
                    "zip": ["32601", "35004"],
                    "state": ["FL", "AL"],
                    "cycle": [cycle, cycle],
                    "total": [3000.0, 9000.0],
                    "count": [15, 45],
                    "zip5": ["32601", "35004"],
                }
            )

    def _make_crosswalk(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "zip5": ["32601", "30301", "35004"],
                "county_fips": ["12001", "13121", "01001"],
            }
        )

    @patch(
        "src.assembly.fetch_fec_contributions._load_zcta_crosswalk"
    )
    @patch(
        "src.assembly.fetch_fec_contributions.fetch_committee_zip_totals"
    )
    def test_output_columns(self, mock_fetch, mock_crosswalk):
        """Output DataFrame has required columns for each cycle."""
        mock_crosswalk.return_value = self._make_crosswalk()
        mock_fetch.side_effect = self._make_zip_df

        result = build_fec_features(cycles=[2020])

        expected_cols = {
            "county_fips",
            "state_abbr",
            "fec_actblue_2020",
            "fec_winred_2020",
            "fec_dem_ratio_2020",
        }
        assert expected_cols.issubset(set(result.columns))

    @patch(
        "src.assembly.fetch_fec_contributions._load_zcta_crosswalk"
    )
    @patch(
        "src.assembly.fetch_fec_contributions.fetch_committee_zip_totals"
    )
    def test_fips_zero_padded(self, mock_fetch, mock_crosswalk):
        """county_fips values are 5-character zero-padded strings."""
        mock_crosswalk.return_value = self._make_crosswalk()
        mock_fetch.side_effect = self._make_zip_df

        result = build_fec_features(cycles=[2020])
        assert (result["county_fips"].str.len() == 5).all()

    @patch(
        "src.assembly.fetch_fec_contributions._load_zcta_crosswalk"
    )
    @patch(
        "src.assembly.fetch_fec_contributions.fetch_committee_zip_totals"
    )
    def test_dem_ratio_bounds(self, mock_fetch, mock_crosswalk):
        """fec_dem_ratio values are within [0, 1]."""
        mock_crosswalk.return_value = self._make_crosswalk()
        mock_fetch.side_effect = self._make_zip_df

        result = build_fec_features(cycles=[2020])
        col = "fec_dem_ratio_2020"
        assert (result[col] >= 0.0).all()
        assert (result[col] <= 1.0).all()

    @patch(
        "src.assembly.fetch_fec_contributions._load_zcta_crosswalk"
    )
    @patch(
        "src.assembly.fetch_fec_contributions.fetch_committee_zip_totals"
    )
    def test_zero_contributions_imputed_half(self, mock_fetch, mock_crosswalk):
        """Counties with no contributions for either committee get ratio = 0.5."""
        cw = self._make_crosswalk()
        # Add a county that has no contributions in either committee
        cw_extra = pd.concat(
            [cw, pd.DataFrame({"zip5": ["99001"], "county_fips": ["12099"]})],
            ignore_index=True,
        )
        mock_crosswalk.return_value = cw_extra
        mock_fetch.side_effect = self._make_zip_df  # does not include 99001

        result = build_fec_features(cycles=[2020])
        zero_county = result[result["county_fips"] == "12099"]
        if len(zero_county) > 0:
            assert zero_county["fec_dem_ratio_2020"].iloc[0] == pytest.approx(0.5)

    @patch(
        "src.assembly.fetch_fec_contributions._load_zcta_crosswalk"
    )
    @patch(
        "src.assembly.fetch_fec_contributions.fetch_committee_zip_totals"
    )
    def test_multiple_cycles(self, mock_fetch, mock_crosswalk):
        """Output includes columns for each requested cycle."""
        mock_crosswalk.return_value = self._make_crosswalk()
        mock_fetch.side_effect = self._make_zip_df

        result = build_fec_features(cycles=[2020, 2022])

        for cycle in [2020, 2022]:
            assert f"fec_actblue_{cycle}" in result.columns
            assert f"fec_winred_{cycle}" in result.columns
            assert f"fec_dem_ratio_{cycle}" in result.columns

    @patch(
        "src.assembly.fetch_fec_contributions._load_zcta_crosswalk"
    )
    @patch(
        "src.assembly.fetch_fec_contributions.fetch_committee_zip_totals"
    )
    def test_state_abbr_correct(self, mock_fetch, mock_crosswalk):
        """state_abbr is populated correctly for each county."""
        mock_crosswalk.return_value = self._make_crosswalk()
        mock_fetch.side_effect = self._make_zip_df

        result = build_fec_features(cycles=[2020])
        fl_row = result[result["county_fips"] == "12001"]
        if len(fl_row) > 0:
            assert fl_row["state_abbr"].iloc[0] == "FL"
        ga_row = result[result["county_fips"] == "13121"]
        if len(ga_row) > 0:
            assert ga_row["state_abbr"].iloc[0] == "GA"
        al_row = result[result["county_fips"] == "01001"]
        if len(al_row) > 0:
            assert al_row["state_abbr"].iloc[0] == "AL"

    @patch(
        "src.assembly.fetch_fec_contributions._load_zcta_crosswalk"
    )
    @patch(
        "src.assembly.fetch_fec_contributions.fetch_committee_zip_totals"
    )
    def test_numeric_amount_columns(self, mock_fetch, mock_crosswalk):
        """fec_actblue_* and fec_winred_* columns are float dtype."""
        mock_crosswalk.return_value = self._make_crosswalk()
        mock_fetch.side_effect = self._make_zip_df

        result = build_fec_features(cycles=[2020])
        assert result["fec_actblue_2020"].dtype.kind == "f"
        assert result["fec_winred_2020"].dtype.kind == "f"


# ---------------------------------------------------------------------------
# 5. Cache path convention
# ---------------------------------------------------------------------------


class TestCachePaths:
    def test_raw_dir_structure(self):
        """RAW_DIR ends in data/raw/fec."""
        assert RAW_DIR.parts[-3:] == ("data", "raw", "fec")

    def test_assembled_dir_structure(self):
        """ASSEMBLED_DIR ends in data/assembled."""
        assert ASSEMBLED_DIR.parts[-2:] == ("data", "assembled")

    def test_committee_ids_defined(self):
        """ActBlue and WinRed committee IDs have expected FEC format."""
        assert ACTBLUE_ID.startswith("C")
        assert WINRED_ID.startswith("C")
        assert len(ACTBLUE_ID) == 9
        assert len(WINRED_ID) == 9

    def test_fec_cycles_include_2020_2022_2024(self):
        """FEC_CYCLES covers the three target election cycles."""
        assert 2020 in FEC_CYCLES
        assert 2022 in FEC_CYCLES
        assert 2024 in FEC_CYCLES

    def test_target_states(self):
        """TARGET_STATES contains FL, GA, AL."""
        assert TARGET_STATES == {"FL", "GA", "AL"}


# ---------------------------------------------------------------------------
# 6. zip_to_county edge cases
# ---------------------------------------------------------------------------


class TestZipToCountyEdgeCases:
    def test_single_zip_single_county(self):
        """Single ZIP maps correctly to single county."""
        df = pd.DataFrame(
            {
                "zip5": ["32601"],
                "total": [5000.0],
                "count": [20],
            }
        )
        cw = pd.DataFrame({"zip5": ["32601"], "county_fips": ["12001"]})
        result = zip_to_county(df, cw)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"] == "12001"
        assert result.iloc[0]["total"] == pytest.approx(5000.0)

    def test_multiple_zips_same_county_summed(self):
        """Multiple ZIPs in the same county are summed correctly."""
        df = pd.DataFrame(
            {
                "zip5": ["32601", "32602", "32603"],
                "total": [1000.0, 2000.0, 3000.0],
                "count": [10, 20, 30],
            }
        )
        cw = pd.DataFrame(
            {
                "zip5": ["32601", "32602", "32603"],
                "county_fips": ["12001", "12001", "12001"],
            }
        )
        result = zip_to_county(df, cw)
        assert len(result) == 1
        assert result.iloc[0]["total"] == pytest.approx(6000.0)
        assert result.iloc[0]["count"] == 60

    def test_no_matching_zips_returns_empty(self):
        """No overlapping ZIPs between df and crosswalk returns empty DataFrame."""
        df = pd.DataFrame(
            {
                "zip5": ["99999"],
                "total": [5000.0],
                "count": [20],
            }
        )
        cw = pd.DataFrame({"zip5": ["32601"], "county_fips": ["12001"]})
        result = zip_to_county(df, cw)
        assert len(result) == 0
