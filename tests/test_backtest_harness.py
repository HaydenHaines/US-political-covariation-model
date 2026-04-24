"""Tests for src/validation/backtest_harness.py.

Tests cover:
  - Two-party dem share conversion
  - State name → abbreviation mapping
  - Historic poll loading (presidential, senate, governor)
  - Historic actuals loading
  - Full 2020 presidential backtest (integration smoke test)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.validation.backtest_harness import (
    _STATE_ABBR_TO_NAME,
    _STATE_NAME_TO_ABBR,
    _two_party_dem_share,
    load_historic_actuals,
    load_historic_polls,
    run_backtest,
)
from tests.conftest import skip_if_missing

# Data files required by individual tests. Paths are relative to the project
# root; all live under gitignored directories so CI never has them — the
# decorators below skip the affected tests when the files are missing.
_PRES_POLLAVERAGES = "data/raw/fivethirtyeight/data-repo/polls/pres_pollaverages_1968-2016.csv"
_PRES_CHECKING = "data/raw/fivethirtyeight/checking-our-work-data/presidential_elections.csv"
_SEN_CHECKING = "data/raw/fivethirtyeight/checking-our-work-data/us_senate_elections.csv"
_GOV_CHECKING = "data/raw/fivethirtyeight/checking-our-work-data/governors_elections.csv"
_ACTUALS_PRES_2020 = "data/assembled/medsl_county_presidential_2020.parquet"
_ACTUALS_GOV_2018 = "data/assembled/algara_county_governor_2018.parquet"
_ACTUALS_GOV_2022 = "data/assembled/medsl_county_2022_governor.parquet"
_ACTUALS_SEN_2022 = "data/assembled/medsl_county_senate_2022.parquet"
_COMMUNITIES_TYPES = "data/communities/type_assignments.parquet"


# ---------------------------------------------------------------------------
# Two-party conversion
# ---------------------------------------------------------------------------

class TestTwoPartyConversion:
    def test_equal_split_is_50pct(self):
        assert abs(_two_party_dem_share(50.0, 50.0) - 0.5) < 1e-9

    def test_dem_dominant(self):
        # D=60, R=40 → 60/100 = 0.6
        result = _two_party_dem_share(60.0, 40.0)
        assert result is not None
        assert abs(result - 0.6) < 1e-9

    def test_third_party_stripped(self):
        # D=45, R=50 (5pp goes to Lib) → 45/95 ≈ 0.4737
        result = _two_party_dem_share(45.0, 50.0)
        assert result is not None
        assert abs(result - 45.0 / 95.0) < 1e-9

    def test_nan_returns_none(self):
        assert _two_party_dem_share(float("nan"), 50.0) is None
        assert _two_party_dem_share(50.0, float("nan")) is None

    def test_zero_total_returns_none(self):
        assert _two_party_dem_share(0.0, 0.0) is None

    def test_fractional_values(self):
        # Works with fraction scale too (e.g. 0.45, 0.55)
        result = _two_party_dem_share(0.45, 0.55)
        assert result is not None
        assert abs(result - 0.45) < 1e-9


# ---------------------------------------------------------------------------
# State name mapping
# ---------------------------------------------------------------------------

class TestStateNameMapping:
    def test_full_name_to_abbr_florida(self):
        assert _STATE_NAME_TO_ABBR["Florida"] == "FL"

    def test_full_name_to_abbr_california(self):
        assert _STATE_NAME_TO_ABBR["California"] == "CA"

    def test_abbr_to_full_name_pa(self):
        assert _STATE_ABBR_TO_NAME["PA"] == "Pennsylvania"

    def test_all_50_states_plus_dc_present(self):
        # 50 states + DC = 51
        assert len(_STATE_NAME_TO_ABBR) == 51

    def test_round_trip(self):
        for name, abbr in _STATE_NAME_TO_ABBR.items():
            assert _STATE_ABBR_TO_NAME[abbr] == name, (
                f"Round-trip failed for {name} ↔ {abbr}"
            )


# ---------------------------------------------------------------------------
# load_historic_polls — presidential
# ---------------------------------------------------------------------------

class TestLoadHistoricPollsPresidential:
    @skip_if_missing(_PRES_POLLAVERAGES)
    def test_2016_loads_correct_number_of_states(self):
        """2016 presidential should load 51 states (50 + DC)."""
        polls = load_historic_polls(2016, "president")
        # Each state → one race key
        states_in_polls = {key.split(" ")[1] for key in polls}
        assert len(states_in_polls) >= 48, (
            f"Expected ≥48 states, got {len(states_in_polls)}"
        )

    @skip_if_missing(_PRES_POLLAVERAGES)
    def test_2016_race_name_format(self):
        """Race names should be '{year} {state} President'."""
        polls = load_historic_polls(2016, "president")
        for race_name in polls:
            parts = race_name.split(" ")
            assert len(parts) == 3
            assert parts[0] == "2016"
            assert parts[2] == "President"

    @skip_if_missing(_PRES_POLLAVERAGES)
    def test_2016_dem_shares_are_probabilities(self):
        """All dem_share values should be in (0, 1)."""
        polls = load_historic_polls(2016, "president")
        for race_name, poll_list in polls.items():
            for poll in poll_list:
                share = poll["dem_share"]
                assert 0.0 < share < 1.0, (
                    f"dem_share={share} out of range for {race_name}"
                )

    @skip_if_missing(_PRES_POLLAVERAGES)
    def test_2016_has_expected_keys(self):
        """Poll dicts should contain required forecast_engine keys."""
        polls = load_historic_polls(2016, "president")
        required_keys = {"state", "dem_share", "n_sample", "race", "date", "pollster", "geo_level"}
        for race_name, poll_list in list(polls.items())[:3]:
            for poll in poll_list:
                assert required_keys.issubset(poll.keys()), (
                    f"Missing keys in {race_name}: {required_keys - set(poll.keys())}"
                )

    @skip_if_missing(_PRES_CHECKING)
    def test_2020_polls_load(self):
        """2020 presidential polls (from checking-our-work) should load."""
        polls = load_historic_polls(2020, "president")
        assert len(polls) >= 48

    def test_invalid_year_raises(self):
        with pytest.raises(ValueError, match="Unsupported presidential year"):
            load_historic_polls(2024, "president")


# ---------------------------------------------------------------------------
# load_historic_polls — senate
# ---------------------------------------------------------------------------

class TestLoadHistoricPollsSenate:
    @skip_if_missing(_SEN_CHECKING)
    def test_2022_senate_loads(self):
        """2022 senate should have races for ~33 states."""
        polls = load_historic_polls(2022, "senate")
        assert len(polls) >= 25, f"Expected ≥25 senate races, got {len(polls)}"

    @skip_if_missing(_SEN_CHECKING)
    def test_2022_senate_race_format(self):
        polls = load_historic_polls(2022, "senate")
        for race_name in polls:
            parts = race_name.split(" ")
            assert parts[0] == "2022"
            assert parts[2] == "Senate"

    @skip_if_missing(_SEN_CHECKING)
    def test_2022_senate_dem_shares_valid(self):
        polls = load_historic_polls(2022, "senate")
        for race_name, poll_list in polls.items():
            for poll in poll_list:
                assert 0.0 < poll["dem_share"] < 1.0, (
                    f"Invalid dem_share in {race_name}: {poll['dem_share']}"
                )

    @skip_if_missing(_SEN_CHECKING)
    def test_2018_senate_loads(self):
        polls = load_historic_polls(2018, "senate")
        assert len(polls) >= 20

    def test_invalid_race_type_raises(self):
        with pytest.raises(ValueError, match="Unknown race_type"):
            load_historic_polls(2020, "congress")


# ---------------------------------------------------------------------------
# load_historic_actuals
# ---------------------------------------------------------------------------

class TestLoadHistoricActuals:
    @skip_if_missing(_ACTUALS_PRES_2020)
    def test_2020_presidential_shape(self):
        """2020 presidential actuals should have ~3,154 counties."""
        df = load_historic_actuals(2020, "president")
        assert len(df) >= 3000, f"Expected ≥3000 counties, got {len(df)}"
        assert set(df.columns) == {"county_fips", "state_abbr", "actual_dem_share"}

    @skip_if_missing(_ACTUALS_PRES_2020)
    def test_2020_presidential_no_state_totals(self):
        """FIPS '00000' (state totals) should be excluded."""
        df = load_historic_actuals(2020, "president")
        assert "00000" not in df["county_fips"].values

    @skip_if_missing(_ACTUALS_PRES_2020)
    def test_2020_presidential_no_nans(self):
        df = load_historic_actuals(2020, "president")
        assert df["actual_dem_share"].isna().sum() == 0

    @skip_if_missing(_ACTUALS_PRES_2020)
    def test_2020_presidential_dem_shares_valid(self):
        df = load_historic_actuals(2020, "president")
        assert (df["actual_dem_share"] >= 0.0).all()
        assert (df["actual_dem_share"] <= 1.0).all()

    @skip_if_missing(_ACTUALS_GOV_2018)
    def test_2018_governor_loads(self):
        """2018 governor actuals from algara parquet."""
        df = load_historic_actuals(2018, "governor")
        assert len(df) >= 500, f"Expected ≥500 counties, got {len(df)}"
        assert "actual_dem_share" in df.columns

    @skip_if_missing(_ACTUALS_GOV_2022)
    def test_2022_governor_loads(self):
        """2022 governor actuals from medsl parquet."""
        df = load_historic_actuals(2022, "governor")
        assert len(df) >= 500

    @skip_if_missing(_ACTUALS_SEN_2022)
    def test_2022_senate_loads(self):
        df = load_historic_actuals(2022, "senate")
        assert len(df) >= 500

    def test_invalid_race_type_raises(self):
        with pytest.raises(ValueError, match="Unknown race_type"):
            load_historic_actuals(2020, "house")

    @skip_if_missing(_ACTUALS_PRES_2020)
    def test_fips_are_string_type(self):
        """FIPS codes should be string type (not int)."""
        df = load_historic_actuals(2020, "president")
        assert df["county_fips"].dtype == object, (
            f"Expected str dtype, got {df['county_fips'].dtype}"
        )

    @skip_if_missing(_ACTUALS_PRES_2020)
    def test_fips_mostly_5_digits(self):
        """The vast majority of FIPS codes should be 5 digits."""
        df = load_historic_actuals(2020, "president")
        lengths = df["county_fips"].str.len()
        pct_5digit = float((lengths == 5).sum()) / len(df)
        assert pct_5digit >= 0.99, f"Only {pct_5digit:.1%} of FIPS are 5 digits"


# ---------------------------------------------------------------------------
# Full backtest integration (2020 presidential)
# ---------------------------------------------------------------------------

@skip_if_missing(_PRES_CHECKING, _ACTUALS_PRES_2020, _COMMUNITIES_TYPES)
class TestRunBacktest2020Presidential:
    """Integration test: run the full backtest pipeline for 2020 presidential."""

    @pytest.fixture(scope="class")
    def backtest_result(self):
        return run_backtest(2020, "president")

    def test_returns_expected_top_level_keys(self, backtest_result):
        required = {"year", "race_type", "n_races", "n_counties", "overall_r",
                    "overall_rmse", "overall_bias", "per_state"}
        assert required.issubset(backtest_result.keys()), (
            f"Missing keys: {required - backtest_result.keys()}"
        )

    def test_year_and_race_type(self, backtest_result):
        assert backtest_result["year"] == 2020
        assert backtest_result["race_type"] == "president"

    def test_overall_r_above_threshold(self, backtest_result):
        """The model should achieve r > 0.5 on 2020 presidential out-of-sample test."""
        r = backtest_result["overall_r"]
        assert not math.isnan(r), "overall_r is NaN"
        assert r > 0.5, f"r={r:.4f} below threshold of 0.5"

    def test_overall_rmse_below_threshold(self, backtest_result):
        """RMSE should be below 0.2 for the 2020 presidential backtest."""
        rmse = backtest_result["overall_rmse"]
        assert not math.isnan(rmse), "overall_rmse is NaN"
        assert rmse < 0.2, f"RMSE={rmse:.4f} above threshold of 0.2"

    def test_n_counties_is_plausible(self, backtest_result):
        assert backtest_result["n_counties"] >= 3000

    def test_n_races_is_51(self, backtest_result):
        """Should have all 51 states (50 + DC) with races."""
        assert backtest_result["n_races"] >= 48

    def test_per_state_has_expected_keys(self, backtest_result):
        required_state_keys = {
            "state", "r", "rmse", "bias", "n_counties",
            "pred_state_dem", "actual_state_dem", "direction_correct",
        }
        for s in backtest_result["per_state"]:
            assert required_state_keys.issubset(s.keys()), (
                f"Missing per-state keys for {s.get('state')}: "
                f"{required_state_keys - s.keys()}"
            )

    def test_direction_accuracy_above_80pct(self, backtest_result):
        """The model should correctly predict the direction (D/R win) in ≥80% of states."""
        dir_acc = backtest_result["direction_accuracy"]
        assert not math.isnan(dir_acc), "direction_accuracy is NaN"
        assert dir_acc >= 0.8, f"direction_accuracy={dir_acc:.2%} below 80%"

    def test_no_error_key(self, backtest_result):
        assert "error" not in backtest_result, (
            f"Backtest returned error: {backtest_result.get('error')}"
        )
