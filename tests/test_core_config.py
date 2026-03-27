"""Tests for src/core/config.py — central configuration loader.

Covers: load(), get_state_fips(), eager constants, pair formatting.
"""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from src.core import config


# ---------------------------------------------------------------------------
# Eager constants populated at import time
# ---------------------------------------------------------------------------

class TestEagerConstants:
    def test_states_is_nonempty_dict(self):
        assert isinstance(config.STATES, dict)
        assert len(config.STATES) >= 50  # 50 states + DC

    def test_all_state_fips_are_two_digit_strings(self):
        for abbr, fips in config.STATES.items():
            assert len(fips) == 2, f"{abbr} has FIPS '{fips}' (not 2 chars)"
            assert fips.isdigit(), f"{abbr} FIPS '{fips}' is not all digits"

    def test_state_abbr_is_inverse_of_states(self):
        for abbr, fips in config.STATES.items():
            assert config.STATE_ABBR[fips] == abbr

    def test_state_abbr_same_length(self):
        assert len(config.STATE_ABBR) == len(config.STATES)

    def test_well_known_states_present(self):
        """Spot-check a handful of well-known states."""
        assert "FL" in config.STATES
        assert "CA" in config.STATES
        assert "TX" in config.STATES
        assert "NY" in config.STATES
        assert "DC" in config.STATES

    def test_fl_fips_is_12(self):
        assert config.STATES["FL"] == "12"

    def test_dc_fips_is_11(self):
        assert config.STATES["DC"] == "11"


class TestPairConstants:
    def test_pres_pairs_non_empty(self):
        assert len(config.PRES_PAIRS) > 0

    def test_pres_pairs_are_two_digit_tuples(self):
        for a, b in config.PRES_PAIRS:
            assert len(a) == 2 and a.isdigit()
            assert len(b) == 2 and b.isdigit()

    def test_gov_pairs_non_empty(self):
        assert len(config.GOV_PAIRS) > 0

    def test_senate_pairs_non_empty(self):
        assert len(config.SENATE_PAIRS) > 0

    def test_holdout_pres_pairs_non_empty(self):
        assert len(config.HOLDOUT_PRES_PAIRS) > 0

    def test_holdout_pairs_format_matches_pres_pairs(self):
        """Holdout pairs should use the same 2-digit tuple format as PRES_PAIRS."""
        for a, b in config.HOLDOUT_PRES_PAIRS:
            assert len(a) == 2 and a.isdigit()
            assert len(b) == 2 and b.isdigit()
            # The later year should be strictly greater
            assert int(b) > int(a) or (int(a) > 90 and int(b) < 10)  # handles century wrap


class TestElectionYears:
    def test_pres_years_every_four(self):
        for y in config.PRES_YEARS:
            assert y % 4 == 0, f"Presidential year {y} not divisible by 4"

    def test_pres_years_sorted(self):
        assert config.PRES_YEARS == sorted(config.PRES_YEARS)

    def test_gov_years_sorted(self):
        assert config.GOV_YEARS == sorted(config.GOV_YEARS)

    def test_senate_years_sorted(self):
        assert config.SENATE_YEARS == sorted(config.SENATE_YEARS)


class TestVoteShareConfig:
    def test_vote_share_type_known(self):
        assert config.VOTE_SHARE_TYPE in ("total", "twoparty")

    def test_shift_type_known(self):
        assert config.SHIFT_TYPE in ("logodds", "raw")

    def test_logodds_epsilon_small_positive(self):
        assert 0 < config.LOGODDS_EPSILON < 0.1


# ---------------------------------------------------------------------------
# get_state_fips()
# ---------------------------------------------------------------------------

class TestGetStateFips:
    def test_no_filter_returns_all(self):
        result = config.get_state_fips()
        assert result == config.STATES

    def test_no_filter_returns_copy(self):
        """Should return a copy, not the original dict."""
        result = config.get_state_fips()
        result["ZZ"] = "99"
        assert "ZZ" not in config.STATES

    def test_filter_single_state(self):
        result = config.get_state_fips(["FL"])
        assert result == {"FL": "12"}

    def test_filter_multiple_states(self):
        result = config.get_state_fips(["FL", "GA", "AL"])
        assert set(result.keys()) == {"FL", "GA", "AL"}

    def test_unknown_state_raises(self):
        with pytest.raises(ValueError, match="Unknown state abbreviation"):
            config.get_state_fips(["XX"])

    def test_unknown_mixed_with_valid_raises(self):
        with pytest.raises(ValueError):
            config.get_state_fips(["FL", "NOPE"])

    def test_empty_list_returns_empty(self):
        result = config.get_state_fips([])
        assert result == {}


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_default_returns_dict(self):
        cfg = config.load()
        assert isinstance(cfg, dict)

    def test_load_has_geography_key(self):
        cfg = config.load()
        assert "geography" in cfg

    def test_load_has_election_key(self):
        cfg = config.load()
        assert "election" in cfg

    def test_load_custom_path(self, tmp_path):
        """Load from a custom YAML path."""
        custom = tmp_path / "test.yaml"
        custom.write_text(yaml.dump({"geography": {"state_fips": {"ZZ": "99"}}}))
        cfg = config.load(custom)
        assert cfg["geography"]["state_fips"]["ZZ"] == "99"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            config.load(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Data file config
# ---------------------------------------------------------------------------

class TestDataFileConfig:
    def test_pres_files_is_dict(self):
        assert isinstance(config.PRES_FILES, dict)

    def test_gov_files_is_dict(self):
        assert isinstance(config.GOV_FILES, dict)

    def test_senate_files_is_dict(self):
        assert isinstance(config.SENATE_FILES, dict)

    def test_spine_file_is_string(self):
        assert isinstance(config.SPINE_FILE, str)
