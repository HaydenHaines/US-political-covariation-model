"""Tests for the Cygnal poll report crosstab parser."""

from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from parse_cygnal_report import (
    parse_demographic_vote_shares,
    parse_header,
    parse_cygnal_report,
    parse_cygnal_text,
    parse_sample_composition,
    two_party_dem_share,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "cygnal_crosstab_extract.txt"
)


class TestTwoPartyConversion:
    def test_basic_conversion(self):
        assert two_party_dem_share(36, 55) == pytest.approx(36 / 91, abs=0.001)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None


class TestHeaderParsing:
    def test_extracts_sample_and_dates(self):
        result = parse_header(FIXTURE.read_text())
        assert result["n_sample"] == 800
        assert result["date_start"] == "2026-04-14"
        assert result["date_end"] == "2026-04-15"

    def test_notes_contains_voter_type(self):
        result = parse_header(FIXTURE.read_text())
        assert "Likely Voters" in result.get("notes", "")

    def test_empty_text_returns_empty(self):
        assert parse_header("") == {}


class TestSampleComposition:
    def test_race_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_race_white"] == pytest.approx(0.68)
        assert result["xt_race_black"] == pytest.approx(0.12)
        assert result["xt_race_hispanic"] == pytest.approx(0.13)
        assert result["xt_race_asian"] == pytest.approx(0.05)

    def test_education_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_education_college"] == pytest.approx(0.41)
        assert result["xt_education_noncollege"] == pytest.approx(0.59)

    def test_senior_age_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_age_senior"] == pytest.approx(0.27)

    def test_empty_text_returns_empty(self):
        assert parse_sample_composition("") == {}


class TestDemographicVoteShares:
    def test_race_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_race_white"] == pytest.approx(36 / (36 + 55))
        assert result["xt_vote_race_black"] == pytest.approx(79 / (79 + 11))
        assert result["xt_vote_race_hispanic"] == pytest.approx(51 / (51 + 36))

    def test_education_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_education_college"] == pytest.approx(52 / (52 + 37))
        assert result["xt_vote_education_noncollege"] == pytest.approx(38 / (38 + 51))

    def test_senior_age_vote_share(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_age_senior"] == pytest.approx(41 / (41 + 49))

    def test_gender_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_gender_men"] == pytest.approx(41 / (41 + 50))
        assert result["xt_vote_gender_women"] == pytest.approx(50 / (50 + 39))

    def test_compact_percent_rows(self):
        text = """\
        By Race
        White Black Hispanic 65+ College Non-College
        36% 79% 51% 41% 52% 38%
        55% 11% 36% 49% 37% 51%
        9% 10% 13% 10% 11% 11%
        """
        result = parse_demographic_vote_shares(text)
        assert result["xt_vote_race_white"] == pytest.approx(36 / (36 + 55))
        assert result["xt_vote_age_senior"] == pytest.approx(41 / (41 + 49))
        assert result["xt_vote_education_noncollege"] == pytest.approx(38 / (38 + 51))


class TestFullParsing:
    def test_parse_text_returns_poll_compatible_fields(self):
        result = parse_cygnal_text(FIXTURE.read_text())
        assert result["pollster"] == "Cygnal"
        assert result["methodology"] == "IVR"
        assert result["n_sample"] == 800
        assert result["xt_race_white"] == pytest.approx(0.68)
        assert result["xt_education_college"] == pytest.approx(0.41)
        assert result["xt_age_senior"] == pytest.approx(0.27)
        assert result["xt_vote_race_white"] == pytest.approx(36 / (36 + 55))
        assert result["xt_vote_education_noncollege"] == pytest.approx(38 / (38 + 51))
        assert result["xt_vote_age_senior"] == pytest.approx(41 / (41 + 49))
        assert result["xt_vote_gender_men"] == pytest.approx(41 / (41 + 50))
        assert result["xt_vote_gender_women"] == pytest.approx(50 / (50 + 39))

    def test_parse_text_file(self):
        result = parse_cygnal_report(FIXTURE)
        assert result["xt_vote_race_black"] == pytest.approx(79 / (79 + 11))
