"""Tests for the Trafalgar Group report crosstab parser."""

from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from parse_trafalgar_report import (
    parse_demographic_vote_shares,
    parse_header,
    parse_trafalgar_report,
    parse_trafalgar_text,
    parse_sample_composition,
    two_party_dem_share,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "trafalgar_crosstab_extract.txt"
)


class TestTwoPartyConversion:
    def test_basic_conversion(self):
        assert two_party_dem_share(37, 56) == pytest.approx(37 / 93, abs=0.001)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None


class TestHeaderParsing:
    def test_extracts_sample_and_dates(self):
        result = parse_header(FIXTURE.read_text())
        assert result["n_sample"] == 952
        assert result["date_start"] == "2026-03-18"
        assert result["date_end"] == "2026-03-19"

    def test_notes_contains_voter_type(self):
        result = parse_header(FIXTURE.read_text())
        assert "Likely General Election Voters" in result.get("notes", "")

    def test_empty_text_returns_empty(self):
        assert parse_header("") == {}


class TestSampleComposition:
    def test_race_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_race_white"] == pytest.approx(0.70)
        assert result["xt_race_black"] == pytest.approx(0.12)
        assert result["xt_race_hispanic"] == pytest.approx(0.11)
        assert result["xt_race_asian"] == pytest.approx(0.05)

    def test_education_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_education_college"] == pytest.approx(0.42)
        assert result["xt_education_noncollege"] == pytest.approx(0.58)

    def test_senior_age_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_age_senior"] == pytest.approx(0.28)

    def test_empty_text_returns_empty(self):
        assert parse_sample_composition("") == {}


class TestDemographicVoteShares:
    def test_race_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_race_white"] == pytest.approx(37 / (37 + 56))
        assert result["xt_vote_race_black"] == pytest.approx(78 / (78 + 12))
        assert result["xt_vote_race_hispanic"] == pytest.approx(50 / (50 + 36))

    def test_education_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_education_college"] == pytest.approx(51 / (51 + 40))
        assert result["xt_vote_education_noncollege"] == pytest.approx(37 / (37 + 54))

    def test_senior_age_vote_share(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_age_senior"] == pytest.approx(40 / (40 + 52))

    def test_compact_percent_rows(self):
        text = """\
        By Race
        White Black Hispanic 65+ College Non-College
        37% 78% 50% 40% 51% 37%
        56% 12% 36% 52% 40% 54%
        7% 10% 14% 8% 9% 9%
        """
        result = parse_demographic_vote_shares(text)
        assert result["xt_vote_race_white"] == pytest.approx(37 / (37 + 56))
        assert result["xt_vote_age_senior"] == pytest.approx(40 / (40 + 52))
        assert result["xt_vote_education_noncollege"] == pytest.approx(37 / (37 + 54))


class TestFullParsing:
    def test_parse_text_returns_poll_compatible_fields(self):
        result = parse_trafalgar_text(FIXTURE.read_text())
        assert result["pollster"] == "Trafalgar Group"
        assert result["methodology"] == "Multi-Mode"
        assert result["n_sample"] == 952
        assert result["xt_race_white"] == pytest.approx(0.70)
        assert result["xt_education_college"] == pytest.approx(0.42)
        assert result["xt_age_senior"] == pytest.approx(0.28)
        assert result["xt_vote_race_white"] == pytest.approx(37 / (37 + 56))
        assert result["xt_vote_education_noncollege"] == pytest.approx(37 / (37 + 54))
        assert result["xt_vote_age_senior"] == pytest.approx(40 / (40 + 52))

    def test_parse_text_file(self):
        result = parse_trafalgar_report(FIXTURE)
        assert result["xt_vote_race_black"] == pytest.approx(78 / (78 + 12))
