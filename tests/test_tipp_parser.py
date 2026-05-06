"""Tests for the TIPP Insights report crosstab parser."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from parse_tipp_report import (
    parse_demographic_vote_shares,
    parse_header,
    parse_sample_composition,
    parse_tipp_report,
    parse_tipp_text,
    two_party_dem_share,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tipp_crosstab_extract.txt"


class TestTwoPartyConversion:
    def test_basic_conversion(self):
        assert two_party_dem_share(39, 52) == pytest.approx(39 / 91, abs=0.001)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None


class TestHeaderParsing:
    def test_extracts_sample_and_dates(self):
        result = parse_header(FIXTURE.read_text())
        assert result["n_sample"] == 1159
        assert result["date_start"] == "2026-04-22"
        assert result["date_end"] == "2026-04-24"

    def test_notes_contains_voter_type(self):
        result = parse_header(FIXTURE.read_text())
        assert "Likely Voters" in result.get("notes", "")

    def test_empty_text_returns_empty(self):
        assert parse_header("") == {}


class TestSampleComposition:
    def test_race_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_race_white"] == pytest.approx(0.66)
        assert result["xt_race_black"] == pytest.approx(0.13)
        assert result["xt_race_hispanic"] == pytest.approx(0.14)
        assert result["xt_race_asian"] == pytest.approx(0.04)

    def test_gender_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_gender_men"] == pytest.approx(0.47)
        assert result["xt_gender_women"] == pytest.approx(0.53)

    def test_age_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_age_18_44"] == pytest.approx(0.32)
        assert result["xt_age_45_64"] == pytest.approx(0.40)
        assert result["xt_age_senior"] == pytest.approx(0.28)

    def test_party_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_party_republican"] == pytest.approx(0.37)
        assert result["xt_party_democrat"] == pytest.approx(0.35)
        assert result["xt_party_independent"] == pytest.approx(0.28)

    def test_empty_text_returns_empty(self):
        assert parse_sample_composition("") == {}


class TestDemographicVoteShares:
    def test_race_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_race_white"] == pytest.approx(39 / (39 + 52))
        assert result["xt_vote_race_black"] == pytest.approx(82 / (82 + 10))
        assert result["xt_vote_race_hispanic"] == pytest.approx(54 / (54 + 34))
        assert result["xt_vote_race_asian"] == pytest.approx(48 / (48 + 40))

    def test_gender_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_gender_men"] == pytest.approx(43 / (43 + 48))
        assert result["xt_vote_gender_women"] == pytest.approx(52 / (52 + 39))

    def test_senior_age_vote_share(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_age_senior"] == pytest.approx(42 / (42 + 49))

    def test_party_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_party_republican"] == pytest.approx(8 / (8 + 86))
        assert result["xt_vote_party_democrat"] == pytest.approx(91 / (91 + 5))
        assert result["xt_vote_party_independent"] == pytest.approx(45 / (45 + 42))

    def test_compact_percent_rows(self):
        text = """\
        By Race
        White Black Hispanic 65+ Rep Dem Ind
        39% 82% 54% 42% 8% 91% 45%
        52% 10% 34% 49% 86% 5% 42%
        9% 8% 12% 9% 6% 4% 13%
        """
        result = parse_demographic_vote_shares(text)
        assert result["xt_vote_race_white"] == pytest.approx(39 / (39 + 52))
        assert result["xt_vote_age_senior"] == pytest.approx(42 / (42 + 49))
        assert result["xt_vote_party_independent"] == pytest.approx(45 / (45 + 42))


class TestFullParsing:
    def test_parse_text_returns_poll_compatible_fields(self):
        result = parse_tipp_text(FIXTURE.read_text())
        assert result["pollster"] == "TIPP Insights"
        assert result["methodology"] == "Online"
        assert result["n_sample"] == 1159
        assert result["xt_race_white"] == pytest.approx(0.66)
        assert result["xt_gender_women"] == pytest.approx(0.53)
        assert result["xt_age_senior"] == pytest.approx(0.28)
        assert result["xt_party_democrat"] == pytest.approx(0.35)
        assert result["xt_vote_race_white"] == pytest.approx(39 / (39 + 52))
        assert result["xt_vote_gender_men"] == pytest.approx(43 / (43 + 48))
        assert result["xt_vote_party_independent"] == pytest.approx(45 / (45 + 42))

    def test_parse_text_file(self):
        result = parse_tipp_report(FIXTURE)
        assert result["xt_vote_race_black"] == pytest.approx(82 / (82 + 10))

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "parse_tipp_report.py"), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--update" in result.stdout
