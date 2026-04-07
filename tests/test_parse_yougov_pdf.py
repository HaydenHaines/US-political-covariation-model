"""Tests for the YouGov/Economist PDF crosstab parser."""

from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from parse_yougov_pdf import (
    classify_response_row,
    find_generic_ballot_page,
    parse_generic_ballot_page,
    parse_header,
    parse_yougov_pdf,
    two_party_dem_share,
)

SAMPLE_PDF = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "yougov"
    / "econTabReport_o84FoNw.pdf"
)


# ---------------------------------------------------------------------------
# Two-party conversion
# ---------------------------------------------------------------------------


class TestTwoPartyConversion:
    def test_basic_conversion(self):
        # 34% D, 43% R -> 34/77 = 0.4416
        result = two_party_dem_share(34, 43)
        assert result == pytest.approx(0.4416, abs=0.001)

    def test_even_split(self):
        assert two_party_dem_share(50, 50) == pytest.approx(0.5)

    def test_all_dem(self):
        assert two_party_dem_share(100, 0) == pytest.approx(1.0)

    def test_all_rep(self):
        assert two_party_dem_share(0, 100) == pytest.approx(0.0)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None

    def test_strong_dem(self):
        # 65% D, 6% R -> 65/71 = 0.9155
        result = two_party_dem_share(65, 6)
        assert result == pytest.approx(0.9155, abs=0.001)


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------


class TestRowClassification:
    def test_democratic_party_candidate(self):
        line = "TheDemocraticPartycandidate 39% 33% 44%"
        assert classify_response_row(line) == "dem"

    def test_republican_party_candidate(self):
        line = "TheRepublicanPartycandidate 36% 43% 29%"
        assert classify_response_row(line) == "rep"

    def test_other(self):
        assert classify_response_row("Other 2% 2% 1%") == "other"

    def test_notsure(self):
        assert classify_response_row("Notsure 11% 10% 12%") == "notsure"

    def test_would_not_vote(self):
        assert classify_response_row("Iwouldnotvote 13% 12% 13%") == "notvote"

    def test_totals(self):
        assert classify_response_row("Totals 101% 100% 99%") == "totals"

    def test_unrecognized_returns_none(self):
        assert classify_response_row("Some random text") is None

    def test_empty_returns_none(self):
        assert classify_response_row("") is None


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
class TestDateExtraction:
    def test_extracts_dates(self):
        import pdfplumber

        with pdfplumber.open(SAMPLE_PDF) as pdf:
            header = parse_header(pdf)
        assert header["date_start"] == "2026-03-20"
        assert header["date_end"] == "2026-03-23"
        assert header["n_total"] == 1665


# ---------------------------------------------------------------------------
# Page detection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
class TestPageDetection:
    def test_finds_generic_ballot_page(self):
        import pdfplumber

        with pdfplumber.open(SAMPLE_PDF) as pdf:
            page_idx = find_generic_ballot_page(pdf)
        # Page 59 in the PDF = 0-indexed 58.
        assert page_idx == 58

    def test_skips_table_of_contents(self):
        """The TOC on page 2 mentions GenericCongressionalVote but has no data."""
        import pdfplumber

        with pdfplumber.open(SAMPLE_PDF) as pdf:
            page_idx = find_generic_ballot_page(pdf)
        # Should NOT return page 1 (the TOC page).
        assert page_idx != 1


# ---------------------------------------------------------------------------
# Full PDF parsing (integration test against real Mar 20-23 sample)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
class TestParseSamplePdf:
    @pytest.fixture(scope="class")
    def extracted(self):
        return parse_yougov_pdf(SAMPLE_PDF)

    def test_topline(self, extracted):
        # 39/(39+36) = 0.52
        assert extracted["gb_topline"] == pytest.approx(0.52, abs=0.001)

    def test_white_two_party(self, extracted):
        # White: 34/(34+43) = 0.4416
        assert extracted["gb_vote_race_white"] == pytest.approx(0.4416, abs=0.001)

    def test_black_two_party(self, extracted):
        # Black: 65/(65+6) = 0.9155
        assert extracted["gb_vote_race_black"] == pytest.approx(0.9155, abs=0.001)

    def test_hispanic_two_party(self, extracted):
        # Hispanic: 41/(41+34) = 0.5467
        assert extracted["gb_vote_race_hispanic"] == pytest.approx(0.5467, abs=0.001)

    def test_education_college(self, extracted):
        # College grad: 48/(48+32) = 0.6
        assert extracted["gb_vote_education_college"] == pytest.approx(0.6, abs=0.001)

    def test_education_noncollege(self, extracted):
        # No degree: 34/(34+38) = 0.4722
        assert extracted["gb_vote_education_noncollege"] == pytest.approx(
            0.4722, abs=0.001
        )

    def test_age_young(self, extracted):
        # 18-29: 39/(39+29) = 0.5735
        assert extracted["gb_vote_age_young"] == pytest.approx(0.5735, abs=0.001)

    def test_age_senior(self, extracted):
        # 65+: 46/(46+40) = 0.5349
        assert extracted["gb_vote_age_senior"] == pytest.approx(0.5349, abs=0.001)

    def test_gender_male(self, extracted):
        # Male: 33/(33+43) = 0.4342
        assert extracted["gb_vote_gender_male"] == pytest.approx(0.4342, abs=0.001)

    def test_gender_female(self, extracted):
        # Female: 44/(44+29) = 0.6027
        assert extracted["gb_vote_gender_female"] == pytest.approx(0.6027, abs=0.001)

    def test_at_least_ten_demographic_values(self, extracted):
        gb_keys = [k for k in extracted if k.startswith("gb_vote_")]
        assert len(gb_keys) >= 10

    def test_sample_sizes_present(self, extracted):
        assert extracted["n_total"] == 1665
        assert extracted["n_total_ballot"] == 1664
        assert extracted["n_white"] == 1096
        assert extracted["n_black"] == 208

    def test_party_id_breakdown(self, extracted):
        # Dem party ID: 91/(91+1) = 0.9891
        assert extracted["gb_vote_party_dem"] == pytest.approx(0.9891, abs=0.001)
        # Rep party ID: 2/(2+87) = 0.0225
        assert extracted["gb_vote_party_rep"] == pytest.approx(0.0225, abs=0.001)

    def test_dates(self, extracted):
        assert extracted["date_start"] == "2026-03-20"
        assert extracted["date_end"] == "2026-03-23"


# ---------------------------------------------------------------------------
# Missing generic ballot question (graceful handling)
# ---------------------------------------------------------------------------


class TestMissingQuestion:
    def test_no_generic_ballot_raises(self, tmp_path):
        """A PDF without the GenericCongressionalVote question should raise."""
        # We can't easily create a fake PDF, so test the page text parser
        # with text that has no matching rows.
        text = "Some random survey about cats\nNo percentages here\n"
        result = parse_generic_ballot_page(text)
        # Should return empty dict — no dem/rep rows found.
        assert "dem_t1" not in result
        assert "rep_t1" not in result


# ---------------------------------------------------------------------------
# Synthetic page parsing
# ---------------------------------------------------------------------------


class TestSyntheticParsing:
    """Test parsing against synthetic text matching the known PDF format."""

    SYNTHETIC_PAGE = (
        "The Economist/YouGov Poll\n"
        "March 20 - 23, 2026 - 1665 U.S. Adult Citizens\n"
        "41. GenericCongressionalVote\n"
        "IftheelectionsforU.S.Congresswerebeingheldtoday\n"
        "Sex Race Age Education\n"
        "Total Male Female White Black Hispanic 18-29 30-44 45-64 65+ Nodegree Collegegrad\n"
        "TheDemocraticPartycandidate 39% 33% 44% 34% 65% 41% 39% 41% 33% 46% 34% 48%\n"
        "TheRepublicanPartycandidate 36% 43% 29% 43% 6% 34% 29% 31% 40% 40% 38% 32%\n"
        "Other 2% 2% 1% 2% 1% 2% 0% 1% 4% 1% 1% 2%\n"
        "Notsure 11% 10% 12% 9% 14% 9% 13% 12% 12% 6% 11% 10%\n"
        "Iwouldnotvote 13% 12% 13% 12% 15% 14% 19% 15% 11% 7% 16% 7%\n"
        "Totals 101% 100% 99% 100% 101% 100% 100% 100% 100% 100% 100% 99%\n"
        "UnweightedN (1,664) (784) (880) (1,096) (208) (252) (352) (437) (509) (366) (1,064) (600)\n"
        "2024Vote Reg Ideology MAGA PartyID\n"
        "Total Harris Trump Voters Lib Mod Con Supporter Dem Ind Rep\n"
        "TheDemocraticPartycandidate 39% 89% 5% 45% 84% 41% 5% 2% 91% 28% 2%\n"
        "TheRepublicanPartycandidate 36% 2% 84% 42% 4% 21% 82% 85% 1% 21% 87%\n"
        "Other 2% 2% 1% 2% 2% 2% 1% 1% 0% 4% 0%\n"
        "Notsure 11% 7% 9% 10% 5% 19% 6% 6% 4% 20% 6%\n"
        "Iwouldnotvote 13% 0% 1% 2% 6% 17% 6% 5% 3% 27% 6%\n"
        "Totals 101% 100% 100% 101% 101% 100% 100% 99% 99% 100% 101%\n"
        "UnweightedN (1,664) (667) (571) (1,501) (516) (501) (519) (412) (572) (639) (453)\n"
    )

    def test_parse_both_tables(self):
        result = parse_generic_ballot_page(self.SYNTHETIC_PAGE)
        assert "dem_t1" in result
        assert "rep_t1" in result
        assert "dem_t2" in result
        assert "rep_t2" in result

    def test_table1_dem_values(self):
        result = parse_generic_ballot_page(self.SYNTHETIC_PAGE)
        # Table 1 Dem row: 39, 33, 44, 34, 65, 41, 39, 41, 33, 46, 34, 48
        assert result["dem_t1"] == [39, 33, 44, 34, 65, 41, 39, 41, 33, 46, 34, 48]

    def test_table1_rep_values(self):
        result = parse_generic_ballot_page(self.SYNTHETIC_PAGE)
        assert result["rep_t1"] == [36, 43, 29, 43, 6, 34, 29, 31, 40, 40, 38, 32]

    def test_unweighted_n_table1(self):
        result = parse_generic_ballot_page(self.SYNTHETIC_PAGE)
        assert result["n_t1"][0] == 1664  # Total
        assert result["n_t1"][3] == 1096  # White

    def test_unweighted_n_table2(self):
        result = parse_generic_ballot_page(self.SYNTHETIC_PAGE)
        assert result["n_t2"][0] == 1664  # Total
        assert result["n_t2"][8] == 572  # Dem party ID

    def test_empty_text(self):
        result = parse_generic_ballot_page("")
        assert result == {}
