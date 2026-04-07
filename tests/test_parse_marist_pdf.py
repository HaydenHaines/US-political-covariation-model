"""Tests for the Marist PDF crosstab parser."""

from pathlib import Path

import pytest

# Import from scripts — add to path so we can import directly.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from parse_marist_pdf import (
    extract_nos_page_text,
    find_question_pages,
    parse_crosstab_text,
    parse_marist_pdf,
    parse_marist_pdf_composition,
    parse_sample_composition,
    two_party_dem_share,
)

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "raw" / "marist" / "NYS_202602201349.pdf"


# ---------------------------------------------------------------------------
# Two-party conversion
# ---------------------------------------------------------------------------

class TestTwoPartyConversion:
    def test_basic_conversion(self):
        # 50% D, 33% R → 50/83 ≈ 0.6024
        result = two_party_dem_share(50, 33)
        assert result == pytest.approx(0.6024, abs=0.001)

    def test_even_split(self):
        assert two_party_dem_share(50, 50) == pytest.approx(0.5)

    def test_all_dem(self):
        assert two_party_dem_share(100, 0) == pytest.approx(1.0)

    def test_all_rep(self):
        assert two_party_dem_share(0, 100) == pytest.approx(0.0)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None

    def test_strong_dem(self):
        # 79% D, 8% R → 79/87 ≈ 0.9080
        result = two_party_dem_share(79, 8)
        assert result == pytest.approx(0.9080, abs=0.001)


# ---------------------------------------------------------------------------
# Question code detection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
class TestQuestionCodeDetection:
    def test_finds_nygov26(self):
        import pdfplumber
        with pdfplumber.open(SAMPLE_PDF) as pdf:
            pages = find_question_pages(pdf, "NYGOV26")
        assert pages == [12]  # 0-indexed, page 13

    def test_finds_hock105(self):
        import pdfplumber
        with pdfplumber.open(SAMPLE_PDF) as pdf:
            pages = find_question_pages(pdf, "HOCK105")
        assert len(pages) >= 1
        assert pages[0] == 2  # Page 3

    def test_missing_code_returns_empty(self):
        import pdfplumber
        with pdfplumber.open(SAMPLE_PDF) as pdf:
            pages = find_question_pages(pdf, "NONEXISTENT99")
        assert pages == []


# ---------------------------------------------------------------------------
# Full PDF parsing (integration test against the real sample)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
class TestParseSamplePdf:
    @pytest.fixture(scope="class")
    def extracted(self):
        return parse_marist_pdf(SAMPLE_PDF, "NYGOV26")

    def test_topline_dem_share(self, extracted):
        # Hochul 50% vs Blakeman 33% → 50/83
        assert extracted["dem_share_topline"] == pytest.approx(0.6024, abs=0.001)

    def test_white_vote_share(self, extracted):
        # White: 46/(46+41) = 0.5287
        assert extracted["xt_vote_race_white"] == pytest.approx(0.5287, abs=0.001)

    def test_black_vote_share(self, extracted):
        # Black: 79/(79+8) = 0.9080
        assert extracted["xt_vote_race_black"] == pytest.approx(0.9080, abs=0.001)

    def test_hispanic_vote_share(self, extracted):
        # Latino: 56/(56+18) = 0.7568
        assert extracted["xt_vote_race_hispanic"] == pytest.approx(0.7568, abs=0.001)

    def test_education_college(self, extracted):
        # College grad: 57/(57+31) = 0.6477
        assert extracted["xt_vote_education_college"] == pytest.approx(0.6477, abs=0.001)

    def test_education_noncollege(self, extracted):
        # Not college grad: 46/(46+35) = 0.5679
        assert extracted["xt_vote_education_noncollege"] == pytest.approx(0.5679, abs=0.001)

    def test_age_senior(self, extracted):
        # 60 or older: 54/(54+35) = 0.6067
        assert extracted["xt_vote_age_senior"] == pytest.approx(0.6067, abs=0.001)

    def test_at_least_six_xt_values(self, extracted):
        xt_keys = [k for k in extracted if k.startswith("xt_vote_")]
        assert len(xt_keys) >= 6

    def test_no_asian_in_this_pdf(self, extracted):
        # Marist doesn't break out Asian — should be absent.
        assert "xt_vote_race_asian" not in extracted


# ---------------------------------------------------------------------------
# Missing demographics (graceful handling)
# ---------------------------------------------------------------------------

class TestMissingDemographics:
    def test_empty_text_returns_empty(self):
        result = parse_crosstab_text("")
        assert result == {}

    def test_text_without_demographics(self):
        text = "Some random text\nNo percentages here\n"
        result = parse_crosstab_text(text)
        assert result == {}

    def test_partial_demographics(self):
        # Only has education, no race or age.
        text = (
            "NYS Registered Voters 50% 33% 2% 15%\n"
            "Education Not college graduate 46% 35% 2% 17%\n"
            "College graduate 57% 31% 2% 11%\n"
        )
        result = parse_crosstab_text(text)
        assert "xt_vote_education_college" in result
        assert "xt_vote_education_noncollege" in result
        # Race columns should be absent, not None.
        assert "xt_vote_race_white" not in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_question_code_raises(self):
        if not SAMPLE_PDF.exists():
            pytest.skip("Sample PDF not available")
        with pytest.raises(ValueError, match="not found"):
            parse_marist_pdf(SAMPLE_PDF, "FAKE_CODE_999")


# ---------------------------------------------------------------------------
# Sample composition (Nature of the Sample) parsing
# ---------------------------------------------------------------------------

# Minimal representation of the Marist NOS page 2 text layout.
NOS_SAMPLE_TEXT = """\
Nature of the Sample
NYS Adults NYS Registered Voters
Column % Column %
NYS Adults 100%
NY Registered Voters 93% 100%
Party Registration Democrat n/a 45%
Gender Men 48% 49%
Women 52% 51%
Age Under 45 45% 41%
45 or older 55% 59%
Age 18 to 29 20% 17%
30 to 44 25% 24%
45 to 59 24% 25%
60 or older 30% 34%
Race/Ethnicity White 57% 63%
Black 14% 13%
Latino 16% 13%
Other 12% 11%
Education Not college graduate 58% 56%
College graduate 42% 44%
Area Description Big city 36% 27%
Small city 11% 12%
Suburban 32% 37%
Small town 11% 13%
Rural 10% 11%
"""


class TestParseSampleComposition:
    """Tests for parse_sample_composition() using synthetic NOS text."""

    def test_race_white(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert result["xt_race_white"] == pytest.approx(0.63)

    def test_race_black(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert result["xt_race_black"] == pytest.approx(0.13)

    def test_race_hispanic(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert result["xt_race_hispanic"] == pytest.approx(0.13)

    def test_education_college(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert result["xt_education_college"] == pytest.approx(0.44)

    def test_education_noncollege(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert result["xt_education_noncollege"] == pytest.approx(0.56)

    def test_age_senior(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert result["xt_age_senior"] == pytest.approx(0.34)

    def test_no_asian(self):
        # "Other" race should not map to xt_race_asian.
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert "xt_race_asian" not in result

    def test_no_urbanicity(self):
        # Urbanicity is not in the NOS table → should be absent.
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert "xt_urbanicity_urban" not in result
        assert "xt_urbanicity_rural" not in result

    def test_no_evangelical(self):
        # Religion is not in the NOS table → should be absent.
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert "xt_religion_evangelical" not in result

    def test_returns_fractions_not_percentages(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        for value in result.values():
            assert 0.0 <= value <= 1.0, f"Value {value} out of [0, 1] range"

    def test_empty_text_returns_empty(self):
        assert parse_sample_composition("") == {}

    def test_no_two_column_percentages_returns_empty(self):
        # Text with only single-column percentages — no NOS rows.
        text = "Some text 50%\nAnother line 30%\n"
        assert parse_sample_composition(text) == {}

    def test_six_composition_values_in_sample_text(self):
        result = parse_sample_composition(NOS_SAMPLE_TEXT)
        assert len(result) == 6


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
class TestParseSampleCompositionFromRealPdf:
    """Integration tests against the actual Marist NY Governor PDF."""

    @pytest.fixture(scope="class")
    def composition(self):
        return parse_marist_pdf_composition(SAMPLE_PDF)

    def test_race_white_from_pdf(self, composition):
        # Page 2: White = 63% (registered voters)
        assert composition["xt_race_white"] == pytest.approx(0.63)

    def test_race_black_from_pdf(self, composition):
        # Page 2: Black = 13%
        assert composition["xt_race_black"] == pytest.approx(0.13)

    def test_race_hispanic_from_pdf(self, composition):
        # Page 2: Latino = 13%
        assert composition["xt_race_hispanic"] == pytest.approx(0.13)

    def test_education_college_from_pdf(self, composition):
        # Page 2: College graduate = 44%
        assert composition["xt_education_college"] == pytest.approx(0.44)

    def test_education_noncollege_from_pdf(self, composition):
        # Page 2: Not college graduate = 56%
        assert composition["xt_education_noncollege"] == pytest.approx(0.56)

    def test_age_senior_from_pdf(self, composition):
        # Page 2: 60 or older = 34%
        assert composition["xt_age_senior"] == pytest.approx(0.34)

    def test_six_values_extracted_from_pdf(self, composition):
        assert len(composition) == 6

    def test_extract_nos_page_text_returns_string(self):
        text = extract_nos_page_text(SAMPLE_PDF)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_invalid_page_index_raises(self):
        with pytest.raises(ValueError, match="page"):
            extract_nos_page_text(SAMPLE_PDF, page_index=9999)
