"""Tests for Emerson crosstab second-tab vote share parsing and forecast wiring.

Covers:
1. GID discovery from mocked htmlview HTML.
2. Crosstab tab identification from mocked CSV content.
3. Vote share parsing from realistic crosstab CSV data.
4. Demographic label → xt_vote_* column mapping.
5. Party inference from question text.
6. Fallback to topline dem_share when no vote share is available.
7. Integration: _extract_crosstabs_from_xt uses per-group dem_share when present.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.scrape_emerson_crosstabs import (
    _CROSSTAB_ROW_LABEL_MAP,
    _find_general_election_question,
    _infer_dem_candidate_column,
    _parse_csv_lines,
    discover_sheet_gids,
    identify_crosstab_gid,
    parse_crosstab_vote_shares,
)
from src.prediction.forecast_engine import _extract_crosstabs_from_xt


# ---------------------------------------------------------------------------
# Helpers: mock CSV factories
# ---------------------------------------------------------------------------

def _make_crosstab_csv(
    question: str = "If the 2026 general election for U.S. Senate were held today, for whom would you vote?",
    dem_name: str = "Jon Ossoff",
    rep_name: str = "Bo Hines",
    group_rows: list[tuple[str, str, str]] | None = None,
) -> str:
    """Build a minimal crosstab tab CSV in Emerson format.

    Column layout (0-indexed):
      Col 0: blank
      Col 1: group label in data rows, blank in header rows
      Col 2: blank in data rows, Count header for Dem in row 2
      Col 3: Dem Row N % header in row 2, Dem pct value in data rows
      Col 4: blank in data rows, Count header for Rep in row 2
      Col 5: Rep Row N % header in row 2, Rep pct value in data rows

    Row layout:
      Row 0: question text in col 0, rest blank
      Row 1: candidate names (Dem at col 3, Rep at col 5) — aligned with data
      Row 2: alternating Count/Row N % headers starting at col 2
      Rows 3+: demographic group rows

    Each group row tuple is (label, dem_pct_str, rep_pct_str).
    """
    if group_rows is None:
        group_rows = [
            ("White or Caucasian", "42.0", "55.0"),
            ("Black or African American", "80.0", "15.0"),
            ("Hispanic or Latino", "58.0", "38.0"),
            ("Asian", "65.0", "30.0"),
            ("College graduate", "60.0", "37.0"),
            ("Postgraduate", "63.0", "34.0"),
            ("High school or less", "35.0", "61.0"),
            ("60-69", "47.0", "50.0"),
            ("70 or more", "44.0", "52.0"),
        ]

    # Row 0: question text in col 0, rest blank
    # The question spans right — other cols are blank placeholders
    row0 = f'"{question}",,,,,,'
    # Row 1: candidate names aligned with their data columns.
    # Col 3 = Dem Row N % → that's where we put the Dem name.
    # Col 5 = Rep Row N % → that's where we put the Rep name.
    row1 = f",,, {dem_name},, {rep_name}"
    # Row 2: Count / Row N % headers for each candidate starting at col 2.
    # Cols 0–1 are blank (label area), then pairs for each candidate.
    row2 = ",,Count,Row N %,Count,Row N %"

    lines = [row0, row1, row2]
    for label, dem_pct, rep_pct in group_rows:
        # Col 0: blank, Col 1: label, Col 2: dem count, Col 3: dem pct,
        # Col 4: rep count, Col 5: rep pct
        lines.append(f',"{label}",100,{dem_pct},100,{rep_pct}')

    return "\n".join(lines)


def _make_demographics_csv() -> str:
    """Build a minimal demographics tab CSV (does NOT have 'Row N %' in row 2)."""
    return (
        "For statistical purposes only, can you please tell me your ethnicity?,,\n"
        ",White or Caucasian,500,62.5\n"
        ",Black or African American,80,10.0\n"
        ",Hispanic or Latino,60,7.5\n"
        ",Asian,40,5.0\n"
    )


# ---------------------------------------------------------------------------
# GID discovery
# ---------------------------------------------------------------------------

def _make_mock_response(text: str = "", status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response-like object."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestDiscoverSheetGids:
    def test_extracts_gids_from_html(self):
        """Should extract GID strings from a mocked htmlview page."""
        sheet_id = "abc123"
        fake_html = """
        <html>
        <a href="#gid=0">Tab 1</a>
        <a href="#gid=1234567890">Tab 2</a>
        <a href="#gid=9876543210">Tab 3</a>
        </html>
        """
        with patch("scripts.scrape_emerson_crosstabs._get",
                   return_value=_make_mock_response(text=fake_html)):
            gids = discover_sheet_gids(sheet_id)
        # All three GIDs should be found, deduplicated
        assert "0" in gids
        assert "1234567890" in gids
        assert "9876543210" in gids

    def test_deduplicates_repeated_gids(self):
        """GIDs appearing multiple times in the HTML should only appear once."""
        sheet_id = "abc123"
        fake_html = """
        <a href="#gid=0">Tab 1</a>
        <a href="#gid=0">Tab 1 again</a>
        <a href="#gid=555">Tab 2</a>
        """
        with patch("scripts.scrape_emerson_crosstabs._get",
                   return_value=_make_mock_response(text=fake_html)):
            gids = discover_sheet_gids(sheet_id)
        assert gids.count("0") == 1
        assert "555" in gids

    def test_returns_empty_on_http_error(self):
        """Should return empty list when the htmlview page returns an HTTP error."""
        sheet_id = "bad_id"
        with patch("scripts.scrape_emerson_crosstabs._get",
                   side_effect=requests.RequestException("404 Not Found")):
            gids = discover_sheet_gids(sheet_id)
        assert gids == []

    def test_returns_empty_when_no_gids_in_html(self):
        """Should return empty list when the HTML has no gid= patterns."""
        sheet_id = "abc123"
        with patch("scripts.scrape_emerson_crosstabs._get",
                   return_value=_make_mock_response(text="<html>No GIDs here</html>")):
            gids = discover_sheet_gids(sheet_id)
        assert gids == []


# ---------------------------------------------------------------------------
# Crosstab tab identification
# ---------------------------------------------------------------------------

class TestIdentifyCrosstabGid:
    def test_identifies_crosstab_tab_by_row_n_percent(self):
        """Should return the GID of the tab that contains 'Row N %' in row 2."""
        sheet_id = "abc123"
        gid_demographics = "0"
        gid_crosstab = "999"

        side_effects = [
            _make_mock_response(text=_make_demographics_csv()),   # gid 0 → no marker
            _make_mock_response(text=_make_crosstab_csv()),       # gid 999 → marker found
        ]

        with patch("scripts.scrape_emerson_crosstabs._get",
                   side_effect=side_effects):
            with patch("time.sleep"):  # suppress delays in tests
                result = identify_crosstab_gid(
                    sheet_id, [gid_demographics, gid_crosstab]
                )
        assert result == gid_crosstab

    def test_returns_none_when_no_crosstab_tab(self):
        """Should return None when no tab has 'Row N %' in row 2."""
        sheet_id = "abc123"
        gid = "0"
        with patch("scripts.scrape_emerson_crosstabs._get",
                   return_value=_make_mock_response(text=_make_demographics_csv())):
            with patch("time.sleep"):
                result = identify_crosstab_gid(sheet_id, [gid])
        assert result is None

    def test_returns_none_for_empty_gid_list(self):
        """Should return None without making any requests when gids list is empty."""
        result = identify_crosstab_gid("abc123", [])
        assert result is None

    def test_skips_inaccessible_tabs(self):
        """Should skip tabs that return HTTP errors and continue checking others."""
        sheet_id = "abc123"
        gid_bad = "111"
        gid_good = "222"

        side_effects = [
            requests.RequestException("403 Forbidden"),           # gid_bad → error
            _make_mock_response(text=_make_crosstab_csv()),       # gid_good → found
        ]

        with patch("scripts.scrape_emerson_crosstabs._get",
                   side_effect=side_effects):
            with patch("time.sleep"):
                result = identify_crosstab_gid(sheet_id, [gid_bad, gid_good])
        assert result == gid_good


# ---------------------------------------------------------------------------
# Vote share parsing — general election question detection
# ---------------------------------------------------------------------------

class TestFindGeneralElectionQuestion:
    def test_finds_general_election_question(self):
        """Should return the cell text containing 'general election'."""
        csv_text = _make_crosstab_csv(
            question="If the 2026 general election for U.S. Senate were held today..."
        )
        rows = _parse_csv_lines(csv_text)
        question = _find_general_election_question(rows)
        assert question is not None
        assert "general election" in question.lower()

    def test_returns_none_when_no_general_election_row(self):
        """Should return None when row 0 has no 'general election' cell."""
        csv_text = "Primary question,,\nCandidate A,,\nCount,Row N %,\n"
        rows = _parse_csv_lines(csv_text)
        result = _find_general_election_question(rows)
        assert result is None

    def test_returns_none_for_empty_csv(self):
        """Should return None for completely empty input."""
        rows = _parse_csv_lines("")
        result = _find_general_election_question(rows)
        assert result is None


# ---------------------------------------------------------------------------
# Vote share parsing — Democrat column identification
# ---------------------------------------------------------------------------

class TestInferDemCandidateColumn:
    def _make_rows(
        self,
        question: str,
        dem_name: str = "Jon Ossoff",
        rep_name: str = "Bo Hines",
    ) -> list[list[str]]:
        csv_text = _make_crosstab_csv(
            question=question, dem_name=dem_name, rep_name=rep_name
        )
        return _parse_csv_lines(csv_text)

    def test_identifies_dem_column_from_democrat_keyword(self):
        """Should find the 'Row N %' column for the Democrat when the question
        contains the word 'Democrat' near the candidate name."""
        question = (
            "If the 2026 general election for U.S. Senate were held today, "
            "for whom would you vote: Democrat Jon Ossoff or Republican Bo Hines?"
        )
        rows = self._make_rows(question, dem_name="Jon Ossoff", rep_name="Bo Hines")
        col = _infer_dem_candidate_column(rows, question)
        assert col is not None
        # Column 2 should be "Row N %" for Ossoff (col 1 = Count, col 2 = Row N %)
        assert rows[2][col].lower() == "row n %"

    def test_identifies_dem_column_from_d_marker_in_name(self):
        """Should find the Dem column when candidate name cell contains '(D)'."""
        # Candidate names in row 1 at col 3 (aligned with the Row N % col 3 in row 2).
        csv_text = (
            '"If the 2026 general election for U.S. Senate were held today?",,,,,\n'
            ",,,Jon Ossoff (D),,Bo Hines (R)\n"
            ",,Count,Row N %,Count,Row N %\n"
            ',White or Caucasian,100,42.0,100,55.0\n'
        )
        rows = _parse_csv_lines(csv_text)
        question = _find_general_election_question(rows)
        col = _infer_dem_candidate_column(rows, question)
        assert col is not None
        assert rows[2][col].lower() == "row n %"

    def test_returns_none_when_no_party_signal(self):
        """Should return None when neither the question nor names indicate party."""
        # Neither the question text nor candidate names contain party indicators.
        csv_text = (
            '"If the general election were held today?",,,,,\n'
            ",,,Alice,,Bob\n"
            ",,Count,Row N %,Count,Row N %\n"
        )
        rows = _parse_csv_lines(csv_text)
        question = _find_general_election_question(rows)
        if question is None:
            pytest.skip("Question not found — precondition failure")
        col = _infer_dem_candidate_column(rows, question)
        assert col is None


# ---------------------------------------------------------------------------
# Vote share parsing — full parse_crosstab_vote_shares
# ---------------------------------------------------------------------------

class TestParseCrosstabVoteShares:
    def test_parses_race_vote_shares(self):
        """Should extract vote shares for all race groups."""
        csv_text = _make_crosstab_csv(
            question=(
                "If the 2026 general election for U.S. Senate were held today, "
                "for whom would you vote: Democrat Jon Ossoff or Republican Bo Hines?"
            ),
            dem_name="Jon Ossoff",
            rep_name="Bo Hines",
            group_rows=[
                ("White or Caucasian", "42.0", "55.0"),
                ("Black or African American", "80.0", "15.0"),
                ("Hispanic or Latino", "58.0", "38.0"),
                ("Asian", "65.0", "30.0"),
            ],
        )
        result = parse_crosstab_vote_shares(csv_text)
        assert "xt_vote_race_white" in result
        assert "xt_vote_race_black" in result
        assert "xt_vote_race_hispanic" in result
        assert "xt_vote_race_asian" in result
        # Values should be on 0–1 scale, not 0–100
        assert result["xt_vote_race_white"] == pytest.approx(0.42)
        assert result["xt_vote_race_black"] == pytest.approx(0.80)

    def test_parses_education_vote_shares(self):
        """Should extract college/noncollege vote shares."""
        csv_text = _make_crosstab_csv(
            question=(
                "If the 2026 general election for U.S. Senate were held today, "
                "for whom would you vote: Democrat Jane Smith or Republican John Doe?"
            ),
            dem_name="Jane Smith",
            rep_name="John Doe",
            group_rows=[
                ("College graduate", "60.0", "37.0"),
                ("Postgraduate", "64.0", "33.0"),
                ("High school or less", "35.0", "62.0"),
            ],
        )
        result = parse_crosstab_vote_shares(csv_text)
        # College graduate and Postgraduate both map to xt_vote_education_college
        # — they are averaged together.
        assert "xt_vote_education_college" in result
        assert result["xt_vote_education_college"] == pytest.approx(0.62)
        assert "xt_vote_education_noncollege" in result
        assert result["xt_vote_education_noncollege"] == pytest.approx(0.35)

    def test_parses_senior_age_vote_share(self):
        """Should average 60-69 and 70+ vote shares into xt_vote_age_senior."""
        csv_text = _make_crosstab_csv(
            question=(
                "If the 2026 general election for U.S. Senate were held today, "
                "for whom would you vote: Democrat Jane Smith or Republican John Doe?"
            ),
            dem_name="Jane Smith",
            rep_name="John Doe",
            group_rows=[
                ("60-69", "48.0", "49.0"),
                ("70 or more", "44.0", "53.0"),
            ],
        )
        result = parse_crosstab_vote_shares(csv_text)
        assert "xt_vote_age_senior" in result
        # Average of 0.48 and 0.44
        assert result["xt_vote_age_senior"] == pytest.approx(0.46)

    def test_returns_empty_dict_for_missing_general_election_question(self):
        """Should return {} when no 'general election' question is in row 0."""
        csv_text = (
            "Primary poll question,,\n"
            ",Candidate A,,Candidate B\n"
            ",Count,Row N %,Count,Row N %\n"
            ",White or Caucasian,100,42.0,100,55.0\n"
        )
        result = parse_crosstab_vote_shares(csv_text)
        assert result == {}

    def test_returns_empty_dict_for_too_few_rows(self):
        """Should return {} when the CSV has fewer than 3 rows."""
        result = parse_crosstab_vote_shares("row0\nrow1\n")
        assert result == {}

    def test_values_are_between_zero_and_one(self):
        """All vote share values should be in [0, 1] range."""
        csv_text = _make_crosstab_csv(
            question=(
                "If the 2026 general election for U.S. Senate were held today, "
                "for whom would you vote: Democrat Jon Ossoff or Republican Bo Hines?"
            ),
        )
        result = parse_crosstab_vote_shares(csv_text)
        for col, val in result.items():
            assert 0.0 <= val <= 1.0, f"{col}={val} is outside [0, 1]"


# ---------------------------------------------------------------------------
# Demographic label → xt_vote_* column mapping
# ---------------------------------------------------------------------------

class TestCrosstabRowLabelMap:
    """Verify that the label map covers expected demographic groups correctly."""

    def _first_match(self, label: str) -> str | None:
        """Return the first xt_vote_* column that matches the given label."""
        label_lower = label.lower()
        for pattern, xt_col in _CROSSTAB_ROW_LABEL_MAP:
            if pattern in label_lower:
                return xt_col
        return None

    def test_white_maps_correctly(self):
        assert self._first_match("White or Caucasian") == "xt_vote_race_white"

    def test_black_maps_correctly(self):
        assert self._first_match("Black or African American") == "xt_vote_race_black"

    def test_hispanic_maps_correctly(self):
        assert self._first_match("Hispanic or Latino") == "xt_vote_race_hispanic"

    def test_asian_maps_correctly(self):
        assert self._first_match("Asian") == "xt_vote_race_asian"

    def test_college_graduate_maps_correctly(self):
        assert self._first_match("College graduate") == "xt_vote_education_college"

    def test_postgraduate_maps_correctly(self):
        assert self._first_match("Postgraduate") == "xt_vote_education_college"

    def test_high_school_maps_to_noncollege(self):
        assert self._first_match("High school or less") == "xt_vote_education_noncollege"

    def test_some_college_maps_to_noncollege(self):
        assert self._first_match("Some college") == "xt_vote_education_noncollege"

    def test_sixty_sixty_nine_maps_to_senior(self):
        assert self._first_match("60-69") == "xt_vote_age_senior"

    def test_seventy_plus_maps_to_senior(self):
        assert self._first_match("70 or more") == "xt_vote_age_senior"

    def test_non_college_takes_priority_over_college(self):
        """'Non-college' should match the noncollege key, not the college key.
        This is the priority ordering gotcha — if 'college' came first in the
        map it would incorrectly match non-college rows.
        """
        result = self._first_match("Non-college educated")
        assert result == "xt_vote_education_noncollege"


# ---------------------------------------------------------------------------
# Integration: _extract_crosstabs_from_xt uses per-group dem_share
# ---------------------------------------------------------------------------

class TestExtractCrosstabsFromXtVoteShares:
    """Verify that _extract_crosstabs_from_xt prefers per-group vote shares."""

    def test_uses_per_group_share_when_present(self):
        """When xt_vote_race_black is present, that poll's Black crosstab entry
        should use 0.80 as its dem_share, not the topline 0.52."""
        poll = {
            "dem_share": 0.52,
            "n_sample": 800,
            "state": "GA",
            "xt_race_black": 0.13,         # 13% of sample is Black
            "xt_vote_race_black": 0.80,    # 80% of Black respondents chose Dem
        }
        crosstabs = _extract_crosstabs_from_xt(poll)
        assert crosstabs is not None

        black_entry = next(
            (c for c in crosstabs
             if c["demographic_group"] == "race" and c["group_value"] == "black"),
            None,
        )
        assert black_entry is not None
        # Should use the per-group share, not the topline
        assert black_entry["dem_share"] == pytest.approx(0.80)

    def test_falls_back_to_topline_when_no_vote_share(self):
        """When xt_vote_race_white is absent, the white crosstab entry should
        fall back to the topline dem_share (0.52)."""
        poll = {
            "dem_share": 0.52,
            "n_sample": 800,
            "state": "GA",
            "xt_race_white": 0.65,  # 65% of sample is white — no vote share column
        }
        crosstabs = _extract_crosstabs_from_xt(poll)
        assert crosstabs is not None

        white_entry = next(
            (c for c in crosstabs
             if c["demographic_group"] == "race" and c["group_value"] == "white"),
            None,
        )
        assert white_entry is not None
        # No xt_vote_race_white → fall back to topline
        assert white_entry["dem_share"] == pytest.approx(0.52)

    def test_mixed_poll_some_groups_have_vote_share(self):
        """Some groups have per-group vote shares, others don't.
        Each should independently use the best available value."""
        poll = {
            "dem_share": 0.52,
            "n_sample": 800,
            "state": "GA",
            "xt_race_black": 0.13,
            "xt_vote_race_black": 0.80,    # per-group share available
            "xt_race_white": 0.65,          # no xt_vote_race_white → uses topline
            "xt_education_college": 0.45,
            "xt_vote_education_college": 0.68,  # per-group share available
        }
        crosstabs = _extract_crosstabs_from_xt(poll)
        assert crosstabs is not None

        entries = {
            (c["demographic_group"], c["group_value"]): c["dem_share"]
            for c in crosstabs
        }
        # Black: has per-group share
        assert entries[("race", "black")] == pytest.approx(0.80)
        # White: no per-group share → topline
        assert entries[("race", "white")] == pytest.approx(0.52)
        # Education college: has per-group share
        assert entries[("education", "college")] == pytest.approx(0.68)

    def test_returns_none_when_no_xt_columns(self):
        """A poll with no xt_* columns should return None (Tier 3 fallback)."""
        poll = {"dem_share": 0.52, "n_sample": 800, "state": "GA"}
        assert _extract_crosstabs_from_xt(poll) is None

    def test_ignores_invalid_vote_share_values(self):
        """An xt_vote_* value that cannot be parsed as float should be ignored,
        and the entry should fall back to the topline dem_share."""
        poll = {
            "dem_share": 0.52,
            "n_sample": 800,
            "state": "GA",
            "xt_race_black": 0.13,
            "xt_vote_race_black": "N/A",  # unparseable
        }
        crosstabs = _extract_crosstabs_from_xt(poll)
        assert crosstabs is not None

        black_entry = next(
            (c for c in crosstabs
             if c["demographic_group"] == "race" and c["group_value"] == "black"),
            None,
        )
        assert black_entry is not None
        # Should fall back to topline since "N/A" is not a valid float
        assert black_entry["dem_share"] == pytest.approx(0.52)

    def test_ignores_out_of_range_vote_share(self):
        """An xt_vote_* value outside [0, 1] is nonsensical — fall back to topline."""
        poll = {
            "dem_share": 0.52,
            "n_sample": 800,
            "state": "GA",
            "xt_race_black": 0.13,
            "xt_vote_race_black": 1.5,  # out of range (someone stored raw %)
        }
        crosstabs = _extract_crosstabs_from_xt(poll)
        assert crosstabs is not None

        black_entry = next(
            (c for c in crosstabs
             if c["demographic_group"] == "race" and c["group_value"] == "black"),
            None,
        )
        assert black_entry is not None
        # 1.5 is out of [0,1] → fall back to topline
        assert black_entry["dem_share"] == pytest.approx(0.52)

    def test_pct_of_sample_is_from_composition_not_vote_share(self):
        """The pct_of_sample field should still come from the xt_race_* composition
        column, not from the xt_vote_* column.  They measure different things."""
        poll = {
            "dem_share": 0.52,
            "n_sample": 800,
            "state": "GA",
            "xt_race_black": 0.13,
            "xt_vote_race_black": 0.80,
        }
        crosstabs = _extract_crosstabs_from_xt(poll)
        assert crosstabs is not None

        black_entry = next(
            (c for c in crosstabs
             if c["demographic_group"] == "race" and c["group_value"] == "black"),
            None,
        )
        assert black_entry is not None
        # pct_of_sample comes from xt_race_black (0.13), not xt_vote_race_black (0.80)
        assert black_entry["pct_of_sample"] == pytest.approx(0.13)
        assert black_entry["dem_share"] == pytest.approx(0.80)
