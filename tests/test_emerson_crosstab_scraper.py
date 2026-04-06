"""
Tests for scripts/scrape_emerson_crosstabs.py.

All tests use fixture data (no live HTTP calls).
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import Optional

import pytest

# Make the scripts/ directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from scrape_emerson_crosstabs import (
    XT_COLUMNS,
    extract_sheet_url,
    infer_state_from_url,
    match_and_update_polls,
    parse_demographics,
)


# ---------------------------------------------------------------------------
# Fixtures: representative CSV text from each known Emerson sheet
# ---------------------------------------------------------------------------

# Florida sheet demographics section (lines 200–242 of the real export)
FL_CSV_FRAGMENT = """\
For statistical purposes only can you please tell me your ethnicity?,,,
,,Frequency,Valid Percent
,Hispanic or Latino of any race,241,21.4
,White or Caucasian,625,55.6
,Black or African American,200,17.8
,Asian,11,1.0
,Other or multiple races,47,4.2
,Total,1125,100.0
,,,
What is your age range?,,,
,,Frequency,Valid Percent
,18-29 years,104,9.2
,30-39 years,128,11.4
,40-49 years,153,13.6
,50-59 years,187,16.6
,60-69 years,247,22.0
,70 or more years,306,27.2
,Total,1125,100.0
,,,
What is the highest level of education you have attained?,,,
,,Frequency,Valid Percent
,High school or less,182,16.2
,Vocational/technical school,84,7.5
,Associate Degree/some college,316,28.1
,College graduate,318,28.3
,Postgraduate or higher,224,19.9
,Total,1125,100.0
,,,
"""

# Georgia sheet (slightly different age grouping, no 18-29 split)
GA_CSV_FRAGMENT = """\
For statistical purposes only can you please tell me your ethnicity?,,,
,,Frequency,Valid Percent
,Hispanic or Latino of any race,81,8.1
,White or Caucasian,602,60.2
,Black or African American,286,28.6
,Asian,17,1.7
,Other or multiple races,14,1.4
,Total,1000,100.0
,,,
Age,,,
,,Frequency,Valid Percent
,18-39 years,267,26.7
,40-49 years,165,16.5
,50-59 years,214,21.4
,60-69 years,187,18.7
,70 or more years,168,16.8
,Total,1000,100.0
,,,
What is the highest level of education you have attained?,,,
,,Frequency,Valid Percent
,High school or less,170,17.0
,Vocational/technical school,90,9.0
,Associate Degree/some college,346,34.6
,College graduate,238,23.8
,Postgraduate or higher,157,15.7
,Total,1000,100.0
,,,
"""

# Maine sheet (simplified race: only White / Non-white)
ME_CSV_FRAGMENT = """\
Race,,,
,,Frequency,Valid Percent
Valid,Non-white,95,8.8
,White,980,91.2
,Total,1075,100.0
,,,
What is your age range?,,,
,,Frequency,Valid Percent
,18-39 years,216,20.1
,40-49 years,165,15.4
,50-59 years,208,19.4
,60-69 years,231,21.5
,70 or more years,254,23.7
,Total,1075,100.0
,,,
What is the highest level of education you have attained?,,,
,,Frequency,Valid Percent
,High school or less,194,18.0
,Vocational/technical school,101,9.4
,Associate Degree/some college,307,28.6
,College graduate,266,24.7
,Postgraduate or higher,208,19.3
,Total,1075,100.0
,,,
"""

# Texas sheet (ethnicity present, bare "Age," header)
TX_CSV_FRAGMENT = """\
For statistical purposes only can you please tell me your ethnicity?,,,
,,Frequency,Valid Percent
,Hispanic or Latino of any race,243,28.6
,White or Caucasian,339,39.9
,Black or African American,226,26.6
,Asian,30,3.5
,Other or multiple races,12,1.4
,Total,850,100.0
,,,
Age,,,
,,Frequency,Valid Percent
,18-29 years,84,9.9
,30-39 years,102,12.0
,40-49 years,121,14.2
,50-59 years,147,17.3
,60-69 years,177,20.8
,70 or more years,219,25.8
,Total,850,100.0
,,,
What is the highest level of education you have attained?,,,
,,Frequency,Valid Percent
,High school or less,138,16.2
,Vocational/technical school,76,8.9
,Associate Degree/some college,250,29.4
,College graduate,222,26.1
,Postgraduate or higher,164,19.3
,Total,850,100.0
,,,
"""


# ---------------------------------------------------------------------------
# Tests: parse_demographics — Florida
# ---------------------------------------------------------------------------

class TestParseDemographicsFL:
    def test_race_white(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_race_white"] == pytest.approx(0.556, abs=0.001)

    def test_race_black(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_race_black"] == pytest.approx(0.178, abs=0.001)

    def test_race_hispanic(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_race_hispanic"] == pytest.approx(0.214, abs=0.001)

    def test_race_asian(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_race_asian"] == pytest.approx(0.010, abs=0.001)

    def test_age_senior(self):
        # 60-69 (22.0) + 70+ (27.2) = 49.2%
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_age_senior"] == pytest.approx(0.492, abs=0.001)

    def test_education_college(self):
        # College grad (28.3) + Postgrad (19.9) = 48.2%
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_education_college"] == pytest.approx(0.482, abs=0.001)

    def test_education_noncollege(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_education_noncollege"] == pytest.approx(0.518, abs=0.001)

    def test_education_sums_to_one(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        total = d["xt_education_college"] + d["xt_education_noncollege"]
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_unavailable_fields_are_none(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        assert d["xt_urbanicity_urban"] is None
        assert d["xt_urbanicity_rural"] is None
        assert d["xt_religion_evangelical"] is None

    def test_all_xt_columns_present(self):
        d = parse_demographics(FL_CSV_FRAGMENT)
        for col in XT_COLUMNS:
            assert col in d


# ---------------------------------------------------------------------------
# Tests: parse_demographics — Georgia (different age grouping)
# ---------------------------------------------------------------------------

class TestParseDemographicsGA:
    def test_race_white(self):
        d = parse_demographics(GA_CSV_FRAGMENT)
        assert d["xt_race_white"] == pytest.approx(0.602, abs=0.001)

    def test_race_black(self):
        d = parse_demographics(GA_CSV_FRAGMENT)
        assert d["xt_race_black"] == pytest.approx(0.286, abs=0.001)

    def test_age_senior(self):
        # 60-69 (18.7) + 70+ (16.8) = 35.5%
        d = parse_demographics(GA_CSV_FRAGMENT)
        assert d["xt_age_senior"] == pytest.approx(0.355, abs=0.001)

    def test_education_college(self):
        # 23.8 + 15.7 = 39.5%
        d = parse_demographics(GA_CSV_FRAGMENT)
        assert d["xt_education_college"] == pytest.approx(0.395, abs=0.001)


# ---------------------------------------------------------------------------
# Tests: parse_demographics — Maine (simplified race)
# ---------------------------------------------------------------------------

class TestParseDemographicsME:
    def test_race_white_available(self):
        d = parse_demographics(ME_CSV_FRAGMENT)
        assert d["xt_race_white"] == pytest.approx(0.912, abs=0.001)

    def test_race_other_not_available(self):
        # Detailed breakdown absent — these should be None
        d = parse_demographics(ME_CSV_FRAGMENT)
        assert d["xt_race_black"] is None
        assert d["xt_race_hispanic"] is None
        assert d["xt_race_asian"] is None

    def test_age_senior(self):
        # 60-69 (21.5) + 70+ (23.7) = 45.2%
        d = parse_demographics(ME_CSV_FRAGMENT)
        assert d["xt_age_senior"] == pytest.approx(0.452, abs=0.001)

    def test_education_college(self):
        # 24.7 + 19.3 = 44.0%
        d = parse_demographics(ME_CSV_FRAGMENT)
        assert d["xt_education_college"] == pytest.approx(0.440, abs=0.001)


# ---------------------------------------------------------------------------
# Tests: parse_demographics — Texas (bare "Age" header)
# ---------------------------------------------------------------------------

class TestParseDemographicsTX:
    def test_race_hispanic(self):
        d = parse_demographics(TX_CSV_FRAGMENT)
        assert d["xt_race_hispanic"] == pytest.approx(0.286, abs=0.001)

    def test_age_senior(self):
        # 60-69 (20.8) + 70+ (25.8) = 46.6%
        d = parse_demographics(TX_CSV_FRAGMENT)
        assert d["xt_age_senior"] == pytest.approx(0.466, abs=0.001)

    def test_education_college(self):
        # 26.1 + 19.3 = 45.4%
        d = parse_demographics(TX_CSV_FRAGMENT)
        assert d["xt_education_college"] == pytest.approx(0.454, abs=0.001)


# ---------------------------------------------------------------------------
# Tests: extract_sheet_url
# ---------------------------------------------------------------------------

class TestExtractSheetUrl:
    def test_standard_edit_url(self):
        html = """
        <p>See <a href="https://docs.google.com/spreadsheets/d/1b8oV9f1Zzl_Uew5WPkIIeFLjVMJuHNf7/edit?usp=sharing">FULL RESULTS</a></p>
        """
        url = extract_sheet_url(html)
        assert url == "https://docs.google.com/spreadsheets/d/1b8oV9f1Zzl_Uew5WPkIIeFLjVMJuHNf7/export?format=csv"

    def test_gid_fragment_url(self):
        html = """
        <a href="https://docs.google.com/spreadsheets/d/1jm6cifqqT8NRH_qpVamcP3BDevlPuVtI/edit?gid=619517400#gid=619517400">FULL RESULTS</a>
        """
        url = extract_sheet_url(html)
        assert url == "https://docs.google.com/spreadsheets/d/1jm6cifqqT8NRH_qpVamcP3BDevlPuVtI/export?format=csv"

    def test_no_sheet_link_returns_none(self):
        html = "<p>No spreadsheet here</p>"
        assert extract_sheet_url(html) is None

    def test_extracts_sheet_id_only(self):
        html = 'href="https://docs.google.com/spreadsheets/d/SHEET_ID_XYZ/export?format=csv"'
        url = extract_sheet_url(html)
        assert "SHEET_ID_XYZ" in url


# ---------------------------------------------------------------------------
# Tests: infer_state_from_url
# ---------------------------------------------------------------------------

class TestInferStateFromUrl:
    def test_florida(self):
        url = "https://emersoncollegepolling.com/florida-2026-poll-donalds-leads/"
        assert infer_state_from_url(url) == "FL"

    def test_georgia(self):
        url = "https://emersoncollegepolling.com/georgia-2026-poll-senator-ossoff/"
        assert infer_state_from_url(url) == "GA"

    def test_maine(self):
        url = "https://emersoncollegepolling.com/maine-2026-poll-platner-leads/"
        assert infer_state_from_url(url) == "ME"

    def test_texas(self):
        url = "https://emersoncollegepolling.com/texas-2026-primary-poll-talarico/"
        assert infer_state_from_url(url) == "TX"

    def test_north_carolina(self):
        url = "https://emersoncollegepolling.com/north-carolina-2026-poll-xxx/"
        assert infer_state_from_url(url) == "NC"

    def test_unknown_state_returns_none(self):
        url = "https://emersoncollegepolling.com/unknown-2026-poll-xxx/"
        assert infer_state_from_url(url) is None


# ---------------------------------------------------------------------------
# Tests: match_and_update_polls
# ---------------------------------------------------------------------------

def _make_poll_rows() -> list[dict]:
    """Create a minimal set of poll rows for matching tests."""
    base = {col: "" for col in XT_COLUMNS}
    return [
        {**base, "race": "2026 FL Governor", "geography": "FL", "pollster": "Emerson College",
         "n_sample": "1125.0", "date": "2026-03-31"},
        {**base, "race": "2026 FL Senate", "geography": "FL", "pollster": "Emerson College",
         "n_sample": "1125.0", "date": "2026-03-31"},
        {**base, "race": "2026 GA Senate", "geography": "GA", "pollster": "Emerson College",
         "n_sample": "1000.0", "date": "2026-03-02"},
        {**base, "race": "2026 AZ Governor", "geography": "AZ", "pollster": "Some Other Firm",
         "n_sample": "900.0", "date": "2026-01-01"},
    ]


class TestMatchAndUpdatePolls:
    def test_updates_matching_state(self):
        rows = _make_poll_rows()
        demographics = {"xt_race_white": 0.556, "xt_age_senior": 0.492}
        # Fill remaining xt_ keys with None
        demographics = {**{col: None for col in XT_COLUMNS}, **demographics}
        count = match_and_update_polls(rows, "FL", demographics)
        assert count == 2  # FL Governor + FL Senate

    def test_does_not_update_other_state(self):
        rows = _make_poll_rows()
        demographics = {col: 0.5 for col in XT_COLUMNS}
        match_and_update_polls(rows, "FL", demographics)
        # GA row should not be touched
        ga_row = next(r for r in rows if r["geography"] == "GA")
        assert ga_row["xt_race_white"] == ""

    def test_does_not_update_wrong_pollster(self):
        rows = _make_poll_rows()
        demographics = {col: 0.5 for col in XT_COLUMNS}
        match_and_update_polls(rows, "AZ", demographics)
        az_row = next(r for r in rows if r["geography"] == "AZ")
        assert az_row["xt_race_white"] == ""

    def test_idempotent_on_rerun(self):
        rows = _make_poll_rows()
        demographics = {**{col: None for col in XT_COLUMNS}, "xt_race_white": 0.556}
        first = match_and_update_polls(rows, "FL", demographics)
        second = match_and_update_polls(rows, "FL", demographics)
        assert first == 2
        assert second == 0  # already set, nothing changed

    def test_n_sample_filtering(self):
        rows = _make_poll_rows()
        # Add a second FL poll with a different n_sample (different survey)
        rows.append({
            **{col: "" for col in XT_COLUMNS},
            "race": "2026 FL Generic", "geography": "FL",
            "pollster": "Emerson College", "n_sample": "800.0", "date": "2026-01-01",
        })
        demographics = {**{col: None for col in XT_COLUMNS}, "xt_race_white": 0.556}
        # With n_sample=1125, should only update the 1125-sample rows
        count = match_and_update_polls(rows, "FL", demographics, n_sample=1125.0)
        assert count == 2

    def test_xt_values_written_as_strings(self):
        rows = _make_poll_rows()
        demographics = {**{col: None for col in XT_COLUMNS}, "xt_race_white": 0.556}
        match_and_update_polls(rows, "GA", demographics)
        ga_row = next(r for r in rows if r["geography"] == "GA")
        assert ga_row["xt_race_white"] == "0.556000"

    def test_dry_run_does_not_write(self):
        rows = _make_poll_rows()
        demographics = {**{col: None for col in XT_COLUMNS}, "xt_race_white": 0.556}
        count = match_and_update_polls(rows, "FL", demographics, dry_run=True)
        # Count says updates would happen
        assert count == 2
        # But no actual write
        for row in rows:
            if row["geography"] == "FL":
                assert row["xt_race_white"] == ""

    def test_none_values_not_overwritten(self):
        """None values in demographics should not overwrite existing non-empty values."""
        rows = _make_poll_rows()
        # Pre-populate
        for row in rows:
            if row["geography"] == "GA":
                row["xt_race_white"] = "0.602000"
        demographics = {col: None for col in XT_COLUMNS}
        count = match_and_update_polls(rows, "GA", demographics)
        ga_row = next(r for r in rows if r["geography"] == "GA")
        # Should remain unchanged since all values are None
        assert ga_row["xt_race_white"] == "0.602000"
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: xt_* value range validation
# ---------------------------------------------------------------------------

class TestXtValueRanges:
    @pytest.mark.parametrize("csv_fragment", [FL_CSV_FRAGMENT, GA_CSV_FRAGMENT, TX_CSV_FRAGMENT])
    def test_all_values_in_0_1(self, csv_fragment: str):
        d = parse_demographics(csv_fragment)
        for key, val in d.items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    @pytest.mark.parametrize("csv_fragment", [FL_CSV_FRAGMENT, GA_CSV_FRAGMENT, TX_CSV_FRAGMENT])
    def test_education_sums_to_one(self, csv_fragment: str):
        d = parse_demographics(csv_fragment)
        if d["xt_education_college"] is not None and d["xt_education_noncollege"] is not None:
            assert d["xt_education_college"] + d["xt_education_noncollege"] == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("csv_fragment", [FL_CSV_FRAGMENT, GA_CSV_FRAGMENT, TX_CSV_FRAGMENT])
    def test_age_senior_reasonable(self, csv_fragment: str):
        """Senior share (60+) should be between 20% and 70% for US state polls."""
        d = parse_demographics(csv_fragment)
        if d["xt_age_senior"] is not None:
            assert 0.20 <= d["xt_age_senior"] <= 0.70


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_csv(self):
        d = parse_demographics("")
        for col in XT_COLUMNS:
            assert d[col] is None

    def test_partial_education_only(self):
        csv_text = """\
What is the highest level of education you have attained?,,,
,,Frequency,Valid Percent
,College graduate,300,30.0
,Postgraduate or higher,200,20.0
,Total,1000,100.0
"""
        d = parse_demographics(csv_text)
        assert d["xt_education_college"] == pytest.approx(0.50, abs=0.001)
        assert d["xt_education_noncollege"] == pytest.approx(0.50, abs=0.001)
        assert d["xt_race_white"] is None
        assert d["xt_age_senior"] is None

    def test_malformed_percent_value_returns_none(self):
        csv_text = """\
For statistical purposes only can you please tell me your ethnicity?,,,
,,Frequency,Valid Percent
,White or Caucasian,n/a,n/a
,Total,1000,100.0
"""
        d = parse_demographics(csv_text)
        assert d["xt_race_white"] is None

    def test_compact_csv_format_with_pct_suffix(self):
        """OH uses compact 3-column layout with % suffixes: label,N,%"""
        csv_text = """\
"For statistical purposes only, can you please tell me your ethnicity?",,
,N,%
Hispanic or Latino of any race,33,3.8%
White or Caucasian,691,81.3%
Black or African American,99,11.6%
Asian,9,1.1%
Other or multiple races,18,2.2%
Total,850,100.0%
,,
What is your age range?,,
,N,%
18-29 years,130,15.3%
30-39 years,130,15.3%
40-49 years,139,16.3%
50-59 years,173,20.4%
60-69 years,139,16.3%
70 or more years,139,16.3%
Total,850,100.0%
,,
What is the highest level of education you have attained?,,
,N,%
High school or less,235,27.6%
Vocational/technical school,84,9.9%
Associate Degree/some college,229,27.0%
College graduate,179,21.1%
Postgraduate or higher,122,14.4%
Total,850,100.0%
"""
        d = parse_demographics(csv_text)
        assert d["xt_race_white"] == pytest.approx(0.813, abs=0.001)
        assert d["xt_race_black"] == pytest.approx(0.116, abs=0.001)
        assert d["xt_race_hispanic"] == pytest.approx(0.038, abs=0.001)
        assert d["xt_race_asian"] == pytest.approx(0.011, abs=0.001)
        assert d["xt_age_senior"] == pytest.approx(0.326, abs=0.001)
        assert d["xt_education_college"] == pytest.approx(0.355, abs=0.001)
        assert d["xt_education_noncollege"] == pytest.approx(0.645, abs=0.001)
