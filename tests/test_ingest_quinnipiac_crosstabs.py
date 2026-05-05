"""Tests for tools/ingest_quinnipiac_crosstabs.py.

Coverage:
  1. parse_pdf_crosstabs — parses a synthetic PDF-text fixture representing
     both the "detailed" format (PA-style: party, gender, education, race, age)
     and the "simple" format (NJ-style: party, gender only).
  2. build_crosstab_records — constructs DB-ready dicts with correct poll_id,
     NULL pct_of_sample, and correct dem_share values.
  3. ingest_to_db — upserts records into a fresh DuckDB connection, verifies
     row count and idempotency (re-run replaces rather than duplicates rows).
  4. _column_map — unit tests for column header → demographic group mapping.
  5. _parse_pct — unit tests for percentage string parsing.
  6. main() CLI — smoke test with a synthetic PDF that covers all code paths.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pdfplumber
import pytest

# Make tools importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from ingest_quinnipiac_crosstabs import (
    _column_map,
    _parse_pct,
    _parse_tables_from_block,
    build_crosstab_records,
    ingest_to_db,
    main,
    parse_pdf_crosstabs,
)
from src.db.domains.polling import _make_poll_id, create_tables


# ---------------------------------------------------------------------------
# Fixtures: synthetic Q1 block text
# ---------------------------------------------------------------------------

# Represents a PA-style detailed poll with education, race, age, and gender.
DETAILED_Q1_BLOCK = """\
1. If the election for governor were being held today, and the candidates were Josh Shapiro
the Democrat and Stacy Garrity the Republican, for whom would you vote? (INCLUDES LEANERS)
REGISTERED VOTERS......................................
WHITE........
4 YR COLL DEG
Tot Rep Dem Ind Men Wom Yes No
Shapiro 55% 13% 98% 61% 48% 61% 68% 41%
Garrity 39 82 1 27 45 34 30 53
SMONE ELSE(VOL) 1 1 - 2 1 1 - -
WLDN'T VOTE(VOL) - - - - - - - -
UNDECIDED(VOL) 4 4 1 8 4 4 2 4
REFUSED 1 - 1 1 2 1 1 1
AGE IN YRS.............. WHITE.....
18-34 35-49 50-64 65+ Men Wom Wht Blk
Shapiro 64% 57% 49% 56% 45% 57% 52% 89%
Garrity 27 38 47 39 50 39 44 7
SMONE ELSE(VOL) 2 2 - - - - - 1
WLDN'T VOTE(VOL) - 1 - - - - - 2
UNDECIDED(VOL) 6 2 4 3 3 3 3 2
REFUSED 1 - - 1 1 1 1 -
"""

# Represents an NJ-style simple poll with only party and gender.
SIMPLE_Q1_BLOCK = """\
1. If the election for governor were being held today, and the candidates were Mikie Sherrill
the Democrat, Jack Ciattarelli the Republican, for whom would you vote?
LIKELY VOTERS.........................
Tot Rep Dem Ind Men Wom
Sherrill 51% 4% 94% 44% 43% 57%
Ciattarelli 43 93 1 47 50 37
SMONE ELSE(VOL) - - - - - -
WLDN'T VOTE(VOL) - - - - - -
UNDECIDED(VOL) 3 1 3 2 2 4
REFUSED 1 1 - 3 1 1
"""

# Simulates a full Quinnipiac PDF text with detailed Q1 + a Q1a sub-question.
DETAILED_PDF_TEXT = DETAILED_Q1_BLOCK + """\
1a. (If candidate chosen q1) Would you say you are very enthusiastic about supporting
(candidate of choice)?
LIKELY VOTERS...................
CANDIDATE CHOSEN Q1.............
Tot Sherrill Garrity
Very enthusiastic 53% 45% 62%
"""

SIMPLE_PDF_TEXT = SIMPLE_Q1_BLOCK + """\
1a. (If candidate chosen q1) Would you say you are very enthusiastic?
LIKELY VOTERS...................
Tot Sherrill Ciattarelli
Very enthusiastic 47% 42% 55%
"""


def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal PDF with one page containing the given text.

    Uses reportlab if available, otherwise creates a placeholder byte stream
    and monkeypatches pdfplumber to return the raw text. This avoids a hard
    dependency on reportlab in the test suite.
    """
    # We'll just mock extract_text to return our text instead of building a real PDF.
    # This is handled in the test via patching pdfplumber.open().
    return text.encode("utf-8")  # placeholder


def _patch_pdf(text: str):
    """Context manager: patches pdfplumber.open() to yield pages with given text."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = text
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]
    return patch("ingest_quinnipiac_crosstabs.pdfplumber.open", return_value=mock_pdf)


# ---------------------------------------------------------------------------
# Unit: _parse_pct
# ---------------------------------------------------------------------------

class TestParsePct:
    def test_percentage_string(self):
        assert _parse_pct("55%") == pytest.approx(0.55)

    def test_plain_number(self):
        assert _parse_pct("39") == pytest.approx(0.39)

    def test_dash(self):
        assert _parse_pct("-") is None

    def test_double_dash(self):
        assert _parse_pct("--") is None

    def test_empty(self):
        assert _parse_pct("") is None

    def test_zero(self):
        assert _parse_pct("0") == pytest.approx(0.0)

    def test_decimal(self):
        assert _parse_pct("4.5%") == pytest.approx(0.045)


# ---------------------------------------------------------------------------
# Unit: _column_map
# ---------------------------------------------------------------------------

class TestColumnMap:
    def test_standard_party_gender_no_education(self):
        headers = ["Tot", "Rep", "Dem", "Ind", "Men", "Wom"]
        mapping = _column_map(headers, context="LIKELY VOTERS")
        assert mapping[0] is None          # Tot
        assert mapping[1] == ("party", "republican")
        assert mapping[2] == ("party", "democrat")
        assert mapping[3] == ("party", "independent")
        assert mapping[4] == ("gender", "men")
        assert mapping[5] == ("gender", "women")

    def test_education_columns_detected(self):
        headers = ["Tot", "Rep", "Dem", "Ind", "Men", "Wom", "Yes", "No"]
        context = "WHITE........\n4 YR COLL DEG"
        mapping = _column_map(headers, context)
        assert mapping[6] == ("education", "college")   # Yes
        assert mapping[7] == ("education", "noncollege")  # No

    def test_yes_no_without_education_context_is_none(self):
        headers = ["Tot", "Rep", "Dem", "Ind", "Men", "Wom", "Yes", "No"]
        mapping = _column_map(headers, context="LIKELY VOTERS")
        assert mapping[6] is None  # Yes without COLL DEG context → skip
        assert mapping[7] is None

    def test_race_age_table(self):
        headers = ["18-34", "35-49", "50-64", "65+", "Men", "Wom", "Wht", "Blk"]
        mapping = _column_map(headers, context="AGE IN YRS.............. WHITE.....")
        assert mapping[0] == ("age", "18_34")
        assert mapping[1] == ("age", "35_49")
        assert mapping[2] == ("age", "50_64")
        assert mapping[3] == ("age", "65_plus")
        assert mapping[4] is None   # White Men → skip
        assert mapping[5] is None   # White Women → skip
        assert mapping[6] == ("race", "white")
        assert mapping[7] == ("race", "black")


# ---------------------------------------------------------------------------
# Unit: _parse_tables_from_block
# ---------------------------------------------------------------------------

class TestParseTablesFromBlock:
    def test_detailed_block_shapiro(self):
        rows = _parse_tables_from_block(DETAILED_Q1_BLOCK, "Shapiro")
        group_map = {(r["demographic_group"], r["group_value"]): r["dem_share"] for r in rows}

        assert group_map[("party", "republican")] == pytest.approx(0.13)
        assert group_map[("party", "democrat")] == pytest.approx(0.98)
        assert group_map[("party", "independent")] == pytest.approx(0.61)
        assert group_map[("gender", "men")] == pytest.approx(0.48)
        assert group_map[("gender", "women")] == pytest.approx(0.61)
        assert group_map[("education", "college")] == pytest.approx(0.68)
        assert group_map[("education", "noncollege")] == pytest.approx(0.41)
        assert group_map[("race", "white")] == pytest.approx(0.52)
        assert group_map[("race", "black")] == pytest.approx(0.89)
        assert group_map[("age", "18_34")] == pytest.approx(0.64)
        assert group_map[("age", "65_plus")] == pytest.approx(0.56)

    def test_detailed_block_no_duplicate_groups(self):
        rows = _parse_tables_from_block(DETAILED_Q1_BLOCK, "Shapiro")
        group_keys = [(r["demographic_group"], r["group_value"]) for r in rows]
        assert len(group_keys) == len(set(group_keys)), "Duplicate demographic groups found"

    def test_simple_block_sherrill(self):
        rows = _parse_tables_from_block(SIMPLE_Q1_BLOCK, "Sherrill")
        group_map = {(r["demographic_group"], r["group_value"]): r["dem_share"] for r in rows}

        assert group_map[("party", "republican")] == pytest.approx(0.04)
        assert group_map[("party", "democrat")] == pytest.approx(0.94)
        assert group_map[("party", "independent")] == pytest.approx(0.44)
        assert group_map[("gender", "men")] == pytest.approx(0.43)
        assert group_map[("gender", "women")] == pytest.approx(0.57)
        # No education or race in simple format
        assert ("education", "college") not in group_map
        assert ("race", "white") not in group_map

    def test_empty_block_returns_empty(self):
        rows = _parse_tables_from_block("No table data here", "Shapiro")
        assert rows == []


# ---------------------------------------------------------------------------
# Integration: parse_pdf_crosstabs (with pdfplumber mocked)
# ---------------------------------------------------------------------------

class TestParsePdfCrosstabs:
    def test_detailed_pdf_extracts_13_groups(self):
        with _patch_pdf(DETAILED_PDF_TEXT):
            records, dem_name = parse_pdf_crosstabs(b"fake-pdf-bytes")
        assert dem_name is not None
        assert "Shapiro" in dem_name
        assert len(records) == 13  # party×3, gender×2, edu×2, race×2, age×4

    def test_simple_pdf_extracts_5_groups(self):
        with _patch_pdf(SIMPLE_PDF_TEXT):
            records, dem_name = parse_pdf_crosstabs(b"fake-pdf-bytes")
        assert dem_name is not None
        assert "Sherrill" in dem_name
        assert len(records) == 5   # party×3, gender×2

    def test_no_election_question_returns_empty(self):
        with _patch_pdf("This PDF contains no election question."):
            records, dem_name = parse_pdf_crosstabs(b"fake-pdf-bytes")
        assert records == []
        assert dem_name is None


# ---------------------------------------------------------------------------
# Integration: build_crosstab_records
# ---------------------------------------------------------------------------

class TestBuildCrosstabRecords:
    def test_poll_id_attached(self):
        parsed = [{"demographic_group": "race", "group_value": "black", "dem_share": 0.89}]
        poll_id = "abc123"
        records = build_crosstab_records(parsed, poll_id, total_n=1000)
        assert all(r["poll_id"] == poll_id for r in records)

    def test_pct_of_sample_is_none(self):
        parsed = [{"demographic_group": "gender", "group_value": "men", "dem_share": 0.45}]
        records = build_crosstab_records(parsed, "pid", total_n=None)
        assert records[0]["pct_of_sample"] is None

    def test_n_sample_is_none(self):
        parsed = [{"demographic_group": "gender", "group_value": "women", "dem_share": 0.55}]
        records = build_crosstab_records(parsed, "pid", total_n=None)
        assert records[0]["n_sample"] is None

    def test_dem_share_preserved(self):
        parsed = [{"demographic_group": "race", "group_value": "white", "dem_share": 0.52}]
        records = build_crosstab_records(parsed, "pid", total_n=1579)
        assert records[0]["dem_share"] == pytest.approx(0.52)


# ---------------------------------------------------------------------------
# Integration: ingest_to_db
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh_db():
    con = duckdb.connect(":memory:")
    create_tables(con)
    yield con
    con.close()


class TestIngestToDb:
    def test_inserts_records(self, fresh_db):
        records = [
            {"poll_id": "pid1", "demographic_group": "race", "group_value": "black",
             "dem_share": 0.89, "n_sample": None, "pct_of_sample": None},
            {"poll_id": "pid1", "demographic_group": "gender", "group_value": "men",
             "dem_share": 0.45, "n_sample": None, "pct_of_sample": None},
        ]
        n = ingest_to_db(fresh_db, records)
        assert n == 2
        count = fresh_db.execute("SELECT count(*) FROM poll_crosstabs WHERE poll_id='pid1'").fetchone()[0]
        assert count == 2

    def test_dry_run_does_not_write(self, fresh_db):
        records = [
            {"poll_id": "pid_dry", "demographic_group": "race", "group_value": "white",
             "dem_share": 0.52, "n_sample": None, "pct_of_sample": None},
        ]
        n = ingest_to_db(fresh_db, records, dry_run=True)
        assert n == 1
        count = fresh_db.execute("SELECT count(*) FROM poll_crosstabs WHERE poll_id='pid_dry'").fetchone()[0]
        assert count == 0

    def test_idempotent_rerun_replaces_not_duplicates(self, fresh_db):
        records = [
            {"poll_id": "pid2", "demographic_group": "party", "group_value": "republican",
             "dem_share": 0.13, "n_sample": None, "pct_of_sample": None},
        ]
        ingest_to_db(fresh_db, records)
        ingest_to_db(fresh_db, records)  # second run
        count = fresh_db.execute("SELECT count(*) FROM poll_crosstabs WHERE poll_id='pid2'").fetchone()[0]
        assert count == 1  # not 2

    def test_empty_records_returns_zero(self, fresh_db):
        n = ingest_to_db(fresh_db, [])
        assert n == 0


# ---------------------------------------------------------------------------
# Smoke: main() CLI with mocked PDF
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_main_dry_run_detailed(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        # Create the table first
        con = duckdb.connect(db_path)
        create_tables(con)
        con.close()

        with _patch_pdf(DETAILED_PDF_TEXT):
            ret = main([
                "--pdf", "/fake/path.pdf",
                "--race", "2026 PA Governor",
                "--geography", "PA",
                "--date", "2025-10-01",
                "--cycle", "2026",
                "--db", db_path,
                "--dry-run",
            ])
        assert ret == 0

    def test_main_writes_to_db(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        con = duckdb.connect(db_path)
        create_tables(con)
        con.close()

        with _patch_pdf(SIMPLE_PDF_TEXT):
            ret = main([
                "--pdf", "/fake/path.pdf",
                "--race", "2026 NJ Governor",
                "--geography", "NJ",
                "--date", "2025-10-30",
                "--cycle", "2026",
                "--db", db_path,
            ])
        assert ret == 0
        con = duckdb.connect(db_path)
        count = con.execute("SELECT count(*) FROM poll_crosstabs").fetchone()[0]
        con.close()
        assert count == 5  # NJ simple: party×3 + gender×2

    def test_main_missing_pdf_returns_error(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        ret = main([
            "--pdf", "/nonexistent/path.pdf",
            "--race", "2026 NJ Governor",
            "--geography", "NJ",
            "--date", "2025-10-30",
            "--db", db_path,
        ])
        assert ret == 1

    def test_main_url_download_failure_returns_error(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        import requests as req

        with patch("ingest_quinnipiac_crosstabs.requests.get") as mock_get:
            mock_get.side_effect = req.RequestException("network error")
            ret = main([
                "--url", "https://poll.qu.edu/images/polling/pa/fake.pdf",
                "--race", "2026 PA Governor",
                "--geography", "PA",
                "--date", "2025-10-01",
                "--db", db_path,
            ])
        assert ret == 1
