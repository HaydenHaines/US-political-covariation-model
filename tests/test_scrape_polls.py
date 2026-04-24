"""Tests for scripts/scrape_2026_polls.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Import from the scraper module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from scrape_2026_polls import (
    RACE_CONFIG,
    _GB_RACE_LABEL,
    build_output_df,
    dedup_key,
    deduplicate,
    extract_pct,
    extract_sample_size,
    merge_with_existing,
    normalize_pollster,
    parse_poll_date,
    scrape_270towin,
    scrape_generic_ballot_rcp,
    scrape_rcp,
    scrape_wikipedia,
    two_party_share,
)

import pandas as pd


# ============================================================================
# Pollster name normalization
# ============================================================================
class TestNormalizePollster:
    def test_exact_match(self):
        assert normalize_pollster("Emerson College") == "Emerson College"

    def test_case_insensitive(self):
        assert normalize_pollster("emerson college") == "Emerson College"
        assert normalize_pollster("EMERSON COLLEGE") == "Emerson College"

    def test_variant_mapping(self):
        assert normalize_pollster("Emerson College Polling") == "Emerson College"
        assert normalize_pollster("Mason Dixon") == "Mason-Dixon"
        assert normalize_pollster("Mason-Dixon Polling") == "Mason-Dixon"

    def test_footnote_removal(self):
        assert normalize_pollster("Emerson College[1]") == "Emerson College"
        assert normalize_pollster("Quinnipiac[a]") == "Quinnipiac University"

    def test_unknown_pollster_passthrough(self):
        assert normalize_pollster("Acme Polling Inc.") == "Acme Polling Inc."

    def test_empty_and_none(self):
        assert normalize_pollster("") == ""
        assert normalize_pollster(None) == ""

    def test_whitespace_handling(self):
        assert normalize_pollster("  Emerson College  ") == "Emerson College"

    def test_multiple_aliases(self):
        assert normalize_pollster("Fox News") == "FOX News"
        assert normalize_pollster("FOX News Poll") == "FOX News"
        assert normalize_pollster("Trafalgar Group") == "Trafalgar Group"
        assert normalize_pollster("The Trafalgar Group") == "Trafalgar Group"

    def test_partisan_tag_aliases(self):
        assert normalize_pollster("Cygnal (R)") == "Cygnal"
        assert normalize_pollster("Quantus Insights (R)") == "Quantus Insights"
        assert normalize_pollster("Tyson Group (R)") == "Tyson Group"
        assert (
            normalize_pollster("Bendixen & Amandi International (D)")
            == "Bendixen & Amandi International"
        )


# ============================================================================
# Two-party share conversion
# ============================================================================
class TestTwoPartyShare:
    def test_basic_conversion(self):
        assert two_party_share(45.0, 55.0) == 0.45

    def test_even_split(self):
        assert two_party_share(50.0, 50.0) == 0.5

    def test_with_undecided(self):
        # D=40, R=45 (15% undecided) -> 40/85 = 0.4706
        result = two_party_share(40.0, 45.0)
        assert result is not None
        assert abs(result - 0.4706) < 0.001

    def test_sanity_lower_bound(self):
        assert two_party_share(10.0, 90.0) is None

    def test_sanity_upper_bound(self):
        assert two_party_share(90.0, 10.0) is None

    def test_zero_values(self):
        assert two_party_share(0.0, 50.0) is None
        assert two_party_share(50.0, 0.0) is None

    def test_negative_values(self):
        assert two_party_share(-5.0, 50.0) is None

    def test_rounding(self):
        result = two_party_share(43.0, 52.0)
        assert result is not None
        assert result == round(43.0 / 95.0, 4)

    def test_boundary_values(self):
        # Just inside range
        assert two_party_share(16.0, 84.0) is not None  # 0.16
        assert two_party_share(84.0, 16.0) is not None  # 0.84
        # Just outside range
        assert two_party_share(14.0, 86.0) is None  # 0.14


# ============================================================================
# Date parsing
# ============================================================================
class TestParsePollDate:
    def test_iso_format(self):
        assert parse_poll_date("2026-03-04") == "2026-03-04"

    def test_us_format(self):
        assert parse_poll_date("March 4, 2026") == "2026-03-04"

    def test_range_uses_end_date(self):
        result = parse_poll_date("March 1-4, 2026")
        assert result == "2026-03-04"

    def test_cross_month_range(self):
        result = parse_poll_date("February 28 - March 4, 2026")
        assert result == "2026-03-04"

    def test_empty_string(self):
        assert parse_poll_date("") is None

    def test_none(self):
        assert parse_poll_date(None) is None

    def test_footnote_removal(self):
        result = parse_poll_date("March 4, 2026[1]")
        assert result == "2026-03-04"

    def test_slash_format(self):
        result = parse_poll_date("3/09/2026")
        assert result == "2026-03-09"


# ============================================================================
# Extract percentage
# ============================================================================
class TestExtractPct:
    def test_plain_number(self):
        assert extract_pct("45") == 45.0

    def test_with_percent_sign(self):
        assert extract_pct("45%") == 45.0

    def test_with_footnote(self):
        assert extract_pct("45[1]") == 45.0

    def test_float(self):
        assert extract_pct("45.3%") == 45.3

    def test_nan(self):
        assert extract_pct(float("nan")) is None

    def test_non_numeric(self):
        assert extract_pct("N/A") is None

    def test_zero(self):
        assert extract_pct("0") is None

    def test_hundred(self):
        assert extract_pct("100") is None


# ============================================================================
# Extract sample size
# ============================================================================
class TestExtractSampleSize:
    def test_plain_number(self):
        assert extract_sample_size("800") == 800

    def test_with_comma(self):
        assert extract_sample_size("1,200") == 1200

    def test_with_type_label(self):
        assert extract_sample_size("800 LV") == 800

    def test_with_parenthetical(self):
        assert extract_sample_size("800 (LV)") == 800

    def test_too_small(self):
        assert extract_sample_size("10") is None

    def test_nan(self):
        assert extract_sample_size(float("nan")) is None

    def test_with_moe(self):
        # 270toWin format: "560 LV +/-4.1%"
        assert extract_sample_size("560 LV") == 560


# ============================================================================
# Deduplication
# ============================================================================
class TestDeduplication:
    def _make_poll(self, pollster, date, race, source, dem_pct=45.0, rep_pct=55.0):
        return {
            "race": race,
            "pollster": pollster,
            "pollster_raw": pollster,
            "date": date,
            "n_sample": 800,
            "dem_pct": dem_pct,
            "rep_pct": rep_pct,
            "dem_share": dem_pct / (dem_pct + rep_pct),
            "source": source,
            "sample_type": "",
        }

    def test_no_duplicates(self):
        polls = [
            self._make_poll("Emerson College", "2026-03-01", "2026 FL Governor", "wikipedia"),
            self._make_poll("Mason-Dixon", "2026-03-05", "2026 FL Governor", "270towin"),
        ]
        result = deduplicate(polls)
        assert len(result) == 2

    def test_duplicate_prefers_270towin_over_wiki(self):
        polls = [
            self._make_poll(
                "Emerson College", "2026-03-01", "2026 FL Governor", "wikipedia", 44.0, 56.0
            ),
            self._make_poll(
                "Emerson College", "2026-03-01", "2026 FL Governor", "270towin", 45.0, 55.0
            ),
        ]
        result = deduplicate(polls)
        assert len(result) == 1
        assert result[0]["source"] == "270towin"
        assert result[0]["dem_pct"] == 45.0

    def test_duplicate_prefers_rcp_over_wiki(self):
        polls = [
            self._make_poll(
                "Emerson College", "2026-03-01", "2026 FL Governor", "wikipedia", 44.0, 56.0
            ),
            self._make_poll(
                "Emerson College", "2026-03-01", "2026 FL Governor", "rcp", 46.0, 54.0
            ),
        ]
        result = deduplicate(polls)
        assert len(result) == 1
        assert result[0]["source"] == "rcp"
        assert result[0]["dem_pct"] == 46.0

    def test_duplicate_prefers_270towin_over_rcp(self):
        polls = [
            self._make_poll(
                "Emerson College", "2026-03-01", "2026 FL Governor", "rcp", 46.0, 54.0
            ),
            self._make_poll(
                "Emerson College", "2026-03-01", "2026 FL Governor", "270towin", 45.0, 55.0
            ),
        ]
        result = deduplicate(polls)
        assert len(result) == 1
        assert result[0]["source"] == "270towin"
        assert result[0]["dem_pct"] == 45.0

    def test_different_races_not_deduped(self):
        polls = [
            self._make_poll("Emerson College", "2026-03-01", "2026 FL Governor", "wikipedia"),
            self._make_poll("Emerson College", "2026-03-01", "2026 FL Senate", "wikipedia"),
        ]
        result = deduplicate(polls)
        assert len(result) == 2

    def test_different_dates_not_deduped(self):
        polls = [
            self._make_poll("Emerson College", "2026-03-01", "2026 FL Governor", "wikipedia"),
            self._make_poll("Emerson College", "2026-03-15", "2026 FL Governor", "wikipedia"),
        ]
        result = deduplicate(polls)
        assert len(result) == 2

    def test_dedup_key_case_insensitive(self):
        p = self._make_poll("Emerson College", "2026-03-01", "2026 FL Governor", "wikipedia")
        key = dedup_key(p)
        assert key[0] == "emerson college"


# ============================================================================
# Output DataFrame schema
# ============================================================================
class TestBuildOutputDf:
    def test_correct_columns(self):
        polls = [
            {
                "race": "2026 FL Governor",
                "pollster": "Emerson College",
                "pollster_raw": "Emerson College",
                "date": "2026-03-01",
                "n_sample": 800,
                "dem_pct": 45.0,
                "rep_pct": 55.0,
                "dem_share": 0.45,
                "source": "wikipedia",
                "sample_type": "",
            }
        ]
        df = build_output_df(polls)
        expected_cols = [
            "race", "geography", "geo_level", "dem_share", "n_sample", "date",
            "pollster", "notes",
        ]
        assert list(df.columns) == expected_cols

    def test_geography_mapping(self):
        polls = [
            {
                "race": "2026 GA Senate",
                "pollster": "Cygnal",
                "pollster_raw": "Cygnal",
                "date": "2026-02-01",
                "n_sample": 700,
                "dem_pct": 48.0,
                "rep_pct": 52.0,
                "dem_share": 0.48,
                "source": "270towin",
                "sample_type": "LV",
            }
        ]
        df = build_output_df(polls)
        assert df.iloc[0]["geography"] == "GA"
        assert df.iloc[0]["geo_level"] == "state"

    def test_notes_contain_raw_pcts(self):
        polls = [
            {
                "race": "2026 FL Governor",
                "pollster": "Mason-Dixon",
                "pollster_raw": "Mason-Dixon",
                "date": "2026-01-15",
                "n_sample": 900,
                "dem_pct": 43.0,
                "rep_pct": 52.0,
                "dem_share": 0.4526,
                "source": "wikipedia",
                "sample_type": "",
            }
        ]
        df = build_output_df(polls)
        notes = df.iloc[0]["notes"]
        assert "D=43.0%" in notes
        assert "R=52.0%" in notes
        assert "src=wikipedia" in notes

    def test_notes_contain_sample_type(self):
        polls = [
            {
                "race": "2026 FL Governor",
                "pollster": "X",
                "pollster_raw": "X",
                "date": "2026-01-15",
                "n_sample": 900,
                "dem_pct": 43.0,
                "rep_pct": 52.0,
                "dem_share": 0.4526,
                "source": "270towin",
                "sample_type": "LV",
            }
        ]
        df = build_output_df(polls)
        assert "LV" in df.iloc[0]["notes"]

    def test_sorted_by_race_and_date(self):
        polls = [
            {
                "race": "2026 GA Senate",
                "pollster": "B",
                "pollster_raw": "B",
                "date": "2026-03-01",
                "n_sample": 800,
                "dem_pct": 48.0,
                "rep_pct": 52.0,
                "dem_share": 0.48,
                "source": "wikipedia",
                "sample_type": "",
            },
            {
                "race": "2026 FL Governor",
                "pollster": "A",
                "pollster_raw": "A",
                "date": "2026-02-01",
                "n_sample": 700,
                "dem_pct": 45.0,
                "rep_pct": 55.0,
                "dem_share": 0.45,
                "source": "wikipedia",
                "sample_type": "",
            },
        ]
        df = build_output_df(polls)
        assert df.iloc[0]["race"] == "2026 FL Governor"
        assert df.iloc[1]["race"] == "2026 GA Senate"

    def test_empty_input(self):
        df = build_output_df([])
        assert list(df.columns) == [
            "race", "geography", "geo_level", "dem_share", "n_sample",
            "date", "pollster", "notes",
        ]
        assert len(df) == 0


# ============================================================================
# Wikipedia parsing with mock HTML (general election table)
# ============================================================================
WIKI_FIXTURE_HTML = """
<html><body>
<h2>Opinion polling</h2>
<h3>General election</h3>
<table class="wikitable">
<tr><th>Poll source</th><th>Date(s)</th><th>Sample size</th>
<th>Jane Smith (D)</th><th>Bob Jones (R)</th><th>Margin</th></tr>
<tr><td>Emerson College[1]</td><td>March 1-4, 2026</td><td>800 (LV)</td>
<td>45%</td><td>52%</td><td>R+7</td></tr>
<tr><td>Quinnipiac University</td><td>February 20, 2026</td><td>920 (RV)</td>
<td>43%</td><td>50%</td><td>R+7</td></tr>
<tr><td>Mason-Dixon</td><td>January 15, 2026</td><td>750</td>
<td>44%</td><td>53%</td><td>R+9</td></tr>
</table>
<h3>Republican primary</h3>
<table class="wikitable">
<tr><th>Poll source</th><th>Date(s)</th><th>Sample size</th>
<th>Bob Jones</th><th>Tom White</th><th>Other</th></tr>
<tr><td>Emerson College</td><td>March 1, 2026</td><td>500</td>
<td>35%</td><td>28%</td><td>37%</td></tr>
</table>
<table class="wikitable">
<tr><th>Other data</th><th>Value</th></tr>
<tr><td>Foo</td><td>Bar</td></tr>
</table>
</body></html>
"""

# Candidate lists for the fixture
FIXTURE_DEM_CANDIDATES = ["smith", "jane smith"]
FIXTURE_REP_CANDIDATES = ["jones", "bob jones"]


class TestWikipediaParsing:
    @patch("scrape_2026_polls.fetch_html")
    def test_parse_wiki_general_election_only(self, mock_fetch):
        mock_fetch.return_value = WIKI_FIXTURE_HTML
        polls = scrape_wikipedia(
            "2026 FL Governor", "http://fake-wiki-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        # Should only get 3 general election polls, not the primary
        assert len(polls) == 3
        assert polls[0]["pollster"] == "Emerson College"
        assert polls[0]["dem_pct"] == 45.0
        assert polls[0]["rep_pct"] == 52.0
        assert polls[0]["source"] == "wikipedia"

    @patch("scrape_2026_polls.fetch_html")
    def test_wiki_sample_size_extraction(self, mock_fetch):
        mock_fetch.return_value = WIKI_FIXTURE_HTML
        polls = scrape_wikipedia(
            "2026 FL Governor", "http://fake-wiki-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["n_sample"] == 800
        assert polls[1]["n_sample"] == 920

    @patch("scrape_2026_polls.fetch_html")
    def test_wiki_date_parsing(self, mock_fetch):
        mock_fetch.return_value = WIKI_FIXTURE_HTML
        polls = scrape_wikipedia(
            "2026 FL Governor", "http://fake-wiki-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["date"] == "2026-03-04"
        assert polls[1]["date"] == "2026-02-20"

    @patch("scrape_2026_polls.fetch_html")
    def test_wiki_network_error(self, mock_fetch):
        mock_fetch.return_value = None
        polls = scrape_wikipedia(
            "2026 FL Governor", "http://fake-wiki-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls == []

    @patch("scrape_2026_polls.fetch_html")
    def test_wiki_no_tables(self, mock_fetch):
        mock_fetch.return_value = "<html><body><p>No tables here</p></body></html>"
        polls = scrape_wikipedia(
            "2026 FL Governor", "http://fake-wiki-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls == []


# ============================================================================
# 270toWin parsing with mock HTML
# ============================================================================
TTW_FIXTURE_HTML = """
<html><body>
<table>
<tr><th>Source</th><th>Date</th><th>Sample</th><th>MoE</th>
<th>Jane Smith (D)</th><th>Bob Jones (R)</th><th>Spread</th></tr>
<tr><td>Emerson College</td><td>3/04/2026</td><td>800 LV</td><td>3.5%</td>
<td>45%</td><td>52%</td><td>Jones +7</td></tr>
<tr><td>Cygnal</td><td>2/10/2026</td><td>600 RV</td><td>4.0%</td>
<td>42%</td><td>51%</td><td>Jones +9</td></tr>
</table>
<table>
<tr><th>Source</th><th>Date</th><th>Sample</th>
<th>Bob Jones</th><th>Tom White</th><th>Other</th></tr>
<tr><td>SomePolls</td><td>1/15/2026</td><td>500</td>
<td>35%</td><td>28%</td><td>37%</td></tr>
</table>
</body></html>
"""


class TestTTWParsing:
    @patch("scrape_2026_polls.fetch_html")
    def test_parse_ttw_general_election_only(self, mock_fetch):
        mock_fetch.return_value = TTW_FIXTURE_HTML
        polls = scrape_270towin(
            "2026 GA Senate", "http://fake-ttw-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        # Should get 2 general election polls, skip the primary table
        assert len(polls) == 2
        assert polls[0]["pollster"] == "Emerson College"
        assert polls[0]["source"] == "270towin"

    @patch("scrape_2026_polls.fetch_html")
    def test_ttw_network_error(self, mock_fetch):
        mock_fetch.return_value = None
        polls = scrape_270towin(
            "2026 GA Senate", "http://fake-ttw-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls == []

    @patch("scrape_2026_polls.fetch_html")
    def test_ttw_cygnal_normalized(self, mock_fetch):
        mock_fetch.return_value = TTW_FIXTURE_HTML
        polls = scrape_270towin(
            "2026 GA Senate", "http://fake-ttw-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[1]["pollster"] == "Cygnal"

    @patch("scrape_2026_polls.fetch_html")
    def test_ttw_sample_type_extracted(self, mock_fetch):
        mock_fetch.return_value = TTW_FIXTURE_HTML
        polls = scrape_270towin(
            "2026 GA Senate", "http://fake-ttw-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["sample_type"] == "LV"
        assert polls[1]["sample_type"] == "RV"

    @patch("scrape_2026_polls.fetch_html")
    def test_ttw_date_parsing_slash_format(self, mock_fetch):
        mock_fetch.return_value = TTW_FIXTURE_HTML
        polls = scrape_270towin(
            "2026 GA Senate", "http://fake-ttw-url",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["date"] == "2026-03-04"
        assert polls[1]["date"] == "2026-02-10"


# ============================================================================
# RCP parsing with mock HTML (Next.js embedded JSON format)
# ============================================================================

# RCP serves data via Next.js flight protocol: embedded JSON inside
# self.__next_f.push() script calls.  Table rows are empty; data lives in
# the "polls" array within the decoded JSON payload.
_RCP_POLLS_JSON = [
    {
        "id": "9999",
        "type": "rcp_average",
        "pollster": "rcp_average",
        "date": "3/20 - 4/1",
        "data_end_date": "2026/04/01",
        "sampleSize": "",
        "candidate": [
            {"name": "Smith", "affiliation": "Democrat", "value": "46"},
            {"name": "Jones", "affiliation": "Republican", "value": "50"},
        ],
    },
    {
        "id": "111",
        "type": "poll_rcp_avg",
        "pollster": "Emerson",
        "pollster_group_name": "Emerson College",
        "date": "3/31 - 4/1",
        "data_end_date": "2026/04/01",
        "sampleSize": "987 LV",
        "candidate": [
            {"name": "Smith", "affiliation": "Democrat", "value": "45"},
            {"name": "Jones", "affiliation": "Republican", "value": "52"},
        ],
    },
    {
        "id": "222",
        "type": "poll_rcp_avg",
        "pollster": "Cygnal",
        "pollster_group_name": "Cygnal",
        "date": "3/20 - 3/22",
        "data_end_date": "2026/03/22",
        "sampleSize": "556 RV",
        "candidate": [
            {"name": "Smith", "affiliation": "Democrat", "value": "47"},
            {"name": "Jones", "affiliation": "Republican", "value": "48"},
        ],
    },
]


def _build_rcp_fixture_html(polls_data) -> str:
    """Build a minimal RCP-format HTML fixture with embedded Next.js JSON."""
    # Build the encoded payload the same way RCP embeds it
    inner_obj = f'{{"polls":{json.dumps(polls_data)}}}'
    # Escape as a JSON string value (double-encode)
    encoded = json.dumps(inner_obj)[1:-1]  # strip outer quotes
    return f"""<html><body>
<script>self.__next_f.push([1,"{encoded}"]);</script>
</body></html>"""


RCP_FIXTURE_HTML = _build_rcp_fixture_html(_RCP_POLLS_JSON)


class TestRCPParsing:
    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_filters_average_row(self, mock_fetch):
        mock_fetch.return_value = RCP_FIXTURE_HTML
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        # Should get 2 polls — RCP Average row must be excluded
        assert len(polls) == 2

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_source_label(self, mock_fetch):
        mock_fetch.return_value = RCP_FIXTURE_HTML
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert all(p["source"] == "rcp" for p in polls)

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_pollster_normalized(self, mock_fetch):
        mock_fetch.return_value = RCP_FIXTURE_HTML
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["pollster"] == "Emerson College"
        assert polls[1]["pollster"] == "Cygnal"

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_date_from_data_end_date(self, mock_fetch):
        mock_fetch.return_value = RCP_FIXTURE_HTML
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        # data_end_date "2026/04/01" → "2026-04-01"
        assert polls[0]["date"] == "2026-04-01"
        # data_end_date "2026/03/22" → "2026-03-22"
        assert polls[1]["date"] == "2026-03-22"

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_sample_type(self, mock_fetch):
        mock_fetch.return_value = RCP_FIXTURE_HTML
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["sample_type"] == "LV"
        assert polls[1]["sample_type"] == "RV"

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_sample_size(self, mock_fetch):
        mock_fetch.return_value = RCP_FIXTURE_HTML
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["n_sample"] == 987
        assert polls[1]["n_sample"] == 556

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_network_error(self, mock_fetch):
        mock_fetch.return_value = None
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls == []

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_percentages(self, mock_fetch):
        mock_fetch.return_value = RCP_FIXTURE_HTML
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls[0]["dem_pct"] == 45.0
        assert polls[0]["rep_pct"] == 52.0

    @patch("scrape_2026_polls.fetch_html")
    def test_parse_rcp_no_json_returns_empty(self, mock_fetch):
        """Pages with no embedded poll JSON should return empty list gracefully."""
        mock_fetch.return_value = "<html><body><p>No data</p></body></html>"
        polls = scrape_rcp(
            "2026 GA Senate", "/polls/senate/general/2026/georgia/ossoff-vs-carter",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        assert polls == []


# ============================================================================
# Integration: end-to-end with mocked HTTP
# ============================================================================
class TestEndToEnd:
    @patch("scrape_2026_polls.fetch_html")
    def test_full_pipeline_with_dedup(self, mock_fetch):
        """Both sources return the same Emerson poll; should deduplicate to 270toWin."""
        mock_fetch.side_effect = [WIKI_FIXTURE_HTML, TTW_FIXTURE_HTML]

        wiki_polls = scrape_wikipedia(
            "2026 FL Governor", "http://wiki",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )
        ttw_polls = scrape_270towin(
            "2026 FL Governor", "http://ttw",
            FIXTURE_DEM_CANDIDATES, FIXTURE_REP_CANDIDATES,
        )

        all_polls = wiki_polls + ttw_polls
        deduped = deduplicate(all_polls)

        # Emerson on same date (2026-03-04) should be deduped
        emerson_polls = [p for p in deduped if p["pollster"] == "Emerson College"]
        assert len(emerson_polls) == 1
        assert emerson_polls[0]["source"] == "270towin"

        df = build_output_df(deduped)
        assert "race" in df.columns
        assert "geography" in df.columns
        assert "dem_share" in df.columns
        assert all(0.15 < s < 0.85 for s in df["dem_share"])


# ============================================================================
# Race config integrity
# ============================================================================
class TestRaceConfig:
    def test_all_races_have_required_keys(self):
        for label, cfg in RACE_CONFIG.items():
            assert "state" in cfg, f"{label} missing state"
            assert "wiki_url" in cfg, f"{label} missing wiki_url"
            assert "ttw_url" in cfg, f"{label} missing ttw_url"
            assert "rcp_urls" in cfg, f"{label} missing rcp_urls"
            assert "dem_candidates" in cfg, f"{label} missing dem_candidates"
            assert "rep_candidates" in cfg, f"{label} missing rep_candidates"
            assert len(cfg["dem_candidates"]) > 0, f"{label} has empty dem_candidates"
            assert len(cfg["rep_candidates"]) > 0, f"{label} has empty rep_candidates"
            # rcp_urls must be a list (may be empty for races RCP does not track)
            assert isinstance(cfg["rcp_urls"], list), f"{label} rcp_urls must be a list"

    def test_races_configured(self):
        # Original 18 + 9 new RCP-only races
        assert len(RACE_CONFIG) == 27

    def test_original_18_races_present(self):
        """All original 18 tracked races must still be in RACE_CONFIG."""
        original_18 = {
            # Original 6
            "2026 FL Governor", "2026 FL Senate",
            "2026 GA Governor", "2026 GA Senate",
            "2026 AL Governor", "2026 AL Senate",
            # 12 national competitive races added in S213
            "2026 IA Senate", "2026 ME Senate",
            "2026 MI Senate", "2026 MN Senate",
            "2026 NC Senate", "2026 NH Senate", "2026 OR Senate",
            "2026 MI Governor", "2026 OH Governor",
            "2026 PA Governor", "2026 TX Governor", "2026 WI Governor",
        }
        assert original_18.issubset(set(RACE_CONFIG.keys()))

    def test_new_rcp_races_present(self):
        """New races sourced from RCP must be in RACE_CONFIG."""
        new_races = {
            "2026 TX Senate", "2026 MA Senate", "2026 OH Senate",
            "2026 NY Governor", "2026 NV Governor", "2026 AZ Governor",
            "2026 MN Governor", "2026 MA Governor", "2026 NH Governor",
        }
        assert new_races.issubset(set(RACE_CONFIG.keys()))

    def test_states_are_valid(self):
        valid_states = {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
            "DC",
        }
        for label, cfg in RACE_CONFIG.items():
            assert cfg["state"] in valid_states, f"{label} has invalid state {cfg['state']}"


# ============================================================================
# Generic ballot RCP scraping
# ============================================================================

# Generic ballot RCP fixture — uses "Democrats"/"Republicans" as candidate
# names with Democrat/Republican affiliations, matching the real RCP GB page.
_GB_POLLS_JSON = [
    {
        "id": "8888",
        "type": "rcp_average",
        "pollster": "rcp_average",
        "date": "3/20 - 4/1",
        "data_end_date": "2026/04/01",
        "sampleSize": "",
        "candidate": [
            {"name": "Democrats", "affiliation": "Democrat", "value": "46"},
            {"name": "Republicans", "affiliation": "Republican", "value": "50"},
        ],
    },
    {
        "id": "333",
        "type": "poll_rcp_avg",
        "pollster": "Emerson",
        "pollster_group_name": "Emerson College",
        "date": "3/31 - 4/1",
        "data_end_date": "2026/04/01",
        "sampleSize": "1200 LV",
        "candidate": [
            {"name": "Democrats", "affiliation": "Democrat", "value": "47"},
            {"name": "Republicans", "affiliation": "Republican", "value": "48"},
        ],
    },
    {
        "id": "444",
        "type": "poll_rcp_avg",
        "pollster": "Quinnipiac",
        "pollster_group_name": "Quinnipiac University",
        "date": "3/20 - 3/25",
        "data_end_date": "2026/03/25",
        "sampleSize": "800 RV",
        "candidate": [
            {"name": "Democrats", "affiliation": "Democrat", "value": "45"},
            {"name": "Republicans", "affiliation": "Republican", "value": "51"},
        ],
    },
]

GB_RCP_FIXTURE_HTML = _build_rcp_fixture_html(_GB_POLLS_JSON)


class TestGenericBallotRCPScraping:
    @patch("scrape_2026_polls.fetch_html")
    def test_returns_polls_not_average(self, mock_fetch):
        """Should return 2 polls and exclude the RCP Average row."""
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        assert len(polls) == 2

    @patch("scrape_2026_polls.fetch_html")
    def test_race_label(self, mock_fetch):
        """All returned polls must have race == '2026 Generic Ballot'."""
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        assert all(p["race"] == _GB_RACE_LABEL for p in polls)

    @patch("scrape_2026_polls.fetch_html")
    def test_source_label(self, mock_fetch):
        """All returned polls must have source == 'rcp'."""
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        assert all(p["source"] == "rcp" for p in polls)

    @patch("scrape_2026_polls.fetch_html")
    def test_pollster_normalized(self, mock_fetch):
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        pollsters = {p["pollster"] for p in polls}
        assert "Emerson College" in pollsters
        assert "Quinnipiac University" in pollsters

    @patch("scrape_2026_polls.fetch_html")
    def test_dem_rep_percentages(self, mock_fetch):
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        # First poll (Emerson): D=47, R=48
        emerson = next(p for p in polls if p["pollster"] == "Emerson College")
        assert emerson["dem_pct"] == 47.0
        assert emerson["rep_pct"] == 48.0

    @patch("scrape_2026_polls.fetch_html")
    def test_sample_type(self, mock_fetch):
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        emerson = next(p for p in polls if p["pollster"] == "Emerson College")
        quinnipiac = next(p for p in polls if p["pollster"] == "Quinnipiac University")
        assert emerson["sample_type"] == "LV"
        assert quinnipiac["sample_type"] == "RV"

    @patch("scrape_2026_polls.fetch_html")
    def test_sample_size(self, mock_fetch):
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        emerson = next(p for p in polls if p["pollster"] == "Emerson College")
        assert emerson["n_sample"] == 1200

    @patch("scrape_2026_polls.fetch_html")
    def test_date_parsed(self, mock_fetch):
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        emerson = next(p for p in polls if p["pollster"] == "Emerson College")
        assert emerson["date"] == "2026-04-01"

    @patch("scrape_2026_polls.fetch_html")
    def test_dem_share_in_range(self, mock_fetch):
        mock_fetch.return_value = GB_RCP_FIXTURE_HTML
        polls = scrape_generic_ballot_rcp()
        for p in polls:
            assert 0.15 < p["dem_share"] < 0.85

    @patch("scrape_2026_polls.fetch_html")
    def test_network_error_returns_empty(self, mock_fetch):
        mock_fetch.return_value = None
        polls = scrape_generic_ballot_rcp()
        assert polls == []

    @patch("scrape_2026_polls.fetch_html")
    def test_no_json_returns_empty(self, mock_fetch):
        mock_fetch.return_value = "<html><body>no data</body></html>"
        polls = scrape_generic_ballot_rcp()
        assert polls == []


class TestBuildOutputDfGenericBallot:
    """Tests for build_output_df behaviour with generic ballot polls."""

    def _make_gb_poll(self, pollster="Emerson College", date="2026-04-01"):
        return {
            "race": _GB_RACE_LABEL,
            "pollster": pollster,
            "pollster_raw": pollster,
            "date": date,
            "n_sample": 1200,
            "dem_pct": 47.0,
            "rep_pct": 48.0,
            "dem_share": round(47.0 / 95.0, 4),
            "source": "rcp",
            "sample_type": "LV",
        }

    def test_gb_polls_geo_level_national(self):
        """Generic ballot polls must have geo_level='national'."""
        df = build_output_df([self._make_gb_poll()])
        assert df.iloc[0]["geo_level"] == "national"

    def test_gb_polls_geography_us(self):
        """Generic ballot polls must have geography='US'."""
        df = build_output_df([self._make_gb_poll()])
        assert df.iloc[0]["geography"] == "US"

    def test_gb_polls_race_label_preserved(self):
        """Race label must survive the round-trip through build_output_df."""
        df = build_output_df([self._make_gb_poll()])
        assert df.iloc[0]["race"] == _GB_RACE_LABEL

    def test_state_polls_unaffected(self):
        """Regular state polls must still get geo_level='state' and a state code."""
        state_poll = {
            "race": "2026 GA Senate",
            "pollster": "Cygnal",
            "pollster_raw": "Cygnal",
            "date": "2026-02-01",
            "n_sample": 700,
            "dem_pct": 48.0,
            "rep_pct": 52.0,
            "dem_share": 0.48,
            "source": "rcp",
            "sample_type": "LV",
        }
        df = build_output_df([state_poll])
        assert df.iloc[0]["geo_level"] == "state"
        assert df.iloc[0]["geography"] == "GA"

    def test_mixed_polls_both_preserved(self):
        """A mix of GB + state polls must each get correct geo metadata."""
        polls = [
            self._make_gb_poll(),
            {
                "race": "2026 GA Senate",
                "pollster": "Cygnal",
                "pollster_raw": "Cygnal",
                "date": "2026-02-01",
                "n_sample": 700,
                "dem_pct": 48.0,
                "rep_pct": 52.0,
                "dem_share": 0.48,
                "source": "rcp",
                "sample_type": "LV",
            },
        ]
        df = build_output_df(polls)
        gb_row = df[df["race"] == _GB_RACE_LABEL].iloc[0]
        state_row = df[df["race"] == "2026 GA Senate"].iloc[0]
        assert gb_row["geo_level"] == "national"
        assert gb_row["geography"] == "US"
        assert state_row["geo_level"] == "state"
        assert state_row["geography"] == "GA"


# ============================================================================
# merge_with_existing — preserve xt_* / methodology enrichment on re-scrape
# ============================================================================
class TestMergeWithExisting:
    _SCRAPER_COLS = ["race", "geography", "geo_level", "dem_share",
                     "n_sample", "date", "pollster", "notes"]

    def _new_row(self, **overrides):
        base = {
            "race": "2026 AZ Governor",
            "geography": "AZ",
            "geo_level": "state",
            "dem_share": 0.5057,
            "n_sample": 850.0,
            "date": "2025-11-10",
            "pollster": "Emerson College",
            "notes": "D=44% R=43%",
        }
        base.update(overrides)
        return base

    def test_rescrape_preserves_xt_enrichment(self):
        """The regression: xt_* values on an existing row must survive a
        re-scrape that produces the same race+date+pollster but no xt_*
        columns. Previously ``drop_duplicates(keep="last")`` wiped them."""
        existing = pd.DataFrame([
            {**self._new_row(),
             "methodology": "mixed",
             "xt_education_college": 0.355,
             "xt_race_white": 0.712,
             "xt_vote_race_white": 0.416},
        ])
        new = pd.DataFrame([self._new_row()])

        merged = merge_with_existing(new, existing)

        assert len(merged) == 1
        row = merged.iloc[0]
        assert row["xt_education_college"] == 0.355
        assert row["xt_race_white"] == 0.712
        assert row["xt_vote_race_white"] == 0.416
        assert row["methodology"] == "mixed"

    def test_rescrape_prefers_fresh_scraper_cols(self):
        """Scraper-produced columns (dem_share, n_sample, notes) on the
        fresh row win over the stale existing values."""
        existing = pd.DataFrame([
            {**self._new_row(dem_share=0.47, n_sample=800, notes="stale"),
             "xt_education_college": 0.355},
        ])
        new = pd.DataFrame([
            self._new_row(dem_share=0.5057, n_sample=850, notes="fresh"),
        ])

        merged = merge_with_existing(new, existing)

        assert len(merged) == 1
        row = merged.iloc[0]
        assert row["dem_share"] == 0.5057
        assert row["n_sample"] == 850
        assert row["notes"] == "fresh"
        # enrichment still preserved
        assert row["xt_education_college"] == 0.355

    def test_new_row_has_no_existing_match(self):
        """A brand-new race+date+pollster triple is appended, not merged."""
        existing = pd.DataFrame([
            {**self._new_row(race="2026 OH Governor", geography="OH"),
             "xt_education_college": 0.40},
        ])
        new = pd.DataFrame([self._new_row()])

        merged = merge_with_existing(new, existing)

        assert len(merged) == 2
        az = merged[merged["race"] == "2026 AZ Governor"].iloc[0]
        oh = merged[merged["race"] == "2026 OH Governor"].iloc[0]
        # New row gets NaN for enrichment columns (nothing to carry)
        assert pd.isna(az["xt_education_college"])
        # Existing row keeps its enrichment
        assert oh["xt_education_college"] == 0.40

    def test_existing_row_with_no_fresh_match_is_preserved(self):
        """Polls in the existing CSV that weren't re-scraped (e.g. because
        RCP returned 403) must survive the merge unchanged."""
        existing = pd.DataFrame([
            {**self._new_row(race="2026 OH Governor", geography="OH",
                             dem_share=0.45),
             "xt_education_college": 0.40},
        ])
        new = pd.DataFrame([self._new_row()])  # different race entirely

        merged = merge_with_existing(new, existing)

        oh_rows = merged[merged["race"] == "2026 OH Governor"]
        assert len(oh_rows) == 1
        assert oh_rows.iloc[0]["dem_share"] == 0.45
        assert oh_rows.iloc[0]["xt_education_college"] == 0.40

    def test_no_enrichment_cols_behaves_like_plain_dedup(self):
        """When the existing CSV has no extra columns, merge collapses to
        the original concat+drop_duplicates semantics."""
        existing = pd.DataFrame([
            self._new_row(dem_share=0.47, notes="old"),
        ])
        new = pd.DataFrame([self._new_row(dem_share=0.5057, notes="new")])

        merged = merge_with_existing(new, existing)

        assert len(merged) == 1
        assert merged.iloc[0]["dem_share"] == 0.5057
        assert merged.iloc[0]["notes"] == "new"

    def test_duplicate_existing_rows_deduped_by_last(self):
        """If the existing CSV has duplicates on the key (shouldn't happen
        but could from hand-edits), the last-seen enrichment is used."""
        existing = pd.DataFrame([
            {**self._new_row(notes="first"), "xt_education_college": 0.30},
            {**self._new_row(notes="second"), "xt_education_college": 0.35},
        ])
        new = pd.DataFrame([self._new_row(notes="fresh")])

        merged = merge_with_existing(new, existing)

        assert len(merged) == 1
        assert merged.iloc[0]["xt_education_college"] == 0.35
