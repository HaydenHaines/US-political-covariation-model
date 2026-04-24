"""Tests for scripts/scrape_rcp_primaries.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scrape_rcp_primaries import (  # noqa: E402
    OUTPUT_COLUMNS,
    PRIMARY_RACE_CONFIG,
    build_primary_output_df,
    deduplicate_primaries,
    primary_dedup_key,
    scrape_rcp_primary,
)

# ---------------------------------------------------------------------------
# Fixture: an RCP primary page with multiple candidates per poll.
# ---------------------------------------------------------------------------
_PRIMARY_POLLS_JSON = [
    {
        "id": "9001",
        "type": "rcp_average",
        "pollster": "rcp_average",
        "date": "3/15 - 3/30",
        "data_end_date": "2026/03/30",
        "sampleSize": "",
        "candidate": [
            {"name": "Carter", "value": "32"},
            {"name": "Collins", "value": "28"},
            {"name": "Dooley", "value": "14"},
        ],
    },
    {
        "id": "1001",
        "type": "poll_rcp_avg",
        "pollster": "Emerson",
        "pollster_group_name": "Emerson College",
        "date": "3/28 - 3/30",
        "data_end_date": "2026/03/30",
        "sampleSize": "600 LV",
        "candidate": [
            {"name": "Carter", "value": "31"},
            {"name": "Collins", "value": "29"},
            {"name": "Dooley", "value": "13"},
            {"name": "Undecided", "value": "27"},
        ],
    },
    {
        "id": "1002",
        "type": "poll_rcp_avg",
        "pollster": "Cygnal",
        "pollster_group_name": "Cygnal",
        "date": "3/10 - 3/12",
        "data_end_date": "2026/03/12",
        "sampleSize": "500 RV",
        "candidate": [
            {"name": "Carter", "value": "35"},
            {"name": "Collins", "value": "25"},
            {"name": "Dooley", "value": "12"},
        ],
    },
]


def _build_rcp_fixture_html(polls_data) -> str:
    inner_obj = f'{{"polls":{json.dumps(polls_data)}}}'
    encoded = json.dumps(inner_obj)[1:-1]
    return f'<html><body>\n<script>self.__next_f.push([1,"{encoded}"]);</script>\n</body></html>'


PRIMARY_FIXTURE_HTML = _build_rcp_fixture_html(_PRIMARY_POLLS_JSON)


# ---------------------------------------------------------------------------
# Scrape behaviour
# ---------------------------------------------------------------------------
class TestScrapeRCPPrimary:
    @patch("scrape_rcp_primaries.fetch_html")
    def test_skips_rcp_average_row(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary(
            "2026 GA Senate Republican Primary",
            "GA",
            "R",
            "/polls/senate/republican-primary/2026/georgia/carter-vs-collins-vs-dooley",
        )
        assert len(polls) == 2

    @patch("scrape_rcp_primaries.fetch_html")
    def test_race_key_propagates(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary(
            "2026 GA Senate Republican Primary",
            "GA",
            "R",
            "/fake-url",
        )
        assert all(p["race_key"] == "2026 GA Senate Republican Primary" for p in polls)

    @patch("scrape_rcp_primaries.fetch_html")
    def test_is_primary_flag_set(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        assert all(p["is_primary"] is True for p in polls)

    @patch("scrape_rcp_primaries.fetch_html")
    def test_geography_and_party_fields(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        assert all(p["geography"] == "GA" for p in polls)
        assert all(p["party"] == "R" for p in polls)
        assert all(p["geo_level"] == "state" for p in polls)

    @patch("scrape_rcp_primaries.fetch_html")
    def test_candidates_preserved_and_sorted(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        # Emerson has 4 candidates (including Undecided); they must be sorted by pct desc.
        emerson = next(p for p in polls if p["pollster"] == "Emerson College")
        pcts = [c["pct"] for c in emerson["candidates"]]
        assert pcts == sorted(pcts, reverse=True)
        assert emerson["candidates"][0]["name"] == "Carter"
        assert emerson["candidates"][0]["pct"] == 31.0

    @patch("scrape_rcp_primaries.fetch_html")
    def test_n_sample_and_type_extracted(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        emerson = next(p for p in polls if p["pollster"] == "Emerson College")
        cygnal = next(p for p in polls if p["pollster"] == "Cygnal")
        assert emerson["n_sample"] == 600
        assert emerson["sample_type"] == "LV"
        assert cygnal["n_sample"] == 500
        assert cygnal["sample_type"] == "RV"

    @patch("scrape_rcp_primaries.fetch_html")
    def test_date_parsing(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        dates = {p["pollster"]: p["date"] for p in polls}
        assert dates["Emerson College"] == "2026-03-30"
        assert dates["Cygnal"] == "2026-03-12"

    @patch("scrape_rcp_primaries.fetch_html")
    def test_pollster_normalized(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        names = {p["pollster"] for p in polls}
        assert names == {"Emerson College", "Cygnal"}

    @patch("scrape_rcp_primaries.fetch_html")
    def test_network_error_returns_empty(self, mock_fetch):
        mock_fetch.return_value = None
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        assert polls == []

    @patch("scrape_rcp_primaries.fetch_html")
    def test_empty_json_returns_empty(self, mock_fetch):
        mock_fetch.return_value = "<html><body>no data</body></html>"
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        assert polls == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class TestDeduplicatePrimaries:
    def _make(self, pollster, date, race_key, candidates):
        return {
            "race_key": race_key,
            "geography": "GA",
            "geo_level": "state",
            "party": "R",
            "pollster": pollster,
            "pollster_raw": pollster,
            "date": date,
            "n_sample": 800,
            "candidates": candidates,
            "source": "rcp",
            "sample_type": "LV",
            "is_primary": True,
        }

    def test_no_duplicates(self):
        polls = [
            self._make(
                "Emerson College",
                "2026-03-01",
                "2026 GA Senate R Primary",
                [{"name": "A", "pct": 40}, {"name": "B", "pct": 30}],
            ),
            self._make(
                "Cygnal", "2026-03-05", "2026 GA Senate R Primary", [{"name": "A", "pct": 42}, {"name": "B", "pct": 28}]
            ),
        ]
        assert len(deduplicate_primaries(polls)) == 2

    def test_prefers_richer_candidate_list(self):
        """Same pollster/date/race but one URL has more candidates — keep the richer one."""
        two_way = self._make(
            "Emerson College",
            "2026-03-01",
            "2026 GA Senate R Primary",
            [{"name": "A", "pct": 50}, {"name": "B", "pct": 40}],
        )
        three_way = self._make(
            "Emerson College",
            "2026-03-01",
            "2026 GA Senate R Primary",
            [{"name": "A", "pct": 45}, {"name": "B", "pct": 35}, {"name": "C", "pct": 10}],
        )
        result = deduplicate_primaries([two_way, three_way])
        assert len(result) == 1
        assert len(result[0]["candidates"]) == 3

    def test_dedup_key(self):
        p = self._make("Emerson College", "2026-03-01", "2026 GA Senate R Primary", [{"name": "A", "pct": 50}])
        assert primary_dedup_key(p) == ("2026 GA Senate R Primary", "2026-03-01", "emerson college")

    def test_different_races_kept(self):
        polls = [
            self._make("Emerson College", "2026-03-01", "2026 GA Senate R Primary", [{"name": "A", "pct": 40}]),
            self._make("Emerson College", "2026-03-01", "2026 TX Senate R Primary", [{"name": "X", "pct": 45}]),
        ]
        assert len(deduplicate_primaries(polls)) == 2


# ---------------------------------------------------------------------------
# Output DataFrame schema
# ---------------------------------------------------------------------------
class TestBuildPrimaryOutputDf:
    def _make(self, race_key="2026 GA Senate Republican Primary"):
        return {
            "race_key": race_key,
            "geography": "GA",
            "geo_level": "state",
            "party": "R",
            "pollster": "Emerson College",
            "pollster_raw": "Emerson College",
            "date": "2026-03-30",
            "n_sample": 600,
            "candidates": [
                {"name": "Carter", "pct": 31.0},
                {"name": "Collins", "pct": 29.0},
                {"name": "Dooley", "pct": 13.0},
            ],
            "source": "rcp",
            "sample_type": "LV",
            "is_primary": True,
        }

    def test_columns(self):
        df = build_primary_output_df([self._make()])
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_candidates_json_roundtrip(self):
        df = build_primary_output_df([self._make()])
        json_str = df.iloc[0]["candidates_json"]
        parsed = json.loads(json_str)
        assert len(parsed) == 3
        assert parsed[0]["name"] == "Carter"
        assert parsed[0]["pct"] == 31.0

    def test_is_primary_true(self):
        df = build_primary_output_df([self._make()])
        # pandas/numpy boxes bool into np.True_; compare by value not identity.
        assert bool(df.iloc[0]["is_primary"]) is True

    def test_notes_include_leader(self):
        df = build_primary_output_df([self._make()])
        notes = df.iloc[0]["notes"]
        assert "lead=Carter@31.0%" in notes
        assert "LV" in notes
        assert "src=rcp" in notes

    def test_empty_input_returns_empty_df_with_columns(self):
        df = build_primary_output_df([])
        assert list(df.columns) == OUTPUT_COLUMNS
        assert len(df) == 0

    def test_sorted_by_race_and_date(self):
        polls = [
            self._make(race_key="2026 TX Senate Republican Primary"),
            self._make(race_key="2026 GA Senate Republican Primary"),
        ]
        polls[0]["date"] = "2026-03-15"
        polls[1]["date"] = "2026-03-30"
        df = build_primary_output_df(polls)
        assert df.iloc[0]["race_key"] == "2026 GA Senate Republican Primary"
        assert df.iloc[1]["race_key"] == "2026 TX Senate Republican Primary"


# ---------------------------------------------------------------------------
# Race config integrity
# ---------------------------------------------------------------------------
class TestPrimaryRaceConfig:
    def test_all_races_have_required_keys(self):
        for race_key, cfg in PRIMARY_RACE_CONFIG.items():
            assert "state" in cfg, f"{race_key} missing state"
            assert "party" in cfg, f"{race_key} missing party"
            assert "rcp_urls" in cfg, f"{race_key} missing rcp_urls"
            assert isinstance(cfg["rcp_urls"], list), f"{race_key} rcp_urls not a list"
            assert len(cfg["rcp_urls"]) >= 1, f"{race_key} has no URLs"

    def test_party_values_valid(self):
        for race_key, cfg in PRIMARY_RACE_CONFIG.items():
            assert cfg["party"] in {"R", "D"}, f"{race_key} has invalid party {cfg['party']}"

    def test_state_codes_uppercase_two_letter(self):
        for race_key, cfg in PRIMARY_RACE_CONFIG.items():
            assert len(cfg["state"]) == 2 and cfg["state"].isupper(), f"{race_key} has malformed state {cfg['state']!r}"

    def test_race_keys_indicate_primary(self):
        """Race keys must include 'Primary' for downstream identification."""
        for race_key in PRIMARY_RACE_CONFIG:
            assert "Primary" in race_key, f"{race_key} missing 'Primary' marker"

    def test_race_keys_indicate_party(self):
        """Race keys must include the party name so humans can scan them."""
        for race_key, cfg in PRIMARY_RACE_CONFIG.items():
            if cfg["party"] == "R":
                assert "Republican" in race_key, f"{race_key} should say 'Republican'"
            else:
                assert "Democratic" in race_key, f"{race_key} should say 'Democratic'"

    def test_urls_start_with_slash(self):
        for race_key, cfg in PRIMARY_RACE_CONFIG.items():
            for url in cfg["rcp_urls"]:
                assert url.startswith("/polls/"), f"{race_key} URL {url!r} must be a site-relative path"
                assert "primary" in url.lower(), f"{race_key} URL {url!r} must target a primary page"


# ---------------------------------------------------------------------------
# End-to-end: scrape + dedupe + build_df
# ---------------------------------------------------------------------------
class TestEndToEnd:
    @patch("scrape_rcp_primaries.fetch_html")
    def test_full_pipeline(self, mock_fetch):
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary(
            "2026 GA Senate Republican Primary",
            "GA",
            "R",
            "/polls/senate/republican-primary/2026/georgia/carter-vs-collins-vs-dooley",
        )
        deduped = deduplicate_primaries(polls)
        df = build_primary_output_df(deduped)
        assert len(df) == 2
        assert set(df.columns) == set(OUTPUT_COLUMNS)
        assert (df["is_primary"] == True).all()  # noqa: E712
        assert (df["party"] == "R").all()
        assert (df["geography"] == "GA").all()

    @patch("scrape_rcp_primaries.fetch_html")
    def test_merge_semantics_keep_last(self, mock_fetch, tmp_path):
        """Simulate merge-with-existing: last row wins on duplicate key."""
        mock_fetch.return_value = PRIMARY_FIXTURE_HTML
        polls = scrape_rcp_primary("2026 GA Senate Republican Primary", "GA", "R", "/u")
        df_new = build_primary_output_df(deduplicate_primaries(polls))

        # Pretend there's an older row with the same (race_key, date, pollster)
        # but stale candidate data.
        stale = df_new.iloc[[0]].copy()
        stale.iloc[0, stale.columns.get_loc("candidates_json")] = "[]"
        existing = stale

        merged = pd.concat([existing, df_new], ignore_index=True).drop_duplicates(
            subset=["race_key", "date", "pollster"], keep="last"
        )
        # Fresh scrape wins, so candidates_json must not be "[]"
        assert len(merged) == len(df_new)
        assert (merged["candidates_json"] != "[]").all()
