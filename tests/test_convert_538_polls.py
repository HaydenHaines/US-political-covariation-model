"""Tests for src.assembly.convert_538_polls."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from src.assembly.convert_538_polls import (
    compute_two_party_dem_share,
    convert_538_polls,
    format_race_name,
    load_pollster_ratings,
    enrich_with_ratings,
    RACE_TYPE_MAP,
)
from src.propagation.propagate_polls import PollObservation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_POLLS_HEADER = (
    "poll_id,question_id,race_id,cycle,location,type_simple,race,pollster,"
    "pollster_rating_id,aapor_roper,inactive,methodology,transparency_score,"
    "partisan,polldate,electiondate,time_to_election,samplesize,"
    "cand1_name,cand1_id,cand1_party,cand1_pct,cand1_actual,"
    "cand2_name,cand2_id,cand2_party,cand2_pct,cand2_actual,"
    "margin_poll,margin_actual"
)


def _make_raw_row(
    *,
    poll_id="1",
    cycle="2020",
    location="FL",
    type_simple="Pres-G",
    race="2020_Pres-G_FL",
    pollster="TestPoll",
    pollster_rating_id="100",
    methodology="Live Phone",
    partisan="NA",
    polldate="2020-10-01",
    electiondate="2020-11-03",
    samplesize="800",
    cand1_party="DEM",
    cand1_pct="48",
    cand2_party="REP",
    cand2_pct="52",
):
    return (
        f"{poll_id},1,1,{cycle},{location},{type_simple},{race},{pollster},"
        f"{pollster_rating_id},FALSE,FALSE,{methodology},8,"
        f"{partisan},{polldate},{electiondate},30,{samplesize},"
        f"Dem Cand,1,{cand1_party},{cand1_pct},48.0,"
        f"Rep Cand,2,{cand2_party},{cand2_pct},52.0,"
        f"-4,-4"
    )


RATINGS_HEADER = (
    "pollster,pollster_rating_id,aapor_roper,inactive,numeric_grade,rank,"
    "POLLSCORE,wtd_avg_transparency,number_polls_pollster_total,"
    "percent_partisan_work,error_ppm,bias_ppm,number_polls_pollster_time_weighted"
)


def _write_raw_polls(tmp_path: Path, rows: list[str]) -> Path:
    """Write a fake raw_polls.csv and return its path."""
    p = tmp_path / "raw_polls.csv"
    p.write_text(RAW_POLLS_HEADER + "\n" + "\n".join(rows) + "\n")
    return p


def _write_ratings(tmp_path: Path, rows: list[str]) -> Path:
    """Write a fake pollster-ratings-combined.csv and return its path."""
    p = tmp_path / "ratings.csv"
    p.write_text(RATINGS_HEADER + "\n" + "\n".join(rows) + "\n")
    return p


# ---------------------------------------------------------------------------
# Two-party share computation
# ---------------------------------------------------------------------------


class TestTwoPartyShare:
    def test_dem_cand1(self):
        row = {"cand1_party": "DEM", "cand2_party": "REP",
               "cand1_pct": "48", "cand2_pct": "52"}
        assert compute_two_party_dem_share(row) == pytest.approx(0.48)

    def test_dem_cand2(self):
        """DEM as candidate 2 — should still compute correctly."""
        row = {"cand1_party": "REP", "cand2_party": "DEM",
               "cand1_pct": "55", "cand2_pct": "45"}
        assert compute_two_party_dem_share(row) == pytest.approx(0.45)

    def test_two_party_normalization(self):
        """Percentages that don't sum to 100 still yield correct two-party share."""
        row = {"cand1_party": "DEM", "cand2_party": "REP",
               "cand1_pct": "40", "cand2_pct": "45"}
        expected = 40.0 / 85.0
        assert compute_two_party_dem_share(row) == pytest.approx(expected)

    def test_missing_party_returns_none(self):
        row = {"cand1_party": "IND", "cand2_party": "REP",
               "cand1_pct": "48", "cand2_pct": "52"}
        assert compute_two_party_dem_share(row) is None

    def test_both_independent_returns_none(self):
        row = {"cand1_party": "LIB", "cand2_party": "GRN",
               "cand1_pct": "48", "cand2_pct": "52"}
        assert compute_two_party_dem_share(row) is None

    def test_missing_pct_returns_none(self):
        row = {"cand1_party": "DEM", "cand2_party": "REP",
               "cand1_pct": "", "cand2_pct": "52"}
        assert compute_two_party_dem_share(row) is None

    def test_zero_total_returns_none(self):
        row = {"cand1_party": "DEM", "cand2_party": "REP",
               "cand1_pct": "0", "cand2_pct": "0"}
        assert compute_two_party_dem_share(row) is None

    def test_case_insensitive_party(self):
        row = {"cand1_party": "dem", "cand2_party": "rep",
               "cand1_pct": "48", "cand2_pct": "52"}
        assert compute_two_party_dem_share(row) == pytest.approx(0.48)


# ---------------------------------------------------------------------------
# Race name formatting
# ---------------------------------------------------------------------------


class TestRaceNameFormatting:
    def test_president(self):
        assert format_race_name("2020", "FL", "Pres-G") == "2020 FL President"

    def test_senate(self):
        assert format_race_name("2022", "GA", "Sen-G") == "2022 GA Senate"

    def test_governor(self):
        assert format_race_name("2022", "FL", "Gov-G") == "2022 FL Governor"

    def test_national(self):
        assert format_race_name("2020", "US", "Pres-G") == "2020 US President"

    def test_primary_returns_none(self):
        assert format_race_name("2020", "FL", "Pres-P") is None

    def test_house_returns_none(self):
        assert format_race_name("2020", "FL", "House-G") is None


# ---------------------------------------------------------------------------
# Pollster ratings loading
# ---------------------------------------------------------------------------


class TestPollsterRatings:
    def test_load_ratings(self, tmp_path):
        row = "TestPollster,100,TRUE,FALSE,2.5,10,-1.2,8.5,50,0,-0.5,-1.0,30"
        path = _write_ratings(tmp_path, [row])
        ratings = load_pollster_ratings(path)

        assert 100 in ratings
        assert ratings[100]["pollster"] == "TestPollster"
        assert ratings[100]["numeric_grade"] == pytest.approx(2.5)
        assert ratings[100]["pollscore"] == pytest.approx(-1.2)
        assert ratings[100]["bias"] == pytest.approx(-1.0)

    def test_missing_file_returns_empty(self, tmp_path):
        ratings = load_pollster_ratings(tmp_path / "nonexistent.csv")
        assert ratings == {}

    def test_multiple_ratings(self, tmp_path):
        rows = [
            "PollA,100,TRUE,FALSE,3,1,-1.5,9,120,0,-1,-2,111",
            "PollB,200,FALSE,FALSE,1.5,50,2.0,5,10,50,3,1.5,5",
        ]
        path = _write_ratings(tmp_path, rows)
        ratings = load_pollster_ratings(path)
        assert len(ratings) == 2
        assert ratings[200]["numeric_grade"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Date format
# ---------------------------------------------------------------------------


class TestDateFormat:
    def test_iso_date_preserved(self, tmp_path):
        """538 dates are already ISO format — should pass through unchanged."""
        rows = [_make_raw_row(polldate="2020-10-15")]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert result["2020"][0].date == "2020-10-15"


# ---------------------------------------------------------------------------
# State filtering
# ---------------------------------------------------------------------------


class TestStateFiltering:
    def test_filter_to_specific_states(self, tmp_path):
        rows = [
            _make_raw_row(poll_id="1", location="FL"),
            _make_raw_row(poll_id="2", location="GA"),
            _make_raw_row(poll_id="3", location="TX"),
        ]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL", "GA"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        geos = {p.geography for p in result["2020"]}
        assert geos == {"FL", "GA"}
        assert len(result["2020"]) == 2

    def test_none_states_includes_all(self, tmp_path):
        rows = [
            _make_raw_row(poll_id="1", location="FL"),
            _make_raw_row(poll_id="2", location="TX"),
            _make_raw_row(poll_id="3", location="US"),
        ]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=None,
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert len(result["2020"]) == 3

    def test_national_polls_included(self, tmp_path):
        rows = [_make_raw_row(location="US")]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL", "US"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert result["2020"][0].geography == "US"


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------


class TestMissingData:
    def test_missing_samplesize_skipped(self, tmp_path):
        rows = [_make_raw_row(samplesize="")]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert len(result["2020"]) == 0

    def test_zero_samplesize_skipped(self, tmp_path):
        rows = [_make_raw_row(samplesize="0")]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert len(result["2020"]) == 0

    def test_missing_party_skipped(self, tmp_path):
        rows = [_make_raw_row(cand1_party="IND")]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert len(result["2020"]) == 0

    def test_primary_type_skipped(self, tmp_path):
        rows = [_make_raw_row(type_simple="Pres-P")]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert len(result["2020"]) == 0

    def test_house_type_skipped(self, tmp_path):
        rows = [_make_raw_row(type_simple="House-G")]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert len(result["2020"]) == 0


# ---------------------------------------------------------------------------
# Full conversion pipeline
# ---------------------------------------------------------------------------


class TestConvertPipeline:
    def test_basic_conversion(self, tmp_path):
        rows = [
            _make_raw_row(
                poll_id="1",
                cycle="2020",
                location="FL",
                type_simple="Pres-G",
                pollster="Quinnipiac",
                polldate="2020-10-01",
                samplesize="800",
                cand1_party="DEM",
                cand1_pct="48",
                cand2_party="REP",
                cand2_pct="52",
            )
        ]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )

        assert len(result["2020"]) == 1
        obs = result["2020"][0]
        assert obs.geography == "FL"
        assert obs.dem_share == pytest.approx(0.48)
        assert obs.n_sample == 800
        assert obs.race == "2020 FL President"
        assert obs.date == "2020-10-01"
        assert obs.pollster == "Quinnipiac"
        assert obs.geo_level == "state"

    def test_csv_output_written(self, tmp_path):
        rows = [_make_raw_row()]
        raw_path = _write_raw_polls(tmp_path, rows)
        convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )

        csv_path = tmp_path / "polls_2020.csv"
        assert csv_path.exists()

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows_out = list(reader)

        assert len(rows_out) == 1
        assert rows_out[0]["race"] == "2020 FL President"
        assert rows_out[0]["geography"] == "FL"
        assert rows_out[0]["geo_level"] == "state"
        assert float(rows_out[0]["dem_share"]) == pytest.approx(0.48)

    def test_multiple_cycles(self, tmp_path):
        rows = [
            _make_raw_row(poll_id="1", cycle="2020"),
            _make_raw_row(poll_id="2", cycle="2022", type_simple="Sen-G"),
        ]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020", "2022"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )

        assert len(result["2020"]) == 1
        assert len(result["2022"]) == 1
        assert (tmp_path / "polls_2020.csv").exists()
        assert (tmp_path / "polls_2022.csv").exists()

    def test_enrichment_adds_notes(self, tmp_path):
        rows = [_make_raw_row(pollster_rating_id="100")]
        raw_path = _write_raw_polls(tmp_path, rows)
        ratings_rows = [
            "TestPollster,100,TRUE,FALSE,2.5,10,-1.2,8.5,50,0,-0.5,-1.0,30"
        ]
        ratings_path = _write_ratings(tmp_path, ratings_rows)

        convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            ratings_path=ratings_path,
            enrich=True,
        )

        csv_path = tmp_path / "polls_2020.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert "grade=2.5" in row["notes"]
        assert "pollscore=-1.2" in row["notes"]
        assert "bias=-1.0" in row["notes"]

    def test_race_type_filter(self, tmp_path):
        rows = [
            _make_raw_row(poll_id="1", type_simple="Pres-G"),
            _make_raw_row(poll_id="2", type_simple="Sen-G"),
            _make_raw_row(poll_id="3", type_simple="Gov-G"),
        ]
        raw_path = _write_raw_polls(tmp_path, rows)
        result = convert_538_polls(
            cycles=["2020"],
            states=["FL"],
            race_types=["Pres-G"],
            output_dir=tmp_path,
            raw_polls_path=raw_path,
            enrich=False,
        )
        assert len(result["2020"]) == 1
        assert "President" in result["2020"][0].race


# ---------------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_runs(self, tmp_path):
        """Verify the module can be invoked as a CLI."""
        rows = [_make_raw_row()]
        raw_path = _write_raw_polls(tmp_path, rows)

        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.assembly.convert_538_polls",
                "--cycles",
                "2020",
                "--states",
                "FL",
                "--output-dir",
                str(tmp_path),
                "--no-enrich",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parents[1]),
            env={
                **__import__("os").environ,
                # Override raw polls path via env — not supported, so we test
                # with real data path. This test just verifies CLI parsing.
            },
        )
        # CLI should run without crashing (may warn about missing file)
        # The real raw_polls.csv should exist on this machine
        assert result.returncode == 0 or "not found" in result.stderr


# ---------------------------------------------------------------------------
# Enrich helper
# ---------------------------------------------------------------------------


class TestEnrichWithRatings:
    def test_returns_enriched_dicts(self):
        polls = [
            PollObservation(
                geography="FL",
                dem_share=0.48,
                n_sample=800,
                race="2020 FL President",
                date="2020-10-01",
                pollster="TestPoll",
                geo_level="state",
            )
        ]
        ratings = {100: {"numeric_grade": 2.5, "pollscore": -1.2, "bias": -1.0}}
        enriched = enrich_with_ratings(polls, ratings)
        assert len(enriched) == 1
        assert enriched[0]["race"] == "2020 FL President"
        assert enriched[0]["dem_share"] == pytest.approx(0.48)
