"""Tests for tools/ingest_emerson_crosstabs.py.

Coverage:
  1. build_crosstab_records — parses a real Emerson Sheet snapshot (AZ Governor
     data from polls_2026.csv) into poll_crosstabs dicts with correct values.
  2. Poll-linkage logic — verifies that the poll_id computed by the tool matches
     the ID that polling.ingest() would assign to the same row, and that records
     are correctly inserted/upserted into DuckDB.
  3. Tier 2 acceptance — an ingested poll's crosstab records, when passed to
     build_W_poll, return a list (Tier 2) rather than an ndarray (Tier 1/3).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from ingest_emerson_crosstabs import (
    EMERSON_POLLSTER,
    build_crosstab_records,
    ingest_to_db,
    load_emerson_poll_rows,
    _compute_poll_id,
)
from src.db.domains.polling import _make_poll_id, create_tables


# ---------------------------------------------------------------------------
# Real Emerson data fixture — AZ Governor 2025-11-10 (from polls_2026.csv)
# CSV header order:
#   race, geography, geo_level, dem_share, n_sample, date, pollster, notes,
#   methodology, xt_education_college, xt_education_noncollege, xt_race_white,
#   xt_race_black, xt_race_hispanic, xt_race_asian, xt_urbanicity_urban,
#   xt_urbanicity_rural, xt_age_senior, xt_religion_evangelical,
#   xt_vote_race_white, xt_vote_race_black, xt_vote_race_hispanic,
#   xt_vote_race_asian, xt_vote_education_college, xt_vote_education_noncollege,
#   xt_vote_age_senior
# ---------------------------------------------------------------------------

AZ_GOV_ROW: dict[str, str] = {
    "race": "2026 AZ Governor",
    "geography": "AZ",
    "geo_level": "state",
    "dem_share": "0.5057",
    "n_sample": "850.0",
    "date": "2025-11-10",
    "pollster": EMERSON_POLLSTER,
    "notes": "D=44.0% R=43.0%; RV; src=rcp",
    "methodology": "mixed",
    # Composition (pct_of_sample)
    "xt_education_college": "0.355",
    "xt_education_noncollege": "0.645",
    "xt_race_white": "0.712",
    "xt_race_black": "0.026",
    "xt_race_hispanic": "0.187",
    "xt_race_asian": "0.022",
    "xt_urbanicity_urban": "",
    "xt_urbanicity_rural": "",
    "xt_age_senior": "0.39",
    "xt_religion_evangelical": "",
    # Per-group vote shares (dem_share for Tier 2)
    "xt_vote_race_white": "0.416",
    "xt_vote_race_black": "0.572",
    "xt_vote_race_hispanic": "0.474",
    "xt_vote_race_asian": "0.477",
    "xt_vote_education_college": "0.545",
    "xt_vote_education_noncollege": "0.403",
    "xt_vote_age_senior": "0.4385",
}

GA_SENATE_ROW: dict[str, str] = {
    "race": "2026 GA Senate",
    "geography": "GA",
    "geo_level": "state",
    "dem_share": "0.5275",
    "n_sample": "1000.0",
    "date": "2026-03-02",
    "pollster": EMERSON_POLLSTER,
    "notes": "D=48.0% R=43.0%; LV; src=rcp",
    "methodology": "mixed",
    "xt_education_college": "0.395",
    "xt_education_noncollege": "0.605",
    "xt_race_white": "0.602",
    "xt_race_black": "0.286",
    "xt_race_hispanic": "0.081",
    "xt_race_asian": "0.017",
    "xt_urbanicity_urban": "",
    "xt_urbanicity_rural": "",
    "xt_age_senior": "0.355",
    "xt_religion_evangelical": "",
    "xt_vote_race_white": "0.348",
    "xt_vote_race_black": "0.768",
    "xt_vote_race_hispanic": "0.361",
    "xt_vote_race_asian": "0.596",
    "xt_vote_education_college": "0.585",
    "xt_vote_education_noncollege": "0.389",
    "xt_vote_age_senior": "0.455",
}


def _poll_id(row: dict[str, str], cycle: str = "2026") -> str:
    return _compute_poll_id(row, cycle)


def _make_mem_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


# ---------------------------------------------------------------------------
# 1. build_crosstab_records — parsing a real Emerson Sheet snapshot
# ---------------------------------------------------------------------------

class TestBuildCrosstabRecordsAZGov:
    """Parses the real AZ Governor 2025-11-10 Emerson poll data."""

    @pytest.fixture(autouse=True)
    def _records(self):
        self.poll_id = _poll_id(AZ_GOV_ROW)
        self.records = build_crosstab_records(AZ_GOV_ROW, self.poll_id)

    def test_produces_records(self):
        assert len(self.records) > 0

    def test_all_poll_ids_match(self):
        for rec in self.records:
            assert rec["poll_id"] == self.poll_id

    def test_race_white_pct_of_sample(self):
        white = next(r for r in self.records
                     if r["demographic_group"] == "race" and r["group_value"] == "white")
        assert white["pct_of_sample"] == pytest.approx(0.712, abs=1e-3)

    def test_race_white_dem_share(self):
        white = next(r for r in self.records
                     if r["demographic_group"] == "race" and r["group_value"] == "white")
        assert white["dem_share"] == pytest.approx(0.416, abs=1e-3)

    def test_race_black_present(self):
        black = next((r for r in self.records
                      if r["demographic_group"] == "race" and r["group_value"] == "black"), None)
        assert black is not None
        assert black["pct_of_sample"] == pytest.approx(0.026, abs=1e-3)
        assert black["dem_share"] == pytest.approx(0.572, abs=1e-3)

    def test_education_college_dem_share(self):
        college = next(r for r in self.records
                       if r["demographic_group"] == "education" and r["group_value"] == "college")
        assert college["dem_share"] == pytest.approx(0.545, abs=1e-3)

    def test_age_senior_present(self):
        senior = next((r for r in self.records
                       if r["demographic_group"] == "age" and r["group_value"] == "senior"), None)
        assert senior is not None
        assert senior["pct_of_sample"] == pytest.approx(0.39, abs=1e-3)
        assert senior["dem_share"] == pytest.approx(0.4385, abs=1e-3)

    def test_n_sample_computed(self):
        # n_sample = int(850 * pct_of_sample) for records where total_n is known
        white = next(r for r in self.records
                     if r["demographic_group"] == "race" and r["group_value"] == "white")
        assert white["n_sample"] == int(850 * 0.712)

    def test_empty_xt_columns_not_included(self):
        # xt_urbanicity_urban and xt_religion_evangelical are empty in AZ row
        urbanicity = [r for r in self.records if r["demographic_group"] == "urbanicity"]
        evangelical = [r for r in self.records if r["group_value"] == "evangelical"]
        assert len(urbanicity) == 0
        assert len(evangelical) == 0

    def test_all_records_have_required_keys(self):
        required = {"poll_id", "demographic_group", "group_value",
                    "pct_of_sample", "dem_share", "n_sample"}
        for rec in self.records:
            assert required.issubset(rec.keys()), (
                f"Record missing keys: {required - rec.keys()}"
            )

    def test_pct_of_sample_values_in_range(self):
        for rec in self.records:
            assert 0.0 < rec["pct_of_sample"] <= 1.0, (
                f"pct_of_sample out of range: {rec}"
            )

    def test_dem_share_values_in_range(self):
        for rec in self.records:
            ds = rec["dem_share"]
            if ds is not None:
                assert 0.0 <= ds <= 1.0, f"dem_share out of range: {rec}"


class TestBuildCrosstabRecordsEdgeCases:
    def test_row_without_xt_produces_no_records(self):
        row = {"race": "test", "geography": "GA", "dem_share": "0.50",
               "n_sample": "600", "date": "2026-01-01", "pollster": EMERSON_POLLSTER}
        records = build_crosstab_records(row, "testid")
        assert records == []

    def test_missing_vote_column_falls_back_to_topline(self):
        """When xt_vote_* is absent, dem_share should be the poll's topline dem_share."""
        row = {
            "race": "test", "geography": "FL", "dem_share": "0.48",
            "n_sample": "1000", "date": "2026-01-01", "pollster": EMERSON_POLLSTER,
            "xt_race_white": "0.70",
            # xt_vote_race_white deliberately absent
        }
        records = build_crosstab_records(row, "testid")
        assert len(records) == 1
        # Must fall back to topline dem_share, not None (which would skip the row)
        assert records[0]["dem_share"] == pytest.approx(0.48, abs=1e-3)

    def test_zero_pct_xt_value_skipped(self):
        row = {
            "race": "test", "geography": "AZ", "dem_share": "0.52",
            "n_sample": "700", "date": "2026-01-01", "pollster": EMERSON_POLLSTER,
            "xt_race_white": "0.0",    # zero — should be skipped
            "xt_race_black": "0.13",   # valid
        }
        records = build_crosstab_records(row, "testid")
        groups = {r["group_value"] for r in records}
        assert "white" not in groups
        assert "black" in groups

    def test_missing_n_sample_produces_none_n_sample(self):
        row = {
            "race": "test", "geography": "GA", "dem_share": "0.50",
            "date": "2026-01-01", "pollster": EMERSON_POLLSTER,
            "xt_race_white": "0.65",
            # n_sample absent
        }
        records = build_crosstab_records(row, "testid")
        assert len(records) == 1
        assert records[0]["n_sample"] is None


# ---------------------------------------------------------------------------
# 2. Poll-linkage logic
# ---------------------------------------------------------------------------

class TestPollLinkage:
    """Verifies the poll_id computation and DuckDB insertion/idempotency."""

    def test_poll_id_matches_domain_compute(self):
        """poll_id from _compute_poll_id must equal what polling.ingest() computes."""
        row = AZ_GOV_ROW
        cycle = "2026"
        tool_id = _compute_poll_id(row, cycle)
        domain_id = _make_poll_id(
            race=row["race"],
            geography=row["geography"],
            date=row["date"],
            pollster=row["pollster"],
            cycle=cycle,
        )
        assert tool_id == domain_id

    def test_ingest_creates_records_in_db(self):
        """Records from build_crosstab_records should appear in poll_crosstabs."""
        con = _make_mem_db()
        poll_id = _poll_id(AZ_GOV_ROW)
        records = build_crosstab_records(AZ_GOV_ROW, poll_id)
        n = ingest_to_db(con, records)
        assert n == len(records)
        stored = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
        assert stored == len(records)

    def test_ingest_persists_dem_share(self):
        """dem_share must be stored (not NULL) for rows with xt_vote_* data."""
        con = _make_mem_db()
        poll_id = _poll_id(AZ_GOV_ROW)
        records = build_crosstab_records(AZ_GOV_ROW, poll_id)
        ingest_to_db(con, records)
        row = con.execute(
            "SELECT dem_share FROM poll_crosstabs "
            "WHERE demographic_group='race' AND group_value='white'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.416, abs=1e-3)

    def test_ingest_is_idempotent(self):
        """Running ingest twice for the same poll should not duplicate rows."""
        con = _make_mem_db()
        poll_id = _poll_id(AZ_GOV_ROW)
        records = build_crosstab_records(AZ_GOV_ROW, poll_id)
        ingest_to_db(con, records)
        ingest_to_db(con, records)  # second run
        count = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
        assert count == len(records)

    def test_ingest_overwrites_null_dem_share_rows(self):
        """Existing rows with dem_share=NULL (from polling.ingest) are replaced."""
        con = _make_mem_db()
        create_tables(con)
        poll_id = _poll_id(AZ_GOV_ROW)
        # Simulate what polling.ingest() produces: pct_of_sample but dem_share=NULL.
        null_rows = pd.DataFrame([{
            "poll_id": poll_id,
            "demographic_group": "race",
            "group_value": "white",
            "dem_share": None,
            "n_sample": None,
            "pct_of_sample": 0.712,
        }])
        con.register("_tmp_seed", null_rows)
        con.execute("INSERT INTO poll_crosstabs SELECT * FROM _tmp_seed")
        con.unregister("_tmp_seed")

        # Now run the tool — should replace the NULL row with a real dem_share.
        records = build_crosstab_records(AZ_GOV_ROW, poll_id)
        ingest_to_db(con, records)

        row = con.execute(
            "SELECT dem_share FROM poll_crosstabs "
            "WHERE demographic_group='race' AND group_value='white'"
        ).fetchone()
        assert row[0] == pytest.approx(0.416, abs=1e-3)

    def test_dry_run_does_not_write(self):
        """dry_run=True must not insert any records."""
        con = _make_mem_db()
        poll_id = _poll_id(AZ_GOV_ROW)
        records = build_crosstab_records(AZ_GOV_ROW, poll_id)
        ingest_to_db(con, records, dry_run=True)
        # Table is created by ingest_to_db but no rows are inserted.
        count = con.execute(
            "SELECT COUNT(*) FROM poll_crosstabs"
        ).fetchone()[0]
        assert count == 0

    def test_dry_run_returns_correct_count(self):
        con = _make_mem_db()
        poll_id = _poll_id(AZ_GOV_ROW)
        records = build_crosstab_records(AZ_GOV_ROW, poll_id)
        n = ingest_to_db(con, records, dry_run=True)
        assert n == len(records)

    def test_multiple_polls_ingested_separately(self):
        """Two distinct polls must produce records under distinct poll_ids."""
        con = _make_mem_db()
        az_id = _poll_id(AZ_GOV_ROW)
        ga_id = _poll_id(GA_SENATE_ROW)
        assert az_id != ga_id

        az_records = build_crosstab_records(AZ_GOV_ROW, az_id)
        ga_records = build_crosstab_records(GA_SENATE_ROW, ga_id)
        ingest_to_db(con, az_records + ga_records)

        count = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
        assert count == len(az_records) + len(ga_records)

    def test_load_emerson_poll_rows_from_csv(self, tmp_path: Path):
        """load_emerson_poll_rows filters to Emerson rows with xt_* data."""
        polls_dir = tmp_path / "data" / "polls"
        polls_dir.mkdir(parents=True)
        csv_path = polls_dir / "polls_2026.csv"
        fieldnames = list(AZ_GOV_ROW.keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            # Emerson row with xt_* data — should be included
            writer.writerow(AZ_GOV_ROW)
            # Non-Emerson row — should be excluded
            writer.writerow({**AZ_GOV_ROW, "pollster": "Siena"})
            # Emerson row with no xt_* data — should be excluded
            blank = {k: ("" if k.startswith("xt_") else v)
                     for k, v in AZ_GOV_ROW.items()}
            writer.writerow(blank)

        rows = load_emerson_poll_rows(csv_path)
        assert len(rows) == 1
        assert rows[0]["geography"] == "AZ"

    def test_load_emerson_poll_rows_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_emerson_poll_rows(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# 3. Tier 2 acceptance criterion
# ---------------------------------------------------------------------------

class TestTier2AcceptanceCriteria:
    """Ingested crosstab records trigger Tier 2 (list return) from build_W_poll."""

    def _make_type_profiles(self, J: int = 4) -> pd.DataFrame:
        return pd.DataFrame({
            "pct_bachelors_plus": np.linspace(0.15, 0.65, J),
            "pct_white_nh":       np.linspace(0.90, 0.40, J),
            "pct_black":          np.linspace(0.02, 0.35, J),
            "pct_hispanic":       np.linspace(0.03, 0.30, J),
            "pct_asian":          np.linspace(0.01, 0.12, J),
            "evangelical_share":  np.linspace(0.50, 0.05, J),
        })

    def test_ingested_poll_triggers_tier2(self):
        """build_W_poll with ingested poll_crosstabs returns a list (Tier 2)."""
        from src.prediction.poll_enrichment import build_W_poll

        J = 4
        type_profiles = self._make_type_profiles(J)
        state_type_weights = np.ones(J) / J

        poll = {
            "dem_share": float(AZ_GOV_ROW["dem_share"]),
            "n_sample": int(float(AZ_GOV_ROW["n_sample"])),
            "state": AZ_GOV_ROW["geography"],
        }

        poll_id = _poll_id(AZ_GOV_ROW)
        crosstab_records = build_crosstab_records(AZ_GOV_ROW, poll_id)

        # Convert DB record format to the list[dict] format expected by build_W_poll.
        crosstabs = [
            {
                "demographic_group": r["demographic_group"],
                "group_value": r["group_value"],
                "pct_of_sample": r["pct_of_sample"],
                "dem_share": r["dem_share"],
            }
            for r in crosstab_records
            if r["dem_share"] is not None
        ]

        result = build_W_poll(
            poll=poll,
            type_profiles=type_profiles,
            state_type_weights=state_type_weights,
            poll_crosstabs=crosstabs,
        )

        # Tier 2 returns a list of observation dicts; Tier 1/3 returns an ndarray.
        assert isinstance(result, list), (
            "Expected Tier 2 (list) result when poll_crosstabs is populated"
        )
        assert len(result) >= 1
        for obs in result:
            assert "W" in obs and "y" in obs and "sigma" in obs
            assert obs["W"].shape == (J,)
            assert abs(obs["W"].sum() - 1.0) < 1e-9

    def test_ingested_from_db_triggers_tier2(self):
        """Records queried back from DuckDB feed Tier 2 correctly."""
        from src.prediction.poll_enrichment import build_W_poll

        J = 4
        type_profiles = self._make_type_profiles(J)
        state_type_weights = np.ones(J) / J

        con = _make_mem_db()
        poll_id = _poll_id(AZ_GOV_ROW)
        records = build_crosstab_records(AZ_GOV_ROW, poll_id)
        ingest_to_db(con, records)

        # Query back the records as build_W_poll expects them.
        rows = con.execute(
            "SELECT demographic_group, group_value, pct_of_sample, dem_share "
            "FROM poll_crosstabs WHERE poll_id = ?",
            [poll_id],
        ).fetchall()

        crosstabs = [
            {"demographic_group": r[0], "group_value": r[1],
             "pct_of_sample": r[2], "dem_share": r[3]}
            for r in rows
            if r[3] is not None  # skip rows where dem_share is NULL
        ]

        poll = {
            "dem_share": float(AZ_GOV_ROW["dem_share"]),
            "n_sample": int(float(AZ_GOV_ROW["n_sample"])),
        }

        result = build_W_poll(
            poll=poll,
            type_profiles=type_profiles,
            state_type_weights=state_type_weights,
            poll_crosstabs=crosstabs,
        )

        assert isinstance(result, list), "Expected Tier 2 after DB round-trip"
        assert len(result) >= 1
