"""Tests for fetch_algara_amlani.py — Algara & Amlani county governor returns.

Dataset schema (from doi:10.7910/DVN/DGUMFI):
  - election_year: float (e.g. 2006.0)
  - fips: str, 5-char zero-padded (e.g. '12001')
  - state: str abbreviation (e.g. 'FL')
  - sfips: str (numeric string, e.g. '12')
  - office: str, always 'GOV' in the governor file
  - election_type: str, 'G' for general
  - democratic_raw_votes: float
  - republican_raw_votes: float
  - gov_raw_county_vote_totals_two_party: float
  - county_name, dem_nominee, rep_nominee, seat_status, etc.

Tests use synthetic DataFrames with ACTUAL column names from the dataset.
No network access required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_algara_amlani import (
    GOV_YEARS,
    STATES,
    aggregate_county_year,
    filter_governor_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    state: str,
    fips: str,
    election_year: float,
    office: str = "GOV",
    election_type: str = "G",
    dem_votes: float = 10000.0,
    rep_votes: float = 15000.0,
    two_party_total: float = 25000.0,
) -> dict:
    return {
        "election_id": f"{int(election_year)} {state}",
        "election_year": election_year,
        "fips": fips,
        "county_name": "TEST COUNTY",
        "state": state,
        "sfips": fips[:2],
        "office": office,
        "election_type": election_type,
        "seat_status": "Republican Open Seat",
        "democratic_raw_votes": dem_votes,
        "dem_nominee": "Dem Candidate",
        "republican_raw_votes": rep_votes,
        "rep_nominee": "Rep Candidate",
        "gov_raw_county_vote_totals_two_party": two_party_total,
        "raw_county_vote_totals": two_party_total,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: Non-governor races are excluded
# ---------------------------------------------------------------------------


def test_filter_governor_only():
    """Rows with office != 'GOV' must be dropped."""
    rows = [
        _make_row("FL", "12001", 2006.0, office="GOV"),
        _make_row("FL", "12003", 2006.0, office="SEN"),  # should be excluded
        _make_row("GA", "13001", 2006.0, office="GOV"),
    ]
    df = _make_df(rows)
    filtered = filter_governor_rows(df)
    assert len(filtered) == 2
    assert set(filtered["office"].unique()) == {"GOV"}
    assert "12003" not in filtered["fips"].values


# ---------------------------------------------------------------------------
# Test 2: Only FL (12), GA (13), AL (01) are kept
# ---------------------------------------------------------------------------


def test_filter_target_states_only():
    """Only FL, GA, AL rows should survive after filtering."""
    rows = [
        _make_row("FL", "12001", 2006.0),
        _make_row("GA", "13001", 2006.0),
        _make_row("AL", "01001", 2006.0),
        _make_row("TX", "48001", 2006.0),  # should be excluded
        _make_row("CA", "06001", 2006.0),  # should be excluded
    ]
    df = _make_df(rows)
    filtered = filter_governor_rows(df)
    assert len(filtered) == 3
    assert set(filtered["state"].unique()) == {"FL", "GA", "AL"}
    assert "48001" not in filtered["fips"].values
    assert "06001" not in filtered["fips"].values


# ---------------------------------------------------------------------------
# Test 3: dem_share = dem_votes / (dem_votes + rep_votes)
# ---------------------------------------------------------------------------


def test_aggregate_dem_share():
    """dem_share must equal democratic_raw_votes / two_party_total."""
    rows = [
        _make_row("FL", "12001", 2006.0, dem_votes=30000.0, rep_votes=70000.0, two_party_total=100000.0),
        _make_row("FL", "12003", 2006.0, dem_votes=50000.0, rep_votes=50000.0, two_party_total=100000.0),
        _make_row("GA", "13001", 2006.0, dem_votes=20000.0, rep_votes=80000.0, two_party_total=100000.0),
    ]
    df = _make_df(rows)
    out = aggregate_county_year(df, 2006)

    fl1 = out[out["county_fips"] == "12001"].iloc[0]
    assert abs(fl1["gov_dem_share_2006"] - 0.30) < 1e-9

    fl3 = out[out["county_fips"] == "12003"].iloc[0]
    assert abs(fl3["gov_dem_share_2006"] - 0.50) < 1e-9

    ga1 = out[out["county_fips"] == "13001"].iloc[0]
    assert abs(ga1["gov_dem_share_2006"] - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# Test 4: Output has exactly the required columns
# ---------------------------------------------------------------------------


def test_aggregate_columns():
    """Output must have exactly county_fips, state_abbr, and the four year-suffixed cols."""
    year = 2006
    rows = [
        _make_row("FL", "12001", float(year)),
        _make_row("GA", "13001", float(year)),
    ]
    df = _make_df(rows)
    out = aggregate_county_year(df, year)

    expected_cols = {
        "county_fips",
        "state_abbr",
        f"gov_dem_{year}",
        f"gov_rep_{year}",
        f"gov_total_{year}",
        f"gov_dem_share_{year}",
    }
    assert set(out.columns) == expected_cols, (
        f"Expected {sorted(expected_cols)}, got {sorted(out.columns)}"
    )


# ---------------------------------------------------------------------------
# Test 5: GOV_YEARS contains [2002, 2006, 2010, 2014, 2018]
# ---------------------------------------------------------------------------


def test_gov_years_coverage():
    """GOV_YEARS must include the four midterm governor cycles we need."""
    required = [2002, 2006, 2010, 2014, 2018]
    for y in required:
        assert y in GOV_YEARS, f"GOV_YEARS missing year {y}"
