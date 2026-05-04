"""Tests for fetch_algara_presidential.py — Algara & Amlani county presidential returns.

Dataset schema (from doi:10.7910/DVN/DGUMFI):
  - election_year: float (e.g. 1976.0)
  - fips: str, 5-char zero-padded (e.g. '12001')
  - state: str abbreviation (e.g. 'FL')
  - office: str, 'PRES' in the presidential file
  - democratic_raw_votes: float
  - republican_raw_votes: float
  - raw_county_vote_totals: float

Tests use synthetic DataFrames. No network access required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_algara_presidential import (
    PRES_YEARS,
    aggregate_county_year,
    filter_presidential_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    state: str,
    fips: str,
    election_year: float,
    dem_votes: float = 10000.0,
    rep_votes: float = 15000.0,
    total_votes: float = 25000.0,
) -> dict:
    return {
        "election_year": election_year,
        "fips": fips,
        "state": state,
        "democratic_raw_votes": dem_votes,
        "republican_raw_votes": rep_votes,
        "raw_county_vote_totals": total_votes,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: filter_presidential_rows selects only the requested year
# ---------------------------------------------------------------------------


def test_filter_selects_correct_year():
    rows = [
        _make_row("FL", "12001", 1976.0),
        _make_row("FL", "12003", 1980.0),
        _make_row("GA", "13001", 1976.0),
    ]
    df = _make_df(rows)
    filtered = filter_presidential_rows(df, 1976)
    assert len(filtered) == 2
    assert set(filtered["state"].unique()) == {"FL", "GA"}
    assert all(filtered["election_year"] == 1976.0)


# ---------------------------------------------------------------------------
# Test 2: Alaska excluded pre-1972
# ---------------------------------------------------------------------------


def test_filter_excludes_alaska_pre_1972():
    rows = [
        _make_row("AK", "02020", 1968.0),  # Alaska pre-1972 — must be dropped
        _make_row("FL", "12001", 1968.0),  # Florida — must be kept
    ]
    df = _make_df(rows)
    filtered = filter_presidential_rows(df, 1968)
    assert len(filtered) == 1
    assert filtered["state"].iloc[0] == "FL"


def test_filter_keeps_alaska_from_1972():
    rows = [
        _make_row("AK", "02020", 1972.0),  # Alaska 1972 — must be kept
        _make_row("FL", "12001", 1972.0),
    ]
    df = _make_df(rows)
    filtered = filter_presidential_rows(df, 1972)
    assert len(filtered) == 2
    assert "AK" in filtered["state"].values


def test_filter_keeps_alaska_post_1972():
    rows = [
        _make_row("AK", "02020", 1976.0),  # Alaska post-1972 — must be kept
        _make_row("FL", "12001", 1976.0),
    ]
    df = _make_df(rows)
    filtered = filter_presidential_rows(df, 1976)
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# Test 3: aggregate_county_year produces correct column names
# ---------------------------------------------------------------------------


def test_aggregate_columns():
    year = 1976
    rows = [
        _make_row("FL", "12001", float(year)),
        _make_row("GA", "13001", float(year)),
    ]
    df = _make_df(rows)
    out = aggregate_county_year(df, year)

    expected_cols = {
        "county_fips",
        "state_abbr",
        f"pres_dem_{year}",
        f"pres_rep_{year}",
        f"pres_total_{year}",
        f"pres_dem_share_{year}",
    }
    assert set(out.columns) == expected_cols, (
        f"Expected {sorted(expected_cols)}, got {sorted(out.columns)}"
    )


# ---------------------------------------------------------------------------
# Test 4: dem_share computation
# ---------------------------------------------------------------------------


def test_aggregate_dem_share():
    year = 1980
    rows = [
        _make_row("FL", "12001", float(year), dem_votes=40000.0, rep_votes=60000.0, total_votes=100000.0),
        _make_row("GA", "13001", float(year), dem_votes=25000.0, rep_votes=75000.0, total_votes=100000.0),
    ]
    df = _make_df(rows)
    out = aggregate_county_year(df, year)

    fl1 = out[out["county_fips"] == "12001"].iloc[0]
    assert abs(fl1[f"pres_dem_share_{year}"] - 0.40) < 1e-9

    ga1 = out[out["county_fips"] == "13001"].iloc[0]
    assert abs(ga1[f"pres_dem_share_{year}"] - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# Test 5: county_fips is zero-padded to 5 digits
# ---------------------------------------------------------------------------


def test_aggregate_fips_zero_padded():
    year = 1960
    rows = [
        _make_row("AL", "1001", float(year)),   # raw 4-digit — should become "01001"
        _make_row("FL", "12001", float(year)),  # already 5-digit
    ]
    df = _make_df(rows)
    out = aggregate_county_year(df, year)
    assert "01001" in out["county_fips"].values
    assert "12001" in out["county_fips"].values


# ---------------------------------------------------------------------------
# Test 6: PRES_YEARS covers the expected range
# ---------------------------------------------------------------------------


def test_pres_years_coverage():
    required = [1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000]
    for y in required:
        assert y in PRES_YEARS, f"PRES_YEARS missing year {y}"


# ---------------------------------------------------------------------------
# Test 7: fallback to two-party sum when raw_county_vote_totals is 0
# ---------------------------------------------------------------------------


def test_aggregate_fallback_to_two_party_sum():
    year = 1964
    rows = [
        _make_row("FL", "12001", float(year), dem_votes=30000.0, rep_votes=70000.0, total_votes=0.0),
    ]
    df = _make_df(rows)
    out = aggregate_county_year(df, year)
    assert out[f"pres_total_{year}"].iloc[0] == 100000.0
    assert abs(out[f"pres_dem_share_{year}"].iloc[0] - 0.30) < 1e-9
