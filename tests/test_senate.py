"""Tests for the MEDSL county Senate fetcher and Senate shift computation.

Uses synthetic DataFrames — no network access required.

Covers:
  - filter_senate_rows: office filter, state filter, party filter
  - aggregate_county_year: dem_share calculation, output columns,
    uncontested drop, FIPS zero-padding, multi-race same year blending
  - _extract_senate_from_2024_precinct: 2024 MEDSL precinct extraction
  - compute_senate_shift: math, sign convention, turnout
  - build_multiyear_shifts: Senate pairs forwarded correctly
  - Config constants: SENATE_YEARS, SENATE_PAIRS, SENATE_FILES
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_county_shifts_multiyear import (
    EPSILON,
    SENATE_PAIRS,
    TRAINING_SHIFT_COLS,
    build_multiyear_shifts,
    compute_senate_shift,
)
from src.assembly.fetch_medsl_county_senate import (
    SENATE_YEARS,
    _extract_senate_from_2024_precinct,
    aggregate_county_year,
    filter_senate_rows,
)
from src.core import config as _cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_senate_row(
    state_po: str,
    county_fips: str,
    year: int,
    office: str = "US SENATE",
    party_simplified: str = "DEMOCRAT",
    candidatevotes: int = 10000,
    totalvotes: int = 25000,
) -> dict:
    return {
        "year": year,
        "state_po": state_po,
        "county_fips": county_fips,
        "office": office,
        "party_simplified": party_simplified,
        "candidatevotes": candidatevotes,
        "totalvotes": totalvotes,
    }


def _raw_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _logit(p: float) -> float:
    p_c = min(max(p, EPSILON), 1 - EPSILON)
    return float(np.log(p_c / (1 - p_c)))


# ---------------------------------------------------------------------------
# filter_senate_rows
# ---------------------------------------------------------------------------


def test_filter_drops_state_senate():
    """'STATE SENATE' rows must be dropped; only US Senate kept."""
    rows = [
        _make_senate_row("FL", "12001", 2018, office="US SENATE"),
        _make_senate_row("FL", "12001", 2018, office="STATE SENATE"),  # drop
        _make_senate_row("GA", "13001", 2018, office="US SENATE"),
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    assert len(filtered) == 2
    assert all("STATE" not in o.upper() for o in filtered["office"])


def test_filter_drops_other_states():
    """All configured-state rows survive (TX and NY are now in scope — national expansion)."""
    rows = [
        _make_senate_row("FL", "12001", 2018),
        _make_senate_row("GA", "13001", 2018),
        _make_senate_row("AL", "01001", 2018),
        _make_senate_row("TX", "48001", 2018),  # TX now in scope
        _make_senate_row("NY", "36001", 2018),  # NY now in scope
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    assert len(filtered) == 5
    assert set(filtered["state_po"]) == {"FL", "GA", "AL", "TX", "NY"}


def test_filter_drops_third_party():
    """Only DEMOCRAT and REPUBLICAN rows should survive."""
    rows = [
        _make_senate_row("FL", "12001", 2018, party_simplified="DEMOCRAT"),
        _make_senate_row("FL", "12001", 2018, party_simplified="REPUBLICAN"),
        _make_senate_row("FL", "12001", 2018, party_simplified="LIBERTARIAN"),  # drop
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    assert set(filtered["party_simplified"]) == {"DEMOCRAT", "REPUBLICAN"}


# ---------------------------------------------------------------------------
# aggregate_county_year — basic dem_share
# ---------------------------------------------------------------------------


def test_aggregate_dem_share():
    """dem_share = dem_votes / totalvotes (all-candidate denominator)."""
    rows = [
        _make_senate_row("FL", "12001", 2018, party_simplified="DEMOCRAT",
                         candidatevotes=30000, totalvotes=100000),
        _make_senate_row("FL", "12001", 2018, party_simplified="REPUBLICAN",
                         candidatevotes=65000, totalvotes=100000),
        _make_senate_row("GA", "13001", 2018, party_simplified="DEMOCRAT",
                         candidatevotes=50000, totalvotes=100000),
        _make_senate_row("GA", "13001", 2018, party_simplified="REPUBLICAN",
                         candidatevotes=50000, totalvotes=100000),
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    out = aggregate_county_year(filtered, 2018)

    fl1 = out[out["county_fips"] == "12001"].iloc[0]
    assert abs(fl1["senate_dem_share_2018"] - 30000 / 100000) < 1e-9

    ga1 = out[out["county_fips"] == "13001"].iloc[0]
    assert abs(ga1["senate_dem_share_2018"] - 50000 / 100000) < 1e-9


def test_aggregate_columns():
    """Output must have exactly the six canonical columns."""
    year = 2018
    rows = [
        _make_senate_row("FL", "12001", year, party_simplified="DEMOCRAT",
                         candidatevotes=30000, totalvotes=95000),
        _make_senate_row("FL", "12001", year, party_simplified="REPUBLICAN",
                         candidatevotes=65000, totalvotes=95000),
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    out = aggregate_county_year(filtered, year)

    expected = {
        "county_fips", "state_abbr",
        f"senate_dem_{year}", f"senate_rep_{year}",
        f"senate_total_{year}", f"senate_dem_share_{year}",
    }
    assert set(out.columns) == expected


def test_aggregate_fips_zero_padded():
    """county_fips must be a 5-character zero-padded string."""
    rows = [
        # Simulate MEDSL float representation (e.g. 1001.0 → "01001")
        _make_senate_row("AL", 1001, 2020, party_simplified="DEMOCRAT",
                         candidatevotes=10000, totalvotes=25000),
        _make_senate_row("AL", 1001, 2020, party_simplified="REPUBLICAN",
                         candidatevotes=15000, totalvotes=25000),
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    out = aggregate_county_year(filtered, 2020)
    assert all(out["county_fips"].str.len() == 5)
    assert "01001" in out["county_fips"].values


def test_aggregate_uncontested_dropped():
    """Counties where either party has zero votes are dropped."""
    rows = [
        # Contested county
        _make_senate_row("FL", "12001", 2022, party_simplified="DEMOCRAT",
                         candidatevotes=40000, totalvotes=100000),
        _make_senate_row("FL", "12001", 2022, party_simplified="REPUBLICAN",
                         candidatevotes=60000, totalvotes=100000),
        # Uncontested: no Republican
        _make_senate_row("FL", "12003", 2022, party_simplified="DEMOCRAT",
                         candidatevotes=50000, totalvotes=50000),
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    out = aggregate_county_year(filtered, 2022)
    assert "12003" not in out["county_fips"].values
    assert "12001" in out["county_fips"].values


def test_aggregate_empty_year():
    """Requesting a year with no data returns an empty DataFrame with correct columns."""
    rows = [
        _make_senate_row("FL", "12001", 2018, party_simplified="DEMOCRAT",
                         candidatevotes=30000, totalvotes=95000),
        _make_senate_row("FL", "12001", 2018, party_simplified="REPUBLICAN",
                         candidatevotes=65000, totalvotes=95000),
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    out = aggregate_county_year(filtered, 2010)  # not in data
    assert len(out) == 0
    assert "county_fips" in out.columns
    assert "senate_dem_share_2010" in out.columns


def test_aggregate_multi_race_blending():
    """When a state has two Senate races in the same year (e.g. GA 2020 regular +
    special), candidatevotes are summed across races per party per county."""
    rows = [
        # Race 1 — regular seat
        _make_senate_row("GA", "13001", 2020, party_simplified="DEMOCRAT",
                         candidatevotes=30000, totalvotes=80000),
        _make_senate_row("GA", "13001", 2020, party_simplified="REPUBLICAN",
                         candidatevotes=50000, totalvotes=80000),
        # Race 2 — special seat (same county, same year, same party labels)
        _make_senate_row("GA", "13001", 2020, party_simplified="DEMOCRAT",
                         candidatevotes=20000, totalvotes=60000),
        _make_senate_row("GA", "13001", 2020, party_simplified="REPUBLICAN",
                         candidatevotes=40000, totalvotes=60000),
    ]
    df = _raw_df(rows)
    filtered = filter_senate_rows(df)
    out = aggregate_county_year(filtered, 2020)

    ga1 = out[out["county_fips"] == "13001"].iloc[0]
    # dem total = 30000 + 20000 = 50000; total = 80000 + 60000 = 140000
    assert abs(ga1["senate_dem_2020"] - 50000) < 1e-6
    assert abs(ga1["senate_total_2020"] - 140000) < 1e-6
    assert abs(ga1["senate_dem_share_2020"] - 50000 / 140000) < 1e-6


# ---------------------------------------------------------------------------
# compute_senate_shift
# ---------------------------------------------------------------------------


@pytest.fixture
def early_sen():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "senate_dem_share_2002": [0.45, 0.52, 0.38],
        "senate_total_2002": [80000, 70000, 35000],
    })


@pytest.fixture
def late_sen():
    return pd.DataFrame({
        "county_fips": ["12001", "13001", "01001"],
        "senate_dem_share_2008": [0.50, 0.55, 0.40],
        "senate_total_2008": [90000, 75000, 38000],
    })


def test_senate_d_shift_math(early_sen, late_sen):
    """Senate D shift = logit(later) - logit(earlier)."""
    result = compute_senate_shift(early_sen, late_sen, "02", "08")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    expected = _logit(0.50) - _logit(0.45)
    assert abs(fl_row["sen_d_shift_02_08"] - expected) < 1e-6


def test_senate_r_shift_is_negative_d(early_sen, late_sen):
    """R shift must be the negative of D shift."""
    result = compute_senate_shift(early_sen, late_sen, "02", "08")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    assert abs(fl_row["sen_r_shift_02_08"] + fl_row["sen_d_shift_02_08"]) < 1e-10


def test_senate_turnout_shift(early_sen, late_sen):
    """Senate turnout shift is raw proportional change."""
    result = compute_senate_shift(early_sen, late_sen, "02", "08")
    fl_row = result[result["county_fips"] == "12001"].iloc[0]
    expected = (90000 - 80000) / 80000
    assert abs(fl_row["sen_turnout_shift_02_08"] - expected) < 1e-9


def test_senate_shift_output_columns(early_sen, late_sen):
    """compute_senate_shift returns exactly county_fips + 3 shift cols."""
    result = compute_senate_shift(early_sen, late_sen, "02", "08")
    expected_cols = {
        "county_fips",
        "sen_d_shift_02_08",
        "sen_r_shift_02_08",
        "sen_turnout_shift_02_08",
    }
    assert set(result.columns) == expected_cols


def test_senate_shift_inner_join(early_sen, late_sen):
    """Counties present in only one year are excluded (inner join)."""
    # Add an extra county to early only
    extra = pd.DataFrame({
        "county_fips": ["12099"],
        "senate_dem_share_2002": [0.50],
        "senate_total_2002": [5000],
    })
    early_extra = pd.concat([early_sen, extra], ignore_index=True)
    result = compute_senate_shift(early_extra, late_sen, "02", "08")
    assert "12099" not in result["county_fips"].values
    assert len(result) == 3


# ---------------------------------------------------------------------------
# build_multiyear_shifts — senate_pairs forwarded
# ---------------------------------------------------------------------------


def test_build_multiyear_includes_senate_cols(early_sen, late_sen):
    """build_multiyear_shifts includes Senate shift columns when senate_pairs provided."""
    spine = pd.DataFrame({"county_fips": ["12001", "13001", "01001"]})
    result = build_multiyear_shifts(
        spine,
        pres_pairs=[],
        gov_pairs=[],
        senate_pairs=[("02", "08", early_sen, late_sen)],
    )
    assert "sen_d_shift_02_08" in result.columns
    assert "sen_r_shift_02_08" in result.columns
    assert "sen_turnout_shift_02_08" in result.columns


def test_build_multiyear_senate_none_no_error():
    """build_multiyear_shifts with senate_pairs=None does not raise."""
    spine = pd.DataFrame({"county_fips": ["12001", "13001"]})
    result = build_multiyear_shifts(spine, pres_pairs=[], gov_pairs=[], senate_pairs=None)
    assert len(result) == 2


def test_build_multiyear_senate_missing_county_zero_filled(early_sen, late_sen):
    """Counties on spine but not in Senate data are zero-filled."""
    spine = pd.DataFrame({"county_fips": ["12001", "13001", "01001", "12099"]})
    result = build_multiyear_shifts(
        spine,
        pres_pairs=[],
        gov_pairs=[],
        senate_pairs=[("02", "08", early_sen, late_sen)],
    )
    row = result[result["county_fips"] == "12099"].iloc[0]
    assert row["sen_d_shift_02_08"] == 0.0


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _extract_senate_from_2024_precinct
# ---------------------------------------------------------------------------


def _make_2024_precinct_row(
    state_po: str = "FL",
    county_fips: str = "12001",
    office: str = "US SENATE",
    stage: str = "gen",
    party_simplified: str = "DEMOCRAT",
    votes: int = 10000,
    writein: bool = False,
    mode: str = "TOTAL",
) -> dict:
    return {
        "state_po": state_po,
        "county_fips": county_fips,
        "office": office,
        "stage": stage,
        "party_simplified": party_simplified,
        "votes": votes,
        "writein": writein,
        "mode": mode,
    }


def test_extract_2024_basic():
    """Basic extraction: D+R rows from a single county produce correct output."""
    rows = [
        _make_2024_precinct_row(party_simplified="DEMOCRAT", votes=40000),
        _make_2024_precinct_row(party_simplified="REPUBLICAN", votes=55000),
        _make_2024_precinct_row(party_simplified="LIBERTARIAN", votes=5000),
    ]
    df = pd.DataFrame(rows)
    result = _extract_senate_from_2024_precinct(df, "FL")
    assert len(result) == 3  # D + R + OTHER in county-party agg
    assert "12001" in result["county_fips"].values


def test_extract_2024_filters_state_senate():
    """STATE SENATE rows must be excluded — only US SENATE kept."""
    rows = [
        _make_2024_precinct_row(office="US SENATE", votes=40000),
        _make_2024_precinct_row(office="STATE SENATE", votes=10000),
        _make_2024_precinct_row(office="US SENATE", party_simplified="REPUBLICAN", votes=55000),
    ]
    df = pd.DataFrame(rows)
    result = _extract_senate_from_2024_precinct(df, "FL")
    # Only the two US SENATE rows survive filtering (D + R)
    assert len(result) == 2


def test_extract_2024_empty_when_no_senate():
    """Returns empty DataFrame when state has no Senate race (e.g. only President)."""
    rows = [
        _make_2024_precinct_row(office="US PRESIDENT", votes=40000),
        _make_2024_precinct_row(office="US PRESIDENT", party_simplified="REPUBLICAN", votes=55000),
    ]
    df = pd.DataFrame(rows)
    result = _extract_senate_from_2024_precinct(df, "TX")
    assert result.empty


def test_extract_2024_excludes_writeins():
    """Writein rows should be excluded from Senate extraction."""
    rows = [
        _make_2024_precinct_row(party_simplified="DEMOCRAT", votes=40000, writein=False),
        _make_2024_precinct_row(party_simplified="REPUBLICAN", votes=55000, writein=False),
        _make_2024_precinct_row(party_simplified="OTHER", votes=100, writein=True),
    ]
    df = pd.DataFrame(rows)
    result = _extract_senate_from_2024_precinct(df, "FL")
    total_votes = result["candidatevotes"].sum()
    assert total_votes == 95000  # writein excluded


def test_extract_2024_fips_zero_padded():
    """county_fips should be zero-padded to 5 characters."""
    rows = [
        _make_2024_precinct_row(county_fips="1001.0", party_simplified="DEMOCRAT", votes=10000),
        _make_2024_precinct_row(county_fips="1001.0", party_simplified="REPUBLICAN", votes=15000),
    ]
    df = pd.DataFrame(rows)
    result = _extract_senate_from_2024_precinct(df, "AL")
    assert all(len(f) == 5 for f in result["county_fips"].values)
    assert "01001" in result["county_fips"].values


def test_extract_2024_mode_dedup():
    """When TOTAL mode rows exist, non-TOTAL mode rows should be dropped."""
    rows = [
        # TOTAL mode rows (should be kept)
        _make_2024_precinct_row(party_simplified="DEMOCRAT", votes=40000, mode="TOTAL"),
        _make_2024_precinct_row(party_simplified="REPUBLICAN", votes=55000, mode="TOTAL"),
        # ABSENTEE mode rows (should be dropped because TOTAL exists)
        _make_2024_precinct_row(party_simplified="DEMOCRAT", votes=15000, mode="ABSENTEE"),
        _make_2024_precinct_row(party_simplified="REPUBLICAN", votes=20000, mode="ABSENTEE"),
    ]
    df = pd.DataFrame(rows)
    result = _extract_senate_from_2024_precinct(df, "FL")
    total_votes = result["candidatevotes"].sum()
    assert total_votes == 95000  # only TOTAL mode rows


def test_extract_2024_party_normalization():
    """DEMOCRATIC should be normalized to DEMOCRAT."""
    rows = [
        _make_2024_precinct_row(party_simplified="DEMOCRATIC", votes=40000),
        _make_2024_precinct_row(party_simplified="REPUBLICAN", votes=55000),
    ]
    df = pd.DataFrame(rows)
    result = _extract_senate_from_2024_precinct(df, "FL")
    assert "DEMOCRAT" in result["party_simplified"].values
    assert "DEMOCRATIC" not in result["party_simplified"].values


def test_extract_2024_missing_columns_returns_empty():
    """Returns empty DataFrame when required columns are missing."""
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    result = _extract_senate_from_2024_precinct(df, "FL")
    assert result.empty


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------


def test_senate_years_in_config():
    """SENATE_YEARS must include the expected contested cycles including 2024."""
    required = [2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    for y in required:
        assert y in SENATE_YEARS, f"SENATE_YEARS missing year {y}"


def test_senate_pairs_in_config():
    """SENATE_PAIRS must include the 6-year same-seat pairs including 18→24."""
    expected = {
        ("02", "08"), ("04", "10"), ("06", "12"),
        ("08", "14"), ("10", "16"), ("12", "18"),
        ("14", "20"), ("16", "22"), ("18", "24"),
    }
    actual = set(SENATE_PAIRS)
    assert actual == expected, f"SENATE_PAIRS mismatch.\nExpected: {sorted(expected)}\nGot: {sorted(actual)}"


def test_senate_files_in_config():
    """SENATE_FILES must have an entry for each 2-digit year suffix including 24."""
    senate_files = _cfg.SENATE_FILES
    required_keys = {"02", "04", "06", "08", "10", "12", "14", "16", "18", "20", "22", "24"}
    for k in required_keys:
        assert k in senate_files, f"SENATE_FILES missing key '{k}'"
        assert senate_files[k].startswith("medsl_county_senate_"), (
            f"SENATE_FILES['{k}'] has unexpected filename: {senate_files[k]}"
        )


def test_training_shift_cols_include_senate():
    """TRAINING_SHIFT_COLS must include at least one Senate shift dimension."""
    senate_cols = [c for c in TRAINING_SHIFT_COLS if c.startswith("sen_")]
    assert len(senate_cols) > 0, "No Senate columns found in TRAINING_SHIFT_COLS"
    # 9 pairs × 3 dims = 27 Senate dims (added 18→24 pair)
    assert len(senate_cols) == 27, f"Expected 27 Senate dims, got {len(senate_cols)}"
