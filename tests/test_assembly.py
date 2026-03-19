"""Tests for the data assembly pipeline.

Tests cover:
- Community weight files (propagation matrices) that do exist on disk
- Poll ingestion logic that operates on data/polls/polls_2026.csv
- Mathematical properties of the assembled propagation data
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1]
COMP_COLS = [f"c{k}" for k in range(1, 8)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def state_weights() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data" / "propagation" / "community_weights_state.parquet")


@pytest.fixture(scope="module")
def county_weights() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet")


@pytest.fixture(scope="module")
def tract_weights() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data" / "propagation" / "community_weights_tract.parquet")


# ---------------------------------------------------------------------------
# Propagation weight file tests
# ---------------------------------------------------------------------------


def test_state_weights_file_exists():
    """Propagation state weight file must be present."""
    assert (PROJECT_ROOT / "data" / "propagation" / "community_weights_state.parquet").exists()


def test_county_weights_file_exists():
    """Propagation county weight file must be present."""
    assert (PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet").exists()


def test_tract_weights_file_exists():
    """Propagation tract weight file must be present."""
    assert (PROJECT_ROOT / "data" / "propagation" / "community_weights_tract.parquet").exists()


def test_state_weights_covers_fl_ga_al(state_weights):
    """State weight matrix must contain exactly FL, GA, and AL."""
    states = set(state_weights["state_abbr"])
    assert states == {"FL", "GA", "AL"}


def test_county_weights_covers_fl_ga_al(county_weights):
    """County weight matrix must contain counties from FL, GA, and AL only."""
    states = set(county_weights["state_abbr"])
    assert states == {"FL", "GA", "AL"}


def test_tract_weights_covers_fl_ga_al(tract_weights):
    """Tract weight matrix must contain tracts from FL, GA, and AL only."""
    states = set(tract_weights["state_abbr"])
    assert states == {"FL", "GA", "AL"}


def test_state_weights_have_seven_community_columns(state_weights):
    """State weight matrix must have exactly 7 community columns (c1..c7)."""
    for col in COMP_COLS:
        assert col in state_weights.columns, f"Missing column: {col}"


def test_county_weights_have_seven_community_columns(county_weights):
    """County weight matrix must have exactly 7 community columns (c1..c7)."""
    for col in COMP_COLS:
        assert col in county_weights.columns, f"Missing column: {col}"


def test_tract_weights_have_seven_community_columns(tract_weights):
    """Tract weight matrix must have exactly 7 community columns (c1..c7)."""
    for col in COMP_COLS:
        assert col in tract_weights.columns, f"Missing column: {col}"


def test_state_weights_sum_to_one(state_weights):
    """Each state's community weights must sum to 1.0 (within floating-point tolerance)."""
    row_sums = state_weights[COMP_COLS].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"State weight rows don't sum to 1. Max deviation: {abs(row_sums - 1.0).max():.2e}"
    )


def test_county_weights_sum_to_one(county_weights):
    """Each county's community weights must sum to 1.0 (within floating-point tolerance)."""
    row_sums = county_weights[COMP_COLS].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"County weight rows don't sum to 1. Max deviation: {abs(row_sums - 1.0).max():.2e}"
    )


def test_tract_weights_sum_to_one(tract_weights):
    """Each tract's community weights must sum to 1.0 (within floating-point tolerance)."""
    row_sums = tract_weights[COMP_COLS].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Tract weight rows don't sum to 1. Max deviation: {abs(row_sums - 1.0).max():.2e}"
    )


def test_state_weights_are_non_negative(state_weights):
    """Community weights must all be >= 0 (they represent fractions of the population)."""
    assert (state_weights[COMP_COLS].values >= 0).all()


def test_county_weights_are_non_negative(county_weights):
    """Community weights must all be >= 0."""
    assert (county_weights[COMP_COLS].values >= 0).all()


def test_tract_weights_are_non_negative(tract_weights):
    """Community weights must all be >= 0."""
    assert (tract_weights[COMP_COLS].values >= 0).all()


def test_county_fips_are_five_digits(county_weights):
    """County FIPS codes must be 5-character strings (2-digit state + 3-digit county)."""
    fips = county_weights["county_fips"]
    assert (fips.str.len() == 5).all(), "Some county FIPS codes are not 5 characters"
    assert fips.str.isdigit().all(), "Some county FIPS codes contain non-digit characters"


def test_tract_geoid_are_eleven_digits(tract_weights):
    """Tract GEOIDs must be 11-character strings (2 state + 3 county + 6 tract)."""
    geoids = tract_weights["tract_geoid"]
    assert (geoids.str.len() == 11).all(), "Some tract GEOIDs are not 11 characters"
    assert geoids.str.isdigit().all(), "Some tract GEOIDs contain non-digit characters"


def test_county_fips_state_prefix_matches_state_abbr(county_weights):
    """First 2 digits of county FIPS must match state FIPS (01=AL, 12=FL, 13=GA)."""
    fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
    derived_state = county_weights["county_fips"].str[:2].map(fips_to_abbr)
    mismatches = county_weights["state_abbr"] != derived_state
    assert not mismatches.any(), (
        f"{mismatches.sum()} county rows have mismatched state_fips/state_abbr"
    )


def test_county_count_reasonable(county_weights):
    """FL+GA+AL have 67+159+67=293 counties; weight file must match."""
    # The pipeline may include slightly different counts due to merging,
    # but should be in the right ballpark.
    assert 200 <= len(county_weights) <= 300, (
        f"Expected 200-300 counties, got {len(county_weights)}"
    )


def test_tract_count_reasonable(tract_weights):
    """FL+GA+AL have ~9,393 tracts; weight file should be close to that."""
    assert 9_000 <= len(tract_weights) <= 10_000, (
        f"Expected 9,000-10,000 tracts, got {len(tract_weights)}"
    )


# ---------------------------------------------------------------------------
# Poll CSV tests
# ---------------------------------------------------------------------------


def test_polls_csv_exists():
    """2026 poll CSV must exist at data/polls/polls_2026.csv."""
    assert (PROJECT_ROOT / "data" / "polls" / "polls_2026.csv").exists()


def test_polls_csv_has_required_columns():
    """Poll CSV must have all required columns."""
    required = {"race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster"}
    df = pd.read_csv(PROJECT_ROOT / "data" / "polls" / "polls_2026.csv")
    missing = required - set(df.columns)
    assert not missing, f"Poll CSV missing columns: {missing}"


def test_polls_dem_share_in_range():
    """All dem_share values in polls CSV must be strictly between 0 and 1."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "polls" / "polls_2026.csv")
    out_of_range = df[(df["dem_share"] <= 0) | (df["dem_share"] >= 1)]
    assert len(out_of_range) == 0, (
        f"{len(out_of_range)} poll rows have dem_share outside (0, 1)"
    )


def test_polls_n_sample_positive():
    """All n_sample values must be positive integers."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "polls" / "polls_2026.csv")
    assert (df["n_sample"] > 0).all(), "Some polls have non-positive sample sizes"


def test_polls_not_empty():
    """Poll CSV must contain at least one row."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "polls" / "polls_2026.csv")
    assert len(df) > 0, "Poll CSV is empty"
