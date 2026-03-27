"""Tests for src/tracts/vest_columns.py — VEST column parsing and vote extraction.

Uses synthetic GeoDataFrames to avoid needing real VEST shapefiles.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.tracts.vest_columns import (
    GA_ALT_PATTERN,
    PARTY_CODES,
    RACE_CODES,
    VEST_PATTERN,
    _find_precinct_id,
    extract_vest_votes,
    find_vote_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gdf(columns: dict, n: int = 5) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with given column names and random int data."""
    data = {col: np.random.randint(100, 10000, n) for col in columns}
    geoms = [box(i, 0, i + 1, 1) for i in range(n)]
    return gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# VEST_PATTERN regex
# ---------------------------------------------------------------------------

class TestVestPattern:
    def test_standard_presidential(self):
        m = VEST_PATTERN.match("G16PREDCLI")
        assert m is not None
        assert m.groups() == ("16", "PRE", "D", "CLI")

    def test_standard_republican(self):
        m = VEST_PATTERN.match("G20PRERTRU")
        assert m is not None
        assert m.groups() == ("20", "PRE", "R", "TRU")

    def test_governor(self):
        m = VEST_PATTERN.match("G18GOVRDES")
        assert m is not None
        assert m.groups() == ("18", "GOV", "R", "DES")

    def test_senate(self):
        m = VEST_PATTERN.match("G20USSDOSS")
        assert m is not None
        assert m.groups() == ("20", "USS", "D", "OSS")

    def test_write_in(self):
        m = VEST_PATTERN.match("G16PREOWRI")
        assert m is not None
        assert m.group(3) == "O"

    def test_rejects_non_g_prefix(self):
        assert VEST_PATTERN.match("X16PREDCLI") is None

    def test_rejects_short_candidate(self):
        """Candidate code must be 3+ chars."""
        assert VEST_PATTERN.match("G16PREDCL") is None

    def test_rejects_geoid_column(self):
        """GEOID20 should not match (not a vote column)."""
        assert VEST_PATTERN.match("GEOID20") is None

    def test_rejects_lowercase(self):
        assert VEST_PATTERN.match("g16predcli") is None


class TestGaAltPattern:
    def test_ga_county_certified(self):
        m = GA_ALT_PATTERN.match("C20PREDCLI")
        assert m is not None
        assert m.groups() == ("C", "20", "PRE", "D", "CLI")

    def test_ga_special(self):
        m = GA_ALT_PATTERN.match("S20USSDWAR")
        assert m is not None
        assert m.group(1) == "S"

    def test_ga_runoff(self):
        m = GA_ALT_PATTERN.match("R21USSROSS")
        assert m is not None
        assert m.group(1) == "R"

    def test_standard_g_doesnt_match(self):
        assert GA_ALT_PATTERN.match("G20PREDCLI") is None


# ---------------------------------------------------------------------------
# RACE_CODES and PARTY_CODES
# ---------------------------------------------------------------------------

class TestCodeMappings:
    def test_race_codes_has_president(self):
        assert RACE_CODES["PRE"] == "president"

    def test_race_codes_has_governor(self):
        assert RACE_CODES["GOV"] == "governor"

    def test_race_codes_has_senate(self):
        assert RACE_CODES["USS"] == "us_senate"

    def test_party_codes_dem_rep(self):
        assert PARTY_CODES["D"] == "dem"
        assert PARTY_CODES["R"] == "rep"

    def test_party_codes_third_parties(self):
        assert PARTY_CODES["L"] == "lib"
        assert PARTY_CODES["G"] == "grn"


# ---------------------------------------------------------------------------
# find_vote_columns
# ---------------------------------------------------------------------------

class TestFindVoteColumns:
    def test_finds_2016_presidential(self):
        gdf = _make_gdf({"G16PREDCLI": 0, "G16PRERTRU": 0, "G16PREOWRI": 0})
        result = find_vote_columns(gdf, 2016, "president")
        assert "dem" in result
        assert "rep" in result
        assert "G16PREDCLI" in result["dem"]
        assert "G16PRERTRU" in result["rep"]

    def test_filters_by_year(self):
        gdf = _make_gdf({"G16PREDCLI": 0, "G20PREDBID": 0})
        result = find_vote_columns(gdf, 2016, "president")
        assert "G20PREDBID" not in result.get("dem", [])

    def test_filters_by_race(self):
        gdf = _make_gdf({"G16PREDCLI": 0, "G16GOVRDES": 0})
        result = find_vote_columns(gdf, 2016, "president")
        assert "G16GOVRDES" not in result.get("rep", [])

    def test_unknown_race_raises(self):
        gdf = _make_gdf({"G16PREDCLI": 0})
        with pytest.raises(ValueError, match="Unknown race"):
            find_vote_columns(gdf, 2016, "dogcatcher")

    def test_empty_when_no_match(self):
        gdf = _make_gdf({"G16GOVRDES": 0})
        result = find_vote_columns(gdf, 2016, "president")
        assert result == {}

    def test_ga_alt_not_included_by_default(self):
        gdf = _make_gdf({"C20PREDCLI": 0, "G20PREDCLI": 0})
        result = find_vote_columns(gdf, 2020, "president")
        all_cols = [c for party_cols in result.values() for c in party_cols]
        assert "C20PREDCLI" not in all_cols

    def test_ga_alt_included_when_flag_set(self):
        gdf = _make_gdf({"C20PREDCLI": 0})
        result = find_vote_columns(gdf, 2020, "president", include_alt_prefixes=True)
        all_cols = [c for party_cols in result.values() for c in party_cols]
        assert "C20PREDCLI" in all_cols


# ---------------------------------------------------------------------------
# _find_precinct_id
# ---------------------------------------------------------------------------

class TestFindPrecinctId:
    def test_fl_specific_column(self):
        gdf = _make_gdf({"COUNTY": 0, "PRECINCT": 0, "other": 0})
        assert _find_precinct_id(gdf, "FL") == "COUNTY"

    def test_ga_specific_column(self):
        gdf = _make_gdf({"PRECINCT_I": 0, "other": 0})
        assert _find_precinct_id(gdf, "GA") == "PRECINCT_I"

    def test_al_geoid_column(self):
        gdf = _make_gdf({"GEOID20": 0, "other": 0})
        assert _find_precinct_id(gdf, "AL") == "GEOID20"

    def test_fallback_geoid_pattern(self):
        gdf = _make_gdf({"GEOID22": 0})
        result = _find_precinct_id(gdf, "TX")  # not in state-specific list
        assert result == "GEOID22"

    def test_returns_empty_when_no_match(self):
        gdf = _make_gdf({"votes": 0, "name": 0})
        result = _find_precinct_id(gdf, "ZZ")
        assert result == ""


# ---------------------------------------------------------------------------
# extract_vest_votes
# ---------------------------------------------------------------------------

class TestExtractVestVotes:
    def test_basic_extraction(self):
        data = {
            "G16PREDCLI": [100, 200],
            "G16PRERTRU": [150, 250],
            "G16PREOWRI": [10, 20],
            "PRECINCT": ["P1", "P2"],
        }
        gdf = gpd.GeoDataFrame(
            data, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326"
        )
        result = extract_vest_votes(gdf, "FL", 2016, "president")
        assert list(result.columns) == ["precinct_id", "votes_dem", "votes_rep", "votes_total", "geometry"]
        assert result["votes_dem"].tolist() == [100, 200]
        assert result["votes_rep"].tolist() == [150, 250]
        assert result["votes_total"].tolist() == [260, 470]

    def test_raises_when_no_columns_found(self):
        gdf = _make_gdf({"other": 0})
        with pytest.raises(ValueError, match="No VEST columns found"):
            extract_vest_votes(gdf, "FL", 2016, "president")

    def test_multiple_dem_columns_summed(self):
        """Multiple D candidates should be summed."""
        data = {
            "G16PREDDOE": [100],
            "G16PREDJOE": [50],
            "G16PRERTRU": [200],
        }
        gdf = gpd.GeoDataFrame(data, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        result = extract_vest_votes(gdf, "FL", 2016, "president")
        assert result["votes_dem"].iloc[0] == 150
        assert result["votes_total"].iloc[0] == 350

    def test_preserves_geometry(self):
        gdf = gpd.GeoDataFrame(
            {"G20PREDBID": [100], "G20PRERTRU": [150]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        result = extract_vest_votes(gdf, "FL", 2020, "president")
        assert result.crs is not None
        assert result.geometry is not None

    def test_index_fallback_when_no_id_col(self):
        """When no precinct ID column is found, index should be used."""
        gdf = gpd.GeoDataFrame(
            {"G20PREDBID": [100, 200], "G20PRERTRU": [150, 250]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        result = extract_vest_votes(gdf, "ZZ", 2020, "president")
        # Should fall back to string index
        assert result["precinct_id"].tolist() == ["0", "1"]
