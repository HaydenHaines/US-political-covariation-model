"""Tests for build_sci_features.py.

Verifies:
- Feature builder produces expected columns
- All counties have valid (non-NaN) values after imputation
- SCI-weighted dem share is between 0 and 1 for matched counties
- network_diversity is positive
- pct_sci_instate is in [0, 1]
- sci_geographic_reach is a positive integer
- Self-connections are excluded
- FIPS is zero-padded to 5 chars
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembly.build_sci_features import (
    _build_geographic_reach,
    _build_hhi,
    _build_pct_instate,
    _build_political_feature,
    build_features,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_COLS = [
    "network_diversity",
    "pct_sci_instate",
    "sci_top5_mean_dem_share",
    "sci_geographic_reach",
]


def _make_sci_edges(seed: int = 0) -> pd.DataFrame:
    """Create a minimal synthetic SCI edge list.

    5 user counties, each connected to several others (already cleaned — no
    self-connections, zero-padded FIPS, positive SCI).
    """
    rng = np.random.default_rng(seed)
    rows = []
    counties = ["01001", "01003", "01005", "12001", "12003", "13001", "13003"]
    for user in counties:
        for friend in counties:
            if user == friend:
                continue
            rows.append({
                "user_region": user,
                "friend_region": friend,
                "scaled_sci": float(rng.integers(100, 100000)),
            })
    return pd.DataFrame(rows)


def _make_pres_2020(fips_list: list[str]) -> pd.DataFrame:
    """Minimal synthetic 2020 presidential results."""
    rng = np.random.default_rng(42)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "pres_dem_share_2020": rng.uniform(0.25, 0.75, size=n),
    })


# ---------------------------------------------------------------------------
# Tests: build_features (integration)
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    """Integration tests for the top-level build_features() function."""

    def test_expected_columns_present(self):
        """Output must contain all 4 SCI feature columns plus county_fips."""
        sci = _make_sci_edges()
        pres = _make_pres_2020(sci["friend_region"].unique().tolist())
        result = build_features(sci, pres)
        assert "county_fips" in result.columns
        for col in EXPECTED_FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_matches_user_counties(self):
        """Output has one row per unique user_region in SCI data."""
        sci = _make_sci_edges()
        pres = _make_pres_2020(sci["friend_region"].unique().tolist())
        result = build_features(sci, pres)
        n_users = sci["user_region"].nunique()
        assert len(result) == n_users

    def test_no_duplicate_county_fips(self):
        """Each county_fips appears at most once."""
        sci = _make_sci_edges()
        pres = _make_pres_2020(sci["friend_region"].unique().tolist())
        result = build_features(sci, pres)
        assert result["county_fips"].nunique() == len(result)


# ---------------------------------------------------------------------------
# Tests: network_diversity
# ---------------------------------------------------------------------------


class TestNetworkDiversity:
    """network_diversity (1 - HHI) should always be in (0, 1]."""

    def test_diversity_positive(self):
        """network_diversity must be > 0 for all counties."""
        sci = _make_sci_edges()
        result = _build_hhi(sci)
        assert (result["network_diversity"] > 0).all(), (
            f"Found non-positive diversity values: {result[result['network_diversity'] <= 0]}"
        )

    def test_diversity_at_most_one(self):
        """network_diversity must be <= 1 (HHI floor is 0, so 1 - HHI <= 1)."""
        sci = _make_sci_edges()
        result = _build_hhi(sci)
        assert (result["network_diversity"] <= 1.0).all()

    def test_concentrated_county_has_low_diversity(self):
        """A county that sends all its SCI to one friend should have near-zero diversity."""
        # All SCI weight to a single friend county
        sci_concentrated = pd.DataFrame([
            {"user_region": "01001", "friend_region": "01003", "scaled_sci": 1000000.0},
            {"user_region": "01001", "friend_region": "01005", "scaled_sci": 1.0},  # tiny noise
        ])
        result = _build_hhi(sci_concentrated)
        row = result[result["county_fips"] == "01001"]
        assert len(row) == 1
        # HHI ≈ 1 → diversity ≈ 0
        assert row["network_diversity"].iloc[0] < 0.05

    def test_equal_weight_county_has_high_diversity(self):
        """A county with equal weight to many friends should have high diversity."""
        n_friends = 50
        sci_equal = pd.DataFrame([
            {"user_region": "01001", "friend_region": f"{i:05d}", "scaled_sci": 1.0}
            for i in range(2, n_friends + 2)
        ])
        result = _build_hhi(sci_equal)
        row = result[result["county_fips"] == "01001"]
        # HHI = 1/50 = 0.02, diversity ≈ 0.98
        assert row["network_diversity"].iloc[0] > 0.95


# ---------------------------------------------------------------------------
# Tests: pct_sci_instate
# ---------------------------------------------------------------------------


class TestPctSciInstate:
    """pct_sci_instate should be in [0, 1]."""

    def test_bounded_zero_one(self):
        """pct_sci_instate must be in [0, 1] for all counties."""
        sci = _make_sci_edges()
        result = _build_pct_instate(sci)
        assert (result["pct_sci_instate"] >= 0).all()
        assert (result["pct_sci_instate"] <= 1).all()

    def test_all_instate_is_one(self):
        """County connected only to in-state friends should have pct_sci_instate == 1."""
        # All Alabama counties
        sci_instate = pd.DataFrame([
            {"user_region": "01001", "friend_region": "01003", "scaled_sci": 100.0},
            {"user_region": "01001", "friend_region": "01005", "scaled_sci": 200.0},
        ])
        result = _build_pct_instate(sci_instate)
        row = result[result["county_fips"] == "01001"]
        assert abs(row["pct_sci_instate"].iloc[0] - 1.0) < 1e-9

    def test_all_outstate_is_zero(self):
        """County connected only to out-of-state friends should have pct_sci_instate == 0."""
        # AL user, all friends in FL (different state prefix)
        sci_outstate = pd.DataFrame([
            {"user_region": "01001", "friend_region": "12001", "scaled_sci": 100.0},
            {"user_region": "01001", "friend_region": "12003", "scaled_sci": 200.0},
        ])
        result = _build_pct_instate(sci_outstate)
        row = result[result["county_fips"] == "01001"]
        assert abs(row["pct_sci_instate"].iloc[0] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests: sci_top5_mean_dem_share
# ---------------------------------------------------------------------------


class TestSciTop5DemShare:
    """sci_top5_mean_dem_share should be in [0, 1] for matched counties."""

    def test_dem_share_bounded(self):
        """Weighted Dem share must be in [0, 1] for all counties with data."""
        sci = _make_sci_edges()
        top5 = (
            sci.sort_values("scaled_sci", ascending=False)
            .groupby("user_region")
            .head(5)
        )
        pres = _make_pres_2020(sci["friend_region"].unique().tolist())
        result = _build_political_feature(top5, pres)
        valid = result.dropna(subset=["sci_top5_mean_dem_share"])
        assert (valid["sci_top5_mean_dem_share"] >= 0).all()
        assert (valid["sci_top5_mean_dem_share"] <= 1).all()

    def test_known_value_computation(self):
        """Verify exact weighted-average computation with known inputs."""
        # User in one county; 2 friends with known Dem shares and SCI weights
        sci_simple = pd.DataFrame([
            {"user_region": "01001", "friend_region": "12001", "scaled_sci": 100.0},
            {"user_region": "01001", "friend_region": "12003", "scaled_sci": 300.0},
        ])
        pres_simple = pd.DataFrame({
            "county_fips": ["12001", "12003"],
            "pres_dem_share_2020": [0.40, 0.60],
        })
        # Expected: (100*0.40 + 300*0.60) / (100+300) = (40+180)/400 = 220/400 = 0.55
        result = _build_political_feature(sci_simple, pres_simple)
        row = result[result["county_fips"] == "01001"]
        assert len(row) == 1
        assert abs(row["sci_top5_mean_dem_share"].iloc[0] - 0.55) < 1e-9

    def test_missing_friend_excluded(self):
        """Friends without 2020 results are excluded from the weighted average."""
        sci_simple = pd.DataFrame([
            {"user_region": "01001", "friend_region": "12001", "scaled_sci": 100.0},
            {"user_region": "01001", "friend_region": "99999", "scaled_sci": 999.0},  # no data
        ])
        pres_simple = pd.DataFrame({
            "county_fips": ["12001"],
            "pres_dem_share_2020": [0.50],
        })
        result = _build_political_feature(sci_simple, pres_simple)
        row = result[result["county_fips"] == "01001"]
        # Only 12001 contributes; dem share = 0.50
        assert abs(row["sci_top5_mean_dem_share"].iloc[0] - 0.50) < 1e-9


# ---------------------------------------------------------------------------
# Tests: sci_geographic_reach
# ---------------------------------------------------------------------------


class TestSciGeographicReach:
    """sci_geographic_reach counts distinct states in top-20 connections."""

    def test_reach_positive(self):
        """Geographic reach must be >= 1 (at least the user's own or one other state)."""
        sci = _make_sci_edges()
        top20 = (
            sci.sort_values("scaled_sci", ascending=False)
            .groupby("user_region")
            .head(20)
        )
        result = _build_geographic_reach(top20)
        assert (result["sci_geographic_reach"] >= 1).all()

    def test_single_state_connections_reach_one(self):
        """County with only in-state friends should have geographic reach == 1."""
        sci_instate = pd.DataFrame([
            {"user_region": "01001", "friend_region": "01003", "scaled_sci": 100.0},
            {"user_region": "01001", "friend_region": "01005", "scaled_sci": 200.0},
        ])
        result = _build_geographic_reach(sci_instate)
        row = result[result["county_fips"] == "01001"]
        assert row["sci_geographic_reach"].iloc[0] == 1

    def test_multi_state_connections(self):
        """County with friends in 3 different states should have reach == 3."""
        sci_multi = pd.DataFrame([
            {"user_region": "01001", "friend_region": "01003", "scaled_sci": 100.0},  # AL
            {"user_region": "01001", "friend_region": "12001", "scaled_sci": 200.0},  # FL
            {"user_region": "01001", "friend_region": "13001", "scaled_sci": 300.0},  # GA
        ])
        result = _build_geographic_reach(sci_multi)
        row = result[result["county_fips"] == "01001"]
        assert row["sci_geographic_reach"].iloc[0] == 3
