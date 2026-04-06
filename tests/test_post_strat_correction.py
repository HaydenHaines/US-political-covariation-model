"""Tests for post-stratification correction in Tier 2 W vector construction.

These tests verify that:
  1. When a poll oversamples a demographic group, sigma increases (less influence).
  2. The correction factor math is exact: sub_n = n_sample * pop_share.
  3. When population_shares is None, the original behavior is preserved.
  4. The population_vectors module computes correct state vectors.
  5. build_W_poll passes population_shares through to build_W_from_crosstabs.

Background: Tier 2 crosstab observations use pct_of_sample to compute sub_n,
which controls sigma (the observation uncertainty). If a poll oversamples college
grads at 55% when the population is 30%, sub_n = n * 0.55 inflates the sample
size and gives an artificially precise observation. Post-stratification corrects
this: sub_n = n * pop_share (= n * 0.30) restores the correct effective sample
size, increasing sigma and reducing the oversampled group's influence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.poll_enrichment import build_W_from_crosstabs, build_W_poll
from src.prediction.population_vectors import (
    XT_TO_TYPE_PROFILE_COL,
    build_state_population_vectors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_type_profiles(n_types: int = 4) -> pd.DataFrame:
    """Minimal type_profiles with columns referenced by _map_demographic_to_types."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "type_id": list(range(n_types)),
        "pct_bachelors_plus": rng.uniform(0.10, 0.65, n_types),
        "pct_white_nh": rng.uniform(0.20, 0.90, n_types),
        "pct_black": rng.uniform(0.01, 0.40, n_types),
        "pct_hispanic": rng.uniform(0.02, 0.50, n_types),
        "pct_asian": rng.uniform(0.01, 0.20, n_types),
        "evangelical_share": rng.uniform(0.05, 0.80, n_types),
    })


def _make_poll(dem_share: float = 0.52, n_sample: int = 600) -> dict:
    """Minimal poll dict."""
    return {"dem_share": dem_share, "n_sample": n_sample, "state": "WI"}


def _make_state_type_weights(n_types: int = 4) -> np.ndarray:
    """Uniform state type weights."""
    w = np.ones(n_types) / n_types
    return w


def _make_county_assignments(n_counties: int = 6, n_types: int = 4) -> pd.DataFrame:
    """Soft-membership county assignments."""
    rng = np.random.default_rng(7)
    fips = [f"{i:05d}" for i in range(1, n_counties + 1)]
    scores = rng.dirichlet(alpha=np.ones(n_types), size=n_counties)
    score_cols = {f"type_{j}_score": scores[:, j] for j in range(n_types)}
    return pd.DataFrame({"county_fips": fips, **score_cols})


def _make_county_votes(
    county_fips: list[str], state_abbr: list[str], votes: list[int]
) -> pd.DataFrame:
    return pd.DataFrame({
        "county_fips": county_fips,
        "state_abbr": state_abbr,
        "pres_total_2020": votes,
    })


# ---------------------------------------------------------------------------
# 1. Post-stratification correction math
# ---------------------------------------------------------------------------


class TestPostStratCorrection:
    """Verify the sigma correction when population shares differ from poll shares."""

    def setup_method(self):
        self.type_profiles = _make_type_profiles()
        self.state_weights = _make_state_type_weights()
        self.poll = _make_poll(dem_share=0.52, n_sample=1000)

    def test_oversampled_group_gets_larger_sigma(self):
        """Oversampled group (poll_share > pop_share) should have larger sigma than raw.

        Without correction: sub_n = n * poll_share (inflated)
        With correction:    sub_n = n * pop_share  (correct)
        Since poll_share > pop_share, corrected sub_n < raw sub_n → sigma increases.
        """
        poll_share = 0.55   # poll sampled 55% college-educated
        pop_share = 0.30    # state population is 30% college-educated

        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": 0.52,
        }]

        # Without post-stratification
        obs_raw = build_W_from_crosstabs(
            self.poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )
        sigma_raw = obs_raw[0]["sigma"]

        # With post-stratification
        obs_corrected = build_W_from_crosstabs(
            self.poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_education_college": pop_share},
        )
        sigma_corrected = obs_corrected[0]["sigma"]

        # Oversampling correction should increase sigma (more uncertainty)
        assert sigma_corrected > sigma_raw, (
            f"Expected corrected sigma ({sigma_corrected:.6f}) > raw sigma ({sigma_raw:.6f}) "
            "when poll oversamples the group"
        )

    def test_undersampled_group_gets_smaller_sigma(self):
        """Undersampled group (poll_share < pop_share) should have smaller sigma.

        When a group is undersampled, the correction bumps up effective sample size,
        reducing sigma — reflecting that we have good reason to believe the group's
        representation in the state is larger than the poll suggests.
        """
        poll_share = 0.08   # poll only sampled 8% Black respondents
        pop_share = 0.20    # state population is 20% Black

        crosstabs = [{
            "demographic_group": "race",
            "group_value": "black",
            "pct_of_sample": poll_share,
            "dem_share": 0.52,
        }]

        obs_raw = build_W_from_crosstabs(
            self.poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )
        obs_corrected = build_W_from_crosstabs(
            self.poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_race_black": pop_share},
        )

        # Undersampling correction should decrease sigma (more effective representation)
        assert obs_corrected[0]["sigma"] < obs_raw[0]["sigma"], (
            "Expected corrected sigma to be smaller when poll undersamples the group"
        )

    def test_correction_factor_math_is_exact(self):
        """Verify sub_n = n_sample * pop_share exactly (not poll_share)."""
        n_sample = 800
        poll_share = 0.60
        pop_share = 0.35
        dem_share = 0.52

        poll = _make_poll(dem_share=dem_share, n_sample=n_sample)
        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": dem_share,
        }]

        obs = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_education_college": pop_share},
        )

        # Expected: sub_n = int(800 * 0.35) = 280
        expected_sub_n = int(n_sample * pop_share)
        expected_sigma = np.sqrt(dem_share * (1 - dem_share) / expected_sub_n)

        assert abs(obs[0]["sigma"] - expected_sigma) < 1e-10, (
            f"sigma {obs[0]['sigma']:.8f} != expected {expected_sigma:.8f} "
            f"(expected sub_n={expected_sub_n})"
        )

    def test_representative_group_has_same_sigma(self):
        """When poll_share == pop_share, correction is 1.0 and sigma is unchanged."""
        share = 0.35
        dem_share = 0.52
        n_sample = 600

        poll = _make_poll(dem_share=dem_share, n_sample=n_sample)
        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": share,
            "dem_share": dem_share,
        }]

        obs_raw = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )
        obs_corrected = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_education_college": share},
        )

        assert abs(obs_raw[0]["sigma"] - obs_corrected[0]["sigma"]) < 1e-10, (
            "When poll_share == pop_share, correction must have no effect"
        )

    def test_correction_handles_multiple_groups(self):
        """Post-stratification applies independently to each crosstab group."""
        n_sample = 1000
        dem_share = 0.52

        poll = _make_poll(dem_share=dem_share, n_sample=n_sample)
        crosstabs = [
            {
                "demographic_group": "education",
                "group_value": "college",
                "pct_of_sample": 0.55,   # oversampled
                "dem_share": dem_share,
            },
            {
                "demographic_group": "race",
                "group_value": "black",
                "pct_of_sample": 0.08,   # undersampled
                "dem_share": dem_share,
            },
        ]
        population_shares = {
            "xt_education_college": 0.30,  # pop share < poll share → bigger sigma
            "xt_race_black": 0.20,         # pop share > poll share → smaller sigma
        }

        obs_raw = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )
        obs_corrected = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=population_shares,
        )

        # Both groups must be present in both outputs
        assert len(obs_raw) == 2
        assert len(obs_corrected) == 2

        # Sigma relationships hold per-group
        # Observation order tracks crosstabs list order
        assert obs_corrected[0]["sigma"] > obs_raw[0]["sigma"], (
            "Oversampled group should get larger sigma after correction"
        )
        assert obs_corrected[1]["sigma"] < obs_raw[1]["sigma"], (
            "Undersampled group should get smaller sigma after correction"
        )

    def test_w_vector_unchanged_by_correction(self):
        """Post-stratification changes sigma only — the W vector is unaffected."""
        poll_share = 0.55
        pop_share = 0.30
        dem_share = 0.52

        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": dem_share,
        }]

        obs_raw = build_W_from_crosstabs(
            self.poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )
        obs_corrected = build_W_from_crosstabs(
            self.poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_education_college": pop_share},
        )

        # The W vector must be identical — correction only changes sigma
        np.testing.assert_array_equal(
            obs_raw[0]["W"], obs_corrected[0]["W"],
            err_msg="Post-stratification should not change the W vector",
        )


# ---------------------------------------------------------------------------
# 2. Fallback behavior (no population_shares)
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """Verify that passing population_shares=None preserves the original behavior."""

    def setup_method(self):
        self.type_profiles = _make_type_profiles()
        self.state_weights = _make_state_type_weights()
        self.poll = _make_poll()

    def test_no_population_shares_uses_raw_pct(self):
        """When population_shares=None, sub_n = n_sample * pct_of_sample (original)."""
        n_sample = 600
        poll_share = 0.45
        dem_share = 0.52

        poll = _make_poll(dem_share=dem_share, n_sample=n_sample)
        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": dem_share,
        }]

        obs = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )

        expected_sub_n = int(n_sample * poll_share)
        expected_sigma = np.sqrt(dem_share * (1 - dem_share) / expected_sub_n)
        assert abs(obs[0]["sigma"] - expected_sigma) < 1e-10

    def test_missing_xt_col_in_population_shares_falls_back(self):
        """When xt_ col is absent from population_shares, raw pct_of_sample is used."""
        n_sample = 600
        poll_share = 0.45
        dem_share = 0.52

        poll = _make_poll(dem_share=dem_share, n_sample=n_sample)
        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": dem_share,
        }]

        # Provide population_shares but without the matching xt_education_college key.
        obs_corrected = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_race_black": 0.12},  # different group — no match
        )
        obs_raw = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )

        # With no matching key, corrected should equal raw
        assert abs(obs_corrected[0]["sigma"] - obs_raw[0]["sigma"]) < 1e-10, (
            "Fallback to raw sigma expected when xt_col absent from population_shares"
        )

    def test_zero_pop_share_falls_back_to_raw(self):
        """When population_shares maps to 0.0, fall back to raw pct_of_sample."""
        n_sample = 600
        poll_share = 0.45
        dem_share = 0.52

        poll = _make_poll(dem_share=dem_share, n_sample=n_sample)
        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": dem_share,
        }]

        # pop_share = 0 means division would be undefined — fall back to raw
        obs_zero_pop = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_education_college": 0.0},
        )
        obs_raw = build_W_from_crosstabs(
            poll, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )

        assert abs(obs_zero_pop[0]["sigma"] - obs_raw[0]["sigma"]) < 1e-10

    def test_build_W_poll_passes_population_shares_to_tier2(self):
        """build_W_poll must forward population_shares to build_W_from_crosstabs."""
        n_sample = 1000
        poll_share = 0.60
        pop_share = 0.30
        dem_share = 0.52

        # Construct a poll dict that will trigger Tier 2 (has xt_ fields)
        poll_with_xt = {
            "dem_share": dem_share,
            "n_sample": n_sample,
            "state": "WI",
            "xt_education_college": poll_share,
        }

        # build_W_poll with crosstabs (Tier 2)
        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": dem_share,
        }]

        # Call build_W_from_crosstabs directly with and without pop_shares
        obs_raw = build_W_from_crosstabs(
            poll_with_xt, crosstabs, self.type_profiles, self.state_weights,
            population_shares=None,
        )
        obs_corrected = build_W_from_crosstabs(
            poll_with_xt, crosstabs, self.type_profiles, self.state_weights,
            population_shares={"xt_education_college": pop_share},
        )

        # Corrected should have larger sigma for the oversampled group
        assert obs_corrected[0]["sigma"] > obs_raw[0]["sigma"]

        # Also verify build_W_poll routes population_shares through correctly
        result_corrected = build_W_poll(
            poll=poll_with_xt,
            type_profiles=self.type_profiles,
            state_type_weights=self.state_weights,
            poll_crosstabs=crosstabs,
            population_shares={"xt_education_college": pop_share},
        )
        assert isinstance(result_corrected, list)
        assert result_corrected[0]["sigma"] == obs_corrected[0]["sigma"]

    def test_build_W_poll_without_population_shares(self):
        """build_W_poll with population_shares=None matches original behavior."""
        n_sample = 600
        poll_share = 0.45
        dem_share = 0.52

        poll = {"dem_share": dem_share, "n_sample": n_sample, "state": "WI"}
        crosstabs = [{
            "demographic_group": "education",
            "group_value": "college",
            "pct_of_sample": poll_share,
            "dem_share": dem_share,
        }]

        result = build_W_poll(
            poll=poll,
            type_profiles=self.type_profiles,
            state_type_weights=self.state_weights,
            poll_crosstabs=crosstabs,
            population_shares=None,
        )
        assert isinstance(result, list)
        expected_sub_n = int(n_sample * poll_share)
        expected_sigma = np.sqrt(dem_share * (1 - dem_share) / expected_sub_n)
        assert abs(result[0]["sigma"] - expected_sigma) < 1e-10


# ---------------------------------------------------------------------------
# 3. Population vectors module tests
# ---------------------------------------------------------------------------


class TestPopulationVectorsModule:
    """Verify that build_state_population_vectors produces correct results."""

    def setup_method(self):
        self.n_types = 4
        self.type_profiles = pd.DataFrame({
            "type_id": list(range(self.n_types)),
            "pct_white_nh":      [0.9, 0.3, 0.5, 0.7],
            "pct_black":         [0.05, 0.60, 0.20, 0.10],
            "pct_hispanic":      [0.03, 0.08, 0.25, 0.15],
            "pct_asian":         [0.02, 0.02, 0.05, 0.05],
            "pct_bachelors_plus": [0.45, 0.15, 0.30, 0.20],
            "evangelical_share": [0.30, 0.20, 0.15, 0.60],
        })
        rng = np.random.default_rng(7)
        n_counties = 6
        fips = [f"{i:05d}" for i in range(1, n_counties + 1)]
        scores = rng.dirichlet(alpha=np.ones(self.n_types), size=n_counties)
        score_cols = {f"type_{j}_score": scores[:, j] for j in range(self.n_types)}
        self.county_assignments = pd.DataFrame({"county_fips": fips, **score_cols})

        states = ["AA", "AA", "AA", "BB", "BB", "BB"]
        votes = [10000, 20000, 30000, 15000, 25000, 5000]
        self.county_votes = _make_county_votes(fips, states, votes)

    def test_xt_to_type_profile_col_completeness(self):
        """XT_TO_TYPE_PROFILE_COL must map to valid type_profiles columns."""
        tp = self.type_profiles
        for xt_col, profile_col in XT_TO_TYPE_PROFILE_COL.items():
            assert profile_col in tp.columns, (
                f"XT_TO_TYPE_PROFILE_COL['{xt_col}'] = '{profile_col}' not in type_profiles"
            )

    def test_returns_both_states(self):
        """Should produce a vector for every state in county_votes."""
        vecs = build_state_population_vectors(
            self.type_profiles,
            self.county_assignments,
            self.county_votes,
            ["xt_race_white", "xt_race_black"],
        )
        assert "AA" in vecs
        assert "BB" in vecs

    def test_values_are_proportions(self):
        """All demographic shares should be in [0, 1]."""
        vecs = build_state_population_vectors(
            self.type_profiles,
            self.county_assignments,
            self.county_votes,
            ["xt_race_white", "xt_race_black", "xt_education_college"],
        )
        for state, vec in vecs.items():
            for col, val in vec.items():
                assert 0.0 <= val <= 1.0, f"{state}:{col} = {val} out of [0,1]"

    def test_keys_limited_to_mappable_xt_cols(self):
        """Output keys must only include xt_ columns present in XT_TO_TYPE_PROFILE_COL."""
        xt_cols = ["xt_race_white", "xt_race_black", "xt_urbanicity_rural"]
        vecs = build_state_population_vectors(
            self.type_profiles,
            self.county_assignments,
            self.county_votes,
            xt_cols,
        )
        for state_vec in vecs.values():
            # xt_urbanicity_rural has no mapping → should not appear
            assert "xt_urbanicity_rural" not in state_vec

    def test_vote_weighting_changes_result(self):
        """A county with much higher vote total should dominate state demographics."""
        fips = self.county_assignments["county_fips"].tolist()

        # Scenario A: county 0 dominates AA
        votes_a = [100_000, 1, 1, 1, 1, 1]
        cv_a = _make_county_votes(fips, ["AA", "AA", "AA", "BB", "BB", "BB"], votes_a)
        vecs_a = build_state_population_vectors(
            self.type_profiles, self.county_assignments, cv_a, ["xt_race_white"]
        )

        # Scenario B: county 2 dominates AA
        votes_b = [1, 1, 100_000, 1, 1, 1]
        cv_b = _make_county_votes(fips, ["AA", "AA", "AA", "BB", "BB", "BB"], votes_b)
        vecs_b = build_state_population_vectors(
            self.type_profiles, self.county_assignments, cv_b, ["xt_race_white"]
        )

        # If counties 0 and 2 differ in soft membership, state vectors should differ
        n_types = self.n_types
        score_col_names = [f"type_{j}_score" for j in range(n_types)]
        scores_0 = self.county_assignments.iloc[0][score_col_names].astype(float).values
        scores_2 = self.county_assignments.iloc[2][score_col_names].astype(float).values
        if not np.allclose(scores_0, scores_2):
            assert vecs_a["AA"]["xt_race_white"] != vecs_b["AA"]["xt_race_white"], (
                "Different dominant counties should produce different state vectors"
            )

    def test_vector_is_weighted_average_of_county_estimates(self):
        """Manually verify the vote-weighted computation for a simple 2-county state."""
        # Use 2 types so the math is easy to verify by hand.
        n_types = 2
        type_profiles = pd.DataFrame({
            "type_id": [0, 1],
            "pct_black": [0.60, 0.10],
        })
        # Two counties: county A has 100% type_0 membership, county B has 100% type_1
        county_assignments = pd.DataFrame({
            "county_fips": ["00001", "00002"],
            "type_0_score": [1.0, 0.0],
            "type_1_score": [0.0, 1.0],
        })
        # County A: 1000 votes, county B: 4000 votes → BB = 80% of state
        county_votes = pd.DataFrame({
            "county_fips": ["00001", "00002"],
            "state_abbr": ["ZZ", "ZZ"],
            "pres_total_2020": [1000, 4000],
        })

        vecs = build_state_population_vectors(
            type_profiles, county_assignments, county_votes, ["xt_race_black"]
        )

        # county A pct_black = 0.60 (pure type_0), county B pct_black = 0.10 (pure type_1)
        # vote-weighted: (1000 * 0.60 + 4000 * 0.10) / 5000 = (600 + 400) / 5000 = 0.20
        expected = (1000 * 0.60 + 4000 * 0.10) / 5000
        assert abs(vecs["ZZ"]["xt_race_black"] - expected) < 1e-10, (
            f"Expected {expected:.4f}, got {vecs['ZZ']['xt_race_black']:.4f}"
        )

    def test_mismatched_score_columns_raises(self):
        """Mismatched number of score columns vs type_profiles rows should raise ValueError."""
        # type_profiles has 4 types but assignments only have 2 score columns
        county_assignments_small = pd.DataFrame({
            "county_fips": ["00001"],
            "type_0_score": [0.5],
            "type_1_score": [0.5],
        })
        with pytest.raises(ValueError, match="Score columns.*don't match type profiles"):
            build_state_population_vectors(
                self.type_profiles,
                county_assignments_small,
                self.county_votes,
                ["xt_race_white"],
            )

    def test_no_mappable_columns_raises(self):
        """Requesting only unmappable xt_ columns should raise ValueError."""
        with pytest.raises(ValueError, match="No mappable xt_ columns"):
            build_state_population_vectors(
                self.type_profiles,
                self.county_assignments,
                self.county_votes,
                xt_cols=["xt_urbanicity_rural", "xt_age_senior"],  # neither is mappable
            )
