"""Tests for poll propagation model.

Covers:
  - PollObservation dataclass: sigma computation, validation
  - CommunityPosterior: credible intervals, to_dataframe
  - bayesian_poll_update: Kalman filter update mechanics
    - posterior is narrower than prior (uncertainty reduction)
    - single perfect poll with all-in-one-community recovers that community's value
    - multiple polls stack correctly
    - no polls returns prior unchanged
    - propagation respects covariance structure (correlated communities move together)
  - load_weight_vector: real data round-trip
  - load_polls / ingest_polls: CSV ingestion filtering and validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.propagation.propagate_polls import (
    COMP_COLS,
    K,
    CommunityPosterior,
    PollObservation,
    bayesian_poll_update,
    load_weight_vector,
)
from src.assembly.ingest_polls import load_polls, list_races


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diagonal_prior(k: int = K, variance: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Return a simple diagonal Gaussian prior (mu, Sigma) for testing."""
    mu = np.full(k, 0.5)
    Sigma = np.eye(k) * variance
    return mu, Sigma


def _correlated_prior(k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Return a prior with off-diagonal correlations for covariance-propagation tests."""
    rng = np.random.default_rng(99)
    A = rng.standard_normal((k, k)) * 0.05
    Sigma = A @ A.T + np.eye(k) * 0.005
    mu = np.linspace(0.35, 0.65, k)
    return mu, Sigma


# ---------------------------------------------------------------------------
# PollObservation tests
# ---------------------------------------------------------------------------


def test_poll_sigma_formula():
    """sigma property must equal sqrt(p*(1-p)/n) — binomial standard error."""
    poll = PollObservation("FL", dem_share=0.50, n_sample=1000)
    expected = np.sqrt(0.50 * 0.50 / 1000)
    assert abs(poll.sigma - expected) < 1e-12


def test_poll_sigma_decreases_with_larger_n():
    """Larger sample size must give smaller standard error."""
    small = PollObservation("FL", dem_share=0.45, n_sample=400)
    large = PollObservation("FL", dem_share=0.45, n_sample=2000)
    assert large.sigma < small.sigma


def test_poll_sigma_larger_near_50pct():
    """Variance is maximized at 50%; dem_share=0.5 gives larger sigma than extreme shares."""
    mid = PollObservation("FL", dem_share=0.50, n_sample=1000)
    extreme = PollObservation("FL", dem_share=0.10, n_sample=1000)
    assert mid.sigma > extreme.sigma


def test_poll_repr_contains_geography():
    """repr must include the geography string for legibility."""
    poll = PollObservation("GA", dem_share=0.48, n_sample=800)
    assert "GA" in repr(poll)


# ---------------------------------------------------------------------------
# CommunityPosterior tests
# ---------------------------------------------------------------------------


def test_community_posterior_credible_interval_shape():
    """credible_interval must return two arrays of length K."""
    mu = np.linspace(0.35, 0.65, K)
    Sigma = np.eye(K) * 0.01
    post = CommunityPosterior(mu=mu, sigma=Sigma, comps=COMP_COLS)
    lo, hi = post.credible_interval(0.90)
    assert lo.shape == (K,)
    assert hi.shape == (K,)


def test_community_posterior_credible_interval_ordering():
    """Lower bound must be <= mu <= upper bound for every community."""
    mu = np.linspace(0.35, 0.65, K)
    Sigma = np.eye(K) * 0.01
    post = CommunityPosterior(mu=mu, sigma=Sigma, comps=COMP_COLS)
    lo, hi = post.credible_interval(0.90)
    assert (lo <= mu).all(), "Some lower CI bounds exceed mu"
    assert (hi >= mu).all(), "Some upper CI bounds are below mu"


def test_community_posterior_wider_ci_at_higher_level():
    """90% CI must be narrower than 95% CI (wider at higher confidence level)."""
    mu = np.full(K, 0.50)
    Sigma = np.eye(K) * 0.02
    post = CommunityPosterior(mu=mu, sigma=Sigma, comps=COMP_COLS)
    lo90, hi90 = post.credible_interval(0.90)
    lo95, hi95 = post.credible_interval(0.95)
    assert (hi95 - lo95 >= hi90 - lo90).all(), "95% CI should be at least as wide as 90% CI"


def test_community_posterior_to_dataframe_columns():
    """to_dataframe must produce a DataFrame with expected column set."""
    mu = np.full(K, 0.50)
    Sigma = np.eye(K) * 0.01
    post = CommunityPosterior(mu=mu, sigma=Sigma, comps=COMP_COLS)
    df = post.to_dataframe()
    expected_cols = {"component", "label", "mu_post", "std_post", "lo90", "hi90"}
    assert expected_cols.issubset(set(df.columns))


def test_community_posterior_to_dataframe_row_count():
    """to_dataframe must have exactly K rows (one per community)."""
    mu = np.full(K, 0.50)
    Sigma = np.eye(K) * 0.01
    post = CommunityPosterior(mu=mu, sigma=Sigma, comps=COMP_COLS)
    df = post.to_dataframe()
    assert len(df) == K


# ---------------------------------------------------------------------------
# bayesian_poll_update tests
# ---------------------------------------------------------------------------


def test_no_polls_returns_prior():
    """With zero polls, the posterior must equal the prior exactly."""
    mu, Sigma = _diagonal_prior()
    posterior = bayesian_poll_update(mu, Sigma, polls=[])
    assert np.allclose(posterior.mu, mu, atol=1e-12)
    assert np.allclose(posterior.sigma, Sigma, atol=1e-12)


def test_poll_update_reduces_uncertainty():
    """Adding a poll must reduce posterior variance (tighter credible intervals)."""
    mu, Sigma = _diagonal_prior(variance=0.01)
    # Weight vector for synthetic geography: unit vector for community 0
    W_lookup = {"TEST": np.eye(K)[0]}  # poll is pure community 1
    poll = PollObservation("TEST", dem_share=0.55, n_sample=1000, geo_level="state")
    posterior = bayesian_poll_update(mu, Sigma, polls=[poll], weight_lookup=W_lookup)

    prior_std = np.sqrt(np.diag(Sigma))
    post_std = np.sqrt(np.diag(posterior.sigma))
    # At least one community must have reduced uncertainty
    assert (post_std <= prior_std + 1e-12).all(), (
        "Posterior uncertainty should not exceed prior uncertainty in any community"
    )
    # The directly observed community (c1) must have strictly reduced uncertainty
    assert post_std[0] < prior_std[0], (
        "Directly observed community's posterior std should be < prior std"
    )


def test_poll_update_moves_mean_toward_observation():
    """After a high-dem poll, posterior mean should shift up relative to prior."""
    mu, Sigma = _diagonal_prior(variance=0.01)
    prior_mean = mu.copy()
    # Weight: equal across all communities (statewide poll)
    W_equal = np.full(K, 1.0 / K)
    W_lookup = {"TEST": W_equal}
    poll = PollObservation("TEST", dem_share=0.70, n_sample=500, geo_level="state")
    posterior = bayesian_poll_update(mu, Sigma, polls=[poll], weight_lookup=W_lookup)
    # Posterior aggregate should be higher than prior aggregate
    prior_agg = float(W_equal @ prior_mean)
    post_agg = float(W_equal @ posterior.mu)
    assert post_agg > prior_agg, (
        f"Posterior aggregate ({post_agg:.3f}) should be higher than prior ({prior_agg:.3f}) "
        "after high-dem poll"
    )


def test_poll_update_moves_mean_toward_low_dem_observation():
    """After a low-dem poll, posterior mean should shift down relative to prior."""
    mu, Sigma = _diagonal_prior(variance=0.01)
    W_equal = np.full(K, 1.0 / K)
    W_lookup = {"TEST": W_equal}
    poll = PollObservation("TEST", dem_share=0.30, n_sample=500, geo_level="state")
    posterior = bayesian_poll_update(mu, Sigma, polls=[poll], weight_lookup=W_lookup)
    prior_agg = float(W_equal @ mu)
    post_agg = float(W_equal @ posterior.mu)
    assert post_agg < prior_agg, (
        f"Posterior aggregate ({post_agg:.3f}) should be lower than prior ({prior_agg:.3f}) "
        "after low-dem poll"
    )


def test_precise_poll_dominates_diffuse_prior():
    """A poll with a huge sample (very precise) should override an uninformative prior."""
    mu, Sigma = _diagonal_prior(variance=0.05)  # quite uninformative
    W_lookup = {"TEST": np.full(K, 1.0 / K)}
    # Extremely large sample → near-certain observation
    poll = PollObservation("TEST", dem_share=0.60, n_sample=10_000_000, geo_level="state")
    posterior = bayesian_poll_update(mu, Sigma, polls=[poll], weight_lookup=W_lookup)
    post_agg = float(np.full(K, 1.0 / K) @ posterior.mu)
    assert abs(post_agg - 0.60) < 0.02, (
        f"With a very precise poll at 0.60, posterior aggregate should be near 0.60; got {post_agg:.3f}"
    )


def test_multiple_polls_stack_additively():
    """Two polls in the same geography should narrow uncertainty more than one poll."""
    mu, Sigma = _diagonal_prior(variance=0.01)
    W_lookup = {"TEST": np.full(K, 1.0 / K)}
    one_poll = [PollObservation("TEST", dem_share=0.50, n_sample=1000, geo_level="state")]
    two_polls = [
        PollObservation("TEST", dem_share=0.50, n_sample=1000, geo_level="state"),
        PollObservation("TEST", dem_share=0.50, n_sample=1000, geo_level="state"),
    ]
    post1 = bayesian_poll_update(mu, Sigma, one_poll, weight_lookup=W_lookup)
    post2 = bayesian_poll_update(mu, Sigma, two_polls, weight_lookup=W_lookup)
    # Two polls should give smaller posterior variances
    std1 = np.sqrt(np.diag(post1.sigma))
    std2 = np.sqrt(np.diag(post2.sigma))
    assert (std2 <= std1 + 1e-12).all(), "Two polls should give tighter posterior than one poll"


def test_poll_update_respects_covariance():
    """A poll that constrains one community should shift correlated communities too."""
    # Build a prior where c1 and c2 are strongly positively correlated
    mu = np.full(K, 0.50)
    Sigma = np.eye(K) * 0.005
    Sigma[0, 1] = 0.004  # strong positive correlation between c1 and c2
    Sigma[1, 0] = 0.004

    # Poll is a pure c1 observation (high dem share)
    W_c1 = np.zeros(K)
    W_c1[0] = 1.0
    W_lookup = {"TEST_C1": W_c1}
    poll = PollObservation("TEST_C1", dem_share=0.70, n_sample=2000, geo_level="state")
    posterior = bayesian_poll_update(mu, Sigma, [poll], weight_lookup=W_lookup)

    # c1 should shift up
    assert posterior.mu[0] > mu[0], "c1 should shift toward the poll"
    # c2 is positively correlated with c1 — it should also shift up (though less)
    assert posterior.mu[1] > mu[1], (
        "c2 is positively correlated with c1; it should also shift up when c1 is observed high"
    )


def test_posterior_sigma_is_symmetric():
    """Bayesian update must produce a symmetric posterior covariance matrix."""
    mu, Sigma = _diagonal_prior()
    W_lookup = {"TEST": np.full(K, 1.0 / K)}
    poll = PollObservation("TEST", dem_share=0.48, n_sample=800, geo_level="state")
    posterior = bayesian_poll_update(mu, Sigma, [poll], weight_lookup=W_lookup)
    assert np.allclose(posterior.sigma, posterior.sigma.T, atol=1e-12), (
        "Posterior covariance matrix is not symmetric"
    )


def test_posterior_sigma_is_positive_semidefinite():
    """Bayesian update must produce a PSD posterior covariance matrix."""
    mu, Sigma = _diagonal_prior()
    W_lookup = {"TEST": np.full(K, 1.0 / K)}
    poll = PollObservation("TEST", dem_share=0.52, n_sample=1200, geo_level="state")
    posterior = bayesian_poll_update(mu, Sigma, [poll], weight_lookup=W_lookup)
    eigenvalues = np.linalg.eigvalsh(posterior.sigma)
    assert (eigenvalues >= -1e-10).all(), (
        f"Posterior covariance has negative eigenvalues: {eigenvalues.min():.4e}"
    )


# ---------------------------------------------------------------------------
# load_weight_vector integration tests (reads real data files)
# ---------------------------------------------------------------------------


def test_load_weight_vector_state_fl():
    """load_weight_vector must return a (K,) array for state='FL'."""
    W = load_weight_vector("FL", "state")
    assert W.shape == (K,)


def test_load_weight_vector_state_ga():
    """load_weight_vector must return a (K,) array for state='GA'."""
    W = load_weight_vector("GA", "state")
    assert W.shape == (K,)


def test_load_weight_vector_state_al():
    """load_weight_vector must return a (K,) array for state='AL'."""
    W = load_weight_vector("AL", "state")
    assert W.shape == (K,)


def test_load_weight_vector_state_sums_to_one():
    """State weight vectors must sum to 1.0."""
    for state in ["FL", "GA", "AL"]:
        W = load_weight_vector(state, "state")
        assert abs(W.sum() - 1.0) < 1e-6, (
            f"State weight vector for {state} sums to {W.sum():.6f}, expected 1.0"
        )


def test_load_weight_vector_state_non_negative():
    """State weight vectors must have all non-negative entries."""
    for state in ["FL", "GA", "AL"]:
        W = load_weight_vector(state, "state")
        assert (W >= 0).all(), f"State {state} has negative community weights"


def test_load_weight_vector_county():
    """load_weight_vector must work for a known FL county (Miami-Dade = 12086)."""
    W = load_weight_vector("12086", "county")
    assert W.shape == (K,)
    assert abs(W.sum() - 1.0) < 1e-6


def test_load_weight_vector_invalid_state_raises():
    """load_weight_vector must raise ValueError for an unknown geography."""
    with pytest.raises(ValueError, match="not found"):
        load_weight_vector("ZZ", "state")


def test_load_weight_vector_invalid_geo_level_raises():
    """load_weight_vector must raise ValueError for unsupported geo_level."""
    with pytest.raises(ValueError, match="Unsupported geo_level"):
        load_weight_vector("FL", "district")


# ---------------------------------------------------------------------------
# load_polls integration tests (reads real polls_2026.csv)
# ---------------------------------------------------------------------------


def test_load_polls_returns_list():
    """load_polls must return a list (possibly empty) for a valid cycle."""
    polls = load_polls("2026")
    assert isinstance(polls, list)


def test_load_polls_2026_not_empty():
    """2026 poll CSV has data; load_polls should return at least one poll."""
    polls = load_polls("2026")
    assert len(polls) > 0, "No polls loaded for 2026 cycle"


def test_load_polls_returns_poll_observations():
    """All loaded polls must be PollObservation instances."""
    polls = load_polls("2026")
    for p in polls:
        assert isinstance(p, PollObservation), f"Expected PollObservation, got {type(p)}"


def test_load_polls_dem_share_in_range():
    """All loaded polls must have dem_share strictly in (0, 1)."""
    polls = load_polls("2026")
    for p in polls:
        assert 0 < p.dem_share < 1, (
            f"Poll {p!r} has dem_share={p.dem_share} outside (0, 1)"
        )


def test_load_polls_n_sample_positive():
    """All loaded polls must have positive sample size."""
    polls = load_polls("2026")
    for p in polls:
        assert p.n_sample > 0, f"Poll {p!r} has non-positive n_sample"


def test_load_polls_sorted_by_date():
    """load_polls should return polls sorted chronologically (ascending date)."""
    polls = load_polls("2026")
    dates = [p.date for p in polls]
    assert dates == sorted(dates), "Polls are not sorted by date"


def test_load_polls_race_filter():
    """race filter must restrict results to matching polls."""
    all_polls = load_polls("2026")
    fl_polls = load_polls("2026", race="FL Senate")
    assert len(fl_polls) <= len(all_polls)
    for p in fl_polls:
        assert "FL Senate".lower() in p.race.lower()


def test_load_polls_geography_filter():
    """geography filter must return only polls for the specified geography."""
    fl_polls = load_polls("2026", geography="FL")
    for p in fl_polls:
        assert p.geography == "FL"


def test_load_polls_after_filter():
    """after filter must exclude polls before the cutoff date."""
    cutoff = "2026-02-01"
    recent = load_polls("2026", after=cutoff)
    for p in recent:
        assert p.date >= cutoff, f"Poll {p!r} predates cutoff {cutoff}"


def test_list_races_returns_nonempty_list():
    """list_races must return at least one race for the 2026 cycle."""
    races = list_races("2026")
    assert isinstance(races, list)
    assert len(races) > 0


def test_list_races_returns_sorted_strings():
    """list_races must return a sorted list of non-empty strings."""
    races = list_races("2026")
    assert races == sorted(races), "Race list is not sorted"
    for r in races:
        assert isinstance(r, str) and r.strip(), "Race name should be a non-empty string"
