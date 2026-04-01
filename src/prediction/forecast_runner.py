"""Single-race forecast: Bayesian poll update through type structure.

This module provides predict_race() — the core mathematical operation of the
type-primary forecast pipeline. Given type scores, covariance, priors, and
polls, it produces county-level predictions with credible intervals.

Design:
  - County-level priors provide each county's own historical baseline.
  - Types determine comovement only (how polls shift predictions via covariance).
  - Multiple polls are stacked into one Bayesian update, preserving geographic
    information when polls cover different states.
  - When county_priors is None, falls back to type-mean weighted predictions
    (legacy path; present for backwards compatibility).

See predict_2026_types.py for the full pipeline orchestration.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as _cfg

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# State FIPS -> abbreviation (all 50 states + DC, sourced from config/model.yaml)
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR


def predict_race(
    race: str,
    type_scores: np.ndarray,
    type_covariance: np.ndarray,
    type_priors: np.ndarray,
    county_fips: list[str],
    polls: list[
        tuple[float, int, str] | tuple[float, int, str, np.ndarray | None]
    ]
    | None = None,
    states: list[str] | None = None,
    county_names: list[str] | None = None,
    state_filter: str | None = None,
    county_priors: np.ndarray | None = None,
    prior_weight: float = 1.0,
    generic_ballot_shift: float = 0.0,
) -> pd.DataFrame:
    """Produce county-level predictions from type structure.

    Uses county-level priors (each county's own historical baseline) when
    provided. Types determine only the covariance structure (how poll
    observations shift predictions). The prediction formula is:

        county_pred = county_prior + type_covariance_adjustment

    where the adjustment comes from the Bayesian update on type means, and
    each county's adjustment is the score-weighted shift in type means.

    When county_priors is None, falls back to type-mean weighted predictions (legacy).

    Multiple polls are stacked into a single Bayesian update: each poll
    contributes one row to the W matrix (n_polls × J), enabling exact
    multi-poll inference rather than collapsing to a single effective poll.
    This preserves geographic information when polls cover different states.

    Parameters
    ----------
    race : str
        Election race label (e.g. "FL Senate").
    type_scores : ndarray of shape (N, J)
        County type scores (soft membership, can be negative).
    type_covariance : ndarray of shape (J, J)
        Type covariance matrix.
    type_priors : ndarray of shape (J,)
        Prior Dem share per type (used for Bayesian update baseline).
    county_fips : list[str]
        FIPS codes for each county (length N).
    polls : list of (dem_share, n, state_abbr) tuples or None
        Poll observations. Each tuple is one poll: Democratic two-party
        share (0-1), sample size, and the state abbreviation whose type
        composition defines the observation equation (W row). Multiple
        polls are stacked into a single multi-row Bayesian update.
        None = use prior only (no poll adjustment).
    states : list[str] or None
        State abbreviation per county. Derived from FIPS if None.
    county_names : list[str] or None
        County names. Set to empty string if None.
    state_filter : str or None
        If provided, filter output rows to this state abbreviation.
        Does NOT affect which polls are applied — poll state comes
        from each tuple in `polls`.
    county_priors : ndarray of shape (N,) or None
        Per-county prior Dem share (historical baseline). When provided,
        predictions use county baselines + type covariance adjustments.
        When None, falls back to type-mean weighted predictions (legacy).

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, state, county_name, pred_dem_share,
        ci_lower, ci_upper, dominant_type, super_type
    """
    N, J = type_scores.shape
    assert len(county_fips) == N
    assert type_covariance.shape == (J, J)
    assert len(type_priors) == J
    if county_priors is not None:
        assert len(county_priors) == N

    # Apply generic ballot shift to county priors before prediction
    if generic_ballot_shift != 0.0 and county_priors is not None:
        from src.prediction.generic_ballot import apply_gb_shift

        county_priors = apply_gb_shift(county_priors, generic_ballot_shift)

    # Derive states from FIPS if not provided
    if states is None:
        states = [_STATE_FIPS_TO_ABBR.get(f[:2], "??") for f in county_fips]
    if county_names is None:
        county_names = [""] * N

    # Apply prior_weight scaling to type covariance before any Bayesian update.
    # Lower weight = less informative prior = polls pull predictions further from baseline.
    # At pw=0 the user wants "trust only polls" — inflate covariance enormously
    # so the Bayesian update posterior collapses onto the poll likelihood.
    type_cov = _scale_prior_covariance(type_covariance, prior_weight)

    # Run Bayesian update if polls are provided, otherwise keep the prior
    type_means = type_priors.copy().astype(float)
    if polls:
        type_means, type_cov = _apply_poll_updates(
            polls, type_means, type_cov, type_scores, states, J
        )

    # Map updated type estimates back to county-level predictions
    pred_dem_share = _compute_county_predictions(
        type_scores, type_means, type_priors, county_priors, prior_weight
    )

    # Compute 90% credible intervals from type covariance diagonal
    ci_lower, ci_upper = _compute_credible_intervals(pred_dem_share, type_scores, type_cov)

    result = _assemble_result(
        county_fips, states, county_names, pred_dem_share, ci_lower, ci_upper, type_scores
    )

    if state_filter is not None:
        result = result[result["state"] == state_filter].reset_index(drop=True)

    return result


def _scale_prior_covariance(type_covariance: np.ndarray, prior_weight: float) -> np.ndarray:
    """Scale the type covariance matrix to reflect prior_weight.

    prior_weight=1.0 → no change (default, full Ridge model trust).
    prior_weight<1.0 → divide by weight, making prior less informative.
    prior_weight=0.0 → multiply by 1e6, collapsing posterior onto poll likelihood.
    """
    type_cov = type_covariance.copy().astype(float)
    if prior_weight == 0.0:
        type_cov = type_cov * 1e6
    elif prior_weight != 1.0:
        type_cov = type_cov / prior_weight
    return type_cov


def _build_poll_W_row(
    poll_tuple: tuple,
    type_scores: np.ndarray,
    states: list[str],
    J: int,
) -> np.ndarray:
    """Build the observation equation row W for a single poll.

    W is the type composition of the polled geography. This tells the Bayesian
    update which combination of type means the poll is observing.

    Three cases:
    - Explicit W override (crosstab-derived): use directly, normalize.
    - Poll state known: average absolute type scores across that state's counties.
    - No state info: uniform weight across all types (fallback).
    """
    w_override = poll_tuple[3] if len(poll_tuple) > 3 else None
    poll_state = poll_tuple[2]

    if w_override is not None:
        # Crosstab-derived W vector: normalize and use directly
        w_sum = float(w_override.sum())
        return w_override / w_sum if w_sum > 0 else np.ones(J) / J

    if poll_state:
        state_mask = np.array([s == poll_state for s in states])
        if state_mask.any():
            state_scores = type_scores[state_mask]
            W_row = np.abs(state_scores).mean(axis=0)
            W_sum = W_row.sum()
            return W_row / W_sum if W_sum > 0 else np.ones(J) / J

    # Fallback: uniform type weights
    return np.ones(J) / J


def _apply_poll_updates(
    polls: list,
    type_means: np.ndarray,
    type_cov: np.ndarray,
    type_scores: np.ndarray,
    states: list[str],
    J: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack all polls into a single Bayesian update and return posterior.

    Each poll contributes one W row (the type composition of its state).
    Stacking all rows into a single update preserves geographic information
    across polls covering different states, vs. collapsing to one scalar.
    """
    W_rows = []
    y_vals = []
    sigma_vals = []

    for poll_tuple in polls:
        dem_share = poll_tuple[0]
        n = poll_tuple[1]
        W_row = _build_poll_W_row(poll_tuple, type_scores, states, J)
        W_rows.append(W_row)
        y_vals.append(dem_share)
        sigma_vals.append(np.sqrt(dem_share * (1 - dem_share) / n))

    return _bayesian_update(
        mu_prior=type_means,
        sigma_prior=type_cov,
        W=np.array(W_rows),
        y=np.array(y_vals),
        sigma_polls=np.array(sigma_vals),
    )


def _compute_county_predictions(
    type_scores: np.ndarray,
    type_means: np.ndarray,
    type_priors: np.ndarray,
    county_priors: np.ndarray | None,
    prior_weight: float,
) -> np.ndarray:
    """Map type-level estimates to county-level Dem share predictions.

    Two modes:
    - County-prior mode: county baseline + score-weighted type shift. The
      prior_weight slider blends between the Ridge county baseline (pw=1) and
      the type-mean baseline (pw=0), controlling how much historical county
      specificity is preserved vs. type-average behavior.
    - Legacy mode (county_priors is None): pure type-mean weighted prediction.
    """
    abs_scores = np.abs(type_scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)  # avoid div by zero

    if county_priors is not None:
        # 1. Compute the type-level shift from the Bayesian update
        type_shift = type_means - type_priors.astype(float)
        # 2. Each county's adjustment = score-weighted average of type shifts
        county_adjustment = (abs_scores * type_shift[None, :]).sum(axis=1) / weight_sums
        # 3. Blend county priors toward type-weighted baseline using prior_weight.
        #    pw=1.0 → use county_priors (Ridge model); pw=0.0 → use type-mean baseline.
        type_prior_baseline = (
            (abs_scores * type_priors.astype(float)[None, :]).sum(axis=1) / weight_sums
        )
        effective_priors = (
            prior_weight * county_priors.astype(float)
            + (1 - prior_weight) * type_prior_baseline
        )
        # 4. Final prediction = blended baseline + adjustment
        pred_dem_share = effective_priors + county_adjustment
    else:
        # Legacy: type-mean weighted predictions
        pred_dem_share = (abs_scores * type_means[None, :]).sum(axis=1) / weight_sums

    return np.clip(pred_dem_share, 0.0, 1.0)


def _compute_credible_intervals(
    pred_dem_share: np.ndarray,
    type_scores: np.ndarray,
    type_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 90% credible intervals from the posterior type covariance.

    County-level standard deviation is the score-weighted average of the
    per-type posterior standard deviations (diagonal of Sigma_post).
    """
    abs_scores = np.abs(type_scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    type_std = np.sqrt(np.diag(type_cov))
    county_std = (abs_scores * type_std[None, :]).sum(axis=1) / weight_sums

    ci_lower = np.clip(pred_dem_share - 1.645 * county_std, 0.0, 1.0)
    ci_upper = np.clip(pred_dem_share + 1.645 * county_std, 0.0, 1.0)
    return ci_lower, ci_upper


def _assemble_result(
    county_fips: list[str],
    states: list[str],
    county_names: list[str],
    pred_dem_share: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    type_scores: np.ndarray,
) -> pd.DataFrame:
    """Pack all county-level arrays into the output DataFrame.

    super_type is set to -1 here; callers should join against
    type_assignments.parquet (which has the authoritative super_type from
    nest_types) to populate this column.
    """
    N = len(county_fips)
    dominant_type = np.argmax(np.abs(type_scores), axis=1)
    super_type = np.full(N, -1, dtype=int)

    return pd.DataFrame({
        "county_fips": county_fips,
        "state": states,
        "county_name": county_names,
        "pred_dem_share": pred_dem_share,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "dominant_type": dominant_type,
        "super_type": super_type,
    })


def _bayesian_update(
    mu_prior: np.ndarray,
    sigma_prior: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    sigma_polls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian Bayesian update: posterior mean and covariance.

    Same mathematical formulation as predict_2026_hac.bayesian_update.
    """
    R = np.diag(sigma_polls ** 2)
    sigma_prior_inv = np.linalg.inv(
        sigma_prior + np.eye(len(mu_prior)) * 1e-8
    )
    sigma_post_inv = sigma_prior_inv + W.T @ np.linalg.inv(R) @ W
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (
        sigma_prior_inv @ mu_prior + W.T @ np.linalg.solve(R, y)
    )
    return mu_post, sigma_post
