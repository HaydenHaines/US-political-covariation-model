"""Forecast engine: θ_prior → θ_national → δ_race → county predictions.

This module orchestrates the hierarchical poll decomposition model.
Voters move slowly (θ_prior from decade of elections); polls move quickly
(θ_national captures current sentiment; δ_race captures candidate effects).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.prediction.national_environment import estimate_theta_national
from src.prediction.candidate_effects import estimate_delta_race
from src.propagation.propagate_polls import PollObservation
from src.propagation.poll_weighting import apply_all_weights

if TYPE_CHECKING:
    import pandas as pd


def prepare_polls(
    polls_by_race: dict[str, list[dict]],
    reference_date: str,
    half_life_days: float = 30.0,
) -> dict[str, list[dict]]:
    """Apply quality weighting to raw poll dicts.

    Converts dicts → PollObservation → apply_all_weights → back to dicts.
    Returns polls with adjusted dem_share (house effects) and n_sample
    (time decay, pollster grade, pre-primary discount).
    """
    if not polls_by_race:
        return {}

    # Flatten all polls, keeping race labels and original notes
    all_obs: list[PollObservation] = []
    all_notes: list[str] = []
    race_labels: list[str] = []

    for race_id, polls in polls_by_race.items():
        for p in polls:
            obs = PollObservation(
                geography=p.get("state", ""),
                dem_share=p["dem_share"],
                n_sample=int(p["n_sample"]),
                race=race_id,
                date=p.get("date", ""),
                pollster=p.get("pollster", ""),
                geo_level=p.get("geo_level", "state"),
            )
            all_obs.append(obs)
            all_notes.append(p.get("notes", ""))
            race_labels.append(race_id)

    # Apply all quality adjustments (house effects, primary discount, time decay, grade)
    weighted = apply_all_weights(
        all_obs,
        reference_date=reference_date,
        half_life_days=half_life_days,
        poll_notes=all_notes,
    )

    # Reconstruct dicts grouped by race, preserving original notes
    result: dict[str, list[dict]] = {}
    for obs, notes, race_id in zip(weighted, all_notes, race_labels):
        d = {
            "dem_share": obs.dem_share,
            "n_sample": obs.n_sample,
            "state": obs.geography,
            "date": obs.date,
            "pollster": obs.pollster,
            "notes": notes,
        }
        result.setdefault(race_id, []).append(d)

    return result


def compute_theta_prior(
    type_scores: np.ndarray,  # (n_counties, J) — soft membership
    county_priors: np.ndarray,  # (n_counties,) — baseline Dem share
) -> np.ndarray:
    """Convert county-level priors to type-level θ_prior.

    θ_prior[j] = Σ_c W[c,j] · prior[c] / Σ_c W[c,j]
    Weighted average of county priors by type membership.
    """
    # Ensure non-negative weights (soft membership should already be non-negative)
    W = np.abs(type_scores)
    type_totals = W.sum(axis=0)  # (J,)
    # Avoid division by zero for types with no member counties
    type_totals = np.where(type_totals > 0, type_totals, 1.0)
    theta = (W.T @ county_priors) / type_totals  # (J,)
    return theta


@dataclass
class ForecastResult:
    """Result for a single race."""

    theta_prior: np.ndarray  # (J,)
    theta_national: np.ndarray  # (J,)
    delta_race: np.ndarray  # (J,)
    county_preds_national: np.ndarray  # (n_counties,) — θ_national mode
    county_preds_local: np.ndarray  # (n_counties,) — θ_national + δ mode
    n_polls: int


def build_W_state(
    state: str,
    type_scores: np.ndarray,  # (n_counties, J)
    states: list[str],
    county_votes: np.ndarray,  # (n_counties,)
) -> np.ndarray:
    """Build W vector for a state: vote-weighted mean of county type memberships."""
    mask = np.array([s == state for s in states])
    if not mask.any():
        J = type_scores.shape[1]
        return np.ones(J) / J  # Uniform fallback

    state_scores = np.abs(type_scores[mask])
    state_votes = county_votes[mask]

    if state_votes.sum() > 0:
        weights = state_votes / state_votes.sum()
        W = (state_scores * weights[:, np.newaxis]).sum(axis=0)
    else:
        W = state_scores.mean(axis=0)

    W_sum = W.sum()
    return W / W_sum if W_sum > 0 else np.ones(type_scores.shape[1]) / type_scores.shape[1]


def _build_poll_arrays(
    polls_by_race: dict[str, list[dict]],
    type_scores: np.ndarray,
    states: list[str],
    county_votes: np.ndarray,
    w_builder: callable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build W, y, sigma arrays from all polls across all races.

    When w_builder is provided, it is called for each poll to produce
    a poll-specific W vector (or list of observation dicts for Tier 2).
    When w_builder is None, falls back to build_W_state (current behavior).

    Returns: (W_all, y_all, sigma_all, race_labels)
    """
    W_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    sigma_vals: list[float] = []
    race_labels: list[str] = []

    for race_id, polls in polls_by_race.items():
        for p in polls:
            state = p["state"]
            dem_share = p["dem_share"]
            n_sample = p["n_sample"]

            if w_builder is not None:
                result = w_builder(p)
                if isinstance(result, list):
                    # Tier 2: multiple observations per poll (crosstab-expanded)
                    for obs in result:
                        W_rows.append(obs["W"])
                        y_vals.append(obs["y"])
                        sigma_vals.append(obs["sigma"])
                        race_labels.append(race_id)
                    continue
                else:
                    W_row = result
            else:
                W_row = build_W_state(state, type_scores, states, county_votes)

            sigma = np.sqrt(dem_share * (1 - dem_share) / max(n_sample, 1))

            W_rows.append(W_row)
            y_vals.append(dem_share)
            sigma_vals.append(max(sigma, 1e-6))  # Floor to avoid div-by-zero
            race_labels.append(race_id)

    J = type_scores.shape[1]
    if not W_rows:
        return np.empty((0, J)), np.empty(0), np.empty(0), []

    return (
        np.array(W_rows),
        np.array(y_vals),
        np.array(sigma_vals),
        race_labels,
    )


def run_forecast(
    type_scores: np.ndarray,  # (n_counties, J)
    county_priors: np.ndarray,  # (n_counties,)
    states: list[str],  # (n_counties,) state per county
    county_votes: np.ndarray,  # (n_counties,) votes per county
    polls_by_race: dict[str, list[dict]],  # race_id -> list of poll dicts
    races: list[str],  # all race IDs to forecast
    lam: float = 1.0,  # θ_national regularization
    mu: float = 1.0,  # δ_race regularization
    generic_ballot_shift: float = 0.0,
    w_vector_mode: str = "core",
    reference_date: str | None = None,
    type_profiles: pd.DataFrame | None = None,
) -> dict[str, ForecastResult]:
    """Run the full hierarchical forecast for all races.

    1. Compute θ_prior from county priors
    2. Apply poll quality weighting (if reference_date provided)
    3. Estimate θ_national from all polls pooled
    4. For each race, estimate δ_race from residuals
    5. Produce county predictions in both modes
    """
    J = type_scores.shape[1]

    # Apply generic ballot shift to county priors
    adjusted_priors = county_priors + generic_ballot_shift

    # Step 1: θ_prior
    theta_prior = compute_theta_prior(type_scores, adjusted_priors)

    # Step 1.5: Apply poll quality weighting
    working_polls = polls_by_race
    if reference_date:
        working_polls = prepare_polls(polls_by_race, reference_date)

    # Step 1.6: Build W vector builder if type_profiles available
    w_builder = None
    if type_profiles is not None:
        from src.prediction.poll_enrichment import build_W_poll

        # Precompute state-level type weights for W vector construction;
        # cache avoids redundant vote-weighted aggregation across polls in same state.
        state_type_weight_cache: dict[str, np.ndarray] = {}

        def _w_builder(poll: dict) -> np.ndarray | list[dict]:
            st = poll["state"]
            if st not in state_type_weight_cache:
                state_type_weight_cache[st] = build_W_state(
                    st, type_scores, states, county_votes,
                )
            return build_W_poll(
                poll=poll,
                type_profiles=type_profiles,
                state_type_weights=state_type_weight_cache[st],
                w_vector_mode=w_vector_mode,
            )

        w_builder = _w_builder

    # Step 2: Build poll arrays and estimate θ_national
    W_all, y_all, sigma_all, race_labels = _build_poll_arrays(
        working_polls, type_scores, states, county_votes,
        w_builder=w_builder,
    )
    theta_national = estimate_theta_national(W_all, y_all, sigma_all, theta_prior, lam)

    # Step 3 & 4: Per-race δ and predictions
    results: dict[str, ForecastResult] = {}
    for race_id in races:
        race_polls = working_polls.get(race_id, [])
        n_polls = len(race_polls)

        if n_polls > 0:
            race_W, race_y, race_sigma, _ = _build_poll_arrays(
                {race_id: race_polls}, type_scores, states, county_votes,
                w_builder=w_builder,
            )
            residuals = race_y - race_W @ theta_national
            delta = estimate_delta_race(race_W, residuals, race_sigma, J, mu)
        else:
            delta = np.zeros(J)

        county_preds_national = type_scores @ theta_national
        county_preds_local = type_scores @ (theta_national + delta)

        results[race_id] = ForecastResult(
            theta_prior=theta_prior,
            theta_national=theta_national,
            delta_race=delta,
            county_preds_national=county_preds_national,
            county_preds_local=county_preds_local,
            n_polls=n_polls,
        )

    return results
