"""Forecast engine: θ_prior → θ_national → δ_race → county predictions.

This module orchestrates the hierarchical poll decomposition model.
Voters move slowly (θ_prior from decade of elections); polls move quickly
(θ_national captures current sentiment; δ_race captures candidate effects).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.prediction.candidate_effects import estimate_delta_race
from src.prediction.national_environment import estimate_theta_national
from src.propagation.poll_weighting import apply_all_weights
from src.propagation.propagate_polls import PollObservation

if TYPE_CHECKING:
    import pandas as pd


def prepare_polls(
    polls_by_race: dict[str, list[dict]],
    reference_date: str,
    half_life_days: float = 30.0,
    pre_primary_discount: float = 0.5,
    accuracy_path: Path | None = None,
    methodology_weights: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    """Apply quality weighting to raw poll dicts.

    Converts dicts → PollObservation → apply_all_weights → back to dicts.
    Returns polls with adjusted dem_share (house effects) and n_sample
    (time decay, pollster grade, pre-primary discount, methodology quality).

    Parameters
    ----------
    half_life_days:
        Exponential decay half-life.  Comes from prediction_params.json
        ``poll_weighting.half_life_days``; defaults to 30.0.
    pre_primary_discount:
        Multiplicative n_sample factor for pre-primary polls.  Comes from
        prediction_params.json ``poll_weighting.pre_primary_discount``; defaults to 0.5.
    accuracy_path:
        Optional path to pollster_accuracy.json.  When provided, RMSE-based
        quality weights are used in place of grade-based weights for any
        pollster that appears in the accuracy data.
    methodology_weights:
        Optional mapping of methodology strings (e.g. "phone", "online") to
        quality multipliers.  When provided, each poll's n_sample is multiplied
        by the factor for its methodology tag (from the "methodology" key in the
        poll dict).  When None, methodology weighting is skipped.  Pass
        ``_DEFAULT_METHODOLOGY_WEIGHTS`` from ``poll_methodology`` to use the
        built-in defaults loaded from prediction_params.json.
    """
    if not polls_by_race:
        return {}

    # Flatten all polls, keeping race labels, original notes, and methodology tags
    all_obs: list[PollObservation] = []
    all_notes: list[str] = []
    all_methodologies: list[str | None] = []
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
            # Extract methodology tag; None = missing (treated as neutral)
            raw_method = p.get("methodology", None)
            all_methodologies.append(raw_method if raw_method else None)
            race_labels.append(race_id)

    # Apply all quality adjustments:
    # house effects, primary discount, time decay, grade/RMSE, methodology.
    # Pass poll_methodologies=None when methodology_weights is None to skip
    # the methodology step entirely (caller opted out by omitting weights).
    effective_methodologies = all_methodologies if methodology_weights is not None else None
    weighted = apply_all_weights(
        all_obs,
        reference_date=reference_date,
        half_life_days=half_life_days,
        poll_notes=all_notes,
        primary_discount_factor=pre_primary_discount,
        accuracy_path=accuracy_path,
        poll_methodologies=effective_methodologies,
        methodology_weights=methodology_weights,
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


def _extract_crosstabs_from_xt(poll: dict) -> list[dict] | None:
    """Extract xt_* fields from a poll dict and convert to crosstab observation dicts.

    Each xt_ field encodes the fraction of the poll sample in a demographic group
    (e.g. ``xt_race_black=0.13`` means 13% of respondents identify as Black).
    When a matching ``xt_vote_<group>_<value>`` column is present (parsed from the
    Emerson crosstab second tab), we use that as the per-group dem_share observation.
    Otherwise we fall back to the poll's topline dem_share as a conservative proxy.

    This transforms Tier 2 from "same y, different W" (topline proxy) to
    "different y AND different W" (per-group vote share + demographic W vector)
    whenever crosstab data is available — the highest-ROI improvement to poll
    signal extraction short of per-precinct vote allocation.

    The resulting crosstab list feeds ``build_W_from_crosstabs()``, which constructs
    a demographic-specific W vector per group.  Each W is weighted by the type-profile
    column for that demographic, concentrating signal on types that demographically
    resemble the polled group.

    Returns None when no xt_ data is present so the caller falls through to Tier 1/3.
    """
    dem_share = poll.get("dem_share")
    n_sample = poll.get("n_sample", 600)

    if dem_share is None:
        return None

    crosstabs: list[dict] = []
    for key, value in poll.items():
        if not key.startswith("xt_"):
            continue
        # xt_vote_* columns contain per-group vote shares (e.g. xt_vote_race_black
        # = 0.80 means 80% of Black respondents chose Dem).  These are referenced
        # by the vote_key lookup below — they are NOT sample composition columns
        # and must not be parsed as pct_of_sample.
        if key.startswith("xt_vote_"):
            continue
        # Column naming convention: xt_<group>_<value>
        # e.g. xt_race_black → group="race", value="black"
        remainder = key[3:]  # strip "xt_" prefix
        parts = remainder.split("_", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        try:
            pct = float(value)
        except (TypeError, ValueError):
            continue
        # Skip missing/NaN values (float('nan') != float('nan'))
        if pct != pct or pct <= 0:
            continue
        # Check whether we have a per-group vote share for this demographic.
        # Per-group shares live in xt_vote_<group>_<value> columns (e.g.
        # xt_vote_race_black = 0.80 means 80% of Black respondents chose Dem).
        # These come from the Emerson crosstab second tab and are much more
        # informative than the topline — they give both a different W vector
        # (routing to matching community types) AND a different y observation
        # (the actual vote share for that group, not just the poll-wide average).
        vote_key = f"xt_vote_{parts[0]}_{parts[1]}"
        group_dem_share_raw = poll.get(vote_key)
        group_dem_share: float | None = None
        if group_dem_share_raw is not None:
            try:
                parsed = float(group_dem_share_raw)
                # Guard against NaN / nonsensical values
                if parsed == parsed and 0.0 <= parsed <= 1.0:
                    group_dem_share = parsed
            except (TypeError, ValueError):
                pass

        crosstabs.append({
            "demographic_group": parts[0],
            "group_value": parts[1],
            "pct_of_sample": pct,
            # Use per-group vote share when available (from crosstab second tab),
            # otherwise fall back to the topline dem_share as a conservative proxy.
            # Per-group shares make Tier 2 much more informative: they inject
            # genuine cross-group variation into the y observations, not just
            # different W vectors with the same y.
            "dem_share": group_dem_share if group_dem_share is not None else float(dem_share),
        })

    return crosstabs if crosstabs else None


def _extract_raw_demographics(poll: dict) -> dict[str, float] | None:
    """Extract xt_* fields from a poll dict and map them to type_profiles column names.

    Polls loaded from polls_2026.csv carry xt_* keys (e.g. ``xt_education_college``)
    when crosstab data is available.  The Tier 1 W builder (``build_W_from_raw_sample``)
    expects keys that match ``type_profiles`` column names rather than xt_ keys, so we
    translate them here using the same dimension map that the crosstab W builder uses.

    Returns None when no xt_ data is present so that callers can fall back to Tier 3.
    """
    # Lazy import avoids a circular dependency: forecast_engine → poll_enrichment,
    # but crosstab_w_builder is in propagation so there is no cycle.
    from src.propagation.crosstab_w_builder import CROSSTAB_DIMENSION_MAP

    raw: dict[str, float] = {}
    for key, value in poll.items():
        if not key.startswith("xt_"):
            continue
        # xt_education_college → education_college (strip the "xt_" prefix)
        dim_key = key[3:]  # e.g. "education_college"
        col = CROSSTAB_DIMENSION_MAP.get(dim_key)
        # Inverted dimensions (noncollege, rural) map to None in the dimension
        # map because they are derived from their parent.  Skip them here —
        # build_W_from_raw_sample computes similarity directly from the columns
        # present, so including the parent (college) is sufficient.
        if col is not None:
            raw[col] = float(value)

    return raw if raw else None


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
    half_life_days: float = 30.0,  # poll time-decay half-life; see prediction_params.json
    pre_primary_discount: float = 0.5,  # n_sample factor for pre-primary polls
    accuracy_path: Path | None = None,  # path to pollster_accuracy.json for RMSE weights
    methodology_weights: dict[str, float] | None = None,  # see prediction_params.json
    state_population_vectors: dict[str, dict[str, float]] | None = None,
    # ^^ Optional: maps state_abbr → {xt_col: population_share}.
    # When provided, Tier 2 crosstab observations are post-stratification corrected
    # so that oversampled demographic groups don't get artificially low sigma.
    # Precompute with src.prediction.population_vectors.build_state_population_vectors.
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

    # Step 1.5: Apply poll quality weighting.
    # Parameters come from prediction_params.json via the caller
    # (predict_2026_types.run).  Function defaults serve as fallbacks
    # when called from tests or other contexts that don't supply config.
    working_polls = polls_by_race
    if reference_date:
        working_polls = prepare_polls(
            polls_by_race,
            reference_date,
            half_life_days=half_life_days,
            pre_primary_discount=pre_primary_discount,
            accuracy_path=accuracy_path,
            methodology_weights=methodology_weights,
        )

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
            # Tier 2 (highest priority): extract xt_* fields as per-group
            # crosstab dicts.  Each group becomes a separate Bayesian observation
            # with a demographic-specific W vector.  When xt_ data is absent
            # this returns None and we fall through to Tier 1.
            crosstabs = _extract_crosstabs_from_xt(poll)
            # Tier 1 fallback: map xt_* keys to type_profiles column names for
            # a single W vector shift.  Returns None when no xt_ data is present,
            # signalling Tier 3 (methodology-only adjustments).
            raw_demographics = _extract_raw_demographics(poll) if crosstabs is None else None
            # Look up population shares for this poll's state so that Tier 2
            # crosstab observations can apply post-stratification correction.
            # If state_population_vectors is not available, pass None and the
            # correction is skipped (preserving original behavior).
            pop_shares = (
                state_population_vectors.get(st) if state_population_vectors is not None else None
            )
            return build_W_poll(
                poll=poll,
                type_profiles=type_profiles,
                state_type_weights=state_type_weight_cache[st],
                poll_crosstabs=crosstabs,
                raw_sample_demographics=raw_demographics,
                w_vector_mode=w_vector_mode,
                population_shares=pop_shares,
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
