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

    core_keys = {
        "dem_share",
        "n_sample",
        "state",
        "date",
        "pollster",
        "notes",
        "geo_level",
    }

    # Flatten all polls, keeping race labels, original notes, methodology tags,
    # and enrichment metadata that downstream W builders consume.
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
                metadata={k: v for k, v in p.items() if k not in core_keys},
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

    # Reconstruct dicts grouped by race, preserving original enrichment metadata
    # while replacing topline fields with weighted values.
    result: dict[str, list[dict]] = {}
    for obs, notes, race_id in zip(weighted, all_notes, race_labels):
        d = dict(obs.metadata)
        d.update({
            "dem_share": obs.dem_share,
            "n_sample": obs.n_sample,
            "state": obs.geography,
            "date": obs.date,
            "pollster": obs.pollster,
            "notes": notes,
            "geo_level": obs.geo_level,
        })
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
    generic_ballot_shift: float | np.ndarray = 0.0,
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
    poll_blend_scale: float = 5.0,
    # ^^ The k parameter in alpha = 1/(1 + n_polls/k).  Controls how quickly
    # the model transitions from trusting county priors (few polls) to trusting
    # type-projected predictions (many polls).  k=5: at 5 polls, 50/50 blend.
    race_adjustments: dict[str, dict] | None = None,
    # ^^ Per-race prior overrides.  Maps race_id -> {"prior_dem_share_override": float}.
    # When set, the state-mean of county priors is shifted to the target value
    # before the Bayesian update.  Used for structural factors the model can't
    # capture (RCV dynamics, unusual candidate effects).
    ctov_adjustments: dict[str, np.ndarray] | None = None,
    # ^^ Per-race type-level CTOV adjustments from candidate sabermetrics.
    # Maps race_id -> ndarray of shape (J,).  Applied as:
    #   adjusted_prior[c] += type_scores[c] @ ctov  for c in state
    # Unlike race_adjustments (uniform shift), CTOV shifts counties differently
    # based on their type composition.  Loaded by candidate_ctov.load_ctov_adjustments().
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

        # Apply per-race prior override if configured (e.g., RCV states).
        # Shifts all county priors in the race's state so the vote-weighted
        # state mean matches the target.  Preserves relative county structure.
        race_priors = adjusted_priors
        if race_adjustments:
            adj = race_adjustments.get(race_id)
            if adj and "prior_dem_share_override" in adj:
                target = adj["prior_dem_share_override"]
                # Extract the state from the race_id (e.g., "2026 AK Senate" -> "AK")
                race_parts = race_id.split()
                state_abbr = race_parts[1] if len(race_parts) >= 3 else None
                if state_abbr:
                    state_mask = np.array([s == state_abbr for s in states])
                    if state_mask.any():
                        # Vote-weighted state mean of current priors
                        state_votes_masked = county_votes[state_mask]
                        vote_total = state_votes_masked.sum()
                        if vote_total > 0:
                            current_mean = (
                                adjusted_priors[state_mask] * state_votes_masked
                            ).sum() / vote_total
                        else:
                            current_mean = adjusted_priors[state_mask].mean()
                        shift = target - current_mean
                        # Apply uniform shift to this state's counties only
                        race_priors = adjusted_priors.copy()
                        race_priors[state_mask] += shift

        # Apply type-level CTOV adjustment from candidate sabermetrics.
        # Unlike race_adjustments (uniform shift), CTOV shifts each county
        # differently based on its type composition — Graham's evangelical
        # overperformance shifts rural evangelical counties more than urban ones.
        # Scale factor (0.3) and cap (±5pp) prevent extreme values from
        # dominating — a county with 93% in one type shouldn't shift 23pp.
        if ctov_adjustments:
            ctov_vec = ctov_adjustments.get(race_id)
            if ctov_vec is not None:
                race_parts = race_id.split()
                state_abbr = race_parts[1] if len(race_parts) >= 3 else None
                if state_abbr:
                    state_mask_ctov = np.array([s == state_abbr for s in states])
                    if state_mask_ctov.any():
                        from src.prediction.candidate_ctov import (
                            CTOV_MAX_SHIFT,
                            CTOV_SCALE,
                        )

                        if race_priors is adjusted_priors:
                            race_priors = adjusted_priors.copy()
                        raw_shift = type_scores[state_mask_ctov] @ ctov_vec
                        capped_shift = np.clip(
                            raw_shift * CTOV_SCALE, -CTOV_MAX_SHIFT, CTOV_MAX_SHIFT,
                        )
                        race_priors[state_mask_ctov] += capped_shift

        if n_polls > 0:
            race_W, race_y, race_sigma, _ = _build_poll_arrays(
                {race_id: race_polls}, type_scores, states, county_votes,
                w_builder=w_builder,
            )
            residuals = race_y - race_W @ theta_national
            delta = estimate_delta_race(race_W, residuals, race_sigma, J, mu)
        else:
            delta = np.zeros(J)

        # County-level residual blending with poll-count adaptive weight.
        #
        # Problem: purely type-based projection (type_scores @ theta)
        # compresses all counties to type means, making NJ R+1 despite
        # being D+16.  But using county priors alone ignores poll signal,
        # breaking well-polled races like GA.
        #
        # Solution: blend county priors with type projection, weighting
        # by how many polls inform the race.  With zero polls, trust
        # county priors fully (NJ gets its D+16 lean).  With many polls,
        # trust the type projection (GA matches its D+2.6 polling).
        #
        # alpha = 1 / (1 + n_polls / k), where k controls the transition.
        # k=5: at 5 polls, weight is 50/50.  At 15 polls, 75% type model.
        # poll_blend_scale is passed as a parameter to run_forecast() for tuning.
        alpha = 1.0 / (1.0 + n_polls / poll_blend_scale)
        type_proj_national = type_scores @ theta_national
        type_proj_local = type_scores @ (theta_national + delta)
        county_preds_national = alpha * race_priors + (1 - alpha) * type_proj_national
        county_preds_local = alpha * race_priors + (1 - alpha) * type_proj_local

        results[race_id] = ForecastResult(
            theta_prior=theta_prior,
            theta_national=theta_national,
            delta_race=delta,
            county_preds_national=county_preds_national,
            county_preds_local=county_preds_local,
            n_polls=n_polls,
        )

    return results


def _xt_delta_from_polls(
    polls_by_race: dict[str, list[dict]],
    type_scores: np.ndarray,
    county_priors: np.ndarray,
    states: list[str],
    county_votes: np.ndarray,
    all_race_ids: list[str],
    **run_forecast_kwargs,
) -> dict:
    """Compute enriched-vs-stripped xt_ impact from pre-loaded data.

    Called by make_xt_impact_report(). Split out for testability.
    Does NOT write to disk or mutate global state.

    Parameters
    ----------
    polls_by_race:
        Poll dicts keyed by race ID.  May contain xt_* fields.
    run_forecast_kwargs:
        Any kwargs forwarded to run_forecast() — e.g. reference_date,
        type_profiles, lam, mu, generic_ballot_shift.  reference_date
        is also used as the report_date in the returned dict.
    """
    from datetime import date as _date

    polls_stripped = {
        race_id: [
            {k: v for k, v in p.items() if not k.startswith("xt_") and k != "methodology"}
            for p in polls
        ]
        for race_id, polls in polls_by_race.items()
    }

    common = dict(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes,
        races=all_race_ids,
        **run_forecast_kwargs,
    )
    results_enriched = run_forecast(polls_by_race=polls_by_race, **common)
    results_stripped = run_forecast(polls_by_race=polls_stripped, **common)

    xt_race_counts = {
        race_id: sum(1 for p in polls if any(k.startswith("xt_") for k in p))
        for race_id, polls in polls_by_race.items()
    }

    states_arr = np.array(states)

    def _state_pred(result: ForecastResult, state_abbr: str) -> float | None:
        mask = states_arr == state_abbr
        if not mask.any():
            return None
        votes = county_votes[mask]
        if votes.sum() <= 0:
            return float(np.mean(result.county_preds_local[mask]))
        return float(np.average(result.county_preds_local[mask], weights=votes))

    enriched_deltas: dict[str, float] = {}
    for race_id in sorted(results_enriched):
        if race_id not in results_stripped:
            continue
        parts = race_id.split()
        state = parts[1] if len(parts) > 1 else None
        if state is None:
            continue
        p_e = _state_pred(results_enriched[race_id], state)
        p_s = _state_pred(results_stripped[race_id], state)
        if p_e is not None and p_s is not None:
            enriched_deltas[race_id] = (p_e - p_s) * 100.0

    xt_deltas = [
        abs(enriched_deltas[r])
        for r in enriched_deltas
        if xt_race_counts.get(r, 0) > 0
    ]
    report_date = run_forecast_kwargs.get("reference_date") or str(_date.today())

    return {
        "enriched_deltas": enriched_deltas,
        "mean_delta": float(np.mean(xt_deltas)) if xt_deltas else 0.0,
        "max_delta": float(np.max(xt_deltas)) if xt_deltas else 0.0,
        "races_with_xt": sum(1 for c in xt_race_counts.values() if c > 0),
        "report_date": report_date,
    }


def make_xt_impact_report(races: list[str] | None = None) -> dict:
    """Run enriched vs. xt_-stripped forecast comparison and return a summary dict.

    Converts the core comparison logic from
    scripts/experiments/compare_xt_impact_v2.py into a reusable function.
    Does NOT write to disk or mutate global state.

    Parameters
    ----------
    races:
        Optional list of race IDs to include.  When None, all state-level
        races found in polls_2026.csv are included.

    Returns
    -------
    dict with keys:
        enriched_deltas   {race_id: float}  enriched minus stripped (pp)
        mean_delta        float  mean |delta| over races with xt_ polls
        max_delta         float  max |delta| over races with xt_ polls
        races_with_xt     int    count of races with at least one xt_ poll
        report_date       str    ISO date of the run
    """
    import json
    from datetime import date as _date
    from pathlib import Path as _Path

    import pandas as _pd

    from src.core import config as _cfg
    from src.assembly.define_races import load_races
    from src.prediction.county_priors import load_county_priors_with_ridge
    from src.prediction.generic_ballot import compute_gb_shift

    project_root = _Path(__file__).resolve().parents[2]

    ta_df = _pd.read_parquet(project_root / "data" / "communities" / "type_assignments.parquet")
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores_loaded = ta_df[score_cols].values
    states_list = [_cfg.STATE_ABBR.get(f[:2], "??") for f in county_fips]

    county_votes = np.ones(len(county_fips))
    votes_path = project_root / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    if votes_path.exists():
        vdf = _pd.read_parquet(votes_path)
        votes_col = "pres_total_2024" if "pres_total_2024" in vdf.columns else "totalvotes"
        if "county_fips" in vdf.columns and votes_col in vdf.columns:
            vmap = dict(zip(
                vdf["county_fips"].astype(str).str.zfill(5),
                vdf[votes_col],
            ))
            county_votes = np.array([float(vmap.get(f, 1.0)) for f in county_fips])

    county_priors_loaded = load_county_priors_with_ridge(county_fips)

    tp_path = project_root / "data" / "communities" / "type_profiles.parquet"
    type_profiles = _pd.read_parquet(tp_path) if tp_path.exists() else None

    polls_path = project_root / "data" / "polls" / "polls_2026.csv"
    polls_df = _pd.read_csv(polls_path)
    if "geography" in polls_df.columns and "state" not in polls_df.columns:
        polls_df = polls_df.rename(columns={"geography": "state"})
    if "geo_level" in polls_df.columns:
        polls_df = polls_df[polls_df["geo_level"] == "state"].copy()

    xt_cols = [c for c in polls_df.columns if c.startswith("xt_")]

    polls_by_race: dict[str, list[dict]] = {}
    for race_id, grp in polls_df.groupby("race"):
        if str(race_id).startswith("2026 Generic Ballot"):
            continue
        if races is not None and str(race_id) not in races:
            continue
        race_dicts = []
        for _, row in grp.iterrows():
            d: dict = {
                "dem_share": float(row["dem_share"]),
                "n_sample": int(row["n_sample"]) if _pd.notna(row["n_sample"]) else 600,
                "state": str(row["state"]),
                "date": str(row["date"]) if _pd.notna(row.get("date")) else "",
                "pollster": str(row["pollster"]) if _pd.notna(row.get("pollster")) else "",
                "notes": str(row["notes"]) if _pd.notna(row.get("notes")) else "",
            }
            method = row.get("methodology")
            if method is not None and _pd.notna(method):
                d["methodology"] = str(method)
            for col in xt_cols:
                val = row.get(col)
                if val is not None and _pd.notna(val):
                    d[col] = float(val)
            race_dicts.append(d)
        if race_dicts:
            polls_by_race[str(race_id)] = race_dicts

    params_path = project_root / "data" / "config" / "prediction_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    fc = params.get("forecast", {})
    pw = params.get("poll_weighting", {})

    registry = load_races(2026)
    all_race_ids = [r.race_id for r in registry]
    if races is not None:
        all_race_ids = [r for r in all_race_ids if r in races]

    gb_shift = compute_gb_shift().shift
    accuracy_path = project_root / "data" / "config" / "pollster_accuracy.json"

    return _xt_delta_from_polls(
        polls_by_race=polls_by_race,
        type_scores=type_scores_loaded,
        county_priors=county_priors_loaded,
        states=states_list,
        county_votes=county_votes,
        all_race_ids=all_race_ids,
        reference_date=str(_date.today()),
        lam=fc.get("lam", 1.0),
        mu=fc.get("mu", 1.0),
        generic_ballot_shift=gb_shift,
        w_vector_mode=fc.get("w_vector_mode", "core"),
        type_profiles=type_profiles,
        half_life_days=pw.get("half_life_days", 30.0),
        pre_primary_discount=pw.get("pre_primary_discount", 0.5),
        accuracy_path=accuracy_path if accuracy_path.exists() else None,
        methodology_weights=pw.get("methodology_weights"),
        poll_blend_scale=fc.get("poll_blend_scale", 5.0),
    )
