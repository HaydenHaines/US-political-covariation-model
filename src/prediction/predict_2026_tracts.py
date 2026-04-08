"""2026 predictions using tract-level data (tract-primary pipeline).

Loads tract type assignments, Ridge priors, and 2024 presidential vote totals.
Feeds tract data into the existing forecast engine (which is geography-agnostic —
it accepts any N×J input and calls the results "county" internally, but works
identically for tracts).

Design: the forecast engine's parameter names (county_priors, county_votes, etc.)
are internal labels that happen to use "county" terminology. We pass tract arrays
into the same interface. There are 200+ occurrences of these names across 34 files,
so renaming is out of scope — instead, this module passes tracts in and documents
the mapping clearly.

Input → forecast_engine parameter mapping:
  tract type scores  (N_tracts, J)  → type_scores
  ridge tract priors (N_tracts,)    → county_priors
  tract state labels (N_tracts,)    → states
  tract vote counts  (N_tracts,)    → county_votes

Behavior layer (τ and δ):
  For off-cycle races (governor, senate), priors are adjusted before passing to
  the engine: tau shifts the effective electorate composition, delta shifts
  residual choice. For presidential races, no adjustment (τ=1, δ=0 effectively).

Inputs:
  data/communities/tract_type_assignments.parquet  — 80,507 tracts × type scores
  data/models/ridge_model/ridge_tract_priors.parquet — 79,660 tracts × ridge_pred
  data/assembled/tract_elections.parquet            — tract votes by year + race
  data/behavior/tau.npy                             — (J,) per-type turnout ratio
  data/behavior/delta_ces_governor.npy              — (J,) CES governor δ (preferred)
  data/behavior/delta_ces_senate.npy                — (J,) CES senate δ (preferred)
  data/behavior/delta.npy                           — (J,) model-computed δ (fallback)
  data/polls/polls_2026.csv                         — state-level polls

Outputs:
  data/predictions/tract_predictions_2026.parquet   — tract-level predictions
  data/predictions/tract_state_predictions_2026.json — vote-weighted state aggregations
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior.voter_behavior import apply_behavior_adjustment
from src.core import config as _cfg
from src.prediction.forecast_engine import run_forecast

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# State FIPS → abbreviation: derives state from tract GEOID first 2 digits.
# Tract GEOIDs are 11-digit strings: SSCCCTTTTTT (SS=state, CCC=county, TTTTTT=tract).
# We reuse the same dict used by predict_2026_types, sourced from config/model.yaml.
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR

# Race types that are off-cycle (governor + senate variants).
# These receive τ/δ behavior adjustment before the Bayesian update.
# Presidential races get no adjustment (they define the τ baseline).
_OFFCYCLE_RACE_TYPES = {"governor", "senate"}

# Races from the registry that are off-cycle by race_id convention.
# The registry race_type field uses lowercase: "senate", "governor", "president".
_OFFCYCLE_REGISTRY_TYPES = {"senate", "governor"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tract_type_scores() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load tract type membership scores and dominant type assignments.

    Returns
    -------
    tract_geoids : list[str]
        11-digit GEOID strings for each tract (N,).
    type_scores : ndarray of shape (N, J)
        Soft membership scores. Row-normalized within each tract.
    dominant_types : ndarray of shape (N,) dtype int
        Index of the highest-weight type for each tract. Used to look up
        the τ and δ adjustment for that tract's primary community type.
    """
    path = PROJECT_ROOT / "data" / "communities" / "tract_type_assignments.parquet"
    log.info("Loading tract type assignments from %s", path)
    df = pd.read_parquet(path)

    # Deduplicate: DRA pipeline can produce duplicate GEOIDs (see CLAUDE.md gotcha).
    n_before = len(df)
    df = df.drop_duplicates(subset="tract_geoid")
    if len(df) < n_before:
        log.warning("Dropped %d duplicate tract GEOIDs", n_before - len(df))

    tract_geoids = df["tract_geoid"].astype(str).str.zfill(11).tolist()

    score_cols = sorted([c for c in df.columns if c.startswith("type_") and c.endswith("_score")])
    type_scores = df[score_cols].values.astype(float)  # (N, J)

    dominant_types = df["dominant_type"].values.astype(int)  # (N,)

    log.info(
        "Loaded %d tracts, J=%d types",
        len(tract_geoids),
        type_scores.shape[1],
    )
    return tract_geoids, type_scores, dominant_types


def load_tract_priors(tract_geoids: list[str]) -> np.ndarray:
    """Load Ridge-model tract priors aligned to tract_geoids order.

    Tracts not in the Ridge output (coverage gap) fall back to the national
    mean Ridge prediction (≈ 0.45). This handles the ~800-tract gap between
    tract_type_assignments (80,507) and ridge_tract_priors (79,660).

    Returns
    -------
    priors : ndarray of shape (N,) — Dem share prior per tract.
    """
    path = PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_tract_priors.parquet"
    log.info("Loading Ridge tract priors from %s", path)
    rp = pd.read_parquet(path)

    prior_map = dict(zip(
        rp["tract_geoid"].astype(str).str.zfill(11),
        rp["ridge_pred_dem_share"].values,
    ))

    # Compute fallback as mean of available priors (avoids hardcoded constant).
    fallback = float(np.mean(list(prior_map.values()))) if prior_map else 0.45
    n_missing = sum(1 for g in tract_geoids if g not in prior_map)
    if n_missing:
        log.warning(
            "%d tracts missing from Ridge priors — using mean fallback %.3f",
            n_missing, fallback,
        )

    priors = np.array([prior_map.get(g, fallback) for g in tract_geoids])
    log.info(
        "Tract priors loaded: mean=%.3f std=%.3f range=[%.3f, %.3f]",
        priors.mean(), priors.std(), priors.min(), priors.max(),
    )
    return priors


def load_tract_votes(tract_geoids: list[str]) -> np.ndarray:
    """Load 2024 presidential total vote counts per tract for vote weighting.

    Vote weighting is critical for state-level aggregation — without it, small
    low-turnout tracts get the same weight as large urban tracts, biasing results.
    Tracts without 2024 presidential data fall back to the median tract vote count
    (not 1.0), so the fallback doesn't cause extreme distortion.

    Returns
    -------
    votes : ndarray of shape (N,) — 2024 presidential total votes per tract.
    """
    path = PROJECT_ROOT / "data" / "assembled" / "tract_elections.parquet"
    log.info("Loading tract vote counts from %s", path)
    te = pd.read_parquet(path, columns=["tract_geoid", "year", "race_type", "total_votes"])

    pres_2024 = te[(te["year"] == 2024) & (te["race_type"] == "PRES")]
    votes_map = dict(zip(
        pres_2024["tract_geoid"].astype(str).str.zfill(11),
        pres_2024["total_votes"].values,
    ))

    fallback = float(np.median(list(votes_map.values()))) if votes_map else 1000.0
    n_missing = sum(1 for g in tract_geoids if g not in votes_map)
    if n_missing:
        log.info(
            "%d tracts missing 2024 presidential votes — using median fallback %.0f",
            n_missing, fallback,
        )

    votes = np.array([votes_map.get(g, fallback) for g in tract_geoids])
    log.info(
        "Tract votes loaded: %d tracts, total=%.0f M votes",
        len(votes), votes.sum() / 1e6,
    )
    return votes


def derive_tract_states(tract_geoids: list[str]) -> list[str]:
    """Derive state abbreviation from tract GEOID (first 2 digits = state FIPS).

    Tract GEOIDs are 11-character strings: SSCCCTTTTTT.
    The first 2 characters are the state FIPS code, which maps directly to the
    same STATE_ABBR dict used by the county pipeline (sourced from config/model.yaml).

    Returns
    -------
    states : list[str] — state abbreviation per tract (e.g. "FL", "GA").
             Unknown FIPS codes map to "??" (defensive fallback).
    """
    return [_STATE_FIPS_TO_ABBR.get(g[:2], "??") for g in tract_geoids]


def load_behavior_layer() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load per-type τ and race-specific CES-derived δ arrays.

    τ (tau): turnout ratio from the model-computed behavior layer.
    δ (delta): per-type choice shift from CES external validation (248K voters).
               Governor and Senate races get separate δ because the CES shows
               different type-level behavior across race types (r=0.39 between
               governor and senate δ).

    Falls back to model-computed δ if CES files are missing.

    Returns
    -------
    tau : ndarray of shape (J,)
    deltas : dict mapping race type ("governor", "senate") to ndarray of shape (J,).
             Also includes "default" key for backward compatibility.
    """
    tau_path = PROJECT_ROOT / "data" / "behavior" / "tau.npy"
    tau = np.load(tau_path)

    # Prefer CES-derived δ (empirical, from 248K validated voters).
    # CES δ correlates r=0.894 with model type structure; model-computed δ
    # has r=-0.008 correlation with CES δ (i.e., no signal).
    deltas: dict[str, np.ndarray] = {}
    for race in ("governor", "senate"):
        ces_path = PROJECT_ROOT / "data" / "behavior" / f"delta_ces_{race}.npy"
        if ces_path.exists():
            deltas[race] = np.load(ces_path)
            log.info(
                "CES δ (%s): J=%d, mean=%.4f, std=%.4f",
                race, len(deltas[race]), deltas[race].mean(), deltas[race].std(),
            )
        else:
            log.warning("CES δ for %s not found at %s, falling back to model δ", race, ces_path)

    # Fallback: model-computed δ (weak signal but better than nothing).
    if not deltas:
        delta_path = PROJECT_ROOT / "data" / "behavior" / "delta.npy"
        model_delta = np.load(delta_path)
        deltas["governor"] = model_delta
        deltas["senate"] = model_delta
        log.warning("Using model-computed δ (no CES data available)")

    # Default key for any code that doesn't specify race type.
    deltas["default"] = deltas.get("governor", deltas.get("senate"))

    log.info("Behavior layer: J=%d, τ=[%.3f, %.3f]", len(tau), tau.min(), tau.max())
    return tau, deltas


def load_polls() -> tuple[dict[str, list[dict]], dict[str, list[tuple[float, int, str]]]]:
    """Load and prepare state-level polls from polls_2026.csv.

    Returns
    -------
    polls_by_race : dict[str, list[dict]]
        Rich dict format for run_forecast (includes xt_* demographic fields).
    poll_lookup : dict[str, list[tuple]]
        Legacy flat format used for unmatched-race warnings.
    """
    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    log.info("Loading polls from %s", polls_path)
    polls_df = pd.read_csv(polls_path)

    # Normalize geography column name — some pipeline versions use "geography"
    if "geography" in polls_df.columns and "state" not in polls_df.columns:
        polls_df = polls_df.rename(columns={"geography": "state"})

    # Keep only state-level, non-generic-ballot polls.
    if "geo_level" in polls_df.columns:
        state_polls = polls_df[polls_df["geo_level"] == "state"].copy()
    else:
        state_polls = polls_df.copy()

    xt_cols = [c for c in state_polls.columns if c.startswith("xt_")]

    # Build legacy poll_lookup for warning checks.
    poll_lookup: dict[str, list[tuple[float, int, str]]] = {}
    for race, group in state_polls.groupby("race"):
        if str(race).startswith("2026 Generic Ballot"):
            continue
        race_polls = [
            (
                float(row["dem_share"]),
                int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
                str(row["state"]),
            )
            for _, row in group.iterrows()
            if row.get("geo_level", "state") == "state"
        ]
        if race_polls:
            poll_lookup[race] = race_polls

    # Build rich dict format with xt_* demographic fields for poll enrichment.
    polls_by_race: dict[str, list[dict]] = {}
    for race_id, race_polls_list in poll_lookup.items():
        race_rows = state_polls[
            (state_polls["race"] == race_id)
            & (state_polls.get("geo_level", pd.Series(["state"] * len(state_polls))) == "state")
        ]
        race_dicts: list[dict] = []
        for _, row in race_rows.iterrows():
            d: dict = {
                "dem_share": float(row["dem_share"]),
                "n_sample": int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
                "state": str(row["state"]),
            }
            for col in xt_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    d[col] = float(val)
            race_dicts.append(d)
        if race_dicts:
            polls_by_race[race_id] = race_dicts

    log.info("Loaded %d races with polls", len(polls_by_race))
    return polls_by_race, poll_lookup


# ---------------------------------------------------------------------------
# Behavior adjustment
# ---------------------------------------------------------------------------


def adjust_priors_for_race_type(
    priors: np.ndarray,
    type_scores: np.ndarray,
    tau: np.ndarray,
    deltas: dict[str, np.ndarray],
    race_type: str,
) -> np.ndarray:
    """Adjust tract priors for the given race type using behavior layer parameters.

    Presidential races: no adjustment (τ and δ were estimated relative to the
    presidential baseline — applying them would double-count).

    Off-cycle races (governor, senate): apply τ and race-specific CES δ via
    apply_behavior_adjustment. Governor and senate races get different δ because
    CES data shows different type-level behavior across race types.

    Args:
        priors: Ridge tract priors, shape (N,).
        type_scores: Soft membership matrix, shape (N, J).
        tau: Per-type turnout ratio, shape (J,).
        deltas: Dict mapping race type to per-type choice shift arrays.
        race_type: Registry race_type string ("president", "senate", "governor").

    Returns:
        Adjusted priors, shape (N,). Clipped to [0, 1].
    """
    is_offcycle = race_type in _OFFCYCLE_REGISTRY_TYPES
    delta = deltas.get(race_type, deltas["default"])
    return apply_behavior_adjustment(priors, type_scores, tau, delta, is_offcycle)


# ---------------------------------------------------------------------------
# State-level aggregation
# ---------------------------------------------------------------------------


def aggregate_to_states(
    tract_preds: np.ndarray,
    tract_votes: np.ndarray,
    states: list[str],
) -> dict[str, float]:
    """Aggregate tract-level predictions to state level via vote-weighting.

    State prediction = SUM(pred * votes) / SUM(votes) across all tracts in state.
    This matches how the county pipeline aggregates: state_pred is the vote-weighted
    mean of local predictions, not the simple average.

    States with no tracts return NaN (filtered out downstream).

    Args:
        tract_preds: Predicted Dem share per tract, shape (N,).
        tract_votes: Total votes per tract for weighting, shape (N,).
        states: State abbreviation per tract, length N.

    Returns:
        dict mapping state_abbr → weighted Dem share prediction.
    """
    state_arr = np.array(states)
    unique_states = sorted(set(states))

    result: dict[str, float] = {}
    for st in unique_states:
        if st == "??":
            continue
        mask = state_arr == st
        preds_st = tract_preds[mask]
        votes_st = tract_votes[mask]
        total_votes = votes_st.sum()
        if total_votes > 0:
            result[st] = float((preds_st * votes_st).sum() / total_votes)
        else:
            result[st] = float(preds_st.mean())

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run() -> None:
    """Load tract inputs and produce 2026 tract-level and state-level predictions."""
    from src.assembly.define_races import load_races
    from src.prediction.fundamentals import (
        compute_fundamentals_shift,
        load_fundamentals_snapshot,
    )
    from src.prediction.generic_ballot import compute_gb_shift

    # Load prediction hyperparameters from the shared config file.
    params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
    all_params = json.loads(params_path.read_text())
    forecast_params = all_params["forecast"]
    lam: float = forecast_params["lam"]
    mu: float = forecast_params["mu"]
    w_vector_mode: str = forecast_params["w_vector_mode"]

    poll_weighting_params = all_params.get("poll_weighting", {})
    half_life_days = float(poll_weighting_params.get("half_life_days", 30.0))
    pre_primary_discount = float(poll_weighting_params.get("pre_primary_discount", 0.5))
    use_pollster_rmse = bool(poll_weighting_params.get("use_pollster_rmse_weights", True))
    accuracy_path = (
        PROJECT_ROOT / "data" / "experiments" / "pollster_accuracy.json"
        if use_pollster_rmse
        else None
    )

    from src.propagation.poll_methodology import (
        _DEFAULT_METHODOLOGY_WEIGHTS,
        load_methodology_weights,
    )
    methodology_weights = (
        load_methodology_weights(params_path)
        if poll_weighting_params.get("methodology_weights")
        else dict(_DEFAULT_METHODOLOGY_WEIGHTS)
    )

    fundamentals_params = all_params.get("fundamentals", {})
    fundamentals_enabled = bool(fundamentals_params.get("enabled", False))
    fundamentals_weight = float(fundamentals_params.get("fundamentals_weight", 0.3))

    # --- Load tract data ---
    tract_geoids, type_scores, dominant_types = load_tract_type_scores()
    tract_priors = load_tract_priors(tract_geoids)
    tract_votes = load_tract_votes(tract_geoids)
    states = derive_tract_states(tract_geoids)
    tau, deltas = load_behavior_layer()
    polls_by_race, poll_lookup = load_polls()

    J = type_scores.shape[1]
    log.info("Tract pipeline: N=%d tracts, J=%d types", len(tract_geoids), J)

    # Generic ballot and fundamentals shift (same logic as county pipeline).
    gb_info = compute_gb_shift()
    gb_shift = gb_info.shift
    log.info(
        "Generic ballot shift: %+.1f pp (%d polls, source=%s)",
        gb_shift * 100, gb_info.n_polls, gb_info.source,
    )

    if fundamentals_enabled:
        try:
            snapshot = load_fundamentals_snapshot()
            fund_info = compute_fundamentals_shift(snapshot)
            w = fundamentals_weight
            combined_shift = w * fund_info.shift + (1 - w) * gb_shift
            log.info(
                "Fundamentals shift: %+.1f pp (weight=%.0f%%), combined: %+.1f pp",
                fund_info.shift * 100, w * 100, combined_shift * 100,
            )
            gb_shift = combined_shift
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Fundamentals unavailable, using GB only: %s", exc)

    # Race registry — we run all 2026 races.
    registry = load_races(2026)
    all_race_ids = [r.race_id for r in registry]
    # Build race_id → race_type lookup for behavior adjustment decisions.
    race_type_lookup = {r.race_id: r.race_type for r in registry}
    log.info("Race registry: %d races", len(registry))

    # --- Run forecast per race ---
    # Because τ/δ adjustment changes the priors (which flow into θ_prior), we need
    # to run the forecast separately for presidential vs off-cycle races.
    # Presidential races use raw Ridge priors; off-cycle races use τ/δ-adjusted priors.
    #
    # Grouping races by race_type and running two batch calls is more efficient than
    # per-race calls. The forecast engine pools all poll observations for θ_national
    # estimation — so grouping correctly is important.
    #
    # Implementation choice: run two separate run_forecast calls (presidential + offcycle).
    # This avoids mixing the two prior adjustments in a single engine call and makes
    # the behavior layer integration transparent.

    pres_race_ids = [rid for rid in all_race_ids if race_type_lookup.get(rid) == "president"]
    n_offcycle = sum(1 for rid in all_race_ids if race_type_lookup.get(rid) in _OFFCYCLE_REGISTRY_TYPES)

    log.info(
        "Running forecast: %d presidential races, %d off-cycle races",
        len(pres_race_ids), n_offcycle,
    )

    all_results: dict = {}

    # Presidential races: use unadjusted priors + GB shift
    if pres_race_ids:
        pres_priors = tract_priors + gb_shift
        pres_results = run_forecast(
            type_scores=type_scores,
            county_priors=pres_priors,          # tract priors aliased as county_priors
            states=states,
            county_votes=tract_votes,           # tract votes aliased as county_votes
            polls_by_race={
                rid: polls_by_race[rid]
                for rid in pres_race_ids
                if rid in polls_by_race
            },
            races=pres_race_ids,
            lam=lam,
            mu=mu,
            generic_ballot_shift=0.0,           # shift already baked into pres_priors
            w_vector_mode=w_vector_mode,
            reference_date=str(date.today()),
            half_life_days=half_life_days,
            pre_primary_discount=pre_primary_discount,
            accuracy_path=accuracy_path,
            methodology_weights=methodology_weights,
        )
        all_results.update(pres_results)

    # Off-cycle races: behavior layer FULLY INERT (both δ and τ no-ops).
    #
    # δ is DISABLED based on CES temporal stability analysis (S501):
    #   - Governor δ has mean cross-year r=0.091 (cycle noise, not type property)
    #   - CES δ backtest: r drops from 0.839 to 0.777 when applied
    #   - Model δ backtest: r drops from 0.839 to 0.793
    # τ is stable (r=0.599) but PRACTICALLY NEGLIGIBLE at J=100 (S504):
    #   - With δ=0, τ has no effect — adjust_priors computes τ-weighted δ, which is 0.
    #   - Even if τ were applied to type_scores directly, avg composition shift
    #     is only 1.14% per tract (τ range [0.733, 1.027], std=0.072).
    #   - Correct τ integration would reweight type_scores before forecast_engine,
    #     but the effect size is < 0.5pp on state predictions. Not worth the complexity.
    # Race-specific δ infrastructure retained for future use if pooling improves.
    #
    # Zero out δ (τ passes through adjust_priors but has no effect with δ=0):
    zero_deltas = {k: np.zeros_like(v) for k, v in deltas.items()}

    gov_race_ids = [
        rid for rid in all_race_ids
        if race_type_lookup.get(rid) == "governor"
    ]
    sen_race_ids = [
        rid for rid in all_race_ids
        if race_type_lookup.get(rid) == "senate"
    ]

    for race_type, race_ids in [("governor", gov_race_ids), ("senate", sen_race_ids)]:
        if not race_ids:
            continue
        adjusted_priors = adjust_priors_for_race_type(
            tract_priors, type_scores, tau, zero_deltas, race_type=race_type,
        )
        adjusted_priors = adjusted_priors + gb_shift
        results = run_forecast(
            type_scores=type_scores,
            county_priors=adjusted_priors,
            states=states,
            county_votes=tract_votes,
            polls_by_race={
                rid: polls_by_race[rid]
                for rid in race_ids
                if rid in polls_by_race
            },
            races=race_ids,
            lam=lam,
            mu=mu,
            generic_ballot_shift=0.0,           # shift already baked into adjusted_priors
            w_vector_mode=w_vector_mode,
            reference_date=str(date.today()),
            half_life_days=half_life_days,
            pre_primary_discount=pre_primary_discount,
            accuracy_path=accuracy_path,
            methodology_weights=methodology_weights,
        )
        all_results.update(results)

    # --- Build output DataFrame ---
    all_predictions: list[pd.DataFrame] = []
    for race_id, fr in all_results.items():
        for mode in ("national", "local"):
            preds = fr.county_preds_national if mode == "national" else fr.county_preds_local
            # Aggregate to state level for state_pred column
            state_preds_map = aggregate_to_states(preds, tract_votes, states)
            state_pred_col = [state_preds_map.get(s, float("nan")) for s in states]

            all_predictions.append(pd.DataFrame({
                "tract_geoid": tract_geoids,
                "state": states,
                "pred_dem_share": preds,
                "state_pred_dem_share": state_pred_col,
                "race": race_id,
                "forecast_mode": mode,
            }))

    output = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # --- Write tract-level parquet ---
    out_dir = PROJECT_ROOT / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "tract_predictions_2026.parquet"
    output.to_parquet(parquet_path, index=False)
    log.info("Saved %d tract predictions to %s", len(output), parquet_path)
    print(f"Saved {len(output):,} tract-level predictions to {parquet_path}")

    # --- Write state-level JSON ---
    # Build state aggregations for all races and both modes.
    state_json: dict[str, dict[str, dict[str, float]]] = {}
    for race_id, fr in all_results.items():
        state_json[race_id] = {}
        for mode in ("national", "local"):
            preds = fr.county_preds_national if mode == "national" else fr.county_preds_local
            state_preds_map = aggregate_to_states(preds, tract_votes, states)
            state_json[race_id][mode] = state_preds_map

    json_path = out_dir / "tract_state_predictions_2026.json"
    with open(json_path, "w") as f:
        json.dump(state_json, f, indent=2)
    log.info("Saved state-level predictions to %s", json_path)
    print(f"Saved state-level predictions to {json_path}")

    # --- Summary output ---
    _print_summary(all_results, tract_geoids, states, tract_votes)

    if poll_lookup:
        unmatched = set(poll_lookup.keys()) - set(all_race_ids)
        if unmatched:
            log.warning("Polls for unregistered races (ignored): %s", unmatched)


def _print_summary(
    all_results: dict,
    tract_geoids: list[str],
    states: list[str],
    tract_votes: np.ndarray,
) -> None:
    """Print a human-readable summary of tract counts and state predictions."""
    # Tracts per state
    state_arr = np.array(states)
    unique_states = sorted(s for s in set(states) if s != "??")
    print("\n--- Tracts per state ---")
    for st in unique_states[:10]:  # show first 10 states
        n = (state_arr == st).sum()
        print(f"  {st}: {n:,} tracts")
    print(f"  ... ({len(unique_states)} states total, {len(tract_geoids):,} tracts)")

    # State predictions for selected key races
    key_races = [
        "2026 GA Senate",
        "2026 TX Senate",
        "2026 NC Senate",
        "2026 AZ Senate",
        "2026 FL Senate",
        "2026 OH Senate",
        "2026 PA Senate",
        "2026 WI Senate",
    ]
    print("\n--- State predictions (national mode, key races) ---")
    for race_id in key_races:
        if race_id not in all_results:
            continue
        fr = all_results[race_id]
        state_preds = aggregate_to_states(
            fr.county_preds_national, tract_votes, states
        )
        st = race_id.split(" ")[1]  # e.g. "GA" from "2026 GA Senate"
        pred = state_preds.get(st, float("nan"))
        print(f"  {race_id}: {pred:.3f} Dem share ({fr.n_polls} polls)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run()
