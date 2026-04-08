"""2026 predictions using type structure (type-primary pipeline).

Loads type assignments, type covariance, county-level historical baselines,
and polls. Performs Gaussian Bayesian update through type structure to produce
county-level 2026 predictions.

Key design: county-level priors, type-level covariance.
  - Each county's prior prediction = its own historical baseline (mean Dem share)
  - Types determine comovement only (how polls adjust predictions)
  - The Bayesian update shifts type means; the shift propagates to counties via
    type scores, but is added to county baselines (not type baselines)

Inputs:
  data/communities/type_assignments.parquet       — county type scores
  data/covariance/type_covariance.parquet          — J x J covariance
  data/communities/type_profiles.parquet           — type demographic profiles
  data/polls/polls_2026.csv                        — poll observations
  data/assembled/medsl_county_presidential_*.parquet — county historical results

Outputs:
  data/predictions/county_predictions_2026_types.parquet
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as _cfg
from src.prediction.county_priors import load_county_priors_with_ridge
from src.prediction.forecast_engine import compute_theta_prior, run_forecast
from src.prediction.fundamentals import (
    compute_fundamentals_shift,
    load_fundamentals_snapshot,
)
from src.prediction.generic_ballot import compute_gb_shift

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# State FIPS -> abbreviation (all 50 states + DC, sourced from config/model.yaml)
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR

# ---------------------------------------------------------------------------
# Forecast hyperparameters — loaded from data/config/prediction_params.json.
# lam: θ_national regularization strength. Calibrated 2026-04-01 via
#      scripts/calibrate_lam_mu.py. lam=1.0 retained (see prediction_params.json
#      for full calibration notes). Sweep range [0.1, 20]; RMSE improvement is
#      monotonic but small (~0.0001) and is a validation-data artifact.
# mu:  δ_race regularization strength. Calibrated 2026-04-01. mu has no
#      measurable impact on RMSE across [0.1, 20] — delta_race is underdetermined
#      with state-level polls only. mu=1.0 retained.
# w_vector_mode: W vector construction tier ("core" vs "full" — benchmark first)
# ---------------------------------------------------------------------------
_PARAMS_PATH = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
_all_params: dict = json.loads(_PARAMS_PATH.read_text())
_forecast_params: dict = _all_params["forecast"]
_LAM: float = _forecast_params["lam"]
_MU: float = _forecast_params["mu"]
_W_VECTOR_MODE: str = _forecast_params["w_vector_mode"]

# Poll weighting hyperparameters.  These live in prediction_params.json so they
# can be swept via scripts/experiments/sweep_poll_half_life.py without touching
# source code.  Function-level defaults in poll_decay.py serve as fallbacks.
_poll_weighting_params: dict = _all_params.get("poll_weighting", {})
_HALF_LIFE_DAYS: float = float(_poll_weighting_params.get("half_life_days", 30.0))
_PRE_PRIMARY_DISCOUNT: float = float(_poll_weighting_params.get("pre_primary_discount", 0.5))
# When True, use empirical RMSE data from pollster_accuracy.json to weight polls.
# Falls back to Silver Bulletin / 538-grade weights for unknown pollsters.
_USE_POLLSTER_RMSE_WEIGHTS: bool = bool(_poll_weighting_params.get("use_pollster_rmse_weights", True))
_ACCURACY_PATH: Path | None = (
    PROJECT_ROOT / "data" / "experiments" / "pollster_accuracy.json"
    if _USE_POLLSTER_RMSE_WEIGHTS
    else None
)
# Methodology-based quality multipliers.  Loaded from prediction_params.json so
# weights can be updated without code changes.  Falls back to module defaults.
from src.propagation.poll_methodology import (  # noqa: E402
    _DEFAULT_METHODOLOGY_WEIGHTS,
    load_methodology_weights,
)

_METHODOLOGY_WEIGHTS: dict[str, float] = (
    load_methodology_weights(_PARAMS_PATH)
    if _poll_weighting_params.get("methodology_weights")
    else dict(_DEFAULT_METHODOLOGY_WEIGHTS)
)
# Fundamentals model configuration.  When enabled, the structural fundamentals
# shift is blended with the generic ballot shift to produce a combined national
# environment prior.  Weight of 0.0 = pure GB (old behavior), 1.0 = pure fundamentals.
_fundamentals_params: dict = _all_params.get("fundamentals", {})
_FUNDAMENTALS_ENABLED: bool = bool(_fundamentals_params.get("enabled", False))
_FUNDAMENTALS_WEIGHT: float = float(_fundamentals_params.get("fundamentals_weight", 0.3))


def _load_type_data() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load type assignments, covariance, and priors from disk.

    Returns
    -------
    county_fips : list[str]
    type_scores : ndarray of shape (N, J)
    type_covariance : ndarray of shape (J, J)
    type_priors : ndarray of shape (J,)
    """
    type_assignments_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    type_cov_path = PROJECT_ROOT / "data" / "covariance" / "type_covariance.parquet"

    log.info("Loading type assignments from %s", type_assignments_path)
    ta_df = pd.read_parquet(type_assignments_path)
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values
    J = type_scores.shape[1]

    log.info("Loading type covariance from %s", type_cov_path)
    cov_df = pd.read_parquet(type_cov_path)
    type_covariance = cov_df.values[:J, :J]

    # Load type priors from 2024 actual results (population-weighted Dem share per type).
    # These are used as the baseline for the Bayesian update (type-level).
    type_priors = np.full(J, 0.45)
    priors_path = PROJECT_ROOT / "data" / "communities" / "type_priors.parquet"
    if priors_path.exists():
        priors_df = pd.read_parquet(priors_path)
        if "prior_dem_share" in priors_df.columns:
            for _, row in priors_df.iterrows():
                t = int(row["type_id"])
                if t < J:
                    type_priors[t] = row["prior_dem_share"]
    log.info("Type priors: %s", np.round(type_priors, 3))

    return county_fips, type_scores, type_covariance, type_priors


def _load_county_metadata(county_fips: list[str]) -> tuple[list[str], list[str]]:
    """Derive state abbreviations and county names from FIPS codes.

    Returns
    -------
    states : list[str]
    county_names : list[str]
    """
    states = [_STATE_FIPS_TO_ABBR.get(f[:2], "??") for f in county_fips]
    county_names = [""] * len(county_fips)

    crosswalk_path = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"
    if crosswalk_path.exists():
        xwalk = pd.read_csv(crosswalk_path, dtype=str)
        xwalk["county_fips"] = xwalk["county_fips"].str.zfill(5)
        name_map = dict(zip(xwalk["county_fips"], xwalk["county_name"]))
        county_names = [name_map.get(f, "") for f in county_fips]

    return states, county_names


def _load_county_votes(county_fips: list[str]) -> np.ndarray:
    """Load 2024 total vote counts per county for vote-weighted W construction.

    Falls back to equal weight (1.0 per county) if the file is unavailable.
    """
    county_votes = np.ones(len(county_fips))
    votes_path = PROJECT_ROOT / "data" / "raw" / "medsl_county_presidential_2024.parquet"
    if votes_path.exists():
        vdf = pd.read_parquet(votes_path)
        if "county_fips" in vdf.columns and "totalvotes" in vdf.columns:
            vmap = dict(zip(
                vdf["county_fips"].astype(str).str.zfill(5),
                vdf["totalvotes"],
            ))
            county_votes = np.array([vmap.get(f, 1.0) for f in county_fips])
    return county_votes


def _load_polls(
    county_fips: list[str],
) -> tuple[dict[str, list[dict]], dict[str, list[tuple[float, int, str]]]]:
    """Load and prepare state-level polls from polls_2026.csv.

    Returns
    -------
    polls_by_race : dict[str, list[dict]]
        For the new forecast engine (run_forecast).
    poll_lookup : dict[str, list[tuple]]
        Legacy format used for unmatched-race warnings.
    """
    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    log.info("Loading polls from %s", polls_path)
    polls = pd.read_csv(polls_path)

    # Normalize column name
    if "geography" in polls.columns and "state" not in polls.columns:
        polls = polls.rename(columns={"geography": "state"})

    # Keep only state-level polls (drop district or national polls)
    if "geo_level" in polls.columns:
        poll_agg = polls[polls["geo_level"] == "state"].copy()
    else:
        poll_agg = polls.copy()

    # Build race -> list of poll tuples (legacy format used for warning check)
    poll_lookup: dict[str, list[tuple[float, int, str]]] = {}
    if len(poll_agg) > 0:
        for race, race_group in poll_agg.groupby("race"):
            if race.startswith("2026 Generic Ballot"):
                continue
            race_polls = [
                (
                    float(row["dem_share"]),
                    int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
                    str(row["state"]),
                )
                for _, row in race_group.iterrows()
                if row.get("geo_level", "state") == "state"
            ]
            if race_polls:
                poll_lookup[race] = race_polls

    # Build the rich dict format expected by run_forecast, including any
    # xt_* demographic composition columns present in the CSV.  These enable
    # Tier 1 W vector construction (raw_sample_demographics) inside the
    # forecast engine — polls without xt_ data fall back to Tier 3 as before.
    xt_cols = [c for c in poll_agg.columns if c.startswith("xt_")]
    polls_by_race: dict[str, list[dict]] = {}
    for race_id, poll_list in poll_lookup.items():
        race_dicts: list[dict] = []
        # Re-iterate the filtered DataFrame rows to capture xt_ columns.
        # poll_lookup already filtered to state-level non-generic-ballot rows,
        # so we match on race and use the same geo_level guard.
        race_rows = poll_agg[
            (poll_agg["race"] == race_id)
            & (poll_agg.get("geo_level", pd.Series(["state"] * len(poll_agg))) == "state")
        ]
        for _, row in race_rows.iterrows():
            poll_dict: dict = {
                "dem_share": float(row["dem_share"]),
                "n_sample": int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
                "state": str(row["state"]),
            }
            # Attach xt_ fields that have non-null values; downstream code
            # maps these to type_profiles columns before passing to build_W_poll.
            for col in xt_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    poll_dict[col] = float(val)
            race_dicts.append(poll_dict)
        if race_dicts:
            polls_by_race[race_id] = race_dicts

    return polls_by_race, poll_lookup


def run() -> None:
    """Load inputs from data/ and produce type-based 2026 predictions."""
    county_fips, type_scores, type_covariance, type_priors = _load_type_data()
    county_prior_values = load_county_priors_with_ridge(county_fips)
    states, county_names = _load_county_metadata(county_fips)
    county_votes = _load_county_votes(county_fips)
    polls_by_race, poll_lookup = _load_polls(county_fips)

    # Apply voter behavior layer (τ/δ) adjustment.
    # All 2026 races are off-cycle (governor/senate), so the adjustment applies
    # universally.  The δ per type captures the systematic Dem share shift in
    # off-cycle elections beyond what differential turnout (τ) explains.
    # We compute each county's adjustment as the type-membership-weighted mean δ.
    behavior_dir = PROJECT_ROOT / "data" / "behavior"
    tau_path = behavior_dir / "tau.npy"
    delta_path = behavior_dir / "delta.npy"
    if tau_path.exists() and delta_path.exists():
        tau = np.load(tau_path)
        delta = np.load(delta_path)
        # Per-county adjustment = weighted sum of δ_j by type membership scores.
        # type_scores shape: (n_counties, J); delta shape: (J,).
        county_delta_shift = type_scores @ delta  # (n_counties,)
        county_prior_values = np.clip(county_prior_values + county_delta_shift, 0.0, 1.0)
        log.info(
            "Behavior layer applied: mean δ shift = %+.3f pp (τ mean=%.3f, δ mean=%+.4f)",
            county_delta_shift.mean() * 100, tau.mean(), delta.mean(),
        )
    else:
        log.warning("Behavior layer data not found at %s, skipping adjustment", behavior_dir)

    # Load type profiles for W vector enrichment
    type_profiles_path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    type_profiles_df = None
    if type_profiles_path.exists():
        type_profiles_df = pd.read_parquet(type_profiles_path)
        log.info("Loaded type profiles for poll enrichment: %d types", len(type_profiles_df))

    # Precompute state-level population demographic vectors for post-stratification
    # correction in Tier 2 W construction.  This corrects for polls that oversample
    # demographic groups (e.g., college-educated at 55% vs population 30%), which
    # would otherwise give those groups an artificially low sigma and outsized influence.
    # Requires: type_profiles, county_type_assignments, and 2020 presidential vote totals.
    state_population_vectors: dict | None = None
    county_assignments_path = PROJECT_ROOT / "data" / "communities" / "county_type_assignments_full.parquet"
    county_votes_2020_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2020.parquet"
    if (
        type_profiles_df is not None
        and county_assignments_path.exists()
        and county_votes_2020_path.exists()
    ):
        from src.prediction.population_vectors import (
            XT_TO_TYPE_PROFILE_COL,
            build_state_population_vectors,
        )
        county_assignments_df = pd.read_parquet(county_assignments_path)
        county_votes_2020_df = pd.read_parquet(county_votes_2020_path)
        # Only compute vectors for the xt_ columns that are actually present in poll data.
        all_xt_cols = list(XT_TO_TYPE_PROFILE_COL.keys())
        try:
            state_population_vectors = build_state_population_vectors(
                type_profiles_df, county_assignments_df, county_votes_2020_df, all_xt_cols
            )
            log.info(
                "Built state population vectors for post-stratification (%d states)",
                len(state_population_vectors),
            )
        except Exception as exc:
            log.warning("Could not build state population vectors, skipping post-strat: %s", exc)
            state_population_vectors = None

    # Compute generic ballot shift from national polls.  This adjusts county
    # priors toward the current national environment before the Bayesian update.
    gb_info = compute_gb_shift()
    gb_shift = gb_info.shift
    log.info(
        "Generic ballot shift: %+.1f pp (%d polls, source=%s)",
        gb_shift * 100, gb_info.n_polls, gb_info.source,
    )

    # Optionally blend in the structural fundamentals shift (approval, GDP,
    # unemployment, CPI).  The combined shift is:
    #   shift = w * fundamentals + (1 - w) * generic_ballot
    # where w = fundamentals_weight from prediction_params.json.
    if _FUNDAMENTALS_ENABLED:
        try:
            snapshot = load_fundamentals_snapshot()
            fund_info = compute_fundamentals_shift(snapshot)
            w = _FUNDAMENTALS_WEIGHT
            combined_shift = w * fund_info.shift + (1 - w) * gb_shift
            log.info(
                "Fundamentals shift: %+.1f pp (LOO RMSE=%.1f pp, weight=%.0f%%)",
                fund_info.shift * 100, fund_info.loo_rmse * 100, w * 100,
            )
            log.info(
                "Combined environment shift: %+.1f pp (was %+.1f pp GB-only)",
                combined_shift * 100, gb_shift * 100,
            )
            gb_shift = combined_shift
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Fundamentals model unavailable, using GB only: %s", exc)

    # Iterate over ALL registered races using the hierarchical forecast engine.
    # Produces dual-mode output: "national" (θ_national only) and
    # "local" (θ_national + δ_race) for every race.
    from src.assembly.define_races import load_races

    registry = load_races(2026)
    all_race_ids = [r.race_id for r in registry]
    log.info("Race registry: %d races loaded", len(registry))

    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_prior_values,
        states=states,
        county_votes=county_votes,
        polls_by_race=polls_by_race,
        races=all_race_ids,
        lam=_LAM,
        mu=_MU,
        generic_ballot_shift=gb_shift,
        w_vector_mode=_W_VECTOR_MODE,
        reference_date=str(date.today()),
        type_profiles=type_profiles_df,
        half_life_days=_HALF_LIFE_DAYS,
        pre_primary_discount=_PRE_PRIMARY_DISCOUNT,
        accuracy_path=_ACCURACY_PATH,
        methodology_weights=_METHODOLOGY_WEIGHTS,
        state_population_vectors=state_population_vectors,
    )

    # Convert ForecastResult → DataFrame rows (both modes per race)
    all_predictions = []
    for race_id, fr in forecast_results.items():
        for mode in ("national", "local"):
            preds = fr.county_preds_national if mode == "national" else fr.county_preds_local
            row = pd.DataFrame({
                "county_fips": county_fips,
                "state": states,
                "county_name": county_names,
                "pred_dem_share": preds,
                "race": race_id,
                "forecast_mode": mode,
            })
            all_predictions.append(row)

    if poll_lookup:
        unmatched = set(poll_lookup.keys()) - set(all_race_ids)
        if unmatched:
            log.warning("Polls for unregistered races (ignored): %s", unmatched)

    # Generate baseline (pure model prior, no polls, no GB shift)
    theta_baseline = compute_theta_prior(type_scores, county_prior_values)
    baseline_preds = type_scores @ theta_baseline
    all_predictions.append(pd.DataFrame({
        "county_fips": county_fips,
        "state": states,
        "county_name": county_names,
        "pred_dem_share": baseline_preds,
        "race": "baseline",
        "forecast_mode": "national",
    }))

    output = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    out_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)
    log.info("Saved %d predictions to %s", len(output), out_path)
    print(f"Saved {len(output)} county predictions to {out_path}")

    if len(output):
        print(output.groupby(["state", "race"])["pred_dem_share"].describe().round(3))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run()
