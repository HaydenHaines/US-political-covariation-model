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

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# State FIPS -> abbreviation (all 50 states + DC, sourced from config/model.yaml)
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR

# ---------------------------------------------------------------------------
# Forecast hyperparameters — loaded from data/config/prediction_params.json.
# lam: θ_national regularization strength (see GitHub #91 for calibration plan)
# mu: δ_race regularization strength (see GitHub #91 for calibration plan)
# w_vector_mode: W vector construction tier ("core" vs "full" — benchmark first)
# ---------------------------------------------------------------------------
_PARAMS_PATH = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
_forecast_params: dict = json.loads(_PARAMS_PATH.read_text())["forecast"]
_LAM: float = _forecast_params["lam"]
_MU: float = _forecast_params["mu"]
_W_VECTOR_MODE: str = _forecast_params["w_vector_mode"]


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

    # Convert to dict format expected by run_forecast
    polls_by_race: dict[str, list[dict]] = {
        race_id: [
            {"dem_share": p[0], "n_sample": p[1], "state": p[2]}
            for p in poll_list
        ]
        for race_id, poll_list in poll_lookup.items()
    }

    return polls_by_race, poll_lookup


def run() -> None:
    """Load inputs from data/ and produce type-based 2026 predictions."""
    county_fips, type_scores, type_covariance, type_priors = _load_type_data()
    county_prior_values = load_county_priors_with_ridge(county_fips)
    states, county_names = _load_county_metadata(county_fips)
    county_votes = _load_county_votes(county_fips)
    polls_by_race, poll_lookup = _load_polls(county_fips)

    # Load type profiles for W vector enrichment
    type_profiles_path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    type_profiles_df = None
    if type_profiles_path.exists():
        type_profiles_df = pd.read_parquet(type_profiles_path)
        log.info("Loaded type profiles for poll enrichment: %d types", len(type_profiles_df))

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
        w_vector_mode=_W_VECTOR_MODE,
        reference_date=str(date.today()),
        type_profiles=type_profiles_df,
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
