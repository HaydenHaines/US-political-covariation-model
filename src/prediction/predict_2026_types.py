"""Type-primary forecast pipeline.

Loads type assignments, type covariance, county-level historical baselines,
and polls. Performs Gaussian Bayesian update through type structure to produce
county-level predictions.

Key design: county-level priors, type-level covariance.
  - Each county's prior prediction = its own historical baseline (mean Dem share)
  - Types determine comovement only (how polls adjust predictions)
  - The Bayesian update shifts type means; the shift propagates to counties via
    type scores, but is added to county baselines (not type baselines)

Public API (for backtest harness and other callers):
  - ForecastParams: dataclass holding all hyperparameters
  - load_forecast_params(): load ForecastParams from prediction_params.json
  - load_type_data(): load type assignments, covariance, and priors
  - load_county_metadata(): derive state abbreviations and county names
  - load_county_votes(): load vote counts for W vector construction
  - load_polls(): load and prepare poll data
  - run_forecast_pipeline(): run the full pipeline with custom parameters

Convenience entry point:
  - run(): loads defaults from disk and calls run_forecast_pipeline()

Default data paths (used by run()):
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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as _cfg
from src.prediction.county_priors import (
    load_county_priors_with_ridge,
    load_county_priors_with_ridge_governor,
)
from src.prediction.early_results import (
    extract_gb_observations,
    load_early_results,
    merge_early_results,
)
from src.prediction.forecast_engine import run_forecast
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
# ForecastParams — all hyperparameters in one place for parameterized runs.
# ---------------------------------------------------------------------------

@dataclass
class ForecastParams:
    """All forecast hyperparameters needed by the pipeline.

    Load from disk with ``load_forecast_params()``, or construct directly
    to override values (e.g., in backtests or parameter sweeps).
    """

    # θ_national regularization strength.  Higher values trust priors more.
    lam: float = 1.0
    # δ_race regularization strength.  Largely irrelevant with sparse state polls.
    mu: float = 1.0
    # W vector construction tier ("core" vs "full").
    w_vector_mode: str = "core"
    # Controls blend between county priors (few polls) and type-projected
    # predictions (many polls).  k in alpha = 1/(1 + n_polls/k).
    poll_blend_scale: float = 5.0

    # Poll weighting
    half_life_days: float = 30.0
    pre_primary_discount: float = 0.5
    accuracy_path: Path | None = None
    methodology_weights: dict[str, float] = field(default_factory=dict)

    # Fundamentals blending
    fundamentals_enabled: bool = False
    fundamentals_weight: float = 0.3

    # State-level economic adjustment (QCEW-based)
    state_econ_enabled: bool = False
    state_econ_sensitivity: float = 0.5

    # Per-race prior overrides (e.g., RCV states, unusual candidate dynamics).
    # Maps race_id -> {"prior_dem_share_override": float, "notes": str}.
    race_adjustments: dict[str, dict] = field(default_factory=dict)


def load_forecast_params(
    params_path: Path | None = None,
) -> ForecastParams:
    """Load forecast parameters from prediction_params.json.

    Parameters
    ----------
    params_path : Path or None
        Path to prediction_params.json.  Defaults to
        ``PROJECT_ROOT / "data" / "config" / "prediction_params.json"``.

    Returns
    -------
    ForecastParams
        Populated dataclass with all hyperparameters.
    """
    if params_path is None:
        params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"

    all_params: dict = json.loads(params_path.read_text())
    forecast_section: dict = all_params["forecast"]
    poll_section: dict = all_params.get("poll_weighting", {})
    fund_section: dict = all_params.get("fundamentals", {})

    use_rmse = bool(poll_section.get("use_pollster_rmse_weights", True))
    accuracy_path: Path | None = (
        PROJECT_ROOT / "data" / "experiments" / "pollster_accuracy.json"
        if use_rmse
        else None
    )

    # Methodology weights: load from JSON if present, else use module defaults.
    from src.propagation.poll_methodology import (
        _DEFAULT_METHODOLOGY_WEIGHTS,
        load_methodology_weights,
    )

    methodology_weights: dict[str, float] = (
        load_methodology_weights(params_path)
        if poll_section.get("methodology_weights")
        else dict(_DEFAULT_METHODOLOGY_WEIGHTS)
    )

    state_econ_section: dict = all_params.get("state_economics", {})

    # Race-level prior overrides.  Strip keys starting with "_" (comments).
    raw_adjustments: dict = all_params.get("race_adjustments", {})
    race_adjustments: dict[str, dict] = {
        k: v for k, v in raw_adjustments.items() if not k.startswith("_")
    }

    return ForecastParams(
        lam=float(forecast_section["lam"]),
        mu=float(forecast_section["mu"]),
        w_vector_mode=str(forecast_section["w_vector_mode"]),
        poll_blend_scale=float(forecast_section.get("poll_blend_scale", 5.0)),
        half_life_days=float(poll_section.get("half_life_days", 30.0)),
        pre_primary_discount=float(poll_section.get("pre_primary_discount", 0.5)),
        accuracy_path=accuracy_path,
        methodology_weights=methodology_weights,
        fundamentals_enabled=bool(fund_section.get("enabled", False)),
        fundamentals_weight=float(fund_section.get("fundamentals_weight", 0.3)),
        state_econ_enabled=bool(state_econ_section.get("enabled", False)),
        state_econ_sensitivity=float(state_econ_section.get("sensitivity", 0.5)),
        race_adjustments=race_adjustments,
    )


def load_type_data() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
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


def load_county_metadata(county_fips: list[str]) -> tuple[list[str], list[str]]:
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


def load_county_votes(county_fips: list[str]) -> np.ndarray:
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


def load_polls(
    county_fips: list[str],
    polls_path: Path | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[tuple[float, int, str]]]]:
    """Load and prepare state-level polls from a CSV file.

    Parameters
    ----------
    county_fips : list[str]
        County FIPS codes (used for context; not directly filtered).
    polls_path : Path or None
        Path to the polls CSV.  Defaults to
        ``PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"``.

    Returns
    -------
    polls_by_race : dict[str, list[dict]]
        For the new forecast engine (run_forecast).
    poll_lookup : dict[str, list[tuple]]
        Legacy format used for unmatched-race warnings.
    """
    if polls_path is None:
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


def run_forecast_pipeline(
    *,
    year: int = 2026,
    params: ForecastParams | None = None,
    polls_path: Path | None = None,
    output_path: Path | None = None,
    reference_date: str | None = None,
    include_baseline: bool = True,
) -> pd.DataFrame:
    """Run the full type-primary forecast pipeline with configurable parameters.

    This is the primary public entry point for producing county-level forecasts.
    It loads type data, county priors, polls, and fundamentals, then runs the
    hierarchical forecast engine for all registered races.

    Parameters
    ----------
    year : int
        Election year.  Used to load the race registry (``load_races(year)``).
    params : ForecastParams or None
        Forecast hyperparameters.  When None, loads defaults from
        ``prediction_params.json`` via ``load_forecast_params()``.
    polls_path : Path or None
        Path to the polls CSV.  Defaults to ``data/polls/polls_{year}.csv``.
    output_path : Path or None
        Where to write the output parquet.  When None, writes to
        ``data/predictions/county_predictions_{year}_types.parquet``.
        Pass ``Path("/dev/null")`` to skip writing entirely.
    reference_date : str or None
        ISO date string for poll time-decay weighting.  Defaults to today.
    include_baseline : bool
        Whether to include a "baseline" race row (county priors with no polls).

    Returns
    -------
    pd.DataFrame
        County-level predictions with columns: county_fips, state,
        county_name, pred_dem_share, race, forecast_mode.
    """
    if params is None:
        params = load_forecast_params()

    if polls_path is None:
        polls_path = PROJECT_ROOT / "data" / "polls" / f"polls_{year}.csv"

    if output_path is None:
        output_path = (
            PROJECT_ROOT / "data" / "predictions"
            / f"county_predictions_{year}_types.parquet"
        )

    if reference_date is None:
        reference_date = str(date.today())

    county_fips, type_scores, type_covariance, type_priors = load_type_data()

    # Load race-type-specific county priors.  Governor races use priors trained
    # on governor outcomes; Senate and other races use presidential priors.
    # This corrects the structural mismatch from using presidential-trained priors
    # for governor forecasts (backtest showed presidential baseline r=0.770 beat
    # model r=0.715 because the priors were trained on the wrong target).
    county_prior_values_pres = load_county_priors_with_ridge(county_fips)
    county_prior_values_gov = load_county_priors_with_ridge_governor(county_fips)

    states, county_names = load_county_metadata(county_fips)
    county_votes = load_county_votes(county_fips)
    polls_by_race, poll_lookup = load_polls(county_fips, polls_path=polls_path)

    # Inject early-cycle election results (special elections, off-year races) as
    # additional poll-like observations before the Bayesian update.
    #
    # State-level results (WI SC → WI Governor, GA-14 → GA Senate) are merged
    # directly into polls_by_race; they'll flow through prepare_polls() and
    # receive normal time-decay and quality weighting.
    #
    # Generic Ballot results (VA/NJ 2025 governor) cannot go through
    # polls_by_race because compute_gb_shift() reads from a file, not from
    # the in-memory dict.  We extract them as (dem_share, n_sample) tuples
    # and pass them via the extra_gb_polls parameter added to compute_gb_shift().
    early_by_race = load_early_results()
    gb_early_polls = extract_gb_observations(early_by_race)
    polls_by_race = merge_early_results(polls_by_race, early_by_race)
    log.info(
        "Early results merged: %d GB observations, %d state-level races affected",
        len(gb_early_polls),
        sum(1 for r in early_by_race if r != "2026 Generic Ballot"),
    )

    # Behavior layer (τ/δ) is DISABLED pending a more sophisticated integration.
    # Backtest (governor-backtest-2022-S492.md) showed the flat δ adjustment REDUCES
    # correlation vs 2022 actuals (r 0.729 → 0.715). The mean δ = -0.011 (R shift)
    # hurts D-leaning states more than it helps R-leaning ones. The behavior layer
    # data (data/behavior/) and compute modules (src/behavior/) are retained for
    # future use when full τ-reweighted type composition adjustment is implemented.

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
    # Early-cycle GB observations (VA/NJ governor) are passed directly here
    # because compute_gb_shift() reads from a file path rather than the
    # in-memory polls dict — passing extra_gb_polls is cleaner than appending
    # to the CSV or duplicating the file-reading logic.
    gb_info = compute_gb_shift(extra_gb_polls=gb_early_polls if gb_early_polls else None)
    gb_shift = gb_info.shift
    log.info(
        "Generic ballot shift: %+.1f pp (%d polls, source=%s)",
        gb_shift * 100, gb_info.n_polls, gb_info.source,
    )

    # Optionally blend in the structural fundamentals shift (approval, GDP,
    # unemployment, CPI).  The combined shift is:
    #   shift = w * fundamentals + (1 - w) * generic_ballot
    # where w = fundamentals_weight from prediction_params.json.
    if params.fundamentals_enabled:
        try:
            snapshot = load_fundamentals_snapshot()
            fund_info = compute_fundamentals_shift(snapshot)
            w = params.fundamentals_weight
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

    # Apply state-level economic adjustment from QCEW data.
    # This modulates the national fundamentals shift by each state's relative
    # employment growth: states with faster job growth get a smaller in-party
    # penalty.  The result is a per-county shift array instead of a scalar.
    if params.state_econ_enabled:
        try:
            from src.prediction.state_economics import compute_state_econ_adjustment

            gb_shift = compute_state_econ_adjustment(
                county_fips=county_fips,
                states=states,
                national_shift=float(gb_shift) if isinstance(gb_shift, (int, float)) else float(gb_shift.mean()),
                econ_sensitivity=params.state_econ_sensitivity,
            )
            log.info(
                "State econ adjustment applied: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                gb_shift.mean(), gb_shift.std(), gb_shift.min(), gb_shift.max(),
            )
        except (FileNotFoundError, ImportError, ValueError) as exc:
            log.warning("State econ adjustment unavailable, using national shift: %s", exc)

    # Iterate over ALL registered races using the hierarchical forecast engine.
    # Produces dual-mode output: "national" (θ_national only) and
    # "local" (θ_national + δ_race) for every race.
    #
    # Governor races use governor-trained priors; Senate and other races use
    # presidential priors.  We split the race registry by type and call
    # run_forecast() twice — once per prior set — then merge results.
    from src.assembly.define_races import load_races

    registry = load_races(year)
    all_race_ids = [r.race_id for r in registry]
    governor_race_ids = [r.race_id for r in registry if r.race_type == "governor"]
    non_governor_race_ids = [r.race_id for r in registry if r.race_type != "governor"]

    log.info(
        "Race registry: %d races (%d governor, %d other)",
        len(registry),
        len(governor_race_ids),
        len(non_governor_race_ids),
    )

    # Build the shared kwargs to avoid repetition.
    shared_forecast_kwargs = dict(
        type_scores=type_scores,
        states=states,
        county_votes=county_votes,
        polls_by_race=polls_by_race,
        lam=params.lam,
        mu=params.mu,
        generic_ballot_shift=gb_shift,
        w_vector_mode=params.w_vector_mode,
        reference_date=reference_date,
        type_profiles=type_profiles_df,
        half_life_days=params.half_life_days,
        pre_primary_discount=params.pre_primary_discount,
        accuracy_path=params.accuracy_path,
        methodology_weights=params.methodology_weights,
        state_population_vectors=state_population_vectors,
        poll_blend_scale=params.poll_blend_scale,
        race_adjustments=params.race_adjustments,
    )

    forecast_results: dict = {}

    # Governor races: use governor Ridge priors (governor-trained).
    if governor_race_ids:
        log.info("Running governor forecast with governor-trained priors...")
        gov_results = run_forecast(
            county_priors=county_prior_values_gov,
            races=governor_race_ids,
            **shared_forecast_kwargs,
        )
        forecast_results.update(gov_results)

    # Senate and all other races: use presidential Ridge priors.
    if non_governor_race_ids:
        log.info("Running non-governor forecast with presidential priors...")
        other_results = run_forecast(
            county_priors=county_prior_values_pres,
            races=non_governor_race_ids,
            **shared_forecast_kwargs,
        )
        forecast_results.update(other_results)

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

    # Generate baseline: county Ridge priors without polls or GB shift.
    # With residual blending, baseline = priors + W @ (θ_prior - θ_prior) = priors.
    # This is the correct structural baseline: each county's own historical lean.
    if include_baseline:
        baseline_preds = county_prior_values_pres
        all_predictions.append(pd.DataFrame({
            "county_fips": county_fips,
            "state": states,
            "county_name": county_names,
            "pred_dem_share": baseline_preds,
            "race": "baseline",
            "forecast_mode": "national",
        }))

    output = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # Write output if a real path was given (not /dev/null).
    if str(output_path) != "/dev/null":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(output_path, index=False)
        log.info("Saved %d predictions to %s", len(output), output_path)

    return output


def run() -> None:
    """Load inputs from data/ and produce type-based 2026 predictions.

    Convenience wrapper around ``run_forecast_pipeline()`` with defaults
    for 2026.  Prints summary statistics to stdout.
    """
    output = run_forecast_pipeline(year=2026)

    out_path = (
        PROJECT_ROOT / "data" / "predictions"
        / "county_predictions_2026_types.parquet"
    )
    print(f"Saved {len(output)} county predictions to {out_path}")

    if len(output):
        print(output.groupby(["state", "race"])["pred_dem_share"].describe().round(3))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run()
