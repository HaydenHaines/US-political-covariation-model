"""Historic backtesting harness for the WetherVane prediction model.

This module runs the CURRENT (frozen) prediction model against historic elections
to evaluate model accuracy.  The model is never modified — we feed historic
state-level polls into the same forecast_engine.run_forecast() pipeline and
compare county predictions against county actuals.

Design principle: the model stays frozen.  Historic polls are translated into
the same dict format the forecast engine expects, then results are compared to
county-level election returns.

Supported race types and years:
  - president: 2008, 2012, 2016, 2020
  - senate:    2010, 2012, 2014, 2016, 2018, 2020, 2022
  - governor:  2010, 2018, 2022

Poll sources by race type:
  - President 2008-2016: 538 pres_pollaverages, two-party conversion D/(D+R)
  - President 2020+:     538 checking-our-work polls-plus final forecast
  - Senate/Governor:     538 checking-our-work polls-plus final forecast
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.county_priors import (
    load_county_priors_with_ridge,
    load_county_priors_with_ridge_governor,
)
from src.prediction.forecast_engine import run_forecast

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# State name ↔ abbreviation mappings
# ---------------------------------------------------------------------------

# Full state name → 2-letter abbreviation (used by 538 full-name states).
# Senate/Governor 538 data uses abbreviations; Presidential pollaverages uses full names.
_STATE_NAME_TO_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}

# Abbreviation → full name (reverse of above).
_STATE_ABBR_TO_NAME: dict[str, str] = {v: k for k, v in _STATE_NAME_TO_ABBR.items()}

# Dem candidates for pres_pollaverages (used to identify the D candidate per cycle).
_PRES_DEM_CANDIDATES: dict[int, str] = {
    2008: "Barack Obama",
    2012: "Barack Obama",
    2016: "Hillary Rodham Clinton",
}

# Rep candidates for pres_pollaverages (used to identify the R candidate per cycle
# for two-party dem share calculation).
_PRES_REP_CANDIDATES: dict[int, str] = {
    2008: "John McCain",
    2012: "Mitt Romney",
    2016: "Donald Trump",
}

# Default n_sample to use for state-level poll averages (no explicit sample size).
# A large number gives high weight; appropriate for final-day model averages.
_DEFAULT_N_SAMPLE = 2000

# Backtest years per race type
_PRESIDENTIAL_YEARS = [2008, 2012, 2016, 2020]
_SENATE_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
_GOVERNOR_YEARS = [2010, 2018, 2022]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _two_party_dem_share(dem: float, rep: float) -> float | None:
    """Compute two-party Democrat share: D / (D + R).

    Returns None when the denominator is zero or either value is NaN.

    The 538 projected_voteshare values represent full-electorate shares (may not
    sum to 100 due to third parties).  We strip third parties by dividing only
    by D+R total.
    """
    if pd.isna(dem) or pd.isna(rep):
        return None
    total = dem + rep
    if total <= 0:
        return None
    return float(dem / total)


def _load_538_final_forecast(
    csv_path: Path,
    year: int,
    race_type: str,
) -> dict[str, float]:
    """Load final 538 polls-plus forecast from checking-our-work CSV.

    Returns dict mapping state_abbr → two-party Dem share.

    Uses forecast_type='polls-plus' and the forecast date closest to election day
    (the idxmax of forecast_date per state/party).  The file uses state
    abbreviations for senate/governor and abbreviations for president in the
    checking-our-work dataset.

    Parameters
    ----------
    csv_path : Path
        Path to the 538 checking-our-work CSV (presidential_elections.csv,
        us_senate_elections.csv, or governors_elections.csv).
    year : int
        Election year to extract.
    race_type : str
        One of "president", "senate", "governor" — used only for logging.
    """
    df = pd.read_csv(csv_path)
    df = df[df["year"] == year].copy()

    # Some files have special election flag — exclude specials to focus on general.
    if "special" in df.columns:
        df = df[df["special"].isin([False, 0, "false", "False"]) | df["special"].isna()]

    # 538's forecast type naming varies by era and race type:
    # Presidential (2008–2020): "polls-plus", "polls-only", "now-cast"
    # Senate/Governor (2018+): "classic", "lite", "deluxe"
    # Use polls-plus or classic (whichever exists) as primary; fall back to all.
    preferred_types = {"polls-plus", "classic"}
    polls_plus = df[df["forecast_type"].isin(preferred_types)]
    if len(polls_plus) == 0:
        log.warning(
            "%s %d: no polls-plus/classic forecasts found, using all forecast types",
            race_type, year,
        )
        polls_plus = df

    # Convert forecast_date to datetime and take the row closest to election day
    # (idxmax over forecast_date per state+party combination).
    polls_plus = polls_plus.copy()
    polls_plus["forecast_date"] = pd.to_datetime(polls_plus["forecast_date"])
    idx = polls_plus.groupby(["state", "party"])["forecast_date"].idxmax()
    final = polls_plus.loc[idx].copy()

    # Build state → dem_share mapping using two-party conversion.
    # Normalize state identifiers: 538 data varies between full names ("Alabama")
    # and abbreviations ("AL") depending on year and race type.  We always output
    # abbreviations so that polls_by_race keys are consistent with actuals.
    dem_rows = final[final["party"] == "D"].set_index("state")
    rep_rows = final[final["party"] == "R"].set_index("state")

    state_dem_shares: dict[str, float] = {}
    for state_raw in dem_rows.index:
        if state_raw not in rep_rows.index:
            continue
        dem_pct = dem_rows.loc[state_raw, "projected_voteshare"]
        rep_pct = rep_rows.loc[state_raw, "projected_voteshare"]
        two_party = _two_party_dem_share(dem_pct, rep_pct)
        if two_party is None:
            continue

        # Normalize to 2-letter abbreviation.
        # If already 2 letters (abbreviation), use as-is.
        # If it's a full name, look it up in the mapping.
        # If the lookup fails, skip and warn.
        if len(state_raw) == 2 and state_raw.isupper():
            state_abbr = state_raw
        else:
            state_abbr = _STATE_NAME_TO_ABBR.get(state_raw)
            if state_abbr is None:
                log.debug("%s %d: unknown state '%s' — skipping", race_type, year, state_raw)
                continue

        state_dem_shares[state_abbr] = two_party

    log.info(
        "%s %d: loaded %d state forecasts from %s",
        race_type, year, len(state_dem_shares), csv_path.name,
    )
    return state_dem_shares


def _load_pres_poll_averages(year: int) -> dict[str, float]:
    """Load presidential poll averages from the pres_pollaverages_1968-2016 CSV.

    Returns dict mapping state_abbr → two-party Dem share (final model date).

    This CSV covers 1968–2016 and uses full state names.  We extract the final
    model date for the D and R candidates, compute two-party share, and convert
    full state names to abbreviations.

    Parameters
    ----------
    year : int
        Election cycle year (must be in 2008–2016).
    """
    csv_path = (
        PROJECT_ROOT
        / "data" / "raw" / "fivethirtyeight"
        / "data-repo" / "polls" / "pres_pollaverages_1968-2016.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"pres_pollaverages CSV not found: {csv_path}")

    dem_candidate = _PRES_DEM_CANDIDATES.get(year)
    rep_candidate = _PRES_REP_CANDIDATES.get(year)
    if dem_candidate is None or rep_candidate is None:
        raise ValueError(f"No candidate names configured for presidential year {year}")

    df = pd.read_csv(csv_path)
    df = df[df["cycle"] == year].copy()
    df["modeldate"] = pd.to_datetime(df["modeldate"])

    # Get the final model date for each candidate per state.
    dem_df = df[df["candidate_name"] == dem_candidate].copy()
    rep_df = df[df["candidate_name"] == rep_candidate].copy()

    if len(dem_df) == 0:
        raise ValueError(f"No data for D candidate '{dem_candidate}' in {year}")
    if len(rep_df) == 0:
        raise ValueError(f"No data for R candidate '{rep_candidate}' in {year}")

    # Take the row with the latest modeldate per state for each candidate.
    dem_final = dem_df.loc[dem_df.groupby("state")["modeldate"].idxmax()].set_index("state")
    rep_final = rep_df.loc[rep_df.groupby("state")["modeldate"].idxmax()].set_index("state")

    state_dem_shares: dict[str, float] = {}
    for full_name in dem_final.index:
        if full_name not in rep_final.index:
            continue
        abbr = _STATE_NAME_TO_ABBR.get(full_name)
        if abbr is None:
            log.warning("Unknown state name '%s' — skipping", full_name)
            continue
        dem_pct = dem_final.loc[full_name, "pct_estimate"]
        rep_pct = rep_final.loc[full_name, "pct_estimate"]
        two_party = _two_party_dem_share(dem_pct, rep_pct)
        if two_party is not None:
            state_dem_shares[abbr] = two_party

    log.info(
        "president %d (pollaverages): loaded %d state forecasts", year, len(state_dem_shares),
    )
    return state_dem_shares


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_historic_polls(year: int, race_type: str) -> dict[str, list[dict]]:
    """Load historic state-level poll averages from 538 data.

    For presidential 2008-2016: uses pres_pollaverages (final model date, two-party D/(D+R)).
    For presidential 2020:      uses checking-our-work polls-plus final forecast.
    For senate/governor:        uses checking-our-work polls-plus final forecast.

    Returns dict mapping race_name → list of poll dicts compatible with forecast_engine.
    Race name format: "{year} {state_abbr} {RaceType}", e.g. "2018 FL Governor".

    Parameters
    ----------
    year : int
        Election year.
    race_type : str
        One of "president", "senate", "governor".
    """
    race_type = race_type.lower()

    # --- Source selection ---
    if race_type == "president":
        if year in (2008, 2012, 2016):
            state_shares = _load_pres_poll_averages(year)
        elif year == 2020:
            csv_path = (
                PROJECT_ROOT
                / "data" / "raw" / "fivethirtyeight"
                / "checking-our-work-data" / "presidential_elections.csv"
            )
            state_shares = _load_538_final_forecast(csv_path, year, "president")
        else:
            raise ValueError(f"Unsupported presidential year: {year}")

    elif race_type == "senate":
        csv_path = (
            PROJECT_ROOT
            / "data" / "raw" / "fivethirtyeight"
            / "checking-our-work-data" / "us_senate_elections.csv"
        )
        state_shares = _load_538_final_forecast(csv_path, year, "senate")

    elif race_type == "governor":
        csv_path = (
            PROJECT_ROOT
            / "data" / "raw" / "fivethirtyeight"
            / "checking-our-work-data" / "governors_elections.csv"
        )
        state_shares = _load_538_final_forecast(csv_path, year, "governor")

    else:
        raise ValueError(f"Unknown race_type '{race_type}'. Must be president, senate, or governor.")

    # --- Build polls_by_race format ---
    # Race name convention: "{year} {state_abbr} {RaceType}"
    # e.g. "2020 FL President", "2018 FL Governor", "2022 PA Senate"
    race_type_label = race_type.capitalize()
    polls_by_race: dict[str, list[dict]] = {}
    for state_abbr, dem_share in state_shares.items():
        race_name = f"{year} {state_abbr} {race_type_label}"
        polls_by_race[race_name] = [{
            "state": state_abbr,
            "dem_share": dem_share,
            "n_sample": _DEFAULT_N_SAMPLE,
            "race": race_name,
            "date": f"{year}-11-01",  # Approximate election day
            "pollster": "538_model_average",
            "geo_level": "state",
        }]

    return polls_by_race


def load_historic_actuals(year: int, race_type: str) -> pd.DataFrame:
    """Load county-level actual results for a given year and race type.

    Returns a DataFrame with columns:
      - county_fips (str, 5-digit zero-padded)
      - state_abbr (str)
      - actual_dem_share (float)

    Rows with missing dem_share (NaN) or fips '00000' (state totals) are excluded.

    Parameters
    ----------
    year : int
        Election year.
    race_type : str
        One of "president", "senate", "governor".
    """
    race_type = race_type.lower()
    assembled_dir = PROJECT_ROOT / "data" / "assembled"

    if race_type == "president":
        path = assembled_dir / f"medsl_county_presidential_{year}.parquet"
        share_col = f"pres_dem_share_{year}"
    elif race_type == "senate":
        path = assembled_dir / f"medsl_county_senate_{year}.parquet"
        share_col = f"senate_dem_share_{year}"
    elif race_type == "governor":
        if year == 2022:
            path = assembled_dir / "medsl_county_2022_governor.parquet"
            share_col = "gov_dem_share_2022"
        else:
            path = assembled_dir / f"algara_county_governor_{year}.parquet"
            share_col = f"gov_dem_share_{year}"
    else:
        raise ValueError(f"Unknown race_type '{race_type}'. Must be president, senate, or governor.")

    if not path.exists():
        raise FileNotFoundError(f"Actuals file not found: {path}")

    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    if share_col not in df.columns:
        raise KeyError(f"Expected column '{share_col}' not found in {path}. Available: {df.columns.tolist()}")

    result = df[["county_fips", "state_abbr", share_col]].rename(
        columns={share_col: "actual_dem_share"}
    ).copy()

    # Drop state-level aggregate rows (FIPS '00000') and rows with missing data.
    result = result[result["county_fips"] != "00000"]
    result = result.dropna(subset=["actual_dem_share"])

    log.info(
        "Actuals %s %d: %d counties loaded from %s",
        race_type, year, len(result), path.name,
    )
    return result.reset_index(drop=True)


def _load_type_data_for_backtest() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load type assignments and covariance from the current (frozen) model.

    Returns
    -------
    county_fips : list[str]
        All county FIPS codes (zero-padded to 5 digits).
    type_scores : ndarray of shape (N, J)
        Soft type membership per county.
    type_covariance : ndarray of shape (J, J)
        Ledoit-Wolf regularized type covariance matrix.
    """
    type_assignments_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    type_cov_path = PROJECT_ROOT / "data" / "covariance" / "type_covariance.parquet"

    ta_df = pd.read_parquet(type_assignments_path)
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values

    cov_df = pd.read_parquet(type_cov_path)
    J = type_scores.shape[1]
    type_covariance = cov_df.values[:J, :J]

    log.info(
        "Loaded type data: %d counties, J=%d types", len(county_fips), J,
    )
    return county_fips, type_scores, type_covariance


def _county_metadata(county_fips: list[str]) -> list[str]:
    """Derive state abbreviation per county from FIPS codes.

    Uses the STATE_FIPS_TO_ABBR mapping from the core config module.
    """
    from src.core import config as cfg
    return [cfg.STATE_ABBR.get(f[:2], "??") for f in county_fips]


def _compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> dict:
    """Compute correlation, RMSE, and bias between predicted and actual dem share.

    Parameters
    ----------
    predictions : ndarray of shape (N,)
    actuals : ndarray of shape (N,)

    Returns
    -------
    dict with keys: r, rmse, bias, n
    """
    n = len(predictions)
    if n < 2:
        return {"r": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n": n}

    residuals = predictions - actuals
    bias = float(np.mean(residuals))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Pearson correlation — guard against constant arrays.
    pred_std = np.std(predictions)
    actual_std = np.std(actuals)
    if pred_std < 1e-9 or actual_std < 1e-9:
        r = float("nan")
    else:
        r = float(np.corrcoef(predictions, actuals)[0, 1])

    return {"r": r, "rmse": rmse, "bias": bias, "n": n}


def run_backtest(year: int, race_type: str) -> dict:
    """Run the current model against a historic election and return accuracy metrics.

    Pipeline:
      1. Load historic state-level polls → polls_by_race (forecast_engine format)
      2. Load current (frozen) type assignments, covariance, and county priors
      3. Call run_forecast() with historic polls
      4. Load county-level actuals for the year
      5. Merge predictions with actuals; compute accuracy metrics

    The model is not modified in any way.  For governor races, governor Ridge priors
    are used (blended 70/30 with presidential priors per the calibrated _GOVERNOR_BLEND_WEIGHT).

    Returns
    -------
    dict with keys:
      year, race_type, n_races, n_counties,
      overall_r, overall_rmse, overall_bias,
      per_state: list of dicts with:
        state, r, rmse, bias, n_counties,
        pred_state_dem, actual_state_dem, direction_correct
    """
    race_type = race_type.lower()
    log.info("Running backtest: %s %d", race_type, year)

    # Load polls (state-level forecasts treated as polls).
    polls_by_race = load_historic_polls(year, race_type)
    if not polls_by_race:
        log.warning("%s %d: no polls loaded, aborting backtest", race_type, year)
        return {"year": year, "race_type": race_type, "error": "no_polls"}

    # Extract which states had races (to build the races list).
    race_ids = list(polls_by_race.keys())
    states_with_races = {race_id.split(" ")[1] for race_id in race_ids}
    log.info("%s %d: %d states have races", race_type, year, len(states_with_races))

    # Load current (frozen) model type data.
    county_fips, type_scores, _ = _load_type_data_for_backtest()
    states = _county_metadata(county_fips)

    # County vote counts for W vector construction; use equal weights as fallback.
    # 2024 presidential vote counts are used regardless of race type — this is correct
    # because they reflect population size, not outcome, and the model is frozen on 2024.
    county_votes_arr = np.ones(len(county_fips))
    votes_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    if votes_path.exists():
        vdf = pd.read_parquet(votes_path)
        if "county_fips" in vdf.columns and "pres_total_2024" in vdf.columns:
            vmap = dict(zip(
                vdf["county_fips"].astype(str).str.zfill(5),
                vdf["pres_total_2024"],
            ))
            county_votes_arr = np.array([float(vmap.get(f, 1.0)) for f in county_fips])
        elif "county_fips" in vdf.columns and "totalvotes" in vdf.columns:
            vmap = dict(zip(
                vdf["county_fips"].astype(str).str.zfill(5),
                vdf["totalvotes"],
            ))
            county_votes_arr = np.array([float(vmap.get(f, 1.0)) for f in county_fips])

    # Load county priors from the current (frozen) Ridge model.
    # Governor races use the blended governor/presidential priors.
    if race_type == "governor":
        county_priors = load_county_priors_with_ridge_governor(county_fips)
    else:
        county_priors = load_county_priors_with_ridge(county_fips)

    # Run the frozen forecast engine with historic polls.
    # No generic ballot shift, no fundamentals: the poll itself IS the signal.
    # reference_date is approximate (day before election) to allow time decay to be
    # essentially flat (all polls have the same "election eve" date).
    reference_date = f"{year}-11-01"
    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes_arr,
        polls_by_race=polls_by_race,
        races=race_ids,
        lam=1.0,
        mu=1.0,
        generic_ballot_shift=0.0,
        w_vector_mode="core",
        reference_date=reference_date,
    )

    # Load county-level actuals for the year.
    actuals_df = load_historic_actuals(year, race_type)

    # For each race, merge predicted county dem_share with actual.
    # The model produces county predictions for each race; we match by state.
    all_pred: list[float] = []
    all_actual: list[float] = []
    per_state_results: list[dict] = []

    for race_id, fr in forecast_results.items():
        # Race id format: "{year} {state_abbr} {RaceType}"
        parts = race_id.split(" ")
        if len(parts) < 3:
            continue
        state_abbr = parts[1]

        # Filter actuals to this state.
        state_actuals = actuals_df[actuals_df["state_abbr"] == state_abbr].copy()
        if len(state_actuals) == 0:
            log.debug("%s: no actuals for state %s — skipping", race_id, state_abbr)
            continue

        # Build county_fips → pred_dem_share from the "national" mode forecast.
        # "national" mode: theta_national-based, no candidate effects (delta_race).
        # This is the correct mode for backtesting the structural model.
        fips_to_pred = dict(zip(county_fips, fr.county_preds_national))

        state_actuals = state_actuals.copy()
        state_actuals["pred_dem_share"] = state_actuals["county_fips"].map(fips_to_pred)
        state_actuals = state_actuals.dropna(subset=["pred_dem_share", "actual_dem_share"])

        if len(state_actuals) == 0:
            continue

        preds_arr = state_actuals["pred_dem_share"].values
        actuals_arr = state_actuals["actual_dem_share"].values

        # State-level vote-weighted aggregate Dem share (unweighted mean as approximation).
        pred_state_dem = float(np.mean(preds_arr))
        actual_state_dem = float(np.mean(actuals_arr))
        direction_correct = (pred_state_dem > 0.5) == (actual_state_dem > 0.5)

        state_metrics = _compute_metrics(preds_arr, actuals_arr)
        per_state_results.append({
            "state": state_abbr,
            "r": state_metrics["r"],
            "rmse": state_metrics["rmse"],
            "bias": state_metrics["bias"],
            "n_counties": state_metrics["n"],
            "pred_state_dem": pred_state_dem,
            "actual_state_dem": actual_state_dem,
            "direction_correct": direction_correct,
        })

        all_pred.extend(preds_arr.tolist())
        all_actual.extend(actuals_arr.tolist())

    if not all_pred:
        log.warning("%s %d: no matched counties — backtest produced no results", race_type, year)
        return {"year": year, "race_type": race_type, "error": "no_matched_counties"}

    overall = _compute_metrics(np.array(all_pred), np.array(all_actual))
    n_direction_correct = sum(1 for s in per_state_results if s["direction_correct"])
    n_total_states = len(per_state_results)

    result = {
        "year": year,
        "race_type": race_type,
        "n_races": len(per_state_results),
        "n_counties": len(all_pred),
        "overall_r": overall["r"],
        "overall_rmse": overall["rmse"],
        "overall_bias": overall["bias"],
        "direction_accuracy": n_direction_correct / n_total_states if n_total_states > 0 else float("nan"),
        "per_state": per_state_results,
    }

    log.info(
        "%s %d: r=%.3f, RMSE=%.4f, bias=%+.4f (%d states, %d counties)",
        race_type, year,
        overall["r"], overall["rmse"], overall["bias"],
        len(per_state_results), len(all_pred),
    )
    return result


def run_all_backtests() -> list[dict]:
    """Run backtests for all available historic elections.

    Presidential: 2008, 2012, 2016, 2020
    Senate:       2010, 2012, 2014, 2016, 2018, 2020, 2022
    Governor:     2010, 2018, 2022

    Returns list of backtest result dicts (one per year/race_type combination).
    Failed backtests (error key present) are included but logged as warnings.
    """
    all_results: list[dict] = []

    for year in _PRESIDENTIAL_YEARS:
        try:
            result = run_backtest(year, "president")
            all_results.append(result)
        except Exception as exc:
            log.error("Backtest failed: president %d — %s", year, exc)
            all_results.append({"year": year, "race_type": "president", "error": str(exc)})

    for year in _SENATE_YEARS:
        try:
            result = run_backtest(year, "senate")
            all_results.append(result)
        except Exception as exc:
            log.error("Backtest failed: senate %d — %s", year, exc)
            all_results.append({"year": year, "race_type": "senate", "error": str(exc)})

    for year in _GOVERNOR_YEARS:
        try:
            result = run_backtest(year, "governor")
            all_results.append(result)
        except Exception as exc:
            log.error("Backtest failed: governor %d — %s", year, exc)
            all_results.append({"year": year, "race_type": "governor", "error": str(exc)})

    return all_results


def _print_summary_table(results: list[dict]) -> None:
    """Print a compact summary table of backtest results to stdout."""
    header = f"{'Race':<22}  {'r':>6}  {'RMSE':>7}  {'Bias':>7}  {'DirAcc':>7}  {'States':>6}  {'Counties':>8}"
    print()
    print(header)
    print("-" * len(header))

    for r in results:
        race_label = f"{r['race_type'].capitalize()} {r['year']}"
        if "error" in r:
            print(f"{race_label:<22}  ERROR: {r['error']}")
            continue
        overall_r = r.get("overall_r", float("nan"))
        rmse = r.get("overall_rmse", float("nan"))
        bias = r.get("overall_bias", float("nan"))
        dir_acc = r.get("direction_accuracy", float("nan"))
        n_states = r.get("n_races", 0)
        n_counties = r.get("n_counties", 0)
        print(
            f"{race_label:<22}  {overall_r:>6.3f}  {rmse:>7.4f}  {bias:>+7.4f}  "
            f"{dir_acc:>6.1%}  {n_states:>6}  {n_counties:>8}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WetherVane historic backtest harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/validation/backtest_harness.py --year 2020 --race-type president
  python src/validation/backtest_harness.py --year 2022 --race-type senate
  python src/validation/backtest_harness.py --all
        """,
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Election year (required unless --all)",
    )
    parser.add_argument(
        "--race-type", type=str, default=None,
        choices=["president", "senate", "governor"],
        help="Race type (required unless --all)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all available backtests and print summary table",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-state results",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import sys

    # Ensure the project root is in sys.path so src imports work.
    _proj_root = str(Path(__file__).resolve().parents[2])
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

    # Re-import after path fix (no-op if already available).
    args = _build_arg_parser().parse_args()

    if args.all:
        print("Running all historic backtests...")
        results = run_all_backtests()
        _print_summary_table(results)

    elif args.year is not None and args.race_type is not None:
        result = run_backtest(args.year, args.race_type)
        results = [result]
        _print_summary_table(results)

        if args.verbose and "per_state" in result:
            print(f"\nPer-state results for {args.race_type.capitalize()} {args.year}:")
            per_state = sorted(result["per_state"], key=lambda x: x["r"], reverse=True)
            hdr = f"  {'State':>5}  {'r':>6}  {'RMSE':>7}  {'Bias':>7}  {'PredDem':>8}  {'ActualDem':>9}  {'DirOK':>6}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for s in per_state:
                print(
                    f"  {s['state']:>5}  {s['r']:>6.3f}  {s['rmse']:>7.4f}  "
                    f"{s['bias']:>+7.4f}  {s['pred_state_dem']:>8.3f}  "
                    f"{s['actual_state_dem']:>9.3f}  {'yes' if s['direction_correct'] else 'no':>6}"
                )

    else:
        _build_arg_parser().print_help()
        raise SystemExit(1)
