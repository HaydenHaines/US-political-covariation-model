"""Parameter sweep framework for optimizing forecast engine hyperparameters.

Runs the forecast engine against 11 historic elections (2008–2022) with
parameterized controls for:
  - lam: θ_national regularization
  - mu: δ_race regularization
  - poll_blend_scale: the k parameter in alpha = 1/(1 + n_polls/k)
  - use_year_adaptive_priors: use prior-election actuals instead of 2024 Ridge priors

The key insight is year-adaptive priors: for backtesting year Y, use the most
recent presidential election BEFORE Y as county priors.  This eliminates the
temporal gradient artifact (r=0.826 in 2008 → r=0.919 in 2020) caused by the
2024-trained Ridge model being more informative for recent elections.

Usage:
  uv run python -m src.validation.backtest_sweep --compare-priors
  uv run python -m src.validation.backtest_sweep --quick
  uv run python -m src.validation.backtest_sweep --full
"""
from __future__ import annotations

import argparse
import itertools
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.forecast_engine import run_forecast
from src.validation.backtest_harness import (
    _GOVERNOR_YEARS,
    _PRESIDENTIAL_YEARS,
    _SENATE_YEARS,
    _compute_metrics,
    _county_metadata,
    _load_type_data_for_backtest,
    load_historic_actuals,
    load_historic_polls,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Available presidential election years with county-level actuals on disk.
# Used to find the most recent prior election for year-adaptive priors.
_PRESIDENTIAL_ACTUALS_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]

# Default fallback when a county has no historical prior.
# Slightly R-leaning, consistent with national presidential baselines.
_FALLBACK_DEM_SHARE = 0.45

# Default race configs: all 14 elections the harness supports.
_ALL_RACE_CONFIGS: list[tuple[int, str]] = (
    [(y, "president") for y in _PRESIDENTIAL_YEARS]
    + [(y, "senate") for y in _SENATE_YEARS]
    + [(y, "governor") for y in _GOVERNOR_YEARS]
)


# ---------------------------------------------------------------------------
# Year-adaptive priors
# ---------------------------------------------------------------------------

def _find_prior_presidential_year(target_year: int) -> int | None:
    """Return the most recent presidential election year strictly before target_year.

    Returns None if no suitable year exists on disk.
    """
    candidates = [y for y in _PRESIDENTIAL_ACTUALS_YEARS if y < target_year]
    return max(candidates) if candidates else None


def build_blended_priors(
    county_fips: list[str],
    target_year: int,
    prior_decay: float = 0.0,
    assembled_dir: Path | None = None,
) -> np.ndarray:
    """Load and blend all prior presidential actuals before target_year.

    Each historical election is weighted by exponential decay based on its
    distance (in 4-year election cycles) from target_year:

        weight_i = prior_decay ^ (distance_i / 4)

    where distance_i = target_year - election_year_i (positive integer).

    Weights are then normalized to sum to 1.

    Special cases:
    - prior_decay=0.0 → only the most recent election gets non-zero weight,
      producing results identical to build_year_adaptive_priors().
    - prior_decay=1.0 → all prior elections weighted equally.
    - prior_decay=0.5 → 4 years back gets 0.5 weight relative to most recent.

    Missing counties fall back to _FALLBACK_DEM_SHARE in each election's
    data, so the blended value is a weighted average of whatever coverage
    exists across the available election files.

    Parameters
    ----------
    county_fips : list[str]
        Zero-padded 5-digit FIPS codes defining the county ordering.
    target_year : int
        The election year being backtested. Only elections STRICTLY before
        this year are included.
    prior_decay : float
        Exponential decay base in [0, 1].  See decay formula above.
    assembled_dir : Path or None
        Directory containing MEDSL presidential parquet files.

    Returns
    -------
    ndarray of shape (N,)
        Blended prior Dem share per county.
    """
    if assembled_dir is None:
        assembled_dir = PROJECT_ROOT / "data" / "assembled"

    # All prior years strictly before target_year, sorted ascending (oldest first).
    prior_years = sorted(y for y in _PRESIDENTIAL_ACTUALS_YEARS if y < target_year)

    if not prior_years:
        log.warning(
            "No presidential actuals before %d — falling back to %.2f for all counties",
            target_year, _FALLBACK_DEM_SHARE,
        )
        return np.full(len(county_fips), _FALLBACK_DEM_SHARE)

    # When prior_decay=0.0, only the most recent election contributes.
    # This exactly replicates build_year_adaptive_priors() behavior.
    if prior_decay == 0.0:
        return build_year_adaptive_priors(county_fips, target_year, assembled_dir=assembled_dir)

    # Compute unnormalized weights: w_i = prior_decay ^ ((target_year - year_i) / 4)
    # The exponent is in election-cycle units (4 years = 1 cycle).
    # More recent elections get a higher weight because their exponent is smaller.
    distances = [target_year - y for y in prior_years]  # positive integers
    raw_weights = np.array([prior_decay ** (d / 4) for d in distances], dtype=float)
    weights = raw_weights / raw_weights.sum()

    # Load per-year share arrays, blending as we go.
    blended = np.zeros(len(county_fips), dtype=float)

    for year, weight in zip(prior_years, weights):
        path = assembled_dir / f"medsl_county_presidential_{year}.parquet"
        share_col = f"pres_dem_share_{year}"

        if not path.exists():
            log.warning(
                "Blended prior: file missing for %d (%s) — substituting %.2f for all counties",
                year, path, _FALLBACK_DEM_SHARE,
            )
            blended += weight * _FALLBACK_DEM_SHARE
            continue

        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

        if share_col not in df.columns:
            raise KeyError(
                f"Expected column '{share_col}' not in {path}. Got: {df.columns.tolist()}"
            )

        fips_to_share = dict(zip(df["county_fips"], df[share_col]))
        year_shares = np.array([
            float(fips_to_share.get(f, _FALLBACK_DEM_SHARE))
            if not pd.isna(fips_to_share.get(f, _FALLBACK_DEM_SHARE))
            else _FALLBACK_DEM_SHARE
            for f in county_fips
        ])
        blended += weight * year_shares

    n_years = len(prior_years)
    log.info(
        "Blended priors for %d: decay=%.2f, %d prior elections %s, weights=%s",
        target_year, prior_decay, n_years, prior_years,
        [f"{w:.3f}" for w in weights],
    )
    return blended


def build_year_adaptive_priors(
    county_fips: list[str],
    target_year: int,
    assembled_dir: Path | None = None,
) -> np.ndarray:
    """Load the most recent presidential actuals before target_year as county priors.

    For backtesting year Y, the prior-election actuals (e.g. 2004 for Y=2008)
    serve as a much better baseline than the 2024 Ridge model, which introduces
    a temporal information leak.

    Parameters
    ----------
    county_fips : list[str]
        Zero-padded 5-digit FIPS codes defining the county ordering.
    target_year : int
        The election year being backtested.
    assembled_dir : Path or None
        Directory containing MEDSL presidential parquet files.

    Returns
    -------
    ndarray of shape (N,)
        Prior Dem share per county.  Missing counties get _FALLBACK_DEM_SHARE.
    """
    if assembled_dir is None:
        assembled_dir = PROJECT_ROOT / "data" / "assembled"

    prior_year = _find_prior_presidential_year(target_year)
    if prior_year is None:
        log.warning(
            "No presidential actuals before %d — falling back to %.2f for all counties",
            target_year, _FALLBACK_DEM_SHARE,
        )
        return np.full(len(county_fips), _FALLBACK_DEM_SHARE)

    path = assembled_dir / f"medsl_county_presidential_{prior_year}.parquet"
    if not path.exists():
        log.warning(
            "Presidential actuals file missing: %s — falling back to %.2f",
            path, _FALLBACK_DEM_SHARE,
        )
        return np.full(len(county_fips), _FALLBACK_DEM_SHARE)

    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    share_col = f"pres_dem_share_{prior_year}"

    if share_col not in df.columns:
        raise KeyError(f"Expected column '{share_col}' not in {path}. Got: {df.columns.tolist()}")

    fips_to_share = dict(zip(df["county_fips"], df[share_col]))

    priors = np.array([
        float(fips_to_share.get(f, _FALLBACK_DEM_SHARE))
        if not pd.isna(fips_to_share.get(f, _FALLBACK_DEM_SHARE))
        else _FALLBACK_DEM_SHARE
        for f in county_fips
    ])

    n_found = sum(1 for f in county_fips if f in fips_to_share)
    log.info(
        "Year-adaptive priors for %d: using %d actuals, %d/%d counties matched (%.1f%%)",
        target_year, prior_year, n_found, len(county_fips),
        100 * n_found / len(county_fips) if county_fips else 0,
    )
    return priors


# ---------------------------------------------------------------------------
# Parameterized backtest runner
# ---------------------------------------------------------------------------

def _load_county_votes(county_fips: list[str]) -> np.ndarray:
    """Load 2024 presidential vote counts for population weighting.

    Falls back to equal weights if the file is missing.
    """
    votes_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    if not votes_path.exists():
        return np.ones(len(county_fips))

    vdf = pd.read_parquet(votes_path)
    if "county_fips" not in vdf.columns:
        return np.ones(len(county_fips))

    vdf["county_fips"] = vdf["county_fips"].astype(str).str.zfill(5)
    total_col = "pres_total_2024" if "pres_total_2024" in vdf.columns else "totalvotes"
    if total_col not in vdf.columns:
        return np.ones(len(county_fips))

    vmap = dict(zip(vdf["county_fips"], vdf[total_col]))
    return np.array([float(vmap.get(f, 1.0)) for f in county_fips])


def run_backtest_with_params(
    year: int,
    race_type: str,
    params: dict,
) -> dict:
    """Run a single backtest with overridable forecast engine parameters.

    This is a parameterized version of backtest_harness.run_backtest() that
    accepts tuning parameters instead of using hardcoded defaults.

    Parameters
    ----------
    year : int
        Election year to backtest.
    race_type : str
        One of "president", "senate", "governor".
    params : dict
        Override parameters. Supported keys:
          lam (float): θ_national regularization. Default 1.0.
          mu (float): δ_race regularization. Default 1.0.
          poll_blend_scale (float): k in alpha = 1/(1+n/k). Default 5.0.
          use_year_adaptive_priors (bool): Use prior-election actuals. Default False.
          use_local_preds (bool): Use county_preds_local (with δ_race) for
              per-race accuracy. Default True.
          prior_decay (float): Exponential decay base for blending multiple
              prior elections.  0.0 = most-recent-only (default).  1.0 =
              equal weight across all prior elections.  Only used when
              use_year_adaptive_priors=True.
          half_life_days (float): Poll time-decay half-life in days.
              Default 30.0.  Passed directly to run_forecast().

    Returns
    -------
    dict with keys: year, race_type, r, rmse, bias, direction_accuracy, n_counties, n_races, params
    """
    race_type = race_type.lower()

    # Extract params with defaults.
    lam = params.get("lam", 1.0)
    mu = params.get("mu", 1.0)
    poll_blend_scale = params.get("poll_blend_scale", 5.0)
    use_year_adaptive = params.get("use_year_adaptive_priors", False)
    use_local = params.get("use_local_preds", True)
    prior_decay = params.get("prior_decay", 0.0)
    half_life_days = params.get("half_life_days", 30.0)

    # Load polls.
    polls_by_race = load_historic_polls(year, race_type)
    if not polls_by_race:
        return {
            "year": year, "race_type": race_type,
            "r": float("nan"), "rmse": float("nan"),
            "bias": float("nan"), "direction_accuracy": float("nan"),
            "n_counties": 0, "n_races": 0, "params": params,
            "error": "no_polls",
        }

    race_ids = list(polls_by_race.keys())

    # Load shared model infrastructure (type scores, covariance — frozen).
    county_fips, type_scores, _ = _load_type_data_for_backtest()
    states = _county_metadata(county_fips)
    county_votes_arr = _load_county_votes(county_fips)

    # Load county priors: year-adaptive (single or blended) or frozen Ridge.
    if use_year_adaptive:
        # prior_decay > 0 → weighted blend of all prior elections.
        # prior_decay == 0 → single most-recent election (original behavior).
        county_priors = build_blended_priors(county_fips, year, prior_decay=prior_decay)
    else:
        # Frozen 2024 Ridge priors (same as default backtest_harness behavior).
        from src.prediction.county_priors import (
            load_county_priors_with_ridge,
            load_county_priors_with_ridge_governor,
        )
        if race_type == "governor":
            county_priors = load_county_priors_with_ridge_governor(county_fips)
        else:
            county_priors = load_county_priors_with_ridge(county_fips)

    reference_date = f"{year}-11-01"

    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes_arr,
        polls_by_race=polls_by_race,
        races=race_ids,
        lam=lam,
        mu=mu,
        generic_ballot_shift=0.0,
        w_vector_mode="core",
        reference_date=reference_date,
        poll_blend_scale=poll_blend_scale,
        half_life_days=half_life_days,
    )

    # Load actuals and compute metrics.
    actuals_df = load_historic_actuals(year, race_type)

    all_pred: list[float] = []
    all_actual: list[float] = []
    n_direction_correct = 0
    n_states = 0

    for race_id, fr in forecast_results.items():
        parts = race_id.split(" ")
        if len(parts) < 3:
            continue
        state_abbr = parts[1]

        state_actuals = actuals_df[actuals_df["state_abbr"] == state_abbr].copy()
        if len(state_actuals) == 0:
            continue

        # Choose prediction mode: local (with δ_race) or national (structural only).
        preds_source = fr.county_preds_local if use_local else fr.county_preds_national
        fips_to_pred = dict(zip(county_fips, preds_source))

        state_actuals = state_actuals.copy()
        state_actuals["pred_dem_share"] = state_actuals["county_fips"].map(fips_to_pred)
        state_actuals = state_actuals.dropna(subset=["pred_dem_share", "actual_dem_share"])

        if len(state_actuals) == 0:
            continue

        preds_arr = state_actuals["pred_dem_share"].values
        actuals_arr = state_actuals["actual_dem_share"].values

        pred_state = float(np.mean(preds_arr))
        actual_state = float(np.mean(actuals_arr))
        if (pred_state > 0.5) == (actual_state > 0.5):
            n_direction_correct += 1
        n_states += 1

        all_pred.extend(preds_arr.tolist())
        all_actual.extend(actuals_arr.tolist())

    if not all_pred:
        return {
            "year": year, "race_type": race_type,
            "r": float("nan"), "rmse": float("nan"),
            "bias": float("nan"), "direction_accuracy": float("nan"),
            "n_counties": 0, "n_races": 0, "params": params,
            "error": "no_matched_counties",
        }

    metrics = _compute_metrics(np.array(all_pred), np.array(all_actual))
    dir_acc = n_direction_correct / n_states if n_states > 0 else float("nan")

    return {
        "year": year,
        "race_type": race_type,
        "r": metrics["r"],
        "rmse": metrics["rmse"],
        "bias": metrics["bias"],
        "direction_accuracy": dir_acc,
        "n_counties": len(all_pred),
        "n_races": n_states,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Parameter grid sweep
# ---------------------------------------------------------------------------

def sweep_parameters(
    param_grid: dict[str, list],
    race_configs: list[tuple[int, str]] | None = None,
) -> pd.DataFrame:
    """Run a grid sweep over forecast engine parameters across historic elections.

    Parameters
    ----------
    param_grid : dict
        Maps parameter names to lists of values to try.
        Example: {"lam": [0.5, 1.0, 2.0], "mu": [0.5, 1.0], "poll_blend_scale": [3.0, 5.0, 10.0]}
    race_configs : list of (year, race_type) or None
        Which elections to backtest.  Defaults to all 14.

    Returns
    -------
    DataFrame with columns: [param values..., year, race_type, r, rmse, bias, direction_accuracy]
    """
    if race_configs is None:
        race_configs = _ALL_RACE_CONFIGS

    # Build all parameter combinations from the grid.
    param_names = sorted(param_grid.keys())
    param_values_lists = [param_grid[name] for name in param_names]
    combos = list(itertools.product(*param_values_lists))
    n_combos = len(combos)
    n_races = len(race_configs)
    total = n_combos * n_races

    print(f"Sweep: {n_combos} param combos × {n_races} elections = {total} runs")
    print(f"Parameters: {param_names}")

    rows: list[dict] = []
    start = time.time()

    for combo_idx, combo in enumerate(combos):
        params = dict(zip(param_names, combo))

        for race_idx, (year, race_type) in enumerate(race_configs):
            run_num = combo_idx * n_races + race_idx + 1
            if run_num % 10 == 0 or run_num == 1:
                elapsed = time.time() - start
                rate = run_num / elapsed if elapsed > 0 else 0
                eta = (total - run_num) / rate if rate > 0 else 0
                print(
                    f"  [{run_num}/{total}] {race_type} {year} "
                    f"| {_format_params(params)} "
                    f"| {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
                )

            result = run_backtest_with_params(year, race_type, params)

            row = {name: val for name, val in zip(param_names, combo)}
            row["year"] = year
            row["race_type"] = race_type
            row["r"] = result["r"]
            row["rmse"] = result["rmse"]
            row["bias"] = result["bias"]
            row["direction_accuracy"] = result["direction_accuracy"]
            rows.append(row)

    elapsed = time.time() - start
    print(f"Sweep complete: {total} runs in {elapsed:.1f}s ({elapsed/total:.2f}s/run)")

    return pd.DataFrame(rows)


def _format_params(params: dict) -> str:
    """Format parameter dict as a compact string for progress output."""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, bool):
            parts.append(f"{k}={'Y' if v else 'N'}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Prior comparison
# ---------------------------------------------------------------------------

def compare_priors(
    race_configs: list[tuple[int, str]] | None = None,
) -> pd.DataFrame:
    """Run all elections twice — Ridge priors vs year-adaptive — and return comparison.

    Returns a DataFrame with columns:
      year, race_type, ridge_r, ridge_rmse, ridge_bias,
      adaptive_r, adaptive_rmse, adaptive_bias, r_delta, rmse_delta
    """
    if race_configs is None:
        race_configs = _ALL_RACE_CONFIGS

    rows: list[dict] = []
    total = len(race_configs)

    for i, (year, race_type) in enumerate(race_configs):
        print(f"  [{i+1}/{total}] {race_type.capitalize()} {year}...", end=" ", flush=True)

        ridge_result = run_backtest_with_params(
            year, race_type, {"use_year_adaptive_priors": False},
        )
        adaptive_result = run_backtest_with_params(
            year, race_type, {"use_year_adaptive_priors": True},
        )

        print(
            f"Ridge r={ridge_result['r']:.3f} → Adaptive r={adaptive_result['r']:.3f} "
            f"(Δ={adaptive_result['r'] - ridge_result['r']:+.3f})"
        )

        rows.append({
            "year": year,
            "race_type": race_type,
            "ridge_r": ridge_result["r"],
            "ridge_rmse": ridge_result["rmse"],
            "ridge_bias": ridge_result["bias"],
            "ridge_dir_acc": ridge_result["direction_accuracy"],
            "adaptive_r": adaptive_result["r"],
            "adaptive_rmse": adaptive_result["rmse"],
            "adaptive_bias": adaptive_result["bias"],
            "adaptive_dir_acc": adaptive_result["direction_accuracy"],
            "r_delta": adaptive_result["r"] - ridge_result["r"],
            "rmse_delta": adaptive_result["rmse"] - ridge_result["rmse"],
        })

    return pd.DataFrame(rows)


def _print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted comparison table of Ridge vs year-adaptive priors."""
    print()
    header = (
        f"{'Race':<22}  {'Ridge r':>8}  {'Adapt r':>8}  {'Δr':>7}  "
        f"{'Ridge RMSE':>10}  {'Adapt RMSE':>10}  {'ΔRMSE':>8}"
    )
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        label = f"{row['race_type'].capitalize()} {row['year']}"
        print(
            f"{label:<22}  {row['ridge_r']:>8.3f}  {row['adaptive_r']:>8.3f}  "
            f"{row['r_delta']:>+7.3f}  {row['ridge_rmse']:>10.4f}  "
            f"{row['adaptive_rmse']:>10.4f}  {row['rmse_delta']:>+8.4f}"
        )

    # Summary row.
    print("-" * len(header))
    mean_ridge_r = df["ridge_r"].mean()
    mean_adapt_r = df["adaptive_r"].mean()
    mean_ridge_rmse = df["ridge_rmse"].mean()
    mean_adapt_rmse = df["adaptive_rmse"].mean()
    print(
        f"{'MEAN':<22}  {mean_ridge_r:>8.3f}  {mean_adapt_r:>8.3f}  "
        f"{mean_adapt_r - mean_ridge_r:>+7.3f}  {mean_ridge_rmse:>10.4f}  "
        f"{mean_adapt_rmse:>10.4f}  {mean_adapt_rmse - mean_ridge_rmse:>+8.4f}"
    )
    print()


def _print_sweep_summary(df: pd.DataFrame, param_names: list[str]) -> None:
    """Print a summary of sweep results grouped by parameter combination."""
    print()
    print("=== Sweep Summary (mean across elections) ===")

    grouped = df.groupby(param_names).agg(
        mean_r=("r", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_bias=("bias", "mean"),
        mean_dir_acc=("direction_accuracy", "mean"),
    ).reset_index().sort_values("mean_r", ascending=False)

    header_parts = [f"{name:>12}" for name in param_names]
    header = "  ".join(header_parts) + f"  {'mean_r':>8}  {'mean_RMSE':>10}  {'mean_bias':>10}  {'dir_acc':>8}"
    print(header)
    print("-" * len(header))

    for _, row in grouped.head(20).iterrows():
        parts = [f"{row[name]:>12.3f}" if isinstance(row[name], float) else f"{row[name]!s:>12}" for name in param_names]
        print(
            "  ".join(parts)
            + f"  {row['mean_r']:>8.3f}  {row['mean_rmse']:>10.4f}  "
            f"{row['mean_bias']:>+10.4f}  {row['mean_dir_acc']:>7.1%}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WetherVane backtest parameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.validation.backtest_sweep --compare-priors
  python -m src.validation.backtest_sweep --quick
  python -m src.validation.backtest_sweep --full
        """,
    )
    parser.add_argument(
        "--compare-priors", action="store_true",
        help="Compare Ridge priors vs year-adaptive priors for all elections",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a small parameter grid on a subset of elections",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the full parameter grid on all elections",
    )
    return parser


def _quick_grid() -> tuple[dict[str, list], list[tuple[int, str]]]:
    """Small grid for quick iteration: 3-5 values per param, subset of elections."""
    param_grid = {
        "lam": [0.5, 1.0, 3.0],
        "mu": [0.5, 1.0, 3.0],
        "poll_blend_scale": [3.0, 5.0, 10.0],
        "use_year_adaptive_priors": [True, False],
        "prior_decay": [0.0, 0.3, 0.5],
        "half_life_days": [15.0, 30.0, 60.0],
    }
    # Subset: 2 presidential + 2 senate + 1 governor = 5 elections.
    race_configs = [
        (2008, "president"),
        (2020, "president"),
        (2016, "senate"),
        (2022, "senate"),
        (2018, "governor"),
    ]
    return param_grid, race_configs


def _full_grid() -> tuple[dict[str, list], list[tuple[int, str]]]:
    """Full grid for thorough optimization."""
    param_grid = {
        "lam": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
        "mu": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
        "poll_blend_scale": [2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
        "use_year_adaptive_priors": [True, False],
        "prior_decay": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
        "half_life_days": [7.0, 14.0, 21.0, 30.0, 45.0, 60.0, 90.0],
    }
    return param_grid, _ALL_RACE_CONFIGS


def _save_results(df: pd.DataFrame, prefix: str = "backtest_sweep") -> Path:
    """Save sweep results CSV to data/experiments/."""
    out_dir = PROJECT_ROOT / "data" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{prefix}_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # Suppress noisy sub-module logging during sweeps.
    logging.getLogger("src.prediction").setLevel(logging.WARNING)
    logging.getLogger("src.propagation").setLevel(logging.WARNING)
    logging.getLogger("src.validation.backtest_harness").setLevel(logging.WARNING)

    args = _build_arg_parser().parse_args()

    if args.compare_priors:
        print("Comparing Ridge priors vs year-adaptive priors across all elections...")
        df = compare_priors()
        _print_comparison_table(df)
        path = _save_results(df, prefix="prior_comparison")
        print(f"Results saved to {path}")

    elif args.quick:
        print("Running quick parameter sweep...")
        param_grid, race_configs = _quick_grid()
        df = sweep_parameters(param_grid, race_configs)
        param_names = sorted(param_grid.keys())
        _print_sweep_summary(df, param_names)
        path = _save_results(df)
        print(f"Results saved to {path}")

    elif args.full:
        print("Running full parameter sweep (this will take a while)...")
        param_grid, race_configs = _full_grid()
        df = sweep_parameters(param_grid, race_configs)
        param_names = sorted(param_grid.keys())
        _print_sweep_summary(df, param_names)
        path = _save_results(df)
        print(f"Results saved to {path}")

    else:
        _build_arg_parser().print_help()
        raise SystemExit(1)
