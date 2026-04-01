"""Calibrate lam and mu hyperparameters for the forecast engine.

lam  — θ_national regularization strength (how much polls can pull type means
        away from their historical priors). Higher lam = more prior trust, less
        poll influence. At lam→∞ the model ignores polls entirely.
mu   — δ_race regularization strength (how much candidate effect can deviate
        from the national environment). Higher mu = more shrinkage toward zero
        (no candidate effect), lower mu = more latitude for race-specific swings.

Calibration strategy
--------------------
We simulate past elections where we know both the pre-election state polls
*and* the actual county-level Dem share outcomes. For each (lam, mu) pair:

  1. Use county-level historical results from before the election year as the
     county prior (e.g. for evaluating 2022, use 2020 presidential as priors).
  2. Feed in state-level polls for that election cycle.
  3. Run run_forecast() to produce county predictions.
  4. Compare predictions to actual county Dem shares.
  5. Report LOO RMSE and Pearson r.

Validation elections used
--------------------------
  - 2020 presidential (polls from data/polls/polls_2020.csv, prior = 2016 pres)
  - 2022 gubernatorial / Senate (polls from data/polls/polls_2022.csv, prior = 2020 pres)

These represent both presidential and off-cycle midterm conditions, giving
calibration coverage across two fundamentally different electoral environments.

Usage
-----
    uv run python scripts/calibrate_lam_mu.py
    uv run python scripts/calibrate_lam_mu.py --lam-min 0.1 --lam-max 20 --lam-steps 15
    uv run python scripts/calibrate_lam_mu.py --lam-min 0.5 --lam-max 5 --lam-steps 10 \\
                                               --mu-min 0.1 --mu-max 5 --mu-steps 10
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Path resolution (worktree-safe: walk up to find data/)
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_WORKTREE_ROOT = _THIS_FILE.parents[1]


def _find_data_root() -> Path:
    """Return the Path that actually contains data/assembled/."""
    candidate = _WORKTREE_ROOT
    if (candidate / "data" / "assembled").is_dir():
        return candidate
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(candidate),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        main_root = Path(common).resolve().parent
        if (main_root / "data" / "assembled").is_dir():
            return main_root
    except Exception:
        pass
    return candidate


PROJECT_ROOT = _WORKTREE_ROOT
DATA_ROOT = _find_data_root()

sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.forecast_engine import (  # noqa: E402
    build_W_state,
    compute_theta_prior,
    run_forecast,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_type_model() -> tuple[list[str], np.ndarray, int]:
    """Load county FIPS codes, type score matrix, and J from type_assignments.

    Returns
    -------
    county_fips : list[str]
    type_scores : ndarray of shape (N, J)
    J : int
    """
    ta_path = DATA_ROOT / "data" / "communities" / "type_assignments.parquet"
    ta = pd.read_parquet(ta_path)
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)

    score_cols = sorted([c for c in ta.columns if c.endswith("_score")])
    type_scores = ta[score_cols].values
    county_fips = ta["county_fips"].tolist()
    J = type_scores.shape[1]

    log.info("Loaded type model: %d counties × %d types", len(county_fips), J)
    return county_fips, type_scores, J


def load_county_dem_share(year: int) -> dict[str, float]:
    """Load county-level Dem share from assembled MEDSL parquet for a given year.

    Returns a dict mapping county_fips -> dem_share, using the presidential
    Dem share column for presidential years, or the best available column for
    off-cycle years.

    Note: For 2022, we use the MEDSL 2022 governor file which contains
    governor race results. The column naming follows the MEDSL parquet convention.
    """
    # Presidential years use the standard MEDSL presidential parquet
    pres_path = DATA_ROOT / "data" / "assembled" / f"medsl_county_presidential_{year}.parquet"
    if pres_path.exists():
        df = pd.read_parquet(pres_path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"pres_dem_share_{year}"
        if share_col in df.columns:
            return dict(zip(df["county_fips"], df[share_col].fillna(np.nan)))

    # 2022 off-cycle: use governor results
    # Column naming: gov_dem_share_2022 (from medsl_county_2022_governor.parquet)
    gov_path = DATA_ROOT / "data" / "assembled" / "medsl_county_2022_governor.parquet"
    if year == 2022 and gov_path.exists():
        df = pd.read_parquet(gov_path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        # Explicit known column first (from MEDSL 2022 governor build)
        if "gov_dem_share_2022" in df.columns:
            return dict(zip(df["county_fips"], df["gov_dem_share_2022"].fillna(np.nan)))
        # Fallback: any dem+share column
        dem_cols = [c for c in df.columns if "dem" in c.lower() and "share" in c.lower()]
        if dem_cols:
            return dict(zip(df["county_fips"], df[dem_cols[0]].fillna(np.nan)))
        # Fallback: compute from raw vote columns
        for dem_col, rep_col in [
            ("gov_dem_2022", "gov_rep_2022"),
            ("dem_votes", "rep_votes"),
        ]:
            if dem_col in df.columns and rep_col in df.columns:
                total = df[dem_col] + df[rep_col]
                df["dem_share"] = df[dem_col] / total.replace(0, np.nan)
                return dict(zip(df["county_fips"], df["dem_share"].fillna(np.nan)))

    return {}


def load_state_abbreviations() -> dict[str, str]:
    """Load FIPS -> state abbreviation mapping from config."""
    from src.core import config as _cfg
    return _cfg.STATE_ABBR


def load_polls_for_year(year: int) -> dict[str, list[dict]]:
    """Load state-level polls from data/polls/polls_{year}.csv.

    Returns polls_by_race dict in the format expected by run_forecast().
    Filters to race type matching year (presidential for 2020, Senate/gov for 2022).
    """
    polls_path = DATA_ROOT / "data" / "polls" / f"polls_{year}.csv"
    if not polls_path.exists():
        log.warning("Poll file not found: %s", polls_path)
        return {}

    df = pd.read_csv(polls_path)
    if "geography" in df.columns and "state" not in df.columns:
        df = df.rename(columns={"geography": "state"})

    # Filter to state-level only
    if "geo_level" in df.columns:
        df = df[df["geo_level"] == "state"].copy()

    # Drop generic ballot rows (not a specific state race)
    if "race" in df.columns:
        df = df[~df["race"].str.contains("Generic Ballot", case=False, na=False)]

    polls_by_race: dict[str, list[dict]] = {}
    if len(df) == 0:
        return {}

    for race, grp in df.groupby("race"):
        race_polls = []
        for _, row in grp.iterrows():
            if pd.isna(row.get("dem_share")):
                continue
            n = int(row["n_sample"]) if pd.notna(row.get("n_sample")) else 600
            race_polls.append({
                "dem_share": float(row["dem_share"]),
                "n_sample": n,
                "state": str(row["state"]),
                "date": str(row.get("date", "")),
                "pollster": str(row.get("pollster", "")),
                "notes": str(row.get("notes", "")),
            })
        if race_polls:
            polls_by_race[race] = race_polls

    log.info("Loaded %d races / %d polls for %d", len(polls_by_race), sum(len(v) for v in polls_by_race.values()), year)
    return polls_by_race


def load_county_votes_for_year(year: int, county_fips: list[str]) -> np.ndarray:
    """Load total votes per county for a given presidential year (for W weights)."""
    pres_path = DATA_ROOT / "data" / "assembled" / f"medsl_county_presidential_{year}.parquet"
    if pres_path.exists():
        df = pd.read_parquet(pres_path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        # Find total votes column
        total_cols = [c for c in df.columns if "total" in c.lower()]
        if total_cols:
            vmap = dict(zip(df["county_fips"], df[total_cols[0]].fillna(1.0)))
            return np.array([vmap.get(f, 1.0) for f in county_fips])
    return np.ones(len(county_fips))


# ---------------------------------------------------------------------------
# Single (lam, mu) evaluation
# ---------------------------------------------------------------------------


def evaluate_lam_mu_for_cycle(
    lam: float,
    mu: float,
    prior_year: int,
    target_year: int,
    county_fips: list[str],
    type_scores: np.ndarray,
    state_abbr: dict[str, str],
) -> dict | None:
    """Evaluate one (lam, mu) pair for one election cycle.

    Uses prior_year presidential results as county priors, feeds in
    target_year polls, runs the forecast, and compares to actual
    target_year county Dem shares.

    Returns None if insufficient data (< 100 matched counties).
    """
    J = type_scores.shape[1]

    # Build county priors from prior_year actuals
    prior_dem = load_county_dem_share(prior_year)
    county_priors = np.array([prior_dem.get(f, 0.45) for f in county_fips])
    # Fill NaN (counties with missing prior) with national average
    nan_mask = np.isnan(county_priors)
    if nan_mask.any():
        fallback = float(np.nanmean(county_priors)) if not np.all(nan_mask) else 0.45
        county_priors[nan_mask] = fallback

    # Load polls for target year
    polls_by_race = load_polls_for_year(target_year)
    if not polls_by_race:
        log.warning("No polls for %d — skipping", target_year)
        return None

    # Build states array (FIPS[:2] -> abbreviation)
    states = [state_abbr.get(f[:2], "??") for f in county_fips]

    # Vote weights: use prior year presidential totals for W vector construction
    county_votes = load_county_votes_for_year(prior_year, county_fips)

    # All unique race IDs
    all_race_ids = list(polls_by_race.keys())

    # Run forecast
    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes,
        polls_by_race=polls_by_race,
        races=all_race_ids,
        lam=lam,
        mu=mu,
        w_vector_mode="core",
        reference_date=None,  # No time-decay; use raw poll weights
    )

    # Load actual results for target year
    actual_dem = load_county_dem_share(target_year)

    # Aggregate predictions across races using vote-weighted mean per county.
    # We average the "local" (θ_national + δ_race) predictions across all races
    # that produced predictions, since different state races cover different counties.
    # This tests whether lam/mu produce well-calibrated county predictions overall.
    state_race_preds: dict[str, list[float]] = {}  # county_fips -> list of race preds
    for race_id, fr in forecast_results.items():
        for i, fips in enumerate(county_fips):
            state_race_preds.setdefault(fips, []).append(fr.county_preds_local[i])

    # Build arrays aligned on counties with actual data
    actual_arr = []
    pred_arr = []
    for fips, pred_list in state_race_preds.items():
        actual = actual_dem.get(fips)
        if actual is None or np.isnan(actual):
            continue
        actual_arr.append(actual)
        pred_arr.append(float(np.mean(pred_list)))

    if len(actual_arr) < 100:
        log.warning("Insufficient matched counties (%d) for %d — skipping", len(actual_arr), target_year)
        return None

    actual_np = np.array(actual_arr)
    pred_np = np.array(pred_arr)

    rmse = float(np.sqrt(np.mean((actual_np - pred_np) ** 2)))
    mae = float(np.mean(np.abs(actual_np - pred_np)))
    bias = float(np.mean(pred_np - actual_np))

    r = float("nan")
    if len(actual_np) >= 2 and np.std(actual_np) > 1e-10 and np.std(pred_np) > 1e-10:
        r, _ = pearsonr(actual_np, pred_np)

    return {
        "lam": lam,
        "mu": mu,
        "year": target_year,
        "n_counties": len(actual_arr),
        "n_races": len(forecast_results),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "bias": round(bias, 6),
        "pearson_r": round(float(r), 6) if not np.isnan(r) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


def run_sweep(
    lam_values: list[float],
    mu_values: list[float],
    eval_cycles: list[tuple[int, int]],
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep (lam, mu) grid over the specified election cycles.

    Parameters
    ----------
    lam_values : list of lam to test
    mu_values : list of mu to test
    eval_cycles : list of (prior_year, target_year) tuples
    verbose : bool

    Returns
    -------
    DataFrame with one row per (lam, mu, year) combination.
    """
    county_fips, type_scores, J = load_type_model()
    state_abbr = load_state_abbreviations()

    if verbose:
        print(f"Type model: {len(county_fips)} counties × {J} types")
        print(f"Evaluation cycles: {eval_cycles}")
        print(f"Grid: {len(lam_values)} lam × {len(mu_values)} mu = {len(lam_values)*len(mu_values)} combinations")
        print(f"Total evaluations: {len(lam_values)*len(mu_values)*len(eval_cycles)}")
        print()

    rows = []
    total = len(lam_values) * len(mu_values) * len(eval_cycles)
    done = 0

    for lam in lam_values:
        for mu in mu_values:
            for prior_year, target_year in eval_cycles:
                result = evaluate_lam_mu_for_cycle(
                    lam=lam,
                    mu=mu,
                    prior_year=prior_year,
                    target_year=target_year,
                    county_fips=county_fips,
                    type_scores=type_scores,
                    state_abbr=state_abbr,
                )
                done += 1

                if result is not None:
                    rows.append(result)
                    if verbose:
                        r_str = f"{result['pearson_r']:.4f}" if not np.isnan(result["pearson_r"]) else "  NaN"
                        print(
                            f"  [{done:>4}/{total}]  lam={lam:>6.2f}  mu={mu:>6.2f}"
                            f"  year={target_year}"
                            f"  RMSE={result['rmse']:.4f}  r={r_str}"
                            f"  n={result['n_counties']}"
                        )
                else:
                    if verbose:
                        print(f"  [{done:>4}/{total}]  lam={lam:>6.2f}  mu={mu:>6.2f}  year={target_year}  SKIPPED")

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Summary and recommendation
# ---------------------------------------------------------------------------


def summarize_sweep(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-year results to (lam, mu) mean RMSE and r across cycles.

    Returns a DataFrame sorted by mean_rmse ascending.
    """
    if results.empty:
        return pd.DataFrame()

    summary = (
        results
        .groupby(["lam", "mu"])
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_r=("pearson_r", "mean"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
            n_cycles=("year", "count"),
        )
        .reset_index()
        .sort_values("mean_rmse")
    )
    return summary


def print_summary(summary: pd.DataFrame, current_lam: float, current_mu: float) -> None:
    """Print the top results table with the current defaults highlighted."""
    print()
    print("=" * 80)
    print("LAM / MU SWEEP — Forecast Engine Regularization (Plan C calibration)")
    print("Lower RMSE = better. Higher r = better.")
    print("=" * 80)

    header = (
        f"{'lam':>7}  {'mu':>7}  {'mean_RMSE':>10}  {'mean_r':>8}  "
        f"{'mean_MAE':>9}  {'mean_bias':>10}  {'cycles':>7}"
    )
    print(header)
    print("-" * 80)

    if summary.empty:
        print("  (no results)")
        return

    best_idx = summary["mean_rmse"].idxmin()
    best_row = summary.loc[best_idx]

    for _, row in summary.iterrows():
        marker = ""
        if abs(row["lam"] - current_lam) < 1e-9 and abs(row["mu"] - current_mu) < 1e-9:
            marker = " <-- CURRENT"
        if abs(row["lam"] - best_row["lam"]) < 1e-9 and abs(row["mu"] - best_row["mu"]) < 1e-9:
            marker = " <-- BEST" + (" / CURRENT" if marker else "")

        r_str = f"{row['mean_r']:.4f}" if pd.notna(row["mean_r"]) else "   NaN"
        print(
            f"{row['lam']:>7.2f}  {row['mu']:>7.2f}  {row['mean_rmse']:>10.5f}  "
            f"{r_str:>8}  {row['mean_mae']:>9.5f}  {row['mean_bias']:>+10.5f}  "
            f"{int(row['n_cycles']):>7}{marker}"
        )


def make_recommendation(
    summary: pd.DataFrame,
    current_lam: float,
    current_mu: float,
    improvement_threshold: float = 0.001,
) -> dict:
    """Build a recommendation dict from the sweep summary.

    Returns dict with keys: recommended_lam, recommended_mu, best_rmse,
    current_rmse, delta_rmse, recommendation.
    """
    if summary.empty:
        return {"recommendation": "No results — cannot calibrate."}

    best_row = summary.iloc[0]  # already sorted by mean_rmse ascending

    current_rows = summary[
        ((summary["lam"] - current_lam).abs() < 1e-9)
        & ((summary["mu"] - current_mu).abs() < 1e-9)
    ]

    rec: dict = {
        "recommended_lam": float(best_row["lam"]),
        "recommended_mu": float(best_row["mu"]),
        "best_rmse": float(best_row["mean_rmse"]),
        "best_r": float(best_row["mean_r"]) if pd.notna(best_row["mean_r"]) else float("nan"),
    }

    if len(current_rows) > 0:
        cur = current_rows.iloc[0]
        rec["current_rmse"] = float(cur["mean_rmse"])
        rec["delta_rmse"] = float(rec["best_rmse"] - rec["current_rmse"])
        delta = rec["delta_rmse"]
        improvement = -delta  # negative delta = improvement

        if improvement < improvement_threshold:
            rec["recommendation"] = (
                f"Current defaults (lam={current_lam}, mu={current_mu}) are near-optimal. "
                f"Best RMSE={rec['best_rmse']:.5f} vs current RMSE={rec['current_rmse']:.5f} "
                f"(Δ={delta:+.5f}). No change recommended."
            )
        else:
            rec["recommendation"] = (
                f"Update lam={current_lam} → {rec['recommended_lam']:.2f}, "
                f"mu={current_mu} → {rec['recommended_mu']:.2f}. "
                f"RMSE improves {rec['current_rmse']:.5f} → {rec['best_rmse']:.5f} "
                f"(Δ={delta:+.5f}, improvement={improvement:.5f}). "
                "Update data/config/prediction_params.json."
            )
    else:
        rec["recommendation"] = (
            f"Current defaults (lam={current_lam}, mu={current_mu}) not in sweep grid. "
            f"Best found: lam={rec['recommended_lam']:.2f}, mu={rec['recommended_mu']:.2f} "
            f"(RMSE={rec['best_rmse']:.5f})."
        )

    return rec


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Calibrate lam/mu hyperparameters for the forecast engine."
    )
    parser.add_argument("--lam-min", type=float, default=0.1,
                        help="Minimum lam value (default: 0.1)")
    parser.add_argument("--lam-max", type=float, default=10.0,
                        help="Maximum lam value (default: 10.0)")
    parser.add_argument("--lam-steps", type=int, default=10,
                        help="Number of lam values in the grid (default: 10)")
    parser.add_argument("--mu-min", type=float, default=0.1,
                        help="Minimum mu value (default: 0.1)")
    parser.add_argument("--mu-max", type=float, default=10.0,
                        help="Maximum mu value (default: 10.0)")
    parser.add_argument("--mu-steps", type=int, default=10,
                        help="Number of mu values in the grid (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data/experiments/lam_mu_sweep.csv)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-evaluation output")
    args = parser.parse_args()

    # Build grids on a log scale (small values matter more for regularization)
    lam_values = list(np.logspace(
        np.log10(args.lam_min), np.log10(args.lam_max), args.lam_steps
    ).round(4))
    mu_values = list(np.logspace(
        np.log10(args.mu_min), np.log10(args.mu_max), args.mu_steps
    ).round(4))

    # Evaluation cycles: (prior_year, target_year)
    # 2020: presidential cycle — tests national poll propagation
    # 2022: off-cycle midterm — tests candidate effect regularization (mu)
    eval_cycles = [(2016, 2020), (2020, 2022)]

    # Load current defaults from config
    params_path = DATA_ROOT / "data" / "config" / "prediction_params.json"
    current_params = json.loads(params_path.read_text())["forecast"]
    current_lam = float(current_params["lam"])
    current_mu = float(current_params["mu"])

    print(f"Data root: {DATA_ROOT}")
    print(f"Current defaults: lam={current_lam}, mu={current_mu}")
    print(f"Sweep lam: {args.lam_min}..{args.lam_max} ({args.lam_steps} steps, log scale)")
    print(f"Sweep mu: {args.mu_min}..{args.mu_max} ({args.mu_steps} steps, log scale)")
    print(f"Evaluation cycles: {eval_cycles}")
    print()

    results = run_sweep(
        lam_values=lam_values,
        mu_values=mu_values,
        eval_cycles=eval_cycles,
        verbose=not args.quiet,
    )

    if results.empty:
        print("ERROR: No results produced. Check that assembled data and poll files exist.")
        sys.exit(1)

    # Save raw results
    out_dir = DATA_ROOT / "data" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.output) if args.output else out_dir / "lam_mu_sweep_raw.csv"
    results.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    # Summarize and print
    summary = summarize_sweep(results)
    print_summary(summary, current_lam, current_mu)

    # Save summary
    summary_path = out_dir / "lam_mu_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Recommendation
    rec = make_recommendation(summary, current_lam, current_mu)
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"  {rec['recommendation']}")
    print()
    print(f"  Best values:    lam={rec['recommended_lam']:.4f}  mu={rec['recommended_mu']:.4f}")
    print(f"  Best RMSE:      {rec['best_rmse']:.5f}")
    if "best_r" in rec:
        print(f"  Best Pearson r: {rec['best_r']:.4f}")
    if "current_rmse" in rec:
        print(f"  Current RMSE:   {rec['current_rmse']:.5f}  (Δ={rec['delta_rmse']:+.5f})")

    # Save recommendation JSON
    rec_path = out_dir / "lam_mu_recommendation.json"
    rec_path.write_text(json.dumps(rec, indent=2))
    print(f"\nRecommendation saved to {rec_path}")

    return rec


if __name__ == "__main__":
    main()
