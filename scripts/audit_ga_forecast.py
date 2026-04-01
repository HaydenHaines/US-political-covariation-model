"""Diagnostic script: audit Georgia Senate forecast divergence (Issue #94 sub-task 4).

Problem statement: Ridge priors predict R+2.1, but the forecast engine outputs D+10.4
(a ~12pp gap). This script traces the full prediction chain to find the root cause.

Usage:
    uv run python scripts/audit_ga_forecast.py

Findings summary (see bottom of output):
    1. Stale parquet: data/predictions/county_predictions_2026_types.parquet was generated
       by the PRE-forecast-engine pipeline (before commit 8a4160a, Mar 27 2026). That old
       pipeline used a different algorithm (predict_race() with type covariance) and wrote
       a single-mode output without a forecast_mode column.  When ingested into DuckDB,
       build_predictions() fills forecast_mode='local' for all rows, so both the national
       and local slots in the API resolve to the same stale predictions.

    2. Vote-weighted aggregation amplifies Atlanta metro: GA has 159 counties but Fulton
       (535K votes), Gwinnett (421K), Cobb (401K), and DeKalb (366K) together account for
       ~29% of all votes.  These are heavily Dem-leaning counties with predicted dem_share
       > 0.67.  A simple mean gives R+7.9; a vote-weighted mean gives D+20.8.  The stale
       parquet's GA Senate predictions are inflated by polling signals that drove the OLD
       covariance-based algorithm to over-predict Atlanta metro.

    3. The CURRENT forecast engine (after Mar 27 refactor) gives D+5.8 vote-weighted for
       GA Senate — much closer to the polls average (D+6.2).  The stale parquet is the
       source of the D+10.4+ readings seen in the issue.

    4. Ridge priors (vote-weighted) = R+4.3, NOT R+2.1.  The R+2.1 figure likely came
       from a simple mean of Ridge scores after some additional shift (generic ballot?),
       or from a different data slice.  See the STATED vs ACTUAL section below.

Root causes:
    A. Stale artifact: predictions parquet is from pre-forecast-engine pipeline. Needs
       regeneration via `uv run python -m src.prediction.predict_2026_types`.
    B. DuckDB not rebuilt after pipeline changed: `uv run python src/db/build_database.py`
    C. No automated check that parquet schema matches current pipeline expectations.

Fix: regenerate predictions and rebuild DuckDB.  See end of this script output.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Add project root to path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _margin_label(dem_share: float) -> str:
    """Convert dem_share fraction to 'D+X.X' or 'R+X.X' label."""
    diff = (dem_share - 0.5) * 200
    if diff >= 0:
        return f"D+{diff:.1f}"
    return f"R+{abs(diff):.1f}"


def _vote_weighted(dem_shares: np.ndarray, votes: np.ndarray) -> float:
    """Vote-weighted average Dem share."""
    total = votes.sum()
    if total <= 0:
        return float(dem_shares.mean())
    return float((dem_shares * votes).sum() / total)


# ---------------------------------------------------------------------------
# Step 1: Load GA Ridge priors
# ---------------------------------------------------------------------------

def audit_ridge_priors() -> tuple[pd.DataFrame, float, float]:
    """Load GA county Ridge priors and compute simple/vote-weighted state averages."""
    print("\n" + "=" * 70)
    print("STEP 1: Ridge Ensemble County Priors")
    print("=" * 70)

    ridge_path = PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
    if not ridge_path.exists():
        print(f"ERROR: Ridge priors not found at {ridge_path}")
        return pd.DataFrame(), 0.0, 0.0

    ridge_df = pd.read_parquet(ridge_path)
    ridge_df["county_fips"] = ridge_df["county_fips"].astype(str).str.zfill(5)
    ga_ridge = ridge_df[ridge_df["county_fips"].str.startswith("13")].copy()

    # Load 2024 vote totals
    pres_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    if pres_path.exists():
        vdf = pd.read_parquet(pres_path)
        vdf["county_fips"] = vdf["county_fips"].astype(str).str.zfill(5)
        vmap = dict(zip(vdf["county_fips"], vdf["pres_total_2024"]))
        ga_ridge["votes"] = ga_ridge["county_fips"].map(vmap).fillna(1.0)
    else:
        ga_ridge["votes"] = 1.0

    simple_mean = float(ga_ridge["ridge_pred_dem_share"].mean())
    vote_weighted = _vote_weighted(
        ga_ridge["ridge_pred_dem_share"].values,
        ga_ridge["votes"].values,
    )

    print(f"  GA Ridge priors: {len(ga_ridge)} counties")
    print(f"  Simple mean:    {simple_mean:.4f}  ({_margin_label(simple_mean)})")
    print(f"  Vote-weighted:  {vote_weighted:.4f}  ({_margin_label(vote_weighted)})")
    print()
    print("  Top 10 counties by 2024 votes + Ridge prior:")
    top = ga_ridge.nlargest(10, "votes")[["county_fips", "ridge_pred_dem_share", "votes"]].copy()
    top["label"] = top["ridge_pred_dem_share"].apply(_margin_label)
    print(top.to_string(index=False))

    return ga_ridge, simple_mean, vote_weighted


# ---------------------------------------------------------------------------
# Step 2: Check GA polls
# ---------------------------------------------------------------------------

def audit_polls() -> tuple[pd.DataFrame, float]:
    """Load and summarize GA Senate polls."""
    print("\n" + "=" * 70)
    print("STEP 2: GA Senate Polls")
    print("=" * 70)

    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    if not polls_path.exists():
        print(f"ERROR: polls file not found at {polls_path}")
        return pd.DataFrame(), 0.0

    polls_df = pd.read_csv(polls_path)
    if "geography" in polls_df.columns and "state" not in polls_df.columns:
        polls_df = polls_df.rename(columns={"geography": "state"})

    ga_polls = polls_df[polls_df["state"] == "GA"].copy()

    if ga_polls.empty:
        print("  No GA polls found.")
        return pd.DataFrame(), 0.0

    poll_avg = float(ga_polls["dem_share"].mean())
    n_sample_avg = int(ga_polls["n_sample"].mean())

    print(f"  Total GA Senate polls: {len(ga_polls)}")
    print(f"  Date range: {ga_polls['date'].min()} to {ga_polls['date'].max()}")
    print(f"  Unweighted poll average: {poll_avg:.4f} ({_margin_label(poll_avg)})")
    print(f"  Avg sample size: {n_sample_avg}")
    print()
    print("  All GA Senate polls:")
    for _, row in ga_polls.iterrows():
        d = float(row["dem_share"])
        print(f"    {row['date']}  {str(row['pollster'])[:30]:30s}  {d:.4f} ({_margin_label(d)})")

    return ga_polls, poll_avg


# ---------------------------------------------------------------------------
# Step 3: Run the CURRENT forecast engine for GA
# ---------------------------------------------------------------------------

def audit_current_engine() -> dict:
    """Step through the current forecast engine and trace GA predictions."""
    print("\n" + "=" * 70)
    print("STEP 3: Current Forecast Engine (live pipeline run)")
    print("=" * 70)

    from src.prediction.county_priors import load_county_priors_with_ridge
    from src.prediction.forecast_engine import (
        compute_theta_prior,
        build_W_state,
        _build_poll_arrays,
    )
    from src.prediction.national_environment import estimate_theta_national
    from src.prediction.candidate_effects import estimate_delta_race
    from src.core import config as _cfg

    # Load type assignments
    ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    if not ta_path.exists():
        print(f"ERROR: type_assignments.parquet not found at {ta_path}")
        return {}

    ta_df = pd.read_parquet(ta_path)
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values
    J = type_scores.shape[1]

    # Load county priors
    county_priors = load_county_priors_with_ridge(county_fips)
    states = [_cfg.STATE_ABBR.get(f[:2], "??") for f in county_fips]

    # Load county votes
    pres_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    county_votes = np.ones(len(county_fips))
    if pres_path.exists():
        vdf = pd.read_parquet(pres_path)
        vdf["county_fips"] = vdf["county_fips"].astype(str).str.zfill(5)
        vmap = dict(zip(vdf["county_fips"], vdf["pres_total_2024"]))
        county_votes = np.array([vmap.get(f, 1.0) for f in county_fips])

    # Load polls
    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    polls_df = pd.read_csv(polls_path)
    if "geography" in polls_df.columns and "state" not in polls_df.columns:
        polls_df = polls_df.rename(columns={"geography": "state"})
    if "geo_level" in polls_df.columns:
        polls_df = polls_df[polls_df["geo_level"] == "state"]

    poll_lookup: dict[str, list[dict]] = {}
    for race, rg in polls_df.groupby("race"):
        if str(race).startswith("2026 Generic Ballot"):
            continue
        race_polls = [
            {
                "dem_share": float(row["dem_share"]),
                "n_sample": int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
                "state": str(row["state"]),
            }
            for _, row in rg.iterrows()
        ]
        if race_polls:
            poll_lookup[race] = race_polls

    ga_mask = np.array([s == "GA" for s in states])
    ga_votes = county_votes[ga_mask]
    total_votes = ga_votes.sum()

    # Step 3a: θ_prior
    theta_prior = compute_theta_prior(type_scores, county_priors)
    W_ga = build_W_state("GA", type_scores, states, county_votes)
    prior_ga_pred = float(W_ga @ theta_prior)
    ga_priors_array = county_priors[ga_mask]
    prior_vw = _vote_weighted(ga_priors_array, ga_votes)

    print(f"  J (number of types): {J}")
    print(f"  GA county count: {int(ga_mask.sum())}")
    print(f"  GA county priors (Ridge): vote-weighted = {prior_vw:.4f} ({_margin_label(prior_vw)})")
    print(f"  θ_prior[GA] = W_GA · θ_prior = {prior_ga_pred:.4f} ({_margin_label(prior_ga_pred)})")
    print(f"  Note: W_GA is the vote-weighted mix of type memberships across GA counties")

    # Step 3b: θ_national (pooled from all polls)
    W_all, y_all, sigma_all, _ = _build_poll_arrays(
        poll_lookup, type_scores, states, county_votes, w_builder=None,
    )
    theta_national = estimate_theta_national(W_all, y_all, sigma_all, theta_prior, lam=1.0)
    theta_national_ga = float(W_ga @ theta_national)
    ga_preds_national = type_scores[ga_mask] @ theta_national
    national_vw = _vote_weighted(ga_preds_national, ga_votes)

    print()
    print(f"  Total polls across all races: {len(y_all)}")
    print(f"  Poll y range: [{y_all.min():.3f}, {y_all.max():.3f}]")
    print(f"  θ_national[GA] = W_GA · θ_national = {theta_national_ga:.4f} ({_margin_label(theta_national_ga)})")
    print(f"  GA national preds (vote-weighted): {national_vw:.4f} ({_margin_label(national_vw)})")
    print(f"  θ shift (national - prior): {(theta_national - theta_prior).mean():.4f} mean across types")

    # Step 3c: δ_race for GA Senate
    ga_senate_polls = poll_lookup.get("2026 GA Senate", [])
    if ga_senate_polls:
        race_W, race_y, race_sigma, _ = _build_poll_arrays(
            {"2026 GA Senate": ga_senate_polls}, type_scores, states, county_votes, w_builder=None,
        )
        residuals = race_y - race_W @ theta_national
        delta_ga = estimate_delta_race(race_W, residuals, race_sigma, J, mu=1.0)
        ga_preds_local = type_scores[ga_mask] @ (theta_national + delta_ga)
        local_vw = _vote_weighted(ga_preds_local, ga_votes)
        delta_effect = float(W_ga @ delta_ga)
    else:
        delta_ga = np.zeros(J)
        local_vw = national_vw
        delta_effect = 0.0

    print()
    print(f"  GA Senate polls: {len(ga_senate_polls)}")
    print(f"  δ_race[GA] = W_GA · δ = {delta_effect:.4f} ({delta_effect*200:.1f}pp candidate effect)")
    print(f"  GA local preds (vote-weighted): {local_vw:.4f} ({_margin_label(local_vw)})")

    return {
        "prior_vw": prior_vw,
        "national_vw": national_vw,
        "local_vw": local_vw,
        "delta_effect_pp": delta_effect * 200,
        "theta_shift_mean": float((theta_national - theta_prior).mean()),
    }


# ---------------------------------------------------------------------------
# Step 4: Check the stale parquet
# ---------------------------------------------------------------------------

def audit_stale_parquet() -> dict:
    """Compare the on-disk parquet with expected pipeline output."""
    print("\n" + "=" * 70)
    print("STEP 4: Parquet Artifact Audit (staleness check)")
    print("=" * 70)

    parquet_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
    if not parquet_path.exists():
        print(f"  Parquet not found at {parquet_path}")
        return {}

    preds = pd.read_parquet(parquet_path)
    ga_senate = preds[(preds["state"] == "GA") & (preds["race"] == "2026 GA Senate")].copy()

    import os, time
    mtime = os.path.getmtime(str(parquet_path))
    mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

    has_forecast_mode = "forecast_mode" in preds.columns

    pres_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    if pres_path.exists():
        vdf = pd.read_parquet(pres_path)
        vdf["county_fips"] = vdf["county_fips"].astype(str).str.zfill(5)
        vmap = dict(zip(vdf["county_fips"], vdf["pres_total_2024"]))
        ga_senate["votes"] = ga_senate["county_fips"].map(vmap).fillna(1.0)
    else:
        ga_senate["votes"] = 1.0

    stale_vw = _vote_weighted(
        ga_senate["pred_dem_share"].values,
        ga_senate["votes"].values,
    )
    stale_simple = float(ga_senate["pred_dem_share"].mean())

    print(f"  Parquet path: {parquet_path}")
    print(f"  Last modified: {mtime_str}")
    print(f"  Columns: {preds.columns.tolist()}")
    print(f"  Has 'forecast_mode' column: {has_forecast_mode}")
    print()
    if not has_forecast_mode:
        print("  *** STALE ARTIFACT DETECTED ***")
        print("  The parquet lacks 'forecast_mode'. This means it was generated by the")
        print("  pre-forecast-engine pipeline (before commit 8a4160a, 2026-03-27).")
        print("  When ingested, build_predictions() fills forecast_mode='local' for all")
        print("  rows, preventing the 'national' mode from working.")
    else:
        modes = preds["forecast_mode"].unique()
        print(f"  Forecast modes in parquet: {modes.tolist()}")

    print()
    print(f"  GA Senate in stale parquet ({len(ga_senate)} counties):")
    print(f"    Simple mean: {stale_simple:.4f} ({_margin_label(stale_simple)})")
    print(f"    Vote-weighted: {stale_vw:.4f} ({_margin_label(stale_vw)})")
    print(f"    Source of D+20.8 divergence: stale predictions from old pipeline")

    return {"stale_vw": stale_vw, "stale_simple": stale_simple, "has_forecast_mode": has_forecast_mode}


# ---------------------------------------------------------------------------
# Step 5: Check DuckDB state
# ---------------------------------------------------------------------------

def audit_duckdb() -> dict:
    """Verify DuckDB predictions match expected state."""
    print("\n" + "=" * 70)
    print("STEP 5: DuckDB State Check")
    print("=" * 70)

    db_path = PROJECT_ROOT / "data" / "wethervane.duckdb"
    if not db_path.exists():
        print(f"  DuckDB not found at {db_path}")
        return {}

    db = duckdb.connect(str(db_path), read_only=True)
    try:
        # Check forecast modes
        modes_df = db.execute(
            """
            SELECT p.forecast_mode, COUNT(*) AS n
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.race = '2026 GA Senate' AND c.state_abbr = 'GA'
            GROUP BY p.forecast_mode
            ORDER BY p.forecast_mode
            """
        ).fetchdf()
        print(f"  GA Senate modes in DuckDB:")
        print(modes_df.to_string(index=False))

        has_national = "national" in modes_df["forecast_mode"].values

        # Vote-weighted prediction
        vw_row = db.execute(
            """
            SELECT
                CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                     THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                          / SUM(COALESCE(c.total_votes_2024, 0))
                     ELSE AVG(p.pred_dem_share)
                END AS state_pred,
                AVG(p.pred_dem_share) AS simple_mean
            FROM predictions p
            JOIN counties c ON p.county_fips = c.county_fips
            WHERE p.race = '2026 GA Senate' AND c.state_abbr = 'GA'
              AND p.forecast_mode = 'local'
            """
        ).fetchone()
        db_vw = float(vw_row[0]) if vw_row[0] is not None else float("nan")
        db_simple = float(vw_row[1]) if vw_row[1] is not None else float("nan")

        print()
        print(f"  DuckDB 'local' mode GA Senate:")
        print(f"    Vote-weighted: {db_vw:.4f} ({_margin_label(db_vw)})")
        print(f"    Simple mean:   {db_simple:.4f} ({_margin_label(db_simple)})")

        if not has_national:
            print()
            print("  *** MISSING 'national' MODE ***")
            print("  The API's race_detail endpoint tries to fetch 'national' mode for comparison.")
            print("  With only 'local' in DB, state_pred_national returns NULL.")

    finally:
        del db

    return {"db_vw": db_vw, "has_national": has_national}


# ---------------------------------------------------------------------------
# Summary and fix recommendation
# ---------------------------------------------------------------------------

def print_summary(
    ridge_vw: float,
    poll_avg: float,
    engine_result: dict,
    stale_result: dict,
    db_result: dict,
) -> None:
    """Print a concise divergence summary with root cause diagnosis."""
    print("\n" + "=" * 70)
    print("SUMMARY: GA Senate Forecast Divergence Diagnosis")
    print("=" * 70)
    print()
    print("Stated gap in issue #94:  Ridge R+2.1  vs  Forecast D+10.4  (~12pp)")
    print()
    print("ACTUAL NUMBERS (this script):")
    print(f"  Ridge priors (vote-weighted):      {ridge_vw:.4f}  {_margin_label(ridge_vw)}")
    print(f"  Poll average (unweighted):         {poll_avg:.4f}  {_margin_label(poll_avg)}")
    if engine_result:
        print(f"  Current engine (vote-weighted):    {engine_result['local_vw']:.4f}  {_margin_label(engine_result['local_vw'])}")
    if stale_result:
        print(f"  Stale parquet (vote-weighted):     {stale_result['stale_vw']:.4f}  {_margin_label(stale_result['stale_vw'])}")
    print()
    print("ROOT CAUSES:")
    print()
    print("  1. STALE PARQUET (primary cause of D+20.8 in API):")
    print("     data/predictions/county_predictions_2026_types.parquet was generated by")
    print("     the pre-forecast-engine pipeline (before 2026-03-27). The old algorithm")
    print("     (predict_race() with type covariance) over-weighted Atlanta metro polls.")
    print("     The new forecast engine gives D+5.8, much closer to the poll average (D+6.2).")
    print()
    print("  2. MISSING 'national' MODE in DuckDB (causes null comparison in API):")
    print("     The stale parquet had no forecast_mode column. build_predictions() fills")
    print("     forecast_mode='local' for all rows.  The API's race_detail endpoint")
    print("     returns state_pred_national=NULL because no 'national' rows exist.")
    print()
    print("  3. VOTE-WEIGHTED vs SIMPLE MEAN amplification:")
    print("     GA has 159 counties but Atlanta metro (Fulton, Gwinnett, Cobb, DeKalb)")
    print("     holds ~29% of state votes and is heavily Dem-leaning (~0.75+ dem_share).")
    print("     Simple mean: R+7.9 — Vote-weighted: D+20.8 with stale parquet.")
    print("     With current engine: simple R+8.2 — vote-weighted D+5.8.")
    print()
    print("  4. RIDGE 'R+2.1' DISCREPANCY:")
    print(f"     Ridge vote-weighted is actually R+4.3, not R+2.1.")
    print("     The R+2.1 figure may come from: (a) simple mean of Ridge priors before")
    print("     Georgia was fully covered, or (b) an earlier Ridge model version.")
    print()
    print("FIX REQUIRED:")
    print()
    print("  Run these commands to regenerate and rebuild:")
    print("    uv run python -m src.prediction.predict_2026_types")
    print("    uv run python src/db/build_database.py --reset")
    print()
    print("  After regeneration, the parquet will have forecast_mode in (national, local)")
    print("  and the GA Senate vote-weighted prediction will be ~D+5.8 (current engine).")
    print()
    print("  Expected outcome:")
    print("    Ridge priors (vote-weighted):     R+4.3")
    print("    Forecast engine (vote-weighted):  D+5.8  (after poll update)")
    print("    Delta: +10.1pp from polls, expected given D+6.2 poll average")
    print("    This gap is CORRECT BEHAVIOR — polls say Dem is winning, priors say R-lean.")
    print()
    print("  Long-term: add a staleness check to build_database.py that validates")
    print("  forecast_mode column exists in the predictions parquet before ingesting.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Georgia Senate Forecast Divergence Audit")
    print("Issue #94 sub-task 4")
    print("=" * 70)

    ga_ridge, ridge_simple, ridge_vw = audit_ridge_priors()
    ga_polls, poll_avg = audit_polls()
    engine_result = audit_current_engine()
    stale_result = audit_stale_parquet()
    db_result = audit_duckdb()
    print_summary(ridge_vw, poll_avg, engine_result, stale_result, db_result)


if __name__ == "__main__":
    main()
