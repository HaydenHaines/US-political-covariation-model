"""Tier 2 crosstab impact comparison (v2) — three runs:

  A) Enriched polls_2026, production code path (prepare_polls preprocessing).
     This is what the live forecast uses today.
  B) Enriched polls_2026, bypassing prepare_polls (reference_date=None).
     This shows what Tier 2 *would* do if xt_ keys survived preprocessing.
  C) Crosstab-stripped polls_2026, same as (A).
     Baseline: what the forecast looks like if every xt_ value were null.

Key insight surfaced by this script (2026-04-24): prepare_polls used to
reconstruct poll dicts from PollObservation topline fields only, dropping
xt_ and methodology keys before the dicts reached the W-builder.  Run A
should now differ from run C when production metadata preservation works.

Task context: follow-up to WV PR #167 (re-scrape regression fix).
After the fix, 25 state-level polls in polls_2026.csv carry xt_ crosstab
data.  This script measures (a) whether those crosstabs move the live
forecast and (b) how that compares with a prepare_polls bypass run.

Outputs a per-race table with enriched/stripped/bypass state-level pred,
theta_national deltas, and pred-change diagnostics suitable for the
brief at knowledge/briefs/2026-04-24-tier2-backtest-restoration.md.
"""
from __future__ import annotations

import copy
import json
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.prediction.county_priors import load_county_priors_with_ridge
from src.prediction.forecast_engine import run_forecast
from src.prediction.generic_ballot import compute_gb_shift

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

def _load_data() -> dict:
    from src.core import config as _cfg
    from src.assembly.define_races import load_races

    ta_df = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet")
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values

    states = [_cfg.STATE_ABBR.get(f[:2], "??") for f in county_fips]

    county_votes = np.ones(len(county_fips))
    votes_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    if votes_path.exists():
        vdf = pd.read_parquet(votes_path)
        votes_col = "pres_total_2024" if "pres_total_2024" in vdf.columns else "totalvotes"
        if "county_fips" in vdf.columns and votes_col in vdf.columns:
            vmap = dict(zip(
                vdf["county_fips"].astype(str).str.zfill(5),
                vdf[votes_col],
            ))
            county_votes = np.array([float(vmap.get(f, 1.0)) for f in county_fips])

    county_priors = load_county_priors_with_ridge(county_fips)

    tp_path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    type_profiles = pd.read_parquet(tp_path) if tp_path.exists() else None

    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    polls_df = pd.read_csv(polls_path)
    if "geography" in polls_df.columns and "state" not in polls_df.columns:
        polls_df = polls_df.rename(columns={"geography": "state"})
    if "geo_level" in polls_df.columns:
        polls_df = polls_df[polls_df["geo_level"] == "state"].copy()

    xt_cols = [c for c in polls_df.columns if c.startswith("xt_")]

    polls_by_race: dict[str, list[dict]] = {}
    for race_id, grp in polls_df.groupby("race"):
        if str(race_id).startswith("2026 Generic Ballot"):
            continue
        race_dicts = []
        for _, row in grp.iterrows():
            d = {
                "dem_share": float(row["dem_share"]),
                "n_sample": int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
                "state": str(row["state"]),
                "date": str(row["date"]) if pd.notna(row.get("date")) else "",
                "pollster": str(row["pollster"]) if pd.notna(row.get("pollster")) else "",
                "notes": str(row["notes"]) if pd.notna(row.get("notes")) else "",
            }
            method = row.get("methodology")
            if method is not None and pd.notna(method):
                d["methodology"] = str(method)
            for col in xt_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    d[col] = float(val)
            race_dicts.append(d)
        if race_dicts:
            polls_by_race[str(race_id)] = race_dicts

    params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}

    registry = load_races(2026)
    all_race_ids = [r.race_id for r in registry]

    gb_shift = compute_gb_shift().shift

    return {
        "type_scores": type_scores,
        "county_priors": county_priors,
        "states": states,
        "county_fips": county_fips,
        "county_votes": county_votes,
        "type_profiles": type_profiles,
        "polls_by_race": polls_by_race,
        "all_race_ids": all_race_ids,
        "gb_shift": gb_shift,
        "params": params,
        "xt_cols": xt_cols,
    }


def _strip_xt_and_methodology(polls_by_race: dict) -> dict:
    """Remove xt_* and methodology keys (simulate regression state)."""
    stripped = {}
    for race_id, polls in polls_by_race.items():
        stripped[race_id] = []
        for p in polls:
            new_p = {
                k: v for k, v in p.items()
                if not k.startswith("xt_") and k != "methodology"
            }
            stripped[race_id].append(new_p)
    return stripped


def _vote_weighted_state_pred(
    result, states: list[str], county_votes: np.ndarray, state_abbr: str,
) -> float | None:
    states_arr = np.array(states)
    mask = states_arr == state_abbr
    if not mask.any():
        return None
    votes = county_votes[mask]
    if votes.sum() <= 0:
        return float(np.mean(result.county_preds_local[mask]))
    return float(np.average(result.county_preds_local[mask], weights=votes))


def _forecast_kwargs(data: dict, reference_date: str | None) -> dict:
    params = data["params"]
    fc = params.get("forecast", {})
    pw = params.get("poll_weighting", {})
    accuracy_path = PROJECT_ROOT / "data" / "config" / "pollster_accuracy.json"
    kwargs = dict(
        type_scores=data["type_scores"],
        county_priors=data["county_priors"],
        states=data["states"],
        county_votes=data["county_votes"],
        races=data["all_race_ids"],
        lam=fc.get("lam", 1.0),
        mu=fc.get("mu", 1.0),
        generic_ballot_shift=data["gb_shift"],
        w_vector_mode=fc.get("w_vector_mode", "core"),
        type_profiles=data["type_profiles"],
        half_life_days=pw.get("half_life_days", 30.0),
        pre_primary_discount=pw.get("pre_primary_discount", 0.5),
        accuracy_path=accuracy_path if accuracy_path.exists() else None,
        methodology_weights=pw.get("methodology_weights"),
        poll_blend_scale=fc.get("poll_blend_scale", 5.0),
    )
    if reference_date is not None:
        kwargs["reference_date"] = reference_date
    return kwargs


def _run_three(data: dict) -> list[dict]:
    polls_enriched = data["polls_by_race"]
    polls_stripped = _strip_xt_and_methodology(polls_enriched)

    ref_date = str(date.today())

    # A: Enriched, production path (reference_date set → prepare_polls strips xt_)
    kwargs_prod = _forecast_kwargs(data, reference_date=ref_date)
    results_A = run_forecast(polls_by_race=polls_enriched, **kwargs_prod)

    # B: Enriched, BYPASS prepare_polls (reference_date=None)
    kwargs_bypass = _forecast_kwargs(data, reference_date=None)
    results_B = run_forecast(polls_by_race=polls_enriched, **kwargs_bypass)

    # C: Stripped, production path
    results_C = run_forecast(polls_by_race=polls_stripped, **kwargs_prod)

    xt_counts: dict[str, int] = {}
    for race_id, polls in polls_enriched.items():
        xt_counts[race_id] = sum(
            1 for p in polls if any(k.startswith("xt_") for k in p)
        )

    rows = []
    for race_id in sorted(results_A.keys()):
        if race_id not in results_B or race_id not in results_C:
            continue
        rA, rB, rC = results_A[race_id], results_B[race_id], results_C[race_id]
        parts = race_id.split()
        state = parts[1] if len(parts) > 1 else None
        pA = _vote_weighted_state_pred(rA, data["states"], data["county_votes"], state)
        pB = _vote_weighted_state_pred(rB, data["states"], data["county_votes"], state)
        pC = _vote_weighted_state_pred(rC, data["states"], data["county_votes"], state)
        theta_AC = float(np.abs(rA.theta_national - rC.theta_national).max())
        theta_BC = float(np.abs(rB.theta_national - rC.theta_national).max())
        rows.append({
            "race": race_id,
            "xt_polls": xt_counts.get(race_id, 0),
            "total_polls": len(polls_enriched.get(race_id, [])),
            "pred_live_enriched": pA,
            "pred_stripped": pC,
            "pred_tier2_bypass": pB,
            "live_vs_stripped_pp": (pA - pC) * 100 if (pA is not None and pC is not None) else None,
            "tier2_vs_stripped_pp": (pB - pC) * 100 if (pB is not None and pC is not None) else None,
            "theta_max_AC": theta_AC,
            "theta_max_BC": theta_BC,
        })
    return rows


def _print_table(rows: list[dict]) -> None:
    print()
    print("=" * 106)
    print("TIER 2 CROSSTAB IMPACT — ENRICHED vs STRIPPED (LIVE) AND TIER 2 BYPASS")
    print("=" * 106)
    print(f"{'Race':<32} {'xt/tot':>8} {'stripped':>9} {'live':>9} {'Δ_live':>8} "
          f"{'tier2':>9} {'Δ_tier2':>9} {'θ_max_live':>11} {'θ_max_tier2':>12}")
    print("-" * 106)
    for r in rows:
        marker = " *" if r["xt_polls"] > 0 else ""
        def f(v, fmt):
            return fmt.format(v) if v is not None else " " * len(fmt.format(0.0))
        print(
            f"{r['race']:<32} "
            f"{r['xt_polls']:>3}/{r['total_polls']:<4} "
            f"{f(r['pred_stripped'], '{:>9.4f}')} "
            f"{f(r['pred_live_enriched'], '{:>9.4f}')} "
            f"{f(r['live_vs_stripped_pp'], '{:>+7.3f}')} "
            f"{f(r['pred_tier2_bypass'], '{:>9.4f}')} "
            f"{f(r['tier2_vs_stripped_pp'], '{:>+8.3f}')} "
            f"{r['theta_max_AC']:>11.6f} "
            f"{r['theta_max_BC']:>12.6f}"
            f"{marker}"
        )
    print("-" * 106)
    xt_rows = [r for r in rows if r["xt_polls"] > 0]
    print(f"* = race has polls with xt_ crosstab data ({len(xt_rows)} races)")
    print()

    def _summary(label, key):
        vals = [abs(r[key]) for r in xt_rows if r[key] is not None]
        if not vals:
            print(f"  {label}: n=0")
            return
        print(f"  {label}: n={len(vals)}  mean |Δ|={np.mean(vals):.3f}pp  "
              f"max |Δ|={np.max(vals):.3f}pp  median |Δ|={np.median(vals):.3f}pp")

    print("Impact on xt_-race state predictions:")
    _summary("Live (enriched) vs stripped     ", "live_vs_stripped_pp")
    _summary("Tier 2 bypass vs stripped       ", "tier2_vs_stripped_pp")
    print()

    def _theta_summary(label, key):
        vals = [r[key] for r in xt_rows]
        if not vals:
            print(f"  {label}: n=0")
            return
        print(f"  {label}: max θ max-diff={np.max(vals):.4f}  "
              f"mean={np.mean(vals):.4f}")

    print("Impact on θ_national (max abs diff across J=100 types):")
    _theta_summary("Live (enriched) vs stripped     ", "theta_max_AC")
    _theta_summary("Tier 2 bypass vs stripped       ", "theta_max_BC")
    print()


def main() -> None:
    data = _load_data()
    rows = _run_three(data)
    _print_table(rows)

    # Write CSV for the brief.
    out = PROJECT_ROOT / "data" / "experiments" / "tier2_impact_2026-04-24.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
