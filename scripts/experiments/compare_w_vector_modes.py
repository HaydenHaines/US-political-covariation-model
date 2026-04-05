"""Compare W vector modes: core vs full vs baseline (no enrichment).

Runs the forecast pipeline three times and reports prediction differences.
Usage: uv run python scripts/compare_w_vector_modes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date
from src.prediction.forecast_engine import run_forecast


def main():
    # Load shared inputs
    ta_df = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet")
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values

    # County priors
    from src.prediction.county_priors import compute_county_priors
    county_priors = compute_county_priors(county_fips)
    ridge_path = PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
    if ridge_path.exists():
        rdf = pd.read_parquet(ridge_path)
        rdf["county_fips"] = rdf["county_fips"].astype(str).str.zfill(5)
        rmap = dict(zip(rdf["county_fips"], rdf["ridge_pred_dem_share"]))
        for i, f in enumerate(county_fips):
            if f in rmap:
                county_priors[i] = rmap[f]

    states = [f[:2] for f in county_fips]  # simplified
    from src.core import config as _cfg
    states = [_cfg.STATE_ABBR.get(f[:2], "??") for f in county_fips]
    county_votes = np.ones(len(county_fips))

    # Type profiles
    tp = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet")

    # Load polls
    polls_csv = pd.read_csv(PROJECT_ROOT / "data" / "polls" / "polls_2026.csv")
    polls_by_race: dict[str, list[dict]] = {}
    for _, row in polls_csv.iterrows():
        race = row["race"]
        if "Senate" not in race:
            continue
        polls_by_race.setdefault(race, []).append({
            "dem_share": float(row["dem_share"]),
            "n_sample": int(row["n_sample"]),
            "state": str(row["geography"]),
            "date": str(row.get("date", "")),
            "pollster": str(row.get("pollster", "")),
            "notes": str(row.get("notes", "")),
        })

    races = list(polls_by_race.keys())
    ref_date = str(date.today())

    modes = {
        "baseline": {"type_profiles": None, "w_vector_mode": "core", "reference_date": None},
        "core": {"type_profiles": tp, "w_vector_mode": "core", "reference_date": ref_date},
        "full": {"type_profiles": tp, "w_vector_mode": "full", "reference_date": ref_date},
    }

    results = {}
    for mode_name, kwargs in modes.items():
        fr = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls_by_race,
            races=races,
            **kwargs,
        )
        # State-level predictions (simple mean across state counties)
        state_preds = {}
        for race_id, res in fr.items():
            st = race_id.split()[1]  # "2026 GA Senate" → "GA"
            mask = np.array([s == st for s in states])
            if mask.any():
                state_preds[race_id] = float(res.county_preds_local[mask].mean())
        results[mode_name] = state_preds

    # Compare
    print(f"\n{'Race':<25s} {'Baseline':>10s} {'Core':>10s} {'Full':>10s} {'Δ core':>8s} {'Δ full':>8s}")
    print("-" * 75)
    for race in sorted(races):
        bl = results["baseline"].get(race, 0.5)
        co = results["core"].get(race, 0.5)
        fu = results["full"].get(race, 0.5)
        d_co = (co - bl) * 100
        d_fu = (fu - bl) * 100
        print(f"{race:<25s} {bl:>10.4f} {co:>10.4f} {fu:>10.4f} {d_co:>+7.2f}pp {d_fu:>+7.2f}pp")

    avg_shift_core = np.mean([abs(results["core"][r] - results["baseline"][r]) * 100
                               for r in races if r in results["baseline"] and r in results["core"]])
    avg_shift_full = np.mean([abs(results["full"][r] - results["baseline"][r]) * 100
                               for r in races if r in results["baseline"] and r in results["full"]])
    print(f"\nAvg |shift| core: {avg_shift_core:.2f}pp")
    print(f"Avg |shift| full: {avg_shift_full:.2f}pp")
    print(f"\nValidation criterion: shifts should average < 2pp")


if __name__ == "__main__":
    main()
