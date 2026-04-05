"""Poll freshness sensitivity sweep: how does half_life_days affect forecasts?

Sweeps half_life_days over a range of values and reports:
  - Number of polls with effective n_sample > 10 after decay per race
  - State-level predictions for key competitive races
  - Mean absolute change from the 30-day baseline

The sweep operates on the *current* poll dataset (polls_2026.csv) against the
current type model, using reference_date = "2026-04-04" for consistency.

We do NOT require the full county model on disk -- if type_assignments.parquet
is missing, the script exits with a clear message.  All other data files fall
back gracefully (equal vote weights, default crosswalks, etc.).

Usage
-----
    uv run python scripts/experiments/sweep_poll_half_life.py
    uv run python scripts/experiments/sweep_poll_half_life.py \\
        --half-lives 15 20 30 45 60 90 120 \\
        --reference-date 2026-04-04 \\
        --n-effective-threshold 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution — worktree-safe
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import config as _cfg  # noqa: E402
from src.prediction.county_priors import load_county_priors_with_ridge  # noqa: E402
from src.prediction.forecast_engine import run_forecast  # noqa: E402
from src.propagation.poll_decay import apply_time_decay  # noqa: E402
from src.propagation.propagate_polls import PollObservation  # noqa: E402

log = logging.getLogger(__name__)

# Baseline half-life (the current production value in prediction_params.json)
_BASELINE_HALF_LIFE: float = 30.0

# Key competitive races to track in the prediction table.
# These are checked against the start of the race_id string (case-insensitive).
_KEY_RACES: list[str] = [
    "2026 FL Senate",
    "2026 GA Senate",
    "2026 PA Senate",
    "2026 OH Senate",
    "2026 MI Senate",
    "2026 WI Senate",
    "2026 TX Senate",
    "2026 AZ Senate",
    "2026 NC Senate",
    "2026 WI Governor",
    "2026 PA Governor",
    "2026 OH Governor",
    "2026 MI Governor",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_type_model() -> tuple[list[str], np.ndarray]:
    """Load county FIPS and soft type scores from disk.

    Returns
    -------
    county_fips : list[str]
    type_scores : ndarray of shape (N, J)
    """
    ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    if not ta_path.exists():
        raise FileNotFoundError(
            f"Type assignments not found at {ta_path}.\n"
            "Run `uv run python -m src.discovery.run_type_discovery` first."
        )
    ta = pd.read_parquet(ta_path)
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)
    score_cols = sorted([c for c in ta.columns if c.endswith("_score")])
    return ta["county_fips"].tolist(), ta[score_cols].values


def _load_county_votes(county_fips: list[str]) -> np.ndarray:
    """Load 2024 total vote counts; falls back to equal weights."""
    votes = np.ones(len(county_fips))
    path = PROJECT_ROOT / "data" / "raw" / "medsl_county_presidential_2024.parquet"
    if path.exists():
        vdf = pd.read_parquet(path)
        if "county_fips" in vdf.columns and "totalvotes" in vdf.columns:
            vmap = dict(zip(
                vdf["county_fips"].astype(str).str.zfill(5),
                vdf["totalvotes"],
            ))
            votes = np.array([vmap.get(f, 1.0) for f in county_fips])
    return votes


def _load_polls_2026() -> tuple[dict[str, list[dict]], list[PollObservation]]:
    """Load 2026 polls as both run_forecast dict and flat PollObservation list.

    Returns
    -------
    polls_by_race : dict[str, list[dict]]
        Format expected by prepare_polls / run_forecast.
    flat_polls : list[PollObservation]
        Flat list (pre-weighting) for decay analysis.
    """
    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    if not polls_path.exists():
        raise FileNotFoundError(f"Poll file not found: {polls_path}")

    df = pd.read_csv(polls_path)
    if "geography" in df.columns and "state" not in df.columns:
        df = df.rename(columns={"geography": "state"})
    if "geo_level" in df.columns:
        df = df[df["geo_level"] == "state"].copy()
    if "race" in df.columns:
        df = df[~df["race"].str.contains("Generic Ballot", case=False, na=False)]

    polls_by_race: dict[str, list[dict]] = {}
    flat_polls: list[PollObservation] = []

    for _, row in df.iterrows():
        try:
            dem_share = float(row["dem_share"])
            n_sample = int(float(row["n_sample"])) if pd.notna(row.get("n_sample")) else 600
        except (ValueError, TypeError):
            continue
        if not (0.0 < dem_share < 1.0) or n_sample <= 0:
            continue

        race = str(row.get("race", "")).strip()
        state = str(row.get("state", "")).strip()
        date = str(row.get("date", "")).strip()
        pollster = str(row.get("pollster", "")).strip()

        d = {
            "dem_share": dem_share,
            "n_sample": n_sample,
            "state": state,
            "date": date,
            "pollster": pollster,
            "notes": str(row.get("notes", "")),
        }
        polls_by_race.setdefault(race, []).append(d)

        flat_polls.append(PollObservation(
            geography=state,
            dem_share=dem_share,
            n_sample=n_sample,
            race=race,
            date=date,
            pollster=pollster,
            geo_level="state",
        ))

    return polls_by_race, flat_polls


# ---------------------------------------------------------------------------
# Effective-poll counting after decay
# ---------------------------------------------------------------------------


def count_effective_polls(
    polls_by_race: dict[str, list[dict]],
    reference_date: str,
    half_life_days: float,
    n_effective_threshold: int = 10,
) -> dict[str, int]:
    """Count polls per race where effective n_sample > threshold after time decay.

    We apply only time decay here (not house effects, quality, or primary discount)
    so the count reflects freshness sensitivity alone.

    Returns
    -------
    dict[str, int]
        Mapping of race_id -> count of polls with n_effective > threshold.
    """
    counts: dict[str, int] = {}
    for race, poll_list in polls_by_race.items():
        obs = [
            PollObservation(
                geography=p["state"],
                dem_share=p["dem_share"],
                n_sample=p["n_sample"],
                race=race,
                date=p.get("date", ""),
                pollster=p.get("pollster", ""),
                geo_level="state",
            )
            for p in poll_list
            if p.get("date") and str(p.get("date", "")).lower() not in ("", "nan", "none")
        ]
        decayed = apply_time_decay(
            obs, reference_date=reference_date, half_life_days=half_life_days,
        )
        counts[race] = sum(1 for d in decayed if d.n_sample > n_effective_threshold)
    return counts


# ---------------------------------------------------------------------------
# Single half-life evaluation
# ---------------------------------------------------------------------------


def _vote_weighted_state_pred(
    state: str,
    county_fips: list[str],
    states: list[str],
    county_votes: np.ndarray,
    county_preds: np.ndarray,
) -> float | None:
    """Compute vote-weighted mean prediction for a state."""
    mask = [s == state for s in states]
    if not any(mask):
        return None
    mask_arr = np.array(mask)
    w = county_votes[mask_arr]
    p = county_preds[mask_arr]
    total = w.sum()
    if total <= 0:
        return float(p.mean())
    return float((p * w).sum() / total)


def evaluate_half_life(
    half_life_days: float,
    county_fips: list[str],
    type_scores: np.ndarray,
    county_priors: np.ndarray,
    states: list[str],
    county_votes: np.ndarray,
    polls_by_race: dict[str, list[dict]],
    reference_date: str,
    n_effective_threshold: int = 10,
) -> dict:
    """Run the full forecast for one half_life_days value.

    Returns a dict with effective poll counts and key-race predictions.
    """
    # Load config lam/mu/mode (never hardcode them here)
    params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
    params = json.loads(params_path.read_text())
    lam: float = params["forecast"]["lam"]
    mu: float = params["forecast"]["mu"]
    w_mode: str = params["forecast"]["w_vector_mode"]

    effective_counts = count_effective_polls(
        polls_by_race,
        reference_date=reference_date,
        half_life_days=half_life_days,
        n_effective_threshold=n_effective_threshold,
    )

    all_race_ids = list(polls_by_race.keys())

    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes,
        polls_by_race=polls_by_race,
        races=all_race_ids,
        lam=lam,
        mu=mu,
        generic_ballot_shift=0.0,
        w_vector_mode=w_mode,
        reference_date=reference_date,
        half_life_days=half_life_days,
        # Use config pre_primary_discount so the sweep isolates half_life_days
        pre_primary_discount=float(
            params.get("poll_weighting", {}).get("pre_primary_discount", 0.5)
        ),
    )

    # State-level predictions for key races (vote-weighted)
    state_preds: dict[str, float] = {}
    for race_id, fr in forecast_results.items():
        # Check if this is a key race
        matched_key = next(
            (k for k in _KEY_RACES if race_id.lower().startswith(k.lower())),
            None,
        )
        if matched_key is None:
            continue

        # Extract the target state abbreviation from the race label
        # Pattern: "2026 FL Senate" -> "FL"
        parts = race_id.split()
        if len(parts) >= 2:
            target_state = parts[1].upper()
        else:
            continue

        pred = _vote_weighted_state_pred(
            target_state,
            county_fips,
            states,
            county_votes,
            fr.county_preds_local,
        )
        if pred is not None:
            state_preds[race_id] = pred

    return {
        "half_life_days": half_life_days,
        "effective_counts": effective_counts,
        "state_preds": state_preds,
        "total_effective_polls": sum(effective_counts.values()),
    }


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def run_sweep(
    half_life_values: list[float],
    reference_date: str,
    n_effective_threshold: int,
    verbose: bool = True,
) -> list[dict]:
    """Run the half-life sweep over all provided values.

    Returns a list of result dicts (one per half_life value).
    """
    if verbose:
        print(f"Loading type model from {PROJECT_ROOT / 'data' / 'communities'} ...")
    county_fips, type_scores = _load_type_model()
    N, J = type_scores.shape
    if verbose:
        print(f"  {N} counties × {J} types")

    if verbose:
        print("Loading county priors ...")
    county_priors = load_county_priors_with_ridge(county_fips)

    states = [_cfg.STATE_ABBR.get(f[:2], "??") for f in county_fips]

    if verbose:
        print("Loading county vote weights ...")
    county_votes = _load_county_votes(county_fips)

    if verbose:
        print("Loading 2026 polls ...")
    polls_by_race, _ = _load_polls_2026()
    total_polls = sum(len(v) for v in polls_by_race.values())
    if verbose:
        print(f"  {total_polls} polls across {len(polls_by_race)} races")
        print()

    results = []
    for hl in half_life_values:
        if verbose:
            print(f"  Evaluating half_life_days={hl:.0f} ...", end=" ", flush=True)
        row = evaluate_half_life(
            half_life_days=hl,
            county_fips=county_fips,
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls_by_race,
            reference_date=reference_date,
            n_effective_threshold=n_effective_threshold,
        )
        results.append(row)
        if verbose:
            print(f"effective_polls={row['total_effective_polls']}")

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], baseline_hl: float = _BASELINE_HALF_LIFE) -> None:
    """Print summary table to stdout."""
    if not results:
        print("No results to display.")
        return

    # Collect all key races that appear in any result
    all_races = sorted({
        race
        for r in results
        for race in r["state_preds"].keys()
    })

    baseline_preds = next(
        (r["state_preds"] for r in results if abs(r["half_life_days"] - baseline_hl) < 0.1),
        None,
    )

    # --- Header ---
    print()
    print("=" * 80)
    print(f"POLL HALF-LIFE SENSITIVITY SWEEP  (reference_date={results[0].get('ref_date', 'N/A')})")
    print(f"Baseline: half_life_days={baseline_hl:.0f}")
    print("=" * 80)

    # --- Section 1: Effective poll counts ---
    print()
    print("Effective polls (n_sample > threshold) per half-life value")
    print("-" * 50)
    hl_header = "  ".join(f"{r['half_life_days']:>6.0f}d" for r in results)
    print(f"{'race':<30} {hl_header}")
    print("-" * 80)

    # Collect all races across all results
    all_count_races = sorted({
        race
        for r in results
        for race in r["effective_counts"].keys()
    })
    for race in all_count_races:
        counts_str = "  ".join(
            f"{r['effective_counts'].get(race, 0):>7d}" for r in results
        )
        print(f"{race:<30} {counts_str}")

    totals_str = "  ".join(f"{r['total_effective_polls']:>7d}" for r in results)
    print("-" * 80)
    print(f"{'TOTAL':<30} {totals_str}")

    # --- Section 2: Key-race predictions ---
    if all_races:
        print()
        print("State-level Dem share predictions for key competitive races")
        print("-" * 80)
        race_header = "  ".join(f"{r['half_life_days']:>6.0f}d" for r in results)
        print(f"{'race':<32} {race_header}  {'|'} chg_vs_{baseline_hl:.0f}d")
        print("-" * 80)

        for race in all_races:
            pred_str_parts = []
            for r in results:
                pred = r["state_preds"].get(race)
                pred_str_parts.append(f"{pred*100:>6.1f}%" if pred is not None else f"{'N/A':>7}")
            pred_row = "  ".join(pred_str_parts)

            # Change vs baseline
            if baseline_preds is not None:
                base_pred = baseline_preds.get(race)
            else:
                base_pred = None
            chg_parts = []
            for r in results:
                p = r["state_preds"].get(race)
                if p is not None and base_pred is not None:
                    chg = (p - base_pred) * 100
                    chg_parts.append(f"{chg:+.1f}pp")
                else:
                    chg_parts.append("  N/A ")
            chg_row = "  ".join(chg_parts)

            print(f"{race:<32} {pred_row}  {'|'} {chg_row}")

    # --- Section 3: Mean absolute change from baseline ---
    print()
    print("Mean absolute change from baseline (vs half_life_days=30d)")
    print("-" * 50)
    for r in results:
        if abs(r["half_life_days"] - baseline_hl) < 0.1:
            print(f"  half_life={r['half_life_days']:.0f}d : (baseline)")
            continue
        changes = []
        for race, pred in r["state_preds"].items():
            if baseline_preds and race in baseline_preds:
                changes.append(abs(pred - baseline_preds[race]) * 100)
        if changes:
            mean_abs_chg = sum(changes) / len(changes)
            max_abs_chg = max(changes)
            print(
                f"  half_life={r['half_life_days']:.0f}d : "
                f"mean_abs_chg={mean_abs_chg:.2f}pp  "
                f"max_abs_chg={max_abs_chg:.2f}pp"
            )
        else:
            print(f"  half_life={r['half_life_days']:.0f}d : (no key race predictions)")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep poll time-decay half_life_days and report forecast sensitivity."
    )
    parser.add_argument(
        "--half-lives",
        nargs="+",
        type=float,
        default=[15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0],
        help="Half-life values in days to sweep (default: 15 20 30 45 60 90 120)",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default="2026-04-04",
        help="Reference date for decay calculation (YYYY-MM-DD, default: 2026-04-04)",
    )
    parser.add_argument(
        "--n-effective-threshold",
        type=int,
        default=10,
        help="Minimum effective n_sample to count a poll as active (default: 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("Poll half-life sensitivity sweep")
    print(f"  half_life values: {args.half_lives}")
    print(f"  reference_date: {args.reference_date}")
    print(f"  n_effective_threshold: {args.n_effective_threshold}")
    print()

    results = run_sweep(
        half_life_values=args.half_lives,
        reference_date=args.reference_date,
        n_effective_threshold=args.n_effective_threshold,
        verbose=True,
    )

    # Attach ref_date to results for display
    for r in results:
        r["ref_date"] = args.reference_date

    print_summary(results, baseline_hl=_BASELINE_HALF_LIFE)


if __name__ == "__main__":
    main()
