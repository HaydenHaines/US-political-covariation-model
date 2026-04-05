"""LOO-optimized J selection sweep.

Sweeps J over a candidate list and evaluates each using both the standard
(inflated) holdout metric and the LOO (honest) holdout metric discovered in
S196. The standard metric inflates by ~0.22 due to type self-prediction in
small types.

Goal: identify the J that maximizes LOO r rather than standard r, as the
standard metric was shown to be biased in favor of smaller types.

Usage:
    uv run python scripts/experiment_j_sweep_loo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types
from src.validation.validate_types import (
    holdout_accuracy_county_prior,
    holdout_accuracy_county_prior_loo,
)

# ── Configuration ─────────────────────────────────────────────────────────────

HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

PRESIDENTIAL_WEIGHT = 8.0
MIN_YEAR = 2008  # Match production pipeline

J_CANDIDATES = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]


def parse_start_year(col: str) -> int | None:
    """Extract the start year from a shift column name like 'pres_d_shift_08_12'.

    Returns the 4-digit start year, or None if not parseable.
    Two-digit years: y >= 50 -> 1900+y, else 2000+y.
    """
    parts = col.split("_")
    # Year pair is last two parts: e.g. '08', '12'
    if len(parts) < 2:
        return None
    try:
        y2_str = parts[-2]  # start year (2-digit)
        y2 = int(y2_str)
        start_year = y2 + (1900 if y2 >= 50 else 2000)
        return start_year
    except (ValueError, IndexError):
        return None


def main() -> None:
    print("=" * 70)
    print("LOO J Selection Sweep — WetherVane S197")
    print("=" * 70)

    # Load shift matrix
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    print(f"\nLoading shift matrix from {shifts_path}...")
    df = pd.read_parquet(shifts_path)
    print(f"  Loaded: {df.shape[0]} counties × {df.shape[1]} columns")

    # All non-FIPS columns
    all_cols = [c for c in df.columns if c != "county_fips"]

    # Identify holdout columns
    holdout_col_names = [c for c in all_cols if c in HOLDOUT_COLUMNS]
    assert len(holdout_col_names) == 3, f"Expected 3 holdout cols, got: {holdout_col_names}"

    # Training columns: not holdout, start year >= MIN_YEAR
    training_col_names = []
    for c in all_cols:
        if c in HOLDOUT_COLUMNS:
            continue
        start_year = parse_start_year(c)
        if start_year is not None and start_year >= MIN_YEAR:
            training_col_names.append(c)

    print(f"  Training columns (start >= {MIN_YEAR}): {len(training_col_names)}")
    print(f"  Holdout columns: {holdout_col_names}")

    # Build full column order: training first, then holdout
    # This establishes the indices into the full scaled matrix
    ordered_cols = training_col_names + holdout_col_names
    training_col_indices = list(range(len(training_col_names)))
    holdout_col_indices = list(range(len(training_col_names), len(ordered_cols)))

    print(f"  Training col indices: [{training_col_indices[0]}..{training_col_indices[-1]}]")
    print(f"  Holdout col indices: {holdout_col_indices}")

    # Extract raw matrix (training + holdout)
    raw_full = df[ordered_cols].values  # N × (D_train + D_hold)
    raw_training = df[training_col_names].values  # N × D_train

    # Apply StandardScaler to the TRAINING columns only (fit on training data)
    print("\nApplying StandardScaler (fit on training columns)...")
    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(raw_training)

    # Scale holdout columns using same scaler's statistics
    # (We need to scale them using each column's own scale for fair comparison)
    # Scale holdout independently
    holdout_raw = df[holdout_col_names].values
    holdout_scaler = StandardScaler()
    scaled_holdout = holdout_scaler.fit_transform(holdout_raw)

    # Full scaled matrix: training + holdout
    scaled_full = np.hstack([scaled_training, scaled_holdout])

    # Apply presidential weighting to training columns (post-scaling, production match)
    pres_train_indices = [i for i, c in enumerate(training_col_names) if "pres_" in c]
    scaled_training_weighted = scaled_training.copy()
    scaled_training_weighted[:, pres_train_indices] *= PRESIDENTIAL_WEIGHT

    # Apply presidential weighting to holdout columns in full matrix
    pres_hold_indices = [
        len(training_col_names) + i
        for i, c in enumerate(holdout_col_names) if "pres_" in c
    ]
    scaled_full_weighted = scaled_full.copy()
    scaled_full_weighted[:, pres_train_indices] *= PRESIDENTIAL_WEIGHT
    scaled_full_weighted[:, pres_hold_indices] *= PRESIDENTIAL_WEIGHT

    print(f"  Applied presidential weight={PRESIDENTIAL_WEIGHT} to:")
    print(f"    {len(pres_train_indices)} training pres cols + {len(pres_hold_indices)} holdout pres cols")

    # ── Sweep ─────────────────────────────────────────────────────────────────

    print(f"\nSweeping J over {J_CANDIDATES}...")
    print("-" * 70)

    results = []

    for j in J_CANDIDATES:
        t0 = time.time()
        print(f"  J={j:3d}  fitting KMeans...", end="", flush=True)

        # Discover types on weighted training matrix
        result = discover_types(scaled_training_weighted, j=j, random_state=42)
        scores = result.scores  # N × J

        # Standard (inflated) holdout metric
        std_result = holdout_accuracy_county_prior(
            scores=scores,
            shift_matrix=scaled_full_weighted,
            training_cols=training_col_indices,
            holdout_cols=holdout_col_indices,
        )
        std_r = std_result["mean_r"]
        std_rmse = std_result["mean_rmse"]

        # LOO (honest) holdout metric
        loo_result = holdout_accuracy_county_prior_loo(
            scores=scores,
            shift_matrix=scaled_full_weighted,
            training_cols=training_col_indices,
            holdout_cols=holdout_col_indices,
        )
        loo_r = loo_result["mean_r"]
        loo_rmse = loo_result["mean_rmse"]

        elapsed = time.time() - t0
        inflation = std_r - loo_r

        print(f"  std_r={std_r:.4f}  loo_r={loo_r:.4f}  inflation={inflation:+.4f}  [{elapsed:.1f}s]")

        results.append({
            "j": j,
            "std_r": std_r,
            "std_rmse": std_rmse,
            "loo_r": loo_r,
            "loo_rmse": loo_rmse,
            "inflation": inflation,
            "per_dim_std_r": std_result["per_dim_r"],
            "per_dim_loo_r": loo_result["per_dim_r"],
        })

    # ── Results table ─────────────────────────────────────────────────────────

    results_df = pd.DataFrame(results)

    best_std_j = int(results_df.loc[results_df["std_r"].idxmax(), "j"])
    best_loo_j = int(results_df.loc[results_df["loo_r"].idxmax(), "j"])

    print("\n" + "=" * 70)
    print("J SWEEP RESULTS")
    print("=" * 70)
    print(f"\n{'J':>5}  {'std_r':>8}  {'loo_r':>8}  {'inflation':>10}  {'std_rmse':>10}  {'loo_rmse':>10}")
    print("-" * 70)
    for row in results:
        marker = ""
        if row["j"] == best_loo_j:
            marker += " <-- LOO OPTIMAL"
        if row["j"] == best_std_j and best_std_j != best_loo_j:
            marker += " <-- STD OPTIMAL"
        if row["j"] == 100:
            marker += " [current]"
        print(
            f"{row['j']:>5}  {row['std_r']:>8.4f}  {row['loo_r']:>8.4f}  "
            f"{row['inflation']:>+10.4f}  {row['std_rmse']:>10.4f}  {row['loo_rmse']:>10.4f}"
            f"{marker}"
        )

    print("-" * 70)
    print(f"\nStandard-optimal J: {best_std_j} (std_r = {results_df.loc[results_df['std_r'].idxmax(), 'std_r']:.4f})")
    print(f"LOO-optimal J:      {best_loo_j} (loo_r = {results_df.loc[results_df['loo_r'].idxmax(), 'loo_r']:.4f})")

    # Per-dimension breakdown at key J values
    print(f"\n{'Per-dimension r at key J values':}")
    print(f"  Holdout dims: {holdout_col_names}")
    for row in results:
        if row["j"] in [best_loo_j, best_std_j, 100]:
            label = f"J={row['j']}"
            if row["j"] == best_loo_j:
                label += " (LOO-opt)"
            if row["j"] == 100:
                label += " (current)"
            std_dims = [f"{r:.4f}" for r in row["per_dim_std_r"]]
            loo_dims = [f"{r:.4f}" for r in row["per_dim_loo_r"]]
            print(f"  {label}")
            print(f"    std_r per dim: {std_dims}")
            print(f"    loo_r per dim: {loo_dims}")

    print("\n" + "=" * 70)

    # Save results
    out_dir = PROJECT_ROOT / "data" / "communities"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_df = results_df[["j", "std_r", "std_rmse", "loo_r", "loo_rmse", "inflation"]]
    out_path = out_dir / "j_sweep_loo_results.parquet"
    save_df.to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    return results_df


if __name__ == "__main__":
    main()
