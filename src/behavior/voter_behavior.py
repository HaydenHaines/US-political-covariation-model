"""Voter behavior layer: per-type turnout ratio (τ) and residual choice shift (δ).

Decomposes the difference between presidential and off-cycle elections into:
  τ (turnout ratio): off-cycle / presidential total votes per type
  δ (choice shift): residual Dem share difference after turnout composition
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OFFCYCLE_RACES = {"governor", "senate"}
PRESIDENTIAL_RACES = {"president"}


def compute_turnout_ratios(
    tract_votes: pd.DataFrame,
    type_scores: pd.DataFrame,
    n_types: int,
) -> np.ndarray:
    """Compute per-type turnout ratio τ = mean(off-cycle votes) / mean(presidential votes).

    Args:
        tract_votes: DataFrame with columns tract_geoid, race, votes_total, year.
        type_scores: DataFrame indexed by GEOID with type_j_score columns.
        n_types: Number of types (J).

    Returns:
        np.ndarray of shape (J,) with turnout ratios clipped to [0.1, 1.5].
    """
    pres = tract_votes[tract_votes["race"].isin(PRESIDENTIAL_RACES)]
    off = tract_votes[tract_votes["race"].isin(OFFCYCLE_RACES)]

    pres_mean = pres.groupby("tract_geoid")["votes_total"].mean()
    off_mean = off.groupby("tract_geoid")["votes_total"].mean()

    # Align to tracts present in both
    common = pres_mean.index.intersection(off_mean.index).intersection(type_scores.index)
    pres_mean = pres_mean.loc[common].values
    off_mean = off_mean.loc[common].values

    # Per-tract ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        tract_ratios = np.where(pres_mean > 0, off_mean / pres_mean, np.nan)

    # Build score matrix for common tracts: (n_tracts, J)
    score_cols = [f"type_{j}_score" for j in range(n_types)]
    W = type_scores.loc[common, score_cols].values  # (n_tracts, J)

    # Weighted average per type
    valid = ~np.isnan(tract_ratios)
    tau = np.zeros(n_types)
    for j in range(n_types):
        weights = W[valid, j]
        total_w = weights.sum()
        if total_w > 0:
            tau[j] = np.average(tract_ratios[valid], weights=weights)
        else:
            tau[j] = 0.7  # fallback

    return np.clip(tau, 0.1, 1.5)


def compute_choice_shifts(
    tract_votes: pd.DataFrame,
    type_scores: pd.DataFrame,
    tau: np.ndarray,
    n_types: int,
) -> np.ndarray:
    """Compute per-type residual choice shift δ = off_dem_share - pres_dem_share.

    Args:
        tract_votes: DataFrame with columns tract_geoid, race, dem_share, year.
        type_scores: DataFrame indexed by GEOID with type_j_score columns.
        tau: Turnout ratios (currently unused; kept for API consistency with future
             turnout-reweighted version).
        n_types: Number of types (J).

    Returns:
        np.ndarray of shape (J,) with per-type δ values.
    """
    pres = tract_votes[tract_votes["race"].isin(PRESIDENTIAL_RACES)]
    off = tract_votes[tract_votes["race"].isin(OFFCYCLE_RACES)]

    pres_share = pres.groupby("tract_geoid")["dem_share"].mean()
    off_share = off.groupby("tract_geoid")["dem_share"].mean()

    common = pres_share.index.intersection(off_share.index).intersection(type_scores.index)
    pres_vals = pres_share.loc[common].values
    off_vals = off_share.loc[common].values

    # Per-tract residual
    tract_residuals = off_vals - pres_vals

    score_cols = [f"type_{j}_score" for j in range(n_types)]
    W = type_scores.loc[common, score_cols].values

    valid = ~np.isnan(tract_residuals)
    delta = np.zeros(n_types)
    for j in range(n_types):
        weights = W[valid, j]
        total_w = weights.sum()
        if total_w > 0:
            delta[j] = np.average(tract_residuals[valid], weights=weights)
        else:
            delta[j] = 0.0

    return delta


def apply_behavior_adjustment(
    tract_priors: np.ndarray,
    type_scores: np.ndarray,
    tau: np.ndarray,
    delta: np.ndarray,
    is_offcycle: bool,
) -> np.ndarray:
    """Adjust tract priors for off-cycle elections using τ and δ.

    Args:
        tract_priors: Dem share priors, shape (n_tracts,).
        type_scores: Type membership scores, shape (n_tracts, J).
        tau: Turnout ratios per type, shape (J,).
        delta: Choice shifts per type, shape (J,).
        is_offcycle: Whether the target election is off-cycle.

    Returns:
        Adjusted priors, shape (n_tracts,), clipped to [0, 1].
    """
    if not is_offcycle:
        return tract_priors.copy()

    # Weight type scores by τ to get off-cycle composition
    # τ-weighted scores: types with lower turnout get downweighted
    offcycle_weights = type_scores * tau[np.newaxis, :]  # (n_tracts, J)
    row_sums = offcycle_weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    offcycle_norm = offcycle_weights / row_sums  # normalized, (n_tracts, J)

    # Per-tract adjustment: weighted sum of δ by off-cycle composition
    adjustment = offcycle_norm @ delta  # (n_tracts,)

    return np.clip(tract_priors + adjustment, 0.0, 1.0)


def train_and_save(
    tract_votes_path: str | Path,
    assignments_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Load data, compute τ and δ, save to disk.

    Args:
        tract_votes_path: Path to tract_votes_dra.parquet.
        assignments_path: Path to national_tract_assignments.parquet.
        output_dir: Directory to write tau.npy, delta.npy, and summary.json.

    Returns:
        Summary dict with tau and delta statistics.
    """
    tract_votes = pd.read_parquet(tract_votes_path)
    assignments = pd.read_parquet(assignments_path)

    # Determine n_types from score columns
    score_cols = [c for c in assignments.columns if c.endswith("_score")]
    n_types = len(score_cols)

    # Set index to GEOID
    if "GEOID" in assignments.columns:
        assignments = assignments.set_index("GEOID")

    tau = compute_turnout_ratios(tract_votes, assignments, n_types)
    delta = compute_choice_shifts(tract_votes, assignments, tau, n_types)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "tau.npy", tau)
    np.save(output_dir / "delta.npy", delta)

    summary = {
        "n_types": n_types,
        "tau_mean": float(tau.mean()),
        "tau_std": float(tau.std()),
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train voter behavior layer (τ + δ)")
    parser.add_argument(
        "--tract-votes",
        default="data/tracts/tract_votes_dra.parquet",
        help="Path to tract votes parquet",
    )
    parser.add_argument(
        "--assignments",
        default="data/tracts/national_tract_assignments.parquet",
        help="Path to tract type assignments parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/behavior",
        help="Output directory for tau.npy, delta.npy, summary.json",
    )
    args = parser.parse_args()

    summary = train_and_save(args.tract_votes, args.assignments, args.output_dir)
    print(f"Voter behavior layer trained: {summary}")
