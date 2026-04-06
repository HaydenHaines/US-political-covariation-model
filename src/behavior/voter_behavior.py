"""Voter behavior layer: per-type turnout ratio (τ) and residual choice shift (δ).

Decomposes the difference between presidential and off-cycle elections into:
  τ (turnout ratio): off-cycle / presidential total votes per type
  δ (residual choice shift): off-cycle Dem share minus presidential Dem share per type

Input data (T.1/T.3 pipeline outputs):
  tract_elections.parquet  — columns: tract_geoid, year, race_type, total_votes, dem_share
  tract_type_assignments.parquet — columns: tract_geoid, type_0_score..type_99_score, dominant_type

The race_type codes used here are:
  PRES                 → presidential (high-turnout baseline)
  GOV                  → governor (off-cycle)
  SEN/SEN_SPEC/SEN_ROFF/SEN_SPECROFF → Senate (off-cycle)
All other race_types (CONG, AG, LTG, etc.) are ignored because they don't provide
the consistent two-party presidential-vs-offcycle comparison needed for τ and δ.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Race types in the T.1 parquet that map to presidential races.
# Only one code exists but we use a set for defensive future-proofing.
PRESIDENTIAL_RACE_TYPES = {"PRES"}

# Senate race_type codes: regular + special + runoff variants.
# All are treated as "off-cycle" regardless of year they fall in.
SENATE_RACE_TYPES = {"SEN", "SEN_SPEC", "SEN_ROFF", "SEN_SPECROFF"}

# Governor codes — could include runoffs in future data expansions.
GOVERNOR_RACE_TYPES = {"GOV", "GOV_ROFF"}

# Combined off-cycle set used for filtering.
OFFCYCLE_RACE_TYPES = SENATE_RACE_TYPES | GOVERNOR_RACE_TYPES

# Internal column name used after mapping race_type → race.
# Keeps the rest of the logic clean and independent of source format.
PRESIDENTIAL_RACES = {"president"}
OFFCYCLE_RACES = {"governor", "senate"}

# Default paths (relative to project root, matching T.1/T.3 outputs).
DEFAULT_TRACT_VOTES_PATH = "data/assembled/tract_elections.parquet"
DEFAULT_ASSIGNMENTS_PATH = "data/communities/tract_type_assignments.parquet"
DEFAULT_OUTPUT_DIR = "data/behavior"

# τ is clipped to this range. Below 0.1 suggests data noise; above 1.5 would
# imply off-cycle outdraws presidential, which doesn't happen systemically.
TAU_CLIP_MIN = 0.1
TAU_CLIP_MAX = 1.5

# Fallback τ for types with no data coverage.
TAU_FALLBACK = 0.7


def _map_race_type_to_race(tract_elections: pd.DataFrame) -> pd.DataFrame:
    """Map T.1 race_type codes to the three canonical race labels.

    Keeps only presidential and off-cycle rows. Non-matching race_types
    (CONG, AG, LTG, AUD, SOS, TREAS, etc.) are dropped because they don't
    provide the consistent two-party comparison needed for τ and δ.

    Args:
        tract_elections: DataFrame with columns including race_type, total_votes.

    Returns:
        Filtered DataFrame with new column `race` in {president, governor, senate}
        and renamed column `votes_total` (from total_votes).
    """
    mapping = {code: "president" for code in PRESIDENTIAL_RACE_TYPES}
    mapping.update({code: "senate" for code in SENATE_RACE_TYPES})
    mapping.update({code: "governor" for code in GOVERNOR_RACE_TYPES})

    known_types = set(mapping.keys())
    df = tract_elections[tract_elections["race_type"].isin(known_types)].copy()
    df["race"] = df["race_type"].map(mapping)
    df = df.rename(columns={"total_votes": "votes_total"})
    return df


def _load_and_prepare_assignments(assignments_path: str | Path) -> pd.DataFrame:
    """Load tract type assignments and set tract_geoid as the index.

    Handles the known DRA gotcha: clustering may produce duplicate GEOIDs.
    We deduplicate before indexing so downstream joins are 1:1.

    Args:
        assignments_path: Path to tract_type_assignments.parquet.

    Returns:
        DataFrame indexed by tract_geoid with type_j_score columns.
    """
    assignments = pd.read_parquet(assignments_path)

    if "tract_geoid" in assignments.columns:
        # Deduplicate: keep first occurrence per GEOID (clustering artifact).
        # The DRA pipeline can produce 112K rows for 81K unique tracts.
        n_before = len(assignments)
        assignments = assignments.drop_duplicates(subset="tract_geoid")
        n_after = len(assignments)
        if n_before != n_after:
            import warnings
            warnings.warn(
                f"Dropped {n_before - n_after} duplicate GEOIDs from assignments "
                f"(kept {n_after} unique tracts).",
                UserWarning,
                stacklevel=3,
            )
        assignments = assignments.set_index("tract_geoid")

    return assignments


def compute_turnout_ratios(
    tract_votes: pd.DataFrame,
    type_scores: pd.DataFrame,
    n_types: int,
) -> np.ndarray:
    """Compute per-type turnout ratio τ = mean(off-cycle votes) / mean(presidential votes).

    τ captures which community types show lower participation in midterm elections.
    For example, high-turnout presidential-only voters (often younger or sporadic) drop
    out in off-cycle races, producing τ < 1.0 for those tract types.

    The per-tract ratio (off_mean / pres_mean) is aggregated to the type level using
    the soft membership scores as weights. Types whose tracts all have τ ≈ 0.7 means
    that community participates at 70% of its presidential rate in off-cycle years.

    Args:
        tract_votes: DataFrame with columns tract_geoid, race, votes_total, year.
                     `race` must be in {president, governor, senate}.
        type_scores: DataFrame indexed by GEOID with type_j_score columns.
        n_types: Number of types (J).

    Returns:
        np.ndarray of shape (J,) with turnout ratios clipped to [TAU_CLIP_MIN, TAU_CLIP_MAX].
    """
    pres = tract_votes[tract_votes["race"].isin(PRESIDENTIAL_RACES)]
    off = tract_votes[tract_votes["race"].isin(OFFCYCLE_RACES)]

    # Average across all years per tract — smooths out individual cycle noise.
    pres_mean = pres.groupby("tract_geoid")["votes_total"].mean()
    off_mean = off.groupby("tract_geoid")["votes_total"].mean()

    # Align to tracts present in both datasets and in the type assignments.
    common = pres_mean.index.intersection(off_mean.index).intersection(type_scores.index)
    pres_mean = pres_mean.loc[common].values
    off_mean = off_mean.loc[common].values

    # Per-tract ratio. Zero presidential turnout → NaN (excluded from weighted avg).
    with np.errstate(divide="ignore", invalid="ignore"):
        tract_ratios = np.where(pres_mean > 0, off_mean / pres_mean, np.nan)

    # Build score matrix for common tracts: shape (n_tracts, J).
    score_cols = [f"type_{j}_score" for j in range(n_types)]
    W = type_scores.loc[common, score_cols].values

    valid = ~np.isnan(tract_ratios)
    tau = np.zeros(n_types)
    for j in range(n_types):
        weights = W[valid, j]
        total_w = weights.sum()
        if total_w > 0:
            tau[j] = np.average(tract_ratios[valid], weights=weights)
        else:
            # Fallback for types with no data coverage in the training set.
            tau[j] = TAU_FALLBACK

    return np.clip(tau, TAU_CLIP_MIN, TAU_CLIP_MAX)


def compute_choice_shifts(
    tract_votes: pd.DataFrame,
    type_scores: pd.DataFrame,
    tau: np.ndarray,
    n_types: int,
) -> np.ndarray:
    """Compute per-type residual choice shift δ = off_dem_share - pres_dem_share.

    δ captures genuine preference differences in off-cycle elections beyond what
    would be explained by which voters show up (τ). A positive δ means the type
    votes more Democratic in off-cycle races; negative means more Republican.

    In practice |δ| is small (< 0.05 for most types) because turnout composition
    already accounts for most of the observed swing. Large δ indicates systematic
    candidate-quality or issue-salience effects.

    Note: τ is accepted as a parameter to keep the API consistent with potential
    future versions that use turnout-reweighted Dem share as the baseline.
    Currently the simple mean difference is used.

    Args:
        tract_votes: DataFrame with columns tract_geoid, race, dem_share, year.
        type_scores: DataFrame indexed by GEOID with type_j_score columns.
        tau: Turnout ratios per type (J,) — reserved for future turnout-reweighted version.
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

    # Per-tract residual: positive = more Dem in off-cycle, negative = more Rep.
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

    For presidential elections this is a no-op. For off-cycle elections the
    adjustment has two components:
      1. τ-reweighting: types with lower off-cycle turnout get down-weighted in
         the composition, shifting the effective electorate toward high-turnout types.
      2. δ-shift: residual preference difference is added to each tract's prior,
         weighted by the τ-adjusted type composition.

    Args:
        tract_priors: Dem share priors, shape (n_tracts,).
        type_scores: Type membership scores, shape (n_tracts, J).
        tau: Turnout ratios per type, shape (J,).
        delta: Choice shifts per type, shape (J,).
        is_offcycle: Whether the target election is off-cycle (governor/senate).

    Returns:
        Adjusted priors, shape (n_tracts,), clipped to [0, 1].
    """
    if not is_offcycle:
        return tract_priors.copy()

    # Reweight type scores by τ to reflect the actual off-cycle electorate composition.
    # Types with lower turnout (τ → 0) shrink in effective representation.
    offcycle_weights = type_scores * tau[np.newaxis, :]  # (n_tracts, J)
    row_sums = offcycle_weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    offcycle_norm = offcycle_weights / row_sums  # normalized, (n_tracts, J)

    # Each tract's adjustment = weighted sum of δ by its off-cycle type composition.
    adjustment = offcycle_norm @ delta  # (n_tracts,)

    return np.clip(tract_priors + adjustment, 0.0, 1.0)


def _print_tau_delta_summary(tau: np.ndarray, delta: np.ndarray, n_tracts: int) -> None:
    """Print τ/δ statistics and sanity checks to stdout."""
    print(f"\n{'='*60}")
    print("Voter Behavior Layer — Training Summary")
    print(f"{'='*60}")
    print(f"Tracts contributing to training: {n_tracts:,}")
    print(f"Types (J): {len(tau)}")

    print(f"\nτ (turnout ratio) distribution:")
    print(f"  mean={tau.mean():.3f}  std={tau.std():.3f}  "
          f"min={tau.min():.3f}  max={tau.max():.3f}")

    top5_tau = np.argsort(tau)[-5:][::-1]
    bot5_tau = np.argsort(tau)[:5]
    print(f"  Highest τ types: {[(int(j), round(float(tau[j]), 3)) for j in top5_tau]}")
    print(f"  Lowest  τ types: {[(int(j), round(float(tau[j]), 3)) for j in bot5_tau]}")

    # Sanity check: most types should show lower off-cycle turnout.
    pct_below_one = (tau < 1.0).mean() * 100
    print(f"  τ < 1.0: {pct_below_one:.0f}% of types "
          f"({'PASS' if pct_below_one > 60 else 'WARN — expected >60%'})")

    print(f"\nδ (residual choice shift) distribution:")
    print(f"  mean={delta.mean():.4f}  std={delta.std():.4f}  "
          f"min={delta.min():.4f}  max={delta.max():.4f}")

    top5_delta = np.argsort(delta)[-5:][::-1]
    bot5_delta = np.argsort(delta)[:5]
    print(f"  Highest δ types: {[(int(j), round(float(delta[j]), 4)) for j in top5_delta]}")
    print(f"  Lowest  δ types: {[(int(j), round(float(delta[j]), 4)) for j in bot5_delta]}")

    pct_small_delta = (np.abs(delta) < 0.1).mean() * 100
    print(f"  |δ| < 0.1: {pct_small_delta:.0f}% of types "
          f"({'PASS' if pct_small_delta > 70 else 'WARN — expected >70%'})")

    print(f"{'='*60}\n")


def train_and_save(
    tract_votes_path: str | Path = DEFAULT_TRACT_VOTES_PATH,
    assignments_path: str | Path = DEFAULT_ASSIGNMENTS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Load T.1/T.3 pipeline outputs, compute τ and δ, save to disk.

    Handles the column mapping from T.1 format (race_type, total_votes) to the
    internal format (race, votes_total) expected by compute_turnout_ratios and
    compute_choice_shifts. Non-relevant race_types (CONG, AG, etc.) are dropped.

    Args:
        tract_votes_path: Path to tract_elections.parquet (T.1 output).
                          Default: data/assembled/tract_elections.parquet
        assignments_path: Path to tract_type_assignments.parquet (T.3 output).
                          Default: data/communities/tract_type_assignments.parquet
        output_dir: Directory to write tau.npy, delta.npy, and summary.json.
                    Default: data/behavior

    Returns:
        Summary dict with τ and δ statistics plus n_tracts.
    """
    tract_elections = pd.read_parquet(tract_votes_path)
    assignments = _load_and_prepare_assignments(assignments_path)

    # Map from T.1 format to internal canonical format.
    # This is the key integration point between T.1/T.3 pipeline and the behavior layer.
    tract_votes = _map_race_type_to_race(tract_elections)

    # Determine J from the score columns (matches T.3 output: type_0_score..type_99_score).
    score_cols = [c for c in assignments.columns if c.endswith("_score")]
    n_types = len(score_cols)

    tau = compute_turnout_ratios(tract_votes, assignments, n_types)
    delta = compute_choice_shifts(tract_votes, assignments, tau, n_types)

    # Count tracts that contributed to training (in both elections and assignments).
    pres = tract_votes[tract_votes["race"].isin(PRESIDENTIAL_RACES)]
    off = tract_votes[tract_votes["race"].isin(OFFCYCLE_RACES)]
    pres_tracts = set(pres["tract_geoid"].unique())
    off_tracts = set(off["tract_geoid"].unique())
    n_tracts = len((pres_tracts & off_tracts) & set(assignments.index))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "tau.npy", tau)
    np.save(output_dir / "delta.npy", delta)

    summary = {
        "n_types": n_types,
        "n_tracts": n_tracts,
        "tau_mean": float(tau.mean()),
        "tau_std": float(tau.std()),
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "tau_pct_below_one": float((tau < 1.0).mean()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_pct_small": float((np.abs(delta) < 0.1).mean()),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_tau_delta_summary(tau, delta, n_tracts)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train voter behavior layer (τ + δ)")
    parser.add_argument(
        "--tract-votes",
        default=DEFAULT_TRACT_VOTES_PATH,
        help="Path to tract_elections.parquet (T.1 output)",
    )
    parser.add_argument(
        "--assignments",
        default=DEFAULT_ASSIGNMENTS_PATH,
        help="Path to tract_type_assignments.parquet (T.3 output)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for tau.npy, delta.npy, summary.json",
    )
    args = parser.parse_args()

    summary = train_and_save(args.tract_votes, args.assignments, args.output_dir)
    print(f"Summary: {summary}")
