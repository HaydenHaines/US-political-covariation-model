"""Experiment: shrinkage lambda sweep for type covariance construction.

Tunes the lambda_shrinkage parameter in construct_type_covariance.py.
Current production value (lambda=0.75) was borrowed from the Economist model,
which used 51 states. Our J=43 types are a different granularity — the optimal
lambda may differ.

For each lambda in [0.10, 0.95] (step 0.05), this script:
  1. Constructs the J×J type covariance using that lambda.
  2. Validates against observed type comovement across elections (validation r).
  3. Computes the Frobenius norm distance from identity (measures informativeness).
  4. Computes the condition number (measures numerical stability).

A high validation_r means the demographic-derived correlation structure matches
how types actually covary across elections. A lambda too high (near raw Pearson)
may overfit to demographic noise. A lambda too low (near all-1s national swing)
loses discriminative power. The sweet spot maximises validation_r.

Results are printed to stdout and saved to data/experiments/lambda_sweep_results.csv.

Usage:
    cd /home/hayden/projects/US-political-covariation-model
    uv run python scripts/experiment_lambda_sweep.py
    uv run python scripts/experiment_lambda_sweep.py --lambda-min 0.1 --lambda-max 0.95
    uv run python scripts/experiment_lambda_sweep.py --lambda-min 0.5 --lambda-max 0.9 --step 0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments"

# ---------------------------------------------------------------------------
# Imports from production pipeline
# ---------------------------------------------------------------------------

from src.covariance.construct_type_covariance import (
    construct_type_covariance,
    validate_covariance,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_type_profiles() -> tuple[pd.DataFrame, list[str]]:
    """Load type demographic profiles from data/communities/type_profiles.parquet.

    Returns
    -------
    profiles : DataFrame
        J rows × demographic columns.
    feature_cols : list[str]
        Column names to use for covariance construction.
    """
    profiles_path = COMMUNITIES_DIR / "type_profiles.parquet"
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Type profiles not found at {profiles_path}.\n"
            "Run `python -m src.description.describe_types` first."
        )
    profiles = pd.read_parquet(profiles_path)
    feature_cols = [
        c for c in profiles.columns
        if profiles[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        and c not in ("type_id", "type_label", "super_type_id")
    ]
    return profiles, feature_cols


def load_type_scores() -> np.ndarray:
    """Load county × type soft-membership matrix.

    Returns
    -------
    type_scores : ndarray of shape (N, J)
    """
    for name in ["type_assignments.parquet", "county_type_assignments.parquet"]:
        path = COMMUNITIES_DIR / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(
            f"County type assignments not found in {COMMUNITIES_DIR}.\n"
            "Run `python -m src.discovery.run_type_discovery` first."
        )
    assignments = pd.read_parquet(path)
    score_cols = [c for c in assignments.columns if c.endswith("_score") and c.startswith("type_")]
    if not score_cols:
        score_cols = [c for c in assignments.columns if assignments[c].dtype in (np.float64, np.float32)]
    return assignments[score_cols].values


def load_shift_matrix() -> tuple[np.ndarray, list[list[int]]]:
    """Load county shift matrix and group columns by election pair.

    Returns
    -------
    shift_matrix : ndarray of shape (N, D)
    election_col_groups : list of lists of int
        Each inner list is the column indices for one election pair (3 cols each).
    """
    if not SHIFTS_PATH.exists():
        raise FileNotFoundError(
            f"Shift matrix not found at {SHIFTS_PATH}.\n"
            "Run `python src/assembly/build_county_shifts_multiyear.py` first."
        )
    shifts_df = pd.read_parquet(SHIFTS_PATH)
    shift_cols = [c for c in shifts_df.columns if c != "county_fips"]
    shift_matrix = shifts_df[shift_cols].values

    # Group by election pair: every 3 consecutive columns = one pair
    D = shift_matrix.shape[1]
    group_size = 3
    election_col_groups = [
        list(range(i, min(i + group_size, D)))
        for i in range(0, D, group_size)
    ]
    return shift_matrix, election_col_groups


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def frobenius_distance_from_identity(C: np.ndarray) -> float:
    """Frobenius norm distance from the all-1s matrix.

    The all-1s matrix represents lambda=0 (pure national swing).
    Because C_final = lambda * C_pearson + (1-lambda) * ones, higher lambda
    moves the matrix away from all-1s and closer to the raw demographic Pearson
    correlation.  A larger distance = more demographic signal retained.

    Parameters
    ----------
    C : ndarray of shape (J, J)
        Correlation matrix.

    Returns
    -------
    float
        ||C - 1||_F  where 1 is the J×J all-ones matrix.
        Increases with lambda (more demographic differentiation).
    """
    J = C.shape[0]
    ones = np.ones((J, J))
    return float(np.linalg.norm(C - ones, "fro"))


def condition_number(C: np.ndarray) -> float:
    """Spectral condition number of the matrix.

    A large condition number indicates near-singularity, which causes numerical
    instability in Bayesian updates.  Ideal range: < 1e6.

    Parameters
    ----------
    C : ndarray of shape (J, J)

    Returns
    -------
    float
        max(eigenvalue) / max(min(eigenvalue), 1e-12)
    """
    eigvals = np.linalg.eigvalsh(C)
    return float(eigvals.max() / max(float(eigvals.min()), 1e-12))


# ---------------------------------------------------------------------------
# Single lambda evaluation
# ---------------------------------------------------------------------------


def evaluate_lambda(
    lam: float,
    type_profiles: pd.DataFrame,
    feature_cols: list[str],
    type_scores: np.ndarray,
    shift_matrix: np.ndarray,
    election_col_groups: list[list[int]],
    sigma_base: float = 0.07,
    floor_negatives: bool = True,
) -> dict:
    """Evaluate one lambda value.

    Parameters
    ----------
    lam : float
        Shrinkage weight on the Pearson correlation vs all-1s matrix.
        0 = pure national swing; 1 = pure demographic Pearson.
    type_profiles : DataFrame
    feature_cols : list[str]
    type_scores : ndarray (N, J)
    shift_matrix : ndarray (N, D)
    election_col_groups : list[list[int]]
    sigma_base : float
    floor_negatives : bool

    Returns
    -------
    dict with keys: lambda, validation_r, frobenius_distance, condition_number
    """
    result = construct_type_covariance(
        type_profiles,
        feature_cols,
        lambda_shrinkage=lam,
        sigma_base=sigma_base,
        floor_negatives=floor_negatives,
    )
    val_r = validate_covariance(result, type_scores, shift_matrix, election_col_groups)
    frob = frobenius_distance_from_identity(result.correlation_matrix)
    cond = condition_number(result.correlation_matrix)

    return {
        "lambda": round(lam, 4),
        "validation_r": round(val_r, 6) if not np.isnan(val_r) else float("nan"),
        "frobenius_distance": round(frob, 6),
        "condition_number": round(cond, 2),
    }


# ---------------------------------------------------------------------------
# Lambda sweep
# ---------------------------------------------------------------------------


def run_lambda_sweep(
    lambda_min: float = 0.10,
    lambda_max: float = 0.95,
    step: float = 0.05,
    sigma_base: float = 0.07,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full lambda sweep.

    Parameters
    ----------
    lambda_min : float
    lambda_max : float
    step : float
    sigma_base : float
    verbose : bool

    Returns
    -------
    DataFrame with columns: lambda, validation_r, frobenius_distance, condition_number
    """
    if verbose:
        print("Loading type profiles...")
    type_profiles, feature_cols = load_type_profiles()
    J = len(type_profiles)
    if verbose:
        print(f"  {J} types × {len(feature_cols)} features")

    if verbose:
        print("Loading type scores and shift matrix...")
    type_scores = load_type_scores()
    shift_matrix, election_col_groups = load_shift_matrix()
    if verbose:
        print(f"  type_scores: {type_scores.shape}")
        print(f"  shift_matrix: {shift_matrix.shape}, {len(election_col_groups)} election groups")
        print()

    # Build lambda range: inclusive of max, rounded to avoid float drift
    n_steps = round((lambda_max - lambda_min) / step) + 1
    lambdas = [round(lambda_min + i * step, 10) for i in range(n_steps)]
    # Clamp to [0.0, 1.0]
    lambdas = [max(0.0, min(1.0, lam)) for lam in lambdas]

    rows = []
    for lam in lambdas:
        row = evaluate_lambda(
            lam,
            type_profiles,
            feature_cols,
            type_scores,
            shift_matrix,
            election_col_groups,
            sigma_base=sigma_base,
        )
        rows.append(row)
        if verbose:
            val_r_str = f"{row['validation_r']:.4f}" if not np.isnan(row["validation_r"]) else "  NaN"
            print(
                f"  lambda={lam:.2f}  validation_r={val_r_str}"
                f"  frobenius={row['frobenius_distance']:.4f}"
                f"  cond={row['condition_number']:.2f}"
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(results: pd.DataFrame, current_lambda: float = 0.75) -> None:
    """Print formatted summary table and recommendation."""
    print()
    print("=" * 75)
    print("LAMBDA SWEEP — Type Covariance Shrinkage")
    print("lambda=0 → all-1s (national swing); lambda=1 → pure demographic correlation")
    print("=" * 75)
    header = f"{'lambda':>8}  {'validation_r':>13}  {'frobenius_dist':>14}  {'cond_number':>12}"
    print(header)
    print("-" * 75)

    valid = results.dropna(subset=["validation_r"])
    if len(valid) == 0:
        print("  (no valid validation_r values)")
        return

    best_idx = valid["validation_r"].idxmax()
    best_lambda = float(valid.loc[best_idx, "lambda"])

    for _, row in results.iterrows():
        lam = float(row["lambda"])
        marker = ""
        if abs(lam - best_lambda) < 1e-9:
            marker = " <-- BEST"
        elif abs(lam - current_lambda) < 1e-9:
            marker = " <-- CURRENT"
        val_r_str = f"{row['validation_r']:>13.4f}" if not np.isnan(row["validation_r"]) else f"{'NaN':>13}"
        print(
            f"{lam:>8.2f}  {val_r_str}  {row['frobenius_distance']:>14.4f}  {row['condition_number']:>12.2f}{marker}"
        )

    print()
    best_row = valid.loc[best_idx]
    current_row = results[results["lambda"].round(4) == round(current_lambda, 4)]

    print(f"Best lambda: {best_lambda:.2f}  (validation_r={float(best_row['validation_r']):.4f})")
    if len(current_row) > 0:
        cur_r = float(current_row.iloc[0]["validation_r"])
        delta = float(best_row["validation_r"]) - cur_r
        print(f"Current lambda: {current_lambda:.2f}  (validation_r={cur_r:.4f})")
        print(f"Delta: {delta:+.4f}")
    print()


def save_results(results: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save results to data/experiments/lambda_sweep_results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "lambda_sweep_results.csv"
    results.to_csv(out_path, index=False)
    return out_path


def make_recommendation(results: pd.DataFrame, current_lambda: float = 0.75) -> str:
    """Return a recommendation string based on sweep results.

    Parameters
    ----------
    results : DataFrame
    current_lambda : float

    Returns
    -------
    str — human-readable recommendation
    """
    valid = results.dropna(subset=["validation_r"])
    if len(valid) == 0:
        return "Cannot make recommendation: no valid validation_r values."

    best_idx = valid["validation_r"].idxmax()
    best_lambda = float(valid.loc[best_idx, "lambda"])
    best_r = float(valid.loc[best_idx, "validation_r"])

    current_row = results[abs(results["lambda"] - current_lambda) < 1e-9]
    if len(current_row) > 0:
        cur_r = float(current_row.iloc[0]["validation_r"])
        delta = best_r - cur_r
        if abs(delta) < 0.005:
            return (
                f"Current lambda={current_lambda:.2f} (r={cur_r:.4f}) is near-optimal. "
                f"Best lambda={best_lambda:.2f} (r={best_r:.4f}), delta={delta:+.4f}. "
                "No change recommended."
            )
        else:
            return (
                f"Recommend changing lambda_shrinkage from {current_lambda:.2f} → {best_lambda:.2f}. "
                f"Validation r: {cur_r:.4f} → {best_r:.4f} ({delta:+.4f}). "
                f"Update config/model.yaml types.lambda_shrinkage = {best_lambda:.2f}."
            )
    return (
        f"Best lambda={best_lambda:.2f} (validation_r={best_r:.4f}). "
        f"Update config/model.yaml types.lambda_shrinkage = {best_lambda:.2f}."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lambda shrinkage sweep for type covariance construction."
    )
    parser.add_argument(
        "--lambda-min", type=float, default=0.10,
        help="Minimum lambda to test (default: 0.10)"
    )
    parser.add_argument(
        "--lambda-max", type=float, default=0.95,
        help="Maximum lambda to test (default: 0.95)"
    )
    parser.add_argument(
        "--step", type=float, default=0.05,
        help="Step size between lambda values (default: 0.05)"
    )
    parser.add_argument(
        "--current-lambda", type=float, default=0.75,
        help="Current production lambda for comparison (default: 0.75)"
    )
    args = parser.parse_args()

    print(
        f"Lambda sweep: lambda={args.lambda_min:.2f}..{args.lambda_max:.2f} "
        f"(step={args.step:.3f}), current={args.current_lambda:.2f}"
    )
    print()

    results = run_lambda_sweep(
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        step=args.step,
        verbose=True,
    )

    print_results(results, current_lambda=args.current_lambda)

    out_path = save_results(results)
    print(f"Results saved to {out_path}")
    print()

    print("RECOMMENDATION")
    print("-" * 40)
    rec = make_recommendation(results, current_lambda=args.current_lambda)
    print(f"  {rec}")
    print()


if __name__ == "__main__":
    main()
