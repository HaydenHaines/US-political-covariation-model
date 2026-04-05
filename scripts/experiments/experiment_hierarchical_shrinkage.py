"""Experiment: Hierarchical shrinkage (partial pooling) on type adjustment.

Hypothesis: counties that are boundary members (low dominant type score) should
trust the type adjustment less and fall back toward their own momentum signal.

Strategies tested:
  - baseline:          current production (flat type adjustment, no confidence weighting)
  - alpha=score:       alpha = dominant_type_score (linear shrinkage toward county momentum)
  - alpha=score^2:     alpha = dominant_type_score^2 (amplify high-confidence counties)
  - alpha=fixed_X.X:   global fixed alpha sweep from 0.0 to 1.0

County own adjustment (leakage-free):
  The holdout is pres_20_24, so we cannot use 2024 data as input.
  We compute county "momentum" as the *trend* in recent training shifts:
    - Split training cols into first-half and second-half
    - county_momentum = (second_half_mean - first_half_mean)
  This captures whether the county has been accelerating or decelerating in
  the direction of the training mean — a forward-looking signal without using
  the holdout.

  Blended county own adjustment:
    county_own_adj = county_training_mean + county_momentum
  i.e., we predict the county will continue its recent trend rather than
  simply revert to its long-run mean.

  Final blend:
    prediction = county_training_mean
                 + alpha * type_adjustment
                 + (1 - alpha) * county_momentum

Usage:
    cd /home/hayden/projects/wethervane
    uv run python scripts/experiment_hierarchical_shrinkage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SHIFTS_PATH = PROJECT_ROOT / "data/shifts/county_shifts_multiyear.parquet"
ASSIGNMENTS_PATH = PROJECT_ROOT / "data/communities/type_assignments.parquet"
MIN_YEAR = 2008


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data() -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Return (shift_matrix, scores, training_cols, holdout_cols)."""
    shifts_df = pd.read_parquet(SHIFTS_PATH)
    all_cols = [c for c in shifts_df.columns if c != "county_fips"]

    holdout_keywords = ["20_24"]
    holdout_col_names = [c for c in all_cols if any(kw in c for kw in holdout_keywords)]
    training_col_names_unfiltered = [c for c in all_cols if c not in holdout_col_names]

    # Filter to min_year (matches production pipeline)
    training_col_names = []
    for c in training_col_names_unfiltered:
        parts = c.split("_")
        try:
            y2 = int(parts[-2])
            y1 = y2 + (1900 if y2 >= 50 else 2000) if len(parts[-2]) == 2 else y2
            if y1 >= MIN_YEAR:
                training_col_names.append(c)
        except (ValueError, IndexError):
            training_col_names.append(c)

    if not training_col_names:
        training_col_names = training_col_names_unfiltered

    used_cols = training_col_names + holdout_col_names
    shift_matrix = shifts_df[used_cols].values

    training_cols = [used_cols.index(c) for c in training_col_names]
    holdout_cols = [used_cols.index(c) for c in holdout_col_names]

    assignments_df = pd.read_parquet(ASSIGNMENTS_PATH)
    score_cols = [
        c for c in assignments_df.columns
        if c.startswith("type_") and c.endswith("_score")
    ]
    if not score_cols:
        score_cols = [
            c for c in assignments_df.columns
            if c not in ("county_fips", "dominant_type")
        ]
    scores = assignments_df[score_cols].values

    print(f"  Counties: {shift_matrix.shape[0]}")
    print(f"  Training dims: {len(training_cols)}")
    print(f"  Holdout dims: {len(holdout_cols)} — {holdout_col_names}")
    print(f"  Types (J): {scores.shape[1]}")
    return shift_matrix, scores, training_cols, holdout_cols


# ── Shared pre-computation ─────────────────────────────────────────────────────


def build_weights(scores: np.ndarray) -> np.ndarray:
    """Normalize absolute scores to weights summing to 1 per county."""
    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return abs_scores / row_sums  # N x J


def compute_dominant_confidence(scores: np.ndarray) -> np.ndarray:
    """Return each county's max type score (confidence in dominant type)."""
    weights = build_weights(scores)
    return weights.max(axis=1)  # N


def compute_county_momentum(
    shift_matrix: np.ndarray,
    training_cols: list[int],
) -> np.ndarray:
    """Leakage-free county momentum: second-half training mean - first-half mean.

    This captures whether a county has been trending more/less D in recent
    elections vs its long-run training average — available without touching
    the holdout.
    """
    training_data = shift_matrix[:, training_cols]
    n_train = len(training_cols)
    mid = n_train // 2
    # If very few training cols, fall back to zero momentum
    if mid == 0:
        return np.zeros(shift_matrix.shape[0])
    first_half = training_data[:, :mid].mean(axis=1)   # N
    second_half = training_data[:, mid:].mean(axis=1)  # N
    return second_half - first_half  # N (positive = accelerating D)


# ── Prediction functions ───────────────────────────────────────────────────────


def predict_baseline(
    weights: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
) -> dict:
    """Current production prediction (flat type adjustment)."""
    weight_sums = weights.sum(axis=0)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)  # N
    type_training_means = (weights.T @ county_training_means) / weight_sums  # J

    per_dim_r, per_dim_rmse = [], []
    for col in holdout_cols:
        actual = shift_matrix[:, col]
        type_holdout_means = (weights.T @ actual) / weight_sums
        type_adjustment = type_holdout_means - type_training_means
        county_adjustment = (weights * type_adjustment[None, :]).sum(axis=1)
        predicted = county_training_means + county_adjustment

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))
        per_dim_rmse.append(float(np.sqrt(np.mean((actual - predicted) ** 2))))

    return {
        "mean_r": float(np.mean(per_dim_r)),
        "mean_rmse": float(np.mean(per_dim_rmse)),
        "per_dim_r": per_dim_r,
        "per_dim_rmse": per_dim_rmse,
    }


def predict_partial_pooling(
    weights: np.ndarray,
    shift_matrix: np.ndarray,
    training_cols: list[int],
    holdout_cols: list[int],
    alpha: np.ndarray,  # N — per-county weight on type_adjustment
    county_momentum: np.ndarray,  # N — leakage-free own signal
) -> dict:
    """Partial-pooling prediction with per-county alpha.

    prediction = county_training_mean
                 + alpha_i * type_adjustment_i
                 + (1 - alpha_i) * county_momentum_i
    """
    weight_sums = weights.sum(axis=0)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    training_data = shift_matrix[:, training_cols]
    county_training_means = training_data.mean(axis=1)  # N
    type_training_means = (weights.T @ county_training_means) / weight_sums  # J

    per_dim_r, per_dim_rmse = [], []
    for col in holdout_cols:
        actual = shift_matrix[:, col]
        type_holdout_means = (weights.T @ actual) / weight_sums
        type_adjustment = type_holdout_means - type_training_means
        # Score-weighted type adjustment per county
        county_type_adj = (weights * type_adjustment[None, :]).sum(axis=1)  # N

        # Blended adjustment
        blended_adj = alpha * county_type_adj + (1.0 - alpha) * county_momentum
        predicted = county_training_means + blended_adj

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))
        per_dim_rmse.append(float(np.sqrt(np.mean((actual - predicted) ** 2))))

    return {
        "mean_r": float(np.mean(per_dim_r)),
        "mean_rmse": float(np.mean(per_dim_rmse)),
        "per_dim_r": per_dim_r,
        "per_dim_rmse": per_dim_rmse,
    }


# ── Main experiment ────────────────────────────────────────────────────────────


def run_experiment() -> None:
    print("=" * 70)
    print("EXPERIMENT: Hierarchical Shrinkage (Partial Pooling) on Type Adjustment")
    print("=" * 70)
    print()

    print("Loading data...")
    shift_matrix, scores, training_cols, holdout_cols = load_data()
    print()

    weights = build_weights(scores)
    dominant_confidence = compute_dominant_confidence(scores)
    county_momentum = compute_county_momentum(shift_matrix, training_cols)

    print(f"Dominant type confidence: "
          f"min={dominant_confidence.min():.3f}, "
          f"mean={dominant_confidence.mean():.3f}, "
          f"max={dominant_confidence.max():.3f}")
    print(f"County momentum (trend):  "
          f"min={county_momentum.min():.4f}, "
          f"mean={county_momentum.mean():.4f}, "
          f"max={county_momentum.max():.4f}")
    print()

    results: list[dict] = []

    # 1. Baseline (production)
    res = predict_baseline(weights, shift_matrix, training_cols, holdout_cols)
    results.append({"strategy": "baseline (α=1.0 flat)", "alpha_desc": "1.0",
                    **res})

    # 2. Momentum only (α=0.0)
    alpha_zero = np.zeros(scores.shape[0])
    res = predict_partial_pooling(weights, shift_matrix, training_cols,
                                   holdout_cols, alpha_zero, county_momentum)
    results.append({"strategy": "momentum only (α=0.0)", "alpha_desc": "0.0",
                    **res})

    # 3. Direct soft score (α = dominant_score)
    alpha_score = dominant_confidence.copy()
    res = predict_partial_pooling(weights, shift_matrix, training_cols,
                                   holdout_cols, alpha_score, county_momentum)
    results.append({"strategy": "α = dominant_score", "alpha_desc": "score",
                    **res})

    # 4. Squared score (amplify high-confidence)
    alpha_score2 = dominant_confidence ** 2
    res = predict_partial_pooling(weights, shift_matrix, training_cols,
                                   holdout_cols, alpha_score2, county_momentum)
    results.append({"strategy": "α = dominant_score²", "alpha_desc": "score²",
                    **res})

    # 5. Fixed alpha sweep 0.1 to 0.9 (0.0 and 1.0 already covered)
    for alpha_val in np.arange(0.1, 1.0, 0.1):
        alpha_fixed = np.full(scores.shape[0], alpha_val)
        res = predict_partial_pooling(weights, shift_matrix, training_cols,
                                       holdout_cols, alpha_fixed, county_momentum)
        results.append({
            "strategy": f"fixed α={alpha_val:.1f}",
            "alpha_desc": f"{alpha_val:.1f}",
            **res
        })

    # ── Print comparison table ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS: Holdout r and RMSE by Strategy")
    print("=" * 70)
    print()
    print(f"{'Strategy':<35} {'r':>8} {'RMSE':>8} {'Δr vs baseline':>15}")
    print("-" * 70)

    baseline_r = results[0]["mean_r"]
    baseline_rmse = results[0]["mean_rmse"]

    for row in results:
        r = row["mean_r"]
        rmse = row["mean_rmse"]
        delta = r - baseline_r
        delta_str = f"{delta:+.4f}" if row["strategy"] != "baseline (α=1.0 flat)" else "  (baseline)"
        print(f"{row['strategy']:<35} {r:>8.4f} {rmse:>8.4f} {delta_str:>15}")

    print("-" * 70)
    print()

    # Best strategy
    best = max(results, key=lambda x: x["mean_r"])
    print(f"Best strategy:  {best['strategy']}")
    print(f"  r    = {best['mean_r']:.4f}  (baseline: {baseline_r:.4f}, "
          f"Δ = {best['mean_r'] - baseline_r:+.4f})")
    print(f"  RMSE = {best['mean_rmse']:.4f}  (baseline: {baseline_rmse:.4f}, "
          f"Δ = {best['mean_rmse'] - baseline_rmse:+.4f})")
    print()

    # Per-dim breakdown for best strategy
    print("Per-dim holdout detail (best strategy):")
    for i, (r_val, rmse_val) in enumerate(
        zip(best["per_dim_r"], best["per_dim_rmse"])
    ):
        b_r = results[0]["per_dim_r"][i]
        b_rmse = results[0]["per_dim_rmse"][i]
        print(f"  dim {i}: r={r_val:.4f} (baseline {b_r:.4f}), "
              f"rmse={rmse_val:.4f} (baseline {b_rmse:.4f})")

    print()
    print("=" * 70)
    print("INTERPRETATION NOTES")
    print("=" * 70)
    print("""
  alpha = weight given to the TYPE adjustment (production signal)
  1 - alpha = weight given to COUNTY MOMENTUM (own recent trend)

  County momentum = (second half training mean) - (first half training mean)
  This captures whether a county has been accelerating or decelerating in
  the D direction within the training window — available without the holdout.

  If alpha=0.0 beats baseline: county-specific momentum is a better predictor
  than type-level comovement for the 2020→2024 shift.

  If intermediate alpha wins: partial pooling helps — type structure is real
  but boundary counties benefit from their own trend signal.

  If baseline (alpha=1.0) still wins: type structure is already optimal;
  individual county momentum adds noise.
""")


if __name__ == "__main__":
    run_experiment()
