"""Ensemble weight sweep: find optimal Ridge/HGB blend ratio.

Current production uses 50/50 Ridge/HGB (ensemble_params.json). This experiment
sweeps w in [0, 0.1, ..., 1.0] where ensemble = w*Ridge + (1-w)*HGB. Uses
20-fold CV for HGB as a fast proxy, with Ridge exact LOO (hat matrix).

If the best weight differs from 0.5 by > 0.01 LOO r, validates with full HGB LOO.

Usage:
    uv run python scripts/experiment_ensemble_weight_sweep.py
    uv run python scripts/experiment_ensemble_weight_sweep.py --full-loo  # force full LOO
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.train_ridge_model import (
    build_feature_matrix,
    compute_county_historical_mean,
    load_target,
)
from src.validation.holdout_accuracy_ridge import (
    _load_and_standardize_demographics,
)

warnings.filterwarnings("ignore")

# HGB params from production config
with open(PROJECT_ROOT / "data" / "config" / "ensemble_params.json") as f:
    _CONFIG = json.load(f)
HGB_PARAMS = {k: v for k, v in _CONFIG["hgb"].items() if k != "random_state"}
HGB_PARAMS["random_state"] = 42


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Ridge LOO via hat matrix (same as exp_hgb_sweep.py)."""
    N, P = X.shape
    X_aug = np.column_stack([np.ones(N), X])
    pen = alpha * np.eye(P + 1)
    pen[0, 0] = 0.0
    A = X_aug.T @ X_aug + pen
    A_inv = np.linalg.inv(A)
    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)
    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    e = y - y_hat
    denom = np.where(np.abs(1.0 - h) < 1e-10, 1e-10, 1.0 - h)
    return y - e / denom


def hgb_kfold_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 20) -> np.ndarray:
    """K-fold CV predictions for HGB (fast proxy for LOO)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    return cross_val_predict(model, X, y, cv=kf)


def hgb_full_loo(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Full LOO for HGB (expensive — N model fits)."""
    N = len(y)
    y_loo = np.empty(N)
    for i in range(N):
        if i % 500 == 0:
            print(f"      LOO {i}/{N}...", flush=True)
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X[mask], y[mask])
        y_loo[i] = m.predict(X[i:i + 1])[0]
    return y_loo


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix X and holdout targets, matching exp_hgb_sweep.py."""
    assembled = PROJECT_ROOT / "data" / "assembled"

    # Load shifts for holdout
    shifts_df = pd.read_parquet(PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet")
    holdout_cols = [c for c in shifts_df.columns if "20_24" in c]
    shift_fips = shifts_df["county_fips"].astype(str).str.zfill(5).values

    # Load production type assignments + features
    ta_df = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet")
    county_fips_ta = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    scores = ta_df[score_cols].values.astype(float)

    demo_df = pd.read_parquet(assembled / "county_features_national.parquet")
    demo_df["county_fips"] = demo_df["county_fips"].astype(str).str.zfill(5)

    county_mean = compute_county_historical_mean(county_fips_ta, assembled)
    y_full = load_target(county_fips_ta, assembled)

    X_all, feature_names, row_mask = build_feature_matrix(
        scores, np.array(county_fips_ta), demo_df, county_mean
    )
    y_matched = y_full[row_mask]
    valid = ~np.isnan(y_matched)
    X = X_all[valid]
    matched_fips = np.array(county_fips_ta)[row_mask][valid]

    # Align holdout
    fips_to_idx = {f: i for i, f in enumerate(shift_fips)}
    holdout_raw = shifts_df[holdout_cols].values.astype(float)
    holdout_rows = []
    for fips in matched_fips:
        if fips in fips_to_idx:
            holdout_rows.append(holdout_raw[fips_to_idx[fips]])
        else:
            holdout_rows.append(np.full(len(holdout_cols), np.nan))
    holdout = np.array(holdout_rows)

    # Drop NaN holdout rows
    holdout_valid = ~np.isnan(holdout).any(axis=1)
    X = X[holdout_valid]
    holdout = holdout[holdout_valid]

    print(f"Data: {X.shape[0]} counties, {X.shape[1]} features, {holdout.shape[1]} holdout dims")
    return X, holdout


def sweep_weights(
    X: np.ndarray,
    holdout: np.ndarray,
    use_full_loo: bool = False,
) -> list[dict]:
    """Sweep ensemble weights and return results."""
    H = holdout.shape[1]
    weights = np.arange(0, 1.05, 0.05)

    # Precompute Ridge LOO (exact, fast)
    print("\nRidge exact LOO (hat matrix)...")
    ridge_preds = []
    for h in range(H):
        y = holdout[:, h]
        alphas = np.logspace(-3, 6, 100)
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        y_loo = ridge_loo_predictions(X, y, float(rcv.alpha_))
        ridge_preds.append(y_loo)
        r, _ = pearsonr(y, y_loo)
        print(f"  dim {h}: Ridge LOO r={r:.4f} (alpha={rcv.alpha_:.2e})")

    # Compute HGB predictions
    if use_full_loo:
        print("\nHGB full LOO (N models per dim — this will take ~30-60 min)...")
        hgb_preds = []
        for h in range(H):
            y = holdout[:, h]
            t0 = time.time()
            y_loo = hgb_full_loo(X, y)
            elapsed = time.time() - t0
            r, _ = pearsonr(y, y_loo)
            hgb_preds.append(y_loo)
            print(f"  dim {h}: HGB LOO r={r:.4f} ({elapsed:.0f}s)")
    else:
        print("\nHGB 20-fold CV (fast proxy)...")
        hgb_preds = []
        for h in range(H):
            y = holdout[:, h]
            t0 = time.time()
            y_cv = hgb_kfold_cv(X, y, n_splits=20)
            elapsed = time.time() - t0
            r, _ = pearsonr(y, y_cv)
            hgb_preds.append(y_cv)
            print(f"  dim {h}: HGB 20-fold CV r={r:.4f} ({elapsed:.1f}s)")

    # Sweep weights
    print(f"\nSweeping {len(weights)} weight ratios (Ridge weight : HGB weight)...")
    results = []
    for w in weights:
        dim_rs = []
        for h in range(H):
            y = holdout[:, h]
            blended = w * ridge_preds[h] + (1 - w) * hgb_preds[h]
            r, _ = pearsonr(y, blended)
            dim_rs.append(float(r))
        mean_r = float(np.mean(dim_rs))
        results.append({
            "ridge_weight": float(w),
            "hgb_weight": float(1 - w),
            "mean_r": mean_r,
            "dim_rs": dim_rs,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble weight sweep")
    parser.add_argument("--full-loo", action="store_true", help="Use full HGB LOO (slow but exact)")
    args = parser.parse_args()

    print("=" * 70)
    print("Ensemble Weight Sweep: Ridge/HGB blend optimization")
    print("=" * 70)
    print(f"Current production: 50% Ridge + 50% HGB")
    print(f"Published baseline: ensemble LOO r ≈ 0.711 (S203)")
    print(f"HGB params: {HGB_PARAMS}")
    print()

    X, holdout = load_data()
    results = sweep_weights(X, holdout, use_full_loo=args.full_loo)

    # Print results table
    results_sorted = sorted(results, key=lambda r: r["mean_r"], reverse=True)
    best = results_sorted[0]
    baseline = next(r for r in results if abs(r["ridge_weight"] - 0.5) < 0.01)

    print("\n" + "=" * 70)
    print("RESULTS — sorted by mean r (descending)")
    print("=" * 70)

    method = "LOO" if args.full_loo else "20-fold CV"
    print(f"\n{'Ridge %':>8} | {'HGB %':>6} | {'Mean r':>8} | {'Δ vs 50/50':>10} | Per-dim r")
    print("-" * 70)
    for r in results_sorted:
        delta = r["mean_r"] - baseline["mean_r"]
        marker = " <<<" if r is best else ""
        dim_str = " ".join(f"{d:.4f}" for d in r["dim_rs"])
        print(f"{r['ridge_weight']:>7.0%} | {r['hgb_weight']:>5.0%} | "
              f"{r['mean_r']:>8.4f} | {delta:>+10.4f} | {dim_str}{marker}")

    print(f"\nMethod: {method}")
    print(f"Baseline (50/50): mean r = {baseline['mean_r']:.4f}")
    print(f"Best ({best['ridge_weight']:.0%}/{best['hgb_weight']:.0%}): "
          f"mean r = {best['mean_r']:.4f} (Δ={best['mean_r'] - baseline['mean_r']:+.4f})")

    if abs(best["ridge_weight"] - 0.5) > 0.05 and not args.full_loo:
        print(f"\nBest weight differs from 50/50. Re-run with --full-loo to validate.")
    elif abs(best["mean_r"] - baseline["mean_r"]) < 0.003:
        print(f"\nImprovement < 0.003 — not worth changing production config.")
    else:
        print(f"\nRECOMMENDATION: Update ensemble_params.json to "
              f"ridge_weight={best['ridge_weight']:.2f}, hgb_weight={best['hgb_weight']:.2f}")

    # Save results
    out_path = PROJECT_ROOT / "data" / "experiments" / "ensemble_weight_sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
