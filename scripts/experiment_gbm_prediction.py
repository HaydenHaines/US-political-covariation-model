"""GBM vs Ridge prediction experiment.

Tests whether gradient boosting (GBM, RF) beats Ridge regression on type scores
(N×101 features) for 2020→2024 presidential holdout prediction.

Methods compared (all 5-fold CV for fair comparison):
  (a) Ridge (5-fold CV) — baseline for direct comparison
  (b) GradientBoostingRegressor (conservative: depth=3, n_est=100)
  (c) RandomForestRegressor (conservative: depth=5, n_est=200)
  (d) LGBMRegressor (if lightgbm available)
  [ref] Ridge LOO (hat matrix) — exact LOO, reported separately

Key question: Do non-linear interactions between type scores improve holdout
prediction enough to justify the overfitting risk at N=3,154?

Usage:
    uv run python scripts/experiment_gbm_prediction.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types

# ── Optionals ─────────────────────────────────────────────────────────────────

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed — skipping LGBM")


# ── Column parsing helpers (copied from experiment_ridge_prediction.py) ───────


def parse_start_year(col: str) -> int | None:
    parts = col.split("_")
    try:
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def is_holdout_col(col: str) -> bool:
    return "20_24" in col


def classify_columns(all_cols: list[str], min_year: int = 2008) -> tuple[list[str], list[str]]:
    holdout = [c for c in all_cols if is_holdout_col(c)]
    training = []
    for c in all_cols:
        if is_holdout_col(c):
            continue
        start = parse_start_year(c)
        if start is None or start >= min_year:
            training.append(c)
    return training, holdout


# ── Ridge LOO via hat matrix (for reference) ──────────────────────────────────


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Ridge LOO via augmented hat-matrix shortcut (unpenalized intercept)."""
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
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return y - e / denom


def ridge_loo_r(X: np.ndarray, y_cols: np.ndarray) -> dict:
    """Exact Ridge LOO r for each holdout column."""
    alphas = np.logspace(-3, 6, 100)
    per_dim_r, per_dim_rmse, best_alphas = [], [], []

    for h in range(y_cols.shape[1]):
        y = y_cols[:, h]
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        best_alpha = float(rcv.alpha_)
        best_alphas.append(best_alpha)

        y_loo = ridge_loo_predictions(X, y, best_alpha)
        r = float(np.clip(pearsonr(y, y_loo)[0], -1.0, 1.0)) if np.std(y_loo) > 1e-10 else 0.0
        per_dim_r.append(r)
        per_dim_rmse.append(float(np.sqrt(np.mean((y - y_loo) ** 2))))

    return {
        "mean_r": float(np.mean(per_dim_r)),
        "mean_rmse": float(np.mean(per_dim_rmse)),
        "per_dim_r": per_dim_r,
        "per_dim_rmse": per_dim_rmse,
        "best_alphas": best_alphas,
    }


# ── 5-fold CV evaluation ──────────────────────────────────────────────────────


def kfold_cv_r(
    model_factory,
    X: np.ndarray,
    y_cols: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """5-fold CV Pearson r and RMSE, averaged over folds and holdout dims.

    For each holdout dim:
      - Run K-fold; collect predictions on held-out folds
      - Compute r and RMSE between actual and OOF predictions
    Then average over dims.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    H = y_cols.shape[1]
    per_dim_r, per_dim_rmse = [], []

    for h in range(H):
        y = y_cols[:, h]
        oof_pred = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X):
            model = model_factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X[train_idx], y[train_idx])
            oof_pred[val_idx] = model.predict(X[val_idx])

        r = float(np.clip(pearsonr(y, oof_pred)[0], -1.0, 1.0)) if np.std(oof_pred) > 1e-10 else 0.0
        per_dim_r.append(r)
        per_dim_rmse.append(float(np.sqrt(np.mean((y - oof_pred) ** 2))))

    return {
        "mean_r": float(np.mean(per_dim_r)),
        "mean_rmse": float(np.mean(per_dim_rmse)),
        "per_dim_r": per_dim_r,
        "per_dim_rmse": per_dim_rmse,
    }


# ── Main experiment ───────────────────────────────────────────────────────────


def run_experiment(
    j: int = 100,
    min_year: int = 2008,
    presidential_weight: float = 8.0,
    temperature: float = 10.0,
    n_splits: int = 5,
) -> None:
    print("=" * 70)
    print("GBM vs Ridge Prediction Experiment")
    print("=" * 70)

    # Load shift data
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    all_cols = [c for c in df.columns if c != "county_fips"]
    training_col_names, holdout_col_names = classify_columns(all_cols, min_year=min_year)

    print(f"\nData: {df.shape[0]} counties, {len(all_cols)} total dims")
    print(f"Training cols (start >= {min_year}): {len(training_col_names)}")
    print(f"Holdout cols: {holdout_col_names}")

    # Build matrices
    used_col_names = training_col_names + holdout_col_names
    full_matrix_raw = df[used_col_names].values.astype(float)
    training_indices = list(range(len(training_col_names)))
    holdout_indices = list(range(len(training_col_names), len(used_col_names)))

    training_raw = full_matrix_raw[:, training_indices]
    holdout_raw = full_matrix_raw[:, holdout_indices]

    pres_indices_in_training = [
        i for i, c in enumerate(training_col_names) if "pres_" in c
    ]

    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    if presidential_weight != 1.0:
        training_scaled[:, pres_indices_in_training] *= presidential_weight
        print(
            f"Presidential weight={presidential_weight} applied to "
            f"{len(pres_indices_in_training)} columns (post-scaling)"
        )

    # Discover types
    print(f"\nRunning KMeans (J={j}, T={temperature})...", end=" ", flush=True)
    type_result = discover_types(training_scaled, j=j, temperature=temperature, random_state=42)
    scores = type_result.scores  # (N, J) soft membership
    print(f"done — scores shape: {scores.shape}")

    county_training_means = training_raw.mean(axis=1)  # (N,)
    X = np.column_stack([scores, county_training_means])  # (N, J+1 = 101)
    N, P = X.shape
    print(f"Feature matrix X: {N} counties × {P} features")
    print(f"Holdout targets: {len(holdout_col_names)} dims")

    print("\n" + "─" * 70)
    print(f"Running experiments (J={j}, {n_splits}-fold CV)...")
    print("─" * 70)

    results: dict[str, dict] = {}

    # ── [ref] Ridge LOO (hat matrix) ─────────────────────────────────────────
    print(f"\n[ref] Ridge LOO (hat matrix, exact)...", end=" ", flush=True)
    res_ref = ridge_loo_r(X, holdout_raw)
    results["ridge_loo"] = res_ref
    print(f"r={res_ref['mean_r']:.4f}, RMSE={res_ref['mean_rmse']:.4f}")
    for i, col in enumerate(holdout_col_names):
        print(f"       {col}: r={res_ref['per_dim_r'][i]:.4f}, RMSE={res_ref['per_dim_rmse'][i]:.4f}  (α={res_ref['best_alphas'][i]:.1f})")

    # ── (a) Ridge 5-fold CV ───────────────────────────────────────────────────
    print(f"\n(a) Ridge ({n_splits}-fold CV)...", end=" ", flush=True)

    def ridge_factory():
        # Use a moderate alpha; Ridge LOO already found best_alphas above
        # For 5-fold we use RidgeCV to auto-select within each fold
        return RidgeCV(alphas=np.logspace(-3, 6, 50), fit_intercept=True)

    res_a = kfold_cv_r(ridge_factory, X, holdout_raw, n_splits=n_splits)
    results["ridge_cv"] = res_a
    print(f"r={res_a['mean_r']:.4f}, RMSE={res_a['mean_rmse']:.4f}")
    for i, col in enumerate(holdout_col_names):
        print(f"       {col}: r={res_a['per_dim_r'][i]:.4f}, RMSE={res_a['per_dim_rmse'][i]:.4f}")

    # ── (b) GradientBoostingRegressor ────────────────────────────────────────
    print(f"\n(b) GBM (depth=3, n_est=100, lr=0.05, subsample=0.8, min_leaf=20)...", end=" ", flush=True)

    def gbm_factory():
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )

    res_b = kfold_cv_r(gbm_factory, X, holdout_raw, n_splits=n_splits)
    results["gbm"] = res_b
    print(f"r={res_b['mean_r']:.4f}, RMSE={res_b['mean_rmse']:.4f}")
    for i, col in enumerate(holdout_col_names):
        print(f"       {col}: r={res_b['per_dim_r'][i]:.4f}, RMSE={res_b['per_dim_rmse'][i]:.4f}")

    # ── (c) RandomForestRegressor ─────────────────────────────────────────────
    print(f"\n(c) RandomForest (depth=5, n_est=200, min_leaf=20)...", end=" ", flush=True)

    def rf_factory():
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        )

    res_c = kfold_cv_r(rf_factory, X, holdout_raw, n_splits=n_splits)
    results["rf"] = res_c
    print(f"r={res_c['mean_r']:.4f}, RMSE={res_c['mean_rmse']:.4f}")
    for i, col in enumerate(holdout_col_names):
        print(f"       {col}: r={res_c['per_dim_r'][i]:.4f}, RMSE={res_c['per_dim_rmse'][i]:.4f}")

    # ── (d) LGBMRegressor (optional) ─────────────────────────────────────────
    if HAS_LGBM:
        print(f"\n(d) LGBM (depth=4, n_est=200, lr=0.05, min_leaf=20)...", end=" ", flush=True)

        def lgbm_factory():
            return LGBMRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )

        res_d = kfold_cv_r(lgbm_factory, X, holdout_raw, n_splits=n_splits)
        results["lgbm"] = res_d
        print(f"r={res_d['mean_r']:.4f}, RMSE={res_d['mean_rmse']:.4f}")
        for i, col in enumerate(holdout_col_names):
            print(f"       {col}: r={res_d['per_dim_r'][i]:.4f}, RMSE={res_d['per_dim_rmse'][i]:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"J={j}, N={N}, P={P}, {n_splits}-fold CV")
    print(f"Features: type scores (J=100) + county_training_mean")
    print("")

    rows = [
        ("[ref] Ridge LOO (exact)", results["ridge_loo"]),
        ("(a)   Ridge 5-fold CV", results["ridge_cv"]),
        ("(b)   GBM (depth=3)", results["gbm"]),
        ("(c)   RandomForest (depth=5)", results["rf"]),
    ]
    if HAS_LGBM:
        rows.append(("(d)   LGBM (depth=4)", results["lgbm"]))

    col_w = 32
    print(f"{'Method':<{col_w}}  {'mean r':>8}  {'mean RMSE':>10}")
    print("-" * (col_w + 24))
    for name, res in rows:
        print(f"{name:<{col_w}}  {res['mean_r']:>8.4f}  {res['mean_rmse']:>10.4f}")
    print("-" * (col_w + 24))

    print("\nPer-dimension r breakdown:")
    header = f"{'Method':<{col_w}}" + "".join(f"  {c:>18}" for c in holdout_col_names)
    print(header)
    print("-" * (col_w + 20 * len(holdout_col_names)))
    for name, res in rows:
        row = f"{name:<{col_w}}"
        for r in res["per_dim_r"]:
            row += f"  {r:>18.4f}"
        print(row)

    # Winner
    print("\n" + "─" * 70)
    ridge_loo_mean_r = results["ridge_loo"]["mean_r"]
    best_name, best_res = max(rows[2:], key=lambda x: x[1]["mean_r"])  # non-baseline
    delta = best_res["mean_r"] - results["ridge_cv"]["mean_r"]
    delta_vs_loo = best_res["mean_r"] - ridge_loo_mean_r

    print(f"Ridge LOO r (reference):   {ridge_loo_mean_r:.4f}")
    print(f"Ridge 5-fold CV r:         {results['ridge_cv']['mean_r']:.4f}")
    print(f"Best non-linear method:    {best_name.strip()} → r={best_res['mean_r']:.4f}")
    print(f"vs Ridge 5-fold CV:        {delta:+.4f}")
    print(f"vs Ridge LOO (reference):  {delta_vs_loo:+.4f}")

    if delta > 0.005:
        print("\nVERDICT: GBM-family BEATS Ridge on 5-fold CV. Worth investigating further.")
    elif delta > 0:
        print("\nVERDICT: Marginal GBM gain (< 0.005). Probably noise. Stick with Ridge.")
    else:
        print("\nVERDICT: Ridge wins. Non-linear models don't help at N=3154 with J=100 type scores.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GBM vs Ridge prediction experiment")
    parser.add_argument("--j", type=int, default=100)
    parser.add_argument("--min-year", type=int, default=2008)
    parser.add_argument("--pres-weight", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    run_experiment(
        j=args.j,
        min_year=args.min_year,
        presidential_weight=args.pres_weight,
        temperature=args.temperature,
        n_splits=args.n_splits,
    )
