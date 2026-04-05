"""Regularization comparison experiment.

Tests whether Lasso or ElasticNet beats Ridge for LOO holdout prediction
from type scores (N×J) + county_training_mean at J=100.

Methods compared (all with LOO or 5-fold CV evaluation):
  (a) RidgeCV — baseline, hat-matrix LOO (exact)
  (b) LassoCV — 5-fold CV r (not true LOO)
  (c) ElasticNetCV — 5-fold CV r (not true LOO)
  (d) Ridge top-K types only (K=20, 40, 60) — features selected by variance

Usage:
    uv run python scripts/experiment_regularization_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types


# ── Column parsing helpers ────────────────────────────────────────────────────


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


# ── Ridge LOO via hat matrix (exact) ─────────────────────────────────────────


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Ridge LOO predictions via augmented hat-matrix shortcut.

    Intercept is not penalized (matching sklearn fit_intercept=True).
    Verified against brute-force LOO in experiment_ridge_prediction.py.
    """
    N, P = X.shape
    X_aug = np.column_stack([np.ones(N), X])  # (N, P+1)

    pen = alpha * np.eye(P + 1)
    pen[0, 0] = 0.0  # unpenalized intercept

    A = X_aug.T @ X_aug + pen
    A_inv = np.linalg.inv(A)

    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)  # hat diagonal

    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    e = y - y_hat

    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return y - e / denom


def ridge_loo_metric(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit RidgeCV (GCV), then compute exact LOO r and RMSE.

    Returns (r, rmse, best_alpha).
    """
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    best_alpha = float(rcv.alpha_)

    y_loo = ridge_loo_predictions(X, y, best_alpha)

    r = float(np.clip(pearsonr(y, y_loo)[0], -1.0, 1.0)) if np.std(y_loo) > 1e-10 else 0.0
    rmse = float(np.sqrt(np.mean((y - y_loo) ** 2)))
    return r, rmse, best_alpha


# ── 5-fold CV metric (for Lasso / ElasticNet) ────────────────────────────────


def cv5_metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute r and RMSE from stacked 5-fold out-of-fold predictions."""
    r = float(np.clip(pearsonr(y_true, y_pred)[0], -1.0, 1.0)) if np.std(y_pred) > 1e-10 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return r, rmse


def lasso_cv5(X: np.ndarray, y: np.ndarray, cv: int = 5) -> tuple[float, float, float]:
    """LassoCV fit + 5-fold OOF predictions. Returns (r, rmse, best_alpha)."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    y_oof = np.zeros_like(y)

    # First fit on full data to get best alpha
    lcv = LassoCV(cv=cv, max_iter=10000, random_state=42)
    lcv.fit(X, y)
    best_alpha = float(lcv.alpha_)

    # OOF predictions with fixed best_alpha
    from sklearn.linear_model import Lasso

    for train_idx, val_idx in kf.split(X):
        model = Lasso(alpha=best_alpha, max_iter=10000)
        model.fit(X[train_idx], y[train_idx])
        y_oof[val_idx] = model.predict(X[val_idx])

    r, rmse = cv5_metric(y, y_oof)
    return r, rmse, best_alpha


def elasticnet_cv5(
    X: np.ndarray,
    y: np.ndarray,
    l1_ratios: list[float] | None = None,
    cv: int = 5,
) -> tuple[float, float, float, float]:
    """ElasticNetCV fit + 5-fold OOF predictions. Returns (r, rmse, best_alpha, best_l1_ratio)."""
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    y_oof = np.zeros_like(y)

    encv = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, max_iter=10000, random_state=42)
    encv.fit(X, y)
    best_alpha = float(encv.alpha_)
    best_l1 = float(encv.l1_ratio_)

    from sklearn.linear_model import ElasticNet

    for train_idx, val_idx in kf.split(X):
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=10000)
        model.fit(X[train_idx], y[train_idx])
        y_oof[val_idx] = model.predict(X[val_idx])

    r, rmse = cv5_metric(y, y_oof)
    return r, rmse, best_alpha, best_l1


# ── Main experiment ───────────────────────────────────────────────────────────


def run_experiment(
    j: int = 100,
    min_year: int = 2008,
    presidential_weight: float = 8.0,
    temperature: float = 10.0,
    top_k_values: tuple[int, ...] = (20, 40, 60),
) -> None:
    print("=" * 72)
    print("Regularization Comparison Experiment")
    print(f"J={j}, presidential_weight={presidential_weight}, T={temperature}")
    print("=" * 72)

    # Load shift data
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    all_cols = [c for c in df.columns if c != "county_fips"]
    training_col_names, holdout_col_names = classify_columns(all_cols, min_year=min_year)

    print(f"\nData: {df.shape[0]} counties, {len(all_cols)} total dims")
    print(f"Training cols (start >= {min_year}): {len(training_col_names)}")
    print(f"Holdout cols: {holdout_col_names}")

    used_col_names = training_col_names + holdout_col_names
    full_matrix_raw = df[used_col_names].values.astype(float)

    training_indices = list(range(len(training_col_names)))
    holdout_indices = list(range(len(training_col_names), len(used_col_names)))

    training_raw = full_matrix_raw[:, training_indices]
    holdout_raw = full_matrix_raw[:, holdout_indices]  # (N, H)

    pres_indices = [i for i, c in enumerate(training_col_names) if "pres_" in c]

    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    if presidential_weight != 1.0:
        training_scaled[:, pres_indices] *= presidential_weight
        print(f"Presidential weight={presidential_weight} applied to {len(pres_indices)} columns")

    # Discover types
    print(f"\nRunning KMeans (J={j}, T={temperature})...", end=" ", flush=True)
    type_result = discover_types(training_scaled, j=j, temperature=temperature, random_state=42)
    scores = type_result.scores  # (N, J) soft membership, row-normalized
    print("done")

    county_training_means = training_raw.mean(axis=1)  # (N,)

    # Features: scores + county_training_mean
    X_base = np.column_stack([scores, county_training_means])  # (N, J+1)
    print(f"\nFeature matrix shape: {X_base.shape}")

    # Collect results per holdout dimension
    H = holdout_raw.shape[1]
    results: list[dict] = []

    for h_idx in range(H):
        y = holdout_raw[:, h_idx]
        col_name = holdout_col_names[h_idx]
        print(f"\n{'─' * 72}")
        print(f"Holdout dim: {col_name}")
        print(f"{'─' * 72}")

        row: dict = {"col": col_name}

        # ── (a) Ridge baseline (LOO, exact) ──────────────────────────────────
        print("  (a) RidgeCV (LOO, exact)...", end=" ", flush=True)
        r_a, rmse_a, alpha_a = ridge_loo_metric(X_base, y)
        row["ridge_r"] = r_a
        row["ridge_rmse"] = rmse_a
        row["ridge_alpha"] = alpha_a
        print(f"r={r_a:.4f}, RMSE={rmse_a:.4f}, alpha={alpha_a:.2f}")

        # ── (b) LassoCV (5-fold CV) ──────────────────────────────────────────
        print("  (b) LassoCV (5-fold CV)...", end=" ", flush=True)
        r_b, rmse_b, alpha_b = lasso_cv5(X_base, y, cv=5)
        row["lasso_r"] = r_b
        row["lasso_rmse"] = rmse_b
        row["lasso_alpha"] = alpha_b
        print(f"r={r_b:.4f}, RMSE={rmse_b:.4f}, alpha={alpha_b:.4f}")

        # ── (c) ElasticNetCV (5-fold CV) ─────────────────────────────────────
        print("  (c) ElasticNetCV (5-fold CV)...", end=" ", flush=True)
        r_c, rmse_c, alpha_c, l1_c = elasticnet_cv5(X_base, y)
        row["en_r"] = r_c
        row["en_rmse"] = rmse_c
        row["en_alpha"] = alpha_c
        row["en_l1"] = l1_c
        print(f"r={r_c:.4f}, RMSE={rmse_c:.4f}, alpha={alpha_c:.4f}, l1_ratio={l1_c:.2f}")

        # ── (d) Ridge top-K types only ────────────────────────────────────────
        # Select top-K types by variance of their score column
        type_vars = np.var(scores, axis=0)  # (J,)
        sorted_idx = np.argsort(type_vars)[::-1]  # descending

        for k in top_k_values:
            top_idx = sorted_idx[:k]
            X_topk = np.column_stack([scores[:, top_idx], county_training_means])
            print(f"  (d-K{k}) Ridge top-{k} types (LOO, exact)...", end=" ", flush=True)
            r_d, rmse_d, alpha_d = ridge_loo_metric(X_topk, y)
            row[f"ridge_top{k}_r"] = r_d
            row[f"ridge_top{k}_rmse"] = rmse_d
            row[f"ridge_top{k}_alpha"] = alpha_d
            print(f"r={r_d:.4f}, RMSE={rmse_d:.4f}, alpha={alpha_d:.2f}")

        results.append(row)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print(f"J={j}, features=type_scores+county_mean")
    print("NOTE: Ridge uses exact hat-matrix LOO; Lasso/EN use 5-fold CV r")
    print("      (5-fold CV r is NOT comparable to LOO r — usually slightly higher)")
    print("=" * 72)

    header = (
        f"{'Holdout dim':<28} {'Ridge(LOO)':>10} {'Lasso(CV5)':>10} {'EN(CV5)':>8}"
        + "".join(f" {'Top-'+str(k)+'(LOO)':>11}" for k in top_k_values)
    )
    print(header)
    print("─" * len(header))

    for row in results:
        line = (
            f"{row['col']:<28} "
            f"{row['ridge_r']:>10.4f} "
            f"{row['lasso_r']:>10.4f} "
            f"{row['en_r']:>8.4f}"
        )
        for k in top_k_values:
            line += f" {row[f'ridge_top{k}_r']:>11.4f}"
        print(line)

    # Mean across holdout dims
    print("─" * len(header))
    mean_ridge = np.mean([r["ridge_r"] for r in results])
    mean_lasso = np.mean([r["lasso_r"] for r in results])
    mean_en = np.mean([r["en_r"] for r in results])
    mean_topk = {k: np.mean([r[f"ridge_top{k}_r"] for r in results]) for k in top_k_values}

    mean_line = f"{'MEAN':<28} {mean_ridge:>10.4f} {mean_lasso:>10.4f} {mean_en:>8.4f}"
    for k in top_k_values:
        mean_line += f" {mean_topk[k]:>11.4f}"
    print(mean_line)

    print("\nRMSE table (lower is better):")
    print(header.replace("Holdout dim", "Holdout dim (RMSE)"))
    print("─" * len(header))
    for row in results:
        line = (
            f"{row['col']:<28} "
            f"{row['ridge_rmse']:>10.4f} "
            f"{row['lasso_rmse']:>10.4f} "
            f"{row['en_rmse']:>8.4f}"
        )
        for k in top_k_values:
            line += f" {row[f'ridge_top{k}_rmse']:>11.4f}"
        print(line)

    mean_ridge_rmse = np.mean([r["ridge_rmse"] for r in results])
    mean_lasso_rmse = np.mean([r["lasso_rmse"] for r in results])
    mean_en_rmse = np.mean([r["en_rmse"] for r in results])
    mean_topk_rmse = {k: np.mean([r[f"ridge_top{k}_rmse"] for r in results]) for k in top_k_values}

    print("─" * len(header))
    mean_rmse_line = (
        f"{'MEAN':<28} {mean_ridge_rmse:>10.4f} {mean_lasso_rmse:>10.4f} {mean_en_rmse:>8.4f}"
    )
    for k in top_k_values:
        mean_rmse_line += f" {mean_topk_rmse[k]:>11.4f}"
    print(mean_rmse_line)

    print("\nAlpha / L1-ratio selected:")
    for row in results:
        alphas_str = (
            f"  {row['col']}: Ridge alpha={row['ridge_alpha']:.2f}, "
            f"Lasso alpha={row['lasso_alpha']:.4f}, "
            f"EN alpha={row['en_alpha']:.4f} l1={row['en_l1']:.2f}"
        )
        for k in top_k_values:
            alphas_str += f", Ridge-top{k} alpha={row[f'ridge_top{k}_alpha']:.2f}"
        print(alphas_str)

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    best_method = max(
        [
            ("Ridge (full, LOO)", mean_ridge),
            ("Lasso (CV5)", mean_lasso),
            ("ElasticNet (CV5)", mean_en),
        ]
        + [(f"Ridge top-{k} (LOO)", mean_topk[k]) for k in top_k_values],
        key=lambda x: x[1],
    )
    print(f"Best mean r: {best_method[0]} → {best_method[1]:.4f}")
    print(f"Ridge baseline (LOO): {mean_ridge:.4f}")
    print(f"Lasso vs Ridge: {mean_lasso - mean_ridge:+.4f} (note: 5-fold CV, not LOO)")
    print(f"ElasticNet vs Ridge: {mean_en - mean_ridge:+.4f} (note: 5-fold CV, not LOO)")
    for k in top_k_values:
        print(f"Ridge top-{k} vs full Ridge (both LOO): {mean_topk[k] - mean_ridge:+.4f}")

    print("\nBaseline from CLAUDE.md:")
    print("  LOO type-mean: 0.448")
    print("  Ridge (scores+county_mean, J=100, LOO): 0.533")
    print(f"  This run Ridge (LOO): {mean_ridge:.4f}")


if __name__ == "__main__":
    run_experiment()
