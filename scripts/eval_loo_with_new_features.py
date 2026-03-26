"""LOO evaluation script for Ridge + HGB ensemble with new QCEW+CHR features.

Uses holdout SHIFT (20→24 electoral shift) as the target, matching the
methodology of exp1b_hgb_full_loo.py (which produced the LOO r baselines
in CLAUDE.md). The feature matrix uses county_features_national.parquet
so it automatically includes QCEW+CHR features after the rebuild.

Baselines (from exp1b / S197-S201):
  Ridge-only LOO r  = 0.650  (Ridge+Demo, S197)
  Ensemble LOO r    = 0.690  (Ridge+HGB 50/50, S201)

Usage:
    uv run python scripts/eval_loo_with_new_features.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


HGB_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}

# Holdout: 2020→2024 shift columns (matching exp1b)
HOLDOUT_PATTERN = "20_24"
# Min start year for training columns (matching exp1b)
MIN_TRAIN_YEAR = 2008
# Presidential weighting applied before StandardScaler (matching production)
PRES_WEIGHT = 8.0
J = 100
TEMPERATURE = 10.0


def parse_start_year(col: str) -> int | None:
    parts = col.split("_")
    try:
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def classify_columns(all_cols: list[str]) -> tuple[list[str], list[str]]:
    holdout = [c for c in all_cols if HOLDOUT_PATTERN in c]
    training = []
    for c in all_cols:
        if HOLDOUT_PATTERN in c:
            continue
        start = parse_start_year(c)
        if start is None or start >= MIN_TRAIN_YEAR:
            training.append(c)
    return training, holdout


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact leave-one-out predictions for Ridge via hat-matrix shortcut."""
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


def hgb_cv_predictions(X: np.ndarray, y: np.ndarray, n_splits: int = 20) -> np.ndarray:
    """20-fold cross-validated predictions for HGB (fast approximation to LOO)."""
    y_oof = np.empty(len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X[train_idx], y[train_idx])
        y_oof[val_idx] = m.predict(X[val_idx])
    return y_oof


def main():
    from src.discovery.run_type_discovery import discover_types

    assembled = PROJECT_ROOT / "data" / "assembled"
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    national_path = assembled / "county_features_national.parquet"

    print("=" * 70)
    print("LOO EVAL: Ridge+HGB Ensemble with QCEW+CHR features")
    print("Target: holdout shift (20→24), matching exp1b baseline methodology")
    print("Baselines: Ridge=0.650, Ensemble=0.690 (S197/S201)")
    print("=" * 70)
    print()

    # ── Load and classify shift columns ─────────────────────────────────────
    df = pd.read_parquet(shifts_path)
    all_cols = [c for c in df.columns if c != "county_fips"]
    training_cols, holdout_cols = classify_columns(all_cols)
    print(f"Training dims: {len(training_cols)}, Holdout dims: {len(holdout_cols)}")
    print(f"Holdout cols: {holdout_cols}")

    mat = df[training_cols + holdout_cols].values.astype(float)
    n_train = len(training_cols)
    training_raw = mat[:, :n_train]
    holdout_raw = mat[:, n_train:]
    county_fips = df["county_fips"].values

    # ── Discover types (matching production pipeline) ────────────────────────
    pres_idx = [i for i, c in enumerate(training_cols) if "pres_" in c]
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    training_scaled[:, pres_idx] *= PRES_WEIGHT

    print(f"Discovering types (J={J}, T={TEMPERATURE}, pw={PRES_WEIGHT})...")
    type_result = discover_types(training_scaled, j=J, temperature=TEMPERATURE, random_state=42)
    scores = type_result.scores
    county_mean = training_raw.mean(axis=1)

    # ── Load county_features_national (includes new QCEW + CHR) ─────────────
    print(f"Loading county_features_national from {national_path} ...")
    demo_df = pd.read_parquet(national_path)
    demo_df["county_fips"] = demo_df["county_fips"].astype(str).str.zfill(5)
    demo_feature_cols = [c for c in demo_df.columns if c != "county_fips"]
    n_demo = len(demo_feature_cols)
    print(f"  {len(demo_df)} counties, {n_demo} demographic features")

    # Check for new feature groups
    qcew_present = sum(1 for c in demo_feature_cols if c in [
        "manufacturing_share", "government_share", "healthcare_share",
        "retail_share", "construction_share", "finance_share",
        "hospitality_share", "industry_diversity_hhi",
    ])
    chr_present = sum(1 for c in demo_feature_cols if c in [
        "adult_smoking_pct", "adult_obesity_pct", "life_expectancy",
        "diabetes_prevalence_pct", "premature_death_rate",
    ])
    print(f"  QCEW features: {qcew_present}/8  |  CHR features: {chr_present}/5 (sample)")
    print()

    # ── Align to counties with both shifts and demographics ──────────────────
    fips_df = pd.DataFrame({"county_fips": county_fips})
    aligned = fips_df.merge(demo_df, on="county_fips", how="inner")
    demo_fips = aligned["county_fips"].values
    demo_mat = aligned[demo_feature_cols].values.astype(float)

    fips_series = pd.Series(county_fips)
    keep_mask = fips_series.isin(set(demo_fips)).values
    fips_in = county_fips[keep_mask]
    demo_idx_map = {f: i for i, f in enumerate(demo_fips)}
    reindex = [demo_idx_map[f] for f in fips_in]
    demo_mat = demo_mat[reindex]

    # Impute any remaining NaN with column median
    for col_i in range(demo_mat.shape[1]):
        col = demo_mat[:, col_i]
        if np.isnan(col).any():
            demo_mat[np.isnan(col), col_i] = np.nanmedian(col)

    scores_in = scores[keep_mask]
    county_mean_in = county_mean[keep_mask]
    holdout_in = holdout_raw[keep_mask]
    N_in = keep_mask.sum()

    demo_scaler = StandardScaler()
    demo_scaled = demo_scaler.fit_transform(demo_mat)

    # X = [type_scores | county_mean | demo_standardized]
    X = np.column_stack([scores_in, county_mean_in, demo_scaled])
    H = holdout_in.shape[1]
    print(f"Working set: {N_in} counties, feature matrix {X.shape}")
    print()

    # ── Ridge exact LOO (per holdout dim, then average) ──────────────────────
    print("Ridge exact LOO...")
    ridge_rs = []
    ridge_loo_preds = []
    for h in range(H):
        y = holdout_in[:, h]
        alphas = np.logspace(-3, 6, 100)
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        y_loo_r = ridge_loo_predictions(X, y, float(rcv.alpha_))
        r, _ = pearsonr(y, y_loo_r)
        ridge_rs.append(float(r))
        ridge_loo_preds.append(y_loo_r)
        print(f"  Dim {h} ({holdout_cols[h]}): LOO r = {r:.4f}  alpha={rcv.alpha_:.4g}")
    ridge_mean = float(np.mean(ridge_rs))
    print(f"  Ridge mean LOO r = {ridge_mean:.4f}")
    print()

    # ── HGB 20-fold CV (per holdout dim) ─────────────────────────────────────
    print("HGB 20-fold CV (fast approximation)...")
    hgb_rs = []
    hgb_cv_preds = []
    for h in range(H):
        y = holdout_in[:, h]
        y_cv = hgb_cv_predictions(X, y)
        r, _ = pearsonr(y, y_cv)
        hgb_rs.append(float(r))
        hgb_cv_preds.append(y_cv)
        print(f"  Dim {h} ({holdout_cols[h]}): 20-fold CV r = {r:.4f}")
    hgb_mean = float(np.mean(hgb_rs))
    print(f"  HGB mean 20-fold CV r = {hgb_mean:.4f}")
    print()

    # ── Ensemble (Ridge LOO + HGB 20-fold) ───────────────────────────────────
    print("Ensemble (50/50 Ridge LOO + HGB 20-fold)...")
    ens_rs = []
    for h in range(H):
        y = holdout_in[:, h]
        blended = 0.5 * ridge_loo_preds[h] + 0.5 * hgb_cv_preds[h]
        r, _ = pearsonr(y, blended)
        ens_rs.append(float(r))
        print(f"  Dim {h}: Ensemble r = {r:.4f}")
    ens_mean = float(np.mean(ens_rs))
    print(f"  Ensemble mean r = {ens_mean:.4f}")
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline Ridge LOO r   = 0.650  (S197, Ridge+Demo, no QCEW/CHR)")
    print(f"  Baseline Ensemble LOO r = 0.690  (S201, Ridge+HGB 50/50)")
    print()
    print(f"  NEW Ridge LOO r        = {ridge_mean:.4f}  (Δ={ridge_mean - 0.650:+.4f})")
    print(f"  NEW HGB 20-fold CV r   = {hgb_mean:.4f}")
    print(f"  NEW Ensemble r         = {ens_mean:.4f}  (Δ={ens_mean - 0.690:+.4f})")
    print()

    best = max(ridge_mean, ens_mean)
    if ens_mean > 0.690:
        print(f"  BEATS ENSEMBLE BASELINE: {ens_mean:.4f} > 0.690  (Δ={ens_mean - 0.690:+.4f})")
    if ridge_mean > 0.650:
        print(f"  BEATS RIDGE BASELINE:    {ridge_mean:.4f} > 0.650  (Δ={ridge_mean - 0.650:+.4f})")
    if best <= 0.690:
        print(f"  No improvement over ensemble baseline. Best = {best:.4f}")

    # Return values for programmatic use
    return {
        "ridge_loo_r": ridge_mean,
        "hgb_cv_r": hgb_mean,
        "ensemble_r": ens_mean,
    }


if __name__ == "__main__":
    main()
