"""Experiment 1: Gradient Boosted Trees (XGBoost / LightGBM)

Tests whether non-linear GBM models beat Ridge LOO r=0.649 on the same
feature set: [type_scores (J=100) + county_mean + ACS + RCMS demographics].

Uses 20-fold CV as a proxy for full LOO (cheap). If promising, runs full
Ridge-style exact LOO comparison.

Current best: Ridge+Demo LOO r=0.649 (N=3,106, S197)

Usage:
    uv run python scripts/experiments/exp1_gradient_boosted_trees.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types

warnings.filterwarnings("ignore")


# ── Shared helpers (copied from experiment_ridge_demographics_full.py) ─────────


def parse_start_year(col: str) -> int | None:
    parts = col.split("_")
    try:
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def is_holdout_col(col: str) -> bool:
    return "20_24" in col


def classify_columns(
    all_cols: list[str], min_year: int = 2008
) -> tuple[list[str], list[str]]:
    holdout = [c for c in all_cols if is_holdout_col(c)]
    training = []
    for c in all_cols:
        if is_holdout_col(c):
            continue
        start = parse_start_year(c)
        if start is None or start >= min_year:
            training.append(c)
    return training, holdout


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Ridge LOO via augmented hat-matrix shortcut."""
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


def build_acs_features(acs: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame()
    feat["county_fips"] = acs["county_fips"]
    pop = acs["pop_total"].replace(0, np.nan)
    feat["pct_white_nh"] = acs["pop_white_nh"] / pop
    feat["pct_black"] = acs["pop_black"] / pop
    feat["pct_asian"] = acs["pop_asian"] / pop
    feat["pct_hispanic"] = acs["pop_hispanic"] / pop
    educ_total = acs["educ_total"].replace(0, np.nan)
    educ_college = (
        acs["educ_bachelors"]
        + acs["educ_masters"]
        + acs["educ_professional"]
        + acs["educ_doctorate"]
    )
    feat["pct_college_plus"] = educ_college / educ_total
    housing = acs["housing_units"].replace(0, np.nan)
    feat["pct_owner_occupied"] = acs["housing_owner"] / housing
    commute = acs["commute_total"].replace(0, np.nan)
    feat["pct_car_commute"] = acs["commute_car"] / commute
    feat["pct_transit"] = acs["commute_transit"] / commute
    feat["pct_wfh"] = acs["commute_wfh"] / commute
    feat["median_hh_income"] = acs["median_hh_income"]
    feat["log_median_income"] = np.log1p(acs["median_hh_income"].clip(lower=1))
    feat["median_age"] = acs["median_age"]
    occ_total = acs["occ_total"].replace(0, np.nan)
    feat["pct_management"] = (acs["occ_mgmt_male"] + acs["occ_mgmt_female"]) / occ_total
    return feat


# ── Data loading ────────────────────────────────────────────────────────────────


def load_all_data():
    assembled = PROJECT_ROOT / "data" / "assembled"
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"

    print("Loading shift data...")
    df = pd.read_parquet(shifts_path)
    all_cols = [c for c in df.columns if c != "county_fips"]
    training_cols, holdout_cols = classify_columns(all_cols, min_year=2008)

    mat = df[training_cols + holdout_cols].values.astype(float)
    n_train = len(training_cols)
    training_raw = mat[:, :n_train]
    holdout_raw = mat[:, n_train:]
    county_fips = df["county_fips"].values

    pres_idx = [i for i, c in enumerate(training_cols) if "pres_" in c]
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    training_scaled[:, pres_idx] *= 8.0

    print(f"  {df.shape[0]} counties, {len(training_cols)} training dims")

    print("Discovering types (J=100, T=10, pw=8.0)...")
    type_result = discover_types(training_scaled, j=100, temperature=10.0, random_state=42)
    scores = type_result.scores
    county_mean = training_raw.mean(axis=1)

    print("Loading ACS + RCMS demographics...")
    acs_raw = pd.read_parquet(assembled / "acs_counties_2022.parquet")
    acs_feat = build_acs_features(acs_raw)
    rcms = pd.read_parquet(assembled / "county_rcms_features.parquet")
    rcms = rcms[[
        "county_fips", "evangelical_share", "mainline_share", "catholic_share",
        "black_protestant_share", "congregations_per_1000", "religious_adherence_rate",
    ]]
    merged = acs_feat.merge(rcms, on="county_fips", how="left")

    # Align to shift matrix
    fips_df = pd.DataFrame({"county_fips": county_fips})
    aligned = fips_df.merge(merged, on="county_fips", how="inner")
    demo_fips = aligned["county_fips"].values
    feat_cols = [c for c in aligned.columns if c != "county_fips"]
    demo_mat = aligned[feat_cols].values.astype(float)

    # Re-align
    fips_series = pd.Series(county_fips)
    demo_fips_set = set(demo_fips)
    mask = fips_series.isin(demo_fips_set).values
    fips_in = county_fips[mask]
    demo_idx_map = {f: i for i, f in enumerate(demo_fips)}
    reindex = [demo_idx_map[f] for f in fips_in]
    demo_mat = demo_mat[reindex]

    # Impute NaNs
    for col_i in range(demo_mat.shape[1]):
        col = demo_mat[:, col_i]
        med = np.nanmedian(col)
        demo_mat[np.isnan(col), col_i] = med

    scores_in = scores[mask]
    county_mean_in = county_mean[mask]
    holdout_in = holdout_raw[mask]

    acs_scaler = StandardScaler()
    demo_scaled = acs_scaler.fit_transform(demo_mat)

    print(f"  Working set: {mask.sum()} counties (inner join)")

    # Build final feature matrix (same as Ridge+Demo best result)
    X = np.column_stack([scores_in, county_mean_in, demo_scaled])
    y = holdout_in[:, 0]  # pres_d_shift_20_24 is the key target

    return X, y, holdout_in, scores_in, county_mean_in, demo_scaled


# ── Cross-validation helpers ────────────────────────────────────────────────────


def cv_r(model, X: np.ndarray, y: np.ndarray, n_splits: int = 20, seed: int = 42) -> float:
    """K-fold CV Pearson r using cross_val_predict."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    r, _ = pearsonr(y, y_pred)
    return float(r)


def ridge_loo_r_single(X: np.ndarray, y: np.ndarray) -> float:
    """Exact Ridge LOO r for a single target."""
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    y_loo = ridge_loo_predictions(X, y, float(rcv.alpha_))
    r, _ = pearsonr(y, y_loo)
    return float(r)


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("EXPERIMENT 1: Gradient Boosted Trees vs Ridge")
    print("Current best: Ridge+Demo LOO r = 0.649 (N~3,106)")
    print("=" * 70)
    print()

    X, y, holdout_in, scores_in, county_mean_in, demo_scaled = load_all_data()
    N, P = X.shape
    H = holdout_in.shape[1]
    print(f"\nFeature matrix: {N} counties x {P} features")
    print(f"Holdout targets: {H}")
    print()

    # ── Baseline: Ridge (exact LOO) on full feature set ───────────────────────
    print("Running Ridge LOO (exact hat-matrix) on all holdout dims...")
    ridge_rs = []
    for h in range(H):
        r = ridge_loo_r_single(X, holdout_in[:, h])
        ridge_rs.append(r)
        print(f"  Holdout dim {h}: LOO r = {r:.4f}")
    ridge_mean = float(np.mean(ridge_rs))
    print(f"  Ridge mean LOO r = {ridge_mean:.4f}")
    print()

    # ── Experiment 1a: HistGradientBoosting (sklearn, handles NaN, fast) ──────
    # Simpler features only (no need to re-standardize for trees)
    print("Experiment 1a: HistGradientBoostingRegressor (sklearn)")
    print("  Using 20-fold CV as LOO proxy...")

    # Trees don't need scaled features — use raw: [scores (already [0,1]) + county_mean + demo_mat_raw]
    # But we'll use the same X for apples-to-apples comparison

    hgb_rs = []
    from sklearn.ensemble import HistGradientBoostingRegressor

    for h in range(H):
        y_h = holdout_in[:, h]
        # Tune: moderate depth, shrinkage
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
        r = cv_r(model, X, y_h, n_splits=20)
        hgb_rs.append(r)
        print(f"  Holdout dim {h}: 20-fold CV r = {r:.4f}")

    hgb_mean = float(np.mean(hgb_rs))
    print(f"  HGB mean 20-fold CV r = {hgb_mean:.4f}")
    print(f"  Δ vs Ridge LOO = {hgb_mean - ridge_mean:+.4f}  (NOTE: 20-fold ≈ LOO proxy)")
    print()

    # ── Experiment 1b: Scores only (J=100) + county_mean, no demographics ─────
    print("Experiment 1b: HGB on scores+county_mean (no demographics, for ablation)")
    X_small = np.column_stack([scores_in, county_mean_in])
    hgb_small_rs = []
    for h in range(H):
        y_h = holdout_in[:, h]
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42,
        )
        r = cv_r(model, X_small, y_h, n_splits=20)
        hgb_small_rs.append(r)
        print(f"  Holdout dim {h}: 20-fold CV r = {r:.4f}")
    hgb_small_mean = float(np.mean(hgb_small_rs))
    print(f"  HGB (no demo) mean 20-fold CV r = {hgb_small_mean:.4f}")
    print()

    # ── Experiment 1c: HGB on demographics only (no type scores) ──────────────
    print("Experiment 1c: HGB on demographics only (no type scores, for ablation)")
    X_demo_only = np.column_stack([county_mean_in, demo_scaled])
    hgb_demo_rs = []
    for h in range(H):
        y_h = holdout_in[:, h]
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42,
        )
        r = cv_r(model, X_demo_only, y_h, n_splits=20)
        hgb_demo_rs.append(r)
        print(f"  Holdout dim {h}: 20-fold CV r = {r:.4f}")
    hgb_demo_mean = float(np.mean(hgb_demo_rs))
    print(f"  HGB (demo only) mean 20-fold CV r = {hgb_demo_mean:.4f}")
    print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 70)
    print(f"  Published best (Ridge+Demo, exact LOO):        r = 0.649")
    print(f"  Ridge LOO (this run, exact):                   r = {ridge_mean:.4f}")
    print(f"  HGB full features (20-fold CV proxy):          r = {hgb_mean:.4f}  Δ={hgb_mean - ridge_mean:+.4f}")
    print(f"  HGB scores+mean only (20-fold CV proxy):       r = {hgb_small_mean:.4f}")
    print(f"  HGB demo only (20-fold CV proxy):              r = {hgb_demo_mean:.4f}")
    print()
    print("NOTE: 20-fold CV r is biased upward vs exact LOO (~+0.02 to +0.04).")
    print("A GBM 20-fold CV r of 0.67+ would be needed to likely beat 0.649 LOO.")
    print()

    best_r = max(hgb_mean, ridge_mean)
    if hgb_mean > ridge_mean + 0.01:
        print("RESULT: HGB shows meaningful improvement — run full LOO for confirmation.")
    elif hgb_mean > ridge_mean:
        print("RESULT: HGB marginally better in 20-fold CV. Likely within noise vs Ridge LOO.")
    else:
        print("RESULT: Ridge matches or beats HGB. Non-linearity not helpful for this target.")


if __name__ == "__main__":
    main()
