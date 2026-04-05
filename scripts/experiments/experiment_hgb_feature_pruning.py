"""Experiment: Does the Ridge feature exclusion list also help HGB?

Tree-based models (HGB) handle noisy features better than Ridge because
tree splits can ignore irrelevant features. The Ridge exclusion list
(74 features pruned → LOO r +0.014) may or may not help HGB.

Tests:
1. HGB with all 114 features (current production)
2. HGB with 40 pruned features (Ridge exclusion list applied)
3. Ensemble (50/50 Ridge+HGB) with pruned HGB

Uses 20-fold CV for HGB (fast proxy for LOO).

Usage:
    uv run python scripts/experiment_hgb_feature_pruning.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.train_ridge_model import (
    build_feature_matrix,
    compute_county_historical_mean,
    load_target,
)
from src.validation.holdout_accuracy_ridge import _load_feature_exclusions

warnings.filterwarnings("ignore")

# Load configs
with open(PROJECT_ROOT / "data" / "config" / "ensemble_params.json") as f:
    _CONFIG = json.load(f)
HGB_PARAMS = {k: v for k, v in _CONFIG["hgb"].items() if k != "random_state"}
HGB_PARAMS["random_state"] = 42

with open(PROJECT_ROOT / "config" / "ridge_feature_exclusions.yaml") as f:
    EXCLUDED = yaml.safe_load(f).get("excluded_features", [])


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Ridge LOO via hat matrix."""
    N, P = X.shape
    X_aug = np.column_stack([np.ones(N), X])
    pen = alpha * np.eye(P + 1)
    pen[0, 0] = 0.0
    A_inv = np.linalg.inv(X_aug.T @ X_aug + pen)
    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)
    e = y - X_aug @ (A_inv @ X_aug.T @ y)
    return y - e / np.where(np.abs(1 - h) < 1e-10, 1e-10, 1 - h)


def main() -> None:
    print("=" * 70)
    print("HGB Feature Pruning Experiment")
    print("=" * 70)
    print(f"Ridge exclusion list: {len(EXCLUDED)} features excluded")
    print(f"HGB params: {HGB_PARAMS}\n")

    assembled = PROJECT_ROOT / "data" / "assembled"

    # Load shifts for holdout
    shifts_df = pd.read_parquet(PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet")
    holdout_cols = [c for c in shifts_df.columns if "20_24" in c]
    shift_fips = shifts_df["county_fips"].astype(str).str.zfill(5).values

    # Load type assignments + features
    ta_df = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet")
    fips_ta = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    scores = ta_df[score_cols].values.astype(float)

    demo_df = pd.read_parquet(assembled / "county_features_national.parquet")
    demo_df["county_fips"] = demo_df["county_fips"].astype(str).str.zfill(5)

    county_mean = compute_county_historical_mean(fips_ta, assembled)
    y_full = load_target(fips_ta, assembled)

    # build_feature_matrix() ALREADY applies the exclusion list, producing
    # the pruned set. We use it for the pruned matrix, then manually build
    # the full matrix by including all demo columns.
    X_pruned_raw, feat_names_pruned, row_mask = build_feature_matrix(
        scores, np.array(fips_ta), demo_df, county_mean
    )

    # Build FULL feature matrix manually: type_scores + county_mean + ALL demo columns
    excluded_set = set(_load_feature_exclusions())
    all_demo_cols = [c for c in demo_df.columns if c != "county_fips"]
    pruned_demo_cols = [c for c in all_demo_cols if c not in excluded_set]

    # Merge to get all demo features for matched rows
    idx_df = pd.DataFrame({"county_fips": np.array(fips_ta), "_row_idx": np.arange(len(fips_ta))})
    merged = idx_df.merge(demo_df[["county_fips"] + all_demo_cols], on="county_fips", how="inner")
    # Same row alignment as build_feature_matrix
    full_row_mask = merged["_row_idx"].values
    full_demo = merged[all_demo_cols].values.astype(float)
    # Impute NaN with column means
    col_means = np.nanmean(full_demo, axis=0)
    for c in range(full_demo.shape[1]):
        nans = np.isnan(full_demo[:, c])
        if nans.any():
            full_demo[nans, c] = col_means[c]
    # Standardize
    from sklearn.preprocessing import StandardScaler as SS
    full_demo = SS().fit_transform(full_demo)
    # Combine: type_scores + county_mean + all_demo
    full_scores = scores[full_row_mask]
    full_cm = county_mean[full_row_mask].reshape(-1, 1)
    X_full_raw = np.hstack([full_scores, full_cm, full_demo])
    feat_names_full = list(score_cols) + ["county_mean"] + all_demo_cols

    y_matched = y_full[row_mask]
    valid = ~np.isnan(y_matched)
    X_pruned = X_pruned_raw[valid]
    X_full = X_full_raw[valid]
    matched_fips = np.array(fips_ta)[row_mask][valid]

    # Align holdout
    fips_to_idx = {f: i for i, f in enumerate(shift_fips)}
    holdout_raw = shifts_df[holdout_cols].values.astype(float)
    h_rows = [holdout_raw[fips_to_idx[f]] if f in fips_to_idx
              else np.full(len(holdout_cols), np.nan) for f in matched_fips]
    holdout = np.array(h_rows)
    hv = ~np.isnan(holdout).any(axis=1)
    X_full = X_full[hv]
    X_pruned = X_pruned[hv]
    holdout = holdout[hv]

    n_type_scores = len(score_cols)
    n_full_demo = len(all_demo_cols)
    n_pruned_demo = len(pruned_demo_cols)
    print(f"Full features: {X_full.shape[1]} ({n_type_scores} types + 1 county_mean + {n_full_demo} demo)")
    print(f"Pruned features: {X_pruned.shape[1]} ({n_type_scores} types + 1 county_mean + {n_pruned_demo} demo)")
    print(f"Excluded: {len(excluded_set)} demo features")
    print(f"Counties: {X_full.shape[0]}, Holdout dims: {holdout.shape[1]}\n")
    feat_names_all = feat_names_full

    H = holdout.shape[1]
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    # Ridge LOO (pruned features — matches production)
    print("Ridge LOO (pruned features, hat matrix)...")
    ridge_rs = []
    ridge_preds = []
    for h in range(H):
        y = holdout[:, h]
        rcv = RidgeCV(alphas=np.logspace(-3, 6, 100), fit_intercept=True, gcv_mode="auto")
        rcv.fit(X_pruned, y)
        y_loo = ridge_loo_predictions(X_pruned, y, float(rcv.alpha_))
        r, _ = pearsonr(y, y_loo)
        ridge_rs.append(r)
        ridge_preds.append(y_loo)
    print(f"  Ridge LOO r: {[f'{r:.4f}' for r in ridge_rs]} mean={np.mean(ridge_rs):.4f}")

    # HGB 20-fold CV: full features
    print("\nHGB 20-fold CV (full features)...")
    hgb_full_rs = []
    hgb_full_preds = []
    for h in range(H):
        y = holdout[:, h]
        model = HistGradientBoostingRegressor(**HGB_PARAMS)
        y_cv = cross_val_predict(model, X_full, y, cv=kf)
        r, _ = pearsonr(y, y_cv)
        hgb_full_rs.append(r)
        hgb_full_preds.append(y_cv)
    print(f"  HGB full r: {[f'{r:.4f}' for r in hgb_full_rs]} mean={np.mean(hgb_full_rs):.4f}")

    # HGB 20-fold CV: pruned features
    print("\nHGB 20-fold CV (pruned features)...")
    hgb_pruned_rs = []
    hgb_pruned_preds = []
    for h in range(H):
        y = holdout[:, h]
        model = HistGradientBoostingRegressor(**HGB_PARAMS)
        y_cv = cross_val_predict(model, X_pruned, y, cv=kf)
        r, _ = pearsonr(y, y_cv)
        hgb_pruned_rs.append(r)
        hgb_pruned_preds.append(y_cv)
    print(f"  HGB pruned r: {[f'{r:.4f}' for r in hgb_pruned_rs]} mean={np.mean(hgb_pruned_rs):.4f}")

    # Ensemble combinations
    print("\nEnsemble 50/50 comparisons...")
    combos = [
        ("Ridge(pruned) + HGB(full)", ridge_preds, hgb_full_preds),
        ("Ridge(pruned) + HGB(pruned)", ridge_preds, hgb_pruned_preds),
    ]
    for name, rpreds, hpreds in combos:
        ens_rs = []
        for h in range(H):
            y = holdout[:, h]
            blended = 0.5 * rpreds[h] + 0.5 * hpreds[h]
            r, _ = pearsonr(y, blended)
            ens_rs.append(r)
        print(f"  {name}: {[f'{r:.4f}' for r in ens_rs]} mean={np.mean(ens_rs):.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} | {'Mean r':>8}")
    print("-" * 50)
    print(f"{'Ridge LOO (pruned)':<35} | {np.mean(ridge_rs):>8.4f}")
    print(f"{'HGB CV (full 114 features)':<35} | {np.mean(hgb_full_rs):>8.4f}")
    print(f"{'HGB CV (pruned 40 features)':<35} | {np.mean(hgb_pruned_rs):>8.4f}")
    delta_hgb = np.mean(hgb_pruned_rs) - np.mean(hgb_full_rs)
    print(f"{'  Δ HGB pruned vs full':<35} | {delta_hgb:>+8.4f}")

    ens_full = np.mean([pearsonr(holdout[:, h], 0.5 * ridge_preds[h] + 0.5 * hgb_full_preds[h])[0] for h in range(H)])
    ens_pruned = np.mean([pearsonr(holdout[:, h], 0.5 * ridge_preds[h] + 0.5 * hgb_pruned_preds[h])[0] for h in range(H)])
    print(f"{'Ensemble Ridge+HGB(full)':<35} | {ens_full:>8.4f}")
    print(f"{'Ensemble Ridge+HGB(pruned)':<35} | {ens_pruned:>8.4f}")
    delta_ens = ens_pruned - ens_full
    print(f"{'  Δ ensemble pruned vs full':<35} | {delta_ens:>+8.4f}")

    if abs(delta_hgb) < 0.003:
        print("\nFINDING: HGB is robust to noisy features — pruning makes negligible difference.")
    elif delta_hgb > 0.003:
        print("\nFINDING: HGB benefits from pruning. Consider applying exclusion list to HGB too.")
    else:
        print("\nFINDING: Pruning HURTS HGB. Keep full features for HGB.")


if __name__ == "__main__":
    main()
