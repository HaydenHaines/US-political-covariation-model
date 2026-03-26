"""Experiment 4: Ensemble of Ridge + GBM

Simple ensemble strategies to combine Ridge and HGB predictions via
averaging or stacking. Uses 20-fold CV for GBM (as LOO proxy).

Current best: Ridge+Demo LOO r = 0.649

Usage:
    uv run python scripts/experiments/exp4_ensemble.py
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
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types

warnings.filterwarnings("ignore")


# ── Helpers ─────────────────────────────────────────────────────────────────────


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


def ridge_loo_r_single(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    y_loo = ridge_loo_predictions(X, y, float(rcv.alpha_))
    r, _ = pearsonr(y, y_loo)
    return float(r), y_loo


def build_acs_base(acs: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame()
    feat["county_fips"] = acs["county_fips"]
    pop = acs["pop_total"].replace(0, np.nan)
    feat["pct_white_nh"] = acs["pop_white_nh"] / pop
    feat["pct_black"] = acs["pop_black"] / pop
    feat["pct_asian"] = acs["pop_asian"] / pop
    feat["pct_hispanic"] = acs["pop_hispanic"] / pop
    educ_total = acs["educ_total"].replace(0, np.nan)
    educ_college = (
        acs["educ_bachelors"] + acs["educ_masters"]
        + acs["educ_professional"] + acs["educ_doctorate"]
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


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    assembled = PROJECT_ROOT / "data" / "assembled"

    print("=" * 70)
    print("EXPERIMENT 4: Ridge + GBM Ensemble")
    print("Baseline LOO r = 0.649 (Ridge+Demo, S197)")
    print("=" * 70)
    print()

    # ── Load data (same pipeline as other experiments) ─────────────────────────
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
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

    print("Discovering types (J=100, T=10, pw=8.0)...")
    type_result = discover_types(training_scaled, j=100, temperature=10.0, random_state=42)
    scores = type_result.scores
    county_mean = training_raw.mean(axis=1)

    acs_raw = pd.read_parquet(assembled / "acs_counties_2022.parquet")
    acs_feat = build_acs_base(acs_raw)
    rcms = pd.read_parquet(assembled / "county_rcms_features.parquet")
    rcms = rcms[["county_fips", "evangelical_share", "mainline_share", "catholic_share",
                 "black_protestant_share", "congregations_per_1000", "religious_adherence_rate"]]
    merged = acs_feat.merge(rcms, on="county_fips", how="left")

    fips_df = pd.DataFrame({"county_fips": county_fips})
    aligned = fips_df.merge(merged, on="county_fips", how="inner")
    demo_fips = aligned["county_fips"].values
    feat_cols = [c for c in aligned.columns if c != "county_fips"]
    demo_mat = aligned[feat_cols].values.astype(float)

    fips_series = pd.Series(county_fips)
    mask = fips_series.isin(set(demo_fips)).values
    fips_in = county_fips[mask]
    demo_idx_map = {f: i for i, f in enumerate(demo_fips)}
    reindex = [demo_idx_map[f] for f in fips_in]
    demo_mat = demo_mat[reindex]

    for col_i in range(demo_mat.shape[1]):
        col = demo_mat[:, col_i]
        if np.isnan(col).any():
            demo_mat[np.isnan(col), col_i] = np.nanmedian(col)

    scores_in = scores[mask]
    county_mean_in = county_mean[mask]
    holdout_in = holdout_raw[mask]
    N_in = mask.sum()

    demo_scaler = StandardScaler()
    demo_scaled = demo_scaler.fit_transform(demo_mat)

    X = np.column_stack([scores_in, county_mean_in, demo_scaled])
    H = holdout_in.shape[1]

    print(f"Working set: {N_in} counties, feature matrix {X.shape}")
    print()

    # ── Per-holdout-dim ensemble ────────────────────────────────────────────────
    kf = KFold(n_splits=20, shuffle=True, random_state=42)
    hgb_model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42,
    )

    results = []

    for h in range(H):
        y = holdout_in[:, h]

        # Ridge (exact LOO)
        ridge_r, ridge_loo = ridge_loo_r_single(X, y)

        # HGB (20-fold CV)
        hgb_loo = cross_val_predict(hgb_model, X, y, cv=kf)
        hgb_r, _ = pearsonr(y, hgb_loo)

        # Ensemble: simple 50/50 average
        ensemble_50_50 = 0.5 * ridge_loo + 0.5 * hgb_loo
        ens_r_5050, _ = pearsonr(y, ensemble_50_50)

        # Ensemble: 70/30 Ridge-heavy (Ridge is exact LOO, HGB is CV proxy — slight advantage to Ridge)
        ensemble_7030 = 0.7 * ridge_loo + 0.3 * hgb_loo
        ens_r_7030, _ = pearsonr(y, ensemble_7030)

        # Ensemble: 30/70 HGB-heavy
        ensemble_3070 = 0.3 * ridge_loo + 0.7 * hgb_loo
        ens_r_3070, _ = pearsonr(y, ensemble_3070)

        # Optimal weight search (in-sample, biased — just for diagnostic)
        best_w, best_r_in = 0.5, 0.0
        for w in np.linspace(0, 1, 21):
            blended = w * ridge_loo + (1 - w) * hgb_loo
            r_in, _ = pearsonr(y, blended)
            if r_in > best_r_in:
                best_r_in, best_w = r_in, w

        print(f"  Holdout dim {h}:")
        print(f"    Ridge (exact LOO):      r = {ridge_r:.4f}")
        print(f"    HGB (20-fold CV proxy): r = {hgb_r:.4f}")
        print(f"    Ensemble 50/50:         r = {ens_r_5050:.4f}")
        print(f"    Ensemble 70/30 Ridge:   r = {ens_r_7030:.4f}")
        print(f"    Ensemble 30/70 HGB:     r = {ens_r_3070:.4f}")
        print(f"    Optimal blend (biased): w_ridge={best_w:.2f}, r_in = {best_r_in:.4f}")
        print()

        results.append({
            "dim": h,
            "ridge_r": ridge_r,
            "hgb_r": float(hgb_r),
            "ens_5050": float(ens_r_5050),
            "ens_7030": float(ens_r_7030),
            "ens_3070": float(ens_r_3070),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    mean_ridge = float(np.mean([r["ridge_r"] for r in results]))
    mean_hgb = float(np.mean([r["hgb_r"] for r in results]))
    mean_5050 = float(np.mean([r["ens_5050"] for r in results]))
    mean_7030 = float(np.mean([r["ens_7030"] for r in results]))
    mean_3070 = float(np.mean([r["ens_3070"] for r in results]))

    print("=" * 70)
    print("EXPERIMENT 4 SUMMARY")
    print("=" * 70)
    print(f"  Published best (S197):                  LOO r = 0.649")
    print(f"  Ridge (exact LOO):                      LOO r = {mean_ridge:.4f}  Δ={mean_ridge-0.649:+.4f}")
    print(f"  HGB (20-fold CV proxy):                  CV r = {mean_hgb:.4f}  (biased ~+0.02 vs LOO)")
    print(f"  Ensemble 50/50:                          ~r  = {mean_5050:.4f}  Δ={mean_5050-mean_ridge:+.4f}")
    print(f"  Ensemble 70% Ridge / 30% HGB:            ~r  = {mean_7030:.4f}  Δ={mean_7030-mean_ridge:+.4f}")
    print(f"  Ensemble 30% Ridge / 70% HGB:            ~r  = {mean_3070:.4f}  Δ={mean_3070-mean_ridge:+.4f}")
    print()

    best_ens = max(mean_5050, mean_7030, mean_3070)
    if best_ens > 0.649:
        print(f"  ENSEMBLE BEATS BASELINE: best ensemble ~r = {best_ens:.4f}")
        print("  NOTE: ensemble metric is a mix of exact LOO (Ridge) + CV proxy (HGB).")
        print("  True LOO would require running GBM in LOO mode — verify with full LOO.")
    else:
        print(f"  ENSEMBLE DOES NOT BEAT BASELINE: best ensemble ~r = {best_ens:.4f}")
        print("  If HGB 20-fold CV is ~0.02 above true LOO, the ensemble is likely <= Ridge LOO.")


if __name__ == "__main__":
    main()
