"""Experiment 1b: Full LOO for HistGradientBoosting

Confirms whether HGB's 20-fold CV advantage over Ridge translates
to true LOO improvement. Runs exact leave-one-out for HGB by
fitting N separate models. This is expensive (~3000 models per holdout dim)
but gives an honest apples-to-apples comparison with Ridge's exact LOO.

Estimated time: ~3-5 minutes per holdout dim (3 dims total = ~10-15 min)

Usage:
    uv run python scripts/experiments/exp1b_hgb_full_loo.py
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


def hgb_full_loo(X: np.ndarray, y: np.ndarray, verbose_every: int = 200) -> np.ndarray:
    """True leave-one-out for HGB. Fits N models, each with one county left out."""
    N = len(y)
    y_loo = np.empty(N)
    model_kwargs = dict(
        max_iter=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42,
    )

    for i in range(N):
        if verbose_every > 0 and i % verbose_every == 0:
            print(f"    LOO iteration {i}/{N}...", flush=True)

        mask = np.ones(N, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[i : i + 1]

        m = HistGradientBoostingRegressor(**model_kwargs)
        m.fit(X_tr, y_tr)
        y_loo[i] = m.predict(X_te)[0]

    return y_loo


def main():
    assembled = PROJECT_ROOT / "data" / "assembled"

    print("=" * 70)
    print("EXPERIMENT 1b: Full LOO for HGB (honest comparison with Ridge LOO)")
    print("Baseline LOO r = 0.649 (Ridge+Demo, S197)")
    print("WARNING: This takes ~10-15 minutes (N=3106 × 3 targets × 1 HGB each)")
    print("=" * 70)
    print()

    # ── Load data ──────────────────────────────────────────────────────────────
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

    # ── Ridge exact LOO (for comparison) ──────────────────────────────────────
    print("Ridge exact LOO (reference)...")
    ridge_rs = []
    for h in range(H):
        y = holdout_in[:, h]
        alphas = np.logspace(-3, 6, 100)
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        y_loo_ridge = ridge_loo_predictions(X, y, float(rcv.alpha_))
        r, _ = pearsonr(y, y_loo_ridge)
        ridge_rs.append(float(r))
        print(f"  Dim {h}: LOO r = {r:.4f}")
    ridge_mean = float(np.mean(ridge_rs))
    print(f"  Ridge mean LOO r = {ridge_mean:.4f}")
    print()

    # ── HGB full LOO (honest) ─────────────────────────────────────────────────
    hgb_rs = []
    hgb_loo_preds = []

    for h in range(H):
        y = holdout_in[:, h]
        print(f"HGB full LOO for holdout dim {h} (this may take a few minutes)...")
        y_loo_hgb = hgb_full_loo(X, y, verbose_every=500)
        r, _ = pearsonr(y, y_loo_hgb)
        hgb_rs.append(float(r))
        hgb_loo_preds.append(y_loo_hgb)
        print(f"  Dim {h}: HGB full LOO r = {r:.4f}")
        print()

    hgb_mean = float(np.mean(hgb_rs))
    print(f"  HGB mean full LOO r = {hgb_mean:.4f}")
    print()

    # ── Ensemble (Ridge LOO + HGB LOO) ─────────────────────────────────────────
    print("Ensemble (Ridge LOO + HGB full LOO)...")
    ridge_loo_preds = []
    for h in range(H):
        y = holdout_in[:, h]
        alphas = np.logspace(-3, 6, 100)
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        y_loo_r = ridge_loo_predictions(X, y, float(rcv.alpha_))
        ridge_loo_preds.append(y_loo_r)

    ens_rs = []
    for h in range(H):
        y = holdout_in[:, h]
        blended = 0.5 * ridge_loo_preds[h] + 0.5 * hgb_loo_preds[h]
        r, _ = pearsonr(y, blended)
        ens_rs.append(float(r))
        print(f"  Dim {h}: Ensemble (50/50) LOO r = {r:.4f}")
    ens_mean = float(np.mean(ens_rs))

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("EXPERIMENT 1b SUMMARY (honest LOO comparison)")
    print("=" * 70)
    print(f"  Published best (Ridge+Demo, S197):   LOO r = 0.649")
    print(f"  Ridge exact LOO (this run):          LOO r = {ridge_mean:.4f}  per-dim = {[f'{r:.3f}' for r in ridge_rs]}")
    print(f"  HGB full LOO (honest):               LOO r = {hgb_mean:.4f}  per-dim = {[f'{r:.3f}' for r in hgb_rs]}")
    print(f"  Ensemble 50/50 (honest LOO):         LOO r = {ens_mean:.4f}  per-dim = {[f'{r:.3f}' for r in ens_rs]}")
    print()

    best = max(ridge_mean, hgb_mean, ens_mean)
    if best > 0.649:
        winner = "HGB" if hgb_mean == best else ("Ensemble" if ens_mean == best else "Ridge")
        print(f"  BEATS BASELINE: {winner} LOO r = {best:.4f}  (Δ={best-0.649:+.4f})")
    else:
        print(f"  BELOW BASELINE: best LOO r = {best:.4f}")
        print("  Interpretation: The non-linear gain in 20-fold CV was inflated.")
        print("  Ridge remains the best production model.")


if __name__ == "__main__":
    main()
