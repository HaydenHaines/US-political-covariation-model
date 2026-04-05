"""Ridge LOO experiment with FULL national ACS demographics (3,144 counties).

Prior experiment used county_acs_features.parquet (293 rows, pilot only),
mostly imputed → got LOO r=0.573 with mostly filled data.

This script uses acs_counties_2022.parquet (3,144 rows) — nearly full national
coverage — plus county_rcms_features.parquet for religion.

Feature sets tested (all via Ridge LOO, hat matrix):
  (1) Baseline:  scores (J=100) + county_mean
  (2) +ACS:      scores + county_mean + ACS demographics
  (3) +ACS+RCMS: scores + county_mean + ACS + religion
  (4) Demo only: ACS + RCMS demographics only (no type scores)
  (5) Mean+Demo: county_mean + ACS + RCMS (no type scores)

Usage:
    uv run python scripts/experiment_ridge_demographics_full.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types


# ── Column helpers ─────────────────────────────────────────────────────────────


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


# ── Ridge LOO via hat matrix ───────────────────────────────────────────────────


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
    denom = np.where(np.abs(1.0 - h) < 1e-10, 1e-10, 1.0 - h)
    return y - e / denom


def ridge_loo_r(X: np.ndarray, holdout_raw: np.ndarray) -> dict:
    """Fit RidgeCV (GCV alpha) then compute exact LOO r for each holdout dim."""
    alphas = np.logspace(-3, 6, 100)
    H = holdout_raw.shape[1]
    per_r: list[float] = []
    per_rmse: list[float] = []
    best_alphas: list[float] = []

    for h in range(H):
        y = holdout_raw[:, h]
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        alpha = float(rcv.alpha_)
        best_alphas.append(alpha)
        y_loo = ridge_loo_predictions(X, y, alpha)
        r, _ = pearsonr(y, y_loo)
        per_r.append(float(np.clip(r, -1.0, 1.0)))
        per_rmse.append(float(np.sqrt(np.mean((y - y_loo) ** 2))))

    return {
        "mean_r": float(np.mean(per_r)),
        "per_dim_r": per_r,
        "mean_rmse": float(np.mean(per_rmse)),
        "per_dim_rmse": per_rmse,
        "best_alphas": best_alphas,
    }


# ── Data loading ───────────────────────────────────────────────────────────────


def load_shifts(min_year: int = 2008):
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)
    all_cols = [c for c in df.columns if c != "county_fips"]
    training_cols, holdout_cols = classify_columns(all_cols, min_year=min_year)
    return df, training_cols, holdout_cols


def build_matrices(df, training_cols, holdout_cols, presidential_weight: float = 8.0):
    """Return training_raw, training_scaled, holdout_raw, county_fips array."""
    used = training_cols + holdout_cols
    mat = df[used].values.astype(float)
    n_train = len(training_cols)

    training_raw = mat[:, :n_train]
    holdout_raw = mat[:, n_train:]
    county_fips = df["county_fips"].values

    pres_idx = [i for i, c in enumerate(training_cols) if "pres_" in c]

    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    if presidential_weight != 1.0:
        training_scaled[:, pres_idx] *= presidential_weight

    return training_raw, training_scaled, holdout_raw, county_fips


def build_acs_features(acs: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage features from raw ACS county population data."""
    feat = pd.DataFrame()
    feat["county_fips"] = acs["county_fips"]

    pop = acs["pop_total"].replace(0, np.nan)

    # Race/ethnicity percentages
    feat["pct_white_nh"] = acs["pop_white_nh"] / pop
    feat["pct_black"] = acs["pop_black"] / pop
    feat["pct_asian"] = acs["pop_asian"] / pop
    feat["pct_hispanic"] = acs["pop_hispanic"] / pop

    # Education (25+ population)
    educ_total = acs["educ_total"].replace(0, np.nan)
    educ_college = (
        acs["educ_bachelors"]
        + acs["educ_masters"]
        + acs["educ_professional"]
        + acs["educ_doctorate"]
    )
    feat["pct_college_plus"] = educ_college / educ_total

    # Housing
    housing = acs["housing_units"].replace(0, np.nan)
    feat["pct_owner_occupied"] = acs["housing_owner"] / housing

    # Commute
    commute = acs["commute_total"].replace(0, np.nan)
    feat["pct_car_commute"] = acs["commute_car"] / commute
    feat["pct_transit"] = acs["commute_transit"] / commute
    feat["pct_wfh"] = acs["commute_wfh"] / commute

    # Income and age (already per-capita)
    feat["median_hh_income"] = acs["median_hh_income"]
    feat["log_median_income"] = np.log1p(acs["median_hh_income"].clip(lower=1))
    feat["median_age"] = acs["median_age"]

    # Management occupation ratio (proxy for white-collar)
    occ_total = acs["occ_total"].replace(0, np.nan)
    feat["pct_management"] = (acs["occ_mgmt_male"] + acs["occ_mgmt_female"]) / occ_total

    return feat


def load_demographics_full(county_fips_arr: np.ndarray):
    """Load ACS 2022 (full national) + RCMS religion, aligned to county_fips_arr."""
    assembled = PROJECT_ROOT / "data" / "assembled"

    # ── ACS 2022 (full national, 3,144 counties) ──────────────────────────────
    acs_raw = pd.read_parquet(assembled / "acs_counties_2022.parquet")
    acs_feat = build_acs_features(acs_raw)

    # ── RCMS religion (3,141 counties) ────────────────────────────────────────
    rcms = pd.read_parquet(assembled / "county_rcms_features.parquet")
    rcms = rcms[
        [
            "county_fips",
            "evangelical_share",
            "mainline_share",
            "catholic_share",
            "black_protestant_share",
            "congregations_per_1000",
            "religious_adherence_rate",
        ]
    ]

    # ── Merge ──────────────────────────────────────────────────────────────────
    merged = acs_feat.merge(rcms, on="county_fips", how="left")

    # ── Align to county_fips_arr order (inner-join on shift matrix) ───────────
    fips_df = pd.DataFrame({"county_fips": county_fips_arr})
    aligned = fips_df.merge(merged, on="county_fips", how="inner")

    feat_cols = [c for c in aligned.columns if c != "county_fips"]
    n_acs = len([c for c in acs_feat.columns if c != "county_fips"])
    n_rcms = len([c for c in rcms.columns if c != "county_fips"])

    return aligned, feat_cols, n_acs, n_rcms


# ── Main experiment ────────────────────────────────────────────────────────────


def main():
    pw = 8.0
    j = 100
    temperature = 10.0

    print("=" * 70)
    print("RIDGE DEMOGRAPHICS FULL — ACS 2022 national coverage (3,144 counties)")
    print("=" * 70)
    print()

    # ── Load shift data ────────────────────────────────────────────────────────
    print("Loading shift data...")
    df, training_cols, holdout_cols = load_shifts(min_year=2008)
    print(
        f"  Shift matrix: {df.shape[0]} counties, {len(training_cols)} training dims"
    )
    print(f"  Holdout dims: {holdout_cols}")
    print()

    # ── Build matrices ─────────────────────────────────────────────────────────
    training_raw, training_scaled, holdout_raw, county_fips = build_matrices(
        df, training_cols, holdout_cols, presidential_weight=pw
    )

    # ── Discover types ─────────────────────────────────────────────────────────
    print(f"Discovering types (J={j}, T={temperature}, pw={pw})...")
    type_result = discover_types(
        training_scaled, j=j, temperature=temperature, random_state=42
    )
    scores = type_result.scores
    county_mean = training_raw.mean(axis=1)
    print(f"  scores shape: {scores.shape}")
    print()

    # ── Load demographics ──────────────────────────────────────────────────────
    print("Loading full ACS 2022 + RCMS demographics...")
    demo_aligned, feat_cols, n_acs, n_rcms = load_demographics_full(county_fips)

    # Counties that have demographic data (inner join)
    demo_fips = demo_aligned["county_fips"].values
    demo_mat = demo_aligned[feat_cols].values.astype(float)

    acs_cols = feat_cols[:n_acs]
    rcms_cols = feat_cols[n_acs:]

    n_all = len(county_fips)
    n_demo = len(demo_fips)
    n_miss_acs = np.isnan(demo_mat[:, :n_acs]).any(axis=1).sum()
    n_miss_rcms = np.isnan(demo_mat[:, n_acs:]).any(axis=1).sum()

    print(f"  Shift matrix counties: {n_all}")
    print(f"  After inner join w/ ACS: {n_demo} ({n_all - n_demo} dropped)")
    print(f"  ACS features: {n_acs}, RCMS features: {n_rcms}")
    print(f"  Rows with any ACS NaN:  {n_miss_acs}")
    print(f"  Rows with any RCMS NaN: {n_miss_rcms}")
    print()

    # Impute remaining NaNs with column median (should be very few)
    for col_i in range(demo_mat.shape[1]):
        col = demo_mat[:, col_i]
        med = np.nanmedian(col)
        demo_mat[np.isnan(col), col_i] = med

    # Check any remaining NaN
    still_nan = np.isnan(demo_mat).sum()
    if still_nan > 0:
        print(f"  WARNING: {still_nan} NaN values remain after median fill!")
    else:
        print("  NaN fill: complete (0 remaining)")
    print()

    # ── Restrict all matrices to the inner-join counties ──────────────────────
    # Build a boolean mask into county_fips array for the matched set
    fips_series = pd.Series(county_fips)
    demo_fips_set = set(demo_fips)
    mask = fips_series.isin(demo_fips_set).values

    # Re-align demo_mat to the masked order (they may differ in sort order)
    # Build index map: county_fips[mask] → demo_mat row
    fips_in = county_fips[mask]
    demo_idx_map = {f: i for i, f in enumerate(demo_fips)}
    reindex = [demo_idx_map[f] for f in fips_in]
    demo_mat_aligned = demo_mat[reindex]

    scores_in = scores[mask]
    county_mean_in = county_mean[mask]
    holdout_in = holdout_raw[mask]

    print(f"  Working set: {mask.sum()} counties (inner join)")
    print()

    # Standardize demographics
    acs_scaler = StandardScaler()
    demo_scaled = acs_scaler.fit_transform(demo_mat_aligned)

    acs_scaled = demo_scaled[:, :n_acs]
    rcms_scaled = demo_scaled[:, n_acs:]

    # ── Run experiments ────────────────────────────────────────────────────────
    print("-" * 70)
    print("Running Ridge LOO experiments...")
    print("-" * 70)
    print()

    results = {}

    # (1) Baseline: scores + county_mean
    print("(1) Baseline: scores + county_mean...")
    X1 = np.column_stack([scores_in, county_mean_in])
    res1 = ridge_loo_r(X1, holdout_in)
    results["baseline"] = res1
    print(
        f"    LOO r = {res1['mean_r']:.4f}  RMSE = {res1['mean_rmse']:.4f}"
        f"  dims = {[f'{r:.3f}' for r in res1['per_dim_r']]}"
        f"  alphas = {[f'{a:.1f}' for a in res1['best_alphas']]}"
    )
    print()

    # (2) scores + county_mean + ACS
    print(f"(2) +ACS ({n_acs} features): scores + county_mean + ACS...")
    X2 = np.column_stack([scores_in, county_mean_in, acs_scaled])
    res2 = ridge_loo_r(X2, holdout_in)
    results["acs"] = res2
    print(
        f"    LOO r = {res2['mean_r']:.4f}  RMSE = {res2['mean_rmse']:.4f}"
        f"  dims = {[f'{r:.3f}' for r in res2['per_dim_r']]}"
        f"  Δ = {res2['mean_r'] - res1['mean_r']:+.4f}"
    )
    print()

    # (3) scores + county_mean + ACS + RCMS
    print(f"(3) +ACS+RCMS ({n_acs + n_rcms} features): scores + county_mean + ACS + religion...")
    X3 = np.column_stack([scores_in, county_mean_in, demo_scaled])
    res3 = ridge_loo_r(X3, holdout_in)
    results["acs_rcms"] = res3
    print(
        f"    LOO r = {res3['mean_r']:.4f}  RMSE = {res3['mean_rmse']:.4f}"
        f"  dims = {[f'{r:.3f}' for r in res3['per_dim_r']]}"
        f"  Δ vs baseline = {res3['mean_r'] - res1['mean_r']:+.4f}"
    )
    print()

    # (4) Demo only (no type scores, no county mean)
    print(f"(4) Demo only: ACS + RCMS ({n_acs + n_rcms} features, no type scores)...")
    X4 = demo_scaled
    res4 = ridge_loo_r(X4, holdout_in)
    results["demo_only"] = res4
    print(
        f"    LOO r = {res4['mean_r']:.4f}  RMSE = {res4['mean_rmse']:.4f}"
        f"  dims = {[f'{r:.3f}' for r in res4['per_dim_r']]}"
    )
    print()

    # (5) county_mean + Demo (no type scores)
    print(f"(5) Mean+Demo: county_mean + ACS + RCMS (no type scores)...")
    X5 = np.column_stack([county_mean_in, demo_scaled])
    res5 = ridge_loo_r(X5, holdout_in)
    results["mean_demo"] = res5
    print(
        f"    LOO r = {res5['mean_r']:.4f}  RMSE = {res5['mean_rmse']:.4f}"
        f"  dims = {[f'{r:.3f}' for r in res5['per_dim_r']]}"
    )
    print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY (pw=8.0, J=100, T=10, inner-join n=" + str(mask.sum()) + ")")
    print("=" * 70)
    print(f"  (1) Baseline (scores+mean):            LOO r = {res1['mean_r']:.4f}")
    print(f"  (2) +ACS ({n_acs} feats):               LOO r = {res2['mean_r']:.4f}  Δ={res2['mean_r']-res1['mean_r']:+.4f}")
    print(f"  (3) +ACS+RCMS ({n_acs+n_rcms} feats):       LOO r = {res3['mean_r']:.4f}  Δ={res3['mean_r']-res1['mean_r']:+.4f}")
    print(f"  (4) Demo only (no types, no mean):     LOO r = {res4['mean_r']:.4f}")
    print(f"  (5) Mean+Demo (no types):              LOO r = {res5['mean_r']:.4f}")
    print()
    print(f"  Types alone add:  {res1['mean_r'] - res5['mean_r']:+.4f} vs mean+demo (no types)")
    print(f"  Demo alone adds:  {res2['mean_r'] - res1['mean_r']:+.4f} vs baseline (no demo)")

    best_name = max(results, key=lambda k: results[k]["mean_r"])
    best_r = results[best_name]["mean_r"]
    label_map = {
        "baseline": "(1) scores+mean",
        "acs": "(2) scores+mean+ACS",
        "acs_rcms": "(3) scores+mean+ACS+RCMS",
        "demo_only": "(4) demo only",
        "mean_demo": "(5) mean+demo",
    }
    print()
    print(f"  BEST: {label_map[best_name]}  LOO r = {best_r:.4f}")
    print(f"  vs published baseline 0.533: Δ = {best_r - 0.533:+.4f}")
    print()
    print("ACS feature list:")
    for c in acs_cols:
        print(f"  {c}")
    print()
    print("RCMS feature list:")
    for c in rcms_cols:
        print(f"  {c}")


if __name__ == "__main__":
    main()
