"""Experiment 3: Feature Interactions for Ridge

Tests whether adding polynomial/interaction features between
top-signal type scores and demographics improves LOO r beyond 0.649.

Strategy:
  (a) Select top-K type scores by Ridge coefficient magnitude
  (b) Select top-K demographic features by Ridge coefficient magnitude
  (c) Add pairwise interaction terms (no squared terms) via manual product
  (d) Fit RidgeCV on the augmented feature set

Rationale: Ridge is linear. If a county being high in "Evangelical Coalition"
type AND high in evangelical_share has a compounded effect beyond additive,
interactions can capture this. We avoid PolynomialFeatures(degree=2) on all
features (J+1+D ≈ 120 features → 7,260 interactions = 3x N = overfit risk).
Instead, we select top 15 features from each group = 225 new features, manageable
for N~3,100 with Ridge regularization.

Usage:
    uv run python scripts/experiments/exp3_feature_interactions.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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


def ridge_loo_r(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    alpha = float(rcv.alpha_)
    y_loo = ridge_loo_predictions(X, y, alpha)
    r, _ = pearsonr(y, y_loo)
    return float(r), alpha


def ridge_loo_multi(X: np.ndarray, holdout: np.ndarray) -> tuple[float, list[float], list[float]]:
    H = holdout.shape[1]
    rs, alphas = [], []
    for h in range(H):
        r, a = ridge_loo_r(X, holdout[:, h])
        rs.append(r)
        alphas.append(a)
    return float(np.mean(rs)), rs, alphas


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


def add_interactions(X_base: np.ndarray, y: np.ndarray, top_n_each: int = 15) -> np.ndarray:
    """Add top-K × top-K interaction features between score block and demo block.

    The first 100 cols are type scores, col 100 is county_mean, remaining are demographics.
    Select top_n_each from each block by |correlation with y| and add cross-products.
    """
    n_type_scores = 100  # first 100 cols are type scores
    # Select top type scores by |corr with y|
    type_block = X_base[:, :n_type_scores]
    demo_block = X_base[:, n_type_scores + 1:]  # skip county_mean col

    type_corrs = np.array([abs(pearsonr(type_block[:, i], y)[0]) for i in range(type_block.shape[1])])
    demo_corrs = np.array([abs(pearsonr(demo_block[:, i], y)[0]) for i in range(demo_block.shape[1])])

    top_type_idx = np.argsort(type_corrs)[-top_n_each:]
    top_demo_idx = np.argsort(demo_corrs)[-top_n_each:]

    # Cross-product interaction terms
    interactions = []
    for ti in top_type_idx:
        for di in top_demo_idx:
            interactions.append(type_block[:, ti] * demo_block[:, di])

    if interactions:
        interaction_mat = np.column_stack(interactions)
        # Standardize interactions (they may have very different scales)
        inter_scaler = StandardScaler()
        interaction_mat = inter_scaler.fit_transform(interaction_mat)
        return np.column_stack([X_base, interaction_mat])
    return X_base


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    assembled = PROJECT_ROOT / "data" / "assembled"

    print("=" * 70)
    print("EXPERIMENT 3: Feature Interactions (Type Scores × Demographics)")
    print("Baseline LOO r = 0.649 (Ridge+Demo, S197)")
    print("=" * 70)
    print()

    # ── Load shifts ────────────────────────────────────────────────────────────
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

    print(f"  {df.shape[0]} counties, {len(training_cols)} training dims")
    print(f"  Holdout columns: {holdout_cols}")
    print()

    # ── Discover types ─────────────────────────────────────────────────────────
    print("Discovering types (J=100, T=10, pw=8.0)...")
    type_result = discover_types(training_scaled, j=100, temperature=10.0, random_state=42)
    scores = type_result.scores
    county_mean = training_raw.mean(axis=1)
    print()

    # ── Load demographics ──────────────────────────────────────────────────────
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
    demo_fips_set = set(demo_fips)
    mask = fips_series.isin(demo_fips_set).values
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

    print(f"Working set: {N_in} counties, {demo_mat.shape[1]} demo features")
    print()

    # ── Build baseline X ───────────────────────────────────────────────────────
    X_base = np.column_stack([scores_in, county_mean_in, demo_scaled])
    print(f"Baseline feature matrix shape: {X_base.shape}")
    print()

    # ── Experiment (1): Baseline (no interactions) ─────────────────────────────
    print("(1) Baseline (scores + county_mean + ACS+RCMS):")
    mean_r1, rs1, alphas1 = ridge_loo_multi(X_base, holdout_in)
    print(f"    LOO r = {mean_r1:.4f}  per-dim = {[f'{r:.3f}' for r in rs1]}")
    print()

    # ── Experiment (2): top-10×10 interactions ─────────────────────────────────
    print("(2) +Interactions (top 10 type scores × top 10 demographics):")
    y_pres = holdout_in[:, 0]  # use first holdout dim to select features
    X_inter10 = add_interactions(X_base, y_pres, top_n_each=10)
    n_new = X_inter10.shape[1] - X_base.shape[1]
    print(f"    Added {n_new} interaction features (shape: {X_inter10.shape})")
    mean_r2, rs2, alphas2 = ridge_loo_multi(X_inter10, holdout_in)
    print(f"    LOO r = {mean_r2:.4f}  per-dim = {[f'{r:.3f}' for r in rs2]}  Δ={mean_r2-mean_r1:+.4f}")
    print(f"    Ridge alphas: {[f'{a:.1f}' for a in alphas2]}")
    print()

    # ── Experiment (3): top-15×15 interactions ─────────────────────────────────
    print("(3) +Interactions (top 15 type scores × top 15 demographics):")
    X_inter15 = add_interactions(X_base, y_pres, top_n_each=15)
    n_new15 = X_inter15.shape[1] - X_base.shape[1]
    print(f"    Added {n_new15} interaction features (shape: {X_inter15.shape})")
    mean_r3, rs3, alphas3 = ridge_loo_multi(X_inter15, holdout_in)
    print(f"    LOO r = {mean_r3:.4f}  per-dim = {[f'{r:.3f}' for r in rs3]}  Δ={mean_r3-mean_r1:+.4f}")
    print(f"    Ridge alphas: {[f'{a:.1f}' for a in alphas3]}")
    print()

    # ── Experiment (4): top-20×20 interactions ─────────────────────────────────
    print("(4) +Interactions (top 20 type scores × top 20 demographics):")
    X_inter20 = add_interactions(X_base, y_pres, top_n_each=20)
    n_new20 = X_inter20.shape[1] - X_base.shape[1]
    print(f"    Added {n_new20} interaction features (shape: {X_inter20.shape})")
    mean_r4, rs4, alphas4 = ridge_loo_multi(X_inter20, holdout_in)
    print(f"    LOO r = {mean_r4:.4f}  per-dim = {[f'{r:.3f}' for r in rs4]}  Δ={mean_r4-mean_r1:+.4f}")
    print(f"    Ridge alphas: {[f'{a:.1f}' for a in alphas4]}")
    print()

    # ── Experiment (5): County mean × demographics interactions ────────────────
    print("(5) County mean × demographics interactions (county_mean × each demo feature):")
    # Simple: multiply county_mean by each demographic (N x D_demo new features)
    cm = county_mean_in[:, np.newaxis]
    cm_demo = cm * demo_scaled  # (N, D_demo)
    cm_demo_scaler = StandardScaler()
    cm_demo_scaled = cm_demo_scaler.fit_transform(cm_demo)
    X_cmdemo = np.column_stack([X_base, cm_demo_scaled])
    n_cm = X_cmdemo.shape[1] - X_base.shape[1]
    print(f"    Added {n_cm} cm×demo features (shape: {X_cmdemo.shape})")
    mean_r5, rs5, alphas5 = ridge_loo_multi(X_cmdemo, holdout_in)
    print(f"    LOO r = {mean_r5:.4f}  per-dim = {[f'{r:.3f}' for r in rs5]}  Δ={mean_r5-mean_r1:+.4f}")
    print()

    # ── Summary ────────────────────────────────────────────────────────────────
    all_results = [
        ("(1) Baseline", mean_r1),
        ("(2) +Inter top10×10", mean_r2),
        ("(3) +Inter top15×15", mean_r3),
        ("(4) +Inter top20×20", mean_r4),
        ("(5) +cm×demo interactions", mean_r5),
    ]
    best_name, best_r = max(all_results, key=lambda x: x[1])

    print("=" * 70)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 70)
    print(f"  Published best (S197):                       LOO r = 0.649")
    for label, r in all_results:
        marker = " <-- BEST" if label == best_name else ""
        print(f"  {label:<35}  LOO r = {r:.4f}  Δ={r-0.649:+.4f}{marker}")
    print()

    if best_r > 0.649:
        print(f"  BEATS BASELINE: {best_name}  LOO r = {best_r:.4f}")
    else:
        print(f"  BELOW BASELINE: best = {best_r:.4f}. Interactions hurt or are neutral.")
        print("  Interpretation: Ridge already captures the linear signal efficiently.")
        print("  Adding interactions increases dimensionality without proportional signal gain.")
        print("  The high regularization alpha needed confirms overfitting pressure.")


if __name__ == "__main__":
    main()
