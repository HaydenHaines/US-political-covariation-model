"""Experiment 2: Extended ACS Feature Set for Ridge

Tests whether adding more ACS-derived features (poverty, urbanicity,
migration, population density, more occupational detail) to the Ridge
model improves LOO r beyond 0.649.

The current model (S197 best) uses:
  - 13 ACS features (race/ethnicity, education, housing tenure,
    commute mode, income, age, management occupation)
  - 6 RCMS features (religion)
  - 100 type scores + county_mean

This experiment adds:
  - Urbanicity: log_pop_density, pop_per_sq_mi
  - Migration: net_migration_rate, avg_inflow_income, migration_diversity,
                inflow_outflow_ratio
  - Additional ACS: poverty proxy (via income quartile), more education tiers,
                    car commute percentage, no-degree ratio
  - Pop size: log_pop_total

All via Ridge with exact LOO hat matrix shortcut.

Usage:
    uv run python scripts/experiments/exp2_more_acs_features.py
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


def ridge_loo_r(X: np.ndarray, y: np.ndarray) -> float:
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    rcv.fit(X, y)
    alpha = float(rcv.alpha_)
    y_loo = ridge_loo_predictions(X, y, alpha)
    r, _ = pearsonr(y, y_loo)
    return float(r), alpha


def ridge_loo_multi(X: np.ndarray, holdout: np.ndarray) -> tuple[float, list[float]]:
    H = holdout.shape[1]
    rs = []
    for h in range(H):
        r, _ = ridge_loo_r(X, holdout[:, h])
        rs.append(r)
    return float(np.mean(rs)), rs


# ── Feature builders ─────────────────────────────────────────────────────────────


def build_acs_base(acs: pd.DataFrame) -> pd.DataFrame:
    """Replicate the existing S197 ACS feature set exactly."""
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


def build_acs_extended(acs: pd.DataFrame) -> pd.DataFrame:
    """Extended ACS feature set — adds new signals on top of base."""
    feat = build_acs_base(acs)

    pop = acs["pop_total"].replace(0, np.nan)
    educ_total = acs["educ_total"].replace(0, np.nan)
    commute = acs["commute_total"].replace(0, np.nan)

    # Population size (log)
    feat["log_pop_total"] = np.log1p(acs["pop_total"].clip(lower=1))

    # No-degree ratio: proportion of 25+ without any college
    educ_no_degree = educ_total - (
        acs["educ_bachelors"] + acs["educ_masters"]
        + acs["educ_professional"] + acs["educ_doctorate"]
    )
    feat["pct_no_degree"] = (educ_no_degree.clip(lower=0)) / educ_total

    # Graduate degree ratio (masters + professional + doctorate)
    educ_grad = (
        acs["educ_masters"] + acs["educ_professional"] + acs["educ_doctorate"]
    )
    feat["pct_grad_degree"] = educ_grad / educ_total

    # Non-citizen proxy: born outside US — not in ACS table, use income-age interaction instead
    # Income-to-age ratio (captures "young rich" vs "old comfortable")
    # Use as a simple interaction proxy without PolynomialFeatures
    median_income = acs["median_hh_income"].clip(lower=1)
    median_age = acs["median_age"].clip(lower=1)
    feat["income_age_ratio"] = np.log1p(median_income) / median_age

    # Renter proportion (complement of owner-occupied)
    housing = acs["housing_units"].replace(0, np.nan)
    feat["pct_renter"] = 1.0 - (acs["housing_owner"] / housing).clip(0, 1)

    # Non-car commuters (transit + wfh + other)
    feat["pct_non_car_commute"] = 1.0 - (acs["commute_car"] / commute).clip(0, 1)

    return feat


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    assembled = PROJECT_ROOT / "data" / "assembled"

    print("=" * 70)
    print("EXPERIMENT 2: Extended ACS Feature Set")
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

    print(f"  {df.shape[0]} counties, {len(training_cols)} training dims, {len(holdout_cols)} holdout dims")
    print(f"  Holdout columns: {holdout_cols}")
    print()

    # ── Discover types ─────────────────────────────────────────────────────────
    print("Discovering types (J=100, T=10, pw=8.0)...")
    type_result = discover_types(training_scaled, j=100, temperature=10.0, random_state=42)
    scores = type_result.scores
    county_mean = training_raw.mean(axis=1)
    print(f"  scores shape: {scores.shape}")
    print()

    # ── Load ACS (base + extended) ─────────────────────────────────────────────
    acs_raw = pd.read_parquet(assembled / "acs_counties_2022.parquet")
    acs_base_feat = build_acs_base(acs_raw)
    acs_ext_feat = build_acs_extended(acs_raw)

    print(f"  ACS base features: {len([c for c in acs_base_feat.columns if c != 'county_fips'])}")
    print(f"  ACS extended features: {len([c for c in acs_ext_feat.columns if c != 'county_fips'])}")

    # ── Load RCMS ──────────────────────────────────────────────────────────────
    rcms = pd.read_parquet(assembled / "county_rcms_features.parquet")
    rcms_cols = ["county_fips", "evangelical_share", "mainline_share", "catholic_share",
                 "black_protestant_share", "congregations_per_1000", "religious_adherence_rate"]
    rcms = rcms[rcms_cols]

    # ── Load urbanicity ───────────────────────────────────────────────────────
    urb = pd.read_parquet(assembled / "county_urbanicity_features.parquet")
    urb_cols = [c for c in urb.columns if c != "county_fips"]
    print(f"  Urbanicity features: {urb_cols}")

    # ── Load migration ────────────────────────────────────────────────────────
    mig = pd.read_parquet(assembled / "county_migration_features.parquet")
    mig_cols = [c for c in mig.columns if c != "county_fips"]
    print(f"  Migration features: {mig_cols}")
    print()

    def align_features(feat_df: pd.DataFrame, county_fips_arr: np.ndarray) -> np.ndarray | None:
        """Merge features with county_fips_arr via inner join. Returns aligned matrix or None."""
        fips_df = pd.DataFrame({"county_fips": county_fips_arr})
        aligned = fips_df.merge(feat_df, on="county_fips", how="inner")
        demo_fips = aligned["county_fips"].values
        feat_c = [c for c in aligned.columns if c != "county_fips"]
        mat = aligned[feat_c].values.astype(float)

        fips_series = pd.Series(county_fips_arr)
        demo_fips_set = set(demo_fips)
        mask = fips_series.isin(demo_fips_set).values
        fips_in = county_fips_arr[mask]
        idx_map = {f: i for i, f in enumerate(demo_fips)}
        reindex = [idx_map[f] for f in fips_in]
        mat_aligned = mat[reindex]

        # Impute NaNs
        for col_i in range(mat_aligned.shape[1]):
            col = mat_aligned[:, col_i]
            if np.isnan(col).any():
                med = np.nanmedian(col)
                mat_aligned[np.isnan(col), col_i] = med

        return mat_aligned, mask

    # ── Build full merged feature set ─────────────────────────────────────────
    # Merge all demographics together before aligning
    ext_merged = acs_ext_feat.merge(rcms, on="county_fips", how="left")
    ext_merged = ext_merged.merge(urb, on="county_fips", how="left")
    ext_merged = ext_merged.merge(mig, on="county_fips", how="left")

    base_merged = acs_base_feat.merge(rcms, on="county_fips", how="left")

    # Get masks and aligned matrices
    ext_mat, ext_mask = align_features(ext_merged, county_fips)
    base_mat, base_mask = align_features(base_merged, county_fips)

    # Verify masks match (they should — same inner-join fips set)
    assert np.array_equal(ext_mask, base_mask), "Masks differ — check data!"
    mask = ext_mask

    scores_in = scores[mask]
    county_mean_in = county_mean[mask]
    holdout_in = holdout_raw[mask]
    N_in = mask.sum()

    ext_feat_count = ext_mat.shape[1]
    base_feat_count = base_mat.shape[1]
    print(f"Working set: {N_in} counties")
    print(f"  Base demographics: {base_feat_count} features")
    print(f"  Extended demographics: {ext_feat_count} features (added {ext_feat_count - base_feat_count})")
    print()

    # Standardize
    base_scaler = StandardScaler()
    base_demo_scaled = base_scaler.fit_transform(base_mat)

    ext_scaler = StandardScaler()
    ext_demo_scaled = ext_scaler.fit_transform(ext_mat)

    # ── Experiments ────────────────────────────────────────────────────────────
    print("-" * 70)
    print("Running Ridge LOO (exact hat-matrix)...")
    print("-" * 70)
    print()

    # (1) Baseline: scores + county_mean + base ACS + RCMS (reproduce S197)
    print("(1) Baseline: scores + county_mean + base ACS + RCMS...")
    X1 = np.column_stack([scores_in, county_mean_in, base_demo_scaled])
    mean_r1, rs1 = ridge_loo_multi(X1, holdout_in)
    print(f"    LOO r = {mean_r1:.4f}  per-dim = {[f'{r:.3f}' for r in rs1]}")
    print()

    # (2) +Extended ACS (adds log_pop, no_degree, grad_degree, income_age_ratio,
    #                     renter, non_car_commute)
    print(f"(2) +Extended ACS ({ext_feat_count} features, adds urbanicity + migration + extra ACS)...")
    X2 = np.column_stack([scores_in, county_mean_in, ext_demo_scaled])
    mean_r2, rs2 = ridge_loo_multi(X2, holdout_in)
    print(f"    LOO r = {mean_r2:.4f}  per-dim = {[f'{r:.3f}' for r in rs2]}  Δ={mean_r2 - mean_r1:+.4f}")
    print()

    # (3) +Extended ACS +urbanicity +migration (already merged, same X2 above)
    # Split out ablations: +urbanicity only
    urb_mat, urb_mask = align_features(
        acs_base_feat.merge(rcms, on="county_fips", how="left").merge(urb, on="county_fips", how="left"),
        county_fips,
    )
    assert np.array_equal(urb_mask, mask)
    urb_scaler = StandardScaler()
    X3 = np.column_stack([scores_in, county_mean_in, urb_scaler.fit_transform(urb_mat)])
    mean_r3, rs3 = ridge_loo_multi(X3, holdout_in)
    print(f"(3) +Urbanicity only (base + urb, {urb_mat.shape[1]} demo feats):")
    print(f"    LOO r = {mean_r3:.4f}  per-dim = {[f'{r:.3f}' for r in rs3]}  Δ={mean_r3 - mean_r1:+.4f}")
    print()

    # +migration only
    mig_mat, mig_mask = align_features(
        acs_base_feat.merge(rcms, on="county_fips", how="left").merge(mig, on="county_fips", how="left"),
        county_fips,
    )
    assert np.array_equal(mig_mask, mask)
    mig_scaler = StandardScaler()
    X4 = np.column_stack([scores_in, county_mean_in, mig_scaler.fit_transform(mig_mat)])
    mean_r4, rs4 = ridge_loo_multi(X4, holdout_in)
    print(f"(4) +Migration only (base + mig, {mig_mat.shape[1]} demo feats):")
    print(f"    LOO r = {mean_r4:.4f}  per-dim = {[f'{r:.3f}' for r in rs4]}  Δ={mean_r4 - mean_r1:+.4f}")
    print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("EXPERIMENT 2 SUMMARY")
    print("=" * 70)
    print(f"  Published best (S197):                      LOO r = 0.649")
    print(f"  (1) Baseline reproduce (base ACS+RCMS):     LOO r = {mean_r1:.4f}  Δ={mean_r1-0.649:+.4f}")
    print(f"  (2) Full extended features:                  LOO r = {mean_r2:.4f}  Δ={mean_r2-mean_r1:+.4f}")
    print(f"  (3) +Urbanicity only:                        LOO r = {mean_r3:.4f}  Δ={mean_r3-mean_r1:+.4f}")
    print(f"  (4) +Migration only:                         LOO r = {mean_r4:.4f}  Δ={mean_r4-mean_r1:+.4f}")
    print()

    best = max(mean_r1, mean_r2, mean_r3, mean_r4)
    if best > 0.649:
        print(f"  BEATS BASELINE: best LOO r = {best:.4f} (Δ={best-0.649:+.4f})")
    else:
        print(f"  BELOW BASELINE: best LOO r = {best:.4f} (Δ={best-0.649:+.4f})")
        print("  Interpretation: Additional features are not providing incremental signal.")
        print("  The base feature set already captures the relevant demographic signal.")


if __name__ == "__main__":
    main()
