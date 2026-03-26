"""2x2 experiment: min_year=[2008,2012] x method=[Ridge+mean, Ridge+demo].

Tests whether the two improvements (fewer dims + demographics) stack.

Context:
  - min_year=2008, Ridge+demo, N=3,106: LOO r=0.650 (current best)
  - min_year=2012 showed improvement in training-window experiment
  - Does combining both push higher?

Usage:
    uv run python scripts/experiment_combined_improvements.py
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
    """Parse start year from column name like 'pres_d_shift_08_12' -> 2008."""
    import re
    m = re.search(r"(\d{2})_(\d{2})$", col)
    if m is None:
        return None
    y2 = int(m.group(1))
    return y2 + (1900 if y2 >= 50 else 2000)


def is_holdout_col(col: str) -> bool:
    return "20_24" in col


def classify_columns(all_cols: list[str], min_year: int) -> tuple[list[str], list[str]]:
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


def ridge_loo_r(X: np.ndarray, holdout_raw: np.ndarray) -> float:
    """Fit RidgeCV (GCV alpha) then compute exact LOO r; return mean over holdout dims."""
    alphas = np.logspace(-3, 6, 100)
    H = holdout_raw.shape[1]
    per_r: list[float] = []

    for h_idx in range(H):
        y = holdout_raw[:, h_idx]
        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        alpha = float(rcv.alpha_)
        y_loo = ridge_loo_predictions(X, y, alpha)
        r, _ = pearsonr(y, y_loo)
        per_r.append(float(np.clip(r, -1.0, 1.0)))

    return float(np.mean(per_r))


# ── Data loading ───────────────────────────────────────────────────────────────


def load_shifts():
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    return pd.read_parquet(shifts_path)


def load_features():
    feat_path = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"
    return pd.read_parquet(feat_path)


# ── Run one cell ───────────────────────────────────────────────────────────────


def run_cell(
    shifts_df: pd.DataFrame,
    features_df: pd.DataFrame,
    min_year: int,
    use_demo: bool,
    pw: float = 8.0,
    j: int = 100,
    temperature: float = 10.0,
) -> float:
    """Run one cell of the 2x2 matrix; return mean LOO r."""
    all_cols = [c for c in shifts_df.columns if c != "county_fips"]
    training_cols, holdout_cols = classify_columns(all_cols, min_year)

    # Inner join with features to align county_fips
    if use_demo:
        merged = shifts_df.merge(features_df, on="county_fips", how="inner")
    else:
        merged = shifts_df.dropna(subset=training_cols + holdout_cols)

    county_fips = merged["county_fips"].values

    # Build training matrix
    training_raw = merged[training_cols].values.astype(float)
    holdout_raw = merged[holdout_cols].values.astype(float)

    pres_idx = [i for i, c in enumerate(training_cols) if "pres_" in c]

    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    if pw != 1.0:
        training_scaled[:, pres_idx] *= pw

    # Discover types
    type_result = discover_types(
        training_scaled, j=j, temperature=temperature, random_state=42
    )
    scores = type_result.scores
    county_mean = training_raw.mean(axis=1)

    if use_demo:
        # Build demographic features from features_df (already inner-joined)
        feat_cols = [c for c in features_df.columns if c != "county_fips"]
        demo_mat = merged[feat_cols].values.astype(float)

        # Impute NaNs with column median
        for col_i in range(demo_mat.shape[1]):
            col = demo_mat[:, col_i]
            if np.isnan(col).any():
                med = np.nanmedian(col)
                demo_mat[np.isnan(col), col_i] = med

        demo_scaler = StandardScaler()
        demo_scaled = demo_scaler.fit_transform(demo_mat)
        X = np.column_stack([scores, county_mean, demo_scaled])
    else:
        X = np.column_stack([scores, county_mean])

    return ridge_loo_r(X, holdout_raw)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    pw = 8.0
    j = 100
    temperature = 10.0

    print("=" * 70)
    print("COMBINED IMPROVEMENTS: min_year x method (2x2 table)")
    print(f"  pw={pw}, J={j}, T={temperature}")
    print("=" * 70)
    print()

    print("Loading data...")
    shifts_df = load_shifts()
    features_df = load_features()

    # Quick sanity checks
    total_counties = len(shifts_df)
    inner_n = len(shifts_df.merge(features_df, on="county_fips", how="inner"))
    all_shift_cols = [c for c in shifts_df.columns if c != "county_fips"]
    train_2008, holdout_cols = classify_columns(all_shift_cols, min_year=2008)
    train_2012, _ = classify_columns(all_shift_cols, min_year=2012)

    print(f"  Shift matrix: {total_counties} counties")
    print(f"  After inner join w/ features: {inner_n} counties")
    print(f"  Training dims @ min_year=2008: {len(train_2008)}")
    print(f"  Training dims @ min_year=2012: {len(train_2012)}")
    print(f"  Holdout dims: {holdout_cols}")
    demo_feat_cols = [c for c in features_df.columns if c != "county_fips"]
    print(f"  Demographic features: {len(demo_feat_cols)}")
    print()

    # 2x2 grid
    min_years = [2008, 2012]
    methods = [
        ("Ridge+mean", False),
        ("Ridge+demo", True),
    ]

    results: dict[tuple, float] = {}

    for min_year in min_years:
        for method_label, use_demo in methods:
            key = (min_year, method_label)
            print(f"Running: min_year={min_year}, method={method_label}...", flush=True)
            loo_r = run_cell(
                shifts_df,
                features_df,
                min_year=min_year,
                use_demo=use_demo,
                pw=pw,
                j=j,
                temperature=temperature,
            )
            results[key] = loo_r
            print(f"  LOO r = {loo_r:.4f}")
            print()

    # ── Print 2x2 table ────────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS: LOO r (2x2 table)")
    print("=" * 70)
    print()

    col_w = 16
    header = f"{'':20}" + "".join(f"{lbl:>{col_w}}" for lbl, _ in methods)
    print(header)
    print("-" * (20 + col_w * len(methods)))

    for min_year in min_years:
        row = f"min_year={min_year:<4}    "
        for method_label, _ in methods:
            val = results[(min_year, method_label)]
            row += f"{val:>{col_w}.4f}"
        print(row)

    print()

    # Deltas
    base = results[(2008, "Ridge+mean")]
    print("Deltas vs (2008, Ridge+mean):")
    for min_year in min_years:
        for method_label, _ in methods:
            key = (min_year, method_label)
            delta = results[key] - base
            marker = " <-- BEST" if results[key] == max(results.values()) else ""
            print(
                f"  min_year={min_year}, {method_label:15}: "
                f"LOO r={results[key]:.4f}  Δ={delta:+.4f}{marker}"
            )

    print()
    best_key = max(results, key=lambda k: results[k])
    print(
        f"BEST: min_year={best_key[0]}, {best_key[1]:15} --> LOO r = {results[best_key]:.4f}"
    )
    print()

    # Stacking check
    improvement_2012 = results[(2012, "Ridge+mean")] - results[(2008, "Ridge+mean")]
    improvement_demo = results[(2008, "Ridge+demo")] - results[(2008, "Ridge+mean")]
    combined = results[(2012, "Ridge+demo")] - results[(2008, "Ridge+mean")]
    synergy = combined - (improvement_2012 + improvement_demo)
    print("Stacking analysis:")
    print(f"  min_year=2012 alone adds:    {improvement_2012:+.4f}")
    print(f"  demographics alone add:      {improvement_demo:+.4f}")
    print(f"  combined adds:               {combined:+.4f}")
    print(f"  synergy (combined - sum):    {synergy:+.4f}")


if __name__ == "__main__":
    main()
