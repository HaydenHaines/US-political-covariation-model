"""Ridge LOO augmentation experiments.

EXPERIMENT A — Presidential weight sweep for Ridge:
  Sweep pw over [1, 2, 4, 6, 8, 10, 12, 16].
  For each pw: scale shifts, discover_types(J=100), get scores.
  Compute Ridge LOO r with scores + county_mean.
  Find optimal pw for Ridge.

EXPERIMENT B — Demographics-augmented Ridge:
  At pw=8.0 (and best pw from A), add ACS + urbanicity + religion features
  to the Ridge feature set.
  Test Ridge LOO r with demographics added.

Usage:
    uv run python scripts/experiment_ridge_augmented.py
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


# ── Column helpers ────────────────────────────────────────────────────────────


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


# ── Ridge LOO via hat matrix ──────────────────────────────────────────────────


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


# ── Data loading ──────────────────────────────────────────────────────────────


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


def load_demographics(county_fips_arr: np.ndarray) -> pd.DataFrame:
    """Load and merge demographic features, return aligned DataFrame indexed by county_fips."""
    assembled = PROJECT_ROOT / "data" / "assembled"

    # ACS demographics_interpolated (most recent year per county)
    acs = pd.read_parquet(assembled / "demographics_interpolated.parquet")
    # Use the most recent year for each county
    acs = acs.sort_values("year").groupby("county_fips").last().reset_index()
    acs_cols = [
        "county_fips",
        "pct_white_nh",
        "pct_black",
        "pct_asian",
        "pct_hispanic",
        "pct_bachelors_plus",
        "pct_owner_occupied",
        "pct_wfh",
        "pct_transit",
        "median_hh_income",
        "median_age",
    ]
    acs = acs[acs_cols]

    # Urbanicity
    urb = pd.read_parquet(assembled / "county_urbanicity_features.parquet")
    urb = urb[["county_fips", "log_pop_density"]]

    # Religion
    rcms = pd.read_parquet(assembled / "county_rcms_features.parquet")
    rcms = rcms[["county_fips", "evangelical_share", "catholic_share", "religious_adherence_rate"]]

    # Merge
    merged = acs.merge(urb, on="county_fips", how="left")
    merged = merged.merge(rcms, on="county_fips", how="left")

    # Align to county_fips_arr order
    fips_df = pd.DataFrame({"county_fips": county_fips_arr})
    aligned = fips_df.merge(merged, on="county_fips", how="left")
    aligned = aligned.set_index("county_fips")

    return aligned


# ── Experiment A: Presidential weight sweep ───────────────────────────────────


def experiment_a(
    df, training_cols, holdout_cols,
    pw_values=(1, 2, 4, 6, 8, 10, 12, 16),
    j: int = 100,
    temperature: float = 10.0,
):
    print("\n" + "=" * 70)
    print("EXPERIMENT A — Presidential weight sweep for Ridge (J=100)")
    print("=" * 70)

    results = []
    best_pw = None
    best_r = -np.inf

    for pw in pw_values:
        training_raw, training_scaled, holdout_raw, county_fips = build_matrices(
            df, training_cols, holdout_cols, presidential_weight=pw
        )

        type_result = discover_types(training_scaled, j=j, temperature=temperature, random_state=42)
        scores = type_result.scores

        county_mean = training_raw.mean(axis=1, keepdims=True)
        X = np.column_stack([scores, county_mean])

        res = ridge_loo_r(X, holdout_raw)
        mean_r = res["mean_r"]
        results.append({"pw": pw, "mean_r": mean_r, "per_dim_r": res["per_dim_r"]})

        marker = ""
        if mean_r > best_r:
            best_r = mean_r
            best_pw = pw
            marker = "  ← best so far"
        print(
            f"  pw={pw:>2}  LOO r={mean_r:.4f}"
            f"  dims={[f'{r:.3f}' for r in res['per_dim_r']]}{marker}"
        )

    print(f"\n  *** Best pw for Ridge: pw={best_pw}  LOO r={best_r:.4f} ***")
    return results, best_pw, best_r


# ── Experiment B: Demographics-augmented Ridge ────────────────────────────────


def experiment_b(
    df, training_cols, holdout_cols,
    pw_values_to_test: list[float],
    j: int = 100,
    temperature: float = 10.0,
):
    print("\n" + "=" * 70)
    print("EXPERIMENT B — Demographics-augmented Ridge")
    print(f"  Testing pw values: {pw_values_to_test}")
    print("=" * 70)

    results = []

    for pw in pw_values_to_test:
        training_raw, training_scaled, holdout_raw, county_fips = build_matrices(
            df, training_cols, holdout_cols, presidential_weight=pw
        )

        type_result = discover_types(training_scaled, j=j, temperature=temperature, random_state=42)
        scores = type_result.scores

        county_mean = training_raw.mean(axis=1)

        # Load demographics aligned to this county set
        demo_df = load_demographics(county_fips)
        demo_feat_cols = [c for c in demo_df.columns]  # all non-fips
        demo_mat = demo_df[demo_feat_cols].values.astype(float)

        # Count coverage
        n_counties = len(county_fips)
        n_demo_missing = np.isnan(demo_mat).any(axis=1).sum()
        print(f"\n  pw={pw}: {n_counties} counties, {len(demo_feat_cols)} demo features")
        print(f"    Missing demo rows: {n_demo_missing}/{n_counties}")

        # Fill missing demographics with column median
        for col_i in range(demo_mat.shape[1]):
            col = demo_mat[:, col_i]
            med = np.nanmedian(col)
            demo_mat[np.isnan(col), col_i] = med

        # Standardize demographics
        demo_scaler = StandardScaler()
        demo_scaled = demo_scaler.fit_transform(demo_mat)

        # B1: scores + county_mean (baseline, no demo)
        X_base = np.column_stack([scores, county_mean])
        res_base = ridge_loo_r(X_base, holdout_raw)

        # B2: scores + county_mean + demographics
        X_demo = np.column_stack([scores, county_mean, demo_scaled])
        res_demo = ridge_loo_r(X_demo, holdout_raw)

        # B3: scores + county_mean + demographics + log_pop_density^2 interaction
        # (just demographics; no extra engineering)

        print(
            f"  pw={pw}  Baseline (scores+mean):       LOO r={res_base['mean_r']:.4f}"
            f"  dims={[f'{r:.3f}' for r in res_base['per_dim_r']]}"
        )
        print(
            f"  pw={pw}  +Demographics ({len(demo_feat_cols)} feats): LOO r={res_demo['mean_r']:.4f}"
            f"  dims={[f'{r:.3f}' for r in res_demo['per_dim_r']]}"
            f"  Δ={res_demo['mean_r']-res_base['mean_r']:+.4f}"
        )

        results.append({
            "pw": pw,
            "n_demo_features": len(demo_feat_cols),
            "demo_feature_names": demo_feat_cols,
            "baseline_r": res_base["mean_r"],
            "baseline_per_dim": res_base["per_dim_r"],
            "demo_r": res_demo["mean_r"],
            "demo_per_dim": res_demo["per_dim_r"],
            "delta": res_demo["mean_r"] - res_base["mean_r"],
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("Loading shift data...")
    df, training_cols, holdout_cols = load_shifts(min_year=2008)
    print(f"  {df.shape[0]} counties, {len(training_cols)} training dims, holdout: {holdout_cols}")

    # Experiment A
    a_results, best_pw, best_r = experiment_a(
        df, training_cols, holdout_cols,
        pw_values=[1, 2, 4, 6, 8, 10, 12, 16],
        j=100,
    )

    # Experiment B — test at pw=8 (current) and best_pw if different
    pw_for_b = [8.0]
    if best_pw != 8 and best_pw is not None:
        pw_for_b = [8.0, float(best_pw)]

    b_results = experiment_b(
        df, training_cols, holdout_cols,
        pw_values_to_test=pw_for_b,
        j=100,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Baseline Ridge LOO r (pw=8, scores+mean):  0.533  (from CLAUDE.md)")
    print()
    print("  Experiment A — Best pw sweep results:")
    for row in sorted(a_results, key=lambda x: -x["mean_r"])[:5]:
        print(f"    pw={row['pw']:>2}  LOO r={row['mean_r']:.4f}")
    print(f"  → Optimal pw for Ridge: {best_pw}  (LOO r={best_r:.4f})")

    print()
    print("  Experiment B — Demographics augmentation:")
    for row in b_results:
        print(
            f"    pw={row['pw']}  baseline={row['baseline_r']:.4f}"
            f"  +demo={row['demo_r']:.4f}  Δ={row['delta']:+.4f}"
        )
    print()

    overall_best = max(
        [{"label": f"A: pw={r['pw']}", "r": r["mean_r"]} for r in a_results]
        + [{"label": f"B: pw={r['pw']} +demo", "r": r["demo_r"]} for r in b_results]
        + [{"label": f"B: pw={r['pw']} baseline", "r": r["baseline_r"]} for r in b_results],
        key=lambda x: x["r"],
    )
    print(f"  BEST overall: {overall_best['label']}  LOO r={overall_best['r']:.4f}")
    print(f"  vs prior 0.533: Δ={overall_best['r'] - 0.533:+.4f}")


if __name__ == "__main__":
    main()
