"""Experiment: Does restricting the training window improve LOO predictive quality?

Tests different min_year values [2000, 2004, 2008, 2012, 2016] for training columns.
For each window, runs KMeans J=100 and evaluates:
  - Ridge LOO r (scores + county_mean)
  - Type-mean LOO r
  - Ridge LOO r with demographics augmented

Holdout target: pres_d_shift_20_24 (always excluded from training).
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

HOLDOUT_COLS = ["pres_d_shift_20_24", "pres_r_shift_20_24", "pres_turnout_shift_20_24"]
PRESIDENTIAL_WEIGHT = 8.0
J = 100
TEMPERATURE = 10.0
RANDOM_STATE = 42


# ── Column year parsing ───────────────────────────────────────────────────────

def col_start_year(col: str) -> int:
    """Parse start year from shift column name.

    Column pattern: ..._YY_YY (e.g., pres_d_shift_08_12 → start year 2008).
    """
    m = re.search(r"_(\d{2})_(\d{2})$", col)
    if not m:
        return -1
    y1_2dig = int(m.group(1))
    y1 = y1_2dig + (1900 if y1_2dig >= 50 else 2000)
    return y1


# ── KMeans type discovery (matches run_type_discovery.py) ────────────────────

def discover_types(shift_matrix: np.ndarray, j: int = J) -> np.ndarray:
    """Return soft membership scores (N × J) via temperature-sharpened inverse distance."""
    km = KMeans(n_clusters=j, random_state=RANDOM_STATE, n_init=10)
    km.fit(shift_matrix)
    centroids = km.cluster_centers_  # J × D

    # Euclidean distances to each centroid
    dists = np.zeros((len(shift_matrix), j))
    for t in range(j):
        dists[:, t] = np.linalg.norm(shift_matrix - centroids[t], axis=1)

    # Temperature-sharpened softmax in log space
    eps = 1e-10
    log_w = -TEMPERATURE * np.log(dists + eps)
    log_w -= log_w.max(axis=1, keepdims=True)
    powered = np.exp(log_w)
    scores = powered / powered.sum(axis=1, keepdims=True)
    return scores


# ── LOO via hat matrix ────────────────────────────────────────────────────────

def ridge_loo_r(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float]:
    """Compute Ridge LOO r and RMSE via hat-matrix shortcut.

    Hat matrix: X_aug = [1, X], penalty alpha*I (intercept unpenalized).
    H = X_aug @ inv(X_aug.T X_aug + P) @ X_aug.T
    y_loo_i = y_i - e_i / (1 - h_ii)
    """
    N = X.shape[0]
    ones = np.ones((N, 1))
    X_aug = np.hstack([ones, X])
    D = X_aug.shape[1]

    penalty = np.eye(D) * alpha
    penalty[0, 0] = 0.0  # don't penalize intercept

    A = X_aug.T @ X_aug + penalty
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    H = X_aug @ A_inv @ X_aug.T  # N × N
    h = np.diag(H)

    # Fit full model for residuals
    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    e = y - y_hat

    # LOO predictions (avoid division by ~1 when h ≈ 1)
    h_capped = np.clip(h, None, 0.9999)
    y_loo = y - e / (1 - h_capped)

    r = np.corrcoef(y, y_loo)[0, 1]
    rmse = np.sqrt(np.mean((y - y_loo) ** 2))
    return float(r), float(rmse)


def best_alpha_ridgecv(X: np.ndarray, y: np.ndarray) -> float:
    """Select best Ridge alpha via RidgeCV (GCV)."""
    alphas = np.logspace(-3, 6, 100)
    rcv = RidgeCV(alphas=alphas, fit_intercept=True)
    rcv.fit(X, y)
    return float(rcv.alpha_)


# ── Type-mean LOO ─────────────────────────────────────────────────────────────

def type_mean_loo_r(scores: np.ndarray, y: np.ndarray) -> float:
    """Type-mean prior LOO r.

    For each county i, compute type-mean from all counties EXCEPT i,
    then predict y_i as weighted average of those leave-i-out type means.
    Approximation: use scores @ type_means, then correct for self-influence.
    Full LOO is O(N²); use the fast hat-matrix route via Ridge with scores.
    """
    # Simpler: treat as Ridge with X=scores, which is already done in ridge_loo_r
    # Here we compute the naive type-mean (not leave-one-out) as a baseline
    # and use Ridge to handle the LOO correction.
    # The "type-mean LOO r" referred to in CLAUDE.md is the one already computed
    # via the Ridge route using scores+county_mean.
    # We compute a simple type-mean prediction (no LOO) just for comparison.
    type_means = (scores.T @ y) / (scores.sum(axis=0) + 1e-10)
    y_pred = scores @ type_means
    r = np.corrcoef(y, y_pred)[0, 1]
    return float(r)


# ── Demographics feature builder ──────────────────────────────────────────────

def build_demo_features(county_fips: np.ndarray) -> pd.DataFrame | None:
    """Load ACS + RCMS, compute pct features, standardize, return aligned DataFrame."""
    acs_path = PROJECT_ROOT / "data" / "assembled" / "acs_counties_2022.parquet"
    rcms_path = PROJECT_ROOT / "data" / "assembled" / "county_rcms_features.parquet"

    if not acs_path.exists():
        print("  [demo] ACS file missing, skipping demographics augmentation")
        return None
    if not rcms_path.exists():
        print("  [demo] RCMS file missing, skipping demographics augmentation")
        return None

    acs = pd.read_parquet(acs_path)
    rcms = pd.read_parquet(rcms_path)

    # Compute percentage features from ACS
    pt = acs["pop_total"].clip(lower=1)
    acs_feats = pd.DataFrame({
        "county_fips": acs["county_fips"],
        "pct_white_nh": acs["pop_white_nh"] / pt,
        "pct_black": acs["pop_black"] / pt,
        "pct_asian": acs["pop_asian"] / pt,
        "pct_hispanic": acs["pop_hispanic"] / pt,
        "median_age": acs["median_age"],
        "log_median_hh_income": np.log1p(acs["median_hh_income"].clip(lower=0)),
        "pct_owner_occupied": acs["housing_owner"] / acs["housing_units"].clip(lower=1),
        "pct_commute_car": acs["commute_car"] / acs["commute_total"].clip(lower=1),
        "pct_commute_transit": acs["commute_transit"] / acs["commute_total"].clip(lower=1),
        "pct_wfh": acs["commute_wfh"] / acs["commute_total"].clip(lower=1),
        "pct_college_plus": (
            acs["educ_bachelors"] + acs["educ_masters"] +
            acs["educ_professional"] + acs["educ_doctorate"]
        ) / acs["educ_total"].clip(lower=1),
        "pct_mgmt": (
            acs["occ_mgmt_male"] + acs["occ_mgmt_female"]
        ) / acs["occ_total"].clip(lower=1),
    })

    # Merge RCMS
    rcms_feats = rcms[["county_fips", "evangelical_share", "catholic_share",
                         "religious_adherence_rate"]].copy()

    demo = acs_feats.merge(rcms_feats, on="county_fips", how="left")

    # Align to our county_fips array
    fips_df = pd.DataFrame({"county_fips": county_fips})
    demo = fips_df.merge(demo, on="county_fips", how="left")

    feature_cols = [c for c in demo.columns if c != "county_fips"]
    X_demo = demo[feature_cols].values.astype(float)

    # Standardize (handle NaNs by filling with col mean before scaling)
    col_means = np.nanmean(X_demo, axis=0)
    inds = np.where(np.isnan(X_demo))
    X_demo[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)
    return X_demo


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment() -> None:
    print("Loading shift matrix...")
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    county_fips = df["county_fips"].values
    N = len(county_fips)

    # Holdout target: D shift 2020→2024
    y = df["pres_d_shift_20_24"].values.astype(float)
    county_mean = y.mean()  # scalar; subtract to get demeaned residual

    # Build demographics once (shared across windows)
    print("Building demographics features...")
    X_demo = build_demo_features(county_fips)
    if X_demo is not None:
        print(f"  Demo features: {X_demo.shape[1]} dims, {N} counties")

    all_shift_cols = [c for c in df.columns
                      if c != "county_fips" and c not in HOLDOUT_COLS]

    min_years = [2000, 2004, 2008, 2012, 2016]

    results = []

    print(f"\n{'='*80}")
    print(f"Experiment: Training window vs. LOO predictive quality (J={J})")
    print(f"Holdout: pres_d_shift_20_24  |  N={N} counties")
    print(f"{'='*80}\n")

    for min_year in min_years:
        # Filter columns to training window
        train_cols = [c for c in all_shift_cols if col_start_year(c) >= min_year]
        n_dims = len(train_cols)

        if n_dims == 0:
            print(f"min_year={min_year}: NO COLUMNS, skipping")
            continue

        print(f"--- min_year={min_year} | {n_dims} training dims ---")
        pres_cols = [c for c in train_cols if "pres_" in c]
        print(f"    Presidential cols: {len(pres_cols)}, Other cols: {n_dims - len(pres_cols)}")

        # Build shift matrix with StandardScaler + presidential weight
        raw = df[train_cols].values.astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(raw)

        pres_indices = [i for i, c in enumerate(train_cols) if "pres_" in c]
        X_scaled[:, pres_indices] *= PRESIDENTIAL_WEIGHT

        # KMeans type discovery
        print(f"    Running KMeans J={J}...")
        scores = discover_types(X_scaled, j=J)  # N × J

        # Type-mean prediction (non-LOO, for reference)
        type_mean_r_naive = type_mean_loo_r(scores, y)

        # Build Ridge feature matrix: scores (J) + county_mean (1 scalar, broadcasts)
        county_mean_col = np.full((N, 1), county_mean)
        X_ridge = np.hstack([scores, county_mean_col])

        # Select best alpha via RidgeCV
        alpha = best_alpha_ridgecv(X_ridge, y)
        ridge_r, ridge_rmse = ridge_loo_r(X_ridge, y, alpha)

        row = {
            "min_year": min_year,
            "n_dims": n_dims,
            "type_mean_r_naive": type_mean_r_naive,
            "ridge_loo_r": ridge_r,
            "ridge_loo_rmse": ridge_rmse,
            "ridge_alpha": alpha,
        }

        # Demographics-augmented Ridge
        if X_demo is not None:
            X_aug = np.hstack([scores, county_mean_col, X_demo])
            alpha_aug = best_alpha_ridgecv(X_aug, y)
            ridge_r_aug, ridge_rmse_aug = ridge_loo_r(X_aug, y, alpha_aug)
            row["ridge_loo_r_aug"] = ridge_r_aug
            row["ridge_loo_rmse_aug"] = ridge_rmse_aug
            row["ridge_alpha_aug"] = alpha_aug
            print(f"    Ridge LOO r (scores+mean): {ridge_r:.4f}  |  +demo: {ridge_r_aug:.4f}  |  type-mean naive: {type_mean_r_naive:.4f}")
        else:
            print(f"    Ridge LOO r (scores+mean): {ridge_r:.4f}  |  type-mean naive: {type_mean_r_naive:.4f}")

        results.append(row)

    # Summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    hdr = f"{'min_year':>10} {'n_dims':>8} {'type_mean_r':>12} {'ridge_loo_r':>12} {'ridge_rmse':>11}"
    if X_demo is not None:
        hdr += f" {'ridge+demo_r':>13} {'ridge+demo_rmse':>15}"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        line = (
            f"{r['min_year']:>10} {r['n_dims']:>8} "
            f"{r['type_mean_r_naive']:>12.4f} "
            f"{r['ridge_loo_r']:>12.4f} "
            f"{r['ridge_loo_rmse']:>11.4f}"
        )
        if X_demo is not None and "ridge_loo_r_aug" in r:
            line += f" {r['ridge_loo_r_aug']:>13.4f} {r['ridge_loo_rmse_aug']:>15.4f}"
        print(line)

    print()
    if results:
        best = max(results, key=lambda r: r["ridge_loo_r"])
        print(f"Best Ridge LOO r (scores+mean): min_year={best['min_year']} → r={best['ridge_loo_r']:.4f} ({best['n_dims']} dims)")
        if X_demo is not None:
            best_aug = max(results, key=lambda r: r.get("ridge_loo_r_aug", -999))
            print(f"Best Ridge LOO r (+demo):       min_year={best_aug['min_year']} → r={best_aug.get('ridge_loo_r_aug', float('nan')):.4f}")

    # Baseline from CLAUDE.md for comparison
    print()
    print("Baselines (from CLAUDE.md / S197):")
    print("  County Ridge LOO r (scores+mean, J=100, 2008+): 0.533")
    print("  County holdout LOO r (type-mean): 0.448")
    print()


if __name__ == "__main__":
    run_experiment()
