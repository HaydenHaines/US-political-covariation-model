"""T3 Column Selection Experiment — Tract-Level Type Discovery (Phase T.3).

Evaluates 4 configurations of the shift matrix for KMeans type discovery:
  A: Presidential-only (4 dims)
  B: Presidential + top-5 off-cycle (9 dims)
  C: Presidential + all 22 off-cycle (26 dims)
  D: Presidential + aggregated off-cycle (6 dims: 4 pres + mean_gov + mean_sen)

For each config:
  1. Select/compute columns
  2. Drop tracts with all-NaN presidential training shifts
  3. Fill remaining NaN with 0
  4. StandardScaler
  5. Apply presidential_weight=8.0 to pres columns
  6. PCA(n=min(15, n_cols), whiten=True)
  7. KMeans J=100, random_state=42, n_init=10
  8. Soft membership at temperature=10
  9. Holdout eval: predict pres_shift_20_24 via type-mean, compute Pearson r

Results saved to data/experiments/t3_column_selection_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.discovery.run_type_discovery import temperature_soft_membership

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "tract_shifts_national.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "experiments" / "t3_column_selection_results.json"

PRES_TRAIN_COLS = ["pres_shift_08_12", "pres_shift_12_16", "pres_shift_16_20"]
PRES_HOLDOUT_COL = "pres_shift_20_24"
ALL_PRES_COLS = PRES_TRAIN_COLS + [PRES_HOLDOUT_COL]

# Top-5 off-cycle by lowest NaN rate (least sparse = most signal per tract)
TOP5_OFFCYCLE_COLS = [
    "gov_shift_18_22_centered",   # 53.1% NaN — least sparse governor
    "sen_shift_16_18_centered",   # 67.1% NaN — least sparse senate
    "sen_shift_22_24_centered",   # 68.9% NaN — 2nd least sparse senate
    "sen_shift_18_20_centered",   # 73.0% NaN
    "sen_shift_16_20_centered",   # 77.1% NaN
]

ALL_OFFCYCLE_COLS = [
    "gov_shift_18_22_centered",
    "gov_shift_16_20_centered",
    "gov_shift_20_24_centered",
    "gov_shift_19_23_centered",
    "gov_shift_14_18_centered",
    "gov_shift_16_18_centered",
    "gov_shift_18_20_centered",
    "gov_shift_17_21_centered",
    "sen_shift_16_17_centered",
    "sen_shift_17_20_centered",
    "sen_shift_20_22_centered",
    "sen_shift_16_20_centered",
    "sen_shift_16_18_centered",
    "sen_shift_18_20_centered",
    "sen_shift_22_24_centered",
    "sen_shift_18_22_centered",
    "sen_shift_20_21_centered",
    "sen_shift_21_22_centered",
    "sen_shift_18_24_centered",
    "sen_shift_14_18_centered",
    "sen_shift_20_24_centered",
    "sen_shift_14_16_centered",
]
GOV_COLS = [c for c in ALL_OFFCYCLE_COLS if c.startswith("gov_")]
SEN_COLS = [c for c in ALL_OFFCYCLE_COLS if c.startswith("sen_")]

PRESIDENTIAL_WEIGHT = 8.0
J = 100
TEMPERATURE = 10.0
RANDOM_STATE = 42
N_INIT = 10


# ── Core computation functions (used by both main and tests) ─────────────────

def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare a shift matrix for clustering.

    Steps:
    1. Drop tracts where ALL training presidential columns are NaN (no signal)
    2. Fill remaining NaN with 0 (for state-centered columns, 0 = state mean)
    3. StandardScaler (before weighting — scale to unit variance first)
    4. Apply presidential weight=8.0 to pres columns (post-scaling)

    Returns
    -------
    X : ndarray of shape (N, D)
        Scaled, weighted shift matrix ready for PCA/KMeans.
    pres_train_mask : ndarray of shape (D,) bool
        True for columns that are presidential training columns (for weighting).
    holdout_vals : ndarray of shape (N,)
        The actual pres_shift_20_24 values for each retained tract.
    """
    # Drop tracts with no training presidential signal
    pres_train_in_features = [c for c in PRES_TRAIN_COLS if c in feature_cols]
    if pres_train_in_features:
        all_pres_nan = df[pres_train_in_features].isna().all(axis=1)
    else:
        # Config has no pres train cols — use all pres train cols to filter
        all_pres_nan = df[PRES_TRAIN_COLS].isna().all(axis=1)

    df_clean = df[~all_pres_nan].copy()

    # Extract feature matrix and fill NaN with 0
    # For state-centered off-cycle columns, NaN means no race in that state/cycle.
    # Filling with 0 means "type moves with its state mean" — a defensible default.
    X_raw = df_clean[feature_cols].fillna(0.0).values.astype(np.float64)

    # Scale to unit variance before weighting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Presidential weight post-scaling — amplify pres signal relative to off-cycle
    pres_train_mask = np.array(
        [c in PRES_TRAIN_COLS for c in feature_cols], dtype=bool
    )
    X_scaled[:, pres_train_mask] *= PRESIDENTIAL_WEIGHT

    holdout_vals = df_clean[PRES_HOLDOUT_COL].values

    return X_scaled, pres_train_mask, holdout_vals


def apply_pca_and_cluster(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA(whiten=True) then KMeans.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Scaled, weighted shift matrix.

    Returns
    -------
    soft_scores : ndarray of shape (N, J)
        Temperature-sharpened soft membership, rows sum to 1.
    labels : ndarray of shape (N,)
        Hard cluster assignments.
    n_pca_dims : int
        Number of PCA components used.
    """
    n_pca = min(15, X.shape[1])
    pca = PCA(n_components=n_pca, whiten=True, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    n_pca_dims = X_pca.shape[1]

    km = KMeans(n_clusters=J, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(X_pca)
    centroids = km.cluster_centers_  # (J, n_pca)

    # Euclidean distance from each tract to each centroid
    dists = np.linalg.norm(X_pca[:, None, :] - centroids[None, :, :], axis=2)  # (N, J)
    soft_scores = temperature_soft_membership(dists, T=TEMPERATURE)

    return soft_scores, labels, n_pca_dims


def compute_holdout_r(
    soft_scores: np.ndarray,
    holdout_vals: np.ndarray,
) -> float:
    """Predict held-out shifts via type-mean and compute Pearson r.

    For each type j, its mean is the weighted average of pres_shift_20_24 over all
    tracts, using soft membership as weights. Each tract's prediction is then
    the soft-membership-weighted sum of type means.

    This is the standard holdout evaluation used by the county model.

    Only tracts where pres_shift_20_24 is non-NaN contribute to the evaluation.
    The type means are computed only from tracts with observed holdout values,
    so tracts with NaN holdout don't distort the type means.

    Parameters
    ----------
    soft_scores : ndarray of shape (N, J)
    holdout_vals : ndarray of shape (N,)
        May contain NaN for tracts with no 2024 presidential data.

    Returns
    -------
    float
        Pearson r between predicted and actual pres_shift_20_24.
    """
    # Restrict to tracts with observed holdout
    has_holdout = ~np.isnan(holdout_vals)
    scores_h = soft_scores[has_holdout]   # (N_h, J)
    actual_h = holdout_vals[has_holdout]  # (N_h,)

    # Type means: weighted average of actual holdout values per type
    # type_mean_j = sum(score_{i,j} * actual_i) / sum(score_{i,j})
    weight_sums = scores_h.sum(axis=0)          # (J,)
    weighted_sums = (scores_h * actual_h[:, None]).sum(axis=0)  # (J,)
    # Avoid division by zero for types with zero total weight
    type_means = np.where(weight_sums > 0, weighted_sums / weight_sums, 0.0)  # (J,)

    # Tract predictions: soft-weighted sum of type means
    predicted_h = scores_h @ type_means  # (N_h,)

    r, _ = pearsonr(predicted_h, actual_h)
    return float(r)


def type_size_summary(labels: np.ndarray) -> dict:
    """Summarize type size distribution."""
    _, counts = np.unique(labels, return_counts=True)
    counts_sorted = sorted(counts.tolist(), reverse=True)
    return {
        "min": int(counts_sorted[-1]),
        "max": int(counts_sorted[0]),
        "median": int(np.median(counts_sorted)),
        "top5": counts_sorted[:5],
        "bottom5": counts_sorted[-5:],
    }


def compute_aggregated_offcycle(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean_gov_shift and mean_sen_shift columns.

    For each tract, average all available (non-NaN) governor shifts and
    all available senate shifts. Tracts with no off-cycle races of a given
    type get NaN, which is later filled to 0.

    Parameters
    ----------
    df : DataFrame with gov_* and sen_* columns

    Returns
    -------
    DataFrame with two new columns: mean_gov_shift, mean_sen_shift
    """
    df = df.copy()
    gov_cols_present = [c for c in GOV_COLS if c in df.columns]
    sen_cols_present = [c for c in SEN_COLS if c in df.columns]

    # nanmean across columns: ignores NaN, returns NaN if all NaN
    df["mean_gov_shift"] = df[gov_cols_present].mean(axis=1)
    df["mean_sen_shift"] = df[sen_cols_present].mean(axis=1)

    return df


# ── Per-config runners ────────────────────────────────────────────────────────

def run_config_a(df: pd.DataFrame) -> dict:
    """Config A: Presidential-only (4 dims, 3 train + holdout structure)."""
    # For Config A, we use pres_shift_08_12, _12_16, _16_20 as features.
    # The holdout is pres_shift_20_24 — not used in clustering, only in eval.
    feature_cols = PRES_TRAIN_COLS
    X, _, holdout_vals = prepare_matrix(df, feature_cols)
    soft_scores, labels, n_pca = apply_pca_and_cluster(X)
    r = compute_holdout_r(soft_scores, holdout_vals)
    return {
        "config": "A",
        "description": "Presidential-only (3 train pres dims)",
        "n_tracts": int(X.shape[0]),
        "n_dims_input": int(X.shape[1]),
        "n_pca_dims": n_pca,
        "holdout_r": round(r, 4),
        "type_sizes": type_size_summary(labels),
    }


def run_config_b(df: pd.DataFrame) -> dict:
    """Config B: Presidential + top-5 off-cycle (8 dims)."""
    feature_cols = PRES_TRAIN_COLS + TOP5_OFFCYCLE_COLS
    X, _, holdout_vals = prepare_matrix(df, feature_cols)
    soft_scores, labels, n_pca = apply_pca_and_cluster(X)
    r = compute_holdout_r(soft_scores, holdout_vals)
    return {
        "config": "B",
        "description": "Presidential + top-5 off-cycle by NaN rate",
        "n_tracts": int(X.shape[0]),
        "n_dims_input": int(X.shape[1]),
        "n_pca_dims": n_pca,
        "holdout_r": round(r, 4),
        "type_sizes": type_size_summary(labels),
        "offcycle_cols": TOP5_OFFCYCLE_COLS,
    }


def run_config_c(df: pd.DataFrame) -> dict:
    """Config C: Presidential + all 22 off-cycle (25 dims)."""
    feature_cols = PRES_TRAIN_COLS + ALL_OFFCYCLE_COLS
    X, _, holdout_vals = prepare_matrix(df, feature_cols)
    soft_scores, labels, n_pca = apply_pca_and_cluster(X)
    r = compute_holdout_r(soft_scores, holdout_vals)
    return {
        "config": "C",
        "description": "Presidential + all 22 off-cycle dims",
        "n_tracts": int(X.shape[0]),
        "n_dims_input": int(X.shape[1]),
        "n_pca_dims": n_pca,
        "holdout_r": round(r, 4),
        "type_sizes": type_size_summary(labels),
    }


def run_config_d(df: pd.DataFrame) -> dict:
    """Config D: Presidential + aggregated off-cycle (5 dims)."""
    df_aug = compute_aggregated_offcycle(df)
    feature_cols = PRES_TRAIN_COLS + ["mean_gov_shift", "mean_sen_shift"]
    X, _, holdout_vals = prepare_matrix(df_aug, feature_cols)
    soft_scores, labels, n_pca = apply_pca_and_cluster(X)
    r = compute_holdout_r(soft_scores, holdout_vals)
    return {
        "config": "D",
        "description": "Presidential + mean_gov_shift + mean_sen_shift",
        "n_tracts": int(X.shape[0]),
        "n_dims_input": int(X.shape[1]),
        "n_pca_dims": n_pca,
        "holdout_r": round(r, 4),
        "type_sizes": type_size_summary(labels),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table of all configs."""
    print()
    print("=" * 70)
    print("T3 Column Selection Experiment — Results")
    print("=" * 70)
    header = f"{'Config':<8} {'Description':<42} {'Tracts':>7} {'Dims':>5} {'PCA':>4} {'r':>6}"
    print(header)
    print("-" * 70)
    for r in results:
        row = (
            f"{r['config']:<8} "
            f"{r['description']:<42} "
            f"{r['n_tracts']:>7,} "
            f"{r['n_dims_input']:>5} "
            f"{r['n_pca_dims']:>4} "
            f"{r['holdout_r']:>6.4f}"
        )
        print(row)
    print("=" * 70)

    best = max(results, key=lambda x: x["holdout_r"])
    print(f"\nBest config: {best['config']} (holdout r={best['holdout_r']:.4f})")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading shift matrix from {SHIFTS_PATH}")
    df = pd.read_parquet(SHIFTS_PATH)
    print(f"Loaded {len(df):,} tracts x {len(df.columns)} columns")

    configs = [
        ("A", run_config_a),
        ("B", run_config_b),
        ("C", run_config_c),
        ("D", run_config_d),
    ]

    results = []
    for name, runner in configs:
        print(f"\nRunning Config {name}...")
        result = runner(df)
        results.append(result)
        print(
            f"  Config {name}: {result['n_tracts']:,} tracts, "
            f"{result['n_dims_input']} input dims, "
            f"{result['n_pca_dims']} PCA dims, "
            f"holdout r={result['holdout_r']:.4f}"
        )

    print_comparison_table(results)

    # Save results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
