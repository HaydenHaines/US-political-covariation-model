"""Experiment: Does enforcing a minimum type size improve LOO r?

Cross-election LOO analysis (S307) shows tiny types (n<10) dominate error.
This experiment tests merging small types into their nearest larger type
and measuring the impact on holdout LOO r.

Strategy: After KMeans, any type with fewer than N_min counties gets merged
into the nearest type (by centroid distance) that has >= N_min counties.
The merged counties get reassigned and soft membership is recomputed.

Usage:
    uv run python scripts/experiment_min_type_size.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import temperature_soft_membership

# Production config
J = 100
PRES_WEIGHT = 8.0
MIN_YEAR = 2008
PCA_N = 15
PCA_WHITEN = True
TEMPERATURE = 10.0
SEED = 42

HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]


def load_and_prepare():
    """Load shift matrix with production preprocessing."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet")
    all_cols = [c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS]

    shift_cols = []
    for c in all_cols:
        parts = c.split("_")
        y2 = int(parts[-2])
        start = y2 + (1900 if y2 >= 50 else 2000)
        if start >= MIN_YEAR:
            shift_cols.append(c)

    shift_raw = df[shift_cols].values.astype(float)
    holdout = df[HOLDOUT_COLUMNS].values.astype(float)

    scaler = StandardScaler()
    shift_scaled = scaler.fit_transform(shift_raw)
    pres_idx = [i for i, c in enumerate(shift_cols) if "pres_" in c]
    shift_scaled[:, pres_idx] *= PRES_WEIGHT

    pca = PCA(n_components=PCA_N, whiten=PCA_WHITEN, random_state=SEED)
    shift_pca = pca.fit_transform(shift_scaled)

    return shift_pca, shift_raw, holdout


def run_kmeans(shift_pca, j=J):
    """Run KMeans and return centroids + labels."""
    km = KMeans(n_clusters=j, random_state=SEED, n_init=10)
    labels = km.fit_predict(shift_pca)
    return km.cluster_centers_, labels


def merge_small_types(shift_pca, centroids, labels, n_min):
    """Merge types with fewer than n_min counties into nearest large type.

    Returns new labels and effective J (number of types after merging).
    """
    unique, counts = np.unique(labels, return_counts=True)
    type_sizes = dict(zip(unique, counts))

    # Identify small and large types
    large_types = {t for t, n in type_sizes.items() if n >= n_min}
    small_types = {t for t, n in type_sizes.items() if n < n_min}

    if not small_types:
        return labels.copy(), len(unique), centroids

    # Build mapping: small type → nearest large type (by centroid distance)
    merge_map = {}
    for st in small_types:
        min_dist = float("inf")
        best_lt = None
        for lt in large_types:
            dist = np.linalg.norm(centroids[st] - centroids[lt])
            if dist < min_dist:
                min_dist = dist
                best_lt = lt
        merge_map[st] = best_lt

    # Apply merges
    new_labels = labels.copy()
    for st, lt in merge_map.items():
        new_labels[new_labels == st] = lt

    # Renumber types to be contiguous (0, 1, ..., J_eff-1)
    unique_new = np.unique(new_labels)
    remap = {old: new for new, old in enumerate(unique_new)}
    new_labels = np.array([remap[l] for l in new_labels])

    # Recompute centroids for merged types
    j_eff = len(unique_new)
    new_centroids = np.zeros((j_eff, shift_pca.shape[1]))
    for t in range(j_eff):
        mask = new_labels == t
        new_centroids[t] = shift_pca[mask].mean(axis=0)

    return new_labels, j_eff, new_centroids


def compute_soft_scores(shift_pca, centroids, j_eff):
    """Compute soft membership scores from centroids."""
    N = shift_pca.shape[0]
    dists = np.zeros((N, j_eff))
    for t in range(j_eff):
        dists[:, t] = np.linalg.norm(shift_pca - centroids[t], axis=1)
    return temperature_soft_membership(dists, T=TEMPERATURE)


def compute_loo_r(weights, holdout, shift_raw):
    """LOO type-mean holdout r (same formula as other experiments)."""
    N, J_ = weights.shape
    county_means = shift_raw.mean(axis=1)
    global_ws = weights.sum(axis=0)
    global_wt = weights.T @ county_means

    rs = []
    for h in range(holdout.shape[1]):
        actual = holdout[:, h]
        global_wh = weights.T @ actual
        predicted = np.zeros(N)
        for i in range(N):
            loo_ws = global_ws - weights[i]
            loo_ws = np.where(loo_ws < 1e-12, 1e-12, loo_ws)
            loo_train = (global_wt - weights[i] * county_means[i]) / loo_ws
            loo_hold = (global_wh - weights[i] * actual[i]) / loo_ws
            predicted[i] = county_means[i] + (weights[i] * (loo_hold - loo_train)).sum()
        r, _ = pearsonr(actual, predicted)
        rs.append(float(r))
    return float(np.mean(rs))


def main():
    print("=" * 70)
    print("Minimum Type Size Enforcement Experiment")
    print("=" * 70)

    shift_pca, shift_raw, holdout = load_and_prepare()
    centroids, labels = run_kmeans(shift_pca)
    N = len(labels)

    # Report baseline type size distribution
    _, counts = np.unique(labels, return_counts=True)
    print(f"\nBaseline: J={J}, N={N}")
    print(f"Type sizes: min={counts.min()}, max={counts.max()}, "
          f"median={np.median(counts):.0f}, mean={counts.mean():.1f}")
    print(f"Types with n<5: {sum(counts < 5)}, n<10: {sum(counts < 10)}, "
          f"n<15: {sum(counts < 15)}, n<20: {sum(counts < 20)}")

    # Test different N_min thresholds
    thresholds = [0, 3, 5, 8, 10, 15, 20, 30]
    results = []

    for n_min in thresholds:
        t0 = time.time()
        if n_min == 0:
            # Baseline: no merging
            scores = compute_soft_scores(shift_pca, centroids, J)
            j_eff = J
            n_merged = 0
        else:
            new_labels, j_eff, new_centroids = merge_small_types(
                shift_pca, centroids, labels, n_min
            )
            scores = compute_soft_scores(shift_pca, new_centroids, j_eff)
            n_merged = J - j_eff

        loo_r = compute_loo_r(scores, holdout, shift_raw)
        elapsed = time.time() - t0

        # Type size stats after merging
        hard_labels = np.argmax(scores, axis=1)
        _, new_counts = np.unique(hard_labels, return_counts=True)

        results.append({
            "n_min": n_min,
            "j_eff": j_eff,
            "n_merged": n_merged,
            "loo_r": loo_r,
            "min_size": int(new_counts.min()),
            "max_size": int(new_counts.max()),
            "elapsed": elapsed,
        })
        print(f"  n_min={n_min:>3}: J_eff={j_eff:>3}, merged={n_merged:>2}, "
              f"LOO r={loo_r:.4f}, sizes=[{new_counts.min()},{new_counts.max()}] ({elapsed:.1f}s)")

    # Results table
    baseline_loo = results[0]["loo_r"]
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'N_min':>5} | {'J_eff':>5} | {'Merged':>6} | {'LOO r':>7} | {'Δ':>7} | {'Min size':>8} | {'Max size':>8}")
    print("-" * 65)
    for r in results:
        delta = r["loo_r"] - baseline_loo
        marker = " <<<" if r == max(results, key=lambda x: x["loo_r"]) else ""
        print(f"{r['n_min']:>5} | {r['j_eff']:>5} | {r['n_merged']:>6} | "
              f"{r['loo_r']:>7.4f} | {delta:>+7.4f} | {r['min_size']:>8} | {r['max_size']:>8}{marker}")

    best = max(results, key=lambda x: x["loo_r"])
    print(f"\nBest: n_min={best['n_min']}, LOO r={best['loo_r']:.4f} "
          f"(Δ={best['loo_r'] - baseline_loo:+.4f} vs baseline)")

    if best["loo_r"] - baseline_loo > 0.003:
        print("RECOMMENDATION: Enforce minimum type size in production.")
    else:
        print("FINDING: Minimum type size enforcement has negligible impact on LOO r.")
        print("Small types are noisy but don't hurt aggregate prediction quality.")


if __name__ == "__main__":
    main()
