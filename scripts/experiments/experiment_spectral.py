"""Spectral clustering experiment (P3.3).

Tests whether spectral clustering finds non-convex clusters that KMeans misses,
and compares holdout r using the same county-prior method as production validation.

Spectral clustering constructs a k-NN affinity graph and partitions the graph's
eigenvectors rather than the raw feature space. This can discover elongated or
manifold-shaped clusters that KMeans' Voronoi-cell geometry misses.

Usage:
    uv run python scripts/experiment_spectral.py

Results are written to:
    data/validation/spectral_experiment_results.parquet
    docs/spectral-experiment-S175.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans, SpectralClustering

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants — must exactly match production pipeline
# ---------------------------------------------------------------------------

HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

PRES_WEIGHT = 2.5
MIN_YEAR = 2008
KMEANS_SEED = 42
KMEANS_N_INIT = 10
TEMPERATURE = 10.0

# Spectral clustering config
SPECTRAL_N_NEIGHBORS = 10
SPECTRAL_SEED = 42

# J values to sweep
J_SWEEP = [20, 30, 40, 43, 50]
PRODUCTION_J = 43

# Baseline (KMeans J=43, county-prior holdout r from CLAUDE.md)
BASELINE_R = 0.828


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    min_year: int = MIN_YEAR,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load shift matrix and build the weighted training matrix.

    Returns
    -------
    X_weighted : ndarray of shape (N, D)
        Presidential×2.5 weighted training matrix (fed to clustering).
    X_raw_training : ndarray of shape (N, D)
        Unweighted training matrix (used for county-prior baseline computation).
    training_cols : list[str]
        Names of training columns (2008+, excluding holdout).
    holdout_cols : list[str]
        Names of holdout columns (2020→2024 presidential).
    """
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    all_cols = [c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS]

    # Filter to >= min_year (parse start year from column suffix: _08_12 → 2008)
    training_cols = []
    for c in all_cols:
        parts = c.split("_")
        try:
            y2d = parts[-2]  # e.g. "08" from "pres_d_shift_08_12"
            if len(y2d) == 2 and y2d.isdigit():
                y = int(y2d)
                year = 2000 + y if y < 50 else 1900 + y
                if year >= min_year:
                    training_cols.append(c)
            else:
                training_cols.append(c)  # keep if unparseable
        except (IndexError, ValueError):
            training_cols.append(c)

    holdout_cols = [c for c in HOLDOUT_COLUMNS if c in df.columns]

    # Build weighted matrix (presidential × 2.5)
    X_raw = df[training_cols].values.copy()
    X_weighted = X_raw.copy()
    for i, col in enumerate(training_cols):
        if col.startswith("pres_"):
            X_weighted[:, i] *= PRES_WEIGHT

    return X_weighted, X_raw, training_cols, holdout_cols


def get_holdout_matrix(holdout_cols: list[str]) -> np.ndarray:
    """Load raw holdout shift values (not weighted)."""
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)
    return df[holdout_cols].values


# ---------------------------------------------------------------------------
# Soft membership
# ---------------------------------------------------------------------------


def temperature_soft_membership(dists: np.ndarray, T: float = TEMPERATURE) -> np.ndarray:
    """Temperature-scaled inverse-distance soft membership (matches production).

    Parameters
    ----------
    dists : ndarray of shape (N, J)
        Euclidean distances to centroids.
    T : float
        Temperature exponent.

    Returns
    -------
    scores : ndarray of shape (N, J)
        Non-negative weights summing to 1 per row.
    """
    if T >= 500.0:
        scores = np.zeros_like(dists)
        scores[np.arange(len(dists)), dists.argmin(axis=1)] = 1.0
        return scores

    eps = 1e-10
    log_w = -T * np.log(dists + eps)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return w / row_sums


def spectral_soft_membership(
    X: np.ndarray,
    labels: np.ndarray,
    J: int,
) -> np.ndarray:
    """Build temperature-scaled soft membership from spectral labels via centroids.

    Spectral clustering produces hard labels. We compute centroids in the
    original feature space and then apply the same T=10 inverse-distance
    soft membership used by KMeans.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Feature matrix (weighted).
    labels : ndarray of shape (N,)
        Hard cluster labels from SpectralClustering.
    J : int
        Number of clusters.

    Returns
    -------
    scores : ndarray of shape (N, J)
        Soft membership weights.
    """
    centroids = np.array([X[labels == j].mean(axis=0) if (labels == j).sum() > 0
                          else np.zeros(X.shape[1])
                          for j in range(J)])
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (N, J)
    return temperature_soft_membership(dists, T=TEMPERATURE)


# ---------------------------------------------------------------------------
# Holdout accuracy — county-prior method (matches validate_types.py)
# ---------------------------------------------------------------------------


def compute_holdout_r_county_prior(
    scores: np.ndarray,
    raw_training_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
) -> dict:
    """Holdout Pearson r using county-level priors + type covariance adjustment.

    Replicates holdout_accuracy_county_prior() from src/validation/validate_types.py
    for self-contained use in this experiment.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        Soft membership scores.
    raw_training_matrix : ndarray of shape (N, D_train)
        Unweighted training shifts (each county's own history).
    holdout_matrix : ndarray of shape (N, D_holdout)
        Actual holdout shifts.

    Returns
    -------
    dict with keys:
        mean_r     float  — mean Pearson r across holdout dims
        per_dim_r  list[float]
        mean_rmse  float
        per_dim_rmse list[float]
    """
    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums  # (N, J)

    weight_sums_per_type = weights.sum(axis=0)  # (J,)
    weight_sums_per_type = np.where(weight_sums_per_type == 0, 1.0, weight_sums_per_type)

    # County baseline: each county's own historical mean
    county_training_means = raw_training_matrix.mean(axis=1)  # (N,)

    # Type-level training mean
    type_training_means = (weights.T @ county_training_means) / weight_sums_per_type  # (J,)

    per_dim_r: list[float] = []
    per_dim_rmse: list[float] = []

    for d in range(holdout_matrix.shape[1]):
        actual = holdout_matrix[:, d]

        # Type-level holdout mean
        type_holdout_means = (weights.T @ actual) / weight_sums_per_type  # (J,)

        # Type adjustment = how much each type moved from training→holdout
        type_adj = type_holdout_means - type_training_means  # (J,)

        # County prediction = own prior + weighted type adjustment
        county_adj = (weights * type_adj[None, :]).sum(axis=1)  # (N,)
        predicted = county_training_means + county_adj

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        per_dim_rmse.append(rmse)

    return {
        "mean_r": float(np.mean(per_dim_r)) if per_dim_r else 0.0,
        "per_dim_r": per_dim_r,
        "mean_rmse": float(np.mean(per_dim_rmse)) if per_dim_rmse else 0.0,
        "per_dim_rmse": per_dim_rmse,
    }


# ---------------------------------------------------------------------------
# Clustering runners
# ---------------------------------------------------------------------------


def run_kmeans(X: np.ndarray, J: int) -> np.ndarray:
    """Run production-equivalent KMeans and return T=10 soft scores."""
    km = KMeans(n_clusters=J, random_state=KMEANS_SEED, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    dists = np.zeros((len(X), J))
    for j in range(J):
        dists[:, j] = np.linalg.norm(X - centroids[j], axis=1)

    return temperature_soft_membership(dists, T=TEMPERATURE), labels


def run_spectral(
    X: np.ndarray,
    J: int,
    n_neighbors: int = SPECTRAL_N_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SpectralClustering with k-NN affinity and return soft scores + labels.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Weighted feature matrix.
    J : int
        Number of clusters.
    n_neighbors : int
        k for k-nearest-neighbors affinity graph construction.

    Returns
    -------
    scores : ndarray of shape (N, J)
        Temperature-scaled soft membership (centroids in original space).
    labels : ndarray of shape (N,)
        Hard cluster labels (0..J-1).
    """
    sc = SpectralClustering(
        n_clusters=J,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=SPECTRAL_SEED,
        assign_labels="kmeans",
        n_jobs=1,
    )
    labels = sc.fit_predict(X)

    # Relabel to 0..J-1 in case SpectralClustering skips indices
    unique_labels = np.unique(labels)
    if len(unique_labels) < J:
        # Some clusters empty — remap to contiguous
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        J_actual = len(unique_labels)
    else:
        J_actual = J

    scores = spectral_soft_membership(X, labels, J_actual)
    return scores, labels


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> list[dict]:
    """Run the spectral vs KMeans comparison across J values.

    Returns
    -------
    list of result dicts with keys:
        method, J, holdout_r, mean_rmse, wall_seconds
    """
    print("=" * 72)
    print("Spectral Clustering Experiment (P3.3)")
    print("=" * 72)
    print(f"\nBaseline: KMeans J={PRODUCTION_J}, holdout r = {BASELINE_R:.3f} (county-prior)")
    print(f"Affinity: nearest_neighbors, n_neighbors={SPECTRAL_N_NEIGHBORS}")
    print(f"Soft membership: T={TEMPERATURE} inverse-distance (same as production)\n")

    X_weighted, X_raw, training_cols, holdout_cols = load_data()
    holdout_matrix = get_holdout_matrix(holdout_cols)

    N, D = X_weighted.shape
    print(f"Data: {N} counties x {D} training dims (presidential×{PRES_WEIGHT} weighted)")
    print(f"Holdout: {len(holdout_cols)} dims ({', '.join(holdout_cols)})\n")

    results = []

    for J in J_SWEEP:
        print(f"--- J = {J} ---")

        # KMeans baseline
        t0 = time.perf_counter()
        scores_km, labels_km = run_kmeans(X_weighted, J)
        km_time = time.perf_counter() - t0
        km_stats = compute_holdout_r_county_prior(scores_km, X_raw, holdout_matrix)
        print(f"  KMeans:   holdout_r={km_stats['mean_r']:.4f}  "
              f"rmse={km_stats['mean_rmse']:.4f}  "
              f"t={km_time:.1f}s")
        results.append({
            "method": "KMeans",
            "J": J,
            "holdout_r": km_stats["mean_r"],
            "mean_rmse": km_stats["mean_rmse"],
            "per_dim_r": km_stats["per_dim_r"],
            "wall_seconds": km_time,
        })

        # Spectral clustering
        t0 = time.perf_counter()
        try:
            scores_sc, labels_sc = run_spectral(X_weighted, J)
            sc_time = time.perf_counter() - t0
            actual_j = len(np.unique(labels_sc))
            sc_stats = compute_holdout_r_county_prior(scores_sc, X_raw, holdout_matrix)
            print(f"  Spectral: holdout_r={sc_stats['mean_r']:.4f}  "
                  f"rmse={sc_stats['mean_rmse']:.4f}  "
                  f"t={sc_time:.1f}s  "
                  f"clusters_found={actual_j}")
            results.append({
                "method": "Spectral",
                "J": J,
                "holdout_r": sc_stats["mean_r"],
                "mean_rmse": sc_stats["mean_rmse"],
                "per_dim_r": sc_stats["per_dim_r"],
                "wall_seconds": sc_time,
                "clusters_found": actual_j,
            })
        except Exception as e:
            sc_time = time.perf_counter() - t0
            print(f"  Spectral: FAILED ({e})  t={sc_time:.1f}s")
            results.append({
                "method": "Spectral",
                "J": J,
                "holdout_r": float("nan"),
                "mean_rmse": float("nan"),
                "per_dim_r": [],
                "wall_seconds": sc_time,
                "error": str(e),
            })

    return results


def print_summary(results: list[dict]) -> None:
    """Print summary comparison table."""
    print("\n" + "=" * 72)
    print("Summary — Holdout r (county-prior method)")
    print("=" * 72)
    df = pd.DataFrame(results)

    for method in ["KMeans", "Spectral"]:
        sub = df[df["method"] == method][["J", "holdout_r", "mean_rmse"]].set_index("J")
        print(f"\n{method}:")
        print(sub.to_string(float_format="%.4f"))

    # Delta at each J
    print("\nDelta (Spectral - KMeans), county-prior holdout r:")
    km_r = {r["J"]: r["holdout_r"] for r in results if r["method"] == "KMeans"}
    sc_r = {r["J"]: r["holdout_r"] for r in results if r["method"] == "Spectral"}
    for J in J_SWEEP:
        if J in km_r and J in sc_r and not (
            pd.isna(km_r[J]) or pd.isna(sc_r[J])
        ):
            delta = sc_r[J] - km_r[J]
            marker = " *** SPECTRAL WINS" if delta > 0.002 else (" (tie)" if abs(delta) <= 0.002 else "")
            print(f"  J={J:3d}: {delta:+.4f}{marker}")

    # Best Spectral J
    sc_results = [r for r in results if r["method"] == "Spectral" and not pd.isna(r["holdout_r"])]
    if sc_results:
        best = max(sc_results, key=lambda r: r["holdout_r"])
        print(f"\nBest Spectral: J={best['J']}, holdout_r={best['holdout_r']:.4f}")
        print(f"Production baseline (KMeans J=43): holdout_r={BASELINE_R:.3f}")
        if best["holdout_r"] > BASELINE_R:
            print("=> Spectral BEATS production baseline")
        else:
            print(f"=> Spectral MISSES baseline by {BASELINE_R - best['holdout_r']:.4f}")


def save_results(results: list[dict]) -> None:
    """Save result records to parquet (excluding list columns)."""
    out_dir = PROJECT_ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tabular summary (drop list columns for parquet)
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, list)}
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = out_dir / "spectral_experiment_results.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")


def write_markdown_report(results: list[dict]) -> None:
    """Write experiment summary to docs/spectral-experiment-S175.md."""
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = docs_dir / "spectral-experiment-S175.md"

    km_results = {r["J"]: r for r in results if r["method"] == "KMeans"}
    sc_results_map = {r["J"]: r for r in results if r["method"] == "Spectral"}
    sc_valid = [r for r in results if r["method"] == "Spectral" and not pd.isna(r["holdout_r"])]
    best_sc = max(sc_valid, key=lambda r: r["holdout_r"]) if sc_valid else None

    lines = [
        "# Spectral Clustering Experiment (P3.3)",
        "",
        "**Date:** 2026-03-23  ",
        "**Session:** S175  ",
        "**Branch:** feat/spectral-clustering-experiment",
        "",
        "## Objective",
        "",
        "Test whether spectral clustering (k-NN affinity) finds non-convex clusters",
        "that KMeans misses, and compare holdout r on the county-prior prediction task.",
        "",
        "## Setup",
        "",
        f"- **Data:** 293 counties (FL+GA+AL), {len([c for c in __import__('pandas').read_parquet(PROJECT_ROOT / 'data' / 'shifts' / 'county_shifts_multiyear.parquet').columns if c != 'county_fips' and c not in HOLDOUT_COLUMNS and _is_training_col(c)])} training dims (2008+, presidential×{PRES_WEIGHT})",
        "- **Holdout:** 2020→2024 presidential shifts (pres_d, pres_r, pres_turnout)",
        "- **KMeans:** n_init=10, random_state=42, T=10 inverse-distance soft membership",
        f"- **Spectral:** affinity=nearest_neighbors, n_neighbors={SPECTRAL_N_NEIGHBORS}, assign_labels=kmeans",
        f"- **Soft membership:** both methods use T={TEMPERATURE} inverse-distance (distance to centroids in original space)",
        f"- **J sweep:** {J_SWEEP}",
        f"- **Metric:** county-prior holdout r (same as production validation)",
        f"- **Production baseline:** KMeans J={PRODUCTION_J}, holdout r={BASELINE_R:.3f}",
        "",
        "## Results",
        "",
        "### Holdout r — county-prior method",
        "",
        "| J | KMeans r | Spectral r | Delta | Winner |",
        "|---|----------|------------|-------|--------|",
    ]

    for J in J_SWEEP:
        km = km_results.get(J)
        sc = sc_results_map.get(J)
        km_r = f"{km['holdout_r']:.4f}" if km and not pd.isna(km["holdout_r"]) else "N/A"
        sc_r_val = sc["holdout_r"] if sc and not pd.isna(sc.get("holdout_r", float("nan"))) else None
        sc_r = f"{sc_r_val:.4f}" if sc_r_val is not None else "N/A"
        if km and sc_r_val is not None and not pd.isna(km["holdout_r"]):
            delta = sc_r_val - km["holdout_r"]
            delta_str = f"{delta:+.4f}"
            winner = "Spectral" if delta > 0.002 else ("KMeans" if delta < -0.002 else "Tie")
        else:
            delta_str = "N/A"
            winner = "N/A"
        lines.append(f"| {J} | {km_r} | {sc_r} | {delta_str} | {winner} |")

    lines += [
        "",
        "### RMSE — county-prior method",
        "",
        "| J | KMeans RMSE | Spectral RMSE |",
        "|---|-------------|---------------|",
    ]
    for J in J_SWEEP:
        km = km_results.get(J)
        sc = sc_results_map.get(J)
        km_rmse = f"{km['mean_rmse']:.4f}" if km and not pd.isna(km.get("mean_rmse", float("nan"))) else "N/A"
        sc_rmse_val = sc.get("mean_rmse") if sc else None
        sc_rmse = f"{sc_rmse_val:.4f}" if sc_rmse_val is not None and not pd.isna(sc_rmse_val) else "N/A"
        lines.append(f"| {J} | {km_rmse} | {sc_rmse} |")

    lines += [""]

    if best_sc:
        beat = best_sc["holdout_r"] > BASELINE_R
        lines += [
            "## Conclusion",
            "",
            f"Best spectral result: **J={best_sc['J']}, holdout r={best_sc['holdout_r']:.4f}**",
            f"Production KMeans baseline: holdout r={BASELINE_R:.3f}",
            "",
        ]
        if beat:
            gain = best_sc["holdout_r"] - BASELINE_R
            lines += [
                f"**Spectral beats production baseline (KMeans J=43 r={BASELINE_R:.3f}) by {gain:.4f} at J={best_sc['J']}.**",
                "",
                "Note: KMeans at the same J may still be competitive — see delta column above.",
                "Consider running a more thorough J sweep before promoting spectral to production.",
            ]
        else:
            gap = BASELINE_R - best_sc["holdout_r"]
            lines += [
                f"**KMeans holds.** Spectral misses the production baseline by {gap:.4f}.",
                "",
                "Interpretation: The county shift space at this sample size (N=293) is",
                "likely well-described by convex Voronoi cells. Spectral clustering's",
                "k-NN graph may add complexity without capturing genuine non-convex",
                "structure at this resolution.",
                "",
                "Spectral remains potentially useful at finer geographic resolution",
                "(tract-level) where non-convex geographic clusters are more plausible.",
            ]
    else:
        lines += [
            "## Conclusion",
            "",
            "Spectral clustering failed for all J values tested. See error messages above.",
        ]

    lines += [
        "",
        "## Method Notes",
        "",
        "Spectral soft membership is computed in the **original feature space** (not",
        "the spectral embedding space). This matches how KMeans computes distances.",
        "An alternative would compute distances in the 2D eigenspace, but the",
        "county-prior prediction task requires soft weights anchored to cluster",
        "centers in the original shift space.",
        "",
        "A future experiment could test spectral soft membership computed in the",
        "normalized Laplacian eigenvector space.",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {out_path}")


def _is_training_col(col: str, min_year: int = MIN_YEAR) -> bool:
    """Check if a column is a training column (≥ min_year, not holdout)."""
    if col in HOLDOUT_COLUMNS:
        return False
    parts = col.split("_")
    try:
        y2d = parts[-2]
        if len(y2d) == 2 and y2d.isdigit():
            y = int(y2d)
            year = 2000 + y if y < 50 else 1900 + y
            return year >= min_year
    except (IndexError, ValueError):
        pass
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    results = run_experiment()
    print_summary(results)
    save_results(results)
    try:
        write_markdown_report(results)
    except Exception as e:
        print(f"Warning: markdown report generation failed: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
