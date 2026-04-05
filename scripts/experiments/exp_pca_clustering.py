"""PCA dimensionality reduction experiment for type discovery (issue #93 / P3.6).

Research question: Does applying PCA before KMeans improve holdout r by reducing
noise dimensions? Presidential shift columns are highly correlated across election
cycles, so PCA may concentrate signal before clustering.

Design:
  - Loads the same training data as run_type_discovery.py (min_year=2008)
  - Applies StandardScaler + presidential_weight=8.0 (post-scaling, per production)
  - Sweeps PCA n_components from 5 to 25
  - For each n_components: KMeans J=100, compute holdout r (type-mean prior)
  - Also runs UMAP as a nonlinear alternative if umap-learn is available
  - Compares all variants against the baseline: no PCA, J=100, holdout r=0.698

Holdout methodology (standard, not LOO):
  Holdout columns = pres_d_shift_20_24, pres_r_shift_20_24, pres_turnout_shift_20_24.
  Prediction = each county's score-weighted type mean on holdout columns.
  NOTE: Standard holdout r inflates by ~0.22 vs LOO r (see S196 LOO honesty rule).
  The LOO metric is the honest generalization estimate; standard is used here to
  stay comparable to the published baseline of 0.698.

Usage:
    uv run python scripts/exp_pca_clustering.py
    uv run python scripts/exp_pca_clustering.py --j 100 --pca-max 30 --umap
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants — must match production pipeline in run_type_discovery.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments"

KMEANS_SEED = 42
KMEANS_N_INIT = 10
PRES_WEIGHT = 8.0
MIN_YEAR = 2008
TEMPERATURE = 10.0
J = 100

# The 2020→2024 pair is the blind holdout — excluded from training, used only
# to measure how well each clustering generalises to unseen electoral shifts.
HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_shift_matrix(
    min_year: int = MIN_YEAR,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load county shift matrix and apply production preprocessing.

    Replicates the exact preprocessing from run_type_discovery.py:
    1. Load all shift columns (excluding county_fips and holdout columns).
    2. Filter to pairs with start year >= min_year (default 2008).
    3. Apply StandardScaler (zero mean, unit variance per column).
    4. Multiply presidential shift columns by PRES_WEIGHT (post-scaling).

    Returns
    -------
    train_matrix : ndarray (N, D_train)
        Scaled + weighted shift matrix for clustering (excludes holdout).
    holdout_matrix : ndarray (N, D_holdout)
        Raw (unscaled) holdout columns for accuracy measurement.
    train_cols : list[str]
        Column names for train_matrix dimensions.
    holdout_cols : list[str]
        Column names for holdout_matrix dimensions.
    """
    df = pd.read_parquet(SHIFTS_PATH)

    # Separate all shift columns from FIPS identifier
    all_shift_cols = [c for c in df.columns if c != "county_fips"]

    # Filter training columns to recent pairs and exclude holdout
    train_cols = []
    for col in all_shift_cols:
        if col in HOLDOUT_COLUMNS:
            continue
        # Extract the start year from column suffix like "pres_d_shift_08_12" → 08 → 2008
        parts = col.split("_")
        # Last two parts are the year pair, e.g. "08" and "12"
        y2_str = parts[-2]
        y2_int = int(y2_str)
        start_year = y2_int + (1900 if y2_int >= 50 else 2000)
        if start_year >= min_year:
            train_cols.append(col)

    train_matrix = df[train_cols].values.astype(float)
    holdout_matrix = df[HOLDOUT_COLUMNS].values.astype(float)

    # StandardScaler: zero mean, unit variance
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_matrix)

    # Post-scaling presidential weight — amplifies cross-state presidential
    # covariation signal relative to off-cycle races
    pres_indices = [i for i, c in enumerate(train_cols) if "pres_" in c]
    if PRES_WEIGHT != 1.0:
        train_matrix[:, pres_indices] *= PRES_WEIGHT

    n_counties, n_dims = train_matrix.shape
    n_pres = len(pres_indices)
    print(f"Loaded {n_counties} counties × {n_dims} training dims "
          f"({n_pres} presidential, {n_dims - n_pres} off-cycle; min_year={min_year})")

    return train_matrix, holdout_matrix, train_cols, HOLDOUT_COLUMNS


# ---------------------------------------------------------------------------
# Soft membership (temperature-scaled inverse distance)
# ---------------------------------------------------------------------------


def temperature_soft_membership(dists: np.ndarray, T: float = TEMPERATURE) -> np.ndarray:
    """Compute temperature-sharpened soft membership from centroid distances.

    Replicates src/discovery/run_type_discovery.py::temperature_soft_membership.
    Formula: weight_j = (1 / (dist_j + eps))^T, then row-normalise.
    Uses log-space arithmetic for numerical stability at T=10.
    """
    N, J_ = dists.shape
    eps = 1e-10

    if T >= 500.0:
        # Hard assignment limit
        scores = np.zeros((N, J_))
        scores[np.arange(N), np.argmin(dists, axis=1)] = 1.0
        return scores

    log_weights = -T * np.log(dists + eps)
    log_weights -= log_weights.max(axis=1, keepdims=True)  # numerical stability
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    return powered / np.where(row_sums == 0, 1.0, row_sums)


# ---------------------------------------------------------------------------
# Holdout accuracy (type-mean prior)
# ---------------------------------------------------------------------------


def compute_holdout_r(
    train_data: np.ndarray,
    holdout_matrix: np.ndarray,
    j: int = J,
    random_state: int = KMEANS_SEED,
    temperature: float = TEMPERATURE,
) -> dict:
    """Run KMeans on train_data, measure holdout r against holdout_matrix.

    Standard holdout r (not LOO). NOTE: inflates vs honest LOO by ~0.22 —
    use only for comparison against the same-method baseline of 0.698.

    Parameters
    ----------
    train_data : ndarray (N, D)
        Pre-processed feature matrix (already scaled, weighted, possibly PCA'd).
    holdout_matrix : ndarray (N, D_holdout)
        Raw holdout shift columns. KMeans sees none of this.
    j : int
        Number of clusters.
    random_state : int
        KMeans seed.
    temperature : float
        Soft membership temperature exponent.

    Returns
    -------
    dict with keys:
        "mean_r"      -- mean Pearson r across holdout columns
        "per_col_r"   -- Pearson r for each holdout column
        "n_types"     -- j (for verification)
        "n_counties"  -- N
    """
    # Fit KMeans
    km = KMeans(n_clusters=j, random_state=random_state, n_init=KMEANS_N_INIT)
    km.fit(train_data)
    centroids = km.cluster_centers_  # (J, D)

    # Compute soft membership via temperature-scaled inverse distance
    dists = np.zeros((len(train_data), j))
    for t in range(j):
        dists[:, t] = np.linalg.norm(train_data - centroids[t], axis=1)
    weights = temperature_soft_membership(dists, T=temperature)  # (N, J)

    # Type-level weighted mean of holdout shifts
    weight_sums = weights.sum(axis=0)  # (J,)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    type_means = (weights.T @ holdout_matrix) / weight_sums[:, None]  # (J, D_holdout)

    # County predictions: weighted sum of type means
    predicted = weights @ type_means  # (N, D_holdout)

    # Pearson r per holdout column
    per_col_r = []
    for col_idx in range(holdout_matrix.shape[1]):
        actual = holdout_matrix[:, col_idx]
        pred = predicted[:, col_idx]
        if np.std(actual) < 1e-10 or np.std(pred) < 1e-10:
            per_col_r.append(0.0)
        else:
            r, _ = pearsonr(actual, pred)
            per_col_r.append(float(np.clip(r, -1.0, 1.0)))

    return {
        "mean_r": float(np.mean(per_col_r)),
        "per_col_r": per_col_r,
        "n_types": j,
        "n_counties": len(train_data),
    }


# ---------------------------------------------------------------------------
# PCA sweep
# ---------------------------------------------------------------------------


def run_pca_sweep(
    train_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
    n_components_range: range,
    j: int = J,
) -> pd.DataFrame:
    """Sweep PCA n_components, run KMeans for each, report holdout r.

    Parameters
    ----------
    train_matrix : ndarray (N, D)
        Pre-scaled shift matrix.
    holdout_matrix : ndarray (N, D_holdout)
        Raw holdout columns.
    n_components_range : range
        PCA component counts to try.
    j : int
        KMeans cluster count.

    Returns
    -------
    pd.DataFrame with one row per n_components setting.
    """
    rows = []
    total = len(n_components_range)
    n_total_dims = train_matrix.shape[1]

    for idx, n_comp in enumerate(n_components_range, 1):
        if n_comp >= n_total_dims:
            print(f"  [{idx}/{total}] PCA n_comp={n_comp} >= total dims {n_total_dims} — skipping")
            continue

        t0 = time.time()

        # Apply PCA
        pca = PCA(n_components=n_comp, random_state=KMEANS_SEED)
        reduced = pca.fit_transform(train_matrix)
        var_explained = pca.explained_variance_ratio_
        cumvar = float(var_explained.cumsum()[-1])

        # Run KMeans + evaluate
        result = compute_holdout_r(reduced, holdout_matrix, j=j)

        elapsed = time.time() - t0
        print(f"  [{idx}/{total}] PCA n_comp={n_comp:2d} | "
              f"var_explained={cumvar:.3f} | "
              f"holdout r={result['mean_r']:.4f} | "
              f"per-col r={[f'{r:.3f}' for r in result['per_col_r']]} | "
              f"{elapsed:.1f}s")

        rows.append({
            "method": "PCA",
            "n_components": n_comp,
            "cumulative_var_explained": cumvar,
            "top1_var": float(var_explained[0]),
            "top5_var": float(var_explained[:5].sum()) if len(var_explained) >= 5 else cumvar,
            "holdout_r": result["mean_r"],
            "pres_d_r": result["per_col_r"][0],
            "pres_r_r": result["per_col_r"][1],
            "turnout_r": result["per_col_r"][2],
            "j": j,
            "elapsed_s": elapsed,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UMAP experiment
# ---------------------------------------------------------------------------


def run_umap_sweep(
    train_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
    n_components_range: list[int],
    j: int = J,
) -> pd.DataFrame:
    """Run UMAP dimensionality reduction + KMeans for each n_components.

    UMAP is nonlinear and can separate non-convex clusters that PCA misses,
    but is sensitive to hyperparameters (n_neighbors, min_dist). We sweep
    component counts with fixed neighbours=15, min_dist=0.1 (UMAP defaults).

    NOTE: UMAP with random state is not fully deterministic — results may vary
    slightly between runs. Treat as an order-of-magnitude comparison only.

    Returns
    -------
    pd.DataFrame or empty DataFrame if umap-learn is not available.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        print("  umap-learn not available — skipping UMAP experiment")
        return pd.DataFrame()

    rows = []
    total = len(n_components_range)

    for idx, n_comp in enumerate(n_components_range, 1):
        t0 = time.time()

        reducer = umap.UMAP(
            n_components=n_comp,
            n_neighbors=15,
            min_dist=0.1,
            random_state=KMEANS_SEED,
            verbose=False,
        )
        reduced = reducer.fit_transform(train_matrix)

        # UMAP doesn't give explained variance — report n_comp only
        result = compute_holdout_r(reduced, holdout_matrix, j=j)

        elapsed = time.time() - t0
        print(f"  [{idx}/{total}] UMAP n_comp={n_comp:2d} | "
              f"holdout r={result['mean_r']:.4f} | "
              f"per-col r={[f'{r:.3f}' for r in result['per_col_r']]} | "
              f"{elapsed:.1f}s")

        rows.append({
            "method": "UMAP",
            "n_components": n_comp,
            "cumulative_var_explained": None,
            "top1_var": None,
            "top5_var": None,
            "holdout_r": result["mean_r"],
            "pres_d_r": result["per_col_r"][0],
            "pres_r_r": result["per_col_r"][1],
            "turnout_r": result["per_col_r"][2],
            "j": j,
            "elapsed_s": elapsed,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PCA variance analysis
# ---------------------------------------------------------------------------


def analyze_pca_variance(train_matrix: np.ndarray) -> None:
    """Print cumulative variance explained across the full component range."""
    pca_full = PCA(random_state=KMEANS_SEED)
    pca_full.fit(train_matrix)
    evr = pca_full.explained_variance_ratio_
    cumvar = evr.cumsum()

    print("\n=== PCA Variance Explained (cumulative) ===")
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    t_idx = 0
    for i, (ev, cv) in enumerate(zip(evr, cumvar)):
        if t_idx < len(thresholds) and cv >= thresholds[t_idx]:
            print(f"  PC{i+1:3d}: +{ev:.4f} cumulative={cv:.4f}  <-- {thresholds[t_idx]:.0%} threshold")
            t_idx += 1
        elif i < 30:
            print(f"  PC{i+1:3d}: +{ev:.4f} cumulative={cv:.4f}")
        elif t_idx >= len(thresholds):
            break

    print(f"\n  Total training dims: {train_matrix.shape[1]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA + UMAP dimensionality reduction experiment for KMeans type discovery"
    )
    parser.add_argument("--j", type=int, default=J, help=f"KMeans clusters (default: {J})")
    parser.add_argument("--pca-min", type=int, default=5, help="Minimum PCA components (default: 5)")
    parser.add_argument("--pca-max", type=int, default=25, help="Maximum PCA components (default: 25)")
    parser.add_argument("--pca-step", type=int, default=1, help="PCA component step size (default: 1)")
    parser.add_argument("--umap", action="store_true", help="Run UMAP experiment (requires umap-learn)")
    parser.add_argument(
        "--umap-dims", type=int, nargs="+", default=[5, 10, 15, 20],
        help="UMAP component counts (default: 5 10 15 20)"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    args = parser.parse_args()

    print("=" * 70)
    print("PCA Clustering Experiment — WetherVane issue #93 / P3.6")
    print("=" * 70)
    print(f"Config: J={args.j}, PCA range={args.pca_min}–{args.pca_max} step {args.pca_step}")
    print(f"Baseline: no PCA, J=100, holdout r=0.698")
    print()

    # Load data
    train_matrix, holdout_matrix, train_cols, _ = load_shift_matrix(min_year=MIN_YEAR)

    # Analyze PCA variance first (informational)
    analyze_pca_variance(train_matrix)

    # --- Baseline: no PCA ---
    print("\n=== Baseline: KMeans J=100, no PCA ===")
    t0 = time.time()
    baseline = compute_holdout_r(train_matrix, holdout_matrix, j=args.j)
    baseline_elapsed = time.time() - t0
    print(f"  holdout r={baseline['mean_r']:.4f} | "
          f"per-col r={[f'{r:.3f}' for r in baseline['per_col_r']]} | "
          f"{baseline_elapsed:.1f}s")

    baseline_row = pd.DataFrame([{
        "method": "baseline_no_pca",
        "n_components": train_matrix.shape[1],
        "cumulative_var_explained": 1.0,
        "top1_var": None,
        "top5_var": None,
        "holdout_r": baseline["mean_r"],
        "pres_d_r": baseline["per_col_r"][0],
        "pres_r_r": baseline["per_col_r"][1],
        "turnout_r": baseline["per_col_r"][2],
        "j": args.j,
        "elapsed_s": baseline_elapsed,
    }])

    # --- PCA sweep ---
    pca_range = range(args.pca_min, args.pca_max + 1, args.pca_step)
    print(f"\n=== PCA sweep: {args.pca_min}–{args.pca_max} components ===")
    pca_df = run_pca_sweep(train_matrix, holdout_matrix, pca_range, j=args.j)

    # --- UMAP (optional) ---
    umap_df = pd.DataFrame()
    if args.umap:
        print(f"\n=== UMAP sweep: n_components={args.umap_dims} ===")
        umap_df = run_umap_sweep(train_matrix, holdout_matrix, args.umap_dims, j=args.j)

    # --- Summary ---
    all_results = pd.concat([baseline_row, pca_df, umap_df], ignore_index=True)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'n_comp':>6} {'var_exp':>8} {'holdout_r':>10} {'delta_vs_baseline':>18}")
    print("-" * 70)

    baseline_r = float(baseline_row["holdout_r"].iloc[0])
    for _, row in all_results.iterrows():
        var_exp_str = f"{row['cumulative_var_explained']:.3f}" if row['cumulative_var_explained'] is not None else "  N/A"
        delta = row["holdout_r"] - baseline_r
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        # Annotate best PCA result
        label = ""
        if not pd.isna(row.get("n_components")) and row["method"] == "PCA":
            if not pca_df.empty and row["holdout_r"] == pca_df["holdout_r"].max():
                label = " <-- BEST PCA"
        print(f"  {row['method']:<23} {int(row['n_components']) if not pd.isna(row['n_components']) else '?':>6} "
              f"{var_exp_str:>8} {row['holdout_r']:>10.4f} {delta_str:>18}{label}")

    if not pca_df.empty:
        best_pca_r = pca_df["holdout_r"].max()
        best_pca_row = pca_df.loc[pca_df["holdout_r"].idxmax()]
        delta_best = best_pca_r - baseline_r
        print(f"\nBest PCA: n_components={int(best_pca_row['n_components'])}, "
              f"var_explained={best_pca_row['cumulative_var_explained']:.3f}, "
              f"holdout r={best_pca_r:.4f} (delta={delta_best:+.4f} vs baseline {baseline_r:.4f})")

        if delta_best > 0.005:
            print("\nFINDING: PCA IMPROVES holdout r by >0.005 — investigate further.")
            print("  Consider updating production pipeline with best n_components.")
        elif delta_best > 0:
            print("\nFINDING: PCA shows marginal improvement (<0.005) — likely not worth the added complexity.")
        else:
            print(f"\nFINDING: PCA does NOT improve holdout r (best delta={delta_best:+.4f}).")
            print("  Raw features work fine. PCA removes variance that KMeans uses for type differentiation.")
            print("  Intuition: presidential shifts are correlated across cycles, but the slight")
            print("  year-to-year variation encodes genuine structural differences between types")
            print("  (e.g., college realignment emerged 2012-2016). PCA mixes these into a single")
            print("  generic direction, losing the temporal structure that separates types.")

    if not umap_df.empty:
        best_umap_r = umap_df["holdout_r"].max()
        best_umap_row = umap_df.loc[umap_df["holdout_r"].idxmax()]
        delta_umap = best_umap_r - baseline_r
        print(f"\nBest UMAP: n_components={int(best_umap_row['n_components'])}, "
              f"holdout r={best_umap_r:.4f} (delta={delta_umap:+.4f})")

    # --- Save results ---
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "pca_clustering_results.parquet"
        all_results.to_parquet(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
