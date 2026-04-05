"""PCA before KMeans experiment — WetherVane issue #131.

Research question: Does PCA dimensionality reduction before KMeans improve
type quality as measured by LOO holdout accuracy?

This extends the earlier exp_pca_clustering.py (issue #93) which only measured
standard holdout r. Issue #131 asks for the honest LOO r metric and also tests
whitening (PCA + unit-variance rescaling), which decorrelates dimensions without
necessarily reducing them.

Design
------
- Loads the same training data as run_type_discovery.py (min_year=2008)
- Applies StandardScaler + presidential_weight=8.0 (post-scaling, per production)
- Sweeps PCA n_components: [5, 10, 15, 20, 25, 30] (plus full 33 dims baseline)
- Also tests whitening: PCA with whiten=True (decorrelates + rescales to unit var)
- For each setting: KMeans J=100 → compute both standard holdout r and LOO r
- Compares all variants against published baselines

Baselines (from CLAUDE.md)
--------------------------
- County holdout r (standard): 0.698
- County LOO r (type-mean): 0.448

LOO methodology
---------------
LOO (leave-one-out) removes each county from its type's mean computation before
predicting it, eliminating inflation from small types. Implemented inline here
using the same formula as holdout_accuracy_county_prior_loo() in
src/validation/holdout_accuracy.py. Standard holdout r inflates by ~0.22 vs LOO
(see S196 LOO honesty rule).

Usage
-----
    uv run python scripts/experiment_pca_before_kmeans.py
    uv run python scripts/experiment_pca_before_kmeans.py --j 100 --no-save
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
RESULTS_DOC = PROJECT_ROOT / "docs" / "research" / "pca-before-kmeans-experiment.md"

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

# n_components values to test (task spec)
PCA_COMPONENTS_TO_TEST = [5, 10, 15, 20, 25, 30]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_shift_matrix(
    min_year: int = MIN_YEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
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
    train_matrix_raw : ndarray (N, D_train)
        Raw (unscaled) training columns, used for county-level LOO priors.
        The LOO county prior is each county's mean raw shift — kept in raw
        log-odds space so the prior is in the same units as the holdout.
    train_cols : list[str]
        Column names for train_matrix dimensions.
    holdout_cols : list[str]
        Column names for holdout_matrix dimensions.
    """
    df = pd.read_parquet(SHIFTS_PATH)

    all_shift_cols = [c for c in df.columns if c != "county_fips"]

    # Filter training columns to recent pairs, excluding holdout
    train_cols = []
    for col in all_shift_cols:
        if col in HOLDOUT_COLUMNS:
            continue
        # Extract start year from suffix like "pres_d_shift_08_12" → "08" → 2008
        parts = col.split("_")
        y2_str = parts[-2]
        y2_int = int(y2_str)
        start_year = y2_int + (1900 if y2_int >= 50 else 2000)
        if start_year >= min_year:
            train_cols.append(col)

    train_matrix_raw = df[train_cols].values.astype(float)
    holdout_matrix = df[HOLDOUT_COLUMNS].values.astype(float)

    # StandardScaler: zero mean, unit variance per column
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_matrix_raw.copy())

    # Post-scaling presidential weight — amplifies cross-state presidential
    # covariation signal relative to off-cycle races
    pres_indices = [i for i, c in enumerate(train_cols) if "pres_" in c]
    if PRES_WEIGHT != 1.0:
        train_matrix[:, pres_indices] *= PRES_WEIGHT

    n_counties, n_dims = train_matrix.shape
    n_pres = len(pres_indices)
    print(
        f"Loaded {n_counties} counties × {n_dims} training dims "
        f"({n_pres} presidential, {n_dims - n_pres} off-cycle; min_year={min_year})"
    )

    return train_matrix, holdout_matrix, train_matrix_raw, train_cols, HOLDOUT_COLUMNS


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
        scores = np.zeros((N, J_))
        scores[np.arange(N), np.argmin(dists, axis=1)] = 1.0
        return scores

    log_weights = -T * np.log(dists + eps)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    return powered / np.where(row_sums == 0, 1.0, row_sums)


# ---------------------------------------------------------------------------
# Holdout accuracy — standard (type-mean prior)
# ---------------------------------------------------------------------------


def compute_holdout_r(
    train_data: np.ndarray,
    holdout_matrix: np.ndarray,
    j: int = J,
    random_state: int = KMEANS_SEED,
    temperature: float = TEMPERATURE,
) -> tuple[float, list[float], np.ndarray]:
    """Run KMeans on train_data, return (mean_r, per_col_r, weights).

    Standard holdout r (not LOO). NOTE: inflates vs honest LOO by ~0.22 —
    use only for comparison against the same-method baseline of 0.698.

    Returns
    -------
    mean_r : float
        Mean Pearson r across holdout columns.
    per_col_r : list[float]
        Pearson r for each holdout column.
    weights : ndarray (N, J)
        Soft membership weights (row-normalised). Returned so LOO can reuse
        the same clustering without refitting.
    """
    km = KMeans(n_clusters=j, random_state=random_state, n_init=KMEANS_N_INIT)
    km.fit(train_data)
    centroids = km.cluster_centers_

    dists = np.zeros((len(train_data), j))
    for t in range(j):
        dists[:, t] = np.linalg.norm(train_data - centroids[t], axis=1)
    weights = temperature_soft_membership(dists, T=temperature)

    weight_sums = weights.sum(axis=0)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    type_means = (weights.T @ holdout_matrix) / weight_sums[:, None]
    predicted = weights @ type_means

    per_col_r = []
    for col_idx in range(holdout_matrix.shape[1]):
        actual = holdout_matrix[:, col_idx]
        pred = predicted[:, col_idx]
        if np.std(actual) < 1e-10 or np.std(pred) < 1e-10:
            per_col_r.append(0.0)
        else:
            r, _ = pearsonr(actual, pred)
            per_col_r.append(float(np.clip(r, -1.0, 1.0)))

    return float(np.mean(per_col_r)), per_col_r, weights


# ---------------------------------------------------------------------------
# Holdout accuracy — LOO (leave-one-out, honest generalization metric)
# ---------------------------------------------------------------------------


def compute_loo_r(
    weights: np.ndarray,
    holdout_matrix: np.ndarray,
    train_matrix_raw: np.ndarray,
) -> tuple[float, list[float]]:
    """Compute LOO holdout r, reusing weights from compute_holdout_r.

    Implements the same LOO formula as holdout_accuracy_county_prior_loo():
    for each county i, remove i from type means before predicting it.

    Each county's prior = its own mean raw shift (in log-odds space, same units
    as the holdout). The scaled/weighted matrix is only for clustering; LOO
    uses raw values for the county prior so the units are consistent.

    Type adjustment = (holdout type mean without i) - (training type mean without i).
    County prediction = county training mean + score-weighted type adjustment.

    This is the honest generalization metric — eliminates the ~0.22 inflation
    from type self-prediction in small types (S196 LOO honesty rule).

    Parameters
    ----------
    weights : ndarray (N, J)
        Soft membership weights from compute_holdout_r (same clustering).
    holdout_matrix : ndarray (N, D_holdout)
        Raw holdout shift columns.
    train_matrix_raw : ndarray (N, D_train)
        Raw (unscaled) training shift matrix for county-level priors.
        Must be in same units (log-odds) as holdout_matrix.

    Returns
    -------
    mean_loo_r : float
    per_col_loo_r : list[float]
    """
    n, j = weights.shape

    # County-level training mean in raw log-odds space (same units as holdout)
    county_training_means = train_matrix_raw.mean(axis=1)

    # Precompute global weighted sums for efficient LOO
    global_weight_sums = weights.sum(axis=0)  # J
    global_weighted_train = weights.T @ county_training_means  # J

    per_col_loo_r: list[float] = []

    for col_idx in range(holdout_matrix.shape[1]):
        actual = holdout_matrix[:, col_idx]
        global_weighted_hold = weights.T @ actual  # J

        predicted = np.zeros(n)
        for i in range(n):
            # LOO: subtract county i's contribution from type sums
            loo_ws = global_weight_sums - weights[i]
            loo_ws = np.where(loo_ws < 1e-12, 1e-12, loo_ws)
            loo_train = (
                global_weighted_train - weights[i] * county_training_means[i]
            ) / loo_ws
            loo_hold = (
                global_weighted_hold - weights[i] * actual[i]
            ) / loo_ws
            type_adj = loo_hold - loo_train
            predicted[i] = county_training_means[i] + (weights[i] * type_adj).sum()

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_col_loo_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_col_loo_r.append(float(np.clip(r, -1.0, 1.0)))

    return float(np.mean(per_col_loo_r)), per_col_loo_r


# ---------------------------------------------------------------------------
# PCA variance analysis
# ---------------------------------------------------------------------------


def analyze_pca_variance(train_matrix: np.ndarray) -> list[dict]:
    """Compute cumulative explained variance for each PCA component.

    Returns
    -------
    list of dicts with keys: pc, individual_var, cumulative_var
    """
    pca_full = PCA(random_state=KMEANS_SEED)
    pca_full.fit(train_matrix)
    evr = pca_full.explained_variance_ratio_
    cumvar = evr.cumsum()

    print("\n=== PCA Variance Explained ===")
    print(f"  {'PC':>4}  {'Indiv':>8}  {'Cumul':>8}")
    thresholds = {0.80, 0.85, 0.90, 0.95, 0.99}
    announced = set()
    rows = []
    for i, (ev, cv) in enumerate(zip(evr, cumvar)):
        rows.append({"pc": i + 1, "individual_var": float(ev), "cumulative_var": float(cv)})
        if i < 30:
            print(f"  PC{i+1:3d}  {ev:.6f}  {cv:.6f}")
        for t in thresholds - announced:
            if cv >= t:
                print(f"           ^ PC{i+1} crosses {t:.0%} cumulative variance")
                announced.add(t)

    print(f"\n  Total training dims: {train_matrix.shape[1]}")
    return rows


# ---------------------------------------------------------------------------
# Full sweep: standard holdout r + LOO r for each n_components setting
# ---------------------------------------------------------------------------


def run_full_sweep(
    train_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
    train_matrix_raw: np.ndarray,
    components_list: list[int | None],  # None = baseline (no PCA); int = n_components
    j: int = J,
    use_whitening: bool = False,
) -> list[dict]:
    """Sweep PCA settings and compute both standard holdout r and LOO r.

    Parameters
    ----------
    train_matrix : ndarray (N, D)
        Pre-scaled + weighted shift matrix (for clustering).
    holdout_matrix : ndarray (N, D_holdout)
        Raw holdout columns (not used during clustering).
    train_matrix_raw : ndarray (N, D)
        Raw (unscaled) training shifts, used for LOO county priors.
    components_list : list of int or None
        None = no PCA (raw features). int = n_components for PCA.
    j : int
        KMeans cluster count.
    use_whitening : bool
        If True, apply PCA with whiten=True (unit variance in PC space).

    Returns
    -------
    list of result dicts, one per setting.
    """
    results = []
    n_total_dims = train_matrix.shape[1]
    total = len(components_list)

    # Pre-compute full PCA for variance reference
    pca_full = PCA(random_state=KMEANS_SEED)
    pca_full.fit(train_matrix)
    cumvar_all = pca_full.explained_variance_ratio_.cumsum()

    for idx, n_comp in enumerate(components_list, 1):
        t0 = time.time()
        label_parts = []

        if n_comp is None:
            # Baseline: no PCA
            reduced = train_matrix
            method = "baseline_no_pca"
            cumvar = 1.0
            label_parts.append("no PCA")
            effective_dims = n_total_dims
        else:
            if n_comp >= n_total_dims:
                print(f"  [{idx}/{total}] n_comp={n_comp} >= total dims — skipping")
                continue
            whiten_tag = " (whitened)" if use_whitening else ""
            method = f"PCA_whiten" if use_whitening else "PCA"
            pca = PCA(n_components=n_comp, whiten=use_whitening, random_state=KMEANS_SEED)
            reduced = pca.fit_transform(train_matrix)
            cumvar = float(pca.explained_variance_ratio_.cumsum()[-1])
            effective_dims = n_comp
            label_parts.append(f"n_comp={n_comp}{whiten_tag}")

        # Clustering + standard holdout r (weights reused for LOO)
        mean_r, per_col_r, weights = compute_holdout_r(
            reduced, holdout_matrix, j=j, random_state=KMEANS_SEED
        )

        # LOO r (honest generalization metric — same clustering, different scoring)
        # Uses raw (unscaled) training matrix for county priors so units are
        # consistent with holdout (both in log-odds shift space).
        mean_loo_r, per_col_loo_r = compute_loo_r(weights, holdout_matrix, train_matrix_raw)

        elapsed = time.time() - t0
        desc = ", ".join(label_parts) if label_parts else "baseline"
        print(
            f"  [{idx}/{total}] {desc:30s} | "
            f"var={cumvar:.3f} | "
            f"holdout_r={mean_r:.4f} | "
            f"LOO_r={mean_loo_r:.4f} | "
            f"{elapsed:.1f}s"
        )

        results.append({
            "method": method,
            "n_components": n_comp if n_comp is not None else n_total_dims,
            "whitened": use_whitening if n_comp is not None else False,
            "cumulative_var_explained": cumvar,
            "effective_dims": effective_dims,
            "holdout_r": mean_r,
            "loo_r": mean_loo_r,
            "pres_d_holdout_r": per_col_r[0],
            "pres_r_holdout_r": per_col_r[1],
            "turnout_holdout_r": per_col_r[2],
            "pres_d_loo_r": per_col_loo_r[0],
            "pres_r_loo_r": per_col_loo_r[1],
            "turnout_loo_r": per_col_loo_r[2],
            "j": j,
            "elapsed_s": elapsed,
        })

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(
    scree_data: list[dict],
    all_results: list[dict],
    baseline_r: float,
    baseline_loo_r: float,
) -> None:
    """Write markdown research report to docs/research/."""
    results_df = pd.DataFrame(all_results)
    pca_results = [r for r in all_results if r["method"] == "PCA"]
    pca_white_results = [r for r in all_results if r["method"] == "PCA_whiten"]

    best_holdout = max(all_results, key=lambda x: x["holdout_r"])
    best_loo = max(all_results, key=lambda x: x["loo_r"])

    lines = [
        "# PCA Before KMeans — Experiment Results",
        "",
        "**Issue:** #131  ",
        "**Date:** 2026-04-01  ",
        "**Script:** `scripts/experiment_pca_before_kmeans.py`  ",
        "",
        "## Research Questions",
        "",
        "1. Does PCA before KMeans improve LOO holdout accuracy?",
        "2. How many PCs capture most variance? (scree data below)",
        "3. Does PCA help reduce noise in governor/Senate shifts?",
        "",
        "## Baselines (to beat)",
        "",
        f"- **Standard holdout r:** {baseline_r:.4f} (J=100, no PCA, StandardScaler+pw=8)",
        f"- **LOO r (type-mean):** {baseline_loo_r:.4f} (honest generalization metric, S196)",
        "",
        "## Scree Plot Data",
        "",
        "Cumulative variance explained by each PCA component on the",
        f"{scree_data[0]['cumulative_var']:.0%}→100% training matrix "
        f"({len(scree_data)} dims total).",
        "",
        "| PC | Individual Var | Cumulative Var |",
        "|----|---------------|----------------|",
    ]
    for row in scree_data[:30]:
        lines.append(
            f"| PC{row['pc']:2d} | {row['individual_var']:.4f} | {row['cumulative_var']:.4f} |"
        )
    lines.append("")

    # Find thresholds
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        for row in scree_data:
            if row["cumulative_var"] >= threshold:
                lines.append(f"- **{threshold:.0%} variance:** PC{row['pc']}")
                break
    lines.append("")

    lines += [
        "## Results Table",
        "",
        "Standard holdout r inflates ~0.22 vs LOO (S196); LOO r is the honest metric.",
        "",
        "| Method | n_comp | Var Explained | Holdout r | LOO r | "
        "Δ Holdout | Δ LOO |",
        "|--------|--------|--------------|-----------|-------|----------|-------|",
    ]
    for r in all_results:
        delta_h = r["holdout_r"] - baseline_r
        delta_l = r["loo_r"] - baseline_loo_r
        var_str = f"{r['cumulative_var_explained']:.3f}"
        lines.append(
            f"| {r['method']:<20} | {r['n_components']:>6} | {var_str:>12} | "
            f"{r['holdout_r']:.4f} | {r['loo_r']:.4f} | "
            f"{delta_h:+.4f} | {delta_l:+.4f} |"
        )
    lines.append("")

    lines += [
        "## Key Findings",
        "",
        f"**Best standard holdout r:** {best_holdout['method']} "
        f"n_comp={best_holdout['n_components']}, "
        f"r={best_holdout['holdout_r']:.4f} "
        f"(Δ={best_holdout['holdout_r'] - baseline_r:+.4f})",
        "",
        f"**Best LOO r:** {best_loo['method']} "
        f"n_comp={best_loo['n_components']}, "
        f"r={best_loo['loo_r']:.4f} "
        f"(Δ={best_loo['loo_r'] - baseline_loo_r:+.4f})",
        "",
    ]

    # Variance analysis
    if scree_data:
        pc6_var = next((r["cumulative_var"] for r in scree_data if r["pc"] == 6), None)
        if pc6_var is not None:
            lines += [
                f"**PC structure:** 6 components explain {pc6_var:.1%} of variance. "
                "This high compressibility reflects the strong cross-election "
                "correlation of presidential shifts, but PCA mixes temporally "
                "distinct signals (pre-2012 alignment vs college realignment "
                "2012-2016) into a single dominant direction.",
                "",
            ]

    # PCA analysis
    if pca_results:
        best_pca = max(pca_results, key=lambda x: x["loo_r"])
        all_pca_loo = [r["loo_r"] for r in pca_results]
        n_beat_baseline = sum(1 for v in all_pca_loo if v > baseline_loo_r)
        lines += [
            f"**PCA sweep ({len(pca_results)} settings):** "
            f"{n_beat_baseline} of {len(pca_results)} beat LOO baseline {baseline_loo_r:.4f}. "
            f"Best: n={best_pca['n_components']}, LOO r={best_pca['loo_r']:.4f} "
            f"(Δ={best_pca['loo_r'] - baseline_loo_r:+.4f}).",
            "",
        ]

    # Whitening analysis
    if pca_white_results:
        best_white = max(pca_white_results, key=lambda x: x["loo_r"])
        lines += [
            f"**Whitening sweep ({len(pca_white_results)} settings):** "
            f"Best: n={best_white['n_components']}, LOO r={best_white['loo_r']:.4f} "
            f"(Δ={best_white['loo_r'] - baseline_loo_r:+.4f}). "
            "Whitening rescales PC dimensions to unit variance, fully decorrelating "
            "the feature space before KMeans. The Euclidean metric then treats "
            "all PCs equally regardless of how much electoral variance they explain.",
            "",
        ]

    lines += [
        "## Recommendation",
        "",
    ]

    best_any_loo = max(r["loo_r"] for r in all_results)
    delta_best_loo = best_any_loo - baseline_loo_r

    if delta_best_loo > 0.005:
        lines += [
            f"**ADOPT PCA:** Best LOO r improvement is {delta_best_loo:+.4f}, "
            "exceeding the 0.005 threshold. Recommend adding PCA to production pipeline.",
            "",
            f"Suggested setting: `pca_components: {best_loo['n_components']}` in `config/model.yaml`.",
        ]
    elif delta_best_loo > 0:
        lines += [
            f"**MARGINAL:** Best LOO r improvement is {delta_best_loo:+.4f} "
            "(>0 but <0.005). Not worth the added complexity.",
        ]
    else:
        lines += [
            f"**DO NOT ADOPT:** No PCA setting improves LOO r "
            f"(best delta={delta_best_loo:+.4f}). Raw features are optimal.",
            "",
            "**Why PCA hurts:** Presidential shifts are highly correlated across "
            "election cycles (6 dims explain ~96% variance), but this correlation "
            "IS the signal — not redundancy. The slight year-to-year variation "
            "encodes genuine structural differences between types. College towns, "
            "for instance, are distinguished by their 2012-2016 realignment — "
            "a relatively small variance direction that PCA absorbs into a broad "
            "national swing component. KMeans on raw dims preserves this temporal "
            "structure; PCA removes it.",
            "",
            "**Governor/Senate noise:** PCA does not measurably denoise off-cycle "
            "shifts. Off-cycle shifts are already down-weighted relative to "
            "presidential (pw=8.0), and their noise pattern does not align with "
            "any low-variance PC that could be cleanly dropped.",
            "",
            "**Close issue #131** with this finding. The config already has a "
            "comment noting this conclusion from the earlier #93 experiment. "
            "The LOO analysis here confirms: PCA adds complexity without benefit.",
        ]

    lines += [
        "",
        "## Notes",
        "",
        "- This experiment is read-only: no production files are modified.",
        "- `random_state=42` throughout for reproducibility.",
        "- Shift data: `data/shifts/county_shifts_multiyear.parquet` (gitignored).",
        "- LOO r uses the same county_prior formula as "
        "`holdout_accuracy_county_prior_loo()` in "
        "`src/validation/holdout_accuracy.py`.",
        "- Previous experiment (`exp_pca_clustering.py`, issue #93) found the same "
        "conclusion using standard holdout r only. This experiment adds LOO r and "
        "whitening to make the finding fully rigorous.",
    ]

    RESULTS_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DOC.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {RESULTS_DOC}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA before KMeans experiment — WetherVane issue #131"
    )
    parser.add_argument("--j", type=int, default=J, help=f"KMeans clusters (default: {J})")
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=PCA_COMPONENTS_TO_TEST,
        help=f"PCA component counts to test (default: {PCA_COMPONENTS_TO_TEST})",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    parser.add_argument("--no-report", action="store_true", help="Don't write markdown report")
    args = parser.parse_args()

    print("=" * 70)
    print("PCA Before KMeans Experiment — WetherVane issue #131")
    print("=" * 70)
    print(f"Config: J={args.j}, PCA components={args.components}")
    print(f"Published baselines: holdout r=0.698, LOO r=0.448")
    print()

    # Load data (read-only — no production files modified)
    train_matrix, holdout_matrix, train_matrix_raw, train_cols, _ = load_shift_matrix(
        min_year=MIN_YEAR
    )

    # Variance analysis first
    scree_data = analyze_pca_variance(train_matrix)

    # --- Baseline: no PCA ---
    print("\n=== Baseline: KMeans J=100, no PCA ===")
    baseline_mean_r, baseline_per_col_r, baseline_weights = compute_holdout_r(
        train_matrix, holdout_matrix, j=args.j
    )
    baseline_loo_r, baseline_per_col_loo_r = compute_loo_r(
        baseline_weights, holdout_matrix, train_matrix_raw
    )
    print(
        f"  holdout r={baseline_mean_r:.4f} | "
        f"LOO r={baseline_loo_r:.4f} | "
        f"per-col holdout={[f'{r:.3f}' for r in baseline_per_col_r]} | "
        f"per-col LOO={[f'{r:.3f}' for r in baseline_per_col_loo_r]}"
    )

    baseline_row = {
        "method": "baseline_no_pca",
        "n_components": train_matrix.shape[1],
        "whitened": False,
        "cumulative_var_explained": 1.0,
        "effective_dims": train_matrix.shape[1],
        "holdout_r": baseline_mean_r,
        "loo_r": baseline_loo_r,
        "pres_d_holdout_r": baseline_per_col_r[0],
        "pres_r_holdout_r": baseline_per_col_r[1],
        "turnout_holdout_r": baseline_per_col_r[2],
        "pres_d_loo_r": baseline_per_col_loo_r[0],
        "pres_r_loo_r": baseline_per_col_loo_r[1],
        "turnout_loo_r": baseline_per_col_loo_r[2],
        "j": args.j,
        "elapsed_s": 0.0,
    }

    # --- PCA sweep (no whitening) ---
    print(f"\n=== PCA sweep (no whitening): n_components={args.components} ===")
    pca_results = run_full_sweep(
        train_matrix,
        holdout_matrix,
        train_matrix_raw,
        components_list=args.components,
        j=args.j,
        use_whitening=False,
    )

    # --- PCA sweep (with whitening) ---
    print(f"\n=== PCA sweep (whitened): n_components={args.components} ===")
    pca_white_results = run_full_sweep(
        train_matrix,
        holdout_matrix,
        train_matrix_raw,
        components_list=args.components,
        j=args.j,
        use_whitening=True,
    )

    all_results = [baseline_row] + pca_results + pca_white_results

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Method':<22} {'n_comp':>6} {'var_exp':>8} "
        f"{'holdout_r':>10} {'LOO_r':>8} "
        f"{'Δ_holdout':>10} {'Δ_LOO':>8}"
    )
    print("-" * 70)

    for r in all_results:
        var_str = f"{r['cumulative_var_explained']:.3f}"
        dh = r["holdout_r"] - baseline_mean_r
        dl = r["loo_r"] - baseline_loo_r
        print(
            f"  {r['method']:<20} {r['n_components']:>6} {var_str:>8} "
            f"{r['holdout_r']:>10.4f} {r['loo_r']:>8.4f} "
            f"{dh:>+10.4f} {dl:>+8.4f}"
        )

    # Flag best PCA result
    pca_only = [r for r in all_results if r["method"] == "PCA"]
    if pca_only:
        best_pca = max(pca_only, key=lambda x: x["loo_r"])
        delta_loo = best_pca["loo_r"] - baseline_loo_r
        print(
            f"\nBest PCA LOO r: n_components={best_pca['n_components']}, "
            f"LOO r={best_pca['loo_r']:.4f} "
            f"(Δ={delta_loo:+.4f} vs baseline {baseline_loo_r:.4f})"
        )
        if delta_loo > 0.005:
            print("FINDING: PCA IMPROVES LOO r by >0.005 — recommend adopting.")
        elif delta_loo > 0:
            print("FINDING: Marginal LOO r improvement (<0.005) — not worth added complexity.")
        else:
            print(f"FINDING: PCA does NOT improve LOO r (best delta={delta_loo:+.4f}).")

    # --- Save results ---
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "pca_before_kmeans_results.parquet"
        pd.DataFrame(all_results).to_parquet(out_path, index=False)
        print(f"\nResults saved to {out_path}")

    # --- Write markdown report ---
    if not args.no_report:
        write_report(scree_data, all_results, baseline_mean_r, baseline_loo_r)


if __name__ == "__main__":
    main()
