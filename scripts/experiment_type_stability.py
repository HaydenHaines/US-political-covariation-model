"""Experiment: Type stability across sub-windows (P4.2).

Compares types discovered from 2008-2016 vs 2016-2024 electoral shifts to
assess whether electoral community types are durable across periods or
period-specific artifacts.

Design:
- Production model uses 33 training dimensions (2008+, presidential x2.5,
  state-centered gov/Senate shifts). Holdout is always pres_*_shift_20_24.
- This experiment splits those 33 dims into two sub-windows:
    Early: election pairs with start year 2008-2015 (24 dims)
    Late:  election pairs with start year 2016+    (9 dims)
- Runs KMeans J=43 independently on each sub-window.
- Compares type assignments using ARI, NMI, and county stability (Hungarian
  matching to align type labels).
- Computes holdout r for each sub-window model on the 2020->2024 blind holdout.
- Bootstrap baseline: 50 KMeans runs with different seeds on the full window
  (measures seed-stability ARI -- if sub-window ARI approaches this, types
  are essentially stable across periods).
- Writes results to docs/type-stability-subwindows-S175.md

Findings vocabulary:
- ARI ~ 0.0: random agreement (types are completely period-specific)
- ARI ~ seed_ARI: types are as stable across periods as across random seeds
  (effectively period-invariant structure)
- 0.3-0.7 range: partial stability -- core structure preserved, some drift

Usage:
    cd /home/hayden/projects/US-political-covariation-model
    uv run python scripts/experiment_type_stability.py
    uv run python scripts/experiment_type_stability.py --j 43 --n-bootstrap 50
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent

SHIFTS_PATH = DATA_ROOT / "data/shifts/county_shifts_multiyear.parquet"
DOCS_DIR = DATA_ROOT / "docs"

KMEANS_SEED = 42
KMEANS_N_INIT = 10
PRES_WEIGHT = 2.5
MIN_YEAR = 2008
TEMPERATURE = 10.0
DEFAULT_J = 43
DEFAULT_N_BOOTSTRAP = 50

# Sub-window split: pairs starting before SPLIT_YEAR go to early window,
# pairs starting >= SPLIT_YEAR go to late window.
SPLIT_YEAR = 2016

BLIND_HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]


# ---------------------------------------------------------------------------
# Year parsing
# ---------------------------------------------------------------------------


def parse_2digit_year(yy: str) -> int:
    """Convert a 2-digit year string to 4-digit year.

    Years 00-29 map to 2000-2029. Years 30-99 map to 1930-1999.
    This covers the full range in county_shifts_multiyear.parquet
    (earliest: 94=1994, latest: 24=2024).
    """
    n = int(yy)
    return 2000 + n if n < 30 else 1900 + n


def get_pair_start_year(col: str) -> int | None:
    """Extract the start year of the election pair encoded in a column name.

    Column naming: <race>_<metric>_shift_<yy1>_<yy2>
    Returns 4-digit start year, or None if pattern does not match.
    """
    m = re.search(r"shift_(\d{2})_(\d{2})$", col)
    if m:
        return parse_2digit_year(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Data loading (mirrors production pipeline preprocessing)
# ---------------------------------------------------------------------------


def load_shift_matrix(
    min_year: int = MIN_YEAR,
    pres_weight: float = PRES_WEIGHT,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the county shift matrix.

    Applies the same preprocessing as the production pipeline:
    - Filter to pairs where start year >= min_year
    - Exclude blind holdout columns (2020->2024)
    - Apply presidential weight (pres_weight) to pres_* columns
    - State-center governor and Senate columns

    Returns
    -------
    X : ndarray of shape (N, D)
        Preprocessed shift matrix.
    shift_cols : list[str]
        Column names (D columns, post-filter).
    county_fips : ndarray of shape (N,)
        County FIPS codes (zero-padded strings).
    pres_mask : ndarray of shape (D,) bool
        True for presidential columns (already weighted).
    gov_sen_mask : ndarray of shape (D,) bool
        True for governor/Senate columns (already state-centered).
    """
    df = pd.read_parquet(SHIFTS_PATH)
    county_fips = df["county_fips"].astype(str).str.zfill(5).values

    all_cols = [c for c in df.columns if c != "county_fips" and c not in BLIND_HOLDOUT_COLUMNS]

    pair_re = re.compile(r"shift_(\d{2})_(\d{2})$")
    shift_cols = []
    for c in all_cols:
        m = pair_re.search(c)
        if m:
            y1 = parse_2digit_year(m.group(1))
            if y1 >= min_year:
                shift_cols.append(c)

    if not shift_cols:
        shift_cols = all_cols

    X_raw = df[shift_cols].values.astype(float)

    pres_mask = np.array([c.startswith("pres_") for c in shift_cols])
    gov_sen_mask = np.array([c.startswith("gov_") or c.startswith("sen_") for c in shift_cols])

    # State-center governor and Senate columns within each state
    state_prefix = np.array([f[:2] for f in county_fips])
    if gov_sen_mask.any():
        X_centered = X_raw.copy()
        for prefix in np.unique(state_prefix):
            idx = state_prefix == prefix
            col_idx = np.where(gov_sen_mask)[0]
            X_centered[np.ix_(idx, col_idx)] -= X_raw[np.ix_(idx, col_idx)].mean(axis=0)
        X_raw = X_centered

    # Apply presidential weight
    weights = np.where(pres_mask, pres_weight, 1.0)
    X = X_raw * weights[None, :]

    return X, shift_cols, county_fips, pres_mask, gov_sen_mask


def split_columns_by_window(
    shift_cols: list[str],
    split_year: int = SPLIT_YEAR,
    min_year: int = MIN_YEAR,
) -> tuple[list[int], list[int]]:
    """Split column indices into early and late sub-windows.

    Early window: pairs with start year in [min_year, split_year)
    Late window:  pairs with start year >= split_year

    Returns
    -------
    early_idx : list of int
        Column indices for early window.
    late_idx : list of int
        Column indices for late window.
    """
    early_idx = []
    late_idx = []
    for i, col in enumerate(shift_cols):
        y1 = get_pair_start_year(col)
        if y1 is None:
            continue
        if min_year <= y1 < split_year:
            early_idx.append(i)
        elif y1 >= split_year:
            late_idx.append(i)
    return early_idx, late_idx


# ---------------------------------------------------------------------------
# KMeans helpers
# ---------------------------------------------------------------------------


def temperature_soft_membership(dists: np.ndarray, T: float = TEMPERATURE) -> np.ndarray:
    """Temperature-sharpened soft membership (matches production implementation)."""
    N, J = dists.shape
    eps = 1e-10

    if T >= 500.0:
        scores = np.zeros((N, J))
        nearest = np.argmin(dists, axis=1)
        scores[np.arange(N), nearest] = 1.0
        return scores

    log_weights = -T * np.log(dists + eps)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return powered / row_sums


def run_kmeans(
    X: np.ndarray,
    j: int,
    random_state: int = KMEANS_SEED,
    n_init: int = KMEANS_N_INIT,
    temperature: float = TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit KMeans and return labels, centroids, soft scores.

    Returns
    -------
    labels : ndarray of shape (N,) -- hard cluster assignments
    centroids : ndarray of shape (J, D)
    scores : ndarray of shape (N, J) -- temperature soft membership
    """
    km = KMeans(n_clusters=j, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    N = X.shape[0]
    dists = np.zeros((N, j))
    for t in range(j):
        dists[:, t] = np.linalg.norm(X - centroids[t], axis=1)
    scores = temperature_soft_membership(dists, T=temperature)

    return labels, centroids, scores


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def hungarian_match(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Find the optimal label permutation of labels_b to align with labels_a.

    Uses the Hungarian algorithm on a cost matrix derived from co-occurrence
    counts. Returns a mapping array `perm` such that perm[b_label] = a_label.

    Parameters
    ----------
    labels_a : ndarray of shape (N,) -- reference labels
    labels_b : ndarray of shape (N,) -- labels to permute

    Returns
    -------
    perm : ndarray of shape (max_label+1,)
        Permutation of b labels to align with a labels.
    remapped_b : ndarray of shape (N,)
        labels_b with optimal permutation applied.
    """
    j_a = int(labels_a.max()) + 1
    j_b = int(labels_b.max()) + 1
    j = max(j_a, j_b)

    # Co-occurrence matrix: cost[i, j] = number of counties where a=i and b=j
    cost = np.zeros((j, j), dtype=int)
    for a, b in zip(labels_a, labels_b):
        cost[int(a), int(b)] += 1

    # Hungarian matching: maximize overlap = minimize negative cost
    row_ind, col_ind = linear_sum_assignment(-cost)

    # Build permutation: b_label -> a_label
    perm = np.arange(j)
    for a_label, b_label in zip(row_ind, col_ind):
        perm[b_label] = a_label

    remapped_b = perm[labels_b]
    return perm, remapped_b


def county_stability_rate(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Fraction of counties assigned to the same type under optimal alignment.

    Applies Hungarian matching to align type labels, then counts exact matches.

    Parameters
    ----------
    labels_a : ndarray of shape (N,) -- reference labels
    labels_b : ndarray of shape (N,) -- labels to compare (will be permuted)

    Returns
    -------
    stability : float in [0, 1]
    """
    _, remapped_b = hungarian_match(labels_a, labels_b)
    return float(np.mean(labels_a == remapped_b))


# ---------------------------------------------------------------------------
# Holdout r computation
# ---------------------------------------------------------------------------


def predict_from_types(
    scores: np.ndarray,
    X_train: np.ndarray,
    X_holdout: np.ndarray,
) -> np.ndarray:
    """Predict holdout columns using type-mean weighted by soft membership.

    Parameters
    ----------
    scores : ndarray of shape (N, J)
    X_train : ndarray of shape (N, D_train) -- used to fit type means
    X_holdout : ndarray of shape (N, D_holdout) -- target predictions

    Returns
    -------
    predicted : ndarray of shape (N, D_holdout)
    """
    weight_sums = scores.sum(axis=0)  # (J,)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    type_means = (scores.T @ X_holdout) / weight_sums[:, None]  # (J, D_holdout)
    return scores @ type_means  # (N, D_holdout)


def compute_holdout_r(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Pearson r across columns of the holdout matrix."""
    r_vals = []
    for d in range(actual.shape[1]):
        a = actual[:, d]
        p = predicted[:, d]
        if np.std(a) < 1e-10 or np.std(p) < 1e-10:
            r_vals.append(0.0)
        else:
            r, _ = pearsonr(a, p)
            r_vals.append(float(r))
    return float(np.mean(r_vals))


def compute_subwindow_holdout_r(
    X_subwindow: np.ndarray,
    X_holdout_raw: np.ndarray,
    j: int,
    temperature: float = TEMPERATURE,
) -> float:
    """Fit KMeans on X_subwindow, predict X_holdout_raw, return mean Pearson r.

    The holdout matrix uses raw (non-weighted) shift values for the 2020->2024
    presidential columns so that r is computed in the original shift space.
    """
    labels, centroids, scores = run_kmeans(X_subwindow, j=j, temperature=temperature)
    predicted = predict_from_types(scores, X_subwindow, X_holdout_raw)
    return compute_holdout_r(X_holdout_raw, predicted)


# ---------------------------------------------------------------------------
# Bootstrap stability test
# ---------------------------------------------------------------------------


def bootstrap_seed_stability(
    X_full: np.ndarray,
    j: int,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    base_seed: int = KMEANS_SEED,
    temperature: float = TEMPERATURE,
    verbose: bool = True,
) -> dict[str, float]:
    """Estimate KMeans seed stability via pairwise ARI across bootstrap runs.

    Runs KMeans n_bootstrap times with different seeds on the full window.
    Computes pairwise ARI between all pairs of runs. The mean pairwise ARI
    is the "seed stability baseline" -- how much the clustering agrees when
    the only difference is random initialization.

    A high seed ARI (>0.8) means KMeans finds consistent types regardless of
    initialization. Sub-window ARI close to seed ARI means cross-period
    stability matches random-seed stability.

    Parameters
    ----------
    X_full : ndarray of shape (N, D)
        Full-window preprocessed shift matrix.
    j : int
        Number of KMeans clusters.
    n_bootstrap : int
        Number of random seeds to try.
    base_seed : int
        Starting random seed. Seeds are base_seed, base_seed+1, ...
    temperature : float
        Soft membership temperature.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        mean_pairwise_ari : float
        std_pairwise_ari  : float
        n_runs            : int
        n_pairs           : int
    """
    all_labels = []
    for i in range(n_bootstrap):
        seed = base_seed + i
        labels, _, _ = run_kmeans(X_full, j=j, random_state=seed, temperature=temperature)
        all_labels.append(labels)

        if verbose and (i + 1) % 10 == 0:
            print(f"    Bootstrap run {i + 1}/{n_bootstrap} done")

    # Pairwise ARI
    ari_vals = []
    for i in range(n_bootstrap):
        for k in range(i + 1, n_bootstrap):
            ari = adjusted_rand_score(all_labels[i], all_labels[k])
            ari_vals.append(ari)

    return {
        "mean_pairwise_ari": float(np.mean(ari_vals)),
        "std_pairwise_ari": float(np.std(ari_vals)),
        "n_runs": n_bootstrap,
        "n_pairs": len(ari_vals),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_type_stability_experiment(
    j: int = DEFAULT_J,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    temperature: float = TEMPERATURE,
    min_year: int = MIN_YEAR,
    split_year: int = SPLIT_YEAR,
    verbose: bool = True,
) -> dict:
    """Run the full type stability sub-window experiment.

    Steps:
    1. Load full production shift matrix (33 dims, 2008+, pres×2.5, state-centered)
    2. Split into early (2008-2015) and late (2016+) sub-windows
    3. Load raw holdout columns (pres_*_shift_20_24) for holdout r
    4. Cluster each sub-window with KMeans J=j
    5. Compare type assignments (ARI, NMI, county stability)
    6. Compute holdout r for each sub-window model
    7. Run bootstrap seed stability on the full window
    8. Return all results in a dict

    Parameters
    ----------
    j : int
        Number of KMeans types.
    n_bootstrap : int
        Number of bootstrap seeds for seed-stability baseline.
    temperature : float
        Soft membership temperature.
    min_year : int
        Minimum start year for shift pairs (production default: 2008).
    split_year : int
        Year that separates early vs late sub-windows (default: 2016).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict with all experiment results.
    """
    if verbose:
        print("=" * 70)
        print(f"Type Stability Sub-Window Experiment (J={j}, split={split_year})")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[1/5] Loading shift matrix (min_year={min_year})...")
    X_full, shift_cols, county_fips, pres_mask, gov_sen_mask = load_shift_matrix(
        min_year=min_year, pres_weight=PRES_WEIGHT
    )
    if verbose:
        print(f"      Shape: {X_full.shape[0]} counties x {X_full.shape[1]} dims")

    # Load raw holdout columns (unweighted, for interpretable r)
    df_raw = pd.read_parquet(SHIFTS_PATH)
    X_holdout_raw = df_raw[BLIND_HOLDOUT_COLUMNS].values.astype(float)

    # Split columns into sub-windows
    early_idx, late_idx = split_columns_by_window(shift_cols, split_year=split_year, min_year=min_year)
    X_early = X_full[:, early_idx]
    X_late = X_full[:, late_idx]

    early_cols = [shift_cols[i] for i in early_idx]
    late_cols = [shift_cols[i] for i in late_idx]

    if verbose:
        print(f"      Early window ({split_year - min_year}-ish years): {len(early_idx)} dims")
        print(f"      Late window  ({split_year}+):                    {len(late_idx)} dims")
        print(f"      Holdout columns: {BLIND_HOLDOUT_COLUMNS}")

    # ------------------------------------------------------------------
    # Cluster each sub-window
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2/5] Clustering early window (seed={KMEANS_SEED})...")
    early_labels, early_centroids, early_scores = run_kmeans(X_early, j=j, temperature=temperature)

    if verbose:
        print(f"      Type sizes (early): {sorted(np.bincount(early_labels).tolist(), reverse=True)[:10]}")
        print(f"\n[2/5] Clustering late window (seed={KMEANS_SEED})...")
    late_labels, late_centroids, late_scores = run_kmeans(X_late, j=j, temperature=temperature)

    if verbose:
        print(f"      Type sizes (late):  {sorted(np.bincount(late_labels).tolist(), reverse=True)[:10]}")

    # ------------------------------------------------------------------
    # Cross-window comparison
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[3/5] Computing cross-window comparison metrics...")

    ari = float(adjusted_rand_score(early_labels, late_labels))
    nmi = float(normalized_mutual_info_score(early_labels, late_labels))
    stability = county_stability_rate(early_labels, late_labels)

    if verbose:
        print(f"      ARI (early vs late): {ari:.4f}")
        print(f"      NMI (early vs late): {nmi:.4f}")
        print(f"      County stability:    {stability:.1%}")

    # ------------------------------------------------------------------
    # Holdout r for each sub-window
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[4/5] Computing holdout r (2020->2024) for each sub-window...")

    early_holdout_r = compute_subwindow_holdout_r(X_early, X_holdout_raw, j=j, temperature=temperature)
    late_holdout_r = compute_subwindow_holdout_r(X_late, X_holdout_raw, j=j, temperature=temperature)
    full_holdout_r = compute_subwindow_holdout_r(X_full, X_holdout_raw, j=j, temperature=temperature)

    if verbose:
        print(f"      Early window holdout r: {early_holdout_r:.4f}")
        print(f"      Late window holdout r:  {late_holdout_r:.4f}")
        print(f"      Full window holdout r:  {full_holdout_r:.4f}")

    # ------------------------------------------------------------------
    # Bootstrap seed stability
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[5/5] Bootstrap seed stability ({n_bootstrap} runs on full window)...")

    seed_stability = bootstrap_seed_stability(
        X_full,
        j=j,
        n_bootstrap=n_bootstrap,
        temperature=temperature,
        verbose=verbose,
    )

    if verbose:
        print(f"      Mean pairwise ARI (seed stability): {seed_stability['mean_pairwise_ari']:.4f} "
              f"+/- {seed_stability['std_pairwise_ari']:.4f} "
              f"({seed_stability['n_pairs']} pairs)")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    seed_ari = seed_stability["mean_pairwise_ari"]
    if ari >= 0.8 * seed_ari:
        interpretation = (
            f"STABLE: Cross-window ARI ({ari:.3f}) is >= 80% of seed ARI ({seed_ari:.3f}). "
            "Types are essentially period-invariant — the same structure emerges from both eras."
        )
    elif ari >= 0.4:
        interpretation = (
            f"PARTIALLY STABLE: Cross-window ARI ({ari:.3f}) is moderate. "
            "Core type structure is preserved but types show meaningful drift across periods."
        )
    elif ari >= 0.15:
        interpretation = (
            f"WEAKLY STABLE: Cross-window ARI ({ari:.3f}) is low. "
            "Types share some structure but significant period-specific components exist."
        )
    else:
        interpretation = (
            f"UNSTABLE: Cross-window ARI ({ari:.3f}) approaches chance. "
            "Types appear to be period-specific artifacts rather than durable structures."
        )

    results = {
        # Sub-window metadata
        "j": j,
        "n_counties": X_full.shape[0],
        "split_year": split_year,
        "min_year": min_year,
        "n_early_dims": len(early_idx),
        "n_late_dims": len(late_idx),
        "n_full_dims": X_full.shape[1],
        "early_cols": early_cols,
        "late_cols": late_cols,
        # Cluster assignments
        "early_labels": early_labels,
        "late_labels": late_labels,
        # Cross-window metrics
        "cross_window_ari": ari,
        "cross_window_nmi": nmi,
        "county_stability_rate": stability,
        # Holdout r
        "early_holdout_r": early_holdout_r,
        "late_holdout_r": late_holdout_r,
        "full_holdout_r": full_holdout_r,
        # Seed stability baseline
        "seed_stability": seed_stability,
        # Interpretation
        "interpretation": interpretation,
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print("INTERPRETATION:")
        print(f"  {interpretation}")
        print(f"{'=' * 70}")

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _render_type_size_hist(labels: np.ndarray, j: int, bins: int = 10) -> str:
    """Render a simple text histogram of type sizes."""
    counts = np.bincount(labels, minlength=j)
    sorted_counts = sorted(counts.tolist(), reverse=True)
    max_count = max(sorted_counts) if sorted_counts else 1
    bar_width = 30
    lines = []
    for i, count in enumerate(sorted_counts[:15]):
        bar = "#" * int(bar_width * count / max_count)
        lines.append(f"  type_{i:02d} ({count:4d} counties): {bar}")
    if j > 15:
        lines.append(f"  ... ({j - 15} more types)")
    return "\n".join(lines)


def write_report(results: dict, output_path: Path) -> None:
    """Write a Markdown report of the experiment results.

    Parameters
    ----------
    results : dict
        Return value from run_type_stability_experiment().
    output_path : Path
        Path to write the Markdown report.
    """
    j = results["j"]
    n_counties = results["n_counties"]
    split_year = results["split_year"]
    min_year = results["min_year"]
    n_early = results["n_early_dims"]
    n_late = results["n_late_dims"]
    n_full = results["n_full_dims"]
    ari = results["cross_window_ari"]
    nmi = results["cross_window_nmi"]
    stability = results["county_stability_rate"]
    early_r = results["early_holdout_r"]
    late_r = results["late_holdout_r"]
    full_r = results["full_holdout_r"]
    ss = results["seed_stability"]
    interpretation = results["interpretation"]

    early_hist = _render_type_size_hist(results["early_labels"], j)
    late_hist = _render_type_size_hist(results["late_labels"], j)

    report = f"""# Type Stability: Sub-Window Analysis (P4.2)

**Session**: S175
**Date**: 2026-03-23
**Question**: Do the same electoral community types emerge from 2008-2016 data as from 2016-2024 data?

---

## Setup

| Parameter | Value |
|-----------|-------|
| Algorithm | KMeans |
| J (types) | {j} |
| Counties | {n_counties} |
| Presidential weight | {PRES_WEIGHT}x |
| Gov/Senate | state-centered |
| Temperature (T) | {TEMPERATURE} |
| Sub-window split | year {split_year} |
| Min training year | {min_year} |
| Holdout | `pres_*_shift_20_24` |
| Bootstrap seeds | {ss["n_runs"]} |

### Early Sub-Window (2008-2015): {n_early} dimensions

Columns: {", ".join(results["early_cols"][:6])} ... ({n_early} total)

### Late Sub-Window ({split_year}+): {n_late} dimensions

Columns: {", ".join(results["late_cols"])}

### Full Window: {n_full} dimensions (production baseline)

---

## Results

### Cross-Window Agreement

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Adjusted Rand Index (ARI) | **{ari:.4f}** | 1.0 = identical, 0.0 = random |
| Normalized Mutual Information (NMI) | **{nmi:.4f}** | 1.0 = identical partitions |
| County stability (Hungarian-matched) | **{stability:.1%}** | % counties in same type after optimal relabeling |

### Seed Stability Baseline (Full Window, {ss["n_runs"]} seeds)

| Metric | Value |
|--------|-------|
| Mean pairwise ARI | **{ss["mean_pairwise_ari"]:.4f}** |
| Std pairwise ARI | {ss["std_pairwise_ari"]:.4f} |
| Pairs evaluated | {ss["n_pairs"]} |

> The seed ARI measures how consistently KMeans finds the same types when only
> the random initialization changes. This is the ceiling for what "perfectly
> stable" types would look like under this algorithm.
>
> **Cross-window ARI / Seed ARI = {ari / ss["mean_pairwise_ari"]:.2f}x**
> (1.0 = sub-window types are as stable as seed variation alone)

### Holdout r (2020→2024 Presidential Shifts)

| Model | Holdout r | Notes |
|-------|-----------|-------|
| Early window (2008-2015) | **{early_r:.4f}** | Older data predicts modern shifts |
| Late window (2016+) | **{late_r:.4f}** | More recent data |
| Full window (production) | **{full_r:.4f}** | Combined 33-dim model |

---

## Type Size Distributions

### Early Window ({n_early} dims) — Top 15 types by size:

```
{early_hist}
```

### Late Window ({n_late} dims) — Top 15 types by size:

```
{late_hist}
```

---

## Interpretation

{interpretation}

### What This Means for Forecasting

- **Cross-window ARI = {ari:.3f}** vs **seed ARI = {ss["mean_pairwise_ari"]:.3f}**
  (ratio: {ari / ss["mean_pairwise_ari"]:.2f}x)

- Holdout r order: early ({early_r:.3f}) vs late ({late_r:.3f}) vs full ({full_r:.3f})
  {"Late > Early as expected: recent data is more predictive of recent shifts." if late_r > early_r
   else "Early >= Late: older structural patterns predict 2020->2024 better than recent shifts."}

- **Implication for 2026 predictions**: {"Types discovered from the full window are grounded in durable structure, not just recent momentum. They should generalize well to future elections." if ari >= 0.4 * ss["mean_pairwise_ari"] else "Types show significant period drift. Consider using time-weighted clustering (see experiment_temporal_weighting.py) for 2026 predictions."}

---

## Methodology Notes

- ARI and NMI are computed on hard cluster assignments (argmax of soft membership).
- County stability uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
  to find the optimal bijective mapping between early-window type labels and
  late-window type labels before computing the match rate.
- Bootstrap seed stability uses pairwise ARI over {ss["n_runs"]} runs × {ss["n_runs"] - 1} / 2 = {ss["n_pairs"]} pairs.
- Holdout r is mean Pearson r across the 3 holdout columns (D, R, turnout shifts).
- All clustering uses the production weighting: pres_* columns × {PRES_WEIGHT}, gov/sen state-centered.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Type stability sub-window experiment (P4.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--j", type=int, default=DEFAULT_J,
        help=f"Number of KMeans types (default: {DEFAULT_J})"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of bootstrap seeds for seed stability (default: {DEFAULT_N_BOOTSTRAP})"
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"Soft membership temperature (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--split-year", type=int, default=SPLIT_YEAR,
        help=f"Year separating early/late sub-windows (default: {SPLIT_YEAR})"
    )
    parser.add_argument(
        "--min-year", type=int, default=MIN_YEAR,
        help=f"Min start year for shift pairs (default: {MIN_YEAR})"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip writing the Markdown report"
    )
    args = parser.parse_args()

    results = run_type_stability_experiment(
        j=args.j,
        n_bootstrap=args.n_bootstrap,
        temperature=args.temperature,
        min_year=args.min_year,
        split_year=args.split_year,
        verbose=True,
    )

    if not args.no_report:
        report_path = DOCS_DIR / "type-stability-subwindows-S175.md"
        write_report(results, report_path)
        print(f"\nReport written to: {report_path}")
    else:
        print("\n(--no-report: skipping Markdown output)")


if __name__ == "__main__":
    main()
