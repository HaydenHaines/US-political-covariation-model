"""KMeans type stability experiment — WetherVane.

Research question: How stable are KMeans type assignments across different
random seeds? If types are unstable (low ARI/NMI), the clustering is partly
driven by KMeans initialisation noise rather than genuine electoral structure.

High stability (ARI > 0.8) would mean the types are real and consistent;
low stability (ARI < 0.5) would mean the electoral landscape does not have
enough cluster separation to reliably anchor 100 types.

Design
------
- Runs the production type discovery pipeline 5 times, one per random seed.
- For each non-reference seed, computes vs. reference seed (seed=42):
    - Adjusted Rand Index (ARI): agreement between two clusterings (1.0=identical,
      0.0=random). Corrects for chance; robust to imbalanced cluster sizes.
    - Normalized Mutual Information (NMI): information-theoretic agreement
      (1.0=identical, 0.0=independent).
- County stability: for each county, what fraction of the non-reference seeds
  place it in the same cluster as the reference? Uses the Hungarian algorithm
  to align type IDs across runs before counting agreement.
- Holdout LOO r per seed: checks that prediction quality is seed-stable.

Hungarian alignment
-------------------
KMeans type IDs are arbitrary labels — "type 7" in one run may correspond
to "type 31" in another. The Hungarian algorithm (linear_sum_assignment on a
negative confusion matrix) finds the optimal 1-to-1 mapping from each run's
type IDs to the reference run's type IDs. After remapping, agreement counts
are meaningful.

Usage
-----
    uv run python scripts/experiment_type_stability.py
    uv run python scripts/experiment_type_stability.py --seeds 42 123 456
    uv run python scripts/experiment_type_stability.py --j 100
    uv run python scripts/experiment_type_stability.py --no-pca
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants — must match production pipeline in run_type_discovery.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments"

# The 2020→2024 shift columns are held out for accuracy evaluation.
# They are excluded from type discovery (as in production) to prevent leakage.
HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

# Production hyperparameters — keep in sync with run_type_discovery.py
# and config/model.yaml. If the model changes, update these constants.
J = 100                # Number of types (KMeans clusters)
MIN_YEAR = 2008        # Exclude shift pairs with start year before this
PRES_WEIGHT = 8.0      # Presidential shift amplifier (post-StandardScaler)
PCA_COMPONENTS = 15    # PCA dimensionality reduction before KMeans
PCA_WHITEN = True      # Whitening: each PC component scaled to unit variance
TEMPERATURE = 10.0     # Soft membership sharpening exponent
KMEANS_N_INIT = 10     # Number of KMeans random initialisations per seed

# Reference seed (seed=42 is the production seed). All other seeds are
# compared against this one for ARI, NMI, and county stability.
REFERENCE_SEED = 42

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Data loading — replicates run_type_discovery.py preprocessing exactly
# ---------------------------------------------------------------------------


def load_shift_matrix(
    min_year: int = MIN_YEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load county shift matrix and apply production preprocessing.

    Replicates the exact preprocessing chain from run_type_discovery.py:
      1. Load all shift columns (excluding county_fips and holdout columns).
      2. Filter to pairs with start year >= min_year (default 2008).
      3. Apply StandardScaler (zero mean, unit variance per column).
      4. Multiply presidential shift columns by PRES_WEIGHT (post-scaling).

    Returns
    -------
    train_matrix : ndarray (N, D)
        Scaled + weighted shift matrix for clustering (excludes holdout).
    holdout_matrix : ndarray (N, 3)
        Raw (unscaled) holdout columns for LOO accuracy measurement.
    train_matrix_raw : ndarray (N, D)
        Raw (unscaled) training columns, used for county-level LOO priors.
        LOO priors are computed in raw log-odds space to match holdout units.
    train_cols : list[str]
        Column names corresponding to train_matrix dimensions.
    """
    df = pd.read_parquet(SHIFTS_PATH)

    all_shift_cols = [c for c in df.columns if c != "county_fips"]

    # Separate holdout from training; filter training to recent pairs
    train_cols = []
    for col in all_shift_cols:
        if col in HOLDOUT_COLUMNS:
            continue
        # Column suffix pattern: e.g. "pres_d_shift_08_12" → start_year_2digit = "08"
        parts = col.split("_")
        y2_str = parts[-2]
        y2_int = int(y2_str)
        # Two-digit years: 00-49 → 2000s, 50-99 → 1900s (no shifts before ~1996)
        start_year = y2_int + (1900 if y2_int >= 50 else 2000)
        if start_year >= min_year:
            train_cols.append(col)

    train_matrix_raw = df[train_cols].values.astype(float)
    holdout_matrix = df[HOLDOUT_COLUMNS].values.astype(float)

    # StandardScaler: normalise each dimension to zero mean + unit variance.
    # Without this, gov/senate shifts (different raw magnitudes) would dominate
    # KMeans Euclidean distance and mask the presidential covariation signal.
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_matrix_raw.copy())

    # Post-scaling presidential weight amplifies the cross-state covariation
    # signal that is the core of type discovery. Applied after scaling so the
    # amplification is relative (8× heavier) rather than additive.
    pres_indices = [i for i, c in enumerate(train_cols) if "pres_" in c]
    if PRES_WEIGHT != 1.0:
        train_matrix[:, pres_indices] *= PRES_WEIGHT

    n_counties, n_dims = train_matrix.shape
    n_pres = len(pres_indices)
    print(
        f"Loaded {n_counties} counties × {n_dims} training dims "
        f"({n_pres} presidential, {n_dims - n_pres} off-cycle; min_year={min_year})"
    )

    return train_matrix, holdout_matrix, train_matrix_raw, train_cols


# ---------------------------------------------------------------------------
# Soft membership (temperature-scaled inverse distance)
# ---------------------------------------------------------------------------


def temperature_soft_membership(dists: np.ndarray, T: float = TEMPERATURE) -> np.ndarray:
    """Compute temperature-sharpened soft membership from centroid distances.

    Replicates src/discovery/run_type_discovery.py::temperature_soft_membership.
    Formula: weight_j = (1 / (dist_j + eps))^T, then row-normalise.
    Uses log-space arithmetic for numerical stability at T=10.

    Parameters
    ----------
    dists : ndarray (N, J)
        Euclidean distances from each county to each centroid.
    T : float
        Temperature exponent. T=10 is production default; T→∞ approaches
        hard (argmax) assignment.

    Returns
    -------
    scores : ndarray (N, J)
        Non-negative weights, each row sums to 1.
    """
    N, J_ = dists.shape
    eps = 1e-10

    if T >= 500.0:
        # Hard assignment shortcut at extreme temperatures
        scores = np.zeros((N, J_))
        scores[np.arange(N), np.argmin(dists, axis=1)] = 1.0
        return scores

    # Log-space: log(weight_j) = T * log(1/(dist_j + eps)) = -T * log(dist_j + eps)
    log_weights = -T * np.log(dists + eps)  # (N, J)

    # Numerically stable softmax: subtract row max before exponentiating
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    return powered / np.where(row_sums == 0, 1.0, row_sums)


# ---------------------------------------------------------------------------
# Single-seed clustering
# ---------------------------------------------------------------------------


def run_one_seed(
    train_matrix: np.ndarray,
    random_state: int,
    j: int = J,
    pca_components: int | None = PCA_COMPONENTS,
    pca_whiten: bool = PCA_WHITEN,
    temperature: float = TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the production clustering pipeline for one random seed.

    Applies optional PCA before KMeans (matching the current production
    config of PCA(n=15, whiten=True) → KMeans(J=100)).

    Parameters
    ----------
    train_matrix : ndarray (N, D)
        Pre-scaled + weighted shift matrix.
    random_state : int
        Random seed for both PCA (if used) and KMeans.
    j : int
        Number of clusters.
    pca_components : int or None
        PCA dimensionality before KMeans. None = skip PCA.
    pca_whiten : bool
        Whiten PCA components (unit variance per PC).
    temperature : float
        Soft membership sharpening exponent.

    Returns
    -------
    labels : ndarray (N,)
        Hard cluster assignment (dominant type) for each county.
    weights : ndarray (N, J)
        Soft membership scores (temperature-scaled inverse distance).
    """
    # Optional PCA — same random_state as KMeans for reproducibility isolation
    if pca_components is not None and pca_components < train_matrix.shape[1]:
        pca = PCA(n_components=pca_components, whiten=pca_whiten, random_state=random_state)
        reduced = pca.fit_transform(train_matrix)
    else:
        reduced = train_matrix

    km = KMeans(n_clusters=j, random_state=random_state, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(reduced)
    centroids = km.cluster_centers_

    # Soft membership via temperature-scaled inverse distance to centroids
    dists = np.zeros((len(reduced), j))
    for t in range(j):
        dists[:, t] = np.linalg.norm(reduced - centroids[t], axis=1)
    weights = temperature_soft_membership(dists, T=temperature)

    return labels, weights


# ---------------------------------------------------------------------------
# Hungarian alignment of type IDs across runs
# ---------------------------------------------------------------------------


def align_labels_to_reference(
    ref_labels: np.ndarray,
    other_labels: np.ndarray,
    j: int = J,
) -> np.ndarray:
    """Remap other_labels to best match ref_labels using the Hungarian algorithm.

    KMeans type IDs are arbitrary labels — "type 7" in one run may correspond
    to "type 31" in another. The Hungarian algorithm finds the optimal 1-to-1
    mapping from other_labels IDs to ref_labels IDs by maximising co-assignment
    count (i.e., maximising the sum of the confusion matrix diagonal after
    permutation).

    Parameters
    ----------
    ref_labels : ndarray (N,)
        Reference clustering (seed=42). Integer type IDs in [0, J).
    other_labels : ndarray (N,)
        Another clustering to align. Integer type IDs in [0, J).
    j : int
        Number of types (cluster count).

    Returns
    -------
    aligned_labels : ndarray (N,)
        other_labels with type IDs remapped to match ref_labels.
        After remapping, ref_labels == aligned_labels gives county-level agreement.
    """
    # C[i, k] = number of counties where ref_label=i and other_label=k
    confusion = np.zeros((j, j), dtype=np.int64)
    for ref_id, other_id in zip(ref_labels, other_labels):
        confusion[ref_id, other_id] += 1

    # Hungarian maximisation: linear_sum_assignment minimises cost, so negate
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # mapping[other_id] = which reference label it maps to
    mapping = np.zeros(j, dtype=np.int64)
    for ref_id, other_id in zip(row_ind, col_ind):
        mapping[other_id] = ref_id

    return mapping[other_labels]


# ---------------------------------------------------------------------------
# LOO r computation (reused from experiment_pca_before_kmeans.py)
# ---------------------------------------------------------------------------


def compute_loo_r(
    weights: np.ndarray,
    holdout_matrix: np.ndarray,
    train_matrix_raw: np.ndarray,
) -> float:
    """Compute LOO holdout r from pre-computed soft membership weights.

    For each county i, removes i from the type means before predicting it.
    This eliminates the ~0.22 inflation from type self-prediction in small
    types (S196 LOO honesty rule).

    The county prior is its own mean raw shift (log-odds space, same units as
    holdout). Type adjustment = (type mean on holdout) - (type mean on training),
    both computed without county i. Prediction: prior + score-weighted adjustment.

    Parameters
    ----------
    weights : ndarray (N, J)
        Soft membership weights from clustering.
    holdout_matrix : ndarray (N, D_holdout)
        Raw holdout shift columns.
    train_matrix_raw : ndarray (N, D_train)
        Raw (unscaled) training shift columns for county-level priors.

    Returns
    -------
    mean_loo_r : float
        Mean Pearson r across holdout columns.
    """
    n, j = weights.shape

    # County-level training prior (mean raw shift across training pairs)
    county_training_means = train_matrix_raw.mean(axis=1)  # (N,)

    # Precompute global weighted sums for O(N) LOO instead of O(N²)
    global_weight_sums = weights.sum(axis=0)          # (J,)
    global_weighted_train = weights.T @ county_training_means  # (J,)

    per_col_loo_r: list[float] = []

    for col_idx in range(holdout_matrix.shape[1]):
        actual = holdout_matrix[:, col_idx]
        global_weighted_hold = weights.T @ actual  # (J,)

        predicted = np.zeros(n)
        for i in range(n):
            # LOO: subtract county i's contribution before computing type means
            loo_ws = global_weight_sums - weights[i]
            loo_ws = np.where(loo_ws < 1e-12, 1e-12, loo_ws)  # avoid division by zero
            loo_train = (global_weighted_train - weights[i] * county_training_means[i]) / loo_ws
            loo_hold = (global_weighted_hold - weights[i] * actual[i]) / loo_ws
            type_adj = loo_hold - loo_train
            predicted[i] = county_training_means[i] + (weights[i] * type_adj).sum()

        if np.std(actual) < 1e-10 or np.std(predicted) < 1e-10:
            per_col_loo_r.append(0.0)
        else:
            r, _ = pearsonr(actual, predicted)
            per_col_loo_r.append(float(np.clip(r, -1.0, 1.0)))

    return float(np.mean(per_col_loo_r))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KMeans type stability experiment — WetherVane"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Random seeds to test (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--j",
        type=int,
        default=J,
        help=f"KMeans cluster count (default: {J})",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Disable PCA (test raw feature clustering)",
    )
    args = parser.parse_args()

    seeds = args.seeds
    j = args.j
    pca_components = None if args.no_pca else PCA_COMPONENTS
    pca_whiten = False if args.no_pca else PCA_WHITEN

    print("=" * 70)
    print("KMeans Type Stability Experiment — WetherVane")
    print("=" * 70)
    pca_tag = f"PCA(n={pca_components}, whiten={pca_whiten})" if pca_components else "no PCA"
    print(f"Config: J={j}, seeds={seeds}, {pca_tag}")
    print(f"Reference seed: {REFERENCE_SEED}")
    print()

    # Load data (read-only — no production files are modified)
    train_matrix, holdout_matrix, train_matrix_raw, _ = load_shift_matrix()

    # -------------------------------------------------------------------
    # Step 1: Run clustering for each seed
    # -------------------------------------------------------------------

    print("\nRunning clustering for each seed...")
    all_labels: dict[int, np.ndarray] = {}
    all_weights: dict[int, np.ndarray] = {}

    for seed in seeds:
        t0 = time.time()
        labels, weights = run_one_seed(
            train_matrix,
            random_state=seed,
            j=j,
            pca_components=pca_components,
            pca_whiten=pca_whiten,
        )
        elapsed = time.time() - t0
        all_labels[seed] = labels
        all_weights[seed] = weights
        print(f"  seed={seed:5d}: {elapsed:.1f}s")

    # -------------------------------------------------------------------
    # Step 2: ARI, NMI, and LOO r vs reference seed
    # -------------------------------------------------------------------

    print()
    ref_labels = all_labels[REFERENCE_SEED]

    # Per-seed results table
    seed_rows: list[dict] = []

    for seed in seeds:
        labels = all_labels[seed]
        weights = all_weights[seed]

        if seed == REFERENCE_SEED:
            # Self-comparison is trivially perfect
            ari = 1.0
            nmi = 1.0
        else:
            ari = float(adjusted_rand_score(ref_labels, labels))
            nmi = float(normalized_mutual_info_score(
                ref_labels, labels, average_method="arithmetic"
            ))

        # LOO r uses the actual weights from this seed's clustering
        loo_r = compute_loo_r(weights, holdout_matrix, train_matrix_raw)

        seed_rows.append({
            "seed": seed,
            "loo_r": loo_r,
            "ari_vs_ref": ari,
            "nmi_vs_ref": nmi,
        })

    # -------------------------------------------------------------------
    # Step 3: County stability via Hungarian-aligned agreement
    # -------------------------------------------------------------------

    print("Aligning type IDs across seeds via Hungarian algorithm...")
    n_counties = len(ref_labels)
    # Count how many non-reference seeds agree with the reference for each county
    agreement_counts = np.zeros(n_counties, dtype=np.int32)

    non_ref_seeds = [s for s in seeds if s != REFERENCE_SEED]
    for seed in non_ref_seeds:
        aligned = align_labels_to_reference(ref_labels, all_labels[seed], j=j)
        agreement_counts += (aligned == ref_labels).astype(np.int32)

    n_non_ref = len(non_ref_seeds)

    # -------------------------------------------------------------------
    # Step 4: Print results
    # -------------------------------------------------------------------

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print()

    # Per-seed table
    header = f"{'Seed':>6} | {'LOO r':>7} | {'ARI vs s42':>10} | {'NMI vs s42':>10}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for row in seed_rows:
        seed_marker = " (ref)" if row["seed"] == REFERENCE_SEED else "      "
        print(
            f"{row['seed']:>6}{seed_marker} | "
            f"{row['loo_r']:>7.4f} | "
            f"{row['ari_vs_ref']:>10.4f} | "
            f"{row['nmi_vs_ref']:>10.4f}"
        )
    print()

    # County stability distribution
    # agreement_counts[i] = number of non-reference seeds that agree with ref.
    # "k/n_non_ref + 1 total seeds" agree when agreement_counts[i] == k.
    total_seeds = len(seeds)
    print("County stability (agreement with reference seed=42):")
    print(f"  (comparing {n_non_ref} non-reference seeds against seed={REFERENCE_SEED})")
    print()

    for k in range(n_non_ref, -1, -1):
        # k non-ref seeds + the reference itself = (k+1) total seeds agree
        count = int((agreement_counts == k).sum())
        frac = count / n_counties
        total_agree = k + 1  # include ref
        bar = "#" * int(frac * 40)
        print(
            f"  {total_agree}/{total_seeds} agreement: {frac:>6.1%} "
            f"({count:>5,} counties)  {bar}"
        )

    print()

    # Overall ARI and NMI summary (excluding reference self-comparison)
    non_ref_rows = [r for r in seed_rows if r["seed"] != REFERENCE_SEED]
    ari_values = [r["ari_vs_ref"] for r in non_ref_rows]
    nmi_values = [r["nmi_vs_ref"] for r in non_ref_rows]
    loo_r_values = [r["loo_r"] for r in seed_rows]

    print(
        f"Overall: ARI mean={np.mean(ari_values):.4f} ± {np.std(ari_values):.4f}, "
        f"NMI mean={np.mean(nmi_values):.4f} ± {np.std(nmi_values):.4f}"
    )
    print(
        f"LOO r across seeds: mean={np.mean(loo_r_values):.4f} "
        f"± {np.std(loo_r_values):.4f} "
        f"(range: {min(loo_r_values):.4f}–{max(loo_r_values):.4f})"
    )

    print()

    # Interpretation — what does the mean ARI actually tell us?
    mean_ari = float(np.mean(ari_values))
    if mean_ari >= 0.80:
        interpretation = (
            "HIGH stability (ARI >= 0.80): types reflect genuine electoral structure "
            "more than initialisation noise."
        )
    elif mean_ari >= 0.60:
        interpretation = (
            "MODERATE stability (0.60 <= ARI < 0.80): types have real structure but "
            "some clusters are near boundaries and flip with different seeds."
        )
    elif mean_ari >= 0.40:
        interpretation = (
            "LOW stability (0.40 <= ARI < 0.60): many types are weakly separated. "
            "Consider reducing J or using a more stable algorithm."
        )
    else:
        interpretation = (
            "VERY LOW stability (ARI < 0.40): clustering is largely driven by "
            "initialisation noise. J is likely too large for the data structure."
        )
    print(f"Interpretation: {interpretation}")

    # -------------------------------------------------------------------
    # Step 5: Save results
    # -------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(seed_rows)
    results_path = OUTPUT_DIR / "type_stability_seed_results.parquet"
    results_df.to_parquet(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Per-county agreement data for further investigation
    ref_county_fips = pd.read_parquet(SHIFTS_PATH)["county_fips"].values
    county_df = pd.DataFrame({
        "county_fips": ref_county_fips,
        "ref_type": ref_labels,
        # Fraction of non-reference seeds that agree with the reference assignment
        "agreement_fraction": (
            agreement_counts / n_non_ref if n_non_ref > 0 else np.ones(n_counties)
        ),
        "agreement_count_nonref": agreement_counts,
    })
    county_path = OUTPUT_DIR / "type_stability_seed_county.parquet"
    county_df.to_parquet(county_path, index=False)
    print(f"County stability saved to {county_path}")


if __name__ == "__main__":
    main()
