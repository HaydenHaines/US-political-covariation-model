"""Experiment: J sweep with leave-one-pair-out cross-validation.

Finds the optimal number of KMeans clusters (J) for county electoral type
discovery. For each J in range(12, 31), holds out each election pair in turn,
trains KMeans on the remaining pairs, predicts the held-out pair using T=10
soft membership, and computes holdout Pearson r and calibration MAE.

Design:
- Data loading replicates the production pipeline exactly:
  presidential×2.5 weighting, state-centered gov/Senate, 2008+ only
- Soft membership uses T=10 (production default from experiment_soft_membership.py)
- Leave-one-pair-out: each fold holds out one election pair (3 columns)
- Prediction: type-mean holdout shift weighted by county soft membership
- Coherence: mean intra-type cosine similarity of county shift vectors
- Results printed to stdout and saved to data/experiments/j_sweep_results.csv

Usage:
    cd /home/hayden/projects/US-political-covariation-model
    uv run python scripts/experiment_j_sweep.py
    uv run python scripts/experiment_j_sweep.py --j-min 12 --j-max 30
    uv run python scripts/experiment_j_sweep.py --j-min 31 --j-max 50 --append
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent

SHIFTS_PATH = DATA_ROOT / "data/shifts/county_shifts_multiyear.parquet"
OUTPUT_DIR = DATA_ROOT / "data/experiments"

# ---------------------------------------------------------------------------
# Production pipeline constants — must match run_type_discovery.py exactly
# ---------------------------------------------------------------------------

KMEANS_SEED = 42
KMEANS_N_INIT = 10
PRES_WEIGHT = 2.5
MIN_YEAR = 2008
TEMPERATURE = 10.0

# The 2020→2024 shift is the final holdout used for blind validation;
# it must be excluded from all J sweep training AND from the CV pairs.
BLIND_HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_shift_matrix(
    min_year: int = MIN_YEAR,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Load and preprocess the county shift matrix.

    Applies the same preprocessing as the production pipeline:
    - Filter to shift pairs where the start year >= min_year
    - Exclude the blind holdout columns (2020→2024)
    - Apply presidential weight (PRES_WEIGHT) to pres_* columns
    - State-center governor and Senate columns

    Parameters
    ----------
    min_year : int
        Only include election pairs with start year >= this value.

    Returns
    -------
    X : ndarray of shape (N, D)
        Preprocessed shift matrix fed to KMeans.
    shift_cols : list[str]
        Column names (D columns, post-filter).
    county_fips : ndarray of shape (N,)
        County FIPS codes (zero-padded strings).
    weights_vec : ndarray of shape (D,)
        Per-column weights applied (for reference; already multiplied into X).
    """
    df = pd.read_parquet(SHIFTS_PATH)
    county_fips = df["county_fips"].astype(str).str.zfill(5).values

    all_cols = [c for c in df.columns if c != "county_fips" and c not in BLIND_HOLDOUT_COLUMNS]

    # Filter to min_year+ by parsing the start year from the column name.
    # Column pattern: <race>_<metric>_shift_<yy1>_<yy2>  e.g. pres_d_shift_08_12
    pair_re = re.compile(r"shift_(\d{2})_\d{2}$")
    shift_cols = []
    for c in all_cols:
        m = pair_re.search(c)
        if m:
            y1 = int("20" + m.group(1))
            if y1 >= min_year:
                shift_cols.append(c)

    if not shift_cols:
        shift_cols = all_cols  # fallback: shouldn't happen with real data

    X_raw = df[shift_cols].values.astype(float)

    # Presidential weight
    weights_vec = np.ones(len(shift_cols))
    for i, c in enumerate(shift_cols):
        if c.startswith("pres_"):
            weights_vec[i] = PRES_WEIGHT

    # State-center governor and Senate columns
    state_prefix = np.array([f[:2] for f in county_fips])
    gov_sen_mask = np.array([c.startswith("gov_") or c.startswith("sen_") for c in shift_cols])
    if gov_sen_mask.any():
        X_centered = X_raw.copy()
        for prefix in np.unique(state_prefix):
            idx = state_prefix == prefix
            col_idx = np.where(gov_sen_mask)[0]
            X_centered[np.ix_(idx, col_idx)] -= X_raw[np.ix_(idx, col_idx)].mean(axis=0)
        X_raw = X_centered

    X = X_raw * weights_vec[None, :]
    return X, shift_cols, county_fips, weights_vec


# ---------------------------------------------------------------------------
# CV fold generation
# ---------------------------------------------------------------------------


def group_columns_by_pair(column_names: list[str]) -> dict[str, list[int]]:
    """Group shift column indices by their election year pair suffix.

    Parameters
    ----------
    column_names : list[str]
        Shift column names (already filtered, without county_fips).

    Returns
    -------
    dict mapping pair key (e.g., '08_12') to list of column indices.
    """
    pair_re = re.compile(r"(\d{2}_\d{2})$")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, col in enumerate(column_names):
        m = pair_re.search(col)
        if m:
            groups[m.group(1)].append(i)
    return dict(groups)


def make_cv_folds(
    column_names: list[str],
    min_year: int = MIN_YEAR,
) -> list[tuple[str, list[int], list[int]]]:
    """Generate leave-one-pair-out CV folds.

    Each fold holds out one election pair's columns. Training uses all
    remaining columns. Only includes pairs where start year >= min_year.

    Parameters
    ----------
    column_names : list[str]
        All shift column names in the matrix (indexed 0..D-1).
    min_year : int
        Minimum start year for pairs to include (both train and holdout).

    Returns
    -------
    list of (pair_key, train_col_indices, holdout_col_indices) tuples.
    """
    pair_groups = group_columns_by_pair(column_names)
    all_col_indices = list(range(len(column_names)))

    folds = []
    for pair_key, holdout_cols in sorted(pair_groups.items()):
        # Parse start year from pair key (e.g., '08_12' → 2008)
        y1 = int("20" + pair_key.split("_")[0])
        if y1 < min_year:
            continue
        train_cols = [i for i in all_col_indices if i not in holdout_cols]
        folds.append((pair_key, train_cols, holdout_cols))
    return folds


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------


def temperature_soft_membership(dists: np.ndarray, T: float = TEMPERATURE) -> np.ndarray:
    """Compute temperature-sharpened soft membership from centroid distances.

    Matches the production implementation in run_type_discovery.py exactly.

    Parameters
    ----------
    dists : ndarray of shape (N, J)
        Euclidean distances from each county to each centroid.
    T : float
        Temperature exponent. T=10.0 is the production default.

    Returns
    -------
    scores : ndarray of shape (N, J)
        Non-negative membership weights, each row sums to 1.
    """
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


def fit_kmeans_get_distances(
    X_train: np.ndarray,
    j: int,
    random_state: int = KMEANS_SEED,
    n_init: int = KMEANS_N_INIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans on training data, return centroid distances for ALL counties.

    The KMeans is fit on X_train (a column-subset of the full matrix).
    Distances are computed from the full (N, D_train) matrix to each centroid.

    Parameters
    ----------
    X_train : ndarray of shape (N, D_train)
        Training columns for all counties.
    j : int
        Number of clusters.
    random_state : int
    n_init : int

    Returns
    -------
    centroids : ndarray of shape (J, D_train)
    labels : ndarray of shape (N,) — hard cluster assignments on training data
    """
    km = KMeans(n_clusters=j, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X_train)
    centroids = km.cluster_centers_
    return centroids, labels


def compute_centroid_distances(
    X: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distances from each row of X to each centroid.

    Parameters
    ----------
    X : ndarray of shape (N, D)
    centroids : ndarray of shape (J, D)

    Returns
    -------
    dists : ndarray of shape (N, J)
    """
    J = centroids.shape[0]
    N = X.shape[0]
    dists = np.zeros((N, J))
    for t in range(J):
        dists[:, t] = np.linalg.norm(X - centroids[t], axis=1)
    return dists


def predict_holdout_columns(
    scores: np.ndarray,
    X_holdout: np.ndarray,
) -> np.ndarray:
    """Predict held-out shift columns as type-mean weighted by membership.

    For each county i and held-out column d:
        pred[i, d] = sum_j(score[i,j] * type_mean[j, d])

    where type_mean[j, d] = sum_i(score[i,j] * X_holdout[i,d]) / sum_i(score[i,j])

    Parameters
    ----------
    scores : ndarray of shape (N, J)
        Soft membership (rows sum to 1).
    X_holdout : ndarray of shape (N, D_holdout)
        Actual held-out shift values.

    Returns
    -------
    predicted : ndarray of shape (N, D_holdout)
    """
    J = scores.shape[1]
    weight_sums = scores.sum(axis=0)  # (J,)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    # type_means[j, d] = weighted mean of column d for type j
    type_means = (scores.T @ X_holdout) / weight_sums[:, None]  # (J, D_holdout)
    return scores @ type_means  # (N, D_holdout)


def compute_holdout_r(
    X_holdout_actual: np.ndarray,
    X_holdout_predicted: np.ndarray,
) -> float:
    """Mean Pearson r across holdout columns.

    Parameters
    ----------
    X_holdout_actual : ndarray of shape (N, D_holdout)
    X_holdout_predicted : ndarray of shape (N, D_holdout)

    Returns
    -------
    float — mean Pearson r (returns 0.0 if any column has zero variance).
    """
    D = X_holdout_actual.shape[1]
    r_vals = []
    for d in range(D):
        a = X_holdout_actual[:, d]
        p = X_holdout_predicted[:, d]
        if np.std(a) < 1e-10 or np.std(p) < 1e-10:
            r_vals.append(0.0)
        else:
            r, _ = pearsonr(a, p)
            r_vals.append(float(r))
    return float(np.mean(r_vals))


def compute_holdout_mae(
    X_holdout_actual: np.ndarray,
    X_holdout_predicted: np.ndarray,
) -> float:
    """Mean absolute error across all holdout column values.

    Parameters
    ----------
    X_holdout_actual : ndarray of shape (N, D_holdout)
    X_holdout_predicted : ndarray of shape (N, D_holdout)

    Returns
    -------
    float — mean absolute error (scalar).
    """
    return float(np.mean(np.abs(X_holdout_actual - X_holdout_predicted)))


def compute_type_coherence(
    X: np.ndarray,
    labels: np.ndarray,
    j: int,
) -> float:
    """Mean intra-type cosine similarity of county shift vectors.

    For each type with >= 2 members, computes the mean pairwise cosine
    similarity among its member counties, then averages across types
    (weighted by type size).

    Parameters
    ----------
    X : ndarray of shape (N, D) — the training shift matrix
    labels : ndarray of shape (N,) — hard KMeans assignments
    j : int — number of types

    Returns
    -------
    float — weighted mean intra-type cosine similarity.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = X / norms

    coherence_vals = []
    weights = []
    for t in range(j):
        idx = np.where(labels == t)[0]
        if len(idx) < 2:
            continue
        members = X_norm[idx]  # (n_t, D)
        # Mean of all pairwise cosine sims = (sum of all dot products - diagonal) / (n*(n-1))
        gram = members @ members.T  # (n_t, n_t)
        n_t = len(idx)
        off_diag_sum = gram.sum() - np.trace(gram)
        mean_cos = off_diag_sum / (n_t * (n_t - 1))
        coherence_vals.append(mean_cos)
        weights.append(n_t)

    if not coherence_vals:
        return float("nan")
    weights_arr = np.array(weights, dtype=float)
    return float(np.average(coherence_vals, weights=weights_arr))


# ---------------------------------------------------------------------------
# J sweep
# ---------------------------------------------------------------------------


def run_j_sweep_fold(
    X: np.ndarray,
    shift_cols: list[str],
    j: int,
    fold: tuple[str, list[int], list[int]],
    temperature: float = TEMPERATURE,
) -> dict[str, float]:
    """Run one fold of the J sweep for a given J.

    Parameters
    ----------
    X : ndarray of shape (N, D) — full preprocessed shift matrix
    shift_cols : list[str] — column names (length D)
    j : int — number of clusters
    fold : (pair_key, train_col_indices, holdout_col_indices)
    temperature : float — soft membership temperature

    Returns
    -------
    dict with keys: pair_key, holdout_r, holdout_mae
    """
    pair_key, train_cols, holdout_cols = fold

    X_train = X[:, train_cols]
    X_holdout = X[:, holdout_cols]

    centroids, labels = fit_kmeans_get_distances(X_train, j)
    dists = compute_centroid_distances(X_train, centroids)
    scores = temperature_soft_membership(dists, T=temperature)

    X_holdout_pred = predict_holdout_columns(scores, X_holdout)

    holdout_r = compute_holdout_r(X_holdout, X_holdout_pred)
    holdout_mae = compute_holdout_mae(X_holdout, X_holdout_pred)

    return {
        "pair_key": pair_key,
        "holdout_r": holdout_r,
        "holdout_mae": holdout_mae,
    }


def run_j_sweep(
    j_range: range | list[int] = range(12, 31),
    min_year: int = MIN_YEAR,
    temperature: float = TEMPERATURE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full J sweep with leave-one-pair-out CV.

    For each J, runs all CV folds and averages metrics across folds. Also
    computes type coherence on the full training matrix.

    Parameters
    ----------
    j_range : range or list of ints
        J values to evaluate.
    min_year : int
        Only use election pairs with start year >= this.
    temperature : float
        Soft membership temperature.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    DataFrame with columns:
        j, mean_r, std_r, mean_mae, std_mae, coherence, n_folds
    """
    if verbose:
        print(f"Loading shift matrix (min_year={min_year})...")
    X, shift_cols, county_fips, weights_vec = load_shift_matrix(min_year=min_year)
    if verbose:
        print(f"  Shape: {X.shape[0]} counties x {X.shape[1]} dims")

    folds = make_cv_folds(shift_cols, min_year=min_year)
    if verbose:
        fold_keys = [f[0] for f in folds]
        print(f"  CV folds ({len(folds)}): {fold_keys}")
        print()

    j_list = list(j_range)
    rows = []

    for j in j_list:
        if verbose:
            print(f"  J={j:2d} ...", end="", flush=True)

        fold_results = []
        for fold in folds:
            try:
                result = run_j_sweep_fold(X, shift_cols, j, fold, temperature=temperature)
                fold_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"\n    WARNING: fold {fold[0]} failed for J={j}: {e}")

        if not fold_results:
            rows.append({
                "j": j,
                "mean_r": float("nan"),
                "std_r": float("nan"),
                "mean_mae": float("nan"),
                "std_mae": float("nan"),
                "coherence": float("nan"),
                "n_folds": 0,
            })
            if verbose:
                print(" no valid folds")
            continue

        r_vals = [fr["holdout_r"] for fr in fold_results]
        mae_vals = [fr["holdout_mae"] for fr in fold_results]

        # Coherence on full matrix (all training columns, hard KMeans labels)
        centroids_full, labels_full = fit_kmeans_get_distances(X, j)
        coherence = compute_type_coherence(X, labels_full, j)

        row = {
            "j": j,
            "mean_r": float(np.mean(r_vals)),
            "std_r": float(np.std(r_vals)),
            "mean_mae": float(np.mean(mae_vals)),
            "std_mae": float(np.std(mae_vals)),
            "coherence": coherence,
            "n_folds": len(fold_results),
        }
        rows.append(row)

        if verbose:
            print(
                f" r={row['mean_r']:.4f} ± {row['std_r']:.4f}"
                f"  MAE={row['mean_mae']:.4f} ± {row['std_mae']:.4f}"
                f"  coh={row['coherence']:.4f}"
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(results: pd.DataFrame, best_j: int) -> None:
    """Print formatted results table to stdout."""
    print()
    print("=" * 90)
    print("J SWEEP — Leave-One-Pair-Out CV (KMeans, T=10 soft membership, 2008+)")
    print("=" * 90)
    header = (
        f"{'J':>4}  {'mean_r':>8}  {'std_r':>7}  {'mean_MAE':>9}  "
        f"{'std_MAE':>8}  {'coherence':>9}  {'n_folds':>7}"
    )
    print(header)
    print("-" * 90)
    for _, row in results.iterrows():
        marker = " <-- BEST" if int(row["j"]) == best_j else ""
        print(
            f"{int(row['j']):>4}  "
            f"{row['mean_r']:>8.4f}  "
            f"{row['std_r']:>7.4f}  "
            f"{row['mean_mae']:>9.4f}  "
            f"{row['std_mae']:>8.4f}  "
            f"{row['coherence']:>9.4f}  "
            f"{int(row['n_folds']):>7}"
            f"{marker}"
        )
    print()
    print(f"Best J by mean holdout r: J={best_j}")
    print()


def save_results(results: pd.DataFrame, output_dir: Path = OUTPUT_DIR, append: bool = False) -> Path:
    """Save results DataFrame to CSV.

    Parameters
    ----------
    results : DataFrame
        New results to save.
    output_dir : Path
        Directory to write to.
    append : bool
        If True and the CSV already exists, merge new rows (by j) with existing
        rows, replacing any duplicate J values with the new data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "j_sweep_results.csv"
    if append and out_path.exists():
        existing = pd.read_csv(out_path)
        # Drop any J values that are being re-run (new data wins)
        existing = existing[~existing["j"].isin(results["j"])]
        combined = pd.concat([existing, results], ignore_index=True)
        combined = combined.sort_values("j").reset_index(drop=True)
        combined.to_csv(out_path, index=False)
        return out_path
    results.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="J sweep: leave-one-pair-out CV for KMeans")
    parser.add_argument("--j-min", type=int, default=12, help="Minimum J to test (default: 12)")
    parser.add_argument("--j-max", type=int, default=30, help="Maximum J to test (default: 30)")
    parser.add_argument("--min-year", type=int, default=MIN_YEAR, help=f"Min start year for election pairs (default: {MIN_YEAR})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help=f"Soft membership temperature (default: {TEMPERATURE})")
    parser.add_argument("--append", action="store_true", help="Append new results to existing CSV instead of overwriting (default: False)")
    args = parser.parse_args()

    j_range = range(args.j_min, args.j_max + 1)

    print(f"J sweep: J={args.j_min}..{args.j_max}, T={args.temperature}, min_year={args.min_year}")
    print()

    results = run_j_sweep(
        j_range=j_range,
        min_year=args.min_year,
        temperature=args.temperature,
        verbose=True,
    )

    # Best J = highest mean_r among valid rows
    valid = results.dropna(subset=["mean_r"])
    if len(valid) == 0:
        print("ERROR: no valid J results produced.")
        return
    best_j = int(valid.loc[valid["mean_r"].idxmax(), "j"])

    print_results(results, best_j)

    out_path = save_results(results, append=args.append)
    print(f"Results saved to {out_path}")

    print()
    print("RECOMMENDATION")
    print("-" * 40)
    best_row = valid.loc[valid["mean_r"].idxmax()]
    current_j = 20
    if best_j == current_j:
        print(f"  Current J={current_j} is already optimal.")
    else:
        current_row = results[results["j"] == current_j]
        if len(current_row) > 0:
            cur_r = float(current_row.iloc[0]["mean_r"])
            print(f"  Recommend changing J from {current_j} → {best_j}.")
            print(f"  Holdout r: {cur_r:.4f} (J={current_j}) → {float(best_row['mean_r']):.4f} (J={best_j})")
        else:
            print(f"  Recommend J={best_j} (mean holdout r={float(best_row['mean_r']):.4f}).")


if __name__ == "__main__":
    main()
