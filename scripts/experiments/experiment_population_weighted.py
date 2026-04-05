"""Experiment: population-weighted KMeans clustering.

Hypothesis: weighting counties by population (or log-population) in KMeans
causes the clustering to prioritize large counties, potentially improving
holdout accuracy because the actual electorate is dominated by large counties.

Current approach: all 293 FL+GA+AL counties are weighted equally in KMeans.
Alternatives tested:
  - unweighted (baseline)
  - sample_weight = population (raw)
  - sample_weight = log(population) (compromise)

Evaluation: leave-one-pair-out CV at J=20 and J=43, using T=10 soft membership.
Metrics: mean holdout Pearson r and calibration MAE across all CV folds.

Usage:
    cd /home/hayden/projects/US-political-covariation-model
    uv run python scripts/experiment_population_weighted.py
    uv run python scripts/experiment_population_weighted.py --j-values 20 43
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
CENSUS_PATH = DATA_ROOT / "data/assembled/census_2020.parquet"
OUTPUT_DIR = DATA_ROOT / "data/experiments"

# ---------------------------------------------------------------------------
# Production pipeline constants — must match run_type_discovery.py exactly
# ---------------------------------------------------------------------------

KMEANS_SEED = 42
KMEANS_N_INIT = 10
PRES_WEIGHT = 2.5
MIN_YEAR = 2008
TEMPERATURE = 10.0

BLIND_HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

# Weighting schemes tested
WEIGHT_SCHEMES = ["unweighted", "population", "log_population"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_shift_matrix(
    min_year: int = MIN_YEAR,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load and preprocess the county shift matrix.

    Applies the same preprocessing as the production pipeline:
    - Filter to shift pairs where start year >= min_year
    - Exclude the blind holdout columns (2020->2024)
    - Apply presidential weight (PRES_WEIGHT) to pres_* columns
    - State-center governor and Senate columns

    Returns
    -------
    X : ndarray of shape (N, D)
        Preprocessed shift matrix fed to KMeans.
    shift_cols : list[str]
        Column names (D columns, post-filter).
    county_fips : ndarray of shape (N,)
        County FIPS codes (zero-padded strings).
    """
    df = pd.read_parquet(SHIFTS_PATH)
    county_fips = df["county_fips"].astype(str).str.zfill(5).values

    all_cols = [c for c in df.columns if c != "county_fips" and c not in BLIND_HOLDOUT_COLUMNS]

    pair_re = re.compile(r"shift_(\d{2})_\d{2}$")
    shift_cols = []
    for c in all_cols:
        m = pair_re.search(c)
        if m:
            y1 = int("20" + m.group(1))
            if y1 >= min_year:
                shift_cols.append(c)

    if not shift_cols:
        shift_cols = all_cols

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
    return X, shift_cols, county_fips


def load_population_weights(county_fips: np.ndarray) -> dict[str, np.ndarray]:
    """Load population data and return weight vectors keyed by scheme name.

    Uses 2020 decennial census pop_total, aligned to the order of county_fips.

    Parameters
    ----------
    county_fips : ndarray of shape (N,)
        County FIPS codes (zero-padded strings), in shift-matrix order.

    Returns
    -------
    dict mapping scheme name -> ndarray of shape (N,):
        "unweighted"    : None (signals sklearn KMeans default)
        "population"    : raw pop_total, normalized to mean=1
        "log_population": log(pop_total), normalized to mean=1
    """
    census = pd.read_parquet(CENSUS_PATH)
    census["county_fips"] = census["county_fips"].astype(str).str.zfill(5)
    fips_to_pop = dict(zip(census["county_fips"], census["pop_total"]))

    pop = np.array([fips_to_pop.get(f, np.nan) for f in county_fips], dtype=float)

    # Fill any missing with median (should not happen for FL+GA+AL)
    median_pop = np.nanmedian(pop)
    pop = np.where(np.isnan(pop), median_pop, pop)
    pop = np.where(pop <= 0, 1.0, pop)  # guard against zeros

    # Normalize to mean=1 so absolute magnitude doesn't affect KMeans inertia scale
    pop_norm = pop / pop.mean()
    log_pop = np.log(pop)
    log_pop_norm = log_pop / log_pop.mean()

    return {
        "unweighted": None,
        "population": pop_norm,
        "log_population": log_pop_norm,
    }


# ---------------------------------------------------------------------------
# CV helpers (reuse logic from experiment_j_sweep.py)
# ---------------------------------------------------------------------------


def group_columns_by_pair(column_names: list[str]) -> dict[str, list[int]]:
    """Group shift column indices by their election year pair suffix."""
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
    """Generate leave-one-pair-out CV folds."""
    pair_groups = group_columns_by_pair(column_names)
    all_col_indices = list(range(len(column_names)))

    folds = []
    for pair_key, holdout_cols in sorted(pair_groups.items()):
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
    """Compute temperature-sharpened soft membership from centroid distances."""
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


def compute_centroid_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances from each row of X to each centroid."""
    J = centroids.shape[0]
    N = X.shape[0]
    dists = np.zeros((N, J))
    for t in range(J):
        dists[:, t] = np.linalg.norm(X - centroids[t], axis=1)
    return dists


def predict_holdout_columns(scores: np.ndarray, X_holdout: np.ndarray) -> np.ndarray:
    """Predict held-out shift columns as type-mean weighted by membership."""
    weight_sums = scores.sum(axis=0)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    type_means = (scores.T @ X_holdout) / weight_sums[:, None]
    return scores @ type_means


def compute_holdout_r(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Pearson r across holdout columns."""
    D = actual.shape[1]
    r_vals = []
    for d in range(D):
        a = actual[:, d]
        p = predicted[:, d]
        if np.std(a) < 1e-10 or np.std(p) < 1e-10:
            r_vals.append(0.0)
        else:
            r, _ = pearsonr(a, p)
            r_vals.append(float(r))
    return float(np.mean(r_vals))


def compute_holdout_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error across all holdout column values."""
    return float(np.mean(np.abs(actual - predicted)))


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------


def run_one_condition(
    X: np.ndarray,
    shift_cols: list[str],
    j: int,
    sample_weight: np.ndarray | None,
    temperature: float = TEMPERATURE,
    min_year: int = MIN_YEAR,
) -> dict[str, float]:
    """Run leave-one-pair-out CV for one (J, weighting) condition.

    Parameters
    ----------
    X : ndarray of shape (N, D) — full preprocessed shift matrix
    shift_cols : list[str]
    j : int — number of KMeans clusters
    sample_weight : ndarray of shape (N,) or None
        Per-county weights passed to KMeans.fit(). None = unweighted.
    temperature : float — T for soft membership
    min_year : int

    Returns
    -------
    dict with keys: mean_r, std_r, mean_mae, std_mae, n_folds
    """
    folds = make_cv_folds(shift_cols, min_year=min_year)
    r_vals = []
    mae_vals = []

    for pair_key, train_cols, holdout_cols in folds:
        X_train = X[:, train_cols]
        X_holdout = X[:, holdout_cols]

        km = KMeans(n_clusters=j, random_state=KMEANS_SEED, n_init=KMEANS_N_INIT)
        km.fit(X_train, sample_weight=sample_weight)
        centroids = km.cluster_centers_

        dists = compute_centroid_distances(X_train, centroids)
        scores = temperature_soft_membership(dists, T=temperature)

        X_holdout_pred = predict_holdout_columns(scores, X_holdout)
        r_vals.append(compute_holdout_r(X_holdout, X_holdout_pred))
        mae_vals.append(compute_holdout_mae(X_holdout, X_holdout_pred))

    return {
        "mean_r": float(np.mean(r_vals)),
        "std_r": float(np.std(r_vals)),
        "mean_mae": float(np.mean(mae_vals)),
        "std_mae": float(np.std(mae_vals)),
        "n_folds": len(r_vals),
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------


def run_experiment(
    j_values: list[int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run population-weighting experiment across all J values and schemes.

    Parameters
    ----------
    j_values : list[int] or None
        J values to test. Defaults to [20, 43].
    verbose : bool

    Returns
    -------
    DataFrame with one row per (j, weight_scheme) combination and columns:
        j, weight_scheme, mean_r, std_r, mean_mae, std_mae, n_folds
    """
    if j_values is None:
        j_values = [20, 43]

    if verbose:
        print("Loading shift matrix...")
    X, shift_cols, county_fips = load_shift_matrix()
    if verbose:
        print(f"  Shape: {X.shape[0]} counties x {X.shape[1]} dims")

    if verbose:
        print("Loading population weights...")
    weight_map = load_population_weights(county_fips)

    folds = make_cv_folds(shift_cols)
    if verbose:
        fold_keys = [f[0] for f in folds]
        print(f"  CV folds ({len(folds)}): {fold_keys}")
        print()

    rows = []
    for j in j_values:
        for scheme in WEIGHT_SCHEMES:
            sample_weight = weight_map[scheme]
            if verbose:
                label = f"J={j:2d}  {scheme:<16}"
                print(f"  Running {label}...", end="", flush=True)

            metrics = run_one_condition(X, shift_cols, j, sample_weight)
            row = {"j": j, "weight_scheme": scheme, **metrics}
            rows.append(row)

            if verbose:
                print(
                    f"  mean_r={metrics['mean_r']:+.4f}  "
                    f"mean_mae={metrics['mean_mae']:.4f}  "
                    f"(n_folds={metrics['n_folds']})"
                )

    results = pd.DataFrame(rows)[
        ["j", "weight_scheme", "mean_r", "std_r", "mean_mae", "std_mae", "n_folds"]
    ]
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_comparison_table(results: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print()
    print("=" * 90)
    print("POPULATION WEIGHTING EXPERIMENT — Leave-one-pair-out CV (T=10 soft membership)")
    print("=" * 90)
    print(
        f"{'J':>4}  {'Weight scheme':<18}  {'Mean r':>8}  {'Std r':>7}  "
        f"{'Mean MAE':>9}  {'Std MAE':>8}  {'Folds':>6}"
    )
    print("-" * 90)

    for j_val in sorted(results["j"].unique()):
        subset = results[results["j"] == j_val]
        # Find best by mean_r
        best_r_idx = subset["mean_r"].idxmax()
        best_mae_idx = subset["mean_mae"].idxmin()

        for idx, row in subset.iterrows():
            markers = []
            if idx == best_r_idx:
                markers.append("best_r")
            if idx == best_mae_idx:
                markers.append("best_mae")
            marker = f"  <-- {', '.join(markers)}" if markers else ""
            print(
                f"{int(row['j']):>4}  {row['weight_scheme']:<18}  "
                f"{row['mean_r']:>+8.4f}  {row['std_r']:>7.4f}  "
                f"{row['mean_mae']:>9.4f}  {row['std_mae']:>8.4f}  "
                f"{int(row['n_folds']):>6}{marker}"
            )
        print()

    print("=" * 90)


def print_summary(results: pd.DataFrame) -> None:
    """Print a concise finding summary."""
    print()
    print("SUMMARY")
    print("-------")
    for j_val in sorted(results["j"].unique()):
        subset = results[results["j"] == j_val]
        baseline = subset[subset["weight_scheme"] == "unweighted"].iloc[0]
        print(f"J={j_val}:")
        for _, row in subset.iterrows():
            if row["weight_scheme"] == "unweighted":
                continue
            delta_r = row["mean_r"] - baseline["mean_r"]
            delta_mae = row["mean_mae"] - baseline["mean_mae"]
            direction_r = "+" if delta_r >= 0 else ""
            direction_mae = "+" if delta_mae >= 0 else ""
            verdict = "BETTER" if delta_r > 0.005 else ("WORSE" if delta_r < -0.005 else "neutral")
            print(
                f"  {row['weight_scheme']:<18} vs unweighted:  "
                f"delta_r={direction_r}{delta_r:.4f}  "
                f"delta_mae={direction_mae}{delta_mae:.5f}  [{verdict}]"
            )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment: population-weighted KMeans clustering"
    )
    parser.add_argument(
        "--j-values",
        type=int,
        nargs="+",
        default=[20, 43],
        metavar="J",
        help="J values to test (default: 20 43)",
    )
    args = parser.parse_args()

    results = run_experiment(j_values=args.j_values, verbose=True)
    print_comparison_table(results)
    print_summary(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "population_weighted_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
