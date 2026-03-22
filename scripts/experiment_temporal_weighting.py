"""Experiment: Temporal weighting in KMeans type discovery.

Hypothesis: weighting recent election shifts more heavily (e.g., 2016→2020
and 2020→2024) than older ones (2008→2012) should produce types that better
predict the next election.

Design:
- Current pipeline uses equal weighting across all shift dimensions.
- Temporal weighting is applied MULTIPLICATIVELY on top of the existing
  presidential×2.5 weight. Implementation: multiply each column by
  sqrt(temporal_weight) before clustering so squared Euclidean distances
  become temporally weighted.
- The "anchor year" for a column is the start year of its election pair
  (e.g., pres_d_shift_08_12 → 2008; gov_d_shift_06_10 → 2006).
  Senate pairs use a 6-year window so the anchor is still the start year.
- Four decay schemes tested:
    equal       : all temporal weights = 1.0 (baseline)
    linear      : weight = (year - 2006) / (2024 - 2006)  (0.11 → 1.0)
    exponential : weight = 2^((year - 2024) / half_life), half_lives = [4, 8, 12]
    step        : weight = 0.0 for pairs starting before 2016, 1.0 otherwise
- For each scheme, applies temporal weights to shift vectors, runs KMeans
  with J=43, T=10, and evaluates with leave-one-pair-out CV (same folds
  as the J-sweep experiment).
- Results printed to stdout and saved to data/experiments/temporal_weighting_results.csv

Usage:
    cd /home/hayden/projects/US-political-covariation-model
    uv run python scripts/experiment_temporal_weighting.py
    uv run python scripts/experiment_temporal_weighting.py --j 43 --temperature 10
"""
from __future__ import annotations

import argparse
import re
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
# Production pipeline constants — must match experiment_j_sweep.py exactly
# ---------------------------------------------------------------------------

KMEANS_SEED = 42
KMEANS_N_INIT = 10
PRES_WEIGHT = 2.5
MIN_YEAR = 2008
TEMPERATURE = 10.0
DEFAULT_J = 43

BLIND_HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

# Reference anchor for linear/exponential decay schemes
EARLIEST_YEAR = 2006  # Earliest possible anchor year in the dataset
LATEST_YEAR = 2024    # Most recent anchor year (the end of the last training pair)
STEP_CUTOFF_YEAR = 2016  # Step scheme: zero weight for pairs starting before this


# ---------------------------------------------------------------------------
# Temporal weight schemes
# ---------------------------------------------------------------------------


def get_anchor_year(col: str) -> int | None:
    """Extract the anchor (start) year from a shift column name.

    Column naming pattern: <race>_<metric>_shift_<yy1>_<yy2>
    Examples: pres_d_shift_08_12, gov_d_shift_06_10, sen_d_shift_02_08

    Returns the 4-digit start year, or None if the column doesn't match.
    """
    m = re.search(r"shift_(\d{2})_\d{2}$", col)
    if m:
        return int("20" + m.group(1))
    return None


def compute_temporal_weights(
    shift_cols: list[str],
    scheme: str,
    half_life: float | None = None,
    step_cutoff: int = STEP_CUTOFF_YEAR,
    earliest_year: int = EARLIEST_YEAR,
    latest_year: int = LATEST_YEAR,
) -> np.ndarray:
    """Compute per-column temporal weights for a given decay scheme.

    Parameters
    ----------
    shift_cols : list[str]
        Column names (post-filter, already excluding blind holdout).
    scheme : str
        One of: 'equal', 'linear', 'exponential', 'step'.
    half_life : float or None
        Half-life in years for 'exponential' scheme. Required when scheme='exponential'.
    step_cutoff : int
        For 'step' scheme: pairs starting before this year get weight 0.
    earliest_year : int
        Earliest anchor year used for linear normalization.
    latest_year : int
        Latest anchor year (denominator for linear scheme).

    Returns
    -------
    weights : ndarray of shape (D,)
        Per-column temporal weights in [0, 1] (or 0 for step-excluded columns).
        These are TEMPORAL weights only — presidential×2.5 is applied separately.
    """
    weights = np.ones(len(shift_cols))

    if scheme == "equal":
        return weights  # all 1.0

    for i, col in enumerate(shift_cols):
        year = get_anchor_year(col)
        if year is None:
            # Cannot parse year — leave at 1.0
            continue

        if scheme == "linear":
            denom = latest_year - earliest_year
            weights[i] = (year - earliest_year) / denom if denom > 0 else 1.0

        elif scheme == "exponential":
            if half_life is None or half_life <= 0:
                raise ValueError(f"half_life must be a positive number for exponential scheme, got {half_life}")
            weights[i] = 2.0 ** ((year - latest_year) / half_life)

        elif scheme == "step":
            weights[i] = 0.0 if year < step_cutoff else 1.0

        else:
            raise ValueError(f"Unknown temporal weighting scheme: {scheme!r}")

    return weights


def scheme_label(scheme: str, half_life: float | None = None) -> str:
    """Return a human-readable label for a scheme."""
    if scheme == "exponential":
        return f"exponential_hl{int(half_life)}"
    return scheme


# ---------------------------------------------------------------------------
# Data loading (mirrors experiment_j_sweep.load_shift_matrix exactly)
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

    Returns
    -------
    X : ndarray of shape (N, D)
        Preprocessed shift matrix (before temporal weighting).
    shift_cols : list[str]
        Column names (D columns, post-filter).
    county_fips : ndarray of shape (N,)
        County FIPS codes (zero-padded strings).
    pres_weights : ndarray of shape (D,)
        Presidential race weights (2.5 for pres_*, 1.0 otherwise),
        already multiplied into X.
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
    pres_weights = np.ones(len(shift_cols))
    for i, c in enumerate(shift_cols):
        if c.startswith("pres_"):
            pres_weights[i] = PRES_WEIGHT

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

    X = X_raw * pres_weights[None, :]
    return X, shift_cols, county_fips, pres_weights


def apply_temporal_weights(X: np.ndarray, temporal_weights: np.ndarray) -> np.ndarray:
    """Apply temporal weights to a preprocessed shift matrix.

    Multiplies each column by sqrt(temporal_weight) so that squared
    Euclidean distances (used by KMeans) are scaled by temporal_weight.

    Zero-weight columns are zeroed out entirely (effectively dropped).

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Shift matrix (already has presidential×2.5 applied).
    temporal_weights : ndarray of shape (D,)
        Per-column temporal weights.

    Returns
    -------
    X_weighted : ndarray of shape (N, D)
    """
    sqrt_weights = np.sqrt(np.maximum(temporal_weights, 0.0))
    return X * sqrt_weights[None, :]


# ---------------------------------------------------------------------------
# CV fold helpers (identical logic to experiment_j_sweep.py)
# ---------------------------------------------------------------------------


def group_columns_by_pair(column_names: list[str]) -> dict[str, list[int]]:
    """Group shift column indices by their election year pair suffix."""
    pair_re = re.compile(r"(\d{2}_\d{2})$")
    from collections import defaultdict
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
# KMeans + evaluation helpers
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


def compute_centroid_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean distances from each row of X to each centroid."""
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


def run_one_fold(
    X_weighted: np.ndarray,
    j: int,
    fold: tuple[str, list[int], list[int]],
    temperature: float = TEMPERATURE,
) -> dict[str, float]:
    """Run one leave-one-pair-out CV fold on a temporally-weighted matrix.

    Parameters
    ----------
    X_weighted : ndarray of shape (N, D)
        Shift matrix with temporal weights already applied (sqrt-scaled).
    j : int
        Number of KMeans clusters.
    fold : (pair_key, train_col_indices, holdout_col_indices)
    temperature : float

    Returns
    -------
    dict with keys: pair_key, holdout_r, holdout_mae
    """
    pair_key, train_cols, holdout_cols = fold

    X_train = X_weighted[:, train_cols]
    X_holdout = X_weighted[:, holdout_cols]

    km = KMeans(n_clusters=j, random_state=KMEANS_SEED, n_init=KMEANS_N_INIT)
    km.fit(X_train)
    centroids = km.cluster_centers_

    dists = compute_centroid_distances(X_train, centroids)
    scores = temperature_soft_membership(dists, T=temperature)

    X_holdout_pred = predict_holdout_columns(scores, X_holdout)

    return {
        "pair_key": pair_key,
        "holdout_r": compute_holdout_r(X_holdout, X_holdout_pred),
        "holdout_mae": compute_holdout_mae(X_holdout, X_holdout_pred),
    }


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------


def build_scheme_list(half_lives: list[float] | None = None) -> list[dict]:
    """Return a list of scheme parameter dicts to evaluate.

    Each dict has keys: label, scheme, half_life (None unless exponential).
    """
    if half_lives is None:
        half_lives = [4.0, 8.0, 12.0]

    schemes = [
        {"label": "equal", "scheme": "equal", "half_life": None},
        {"label": "linear", "scheme": "linear", "half_life": None},
        {"label": "step_2016", "scheme": "step", "half_life": None},
    ]
    for hl in half_lives:
        schemes.append({
            "label": f"exponential_hl{int(hl)}",
            "scheme": "exponential",
            "half_life": hl,
        })
    return schemes


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_temporal_weighting_experiment(
    j: int = DEFAULT_J,
    temperature: float = TEMPERATURE,
    min_year: int = MIN_YEAR,
    half_lives: list[float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the temporal weighting experiment across all schemes.

    For each scheme:
      1. Compute per-column temporal weights
      2. Apply sqrt-scaled weights to the preprocessed shift matrix
      3. Run leave-one-pair-out CV with KMeans J=j, T=temperature
      4. Record mean holdout r and MAE across folds

    Parameters
    ----------
    j : int
        Number of KMeans clusters (default: 43, production value).
    temperature : float
        Soft membership temperature (default: 10.0).
    min_year : int
        Only use election pairs with start year >= this.
    half_lives : list[float] or None
        Half-lives (in years) for exponential schemes. Default: [4, 8, 12].
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    DataFrame with columns:
        scheme, mean_r, std_r, mean_mae, std_mae, n_folds, n_active_dims
    """
    if verbose:
        print(f"Loading shift matrix (min_year={min_year})...")
    X, shift_cols, county_fips, pres_weights = load_shift_matrix(min_year=min_year)
    if verbose:
        print(f"  Shape: {X.shape[0]} counties x {X.shape[1]} dims")

    folds = make_cv_folds(shift_cols, min_year=min_year)
    if verbose:
        fold_keys = [f[0] for f in folds]
        print(f"  CV folds ({len(folds)}): {fold_keys}")
        print()

    schemes = build_scheme_list(half_lives)
    rows = []

    for s in schemes:
        label = s["label"]
        if verbose:
            print(f"  Scheme: {label:30s} ...", end="", flush=True)

        try:
            temp_weights = compute_temporal_weights(
                shift_cols,
                scheme=s["scheme"],
                half_life=s["half_life"],
            )
        except ValueError as e:
            if verbose:
                print(f" ERROR: {e}")
            continue

        X_weighted = apply_temporal_weights(X, temp_weights)

        # Count "active" dims (non-zero temporal weight across all columns in the fold)
        n_active = int((temp_weights > 0).sum())

        fold_results = []
        for fold in folds:
            try:
                result = run_one_fold(X_weighted, j=j, fold=fold, temperature=temperature)
                fold_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"\n    WARNING: fold {fold[0]} failed for scheme={label}: {e}")

        if not fold_results:
            rows.append({
                "scheme": label,
                "mean_r": float("nan"),
                "std_r": float("nan"),
                "mean_mae": float("nan"),
                "std_mae": float("nan"),
                "n_folds": 0,
                "n_active_dims": n_active,
            })
            if verbose:
                print(" no valid folds")
            continue

        r_vals = [fr["holdout_r"] for fr in fold_results]
        mae_vals = [fr["holdout_mae"] for fr in fold_results]

        row = {
            "scheme": label,
            "mean_r": float(np.mean(r_vals)),
            "std_r": float(np.std(r_vals)),
            "mean_mae": float(np.mean(mae_vals)),
            "std_mae": float(np.std(mae_vals)),
            "n_folds": len(fold_results),
            "n_active_dims": n_active,
        }
        rows.append(row)

        if verbose:
            print(
                f" r={row['mean_r']:.4f} ± {row['std_r']:.4f}"
                f"  MAE={row['mean_mae']:.4f} ± {row['std_mae']:.4f}"
                f"  active_dims={n_active}"
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(results: pd.DataFrame) -> None:
    """Print formatted comparison table to stdout."""
    print()
    print("=" * 100)
    print(f"TEMPORAL WEIGHTING EXPERIMENT — Leave-One-Pair-Out CV (KMeans J={DEFAULT_J}, T={TEMPERATURE}, 2008+)")
    print("=" * 100)
    header = (
        f"{'Scheme':<30}  {'mean_r':>8}  {'std_r':>7}  {'mean_MAE':>9}  "
        f"{'std_MAE':>8}  {'active_dims':>11}  {'n_folds':>7}"
    )
    print(header)
    print("-" * 100)

    # Find best scheme by mean_r
    valid = results.dropna(subset=["mean_r"])
    if len(valid) > 0:
        best_idx = valid["mean_r"].idxmax()
    else:
        best_idx = None

    for idx, row in results.iterrows():
        marker = " <-- BEST" if idx == best_idx else ""
        r_str = f"{row['mean_r']:>8.4f}" if pd.notna(row["mean_r"]) else f"{'N/A':>8}"
        mae_str = f"{row['mean_mae']:>9.4f}" if pd.notna(row["mean_mae"]) else f"{'N/A':>9}"
        print(
            f"{row['scheme']:<30}  "
            f"{r_str}  "
            f"{row['std_r']:>7.4f}  "
            f"{mae_str}  "
            f"{row['std_mae']:>8.4f}  "
            f"{int(row['n_active_dims']):>11}  "
            f"{int(row['n_folds']):>7}"
            f"{marker}"
        )

    print()
    if best_idx is not None:
        best_row = valid.loc[best_idx]
        baseline_row = results[results["scheme"] == "equal"]
        if len(baseline_row) > 0 and pd.notna(baseline_row.iloc[0]["mean_r"]):
            baseline_r = float(baseline_row.iloc[0]["mean_r"])
            best_r = float(best_row["mean_r"])
            delta = best_r - baseline_r
            print(f"Best scheme: {best_row['scheme']}  (mean_r = {best_r:.4f})")
            print(f"Baseline (equal): mean_r = {baseline_r:.4f}")
            print(f"Delta vs baseline: {delta:+.4f} ({100*delta/abs(baseline_r):.1f}% relative)")
        else:
            print(f"Best scheme: {best_row['scheme']}  (mean_r = {float(best_row['mean_r']):.4f})")
    print()


def save_results(results: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save results DataFrame to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "temporal_weighting_results.csv"
    results.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal weighting experiment for KMeans type discovery")
    parser.add_argument("--j", type=int, default=DEFAULT_J, help=f"Number of KMeans clusters (default: {DEFAULT_J})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help=f"Soft membership temperature (default: {TEMPERATURE})")
    parser.add_argument("--min-year", type=int, default=MIN_YEAR, help=f"Min start year for election pairs (default: {MIN_YEAR})")
    parser.add_argument(
        "--half-lives",
        type=float,
        nargs="+",
        default=[4.0, 8.0, 12.0],
        help="Half-lives (years) for exponential schemes (default: 4 8 12)",
    )
    args = parser.parse_args()

    print(f"Temporal weighting experiment: J={args.j}, T={args.temperature}, min_year={args.min_year}")
    print(f"Exponential half-lives: {args.half_lives}")
    print()

    results = run_temporal_weighting_experiment(
        j=args.j,
        temperature=args.temperature,
        min_year=args.min_year,
        half_lives=args.half_lives,
        verbose=True,
    )

    print_results(results)

    out_path = save_results(results)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
