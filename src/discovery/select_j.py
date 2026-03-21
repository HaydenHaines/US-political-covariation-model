"""J selection via leave-one-election-pair-out cross-validation.

For each candidate J, holds out each election pair's columns in turn, fits
SVD + varimax on the remaining columns, predicts the held-out columns via
type-mean weighted by scores, and computes Pearson r. Selects J maximizing
mean holdout r.

Usage:
    python -m src.discovery.select_j
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

from src.discovery.run_type_discovery import discover_types


@dataclass
class JSelectionResult:
    best_j: int
    all_results: pd.DataFrame  # columns: j, mean_holdout_r, explained_var, n_params, dof_ratio


def group_columns_by_pair(column_names: list[str]) -> dict[str, list[int]]:
    """Group shift column indices by their election year pair.

    Extracts the year-pair suffix (e.g., '00_04') from column names like
    'pres_d_shift_00_04' and groups column indices sharing the same pair.

    Parameters
    ----------
    column_names : list[str]
        Shift column names (without county_fips).

    Returns
    -------
    dict mapping pair key (e.g., '00_04') to list of column indices.
    """
    pair_pattern = re.compile(r"(\d{2}_\d{2})$")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, col in enumerate(column_names):
        m = pair_pattern.search(col)
        if m:
            groups[m.group(1)].append(i)
    return dict(groups)


def _predict_holdout(
    train_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
    j: int,
    random_state: int,
) -> float:
    """Fit SVD+varimax on train columns, predict holdout via type means.

    For each county, the predicted holdout value is the weighted average of
    type-mean holdout values, weighted by the county's (normalized) type scores.

    Returns mean Pearson r across holdout columns.
    """
    result = discover_types(train_matrix, j=j, random_state=random_state)
    scores = result.scores  # N x J

    # Normalize scores to weights (use absolute scores, then sign-preserve)
    # Simple approach: use softmax on squared scores for weighting
    abs_scores = np.abs(scores)
    row_sums = abs_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = abs_scores / row_sums  # N x J, non-negative, sum to 1

    # Compute type means on holdout columns using same weights
    # type_mean[j, d] = sum_i(weight[i,j] * holdout[i,d]) / sum_i(weight[i,j])
    weight_sums = weights.sum(axis=0, keepdims=True).T  # J x 1
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    type_means = (weights.T @ holdout_matrix) / weight_sums  # J x D_holdout

    # Predict: county_pred[i, d] = sum_j(weight[i,j] * type_mean[j, d])
    predicted = weights @ type_means  # N x D_holdout

    # Pearson r per holdout column, then average
    r_values = []
    for d in range(holdout_matrix.shape[1]):
        actual = holdout_matrix[:, d]
        pred = predicted[:, d]
        if np.std(actual) < 1e-10 or np.std(pred) < 1e-10:
            r_values.append(0.0)
        else:
            r, _ = pearsonr(actual, pred)
            r_values.append(r)
    return float(np.mean(r_values))


def select_j(
    shift_matrix: np.ndarray,
    pair_column_indices: list[list[int]],
    j_candidates: list[int],
    random_state: int = 42,
) -> JSelectionResult:
    """Select J by leave-one-election-pair-out cross-validation.

    Parameters
    ----------
    shift_matrix : ndarray of shape (N, D)
        Full shift matrix (training columns only, no holdout).
    pair_column_indices : list of list[int]
        Each inner list contains column indices for one election pair
        (typically 3: d_shift, r_shift, turnout_shift).
    j_candidates : list[int]
        Candidate J values to evaluate.
    random_state : int
        Random seed.

    Returns
    -------
    JSelectionResult
        Best J and full results table.
    """
    n_counties, n_dims = shift_matrix.shape
    all_col_indices = set(range(n_dims))
    rows = []

    for j in j_candidates:
        # Degrees of freedom check
        n_params = j * (n_counties + n_dims)
        total_cells = n_counties * n_dims
        dof_ratio = total_cells / n_params if n_params > 0 else 0.0

        if dof_ratio < 1.5:
            rows.append({
                "j": j,
                "mean_holdout_r": np.nan,
                "explained_var": np.nan,
                "n_params": n_params,
                "dof_ratio": dof_ratio,
            })
            continue

        # Leave-one-pair-out CV
        holdout_rs = []
        for pair_cols in pair_column_indices:
            train_cols = sorted(all_col_indices - set(pair_cols))
            if len(train_cols) < j:
                continue  # Not enough training columns
            train_matrix = shift_matrix[:, train_cols]
            holdout_matrix = shift_matrix[:, pair_cols]
            r = _predict_holdout(train_matrix, holdout_matrix, j, random_state)
            holdout_rs.append(r)

        mean_r = float(np.mean(holdout_rs)) if holdout_rs else np.nan

        # Explained variance on full matrix
        result = discover_types(shift_matrix, j=j, random_state=random_state)
        explained_var = float(result.explained_variance.sum())

        rows.append({
            "j": j,
            "mean_holdout_r": mean_r,
            "explained_var": explained_var,
            "n_params": n_params,
            "dof_ratio": dof_ratio,
        })

    results_df = pd.DataFrame(rows)

    # Select best J: highest mean holdout r among valid candidates
    valid = results_df.dropna(subset=["mean_holdout_r"])
    if len(valid) == 0:
        best_j = j_candidates[0]  # Fallback
    else:
        best_j = int(valid.loc[valid["mean_holdout_r"].idxmax(), "j"])

    return JSelectionResult(best_j=best_j, all_results=results_df)


# ── CLI entry point ──────────────────────────────────────────────────────────

HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # Load config
    config_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    j_candidates = config["types"]["j_candidates"]

    # Load shift matrix
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    # Separate FIPS and shift columns, exclude holdout
    shift_cols = [c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS]
    shift_matrix = df[shift_cols].values

    print(f"Shift matrix: {shift_matrix.shape[0]} counties x {shift_matrix.shape[1]} dims")
    print(f"J candidates: {j_candidates}")

    # Group columns by election pair
    pair_groups = group_columns_by_pair(shift_cols)
    pair_indices = list(pair_groups.values())
    print(f"Election pairs for CV: {len(pair_indices)} ({list(pair_groups.keys())})")

    # Run J selection
    result = select_j(
        shift_matrix,
        pair_column_indices=pair_indices,
        j_candidates=j_candidates,
    )

    print("\n=== J Selection Results ===")
    print(result.all_results.to_string(index=False))
    print(f"\nBest J: {result.best_j}")

    # Save results
    out_dir = PROJECT_ROOT / "data" / "communities"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "j_selection_results.parquet"
    result.all_results.to_parquet(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
