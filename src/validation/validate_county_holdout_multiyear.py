"""Multi-year county holdout validation.

Trains community discovery on 30-dimensional pre-2024 shifts,
tests against 2020→2024 (3 holdout dims). Reports Pearson r and
MAE at multiple k values. Compares against 3-cycle baseline.

Metric notes
------------
The 3-cycle baseline (6 training dims) reports r by correlating community-
mean pres_d_shift_16_20 (training col 0) against community-mean
pres_d_shift_20_24 (holdout col 0).  For a fair comparison with the multi-
year model (30 training dims), we use the same two cycles: training col 12
(pres_d_shift_16_20, the most recent presidential D-shift in training) vs
holdout col 0.  Calling community_correlation with training_col=12 produces
an apple-to-apples comparison.  The default training_col=0 is preserved for
the unit tests (which use synthetic 30-dim data where all columns are
equivalent).

Usage:
  uv run python -m src.validation.validate_county_holdout_multiyear
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
ADJACENCY_DIR = PROJECT_ROOT / "data" / "communities"

# Baseline results from 3-cycle run (county_shifts.parquet, 6 training dims).
# Those results correlate training col 0 (pres_d_shift_16_20) vs holdout col 0
# (pres_d_shift_20_24) at community level.
BASELINE = {5: 0.983, 7: 0.964, 10: 0.941, 15: 0.934, 20: 0.932}

N_TRAINING_COLS = 30  # first 30 cols after county_fips are training

# Column index of pres_d_shift_16_20 within the 30 training columns.
# This is the most recent presidential D-shift in training and the direct
# counterpart to the holdout's pres_d_shift_20_24 (holdout col 0).
# Verified from county_shifts_multiyear.parquet column ordering.
PRES_D_16_20_COL = 12

K_VALUES = [5, 7, 10, 15, 20]


def split_training_holdout(
    shifts: np.ndarray,
    n_training: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split shift matrix into training and holdout portions."""
    return shifts[:, :n_training], shifts[:, n_training:]


def community_correlation(
    training_means: np.ndarray,
    holdout_means: np.ndarray,
    training_col: int = 0,
) -> tuple[float, float]:
    """Pearson r and MAE between a training-cycle D-shift and holdout D-shift.

    Parameters
    ----------
    training_means:
        Shape (K, n_training_dims).  Community-level means over training data.
    holdout_means:
        Shape (K, n_holdout_dims).  Community-level means over holdout data.
    training_col:
        Which column of training_means to use as the training reference.
        Default 0 (first column) is compatible with the 3-cycle baseline and
        the unit tests.  Pass PRES_D_16_20_COL=12 in main() for the fair
        apple-to-apples comparison against the 3-cycle baseline.
    """
    from scipy.stats import pearsonr
    train_d = training_means[:, training_col]
    holdout_d = holdout_means[:, 0]
    r, _ = pearsonr(train_d, holdout_d)
    mae = float(np.mean(np.abs(train_d - holdout_d)))
    return float(r), mae


def main() -> None:
    import pandas as pd
    from scipy.sparse import load_npz
    from sklearn.cluster._agglomerative import _hc_cut
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    log.info("Loading multi-year county shifts from %s", SHIFTS_PATH)
    df = pd.read_parquet(SHIFTS_PATH)
    shift_cols = [c for c in df.columns if c != "county_fips"]

    # Align to adjacency ordering
    geoids_path = ADJACENCY_DIR / "county_adjacency.fips.txt"
    geoids = geoids_path.read_text().splitlines()
    fips_indexed = df.set_index("county_fips")
    aligned = fips_indexed.reindex(geoids)
    n_missing = aligned[shift_cols[0]].isna().sum()
    if n_missing:
        log.info("Filling %d counties with missing data using column means", n_missing)
        aligned[shift_cols] = aligned[shift_cols].fillna(aligned[shift_cols].mean())
    all_shifts = aligned[shift_cols].values.astype(float)

    train, holdout = split_training_holdout(all_shifts, N_TRAINING_COLS)
    log.info("Training dims: %d | Holdout dims: %d | Counties: %d",
             train.shape[1], holdout.shape[1], train.shape[0])

    # Confirm column identity for the training reference column
    ref_col_name = shift_cols[PRES_D_16_20_COL]
    holdout_col_name = shift_cols[N_TRAINING_COLS]
    log.info("Training reference col [%d]: %s", PRES_D_16_20_COL, ref_col_name)
    log.info("Holdout reference col  [0]: %s", holdout_col_name)

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train)

    adjacency_path = ADJACENCY_DIR / "county_adjacency.npz"
    W = load_npz(str(adjacency_path))

    # Build full tree once
    log.info("Building full Ward dendrogram...")
    model = AgglomerativeClustering(
        linkage="ward", connectivity=W, n_clusters=1, compute_distances=True
    )
    model.fit(train_norm)

    print("\n" + "=" * 75)
    print("Multi-Year County Holdout Validation — 30 training dims vs 3-cycle baseline")
    print("Train: pres 2000–2020 (5 pairs) + gov 2002–2022 (5 pairs)")
    print("Holdout: pres 2020→2024")
    print(f"Metric: community-mean {ref_col_name} (train col {PRES_D_16_20_COL})")
    print(f"        vs community-mean {holdout_col_name} (holdout col 0)")
    print("3-cycle baseline: same two columns but clusters built on only 6 training dims")
    print("=" * 75)
    print(f"{'k':>4}  {'r (multiyear)':>16}  {'r (3-cycle)':>12}  {'delta':>7}  {'MAE':>8}")
    print("-" * 75)

    results = {}
    for k in K_VALUES:
        labels = _hc_cut(k, model.children_, len(geoids))
        unique_labels = np.unique(labels)
        train_means = np.array([train[labels == lbl].mean(axis=0) for lbl in unique_labels])
        holdout_means = np.array([holdout[labels == lbl].mean(axis=0) for lbl in unique_labels])
        r, mae = community_correlation(train_means, holdout_means, training_col=PRES_D_16_20_COL)
        baseline_r = BASELINE.get(k, float("nan"))
        delta = r - baseline_r
        print(f"{k:>4}  {r:>16.4f}  {baseline_r:>12.4f}  {delta:>+7.4f}  {mae:>8.4f}")
        results[k] = {"r": r, "mae": mae, "baseline_r": baseline_r, "delta": delta}

    print()
    log.info("Done.")
    return results


if __name__ == "__main__":
    main()
