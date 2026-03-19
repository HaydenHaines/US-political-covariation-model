"""Temporal holdout validation — train on pre-2024 shifts, test against 2024.

SHIFT_COLS order in shift_vectors.parquet:
  cols 0-2  pres_16_20   (D-shift: dem_share_2020 - dem_share_2016)
  cols 3-5  pres_20_24   (D-shift: dem_share_2024 - dem_share_2020)  ← holdout
  cols 6-8  mid_18_22    (D-shift: dem_share_2022 - dem_share_2018)

Training = cols 0-2 + cols 6-8 (6 dims, no future leak).
Holdout  = cols 3-5 (3 dims, 2024 presidential shift).

Outputs:
    data/validation/holdout_2024_results.parquet  — community-level summary

Usage:
    uv run python src/validation/validate_holdout.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "assembled" / "shift_vectors.parquet"
ADJACENCY_DIR = PROJECT_ROOT / "data" / "communities"
OUTPUT_PATH = PROJECT_ROOT / "data" / "validation" / "holdout_2024_results.parquet"


# ── Core functions ─────────────────────────────────────────────────────────────


def split_training_holdout(
    shifts_9d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split 9-dim shift matrix into 6-dim training and 3-dim holdout.

    Column layout (see module docstring):
      - cols 0-2: pres 16->20 (training)
      - cols 3-5: pres 20->24 (holdout — NOT in training)
      - cols 6-8: mid  18->22 (training)

    Parameters
    ----------
    shifts_9d:
        Array of shape (n_tracts, 9).

    Returns
    -------
    train_6d : np.ndarray
        Shape (n_tracts, 6) — cols 0-2 + cols 6-8.
    holdout_3d : np.ndarray
        Shape (n_tracts, 3) — cols 3-5 only.
    """
    train_6d = np.concatenate([shifts_9d[:, 0:3], shifts_9d[:, 6:9]], axis=1)
    holdout_3d = shifts_9d[:, 3:6]
    return train_6d, holdout_3d


def community_level_prediction_accuracy(
    training_means: np.ndarray,
    holdout_means: np.ndarray,
) -> tuple[float, float]:
    """Pearson correlation and MAE between community-level D-shift means.

    Uses only the first column of each array (the primary D-shift dimension)
    to measure how well training-period community means predict holdout-period
    community means.

    Parameters
    ----------
    training_means:
        Array of shape (n_communities, n_train_dims) — per-community mean
        training shift vectors.
    holdout_means:
        Array of shape (n_communities, n_holdout_dims) — per-community mean
        holdout shift vectors.

    Returns
    -------
    correlation : float
        Pearson r between training[:, 0] and holdout[:, 0].
    mae : float
        Mean absolute error between the same two vectors.
    """
    from scipy.stats import pearsonr

    train_d = training_means[:, 0]
    holdout_d = holdout_means[:, 0]

    correlation, _ = pearsonr(train_d, holdout_d)
    mae = float(np.mean(np.abs(train_d - holdout_d)))
    return float(correlation), mae


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Full holdout pipeline: cluster on pre-2024 shifts, evaluate on 2024."""
    import pandas as pd
    from scipy.sparse import load_npz
    from sklearn.preprocessing import StandardScaler

    from src.description.compare_to_nmf import within_community_variance
    from src.discovery.build_adjacency import build_queen_adjacency
    from src.discovery.cluster_communities import cluster_at_threshold

    # ── 1. Load 9-dim shift vectors ────────────────────────────────────────────
    log.info("Loading shift vectors from %s", SHIFTS_PATH)
    shifts_df = pd.read_parquet(SHIFTS_PATH)
    shift_cols = [c for c in shifts_df.columns if c != "tract_geoid"]
    shifts_9d = shifts_df[shift_cols].values.astype(float)
    log.info("Loaded %d tracts × %d shift dims", *shifts_9d.shape)

    # ── 2. Split training / holdout ────────────────────────────────────────────
    train_6d, holdout_3d = split_training_holdout(shifts_9d)
    log.info("Training dims: %s  |  Holdout dims: %s", train_6d.shape[1], holdout_3d.shape[1])

    # ── 3. Normalise training shifts ───────────────────────────────────────────
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_6d)
    log.info("Training shifts normalised (StandardScaler)")

    # ── 4. Load adjacency, cluster on training shifts ──────────────────────────
    adjacency_path = ADJACENCY_DIR / "adjacency.npz"
    geoids_path = ADJACENCY_DIR / "adjacency.geoids.txt"

    if adjacency_path.exists():
        log.info("Loading adjacency from %s", adjacency_path)
        W = load_npz(str(adjacency_path))
    else:
        log.warning("Adjacency file not found; skipping spatial constraint")
        W = None

    log.info("Clustering at n_clusters=50 on training-only shifts …")
    labels, _ = cluster_at_threshold(train_norm, W, n_clusters=50)
    n_communities = len(np.unique(labels))
    log.info("Produced %d communities", n_communities)

    # ── 5. Within-community variance on holdout shifts ─────────────────────────
    holdout_var = within_community_variance(holdout_3d, labels)
    train_var = within_community_variance(train_6d, labels)
    log.info("Within-community variance — training: %.6f | holdout: %.6f", train_var, holdout_var)

    # ── 6. Community-level prediction accuracy ─────────────────────────────────
    unique_labels = np.unique(labels)
    training_means = np.array([train_6d[labels == k].mean(axis=0) for k in unique_labels])
    holdout_means = np.array([holdout_3d[labels == k].mean(axis=0) for k in unique_labels])

    corr, mae = community_level_prediction_accuracy(training_means, holdout_means)
    log.info("Community-level D-shift correlation (training→holdout): %.4f", corr)
    log.info("Community-level D-shift MAE: %.4f", mae)

    # ── 7. Print report and save ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Temporal Holdout Validation — 2024 Presidential Shift")
    print("Train: pres 16→20 + mid 18→22  |  Holdout: pres 20→24")
    print("=" * 60)
    print(f"  n_tracts           : {shifts_9d.shape[0]:,}")
    print(f"  n_communities      : {n_communities}")
    print(f"  WCV (training)     : {train_var:.6f}")
    print(f"  WCV (holdout)      : {holdout_var:.6f}")
    print(f"  Holdout/Train ratio: {holdout_var / train_var:.3f}  (1.0 = equal structure)")
    print(f"  Community corr     : {corr:.4f}  (training D-shift → holdout D-shift)")
    print(f"  Community MAE      : {mae:.4f}")
    print()

    results_df = pd.DataFrame({
        "community_id": unique_labels,
        "training_d_shift_mean": training_means[:, 0],
        "holdout_d_shift_mean": holdout_means[:, 0],
        "training_wcv": [
            float(within_community_variance(train_6d[labels == k], np.zeros(np.sum(labels == k), dtype=int)))
            for k in unique_labels
        ],
        "holdout_wcv": [
            float(within_community_variance(holdout_3d[labels == k], np.zeros(np.sum(labels == k), dtype=int)))
            for k in unique_labels
        ],
        "n_tracts": [int(np.sum(labels == k)) for k in unique_labels],
    })
    results_df.attrs["community_corr"] = corr
    results_df.attrs["community_mae"] = mae
    results_df.attrs["train_wcv"] = train_var
    results_df.attrs["holdout_wcv"] = holdout_var

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Results saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
