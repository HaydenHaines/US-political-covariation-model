"""Compare shift-discovered communities to the NMF baseline.

Converts NMF soft assignments to hard assignments (argmax), then compares
within-community shift variance against a random spatial baseline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)

# Input/output paths
NMF_WEIGHTS_PATH = Path("data/propagation/community_weights_tract.parquet")
SHIFT_ASSIGNMENTS_PATH = Path("data/communities/community_assignments.parquet")
SHIFTS_PATH = Path(__file__).resolve().parents[2] / "data" / "shifts" / "tract_shifts.parquet"
OUTPUT_PATH = Path("data/validation/nmf_comparison.parquet")


def nmf_hard_assignment(
    weights_df: pd.DataFrame,
    component_cols: list[str],
) -> pd.DataFrame:
    """Convert NMF soft assignments to hard (argmax) assignments.

    Parameters
    ----------
    weights_df:
        DataFrame containing ``tract_geoid`` and one column per NMF component.
    component_cols:
        Ordered list of component column names to use for argmax.

    Returns
    -------
    DataFrame with ``tract_geoid`` and ``nmf_community`` (integer index of
    the dominant component).
    """
    W = weights_df[component_cols].to_numpy()
    community_indices = np.argmax(W, axis=1)
    return pd.DataFrame({
        "tract_geoid": weights_df["tract_geoid"].values,
        "nmf_community": community_indices,
    })


def within_community_variance(
    shifts: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute size-weighted mean within-community variance.

    For each community k, compute the mean squared distance from each member's
    shift vector to the community centroid::

        var_k = mean(||x_i - mu_k||^2)

    Return the population-weighted mean across communities::

        sum(n_k * var_k) / sum(n_k)

    Parameters
    ----------
    shifts:
        Array of shape ``(n_tracts, n_dims)`` with shift vectors.
    labels:
        Integer community label for each tract, shape ``(n_tracts,)``.

    Returns
    -------
    Scalar weighted variance.
    """
    unique_labels = np.unique(labels)
    total_n = 0
    total_var = 0.0

    for k in unique_labels:
        mask = labels == k
        members = shifts[mask]
        n_k = len(members)
        if n_k == 0:
            continue
        centroid = members.mean(axis=0)
        diffs = members - centroid
        var_k = (diffs ** 2).sum(axis=1).mean()
        total_var += n_k * var_k
        total_n += n_k

    if total_n == 0:
        return 0.0
    return float(total_var / total_n)


def random_spatial_variance(
    shifts: np.ndarray,
    W: csr_matrix,
    n_communities: int,
    n_trials: int = 100,
) -> float:
    """Baseline: mean within-community variance of random spatial partitions.

    Randomly assign each node to one of ``n_communities`` groups (uniform),
    compute the within-community variance, and average over ``n_trials``.

    Parameters
    ----------
    shifts:
        Array of shape ``(n_tracts, n_dims)``.
    W:
        Sparse adjacency matrix (used only to determine n_tracts).
    n_communities:
        Number of partitions to create in each trial.
    n_trials:
        Number of random trials to average over.

    Returns
    -------
    Mean within-community variance across trials.
    """
    n_tracts = shifts.shape[0]
    rng = np.random.default_rng(0)
    variances = []

    for _ in range(n_trials):
        labels = rng.integers(0, n_communities, size=n_tracts)
        var = within_community_variance(shifts, labels)
        variances.append(var)

    return float(np.mean(variances))


def main() -> None:
    """Load data, compare NMF vs shift communities, save comparison table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("Loading NMF weights from %s", NMF_WEIGHTS_PATH)
    weights_df = pd.read_parquet(NMF_WEIGHTS_PATH)

    log.info("Loading shift-based assignments from %s", SHIFT_ASSIGNMENTS_PATH)
    shift_assignments = pd.read_parquet(SHIFT_ASSIGNMENTS_PATH)

    log.info("Loading shift vectors from %s", SHIFTS_PATH)
    shifts_df = pd.read_parquet(SHIFTS_PATH)

    # Align all data to common tracts
    tract_ids = (
        set(weights_df["tract_geoid"])
        & set(shift_assignments["tract_geoid"])
        & set(shifts_df["tract_geoid"])
    )
    log.info("Common tracts: %d", len(tract_ids))

    shifts_df = shifts_df[shifts_df["tract_geoid"].isin(tract_ids)].sort_values("tract_geoid")
    shift_assignments = shift_assignments[shift_assignments["tract_geoid"].isin(tract_ids)].sort_values("tract_geoid")
    weights_df = weights_df[weights_df["tract_geoid"].isin(tract_ids)].sort_values("tract_geoid")

    shift_cols = [c for c in shifts_df.columns if c != "tract_geoid"]
    shift_matrix = shifts_df[shift_cols].to_numpy()

    # NMF hard assignment — only numeric weight columns (e.g., c1..c7)
    component_cols = [c for c in weights_df.select_dtypes("number").columns if c != "tract_geoid"]
    nmf_assignments = nmf_hard_assignment(weights_df, component_cols)
    nmf_labels = nmf_assignments.sort_values("tract_geoid")["nmf_community"].to_numpy()

    # Shift-based labels — column may be "community" or "community_id"
    comm_col = "community_id" if "community_id" in shift_assignments.columns else "community"
    shift_labels = shift_assignments[comm_col].to_numpy()

    # Compute variances
    n_communities = len(np.unique(shift_labels))
    var_shift = within_community_variance(shift_matrix, shift_labels)
    var_nmf = within_community_variance(shift_matrix, nmf_labels)

    log.info("Within-community variance — shift: %.6f | NMF: %.6f", var_shift, var_nmf)

    # Build comparison table
    comparison = pd.DataFrame([
        {"method": "shift_discovered", "within_community_variance": var_shift, "n_communities": n_communities},
        {"method": "nmf_baseline", "within_community_variance": var_nmf, "n_communities": len(np.unique(nmf_labels))},
    ])
    print(comparison.to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
