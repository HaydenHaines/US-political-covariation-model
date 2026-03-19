"""Score community borders by the gradient of shift vectors across boundaries.

For each pair of adjacent tracts assigned to different communities, computes
the Euclidean distance between their shift vectors. Groups these pairwise
distances by community-pair and reports the mean gradient and count.

High gradient borders represent genuine political fault lines; low gradient
borders may indicate over-segmentation.

Outputs:
    data/communities/border_gradients.parquet  — community-pair gradient scores

Usage:
    uv run python src/discovery/score_borders.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"

# ── Core functions ─────────────────────────────────────────────────────────────


def compute_border_gradients(
    labels: np.ndarray,
    shifts: np.ndarray,
    W,
    geoids: list[str],
) -> pd.DataFrame:
    """Compute mean shift-vector gradient across each community border.

    For every edge (i, j) in the adjacency graph where labels[i] != labels[j],
    computes the Euclidean distance between shifts[i] and shifts[j]. Results
    are grouped by the sorted (community_a, community_b) pair and aggregated
    as mean gradient and edge count.

    Parameters
    ----------
    labels:
        Integer community label per tract, shape (n_tracts,).
    shifts:
        Shift-vector array of shape (n_tracts, n_dims).
    W:
        Sparse adjacency matrix (scipy CSR) of shape (n_tracts, n_tracts).
    geoids:
        Ordered list of tract geoids corresponding to rows/cols of W.

    Returns
    -------
    pd.DataFrame
        Columns: community_a, community_b, gradient, n_boundary_pairs.
        Sorted by community_a, community_b.
    """
    from scipy.sparse import triu

    # Only examine upper triangle to avoid double-counting edges
    W_upper = triu(W, k=1)
    cx = W_upper.tocoo()

    rows = cx.row
    cols = cx.col

    # Filter to cross-community edges
    cross_mask = labels[rows] != labels[cols]
    cross_rows = rows[cross_mask]
    cross_cols = cols[cross_mask]

    if len(cross_rows) == 0:
        return pd.DataFrame(columns=["community_a", "community_b", "gradient", "n_boundary_pairs"])

    # Compute pairwise Euclidean distances
    diff = shifts[cross_rows] - shifts[cross_cols]
    distances = np.linalg.norm(diff, axis=1)

    # Build sorted community-pair keys
    ca = np.minimum(labels[cross_rows], labels[cross_cols])
    cb = np.maximum(labels[cross_rows], labels[cross_cols])

    # Group and aggregate
    pair_df = pd.DataFrame({
        "community_a": ca,
        "community_b": cb,
        "distance": distances,
    })
    result = (
        pair_df.groupby(["community_a", "community_b"], sort=True)
        .agg(gradient=("distance", "mean"), n_boundary_pairs=("distance", "count"))
        .reset_index()
    )
    result["n_boundary_pairs"] = result["n_boundary_pairs"].astype(int)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Load data, compute border gradients, and save to disk."""
    from scipy.sparse import load_npz

    adjacency_path = COMMUNITIES_DIR / "adjacency.npz"
    geoids_path = COMMUNITIES_DIR / "adjacency.geoids.txt"
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "tract_shifts.parquet"
    assignments_path = COMMUNITIES_DIR / "community_assignments.parquet"

    log.info("Loading adjacency matrix …")
    W = load_npz(str(adjacency_path))
    geoids = geoids_path.read_text().splitlines()

    log.info("Loading shift vectors …")
    shifts_df = pd.read_parquet(shifts_path)
    shift_cols = [c for c in shifts_df.columns if c != "tract_geoid"]
    # Align to adjacency geoid order; fill missing tracts with column means
    shifts_indexed = shifts_df.set_index("tract_geoid")
    aligned = shifts_indexed.reindex(geoids)
    n_missing = aligned[shift_cols[0]].isna().sum()
    if n_missing:
        aligned[shift_cols] = aligned[shift_cols].fillna(aligned[shift_cols].mean())
    shifts = aligned[shift_cols].values

    log.info("Loading community assignments …")
    assignments_df = pd.read_parquet(assignments_path)
    labels = assignments_df["community"].values

    log.info("Computing border gradients …")
    result = compute_border_gradients(labels, shifts, W, geoids)
    log.info("Found %d community borders", len(result))

    out_path = COMMUNITIES_DIR / "border_gradients.parquet"
    result.to_parquet(out_path, index=False)
    log.info("Border gradients saved to %s", out_path)


if __name__ == "__main__":
    main()
