"""Hierarchical nesting of fine types into super-types.

Uses Ward HAC (no spatial constraint) on type loading vectors to group
fine types into a smaller number of super-types for public interpretability.

Usage:
    python -m src.discovery.nest_types
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score


@dataclass
class NestingResult:
    mapping: dict[int, int]  # fine_type_id -> super_type_id
    best_s: int  # selected number of super-types
    silhouette_scores: dict[int, float]  # s -> silhouette score


def nest_types(
    type_loadings: np.ndarray,
    s_candidates: list[int] | None = None,
) -> NestingResult:
    """Group fine types into super-types via Ward HAC on loading vectors.

    Parameters
    ----------
    type_loadings : ndarray of shape (J, D)
        Type shift profiles (rotated loadings from SVD+varimax).
    s_candidates : list[int], optional
        Candidate numbers of super-types. Defaults to [5, 6, 7, 8].

    Returns
    -------
    NestingResult
        Mapping from fine type to super-type, best S, and silhouette scores.
    """
    if s_candidates is None:
        s_candidates = [5, 6, 7, 8]

    j = type_loadings.shape[0]

    # Ward linkage on type loading vectors
    Z = linkage(type_loadings, method="ward")

    # Evaluate each candidate S
    sil_scores: dict[int, float] = {}
    label_maps: dict[int, np.ndarray] = {}

    for s in s_candidates:
        if s >= j:
            # Can't have more super-types than fine types
            sil_scores[s] = -1.0
            label_maps[s] = np.arange(j)
            continue

        labels = fcluster(Z, t=s, criterion="maxclust")
        # Re-index to 0-based contiguous
        unique_labels = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique_labels)}
        labels_0 = np.array([remap[l] for l in labels])
        label_maps[s] = labels_0

        if len(set(labels_0)) < 2:
            sil_scores[s] = -1.0
        else:
            sil_scores[s] = float(silhouette_score(type_loadings, labels_0))

    # Select S with highest silhouette score
    best_s = max(sil_scores, key=lambda s: sil_scores[s])
    best_labels = label_maps[best_s]

    mapping = {i: int(best_labels[i]) for i in range(j)}

    return NestingResult(
        mapping=mapping,
        best_s=best_s,
        silhouette_scores=sil_scores,
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import pandas as pd

    from src.discovery.run_type_discovery import discover_types

    # Load config
    config_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    s_candidates = config["types"]["super_type_count_candidates"]

    # Load type assignments to get J
    type_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    if not type_path.exists():
        raise FileNotFoundError(
            f"Run type discovery first: python -m src.discovery.run_type_discovery\n"
            f"Expected: {type_path}"
        )

    type_df = pd.read_parquet(type_path)
    score_cols = [c for c in type_df.columns if c.startswith("type_") and c.endswith("_score")]
    j = len(score_cols)

    # Re-run discovery to get loadings (or load from saved)
    holdout_cols = [
        "pres_d_shift_20_24",
        "pres_r_shift_20_24",
        "pres_turnout_shift_20_24",
    ]
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    shift_df = pd.read_parquet(shifts_path)
    shift_cols = [c for c in shift_df.columns if c != "county_fips" and c not in holdout_cols]
    shift_matrix = shift_df[shift_cols].values

    result = discover_types(shift_matrix, j=j)

    print(f"Nesting {j} fine types into {s_candidates} super-type candidates...")
    nesting = nest_types(result.loadings, s_candidates=s_candidates)

    print(f"\nSilhouette scores: {nesting.silhouette_scores}")
    print(f"Best S: {nesting.best_s}")
    print(f"Mapping: {nesting.mapping}")

    # Save nesting to the type assignments
    type_df["super_type"] = type_df["dominant_type"].map(nesting.mapping)
    type_df.to_parquet(type_path, index=False)
    print(f"Updated {type_path} with super_type column")


if __name__ == "__main__":
    main()
