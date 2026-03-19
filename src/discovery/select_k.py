"""K selection via holdout accuracy sweep for Ward HAC community discovery.

Sweeps K values and evaluates holdout predictive accuracy. Picks the K
that maximizes Pearson r between community-mean training shifts and
community-mean holdout shifts, subject to a minimum community size constraint.

Usage:
    python src/discovery/select_k.py
    python src/discovery/select_k.py --k-values 5 7 10 15 20 25 30 --min-size 8
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, load_npz
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
# Private sklearn API: stable in sklearn >=1.2. Public alternative would require
# re-fitting at each K (10-50x slower). Verified with sklearn 1.8.0.
from sklearn.cluster._agglomerative import _hc_cut
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
ADJ_NPZ = PROJECT_ROOT / "data" / "communities" / "county_adjacency.npz"
ADJ_FIPS = PROJECT_ROOT / "data" / "communities" / "county_adjacency.fips.txt"
CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"

# Column index within the 30 training cols that corresponds to pres_d_shift_16_20
# (the most recent presidential D-shift in training = direct counterpart to holdout col 0)
_PRES_D_16_20_TRAINING_IDX = 12


@dataclass
class KSweepResult:
    k: int
    holdout_r: float
    min_community_size: int
    community_sizes: list[int] = field(default_factory=list)


def run_k_sweep(
    shifts_df: pd.DataFrame,
    fips_list: list[str],
    adjacency: csr_matrix,
    train_cols: list[str],
    holdout_cols: list[str],
    k_values: Sequence[int],
    min_community_size: int = 8,
    training_comparison_idx: int = _PRES_D_16_20_TRAINING_IDX,
) -> list[KSweepResult]:
    """Run Ward HAC at multiple K values, return holdout accuracy per valid K."""
    # Align shifts to adjacency order
    indexed = shifts_df.set_index("county_fips")
    aligned = indexed.reindex(fips_list)
    n_missing = aligned[train_cols[0]].isna().sum()
    if n_missing:
        log.warning("Filling %d counties with NaN shifts (column means)", n_missing)
        aligned[train_cols + holdout_cols] = aligned[train_cols + holdout_cols].fillna(
            aligned[train_cols + holdout_cols].mean()
        )

    if training_comparison_idx >= len(train_cols):
        raise ValueError(
            f"training_comparison_idx={training_comparison_idx} is out of range "
            f"for train_cols (len={len(train_cols)})"
        )

    train_arr = aligned[train_cols].values        # (N, n_train)
    holdout_arr = aligned[holdout_cols].values    # (N, n_holdout)
    n_leaves = len(fips_list)

    # Normalize training shifts
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_arr)

    # Fit full Ward dendrogram once.
    # NOTE: The K sweep uses unconstrained Ward HAC (no adjacency connectivity)
    # to produce balanced, stable clusters for holdout accuracy estimation.
    # The adjacency constraint is applied in the actual clustering step (Task 2,
    # cluster_communities.py). Unconstrained Ward produces more balanced splits
    # that are better suited for K selection; the spatial contiguity enforcement
    # is then applied once the optimal K is known.
    log.info("Fitting Ward dendrogram (this may take ~10s for 293 counties)...")
    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=None,
        n_clusters=1,
        compute_distances=True,
    )
    model.fit(train_norm)

    results = []
    for k in sorted(k_values):
        if k >= n_leaves:
            log.warning("k=%d >= n_leaves=%d, skipping", k, n_leaves)
            continue

        labels = _hc_cut(k, model.children_, n_leaves)
        sizes = np.bincount(labels)
        min_size = int(sizes.min())

        if min_size < min_community_size:
            log.info("k=%d: min community size %d < %d, skipping", k, min_size, min_community_size)
            continue

        # Community-level means: training col at training_comparison_idx vs holdout col 0
        train_means = np.array([
            train_arr[labels == i, training_comparison_idx].mean() for i in range(k)
        ])
        holdout_means = np.array([
            holdout_arr[labels == i, 0].mean() for i in range(k)
        ])

        if len(np.unique(train_means)) < 2 or len(np.unique(holdout_means)) < 2:
            log.warning("k=%d: degenerate means (constant vector), skipping", k)
            continue

        r = float(pearsonr(train_means, holdout_means).statistic)
        results.append(KSweepResult(
            k=k,
            holdout_r=r,
            min_community_size=min_size,
            community_sizes=sizes.tolist(),
        ))
        log.info("k=%d: holdout_r=%.4f, min_size=%d", k, r, min_size)

    return results


def pick_optimal_k(results: list[KSweepResult]) -> int:
    """Return the K with the highest holdout_r. Raises ValueError if no valid results."""
    if not results:
        raise ValueError("No valid K values found (all failed min community size constraint)")
    return max(results, key=lambda r: r.holdout_r).k


def update_config_k(k: int, config_path: Path = CONFIG_PATH) -> None:
    """Write the chosen K back to config/model.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "clustering" not in cfg:
        raise ValueError(f"config/model.yaml is missing 'clustering' section: {config_path}")
    cfg["clustering"]["k"] = k
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Updated config/model.yaml: clustering.k = %d", k)


def main() -> None:
    parser = argparse.ArgumentParser(description="K selection sweep for Ward HAC")
    parser.add_argument(
        "--k-values", nargs="+", type=int,
        default=[5, 7, 10, 12, 15, 20, 25, 30],
        help="K values to sweep"
    )
    parser.add_argument(
        "--min-size", type=int, default=8,
        help="Minimum counties per community"
    )
    parser.add_argument(
        "--no-update-config", action="store_true",
        help="Print result but do not update config/model.yaml"
    )
    args = parser.parse_args()

    for path in (SHIFTS_PATH, ADJ_NPZ, ADJ_FIPS):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    log.info("Loading shifts from %s", SHIFTS_PATH)
    shifts = pd.read_parquet(SHIFTS_PATH)
    shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)

    fips_list = ADJ_FIPS.read_text().splitlines()
    W = load_npz(str(ADJ_NPZ))

    from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS
    train_cols = TRAINING_SHIFT_COLS
    holdout_cols = HOLDOUT_SHIFT_COLS

    results = run_k_sweep(
        shifts, fips_list, W,
        train_cols=train_cols,
        holdout_cols=holdout_cols,
        k_values=args.k_values,
        min_community_size=args.min_size,
    )

    print("\n=== K Selection Results ===")
    print(f"{'k':>4}  {'holdout_r':>10}  {'min_size':>9}  {'community_sizes'}")
    for r in results:
        print(f"{r.k:>4}  {r.holdout_r:>10.4f}  {r.min_community_size:>9}  {r.community_sizes}")

    if not results:
        log.error("No valid K values — check min_community_size constraint")
        return

    optimal_k = pick_optimal_k(results)
    best = next(r for r in results if r.k == optimal_k)
    print(f"\nOptimal K = {optimal_k} (holdout_r = {best.holdout_r:.4f})")

    if not args.no_update_config:
        update_config_k(optimal_k)
        print(f"Updated config/model.yaml: clustering.k = {optimal_k}")


if __name__ == "__main__":
    main()
