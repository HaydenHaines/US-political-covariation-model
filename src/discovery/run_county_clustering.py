"""Run county-level agglomerative clustering on electoral shift vectors.

Loads county_shifts_multiyear.parquet (30-dim training shifts, 2000–2022)
and county_adjacency.npz. K is read from config/model.yaml (set by select_k.py).

Outputs (Layer 1 — geographic community assignment):
    data/communities/county_community_assignments.parquet — county_fips, community_id
      (also written with legacy column name 'community' for backward compatibility)

Outputs (Layer 2 — electoral type stub):
    data/communities/county_type_assignments_stub.parquet — community_id, type_weight_*, dominant_type_id

Usage:
    uv run python src/discovery/run_county_clustering.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut
from sklearn.preprocessing import StandardScaler

from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
ADJ_NPZ = PROJECT_ROOT / "data" / "communities" / "county_adjacency.npz"
ADJ_FIPS = PROJECT_ROOT / "data" / "communities" / "county_adjacency.fips.txt"
OUT_PATH = PROJECT_ROOT / "data" / "communities" / "county_community_assignments.parquet"


def within_cluster_variance(shifts: np.ndarray, labels: np.ndarray) -> float:
    """Weighted mean within-cluster variance across all clusters."""
    total_var = 0.0
    total_weight = 0.0
    for label in np.unique(labels):
        mask = labels == label
        count = int(mask.sum())
        var = float(np.var(shifts[mask], ddof=0)) if count > 1 else 0.0
        total_var += var * count
        total_weight += count
    return total_var / total_weight if total_weight > 0 else 0.0


def main() -> None:
    # ── Read K from config (set by select_k.py) ───────────────────────────────
    # Use config.load() at runtime — do NOT use module-level cached config
    # because select_k.py writes the K choice after import time.
    from src.core import config as _cfg_mod
    _live_cfg = _cfg_mod.load()
    k_target = _live_cfg["clustering"]["k"]
    if k_target is None:
        raise RuntimeError(
            "clustering.k is null in config/model.yaml. "
            "Run src/discovery/select_k.py first."
        )
    log.info("Using K=%d from config/model.yaml", k_target)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading county shift vectors...")
    shifts_df = pd.read_parquet(SHIFTS_PATH)
    shifts_df["county_fips"] = shifts_df["county_fips"].astype(str).str.zfill(5)

    log.info("Loading adjacency matrix...")
    W = load_npz(str(ADJ_NPZ))
    fips_list = ADJ_FIPS.read_text().splitlines()

    # ── Align shifts to adjacency ordering ────────────────────────────────────
    shifts_indexed = shifts_df.set_index("county_fips")
    aligned = shifts_indexed.reindex(fips_list)
    n_missing = aligned[TRAINING_SHIFT_COLS[0]].isna().sum()
    if n_missing:
        log.warning("Filling %d counties with NaN shifts using column means", n_missing)
        col_means = aligned[TRAINING_SHIFT_COLS].mean()
        aligned[TRAINING_SHIFT_COLS] = aligned[TRAINING_SHIFT_COLS].fillna(col_means)

    # Cluster on the 30 training dimensions only (exclude holdout cols)
    train_shifts = aligned[TRAINING_SHIFT_COLS].values  # (293, 30)

    log.info(
        "Data: %d counties, train dims=%d",
        len(fips_list), train_shifts.shape[1],
    )

    # ── Normalize training shifts ─────────────────────────────────────────────
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_shifts)

    # ── Fit full Ward tree (n_clusters=1 builds the full dendrogram) ──────────
    log.info("Fitting Ward dendrogram (n_clusters=1)...")
    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=W,
        n_clusters=1,
        compute_distances=True,
    )
    model.fit(train_norm)
    n_leaves = len(fips_list)

    log.info("Clustering at k=%d...", k_target)
    labels_final = _hc_cut(k_target, model.children_, n_leaves)

    print(f"\nk={k_target} cluster sizes: {dict(zip(*np.unique(labels_final, return_counts=True)))}")

    # ── Save assignments (Layer 1) ────────────────────────────────────────────
    # Canonical column name is community_id; keep 'community' for backward compat.
    assignments = pd.DataFrame({
        "county_fips": fips_list,
        "community_id": labels_final,
        "community": labels_final,  # legacy alias — downstream scripts may use this
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(OUT_PATH, index=False)
    log.info("Layer 1 community assignments saved to %s", OUT_PATH)

    print(f"\nFinal: {k_target} communities assigned to {len(fips_list)} counties")

    # ── Produce Layer 2 stub (type assignments) ───────────────────────────────
    # Full NMF implementation is Phase 1 work. This stub preserves the pipeline
    # end-to-end so downstream code can depend on the Layer 2 output format.
    try:
        from src.models.type_classifier import run_type_classification
        type_stub_path = OUT_PATH.parent / "county_type_assignments_stub.parquet"
        # Default J=7 matches the historical NMF K=7 canonical choice
        run_type_classification(
            shifts_path=SHIFTS_PATH,
            assignments_path=OUT_PATH,
            shift_cols=TRAINING_SHIFT_COLS,
            j=7,
            output_path=type_stub_path,
        )
        log.info("Layer 2 stub type assignments saved to %s", type_stub_path)
    except Exception as exc:
        log.warning("Layer 2 stub generation failed (non-fatal): %s", exc)


if __name__ == "__main__":
    main()
