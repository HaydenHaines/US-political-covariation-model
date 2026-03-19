"""Generate Phase 1 validation report.

Computes:
  - Holdout Pearson r and MAE at the chosen K
  - Comparison to 3-cycle baseline
  - Community-level predictions vs actuals for 2020->2024

Usage:
    python src/validation/generate_validation_report.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 3-cycle baseline holdout r values (from validate_county_holdout_multiyear.py)
BASELINE = {5: 0.983, 7: 0.964, 10: 0.941, 15: 0.934, 20: 0.932}
PRES_D_16_20_COL = 12  # index within 30 training cols = pres_d_shift_16_20


def generate_report() -> dict:
    from src.core import config as _cfg_mod
    from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS

    # Load K at runtime (not import-time) — select_k.py wrote it after module import
    k = _cfg_mod.load()["clustering"]["k"]
    if k is None:
        raise RuntimeError("clustering.k is null — run select_k.py first")

    shifts = pd.read_parquet(PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet")
    shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)

    fips_list = (PROJECT_ROOT / "data" / "communities" / "county_adjacency.fips.txt").read_text().splitlines()
    W = load_npz(str(PROJECT_ROOT / "data" / "communities" / "county_adjacency.npz"))

    indexed = shifts.set_index("county_fips").reindex(fips_list)
    train_arr = indexed[TRAINING_SHIFT_COLS].values
    holdout_arr = indexed[HOLDOUT_SHIFT_COLS].values

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_arr)

    model = AgglomerativeClustering(linkage="ward", connectivity=W, n_clusters=1, compute_distances=True)
    model.fit(train_norm)
    labels = _hc_cut(k, model.children_, len(fips_list))

    train_means = np.array([train_arr[labels == i, PRES_D_16_20_COL].mean() for i in range(k)])
    holdout_means = np.array([holdout_arr[labels == i, 0].mean() for i in range(k)])

    r = float(pearsonr(train_means, holdout_means).statistic)
    mae = float(np.mean(np.abs(train_means - holdout_means)))

    baseline_r = BASELINE.get(k, None)

    report = {
        "chosen_k": k,
        "holdout_r_multiyear": r,
        "holdout_mae": mae,
        "baseline_3cycle_r": baseline_r,
        "delta_vs_baseline": r - baseline_r if baseline_r is not None else None,
        "community_sizes": np.bincount(labels).tolist(),
    }

    print("\n=== Phase 1 Validation Report ===")
    print(f"Chosen K          : {k}")
    print(f"Holdout r         : {r:.4f}")
    print(f"Holdout MAE       : {mae:.4f}")
    if baseline_r:
        print(f"3-cycle baseline r: {baseline_r:.4f}")
        print(f"Delta             : {r - baseline_r:.4f}")
    print(f"Community sizes   : {np.bincount(labels).tolist()}")

    return report


if __name__ == "__main__":
    generate_report()
