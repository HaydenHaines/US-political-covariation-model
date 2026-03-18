"""
Stage 2 community detection: fit final NMF at K=8.

Fits the model, normalizes membership vectors, prints component profiles,
and saves the two output matrices that all downstream stages consume.

Outputs:
  data/communities/tract_memberships_k8.parquet
      tract_geoid + 8 membership columns (rows sum to 1.0, soft assignment)
      is_uninhabited tracts receive NaN memberships.

  data/communities/components_k8.parquet
      8 rows × 12 feature columns (the H matrix — community "fingerprints")
      Each row describes one community type's feature profile.

Input:  data/assembled/tract_features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "tract_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "communities"

FEATURE_COLS = [
    "pct_white_nh",
    "pct_black",
    "pct_asian",
    "pct_hispanic",
    "log_median_income",
    "pct_mgmt_occ",
    "pct_owner_occ",
    "pct_car_commute",
    "pct_transit_commute",
    "pct_wfh_commute",
    "pct_college_plus",
    "median_age",
]

K = 7


def load_and_normalize(path: Path) -> tuple[np.ndarray, pd.DataFrame, pd.Index, MinMaxScaler]:
    df = pd.read_parquet(path)
    meta = df[["tract_geoid", "is_uninhabited"]].copy()
    populated = df[~df["is_uninhabited"]][FEATURE_COLS].copy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(populated)
    return X, meta, populated.index, scaler


def fit_nmf(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit NMF; return (W, H) unnormalized."""
    model = NMF(n_components=k, init="nndsvda", max_iter=3000, tol=1e-4)
    W = model.fit_transform(X)
    H = model.components_
    log.info("Fit K=%d | err=%.4f | iters=%d | converged=%s",
             k, model.reconstruction_err_, model.n_iter_, model.n_iter_ < 3000)
    return W, H


def normalize_memberships(W: np.ndarray) -> np.ndarray:
    """
    Row-normalize W so each tract's membership vector sums to 1.0.

    This gives soft assignment probabilities: W[i, k] = fraction of tract i
    that belongs to community type k. Required for Stage 3 covariance model.
    """
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for all-zero rows
    return W / row_sums


def print_component_profiles(H: np.ndarray) -> None:
    """
    Print each component's top and bottom features to support manual labeling.

    H rows are normalized to [0, 1] per-feature for comparability across
    features with different scales (income vs. percentages).
    """
    # Normalize each feature column to [0, 1] across components
    H_norm = H.copy()
    col_max = H_norm.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1
    H_norm = H_norm / col_max

    print("\n" + "=" * 70)
    print(f"NMF K={K} component profiles  (feature loadings, normalized 0→1)")
    print("=" * 70)

    for k in range(K):
        row = H_norm[k]
        ranked = sorted(enumerate(row), key=lambda x: -x[1])
        top = [(FEATURE_COLS[i], v) for i, v in ranked[:4]]
        bot = [(FEATURE_COLS[i], v) for i, v in ranked[-3:]]

        top_str = "  ".join(f"{n}={v:.2f}" for n, v in top)
        bot_str = "  ".join(f"{n}={v:.2f}" for n, v in bot)
        print(f"\nComponent {k + 1:>2}:")
        print(f"  HIGH: {top_str}")
        print(f"  LOW:  {bot_str}")

    print("\n" + "=" * 70)
    print("Full H matrix (rows=components, cols=features, values normalized):")
    print("=" * 70)
    h_df = pd.DataFrame(H_norm.round(3), columns=FEATURE_COLS,
                        index=[f"C{k+1}" for k in range(K)])
    print(h_df.to_string())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, meta, populated_idx, scaler = load_and_normalize(INPUT_PATH)
    log.info("Feature matrix: %d populated tracts × %d features", *X.shape)

    W_raw, H = fit_nmf(X, K)
    W = normalize_memberships(W_raw)

    # ── Memberships output ────────────────────────────────────────────────────
    # Build full 9393-row DataFrame; uninhabited tracts get NaN
    comp_cols = [f"c{k+1}" for k in range(K)]

    w_populated = pd.DataFrame(W, index=populated_idx, columns=comp_cols)
    memberships = meta.copy()
    for col in comp_cols:
        memberships[col] = np.nan
    memberships.loc[populated_idx, comp_cols] = w_populated[comp_cols].values

    mem_path = OUTPUT_DIR / f"tract_memberships_k{K}.parquet"
    memberships.to_parquet(mem_path, index=False)
    log.info("Saved memberships → %s  (%d tracts × %d components)",
             mem_path, len(memberships), K)

    # ── Components output ─────────────────────────────────────────────────────
    comp_df = pd.DataFrame(H, columns=FEATURE_COLS,
                           index=[f"c{k+1}" for k in range(K)])
    comp_path = OUTPUT_DIR / f"components_k{K}.parquet"
    comp_df.to_parquet(comp_path)
    log.info("Saved components → %s", comp_path)

    # ── Quick membership stats ────────────────────────────────────────────────
    populated_mem = memberships[~memberships["is_uninhabited"]]
    dominant = populated_mem[comp_cols].idxmax(axis=1)
    print("\n=== Dominant community type per tract (plurality) ===")
    print(dominant.value_counts().sort_index().to_string())

    avg_entropy = -(populated_mem[comp_cols].values * np.log(
        populated_mem[comp_cols].values.clip(1e-10))).sum(axis=1).mean()
    max_entropy = np.log(K)
    print(f"\nAvg membership entropy: {avg_entropy:.3f} / {max_entropy:.3f} max")
    print(f"  (0 = pure membership, {max_entropy:.2f} = equal mixture of all {K})")

    print_component_profiles(H)


if __name__ == "__main__":
    main()
