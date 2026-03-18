"""
Stage 2 community detection: NMF elbow curve analysis.

Fits NMF for K=4..12 with multiple random restarts per K. Reports
reconstruction error to identify where adding another component stops
buying meaningful structure (the elbow).

Features are min-max normalized to [0, 1] before NMF. Uninhabited
tracts (pop=0) are excluded from fitting; they receive NaN membership
vectors in the output.

Input:   data/assembled/tract_features.parquet
Outputs: data/communities/nmf_elbow.csv    (K, restart, error, converged)
         data/communities/nmf_elbow.png    (elbow plot)

Reference: docs/references/RESOURCE_INDEX.md — S2 known gaps
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
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

K_RANGE = range(4, 13)   # K=4..12 inclusive
N_RESTARTS = 15          # random restarts per K for random init; ignored for nndsvda
MAX_ITER = 3000          # raised from 500 — nndsvda typically converges in <300


def load_and_normalize(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load features, exclude uninhabited tracts, min-max normalize.

    Returns:
        X: (n_populated_tracts, n_features) float array, values in [0, 1]
        meta: DataFrame with tract_geoid and is_uninhabited for all 9,393 tracts
    """
    df = pd.read_parquet(path)
    meta = df[["tract_geoid", "is_uninhabited"]].copy()

    # Fit and transform on populated tracts only
    populated = df[~df["is_uninhabited"]][FEATURE_COLS].copy()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(populated)

    n_nan = np.isnan(X).sum()
    if n_nan > 0:
        raise ValueError(f"{n_nan} NaN values in feature matrix after scaling — check imputation")

    log.info("Feature matrix: %d tracts × %d features (min-max scaled)", *X.shape)
    return X, meta, populated.index, scaler


def fit_nmf_single(X: np.ndarray, k: int, random_state: int | None = None) -> tuple[float, bool, int]:
    """
    Fit NMF with nndsvda initialization; return (error, converged, n_iter).

    nndsvda (Non-Negative Double SVD with average fill) provides a deterministic,
    near-optimal starting point. It typically converges in <300 iterations vs 500+
    for random init, and produces more stable, reproducible results.
    nndsvda does not use random_state (deterministic), so random_state is ignored.
    """
    model = NMF(
        n_components=k,
        init="nndsvda",
        max_iter=MAX_ITER,
        tol=1e-4,
    )
    model.fit(X)
    converged = model.n_iter_ < MAX_ITER
    return model.reconstruction_err_, converged, model.n_iter_


def run_elbow(X: np.ndarray) -> pd.DataFrame:
    """
    Run NMF once per K (nndsvda is deterministic — multiple restarts not needed).

    Returns a DataFrame with columns: k, restart, error, converged, n_iter.
    """
    records = []
    for k in K_RANGE:
        t0 = time.time()
        err, converged, n_iter = fit_nmf_single(X, k)
        elapsed = time.time() - t0
        if not converged:
            log.warning("  K=%2d | DID NOT CONVERGE in %d iters — increase MAX_ITER", k, MAX_ITER)
        log.info(
            "  K=%2d | err=%.4f | iters=%d | converged=%s | %.1fs",
            k, err, n_iter, converged, elapsed,
        )
        records.append({"k": k, "restart": 0, "error": err, "converged": converged, "n_iter": n_iter})
    return pd.DataFrame(records)


def plot_elbow(results: pd.DataFrame, output_path: Path) -> None:
    """Plot min and mean reconstruction error vs K."""
    summary = results.groupby("k")["error"].agg(["min", "mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(summary["k"], summary["min"], "o-", color="#2563eb", linewidth=2,
            markersize=6, label="Min error (best restart)")
    ax.fill_between(
        summary["k"],
        summary["mean"] - summary["std"],
        summary["mean"] + summary["std"],
        alpha=0.15, color="#2563eb", label="Mean ± 1 std (across restarts)",
    )
    ax.plot(summary["k"], summary["mean"], "--", color="#2563eb", linewidth=1, alpha=0.6)

    ax.set_xlabel("Number of community types (K)", fontsize=12)
    ax.set_ylabel("NMF reconstruction error (Frobenius norm)", fontsize=12)
    ax.set_title("NMF elbow curve — FL + GA + AL census tracts", fontsize=13)
    ax.set_xticks(list(K_RANGE))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate percent improvement per step
    for i in range(1, len(summary)):
        pct_drop = 100 * (summary["min"].iloc[i - 1] - summary["min"].iloc[i]) / summary["min"].iloc[i - 1]
        ax.annotate(
            f"−{pct_drop:.1f}%",
            xy=(summary["k"].iloc[i], summary["min"].iloc[i]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8, color="#64748b",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Saved elbow plot → %s", output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, meta, populated_idx, scaler = load_and_normalize(INPUT_PATH)

    log.info("Running NMF elbow: K=%d..%d, %d restarts each...",
             min(K_RANGE), max(K_RANGE), N_RESTARTS)
    results = run_elbow(X)

    csv_path = OUTPUT_DIR / "nmf_elbow.csv"
    results.to_csv(csv_path, index=False)
    log.info("Saved elbow data → %s", csv_path)

    plot_elbow(results, OUTPUT_DIR / "nmf_elbow.png")

    # Print summary table
    summary = results.groupby("k")["error"].agg(["min", "mean"]).round(4)
    summary.columns = ["min_error", "mean_error"]
    summary["pct_gain_vs_prev"] = summary["min_error"].pct_change() * 100
    print("\n=== Elbow summary ===")
    print(summary.to_string())
    print("\nElbow typically where |pct_gain_vs_prev| drops below ~2–3%")


if __name__ == "__main__":
    main()
