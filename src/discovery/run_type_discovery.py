"""KMeans-based electoral type discovery.

Discovers J electoral types directly from county shift vectors. Each county
gets a hard cluster assignment (dominant type) and soft membership scores
based on inverse distance to centroids. Types are abstract archetypes:
Rural Conservative, Black Belt, College Town, etc.

Switched from SVD+varimax (which produced degenerate 2-type solutions) to
KMeans after empirical testing showed KMeans achieves holdout r=0.77 vs
SVD+varimax r=0.35 on recent (2008+) data.

Usage:
    python -m src.discovery.run_type_discovery [--j J]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class TypeDiscoveryResult:
    scores: np.ndarray  # N × J soft membership (inverse-distance, row-normalized to sum to 1)
    loadings: np.ndarray  # J × D cluster centroids (type shift profiles)
    dominant_types: np.ndarray  # N array of dominant type indices
    explained_variance: np.ndarray  # J array — fraction of counties per type
    rotation_matrix: np.ndarray  # J × J identity (kept for interface compat)


def varimax(
    Phi: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Varimax rotation of a factor loading matrix (retained for backward compat).

    Parameters
    ----------
    Phi : ndarray of shape (p, k)
        Unrotated factor loading or score matrix.
    gamma : float
        Kaiser normalization parameter (1.0 = standard varimax).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on rotation matrix change.

    Returns
    -------
    rotated : ndarray of shape (p, k)
        Rotated matrix.
    R : ndarray of shape (k, k)
        Orthogonal rotation matrix.
    """
    p, k = Phi.shape
    R = np.eye(k)
    for _ in range(max_iter):
        Lambda = Phi @ R
        # Varimax criterion gradient
        A = Lambda**3 - (gamma / p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0))
        u, _s, vt = np.linalg.svd(Phi.T @ A)
        R_new = u @ vt
        if np.max(np.abs(R_new - R)) < tol:
            R = R_new
            break
        R = R_new
    return Phi @ R, R


def temperature_soft_membership(dists: np.ndarray, T: float) -> np.ndarray:
    """Compute temperature-sharpened soft membership from centroid distances.

    Formula: weight_j = (1 / (dist_j + eps))^T, then row-normalize.

    Numerically stable: operates in log-space to avoid overflow at large T.
    For T >= 500 falls back to hard (argmax) assignment directly.

    Parameters
    ----------
    dists : ndarray of shape (N, J)
        Euclidean distances to centroids (non-negative).
    T : float
        Temperature exponent.  T=1.0 = original inverse-distance baseline.
        T=10.0 = production default (reduces calibration MAE by ~37%).
        T→∞ approaches hard assignment.

    Returns
    -------
    scores : ndarray of shape (N, J)
        Non-negative soft membership weights, each row sums to 1.
    """
    N, J_ = dists.shape
    eps = 1e-10

    if T >= 500.0:
        scores = np.zeros((N, J_))
        nearest = np.argmin(dists, axis=1)
        scores[np.arange(N), nearest] = 1.0
        return scores

    # Log-space: log(weight_j) = T * log(1/(dist_j + eps)) = -T * log(dist_j + eps)
    log_weights = -T * np.log(dists + eps)  # (N, J)

    # Numerically stable softmax: subtract row max before exponentiating
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return powered / row_sums


def discover_types(
    shift_matrix: np.ndarray,
    j: int,
    random_state: int = 42,
    temperature: float = 10.0,
    pca_components: int | None = None,
) -> TypeDiscoveryResult:
    """KMeans clustering on county shift vectors, with optional PCA compression.

    Parameters
    ----------
    shift_matrix : ndarray of shape (N, D)
        County shift vectors (log-odds shifts). N counties, D shift dimensions.
        Should already be scaled and presidential-weighted.
    j : int
        Number of types to discover.
    random_state : int
        Random seed for KMeans initialization.
    temperature : float
        Temperature exponent for soft membership sharpening.
        T=1.0 = original inverse-distance baseline.
        T=10.0 = production default, reduces calibration MAE by ~37%.
        See scripts/experiment_soft_membership.py for derivation.
    pca_components : int or None
        If set, apply PCA to reduce shift_matrix to this many dimensions
        before KMeans. None = no PCA (backward compatible default).
        Presidential shifts are highly correlated, so 33→15 typically retains
        90%+ variance.

    Returns
    -------
    TypeDiscoveryResult
        Soft scores (temperature-scaled inverse-distance), centroids,
        dominant type assignments, type size fractions, and identity
        rotation matrix.
    """
    # Optional PCA dimensionality reduction
    if pca_components is not None and pca_components < shift_matrix.shape[1]:
        pca = PCA(n_components=pca_components, random_state=random_state)
        shift_matrix = pca.fit_transform(shift_matrix)
        cumulative_var = pca.explained_variance_ratio_.cumsum()
        print(f"PCA: {shift_matrix.shape[1]} components retain "
              f"{cumulative_var[-1]:.1%} variance "
              f"(top-5: {', '.join(f'{v:.1%}' for v in cumulative_var[:5])})")

    km = KMeans(n_clusters=j, random_state=random_state, n_init=10)
    labels = km.fit_predict(shift_matrix)
    centroids = km.cluster_centers_  # (J, D)

    # Soft membership: temperature-scaled inverse distance to each centroid
    dists = np.zeros((len(shift_matrix), j))
    for t in range(j):
        dists[:, t] = np.linalg.norm(shift_matrix - centroids[t], axis=1)
    soft_scores = temperature_soft_membership(dists, T=temperature)

    # Type size fractions as "explained variance" analog
    _, counts = np.unique(labels, return_counts=True)
    type_fractions = counts / len(labels)

    return TypeDiscoveryResult(
        scores=soft_scores,
        loadings=centroids,
        dominant_types=labels,
        explained_variance=type_fractions,
        rotation_matrix=np.eye(j),
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="KMeans type discovery")
    parser.add_argument("--j", type=int, default=None, help="Number of types (overrides config)")
    parser.add_argument("--min-year", type=int, default=2008, help="Minimum start year for shift pairs (default: 2008)")
    parser.add_argument("--pca", type=int, default=None, help="PCA components before KMeans (overrides config)")
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    j = args.j or config["types"].get("j")
    if j is None:
        # Try J selection results
        j_results_path = PROJECT_ROOT / "data" / "communities" / "j_selection_results.parquet"
        if j_results_path.exists():
            j_df = pd.read_parquet(j_results_path)
            j = int(j_df.loc[j_df["mean_holdout_r"].idxmax(), "j"])
            print(f"Using best J={j} from J selection results")
        else:
            raise ValueError(
                "No J specified: set types.j in config, use --j, or run select_j first"
            )

    temperature = float(config["types"].get("temperature", 10.0))
    # --pca CLI overrides config; --pca 0 explicitly disables PCA
    if args.pca is not None:
        pca_components = args.pca if args.pca > 0 else None
    else:
        pca_components = config["types"].get("pca_components")
    if pca_components is not None:
        pca_components = int(pca_components)

    # Load shift matrix
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    # Separate FIPS and shift columns, filter to recent data
    county_fips = df["county_fips"].values
    all_shift_cols = [c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS]

    # Filter to recent shift pairs (start year >= min_year)
    shift_cols = []
    for c in all_shift_cols:
        parts = c.split("_")
        y2 = int(parts[-2])
        y1 = y2 + (1900 if y2 >= 50 else 2000)
        if y1 >= args.min_year:
            shift_cols.append(c)

    if not shift_cols:
        shift_cols = all_shift_cols  # fallback to all if filter removes everything

    shift_matrix = df[shift_cols].values

    # Apply StandardScaler, then presidential weighting (post-scaling).
    # Without scaling, gov/senate shifts with different variances dominate
    # KMeans by Euclidean distance magnitude, not signal content.
    presidential_weight = float(config["types"].get("presidential_weight", 4.0))
    scaler = StandardScaler()
    shift_matrix = scaler.fit_transform(shift_matrix)

    if presidential_weight != 1.0:
        pres_indices = [i for i, c in enumerate(shift_cols) if "pres_" in c]
        shift_matrix[:, pres_indices] *= presidential_weight
        print(f"Applied presidential weight={presidential_weight} to {len(pres_indices)} columns (post-scaling)")

    print(f"Shift matrix: {shift_matrix.shape[0]} counties x {shift_matrix.shape[1]} dims (min_year={args.min_year})")
    pca_msg = f", pca_components={pca_components}" if pca_components else ""
    print(f"Discovering J={j} types via KMeans (temperature={temperature}{pca_msg})...")

    result = discover_types(shift_matrix, j=j, temperature=temperature, pca_components=pca_components)

    unique, counts = np.unique(result.dominant_types, return_counts=True)
    print(f"Type sizes: {sorted(counts.tolist(), reverse=True)}")

    # Save results
    out_dir = PROJECT_ROOT / "data" / "communities"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({"county_fips": county_fips})
    for i in range(j):
        out_df[f"type_{i}_score"] = result.scores[:, i]
    out_df["dominant_type"] = result.dominant_types

    out_path = out_dir / "type_assignments.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved type assignments to {out_path}")


if __name__ == "__main__":
    main()
