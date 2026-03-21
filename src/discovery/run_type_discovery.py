"""SVD + varimax rotation for electoral type discovery.

Discovers J electoral types directly from county shift vectors. Each county
gets a rotated score vector (soft membership, can be negative = anti-correlated).
Types are abstract archetypes: Rural Conservative, Black Belt, College Town, etc.

Usage:
    python -m src.discovery.run_type_discovery [--j J]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import TruncatedSVD


@dataclass
class TypeDiscoveryResult:
    scores: np.ndarray  # N × J rotated county scores (soft membership)
    loadings: np.ndarray  # J × D rotated type shift profiles
    dominant_types: np.ndarray  # N array of dominant type indices
    explained_variance: np.ndarray  # J array of variance ratios
    rotation_matrix: np.ndarray  # J × J varimax rotation matrix


def varimax(
    Phi: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Varimax rotation of a factor loading matrix.

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


def discover_types(
    shift_matrix: np.ndarray,
    j: int,
    random_state: int = 42,
) -> TypeDiscoveryResult:
    """SVD + varimax rotation on county shift vectors.

    Parameters
    ----------
    shift_matrix : ndarray of shape (N, D)
        County shift vectors (log-odds shifts). N counties, D shift dimensions.
    j : int
        Number of types to discover.
    random_state : int
        Random seed for SVD initialization.

    Returns
    -------
    TypeDiscoveryResult
        Rotated scores, loadings, dominant type assignments, explained variance,
        and the rotation matrix.
    """
    # Center the shift matrix
    X = shift_matrix - shift_matrix.mean(axis=0)

    # SVD dimensionality reduction
    svd = TruncatedSVD(n_components=j, random_state=random_state)
    scores = svd.fit_transform(X)
    loadings = svd.components_  # shape (j, D)
    explained_variance = svd.explained_variance_ratio_

    # Varimax rotation for interpretability
    rotated_scores, rotation = varimax(scores)
    # Transform loadings to match rotated scores
    # If scores_rot = scores @ R, then loadings_rot = R.T @ loadings
    rotated_loadings = rotation.T @ loadings

    # Dominant type = highest absolute score
    dominant_types = np.argmax(np.abs(rotated_scores), axis=1)

    return TypeDiscoveryResult(
        scores=rotated_scores,
        loadings=rotated_loadings,
        dominant_types=dominant_types,
        explained_variance=explained_variance,
        rotation_matrix=rotation,
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

    parser = argparse.ArgumentParser(description="SVD + varimax type discovery")
    parser.add_argument("--j", type=int, default=None, help="Number of types (overrides config)")
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

    # Load shift matrix
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    # Separate FIPS and shift columns
    county_fips = df["county_fips"].values
    shift_cols = [c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS]
    shift_matrix = df[shift_cols].values

    print(f"Shift matrix: {shift_matrix.shape[0]} counties x {shift_matrix.shape[1]} dims")
    print(f"Discovering J={j} types via SVD + varimax rotation...")

    result = discover_types(shift_matrix, j=j)

    print(f"Explained variance (pre-rotation): {result.explained_variance.sum():.3f}")
    print(f"Dominant type distribution: {np.bincount(result.dominant_types)}")

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
