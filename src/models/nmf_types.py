"""Real NMF type classification for Layer 2 of the Bedrock pipeline.

Takes community shift profiles (K × n_dims) and fits sklearn NMF to
produce J electoral types with soft membership weights.

Layer 2 semantics:
  W[k, j] = community k's weight for type j (normalized to sum to 1)
  H[j, d] = type j's characteristic shift pattern (not normalized)
  dominant_type[k] = argmax(W[k, :])

J selection: sweep J=5,6,7,8; pick based on reconstruction error +
interpretability. Reconstruction error alone is insufficient — the user
must review type profiles and name them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


@dataclass
class NMFResult:
    j: int
    W: np.ndarray           # (K, J) community type weights, rows sum to 1
    H: np.ndarray           # (J, n_dims) type profiles
    dominant_type: np.ndarray  # (K,) argmax of W rows
    reconstruction_error: float


@dataclass
class JSweepEntry:
    j: int
    reconstruction_error: float
    result: NMFResult


def compute_community_profiles(
    shifts: pd.DataFrame,
    assignments: pd.DataFrame,
    shift_cols: list[str],
) -> np.ndarray:
    """Compute mean shift vector per community.

    Parameters
    ----------
    shifts:
        DataFrame with county_fips + shift_cols.
    assignments:
        DataFrame with county_fips + community_id.
    shift_cols:
        Names of shift columns to use.

    Returns
    -------
    profiles: np.ndarray of shape (K, len(shift_cols))
        Row k = mean shift vector for community k.
        Communities are ordered 0, 1, ..., K-1 by community_id.
    """
    merged = shifts.merge(assignments[["county_fips", "community_id"]], on="county_fips")
    k_ids = sorted(merged["community_id"].unique())
    profiles = np.array([
        merged[merged["community_id"] == k][shift_cols].mean().values
        for k in k_ids
    ])
    return profiles


def fit_nmf(
    community_profiles: np.ndarray,
    j: int,
    random_state: int = 42,
) -> NMFResult:
    """Fit NMF to community profiles and return normalized type weights.

    NMF requires non-negative input. Community shift profiles contain
    negative values (log-odds shifts). We apply a MinMax shift to [0, 1]
    before fitting, then extract the membership matrix W.

    The W matrix (community × type) is row-normalized so each community's
    type weights sum to 1 (soft membership probabilities).
    """
    # Shift to non-negative: MinMax scale to [0.01, 1.0]
    scaler = MinMaxScaler(feature_range=(0.01, 1.0))
    profiles_nn = scaler.fit_transform(community_profiles)

    n_samples, n_features = profiles_nn.shape
    # nndsvda requires n_components <= min(n_samples, n_features); fall back to random
    init = "nndsvda" if j <= min(n_samples, n_features) else "random"
    nmf = NMF(
        n_components=j,
        init=init,
        random_state=random_state,
        max_iter=500,
    )
    W_raw = nmf.fit_transform(profiles_nn)   # (K, J)
    H = nmf.components_                       # (J, n_dims) in transformed space

    # Row-normalize W to sum to 1
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # guard against zero rows
    W = W_raw / row_sums

    dominant_type = np.argmax(W, axis=1)

    return NMFResult(
        j=j,
        W=W,
        H=H,
        dominant_type=dominant_type,
        reconstruction_error=float(nmf.reconstruction_err_),
    )


def sweep_j(
    community_profiles: np.ndarray,
    j_values: list[int] | None = None,
    random_state: int = 42,
) -> list[JSweepEntry]:
    """Fit NMF at multiple J values and return reconstruction errors.

    Parameters
    ----------
    community_profiles:
        (K, n_dims) community shift profiles.
    j_values:
        J values to sweep. Defaults to [5, 6, 7, 8].
    random_state:
        Seed for reproducibility.

    Returns
    -------
    List of JSweepEntry sorted by j ascending.
    """
    if j_values is None:
        j_values = [5, 6, 7, 8]
    results = []
    for j in sorted(j_values):
        if j > len(community_profiles):
            continue  # can't have more types than communities
        result = fit_nmf(community_profiles, j=j, random_state=random_state)
        results.append(JSweepEntry(j=j, reconstruction_error=result.reconstruction_error, result=result))
    return results


def run_nmf_classification(
    shifts_path,
    assignments_path,
    shift_cols: list[str],
    j: int,
    output_path,
    random_state: int = 42,
) -> NMFResult:
    """End-to-end NMF classification pipeline.

    Reads shifts and assignments, computes profiles, fits NMF, writes parquet.

    Output parquet columns:
        community_id, type_weight_0, ..., type_weight_{J-1}, dominant_type_id, j
    """
    shifts = pd.read_parquet(shifts_path)
    assignments = pd.read_parquet(assignments_path)
    shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)

    profiles = compute_community_profiles(shifts, assignments, shift_cols)
    result = fit_nmf(profiles, j=j, random_state=random_state)

    k_ids = sorted(assignments["community_id"].unique())
    out = pd.DataFrame({"community_id": k_ids})
    for jj in range(j):
        out[f"type_weight_{jj}"] = result.W[:, jj]
    out["dominant_type_id"] = result.dominant_type
    out["j"] = j

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    return result
