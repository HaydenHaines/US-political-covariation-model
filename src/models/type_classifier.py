"""Layer 2: Electoral type classification via NMF on community shift profiles.

Takes Layer 1 community assignments + shift vectors, computes per-community
mean shift profiles, then fits NMF to discover J electoral types.

Types are abstract archetypes that can appear in multiple non-contiguous communities.
Rural Georgia and rural Washington may be different communities but the same type.

run_type_classification delegates to src.models.nmf_types.run_nmf_classification
for the real sklearn NMF implementation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from src.models.nmf_types import run_nmf_classification

log = logging.getLogger(__name__)


def compute_community_profiles(
    shifts: pd.DataFrame,
    assignments: pd.DataFrame,
    shift_cols: list[str],
) -> pd.DataFrame:
    """Compute mean shift vector for each community.

    Parameters
    ----------
    shifts : DataFrame with county_fips + shift columns
    assignments : DataFrame with county_fips + community_id
    shift_cols : list of shift column names to average

    Returns
    -------
    DataFrame with community_id + one column per shift dim (mean across member counties)
    """
    merged = shifts.merge(assignments[["county_fips", "community_id"]], on="county_fips")
    profiles = (
        merged.groupby("community_id")[shift_cols]
        .mean()
        .reset_index()
    )
    return profiles


def classify_types(
    community_profiles: pd.DataFrame,
    shift_cols: list[str],
    j: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit NMF to community shift profiles -> J electoral types.

    Now backed by real sklearn NMF via src.models.nmf_types.fit_nmf.
    community_profiles must be a DataFrame with a 'community_id' column
    and one column per shift dim.

    Returns
    -------
    DataFrame with community_id + type_weight_{j} for j in range(J)
    + dominant_type_id (argmax of weights)
    """
    from src.models.nmf_types import fit_nmf

    profiles_arr = community_profiles[shift_cols].values
    result = fit_nmf(profiles_arr, j=j, random_state=random_state)

    out = community_profiles[["community_id"]].copy().reset_index(drop=True)
    for j_idx in range(j):
        out[f"type_weight_{j_idx}"] = result.W[:, j_idx]
    out["dominant_type_id"] = result.dominant_type
    return out


def run_type_classification(
    shifts_path: Path,
    assignments_path: Path,
    shift_cols: list[str],
    j: int,
    output_path: Path,
) -> pd.DataFrame:
    """End-to-end type classification pipeline.

    Delegates to src.models.nmf_types.run_nmf_classification for real NMF.
    Returns the NMFResult object from nmf_types.
    """
    return run_nmf_classification(
        shifts_path=shifts_path,
        assignments_path=assignments_path,
        shift_cols=shift_cols,
        j=j,
        output_path=output_path,
    )
