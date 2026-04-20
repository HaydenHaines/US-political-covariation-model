"""Composite scores, fit scoring, and talent pipeline.

Aggregates per-race stats into career CTOV summaries.
Computes candidate-district fit scores for recruitment scouting.

The central concept: a candidate's career-average CTOV (Community-Type
Overperformance Vector) describes *where* they outperform. A district's W
vector (type composition) describes *which types* live there. Fit = dot(CTOV, W):
how well do the candidate's skills match what the district needs?
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CTOV_PATH = PROJECT_ROOT / "data" / "sabermetrics" / "candidate_ctov.parquet"

# Number of top contributing types to include in the `top_types` field.
_N_TOP_TYPES = 5

# Minimum races needed for a CTOV estimate to be considered reliable.
_MIN_RACES_DEFAULT = 2


# ---------------------------------------------------------------------------
# Career CTOV aggregation
# ---------------------------------------------------------------------------


def compute_career_ctovs(
    ctov_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate per-race CTOV rows into a single career-average CTOV per candidate.

    Parameters
    ----------
    ctov_df : pd.DataFrame | None
        Output of the Phase 2 residuals pipeline. Columns: person_id, name, party,
        year, state, office, mvd, ctov_type_0 … ctov_type_N.
        If None, loads from the default path (data/sabermetrics/candidate_ctov.parquet).

    Returns
    -------
    pd.DataFrame
        One row per (person_id, party). Columns:
          person_id, name, party, n_races,
          ctov_type_0 … ctov_type_N  (mean across all races)
    """
    if ctov_df is None:
        if not _CTOV_PATH.exists():
            raise FileNotFoundError(f"CTOV parquet not found at {_CTOV_PATH}. Run the sabermetrics pipeline first.")
        ctov_df = pd.read_parquet(_CTOV_PATH)

    type_cols = [c for c in ctov_df.columns if c.startswith("ctov_type_")]
    if not type_cols:
        raise ValueError("ctov_df has no ctov_type_* columns")

    group_cols = ["person_id", "name", "party"]
    agg: dict[str, str | list] = {col: "mean" for col in type_cols}
    agg["year"] = "count"  # reuse as n_races counter; rename below

    career = (
        ctov_df[group_cols + type_cols + ["year"]]
        .groupby(group_cols, as_index=False)
        .agg({"year": "count", **{col: "mean" for col in type_cols}})
        .rename(columns={"year": "n_races"})
    )
    return career


# ---------------------------------------------------------------------------
# Core fit score
# ---------------------------------------------------------------------------


def compute_fit_score(
    candidate_ctov: np.ndarray,
    district_W: np.ndarray,
) -> float:
    """Candidate-district fit score: dot product of CTOV and district type weights.

    A positive CTOV component means the candidate outperforms in that type.
    A large W component means that type makes up a large share of the district.
    Fit measures how much of the district's type composition the candidate can
    exploit — i.e., how aligned their skill profile is with the local electorate.

    Parameters
    ----------
    candidate_ctov : np.ndarray
        Career-average CTOV vector (J,). Values can be positive or negative.
    district_W : np.ndarray
        District community-type weights (J,). Must sum to 1.

    Returns
    -------
    float
        Fit score. Positive = candidate tends to overperform in the types that
        dominate this district. Negative = candidate underperforms there.
    """
    c = np.asarray(candidate_ctov, dtype=float)
    w = np.asarray(district_W, dtype=float)
    if c.shape != w.shape:
        raise ValueError(f"Shape mismatch: CTOV {c.shape} vs W {w.shape}")
    return float(np.dot(c, w))


# ---------------------------------------------------------------------------
# District ranking
# ---------------------------------------------------------------------------


def rank_candidates_for_district(
    career_ctov_df: pd.DataFrame,
    target_W: np.ndarray,
    party_filter: str | None = None,
    min_races: int = _MIN_RACES_DEFAULT,
) -> pd.DataFrame:
    """Rank candidates by fit score for a target district.

    The Moneyball scouting report: given a district's type composition W,
    which multi-race candidates have skills that match what the district needs?

    Parameters
    ----------
    career_ctov_df : pd.DataFrame
        Output of compute_career_ctovs(). Rows are candidates with their
        career-average CTOV vectors and n_races.
    target_W : np.ndarray
        Target district's community-type weights (J,). Must sum to 1.
    party_filter : str | None
        If "D" or "R", restrict to candidates from that party. None = all parties.
    min_races : int
        Minimum number of races a candidate must have for inclusion.
        Single-race candidates have noisy CTOVs; min_races=2 is the default.

    Returns
    -------
    pd.DataFrame
        Candidates ranked by fit score, highest first. Columns:
          rank, person_id, name, party, n_races, fit_score, top_types
    """
    type_cols = [c for c in career_ctov_df.columns if c.startswith("ctov_type_")]
    if not type_cols:
        raise ValueError("career_ctov_df has no ctov_type_* columns")

    W = np.asarray(target_W, dtype=float)
    if W.shape[0] != len(type_cols):
        raise ValueError(f"W length {W.shape[0]} does not match CTOV dimension {len(type_cols)}")

    pool = career_ctov_df.copy()
    if party_filter is not None:
        pool = pool[pool["party"] == party_filter]
    pool = pool[pool["n_races"] >= min_races]
    if pool.empty:
        return pd.DataFrame(columns=["rank", "person_id", "name", "party", "n_races", "fit_score", "top_types"])

    ctov_matrix = pool[type_cols].to_numpy(dtype=float)
    fit_scores = ctov_matrix @ W  # shape (N_candidates,)

    pool = pool.copy()
    pool["fit_score"] = fit_scores

    # For each candidate, find the top contributing types (ctov_j * W_j largest).
    def _top_types(row: pd.Series) -> list[int]:
        ctov_vec = row[type_cols].to_numpy(dtype=float)
        contributions = ctov_vec * W
        # Return type indices sorted by contribution magnitude (positive first).
        idx = np.argsort(-contributions)
        return [int(i) for i in idx[:_N_TOP_TYPES]]

    pool["top_types"] = pool.apply(_top_types, axis=1)

    result = (
        pool[["person_id", "name", "party", "n_races", "fit_score", "top_types"]]
        .sort_values("fit_score", ascending=False)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", result.index + 1)
    return result
