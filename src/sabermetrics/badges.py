"""Badge derivation for political candidates.

Badges translate CTOV vectors into human-readable archetypes. Each badge
reflects a demographic dimension: does this candidate systematically
over/underperform in communities dominated by a particular demographic trait?

Badge math:
    score = dot(effective_ctov, demographic_feature_per_type)

where effective_ctov is the party-adjusted CTOV (flipped for R candidates so
that positive scores always mean overperformance in that demographic context).

A badge is awarded when |score| > 1 standard deviation above the
cross-candidate mean, ensuring badges are relative (not absolute) measures.

For multi-race candidates, badges use the average CTOV across their career,
not any single race — this is the career signal, not a one-year snapshot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Badge catalog
# ---------------------------------------------------------------------------

# Each badge entry maps a badge name to the type_profiles column it weights.
# Positive weight → higher demographic value → higher badge score.
# Negative weight → lower demographic value → higher badge score (inverted).
# "turnout_monster" is special: it uses state-level MVD directly.
#
# Column names must match what's actually in type_profiles.parquet.
_BADGE_CATALOG: list[dict] = [
    # --- Race & ethnicity ---
    {
        "badge": "Hispanic Appeal",
        "column": "pct_hispanic",
        "direction": 1,
        "description": "Overperforms in Hispanic-dominated communities",
    },
    {
        "badge": "Black Community Strength",
        "column": "pct_black",
        "direction": 1,
        "description": "Overperforms in Black-dominated communities",
    },
    {
        "badge": "White Working Class",
        "column": "pct_white_nh",
        "direction": 1,
        "description": "Overperforms in white non-Hispanic communities",
    },
    {
        "badge": "Asian Community Appeal",
        "column": "pct_asian",
        "direction": 1,
        "description": "Overperforms in Asian-dominated communities",
    },
    # --- Age ---
    {
        "badge": "Senior Whisperer",
        "column": "median_age",
        "direction": 1,
        "description": "Overperforms in older communities",
    },
    # --- Education & class ---
    {
        "badge": "Suburban Professional",
        "column": "pct_bachelors_plus",
        "direction": 1,
        "description": "Overperforms in college-educated suburban communities",
    },
    {
        "badge": "Graduate Elite",
        "column": "pct_graduate",
        "direction": 1,
        "description": "Overperforms in highly-educated graduate-degree communities",
    },
    {
        "badge": "Professional Class",
        "column": "pct_management",
        "direction": 1,
        "description": "Overperforms in management/professional-heavy communities",
    },
    # --- Geography & density ---
    {
        "badge": "Rural Populist",
        "column": "log_pop_density",
        "direction": -1,  # Negative: high density = urban; -1 = rural overperformance
        "description": "Overperforms in low-density rural communities",
    },
    {
        "badge": "Urban Core",
        "column": "pct_transit",
        "direction": 1,
        "description": "Overperforms in transit-heavy urban cores",
    },
    # --- Housing & economics ---
    {
        "badge": "Homeowner Base",
        "column": "pct_owner_occupied",
        "direction": 1,
        "description": "Overperforms in homeowner-heavy communities",
    },
    {
        "badge": "Wealthy Suburb",
        "column": "log_median_hh_income",
        "direction": 1,
        "description": "Overperforms in high-income communities",
    },
    {
        "badge": "Knowledge Economy",
        "column": "pct_wfh",
        "direction": 1,
        "description": "Overperforms in remote-work-heavy knowledge-economy communities",
    },
    # --- Religion ---
    {
        "badge": "Faith Coalition",
        "column": "evangelical_share",
        "direction": 1,
        "description": "Overperforms in evangelical-heavy communities",
    },
    {
        "badge": "Mainline Protestant",
        "column": "mainline_share",
        "direction": 1,
        "description": "Overperforms in mainline Protestant communities",
    },
    {
        "badge": "Catholic Appeal",
        "column": "catholic_share",
        "direction": 1,
        "description": "Overperforms in Catholic-heavy communities",
    },
    {
        "badge": "Black Church Alliance",
        "column": "black_protestant_share",
        "direction": 1,
        "description": "Overperforms in Black Protestant church communities",
    },
    {
        "badge": "Devout Community",
        "column": "religious_adherence_rate",
        "direction": 1,
        "description": "Overperforms in highly religious communities",
    },
    # --- Income composition ---
    {
        "badge": "Wage Earner Base",
        "column": "earnings_share",
        "direction": 1,
        "description": "Overperforms in communities where wages dominate income",
    },
    {
        "badge": "Transfer Community",
        "column": "transfers_share",
        "direction": 1,
        "description": "Overperforms in transfer-payment-dependent communities (retirement, disability)",
    },
    {
        "badge": "Investor Class",
        "column": "investment_share",
        "direction": 1,
        "description": "Overperforms in investment-income-heavy communities",
    },
    # --- Migration ---
    {
        "badge": "Boomtown Appeal",
        "column": "net_migration_rate",
        "direction": 1,
        "description": "Overperforms in fast-growing communities with net in-migration",
    },
    {
        "badge": "Affluent Transplant",
        "column": "avg_inflow_income",
        "direction": 1,
        "description": "Overperforms in communities attracting high-income movers",
    },
]

# "Turnout Monster" badge uses total MVD rather than CTOV decomposition.
# It's awarded to candidates with unusually large |MVD|, regardless of type.
_TURNOUT_MONSTER_BADGE = {
    "badge": "Turnout Monster",
    "description": "Unusually large overperformance relative to structural prediction",
}

# Badges are awarded at 1 std dev threshold above cross-candidate mean.
# This is intentionally permissive: ~16% of candidates get each badge,
# which is enough to be meaningful but not so rare it's useless.
_BADGE_THRESHOLD_STD = 1.0


# ---------------------------------------------------------------------------
# Type profile loader
# ---------------------------------------------------------------------------


def _load_type_profiles() -> pd.DataFrame:
    """Load per-type demographic feature vectors from type_profiles.parquet.

    Returns a DataFrame indexed by type_id with demographic columns.
    Used to project CTOV vectors onto demographic dimensions.
    """
    path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    df = pd.read_parquet(path)
    df = df.set_index("type_id").sort_index()
    return df


# ---------------------------------------------------------------------------
# Effective CTOV: party-adjust so positive = overperformance
# ---------------------------------------------------------------------------


def _effective_ctov(ctov: np.ndarray, party: str) -> np.ndarray:
    """Return the party-adjusted CTOV so positive = candidate overperformance.

    The model is Dem-centric (positive CTOV = D overperforms model).
    For R candidates: negate so positive = R overperforms model.

    Parameters
    ----------
    ctov : ndarray of shape (J,)
        Raw CTOV from compute_ctov() (Dem-centric).
    party : str
        "D" or "R".
    """
    if party == "R":
        return -ctov
    return ctov.copy()


# ---------------------------------------------------------------------------
# Badge score computation
# ---------------------------------------------------------------------------


def _compute_badge_score(
    ctov_effective: np.ndarray,
    feature_per_type: np.ndarray,
    direction: int,
) -> float:
    """Compute the raw badge score for one candidate-badge combination.

    score = direction × dot(ctov_effective, feature_per_type_centered)

    We center the feature vector to remove the overall level — we care
    about which types are high or low *relative to the average*, not
    the absolute demographic level. This ensures that a candidate who
    uniformly overperforms everywhere doesn't spuriously score on all badges.

    Parameters
    ----------
    ctov_effective : ndarray of shape (J,)
        Party-adjusted CTOV.
    feature_per_type : ndarray of shape (J,)
        Demographic feature value for each type.
    direction : int
        +1 or -1 (see _BADGE_CATALOG).
    """
    centered_feature = feature_per_type - feature_per_type.mean()
    raw_score = float(np.dot(ctov_effective, centered_feature))
    return direction * raw_score


# ---------------------------------------------------------------------------
# Career CTOV: average across races for multi-race candidates
# ---------------------------------------------------------------------------


def _career_ctov(person_id: str, ctov_df: pd.DataFrame) -> np.ndarray:
    """Compute the career-average CTOV for a candidate.

    For multi-race candidates, simple average of CTOV vectors.
    For single-race candidates, just their one vector.

    This gives us a stable career signal rather than election-specific noise.
    """
    ctov_cols = [c for c in ctov_df.columns if c.startswith("ctov_type_")]
    person_rows = ctov_df[ctov_df["person_id"] == person_id]
    if len(person_rows) == 0:
        J = len(ctov_cols)
        return np.zeros(J)
    return person_rows[ctov_cols].values.mean(axis=0)


# ---------------------------------------------------------------------------
# Main badge derivation function
# ---------------------------------------------------------------------------


def derive_badges(ctov_df: pd.DataFrame, mvd_df: pd.DataFrame) -> dict:
    """Derive badges for all candidates based on their CTOV and MVD.

    Badge assignment is relative (1 std dev above mean) so the set of
    badges awarded automatically adjusts to the distribution of candidates
    in the registry.

    Parameters
    ----------
    ctov_df : pd.DataFrame
        Output of compute_ctov() — one row per candidate-race.
    mvd_df : pd.DataFrame
        Output of compute_mvd() — used for Turnout Monster badge.

    Returns
    -------
    dict
        Mapping: person_id → {
            "name": str,
            "party": str,
            "n_races": int,
            "badges": list[str],
            "badge_scores": dict[badge_name, float]
        }
    """
    type_profiles = _load_type_profiles()
    ctov_cols = [c for c in ctov_df.columns if c.startswith("ctov_type_")]
    J = len(ctov_cols)

    if J != len(type_profiles):
        log.warning(
            "CTOV dimension J=%d does not match type_profiles rows=%d; "
            "badge scores may be truncated",
            J,
            len(type_profiles),
        )
        # Use the smaller dimension to avoid index errors
        J = min(J, len(type_profiles))

    # --- Step 1: Compute career CTOV and raw badge scores for all candidates ---
    person_ids = ctov_df["person_id"].unique()
    raw_scores: dict[str, dict[str, float]] = {}

    # Also compute Turnout Monster raw scores (mean |MVD| per candidate)
    mvd_by_person: dict[str, float] = (
        mvd_df.groupby("person_id")["mvd"].mean().to_dict()
    )

    for person_id in person_ids:
        person_rows = ctov_df[ctov_df["person_id"] == person_id]
        party = person_rows["party"].iloc[0]
        career_vec = _career_ctov(person_id, ctov_df)[:J]
        effective = _effective_ctov(career_vec, party)

        scores: dict[str, float] = {}
        for badge_def in _BADGE_CATALOG:
            col = badge_def["column"]
            if col not in type_profiles.columns:
                log.warning("Badge column '%s' not in type_profiles, skipping badge", col)
                continue
            feature = type_profiles[col].values[:J].astype(float)
            scores[badge_def["badge"]] = _compute_badge_score(effective, feature, badge_def["direction"])

        # Turnout Monster: mean absolute MVD across all races
        mean_mvd = mvd_by_person.get(person_id, 0.0)
        scores[_TURNOUT_MONSTER_BADGE["badge"]] = abs(mean_mvd)

        raw_scores[person_id] = scores

    # --- Step 2: Compute cross-candidate mean and std for threshold ---
    # One mean/std per badge, computed across all candidates who have that badge.
    all_badge_names = list(next(iter(raw_scores.values())).keys()) if raw_scores else []
    badge_mean: dict[str, float] = {}
    badge_std: dict[str, float] = {}

    for badge_name in all_badge_names:
        vals = np.array([raw_scores[pid][badge_name] for pid in raw_scores if badge_name in raw_scores[pid]])
        badge_mean[badge_name] = float(np.mean(vals))
        badge_std[badge_name] = float(np.std(vals))

    # --- Step 3: Award badges when |score| > mean + threshold × std ---
    result: dict[str, dict] = {}
    for person_id, scores in raw_scores.items():
        person_rows = ctov_df[ctov_df["person_id"] == person_id]
        awarded: list[str] = []

        for badge_name, raw_score in scores.items():
            mean = badge_mean.get(badge_name, 0.0)
            std = badge_std.get(badge_name, 1.0)
            if std < 1e-9:
                # All candidates have the same score — badge is uninformative
                continue
            # Positive badge: score > mean + threshold*std
            # Negative badge: score < mean - threshold*std (exceptionally low)
            # We award the positive version (candidate genuinely skilled in that dimension)
            # and a negative version (candidate notably weak)
            if raw_score > mean + _BADGE_THRESHOLD_STD * std:
                awarded.append(badge_name)
            elif raw_score < mean - _BADGE_THRESHOLD_STD * std:
                awarded.append(f"Low {badge_name}")

        result[person_id] = {
            "name": person_rows["name"].iloc[0],
            "party": person_rows["party"].iloc[0],
            "n_races": len(person_rows),
            "badges": awarded,
            "badge_scores": scores,
        }

    n_with_badges = sum(1 for v in result.values() if v["badges"])
    log.info(
        "derive_badges: %d candidates, %d with at least one badge",
        len(result),
        n_with_badges,
    )
    return result
