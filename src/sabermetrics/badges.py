"""Badge derivation for political candidates.

Badges translate CTOV vectors into human-readable archetypes. Each badge
reflects a demographic dimension: does this candidate systematically
over/underperform in communities dominated by a particular demographic trait?

Badge math:
    residual = career_ctov - party_mean_ctov
    effective = party_adjust(residual)  # negate for R
    score = dot(effective, demographic_feature_per_type)

Party-mean subtraction ensures badges capture what makes a candidate UNIQUE
within their party, not what makes them a generic D or R.  Without it, all
D candidates score similarly on education/income badges (because those are
party coalition patterns, not individual fingerprints).

The effective CTOV is then party-adjusted (flipped for R candidates so that
positive scores always mean overperformance in that demographic context).

A badge is awarded when |score| > 1 standard deviation above the
cross-candidate mean, ensuring badges are relative (not absolute) measures.

For multi-race candidates, badges use the average CTOV across their career,
not any single race — this is the career signal, not a one-year snapshot.

NOTE: Only badges use residual (party-relative) CTOV.  Phase 4 CTOV
prediction integration uses absolute CTOV — we want the full candidate
signal including party component for prediction priors.
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

# Badges are awarded at 1 std dev threshold above the within-party mean.
# This is intentionally permissive: ~16% of candidates get each badge,
# which is enough to be meaningful but not so rare it's useless.
_BADGE_THRESHOLD_STD = 1.0

# Minimum party pool size for within-party thresholding.
# Below this, fall back to global thresholding to avoid std estimates that
# are too noisy to be meaningful (< 20 candidates → std error > ~22% of true std).
_MIN_PARTY_POOL_SIZE = 20

# Threshold for auto-discovered signature badges (within-party z-score).
_SIGNATURE_Z_THRESHOLD = 2.0

# Maximum signature badges per candidate (greedy cosine dedup reduces further).
_MAX_SIGNATURES = 3

# Cosine similarity threshold for signature badge deduplication.
# Types with column-cosine > this are considered redundant and the lower-ranked
# one is skipped so all 3 signatures represent orthogonal skill axes.
_COSINE_DEDUP_THRESHOLD = 0.6


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
            "CTOV dimension J=%d does not match type_profiles rows=%d; badge scores may be truncated",
            J,
            len(type_profiles),
        )
        # Use the smaller dimension to avoid index errors
        J = min(J, len(type_profiles))

    # --- Step 0: Compute party-mean CTOV for residual badge scoring ---
    # Badges capture what makes a candidate UNIQUE within their party, not
    # what makes them a generic D or R.  Subtracting the party mean removes
    # the shared coalition signal so badge scores reflect individual deviation.
    # NOTE: This is ONLY for badges — Phase 4 CTOV prediction uses absolute CTOV.
    party_mean_ctov: dict[str, np.ndarray] = {}
    for party_code in ["D", "R", "I"]:
        party_rows = ctov_df[ctov_df["party"] == party_code]
        if len(party_rows) > 0:
            ctov_matrix = party_rows[ctov_cols].values
            party_mean_ctov[party_code] = ctov_matrix.mean(axis=0)[:J]

    # --- Step 1: Compute career CTOV and raw badge scores for all candidates ---
    person_ids = ctov_df["person_id"].unique()
    raw_scores: dict[str, dict[str, float]] = {}
    # Track party per person for within-party thresholding in Step 2
    party_by_person: dict[str, str] = {}

    # Also compute Turnout Monster raw scores (mean |MVD| per candidate)
    mvd_by_person: dict[str, float] = mvd_df.groupby("person_id")["mvd"].mean().to_dict()

    for person_id in person_ids:
        person_rows = ctov_df[ctov_df["person_id"] == person_id]
        party = person_rows["party"].iloc[0]
        party_by_person[person_id] = party
        career_vec = _career_ctov(person_id, ctov_df)[:J]
        # Subtract party mean to get the RESIDUAL — what makes this candidate
        # different from an average member of their party
        party_mean = party_mean_ctov.get(party, np.zeros(J))
        residual_ctov = career_vec - party_mean
        effective = _effective_ctov(residual_ctov, party)

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

    # --- Step 2: Compute thresholds: within-party where pool >= _MIN_PARTY_POOL_SIZE,
    #             global fallback otherwise. Turnout Monster stays global always. ---
    all_badge_names = list(next(iter(raw_scores.values())).keys()) if raw_scores else []

    # Global fallback thresholds (used for small parties and Turnout Monster)
    global_mean: dict[str, float] = {}
    global_std: dict[str, float] = {}
    for badge_name in all_badge_names:
        vals = np.array([raw_scores[pid][badge_name] for pid in raw_scores if badge_name in raw_scores[pid]])
        global_mean[badge_name] = float(np.mean(vals))
        global_std[badge_name] = float(np.std(vals))

    # Within-party thresholds for parties meeting the pool size floor
    party_mean: dict[tuple[str, str], float] = {}
    party_std: dict[tuple[str, str], float] = {}
    all_parties = ctov_df["party"].unique()
    small_parties: set[str] = set()
    for p in all_parties:
        pids = [pid for pid in raw_scores if party_by_person.get(pid) == p]
        if len(pids) < _MIN_PARTY_POOL_SIZE:
            small_parties.add(p)
            continue
        for badge_name in all_badge_names:
            vals = np.array([raw_scores[pid][badge_name] for pid in pids if badge_name in raw_scores[pid]])
            party_mean[(p, badge_name)] = float(np.mean(vals))
            party_std[(p, badge_name)] = float(np.std(vals))

    if small_parties:
        log.info(
            "derive_badges: parties %s have pool < %d — using global threshold fallback",
            sorted(small_parties),
            _MIN_PARTY_POOL_SIZE,
        )

    # --- Step 3: Award badges using within-party threshold (or global fallback) ---
    result: dict[str, dict] = {}
    for person_id, scores in raw_scores.items():
        person_rows = ctov_df[ctov_df["person_id"] == person_id]
        n_races = len(person_rows)
        p = party_by_person.get(person_id, "")
        is_small_pool = p in small_parties
        provisional = n_races < 2
        awarded: list[str] = []
        details: list[dict] = []

        for badge_name, raw_score in scores.items():
            # Turnout Monster is intentionally global — absolute magnitude, not party-relative.
            # Coalition badges use within-party thresholds so D and R outliers are measured
            # against their own party peers, not each other's coalition structure.
            is_turnout_monster = badge_name == _TURNOUT_MONSTER_BADGE["badge"]
            use_global = is_small_pool or is_turnout_monster
            fallback_reason: str | None = "small_pool" if (is_small_pool and not is_turnout_monster) else None

            if use_global or (p, badge_name) not in party_mean:
                mean = global_mean.get(badge_name, 0.0)
                std = global_std.get(badge_name, 1.0)
            else:
                mean = party_mean[(p, badge_name)]
                std = party_std[(p, badge_name)]

            if std < 1e-9:
                continue

            if raw_score > mean + _BADGE_THRESHOLD_STD * std:
                awarded.append(badge_name)
                details.append(
                    {
                        "name": badge_name,
                        "score": raw_score,
                        "provisional": provisional,
                        "kind": "catalog",
                        "fallback_reason": fallback_reason,
                    }
                )
            elif raw_score < mean - _BADGE_THRESHOLD_STD * std:
                low_name = f"Low {badge_name}"
                awarded.append(low_name)
                details.append(
                    {
                        "name": low_name,
                        "score": raw_score,
                        "provisional": provisional,
                        "kind": "catalog",
                        "fallback_reason": fallback_reason,
                    }
                )

        result[person_id] = {
            "name": person_rows["name"].iloc[0],
            "party": p,
            "n_races": n_races,
            "badges": awarded,
            "badge_scores": scores,
            "badge_details": details,
        }

    n_with_badges = sum(1 for v in result.values() if v["badges"])
    log.info(
        "derive_badges: %d candidates, %d with at least one badge",
        len(result),
        n_with_badges,
    )
    return result


# ---------------------------------------------------------------------------
# Signature badge derivation (auto-discovered)
# ---------------------------------------------------------------------------


def derive_signature_badges(
    ctov_df: pd.DataFrame,
    super_type_names: dict[int, str] | None = None,
) -> dict:
    """Derive auto-discovered signature badges for all candidates.

    Signature badges capture a candidate's most distinctive type-level fingerprint
    — where their residual CTOV is far from party peers.  Unlike catalog badges
    (which project onto preset demographic axes), signature badges are labeled
    directly by community type.

    Algorithm:
    1. Compute party-mean CTOV and per-candidate residual (same as derive_badges).
    2. For each type j, compute within-party std of residuals.
    3. For each candidate, compute z-score = residual[j] / party_std[j].
    4. Candidate types where |z| > _SIGNATURE_Z_THRESHOLD qualify.
    5. Greedy cosine dedup: skip type if cosine(col_j, already_selected) > _COSINE_DEDUP_THRESHOLD.
    6. Cap at _MAX_SIGNATURES per candidate.

    Parameters
    ----------
    ctov_df : pd.DataFrame
        Output of compute_ctov() — one row per candidate-race.
    super_type_names : dict[int, str] | None
        Mapping from type_id → display name.  If None, loads from DuckDB.

    Returns
    -------
    dict
        Mapping: person_id → {
            "signature_badges": list[{
                "name": str,
                "score": float (z-score),
                "type_id": int,
                "kind": "signature",
                "provisional": bool,
            }]
        }
    """
    ctov_cols = [c for c in ctov_df.columns if c.startswith("ctov_type_")]
    J = len(ctov_cols)

    if super_type_names is None:
        super_type_names = _load_super_type_names()

    # Compute party-mean CTOV (same approach as derive_badges Step 0)
    party_mean_ctov: dict[str, np.ndarray] = {}
    for party_code in ctov_df["party"].unique():
        party_rows = ctov_df[ctov_df["party"] == party_code]
        party_mean_ctov[party_code] = party_rows[ctov_cols].values.mean(axis=0)

    # Compute career CTOV and residual per candidate
    person_ids = ctov_df["person_id"].unique()
    residual_by_person: dict[str, np.ndarray] = {}
    party_by_person: dict[str, str] = {}
    n_races_by_person: dict[str, int] = {}
    for pid in person_ids:
        rows = ctov_df[ctov_df["person_id"] == pid]
        p = rows["party"].iloc[0]
        career_vec = rows[ctov_cols].values.mean(axis=0)
        pm = party_mean_ctov.get(p, np.zeros(J))
        residual_by_person[pid] = career_vec - pm
        party_by_person[pid] = p
        n_races_by_person[pid] = len(rows)

    # Compute within-party std per type (used for z-scores)
    # Fall back to global std for small parties.
    party_type_std: dict[str, np.ndarray] = {}
    global_type_std = np.std(np.array(list(residual_by_person.values())), axis=0).clip(min=1e-9)

    for p in ctov_df["party"].unique():
        pids = [pid for pid in person_ids if party_by_person[pid] == p]
        if len(pids) < _MIN_PARTY_POOL_SIZE:
            party_type_std[p] = global_type_std
        else:
            mat = np.array([residual_by_person[pid] for pid in pids])
            party_type_std[p] = np.std(mat, axis=0).clip(min=1e-9)

    # Residual matrix for cosine dedup: shape (n_candidates, J)
    # We use this to measure how correlated two type columns are across the candidate pool.
    residual_matrix = np.array(list(residual_by_person.values()))  # (N, J)
    col_norms = np.linalg.norm(residual_matrix, axis=0).clip(min=1e-9)

    result: dict[str, dict] = {}
    for pid in person_ids:
        residual = residual_by_person[pid]
        p = party_by_person[pid]
        type_std = party_type_std[p]
        provisional = n_races_by_person[pid] < 2

        z_scores = residual / type_std  # shape (J,)

        # Candidate types exceeding the z-score threshold
        qualifying = [(j, float(z_scores[j])) for j in range(J) if abs(z_scores[j]) >= _SIGNATURE_Z_THRESHOLD]
        # Sort by |z| descending
        qualifying.sort(key=lambda x: -abs(x[1]))

        # Greedy cosine dedup: keep type if it's not too similar to any already selected
        selected_type_ids: list[int] = []
        selected_col_vecs: list[np.ndarray] = []

        for type_id, z in qualifying:
            if len(selected_type_ids) >= _MAX_SIGNATURES:
                break
            col_vec = residual_matrix[:, type_id] / col_norms[type_id]
            # Check cosine similarity against already-selected types
            too_similar = any(
                float(np.dot(col_vec, prev_vec)) > _COSINE_DEDUP_THRESHOLD for prev_vec in selected_col_vecs
            )
            if not too_similar:
                selected_type_ids.append(type_id)
                selected_col_vecs.append(col_vec)

        # Build badge list for this candidate
        badges = []
        for type_id in selected_type_ids:
            z = float(z_scores[type_id])
            type_name = super_type_names.get(type_id, f"Type {type_id}")
            prefix = "Signature" if z >= 0 else "Low Signature"
            badges.append(
                {
                    "name": f"{prefix}: {type_name}",
                    "score": z,
                    "type_id": type_id,
                    "kind": "signature",
                    "provisional": provisional,
                }
            )

        result[pid] = {"signature_badges": badges}

    n_with_signatures = sum(1 for v in result.values() if v["signature_badges"])
    log.info(
        "derive_signature_badges: %d candidates, %d with at least one signature",
        len(result),
        n_with_signatures,
    )
    return result


def _load_super_type_names() -> dict[int, str]:
    """Load type display names from DuckDB (authoritative source).

    Falls back to empty dict on failure so signature badges degrade to
    "Type N" labels rather than crashing the pipeline.
    """
    try:
        import duckdb

        db_path = PROJECT_ROOT / "data" / "wethervane.duckdb"
        if not db_path.exists():
            log.warning("DuckDB not found at %s; signature badge labels will be 'Type N'", db_path)
            return {}
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute("SELECT type_id, display_name FROM types").fetchall()
            return {int(r[0]): str(r[1]) for r in rows}
        finally:
            con.close()
    except Exception as exc:
        log.warning("Could not load type display names: %s; using 'Type N' fallback", exc)
        return {}
