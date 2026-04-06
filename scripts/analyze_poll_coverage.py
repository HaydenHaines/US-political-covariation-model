"""Poll coverage diagnostic: identify under/oversampled community types per race.

Issue #94.2 — Undersampled group identification.

For each poll with xt_ demographic composition data (currently 24 Emerson polls),
compare the poll's sampled demographic mix against the true population demographic
mix for that state. Identify which groups are systematically over- or undersampled,
and which community types are most affected.

Output: data/diagnostics/poll_coverage_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path so the src package is importable when this script
# is run directly (e.g. ``python scripts/analyze_poll_coverage.py``).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Data directory: the data/ folder is gitignored and lives only in the canonical
# project location. In worktrees (used for parallel agent isolation) the data is not
# copied, so we fall back to the canonical path when the local one is absent.
_LOCAL_DATA = PROJECT_ROOT / "data"
_CANONICAL_DATA = Path("/home/hayden/projects/wethervane/data")
# Use the canonical data path when the local worktree doesn't have the large data files
# (communities/ and assembled/ are gitignored and not present in worktrees).
DATA_ROOT = (
    _LOCAL_DATA
    if (_LOCAL_DATA / "communities" / "type_profiles.parquet").exists()
    else _CANONICAL_DATA
)

POLLS_PATH = DATA_ROOT / "polls" / "polls_2026.csv"
TYPE_PROFILES_PATH = DATA_ROOT / "communities" / "type_profiles.parquet"
COUNTY_ASSIGNMENTS_PATH = DATA_ROOT / "communities" / "county_type_assignments_full.parquet"
COUNTY_VOTES_PATH = DATA_ROOT / "assembled" / "medsl_county_presidential_2020.parquet"
# Output goes into the project root's data/ regardless, creating the folder if needed.
OUTPUT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "poll_coverage_report.json"

# Thresholds for flagging coverage gaps.
# ratio = poll_share / population_share; 1.0 means perfectly representative.
OVERSAMPLE_THRESHOLD = 1.2   # poll overweights this group by ≥20 %
UNDERSAMPLE_THRESHOLD = 0.8  # poll underweights this group by ≥20 %

# Mapping: xt_ column name → corresponding column in type_profiles.
# Imported from the shared module so the forecast pipeline can use the same mapping.
from src.prediction.population_vectors import (  # noqa: E402
    XT_TO_TYPE_PROFILE_COL,
    build_state_population_vectors,
)

# Human-readable labels for report output.
XT_LABELS: dict[str, str] = {
    "xt_race_white": "White (non-Hispanic)",
    "xt_race_black": "Black",
    "xt_race_hispanic": "Hispanic",
    "xt_race_asian": "Asian",
    "xt_education_college": "College-educated",
    "xt_religion_evangelical": "Evangelical",
}

# Number of top-affected types to report per group gap.
TOP_TYPES_PER_GAP = 5

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets. Returns (polls, type_profiles, county_assignments, county_votes)."""
    polls = pd.read_csv(POLLS_PATH)
    type_profiles = pd.read_parquet(TYPE_PROFILES_PATH).reset_index(drop=True)
    county_assignments = pd.read_parquet(COUNTY_ASSIGNMENTS_PATH)
    county_votes = pd.read_parquet(COUNTY_VOTES_PATH)

    log.info(
        "Loaded %d polls, %d types, %d county assignments, %d county vote records",
        len(polls),
        len(type_profiles),
        len(county_assignments),
        len(county_votes),
    )
    return polls, type_profiles, county_assignments, county_votes


# ---------------------------------------------------------------------------
# Coverage gap computation
# ---------------------------------------------------------------------------


def compute_coverage_ratios(
    poll_vec: dict[str, float],
    population_vec: dict[str, float],
) -> dict[str, float | None]:
    """Compute poll/population ratio for each demographic group.

    A ratio > 1.0 means the poll oversamples this group relative to the population.
    A ratio < 1.0 means the poll undersamples this group.

    Returns a dict of {xt_col: ratio} for columns present in both vectors.
    None is returned when the population share is zero (division undefined).
    """
    ratios: dict[str, float | None] = {}
    for col, poll_share in poll_vec.items():
        pop_share = population_vec.get(col)
        if pop_share is None:
            continue
        if pop_share == 0:
            ratios[col] = None  # Can't compute ratio; population has none of this group
        else:
            ratios[col] = float(poll_share / pop_share)
    return ratios


def find_affected_types(
    xt_col: str,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
    n_top: int = TOP_TYPES_PER_GAP,
) -> list[dict]:
    """Find which types are most 'exposed' to a demographic gap.

    A type is highly affected by an undersampling gap in group G if:
      - It has a high concentration of group G (relative to other types), AND
      - It has meaningful weight in the state's type distribution.

    We rank by (type_demographic_share × state_type_weight) — i.e., the joint
    probability that a randomly chosen voter from this state belongs to this type
    AND belongs to demographic group G.

    Args:
        xt_col: The xt_ column identifying the undersampled group.
        type_profiles: DataFrame with demographic profiles for each type.
        state_type_weights: Vote-weighted type distribution for the relevant state.
        n_top: Number of types to return.

    Returns:
        List of dicts with type_id, display_name, group_share, state_weight, exposure.
    """
    profile_col = XT_TO_TYPE_PROFILE_COL.get(xt_col)
    if profile_col is None or profile_col not in type_profiles.columns:
        return []

    type_group_shares = type_profiles[profile_col].values.astype(float)
    # Exposure = how much of the state's population of this group lives in this type.
    exposure = type_group_shares * state_type_weights

    # Normalize so exposures sum to 1 (proportion of the gap attributable to each type).
    total_exposure = exposure.sum()
    if total_exposure > 0:
        exposure_norm = exposure / total_exposure
    else:
        exposure_norm = exposure.copy()

    top_indices = np.argsort(exposure_norm)[::-1][:n_top]
    results = []
    for idx in top_indices:
        if exposure_norm[idx] <= 0:
            break
        row = type_profiles.iloc[idx]
        results.append(
            {
                "type_id": int(row["type_id"]),
                "display_name": str(row.get("display_name", f"Type {idx}")),
                "group_share": round(float(type_group_shares[idx]), 4),
                "state_weight": round(float(state_type_weights[idx]), 4),
                "exposure": round(float(exposure_norm[idx]), 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Per-race analysis
# ---------------------------------------------------------------------------


def analyze_poll(
    poll_row: pd.Series,
    xt_cols: list[str],
    state_population_vectors: dict[str, dict[str, float]],
    type_profiles: pd.DataFrame,
    county_assignments: pd.DataFrame,
    county_votes: pd.DataFrame,
) -> dict | None:
    """Analyze coverage gaps for a single poll.

    Returns a structured result dict, or None if the poll can't be analyzed
    (e.g., no xt_ data, state not found in population vectors).
    """
    state = str(poll_row.get("geography", ""))
    if not state or state not in state_population_vectors:
        return None

    # Extract xt_ values that are actually present for this poll.
    poll_vec: dict[str, float] = {}
    for col in xt_cols:
        val = poll_row.get(col)
        if pd.notna(val):
            poll_vec[col] = float(val)

    if not poll_vec:
        return None

    pop_vec = state_population_vectors[state]
    ratios = compute_coverage_ratios(poll_vec, pop_vec)

    # Classify each group as over/under/representative.
    gaps: list[dict] = []
    for col, ratio in ratios.items():
        if ratio is None:
            continue
        if ratio > OVERSAMPLE_THRESHOLD:
            status = "oversampled"
        elif ratio < UNDERSAMPLE_THRESHOLD:
            status = "undersampled"
        else:
            status = "representative"

        gap_entry: dict = {
            "demographic_group": col,
            "label": XT_LABELS.get(col, col),
            "poll_share": round(poll_vec[col], 4),
            "population_share": round(pop_vec.get(col, float("nan")), 4),
            "ratio": round(ratio, 4),
            "status": status,
        }

        if status == "undersampled":
            # Compute state type distribution for identifying affected types.
            state_type_weights = _get_state_type_weights(
                state, county_assignments, county_votes
            )
            gap_entry["affected_types"] = find_affected_types(
                col, type_profiles, state_type_weights
            )

        gaps.append(gap_entry)

    n_undersampled = sum(1 for g in gaps if g["status"] == "undersampled")
    n_oversampled = sum(1 for g in gaps if g["status"] == "oversampled")

    return {
        "race": str(poll_row.get("race", "")),
        "state": state,
        "pollster": str(poll_row.get("pollster", "")),
        "date": str(poll_row.get("date", "")),
        "n_sample": int(poll_row.get("n_sample", 0)) if pd.notna(poll_row.get("n_sample")) else None,
        "n_groups_analyzed": len(gaps),
        "n_undersampled": n_undersampled,
        "n_oversampled": n_oversampled,
        "gaps": gaps,
    }


def _get_state_type_weights(
    state_abbr: str,
    county_assignments: pd.DataFrame,
    county_votes: pd.DataFrame,
) -> np.ndarray:
    """Compute vote-weighted type distribution for a single state.

    Returns a (J,) array that sums to 1.0. This is the same calculation used in
    build_state_population_vectors but for a single state, used when identifying
    which types are most exposed to a particular demographic gap.
    """
    score_cols = sorted(
        [c for c in county_assignments.columns if c.startswith("type_") and c.endswith("_score")],
        key=lambda c: int(c.split("_")[1]),
    )

    county_df = county_assignments[["county_fips"] + score_cols].merge(
        county_votes[["county_fips", "state_abbr", "pres_total_2020"]],
        on="county_fips",
        how="inner",
    )
    state_df = county_df[
        (county_df["state_abbr"] == state_abbr)
        & county_df["pres_total_2020"].notna()
    ]

    scores = state_df[score_cols].values.astype(float)
    votes = state_df["pres_total_2020"].values
    total = votes.sum()
    if total == 0:
        return np.ones(len(score_cols)) / len(score_cols)

    weights = (scores * votes[:, None]).sum(axis=0) / total
    return weights


# ---------------------------------------------------------------------------
# Aggregate summary across races
# ---------------------------------------------------------------------------


def build_summary(results: list[dict]) -> dict:
    """Summarize coverage patterns across all analyzed polls.

    Returns per-group counts of undersampled/oversampled/representative polls,
    and the most commonly affected types for each undersampled group.
    """
    from collections import Counter, defaultdict

    group_status_counts: dict[str, Counter] = defaultdict(Counter)
    group_affected_types: dict[str, Counter] = defaultdict(Counter)

    for result in results:
        for gap in result.get("gaps", []):
            col = gap["demographic_group"]
            status = gap["status"]
            group_status_counts[col][status] += 1

            if status == "undersampled":
                for t in gap.get("affected_types", []):
                    key = f"{t['type_id']}: {t['display_name']}"
                    group_affected_types[col][key] += 1

    summary: dict = {"by_group": {}}
    for col in sorted(group_status_counts.keys()):
        counts = group_status_counts[col]
        top_types = [
            {"type_label": label, "n_races_affected": n}
            for label, n in group_affected_types[col].most_common(5)
        ]
        summary["by_group"][col] = {
            "label": XT_LABELS.get(col, col),
            "n_undersampled": counts.get("undersampled", 0),
            "n_oversampled": counts.get("oversampled", 0),
            "n_representative": counts.get("representative", 0),
            "n_total_polls": sum(counts.values()),
            "top_affected_types": top_types,
        }

    # Which groups are most chronically undersampled?
    undersampled_ranking = sorted(
        [(col, v["n_undersampled"]) for col, v in summary["by_group"].items()],
        key=lambda x: x[1],
        reverse=True,
    )
    summary["undersampled_ranking"] = [
        {"group": col, "label": XT_LABELS.get(col, col), "n_polls_undersampled": n}
        for col, n in undersampled_ranking
        if n > 0
    ]

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the poll coverage diagnostic and save results to JSON."""
    log.info("Loading data …")
    polls, type_profiles, county_assignments, county_votes = load_data()

    # Filter to polls that have at least one xt_ value.
    xt_cols = [c for c in polls.columns if c.startswith("xt_")]
    has_xt = polls[xt_cols].notna().any(axis=1)
    xt_polls = polls[has_xt].copy()
    log.info("%d polls have xt_ data (out of %d total)", len(xt_polls), len(polls))

    # Only process xt_ columns that have at least one non-null value across all polls.
    active_xt_cols = [c for c in xt_cols if xt_polls[c].notna().any()]
    # Further restrict to those with a type_profiles mapping (needed for affected-type analysis).
    mappable_xt_cols = [c for c in active_xt_cols if c in XT_TO_TYPE_PROFILE_COL]
    log.info(
        "Active xt_ columns: %s (mappable to type_profiles: %s)",
        active_xt_cols,
        mappable_xt_cols,
    )

    log.info("Building state population demographic vectors …")
    state_population_vectors = build_state_population_vectors(
        type_profiles, county_assignments, county_votes, mappable_xt_cols
    )

    log.info("Analyzing %d polls …", len(xt_polls))
    results = []
    for _, row in xt_polls.iterrows():
        result = analyze_poll(
            row,
            active_xt_cols,
            state_population_vectors,
            type_profiles,
            county_assignments,
            county_votes,
        )
        if result is not None:
            results.append(result)

    log.info("Analyzed %d polls successfully", len(results))

    summary = build_summary(results)

    report = {
        "metadata": {
            "total_polls": int(len(polls)),
            "polls_with_xt_data": int(len(xt_polls)),
            "polls_analyzed": int(len(results)),
            "active_xt_columns": active_xt_cols,
            "mappable_xt_columns": mappable_xt_cols,
            "oversample_threshold": OVERSAMPLE_THRESHOLD,
            "undersample_threshold": UNDERSAMPLE_THRESHOLD,
        },
        "summary": summary,
        "per_poll_results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved to %s", OUTPUT_PATH)

    # Print a human-readable summary to stdout.
    print("\n=== POLL COVERAGE DIAGNOSTIC REPORT ===\n")
    print(f"Polls analyzed: {len(results)} (of {len(xt_polls)} with xt_ data)")
    print(f"Active xt_ columns: {active_xt_cols}")
    print()

    print("--- Coverage gaps by demographic group ---")
    for col, stats in summary["by_group"].items():
        print(
            f"  {stats['label']:30s}  "
            f"under={stats['n_undersampled']:2d}  "
            f"over={stats['n_oversampled']:2d}  "
            f"ok={stats['n_representative']:2d}  "
            f"(of {stats['n_total_polls']} polls)"
        )
        if stats["top_affected_types"]:
            for t in stats["top_affected_types"][:3]:
                print(f"      → {t['type_label']} (in {t['n_races_affected']} races)")

    print()
    if summary["undersampled_ranking"]:
        print("--- Most chronically undersampled groups ---")
        for entry in summary["undersampled_ranking"]:
            print(
                f"  {entry['label']:30s}  undersampled in {entry['n_polls_undersampled']} polls"
            )
    else:
        print("No systematically undersampled groups found at the current thresholds.")

    print(f"\nFull report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
