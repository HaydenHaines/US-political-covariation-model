"""Vote-weighted population demographic vectors per state.

This module extracts the core population vector logic from the poll coverage
diagnostic so it can be shared with the forecast pipeline. The key operation
is: given the model's type profiles and county soft-membership scores, compute
the expected demographic composition of each state's electorate.

This is used by:
  1. ``scripts/analyze_poll_coverage.py`` — diagnostic for identifying coverage gaps.
  2. ``src/prediction/poll_enrichment.py`` — post-stratification correction that
     adjusts effective sample size when a poll oversamples a demographic group.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

# Mapping from xt_ column name (as it appears in poll data) to the corresponding
# column in type_profiles. Only columns with a clear 1:1 equivalent are mapped.
# Inverted or derived columns (noncollege, rural) have no standalone type profile
# column, so they are intentionally absent here.
XT_TO_TYPE_PROFILE_COL: dict[str, str] = {
    "xt_race_white": "pct_white_nh",
    "xt_race_black": "pct_black",
    "xt_race_hispanic": "pct_hispanic",
    "xt_race_asian": "pct_asian",
    "xt_education_college": "pct_bachelors_plus",
    "xt_religion_evangelical": "evangelical_share",
    # xt_education_noncollege is 1 - xt_education_college, not separately profiled.
    # xt_urbanicity_* and xt_age_senior have no direct type_profiles equivalent.
}


def build_state_population_vectors(
    type_profiles: pd.DataFrame,
    county_assignments: pd.DataFrame,
    county_votes: pd.DataFrame,
    xt_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Compute vote-weighted population demographic vectors per state.

    For each state we compute: for each demographic group, the population share
    implied by the community type distribution. The steps are:

    1. For each county, the soft membership scores (type_0_score … type_99_score)
       describe how much of that county belongs to each of the J=100 types.
    2. The expected demographic share for county c is:
         demo(c) = sum_j [ scores(c,j) * type_demo(j) ]
    3. The state-level population vector is the vote-count-weighted average across
       counties in the state.

    This gives us the "ground truth" demographic composition of each state's
    electorate (from the model's perspective).

    Args:
        type_profiles: DataFrame with one row per type, demographic columns.
        county_assignments: DataFrame with county_fips and type_0_score … type_99_score.
        county_votes: DataFrame with county_fips, state_abbr, pres_total_2020.
        xt_cols: The xt_ column names for which to compute population vectors.
            Only columns present in XT_TO_TYPE_PROFILE_COL (and in type_profiles)
            will be included in the output.

    Returns:
        Dict mapping state_abbr → {xt_col: population_share}.
    """
    # Score columns in county_assignments (type_0_score … type_99_score).
    # Sorting by integer index is critical — we need the score vector to align
    # row-wise with type_profiles.
    score_cols = sorted(
        [c for c in county_assignments.columns if c.startswith("type_") and c.endswith("_score")],
        key=lambda c: int(c.split("_")[1]),
    )
    if len(score_cols) != len(type_profiles):
        raise ValueError(
            f"Score columns ({len(score_cols)}) don't match type profiles ({len(type_profiles)})"
        )

    # Merge county soft-membership with vote totals and state labels.
    county_df = county_assignments[["county_fips"] + score_cols].merge(
        county_votes[["county_fips", "state_abbr", "pres_total_2020"]],
        on="county_fips",
        how="inner",
    )
    county_df = county_df.dropna(subset=["state_abbr", "pres_total_2020"])

    # Drop the aggregate row (fips 00000) that appears in MEDSL data — it represents
    # the whole-state sum and would double-count all votes if included.
    county_df = county_df[county_df["county_fips"] != "00000"]
    county_df = county_df[county_df["county_fips"] != 0]

    # Build a (J,) vector of each demographic attribute from type_profiles.
    # XT_TO_TYPE_PROFILE_COL maps each xt_ column to the type_profile column that
    # holds the corresponding demographic share.
    type_demo: dict[str, np.ndarray] = {}
    for xt_col in xt_cols:
        profile_col = XT_TO_TYPE_PROFILE_COL.get(xt_col)
        if profile_col is not None and profile_col in type_profiles.columns:
            type_demo[xt_col] = type_profiles[profile_col].values.astype(float)

    if not type_demo:
        raise ValueError("No mappable xt_ columns found in type_profiles")

    # Compute county-level demographic estimates via matrix multiply:
    # county_demo[i, xt_col] = scores[i, :] @ type_vals
    # This is a dot product of the county's soft-membership over J types with the
    # per-type demographic value — giving the expected demographic share in the county.
    scores = county_df[score_cols].values.astype(float)  # (N_counties, J)
    for xt_col, type_vals in type_demo.items():
        county_df[f"_est_{xt_col}"] = scores @ type_vals

    # Vote-weighted average across counties per state.
    state_vectors: dict[str, dict[str, float]] = {}
    for state, group in county_df.groupby("state_abbr"):
        votes = group["pres_total_2020"].values
        total_votes = votes.sum()
        if total_votes == 0:
            continue
        state_vec: dict[str, float] = {}
        for xt_col in type_demo:
            county_estimates = group[f"_est_{xt_col}"].values
            state_vec[xt_col] = float((county_estimates * votes).sum() / total_votes)
        state_vectors[str(state)] = state_vec

    log.info("Built population vectors for %d states", len(state_vectors))
    return state_vectors
