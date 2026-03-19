"""Build community weight matrices from hard HAC assignments.

Unlike the NMF pipeline (which uses soft c1..c7 membership weights),
the HAC pipeline uses hard assignments: each county belongs 100% to its
community. The state-level W matrix weights communities by their share
of a state's recent vote totals.

Inputs:
  data/communities/county_community_assignments.parquet
  data/assembled/medsl_county_2024_president.parquet  (vote totals)

Outputs:
  data/propagation/community_weights_county_hac.parquet
      county_fips, community_id, state_fips, recent_total
  data/propagation/community_weights_state_hac.parquet
      state_fips, state_abbr, community_0, community_1, ...
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"
OUTPUT_DIR = PROJECT_ROOT / "data" / "propagation"

STATE_FIPS_TO_ABBR = {"12": "FL", "13": "GA", "01": "AL"}


def build_county_weights(
    assignments: pd.DataFrame,
    vote_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Build county-level weight table (hard assignment).

    Returns DataFrame: county_fips, community_id, state_fips, recent_total
    """
    merged = assignments.merge(vote_totals, on="county_fips", how="left")
    return merged[["county_fips", "community_id", "state_fips", "recent_total"]]


def build_state_weights(
    assignments: pd.DataFrame,
    vote_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Build state-level W matrix for poll propagation.

    W[state, k] = vote share of community k within state.
    Columns: state_fips, state_abbr, community_0, ..., community_{K-1}
    """
    merged = assignments.merge(vote_totals, on="county_fips", how="left")
    merged["recent_total"] = merged["recent_total"].fillna(0)
    k_ids = sorted(merged["community_id"].unique())

    rows = []
    for state_fips, state_df in merged.groupby("state_fips"):
        total_votes = state_df["recent_total"].sum()
        row = {"state_fips": state_fips, "state_abbr": STATE_FIPS_TO_ABBR.get(state_fips, "???")}
        for k in k_ids:
            k_votes = state_df[state_df["community_id"] == k]["recent_total"].sum()
            row[f"community_{k}"] = float(k_votes / total_votes) if total_votes > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    # Normalize rows to exactly sum to 1 (handle float rounding)
    weight_cols = [c for c in df.columns if c.startswith("community_")]
    row_sums = df[weight_cols].sum(axis=1)
    df[weight_cols] = df[weight_cols].div(row_sums, axis=0)
    return df


def run() -> None:
    log.info("Loading assignments...")
    assignments = pd.read_parquet(COMMUNITIES_DIR / "county_community_assignments.parquet")
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
    if "community_id" not in assignments.columns and "community" in assignments.columns:
        assignments = assignments.rename(columns={"community": "community_id"})

    log.info("Loading vote totals (2024 president)...")
    pres_2024 = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")
    pres_2024["county_fips"] = pres_2024["county_fips"].astype(str).str.zfill(5)
    vote_totals = pres_2024[["county_fips", "pres_total_2024"]].rename(
        columns={"pres_total_2024": "recent_total"}
    )
    vote_totals["state_fips"] = vote_totals["county_fips"].str[:2]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    county_w = build_county_weights(assignments, vote_totals)
    county_w.to_parquet(OUTPUT_DIR / "community_weights_county_hac.parquet", index=False)
    log.info("Saved county weights: %s", county_w.shape)

    state_w = build_state_weights(assignments, vote_totals)
    state_w.to_parquet(OUTPUT_DIR / "community_weights_state_hac.parquet", index=False)
    log.info("Saved state weights:\n%s", state_w.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
