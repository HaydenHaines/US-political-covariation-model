"""Build county-level social connectedness features from Facebook SCI data.

Reads data/raw/facebook_sci/us_counties.csv (10.3M rows, county pairs) and
outputs data/assembled/county_sci_features.parquet with:

    network_diversity       Herfindahl index of SCI weights. Low HHI = connections
                            spread across many counties (diverse, nationally connected).
                            High HHI = concentrated on a few (insular). Inverted so
                            higher = more diverse for intuitive interpretation.

    sci_top5_mean_dem_share SCI-weighted average 2020 Dem share of a county's top 5
                            most-connected counties (excluding self). Captures the
                            political environment of the social network.

    sci_geographic_reach    Number of distinct states represented in the top-20
                            most-connected counties (excluding self). Higher = broader
                            geographic reach.

    pct_sci_instate         Percentage of total SCI weight flowing to counties in the
                            same state. High = insular state-level network.

Scope: All counties in the SCI data (national, ~3,100 user counties).
Memory strategy: Keep only top-N connections per county during aggregation to
avoid materializing all 10M rows in memory at once.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
SCI_PATH = PROJECT_ROOT / "data" / "raw" / "facebook_sci" / "us_counties.csv"
PRES_2020_PATH = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2020.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_sci_features.parquet"

# Number of top connections to retain per county for geographic/political features
_TOP_N_POLITICAL = 5
_TOP_N_GEOGRAPHIC = 20


def _load_sci(path: Path) -> pd.DataFrame:
    """Load SCI data with FIPS zero-padding and basic filtering."""
    log.info("Loading SCI data from %s", path)
    df = pd.read_csv(
        path,
        usecols=["user_region", "friend_region", "scaled_sci"],
        dtype={"user_region": str, "friend_region": str, "scaled_sci": float},
    )
    log.info("  Raw rows: %d", len(df))

    # Zero-pad FIPS to 5 chars (they may arrive as "1001" for Alabama counties)
    df["user_region"] = df["user_region"].str.zfill(5)
    df["friend_region"] = df["friend_region"].str.zfill(5)

    # Drop self-connections (not useful for social network features)
    self_mask = df["user_region"] == df["friend_region"]
    n_self = self_mask.sum()
    if n_self:
        log.info("  Dropping %d self-connection rows", n_self)
        df = df[~self_mask].copy()

    # Drop zero or negative SCI
    df = df[df["scaled_sci"] > 0].copy()
    log.info("  Rows after cleaning: %d", len(df))
    return df


def _build_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Herfindahl-Hirschman index of SCI weight distribution per user county.

    HHI = sum of squared share of each friend county's SCI weight.
    Range: (0, 1]. High HHI = concentrated (insular). Low HHI = diverse.
    We invert: network_diversity = 1 - HHI so higher = more diverse.
    """
    log.info("Computing network_diversity (1 - HHI) per county...")

    # Total SCI weight per user county
    total = df.groupby("user_region")["scaled_sci"].sum().rename("total_sci")

    # Per-pair share
    merged = df.merge(total, left_on="user_region", right_index=True)
    merged["sci_share"] = merged["scaled_sci"] / merged["total_sci"]

    # HHI = sum of squared shares
    hhi = (
        merged.groupby("user_region")["sci_share"]
        .apply(lambda x: (x ** 2).sum())
        .rename("hhi")
    )

    diversity = (1.0 - hhi).rename("network_diversity").reset_index()
    diversity.columns = ["county_fips", "network_diversity"]
    log.info("  Computed for %d counties", len(diversity))
    return diversity


def _build_pct_instate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fraction of SCI weight going to same-state counties."""
    log.info("Computing pct_sci_instate per county...")

    df = df.copy()
    df["user_state"] = df["user_region"].str[:2]
    df["friend_state"] = df["friend_region"].str[:2]
    df["is_instate"] = (df["user_state"] == df["friend_state"]).astype(float)

    agg = df.groupby("user_region").agg(
        total_sci=("scaled_sci", "sum"),
        instate_sci=("scaled_sci", lambda x: (x * df.loc[x.index, "is_instate"]).sum()),
    )
    agg["pct_sci_instate"] = agg["instate_sci"] / agg["total_sci"]
    result = agg[["pct_sci_instate"]].reset_index()
    result.columns = ["county_fips", "pct_sci_instate"]
    log.info("  Computed for %d counties", len(result))
    return result


def _get_top_n_connections(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return top-N friend counties by scaled_sci for each user county."""
    log.info("Selecting top-%d connections per county...", n)
    top = (
        df.sort_values("scaled_sci", ascending=False)
        .groupby("user_region")
        .head(n)
        .copy()
    )
    log.info("  Top-%d rows: %d", n, len(top))
    return top


def _build_political_feature(
    top5: pd.DataFrame, pres_2020: pd.DataFrame
) -> pd.DataFrame:
    """Compute SCI-weighted Dem share of top-5 connections.

    Parameters
    ----------
    top5:
        DataFrame with user_region, friend_region, scaled_sci (top 5 per user).
    pres_2020:
        Presidential 2020 results with county_fips and pres_dem_share_2020.
    """
    log.info("Computing sci_top5_mean_dem_share...")

    dem_share = pres_2020[["county_fips", "pres_dem_share_2020"]].copy()
    # Drop the aggregate row (county_fips == "00000" represents national summary)
    dem_share = dem_share[dem_share["county_fips"] != "00000"].copy()

    # Join Dem share onto friend counties
    top5_with_dem = top5.merge(
        dem_share.rename(columns={"county_fips": "friend_region",
                                  "pres_dem_share_2020": "friend_dem_share"}),
        on="friend_region",
        how="left",
    )

    # Drop friends without 2020 results (Alaska, some territories, etc.)
    n_missing = top5_with_dem["friend_dem_share"].isna().sum()
    if n_missing:
        log.info("  %d top-5 friend edges lack 2020 Dem share — excluded from weighted avg", n_missing)
    top5_with_dem = top5_with_dem.dropna(subset=["friend_dem_share"])

    # Weighted average Dem share per user county (vectorized)
    top5_with_dem = top5_with_dem.copy()
    top5_with_dem["weighted_dem"] = top5_with_dem["scaled_sci"] * top5_with_dem["friend_dem_share"]
    agg = top5_with_dem.groupby("user_region").agg(
        total_weight=("scaled_sci", "sum"),
        weighted_dem_sum=("weighted_dem", "sum"),
    )
    agg["sci_top5_mean_dem_share"] = agg["weighted_dem_sum"] / agg["total_weight"]
    result = agg[["sci_top5_mean_dem_share"]].reset_index()
    result.columns = ["county_fips", "sci_top5_mean_dem_share"]
    log.info("  Computed for %d counties", len(result))
    return result


def _build_geographic_reach(top20: pd.DataFrame) -> pd.DataFrame:
    """Count distinct states in top-20 connections (geographic reach)."""
    log.info("Computing sci_geographic_reach (distinct states in top-20)...")

    top20 = top20.copy()
    top20["friend_state"] = top20["friend_region"].str[:2]

    result = (
        top20.groupby("user_region")["friend_state"]
        .nunique()
        .rename("sci_geographic_reach")
        .reset_index()
    )
    result.columns = ["county_fips", "sci_geographic_reach"]
    log.info("  Computed for %d counties", len(result))
    return result


def build_features(sci: pd.DataFrame, pres_2020: pd.DataFrame) -> pd.DataFrame:
    """Compute all SCI county features and return a joined DataFrame.

    Parameters
    ----------
    sci:
        Raw SCI edge list (cleaned, no self-connections).
    pres_2020:
        County 2020 presidential results.

    Returns
    -------
    DataFrame with county_fips and 4 SCI feature columns.
    """
    # Feature 1: network diversity (HHI-based)
    diversity = _build_hhi(sci)

    # Feature 2: pct_sci_instate
    pct_instate = _build_pct_instate(sci)

    # Pre-compute top-N subsets (avoid repeated full-sort)
    top20 = _get_top_n_connections(sci, _TOP_N_GEOGRAPHIC)
    # Top-5 is a subset of top-20
    top5 = (
        top20.sort_values("scaled_sci", ascending=False)
        .groupby("user_region")
        .head(_TOP_N_POLITICAL)
        .copy()
    )

    # Feature 3: SCI-weighted Dem share of top-5 connections
    political = _build_political_feature(top5, pres_2020)

    # Feature 4: geographic reach (distinct states in top-20)
    geo_reach = _build_geographic_reach(top20)

    # Join all features on county_fips
    features = diversity.merge(pct_instate, on="county_fips", how="outer")
    features = features.merge(political, on="county_fips", how="left")
    features = features.merge(geo_reach, on="county_fips", how="left")

    features = features.reset_index(drop=True)
    return features


def main() -> None:
    sci_raw = _load_sci(SCI_PATH)

    log.info("Loading 2020 presidential results from %s", PRES_2020_PATH)
    pres_2020 = pd.read_parquet(PRES_2020_PATH)
    log.info("  %d counties in 2020 results", len(pres_2020))

    features = build_features(sci_raw, pres_2020)

    n_counties = len(features)
    n_na = features.isnull().any(axis=1).sum()
    log.info(
        "Built %d county SCI feature rows x %d feature columns | %d counties with ≥1 NaN",
        n_counties,
        len(features.columns) - 1,
        n_na,
    )

    if n_na > 0:
        log.warning("Columns with NaN:")
        for col in features.columns:
            n = features[col].isna().sum()
            if n > 0:
                log.warning("  %-40s  %d NaN", col, n)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved -> %s", OUTPUT_PATH)

    # Sanity prints
    print("\n--- SCI Feature Summary ---")
    for col in ["network_diversity", "pct_sci_instate", "sci_top5_mean_dem_share", "sci_geographic_reach"]:
        if col not in features.columns:
            continue
        s = features[col].dropna()
        print(f"  {col:40s}  min={s.min():.4f}  median={s.median():.4f}  max={s.max():.4f}")

    print("\n--- Top 5 most nationally connected counties (lowest pct_sci_instate) ---")
    print(features.nsmallest(5, "pct_sci_instate")[
        ["county_fips", "pct_sci_instate", "network_diversity", "sci_geographic_reach"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
