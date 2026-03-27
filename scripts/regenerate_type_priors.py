"""Regenerate type_priors.parquet for the current J=100 KMeans model.

The old file was built for J=20 (FL/GA/AL pilot). The national J=100 model
has 100 types, so types 20-99 all defaulted to 0.45 regardless of their
actual partisan lean. This corrupted the Bayesian update in every 2026 forecast.

This script recomputes priors using:
  - Ridge ensemble predictions (primary): model already accounts for demographics,
    geography, and historical patterns. Best available estimate of each county's
    structural partisan baseline.
  - 2024 actual dem_share (fallback): for counties without Ridge predictions.

For each type t, the prior is a soft-membership, vote-weighted mean:

    prior[t] = sum_c( score[c,t] * votes[c] * dem_share[c] )
               / sum_c( score[c,t] * votes[c] )

where score[c,t] is the county's soft membership in type t (sums to 1 across t),
and votes[c] is total 2024 presidential vote (as a population proxy).

Outputs:
  - data/communities/type_priors.parquet  (prior_dem_share col — used by predict_2026_types.py)
  - data/communities/type_profiles.parquet  (adds mean_dem_share col — used by build_database.py)

Usage:
    uv run python scripts/regenerate_type_priors.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Fallback prior for types with zero total weight (no counties assigned)
DEFAULT_PRIOR = 0.45

# Paths
TYPE_ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
RIDGE_PRIORS_PATH = PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
PRES_2024_PATH = PROJECT_ROOT / "data" / "assembled" / "medsl_county_2024_president.parquet"
TYPE_PRIORS_OUT = PROJECT_ROOT / "data" / "communities" / "type_priors.parquet"
TYPE_PROFILES_PATH = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"


def compute_type_priors(
    type_scores: np.ndarray,
    county_fips: list[str],
    dem_share: np.ndarray,
    vote_weights: np.ndarray,
) -> np.ndarray:
    """Compute soft-membership, vote-weighted dem_share prior per type.

    Parameters
    ----------
    type_scores : (N, J) array of soft membership scores (rows sum to 1)
    county_fips : list of N FIPS strings (aligned with type_scores rows)
    dem_share   : (N,) array; NaN where no data available
    vote_weights: (N,) array of total votes (population proxy); must be > 0

    Returns
    -------
    priors : (J,) array of dem_share priors, one per type
    """
    j = type_scores.shape[1]
    priors = np.full(j, DEFAULT_PRIOR)
    valid = ~np.isnan(dem_share)

    for t in range(j):
        w = type_scores[:, t] * vote_weights
        w_valid = w[valid]
        d_valid = dem_share[valid]
        total_w = w_valid.sum()
        if total_w > 0:
            priors[t] = (w_valid * d_valid).sum() / total_w
        else:
            log.warning("Type %d has zero total weight — using default %.2f", t, DEFAULT_PRIOR)

    return priors


def main() -> None:
    log.info("Loading type assignments from %s", TYPE_ASSIGNMENTS_PATH)
    ta = pd.read_parquet(TYPE_ASSIGNMENTS_PATH)
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)

    score_cols = sorted(
        [c for c in ta.columns if c.endswith("_score") and c.startswith("type_")],
        key=lambda c: int(c.split("_")[1]),
    )
    j = len(score_cols)
    n = len(ta)
    log.info("Type assignments: %d counties, J=%d types", n, j)

    type_scores = ta[score_cols].values  # (N, J)
    county_fips = ta["county_fips"].tolist()

    # --- Build dem_share array (Ridge preferred, 2024 actual fallback) ---
    dem_share = np.full(n, np.nan)
    vote_weights = np.ones(n)

    # Load 2024 actuals first (fallback baseline)
    if PRES_2024_PATH.exists():
        pres2024 = pd.read_parquet(PRES_2024_PATH)
        pres2024["county_fips"] = pres2024["county_fips"].astype(str).str.zfill(5)
        share_map = dict(zip(pres2024["county_fips"], pres2024["pres_dem_share_2024"]))
        votes_map = dict(zip(pres2024["county_fips"], pres2024["pres_total_2024"]))
        for i, fips in enumerate(county_fips):
            if fips in share_map:
                dem_share[i] = share_map[fips]
            if fips in votes_map and pd.notna(votes_map[fips]):
                vote_weights[i] = max(votes_map[fips], 1.0)
        n_2024 = np.sum(~np.isnan(dem_share))
        log.info("2024 actual data: %d/%d counties matched", n_2024, n)
    else:
        log.warning("2024 presidential data not found at %s — no fallback", PRES_2024_PATH)

    # Overwrite with Ridge predictions where available (higher accuracy)
    n_ridge = 0
    if RIDGE_PRIORS_PATH.exists():
        ridge_df = pd.read_parquet(RIDGE_PRIORS_PATH)
        ridge_df["county_fips"] = ridge_df["county_fips"].astype(str).str.zfill(5)
        ridge_map = dict(zip(ridge_df["county_fips"], ridge_df["ridge_pred_dem_share"]))
        for i, fips in enumerate(county_fips):
            if fips in ridge_map:
                dem_share[i] = ridge_map[fips]
                n_ridge += 1
        log.info(
            "Ridge predictions: %d/%d counties replaced with Ridge dem_share "
            "(%d remaining on 2024 actuals)",
            n_ridge, n, n - n_ridge,
        )
    else:
        log.warning("Ridge priors not found at %s — using 2024 actuals only", RIDGE_PRIORS_PATH)

    n_missing = np.sum(np.isnan(dem_share))
    if n_missing > 0:
        log.warning("%d counties have no dem_share data; they will not contribute to any type prior", n_missing)

    # --- Compute priors ---
    priors = compute_type_priors(type_scores, county_fips, dem_share, vote_weights)

    log.info(
        "Type priors — range [%.3f, %.3f], mean=%.3f",
        priors.min(), priors.max(), priors.mean(),
    )

    # Sanity check: most partisan types
    sorted_idx = np.argsort(priors)
    log.info(
        "5 most Republican types: %s → priors %s",
        sorted_idx[:5].tolist(),
        np.round(priors[sorted_idx[:5]], 3).tolist(),
    )
    log.info(
        "5 most Democratic types: %s → priors %s",
        sorted_idx[-5:].tolist(),
        np.round(priors[sorted_idx[-5:]], 3).tolist(),
    )

    # --- Write type_priors.parquet ---
    out_df = pd.DataFrame({
        "type_id": np.arange(j, dtype=int),
        "prior_dem_share": priors,
    })
    out_df.to_parquet(TYPE_PRIORS_OUT, index=False)
    log.info("Wrote %d rows to %s", len(out_df), TYPE_PRIORS_OUT)

    # --- Also add mean_dem_share to type_profiles.parquet ---
    # The DB ingestion pipeline (_ingest_type_priors in model.py) reads
    # type_profiles.parquet looking for mean_dem_share. Without this column
    # it falls back to 0.45 for all types.
    if TYPE_PROFILES_PATH.exists():
        profiles = pd.read_parquet(TYPE_PROFILES_PATH)
        # Build a map from type_id → prior so we handle any ordering
        prior_map = dict(zip(out_df["type_id"], out_df["prior_dem_share"]))
        profiles["mean_dem_share"] = profiles["type_id"].map(prior_map).fillna(DEFAULT_PRIOR)
        profiles.to_parquet(TYPE_PROFILES_PATH, index=False)
        log.info("Updated type_profiles.parquet with mean_dem_share column (%d rows)", len(profiles))
    else:
        log.warning("type_profiles.parquet not found — skipping mean_dem_share update")

    # --- Verify: print full table ---
    print("\nType priors (all %d types):" % j)
    print(f"{'type_id':>8}  {'prior_dem_share':>16}")
    print("-" * 28)
    for _, row in out_df.iterrows():
        label = "D" if row["prior_dem_share"] > 0.5 else "R"
        margin = abs(row["prior_dem_share"] - 0.5) * 100
        print(f"  {int(row['type_id']):>6}  {row['prior_dem_share']:>16.4f}  ({label}+{margin:.1f})")


if __name__ == "__main__":
    main()
