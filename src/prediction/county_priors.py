"""County-level prior computation for the type-primary forecast pipeline.

County priors are the historical baseline Dem share estimates used as the
starting point before poll-based Bayesian updates. Two entry points:

  - compute_county_priors(): file-based loader (reads MEDSL parquet files)
  - compute_county_priors_from_data(): pure-function version for testing

The file-based loader attempts Ridge-predicted priors first (from
data/models/ridge_model/ridge_county_priors.parquet) and falls back to
historical presidential results if the Ridge model has not been trained yet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default Dem share when no historical data is available for a county.
# 0.45 is a slightly R-leaning national prior consistent with recent presidential results.
_FALLBACK_DEM_SHARE = 0.45


def compute_county_priors(
    county_fips: list[str],
    assembled_dir: Path | None = None,
) -> np.ndarray:
    """Compute county-level prior Dem share from historical election results.

    Uses the most recent presidential Dem share as the primary prior.
    Falls back to mean across available elections if 2024 is missing.
    Falls back to _FALLBACK_DEM_SHARE (generic prior) if no data available.

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes (zero-padded to 5 digits).
    assembled_dir : Path or None
        Directory containing MEDSL county parquet files.
        Defaults to PROJECT_ROOT / "data" / "assembled".

    Returns
    -------
    ndarray of shape (N,)
        Prior Dem share per county, one per FIPS in county_fips.
    """
    if assembled_dir is None:
        assembled_dir = PROJECT_ROOT / "data" / "assembled"

    N = len(county_fips)
    fips_set = set(county_fips)

    # Load available presidential results (most recent first so index 0 = most recent)
    years = [2024, 2020, 2016, 2012, 2008]
    dem_shares: dict[str, list[float]] = {f: [] for f in county_fips}

    for year in years:
        path = assembled_dir / f"medsl_county_presidential_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        share_col = f"pres_dem_share_{year}"
        if share_col not in df.columns:
            continue
        for _, row in df.iterrows():
            fips = row["county_fips"]
            if fips in fips_set:
                val = row[share_col]
                if pd.notna(val):
                    dem_shares[fips].append(float(val))

    # Build prior array: use most recent available, fall back to _FALLBACK_DEM_SHARE
    priors = np.full(N, _FALLBACK_DEM_SHARE)
    for i, fips in enumerate(county_fips):
        vals = dem_shares[fips]
        if vals:
            priors[i] = vals[0]  # vals[0] is most recent (years list is newest-first)

    return priors


def compute_county_priors_from_data(
    county_fips: list[str],
    dem_share_map: dict[str, float],
    fallback: float = _FALLBACK_DEM_SHARE,
) -> np.ndarray:
    """Compute county-level priors from a pre-built FIPS->dem_share mapping.

    This is the testable pure-function version (no file I/O).

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes.
    dem_share_map : dict[str, float]
        Mapping from FIPS to Dem share (e.g., from 2024 results).
    fallback : float
        Default Dem share for counties not in the map.

    Returns
    -------
    ndarray of shape (N,)
    """
    return np.array([dem_share_map.get(f, fallback) for f in county_fips])


def load_county_priors_with_ridge_governor(
    county_fips: list[str],
    governor_priors_path: Path | None = None,
    presidential_priors_path: Path | None = None,
    assembled_dir: Path | None = None,
) -> np.ndarray:
    """Load governor-specific county priors, falling back to presidential priors.

    Loads from the governor Ridge model first.  For counties not covered by
    the governor model (states that never had a governor race in the training
    data, or counties missing from the feature matrix), falls back to the
    presidential Ridge priors.  Finally falls back to historical presidential
    results for any county still uncovered.

    The two-level fallback is important: governor data is sparse (only states
    with a governor election that year contribute training data), so some
    counties will genuinely have no governor-specific prediction.  The
    presidential prior is a reasonable structural substitute there.

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes (zero-padded to 5 digits).
    governor_priors_path : Path or None
        Path to the governor Ridge county priors parquet.
        Defaults to data/models/ridge_model_governor/ridge_county_priors_governor.parquet.
    presidential_priors_path : Path or None
        Path to the presidential Ridge county priors parquet, used as first
        fallback when a county has no governor prior.
        Defaults to data/models/ridge_model/ridge_county_priors.parquet.
    assembled_dir : Path or None
        Directory containing MEDSL county parquet files (for final fallback).

    Returns
    -------
    ndarray of shape (N,)
        Prior Dem share per county, governor-trained where available.
    """
    if governor_priors_path is None:
        governor_priors_path = (
            PROJECT_ROOT
            / "data"
            / "models"
            / "ridge_model_governor"
            / "ridge_county_priors_governor.parquet"
        )
    if presidential_priors_path is None:
        presidential_priors_path = (
            PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
        )

    # Start with presidential priors (historical or Ridge) as the baseline.
    # This ensures every county has a value before we overlay governor priors.
    county_prior_values = load_county_priors_with_ridge(
        county_fips,
        ridge_priors_path=presidential_priors_path,
        assembled_dir=assembled_dir,
    )

    if not governor_priors_path.exists():
        log.warning(
            "Governor Ridge model not found at %s — using presidential priors for all counties",
            governor_priors_path,
        )
        return county_prior_values

    log.info("Loading governor Ridge county priors from %s", governor_priors_path)
    gov_df = pd.read_parquet(governor_priors_path)
    gov_df["county_fips"] = gov_df["county_fips"].astype(str).str.zfill(5)
    gov_map = dict(zip(gov_df["county_fips"], gov_df["ridge_pred_dem_share"]))

    n_matched = 0
    for i, fips in enumerate(county_fips):
        if fips in gov_map:
            county_prior_values[i] = gov_map[fips]
            n_matched += 1
    n_fallback = len(county_fips) - n_matched

    log.info(
        "Governor priors: %d/%d counties matched; %d using presidential fallback",
        n_matched,
        len(county_fips),
        n_fallback,
    )
    print(
        f"Using governor Ridge priors for {n_matched}/{len(county_fips)} counties "
        f"({n_fallback} fallback to presidential priors)"
    )
    return county_prior_values


def load_county_priors_with_ridge(
    county_fips: list[str],
    ridge_priors_path: Path | None = None,
    assembled_dir: Path | None = None,
) -> np.ndarray:
    """Load county priors, preferring Ridge model predictions over historical data.

    Attempts to load Ridge-predicted priors. Falls back to historical presidential
    Dem share for any county not covered by the Ridge model, and to
    _FALLBACK_DEM_SHARE for counties with no historical data at all.

    Parameters
    ----------
    county_fips : list[str]
        FIPS codes (zero-padded to 5 digits).
    ridge_priors_path : Path or None
        Path to Ridge county priors parquet. Defaults to the standard model path.
    assembled_dir : Path or None
        Directory containing MEDSL county parquet files.

    Returns
    -------
    ndarray of shape (N,)
        Prior Dem share per county.
    """
    if ridge_priors_path is None:
        ridge_priors_path = (
            PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
        )

    # Always compute historical fallback first so we have a baseline for every county
    county_prior_values = compute_county_priors(county_fips, assembled_dir=assembled_dir)

    if ridge_priors_path.exists():
        log.info("Loading Ridge county priors from %s", ridge_priors_path)
        ridge_df = pd.read_parquet(ridge_priors_path)
        ridge_df["county_fips"] = ridge_df["county_fips"].astype(str).str.zfill(5)
        ridge_map = dict(zip(ridge_df["county_fips"], ridge_df["ridge_pred_dem_share"]))

        n_matched = 0
        for i, fips in enumerate(county_fips):
            if fips in ridge_map:
                county_prior_values[i] = ridge_map[fips]
                n_matched += 1
        n_fallback = len(county_fips) - n_matched

        log.info(
            "Ridge priors: %d/%d counties matched; %d using historical fallback",
            n_matched,
            len(county_fips),
            n_fallback,
        )
        print(
            f"Using Ridge priors for {n_matched}/{len(county_fips)} counties "
            f"({n_fallback} fallback to historical)"
        )
    else:
        log.info(
            "Ridge model not found at %s — using historical county priors",
            ridge_priors_path,
        )
        n_with_data = int(np.sum(county_prior_values != _FALLBACK_DEM_SHARE))
        log.info(
            "County priors (historical): %d/%d counties have data, range [%.3f, %.3f]",
            n_with_data,
            len(county_fips),
            county_prior_values.min(),
            county_prior_values.max(),
        )

    return county_prior_values
