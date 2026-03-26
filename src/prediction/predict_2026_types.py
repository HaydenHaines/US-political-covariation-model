"""2026 predictions using type structure (type-primary pipeline).

Loads type assignments, type covariance, county-level historical baselines,
and polls. Performs Gaussian Bayesian update through type structure to produce
county-level 2026 predictions.

Key design: county-level priors, type-level covariance.
  - Each county's prior prediction = its own historical baseline (mean Dem share)
  - Types determine comovement only (how polls adjust predictions)
  - The Bayesian update shifts type means; the shift propagates to counties via
    type scores, but is added to county baselines (not type baselines)

Inputs:
  data/communities/type_assignments.parquet       — county type scores
  data/covariance/type_covariance.parquet          — J x J covariance
  data/communities/type_profiles.parquet           — type demographic profiles
  data/polls/polls_2026.csv                        — poll observations
  data/assembled/medsl_county_presidential_*.parquet — county historical results

Outputs:
  data/predictions/county_predictions_2026_types.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as _cfg

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# State FIPS -> abbreviation (all 50 states + DC, sourced from config/model.yaml)
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR


def compute_county_priors(
    county_fips: list[str],
    assembled_dir: Path | None = None,
) -> np.ndarray:
    """Compute county-level prior Dem share from historical election results.

    Uses the most recent presidential Dem share as the primary prior.
    Falls back to mean across available elections if 2024 is missing.
    Falls back to 0.45 (generic prior) if no data available.

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

    # Load available presidential results (most recent first)
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

    # Build prior array: use most recent available, fall back to mean, then 0.45
    priors = np.full(N, 0.45)
    for i, fips in enumerate(county_fips):
        vals = dem_shares[fips]
        if vals:
            priors[i] = vals[0]  # most recent (2024 first in years list)

    return priors


def compute_county_priors_from_data(
    county_fips: list[str],
    dem_share_map: dict[str, float],
    fallback: float = 0.45,
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


def predict_race(
    race: str,
    type_scores: np.ndarray,
    type_covariance: np.ndarray,
    type_priors: np.ndarray,
    county_fips: list[str],
    polls: list[tuple[float, int, str]] | None = None,
    states: list[str] | None = None,
    county_names: list[str] | None = None,
    state_filter: str | None = None,
    county_priors: np.ndarray | None = None,
) -> pd.DataFrame:
    """Produce county-level predictions from type structure.

    Uses county-level priors (each county's own historical baseline) when
    provided. Types determine only the covariance structure (how poll
    observations shift predictions). The prediction formula is:

        county_pred = county_prior + type_covariance_adjustment

    where the adjustment comes from the Bayesian update on type means, and
    each county's adjustment is the score-weighted shift in type means.

    When county_priors is None, falls back to type-mean priors (legacy).

    Multiple polls are stacked into a single Bayesian update: each poll
    contributes one row to the W matrix (n_polls × J), enabling exact
    multi-poll inference rather than collapsing to a single effective poll.
    This preserves geographic information when polls cover different states.

    Parameters
    ----------
    race : str
        Election race label (e.g. "FL Senate").
    type_scores : ndarray of shape (N, J)
        County type scores (soft membership, can be negative).
    type_covariance : ndarray of shape (J, J)
        Type covariance matrix.
    type_priors : ndarray of shape (J,)
        Prior Dem share per type (used for Bayesian update baseline).
    county_fips : list[str]
        FIPS codes for each county (length N).
    polls : list of (dem_share, n, state_abbr) tuples or None
        Poll observations. Each tuple is one poll: Democratic two-party
        share (0-1), sample size, and the state abbreviation whose type
        composition defines the observation equation (W row). Multiple
        polls are stacked into a single multi-row Bayesian update.
        None = use prior only (no poll adjustment).
    states : list[str] or None
        State abbreviation per county. Derived from FIPS if None.
    county_names : list[str] or None
        County names. Set to empty string if None.
    state_filter : str or None
        If provided, filter output rows to this state abbreviation.
        Does NOT affect which polls are applied — poll state comes
        from each tuple in `polls`.
    county_priors : ndarray of shape (N,) or None
        Per-county prior Dem share (historical baseline). When provided,
        predictions use county baselines + type covariance adjustments.
        When None, falls back to type-mean weighted predictions (legacy).

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, state, county_name, pred_dem_share,
        ci_lower, ci_upper, dominant_type, super_type
    """
    N, J = type_scores.shape
    assert len(county_fips) == N
    assert type_covariance.shape == (J, J)
    assert len(type_priors) == J
    if county_priors is not None:
        assert len(county_priors) == N

    # Derive states from FIPS if not provided
    if states is None:
        states = [_STATE_FIPS_TO_ABBR.get(f[:2], "??") for f in county_fips]
    if county_names is None:
        county_names = [""] * N

    # ── Type-level Bayesian update ──────────────────────────────────────────
    type_means = type_priors.copy().astype(float)
    type_cov = type_covariance.copy().astype(float)

    if polls:
        # Each poll contributes one W row: the type composition of its state.
        # Stacking all rows into a single update preserves geographic information
        # across polls covering different states (vs. collapsing to one scalar).
        W_rows = []
        y_vals = []
        sigma_vals = []
        for dem_share, n, poll_state in polls:
            if poll_state:
                state_mask = np.array([s == poll_state for s in states])
                if state_mask.any():
                    state_scores = type_scores[state_mask]
                    W_row = np.abs(state_scores).mean(axis=0)
                    W_sum = W_row.sum()
                    W_row = W_row / W_sum if W_sum > 0 else np.ones(J) / J
                else:
                    W_row = np.ones(J) / J
            else:
                W_row = np.ones(J) / J
            W_rows.append(W_row)
            y_vals.append(dem_share)
            sigma_vals.append(np.sqrt(dem_share * (1 - dem_share) / n))

        type_means, type_cov = _bayesian_update(
            mu_prior=type_means,
            sigma_prior=type_cov,
            W=np.array(W_rows),
            y=np.array(y_vals),
            sigma_polls=np.array(sigma_vals),
        )

    # ── Map type estimates back to counties ─────────────────────────────────
    abs_scores = np.abs(type_scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)  # avoid div by zero

    if county_priors is not None:
        # County-level prior approach:
        # 1. Compute the type-level shift from the Bayesian update
        type_shift = type_means - type_priors.astype(float)
        # 2. Each county's adjustment = score-weighted average of type shifts
        county_adjustment = (abs_scores * type_shift[None, :]).sum(axis=1) / weight_sums
        # 3. Final prediction = county baseline + adjustment
        pred_dem_share = county_priors.astype(float) + county_adjustment
    else:
        # Legacy: type-mean weighted predictions
        pred_dem_share = (abs_scores * type_means[None, :]).sum(axis=1) / weight_sums

    # Clip to [0, 1]
    pred_dem_share = np.clip(pred_dem_share, 0.0, 1.0)

    # ── Uncertainty from covariance diagonal + type weights ─────────────────
    type_std = np.sqrt(np.diag(type_cov))
    county_std = (abs_scores * type_std[None, :]).sum(axis=1) / weight_sums

    ci_lower = np.clip(pred_dem_share - 1.645 * county_std, 0.0, 1.0)
    ci_upper = np.clip(pred_dem_share + 1.645 * county_std, 0.0, 1.0)

    # Dominant type per county
    dominant_type = np.argmax(np.abs(type_scores), axis=1)

    # Super-type: set to -1 here; the calling code should join against
    # type_assignments.parquet (which has the authoritative super_type from nest_types).
    super_type = np.full(N, -1, dtype=int)

    result = pd.DataFrame({
        "county_fips": county_fips,
        "state": states,
        "county_name": county_names,
        "pred_dem_share": pred_dem_share,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "dominant_type": dominant_type,
        "super_type": super_type,
    })

    if state_filter is not None:
        result = result[result["state"] == state_filter].reset_index(drop=True)

    return result


def _bayesian_update(
    mu_prior: np.ndarray,
    sigma_prior: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    sigma_polls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian Bayesian update: posterior mean and covariance.

    Same mathematical formulation as predict_2026_hac.bayesian_update.
    """
    R = np.diag(sigma_polls ** 2)
    sigma_prior_inv = np.linalg.inv(
        sigma_prior + np.eye(len(mu_prior)) * 1e-8
    )
    sigma_post_inv = sigma_prior_inv + W.T @ np.linalg.inv(R) @ W
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (
        sigma_prior_inv @ mu_prior + W.T @ np.linalg.solve(R, y)
    )
    return mu_post, sigma_post


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Load inputs from data/ and produce type-based 2026 predictions."""
    type_assignments_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    type_cov_path = PROJECT_ROOT / "data" / "covariance" / "type_covariance.parquet"
    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    crosswalk_path = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"

    # Load type assignments (county scores)
    log.info("Loading type assignments from %s", type_assignments_path)
    ta_df = pd.read_parquet(type_assignments_path)
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values
    J = type_scores.shape[1]

    # Load type covariance
    log.info("Loading type covariance from %s", type_cov_path)
    cov_df = pd.read_parquet(type_cov_path)
    type_covariance = cov_df.values[:J, :J]

    # Load type priors from 2024 actual results (population-weighted Dem share per type)
    # These are used as the baseline for the Bayesian update (type-level)
    type_priors = np.full(J, 0.45)
    priors_path = PROJECT_ROOT / "data" / "communities" / "type_priors.parquet"
    if priors_path.exists():
        priors_df = pd.read_parquet(priors_path)
        if "prior_dem_share" in priors_df.columns:
            for _, row in priors_df.iterrows():
                t = int(row["type_id"])
                if t < J:
                    type_priors[t] = row["prior_dem_share"]
    log.info("Type priors: %s", np.round(type_priors, 3))

    # Load county-level priors: Ridge-predicted if available, else historical
    ridge_priors_path = PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
    county_prior_values = compute_county_priors(county_fips)  # baseline fallback

    if ridge_priors_path.exists():
        log.info("Loading Ridge county priors from %s", ridge_priors_path)
        ridge_df = pd.read_parquet(ridge_priors_path)
        ridge_df["county_fips"] = ridge_df["county_fips"].astype(str).str.zfill(5)
        ridge_map = dict(zip(ridge_df["county_fips"], ridge_df["ridge_pred_dem_share"]))
        n_matched = sum(1 for f in county_fips if f in ridge_map)
        n_fallback = len(county_fips) - n_matched
        for i, fips in enumerate(county_fips):
            if fips in ridge_map:
                county_prior_values[i] = ridge_map[fips]
        log.info(
            "Ridge priors: %d/%d counties matched; %d using historical fallback",
            n_matched, len(county_fips), n_fallback,
        )
        print(f"Using Ridge priors for {n_matched}/{len(county_fips)} counties "
              f"({n_fallback} fallback to historical)")
    else:
        log.info(
            "Ridge model not found at %s — using historical county priors",
            ridge_priors_path,
        )
        n_with_data = np.sum(county_prior_values != 0.45)
        log.info(
            "County priors (historical): %d/%d counties have data, range [%.3f, %.3f]",
            n_with_data, len(county_fips),
            county_prior_values.min(), county_prior_values.max(),
        )

    # Derive states and names
    states = [_STATE_FIPS_TO_ABBR.get(f[:2], "??") for f in county_fips]
    county_names = [""] * len(county_fips)
    if crosswalk_path.exists():
        xwalk = pd.read_csv(crosswalk_path, dtype=str)
        xwalk["county_fips"] = xwalk["county_fips"].str.zfill(5)
        name_map = dict(zip(xwalk["county_fips"], xwalk["county_name"]))
        county_names = [name_map.get(f, "") for f in county_fips]

    # Load polls
    log.info("Loading polls from %s", polls_path)
    polls = pd.read_csv(polls_path)
    # Normalize column name
    if "geography" in polls.columns and "state" not in polls.columns:
        polls = polls.rename(columns={"geography": "state"})

    # Group polls by race so each race receives all its polls in a single
    # stacked Bayesian update (rather than collapsing to one effective poll).
    # We keep individual rows — no aggregation — to preserve geographic
    # information when multiple polls cover different states.
    poll_agg = polls[polls.get("geo_level", pd.Series(["state"] * len(polls))) == "state"].copy()
    if "geo_level" in polls.columns:
        poll_agg = polls[polls["geo_level"] == "state"].copy()
    else:
        poll_agg = polls.copy()

    all_predictions = []
    for race, race_group in poll_agg.groupby("race"):
        race_polls = [
            (float(row["dem_share"]), int(row["n_sample"]), str(row["state"]))
            for _, row in race_group.iterrows()
            if row.get("geo_level", "state") == "state"
        ]
        result = predict_race(
            race=race,
            polls=race_polls if race_polls else None,
            type_scores=type_scores,
            type_covariance=type_covariance,
            type_priors=type_priors,
            county_fips=county_fips,
            states=states,
            county_names=county_names,
            county_priors=county_prior_values,
        )
        result["race"] = race
        all_predictions.append(result)

    # Generate national baseline for all counties (no poll adjustment)
    baseline = predict_race(
        race="baseline",
        type_scores=type_scores,
        type_covariance=type_covariance,
        type_priors=type_priors,
        county_fips=county_fips,
        states=states,
        county_names=county_names,
        county_priors=county_prior_values,
    )
    baseline["race"] = "baseline"
    all_predictions.append(baseline)

    if all_predictions:
        output = pd.concat(all_predictions, ignore_index=True)
    else:
        output = pd.DataFrame()

    out_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)
    log.info("Saved %d predictions to %s", len(output), out_path)
    print(f"Saved {len(output)} county predictions to {out_path}")

    if len(output):
        print(output.groupby(["state", "race"])["pred_dem_share"].describe().round(3))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run()
