"""2026 predictions using type structure (type-primary pipeline).

Loads SVD+varimax type scores, type covariance, and polls, performs
Gaussian Bayesian update through type structure, and produces county-level
2026 predictions.

Inputs:
  data/communities/type_assignments.parquet       — county type scores
  data/covariance/type_covariance.parquet          — J x J covariance
  data/communities/type_profiles.parquet           — type demographic profiles
  data/polls/polls_2026.csv                        — poll observations

Outputs:
  data/predictions/county_predictions_2026_types.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# State FIPS -> abbreviation
_STATE_FIPS_TO_ABBR = {"12": "FL", "13": "GA", "01": "AL"}


def predict_race(
    race: str,
    poll_dem_share: float | None,
    poll_n: int | None,
    type_scores: np.ndarray,
    type_covariance: np.ndarray,
    type_priors: np.ndarray,
    county_fips: list[str],
    states: list[str] | None = None,
    county_names: list[str] | None = None,
    state_filter: str | None = None,
) -> pd.DataFrame:
    """Produce county-level predictions from type structure.

    Parameters
    ----------
    race : str
        Election race label (e.g. "FL Senate").
    poll_dem_share : float or None
        Poll Democratic share (0-1). None = use prior only.
    poll_n : int or None
        Poll sample size. Required if poll_dem_share is not None.
    type_scores : ndarray of shape (N, J)
        County type scores (soft membership, can be negative).
    type_covariance : ndarray of shape (J, J)
        Type covariance matrix.
    type_priors : ndarray of shape (J,)
        Prior Dem share per type.
    county_fips : list[str]
        FIPS codes for each county (length N).
    states : list[str] or None
        State abbreviation per county. Derived from FIPS if None.
    county_names : list[str] or None
        County names. Set to empty string if None.
    state_filter : str or None
        If provided, filter output to this state abbreviation.

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

    # Derive states from FIPS if not provided
    if states is None:
        states = [_STATE_FIPS_TO_ABBR.get(f[:2], "??") for f in county_fips]
    if county_names is None:
        county_names = [""] * N

    # ── Type-level Bayesian update ──────────────────────────────────────────
    type_means = type_priors.copy().astype(float)
    type_cov = type_covariance.copy().astype(float)

    if poll_dem_share is not None and poll_n is not None:
        # Decompose state-level poll into type-level signal
        # Weight vector: average type scores for counties in the polled state
        poll_state = state_filter or _extract_state_from_race(race)
        if poll_state:
            state_mask = np.array([s == poll_state for s in states])
            if state_mask.any():
                # W = mean absolute type scores for state counties, normalized
                state_scores = type_scores[state_mask]
                W = np.abs(state_scores).mean(axis=0)
                W_sum = W.sum()
                if W_sum > 0:
                    W = W / W_sum  # normalize to sum to 1
                else:
                    W = np.ones(J) / J

                # Gaussian Bayesian update
                poll_sigma = np.sqrt(
                    poll_dem_share * (1 - poll_dem_share) / poll_n
                )
                type_means, type_cov = _bayesian_update(
                    mu_prior=type_means,
                    sigma_prior=type_cov,
                    W=W.reshape(1, -1),
                    y=np.array([poll_dem_share]),
                    sigma_polls=np.array([poll_sigma]),
                )

    # ── Map type estimates back to counties ─────────────────────────────────
    # county_pred = sum_j(|score_j| * type_mean_j) / sum_j(|score_j|)
    abs_scores = np.abs(type_scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)  # avoid div by zero

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

    # Super-type placeholder (type // 4 as simple grouping; real pipeline
    # reads from hierarchical nesting output)
    super_type = dominant_type // 4

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


def _extract_state_from_race(race: str) -> str | None:
    """Try to extract state abbreviation from race string like 'FL Senate'."""
    for abbr in _STATE_FIPS_TO_ABBR.values():
        if abbr in race:
            return abbr
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Load inputs from data/ and produce type-based 2026 predictions."""
    type_assignments_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    type_cov_path = PROJECT_ROOT / "data" / "covariance" / "type_covariance.parquet"
    type_profiles_path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
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

    # Load type priors from profiles (mean Dem share per type)
    # If not available, use flat prior
    type_priors = np.full(J, 0.45)
    if type_profiles_path.exists():
        profiles = pd.read_parquet(type_profiles_path)
        if "mean_dem_share" in profiles.columns:
            type_priors = profiles["mean_dem_share"].values[:J]
        log.info("Type priors: %s", np.round(type_priors, 3))

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

    # Aggregate polls by (race, state)
    poll_agg = (
        polls.groupby(["race", "geo_level"])
        .agg(
            dem_share=("dem_share", "mean"),
            n_sample=("n_sample", "sum"),
            state=("state", "first"),
        )
        .reset_index()
    )

    all_predictions = []
    for _, poll_row in poll_agg.iterrows():
        race = poll_row["race"]
        poll_dem = float(poll_row["dem_share"])
        poll_n = int(poll_row["n_sample"])
        geo = poll_row["state"]

        result = predict_race(
            race=race,
            poll_dem_share=poll_dem,
            poll_n=poll_n,
            type_scores=type_scores,
            type_covariance=type_covariance,
            type_priors=type_priors,
            county_fips=county_fips,
            states=states,
            county_names=county_names,
            state_filter=geo if len(geo) == 2 else None,
        )
        result["race"] = race
        all_predictions.append(result)

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
