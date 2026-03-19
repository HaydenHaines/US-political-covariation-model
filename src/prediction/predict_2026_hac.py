"""2026 predictions using HAC community structure (Phase 1 pipeline).

Loads the HAC community Sigma (from Task 4), state weight matrix,
and polls, runs the Gaussian Bayesian update, and produces county-level
2026 predictions.

Inputs:
  data/covariance/county_community_sigma.parquet
  data/covariance/county_theta_obs.parquet
  data/propagation/community_weights_state_hac.parquet
  data/propagation/community_weights_county_hac.parquet
  data/polls/polls_2026.csv

Outputs:
  data/predictions/county_predictions_2026_hac.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLLS_PATH = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
SIGMA_PATH = PROJECT_ROOT / "data" / "covariance" / "county_community_sigma.parquet"
STATE_W_PATH = PROJECT_ROOT / "data" / "propagation" / "community_weights_state_hac.parquet"
COUNTY_W_PATH = PROJECT_ROOT / "data" / "propagation" / "community_weights_county_hac.parquet"
THETA_OBS_PATH = PROJECT_ROOT / "data" / "covariance" / "county_theta_obs.parquet"


def bayesian_update(
    mu_prior: np.ndarray,
    sigma_prior: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    sigma_polls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian Bayesian update: posterior mean and covariance."""
    R = np.diag(sigma_polls ** 2)
    sigma_prior_inv = np.linalg.inv(sigma_prior + np.eye(len(mu_prior)) * 1e-8)
    sigma_post_inv = sigma_prior_inv + W.T @ np.linalg.inv(R) @ W
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (sigma_prior_inv @ mu_prior + W.T @ np.linalg.solve(R, y))
    return mu_post, sigma_post


def run() -> None:
    log.info("Loading HAC community Sigma...")
    sigma_df = pd.read_parquet(SIGMA_PATH)
    k_ids = list(sigma_df.index)
    K = len(k_ids)
    Sigma = sigma_df.values.astype(float)

    log.info("Loading state weights...")
    state_w = pd.read_parquet(STATE_W_PATH)
    weight_cols = sorted([c for c in state_w.columns if c.startswith("community_")])

    log.info("Loading county weights...")
    county_w = pd.read_parquet(COUNTY_W_PATH)

    log.info("Loading polls...")
    polls = pd.read_csv(POLLS_PATH)

    # Polls use 'geography' column (state abbreviation), not 'state'
    if "state" not in polls.columns and "geography" in polls.columns:
        polls = polls.rename(columns={"geography": "state"})

    # Community prior from theta_obs (mean across elections per community)
    if THETA_OBS_PATH.exists():
        theta_obs_df = pd.read_parquet(THETA_OBS_PATH)
        mu_prior = theta_obs_df.mean(axis=1).values
        log.info("Using theta_obs prior: %s", mu_prior.round(3))
    else:
        log.warning("No theta_obs found; using flat 0.45 prior")
        mu_prior = np.full(K, 0.45)

    # Average polls by (state, race) to get one observation per race
    poll_agg = (
        polls.groupby(["state", "race"])
        .agg(
            dem_share=("dem_share", "mean"),
            n_sample=("n_sample", "sum"),
        )
        .reset_index()
    )

    predictions = []
    for _, row in poll_agg.iterrows():
        state = row["state"]
        race = row["race"]
        poll_avg = float(row["dem_share"])
        poll_n = float(row.get("n_sample", 1000))
        poll_sigma = np.sqrt(poll_avg * (1 - poll_avg) / poll_n)

        state_row = state_w[state_w["state_abbr"] == state]
        if state_row.empty:
            log.warning("No weight row for state %s, skipping", state)
            continue

        W_mat = state_row[weight_cols].values.astype(float)  # (1, K)
        mu_post, sigma_post = bayesian_update(
            mu_prior=mu_prior,
            sigma_prior=Sigma,
            W=W_mat,
            y=np.array([poll_avg]),
            sigma_polls=np.array([poll_sigma]),
        )

        # Back-project to county level
        state_fips = state_w[state_w["state_abbr"] == state]["state_fips"].values[0]
        state_counties = county_w[county_w["county_fips"].str.startswith(state_fips)]
        for _, county_row in state_counties.iterrows():
            comm_id = county_row["community_id"]
            k_idx = k_ids.index(comm_id)
            pred = float(mu_post[k_idx])
            std = float(np.sqrt(sigma_post[k_idx, k_idx]))
            predictions.append({
                "county_fips": county_row["county_fips"],
                "state_abbr": state,
                "race": race,
                "pred_dem_share": pred,
                "pred_std": std,
                "pred_lo90": pred - 1.645 * std,
                "pred_hi90": pred + 1.645 * std,
                "state_pred": float((W_mat @ mu_post).squeeze()),
                "poll_avg": poll_avg,
            })

    result = pd.DataFrame(predictions)
    out_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_hac.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    log.info("Saved %d predictions to %s", len(result), out_path)
    print(result.groupby(["state_abbr", "race"])["pred_dem_share"].describe().round(3))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
