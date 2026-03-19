"""County-level Stan covariance estimation (Phase 1 pipeline).

Uses Ward HAC community assignments + assembled election parquets to
build theta_obs[K, T] (community vote shares per election), then runs
the Stan factor model to estimate the K×K community covariance matrix Σ.

Key design:
  theta_obs[k, t] = population-weighted mean dem_share for counties in
                    community k, election t.
  k_ref = 1-indexed community with highest mean dem_share (most Democratic
          community). Used to fix the sign of the factor loadings.

Inputs:
  data/communities/county_community_assignments.parquet
  data/assembled/medsl_county_presidential_YYYY.parquet (2016, 2020)
  data/assembled/algara_county_governor_2018.parquet
  data/assembled/medsl_county_2022_governor.parquet
  data/assembled/medsl_county_2024_president.parquet

Outputs:
  data/covariance/county_community_sigma.parquet   — K×K posterior mean Σ
  data/covariance/county_community_rho.parquet     — K×K posterior mean correlation
  data/covariance/county_covariance_summary.csv    — posterior summary
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"
COVARIANCE_DIR = PROJECT_ROOT / "data" / "covariance"
STAN_MODEL = PROJECT_ROOT / "src" / "covariance" / "stan" / "community_covariance.stan"

# Elections used for covariance estimation (label, dem_share_col, total_col, parquet)
_ELECTIONS = [
    ("pres_2016", "pres_dem_share_2016", "pres_total_2016",
     "medsl_county_presidential_2016.parquet"),
    ("gov_2018", "gov_dem_share_2018", "gov_total_2018",
     "algara_county_governor_2018.parquet"),
    ("pres_2020", "pres_dem_share_2020", "pres_total_2020",
     "medsl_county_presidential_2020.parquet"),
    ("gov_2022", "gov_dem_share_2022", "gov_total_2022",
     "medsl_county_2022_governor.parquet"),
    ("pres_2024", "pres_dem_share_2024", "pres_total_2024",
     "medsl_county_2024_president.parquet"),
]


def load_election(parquet_name: str, share_col: str, total_col: str) -> pd.DataFrame:
    """Load an election parquet and return [county_fips, dem_share, total_votes]."""
    path = ASSEMBLED_DIR / parquet_name
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    return df[["county_fips"]].assign(
        dem_share=df[share_col],
        total_votes=df[total_col],
    )


def compute_theta_obs(
    assignments: pd.DataFrame,
    elections: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute community vote shares and standard errors across elections.

    Parameters
    ----------
    assignments:
        DataFrame with county_fips and community_id (0-indexed).
    elections:
        List of election DataFrames, each with county_fips, dem_share, total_votes.

    Returns
    -------
    theta_obs: (K, T) population-weighted community dem_share per election
    theta_se:  (K, T) Kish effective-n standard error per community-election
    obs_mask:  (K, T) 1.0 if observed, 0.0 if missing (all counties NaN)
    """
    k_ids = sorted(assignments["community_id"].unique())
    K = len(k_ids)
    T = len(elections)

    theta_obs = np.full((K, T), np.nan)
    theta_se = np.full((K, T), 0.05)  # fallback SE
    obs_mask = np.zeros((K, T))

    for t, elec_df in enumerate(elections):
        merged = assignments.merge(elec_df, on="county_fips", how="left")
        for k_idx, k_id in enumerate(k_ids):
            mask = merged["community_id"] == k_id
            sub = merged[mask].dropna(subset=["dem_share", "total_votes"])
            if len(sub) == 0:
                obs_mask[k_idx, t] = 0.0
                continue

            w = sub["total_votes"].values
            p = sub["dem_share"].values

            w_sum = w.sum()
            if w_sum == 0:
                obs_mask[k_idx, t] = 0.0
                continue

            theta = float((p * w).sum() / w_sum)
            theta_obs[k_idx, t] = theta
            obs_mask[k_idx, t] = 1.0

            # Kish effective N standard error
            w_sq_sum = (w ** 2).sum()
            n_eff = w_sum ** 2 / w_sq_sum if w_sq_sum > 0 else 1.0
            se = float(np.sqrt(theta * (1 - theta) / max(n_eff, 1)))
            theta_se[k_idx, t] = max(se, 0.001)  # floor at 0.1%

    return theta_obs, theta_se, obs_mask


def identify_k_ref(theta_obs: np.ndarray) -> int:
    """Return 1-indexed Stan k_ref = community with highest mean dem_share."""
    mean_shares = np.nanmean(theta_obs, axis=1)
    return int(np.argmax(mean_shares)) + 1  # Stan is 1-indexed


def run(
    assignments_path=None,
    output_dir=None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = 42,
) -> Path:
    """Run the county-level Stan covariance model and return sigma parquet path."""
    import cmdstanpy

    assignments_path = Path(assignments_path or COMMUNITIES_DIR / "county_community_assignments.parquet")
    output_dir = Path(output_dir or COVARIANCE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading community assignments from %s", assignments_path)
    assignments = pd.read_parquet(assignments_path)
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
    if "community_id" not in assignments.columns and "community" in assignments.columns:
        assignments = assignments.rename(columns={"community": "community_id"})
    K = assignments["community_id"].nunique()
    log.info("K = %d communities", K)

    log.info("Loading %d election cycles for theta_obs...", len(_ELECTIONS))
    elections = []
    for label, share_col, total_col, parquet in _ELECTIONS:
        try:
            elec = load_election(parquet, share_col, total_col)
            elections.append(elec)
            log.info("  Loaded %s (%d counties)", label, len(elec))
        except Exception as e:
            log.warning("  Could not load %s: %s", label, e)
    T = len(elections)
    log.info("T = %d elections for Stan", T)

    theta_obs, theta_se, obs_mask = compute_theta_obs(assignments, elections)

    # Save theta_obs for use by predict_2026_hac.py
    theta_obs_df = pd.DataFrame(
        theta_obs,
        index=sorted(assignments["community_id"].unique()),
        columns=[e[0] for e in _ELECTIONS[:T]],
    )
    theta_obs_df.to_parquet(output_dir / "county_theta_obs.parquet")
    log.info("Saved theta_obs to %s", output_dir / "county_theta_obs.parquet")

    k_ref = identify_k_ref(theta_obs)
    log.info("k_ref = %d (most Democratic community, 1-indexed)", k_ref)

    stan_data = {
        "K": K,
        "T": T,
        "k_ref": k_ref,
        "theta_obs": theta_obs.tolist(),
        "theta_se": theta_se.tolist(),
        "obs_mask": obs_mask.tolist(),
    }

    log.info("Compiling Stan model: %s", STAN_MODEL)
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_MODEL))
    log.info("Running MCMC: %d chains, %d warmup, %d sampling", chains, iter_warmup, iter_sampling)
    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        show_progress=True,
        output_dir=str(output_dir / "stan_draws_county"),
    )

    diag = fit.diagnose()
    log.info("Stan diagnostics:\n%s", diag)

    sigma_draws = fit.stan_variable("Sigma")  # (n_draws, K, K)
    sigma_mean = sigma_draws.mean(axis=0)
    rho_draws = fit.stan_variable("Rho")
    rho_mean = rho_draws.mean(axis=0)

    comm_ids = sorted(assignments["community_id"].unique())
    sigma_df = pd.DataFrame(sigma_mean, index=comm_ids, columns=comm_ids)
    sigma_path = output_dir / "county_community_sigma.parquet"
    sigma_df.to_parquet(sigma_path)
    log.info("Saved Σ to %s  (K=%d, T=%d)", sigma_path, K, T)

    rho_df = pd.DataFrame(rho_mean, index=comm_ids, columns=comm_ids)
    rho_df.to_parquet(output_dir / "county_community_rho.parquet")

    fit.summary().to_csv(output_dir / "county_covariance_summary.csv")

    print("\n=== Community Σ (posterior mean) ===")
    print(sigma_df.round(5).to_string())
    print("\n=== Community ρ (posterior mean correlation) ===")
    print(rho_df.round(3).to_string())

    return sigma_path


if __name__ == "__main__":
    run()
