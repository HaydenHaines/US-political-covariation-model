"""
Stage 4: Poll propagation via analytical Bayesian Gaussian update.

Given:
  - Prior on community vote shares: θ ~ N(μ_prior, Σ_prior)  [from Stage 3]
  - Poll observations: y_p ≈ W_p · θ + ε_p  [spectral unmixing model]
    where W_p is the community weight vector for poll p's geography
    and ε_p ~ N(0, σ_p²) is sampling noise (known from poll N)

The posterior is analytic (Bayesian linear regression with Gaussian prior):
  Σ_post⁻¹ = Σ_prior⁻¹ + Wᵀ R⁻¹ W
  μ_post   = Σ_post (Σ_prior⁻¹ μ_prior + Wᵀ R⁻¹ y)

where R = diag(σ_p²) is the diagonal poll noise matrix.

This is the Kalman filter measurement update applied to the community vector.
Multiple polls from different geographies stack naturally: each adds a row to W
and an entry to y and R. The information accumulates additively.

This approach is exact under the Gaussian assumptions and requires no MCMC.
It serves as the MVP propagation engine. The full Stage 4 model (reverse random
walk + poll bias corrections + time-varying state) wraps this core update.

Inputs:
  data/covariance/covariance_summary.parquet  [Stage 3 posterior]
  data/covariance/community_sigma.parquet     [Stage 3 covariance]
  data/propagation/community_weights_state.parquet
  data/propagation/community_weights_county.parquet
  poll observations (provided as function arguments or from data/polls/)

Outputs:
  Posterior community vote share estimates (μ_post, Σ_post)
  Can be serialized to data/predictions/community_posterior_[date].parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]

COMP_COLS = [f"c{k}" for k in range(1, 8)]
K = 7

LABELS = {
    "c1": "White rural homeowner",
    "c2": "Black urban",
    "c3": "Knowledge worker",
    "c4": "Asian",
    "c5": "Working-class homeowner",
    "c6": "Hispanic low-income",
    "c7": "Generic suburban baseline",
}


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class PollObservation:
    """
    A single poll observation for spectral unmixing.

    geography:   identifier (e.g. "FL", "FL_Miami-Dade", "GA")
    dem_share:   observed Democratic two-party vote share (0-1)
    n_sample:    effective sample size for this geography/race
    race:        election race being polled (for logging)
    date:        poll date string (for logging)
    pollster:    polling firm (for logging)
    geo_level:   "state" | "county" | "district"
    metadata:    optional original poll fields to preserve through weighting
    """
    geography: str
    dem_share: float
    n_sample: int
    race: str = ""
    date: str = ""
    pollster: str = ""
    geo_level: str = "state"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def sigma(self) -> float:
        """Sampling noise: SE of a proportion estimate."""
        p = self.dem_share
        return np.sqrt(p * (1 - p) / self.n_sample)

    def __repr__(self) -> str:
        return (f"Poll({self.geography} {self.race} {self.date}: "
                f"{self.dem_share:.1%} ± {self.sigma:.1%}, N={self.n_sample})")


@dataclass
class CommunityPosterior:
    """
    Posterior distribution over community vote shares after poll updates.

    mu:    (K,) posterior mean vector
    sigma: (K, K) posterior covariance matrix
    comps: list of component names (c1..c7)
    """
    mu: np.ndarray
    sigma: np.ndarray
    comps: list[str]

    def credible_interval(self, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) credible interval bounds at given level."""
        from scipy import stats
        z = stats.norm.ppf(0.5 + level / 2)
        std = np.sqrt(np.diag(self.sigma))
        return self.mu - z * std, self.mu + z * std

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a tidy DataFrame with mu, std, and CI columns."""
        std = np.sqrt(np.diag(self.sigma))
        lo90, hi90 = self.credible_interval(0.90)
        return pd.DataFrame({
            "component": self.comps,
            "label": [LABELS[c] for c in self.comps],
            "mu_post": self.mu,
            "std_post": std,
            "lo90": lo90,
            "hi90": hi90,
        })


# ── Prior loading ─────────────────────────────────────────────────────────────


def load_prior(
    state: Optional[str] = None,
    year: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Stage 3 posterior as prior for Stage 4.

    If state is given (e.g. "FL"), uses state-stratified community vote shares
    as the prior mean.  By default uses the most recent year available in
    community_vote_shares_by_state.parquet; pass year= to pin a specific cycle.

    If state is None, uses the pooled Stan posterior mean (appropriate for
    region-level inference but not state-level prediction).

    The covariance Sigma is always the pooled Stage 3 estimate — it captures
    cross-community comovement structure that is not state-specific.

    Returns (mu_prior, Sigma_prior):
      mu_prior:    (K,) community baseline vote shares
      Sigma_prior: (K, K) community covariance matrix
    """
    sigma_path = PROJECT_ROOT / "data" / "covariance" / "community_sigma.parquet"
    Sigma_prior = pd.read_parquet(sigma_path).values

    if state is not None:
        state_path = PROJECT_ROOT / "data" / "covariance" / "community_vote_shares_by_state.parquet"
        state_df = pd.read_parquet(state_path)
        state_rows = state_df[state_df["state"] == state]

        # Use requested year, or fall back to the most recent available
        if year is not None:
            state_yr = state_rows[state_rows["year"] == year]
            if len(state_yr) == 0:
                raise ValueError(f"No year={year} state-stratified data for state={state}")
        else:
            latest = int(state_rows["year"].max())
            state_yr = state_rows[state_rows["year"] == latest]
            year = latest

        state_yr = state_yr.sort_values("component")
        mu_prior = state_yr["theta_direct"].values
        log.info("State-stratified prior loaded for %s (year=%d): mu = %s",
                 state, year, np.round(mu_prior, 3))
    else:
        # Pooled prior: use Stan posterior means
        summary_path = PROJECT_ROOT / "data" / "covariance" / "covariance_summary.parquet"
        summary = pd.read_parquet(summary_path)
        mu_rows = summary[summary.index.str.startswith("mu[")]
        mu_prior = mu_rows["Mean"].values
        log.info("Pooled prior loaded: mu = %s", np.round(mu_prior, 3))

    log.info("Sigma diagonal: %s", np.round(np.diag(Sigma_prior), 5))
    return mu_prior, Sigma_prior


# ── Weight matrix loading ─────────────────────────────────────────────────────


def load_weight_vector(geography: str, geo_level: str) -> np.ndarray:
    """
    Load the community weight vector W for a given geography.

    Returns (K,) array of community weights that sum to 1.
    W[k] = fraction of this geography's voting population in community k.
    """
    if geo_level == "state":
        weights_df = pd.read_parquet(
            PROJECT_ROOT / "data" / "propagation" / "community_weights_state.parquet"
        )
        row = weights_df[weights_df["state_abbr"] == geography]
    elif geo_level == "county":
        weights_df = pd.read_parquet(
            PROJECT_ROOT / "data" / "propagation" / "community_weights_county.parquet"
        )
        row = weights_df[weights_df["county_fips"] == geography]
    else:
        raise ValueError(f"Unsupported geo_level: {geo_level}")

    if len(row) == 0:
        raise ValueError(f"Geography '{geography}' not found in {geo_level} weights")

    W = row[COMP_COLS].values[0]
    return W.astype(float)


# ── Bayesian Gaussian update (Kalman filter measurement step) ─────────────────


def bayesian_poll_update(
    mu_prior: np.ndarray,
    Sigma_prior: np.ndarray,
    polls: list[PollObservation],
    weight_lookup: Optional[dict[str, np.ndarray]] = None,
) -> CommunityPosterior:
    """
    Analytical Bayesian update: incorporate poll observations into community estimates.

    Implements the Kalman filter measurement update:
      Σ_post⁻¹ = Σ_prior⁻¹ + Wᵀ R⁻¹ W
      μ_post   = Σ_post (Σ_prior⁻¹ μ_prior + Wᵀ R⁻¹ y)

    where:
      W is (n_polls × K) matrix of community weight vectors
      R is (n_polls × n_polls) diagonal matrix of poll noise variances
      y is (n_polls,) vector of observed Democrat shares

    This is exact under the Gaussian prior + Gaussian observation noise assumptions.

    Args:
      mu_prior:    (K,) prior mean vector
      Sigma_prior: (K, K) prior covariance matrix
      polls:       list of PollObservation instances
      weight_lookup: optional pre-loaded {geography: weight_vector} dict
                     (if None, loads from disk for each poll)

    Returns:
      CommunityPosterior with mu_post and Sigma_post
    """
    if not polls:
        log.info("No polls provided — returning prior unchanged")
        return CommunityPosterior(mu=mu_prior.copy(), sigma=Sigma_prior.copy(), comps=COMP_COLS)

    n_polls = len(polls)
    log.info("Updating with %d poll observations:", n_polls)

    # Build W matrix (n_polls × K) and observation vectors
    W = np.zeros((n_polls, K))
    y = np.zeros(n_polls)
    sigma_p = np.zeros(n_polls)

    for i, poll in enumerate(polls):
        log.info("  [%d] %s", i + 1, poll)
        if weight_lookup and poll.geography in weight_lookup:
            W[i] = weight_lookup[poll.geography]
        else:
            W[i] = load_weight_vector(poll.geography, poll.geo_level)
        y[i] = poll.dem_share
        sigma_p[i] = poll.sigma

    # R = diagonal poll noise matrix
    R_inv = np.diag(1.0 / sigma_p ** 2)

    # Precision update (information form)
    Sigma_prior_inv = np.linalg.inv(Sigma_prior)
    Sigma_post_inv = Sigma_prior_inv + W.T @ R_inv @ W
    Sigma_post = np.linalg.inv(Sigma_post_inv)

    # Mean update
    mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + W.T @ R_inv @ y)

    # Verify posterior is narrower than prior
    prior_std = np.sqrt(np.diag(Sigma_prior))
    post_std = np.sqrt(np.diag(Sigma_post))
    info_gain = 1 - (post_std / prior_std)
    log.info("Information gain per community (uncertainty reduction):")
    for k, comp in enumerate(COMP_COLS):
        log.info("  %s: %.1f%% tighter", comp, info_gain[k] * 100)

    return CommunityPosterior(mu=mu_post, sigma=Sigma_post, comps=COMP_COLS)


# ── Validation: roundtrip test ─────────────────────────────────────────────────


def validate_2020_roundtrip_state(state: str, Sigma_prior: np.ndarray) -> CommunityPosterior:
    """
    Validation test: use actual 2020 state results as "perfect polls"
    (N=∞ for each state) and verify the posterior recovers the known
    community vote shares from Stage 3.

    If the model is correctly specified, propagating the true election
    results should pull the posterior toward the known community estimates.
    """
    """
    Validate roundtrip for a single state: use actual 2020 state result as a
    "perfect poll" and verify the posterior is consistent with the state-stratified
    2020 community estimates. With state-stratified prior, the update should be
    small (the prior already reproduces the state result).
    """
    # 2020 actual two-party Democratic vote shares by state (MEDSL):
    actual_2020 = {"FL": 0.483, "GA": 0.501, "AL": 0.371}
    actual_n = {"FL": 11_067_456, "GA": 4_999_960, "AL": 2_323_282}

    mu_prior, _ = load_prior(state=state)

    perfect_poll = PollObservation(
        state, dem_share=actual_2020[state], n_sample=actual_n[state],
        race="2020 President", date="2020-11-03",
        pollster="Actual Result", geo_level="state",
    )
    posterior = bayesian_poll_update(mu_prior, Sigma_prior, [perfect_poll])
    df = posterior.to_dataframe()

    print(f"\n--- {state} 2020 roundtrip (state-stratified prior) ---")
    print(f"  State-level prediction from prior: "
          f"{sum(mu_prior[k] * load_weight_vector(state,'state')[k] for k in range(K)):.1%}  "
          f"(actual: {actual_2020[state]:.1%})")
    print(f"{'Community':<28}  {'Prior':>8}  {'Posterior':>10}  {'90% CI':>16}")
    print("-" * 65)
    for k, (_, row) in enumerate(df.iterrows()):
        shift = row['mu_post'] - mu_prior[k]
        shift_str = f" ({shift:+.1%})" if abs(shift) > 0.001 else ""
        ci_str = f"[{row['lo90']:.1%}, {row['hi90']:.1%}]"
        print(f"  {row['label']:<26}  {mu_prior[k]:.1%}  {row['mu_post']:.1%}{shift_str:>10}  {ci_str}")

    return posterior


# ── Plots ──────────────────────────────────────────────────────────────────────


def plot_posterior_comparison(
    mu_prior: np.ndarray,
    Sigma_prior: np.ndarray,
    posterior: CommunityPosterior,
    title: str,
    path: Path,
) -> None:
    """Bar chart comparing prior vs. posterior community vote shares with CI."""
    df = posterior.to_dataframe()
    prior_std = np.sqrt(np.diag(Sigma_prior))

    n = len(COMP_COLS)
    order = np.argsort(posterior.mu)
    labels = [df.iloc[i]["label"] for i in order]
    post_mu = posterior.mu[order]
    post_std = np.sqrt(np.diag(posterior.sigma))[order]
    prior_mu = mu_prior[order]
    prior_std_ord = prior_std[order]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(n)
    width = 0.35

    ax.barh(y + width/2, prior_mu, width, color="#94a3b8",
            alpha=0.7, label="Prior (Stage 3)", xerr=1.645*prior_std_ord,
            error_kw={"ecolor": "#64748b", "capsize": 3})
    ax.barh(y - width/2, post_mu, width, color="#3b82f6",
            alpha=0.85, label="Posterior (after polls)", xerr=1.645*post_std,
            error_kw={"ecolor": "#1d4ed8", "capsize": 3})

    ax.axvline(0.5, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Estimated Democratic vote share", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.2, 0.85)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log.info("Saved → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, Sigma_prior = load_prior(state=None)  # covariance is always pooled

    print("\n" + "=" * 70)
    print("Stage 4: Community vote share propagation")
    print("=" * 70)

    # ── Validation test: 2020 roundtrip per state ─────────────────────────────
    print("\n=== 2020 roundtrip validation (state-stratified priors) ===")
    print("With state-stratified priors, the prior already reproduces each state's result.")
    print("Propagating the actual result as a poll should produce minimal shifts.")

    for state in ["FL", "GA", "AL"]:
        validate_2020_roundtrip_state(state, Sigma_prior)

    # ── Demo: propagate synthetic early-cycle 2026 FL polls ───────────────────
    print("\n" + "=" * 70)
    print("Demo: propagate hypothetical 2026 FL Senate polls")
    print("(Synthetic polls — replace with real polls from data/polls/ when available)")
    print("=" * 70)

    mu_fl_2020, _ = load_prior(state="FL")
    std_fl = np.sqrt(np.diag(Sigma_prior))
    print("\nFL 2020 community baselines (state-stratified prior):")
    for k, comp in enumerate(COMP_COLS):
        print(f"  {comp} ({LABELS[comp]:<28}): {mu_fl_2020[k]:.1%} ± {std_fl[k]:.1%} (1σ)")

    demo_polls = [
        PollObservation("FL", dem_share=0.46, n_sample=800,
                        race="2026 FL Senate", date="2026-03-01",
                        pollster="Demo Pollster", geo_level="state"),
    ]

    posterior_demo = bayesian_poll_update(mu_fl_2020, Sigma_prior, demo_polls)
    df_demo = posterior_demo.to_dataframe()

    print(f"\n{'Community':<28}  {'2020 Prior':>10}  {'After Poll':>10}  {'Shift':>8}  {'90% CI':>16}")
    print("-" * 78)
    for k, (_, row) in enumerate(df_demo.iterrows()):
        shift = row['mu_post'] - mu_fl_2020[k]
        shift_str = f"{shift:+.1%}"
        ci_str = f"[{row['lo90']:.1%}, {row['hi90']:.1%}]"
        print(f"  {row['label']:<26}  {mu_fl_2020[k]:.1%}  {row['mu_post']:.1%}  "
              f"{shift_str:>8}  {ci_str}")

    plot_posterior_comparison(
        mu_fl_2020, Sigma_prior, posterior_demo,
        title="FL community vote shares: 2020 state-stratified prior vs. hypothetical 2026 FL Senate poll\n"
              "(Demo only — replace with real 2026 polls from data/polls/)",
        path=output_dir / "community_posterior_2026_demo.png",
    )

    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
