"""GMM vs KMeans comparison experiment (P3.1).

Compares Gaussian Mixture Models with KMeans for type discovery.
GMM gives proper probabilistic soft membership (vs temperature-scaled
inverse distance). Tests full and diagonal covariance types.

Usage:
    uv run python scripts/experiment_gmm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_data(min_year: int = 2008) -> tuple[np.ndarray, pd.DataFrame, list[str], list[str]]:
    """Load shift matrix and identify training/holdout columns."""
    shifts_df = pd.read_parquet(PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet")
    all_cols = [c for c in shifts_df.columns if c != "county_fips"]

    # Presidential weighting: 2.5x
    pres_cols = [c for c in all_cols if "pres_" in c]
    training_cols = [c for c in all_cols if c not in [
        "pres_d_shift_20_24", "pres_r_shift_20_24", "pres_turnout_shift_20_24"
    ]]
    holdout_cols = ["pres_d_shift_20_24", "pres_r_shift_20_24", "pres_turnout_shift_20_24"]

    # Filter to min_year
    def get_start_year(col: str) -> int:
        parts = col.split("_")
        for p in parts:
            if len(p) == 2 and p.isdigit():
                y = int(p)
                return 2000 + y if y < 50 else 1900 + y
        return 2000

    training_cols = [c for c in training_cols if get_start_year(c) >= min_year]

    # Build weighted training matrix (presidential x2.5)
    X_train = shifts_df[training_cols].values.copy()
    for i, col in enumerate(training_cols):
        if "pres_" in col:
            X_train[:, i] *= 2.5

    holdout_matrix = shifts_df[holdout_cols].values

    return X_train, shifts_df, training_cols, holdout_cols


def compute_holdout_r(
    scores: np.ndarray,
    holdout_matrix: np.ndarray,
) -> float:
    """Compute holdout accuracy (type-mean approach) — Pearson r."""
    dominant = np.argmax(np.abs(scores), axis=1)
    J = scores.shape[1]

    all_r = []
    for dim in range(holdout_matrix.shape[1]):
        values = holdout_matrix[:, dim]
        type_means = np.array([
            values[dominant == t].mean() if (dominant == t).sum() > 0 else 0.0
            for t in range(J)
        ])
        pred = type_means[dominant]
        r, _ = pearsonr(values, pred)
        all_r.append(r)
    return float(np.mean(all_r))


def compute_holdout_r_county_prior(
    scores: np.ndarray,
    training_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
) -> float:
    """Compute holdout accuracy using county-level priors."""
    dominant = np.argmax(np.abs(scores), axis=1)
    J = scores.shape[1]
    N = scores.shape[0]
    abs_scores = np.abs(scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)

    all_r = []
    for dim in range(holdout_matrix.shape[1]):
        actual = holdout_matrix[:, dim]
        # County prior: mean of training dims for this county
        county_prior = training_matrix.mean(axis=1)

        # Type means from training
        type_train_means = np.array([
            training_matrix[dominant == t].mean() if (dominant == t).sum() > 0 else 0.0
            for t in range(J)
        ])
        # Type holdout means
        type_holdout_means = np.array([
            actual[dominant == t].mean() if (dominant == t).sum() > 0 else 0.0
            for t in range(J)
        ])
        type_shift = type_holdout_means - type_train_means

        # County prediction
        weighted_shift = (abs_scores * type_shift[None, :]).sum(axis=1) / weight_sums
        pred = county_prior + weighted_shift

        r, _ = pearsonr(actual, pred)
        all_r.append(r)
    return float(np.mean(all_r))


def compute_calibration_mae(
    scores: np.ndarray,
    type_priors: np.ndarray,
    actual_dem_share: np.ndarray,
) -> float:
    """Calibration MAE from soft-membership weighted predictions."""
    abs_scores = np.abs(scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    pred = (abs_scores * type_priors[None, :]).sum(axis=1) / weight_sums
    return float(np.mean(np.abs(pred - actual_dem_share)))


def run_kmeans(X: np.ndarray, J: int, T: float = 10.0) -> np.ndarray:
    """Run KMeans and return temperature-scaled soft scores."""
    km = KMeans(n_clusters=J, random_state=42, n_init=10)
    km.fit(X)
    dists = np.zeros((len(X), J))
    for j in range(J):
        dists[:, j] = np.linalg.norm(X - km.cluster_centers_[j], axis=1)

    # Temperature-scaled inverse distance
    eps = 1e-10
    log_weights = -T * np.log(dists + eps)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return powered / row_sums


def run_gmm(X: np.ndarray, J: int, cov_type: str = "full") -> np.ndarray:
    """Run GMM and return posterior membership probabilities."""
    gmm = GaussianMixture(
        n_components=J,
        covariance_type=cov_type,
        random_state=42,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(X)
    probs = gmm.predict_proba(X)  # (N, J) — proper Bayesian soft membership
    return probs


def main() -> None:
    print("=" * 70)
    print("GMM vs KMeans Experiment (P3.1)")
    print("=" * 70)

    X_train, shifts_df, training_cols, holdout_cols = load_data(min_year=2008)
    holdout_matrix = shifts_df[holdout_cols].values
    # For county-prior computation, use raw (unweighted) training matrix
    raw_training = shifts_df[training_cols].values

    N, D = X_train.shape
    print(f"\nData: {N} counties × {D} training dims (presidential×2.5 weighted)")
    print(f"Holdout: {len(holdout_cols)} dims\n")

    results = []

    for J in [20, 30, 43, 50]:
        print(f"\n--- J = {J} ---")

        # KMeans with T=10
        scores_km = run_kmeans(X_train, J, T=10.0)
        r_km = compute_holdout_r(scores_km, holdout_matrix)
        r_km_cp = compute_holdout_r_county_prior(scores_km, raw_training, holdout_matrix)

        print(f"  KMeans T=10:   holdout_r={r_km:.4f}  county_prior_r={r_km_cp:.4f}")
        results.append({"method": "KMeans T=10", "J": J, "holdout_r": r_km, "county_prior_r": r_km_cp})

        # GMM full covariance
        try:
            scores_gmm_full = run_gmm(X_train, J, cov_type="full")
            r_gmm_full = compute_holdout_r(scores_gmm_full, holdout_matrix)
            r_gmm_full_cp = compute_holdout_r_county_prior(scores_gmm_full, raw_training, holdout_matrix)
            print(f"  GMM full:      holdout_r={r_gmm_full:.4f}  county_prior_r={r_gmm_full_cp:.4f}")
            results.append({"method": "GMM full", "J": J, "holdout_r": r_gmm_full, "county_prior_r": r_gmm_full_cp})
        except Exception as e:
            print(f"  GMM full:      FAILED ({e})")

        # GMM diagonal covariance
        scores_gmm_diag = run_gmm(X_train, J, cov_type="diag")
        r_gmm_diag = compute_holdout_r(scores_gmm_diag, holdout_matrix)
        r_gmm_diag_cp = compute_holdout_r_county_prior(scores_gmm_diag, raw_training, holdout_matrix)
        print(f"  GMM diag:      holdout_r={r_gmm_diag:.4f}  county_prior_r={r_gmm_diag_cp:.4f}")
        results.append({"method": "GMM diag", "J": J, "holdout_r": r_gmm_diag, "county_prior_r": r_gmm_diag_cp})

        # GMM spherical (equivalent to KMeans generative model)
        scores_gmm_sph = run_gmm(X_train, J, cov_type="spherical")
        r_gmm_sph = compute_holdout_r(scores_gmm_sph, holdout_matrix)
        r_gmm_sph_cp = compute_holdout_r_county_prior(scores_gmm_sph, raw_training, holdout_matrix)
        print(f"  GMM spherical: holdout_r={r_gmm_sph:.4f}  county_prior_r={r_gmm_sph_cp:.4f}")
        results.append({"method": "GMM spherical", "J": J, "holdout_r": r_gmm_sph, "county_prior_r": r_gmm_sph_cp})

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    df = pd.DataFrame(results)
    # Pivot for readability
    for metric in ["holdout_r", "county_prior_r"]:
        print(f"\n{metric}:")
        pivot = df.pivot(index="method", columns="J", values=metric)
        print(pivot.to_string(float_format="%.4f"))

    # Save results
    out_path = PROJECT_ROOT / "data" / "validation" / "gmm_experiment_results.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
