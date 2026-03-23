"""Construct J×J type covariance matrix from demographic profiles.

Follows the Economist 2020 presidential model approach (Heidemanns/Gelman/Morris):
construct covariance from demographic similarity rather than estimating it from
too-few election observations.

Pipeline:
  1. Min-max scale type demographic profiles → [0, 1] per feature
  2. Pearson correlation across types → J×J matrix C
  3. Floor negative correlations (optional)
  4. Shrink toward all-1s (national swing component): C_final = lam*C + (1-lam)*1
  5. Enforce positive definiteness via spectral truncation
  6. Scale to covariance: Sigma = sigma_base^2 * C_final

Outputs:
  data/covariance/type_correlation.parquet   — J×J correlation matrix
  data/covariance/type_covariance.parquet    — J×J covariance matrix
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"
COVARIANCE_DIR = PROJECT_ROOT / "data" / "covariance"
CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"


@dataclass
class CovarianceResult:
    correlation_matrix: np.ndarray  # J × J constructed correlation
    covariance_matrix: np.ndarray   # J × J scaled covariance
    validation_r: float             # off-diagonal correlation vs observed
    used_hybrid: bool               # whether hybrid fallback was triggered
    sigma_base: float               # base logit-scale sigma


def construct_type_covariance(
    type_profiles: pd.DataFrame,
    feature_columns: list[str],
    lambda_shrinkage: float = 0.75,
    sigma_base: float = 0.07,
    floor_negatives: bool = True,
    max_rank: int | None = None,
) -> CovarianceResult:
    """Construct type covariance matrix from demographic profiles.

    Parameters
    ----------
    type_profiles:
        J rows × demographic columns (from describe_types).
    feature_columns:
        Which columns to use for computing inter-type correlation.
    lambda_shrinkage:
        Weight on the demographic Pearson correlation (vs national-swing all-1s).
        0 → pure national swing (all correlations = 1).
        1 → pure demographic correlation.
    sigma_base:
        Base logit-scale standard deviation; covariance = sigma_base^2 * C.
    floor_negatives:
        If True, floor raw Pearson correlations at 0 before shrinkage.
    max_rank:
        If set, reduce the effective rank of the correlation matrix to this
        value via eigendecomposition (keep top-k eigencomponents).  Matches
        the observed dimensionality of electoral comovement (~18 from 19
        elections) rather than the full demographic feature count.
    """
    J = len(type_profiles)

    # 1. Extract feature matrix: J × F
    X = type_profiles[feature_columns].values.astype(float)

    # 2. Min-max scale each feature to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # avoid division by zero for constant features
    X_scaled = (X - X_min) / X_range

    # 3. Pearson correlation across types (types are rows)
    C = np.corrcoef(X_scaled)  # J × J

    # Guard against NaN from constant rows (all values identical after scaling)
    C = np.where(np.isnan(C), 0.0, C)
    np.fill_diagonal(C, 1.0)

    # 4. Floor negative correlations (configurable)
    if floor_negatives:
        C = np.maximum(C, 0.0)

    # 5. Shrink toward all-1s (national swing component)
    C_ones = np.ones((J, J))
    C_final = lambda_shrinkage * C + (1 - lambda_shrinkage) * C_ones

    # 6. Rank reduction or PD enforcement
    if max_rank is not None and max_rank < J:
        C_final = _rank_reduce(C_final, max_rank)
        log.info("Rank-reduced correlation matrix to effective rank %d (J=%d)", max_rank, J)
    else:
        C_final = _enforce_pd(C_final)

    # 7. Create covariance (uniform sigma, modulated by correlation)
    Sigma = sigma_base**2 * C_final

    return CovarianceResult(
        correlation_matrix=C_final,
        covariance_matrix=Sigma,
        validation_r=np.nan,
        used_hybrid=False,
        sigma_base=sigma_base,
    )


def _enforce_pd(C: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Return nearest positive-definite matrix via spectral truncation."""
    # Symmetrise to eliminate floating-point asymmetry before decomposition
    C = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, floor)
    C_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-symmetrise after reconstruction
    return (C_pd + C_pd.T) / 2.0


def _rank_reduce(C: np.ndarray, k: int) -> np.ndarray:
    """Reduce effective rank of correlation matrix to k via eigendecomposition.

    Keeps the top-k eigenvalues and reconstructs.  Remaining eigenvalues are
    set to a small floor (1e-6) to maintain strict positive definiteness.
    Diagonal is re-normalised to 1.0 so the result remains a correlation matrix.
    """
    C = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)

    # eigh returns ascending order; reverse to descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep top-k, floor the rest
    floor = 1e-6
    eigvals_reduced = np.where(np.arange(len(eigvals)) < k, eigvals, floor)
    # Also floor any negative eigenvalues in top-k
    eigvals_reduced = np.maximum(eigvals_reduced, floor)

    C_reduced = eigvecs @ np.diag(eigvals_reduced) @ eigvecs.T
    C_reduced = (C_reduced + C_reduced.T) / 2.0

    # Re-normalise diagonal to 1.0 (correlation matrix convention)
    d = np.sqrt(np.diag(C_reduced))
    d[d == 0] = 1.0
    C_reduced = C_reduced / np.outer(d, d)

    return C_reduced


def validate_covariance(
    constructed: CovarianceResult,
    type_scores: np.ndarray,       # N × J county type scores
    shift_matrix: np.ndarray,      # N × D shift matrix
    election_col_groups: list[list[int]],  # column indices grouped by election
) -> float:
    """Validate constructed covariance against observed type comovement.

    For each election in ``election_col_groups``, compute type-level shifts as
    the score-weighted mean of county shifts, producing a J-dimensional vector
    per election.  Then estimate the sample covariance of these vectors across
    elections and compare its off-diagonal structure with the constructed
    correlation matrix.

    Returns
    -------
    float
        Pearson r between off-diagonal elements of the constructed correlation
        and those of the observed sample covariance (normalised to correlation),
        in [-1, 1].  Returns NaN when fewer than two elections are available.
    """
    J = type_scores.shape[1]
    T = len(election_col_groups)

    if T < 2:
        log.warning("validate_covariance: need >= 2 elections, got %d — returning NaN", T)
        return float("nan")

    # Compute type-level shift for each election: (T, J)
    type_shifts = np.zeros((T, J))
    for t, col_indices in enumerate(election_col_groups):
        election_shift = shift_matrix[:, col_indices].mean(axis=1)  # (N,)
        # Weighted mean: weight each county by its type score magnitude
        weights = np.abs(type_scores)  # (N, J)
        w_sum = weights.sum(axis=0) + 1e-12  # (J,)
        type_shifts[t] = (weights * election_shift[:, None]).sum(axis=0) / w_sum

    # Sample covariance of type-level shifts across elections: (J, J)
    obs_cov = np.cov(type_shifts.T)  # (J, J)

    # Normalise observed covariance to correlation for fair comparison
    obs_std = np.sqrt(np.diag(obs_cov))
    obs_std[obs_std == 0] = 1.0
    obs_corr = obs_cov / np.outer(obs_std, obs_std)
    np.fill_diagonal(obs_corr, 1.0)

    # Extract off-diagonal elements
    mask = ~np.eye(J, dtype=bool)
    constructed_off = constructed.correlation_matrix[mask]
    observed_off = obs_corr[mask]

    if np.std(constructed_off) < 1e-10 or np.std(observed_off) < 1e-10:
        return 0.0

    r = float(np.corrcoef(constructed_off, observed_off)[0, 1])
    return np.clip(r, -1.0, 1.0)


def apply_hybrid_fallback(
    constructed: CovarianceResult,
    observed_cov: np.ndarray,
    validation_r: float,
    threshold: float = 0.4,
) -> CovarianceResult:
    """Blend constructed with observed covariance if validation fails.

    If ``validation_r >= threshold``, returns the constructed result unchanged
    (with ``validation_r`` populated).  Otherwise blends the two matrices,
    weighting the constructed result by ``max(0.3, validation_r)``.
    """
    if validation_r >= threshold:
        return CovarianceResult(
            correlation_matrix=constructed.correlation_matrix,
            covariance_matrix=constructed.covariance_matrix,
            validation_r=validation_r,
            used_hybrid=False,
            sigma_base=constructed.sigma_base,
        )

    w = max(0.3, validation_r)
    C_hybrid = w * constructed.correlation_matrix + (1 - w) * observed_cov

    # Ensure positive definiteness
    C_hybrid = _enforce_pd(C_hybrid)

    Sigma = constructed.sigma_base**2 * C_hybrid

    return CovarianceResult(
        correlation_matrix=C_hybrid,
        covariance_matrix=Sigma,
        validation_r=validation_r,
        used_hybrid=True,
        sigma_base=constructed.sigma_base,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    try:
        import yaml  # type: ignore[import]
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.warning("Could not load config: %s — using defaults", e)
        return {}


def _load_type_profiles() -> tuple[pd.DataFrame, list[str]]:
    """Load type profiles from data/communities/type_profiles.parquet."""
    profiles_path = COMMUNITIES_DIR / "type_profiles.parquet"
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Type profiles not found at {profiles_path}. "
            "Run `python -m src.description.describe_types` first."
        )
    profiles = pd.read_parquet(profiles_path)
    # Use all numeric columns as feature columns (exclude any id/label cols)
    feature_cols = [
        c for c in profiles.columns
        if profiles[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        and c not in ("type_id", "type_label", "super_type_id")
    ]
    log.info("Loaded type profiles: %d types × %d features", len(profiles), len(feature_cols))
    return profiles, feature_cols


def _load_type_scores_and_shifts() -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Load type scores and shift matrix for validation."""
    # Type scores: county × type soft-membership matrix
    # Try multiple naming conventions for the type assignments file
    for name in ["type_assignments.parquet", "county_type_assignments.parquet"]:
        assignments_path = COMMUNITIES_DIR / name
        if assignments_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"County type assignments not found in {COMMUNITIES_DIR}. "
            "Run `python -m src.discovery.run_type_discovery` first."
        )

    assignments = pd.read_parquet(assignments_path)
    # Score columns: type_X_score pattern (from run_type_discovery.py)
    score_cols = [
        c for c in assignments.columns
        if c.endswith("_score") and c.startswith("type_")
    ]
    if not score_cols:
        # Assume all float columns except fips are scores
        score_cols = [
            c for c in assignments.columns
            if assignments[c].dtype in (np.float64, np.float32)
        ]
    type_scores = assignments[score_cols].values  # (N, J)
    log.info("Loaded type scores: %s", type_scores.shape)

    # Shift matrix: county × shift dimensions
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    if not shifts_path.exists():
        raise FileNotFoundError(
            f"Shift matrix not found at {shifts_path}. "
            "Run `python src/assembly/build_county_shifts_multiyear.py` first."
        )
    shifts_df = pd.read_parquet(shifts_path)
    shift_cols = [c for c in shifts_df.columns if c != "county_fips"]
    shift_matrix = shifts_df[shift_cols].values  # (N, D)
    log.info("Loaded shift matrix: %s", shift_matrix.shape)

    # Group shift columns by election pair (each pair occupies 3 dims: pres/gov/senate)
    D = shift_matrix.shape[1]
    group_size = 3
    election_col_groups = [
        list(range(i, min(i + group_size, D)))
        for i in range(0, D, group_size)
    ]
    return type_scores, shift_matrix, election_col_groups


def main() -> None:
    cfg = _load_config()
    types_cfg = cfg.get("types", {})
    lambda_shrinkage = float(types_cfg.get("lambda_shrinkage", 0.75))
    threshold = float(types_cfg.get("covariance_acceptance_threshold", 0.4))
    sigma_base = 0.07  # logit-scale base sigma
    max_rank_raw = types_cfg.get("covariance_max_rank", None)
    max_rank = int(max_rank_raw) if max_rank_raw is not None else None

    log.info("lambda_shrinkage=%.2f  threshold=%.2f  max_rank=%s", lambda_shrinkage, threshold, max_rank)

    type_profiles, feature_cols = _load_type_profiles()
    J = len(type_profiles)

    log.info("Constructing %d×%d type covariance matrix...", J, J)
    result = construct_type_covariance(
        type_profiles,
        feature_cols,
        lambda_shrinkage=lambda_shrinkage,
        sigma_base=sigma_base,
        max_rank=max_rank,
    )

    try:
        type_scores, shift_matrix, election_col_groups = _load_type_scores_and_shifts()
        validation_r = validate_covariance(result, type_scores, shift_matrix, election_col_groups)
        log.info("Validation r = %.3f (threshold = %.2f)", validation_r, threshold)

        # Observed covariance for hybrid fallback (normalised to correlation scale)
        result = apply_hybrid_fallback(result, np.eye(J), validation_r, threshold)
    except FileNotFoundError as e:
        log.warning("Skipping validation (data missing): %s", e)
        result.validation_r = float("nan")

    COVARIANCE_DIR.mkdir(parents=True, exist_ok=True)

    # Save correlation matrix
    corr_path = COVARIANCE_DIR / "type_correlation.parquet"
    pd.DataFrame(result.correlation_matrix).to_parquet(corr_path)
    log.info("Saved correlation matrix to %s", corr_path)

    # Save covariance matrix
    cov_path = COVARIANCE_DIR / "type_covariance.parquet"
    pd.DataFrame(result.covariance_matrix).to_parquet(cov_path)
    log.info("Saved covariance matrix to %s", cov_path)

    print("\n=== Type Covariance Construction Summary ===")
    print(f"  J (types):          {J}")
    print(f"  lambda_shrinkage:   {lambda_shrinkage}")
    print(f"  sigma_base:         {sigma_base}")
    print(f"  validation_r:       {result.validation_r:.3f}" if not np.isnan(result.validation_r) else "  validation_r:       N/A")
    print(f"  used_hybrid:        {result.used_hybrid}")
    print(f"  Correlation matrix saved to: {corr_path}")
    print(f"  Covariance matrix saved to:  {cov_path}")


if __name__ == "__main__":
    main()
