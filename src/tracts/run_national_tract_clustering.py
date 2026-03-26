"""National tract-level KMeans clustering.

Runs KMeans on the 84K-tract national feature matrix. Handles:
  - Population filtering: drop tracts with <500 votes in 2020
  - NaN imputation: fill with column median before clustering
  - J sweep with holdout validation
  - Super-type nesting
  - Saves results to data/tracts/national_tract_assignments.parquet

Usage:
    python -m src.tracts.run_national_tract_clustering [--j J] [--j-sweep]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Demographic features used for super-type nesting (Ward HAC on profiles).
# NOTE: centroids are in standardized shift space and produce degenerate
# clustering at J=100 — demographic profiles give meaningful separation.
NESTING_DEMO_FEATURES = [
    "pct_white_nh",
    "pct_black",
    "pct_hispanic",
    "pct_asian",
    "pct_ba_plus",
    "median_hh_income",
    "median_age",
    "evangelical_share",
    "catholic_share",
    "black_protestant_share",
    "religious_adherence_rate",
]

from src.discovery.nest_types import nest_types
from src.discovery.run_type_discovery import temperature_soft_membership

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Feature selection ─────────────────────────────────────────────────────────

# Training features: presidential shifts across 3 cycles (2008→2012, 2012→2016, 2016→2020).
# 2020→2024 is the holdout -- excluded from training.
# Gov shifts excluded: 54% coverage causes massive NaN imputation bias.
# Earlier presidential pairs (08→12, 12→16) have 97-99% coverage.
TRAINING_FEATURES = [
    # Presidential shifts (3 training pairs)
    "pres_d_shift_08_12",
    "pres_r_shift_08_12",
    "pres_turnout_shift_08_12",
    "pres_d_shift_12_16",
    "pres_r_shift_12_16",
    "pres_turnout_shift_12_16",
    "pres_d_shift_16_20",
    "pres_r_shift_16_20",
    "pres_turnout_shift_16_20",
    # Presidential levels
    "pres_dem_share_2016",
    "pres_dem_share_2020",
    # Turnout shift
    "turnout_shift_16_20",
]

# Holdout: 2020→2024 presidential shifts (excluded from training)
HOLDOUT_FEATURES = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

# Demographic features for combined-feature run
DEMOGRAPHIC_FEATURES = [
    "pct_white_nh",
    "pct_black",
    "pct_hispanic",
    "pct_asian",
    "pct_foreign_born",
    "pct_ba_plus",
    "pct_no_hs",
    "pct_wwc",
    "median_hh_income",
    "poverty_rate",
    "pct_owner_occupied",
    "median_home_value",
    "pct_multi_unit",
    "median_age",
    "pct_over_65",
    "pct_single_hh",
    "pct_wfh",
    "pct_no_vehicle",
    "pct_veteran",
    # Religion features (county-level RCMS mapped to tracts via FIPS prefix)
    "evangelical_share",
    "catholic_share",
    "black_protestant_share",
    "religious_adherence_rate",
]

POPULATION_MIN_VOTES = 500


def load_and_filter_features(
    features_path: Path,
    use_demographics: bool = False,
    presidential_weight: float = 2.5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load tract features, filter, impute NaN, scale.

    Returns
    -------
    df_filt : DataFrame
        Filtered (pop >=500) full feature table with GEOID.
    X_train : ndarray of shape (N, D_train)
        Scaled training feature matrix.
    X_holdout : ndarray of shape (N, D_holdout)
        Raw (unscaled) holdout matrix for validation.
    """
    df = pd.read_parquet(features_path)
    log.info("Loaded %d tracts × %d columns from %s", len(df), len(df.columns), features_path)

    # Population filter
    before = len(df)
    df = df[df["turnout_2020"] >= POPULATION_MIN_VOTES].copy()
    dropped = before - len(df)
    log.info(
        "Population filter (turnout_2020 >= %d): kept %d/%d tracts (dropped %d)",
        POPULATION_MIN_VOTES, len(df), before, dropped,
    )

    # Select training features
    train_cols = [c for c in TRAINING_FEATURES if c in df.columns]
    if use_demographics:
        demo_cols = [c for c in DEMOGRAPHIC_FEATURES if c in df.columns]
        train_cols = train_cols + demo_cols

    missing_train = [c for c in TRAINING_FEATURES if c not in df.columns]
    if missing_train:
        log.warning("Missing training columns: %s", missing_train)

    log.info("Training features: %d columns", len(train_cols))

    # NaN imputation via column median
    train_df = df[train_cols].copy()
    nan_counts = train_df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        log.info("Imputing NaN in %d columns via column median:", len(nan_cols))
        for col in nan_cols.index:
            med = train_df[col].median()
            n_filled = int(train_df[col].isnull().sum())
            train_df[col] = train_df[col].fillna(med)
            log.info("  %s: filled %d NaNs with median=%.4f", col, n_filled, med)

    # Standard scale first, then apply presidential weight
    # (Weight must be applied AFTER scaling to have any effect,
    # since StandardScaler normalizes away pre-scaling differences.)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.values)

    if presidential_weight != 1.0:
        pres_indices = [i for i, c in enumerate(train_cols) if "pres_" in c]
        X_train[:, pres_indices] *= presidential_weight
        log.info(
            "Applied presidential weight=%.1f to %d columns (post-scaling)",
            presidential_weight, len(pres_indices),
        )

    log.info("Scaled training matrix: %s", X_train.shape)

    # Holdout matrix (raw, for validation)
    holdout_cols = [c for c in HOLDOUT_FEATURES if c in df.columns]
    X_holdout = df[holdout_cols].values
    holdout_nan_count = np.isnan(X_holdout).sum()
    if holdout_nan_count > 0:
        log.info(
            "Holdout matrix has %d NaN entries (tracts without 2024 data) — "
            "excluded from validation metrics",
            holdout_nan_count,
        )

    return df.reset_index(drop=True), X_train, X_holdout


def compute_holdout_r(
    labels: np.ndarray, X_holdout: np.ndarray
) -> tuple[float, list[float]]:
    """Compute mean Pearson r between type means and actual holdout values.

    Tracts with any NaN in holdout are excluded from per-dimension r calc.
    """
    j = int(labels.max()) + 1
    per_dim_r: list[float] = []

    for dim_idx in range(X_holdout.shape[1]):
        col = X_holdout[:, dim_idx]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() < 10:
            continue
        valid_col = col[valid_mask]
        valid_labels = labels[valid_mask]

        # Predict via type mean on valid subset
        type_means = np.zeros(j)
        for t in range(j):
            mask = valid_labels == t
            if mask.sum() > 0:
                type_means[t] = valid_col[mask].mean()

        predicted = type_means[valid_labels]
        if np.std(valid_col) > 1e-10 and np.std(predicted) > 1e-10:
            r, _ = pearsonr(valid_col, predicted)
            per_dim_r.append(float(r))

    mean_r = float(np.mean(per_dim_r)) if per_dim_r else 0.0
    return mean_r, per_dim_r


def build_demographic_profiles(
    df_filt: pd.DataFrame,
    labels: np.ndarray,
    j: int,
) -> np.ndarray:
    """Compute mean demographic profiles per type for super-type nesting.

    Uses NESTING_DEMO_FEATURES averaged per KMeans label.  Missing columns
    are silently skipped.  NaN values are imputed with the column mean.

    Returns
    -------
    ndarray of shape (J, n_features)
        StandardScaler-normalised profile matrix ready for Ward HAC.
    """
    avail = [c for c in NESTING_DEMO_FEATURES if c in df_filt.columns]
    if not avail:
        log.warning(
            "No demographic nesting features found in df_filt. "
            "Falling back to shift-space centroids for nesting."
        )
        return None  # type: ignore[return-value]

    # Assign labels to filtered tracts
    demo_df = df_filt[avail].copy()
    demo_df["_label"] = labels

    # Mean per type (ordered by type id 0..J-1)
    profiles = demo_df.groupby("_label")[avail].mean()
    # Some types might be missing if labels are non-contiguous (shouldn't happen)
    profiles = profiles.reindex(range(j))

    feature_matrix = profiles.values.astype(float)

    # Impute any remaining NaN (rare: type with all-NaN feature)
    col_means = np.nanmean(feature_matrix, axis=0)
    nan_mask = np.isnan(feature_matrix)
    for col_idx in range(feature_matrix.shape[1]):
        feature_matrix[nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    # StandardScaler so all demographics are on equal footing
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    log.info(
        "Built demographic profile matrix: %d types × %d features (%s)",
        feature_matrix.shape[0], feature_matrix.shape[1], avail,
    )
    return feature_matrix


def run_j_sweep(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    j_candidates: list[int],
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[int, float, list[dict]]:
    """Sweep J candidates, return best J, best holdout r, and full results."""
    results: list[dict] = []
    best_j = j_candidates[0]
    best_r = -999.0

    for j in j_candidates:
        log.info("  Fitting KMeans J=%d ...", j)
        km = KMeans(n_clusters=j, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X_train)
        mean_r, per_dim = compute_holdout_r(labels, X_holdout)
        log.info("    J=%d: holdout r=%.4f (dims: %s)", j, mean_r, [f"{x:.3f}" for x in per_dim])

        results.append({"j": int(j), "holdout_r": float(mean_r), "per_dim_r": per_dim})
        if mean_r > best_r:
            best_r = mean_r
            best_j = j

    return best_j, best_r, results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="National tract KMeans clustering")
    parser.add_argument("--j", type=int, default=40, help="Number of types (default: 40)")
    parser.add_argument(
        "--j-sweep",
        action="store_true",
        help="Run J selection sweep over candidates before fitting final model",
    )
    parser.add_argument(
        "--with-demographics",
        action="store_true",
        help="Include ACS demographic features (combined-feature run)",
    )
    parser.add_argument(
        "--presidential-weight",
        type=float,
        default=2.5,
        help="Weight multiplier for presidential features (default: 2.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Soft membership temperature (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path (default: data/tracts/national_tract_assignments.parquet)",
    )
    args = parser.parse_args()

    features_path = PROJECT_ROOT / "data" / "tracts" / "tract_features.parquet"
    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "tracts" / "national_tract_assignments.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("National Tract KMeans Clustering")
    log.info("=" * 60)

    # Load and prepare
    df_filt, X_train, X_holdout = load_and_filter_features(
        features_path,
        use_demographics=args.with_demographics,
        presidential_weight=args.presidential_weight,
    )
    n_tracts = len(df_filt)
    log.info("Training matrix: %d tracts × %d dims", *X_train.shape)

    # J selection or use provided J
    j_selection_results: list[dict] = []
    if args.j_sweep:
        j_candidates = [20, 30, 40, 50, 60, 70, 80]
        log.info("Running J sweep over %s ...", j_candidates)
        best_j, best_r, j_selection_results = run_j_sweep(
            X_train, X_holdout, j_candidates, n_init=10
        )
        log.info("J sweep complete: best J=%d (holdout r=%.4f)", best_j, best_r)
    else:
        best_j = args.j
        log.info("Using J=%d (no sweep)", best_j)

    # Final KMeans fit
    log.info("Fitting final KMeans J=%d on %d tracts ...", best_j, n_tracts)
    km = KMeans(n_clusters=best_j, n_init=10, random_state=42)
    labels = km.fit_predict(X_train)
    centroids = km.cluster_centers_

    # Soft membership (temperature-scaled inverse distance)
    dists = np.zeros((n_tracts, best_j))
    for t in range(best_j):
        dists[:, t] = np.linalg.norm(X_train - centroids[t], axis=1)
    soft_scores = temperature_soft_membership(dists, T=args.temperature)

    # Holdout validation
    final_r, per_dim = compute_holdout_r(labels, X_holdout)
    log.info("Final holdout r=%.4f (per dim: %s)", final_r, [f"{x:.3f}" for x in per_dim])

    # Type size distribution
    unique, counts = np.unique(labels, return_counts=True)
    sorted_counts = sorted(counts.tolist(), reverse=True)
    log.info("Type sizes (sorted): %s", sorted_counts[:20])
    log.info("Min type size: %d, max: %d, median: %.0f", min(counts), max(counts), np.median(counts))

    # Super-type nesting via demographic profiles (NOT centroids).
    # Centroids are in standardized shift space — at J=100 they cluster
    # degenerately, collapsing all types into one super-type.
    # Demographic profiles produce meaningful separation (see CLAUDE.md).
    log.info("Nesting %d types into super-types via demographic profiles ...", best_j)
    demo_profiles = build_demographic_profiles(df_filt, labels, best_j)
    nesting_features = demo_profiles if demo_profiles is not None else centroids
    nesting = nest_types(nesting_features, s_candidates=[5, 6, 7, 8])
    log.info(
        "Best super-types: S=%d (silhouette=%.4f)",
        nesting.best_s,
        nesting.silhouette_scores[nesting.best_s],
    )
    super_type_labels = np.array([nesting.mapping[t] for t in labels])

    # Build output DataFrame (concat all columns at once to avoid fragmentation)
    score_cols = {f"type_{i}_score": soft_scores[:, i] for i in range(best_j)}
    out_df = pd.concat([
        pd.DataFrame({"GEOID": df_filt["GEOID"].values}),
        pd.DataFrame(score_cols),
        pd.DataFrame({"dominant_type": labels, "super_type": super_type_labels}),
    ], axis=1)

    out_df.to_parquet(out_path, index=False)
    log.info("Saved %d tract assignments → %s", len(out_df), out_path)

    # Save validation metrics
    val_path = out_path.parent / "national_tract_validation.json"
    validation = {
        "n_tracts_input": 84415,
        "n_tracts_after_pop_filter": int(n_tracts),
        "pop_filter_threshold": POPULATION_MIN_VOTES,
        "j": int(best_j),
        "temperature": float(args.temperature),
        "presidential_weight": float(args.presidential_weight),
        "with_demographics": bool(args.with_demographics),
        "holdout_r": float(final_r),
        "holdout_per_dim_r": per_dim,
        "holdout_features": HOLDOUT_FEATURES,
        "n_super_types": int(nesting.best_s),
        "super_type_silhouette": {str(s): float(v) for s, v in nesting.silhouette_scores.items()},
        "type_size_min": int(min(counts)),
        "type_size_max": int(max(counts)),
        "type_size_median": float(np.median(counts)),
        "j_selection_results": j_selection_results,
    }
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    log.info("Saved validation → %s", val_path)

    # Summary
    print()
    print("=" * 60)
    print("NATIONAL TRACT CLUSTERING COMPLETE")
    print("=" * 60)
    print(f"  Tracts clustered:  {n_tracts:,} (of 84,415 input)")
    print(f"  J (types):         {best_j}")
    print(f"  Holdout r:         {final_r:.4f}")
    print(f"  Super-types (S):   {nesting.best_s}")
    print(f"  Output:            {out_path}")
    print(f"  Validation:        {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
