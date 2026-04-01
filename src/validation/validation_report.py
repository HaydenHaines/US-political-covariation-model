"""Type validation report generator.

Orchestrates all type validation metrics — coherence, stability, holdout
accuracy, Ridge accuracy, Ridge+demographics, covariance validation, and
per-super-type RMSE — then writes a JSON report to data/validation/.

Entry point for `python -m src.validation.validate_types`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.validation.holdout_accuracy import (
    RMSE_FLAG_THRESHOLD,
    holdout_accuracy,
    holdout_accuracy_county_prior,
    holdout_accuracy_county_prior_loo,
    holdout_accuracy_ridge,
    holdout_accuracy_ridge_augmented,
    rmse_by_super_type,
)
from src.validation.type_coherence import type_coherence
from src.validation.type_stability import type_stability

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> Path:
    """Resolve a string path as absolute or relative to project root."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_shift_data(
    shifts_path: Path,
    min_year: int,
) -> tuple:
    """Load the shift matrix and derive training/holdout column indices.

    Returns
    -------
    (shifts_df, full_matrix, training_col_names, holdout_col_names,
     training_matrix, training_indices, holdout_indices, used_cols)
    """
    import pandas as pd

    log.info("Loading shift matrix from %s", shifts_path)
    shifts_df = pd.read_parquet(shifts_path)

    all_cols = [c for c in shifts_df.columns if c != "county_fips"]

    # 2020→2024 presidential is held out for testing generalization
    holdout_keywords = ["20_24"]
    holdout_col_names = [c for c in all_cols if any(kw in c for kw in holdout_keywords)]
    training_col_names_unfiltered = [c for c in all_cols if c not in holdout_col_names]

    # Filter training columns to min_year so they match type discovery's filter
    training_col_names = []
    for c in training_col_names_unfiltered:
        parts = c.split("_")
        try:
            y2 = int(parts[-2])
            y1 = y2 + (1900 if y2 >= 50 else 2000) if len(parts[-2]) == 2 else y2
            if y1 >= min_year:
                training_col_names.append(c)
        except (ValueError, IndexError):
            training_col_names.append(c)  # keep if year can't be parsed

    if not training_col_names:
        training_col_names = training_col_names_unfiltered  # fallback

    used_cols = training_col_names + holdout_col_names
    full_matrix = shifts_df[used_cols].values
    holdout_indices = [used_cols.index(c) for c in holdout_col_names]

    training_matrix = shifts_df[training_col_names].values
    training_indices = [used_cols.index(c) for c in training_col_names]

    return (
        shifts_df,
        full_matrix,
        training_col_names,
        holdout_col_names,
        training_matrix,
        training_indices,
        holdout_indices,
        used_cols,
    )


def _load_type_assignments(assignments_path: Path) -> tuple:
    """Load type assignment parquet and extract scores and dominant types.

    Returns
    -------
    (assignments_df, scores, dominant_types, score_cols, j)
    """
    import pandas as pd

    log.info("Loading type assignments from %s", assignments_path)
    assignments_df = pd.read_parquet(assignments_path)

    score_cols = [c for c in assignments_df.columns if c.startswith("type_") and c.endswith("_score")]
    if not score_cols:
        score_cols = [c for c in assignments_df.columns if c != "county_fips" and c != "dominant_type"]
    scores = assignments_df[score_cols].values

    if "dominant_type" in assignments_df.columns:
        dominant_types = assignments_df["dominant_type"].values
    else:
        dominant_types = np.argmax(np.abs(scores), axis=1)

    j = scores.shape[1]
    return assignments_df, scores, dominant_types, score_cols, j


def _run_accuracy_validations(
    scores: np.ndarray,
    full_matrix: np.ndarray,
    training_indices: list[int],
    holdout_indices: list[int],
    dominant_types: np.ndarray,
    shifts_df,
    assignments_df,
    score_cols: list[str],
) -> tuple:
    """Run the five holdout accuracy methods and return their result dicts.

    Returns
    -------
    (accuracy, accuracy_county_prior, accuracy_county_prior_loo,
     accuracy_ridge, accuracy_ridge_augmented)
    """
    log.info("Running holdout_accuracy (type-mean prior)...")
    accuracy = holdout_accuracy(scores, full_matrix, holdout_indices, dominant_types)

    log.info("Running holdout_accuracy (county-level prior)...")
    accuracy_county_prior = holdout_accuracy_county_prior(
        scores, full_matrix, training_indices, holdout_indices,
    )

    log.info("Running holdout_accuracy_county_prior_loo...")
    accuracy_county_prior_loo = holdout_accuracy_county_prior_loo(
        scores, full_matrix, training_indices, holdout_indices,
    )

    log.info("Running holdout_accuracy_ridge (LOO via hat matrix)...")
    accuracy_ridge = holdout_accuracy_ridge(
        scores, full_matrix, training_indices, holdout_indices, include_county_mean=True,
    )

    log.info("Running holdout_accuracy_ridge_augmented (Ridge + demographics)...")
    accuracy_ridge_augmented = _run_ridge_augmented(
        scores, full_matrix, training_indices, holdout_indices,
        shifts_df, assignments_df, score_cols,
    )

    return (
        accuracy,
        accuracy_county_prior,
        accuracy_county_prior_loo,
        accuracy_ridge,
        accuracy_ridge_augmented,
    )


def _run_ridge_augmented(
    scores: np.ndarray,
    full_matrix: np.ndarray,
    training_indices: list[int],
    holdout_indices: list[int],
    shifts_df,
    assignments_df,
    score_cols: list[str],
) -> dict | None:
    """Align county FIPS and scores for the Ridge+demographics method.

    The augmented Ridge needs scores aligned to shift matrix row order.
    When both DataFrames have county_fips, we reindex assignments to match
    shifts; otherwise we fall back to the original row order.
    """
    county_fips_arr: np.ndarray | None = None
    scores_for_augmented = scores

    if "county_fips" in shifts_df.columns:
        shifts_fips = shifts_df["county_fips"].values
        if "county_fips" in assignments_df.columns:
            assign_map = assignments_df.set_index("county_fips")[score_cols]
            try:
                scores_aligned = assign_map.reindex(shifts_fips).values
                # Only use aligned scores if join was clean (no missing counties)
                if not np.any(np.isnan(scores_aligned)):
                    scores_for_augmented = scores_aligned
                county_fips_arr = shifts_fips
            except (KeyError, ValueError):
                county_fips_arr = shifts_fips
        else:
            county_fips_arr = shifts_fips

    return holdout_accuracy_ridge_augmented(
        scores_for_augmented,
        full_matrix,
        training_indices,
        holdout_indices,
        county_fips=county_fips_arr,
        include_county_mean=True,
    )


def _compute_super_type_rmse(
    assignments_df,
    shifts_df,
    scores: np.ndarray,
    full_matrix: np.ndarray,
    training_indices: list[int],
    holdout_indices: list[int],
) -> dict[str, float] | None:
    """Compute per-super-type RMSE using the county-prior predictions.

    Re-derives county-prior predictions for the first holdout dimension,
    then breaks down RMSE by super-type label. Returns None if super_type
    column is absent or alignment fails.
    """
    if "super_type" not in assignments_df.columns:
        return None

    # Load display name map if available
    super_type_names_map: dict[int, str] | None = None
    super_types_parquet = PROJECT_ROOT / "data" / "communities" / "super_types.parquet"
    if super_types_parquet.exists():
        try:
            import pandas as pd
            st_df = pd.read_parquet(super_types_parquet)
            if "super_type_id" in st_df.columns and "display_name" in st_df.columns:
                super_type_names_map = dict(zip(st_df["super_type_id"], st_df["display_name"]))
        except (FileNotFoundError, KeyError, ValueError) as e:
            log.warning("Could not load super_types.parquet: %s", e)

    def _rmse_from_labels(st_labels_int: np.ndarray) -> dict[str, float] | None:
        """Re-derive county-prior predictions and compute RMSE by super-type."""
        if not holdout_indices:
            return None
        first_holdout = holdout_indices[0]
        actual_h = full_matrix[:, first_holdout]
        abs_sc = np.abs(scores)
        rs = abs_sc.sum(axis=1, keepdims=True)
        rs = np.where(rs == 0, 1.0, rs)
        wts = abs_sc / rs
        wt_sums = wts.sum(axis=0)
        wt_sums = np.where(wt_sums == 0, 1.0, wt_sums)
        tr_data = full_matrix[:, training_indices]
        c_train_means = tr_data.mean(axis=1)
        type_train_means = (wts.T @ c_train_means) / wt_sums
        type_hold_means = (wts.T @ actual_h) / wt_sums
        type_adj = type_hold_means - type_train_means
        c_adj = (wts * type_adj[None, :]).sum(axis=1)
        predicted_h = c_train_means + c_adj
        return rmse_by_super_type(
            actual_h, predicted_h, st_labels_int,
            super_type_names=super_type_names_map,
        )

    try:
        if "county_fips" in assignments_df.columns and "county_fips" in shifts_df.columns:
            st_series = assignments_df.set_index("county_fips")["super_type"]
            shifts_fips_st = shifts_df["county_fips"].values
            st_labels_aligned = st_series.reindex(shifts_fips_st).values
            if not np.any(np.isnan(st_labels_aligned.astype(float))):
                log.info("Running rmse_by_super_type (FIPS-aligned)...")
                return _rmse_from_labels(st_labels_aligned.astype(int))
            else:
                log.warning("rmse_by_super_type: some counties missing super_type; skipping")
                return None
        else:
            log.info("Running rmse_by_super_type (row-aligned)...")
            return _rmse_from_labels(assignments_df["super_type"].values.astype(int))
    except (KeyError, ValueError, TypeError) as e:
        log.warning("rmse_by_super_type failed: %s", e)
        return None


def _compute_covariance_validation(
    cov_path: Path,
    scores: np.ndarray,
    training_matrix: np.ndarray,
) -> float | None:
    """Load the type covariance matrix and compute validation r.

    Groups training shift columns into election pairs (every 3 dims) then
    calls validate_covariance(). Returns None if the file is missing or
    the computation fails.
    """
    if not cov_path.exists():
        return None

    try:
        import pandas as pd
        from src.covariance.construct_type_covariance import (
            CovarianceResult,
            validate_covariance,
        )

        cov_df = pd.read_parquet(cov_path)
        cov_matrix = cov_df.values

        # Normalize to correlation matrix for validate_covariance
        corr_std = np.sqrt(np.diag(cov_matrix))
        corr_std = np.where(corr_std == 0, 1.0, corr_std)
        corr_matrix = cov_matrix / np.outer(corr_std, corr_std)

        constructed = CovarianceResult(
            correlation_matrix=corr_matrix,
            covariance_matrix=cov_matrix,
            validation_r=float("nan"),
            used_hybrid=False,
            sigma_base=0.07,
        )

        n_train_cols = training_matrix.shape[1]
        election_col_groups = [
            list(range(i, min(i + 3, n_train_cols)))
            for i in range(0, n_train_cols, 3)
        ]
        cov_validation_r = validate_covariance(
            constructed, scores, training_matrix, election_col_groups
        )
        log.info("Covariance validation r = %.3f", cov_validation_r)
        return cov_validation_r
    except (FileNotFoundError, KeyError, ValueError) as e:
        log.warning("Covariance validation failed: %s", e)
        return None


def _print_report_summary(
    full_matrix: np.ndarray,
    j: int,
    n_train: int,
    holdout_indices: list[int],
    coherence: dict,
    stability: dict,
    accuracy: dict,
    accuracy_county_prior: dict,
    accuracy_county_prior_loo: dict,
    accuracy_ridge: dict,
    accuracy_ridge_augmented: dict | None,
    cov_validation_r: float | None,
    rmse_super_type: dict[str, float] | None,
) -> None:
    """Print the human-readable validation summary to stdout."""
    print("\n" + "=" * 65)
    print("Type Validation Report")
    print("=" * 65)
    print(f"  J (types):             {j}")
    print(f"  N (counties):          {full_matrix.shape[0]}")
    print(f"  Training dims:         {n_train}")
    print(f"  Holdout dims:          {len(holdout_indices)}")
    print()
    print(f"  Coherence (mean ratio): {coherence['mean_ratio']:.3f}  (> 0.5 = strong)")
    print(f"  Stability (max angle):  {stability['max_angle_degrees']:.1f} deg  (< 30 = stable)")
    print(f"  Holdout accuracy (r):   {accuracy['mean_r']:.3f}  (> 0.5 = good, type-mean prior)")
    print(f"  Holdout accuracy (r):   {accuracy_county_prior['mean_r']:.3f}  (county-level prior)")
    if "mean_rmse" in accuracy_county_prior:
        print(f"  Holdout RMSE:           {accuracy_county_prior['mean_rmse']:.4f}  (county-level prior)")
    print(f"  Holdout LOO r:          {accuracy_county_prior_loo['mean_r']:.3f}  (county-level prior, LOO)")
    if "mean_rmse" in accuracy_county_prior_loo:
        print(f"  Holdout LOO RMSE:       {accuracy_county_prior_loo['mean_rmse']:.4f}  (county-level prior, LOO)")
    print(f"  Holdout Ridge LOO r:    {accuracy_ridge['mean_r']:.3f}  (Ridge scores+county_mean, LOO)")
    if "mean_rmse" in accuracy_ridge:
        print(f"  Holdout Ridge RMSE:     {accuracy_ridge['mean_rmse']:.4f}  (Ridge, LOO)")
    if accuracy_ridge_augmented is not None:
        n_matched = accuracy_ridge_augmented.get("n_matched_counties", "?")
        n_demo = accuracy_ridge_augmented.get("n_demo_features", "?")
        print(
            f"  Ridge+Demo LOO r:       {accuracy_ridge_augmented['mean_r']:.3f}"
            f"  (Ridge scores+county_mean+demographics, LOO, N={n_matched}, D_demo={n_demo})"
        )
        if "mean_rmse" in accuracy_ridge_augmented:
            print(f"  Ridge+Demo RMSE:        {accuracy_ridge_augmented['mean_rmse']:.4f}  (Ridge+demo, LOO)")
    else:
        print("  Ridge+Demo LOO r:       N/A  (demographics file missing or FIPS unavailable)")
    if cov_validation_r is not None:
        print(f"  Covariance val r:       {cov_validation_r:.3f}  (> 0.4 = acceptable)")
    if rmse_super_type is not None:
        print()
        print(f"  RMSE by super-type  (threshold: {RMSE_FLAG_THRESHOLD:.2f} = flagged):")
        for st_name, st_rmse in sorted(rmse_super_type.items()):
            flag = "  *** HIGH ***" if st_rmse > RMSE_FLAG_THRESHOLD else ""
            print(f"    {st_name:<32s}  {st_rmse:.4f}{flag}")
    print("=" * 65)


def generate_type_validation_report(
    shift_parquet_path: str = "data/shifts/county_shifts_multiyear.parquet",
    type_assignments_path: str = "data/communities/type_assignments.parquet",
    type_covariance_path: str = "data/covariance/type_covariance.parquet",
    type_profiles_path: str = "data/communities/type_profiles.parquet",
    min_year: int = 2008,
) -> dict:
    """Run all type validations and save a JSON report.

    Loads shift matrix and type assignments, runs type_coherence,
    type_stability, and holdout_accuracy, then saves results to
    data/validation/type_validation_report.json.

    Returns
    -------
    dict with all validation results.
    """
    shifts_path = _resolve_path(shift_parquet_path)
    assignments_path = _resolve_path(type_assignments_path)

    (
        shifts_df,
        full_matrix,
        training_col_names,
        holdout_col_names,
        training_matrix,
        training_indices,
        holdout_indices,
        used_cols,
    ) = _load_shift_data(shifts_path, min_year)

    assignments_df, scores, dominant_types, score_cols, j = _load_type_assignments(assignments_path)

    n_train = len(training_col_names)
    mid = n_train // 2
    window_a_cols = list(range(0, mid))
    window_b_cols = list(range(mid, n_train))

    # --- Run validations ---
    log.info("Running type_coherence...")
    coherence = type_coherence(scores, full_matrix, holdout_indices)

    log.info("Running type_stability...")
    stability = type_stability(training_matrix, window_a_cols, window_b_cols, j=j)

    (
        accuracy,
        accuracy_county_prior,
        accuracy_county_prior_loo,
        accuracy_ridge,
        accuracy_ridge_augmented,
    ) = _run_accuracy_validations(
        scores, full_matrix, training_indices, holdout_indices,
        dominant_types, shifts_df, assignments_df, score_cols,
    )

    rmse_super_type = _compute_super_type_rmse(
        assignments_df, shifts_df, scores, full_matrix, training_indices, holdout_indices
    )

    cov_path = _resolve_path(type_covariance_path)
    cov_validation_r = _compute_covariance_validation(cov_path, scores, training_matrix)

    # --- Assemble report ---
    report: dict = {
        "coherence": coherence,
        "stability": stability,
        "holdout_accuracy": accuracy,
        "holdout_accuracy_county_prior": accuracy_county_prior,
        "holdout_accuracy_county_prior_loo": accuracy_county_prior_loo,
        "holdout_accuracy_ridge": accuracy_ridge,
        "holdout_accuracy_ridge_augmented": accuracy_ridge_augmented,
        "covariance_validation_r": cov_validation_r,
        "rmse_by_super_type": rmse_super_type,
        "j": j,
        "n_counties": int(full_matrix.shape[0]),
        "n_training_dims": n_train,
        "n_holdout_dims": len(holdout_indices),
        "holdout_columns": holdout_col_names,
        "min_year": min_year,
    }

    _print_report_summary(
        full_matrix, j, n_train, holdout_indices,
        coherence, stability,
        accuracy, accuracy_county_prior, accuracy_county_prior_loo,
        accuracy_ridge, accuracy_ridge_augmented,
        cov_validation_r, rmse_super_type,
    )

    # Save JSON report
    out_dir = PROJECT_ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "type_validation_report.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    log.info("Report saved to %s", out_path)
    print(f"\n  Full report: {out_path}")

    return report
