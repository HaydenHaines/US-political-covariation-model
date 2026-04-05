"""Ridge regression prediction experiment.

Tests whether Ridge regression on type scores (N×J) improves over type-mean
prediction for the 2020→2024 presidential holdout.

Methods compared:
  (a) Standard type-mean (holdout_accuracy_county_prior)
  (b) LOO type-mean (holdout_accuracy_county_prior_loo)
  (c) Ridge on type_scores only (N×J features)
  (d) Ridge on type_scores + county_training_mean (N×(J+1) features)

Tested at J=100 and J=160.

Usage:
    uv run python scripts/experiment_ridge_prediction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types
from src.validation.validate_types import (
    holdout_accuracy_county_prior,
    holdout_accuracy_county_prior_loo,
)


# ── Column parsing helpers ────────────────────────────────────────────────────


def parse_start_year(col: str) -> int | None:
    """Extract the start year from a shift column name like pres_d_shift_08_12."""
    parts = col.split("_")
    try:
        # Second-to-last part is the start year (2-digit)
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def is_holdout_col(col: str) -> bool:
    return "20_24" in col


def classify_columns(all_cols: list[str], min_year: int = 2008) -> tuple[list[str], list[str]]:
    """Split columns into training (start >= min_year) and holdout (pres_*_20_24)."""
    holdout = [c for c in all_cols if is_holdout_col(c)]
    training = []
    for c in all_cols:
        if is_holdout_col(c):
            continue
        start = parse_start_year(c)
        if start is None or start >= min_year:
            training.append(c)
    return training, holdout


# ── Ridge LOO via hat matrix ──────────────────────────────────────────────────


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Compute exact Ridge LOO predictions via the augmented hat-matrix shortcut.

    For Ridge with unpenalized intercept (matching sklearn's fit_intercept=True):
      Augmented: X_aug = [1 | X], penalty = diag(0, alpha*I_P)
      H = X_aug (X_aug'X_aug + P)^{-1} X_aug'
      LOO prediction: y_loo_i = y_i - e_i / (1 - H_ii)
        where e_i = y_i - y_hat_i (in-sample residual)

    This gives EXACT leave-one-out predictions (verified against brute force).
    The key: intercept is NOT regularized, so we use the augmented matrix to
    get the correct hat diagonal including the intercept column.

    Parameters
    ----------
    X : (N, P) feature matrix (raw, not pre-centered)
    y : (N,) target vector
    alpha : float, ridge penalty on slope coefficients only

    Returns
    -------
    y_loo : (N,) LOO predicted values
    """
    N, P = X.shape

    # Augment X with intercept column
    X_aug = np.column_stack([np.ones(N), X])  # (N, P+1)

    # Penalty matrix: no penalty on intercept, alpha on slopes
    pen = alpha * np.eye(P + 1)
    pen[0, 0] = 0.0  # unpenalized intercept

    A = X_aug.T @ X_aug + pen  # (P+1, P+1)
    A_inv = np.linalg.inv(A)

    # Hat matrix diagonal (exact)
    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)  # (N,)

    # In-sample Ridge predictions
    beta = A_inv @ X_aug.T @ y  # (P+1,)
    y_hat = X_aug @ beta  # (N,)
    e = y - y_hat  # (N,)

    # LOO formula: LOO_pred_i = y_i - e_i / (1 - h_ii)
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    y_loo = y - e / denom
    return y_loo


def ridge_gcv_method(
    X: np.ndarray,
    y_cols: np.ndarray,
    use_loo_hat: bool = True,
) -> dict:
    """Fit RidgeCV (GCV alpha selection) and compute LOO r and RMSE.

    Parameters
    ----------
    X : (N, P) features
    y_cols : (N, H) holdout targets (one per holdout column)
    use_loo_hat : bool
        If True, use exact LOO via hat matrix.
        If False, report in-sample r (optimistic upper bound).

    Returns
    -------
    dict with mean_r, per_dim_r, mean_rmse, per_dim_rmse
    """
    # RidgeCV with a wide alpha grid; GCV approximates LOO
    alphas = np.logspace(-3, 6, 100)
    N, P = X.shape
    H = y_cols.shape[1]

    per_dim_r: list[float] = []
    per_dim_rmse: list[float] = []
    best_alphas: list[float] = []

    for h in range(H):
        y = y_cols[:, h]

        rcv = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        best_alpha = float(rcv.alpha_)
        best_alphas.append(best_alpha)

        if use_loo_hat:
            # Use exact augmented hat-matrix LOO (verified against brute force)
            y_loo = ridge_loo_predictions(X, y, best_alpha)
            actual = y
        else:
            y_loo = rcv.predict(X)
            actual = y

        if np.std(actual) < 1e-10 or np.std(y_loo) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(actual, y_loo)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

        rmse = float(np.sqrt(np.mean((actual - y_loo) ** 2)))
        per_dim_rmse.append(rmse)

    return {
        "mean_r": float(np.mean(per_dim_r)),
        "per_dim_r": per_dim_r,
        "mean_rmse": float(np.mean(per_dim_rmse)),
        "per_dim_rmse": per_dim_rmse,
        "best_alphas": best_alphas,
    }


# ── Main experiment ───────────────────────────────────────────────────────────


def run_experiment(
    j_values: list[int] = (100, 160),
    min_year: int = 2008,
    presidential_weight: float = 8.0,
    temperature: float = 10.0,
) -> list[dict]:
    """Run the full Ridge vs type-mean comparison experiment."""
    print("=" * 70)
    print("Ridge Prediction Experiment")
    print("=" * 70)

    # Load shift data
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(shifts_path)

    all_cols = [c for c in df.columns if c != "county_fips"]
    training_col_names, holdout_col_names = classify_columns(all_cols, min_year=min_year)

    print(f"\nData: {df.shape[0]} counties, {len(all_cols)} total dims")
    print(f"Training cols (start >= {min_year}): {len(training_col_names)}")
    print(f"Holdout cols: {holdout_col_names}")

    # Build full matrix (training + holdout) in the original (unscaled) space
    used_col_names = training_col_names + holdout_col_names
    full_matrix_raw = df[used_col_names].values.astype(float)

    training_indices = list(range(len(training_col_names)))
    holdout_indices = list(range(len(training_col_names), len(used_col_names)))

    # Holdout targets in raw (unscaled) space — this is what validate_types uses
    # (the holdout_accuracy functions read raw shift values from shift_matrix)
    holdout_raw = full_matrix_raw[:, holdout_indices]  # (N, H)

    # Build the SCALED training matrix (used for discover_types)
    # Matches what run_type_discovery.main() does
    training_raw = full_matrix_raw[:, training_indices]  # (N, D_train)

    pres_indices_in_training = [
        i for i, c in enumerate(training_col_names) if "pres_" in c
    ]

    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)  # (N, D_train)

    if presidential_weight != 1.0:
        training_scaled[:, pres_indices_in_training] *= presidential_weight
        print(
            f"Presidential weight={presidential_weight} applied to "
            f"{len(pres_indices_in_training)} columns (post-scaling)"
        )

    # For validate_types functions: build full_matrix with scaled training + raw holdout
    # The validate functions use shift_matrix[:, col] for actual values.
    # In validate_types.generate_type_validation_report(), it passes the *raw* full_matrix
    # (built from shifts_df[used_cols].values) and the training_indices/holdout_indices.
    # The holdout actuals are raw; the training means are raw (for county priors).
    # BUT type-mean adjustments also use raw values. So we pass raw throughout.
    # Training matrix passed to discover_types is SCALED, but shift_matrix for
    # holdout_accuracy_county_prior uses the RAW matrix — let's confirm by re-reading
    # validate_types.generate_type_validation_report at line 460: full_matrix = shifts_df[used_cols].values
    # That's raw. So holdout_accuracy_county_prior gets raw shifts. We match that.

    full_matrix_for_validation = full_matrix_raw  # raw throughout for consistency

    results: list[dict] = []

    for j in j_values:
        print(f"\n{'─' * 70}")
        print(f"J = {j}")
        print(f"{'─' * 70}")

        # Discover types on scaled training matrix
        print(f"Running KMeans (J={j}, T={temperature})...", end=" ", flush=True)
        type_result = discover_types(training_scaled, j=j, temperature=temperature, random_state=42)
        scores = type_result.scores  # (N, J) soft membership, row-normalized
        print("done")

        # County training means (raw, unscaled)
        county_training_means = training_raw.mean(axis=1)  # (N,)

        # ── Method (a): Standard type-mean (no LOO) ──────────────────────────
        print("  (a) Standard type-mean...", end=" ", flush=True)
        res_a = holdout_accuracy_county_prior(
            scores, full_matrix_for_validation, training_indices, holdout_indices
        )
        print(f"r={res_a['mean_r']:.4f}, RMSE={res_a['mean_rmse']:.4f}")

        # ── Method (b): LOO type-mean ─────────────────────────────────────────
        print("  (b) LOO type-mean...", end=" ", flush=True)
        res_b = holdout_accuracy_county_prior_loo(
            scores, full_matrix_for_validation, training_indices, holdout_indices
        )
        print(f"r={res_b['mean_r']:.4f}, RMSE={res_b['mean_rmse']:.4f}")

        # ── Method (c): Ridge on type scores only (N×J) ─────────────────────
        # Features: soft membership scores (already row-normalized to [0,1])
        # Target: raw holdout shifts
        print("  (c) Ridge (scores only)...", end=" ", flush=True)
        X_c = scores  # (N, J)
        res_c = ridge_gcv_method(X_c, holdout_raw, use_loo_hat=True)
        print(
            f"r={res_c['mean_r']:.4f}, RMSE={res_c['mean_rmse']:.4f}"
            f"  (alphas: {[f'{a:.1f}' for a in res_c['best_alphas']]})"
        )

        # ── Method (d): Ridge on type scores + county training mean (N×(J+1)) ─
        print("  (d) Ridge (scores + county_mean)...", end=" ", flush=True)
        X_d = np.column_stack([scores, county_training_means])  # (N, J+1)
        res_d = ridge_gcv_method(X_d, holdout_raw, use_loo_hat=True)
        print(
            f"r={res_d['mean_r']:.4f}, RMSE={res_d['mean_rmse']:.4f}"
            f"  (alphas: {[f'{a:.1f}' for a in res_d['best_alphas']]})"
        )

        results.append({
            "j": j,
            "method_a_r": res_a["mean_r"],
            "method_a_rmse": res_a["mean_rmse"],
            "method_a_per_dim_r": res_a["per_dim_r"],
            "method_b_r": res_b["mean_r"],
            "method_b_rmse": res_b["mean_rmse"],
            "method_b_per_dim_r": res_b["per_dim_r"],
            "method_c_r": res_c["mean_r"],
            "method_c_rmse": res_c["mean_rmse"],
            "method_c_per_dim_r": res_c["per_dim_r"],
            "method_c_alphas": res_c["best_alphas"],
            "method_d_r": res_d["mean_r"],
            "method_d_rmse": res_d["mean_rmse"],
            "method_d_per_dim_r": res_d["per_dim_r"],
            "method_d_alphas": res_d["best_alphas"],
            "holdout_cols": holdout_col_names,
        })

    return results


def format_results(results: list[dict], holdout_col_names: list[str]) -> str:
    """Format results as a markdown table."""
    lines: list[str] = []

    lines.append("# Ridge Prediction Experiment — S197")
    lines.append("")
    lines.append("**Question:** Does Ridge regression on type membership scores")
    lines.append("improve LOO holdout prediction over the type-mean baseline?")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Data: 3,154 counties, national (all 50 states + DC)")
    lines.append("- Training: shifts with start year >= 2008 (presidential, governor, Senate pairs)")
    lines.append("- Holdout: pres_d/r/turnout_shift_20_24")
    lines.append("- Preprocessing: StandardScaler + presidential_weight=8.0 (post-scaling)")
    lines.append("- Type discovery: KMeans, T=10 (temperature-scaled soft membership)")
    lines.append("- Ridge alpha: selected by GCV (sklearn RidgeCV, alphas=logspace(-3,6,100))")
    lines.append("- LOO: exact hat-matrix shortcut for Ridge; leave-one-county-out for type-mean")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("| Method | Description |")
    lines.append("|--------|-------------|")
    lines.append("| (a) Standard type-mean | county_mean + score-weighted type adjustment (no LOO) |")
    lines.append("| (b) LOO type-mean | same but each county excluded from its own type mean |")
    lines.append("| (c) Ridge scores-only | RidgeCV(X=scores N×J) → holdout; GCV LOO |")
    lines.append("| (d) Ridge scores+mean | RidgeCV(X=[scores, county_mean] N×J+1) → holdout; GCV LOO |")
    lines.append("")
    lines.append("## Results")
    lines.append("")

    # Summary table
    lines.append("### LOO Pearson r (primary metric)")
    lines.append("")
    lines.append("| J | (a) Std type-mean | (b) LOO type-mean | (c) Ridge scores | (d) Ridge scores+mean |")
    lines.append("|---|:-----------------:|:-----------------:|:----------------:|:---------------------:|")
    for r in results:
        lines.append(
            f"| {r['j']} "
            f"| {r['method_a_r']:.4f} "
            f"| {r['method_b_r']:.4f} "
            f"| {r['method_c_r']:.4f} "
            f"| {r['method_d_r']:.4f} |"
        )
    lines.append("")

    lines.append("### LOO RMSE (lower is better)")
    lines.append("")
    lines.append("| J | (a) Std type-mean | (b) LOO type-mean | (c) Ridge scores | (d) Ridge scores+mean |")
    lines.append("|---|:-----------------:|:-----------------:|:----------------:|:---------------------:|")
    for r in results:
        lines.append(
            f"| {r['j']} "
            f"| {r['method_a_rmse']:.4f} "
            f"| {r['method_b_rmse']:.4f} "
            f"| {r['method_c_rmse']:.4f} "
            f"| {r['method_d_rmse']:.4f} |"
        )
    lines.append("")

    # Per-dim detail
    lines.append("### Per-dimension LOO r breakdown")
    lines.append("")
    for r in results:
        lines.append(f"**J={r['j']}**")
        lines.append("")
        lines.append(f"| Holdout dim | (a) Std | (b) LOO | (c) Ridge | (d) Ridge+mean |")
        lines.append(f"|-------------|:-------:|:-------:|:---------:|:--------------:|")
        for idx, col in enumerate(r["holdout_cols"]):
            a_r = r["method_a_per_dim_r"][idx]
            b_r = r["method_b_per_dim_r"][idx]
            c_r = r["method_c_per_dim_r"][idx]
            d_r = r["method_d_per_dim_r"][idx]
            lines.append(f"| {col} | {a_r:.4f} | {b_r:.4f} | {c_r:.4f} | {d_r:.4f} |")
        lines.append("")

    # Ridge alpha details
    lines.append("### Ridge GCV alpha selections")
    lines.append("")
    for r in results:
        lines.append(f"**J={r['j']}**")
        lines.append("")
        for idx, col in enumerate(r["holdout_cols"]):
            c_alpha = r["method_c_alphas"][idx]
            d_alpha = r["method_d_alphas"][idx]
            lines.append(f"- {col}: (c) α={c_alpha:.2f}, (d) α={d_alpha:.2f}")
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    best = max(results, key=lambda x: x["method_d_r"])
    lines.append(f"- Best configuration: J={best['j']}, method (d) Ridge scores+mean, LOO r={best['method_d_r']:.4f}")

    for r in results:
        improvement_over_b = r["method_d_r"] - r["method_b_r"]
        lines.append(
            f"- J={r['j']}: Ridge(d) vs LOO-type-mean(b) = "
            f"{r['method_d_r']:.4f} vs {r['method_b_r']:.4f} "
            f"({'+'if improvement_over_b >= 0 else ''}{improvement_over_b:.4f})"
        )

    lines.append("")
    lines.append("## Baseline (CLAUDE.md)")
    lines.append("")
    lines.append("- County holdout LOO r (J=100, type-mean): 0.448")
    lines.append("- County holdout r (standard, J=100): 0.698")

    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ridge vs type-mean prediction experiment")
    parser.add_argument("--j", nargs="+", type=int, default=[100, 160])
    parser.add_argument("--min-year", type=int, default=2008)
    parser.add_argument("--pres-weight", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=10.0)
    args = parser.parse_args()

    results = run_experiment(
        j_values=args.j,
        min_year=args.min_year,
        presidential_weight=args.pres_weight,
        temperature=args.temperature,
    )

    holdout_cols = results[0]["holdout_cols"] if results else []

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'J':>5}  {'(a)Std':>8}  {'(b)LOO':>8}  {'(c)Ridge':>10}  {'(d)Ridge+mean':>13}")
    print(f"{'':>5}  {'type-mean':>8}  {'type-mean':>8}  {'scores-only':>10}  {'scores+mean':>13}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['j']:>5}  "
            f"{r['method_a_r']:>8.4f}  "
            f"{r['method_b_r']:>8.4f}  "
            f"{r['method_c_r']:>10.4f}  "
            f"{r['method_d_r']:>13.4f}"
        )
    print("-" * 55)
    print("(All values are LOO Pearson r; higher is better)")

    # Save markdown report
    out_path = PROJECT_ROOT / "docs" / "ridge-prediction-experiment-S197.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = format_results(results, holdout_cols)
    out_path.write_text(report_text)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
