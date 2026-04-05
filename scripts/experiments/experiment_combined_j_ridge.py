"""Combined J × Ridge sweep experiment.

Sweeps J = [40, 60, 80, 100, 120, 140, 160, 180, 200] and for each J computes:
  (a) Type-mean LOO r  — holdout_accuracy_county_prior_loo (honest baseline)
  (b) Ridge LOO r (scores only)  — RidgeCV(X=scores N×J), hat-matrix LOO
  (c) Ridge LOO r (scores + county_mean)  — RidgeCV(X=[scores, mean] N×(J+1))

Goal: find the optimal (J, prediction_method) pair.

Usage:
    uv run python scripts/experiment_combined_j_ridge.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types
from src.validation.validate_types import holdout_accuracy_county_prior_loo

# ── Configuration ──────────────────────────────────────────────────────────────

J_CANDIDATES = [40, 60, 80, 100, 120, 140, 160, 180, 200]
PRESIDENTIAL_WEIGHT = 8.0
MIN_YEAR = 2008
TEMPERATURE = 10.0
RIDGE_ALPHAS = np.logspace(-2, 4, 50)


# ── Column helpers ─────────────────────────────────────────────────────────────


def parse_start_year(col: str) -> int | None:
    """Return 4-digit start year from a shift col like 'pres_d_shift_08_12'."""
    parts = col.split("_")
    try:
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def classify_columns(
    all_cols: list[str], min_year: int = 2008
) -> tuple[list[str], list[str]]:
    """Split columns into training (start >= min_year) and holdout (pres_*_20_24)."""
    holdout = [c for c in all_cols if "20_24" in c]
    training = []
    for c in all_cols:
        if "20_24" in c:
            continue
        start = parse_start_year(c)
        if start is None or start >= min_year:
            training.append(c)
    return training, holdout


# ── Ridge LOO via augmented hat matrix ────────────────────────────────────────


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Ridge LOO predictions via augmented hat-matrix shortcut.

    Intercept is unpenalized (matching sklearn fit_intercept=True).
    Hat matrix: H = X_aug (X_aug'X_aug + P)^{-1} X_aug'
    where P = diag(0, alpha, alpha, ..., alpha).
    LOO: y_loo_i = y_i - e_i / (1 - H_ii)
    """
    N, P = X.shape
    X_aug = np.column_stack([np.ones(N), X])  # (N, P+1)

    pen = alpha * np.eye(P + 1)
    pen[0, 0] = 0.0  # unpenalized intercept

    A = X_aug.T @ X_aug + pen
    A_inv = np.linalg.inv(A)

    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)  # hat diagonal (N,)
    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    e = y - y_hat

    denom = np.where(np.abs(1.0 - h) < 1e-10, 1e-10, 1.0 - h)
    return y - e / denom


def ridge_loo_r(X: np.ndarray, y_cols: np.ndarray) -> dict:
    """Fit RidgeCV (GCV) per holdout dim and compute exact LOO Pearson r.

    Parameters
    ----------
    X       : (N, P) feature matrix
    y_cols  : (N, H) holdout targets

    Returns
    -------
    dict with mean_r, per_dim_r, best_alphas
    """
    H = y_cols.shape[1]
    per_dim_r: list[float] = []
    best_alphas: list[float] = []

    for h in range(H):
        y = y_cols[:, h]
        rcv = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        alpha = float(rcv.alpha_)
        best_alphas.append(alpha)

        y_loo = ridge_loo_predictions(X, y, alpha)

        if np.std(y) < 1e-10 or np.std(y_loo) < 1e-10:
            per_dim_r.append(0.0)
        else:
            r, _ = pearsonr(y, y_loo)
            per_dim_r.append(float(np.clip(r, -1.0, 1.0)))

    return {
        "mean_r": float(np.mean(per_dim_r)),
        "per_dim_r": per_dim_r,
        "best_alphas": best_alphas,
    }


# ── Main sweep ─────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("Combined J × Ridge Sweep — WetherVane")
    print("=" * 72)

    # Load data
    shifts_path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    print(f"\nLoading: {shifts_path}")
    df = pd.read_parquet(shifts_path)
    print(f"  {df.shape[0]} counties × {df.shape[1]} columns")

    all_cols = [c for c in df.columns if c != "county_fips"]
    training_col_names, holdout_col_names = classify_columns(all_cols, min_year=MIN_YEAR)

    print(f"  Training cols (start >= {MIN_YEAR}): {len(training_col_names)}")
    print(f"  Holdout cols: {holdout_col_names}")

    # Build matrices
    ordered_cols = training_col_names + holdout_col_names
    training_indices = list(range(len(training_col_names)))
    holdout_indices = list(range(len(training_col_names), len(ordered_cols)))

    raw_full = df[ordered_cols].values.astype(float)
    raw_training = raw_full[:, training_indices]
    holdout_raw = raw_full[:, holdout_indices]  # (N, H) — raw, for Ridge targets

    # Scale training, apply presidential weight post-scaling
    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(raw_training)

    pres_train_indices = [
        i for i, c in enumerate(training_col_names) if "pres_" in c
    ]
    scaled_training_weighted = scaled_training.copy()
    scaled_training_weighted[:, pres_train_indices] *= PRESIDENTIAL_WEIGHT

    print(
        f"  Presidential weight={PRESIDENTIAL_WEIGHT} applied to "
        f"{len(pres_train_indices)} training cols (post-scaling)"
    )

    # County training mean (raw, unscaled) — extra feature for Ridge(d)
    county_training_mean = raw_training.mean(axis=1)  # (N,)

    # For validate_types LOO: needs full raw matrix (raw training + raw holdout)
    # holdout_accuracy_county_prior_loo reads raw shift values from shift_matrix
    full_matrix_raw = raw_full  # (N, D_train + D_hold), all raw

    # ── Sweep ──────────────────────────────────────────────────────────────────
    print(f"\nSweeping J over {J_CANDIDATES}...")
    print("-" * 72)

    results = []

    for j in J_CANDIDATES:
        t0 = time.time()
        print(f"  J={j:3d}  KMeans...", end="", flush=True)

        # Discover types
        type_result = discover_types(
            scaled_training_weighted,
            j=j,
            temperature=TEMPERATURE,
            random_state=42,
        )
        scores = type_result.scores  # (N, J) soft membership

        # (a) Type-mean LOO r
        loo_res = holdout_accuracy_county_prior_loo(
            scores, full_matrix_raw, training_indices, holdout_indices
        )
        loo_r_a = loo_res["mean_r"]

        # (b) Ridge (scores only)
        X_b = scores  # (N, J)
        res_b = ridge_loo_r(X_b, holdout_raw)
        loo_r_b = res_b["mean_r"]

        # (c) Ridge (scores + county_training_mean)
        X_c = np.column_stack([scores, county_training_mean])  # (N, J+1)
        res_c = ridge_loo_r(X_c, holdout_raw)
        loo_r_c = res_c["mean_r"]

        elapsed = time.time() - t0
        print(
            f"  (a)LOO={loo_r_a:.4f}  (b)Ridge={loo_r_b:.4f}  "
            f"(c)Ridge+mean={loo_r_c:.4f}  [{elapsed:.1f}s]"
        )

        results.append({
            "j": j,
            "loo_r_a": loo_r_a,
            "loo_r_b": loo_r_b,
            "loo_r_c": loo_r_c,
            "per_dim_a": loo_res["per_dim_r"],
            "per_dim_b": res_b["per_dim_r"],
            "per_dim_c": res_c["per_dim_r"],
            "alphas_b": res_b["best_alphas"],
            "alphas_c": res_c["best_alphas"],
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    best_a_j = max(results, key=lambda x: x["loo_r_a"])["j"]
    best_b_j = max(results, key=lambda x: x["loo_r_b"])["j"]
    best_c_j = max(results, key=lambda x: x["loo_r_c"])["j"]

    best_overall = max(results, key=lambda x: max(x["loo_r_a"], x["loo_r_b"], x["loo_r_c"]))
    best_overall_method = max(
        [("a", best_overall["loo_r_a"]),
         ("b", best_overall["loo_r_b"]),
         ("c", best_overall["loo_r_c"])],
        key=lambda x: x[1],
    )

    print("\n" + "=" * 72)
    print("RESULTS — LOO Pearson r (higher is better)")
    print("=" * 72)
    print(
        f"\n{'J':>5}  {'(a) Type-mean':>13}  {'(b) Ridge scores':>16}  "
        f"{'(c) Ridge+mean':>14}  {'Best':>6}"
    )
    print(f"{'':>5}  {'LOO r':>13}  {'LOO r':>16}  {'LOO r':>14}")
    print("-" * 72)

    for row in results:
        best_val = max(row["loo_r_a"], row["loo_r_b"], row["loo_r_c"])
        best_m = (
            "(a)" if best_val == row["loo_r_a"]
            else "(b)" if best_val == row["loo_r_b"]
            else "(c)"
        )
        markers = []
        if row["j"] == best_a_j:
            markers.append("*A")
        if row["j"] == best_b_j:
            markers.append("*B")
        if row["j"] == best_c_j:
            markers.append("*C")
        if row["j"] == 100:
            markers.append("[curr]")
        marker_str = " ".join(markers)
        print(
            f"{row['j']:>5}  {row['loo_r_a']:>13.4f}  {row['loo_r_b']:>16.4f}  "
            f"{row['loo_r_c']:>14.4f}  {best_m}={best_val:.4f}  {marker_str}"
        )

    print("-" * 72)
    print(f"\n  Best (a) Type-mean LOO:    J={best_a_j}, r={max(r['loo_r_a'] for r in results):.4f}")
    print(f"  Best (b) Ridge scores:     J={best_b_j}, r={max(r['loo_r_b'] for r in results):.4f}")
    print(f"  Best (c) Ridge+mean:       J={best_c_j}, r={max(r['loo_r_c'] for r in results):.4f}")

    overall_best_r = max(
        max(r["loo_r_a"] for r in results),
        max(r["loo_r_b"] for r in results),
        max(r["loo_r_c"] for r in results),
    )
    print(f"\n  ==> OPTIMAL: J={best_overall['j']}, method ({best_overall_method[0]})={best_overall_method[1]:.4f}")
    print(f"      Baseline (J=100, type-mean LOO): 0.448")
    print(f"      Baseline (J=100, Ridge+mean):    0.533")
    print(f"      Improvement over Ridge baseline: {overall_best_r - 0.533:+.4f}")

    # Per-dimension breakdown
    print(f"\n{'Per-dimension LOO r at optimal J values':}")
    print(f"  Holdout dims: {holdout_col_names}")
    shown = set()
    for row in results:
        if row["j"] in {best_a_j, best_b_j, best_c_j, 100} and row["j"] not in shown:
            shown.add(row["j"])
            label = f"J={row['j']}"
            if row["j"] == 100:
                label += " [current]"
            print(f"  {label}")
            for idx, col in enumerate(holdout_col_names):
                print(
                    f"    {col:35s}  (a)={row['per_dim_a'][idx]:.4f}  "
                    f"(b)={row['per_dim_b'][idx]:.4f}  "
                    f"(c)={row['per_dim_c'][idx]:.4f}"
                )

    # Ridge alpha info at optimal J
    print(f"\n  Ridge GCV alphas at J={best_overall['j']}:")
    for idx, col in enumerate(holdout_col_names):
        ab = best_overall["alphas_b"][idx]
        ac = best_overall["alphas_c"][idx]
        print(f"    {col}: (b) α={ab:.2f}, (c) α={ac:.2f}")

    print("\n" + "=" * 72)

    # Save results
    out_dir = PROJECT_ROOT / "data" / "communities"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows = []
    for row in results:
        save_rows.append({
            "j": row["j"],
            "loo_r_type_mean": row["loo_r_a"],
            "loo_r_ridge_scores": row["loo_r_b"],
            "loo_r_ridge_mean": row["loo_r_c"],
        })
    save_df = pd.DataFrame(save_rows)
    out_path = out_dir / "combined_j_ridge_sweep.parquet"
    save_df.to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
