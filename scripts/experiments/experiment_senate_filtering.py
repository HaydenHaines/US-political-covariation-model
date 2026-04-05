"""Experiment: Does filtering low-signal Senate pairs improve holdout accuracy?

Senate elections are noisy: incumbent advantage, candidate quality effects,
uncontested races, and 6-year cycles all inject noise. Hypothesis: removing
low-signal Senate shift pairs sharpens type discovery and improves holdout r.

Strategies tested:
  A. Baseline        -- All 33 dims (current production, min_year=2008)
  B. Drop all Senate -- Presidential + Governor only
  C. Drop low-var Senate -- Drop Senate pairs with std < median Senate std
  D. Drop high-zero Senate -- Drop Senate pairs where >20% counties have zero values
  E. Presidential only -- Just presidential shift pairs

Each strategy runs KMeans J=100, StandardScaler, presidential_weight=8.0.
Evaluates holdout r (type-mean prior) and coherence against 2020→2024 presidential shifts.

Usage:
    uv run python scripts/experiment_senate_filtering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types, temperature_soft_membership
from src.validation.validate_types import holdout_accuracy, type_coherence

# ── constants matching production ────────────────────────────────────────────
J = 100
N_INIT = 10
RANDOM_STATE = 42
PRESIDENTIAL_WEIGHT = 8.0
TEMPERATURE = 10.0
MIN_YEAR = 2008  # match production filter

HOLDOUT_COLS_NAMES = ["pres_d_shift_20_24", "pres_r_shift_20_24", "pres_turnout_shift_20_24"]


def parse_start_year(col: str) -> int:
    """Parse the start year from a shift column name like 'pres_d_shift_08_12'."""
    parts = col.split("_")
    y2_str = parts[-2]   # e.g. '08'
    y2 = int(y2_str)
    # Two-digit year: >=50 → 1900s, <50 → 2000s
    y1 = y2 + (1900 if y2 >= 50 else 2000)
    return y1


def load_shifts() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the shift parquet and return (df, training_cols, holdout_cols)."""
    path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(path)

    all_non_fips = [c for c in df.columns if c != "county_fips"]
    holdout = [c for c in all_non_fips if c in HOLDOUT_COLS_NAMES]
    training_unfiltered = [c for c in all_non_fips if c not in holdout]

    # Apply min_year filter (matching production)
    training = []
    for c in training_unfiltered:
        try:
            y1 = parse_start_year(c)
            if y1 >= MIN_YEAR:
                training.append(c)
        except (ValueError, IndexError):
            training.append(c)  # keep if can't parse

    return df, training, holdout


def analyze_senate_columns(df: pd.DataFrame, training_cols: list[str]) -> pd.DataFrame:
    """Compute per-column signal metrics for Senate shift pairs."""
    sen_cols = [c for c in training_cols if "sen_d_shift" in c]

    rows = []
    for col in sen_cols:
        vals = df[col].values
        std = float(np.std(vals))
        zero_frac = float(np.mean(np.abs(vals) < 1e-9))  # effectively zero

        # Correlation with presidential d_shift of the same period
        # Senate period e.g. sen_d_shift_08_14 spans pres elections inside it.
        # Match to nearest presidential training column with same start year.
        col_parts = col.split("_")
        sen_start = col_parts[-2]  # e.g. '08'
        pres_match = None
        for tc in training_cols:
            if "pres_d_shift" in tc and tc.split("_")[-2] == sen_start:
                pres_match = tc
                break

        if pres_match is not None:
            pres_vals = df[pres_match].values
            if np.std(vals) > 1e-10 and np.std(pres_vals) > 1e-10:
                corr = float(np.corrcoef(vals, pres_vals)[0, 1])
            else:
                corr = float("nan")
        else:
            corr = float("nan")

        rows.append({"col": col, "std": std, "zero_frac": zero_frac, "corr_with_pres": corr})

    return pd.DataFrame(rows)


def run_strategy(
    df: pd.DataFrame,
    training_cols: list[str],
    holdout_cols: list[str],
    label: str,
) -> dict:
    """Run KMeans discovery and evaluate for a given column selection."""
    print(f"\n  [{label}] {len(training_cols)} training dims...")

    X_train = df[training_cols].values
    X_holdout = df[holdout_cols].values

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Presidential weight (post-scaling)
    pres_indices = [i for i, c in enumerate(training_cols) if "pres_" in c]
    if pres_indices and PRESIDENTIAL_WEIGHT != 1.0:
        X_scaled[:, pres_indices] *= PRESIDENTIAL_WEIGHT

    # KMeans
    km = KMeans(n_clusters=J, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_

    # Soft membership (T=10)
    dists = np.zeros((len(X_scaled), J))
    for t in range(J):
        dists[:, t] = np.linalg.norm(X_scaled - centroids[t], axis=1)
    scores = temperature_soft_membership(dists, T=TEMPERATURE)

    dominant_types = np.argmax(scores, axis=1)

    # Full matrix for validation (training + holdout combined)
    full_matrix = np.hstack([X_train, X_holdout])
    holdout_indices_in_full = list(range(X_train.shape[1], full_matrix.shape[1]))

    # Holdout accuracy (type-mean prior)
    acc = holdout_accuracy(scores, full_matrix, holdout_indices_in_full, dominant_types)

    # Coherence
    coh = type_coherence(scores, full_matrix, holdout_indices_in_full)

    return {
        "label": label,
        "n_dims": len(training_cols),
        "holdout_r": acc["mean_r"],
        "per_dim_r": acc["per_dim_r"],
        "coherence": coh["mean_ratio"],
    }


def main() -> None:
    print("=" * 70)
    print("Senate Filtering Experiment")
    print(f"  J={J}, pres_weight={PRESIDENTIAL_WEIGHT}, T={TEMPERATURE}, min_year={MIN_YEAR}")
    print("=" * 70)

    df, training_cols, holdout_cols = load_shifts()
    print(f"\nLoaded {len(df)} counties, {len(training_cols)} training dims, {len(holdout_cols)} holdout dims")

    # Classify columns
    pres_cols = [c for c in training_cols if "pres_d_shift" in c or "pres_r_shift" in c or "pres_turnout_shift" in c]
    gov_cols = [c for c in training_cols if "gov_" in c]
    sen_cols = [c for c in training_cols if "sen_" in c]

    print(f"\nColumn breakdown:")
    print(f"  Presidential: {len(pres_cols)} cols — {[c for c in pres_cols]}")
    print(f"  Governor:     {len(gov_cols)} cols — {[c for c in gov_cols]}")
    print(f"  Senate:       {len(sen_cols)} cols — {[c for c in sen_cols]}")

    # Analyze Senate column signal
    print("\n--- Senate Column Signal Analysis ---")
    sen_analysis = analyze_senate_columns(df, training_cols)
    print(sen_analysis.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    median_sen_std = sen_analysis["std"].median()
    print(f"\nMedian Senate std: {median_sen_std:.4f}")

    # Build column sets for each strategy
    # Low-var Senate: std < median → drop those
    low_var_sen_cols_set = set(
        sen_analysis.loc[sen_analysis["std"] < median_sen_std, "col"].tolist()
    )
    # Expand to include r and turnout triplets
    low_var_sen_full = []
    for c in sen_cols:
        base = "_".join(c.split("_")[:-2] + ["d_shift"] + c.split("_")[-2:])  # normalize
        # Check if the d_shift version is low-var
        d_col = c.replace("_r_shift_", "_d_shift_").replace("_turnout_shift_", "_d_shift_")
        if d_col in low_var_sen_cols_set:
            low_var_sen_full.append(c)

    # High-zero Senate: zero_frac > 0.20 → drop those
    high_zero_sen_d_set = set(
        sen_analysis.loc[sen_analysis["zero_frac"] > 0.20, "col"].tolist()
    )
    high_zero_sen_full = []
    for c in sen_cols:
        d_col = c.replace("_r_shift_", "_d_shift_").replace("_turnout_shift_", "_d_shift_")
        if d_col in high_zero_sen_d_set:
            high_zero_sen_full.append(c)

    print(f"\nLow-var Senate d_cols (std < median): {sorted(low_var_sen_cols_set)}")
    print(f"Low-var Senate full set (incl r/turnout): {len(low_var_sen_full)} cols")
    print(f"High-zero Senate d_cols (zero_frac>0.20): {sorted(high_zero_sen_d_set)}")
    print(f"High-zero Senate full set (incl r/turnout): {len(high_zero_sen_full)} cols")

    # Strategy column sets
    strategies = {
        "A. Baseline (all 33)": training_cols,
        "B. Drop all Senate": [c for c in training_cols if "sen_" not in c],
        "C. Drop low-var Senate": [c for c in training_cols if c not in low_var_sen_full],
        "D. Drop high-zero Senate": [c for c in training_cols if c not in high_zero_sen_full],
        "E. Presidential only": [c for c in training_cols if "pres_" in c],
    }

    print("\n--- Running Strategies ---")
    results = []
    for label, cols in strategies.items():
        result = run_strategy(df, cols, holdout_cols, label)
        results.append(result)

    # Print comparison table
    print("\n")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Strategy':<35} {'Dims':>5} {'Holdout r':>10} {'Coherence':>10}"
    print(header)
    print("-" * 70)

    baseline_r = results[0]["holdout_r"]
    baseline_coh = results[0]["coherence"]

    for r in results:
        delta_r = r["holdout_r"] - baseline_r
        delta_coh = r["coherence"] - baseline_coh
        delta_str = f"({delta_r:+.4f})" if delta_r != 0 else "        "
        print(
            f"{r['label']:<35} {r['n_dims']:>5} {r['holdout_r']:>10.4f} {delta_str:<12} {r['coherence']:>10.4f}"
        )

    print("-" * 70)
    print(f"\nBaseline (production):   holdout r={baseline_r:.4f}, coherence={baseline_coh:.4f}")
    print(f"Production target:       holdout r=0.698, coherence=0.783")
    print()

    # Per-dim holdout r detail
    print("Per-holdout-dim r breakdown (pres_d, pres_r, pres_turnout):")
    print(f"  {'Strategy':<35}  d_shift    r_shift  turnout")
    for r in results:
        dims = r["per_dim_r"]
        if len(dims) >= 3:
            print(f"  {r['label']:<35}  {dims[0]:>8.4f}  {dims[1]:>8.4f}  {dims[2]:>8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
