"""Experiment: temperature-scaled soft membership for calibration improvement.

Hypothesis: sharpening the inverse-distance membership weights (via temperature
parameter T) reduces cross-type averaging and improves calibration.

Current formula: weight = 1/(dist + 1e-10)  → row-normalized  (T=1.0)
New formula:     weight = (1/(dist + 1e-10))^T → row-normalized

At T=1.0 this is identical to the current model.
At T→∞ this approaches hard assignment (all weight on nearest centroid).

Usage:
    cd /home/hayden/projects/US-political-covariation-model
    uv run python scripts/experiment_soft_membership.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Paths — always read data from the main repo root regardless of worktree
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/hayden/projects/US-political-covariation-model")

SHIFTS_PATH = DATA_ROOT / "data/shifts/county_shifts_multiyear.parquet"
TYPE_PRIORS_PATH = DATA_ROOT / "data/communities/type_priors.parquet"
TYPE_PROFILES_PATH = DATA_ROOT / "data/communities/type_profiles.parquet"
SUPER_TYPES_PATH = DATA_ROOT / "data/communities/super_types.parquet"
ACTUALS_PATH = DATA_ROOT / "data/assembled/medsl_county_presidential_2024.parquet"
OUTPUT_DIR = DATA_ROOT / "data/validation"

# KMeans hyperparameters — must exactly match run_type_discovery.py
J = 20
KMEANS_SEED = 42
KMEANS_N_INIT = 10
PRES_WEIGHT = 2.5
MIN_YEAR = 2008

# Holdout columns excluded from training (2020→2024 shift)
HOLDOUT_COLUMNS = [
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
]

TEMPERATURES = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 999.0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_shift_matrix() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load and weight the county shift matrix used for KMeans training.

    Returns
    -------
    X : ndarray of shape (N, D)
        Weighted, standardised shift matrix fed to KMeans.
    shift_cols : list[str]
        Column names selected (2008+, excluding holdout).
    county_fips : ndarray of shape (N,)
        County FIPS codes (string).
    """
    df = pd.read_parquet(SHIFTS_PATH)
    county_fips = df["county_fips"].astype(str).str.zfill(5).values

    all_cols = [c for c in df.columns if c != "county_fips" and c not in HOLDOUT_COLUMNS]

    # Filter to 2008+ (parse year from column name: pres_d_shift_08_12 → y1=2008)
    shift_cols = []
    for c in all_cols:
        parts = c.split("_")
        try:
            si = parts.index("shift")
            y1 = int("20" + parts[si + 1])
        except (ValueError, IndexError):
            continue
        if y1 >= MIN_YEAR:
            shift_cols.append(c)

    X_raw = df[shift_cols].values.astype(float)

    # Apply presidential weight: multiply pres_* cols by PRES_WEIGHT
    weights_vec = np.ones(len(shift_cols))
    for i, c in enumerate(shift_cols):
        if c.startswith("pres_"):
            weights_vec[i] = PRES_WEIGHT

    # State-center governor and Senate columns
    # (subtract per-state mean so they measure deviations from state baseline)
    state_prefix = np.array([f[:2] for f in county_fips])
    gov_sen_mask = np.array([
        c.startswith("gov_") or c.startswith("sen_") for c in shift_cols
    ])
    if gov_sen_mask.any():
        X_centered = X_raw.copy()
        for prefix in np.unique(state_prefix):
            idx = state_prefix == prefix
            col_idx = np.where(gov_sen_mask)[0]
            X_centered[np.ix_(idx, col_idx)] -= X_raw[np.ix_(idx, col_idx)].mean(axis=0)
        X_raw = X_centered

    # Apply column weights
    X = X_raw * weights_vec[None, :]

    return X, shift_cols, county_fips


def compute_centroid_distances(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run KMeans and return raw Euclidean distances to each centroid.

    Parameters
    ----------
    X : ndarray of shape (N, D)

    Returns
    -------
    dists : ndarray of shape (N, J)
        Euclidean distance from each county to each centroid.
    labels : ndarray of shape (N,)
        Hard cluster assignment.
    """
    km = KMeans(n_clusters=J, random_state=KMEANS_SEED, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_  # (J, D)

    dists = np.zeros((len(X), J))
    for t in range(J):
        dists[:, t] = np.linalg.norm(X - centroids[t], axis=1)

    return dists, labels


def load_type_priors() -> np.ndarray:
    """Return type priors array indexed by type_id (length J)."""
    priors_df = pd.read_parquet(TYPE_PRIORS_PATH)
    priors = np.full(J, 0.45)
    for _, row in priors_df.iterrows():
        t = int(row["type_id"])
        if t < J:
            priors[t] = float(row["prior_dem_share"])
    return priors


def load_actuals(county_fips: np.ndarray) -> pd.Series:
    """Return 2024 actual Dem share indexed by county_fips (already zfilled)."""
    df = pd.read_parquet(ACTUALS_PATH)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df = df.set_index("county_fips")["pres_dem_share_2024"]
    return df


def load_super_type_map() -> dict[int, int]:
    """Return mapping from type_id → super_type_id."""
    tp = pd.read_parquet(TYPE_PROFILES_PATH)
    return dict(zip(tp["type_id"], tp["super_type_id"]))


def load_super_type_names() -> dict[int, str]:
    """Return mapping from super_type_id → display_name."""
    st = pd.read_parquet(SUPER_TYPES_PATH)
    return dict(zip(st["super_type_id"], st["display_name"]))


# ---------------------------------------------------------------------------
# Core experiment helpers
# ---------------------------------------------------------------------------


def temperature_soft_membership(dists: np.ndarray, T: float) -> np.ndarray:
    """Compute temperature-sharpened soft membership from centroid distances.

    Formula: weight_j = (1 / (dist_j + eps))^T, then row-normalize.

    Numerically: we work in log-space to avoid overflow at large T, then
    apply a stable softmax.  For T >= 500 we fall back to hard (argmax)
    assignment directly to avoid any residual floating-point issues.

    Parameters
    ----------
    dists : ndarray of shape (N, J)
        Euclidean distances to centroids (non-negative).
    T : float
        Temperature exponent.  T=1.0 = current baseline.  T→∞ = hard assignment.

    Returns
    -------
    scores : ndarray of shape (N, J)
        Non-negative soft membership weights, each row sums to 1.
    """
    N, J_ = dists.shape
    eps = 1e-10

    # Hard assignment shortcut for very large T (avoids fp overflow entirely)
    if T >= 500.0:
        scores = np.zeros((N, J_))
        nearest = np.argmin(dists, axis=1)
        scores[np.arange(N), nearest] = 1.0
        return scores

    # Log-space computation: log(weight_j) = T * log(1/(dist_j + eps))
    #                                       = -T * log(dist_j + eps)
    log_weights = -T * np.log(dists + eps)  # (N, J)

    # Numerically stable softmax: subtract row max before exponentiating
    log_weights -= log_weights.max(axis=1, keepdims=True)
    powered = np.exp(log_weights)
    row_sums = powered.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return powered / row_sums


def county_predictions(scores: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Compute county-level predicted Dem share from soft membership.

    pred_i = sum_j(score_ij * prior_j)   (scores already sum to 1)

    Parameters
    ----------
    scores : ndarray of shape (N, J)
    priors : ndarray of shape (J,)

    Returns
    -------
    ndarray of shape (N,), clipped to [0, 1].
    """
    pred = (scores * priors[None, :]).sum(axis=1)
    return np.clip(pred, 0.0, 1.0)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, Pearson r, bias, and prediction range stats."""
    diff = predicted - actual
    r, _ = stats.pearsonr(actual, predicted) if len(actual) >= 2 else (float("nan"), None)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "pearson_r": float(r),
        "bias": float(np.mean(diff)),          # positive = model too high (over-Dem)
        "pred_min": float(predicted.min()),
        "pred_max": float(predicted.max()),
        "pred_range": float(predicted.max() - predicted.min()),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep() -> tuple[pd.DataFrame, int]:
    """Run the temperature sweep and return results DataFrame + best T index."""
    print("Loading shift matrix and running KMeans...")
    X, shift_cols, county_fips = load_shift_matrix()
    print(f"  Shift matrix: {X.shape[0]} counties x {X.shape[1]} dims")

    dists, labels = compute_centroid_distances(X)
    print(f"  KMeans done. Type sizes (sorted): {sorted(np.bincount(labels).tolist(), reverse=True)}")

    priors = load_type_priors()
    actuals_series = load_actuals(county_fips)

    # Align actuals to county_fips order (keep only counties with actual data)
    valid_mask = np.array([f in actuals_series.index for f in county_fips])
    fips_valid = county_fips[valid_mask]
    dists_valid = dists[valid_mask]
    actuals = actuals_series.loc[fips_valid].values.astype(float)

    print(f"  Counties with 2024 actual data: {valid_mask.sum()} / {len(county_fips)}")
    print(f"  Actual Dem share range: [{actuals.min():.3f}, {actuals.max():.3f}]")
    print()

    rows = []
    for T in TEMPERATURES:
        scores = temperature_soft_membership(dists_valid, T)
        pred = county_predictions(scores, priors)
        m = compute_metrics(actuals, pred)
        m["temperature"] = T
        rows.append(m)

    results = pd.DataFrame(rows)[
        ["temperature", "mae", "rmse", "pearson_r", "bias", "pred_min", "pred_max", "pred_range"]
    ]

    best_idx = int(results["mae"].argmin())
    return results, best_idx, dists_valid, priors, actuals, fips_valid, labels[valid_mask]


def per_super_type_bias(
    dists: np.ndarray,
    priors: np.ndarray,
    actuals: np.ndarray,
    fips: np.ndarray,
    hard_labels: np.ndarray,
    T: float,
) -> pd.DataFrame:
    """Compute per-super-type bias at a given temperature.

    Uses the dominant type (hard assignment) to group counties into super-types,
    then reports mean bias within each super-type group.

    Parameters
    ----------
    dists : ndarray of shape (N, J)
    priors : ndarray of shape (J,)
    actuals : ndarray of shape (N,)
    fips : ndarray of shape (N,)
    hard_labels : ndarray of shape (N,)  — hard KMeans labels
    T : float

    Returns
    -------
    DataFrame with columns: super_type_id, super_type_name, n_counties,
        mean_actual, mean_pred, bias, mae.
    """
    type_to_super = load_super_type_map()
    super_names = load_super_type_names()

    scores = temperature_soft_membership(dists, T)
    pred = county_predictions(scores, priors)

    super_labels = np.array([type_to_super.get(int(lbl), -1) for lbl in hard_labels])

    rows = []
    for sid in sorted(set(super_labels)):
        mask = super_labels == sid
        name = super_names.get(int(sid), f"super_{sid}")
        act_sub = actuals[mask]
        pred_sub = pred[mask]
        bias = float(np.mean(pred_sub - act_sub))
        rows.append({
            "super_type_id": int(sid),
            "super_type_name": name,
            "n_counties": int(mask.sum()),
            "mean_actual": float(act_sub.mean()),
            "mean_pred": float(pred_sub.mean()),
            "bias": bias,
            "mae": float(np.mean(np.abs(pred_sub - act_sub))),
        })

    return pd.DataFrame(rows).sort_values("bias", ascending=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    results, best_idx, dists, priors, actuals, fips, hard_labels = run_sweep()

    # Print sweep table
    print("=" * 80)
    print("SOFT MEMBERSHIP TEMPERATURE SWEEP — 2024 Presidential Calibration")
    print("=" * 80)
    print(f"{'T':>7}  {'MAE':>7}  {'RMSE':>7}  {'Pearson r':>9}  {'Bias':>7}  {'Pred range':>12}")
    print("-" * 72)
    for _, row in results.iterrows():
        marker = " <-- BEST" if int(row.name) == best_idx else ""
        print(
            f"{row['temperature']:>7.1f}  "
            f"{row['mae']:>7.4f}  "
            f"{row['rmse']:>7.4f}  "
            f"{row['pearson_r']:>9.4f}  "
            f"{row['bias']:>+7.4f}  "
            f"[{row['pred_min']:.3f}, {row['pred_max']:.3f}]{marker}"
        )
    print()

    best_T = float(results.iloc[best_idx]["temperature"])
    print(f"Best temperature: T={best_T}")
    print()

    # Per-super-type bias at best T
    print(f"Per-super-type bias at T={best_T}:")
    print("-" * 72)
    st_df = per_super_type_bias(dists, priors, actuals, fips, hard_labels, best_T)
    for _, row in st_df.iterrows():
        print(
            f"  {row['super_type_name']:<30}  n={row['n_counties']:>3}  "
            f"actual={row['mean_actual']:.3f}  pred={row['mean_pred']:.3f}  "
            f"bias={row['bias']:+.4f}  mae={row['mae']:.4f}"
        )
    print()

    # Also show baseline (T=1.0) vs best for comparison
    baseline = results[results["temperature"] == 1.0].iloc[0]
    best = results.iloc[best_idx]
    print("Improvement over baseline (T=1.0):")
    print(f"  MAE:      {baseline['mae']:.4f} → {best['mae']:.4f}  ({best['mae'] - baseline['mae']:+.4f})")
    print(f"  RMSE:     {baseline['rmse']:.4f} → {best['rmse']:.4f}  ({best['rmse'] - baseline['rmse']:+.4f})")
    print(f"  Pearson r:{baseline['pearson_r']:.4f} → {best['pearson_r']:.4f}  ({best['pearson_r'] - baseline['pearson_r']:+.4f})")
    print(f"  Bias:     {baseline['bias']:+.4f} → {best['bias']:+.4f}")
    print(f"  Pred range: [{baseline['pred_min']:.3f},{baseline['pred_max']:.3f}] → [{best['pred_min']:.3f},{best['pred_max']:.3f}]")
    print()

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "soft_membership_sweep.csv"
    results.to_csv(out_path, index=False)
    print(f"Sweep results saved to {out_path}")

    # Recommendation
    print()
    print("RECOMMENDATION")
    print("-" * 40)
    if best_T == 1.0:
        print("  Current T=1.0 is already optimal. No change recommended.")
    else:
        mae_improvement = baseline["mae"] - best["mae"]
        range_improvement = best["pred_range"] - baseline["pred_range"]
        print(f"  Use T={best_T:.1f}.")
        print(f"  MAE improves by {mae_improvement:.4f} ({mae_improvement/baseline['mae']*100:.1f}%).")
        print(f"  Prediction range widens by {range_improvement:.3f} "
              f"(from {baseline['pred_range']:.3f} to {best['pred_range']:.3f}),")
        print(f"  better matching the actual range of [{actuals.min():.3f}, {actuals.max():.3f}].")


if __name__ == "__main__":
    main()
