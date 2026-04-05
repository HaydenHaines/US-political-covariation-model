"""Audit covariance cross-race representation (Issue #87).

Research question: The type covariance Σ is built from all shift dimensions
(presidential + governor + Senate) treated identically. Presidential shifts
have lower raw variance (0.04) than governor (0.36) or Senate (0.15). After
StandardScaler normalization, each shift pair contributes equally — but the
underlying comovement patterns may differ by race type.

This script:
  1. Computes variance contribution of pres/gov/senate dims to total covariance
  2. Computes separate observed Ledoit-Wolf correlations for each race type
  3. Compares those per-race covariances to the current combined covariance
  4. Tests whether a presidential-weighted covariance improves cross-election
     forecast quality (leave-one-election-out validation)
  5. Reports findings clearly

Usage:
    uv run python scripts/audit_covariance_cross_race.py

Expected runtime: ~30 seconds on 3,154 counties, J=100 types.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load type scores and shift matrix. Returns (type_scores, shift_matrix, shift_cols, county_fips)."""
    ta = pd.read_parquet(DATA_DIR / "communities" / "type_assignments.parquet")
    score_cols = [c for c in ta.columns if c.endswith("_score") and c.startswith("type_")]
    type_scores = ta[score_cols].values  # (N, J)

    shifts_df = pd.read_parquet(DATA_DIR / "shifts" / "county_shifts_multiyear.parquet")
    # Exclude the holdout (2020→2024) from analysis
    holdout_cols = ["pres_d_shift_20_24", "pres_r_shift_20_24", "pres_turnout_shift_20_24"]
    shift_cols = [c for c in shifts_df.columns if c != "county_fips" and c not in holdout_cols]
    shift_matrix = shifts_df[shift_cols].values  # (N, D)
    county_fips = shifts_df["county_fips"].tolist()

    log.info("Loaded: %d counties, J=%d types, %d shift dims", len(county_fips), type_scores.shape[1], len(shift_cols))
    return type_scores, shift_matrix, shift_cols, county_fips


# ── Variance contribution analysis ───────────────────────────────────────────

def analyze_variance_contribution(shift_matrix: np.ndarray, shift_cols: list[str]) -> pd.DataFrame:
    """Break down variance contribution by race type (pres / gov / senate).

    Without any pre-scaling, raw variance per dim determines contribution.
    This shows how much each race type dominates raw Euclidean distance.
    """
    races = {"pres": [], "gov": [], "sen": []}
    for i, c in enumerate(shift_cols):
        for r in races:
            if c.startswith(r):
                races[r].append(i)
                break

    total_var = shift_matrix.var(axis=0).sum()

    rows = []
    for race, idxs in races.items():
        if not idxs:
            continue
        race_var = shift_matrix[:, idxs].var(axis=0).sum()
        n_dims = len(idxs)
        rows.append({
            "race_type": race,
            "n_dims": n_dims,
            "n_pairs": n_dims // 3,
            "raw_variance_sum": race_var,
            "pct_of_total_var": 100 * race_var / total_var,
            "mean_var_per_dim": race_var / n_dims,
        })
    return pd.DataFrame(rows)


# ── Type-shift computation ────────────────────────────────────────────────────

def compute_type_shifts(
    type_scores: np.ndarray,   # (N, J)
    shift_matrix: np.ndarray,  # (N, D)
    col_groups: list[list[int]],
) -> np.ndarray:
    """Compute J-vector of type-level shifts per election group.

    Returns shape (T, J) — one J-vector per election group.
    Weight is absolute type score to emphasize strongly-typed counties.
    """
    J = type_scores.shape[1]
    T = len(col_groups)
    weights = np.abs(type_scores)  # (N, J)
    w_sum = weights.sum(axis=0) + 1e-12  # (J,)

    type_shifts = np.zeros((T, J))
    for t, col_idxs in enumerate(col_groups):
        election_shift = shift_matrix[:, col_idxs].mean(axis=1)  # (N,) avg of 3 dims
        type_shifts[t] = (weights * election_shift[:, None]).sum(axis=0) / w_sum
    return type_shifts


# ── Ledoit-Wolf covariance estimation ────────────────────────────────────────

def ledoit_wolf_shrinkage(X: np.ndarray, S: np.ndarray) -> float:
    """Analytical Ledoit-Wolf optimal shrinkage intensity."""
    T, J = X.shape
    X_centered = X - X.mean(axis=0)
    mu = np.trace(S) / J
    delta = S - mu * np.eye(J)
    delta_sq_sum = np.sum(delta ** 2)
    b_bar = 0.0
    for k in range(T):
        xk = X_centered[k:k + 1, :]
        Mk = xk.T @ xk - S
        b_bar += np.sum(Mk ** 2)
    b_bar /= T ** 2
    if delta_sq_sum == 0:
        return 1.0
    return float(np.clip(b_bar / delta_sq_sum, 0.0, 1.0))


def enforce_pd(C: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Return nearest positive-definite matrix via spectral truncation."""
    C = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, floor)
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T + (eigvecs @ np.diag(eigvals) @ eigvecs.T).T) / 2.0


def compute_observed_corr(type_shifts: np.ndarray) -> np.ndarray:
    """Compute Ledoit-Wolf regularized correlation from type-shift matrix.

    type_shifts: (T, J) — T election pairs, J types.
    Returns (J, J) correlation matrix.
    """
    T, J = type_shifts.shape
    if T < 2:
        log.warning("Need >= 2 elections, got %d — returning identity", T)
        return np.eye(J)

    S = np.cov(type_shifts.T)  # (J, J)
    shrinkage = ledoit_wolf_shrinkage(type_shifts, S)
    log.debug("LW shrinkage=%.3f (T=%d, J=%d)", shrinkage, T, J)

    # Normalize to correlation
    obs_std = np.sqrt(np.diag(S))
    obs_std[obs_std == 0] = 1.0
    obs_corr = S / np.outer(obs_std, obs_std)
    np.fill_diagonal(obs_corr, 1.0)
    obs_corr = np.where(np.isnan(obs_corr), 0.0, obs_corr)

    # Shrink toward identity
    obs_corr = (1 - shrinkage) * obs_corr + shrinkage * np.eye(J)
    return enforce_pd(obs_corr)


# ── Per-race covariance analysis ──────────────────────────────────────────────

def compute_per_race_covariances(
    type_scores: np.ndarray,
    shift_matrix: np.ndarray,
    shift_cols: list[str],
) -> dict[str, np.ndarray]:
    """Compute separate type covariances for pres-only, gov-only, senate-only."""
    results = {}
    race_prefixes = {"pres": "pres_d", "gov": "gov_d", "sen": "sen_d"}

    for race_name, prefix in race_prefixes.items():
        # Select D-shift columns only for this race type (ignore R/turnout dups)
        idxs = [i for i, c in enumerate(shift_cols) if c.startswith(prefix)]
        if not idxs:
            log.warning("No %s D-shift columns found", race_name)
            continue

        # Each D-shift column = one election pair. Group as singletons.
        col_groups = [[i] for i in idxs]
        type_shifts = compute_type_shifts(type_scores, shift_matrix, col_groups)
        log.info("%s: %d election pairs → type_shifts shape %s", race_name, len(idxs), type_shifts.shape)

        results[race_name] = compute_observed_corr(type_shifts)

    return results


# ── Cross-race divergence: which types deviate most? ─────────────────────────

def analyze_per_type_cross_race_divergence(
    race_corrs: dict[str, np.ndarray],
    reference: str = "pres",
) -> pd.DataFrame:
    """Compare per-type row of each race's correlation to presidential.

    For each type j, compute mean absolute deviation of its row in
    the gov/senate correlation matrix vs the presidential matrix.

    High divergence = this type's comovement pattern differs across races.
    Low divergence = presidential covariance well-represents this type.
    """
    J = list(race_corrs.values())[0].shape[0]
    ref_corr = race_corrs.get(reference, np.eye(J))

    rows = []
    for race_name, corr in race_corrs.items():
        if race_name == reference:
            continue
        # Mean absolute deviation from reference, per type row
        row_mad = np.abs(corr - ref_corr).mean(axis=1)  # (J,)
        for j in range(J):
            rows.append({
                "type_id": j,
                "comparison": f"{race_name}_vs_{reference}",
                "row_mad": row_mad[j],
                "mean_self_corr": corr[j, :].mean(),
                "mean_pres_corr": ref_corr[j, :].mean(),
            })
    return pd.DataFrame(rows)


# ── Weighted covariance experiment ───────────────────────────────────────────

def compute_weighted_combined_covariance(
    type_scores: np.ndarray,
    shift_matrix: np.ndarray,
    shift_cols: list[str],
    pres_weight: float = 1.0,
    gov_weight: float = 1.0,
    sen_weight: float = 1.0,
) -> np.ndarray:
    """Compute a race-weighted blend of per-race covariances.

    Constructs weighted average of pres/gov/senate correlation matrices,
    then applies LW regularization to the blended result.

    Parameters
    ----------
    pres_weight, gov_weight, sen_weight:
        Relative weights for blending per-race correlations.
        Default = 1.0 for all (equal weight — equivalent to treating all
        elections the same, which is the current behavior).
    """
    race_corrs = compute_per_race_covariances(type_scores, shift_matrix, shift_cols)
    J = type_scores.shape[1]

    weights = {"pres": pres_weight, "gov": gov_weight, "sen": sen_weight}
    total_w = sum(w for r, w in weights.items() if r in race_corrs)

    if total_w == 0:
        return np.eye(J)

    blended = np.zeros((J, J))
    for race_name, corr in race_corrs.items():
        w = weights.get(race_name, 0.0) / total_w
        blended += w * corr

    return enforce_pd(blended)


# ── Leave-one-election-out (LOEO) validation ─────────────────────────────────

def loeo_validate(
    type_scores: np.ndarray,
    shift_matrix: np.ndarray,
    shift_cols: list[str],
    col_groups: list[list[int]],
    label: str = "combined",
) -> float:
    """Leave-one-election-out validation of a covariance construction strategy.

    For each election t, compute LW correlation from all elections except t,
    then compare off-diagonal structure to the full-data correlation.
    Returns mean off-diagonal Pearson r across left-out elections.

    Higher r = more stable covariance (better generalization).
    """
    J = type_scores.shape[1]
    mask = ~np.eye(J, dtype=bool)
    T = len(col_groups)

    if T < 3:
        log.warning("LOEO needs >= 3 elections, got %d for %s", T, label)
        return float("nan")

    # Full correlation (target)
    full_type_shifts = compute_type_shifts(type_scores, shift_matrix, col_groups)
    full_corr = compute_observed_corr(full_type_shifts)
    full_off = full_corr[mask]

    loeo_rs = []
    for t in range(T):
        train_groups = [g for i, g in enumerate(col_groups) if i != t]
        train_type_shifts = compute_type_shifts(type_scores, shift_matrix, train_groups)
        train_corr = compute_observed_corr(train_type_shifts)
        train_off = train_corr[mask]
        if np.std(train_off) < 1e-10 or np.std(full_off) < 1e-10:
            continue
        r = float(np.corrcoef(train_off, full_off)[0, 1])
        loeo_rs.append(r)

    result = float(np.mean(loeo_rs)) if loeo_rs else float("nan")
    log.info("LOEO r for %s: %.4f (from %d held-out elections)", label, result, len(loeo_rs))
    return result


# ── Main analysis ─────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("COVARIANCE CROSS-RACE AUDIT (Issue #87)")
    print("=" * 70)

    type_scores, shift_matrix, shift_cols, county_fips = load_data()
    J = type_scores.shape[1]
    N, D = shift_matrix.shape

    # --- 1. Variance contribution by race type ---
    print("\n--- 1. RAW VARIANCE CONTRIBUTION BY RACE TYPE ---")
    print("(Without StandardScaler — shows how much each race type dominates")
    print(" raw Euclidean distance in KMeans. Higher var = more influence.)\n")
    var_df = analyze_variance_contribution(shift_matrix, shift_cols)
    print(var_df.to_string(index=False))

    # Also check post-StandardScaler (what KMeans actually sees)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(shift_matrix)
    scaled_var_df = analyze_variance_contribution(scaled, shift_cols)
    print("\nPost-StandardScaler variance contribution:")
    print("(After scaling, each dim has var=1 — equal per dim, so pct of total")
    print(" is just n_dims/total_dims per race type.)\n")
    print(scaled_var_df[["race_type", "n_dims", "pct_of_total_var"]].to_string(index=False))

    # --- 2. Per-race separate covariance matrices ---
    print("\n--- 2. PER-RACE COVARIANCE MATRICES ---")
    print("(LW-regularized J=100 type correlation from pres-only, gov-only, senate-only D-shifts)\n")
    race_corrs = compute_per_race_covariances(type_scores, shift_matrix, shift_cols)

    for race_name, corr in race_corrs.items():
        off_diag = corr[~np.eye(J, dtype=bool)]
        print(f"  {race_name:8s}: mean off-diag={off_diag.mean():.4f}, "
              f"std={off_diag.std():.4f}, "
              f"range=[{off_diag.min():.4f}, {off_diag.max():.4f}]")

    # Cross-race correlation between off-diagonal elements
    print("\n  Cross-race correlation of off-diagonal elements:")
    mask = ~np.eye(J, dtype=bool)
    race_names = list(race_corrs.keys())
    for i, r1 in enumerate(race_names):
        for r2 in race_names[i + 1:]:
            off1 = race_corrs[r1][mask]
            off2 = race_corrs[r2][mask]
            r = np.corrcoef(off1, off2)[0, 1]
            print(f"    {r1} vs {r2}: r={r:.4f}")

    # --- 3. Load and compare current production covariance ---
    print("\n--- 3. CURRENT PRODUCTION COVARIANCE COMPARISON ---")
    current_cov = pd.read_parquet(DATA_DIR / "covariance" / "type_covariance.parquet").values
    sigma_base = 0.07
    current_corr = current_cov / (sigma_base ** 2)

    off_diag = current_corr[mask]
    print(f"  Current production Σ: mean off-diag={off_diag.mean():.4f}, "
          f"std={off_diag.std():.4f}")
    for race_name, corr in race_corrs.items():
        r = np.corrcoef(current_corr[mask], corr[mask])[0, 1]
        print(f"  Production vs {race_name}: r={r:.4f}")

    # --- 4. Cross-race divergence analysis ---
    print("\n--- 4. PER-TYPE CROSS-RACE DIVERGENCE ---")
    print("(Which types have most different comovement patterns across races?)\n")
    divergence_df = analyze_per_type_cross_race_divergence(race_corrs, reference="pres")
    for comparison, grp in divergence_df.groupby("comparison"):
        top10 = grp.nlargest(10, "row_mad")[["type_id", "row_mad", "mean_self_corr", "mean_pres_corr"]]
        print(f"  {comparison} — top 10 most divergent types:")
        print(top10.to_string(index=False))
        print(f"  Mean divergence (all types): {grp['row_mad'].mean():.4f}")
        print(f"  Median divergence: {grp['row_mad'].median():.4f}")
        print()

    # --- 5. LOEO validation comparison ---
    print("\n--- 5. LEAVE-ONE-ELECTION-OUT VALIDATION ---")
    print("(Higher r = covariance structure generalizes better across elections)\n")

    holdout_cols_set = {"pres_d_shift_20_24", "pres_r_shift_20_24", "pres_turnout_shift_20_24"}

    # Strategy A: current approach — all election pairs grouped by 3
    all_groups = [list(range(i, min(i + 3, D))) for i in range(0, D, 3)]
    loeo_all = loeo_validate(type_scores, shift_matrix, shift_cols, all_groups, "current (all races equal)")

    # Strategy B: presidential-only covariance
    pres_d_idxs = [i for i, c in enumerate(shift_cols) if c.startswith("pres_d")]
    pres_groups = [[i] for i in pres_d_idxs]
    loeo_pres = loeo_validate(type_scores, shift_matrix, shift_cols, pres_groups, "pres-only")

    # Strategy C: governor-only covariance
    gov_d_idxs = [i for i, c in enumerate(shift_cols) if c.startswith("gov_d")]
    gov_groups = [[i] for i in gov_d_idxs]
    loeo_gov = loeo_validate(type_scores, shift_matrix, shift_cols, gov_groups, "gov-only")

    # Strategy D: senate-only covariance
    sen_d_idxs = [i for i, c in enumerate(shift_cols) if c.startswith("sen_d")]
    sen_groups = [[i] for i in sen_d_idxs]
    loeo_sen = loeo_validate(type_scores, shift_matrix, shift_cols, sen_groups, "senate-only")

    # Strategy E: pres + senate (no gov) — test if gov is noise
    pres_sen_idxs = pres_d_idxs + sen_d_idxs
    pres_sen_groups = [[i] for i in sorted(pres_sen_idxs)]
    loeo_pres_sen = loeo_validate(type_scores, shift_matrix, shift_cols, pres_sen_groups, "pres+senate")

    print(f"\n  LOEO r summary:")
    print(f"    Current (all equal):    {loeo_all:.4f}")
    print(f"    Presidential only:      {loeo_pres:.4f}")
    print(f"    Governor only:          {loeo_gov:.4f}")
    print(f"    Senate only:            {loeo_sen:.4f}")
    print(f"    Presidential + Senate:  {loeo_pres_sen:.4f}")

    # --- 6. Weighted covariance blend test ---
    print("\n--- 6. REWEIGHTED COVARIANCE BLEND TEST ---")
    print("(Test whether blending per-race covariances with different weights improves LOEO)\n")

    weight_configs = [
        ("equal (baseline)", 1.0, 1.0, 1.0),
        ("pres x2", 2.0, 1.0, 1.0),
        ("pres x4 (current discovery weight)", 4.0, 1.0, 1.0),
        ("pres x8 (current discovery weight)", 8.0, 1.0, 1.0),
        ("pres only", 1.0, 0.0, 0.0),
        ("pres+sen", 1.0, 0.0, 1.0),
        ("no gov (pres+sen equal)", 1.0, 0.0, 1.0),
    ]

    for label, pw, gw, sw in weight_configs:
        blended = compute_weighted_combined_covariance(
            type_scores, shift_matrix, shift_cols, pw, gw, sw
        )
        off = blended[mask]
        # Compare to current production
        r_vs_prod = np.corrcoef(off, current_corr[mask])[0, 1]
        print(f"  {label:40s}: mean_off={off.mean():.4f}, "
              f"r_vs_production={r_vs_prod:.4f}")

    # --- 7. Summary and recommendation ---
    print("\n--- 7. FINDINGS SUMMARY ---\n")

    pres_vs_gov = np.corrcoef(
        race_corrs["pres"][mask] if "pres" in race_corrs else np.zeros(J * J - J),
        race_corrs["gov"][mask] if "gov" in race_corrs else np.zeros(J * J - J),
    )[0, 1]
    pres_vs_sen = np.corrcoef(
        race_corrs["pres"][mask] if "pres" in race_corrs else np.zeros(J * J - J),
        race_corrs["sen"][mask] if "sen" in race_corrs else np.zeros(J * J - J),
    )[0, 1]

    print(f"  Presidential covariance agrees with governor covariance:  r={pres_vs_gov:.4f}")
    print(f"  Presidential covariance agrees with Senate covariance:    r={pres_vs_sen:.4f}")
    print()

    if pres_vs_gov > 0.7 and pres_vs_sen > 0.7:
        conclusion = "WEAK CONCERN: Per-race covariances are highly correlated — presidential dominance is not causing systematic errors in cross-race comovement representation."
    elif pres_vs_gov < 0.5 or pres_vs_sen < 0.5:
        conclusion = "SIGNIFICANT CONCERN: Per-race covariances diverge substantially — the combined Σ may understate Senate/governor cross-type comovement patterns."
    else:
        conclusion = "MODERATE CONCERN: Per-race covariances show some divergence — worth investigating targeted reweighting, but current approach is not severely biased."

    print(f"  Conclusion: {conclusion}\n")

    if loeo_pres > loeo_all + 0.01:
        print("  RECOMMENDATION: Presidential-only covariance outperforms combined (by >0.01 LOEO r).")
        print("  Consider removing governor/senate from covariance construction.")
    elif loeo_all > loeo_pres + 0.01:
        print("  RECOMMENDATION: Combined covariance outperforms presidential-only (by >0.01 LOEO r).")
        print("  Current approach is working — do not restrict to presidential.")
    else:
        print("  RECOMMENDATION: No substantial difference between approaches.")
        print("  Current approach is adequate. Issue #87 can be downgraded to low priority.")


if __name__ == "__main__":
    import os
    os.chdir(PROJECT_ROOT)
    main()
