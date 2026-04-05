"""Feature importance analysis for Ridge+demo model (LOO r ~0.717).

Identifies which of the ~114 demographic features in county_features_national.parquet
contribute to (or hurt) the LOO r of the Ridge+demo model. Runs the exact same
preprocessing as holdout_accuracy_ridge_augmented(): inner join on FIPS, NaN imputation
with column means, z-score standardization, PCA(n=15, whiten=True) before KMeans,
hat-matrix LOO.

Approach
--------
1. Compute baseline LOO r with all features (should be ~0.717).
2. For each demographic feature, compute LOO r with THAT FEATURE REMOVED.
3. Marginal contribution = baseline_LOO_r - LOO_r_without_feature.
   - Positive: feature helps (removing hurts).
   - Negative: feature HURTS (removing improves LOO r — drop it).
4. Also test dropping features by logical group (data source / topic).

This mirrors the collinearity finding (S303) where FEC donor density + BEA income
composition features (6 total) hurt LOO r by introducing noisy signal. More such
features likely exist in the full 114-feature set.

Usage
-----
    uv run python scripts/experiment_feature_importance.py

Runtime estimate: ~2–4 minutes for 114 individual feature ablations.

Outputs
-------
    - Baseline LOO r (target: ~0.717)
    - Ranked table of features by marginal contribution
    - Features that HURT performance (drop candidates)
    - Recommended feature set after pruning

Notes
-----
- PCA is applied to training_scaled (shift vectors) BEFORE KMeans, not to demos.
- The LOO is over 2020→2024 presidential holdout (3 dimensions: D shift, R shift, turnout).
- Mean LOO r = average across the 3 holdout dimensions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.run_type_discovery import discover_types

# ── Constants matching production config ─────────────────────────────────────

# KMeans hyperparameters (from PCA-whitening experiment, now production)
J = 100
TEMPERATURE = 10.0
PRESIDENTIAL_WEIGHT = 8.0
RANDOM_STATE = 42
PCA_N_COMPONENTS = 15  # PCA whitening before KMeans (S305)

# Alpha sweep (matches holdout_accuracy_ridge.py)
ALPHAS = np.logspace(-3, 6, 100)

# Holdout: 2020→2024 presidential shifts
HOLDOUT_MARKER = "20_24"
MIN_YEAR = 2008  # earliest training shift pair to include


# ── Column helpers (same as in existing experiment scripts) ──────────────────


def parse_start_year(col: str) -> int | None:
    """Extract start year from a shift column like 'pres_d_shift_08_12' → 2008."""
    parts = col.split("_")
    try:
        y2 = int(parts[-2])
        return y2 + (1900 if y2 >= 50 else 2000)
    except (ValueError, IndexError):
        return None


def classify_columns(all_cols: list[str], min_year: int = 2008) -> tuple[list[str], list[str]]:
    """Split shift columns into training and holdout sets."""
    holdout = [c for c in all_cols if HOLDOUT_MARKER in c]
    training = [
        c for c in all_cols
        if HOLDOUT_MARKER not in c and (parse_start_year(c) or 0) >= min_year
    ]
    return training, holdout


# ── Ridge LOO via hat matrix ─────────────────────────────────────────────────


def ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Ridge LOO predictions via augmented hat-matrix (unpenalized intercept).

    Formula: y_loo_i = y_i - e_i / (1 - H_ii)
    where H = X_aug (X_aug^T X_aug + P)^{-1} X_aug^T, intercept column unpenalized.
    """
    n, p = X.shape
    # Augment with intercept column
    X_aug = np.column_stack([np.ones(n), X])
    pen = alpha * np.eye(p + 1)
    pen[0, 0] = 0.0  # intercept is not regularized
    A = X_aug.T @ X_aug + pen
    A_inv = np.linalg.inv(A)
    h = np.einsum("ij,ij->i", X_aug @ A_inv, X_aug)  # hat matrix diagonal
    beta = A_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    e = y - y_hat
    denom = np.where(np.abs(1.0 - h) < 1e-10, 1e-10, 1.0 - h)
    return y - e / denom


def compute_loo_r(X: np.ndarray, holdout_raw: np.ndarray) -> dict:
    """GCV alpha selection + hat-matrix LOO r for each holdout dimension.

    Returns mean_r, per_dim_r, mean_rmse, best_alphas.
    """
    n_dims = holdout_raw.shape[1]
    per_r: list[float] = []
    per_rmse: list[float] = []
    best_alphas: list[float] = []

    for d in range(n_dims):
        y = holdout_raw[:, d]
        rcv = RidgeCV(alphas=ALPHAS, fit_intercept=True, gcv_mode="auto")
        rcv.fit(X, y)
        alpha = float(rcv.alpha_)
        best_alphas.append(alpha)

        y_loo = ridge_loo_predictions(X, y, alpha)

        if np.std(y) < 1e-10 or np.std(y_loo) < 1e-10:
            per_r.append(0.0)
        else:
            r, _ = pearsonr(y, y_loo)
            per_r.append(float(np.clip(r, -1.0, 1.0)))

        per_rmse.append(float(np.sqrt(np.mean((y - y_loo) ** 2))))

    return {
        "mean_r": float(np.mean(per_r)),
        "per_dim_r": per_r,
        "mean_rmse": float(np.mean(per_rmse)),
        "per_dim_rmse": per_rmse,
        "best_alphas": best_alphas,
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def load_shifts() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load multiyear county shift matrix and split into training/holdout columns."""
    path = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
    df = pd.read_parquet(path)
    all_cols = [c for c in df.columns if c != "county_fips"]
    training_cols, holdout_cols = classify_columns(all_cols, min_year=MIN_YEAR)
    return df, training_cols, holdout_cols


def build_shift_matrices(
    df: pd.DataFrame,
    training_cols: list[str],
    holdout_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return training_raw, training_scaled (pw), holdout_raw, county_fips."""
    mat = df[training_cols + holdout_cols].values.astype(float)
    n_train = len(training_cols)

    training_raw = mat[:, :n_train]
    holdout_raw = mat[:, n_train:]
    county_fips = df["county_fips"].values

    # Presidential columns get extra weight (matches production)
    pres_idx = [i for i, c in enumerate(training_cols) if "pres_" in c]
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_raw)
    if PRESIDENTIAL_WEIGHT != 1.0:
        training_scaled[:, pres_idx] *= PRESIDENTIAL_WEIGHT

    return training_raw, training_scaled, holdout_raw, county_fips


def discover_type_scores(training_scaled: np.ndarray) -> np.ndarray:
    """Run PCA(n=15, whiten=True) on shift vectors then KMeans J=100.

    This matches the production pipeline established in experiment_pca_before_kmeans.py
    (S305): PCA whitening before KMeans improves LOO r from 0.695 to 0.717.
    """
    print(f"  Applying PCA(n={PCA_N_COMPONENTS}, whiten=True) to shift vectors...")
    pca = PCA(n_components=PCA_N_COMPONENTS, whiten=True, random_state=RANDOM_STATE)
    training_pca = pca.fit_transform(training_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance explained: {explained:.1%}")

    print(f"  Running KMeans J={J}, T={TEMPERATURE}...")
    type_result = discover_types(training_pca, j=J, temperature=TEMPERATURE, random_state=RANDOM_STATE)
    return type_result.scores


def load_demographics(county_fips: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load county_features_national.parquet, inner-join, impute NaN, standardize.

    Returns (demo_std_feat, demo_cols, row_mask).
    demo_std_feat: shape (n_matched, n_demo), z-scored.
    demo_cols: list of feature column names.
    row_mask: integer indices into original county_fips array for matched rows.
    """
    path = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"
    demo_df = pd.read_parquet(path)
    demo_cols = [c for c in demo_df.columns if c != "county_fips"]

    # Inner join aligned to shift-matrix FIPS order
    idx_df = pd.DataFrame({"county_fips": county_fips, "_row_idx": np.arange(len(county_fips))})
    merged = idx_df.merge(demo_df, on="county_fips", how="inner")

    row_mask = merged["_row_idx"].values
    demo_raw = merged[demo_cols].values.astype(float)

    # Impute NaN with column means (same as production)
    col_means = np.nanmean(demo_raw, axis=0)
    nan_mask = np.isnan(demo_raw)
    if nan_mask.any():
        r_idx, c_idx = np.where(nan_mask)
        demo_raw[r_idx, c_idx] = col_means[c_idx]

    # Z-score standardization
    demo_mean = demo_raw.mean(axis=0)
    demo_std = demo_raw.std(axis=0)
    demo_std = np.where(demo_std < 1e-10, 1.0, demo_std)
    demo_std_feat = (demo_raw - demo_mean) / demo_std

    return demo_std_feat, demo_cols, row_mask


# ── Feature group definitions ─────────────────────────────────────────────────
# Groups corresponding to data sources for batch ablation.

FEATURE_GROUPS: dict[str, list[str]] = {
    "ACS_race_ethnicity": ["pct_white_nh", "pct_black", "pct_asian", "pct_hispanic"],
    "ACS_education": ["pct_bachelors_plus", "pct_graduate"],
    "ACS_income": ["median_hh_income", "log_median_hh_income"],
    "ACS_housing": ["pct_owner_occupied", "homeownership_pct"],
    "ACS_commute": ["pct_wfh", "pct_transit", "drive_alone_pct"],
    "ACS_demographics": ["median_age", "pop_total", "pct_management"],
    "RCMS_religion": ["evangelical_share", "mainline_share", "catholic_share",
                      "black_protestant_share", "congregations_per_1000", "religious_adherence_rate"],
    "QCEW_industry": ["manufacturing_share", "government_share", "healthcare_share",
                      "retail_share", "construction_share", "finance_share",
                      "hospitality_share", "industry_diversity_hhi"],
    "CHR_health_behaviors": ["adult_smoking_pct", "adult_obesity_pct", "excessive_drinking_pct",
                              "physical_inactivity_pct", "insufficient_sleep_pct",
                              "food_insecurity_pct", "alcohol_impaired_driving_deaths_pct"],
    "CHR_health_outcomes": ["premature_death_rate", "life_expectancy", "uninsured_pct",
                             "children_in_poverty_pct", "diabetes_prevalence_pct",
                             "poor_mental_health_days", "primary_care_physicians_rate",
                             "mental_health_providers_rate", "severe_housing_problems_pct"],
    "CDC_mortality": ["drug_overdose_rate", "despair_death_rate", "allcause_age_adj_rate",
                      "drug_overdose_deaths_rate", "suicide_rate", "firearm_fatalities_rate",
                      "injury_deaths_rate", "homicide_rate"],
    "COVID": ["covid_death_rate", "excess_mortality_ratio", "vax_complete_pct",
              "vax_booster_pct", "vax_dose1_pct"],
    "IRS_migration": ["net_migration_rate", "avg_inflow_income", "migration_diversity",
                      "inflow_outflow_ratio"],
    "Urbanicity": ["log_pop_density", "land_area_sq_mi", "pop_per_sq_mi"],
    "Facebook_SCI": ["network_diversity", "pct_sci_instate", "sci_top5_mean_dem_share",
                     "sci_geographic_reach"],
    "Broadband": ["pct_broadband", "pct_no_internet", "pct_satellite",
                  "pct_cable_fiber", "broadband_gap"],
    "BEA_state_gdp": ["bea_state_gdp_millions", "bea_state_income_per_capita"],
    "BEA_growth": ["bea_gdp_growth_1yr", "bea_gdp_growth_2yr", "bea_income_growth_1yr",
                   "gdp_per_capita", "gdp_growth", "pci", "pci_growth"],
    "VA_disability": ["va_disability_per_1000", "va_disability_pct_100rated", "va_disability_pct_young"],
    "USDA_typology": ["High_Farming_2025", "High_Mining_2025", "High_Manufacturing_2025",
                      "High_Government_2025", "High_Recreation_2025", "Nonspecialized_2025",
                      "Low_PostSecondary_Ed_2025", "Low_Employment_2025",
                      "Population_Loss_2025", "Housing_Stress_2025",
                      "Retirement_Destination_2025", "Persistent_Poverty_1721",
                      "Industry_Dependence_2025"],
    "Transport": ["transport_pop_density", "transport_job_density", "transport_intersection_density",
                  "transport_pct_local_roads", "transport_broadband", "transport_dead_end_proportion",
                  "transport_circuity_avg"],
    "CHR_social": ["social_associations_rate", "voter_turnout_pct", "disconnected_youth_pct",
                   "residential_segregation", "census_participation_pct", "free_reduced_lunch_pct",
                   "single_parent_households_pct"],
}


# ── Main experiment ───────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("FEATURE IMPORTANCE — Ridge+Demo LOO r Analysis")
    print(f"Target baseline: ~0.717 | J={J} | PCA n={PCA_N_COMPONENTS} | pw={PRESIDENTIAL_WEIGHT}")
    print("=" * 72)
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading shift matrix...")
    df, training_cols, holdout_cols = load_shifts()
    training_raw, training_scaled, holdout_raw, county_fips = build_shift_matrices(
        df, training_cols, holdout_cols
    )
    print(f"  Shift matrix: {len(county_fips)} counties, {len(training_cols)} training dims")
    print(f"  Holdout dims: {holdout_cols}")
    print()

    # ── 2. Discover type scores (PCA + KMeans) ────────────────────────────────
    print("Discovering type scores...")
    scores = discover_type_scores(training_scaled)
    print(f"  Scores shape: {scores.shape}")
    print()

    # ── 3. Load demographics ──────────────────────────────────────────────────
    print("Loading demographics (county_features_national.parquet)...")
    demo_std_feat, demo_cols, row_mask = load_demographics(county_fips)
    n_counties = len(row_mask)
    print(f"  Matched counties: {n_counties} (dropped {len(county_fips) - n_counties} unmatched)")
    print(f"  Demo features: {len(demo_cols)}")
    print()

    # ── 4. Subset all arrays to inner-join rows ───────────────────────────────
    scores_sub = scores[row_mask]
    holdout_sub = holdout_raw[row_mask]
    county_mean_sub = training_raw[row_mask].mean(axis=1)  # county historical mean

    # Full feature matrix: [type_scores | county_mean | demo_features]
    X_full = np.column_stack([scores_sub, county_mean_sub, demo_std_feat])
    print(f"  Full feature matrix shape: {X_full.shape}")
    print()

    # ── 5. Baseline LOO r (all features) ─────────────────────────────────────
    print("Computing baseline LOO r (all features)...")
    baseline = compute_loo_r(X_full, holdout_sub)
    baseline_r = baseline["mean_r"]
    print(f"  Baseline LOO r = {baseline_r:.4f}  (target: ~0.717)")
    print(f"  Per-dim r: {[f'{r:.4f}' for r in baseline['per_dim_r']]}")
    print(f"  Per-dim RMSE: {[f'{v:.4f}' for v in baseline['per_dim_rmse']]}")
    print()

    # ── 6. Individual feature ablation ───────────────────────────────────────
    # The full matrix has J (100) + 1 (county_mean) + n_demo columns.
    # Demo features start at index (J + 1).
    demo_start_idx = scores_sub.shape[1] + 1  # after scores + county_mean

    print(f"Running individual ablation for {len(demo_cols)} demo features...")
    print("(This may take 2-4 minutes)")
    print()

    ablation_results: list[dict] = []

    for feat_idx, feat_name in enumerate(demo_cols):
        # Build feature matrix with this column removed
        col_to_drop = demo_start_idx + feat_idx
        cols_to_keep = [c for c in range(X_full.shape[1]) if c != col_to_drop]
        X_ablated = X_full[:, cols_to_keep]

        res = compute_loo_r(X_ablated, holdout_sub)
        delta = baseline_r - res["mean_r"]  # positive → feature helps

        ablation_results.append({
            "feature": feat_name,
            "loo_r_without": res["mean_r"],
            "delta": delta,
            "helps": delta > 0,
        })

        # Print progress every 10 features
        if (feat_idx + 1) % 10 == 0:
            print(f"  [{feat_idx + 1}/{len(demo_cols)}] latest: {feat_name} → "
                  f"LOO r w/o = {res['mean_r']:.4f}  Δ={delta:+.4f}")

    print()

    # ── 7. Group-level ablation ───────────────────────────────────────────────
    print("Running group-level ablation...")
    group_results: list[dict] = []

    # Build index map for fast lookup
    feat_to_idx = {f: i for i, f in enumerate(demo_cols)}

    for group_name, group_feats in FEATURE_GROUPS.items():
        # Only include features that are actually in our demo_cols
        present_feats = [f for f in group_feats if f in feat_to_idx]
        if not present_feats:
            continue

        drop_cols = {demo_start_idx + feat_to_idx[f] for f in present_feats}
        cols_to_keep = [c for c in range(X_full.shape[1]) if c not in drop_cols]
        X_ablated = X_full[:, cols_to_keep]

        res = compute_loo_r(X_ablated, holdout_sub)
        delta = baseline_r - res["mean_r"]

        group_results.append({
            "group": group_name,
            "n_features": len(present_feats),
            "features": present_feats,
            "loo_r_without": res["mean_r"],
            "delta": delta,
        })
        print(f"  {group_name} ({len(present_feats)} feats): LOO r w/o = {res['mean_r']:.4f}  Δ={delta:+.4f}")

    print()

    # ── 8. Print sorted results ───────────────────────────────────────────────
    ablation_results.sort(key=lambda x: x["delta"], reverse=True)

    print("=" * 72)
    print(f"FEATURE IMPORTANCE RESULTS  (baseline LOO r = {baseline_r:.4f})")
    print("=" * 72)
    print()

    # Features that help most (top 20)
    print("TOP 20 MOST HELPFUL FEATURES  (biggest drop in LOO r when removed):")
    print(f"  {'Feature':<45} {'LOO r w/o':>10}  {'Δ':>8}")
    print("  " + "-" * 65)
    for row in ablation_results[:20]:
        marker = " *" if row["delta"] > 0.002 else "  "
        print(f"  {row['feature']:<45} {row['loo_r_without']:>10.4f}  {row['delta']:>+8.4f}{marker}")

    print()

    # Features that HURT (negative delta — removing them improves LOO r)
    hurt_feats = [r for r in ablation_results if r["delta"] < -0.0005]
    print(f"FEATURES THAT HURT PERFORMANCE (n={len(hurt_feats)}):  removing them IMPROVES LOO r")
    print(f"  {'Feature':<45} {'LOO r w/o':>10}  {'Δ':>8}")
    print("  " + "-" * 65)
    if hurt_feats:
        for row in sorted(hurt_feats, key=lambda x: x["delta"]):
            print(f"  {row['feature']:<45} {row['loo_r_without']:>10.4f}  {row['delta']:>+8.4f}  <-- DROP?")
    else:
        print("  (none — no individual features hurt performance)")

    print()

    # Near-zero features (might be safe to drop for parsimony)
    neutral_feats = [r for r in ablation_results if -0.0005 <= r["delta"] <= 0.0005]
    print(f"NEAR-ZERO CONTRIBUTION FEATURES (n={len(neutral_feats)},  |Δ| ≤ 0.0005):")
    print("  These can likely be dropped without hurting LOO r.")
    if neutral_feats:
        for row in sorted(neutral_feats, key=lambda x: x["delta"]):
            print(f"  {row['feature']:<45} {row['loo_r_without']:>10.4f}  {row['delta']:>+8.4f}")
    else:
        print("  (none)")

    print()

    # Group summary
    group_results.sort(key=lambda x: x["delta"], reverse=True)
    print("GROUP-LEVEL ABLATION RESULTS:")
    print(f"  {'Group':<30} {'N':>4}  {'LOO r w/o':>10}  {'Δ':>8}")
    print("  " + "-" * 60)
    for row in group_results:
        print(f"  {row['group']:<30} {row['n_features']:>4}  {row['loo_r_without']:>10.4f}  {row['delta']:>+8.4f}")

    print()

    # ── 9. Recommended drop set ───────────────────────────────────────────────
    # Recommend dropping features where: removing them does NOT hurt LOO r.
    # Conservative threshold: delta <= 0 (hurt or neutral individually)
    drop_candidates = [r for r in ablation_results if r["delta"] <= 0.0]

    # Also test the combined effect of dropping all drop candidates
    if drop_candidates:
        drop_names = {r["feature"] for r in drop_candidates}
        drop_idxs = {demo_start_idx + feat_to_idx[f] for f in drop_names}
        cols_to_keep = [c for c in range(X_full.shape[1]) if c not in drop_idxs]
        X_pruned = X_full[:, cols_to_keep]

        pruned_res = compute_loo_r(X_pruned, holdout_sub)
        pruned_r = pruned_res["mean_r"]
        pruned_delta = baseline_r - pruned_r

        print(f"RECOMMENDED DROP SET ({len(drop_candidates)} features, delta ≤ 0 individually):")
        for r in sorted(drop_candidates, key=lambda x: x["delta"]):
            print(f"  {r['feature']:<45}  Δ={r['delta']:>+8.4f}")

        print()
        print(f"  Combined pruned model ({X_pruned.shape[1]} features):")
        print(f"    LOO r = {pruned_r:.4f}  (was {baseline_r:.4f},  Δ={pruned_delta:>+.4f})")
        print(f"    Retained features: {X_pruned.shape[1] - J - 1} demo + {J} type scores + 1 county mean")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
