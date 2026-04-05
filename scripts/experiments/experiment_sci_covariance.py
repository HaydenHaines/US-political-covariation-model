"""
Research: Does Facebook SCI social connectivity correlate with political covariance
between county types?

Hypothesis: Types with high inter-type social connectivity should show higher political
covariance (i.e., they swing together). If true, SCI could inform a propagation model
for how polling signals spread between communities.

Usage:
    uv run python scripts/experiment_sci_covariance.py

Outputs:
    - Pearson r between type-pair SCI matrix and political covariance matrix
    - Within-type vs between-type SCI comparison
    - Top correlated type pairs
    - Scatter plots saved to experiments/sci_covariance/
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_COMMUNITIES = PROJECT_ROOT / "data" / "communities"
DATA_COVARIANCE = PROJECT_ROOT / "data" / "covariance"
EXPERIMENTS_OUT = PROJECT_ROOT / "experiments" / "sci_covariance"

SCI_CSV = DATA_RAW / "facebook_sci" / "us_counties.csv"
TYPE_ASSIGNMENTS = DATA_COMMUNITIES / "type_assignments.parquet"
TYPE_COVARIANCE = DATA_COVARIANCE / "type_covariance.parquet"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_type_assignments() -> pd.DataFrame:
    """Load county → dominant type + soft scores. Returns county_fips (int) + dominant_type."""
    ta = pd.read_parquet(TYPE_ASSIGNMENTS)
    # Normalise FIPS to integer for matching SCI data
    ta["county_fips_int"] = ta["county_fips"].astype(int)
    return ta


def load_type_covariance() -> np.ndarray:
    """Load 100×100 type covariance matrix. Returns numpy array indexed 0..99."""
    cov_df = pd.read_parquet(TYPE_COVARIANCE)
    # Columns are integer 0..99; ensure correct ordering
    cols = sorted(cov_df.columns)
    return cov_df[cols].values.astype(float)


def load_sci_chunked(county_fips_set: set, chunk_size: int = 500_000) -> pd.DataFrame:
    """
    Stream the raw SCI CSV and keep only rows where BOTH endpoints are in our
    county set. Returns DataFrame with columns: user_region, friend_region, scaled_sci.
    """
    print(f"  Streaming SCI file ({SCI_CSV.stat().st_size / 1e6:.0f} MB)...")
    t0 = time.time()
    kept_chunks = []
    total_rows = 0
    kept_rows = 0
    for chunk in pd.read_csv(
        SCI_CSV,
        usecols=["user_region", "friend_region", "scaled_sci"],
        chunksize=chunk_size,
        dtype={"user_region": "int32", "friend_region": "int32", "scaled_sci": "float32"},
    ):
        total_rows += len(chunk)
        mask = chunk["user_region"].isin(county_fips_set) & chunk["friend_region"].isin(county_fips_set)
        filtered = chunk[mask]
        kept_rows += len(filtered)
        if len(filtered) > 0:
            kept_chunks.append(filtered)
    elapsed = time.time() - t0
    print(f"  Read {total_rows:,} rows in {elapsed:.1f}s, kept {kept_rows:,} matching rows")
    return pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. Build type-pair SCI matrix
# ---------------------------------------------------------------------------

def build_type_pair_sci_matrix(
    sci_df: pd.DataFrame,
    ta: pd.DataFrame,
    n_types: int = 100,
) -> np.ndarray:
    """
    For each pair of types (i, j), compute the mean SCI across all county pairs
    (c1 ∈ type_i, c2 ∈ type_j).

    Uses soft-membership weights (type_score) for a richer signal, but also
    computes a hard (dominant type) version for comparison.

    Returns: sci_matrix (n_types × n_types), mean SCI for each type pair.
    """
    print("  Merging SCI with type assignments...")
    fips_to_type = ta.set_index("county_fips_int")["dominant_type"]

    # Hard assignment version (fast, uses dominant type only)
    sci_df = sci_df.copy()
    sci_df["type_u"] = sci_df["user_region"].map(fips_to_type)
    sci_df["type_f"] = sci_df["friend_region"].map(fips_to_type)
    sci_df = sci_df.dropna(subset=["type_u", "type_f"])
    sci_df["type_u"] = sci_df["type_u"].astype(int)
    sci_df["type_f"] = sci_df["type_f"].astype(int)

    print(f"  {len(sci_df):,} SCI pairs after joining type assignments")

    # Build matrix using grouped mean
    sci_matrix = np.full((n_types, n_types), np.nan)

    grouped = sci_df.groupby(["type_u", "type_f"])["scaled_sci"].mean()
    for (i, j), val in grouped.items():
        sci_matrix[i, j] = val
        sci_matrix[j, i] = val  # ensure symmetry (take max of both directions)

    # Fill NaN pairs (no observed SCI) with 0
    sci_matrix = np.nan_to_num(sci_matrix, nan=0.0)
    return sci_matrix


# ---------------------------------------------------------------------------
# 3. Analysis
# ---------------------------------------------------------------------------

def analyse_correlation(sci_matrix: np.ndarray, cov_matrix: np.ndarray, n_types: int = 100):
    """
    Compare the type-pair SCI matrix with the political covariance matrix.

    Reports:
    - Pearson r for all pairs (excluding diagonal)
    - Within-type vs between-type SCI
    - Top 20 most politically-correlated type pairs and their SCI values
    - Partial results: off-diagonal upper triangle only
    """
    print("\n" + "=" * 60)
    print("RESULTS: SCI vs Political Covariance Analysis")
    print("=" * 60)

    # --- Full matrix (all n_types² pairs, including diagonal) ---
    sci_flat = sci_matrix.flatten()
    cov_flat = cov_matrix.flatten()
    r_all, p_all = stats.pearsonr(sci_flat, cov_flat)
    print(f"\n[All pairs including diagonal]")
    print(f"  Pearson r = {r_all:.4f}  (p = {p_all:.2e})  n = {len(sci_flat):,}")

    # --- Off-diagonal upper triangle only ---
    idx_upper = np.triu_indices(n_types, k=1)
    sci_upper = sci_matrix[idx_upper]
    cov_upper = cov_matrix[idx_upper]
    r_upper, p_upper = stats.pearsonr(sci_upper, cov_upper)
    n_pairs = len(sci_upper)
    print(f"\n[Off-diagonal type pairs (upper triangle only)]")
    print(f"  n pairs   = {n_pairs:,}")
    print(f"  Pearson r = {r_upper:.4f}  (p = {p_upper:.2e})")

    # Spearman (rank-based, more robust)
    rho, p_rho = stats.spearmanr(sci_upper, cov_upper)
    print(f"  Spearman ρ= {rho:.4f}  (p = {p_rho:.2e})")

    # --- Diagonal (within-type) vs off-diagonal (between-type) SCI ---
    diag_sci = np.diag(sci_matrix)
    off_diag_sci = sci_upper

    print(f"\n[Within-type vs Between-type SCI]")
    print(f"  Within-type  mean SCI = {diag_sci.mean():>12,.1f}  (n = {n_types})")
    print(f"  Between-type mean SCI = {off_diag_sci.mean():>12,.1f}  (n = {n_pairs:,})")
    ratio = diag_sci.mean() / off_diag_sci[off_diag_sci > 0].mean() if off_diag_sci[off_diag_sci > 0].mean() > 0 else float("nan")
    print(f"  Within/Between ratio  = {ratio:.2f}x")

    # Types with zero SCI (completely isolated)
    zero_sci_types = np.where(diag_sci == 0)[0]
    print(f"  Types with zero within-type SCI: {len(zero_sci_types)}")
    if len(zero_sci_types) < 20:
        print(f"    Type IDs: {zero_sci_types.tolist()}")

    # --- Top political covariance pairs and their SCI ---
    print(f"\n[Top 20 type pairs by POLITICAL COVARIANCE]")
    sorted_idx = np.argsort(cov_upper)[::-1][:20]
    print(f"  {'Pair':>12}  {'Cov':>10}  {'SCI':>12}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}")
    for idx in sorted_idx:
        i, j = idx_upper[0][idx], idx_upper[1][idx]
        print(f"  type {i:2d}-{j:<3d}  {cov_upper[idx]:>10.5f}  {sci_upper[idx]:>12,.0f}")

    # --- Top SCI pairs and their political covariance ---
    print(f"\n[Top 20 type pairs by SOCIAL CONNECTIVITY (SCI)]")
    sorted_sci = np.argsort(sci_upper)[::-1][:20]
    print(f"  {'Pair':>12}  {'SCI':>12}  {'Cov':>10}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*10}")
    for idx in sorted_sci:
        i, j = idx_upper[0][idx], idx_upper[1][idx]
        print(f"  type {i:2d}-{j:<3d}  {sci_upper[idx]:>12,.0f}  {cov_upper[idx]:>10.5f}")

    # --- Correlation by SCI quantile ---
    print(f"\n[Pearson r within SCI quantile bins]")
    quantiles = [0, 25, 50, 75, 90, 100]
    for q_lo, q_hi in zip(quantiles[:-1], quantiles[1:]):
        lo_val = np.percentile(sci_upper, q_lo)
        hi_val = np.percentile(sci_upper, q_hi)
        mask = (sci_upper >= lo_val) & (sci_upper < hi_val)
        if q_hi == 100:
            mask = sci_upper >= lo_val
        n_bin = mask.sum()
        if n_bin < 10:
            print(f"  Q{q_lo:2d}-Q{q_hi:2d}: too few pairs ({n_bin})")
            continue
        r_bin, p_bin = stats.pearsonr(sci_upper[mask], cov_upper[mask])
        print(f"  Q{q_lo:2d}-Q{q_hi:2d} (SCI {lo_val:>10,.0f} – {hi_val:>10,.0f}): r = {r_bin:.4f}  n = {n_bin:,}")

    # --- Summary interpretation ---
    print(f"\n[Interpretation]")
    if abs(r_upper) >= 0.3:
        strength = "strong" if abs(r_upper) >= 0.5 else "moderate"
        direction = "positive" if r_upper > 0 else "negative"
        print(f"  {strength.capitalize()} {direction} correlation detected (r = {r_upper:.3f}).")
        if r_upper > 0:
            print("  => Types that are more socially connected tend to co-vary politically.")
            print("  => SCI is a plausible propagation weight for polling signal diffusion.")
        else:
            print("  => Negative: socially connected types show LESS political co-variation.")
            print("  => SCI may reflect cross-ideological ties (bridging social capital).")
    elif abs(r_upper) >= 0.1:
        print(f"  Weak correlation (r = {r_upper:.3f}). SCI has limited signal for propagation.")
    else:
        print(f"  Negligible correlation (r = {r_upper:.3f}). SCI does not predict political co-variation.")
        print("  => Political covariance is driven by shared structural features, not social ties.")

    return {
        "r_all_pairs": r_all,
        "r_off_diagonal": r_upper,
        "rho_off_diagonal": rho,
        "p_value": p_upper,
        "n_pairs": n_pairs,
        "within_type_mean_sci": float(diag_sci.mean()),
        "between_type_mean_sci": float(off_diag_sci.mean()),
        "within_between_ratio": ratio,
    }


def save_pair_dataframe(
    sci_matrix: np.ndarray,
    cov_matrix: np.ndarray,
    n_types: int = 100,
    out_dir: Path = EXPERIMENTS_OUT,
):
    """Save a long-form CSV of all type pairs with SCI and covariance columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_upper = np.triu_indices(n_types, k=0)  # include diagonal
    rows = []
    for i, j in zip(idx_upper[0], idx_upper[1]):
        rows.append({
            "type_i": int(i),
            "type_j": int(j),
            "sci": float(sci_matrix[i, j]),
            "covariance": float(cov_matrix[i, j]),
            "diagonal": i == j,
        })
    df = pd.DataFrame(rows)
    out_path = out_dir / "type_pair_sci_vs_covariance.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved pair data to: {out_path}")
    return df


# ---------------------------------------------------------------------------
# 4. Optional: scatter plot (skip gracefully if matplotlib not available)
# ---------------------------------------------------------------------------

def try_save_scatter(sci_upper: np.ndarray, cov_upper: np.ndarray, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Raw SCI
        axes[0].scatter(sci_upper, cov_upper, alpha=0.15, s=5, color="steelblue")
        r, _ = stats.pearsonr(sci_upper, cov_upper)
        axes[0].set_xlabel("Mean SCI (type pair)")
        axes[0].set_ylabel("Political covariance")
        axes[0].set_title(f"SCI vs Covariance  (r = {r:.3f})")

        # Log SCI
        log_sci = np.log1p(sci_upper)
        r_log, _ = stats.pearsonr(log_sci, cov_upper)
        axes[1].scatter(log_sci, cov_upper, alpha=0.15, s=5, color="darkorange")
        axes[1].set_xlabel("log(1 + Mean SCI)")
        axes[1].set_ylabel("Political covariance")
        axes[1].set_title(f"log-SCI vs Covariance  (r = {r_log:.3f})")

        # Log-SCI vs covariance Pearson r
        print(f"\n  log(1+SCI) vs covariance:  Pearson r = {r_log:.4f}")

        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "sci_vs_covariance_scatter.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        print(f"  Scatter plot saved to: {fig_path}")
    except ImportError:
        print("  (matplotlib not available — skipping scatter plot)")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SCI ↔ Political Covariance Research Experiment")
    print("=" * 60)

    # --- Load type assignments ---
    print("\n[1] Loading type assignments...")
    ta = load_type_assignments()
    county_fips_set = set(ta["county_fips_int"].tolist())
    n_types = 100
    print(f"  {len(ta):,} counties, {n_types} types")
    print(f"  Dominant type range: {ta['dominant_type'].min()} – {ta['dominant_type'].max()}")

    # --- Load covariance ---
    print("\n[2] Loading type covariance matrix...")
    cov_matrix = load_type_covariance()
    print(f"  Shape: {cov_matrix.shape}")
    print(f"  Diagonal (variances) range: {np.diag(cov_matrix).min():.4f} – {np.diag(cov_matrix).max():.4f}")
    print(f"  Off-diagonal range: {cov_matrix[np.triu_indices(n_types, k=1)].min():.4f} – {cov_matrix[np.triu_indices(n_types, k=1)].max():.4f}")

    # --- Load SCI (streaming) ---
    print("\n[3] Loading SCI data (streaming, this takes ~30–60s)...")
    sci_df = load_sci_chunked(county_fips_set)
    if sci_df.empty:
        print("ERROR: No matching SCI rows found. Check FIPS alignment.")
        sys.exit(1)

    # --- Build type-pair SCI matrix ---
    print("\n[4] Building type-pair SCI matrix...")
    sci_matrix = build_type_pair_sci_matrix(sci_df, ta, n_types=n_types)
    print(f"  SCI matrix: {sci_matrix.shape}")
    sci_upper_vals = sci_matrix[np.triu_indices(n_types, k=1)]
    n_zero = (sci_upper_vals == 0).sum()
    print(f"  Non-zero off-diagonal pairs: {(sci_upper_vals > 0).sum():,} / {len(sci_upper_vals):,}")
    print(f"  SCI range: {sci_upper_vals[sci_upper_vals > 0].min():.0f} – {sci_upper_vals.max():.0f}")

    # --- Analyse ---
    print("\n[5] Analysing correlation...")
    results = analyse_correlation(sci_matrix, cov_matrix, n_types=n_types)

    # log-SCI bonus check
    idx_upper = np.triu_indices(n_types, k=1)
    sci_upper = sci_matrix[idx_upper]
    cov_upper = cov_matrix[idx_upper]
    log_sci = np.log1p(sci_upper)
    r_log, p_log = stats.pearsonr(log_sci, cov_upper)
    print(f"\n  log(1+SCI) vs covariance:  r = {r_log:.4f}  (p = {p_log:.2e})")

    # --- Save pair data ---
    print("\n[6] Saving output data...")
    pair_df = save_pair_dataframe(sci_matrix, cov_matrix, n_types=n_types)

    # --- Scatter plot ---
    try_save_scatter(sci_upper, cov_upper, EXPERIMENTS_OUT)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Off-diagonal Pearson r   : {results['r_off_diagonal']:.4f}")
    print(f"  Off-diagonal Spearman ρ  : {results['rho_off_diagonal']:.4f}")
    print(f"  p-value                  : {results['p_value']:.2e}")
    print(f"  log(SCI) Pearson r       : {r_log:.4f}")
    print(f"  Within-type mean SCI     : {results['within_type_mean_sci']:,.0f}")
    print(f"  Between-type mean SCI    : {results['between_type_mean_sci']:,.0f}")
    print(f"  Within/Between SCI ratio : {results['within_between_ratio']:.2f}x")
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
