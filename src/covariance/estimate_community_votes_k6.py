"""K=6 vote share estimation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "covariance"
K = 6
COMP_COLS = [f"c{k}" for k in range(1, K + 1)]
LABELS = {
    "c1": "White affluent\n(income+mgmt+car)",
    "c2": "Black urban\n(transit+age)",
    "c3": "Asian/Knowledge worker\n(mgmt+WFH+college)",
    "c4": "Homeowner\n(non-professional)",
    "c5": "Retiree\n(age+slight WFH)",
    "c6": "Hispanic",
}

def load_data():
    mem = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / f"tract_memberships_k{K}.parquet")
    vest = pd.read_parquet(PROJECT_ROOT / "data" / "assembled" / "vest_tracts_2020.parquet")
    df = mem.merge(vest[["tract_geoid", "pres_dem_share_2020", "pres_total_2020"]], on="tract_geoid", how="inner")
    return df[~df["is_uninhabited"]].dropna(subset=COMP_COLS + ["pres_dem_share_2020", "pres_total_2020"])

def run(df):
    W, d, v = df[COMP_COLS].values, df["pres_dem_share_2020"].values, df["pres_total_2020"].values
    sqrt_v = np.sqrt(v)
    theta_nnls, _ = nnls(W * sqrt_v[:, np.newaxis], d * sqrt_v)
    weights = v[:, np.newaxis] * W
    theta_direct = (weights * d[:, np.newaxis]).sum(axis=0) / weights.sum(axis=0)
    d_hat = W @ theta_nnls
    ss_res = np.sum(v * (d - d_hat) ** 2)
    ss_tot = np.sum(v * (d - np.average(d, weights=v)) ** 2)
    return theta_nnls, theta_direct, 1.0 - ss_res / ss_tot

def main():
    df = load_data()
    theta_nnls, theta_direct, r2 = run(df)
    dominant = df[COMP_COLS].idxmax(axis=1)
    counts = dominant.value_counts()
    n = len(df)

    print(f"\n{'='*75}")
    print(f"K=6 community vote shares  (R²={r2:.3f}, NNLS)")
    print(f"{'comp':<6}{'direct':>8}{'NNLS θ':>10}  {'n_dom':>6}  label")
    print("=" * 75)
    for i in np.argsort(theta_direct):
        comp = COMP_COLS[i]
        nd = counts.get(comp, 0)
        lean = "D" if theta_direct[i] > 0.5 else "R"
        margin = abs(theta_direct[i] - 0.5)
        nnls_str = f"{theta_nnls[i]:.1%}" if theta_nnls[i] <= 1.0 else f"{theta_nnls[i]:.1%}(!)"
        label = LABELS[comp].replace("\n", " ")
        print(f"  {comp:<5} {theta_direct[i]:.1%} ({lean}+{margin:.1%})  {nnls_str:>10}  n={nd:>5,} ({100*nd/n:.0f}%)  {label}")

    print(f"\nR² = {r2:.3f}")
    results = pd.DataFrame({"component": COMP_COLS, "label": [LABELS[c] for c in COMP_COLS],
        "dem_share_direct": theta_direct, "dem_share_nnls": theta_nnls,
        "n_dominant_tracts": [counts.get(c, 0) for c in COMP_COLS]})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_DIR / "community_vote_shares_2020_k6.parquet", index=False)

if __name__ == "__main__":
    main()
