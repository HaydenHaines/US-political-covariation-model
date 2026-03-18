"""
K=4 comparison run for community vote share estimation.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "covariance"

K = 4
COMP_COLS = [f"c{k}" for k in range(1, K + 1)]

LABELS = {
    "c1": "White affluent\n(income+mgmt+owner+car)",
    "c2": "Black urban\n(transit)",
    "c3": "Asian/Knowledge worker\n(WFH+college+mgmt)",
    "c4": "Hispanic\n(low-income)",
}


def load_data() -> pd.DataFrame:
    mem = pd.read_parquet(
        PROJECT_ROOT / "data" / "communities" / f"tract_memberships_k{K}.parquet"
    )
    vest = pd.read_parquet(
        PROJECT_ROOT / "data" / "assembled" / "vest_tracts_2020.parquet"
    )
    df = mem.merge(vest[["tract_geoid", "pres_dem_share_2020", "pres_total_2020"]],
                   on="tract_geoid", how="inner")
    df = df[~df["is_uninhabited"]].dropna(
        subset=COMP_COLS + ["pres_dem_share_2020", "pres_total_2020"]
    )
    print(f"Joined dataset: {len(df)} tracts")
    return df


def estimate(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    W = df[COMP_COLS].values
    d = df["pres_dem_share_2020"].values
    v = df["pres_total_2020"].values

    sqrt_v = np.sqrt(v)
    theta_nnls, _ = nnls(W * sqrt_v[:, np.newaxis], d * sqrt_v)

    weights = v[:, np.newaxis] * W
    theta_direct = (weights * d[:, np.newaxis]).sum(axis=0) / weights.sum(axis=0)
    return theta_nnls, theta_direct


def r2(df: pd.DataFrame, theta: np.ndarray) -> float:
    W = df[COMP_COLS].values
    d = df["pres_dem_share_2020"].values
    v = df["pres_total_2020"].values
    d_hat = W @ theta
    ss_res = np.sum(v * (d - d_hat) ** 2)
    ss_tot = np.sum(v * (d - np.average(d, weights=v)) ** 2)
    return 1.0 - ss_res / ss_tot


def main() -> None:
    df = load_data()
    theta_nnls, theta_direct = estimate(df)
    r2_val = r2(df, theta_nnls)

    dominant = df[COMP_COLS].idxmax(axis=1)
    counts = dominant.value_counts()

    print("\n" + "=" * 75)
    print(f"K=4 community vote shares  (R²={r2_val:.3f}, NNLS)")
    print(f"{'comp':<6}{'direct':>8}{'NNLS θ':>10}  {'n_dom':>6}  label")
    print("=" * 75)
    order = np.argsort(theta_direct)
    for i in order:
        comp = COMP_COLS[i]
        n = counts.get(comp, 0)
        lean = "D" if theta_direct[i] > 0.5 else "R"
        margin = abs(theta_direct[i] - 0.5)
        nnls_str = f"{theta_nnls[i]:.1%}" if theta_nnls[i] <= 1.0 else f"{theta_nnls[i]:.1%}(!)"
        label = LABELS[comp].replace("\n", " ")
        print(f"  {comp:<5} {theta_direct[i]:.1%} ({lean}+{margin:.1%})  {nnls_str:>10}  n={n:>5,}  {label}")

    print(f"\nR² = {r2_val:.3f}  (K=4 vs K=8 R²=0.624 vs K=11 R²=0.719)")
    print(f"Dominant tract distribution: {counts.to_dict()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "component": COMP_COLS,
        "label": [LABELS[c] for c in COMP_COLS],
        "dem_share_direct": theta_direct,
        "dem_share_nnls": theta_nnls,
        "n_dominant_tracts": [counts.get(c, 0) for c in COMP_COLS],
    })
    results.to_parquet(OUTPUT_DIR / "community_vote_shares_2020_k4.parquet", index=False)
    print(f"\nSaved → {OUTPUT_DIR / 'community_vote_shares_2020_k4.parquet'}")


if __name__ == "__main__":
    main()
