"""K=7 and K=9 comparison vote share estimation."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "covariance"

LABELS = {
    7: {
        "c1": "White retiree/remote\n(owner-occ+WFH+age)",
        "c2": "Black urban\n(transit+income)",
        "c3": "Knowledge worker\n(mgmt+WFH+college)",
        "c4": "Asian",
        "c5": "Homeowner\n(non-professional)",
        "c6": "Hispanic",
        "c7": "Generic white suburban\n(baseline)",
    },
    9: {
        "c1": "White suburban\n(car-dependent)",
        "c2": "Black urban\n(car+transit+income)",
        "c3": "Retiree manager\n(mgmt+age, no commute)",
        "c4": "WFH urban mix\n(white+Hispanic+transit)",
        "c5": "Educated retiree\n(college+owner+age)",
        "c6": "Hispanic",
        "c7": "Asian",
        "c8": "Homeowner\n(non-professional)",
        "c9": "College-educated\n(no other signal)",
    },
}


def load_data(k: int) -> pd.DataFrame:
    comp_cols = [f"c{i}" for i in range(1, k + 1)]
    mem = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / f"tract_memberships_k{k}.parquet")
    vest = pd.read_parquet(PROJECT_ROOT / "data" / "assembled" / "vest_tracts_2020.parquet")
    df = mem.merge(vest[["tract_geoid", "pres_dem_share_2020", "pres_total_2020"]], on="tract_geoid", how="inner")
    df = df[~df["is_uninhabited"]].dropna(subset=comp_cols + ["pres_dem_share_2020", "pres_total_2020"])
    return df


def run(k: int) -> dict:
    comp_cols = [f"c{i}" for i in range(1, k + 1)]
    df = load_data(k)

    W = df[comp_cols].values
    d = df["pres_dem_share_2020"].values
    v = df["pres_total_2020"].values

    sqrt_v = np.sqrt(v)
    theta_nnls, _ = nnls(W * sqrt_v[:, np.newaxis], d * sqrt_v)

    weights = v[:, np.newaxis] * W
    theta_direct = (weights * d[:, np.newaxis]).sum(axis=0) / weights.sum(axis=0)

    d_hat = W @ theta_nnls
    ss_res = np.sum(v * (d - d_hat) ** 2)
    ss_tot = np.sum(v * (d - np.average(d, weights=v)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    dominant = df[comp_cols].idxmax(axis=1)
    counts = dominant.value_counts()

    return {
        "k": k, "r2": r2, "comp_cols": comp_cols,
        "theta_direct": theta_direct, "theta_nnls": theta_nnls,
        "counts": counts, "n_tracts": len(df),
    }


def print_results(res: dict) -> None:
    k = res["k"]
    print(f"\n{'='*75}")
    print(f"K={k} community vote shares  (R²={res['r2']:.3f}, NNLS)")
    print(f"{'comp':<6}{'direct':>8}{'NNLS θ':>10}  {'n_dom':>6}  label")
    print("=" * 75)
    order = np.argsort(res["theta_direct"])
    for i in order:
        comp = res["comp_cols"][i]
        n = res["counts"].get(comp, 0)
        lean = "D" if res["theta_direct"][i] > 0.5 else "R"
        margin = abs(res["theta_direct"][i] - 0.5)
        nnls_v = res["theta_nnls"][i]
        nnls_str = f"{nnls_v:.1%}" if nnls_v <= 1.0 else f"{nnls_v:.1%}(!)"
        label = LABELS[k][comp].replace("\n", " ")
        pct_dominant = 100 * n / res["n_tracts"]
        print(f"  {comp:<5} {res['theta_direct'][i]:.1%} ({lean}+{margin:.1%})  {nnls_str:>10}  n={n:>5,} ({pct_dominant:.0f}%)  {label}")


def main() -> None:
    res7 = run(7)
    res9 = run(9)

    print_results(res7)
    print_results(res9)

    print(f"\n{'='*75}")
    print("R² comparison across all tested K values:")
    print(f"  K=4:  0.615  (no NNLS pathology, balanced tract distribution)")
    print(f"  K=7:  {res7['r2']:.3f}")
    print(f"  K=8:  0.624  (current model)")
    print(f"  K=9:  {res9['r2']:.3f}")
    print(f"  K=11: 0.719  (97.6% generic-bucket dominance)")

    dom7 = res7["counts"].iloc[0]
    dom9 = res9["counts"].iloc[0]
    print(f"\nLargest dominant bucket:")
    print(f"  K=7:  {dom7:,} tracts ({100*dom7/res7['n_tracts']:.0f}%) — {res7['counts'].index[0]}")
    print(f"  K=9:  {dom9:,} tracts ({100*dom9/res9['n_tracts']:.0f}%) — {res9['counts'].index[0]}")


if __name__ == "__main__":
    main()
