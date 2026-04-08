"""Convert CES validation δ CSVs to per-type npy arrays for the behavior layer.

Reads the CES governor and senate delta CSVs (produced by the CES validation
pipeline in S499-S500) and writes npy arrays compatible with the behavior layer.
Types without enough CES respondents get δ=0 (no adjustment).

Output:
  data/behavior/delta_ces_governor.npy  — (J,) governor-specific δ
  data/behavior/delta_ces_senate.npy    — (J,) senate-specific δ
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

J = 100  # Must match current model type count


def build_ces_delta(race: str) -> np.ndarray:
    """Build a (J,) δ array from CES validation CSV.

    Args:
        race: "governor" or "senate"

    Returns:
        δ array of shape (J,), with 0 for types without CES data.
    """
    csv_path = PROJECT_ROOT / "data" / "validation" / f"ces_{race}_delta.csv"
    df = pd.read_csv(csv_path)

    delta = np.zeros(J, dtype=np.float64)
    for _, row in df.iterrows():
        type_id = int(row["type_id"])
        if 0 <= type_id < J:
            delta[type_id] = row["delta"]

    n_populated = (delta != 0).sum()
    print(f"  {race}: {n_populated}/{J} types populated from CES")
    print(f"    mean={delta.mean():.4f}, std={delta.std():.4f}")
    print(f"    range=[{delta.min():.4f}, {delta.max():.4f}]")
    return delta


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "behavior"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building CES-derived δ arrays:")
    for race in ("governor", "senate"):
        delta = build_ces_delta(race)
        out_path = output_dir / f"delta_ces_{race}.npy"
        np.save(out_path, delta)
        print(f"    → {out_path}")

    print("\nDone. Use these in place of model-computed delta.npy.")


if __name__ == "__main__":
    main()
