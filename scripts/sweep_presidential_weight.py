"""Sweep presidential_weight over candidate values and report K=5 holdout r.

Edits config/model.yaml for each weight, runs the multiyear holdout validator,
records K=5 Pearson r, then commits the best-performing weight to config.

Usage:
    uv run python scripts/sweep_presidential_weight.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"

SWEEP_WEIGHTS = [2.5, 4.0, 5.0, 6.0, 8.0]
TARGET_K = 5
T3_BASELINE = 0.9687  # K=5 r from T3 (presidential_weight=8.0, 10-pair window)

PRES_WEIGHT_PATTERN = re.compile(r"^(\s*presidential_weight:\s*)[\d.]+(.*)$", re.MULTILINE)
K5_PATTERN = re.compile(r"^\s+5\s+([\d.-]+)", re.MULTILINE)


def set_presidential_weight(weight: float) -> None:
    text = CONFIG_PATH.read_text()
    new_text = PRES_WEIGHT_PATTERN.sub(rf"\g<1>{weight}\g<2>", text)
    if new_text == text:
        raise RuntimeError(f"presidential_weight pattern not found in {CONFIG_PATH}")
    CONFIG_PATH.write_text(new_text)


def run_validation() -> tuple[str, float | None]:
    result = subprocess.run(
        ["uv", "run", "python", "-m", "src.validation.validate_county_holdout_multiyear"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = result.stdout + result.stderr
    match = K5_PATTERN.search(result.stdout)
    k5_r = float(match.group(1)) if match else None
    return combined, k5_r


def main() -> None:
    print(f"Presidential weight sweep — T3 baseline K={TARGET_K} r={T3_BASELINE}")
    print(f"Sweeping weights: {SWEEP_WEIGHTS}")
    print()

    results: list[tuple[float, float]] = []

    for weight in SWEEP_WEIGHTS:
        print(f"--- presidential_weight={weight} ---")
        set_presidential_weight(weight)
        output, k5_r = run_validation()
        if k5_r is None:
            print(f"  ERROR: could not parse K=5 r from output")
            print(output[-500:])
        else:
            print(f"  K={TARGET_K} r = {k5_r:.4f}  (delta vs T3 baseline: {k5_r - T3_BASELINE:+.4f})")
            results.append((weight, k5_r))
        print()

    if not results:
        print("No valid results — aborting.", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    print(f"{'weight':>8}  {'K=5 r':>8}  {'delta':>8}")
    print("-" * 50)
    for w, r in results:
        print(f"{w:>8.1f}  {r:>8.4f}  {r - T3_BASELINE:>+8.4f}")

    best_weight, best_r = max(results, key=lambda x: x[1])
    print(f"\nBest: presidential_weight={best_weight}  K=5 r={best_r:.4f}")
    meets_baseline = best_r >= T3_BASELINE
    print(f"Meets T3 baseline ({T3_BASELINE}): {'YES' if meets_baseline else 'NO'}")

    print(f"\nSetting config/model.yaml presidential_weight={best_weight}")
    set_presidential_weight(best_weight)

    print("\nSweep complete. Commit the result with:")
    print(f"  git add config/model.yaml src/validation/validate_county_holdout_multiyear.py")
    print(f"  git commit -m 'feat(P6.1/T4): retune presidential_weight={best_weight} post-1980 trim'")

    return results, best_weight, best_r


if __name__ == "__main__":
    main()
