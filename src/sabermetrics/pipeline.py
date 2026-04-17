"""Sabermetrics pipeline orchestrator — Phase 2.

Runs the full sabermetrics pipeline in order:
  1. Load candidate registry (Phase 1 output)
  2. Compute MVD (state-level residuals) for all candidate-races
  3. Compute CTOV (100-dim type decomposition) for all candidate-races
  4. Compute CEC (cross-election consistency) for multi-race candidates
  5. Derive badges from CTOV + demographic profiles
  6. Write output artifacts to data/sabermetrics/

Output files:
  data/sabermetrics/candidate_residuals.parquet  — MVD per candidate-race
  data/sabermetrics/candidate_ctov.parquet       — CTOV per candidate-race
  data/sabermetrics/candidate_badges.json        — badges per candidate

The backtest harness is called once per (year, race_type) combination and
results are cached in memory. With 5 senate years + 2 governor years, this
is 7 backtest calls (~0.8 sec each = ~6 sec total).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = PROJECT_ROOT / "data" / "sabermetrics"


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def load_registry(
    registry_path: str | Path = "data/sabermetrics/candidate_registry.json",
) -> dict:
    """Load the candidate registry from disk.

    Parameters
    ----------
    registry_path : str | Path
        Path to candidate_registry.json (relative to project root or absolute).
    """
    path = Path(registry_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Registry not found at {path}. "
            "Run the Phase 1 registry builder first: "
            "uv run python -m src.sabermetrics.registry"
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_sabermetrics_pipeline(
    registry_path: str | Path = "data/sabermetrics/candidate_registry.json",
    output_dir: str | Path | None = None,
    overwrite: bool = True,
) -> dict:
    """Run the full Phase 2 sabermetrics pipeline.

    Parameters
    ----------
    registry_path : str | Path
        Path to candidate_registry.json.
    output_dir : str | Path | None
        Directory for output parquet/json files. Defaults to data/sabermetrics/.
    overwrite : bool
        Whether to overwrite existing output files.

    Returns
    -------
    dict
        Summary of pipeline results:
          n_persons, n_races_processed, n_with_ctov, n_multi_race,
          cec_mean, cec_std, n_badges_total
    """
    out_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    residuals_path = out_dir / "candidate_residuals.parquet"
    ctov_path = out_dir / "candidate_ctov.parquet"
    badges_path = out_dir / "candidate_badges.json"

    if not overwrite and residuals_path.exists() and ctov_path.exists() and badges_path.exists():
        log.info("All output files exist and overwrite=False — skipping pipeline")
        return {}

    # --- Step 1: Load registry ---
    log.info("Loading candidate registry...")
    registry = load_registry(registry_path)
    persons = registry["persons"]
    n_persons = len(persons)
    log.info("  Registry: %d persons", n_persons)

    # --- Step 2: Compute MVD ---
    log.info("Computing MVD (state-level residuals)...")
    from src.sabermetrics.residuals import (
        _build_backtest_cache,
        compute_cec_for_all_candidates,
        compute_ctov,
        compute_mvd,
    )

    # Build the backtest cache once and share it with both compute_mvd and
    # compute_ctov. Without sharing, each function independently reruns
    # all backtests — doubling the compute time (~6 sec savings).
    log.info("  Building backtest cache for all (year, office) combinations...")
    backtest_cache = _build_backtest_cache(registry)

    mvd_df = compute_mvd(registry, backtest_cache=backtest_cache)
    log.info("  MVD: %d candidate-races", len(mvd_df))

    # Write residuals immediately (useful even if CTOV fails)
    mvd_df.to_parquet(residuals_path, index=False)
    log.info("  Wrote %s", residuals_path)

    # --- Step 3: Compute CTOV ---
    log.info("Computing CTOV (100-dim type decomposition)...")
    ctov_df = compute_ctov(mvd_df, registry, backtest_cache=backtest_cache)
    log.info("  CTOV: %d candidate-races", len(ctov_df))

    ctov_df.to_parquet(ctov_path, index=False)
    log.info("  Wrote %s", ctov_path)

    # --- Step 4: Compute CEC ---
    log.info("Computing CEC (cross-election consistency)...")
    cec_df = compute_cec_for_all_candidates(ctov_df)
    multi_race_cec = cec_df[cec_df["n_races"] > 1]
    cec_mean = float(multi_race_cec["cec"].mean()) if len(multi_race_cec) > 0 else float("nan")
    cec_std = float(multi_race_cec["cec"].std()) if len(multi_race_cec) > 0 else float("nan")
    log.info(
        "  CEC: %d multi-race candidates, mean=%.3f, std=%.3f",
        len(multi_race_cec),
        cec_mean,
        cec_std,
    )

    # --- Step 5: Derive badges ---
    log.info("Deriving badges from CTOV + demographics...")
    from src.sabermetrics.badges import derive_badges, derive_signature_badges

    badges = derive_badges(ctov_df, mvd_df)
    n_badges_total = sum(len(v["badges"]) for v in badges.values())
    log.info("  Badges: %d total awarded across %d candidates", n_badges_total, len(badges))

    # --- Step 5b: Derive signature (auto-discovered) badges ---
    log.info("Deriving signature badges from residual CTOV...")
    sig_badges = derive_signature_badges(ctov_df)
    n_sig_total = sum(len(v["signature_badges"]) for v in sig_badges.values())
    log.info("  Signature badges: %d total across %d candidates", n_sig_total, len(sig_badges))

    # Merge signature badges into the main badge output and attach CEC scores
    cec_by_person = dict(zip(cec_df["person_id"], cec_df["cec"]))
    n_races_by_person = dict(zip(cec_df["person_id"], cec_df["n_races"]))
    for person_id, badge_entry in badges.items():
        badge_entry["cec"] = cec_by_person.get(person_id, None)
        badge_entry["n_races"] = n_races_by_person.get(person_id, badge_entry.get("n_races", 1))
        # Append signature badge details into badge_details list
        sig_entry = sig_badges.get(person_id, {})
        sig_list = sig_entry.get("signature_badges", [])
        badge_entry.setdefault("badge_details", [])
        badge_entry["badge_details"].extend(sig_list)
        # Keep "badges" list in sync for backward compat
        badge_entry["badges"].extend(d["name"] for d in sig_list)

    with open(badges_path, "w") as f:
        json.dump(badges, f, indent=2)
    log.info("  Wrote %s", badges_path)

    summary = {
        "n_persons": n_persons,
        "n_races_processed": len(mvd_df),
        "n_with_ctov": len(ctov_df),
        "n_multi_race": len(multi_race_cec),
        "cec_mean": cec_mean,
        "cec_std": cec_std,
        "n_badges_total": n_badges_total,
    }

    log.info("Pipeline complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the WetherVane sabermetrics Phase 2 pipeline",
    )
    parser.add_argument(
        "--registry",
        default="data/sabermetrics/candidate_registry.json",
        help="Path to candidate_registry.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sabermetrics",
        help="Output directory for parquet/json files",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip pipeline if output files already exist",
    )
    args = parser.parse_args()

    summary = run_sabermetrics_pipeline(
        registry_path=args.registry,
        output_dir=args.output_dir,
        overwrite=not args.no_overwrite,
    )

    if summary:
        print("\nPipeline summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _main()
