"""Forecast change detection: compare race-level predictions before/after a DuckDB rebuild.

Workflow
--------
1. Before a rebuild, call snapshot_predictions() to capture the current state.
2. Rebuild the DuckDB (run build_database.py / ingest_polls.py etc.).
3. After the rebuild, call snapshot_predictions() again.
4. Pass both snapshots to compute_diff() to identify meaningful changes.
5. Pass diff results to format_summary() to produce a human-readable report.

CLI usage
---------
    # Snapshot before rebuild
    python -m src.reporting.forecast_diff --snapshot --out before.json

    # Snapshot after rebuild
    python -m src.reporting.forecast_diff --snapshot --out after.json

    # Compare
    python -m src.reporting.forecast_diff --before before.json --after after.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict

import duckdb

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "wethervane.duckdb"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class RaceDiff(TypedDict):
    race: str
    before: float
    after: float
    delta: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def snapshot_predictions(db_path: str | Path = DEFAULT_DB) -> dict[str, float]:
    """Return a mapping of race -> average pred_dem_share from the predictions table.

    Only races with at least one non-NULL pred_dem_share row are included.
    The average is taken over all counties for that race (using the most recent
    version_id when multiple versions exist).

    Parameters
    ----------
    db_path:
        Path to the WetherVane DuckDB file.

    Returns
    -------
    dict[str, float]
        ``{race_name: avg_pred_dem_share}`` for every race present in the DB.
        Empty dict if the predictions table is absent or empty.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log.warning("DB not found at %s — returning empty snapshot", db_path)
        return {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Check if table exists
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "predictions" not in tables:
            log.warning("predictions table not found in %s", db_path)
            return {}

        # Use the latest version_id so we compare apples-to-apples.
        # If multiple version_ids exist, pick the most recent.
        #
        # IMPORTANT: filter counties to the race's own state.
        # The predictions table stores all 3,154 US counties for every race,
        # with out-of-state counties carrying the national baseline value (~0.318).
        # Averaging across all counties produces a misleading ~48.3% D number
        # for most races — diluted by ~3,100 out-of-state baseline rows.
        #
        # We join through the `races` table (race_id → state) to restrict the
        # aggregate to the counties that actually vote in each race. If `races`
        # is absent (e.g. older schema or small test DBs), we fall back to the
        # simple county average — correct for single-state test DBs but diluted
        # for national DBs.
        has_races_table = "races" in tables
        has_counties_table = "counties" in tables

        if has_races_table and has_counties_table:
            result = con.execute("""
                WITH latest AS (
                    SELECT MAX(version_id) AS vid
                    FROM predictions
                    WHERE pred_dem_share IS NOT NULL
                )
                SELECT
                    p.race,
                    CASE WHEN SUM(COALESCE(c.total_votes_2024, 0)) > 0
                         THEN SUM(p.pred_dem_share * COALESCE(c.total_votes_2024, 0))
                              / SUM(COALESCE(c.total_votes_2024, 0))
                         ELSE AVG(p.pred_dem_share)
                    END AS avg_pred_dem_share
                FROM predictions p, latest
                JOIN counties c ON p.county_fips = c.county_fips
                JOIN races r ON p.race = r.race_id
                WHERE p.version_id = latest.vid
                  AND p.pred_dem_share IS NOT NULL
                  AND c.state_abbr = r.state
                GROUP BY p.race
                ORDER BY p.race
            """).fetchall()
        else:
            # Fallback for DBs without a races table (e.g. older schema or test DBs).
            result = con.execute("""
                WITH latest AS (
                    SELECT MAX(version_id) AS vid
                    FROM predictions
                    WHERE pred_dem_share IS NOT NULL
                )
                SELECT
                    race,
                    AVG(pred_dem_share) AS avg_pred_dem_share
                FROM predictions, latest
                WHERE version_id = vid
                  AND pred_dem_share IS NOT NULL
                  AND race != 'baseline'
                GROUP BY race
                ORDER BY race
            """).fetchall()

        return {row[0]: float(row[1]) for row in result}
    finally:
        con.close()


def compute_diff(
    before: dict[str, float],
    after: dict[str, float],
    threshold: float = 0.005,
) -> list[RaceDiff]:
    """Compare two prediction snapshots and return races with meaningful changes.

    Parameters
    ----------
    before:
        Snapshot dict returned by snapshot_predictions() before the rebuild.
    after:
        Snapshot dict returned by snapshot_predictions() after the rebuild.
    threshold:
        Minimum absolute change in avg_pred_dem_share to include in results.
        Default 0.005 (0.5 percentage points).

    Returns
    -------
    list[RaceDiff]
        Races where ``|after - before| >= threshold``, sorted by descending
        absolute delta. Races that appear only in one snapshot are included
        with the missing side represented as NaN.
    """
    import math

    all_races = sorted(set(before) | set(after))
    diffs: list[RaceDiff] = []

    for race in all_races:
        b = before.get(race, float("nan"))
        a = after.get(race, float("nan"))

        if math.isnan(b) or math.isnan(a):
            # Race appeared or disappeared — always include.
            delta = float("nan")
        else:
            delta = a - b

        if math.isnan(delta) or abs(delta) >= threshold:
            diffs.append(RaceDiff(race=race, before=b, after=a, delta=delta))

    # Sort: NaN deltas (new/removed races) first, then by descending |delta|.
    def sort_key(d: RaceDiff):
        import math
        return (0 if math.isnan(d["delta"]) else 1, -abs(d["delta"]) if not math.isnan(d["delta"]) else 0)

    diffs.sort(key=sort_key)
    return diffs


def format_summary(diff_results: list[RaceDiff]) -> str:
    """Format diff results as a human-readable summary string.

    Parameters
    ----------
    diff_results:
        Output of compute_diff().

    Returns
    -------
    str
        Multi-line summary. Empty string if no changes detected.
    """
    import math

    if not diff_results:
        return "No meaningful forecast changes detected."

    lines = [
        f"Forecast changes detected in {len(diff_results)} race(s):",
        "",
    ]

    for d in diff_results:
        race = d["race"]
        b = d["before"]
        a = d["after"]
        delta = d["delta"]

        if math.isnan(b):
            lines.append(f"  {race:<40}  NEW RACE   after={a:.1%}")
        elif math.isnan(a):
            lines.append(f"  {race:<40}  REMOVED    before={b:.1%}")
        else:
            direction = "+" if delta > 0 else ""
            lines.append(
                f"  {race:<40}  {b:.1%} -> {a:.1%}  ({direction}{delta:+.2%})"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Snapshot and diff WetherVane race-level predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Path to wethervane.duckdb (default: %(default)s)",
    )
    subparsers = p.add_subparsers(dest="command")

    # --snapshot mode
    snap = subparsers.add_parser("snapshot", help="Capture current predictions to JSON")
    snap.add_argument("--out", required=True, help="Output JSON file path")

    # --diff mode (positional: --before / --after)
    diff = subparsers.add_parser("diff", help="Compare two snapshot JSON files")
    diff.add_argument("--before", required=True, help="JSON snapshot taken before rebuild")
    diff.add_argument("--after", required=True, help="JSON snapshot taken after rebuild")
    diff.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Minimum absolute change to report (default: %(default)s)",
    )

    # Legacy positional: --before / --after at top level (backward compat)
    p.add_argument("--before", help="JSON snapshot taken before rebuild (top-level mode)")
    p.add_argument("--after", help="JSON snapshot taken after rebuild (top-level mode)")
    p.add_argument("--snapshot", action="store_true", help="Take a snapshot (top-level mode)")
    p.add_argument("--out", help="Output JSON file path for top-level --snapshot")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Minimum absolute change to report (default: %(default)s)",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Handle top-level --snapshot / --before --after modes
    if args.before and args.after:
        before = json.loads(Path(args.before).read_text())
        after = json.loads(Path(args.after).read_text())
        threshold = getattr(args, "threshold", 0.005)
        diffs = compute_diff(before, after, threshold=threshold)
        print(format_summary(diffs))
        return

    if getattr(args, "snapshot", False):
        out = getattr(args, "out", None)
        if not out:
            parser.error("--snapshot requires --out")
        snap = snapshot_predictions(args.db)
        Path(out).write_text(json.dumps(snap, indent=2))
        log.info("Snapshot written to %s (%d races)", out, len(snap))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
