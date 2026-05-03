"""Daily forecast snapshot runner.

Runs make_xt_impact_report, writes per-race results to data/forecasts/YYYY-MM-DD.json,
computes delta vs the previous day's snapshot, and prints top-5 races by absolute delta.

The per-race ``win_prob`` value is the enriched xt-impact delta (pp) from
make_xt_impact_report — the amount by which demographic crosstab polls shift the
state-level forecast versus the xt-stripped baseline.  Tracking this value day over
day shows prediction drift as new polls arrive.

Usage (from wethervane repo root):
    python scripts/run_daily_forecast.py
    python scripts/run_daily_forecast.py --races "2026 AZ Senate" "2026 GA Senate"
    python scripts/run_daily_forecast.py --forecast-dir data/custom_forecasts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"


# ---------------------------------------------------------------------------
# Pure helpers (fully testable without mocking)
# ---------------------------------------------------------------------------


def find_previous_snapshot(forecast_dir: Path, today_str: str) -> Path | None:
    """Return the most recent *.json in forecast_dir whose stem is not today_str."""
    if not forecast_dir.exists():
        return None
    candidates = sorted(f for f in forecast_dir.glob("*.json") if f.stem != today_str)
    return candidates[-1] if candidates else None


def load_snapshot_win_probs(path: Path) -> dict[str, float] | None:
    """Load {race: win_prob} from a dated forecast snapshot written by write_snapshot."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    races_data = data.get("races")
    if not isinstance(races_data, list):
        return None
    return {
        entry["race"]: float(entry["win_prob"])
        for entry in races_data
        if "race" in entry and "win_prob" in entry
    }


def compute_deltas(
    today_probs: dict[str, float],
    previous_probs: dict[str, float] | None,
) -> dict[str, float | None]:
    """Return per-race delta_pp_vs_yesterday.

    Returns None for a race when there is no prior baseline (first run or race
    not present in previous snapshot).
    """
    if previous_probs is None:
        return {race: None for race in today_probs}
    return {
        race: round(prob - previous_probs[race], 6) if race in previous_probs else None
        for race, prob in today_probs.items()
    }


def count_xt_polls_per_race(project_root: Path) -> dict[str, int]:
    """Return {race_id: n_polls_with_xt_fields} from polls_2026.csv."""
    import pandas as pd

    polls_path = project_root / "data" / "polls" / "polls_2026.csv"
    if not polls_path.exists():
        return {}
    df = pd.read_csv(polls_path)
    if "race" not in df.columns:
        return {}
    xt_cols = [c for c in df.columns if c.startswith("xt_")]
    counts: dict[str, int] = {}
    for race_id, grp in df.groupby("race"):
        if xt_cols:
            n = int(grp[xt_cols].notna().any(axis=1).sum())
        else:
            n = 0
        counts[str(race_id)] = n
    return counts


def build_race_records(
    enriched_deltas: dict[str, float],
    xt_poll_counts: dict[str, int],
    deltas_vs_prev: dict[str, float | None],
) -> list[dict]:
    """Assemble the per-race output records sorted by race ID."""
    return [
        {
            "race": race_id,
            "win_prob": round(win_prob, 6),
            "enriched_poll_count": xt_poll_counts.get(race_id, 0),
            "delta_pp_vs_yesterday": deltas_vs_prev.get(race_id),
        }
        for race_id, win_prob in sorted(enriched_deltas.items())
    ]


def write_snapshot(records: list[dict], report_date: str, forecast_dir: Path) -> Path:
    """Write dated snapshot JSON and return the path."""
    forecast_dir.mkdir(parents=True, exist_ok=True)
    out_path = forecast_dir / f"{report_date}.json"
    out_path.write_text(
        json.dumps({"date": report_date, "races": records}, indent=2),
        encoding="utf-8",
    )
    return out_path


def print_top_movers(records: list[dict], n: int = 5) -> None:
    """Print the top-n races by absolute delta_pp_vs_yesterday."""
    movers = [r for r in records if r["delta_pp_vs_yesterday"] is not None]
    movers.sort(key=lambda r: -abs(r["delta_pp_vs_yesterday"]))
    top = movers[:n]
    print(f"\nTop {len(top)} race(s) by |delta_pp_vs_yesterday|:")
    if not top:
        print("  (first run — no prior snapshot for comparison)")
        return
    for r in top:
        sign = "+" if r["delta_pp_vs_yesterday"] >= 0 else ""
        print(
            f"  {r['race']}: {sign}{r['delta_pp_vs_yesterday']:.3f}pp"
            f"  (win_prob={r['win_prob']:.3f}pp, xt_polls={r['enriched_poll_count']})"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_daily_forecast(
    races: list[str] | None = None,
    forecast_dir: Path = DEFAULT_FORECAST_DIR,
    _forecast_fn=None,
) -> Path:
    """Run the daily forecast, write snapshot, return path to the written file.

    Parameters
    ----------
    races:
        Optional race-ID filter forwarded to make_xt_impact_report.
    forecast_dir:
        Directory for dated snapshot files.
    _forecast_fn:
        Injected alternative to make_xt_impact_report — used in tests only.
    """
    if _forecast_fn is None:
        from src.prediction.forecast_engine import make_xt_impact_report as _forecast_fn

    report = _forecast_fn(races=races)
    enriched_deltas: dict[str, float] = report["enriched_deltas"]
    report_date: str = report["report_date"]

    xt_poll_counts = count_xt_polls_per_race(PROJECT_ROOT)

    prev_path = find_previous_snapshot(forecast_dir, report_date)
    previous_probs = load_snapshot_win_probs(prev_path) if prev_path else None

    deltas_vs_prev = compute_deltas(enriched_deltas, previous_probs)
    records = build_race_records(enriched_deltas, xt_poll_counts, deltas_vs_prev)
    out_path = write_snapshot(records, report_date, forecast_dir)

    print(
        f"Forecast written: {out_path} ({len(records)} races)\n"
        f"  mean_delta={report['mean_delta']:.3f}pp"
        f"  max_delta={report['max_delta']:.3f}pp"
        f"  races_with_xt={report['races_with_xt']}"
    )
    print_top_movers(records)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--races",
        nargs="+",
        default=None,
        metavar="RACE",
        help="Restrict to these race IDs (e.g. '2026 AZ Senate')",
    )
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=DEFAULT_FORECAST_DIR,
        help="Output directory for dated snapshot files (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        run_daily_forecast(races=args.races, forecast_dir=args.forecast_dir)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
