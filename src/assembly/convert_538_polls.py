"""
Convert FiveThirtyEight raw_polls.csv into our internal poll format.

Source: data/raw/fivethirtyeight/data-repo/pollster-ratings/raw_polls.csv
Target: data/polls/polls_{cycle}.csv

Usage:
  python -m src.assembly.convert_538_polls --cycles 2020 2022 --states FL GA AL
  python -m src.assembly.convert_538_polls --cycles 2020 --all-states
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.propagation.propagate_polls import PollObservation  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_POLLS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fivethirtyeight"
    / "data-repo"
    / "pollster-ratings"
    / "raw_polls.csv"
)

POLLSTER_RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fivethirtyeight"
    / "data-repo"
    / "pollster-ratings"
    / "pollster-ratings-combined.csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "polls"

# Map 538 type_simple → our race label suffix
RACE_TYPE_MAP = {
    "Pres-G": "President",
    "Sen-G": "Senate",
    "Gov-G": "Governor",
}

# Types to skip entirely
SKIP_TYPES = {"Pres-P", "Sen-P", "Gov-P", "House-G", "House-G-US", "House-P"}

OUTPUT_COLUMNS = [
    "race",
    "geography",
    "geo_level",
    "dem_share",
    "n_sample",
    "date",
    "pollster",
    "notes",
]


# ---------------------------------------------------------------------------
# Pollster ratings
# ---------------------------------------------------------------------------


def load_pollster_ratings(
    path: Optional[Path] = None,
) -> dict[int, dict]:
    """
    Load 538 pollster ratings into a dict keyed by pollster_rating_id.

    Returns:
        dict of pollster_rating_id → {
            "pollster": str,
            "numeric_grade": float,
            "pollscore": float,
            "bias": float,
        }
    """
    path = path or POLLSTER_RATINGS_PATH
    if not path.exists():
        log.warning("Pollster ratings file not found: %s", path)
        return {}

    ratings = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rating_id = int(row["pollster_rating_id"])
            except (ValueError, KeyError):
                continue

            def _safe_float(val: str) -> Optional[float]:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

            ratings[rating_id] = {
                "pollster": row.get("pollster", ""),
                "numeric_grade": _safe_float(row.get("numeric_grade", "")),
                "pollscore": _safe_float(row.get("POLLSCORE", "")),
                "bias": _safe_float(row.get("bias_ppm", "")),
            }

    log.info("Loaded %d pollster ratings", len(ratings))
    return ratings


# ---------------------------------------------------------------------------
# Two-party share computation
# ---------------------------------------------------------------------------


def compute_two_party_dem_share(row: dict) -> Optional[float]:
    """
    Compute Democratic two-party share from a 538 raw_polls row.

    Identifies which candidate is DEM and which is REP from cand1_party/cand2_party.
    Returns dem_pct / (dem_pct + rep_pct), or None if parties can't be identified.
    """
    c1_party = row.get("cand1_party", "").strip().upper()
    c2_party = row.get("cand2_party", "").strip().upper()

    try:
        c1_pct = float(row.get("cand1_pct", ""))
        c2_pct = float(row.get("cand2_pct", ""))
    except (ValueError, TypeError):
        return None

    dem_pct = None
    rep_pct = None

    if c1_party == "DEM" and c2_party == "REP":
        dem_pct, rep_pct = c1_pct, c2_pct
    elif c1_party == "REP" and c2_party == "DEM":
        dem_pct, rep_pct = c2_pct, c1_pct
    else:
        # Can't identify D vs R — skip
        return None

    total = dem_pct + rep_pct
    if total <= 0:
        return None

    return dem_pct / total


# ---------------------------------------------------------------------------
# Race name formatting
# ---------------------------------------------------------------------------


def format_race_name(cycle: str, location: str, type_simple: str) -> Optional[str]:
    """
    Format race name like '2020 FL President'.

    Returns None if type_simple is not in RACE_TYPE_MAP.
    """
    race_label = RACE_TYPE_MAP.get(type_simple)
    if race_label is None:
        return None
    return f"{cycle} {location} {race_label}"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def _parse_sample_size(row: dict, row_num: int, skipped: dict) -> Optional[int]:
    """Parse and validate the samplesize field from a raw_polls row.

    Returns the integer sample size, or None if the field is missing,
    unparseable, or non-positive. Logs warnings for bad values and
    increments the 'no_sample' skip counter.
    """
    raw_sample = row.get("samplesize", "").strip()
    if not raw_sample:
        log.warning(
            "Row %d skipped: missing samplesize (poll_id=%s, race=%s)",
            row_num, row.get("poll_id", "?"), row.get("race", "?"),
        )
        skipped["no_sample"] += 1
        return None
    try:
        n = int(float(raw_sample))
    except ValueError:
        log.warning("Row %d skipped: invalid samplesize=%r", row_num, raw_sample)
        skipped["no_sample"] += 1
        return None
    if n <= 0:
        skipped["no_sample"] += 1
        return None
    return n


def _build_notes(
    row: dict,
    ratings: dict[int, dict],
    enrich: bool,
) -> str:
    """Build the notes string for a poll row, optionally enriched with ratings.

    Notes include: method, partisan flag, rating_id, and (if enrich) the
    numeric grade, pollscore, and bias from the pollster ratings lookup.
    """
    methodology = row.get("methodology", "").strip()
    partisan = row.get("partisan", "").strip()
    rating_id_str = row.get("pollster_rating_id", "").strip()

    parts = []
    if methodology:
        parts.append(f"method={methodology}")
    if partisan and partisan != "NA":
        parts.append(f"partisan={partisan}")
    if rating_id_str:
        parts.append(f"rating_id={rating_id_str}")

    if enrich and rating_id_str:
        try:
            rid = int(rating_id_str)
            r = ratings.get(rid)
            if r:
                if r["numeric_grade"] is not None:
                    parts.append(f"grade={r['numeric_grade']}")
                if r["pollscore"] is not None:
                    parts.append(f"pollscore={r['pollscore']}")
                if r["bias"] is not None:
                    parts.append(f"bias={r['bias']}")
        except ValueError:
            pass

    return "; ".join(parts)


def _filter_raw_row(
    row: dict,
    row_num: int,
    cycle_set: set,
    allowed_types: set,
    state_set: Optional[set],
    skipped: dict,
) -> Optional[tuple[str, str, str, int, float]]:
    """Apply all row-level filters and return (cycle, type_simple, location, n_sample, dem_share).

    Returns None with the appropriate skip counter incremented when the row
    fails any filter. Separates the gate-keeping logic from the object building.
    """
    cycle = row.get("cycle", "").strip()
    if cycle not in cycle_set:
        skipped["wrong_cycle"] += 1
        return None

    type_simple = row.get("type_simple", "").strip()
    if type_simple not in allowed_types:
        skipped["wrong_type"] += 1
        return None

    location = row.get("location", "").strip()
    if state_set and location not in state_set:
        skipped["wrong_state"] += 1
        return None

    n_sample = _parse_sample_size(row, row_num, skipped)
    if n_sample is None:
        return None

    dem_share = compute_two_party_dem_share(row)
    if dem_share is None:
        skipped["no_party"] += 1
        return None
    if not (0.0 < dem_share < 1.0):
        skipped["bad_share"] += 1
        return None

    return cycle, type_simple, location, n_sample, dem_share


def _build_poll_record(
    row: dict,
    cycle: str,
    type_simple: str,
    location: str,
    n_sample: int,
    dem_share: float,
    ratings: dict[int, dict],
    enrich: bool,
    skipped: dict,
) -> Optional[tuple[str, PollObservation, dict]]:
    """Build PollObservation and CSV row dict from a validated row's fields.

    Returns None (with skip counter incremented) only if the race name cannot
    be formatted (should not happen after _filter_raw_row, but guards safely).
    """
    race_name = format_race_name(cycle, location, type_simple)
    if race_name is None:
        skipped["wrong_type"] += 1
        return None

    date = row.get("polldate", "").strip()
    pollster = row.get("pollster", "").strip()
    notes = _build_notes(row, ratings, enrich)

    obs = PollObservation(
        geography=location,
        dem_share=round(dem_share, 6),
        n_sample=n_sample,
        race=race_name,
        date=date,
        pollster=pollster,
        geo_level="state",
    )
    csv_row = {
        "race": race_name,
        "geography": location,
        "geo_level": "state",
        "dem_share": f"{dem_share:.6f}",
        "n_sample": str(n_sample),
        "date": date,
        "pollster": pollster,
        "notes": notes,
    }
    return cycle, obs, csv_row


def _process_raw_polls_row(
    row: dict,
    row_num: int,
    cycle_set: set,
    allowed_types: set,
    state_set: Optional[set],
    ratings: dict[int, dict],
    enrich: bool,
    skipped: dict,
) -> Optional[tuple[str, PollObservation, dict]]:
    """Filter and parse a raw_polls CSV row into (cycle, observation, csv_row).

    Returns None (with skip counter updated) if the row fails any gate.
    Delegates filtering to _filter_raw_row and building to _build_poll_record.
    """
    filtered = _filter_raw_row(row, row_num, cycle_set, allowed_types, state_set, skipped)
    if filtered is None:
        return None
    cycle, type_simple, location, n_sample, dem_share = filtered
    return _build_poll_record(row, cycle, type_simple, location, n_sample, dem_share,
                               ratings, enrich, skipped)


def _write_cycle_csv(cycle: str, rows: list[dict], output_dir: Path) -> None:
    """Write the CSV rows for a single cycle to data/polls/polls_{cycle}.csv."""
    if not rows:
        log.warning("No polls found for cycle %s", cycle)
        return
    out_path = output_dir / f"polls_{cycle}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d polls to %s", len(rows), out_path)


def _log_conversion_summary(
    result: dict[str, list[PollObservation]],
    skipped: dict,
) -> None:
    """Log per-cycle state breakdown and total skipped row counts."""
    for cycle, obs_list in sorted(result.items()):
        if obs_list:
            by_state: dict[str, int] = {}
            for obs in obs_list:
                by_state[obs.geography] = by_state.get(obs.geography, 0) + 1
            log.info("Cycle %s: %d polls across %d states", cycle, len(obs_list), len(by_state))
            for state, count in sorted(by_state.items()):
                log.info("  %s: %d polls", state, count)
    log.info(
        "Skipped rows: %s",
        ", ".join(f"{k}={v}" for k, v in skipped.items() if v > 0),
    )


def _resolve_filter_sets(
    cycles: list[str],
    states: Optional[list[str]],
    race_types: Optional[list[str]],
) -> tuple[set, set, Optional[set]]:
    """Resolve the cycle, type, and state filter sets from user arguments."""
    cycle_set = set(cycles)
    allowed_types = (
        set(race_types) & set(RACE_TYPE_MAP.keys())
        if race_types is not None
        else set(RACE_TYPE_MAP.keys())
    )
    state_set = set(states) if states else None
    return cycle_set, allowed_types, state_set


def convert_538_polls(
    cycles: list[str],
    states: Optional[list[str]] = None,
    race_types: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    raw_polls_path: Optional[Path] = None,
    ratings_path: Optional[Path] = None,
    enrich: bool = True,
) -> dict[str, list[PollObservation]]:
    """Convert 538 raw_polls.csv into our internal poll format.

    Returns dict of cycle → list of PollObservation, and writes per-cycle
    CSV files to output_dir. Pass enrich=False to skip rating enrichment.
    """
    raw_path = raw_polls_path or RAW_POLLS_PATH
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raise FileNotFoundError(f"538 raw polls not found: {raw_path}")

    cycle_set, allowed_types, state_set = _resolve_filter_sets(cycles, states, race_types)
    ratings = load_pollster_ratings(ratings_path) if enrich else {}

    result: dict[str, list[PollObservation]] = {c: [] for c in cycles}
    csv_rows: dict[str, list[dict]] = {c: [] for c in cycles}
    skipped = {"no_party": 0, "no_sample": 0, "wrong_type": 0, "wrong_cycle": 0,
               "wrong_state": 0, "bad_share": 0}

    with raw_path.open(newline="", encoding="utf-8") as f:
        for row_num, row in enumerate(csv.DictReader(f), start=2):
            parsed = _process_raw_polls_row(
                row, row_num, cycle_set, allowed_types, state_set,
                ratings, enrich, skipped,
            )
            if parsed is None:
                continue
            cycle, obs, csv_row = parsed
            result[cycle].append(obs)
            csv_rows[cycle].append(csv_row)

    for cycle in cycles:
        _write_cycle_csv(cycle, csv_rows[cycle], output_dir)

    _log_conversion_summary(result, skipped)
    return result


# ---------------------------------------------------------------------------
# Enrichment helper (standalone, for use after loading)
# ---------------------------------------------------------------------------


def enrich_with_ratings(
    polls: list[PollObservation],
    ratings: dict[int, dict],
) -> list[dict]:
    """
    Add pollster quality scores to poll observations.

    Returns a list of dicts with poll fields plus rating fields.
    This is for downstream weighting — not CSV output.
    """
    enriched = []
    for obs in polls:
        entry = {
            "race": obs.race,
            "geography": obs.geography,
            "geo_level": obs.geo_level,
            "dem_share": obs.dem_share,
            "n_sample": obs.n_sample,
            "date": obs.date,
            "pollster": obs.pollster,
            "numeric_grade": None,
            "pollscore": None,
            "bias": None,
        }
        enriched.append(entry)

    return enriched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert 538 raw_polls.csv to internal poll format"
    )
    parser.add_argument("--cycles", nargs="+", required=True,
                        help="Cycle years to convert (e.g., 2020 2022)")
    parser.add_argument("--states", nargs="+", default=None,
                        help="State abbreviations to filter (default: FL GA AL)")
    parser.add_argument("--all-states", action="store_true",
                        help="Include all states (overrides --states)")
    parser.add_argument("--race-types", nargs="+", default=None,
                        help="Race types (e.g., Pres-G Sen-G Gov-G). Default: all general.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data/polls/)")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip pollster rating enrichment")
    return parser


def _print_cycle_summary(result: dict[str, list[PollObservation]]) -> None:
    """Print a per-cycle, per-state/race breakdown to stdout."""
    for cycle, polls in sorted(result.items()):
        print(f"\nCycle {cycle}: {len(polls)} polls")
        by_state_race: dict[str, int] = {}
        for p in polls:
            race_label = p.race.split(" ", 2)[-1] if " " in p.race else p.race
            key = f"  {p.geography} / {race_label}"
            by_state_race[key] = by_state_race.get(key, 0) + 1
        for key, count in sorted(by_state_race.items()):
            print(f"{key}: {count}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    states = None if args.all_states else (args.states or ["FL", "GA", "AL"])
    result = convert_538_polls(
        cycles=args.cycles,
        states=states,
        race_types=args.race_types,
        output_dir=args.output_dir,
        enrich=not args.no_enrich,
    )
    _print_cycle_summary(result)


if __name__ == "__main__":
    main()
