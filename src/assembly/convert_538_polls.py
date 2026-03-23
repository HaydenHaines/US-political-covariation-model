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


def convert_538_polls(
    cycles: list[str],
    states: Optional[list[str]] = None,
    race_types: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    raw_polls_path: Optional[Path] = None,
    ratings_path: Optional[Path] = None,
    enrich: bool = True,
) -> dict[str, list[PollObservation]]:
    """
    Convert 538 raw_polls.csv into our internal poll format.

    Args:
        cycles: list of cycle years to convert (e.g., ["2020", "2022"])
        states: list of state abbreviations to filter, or None for all
        race_types: list of type_simple values (e.g., ["Pres-G", "Sen-G"]),
                    or None for all general election types
        output_dir: directory for output CSVs (default: data/polls/)
        raw_polls_path: override path to raw_polls.csv
        ratings_path: override path to pollster-ratings-combined.csv
        enrich: whether to enrich notes with pollster ratings

    Returns:
        dict of cycle → list of PollObservation
    """
    raw_path = raw_polls_path or RAW_POLLS_PATH
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"538 raw polls not found: {raw_path}")

    # Determine which race types to include
    if race_types is not None:
        allowed_types = set(race_types) & set(RACE_TYPE_MAP.keys())
    else:
        allowed_types = set(RACE_TYPE_MAP.keys())

    cycle_set = set(cycles)
    state_set = set(states) if states else None

    # Load pollster ratings
    ratings = {}
    if enrich:
        ratings = load_pollster_ratings(ratings_path)

    # Read and convert
    result: dict[str, list[PollObservation]] = {c: [] for c in cycles}
    # Also collect CSV rows per cycle for writing
    csv_rows: dict[str, list[dict]] = {c: [] for c in cycles}

    skipped = {"no_party": 0, "no_sample": 0, "wrong_type": 0, "wrong_cycle": 0,
               "wrong_state": 0, "bad_share": 0}

    with raw_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            cycle = row.get("cycle", "").strip()
            if cycle not in cycle_set:
                skipped["wrong_cycle"] += 1
                continue

            type_simple = row.get("type_simple", "").strip()
            if type_simple not in allowed_types:
                skipped["wrong_type"] += 1
                continue

            location = row.get("location", "").strip()
            if state_set and location not in state_set:
                skipped["wrong_state"] += 1
                continue

            # Sample size
            raw_sample = row.get("samplesize", "").strip()
            if not raw_sample:
                log.warning(
                    "Row %d skipped: missing samplesize (poll_id=%s, race=%s)",
                    row_num,
                    row.get("poll_id", "?"),
                    row.get("race", "?"),
                )
                skipped["no_sample"] += 1
                continue
            try:
                n_sample = int(float(raw_sample))
            except ValueError:
                log.warning(
                    "Row %d skipped: invalid samplesize=%r",
                    row_num,
                    raw_sample,
                )
                skipped["no_sample"] += 1
                continue

            if n_sample <= 0:
                skipped["no_sample"] += 1
                continue

            # Two-party share
            dem_share = compute_two_party_dem_share(row)
            if dem_share is None:
                skipped["no_party"] += 1
                continue

            if not (0.0 < dem_share < 1.0):
                skipped["bad_share"] += 1
                continue

            # Format race name
            race_name = format_race_name(cycle, location, type_simple)
            if race_name is None:
                skipped["wrong_type"] += 1
                continue

            # Date (already ISO in 538)
            date = row.get("polldate", "").strip()

            # Pollster
            pollster = row.get("pollster", "").strip()

            # Notes
            methodology = row.get("methodology", "").strip()
            partisan = row.get("partisan", "").strip()
            rating_id_str = row.get("pollster_rating_id", "").strip()

            notes_parts = []
            if methodology:
                notes_parts.append(f"method={methodology}")
            if partisan and partisan != "NA":
                notes_parts.append(f"partisan={partisan}")
            if rating_id_str:
                notes_parts.append(f"rating_id={rating_id_str}")

            # Enrich with pollster quality scores
            if enrich and rating_id_str:
                try:
                    rid = int(rating_id_str)
                    if rid in ratings:
                        r = ratings[rid]
                        if r["numeric_grade"] is not None:
                            notes_parts.append(f"grade={r['numeric_grade']}")
                        if r["pollscore"] is not None:
                            notes_parts.append(f"pollscore={r['pollscore']}")
                        if r["bias"] is not None:
                            notes_parts.append(f"bias={r['bias']}")
                except ValueError:
                    pass

            notes = "; ".join(notes_parts)

            # Build PollObservation
            obs = PollObservation(
                geography=location,
                dem_share=round(dem_share, 6),
                n_sample=n_sample,
                race=race_name,
                date=date,
                pollster=pollster,
                geo_level="state",
            )
            result[cycle].append(obs)

            # Build CSV row
            csv_rows[cycle].append(
                {
                    "race": race_name,
                    "geography": location,
                    "geo_level": "state",
                    "dem_share": f"{dem_share:.6f}",
                    "n_sample": str(n_sample),
                    "date": date,
                    "pollster": pollster,
                    "notes": notes,
                }
            )

    # Write CSVs
    for cycle in cycles:
        rows = csv_rows[cycle]
        if not rows:
            log.warning("No polls found for cycle %s", cycle)
            continue

        out_path = output_dir / f"polls_{cycle}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        log.info("Wrote %d polls to %s", len(rows), out_path)

    # Log summary
    for cycle in cycles:
        obs_list = result[cycle]
        if obs_list:
            # Count by state and race
            by_state: dict[str, int] = {}
            by_race: dict[str, int] = {}
            for obs in obs_list:
                by_state[obs.geography] = by_state.get(obs.geography, 0) + 1
                by_race[obs.race] = by_race.get(obs.race, 0) + 1

            log.info(
                "Cycle %s: %d polls across %d states",
                cycle,
                len(obs_list),
                len(by_state),
            )
            for state, count in sorted(by_state.items()):
                log.info("  %s: %d polls", state, count)

    log.info(
        "Skipped rows: %s",
        ", ".join(f"{k}={v}" for k, v in skipped.items() if v > 0),
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 538 raw_polls.csv to internal poll format"
    )
    parser.add_argument(
        "--cycles",
        nargs="+",
        required=True,
        help="Cycle years to convert (e.g., 2020 2022)",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="State abbreviations to filter (default: FL GA AL)",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Include all states (overrides --states)",
    )
    parser.add_argument(
        "--race-types",
        nargs="+",
        default=None,
        help="Race types (e.g., Pres-G Sen-G Gov-G). Default: all general.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/polls/)",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip pollster rating enrichment",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    states = None if args.all_states else (args.states or ["FL", "GA", "AL"])

    result = convert_538_polls(
        cycles=args.cycles,
        states=states,
        race_types=args.race_types,
        output_dir=args.output_dir,
        enrich=not args.no_enrich,
    )

    # Print summary
    for cycle, polls in sorted(result.items()):
        print(f"\nCycle {cycle}: {len(polls)} polls")
        by_state_race: dict[str, int] = {}
        for p in polls:
            key = f"  {p.geography} / {p.race.split(' ', 2)[-1] if ' ' in p.race else p.race}"
            by_state_race[key] = by_state_race.get(key, 0) + 1
        for key, count in sorted(by_state_race.items()):
            print(f"{key}: {count}")


if __name__ == "__main__":
    main()
