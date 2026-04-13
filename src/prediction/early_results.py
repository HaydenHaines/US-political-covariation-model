"""Early-cycle election results as high-confidence poll-like observations.

Before the main election cycle (November 2026), special elections, off-year
governor races, and judicial elections provide real outcome data.  These
results are stronger signals than polls because they reflect actual voter
turnout — but they require adjustment because:

  1. They are in different contexts (governor vs. Senate, off-year vs. midterm).
  2. They fade in relevance as time passes and conditions change.
  3. They may reflect district-level dynamics that don't generalize statewide.

This module loads those results from a CSV and injects them into the forecast
pipeline as poll-like observations.  The existing time-decay machinery in
``poll_decay.py`` handles relevance fading automatically — an election from
6 months ago decays just like a 6-month-old poll.

Design notes:
  - The CSV shares the exact same schema as polls_2026.csv so it flows
    through prepare_polls() without any special-casing.
  - VA/NJ 2025 governor results feed as Generic Ballot (national signal)
    because neither state has a 2026 race to attach them to.
  - WI SC and GA-14 use adjusted partisan_dem_share values, not raw results.
    See the CSV notes column for the adjustment rationale.
  - n_sample is capped at MAX_EFFECTIVE_N (5000) to prevent any single
    result from dominating the weighted average like 30+ polls.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default path for the early results CSV.
_DEFAULT_EARLY_RESULTS_PATH: Path = (
    PROJECT_ROOT / "data" / "polls" / "early_cycle_results.csv"
)

# Config key in prediction_params.json that controls this feature.
_CONFIG_SECTION: str = "early_results"

# Column name identifying Generic Ballot entries; these need to be extracted
# separately because compute_gb_shift() reads from a file path, not memory.
_GB_RACE_LABEL: str = "2026 Generic Ballot"
_GB_GEO_LEVEL: str = "national"


def _load_params_config(params_path: Path | None = None) -> dict:
    """Load the early_results section from prediction_params.json.

    Returns an empty dict if the section is missing, so callers can use
    .get() with defaults without crashing on missing keys.
    """
    if params_path is None:
        params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
    if not params_path.exists():
        return {}
    try:
        all_params: dict = json.loads(params_path.read_text())
        return all_params.get(_CONFIG_SECTION, {})
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not load early_results config from %s: %s", params_path, exc)
        return {}


def load_early_results(
    path: Path | str | None = None,
    params_path: Path | None = None,
) -> dict[str, list[dict]]:
    """Load early-cycle election results from the CSV.

    Reads the early_cycle_results.csv file and returns rows grouped by race
    in the same format as ``load_polls()`` in predict_2026_types.py.  Generic
    Ballot entries (geo_level="national") are included in the returned dict
    under their race key, but callers that need to feed them to
    ``compute_gb_shift()`` should use ``extract_gb_observations()`` instead.

    Parameters
    ----------
    path:
        Path to the early results CSV.  Defaults to
        ``data/polls/early_cycle_results.csv`` (or the path in
        prediction_params.json ``early_results.path`` if set).
    params_path:
        Path to prediction_params.json.  Only used to read config values
        (``enabled``, ``max_effective_n``).  Defaults to project default.

    Returns
    -------
    dict[str, list[dict]]
        Race label → list of poll-format dicts.  Returns an empty dict when
        early_results is disabled in config or the file does not exist.
    """
    config = _load_params_config(params_path)

    # Feature flag: allow disabling without removing the CSV.
    if not config.get("enabled", True):
        log.info("early_results.enabled=false; skipping early cycle results")
        return {}

    max_n: int = int(config.get("max_effective_n", 5000))

    # Resolve the path: explicit arg > config file > module default.
    if path is None:
        config_path_str = config.get("path")
        if config_path_str:
            resolved_path = Path(config_path_str)
            if not resolved_path.is_absolute():
                resolved_path = PROJECT_ROOT / resolved_path
        else:
            resolved_path = _DEFAULT_EARLY_RESULTS_PATH
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        log.warning(
            "Early results CSV not found at %s; no early cycle data loaded",
            resolved_path,
        )
        return {}

    results: dict[str, list[dict]] = {}
    skipped = 0

    with resolved_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            race = row.get("race", "").strip()
            geography = row.get("geography", "").strip()
            geo_level = row.get("geo_level", "state").strip()
            raw_dem = row.get("dem_share", "").strip()
            raw_n = row.get("n_sample", "").strip()
            date_str = row.get("date", "").strip()
            pollster = row.get("pollster", "").strip()
            notes = row.get("notes", "").strip()

            if not race or not geography:
                skipped += 1
                continue

            try:
                dem_share = float(raw_dem)
                n_sample = int(float(raw_n)) if raw_n else max_n
            except (ValueError, TypeError):
                log.warning("Skipping malformed early result row: %r", row)
                skipped += 1
                continue

            if not (0.0 < dem_share < 1.0) or n_sample <= 0:
                log.warning(
                    "Skipping out-of-range early result: dem_share=%s n=%s race=%s",
                    dem_share,
                    n_sample,
                    race,
                )
                skipped += 1
                continue

            # Cap effective sample size — a single election should not dominate
            # like 30+ polls combined.  max_n defaults to 5000 (≈5 high-quality polls).
            capped_n = min(n_sample, max_n)
            if capped_n < n_sample:
                log.debug(
                    "Capping n_sample for %s (%s): %d → %d",
                    race,
                    geography,
                    n_sample,
                    capped_n,
                )

            poll_dict = {
                "dem_share": dem_share,
                "n_sample": capped_n,
                "state": geography,
                "date": date_str,
                "pollster": pollster,
                "notes": notes,
                "geo_level": geo_level,
            }
            results.setdefault(race, []).append(poll_dict)

    n_results = sum(len(v) for v in results.values())
    log.info(
        "Loaded %d early cycle results across %d races from %s (%d skipped)",
        n_results,
        len(results),
        resolved_path,
        skipped,
    )
    return results


def merge_early_results(
    polls_by_race: dict[str, list[dict]],
    early_by_race: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Merge early results into the main polls dict.

    Early results are just additional observations — they flow through the same
    time-decay, quality-weighting, and Bayesian update machinery as regular polls.
    For races that already have polls, the early results are appended.
    For races with only early results, a new entry is created.

    This is intentionally a shallow merge (no deduplication, no reordering).
    Deduplication happens naturally via time decay: older entries are down-weighted.

    Parameters
    ----------
    polls_by_race:
        Existing polls dict from load_polls().
    early_by_race:
        Early results from load_early_results().

    Returns
    -------
    dict[str, list[dict]]
        Merged dict.  The input dicts are not modified.
    """
    if not early_by_race:
        return polls_by_race

    # Build a new dict to avoid mutating the input.
    merged: dict[str, list[dict]] = {}

    # Copy existing polls.
    for race_id, polls in polls_by_race.items():
        merged[race_id] = list(polls)

    # Append early results.  State-level entries go into polls_by_race as
    # additional observations.  Generic Ballot entries (national, geo_level="national")
    # are excluded here — they're handled separately by compute_gb_shift() via
    # the extra_gb_polls pathway.  Including them here would double-count them
    # if the caller also passes them to compute_gb_shift().
    for race_id, early_polls in early_by_race.items():
        state_polls = [
            p for p in early_polls
            if p.get("geo_level", "state") != _GB_GEO_LEVEL
        ]
        if state_polls:
            merged.setdefault(race_id, []).extend(state_polls)

    return merged


def extract_gb_observations(
    early_by_race: dict[str, list[dict]],
) -> list[tuple[float, int]]:
    """Extract Generic Ballot observations from the early results dict.

    Returns a list of (dem_share, n_sample) tuples suitable for passing
    as ``extra_gb_polls`` to ``compute_gb_shift()``.

    Generic Ballot entries have race=_GB_RACE_LABEL and geo_level="national".
    They cannot go through merge_early_results() (which skips national entries)
    because compute_gb_shift() reads from a file path, not from the in-memory
    polls dict.  This function extracts them so the caller can inject them
    directly into compute_gb_shift().

    Parameters
    ----------
    early_by_race:
        Output of load_early_results().

    Returns
    -------
    list[tuple[float, int]]
        (dem_share, n_sample) pairs, one per Generic Ballot early result row.
    """
    gb_polls = early_by_race.get(_GB_RACE_LABEL, [])
    return [
        (float(p["dem_share"]), int(p["n_sample"]))
        for p in gb_polls
        if p.get("geo_level", "state") == _GB_GEO_LEVEL
    ]
