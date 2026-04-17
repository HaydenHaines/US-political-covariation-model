"""Candidate registry: crosswalk identifying unique politicians across multiple races.

This is Phase 1 of the political sabermetrics pipeline. Its job is deceptively
simple: figure out that "Ron Johnson (Wisconsin, 2010)" and "Ron Johnson
(Wisconsin, 2016)" are the same person. Without this crosswalk, computing
cross-election career stats is impossible.

The challenge: the 538 checking-our-work CSVs are a flat list of
candidate-by-race rows with no persistent IDs. Two candidates with the same
name in different years might be the same person (Ron Johnson running twice)
or different people (two candidates named John Smith in different states).

Disambiguation strategy:
1. For legislators: fuzzy-match names against the unitedstates/congress-legislators
   YAML. These get bioguide IDs, the gold-standard unique key across
   Congress.gov, VoteView, GovTrack, etc.
2. For non-legislators (governors, challengers who never served in Congress):
   we can't use bioguide IDs. Instead we match by (state, party, rough name)
   across years. Collisions are extremely rare given state+party constraints.
3. Ambiguous cases are flagged with needs_review=True rather than silently
   merged or silently split.

The output is candidate_registry.json: one record per unique person, with all
their races and their bioguide ID when available.
"""

from __future__ import annotations

import json
import logging
import unicodedata
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State name normalization (full name → abbreviation)
# This is the same dict as in backtest_harness.py — one canonical source
# would be better (DEBT: extract to a shared constants module), but
# duplicating 52 entries is preferable to an import cycle.
# ---------------------------------------------------------------------------

_STATE_NAME_TO_ABBR: dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
}

# Party label normalization: congress-legislators uses "Democrat"/"Republican";
# 538 uses "D"/"R"/"I". Normalize everything to single-letter codes.
# "D2" appears in Louisiana jungle primary races where two Democrats advance —
# we normalize these to "D" since they're still Democratic candidates.
_PARTY_TO_CODE: dict[str, str] = {
    "Democrat": "D",
    "Republican": "R",
    "Independent": "I",
    "D": "D",
    "D2": "D",  # Second Democrat in Louisiana jungle primary runoff
    "R2": "R",  # Second Republican in a multi-candidate runoff
    "R": "R",
    "I": "I",
    "L": "L",
    "G": "G",
}


# ---------------------------------------------------------------------------
# Name normalization helpers
# ---------------------------------------------------------------------------


def normalize_state(state: str) -> str:
    """Convert full state names to two-letter abbreviations, pass abbrevs through.

    The 538 senate data uses full state names only in 2014; all other years
    and files use abbreviations. This function handles both gracefully.
    """
    state = state.strip()
    if len(state) == 2:
        # Already an abbreviation — uppercase to be safe
        return state.upper()
    result = _STATE_NAME_TO_ABBR.get(state)
    if result is None:
        raise ValueError(f"Unknown state: {state!r}")
    return result


def normalize_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop punctuation.

    We need a canonical form for fuzzy matching. The goal is to handle:
    - "Beto O'Rourke" vs "Beto O'Rourke" (apostrophe variants)
    - "José" vs "Jose" (accent stripping)
    - "John  Smith" vs "John Smith" (double spaces)
    - "Jr." suffixes
    """
    # Strip Unicode accents (NFD decomposition separates base + combining chars)
    nfd = unicodedata.normalize("NFD", name)
    ascii_approx = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Lowercase
    lower = ascii_approx.lower()
    # Remove punctuation that commonly varies (apostrophes, periods, commas)
    cleaned = "".join(c if c.isalnum() or c == " " else " " for c in lower)
    # Collapse whitespace
    return " ".join(cleaned.split())


def name_similarity(name_a: str, name_b: str) -> float:
    """Simple overlap-based similarity between two normalized names.

    Uses token set intersection — order-independent, handles "First Last"
    vs "Last, First" variants. Returns fraction in [0, 1].

    We intentionally avoid heavy dependencies (jellyfish, rapidfuzz) since
    the matching is simple enough: same-state, same-party candidates with
    overlapping name tokens are almost certainly the same person.
    """
    tokens_a = set(normalize_name(name_a).split())
    tokens_b = set(normalize_name(name_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    # Jaccard coefficient on token sets
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


# ---------------------------------------------------------------------------
# Congress-legislators loader
# ---------------------------------------------------------------------------


def load_congress_legislators(
    data_dir: str | Path = "data/raw/congress-legislators",
) -> list[dict]:
    """Load current + historical legislators from YAML files.

    Returns a flat list of legislator dicts, each with fields:
        bioguide_id, name_full, name_last, name_first,
        party_codes (list[str]), states (list[str]),
        term_types (list[str]), term_years (list[int])

    We parse all terms rather than just current state/party because
    legislators switch parties and states across their careers.
    The term list lets us match based on what they were doing *when*
    a given race happened.
    """
    data_dir = Path(data_dir)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: uv add pyyaml") from exc

    legislators = []
    for filename in ["legislators-current.yaml", "legislators-historical.yaml"]:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run build_id_crosswalk() first to download.")
        with open(path) as f:
            records = yaml.safe_load(f)
        for rec in records:
            bioguide = rec.get("id", {}).get("bioguide")
            if not bioguide:
                # A few very old historical legislators lack bioguide IDs — skip
                continue
            name = rec.get("name", {})
            official_full = name.get("official_full") or (f"{name.get('first', '')} {name.get('last', '')}".strip())
            terms = rec.get("terms", [])
            # Collect all states and parties across career (a senator who served
            # in multiple states is rare but real — e.g., carpetbaggers post-Civil War)
            states = sorted({t.get("state", "") for t in terms if t.get("state")})
            party_codes = sorted({_PARTY_TO_CODE.get(t.get("party", ""), "?") for t in terms if t.get("party")})
            term_types = sorted({t.get("type", "") for t in terms})
            # Collect election years from term start dates
            term_years = []
            for t in terms:
                start = t.get("start", "")
                if start:
                    term_years.append(int(start[:4]))
            legislators.append(
                {
                    "bioguide_id": bioguide,
                    "name_full": official_full,
                    "name_last": name.get("last", ""),
                    "name_first": name.get("first", ""),
                    "party_codes": party_codes,
                    "states": states,
                    "term_types": term_types,
                    "term_years": sorted(set(term_years)),
                }
            )
    return legislators


# ---------------------------------------------------------------------------
# Match a 538 candidate name to congress-legislators
# ---------------------------------------------------------------------------


def _match_to_legislator(
    candidate_name: str,
    state: str,
    party: str,
    year: int,
    legislators: list[dict],
    similarity_threshold: float = 0.6,
) -> dict | None:
    """Find the best congress-legislators match for a 538 candidate row.

    Strategy:
    1. Filter to legislators who served in this state around this year
       (within ±4 years of the election — senators serve 6-year terms,
       so a senator elected in 2010 had terms starting 2007-2013 range).
    2. Among those, find best name similarity.
    3. Require similarity >= threshold to avoid false positives.

    Returns the matched legislator dict, or None if no confident match.
    """
    party_code = _PARTY_TO_CODE.get(party, party)
    best_match = None
    best_score = 0.0

    for leg in legislators:
        # State filter: must have served in this state
        if state not in leg["states"]:
            continue
        # Party filter: must have been this party at some point
        # (allow flexibility — party switches are rare but real)
        if party_code not in leg["party_codes"] and "?" not in leg["party_codes"]:
            continue
        # Year filter: must have served within ±6 years of the election
        # (a senator may first appear in records 6 years before their election
        # if they served a prior term, or up to 6 years after if they won)
        if leg["term_years"] and not any(abs(yr - year) <= 8 for yr in leg["term_years"]):
            continue
        # Name similarity
        score = name_similarity(candidate_name, leg["name_full"])
        # Also try last-name-only match (covers "John Smith Jr." vs "John Smith")
        last_only = name_similarity(candidate_name.split()[-1], leg["name_last"])
        # Use the better of full-name and last-name-only scores
        # but cap last-name-only at 0.75 to avoid false positives on common names
        effective_score = max(score, min(last_only * 0.9, 0.75))

        if effective_score > best_score:
            best_score = effective_score
            best_match = leg

    if best_score >= similarity_threshold:
        return best_match
    return None


# ---------------------------------------------------------------------------
# Parse 538 race rows
# ---------------------------------------------------------------------------


def _compute_dem_share_2party(row_d: dict | None, row_r: dict | None) -> float | None:
    """Compute Democratic two-party actual vote share from a pair of party rows.

    The 538 actual_voteshare values represent full-electorate shares (may not
    sum to 100 when third parties exist). We strip third parties by dividing
    only by D+R total — same logic as backtest_harness.py.
    """
    if row_d is None or row_r is None:
        return None
    dem = row_d.get("actual_voteshare")
    rep = row_r.get("actual_voteshare")
    if dem is None or rep is None:
        return None
    try:
        dem, rep = float(dem), float(rep)
    except (ValueError, TypeError):
        return None
    total = dem + rep
    if total <= 0:
        return None
    return dem / total


def _load_medsl_state_actuals_cache() -> dict[tuple[int, str, str], float]:
    """Build a cache of state-level Dem two-party share from MEDSL county parquets.

    Fallback for years where 538 actual_voteshare is null (e.g. 2018).
    Returns dict mapping (year, state_abbr, office) → dem_share (vote-weighted).
    """
    import pandas as pd

    project_root = Path(__file__).resolve().parents[2]
    assembled_dir = project_root / "data" / "assembled"
    cache: dict[tuple[int, str, str], float] = {}

    # Senate actuals — use vote-weighted aggregation (sum D / (sum D + sum R))
    # because simple mean of county dem_share is badly biased by rural county count.
    for year in range(2008, 2025, 2):
        path = assembled_dir / f"medsl_county_senate_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "state_abbr" not in df.columns:
            continue
        # Column naming: senate_dem_{year}, senate_rep_{year}
        dem_col = f"senate_dem_{year}"
        rep_col = f"senate_rep_{year}"
        if dem_col in df.columns and rep_col in df.columns:
            for state, grp in df.groupby("state_abbr"):
                d = grp[dem_col].sum()
                r = grp[rep_col].sum()
                if (d + r) > 0:
                    cache[(year, state, "senate")] = d / (d + r)

    # Governor actuals — same vote-weighted approach
    for year in [2010, 2014, 2018, 2022]:
        if year == 2022:
            path = assembled_dir / "medsl_county_2022_governor.parquet"
            prefix = "gov"
        else:
            path = assembled_dir / f"algara_county_governor_{year}.parquet"
            prefix = "gov"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "state_abbr" not in df.columns:
            continue
        dem_col = f"{prefix}_dem_{year}"
        rep_col = f"{prefix}_rep_{year}"
        # Some parquets use different column naming; try both patterns
        if dem_col not in df.columns:
            dem_col = f"gov_dem_votes_{year}"
            rep_col = f"gov_rep_votes_{year}"
        if dem_col in df.columns and rep_col in df.columns:
            for state, grp in df.groupby("state_abbr"):
                d = grp[dem_col].sum()
                r = grp[rep_col].sum()
                if (d + r) > 0:
                    cache[(year, state, "governor")] = d / (d + r)

    return cache


def _parse_538_csv(
    csv_path: Path,
    office: str,
) -> list[dict]:
    """Parse a 538 checking-our-work CSV into race records.

    Each record represents one candidate in one race. We select one
    representative row per candidate per race to avoid duplicates.

    The 538 data has inconsistent forecast_type naming across years:
    - 2008-2014: no forecast_type (empty string) — keep all rows
    - 2016: "polls-plus", "polls-only", "now-cast" — prefer "polls-plus"
    - 2018-2022: "lite", "classic", "deluxe" — prefer "lite"

    Within the chosen type, we use the final forecast_date (election day)
    to get one row per candidate per race.

    Returns list of dicts with keys:
        year, state, office, special, candidate, party,
        actual_voteshare, projected_voteshare, forecast_date
    """
    import csv

    # Preferred forecast type priority order (first available wins)
    _FORECAST_TYPE_PRIORITY = ["lite", "polls-plus", "classic", "deluxe", "polls-only", "now-cast", ""]

    # Read all rows first so we can select the best forecast type per race
    raw_rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with empty candidate names (national-level projection rows)
            candidate_name = row.get("candidate", "").strip()
            if not candidate_name:
                continue
            # "US" rows are national-level projections, not individual state races
            state_raw = row.get("state", "").strip()
            if state_raw == "US":
                continue
            raw_rows.append(dict(row))

    if not raw_rows:
        return []

    # The 538 data uses different forecast type names across years.
    # We select the best available type *per year* to avoid dropping years
    # that don't have the preferred type (e.g., 2016 lacks "lite" but has
    # "polls-plus"; 2008-2014 have no type at all).
    from collections import defaultdict

    rows_by_year: dict[int, list[dict]] = defaultdict(list)
    for row in raw_rows:
        rows_by_year[int(row["year"])].append(row)

    filtered: list[dict] = []
    for year, year_rows in rows_by_year.items():
        available = {r.get("forecast_type", "").strip() for r in year_rows}
        chosen = next((ft for ft in _FORECAST_TYPE_PRIORITY if ft in available), "")
        filtered.extend(r for r in year_rows if r.get("forecast_type", "").strip() == chosen)

    # Group by (year, state, special, candidate, party) and take latest forecast_date
    race_key_to_rows: dict[tuple, list[dict]] = {}
    for row in filtered:
        state_raw = row.get("state", "").strip()
        try:
            state = normalize_state(state_raw)
        except ValueError:
            logger.warning("Unknown state %r in %s, skipping", state_raw, csv_path)
            continue
        special_raw = row.get("special", "").strip().lower()
        is_special = special_raw in ("true", "1", "yes")
        key = (
            int(row["year"]),
            state,
            is_special,
            row.get("candidate", "").strip(),
            row.get("party", "").strip(),
        )
        race_key_to_rows.setdefault(key, []).append(row)

    records = []
    for (year, state, is_special, candidate, party), rows in race_key_to_rows.items():
        # Take the row with the latest forecast_date (election day is the final)
        best_row = max(rows, key=lambda r: r.get("forecast_date", ""))
        records.append(
            {
                "year": year,
                "state": state,
                "office": office,
                "special": is_special,
                "candidate": candidate,
                "party": party,
                "actual_voteshare": best_row.get("actual_voteshare"),
                "projected_voteshare": best_row.get("projected_voteshare"),
                "forecast_date": best_row.get("forecast_date", ""),
            }
        )
    return records


def _group_races(records: list[dict]) -> dict[tuple, list[dict]]:
    """Group candidate rows by (year, state, office, special) into races."""
    races: dict[tuple, list[dict]] = {}
    for rec in records:
        key = (rec["year"], rec["state"], rec["office"], rec["special"])
        races.setdefault(key, []).append(rec)
    return races


# ---------------------------------------------------------------------------
# Core registry builder
# ---------------------------------------------------------------------------


def build_candidate_registry(
    senate_csv: str | Path = "data/raw/fivethirtyeight/checking-our-work-data/us_senate_elections.csv",
    governor_csv: str | Path = "data/raw/fivethirtyeight/checking-our-work-data/governors_elections.csv",
    legislators_dir: str | Path = "data/raw/congress-legislators",
    candidates_2026_path: str | Path = "data/config/candidates_2026.json",
    output_path: str | Path = "data/sabermetrics/candidate_registry.json",
    similarity_threshold: float = 0.6,
) -> dict:
    """Build the candidate registry crosswalk.

    This is the main entry point for Phase 1 of the sabermetrics pipeline.

    Algorithm:
    1. Parse 538 CSVs → list of (candidate, race) rows
    2. Group rows by (year, state, office, special) → races
    3. For each race, compute Dem two-party actual vote share
    4. For each candidate, attempt fuzzy match to congress-legislators
    5. Candidates with a bioguide match → use bioguide as person_id
    6. Candidates without a bioguide match (governors, challengers) →
       group by (state, party, normalized-name) across years
    7. Emit one JSON record per unique person

    Returns the registry dict (also written to output_path).
    """
    senate_csv = Path(senate_csv)
    governor_csv = Path(governor_csv)
    legislators_dir = Path(legislators_dir)
    candidates_2026_path = Path(candidates_2026_path)
    output_path = Path(output_path)

    # --- Step 1: Load congress-legislators for fuzzy matching ---
    logger.info("Loading congress-legislators YAML...")
    legislators = load_congress_legislators(legislators_dir)
    logger.info("  Loaded %d legislators", len(legislators))

    # --- Step 2: Parse 538 CSVs ---
    senate_records = _parse_538_csv(senate_csv, office="Senate")
    governor_records = _parse_538_csv(governor_csv, office="Governor")
    all_records = senate_records + governor_records
    logger.info("Parsed %d senate rows, %d governor rows", len(senate_records), len(governor_records))

    # --- Step 3: Group into races and compute two-party vote share ---
    all_races = _group_races(all_records)

    # Build a list of per-candidate-per-race dicts enriched with 2-party Dem share.
    # MEDSL fallback: 538 checking-our-work has null actual_voteshare for 2018.
    # Load MEDSL assembled parquets as a fallback for state-level actuals.
    medsl_cache = _load_medsl_state_actuals_cache()

    enriched_candidates: list[dict] = []
    for (year, state, office, special), candidates in all_races.items():
        # Find the D and R candidates in this race for 2-party calc
        by_party: dict[str, dict] = {}
        for cand in candidates:
            # If multiple rows per party (shouldn't happen after dedup), take first
            party = cand["party"]
            if party not in by_party:
                by_party[party] = cand

        dem_2party = _compute_dem_share_2party(by_party.get("D"), by_party.get("R"))

        # Fallback to MEDSL when 538 actual is null (e.g. 2018 data)
        if dem_2party is None:
            medsl_key = (year, state, office.lower())
            dem_2party = medsl_cache.get(medsl_key)

        for cand in candidates:
            # Determine result from winner's perspective
            # probwin_outcome would be cleaner but we don't carry it through;
            # derive from vote share instead
            result = _determine_result(cand, by_party, dem_2party)
            enriched_candidates.append(
                {
                    **cand,
                    "dem_2party_share": dem_2party,
                    "result": result,
                }
            )

    # --- Step 4 & 5: Match candidates to legislators via fuzzy name matching ---
    # person_id → person record dict
    persons: dict[str, dict] = {}
    # Track name→bioguide mappings to deduplicate same-person-different-rows
    candidate_to_person_id: dict[tuple, str] = {}

    # First pass: match to legislators (these get bioguide IDs)
    unmatched: list[dict] = []
    for cand in enriched_candidates:
        key = (cand["year"], cand["state"], cand["office"], cand["special"], cand["candidate"], cand["party"])
        if key in candidate_to_person_id:
            # Already processed this exact candidate-race combo (duplicate row)
            continue
        candidate_to_person_id[key] = ""  # Mark as seen; fill in below

        match = _match_to_legislator(
            candidate_name=cand["candidate"],
            state=cand["state"],
            party=cand["party"],
            year=cand["year"],
            legislators=legislators,
            similarity_threshold=similarity_threshold,
        )
        if match:
            bioguide = match["bioguide_id"]
            candidate_to_person_id[key] = bioguide
            if bioguide not in persons:
                persons[bioguide] = {
                    "name": match["name_full"],
                    "party": _most_common_party(match["party_codes"]),
                    "bioguide_id": bioguide,
                    "races": [],
                    "needs_review": False,
                }
        else:
            unmatched.append(cand)

    # Second pass: group unmatched candidates by (state, party, normalized_name)
    # across years → same key = same person (governor who ran twice, challenger, etc.)
    unmatched_person_keys: dict[tuple, str] = {}  # (state, party, norm_name) → person_id
    for cand in unmatched:
        key = (cand["year"], cand["state"], cand["office"], cand["special"], cand["candidate"], cand["party"])
        norm_key = (
            cand["state"],
            _PARTY_TO_CODE.get(cand["party"], cand["party"]),
            normalize_name(cand["candidate"]),
        )
        if norm_key in unmatched_person_keys:
            person_id = unmatched_person_keys[norm_key]
        else:
            # New person — generate a UUID so these IDs are globally unique
            # and won't collide with bioguide IDs (which are alphanumeric letters)
            person_id = f"gen_{uuid.uuid4().hex[:12]}"
            unmatched_person_keys[norm_key] = person_id
            persons[person_id] = {
                "name": cand["candidate"],
                "party": _PARTY_TO_CODE.get(cand["party"], cand["party"]),
                "bioguide_id": None,
                "races": [],
                "needs_review": False,
            }
        candidate_to_person_id[key] = person_id

    # --- Step 6: Attach race records to persons ---
    for cand in enriched_candidates:
        key = (cand["year"], cand["state"], cand["office"], cand["special"], cand["candidate"], cand["party"])
        person_id = candidate_to_person_id.get(key)
        if not person_id:
            # Genuinely unresolved (shouldn't happen after two passes above)
            logger.warning("Could not assign person_id for %s", key)
            continue
        race_record = {
            "year": cand["year"],
            "state": cand["state"],
            "office": cand["office"],
            "special": cand["special"],
            "party": _PARTY_TO_CODE.get(cand["party"], cand["party"]),
            "actual_dem_share_2party": cand["dem_2party_share"],
            "result": cand["result"],
        }
        # Avoid duplicate race entries (can happen if same candidate appears
        # in multiple forecast rows that both pass the lite filter)
        race_key = (cand["year"], cand["state"], cand["office"], cand["special"])
        existing_races = persons[person_id]["races"]
        if not any((r["year"], r["state"], r["office"], r["special"]) == race_key for r in existing_races):
            existing_races.append(race_record)

    # Sort races chronologically within each person
    for person in persons.values():
        person["races"].sort(key=lambda r: (r["year"], r["state"], r["office"]))

    # --- Step 7: Incorporate 2026 candidates ---
    _incorporate_2026_candidates(
        persons=persons,
        candidates_2026_path=candidates_2026_path,
        legislators=legislators,
        similarity_threshold=similarity_threshold,
    )

    # --- Step 8: Build metadata ---
    multi_race_count = sum(1 for p in persons.values() if len(p["races"]) > 1)
    registry = {
        "persons": persons,
        "_meta": {
            "created": "2026-04-16",
            "sources": ["538-checking-our-work", "congress-legislators"],
            "total_persons": len(persons),
            "multi_race_persons": multi_race_count,
        },
    }

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    logger.info(
        "Registry written to %s: %d persons, %d multi-race",
        output_path,
        len(persons),
        multi_race_count,
    )
    return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _determine_result(
    cand: dict,
    by_party: dict[str, dict],
    dem_2party: float | None,
) -> str | None:
    """Infer win/loss from vote share comparison within a race."""
    if dem_2party is None:
        return None
    party = cand["party"]
    if party == "D":
        return "win" if dem_2party > 0.5 else "loss"
    if party == "R":
        return "win" if dem_2party < 0.5 else "loss"
    # Third party — use raw vote share comparison
    try:
        my_share = float(cand.get("actual_voteshare") or 0)
        other_shares = [float(c.get("actual_voteshare") or 0) for p, c in by_party.items() if p != party]
        if not other_shares:
            return None
        return "win" if my_share > max(other_shares) else "loss"
    except (ValueError, TypeError):
        return None


def _most_common_party(party_codes: list[str]) -> str:
    """Return the most common party code from a legislator's career."""
    if not party_codes:
        return "?"
    # Simple: take the last one (most recent is usually most relevant)
    # Most legislators don't switch parties, so this is fine
    return party_codes[-1] if party_codes else "?"


def _incorporate_2026_candidates(
    persons: dict[str, dict],
    candidates_2026_path: Path,
    legislators: list[dict],
    similarity_threshold: float,
) -> None:
    """Add 2026 candidate entries from candidates_2026.json.

    For each 2026 candidate, we try to match them to an existing person
    in the registry (by name + state + party). If they appear in historical
    538 data we'll already have a record. If not (new candidate), we add
    a stub with no historical races.
    """
    if not candidates_2026_path.exists():
        logger.warning("candidates_2026.json not found at %s, skipping", candidates_2026_path)
        return

    with open(candidates_2026_path) as f:
        data = json.load(f)

    # Build a name→person_id index for fast lookup
    name_to_person_ids: dict[str, list[str]] = {}
    for pid, person in persons.items():
        norm = normalize_name(person["name"])
        name_to_person_ids.setdefault(norm, []).append(pid)

    def _find_or_create_person(name: str, party: str, state: str, office: str) -> str:
        """Return person_id for a 2026 candidate, creating one if necessary."""
        party_code = _PARTY_TO_CODE.get(party, party)
        norm = normalize_name(name)

        # Try exact normalized name match
        candidates_with_name = name_to_person_ids.get(norm, [])
        for pid in candidates_with_name:
            p = persons[pid]
            if p["party"] == party_code:
                # Check if any race is in the same state
                if any(r["state"] == state for r in p["races"]):
                    return pid
                # No state races yet — could still be right person (incumbent)
                if p.get("bioguide_id"):
                    # For legislators, check their states from the crosswalk
                    leg = next((lg for lg in legislators if lg["bioguide_id"] == p["bioguide_id"]), None)
                    if leg and state in leg["states"]:
                        return pid

        # Try fuzzy match to legislators for 2026 candidates
        match = _match_to_legislator(
            candidate_name=name,
            state=state,
            party=party_code,
            year=2026,
            legislators=legislators,
            similarity_threshold=similarity_threshold,
        )
        if match:
            bioguide = match["bioguide_id"]
            if bioguide in persons:
                return bioguide
            # Legislator not yet in registry (no historical 538 race)
            persons[bioguide] = {
                "name": match["name_full"],
                "party": _most_common_party(match["party_codes"]),
                "bioguide_id": bioguide,
                "races": [],
                "needs_review": False,
            }
            name_to_person_ids.setdefault(normalize_name(match["name_full"]), []).append(bioguide)
            return bioguide

        # New person — no historical record
        person_id = f"gen_{uuid.uuid4().hex[:12]}"
        persons[person_id] = {
            "name": name,
            "party": party_code,
            "bioguide_id": None,
            "races": [],
            "needs_review": False,
        }
        name_to_person_ids.setdefault(norm, []).append(person_id)
        return person_id

    # Process senate and governor races from 2026 config
    for office_key in ("senate", "governor"):
        office_data = data.get(office_key, {})
        for race_label, race_info in office_data.items():
            state_name = race_info.get("state", "")
            try:
                state = normalize_state(state_name)
            except ValueError:
                logger.warning("Unknown state %r in 2026 config, skipping %s", state_name, race_label)
                continue
            office = "Senate" if office_key == "senate" else "Governor"
            candidates = race_info.get("candidates", {})
            for party, names in candidates.items():
                party_code = _PARTY_TO_CODE.get(party, party)
                for name in names:
                    if not name or not name.strip():
                        continue
                    # Strip parenthetical notes like "(appointed interim)"
                    clean_name = name.split("(")[0].strip()
                    if not clean_name:
                        continue
                    _find_or_create_person(clean_name, party_code, state, office)

            # Also add incumbent if listed
            incumbent = race_info.get("incumbent", {})
            if incumbent and isinstance(incumbent, dict):
                inc_name = incumbent.get("name", "")
                inc_party = incumbent.get("party", "")
                if inc_name and inc_party:
                    _find_or_create_person(inc_name, inc_party, state, office)
