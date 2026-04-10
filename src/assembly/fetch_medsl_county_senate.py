"""Fetch county-level U.S. Senate returns nationally (all 50 states + DC).

Data sources by year:
  2024: MEDSL 2024-elections-official GitHub repo (per-state precinct ZIPs)
        https://github.com/MEDSL/2024-elections-official/tree/main/individual_states
        Same source used for presidential 2024 data (fetch_2024_president.py).
        Each ZIP contains multi-office precinct-level CSVs; we filter to Senate
        and aggregate to county.
  2022: MEDSL county-level results
        doi:10.7910/DVN/YB60EJ  (senate_2022.tab, has county_fips)
  2016/2018/2020: VEST Senate Precinct-Level Returns, aggregated to county
        2016: doi:10.7910/DVN/NLTQAD  (2016-precinct-senate.tab)
        2018: doi:10.7910/DVN/DGNAFS  (SENATE_precinct_general.tab)
        2020: doi:10.7910/DVN/ER9XTV  (2020-SENATE-precinct-general.csv)
  2002-2014: Algara & Amlani County Electoral Dataset
        doi:10.7910/DVN/DGUMFI
        file: dataverse_shareable_us_senate_county_returns_1908_2020.Rdata
        Covers general-election Senate returns at county level from 1908-2020.
        Not every state has a Senate race every cycle (each seat is contested
        every 6 years), so some year-state combos produce 0 counties — correct.

NOTE: Multiple Senate races may exist in the same state-year (e.g. GA 2020
special + regular). When that happens candidatevotes are summed across races
per county. This produces a blended dem_share weighted by turnout.

Output (one parquet per election year, data/assembled/):
  medsl_county_senate_{year}.parquet
  Columns: county_fips, state_abbr, senate_dem_{year}, senate_rep_{year},
           senate_total_{year}, senate_dem_share_{year}

Caches:
  data/raw/medsl/2024/{abbr}24.zip  (per-state MEDSL 2024 precinct ZIPs)
  data/raw/medsl/senate_2022.tab
  data/raw/medsl/senate_precinct_2016.tab
  data/raw/medsl/senate_precinct_2018.tab
  data/raw/medsl/senate_precinct_2020.csv
  data/raw/algara_amlani/dataverse_shareable_us_senate_county_returns_1908_2020.Rdata
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medsl"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

DATAVERSE_API = "https://dataverse.harvard.edu/api"

STATES: dict[str, str] = _cfg.STATE_ABBR   # fips_prefix → abbr  e.g. "12" → "FL"
STATE_PO_SET: set[str] = set(STATES.values())  # all 50 states + DC

# Public constants consumed by tests and downstream modules
SENATE_YEARS: list[int] = _cfg.SENATE_YEARS

# ── Source definitions ─────────────────────────────────────────────────────────

# MEDSL 2022 county-level Senate (has county_fips column directly)
MEDSL_2022_FILE_ID = 7412054   # senate_2022.tab inside doi:10.7910/DVN/YB60EJ
MEDSL_2022_CACHE = RAW_DIR / "senate_2022.tab"

# VEST precinct-level Senate (have county_fips; need county aggregation)
# Filenames and Dataverse file IDs for the main tabular data files:
VEST_SOURCES: dict[int, tuple[int, Path, str]] = {
    # year → (file_id, cache_path, separator)
    2016: (3345325, RAW_DIR / "senate_precinct_2016.tab", "\t"),
    2018: (6692658, RAW_DIR / "senate_precinct_2018.tab", "\t"),
    2020: (6100391, RAW_DIR / "senate_precinct_2020.csv", ","),
}

# Algara & Amlani county-level Senate returns 1908-2020 (doi:10.7910/DVN/DGUMFI)
# Covers 2002-2014 (and beyond, but VEST/MEDSL preferred for 2016+).
ALGARA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "algara_amlani"
ALGARA_SENATE_FILE_ID = 5028534
ALGARA_SENATE_FILENAME = "dataverse_shareable_us_senate_county_returns_1908_2020.Rdata"
ALGARA_SENATE_CACHE = ALGARA_RAW_DIR / ALGARA_SENATE_FILENAME

# MEDSL 2024-elections-official GitHub repo (per-state precinct ZIPs).
# Same ZIP files used by fetch_2024_president.py — they contain all offices
# including Senate.  We reuse the same cache directory to avoid double-downloading.
MEDSL_2024_BASE_URL = (
    "https://github.com/MEDSL/2024-elections-official/raw/main/individual_states"
)
MEDSL_2024_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medsl" / "2024"
# Build per-state ZIP info from config: abbr → (zip_filename, fips_prefix)
MEDSL_2024_STATES: dict[str, tuple[str, str]] = {
    abbr: (f"{abbr.lower()}24.zip", fips)
    for abbr, fips in _cfg.STATES.items()
}


# ── Download helpers ───────────────────────────────────────────────────────────

def _download_file(file_id: int, cache_path: Path) -> Path:
    """Download a single Dataverse file by its numeric file ID."""
    if cache_path.exists():
        log.info("Using cached file: %s", cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    log.info("Downloading file_id=%d → %s ...", file_id, cache_path.name)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(cache_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=131072):
                fh.write(chunk)
    log.info("Saved → %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)
    return cache_path


# ── Shared aggregation logic ───────────────────────────────────────────────────

def _normalise_fips(series: pd.Series) -> pd.Series:
    """Convert county_fips (float or int or str) to zero-padded 5-char string.

    Raises ValueError if any values cannot be parsed as numeric (NaN after coerce),
    since "00000" is not a valid FIPS code and would silently corrupt join keys.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    bad = numeric.isna()
    if bad.any():
        raise ValueError(
            f"_normalise_fips: {bad.sum()} unparseable FIPS value(s): "
            f"{series[bad].unique().tolist()[:5]}"
        )
    return numeric.astype(int).astype(str).str.zfill(5)


def _sum_party_votes(df: pd.DataFrame, year: int) -> tuple["pd.Series", "pd.Series", "pd.Series"]:
    """Sum Democratic, Republican, and total candidatevotes by county.

    Returns (dem_series, rep_series, total_series) indexed by county_fips,
    named with the year-suffixed column names used downstream.
    """
    dem = (
        df[df["party_simplified"] == "DEMOCRAT"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"senate_dem_{year}")
    )
    rep = (
        df[df["party_simplified"] == "REPUBLICAN"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"senate_rep_{year}")
    )
    # Total = ALL candidate votes (D + R + other/write-in).
    # This is the correct denominator for dem_share on a total-vote basis.
    total = (
        df.groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"senate_total_{year}")
    )
    return dem, rep, total


def _drop_uncontested(result: "pd.DataFrame", year: int) -> "pd.DataFrame":
    """Drop counties where either party received zero votes (uncontested races).

    Uncontested counties distort log-odds shifts, so we remove them entirely
    rather than imputing. Logs a warning when any are dropped.
    """
    dem_col, rep_col = f"senate_dem_{year}", f"senate_rep_{year}"
    contested = (result[dem_col] > 0) & (result[rep_col] > 0)
    n_dropped = (~contested).sum()
    if n_dropped:
        log.warning("Year %d: dropping %d uncontested/missing counties", year, n_dropped)
    return result[contested].copy()


def _aggregate_to_county(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate precinct/county rows to one row per county for a given year.

    Expects columns: county_fips, party_simplified, candidatevotes.
    Multiple races per county (e.g. GA specials) are summed.
    Uncontested counties (dem or rep = 0) are dropped.

    Returns county_fips, state_abbr, senate_dem_{year}, senate_rep_{year},
    senate_total_{year}, senate_dem_share_{year}.
    """
    df = df.copy()
    df["county_fips"] = _normalise_fips(df["county_fips"])

    dem, rep, total = _sum_party_votes(df, year)
    result = pd.concat([dem, rep, total], axis=1).reset_index()
    result[f"senate_dem_share_{year}"] = (
        result[f"senate_dem_{year}"] / result[f"senate_total_{year}"]
    )
    result = _drop_uncontested(result, year)
    result["state_abbr"] = result["county_fips"].str[:2].map(STATES)
    dem_col, rep_col = f"senate_dem_{year}", f"senate_rep_{year}"
    return result[["county_fips", "state_abbr",
                   dem_col, rep_col,
                   f"senate_total_{year}", f"senate_dem_share_{year}"]]


# ── Public-API wrappers (used by tests and build_county_shifts_multiyear) ──────

def filter_senate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only D/R U.S. Senate rows for all configured states (50 + DC).

    Expects columns: office, party_simplified, state_po (or state_postal for
    2016 VEST).  This is the public-API entry-point used by test_senate.py and
    build_county_shifts_multiyear.
    """
    state_col = "state_po" if "state_po" in df.columns else "state_postal"
    mask = (
        (df["office"].str.upper().str.contains("SENATE", na=False))
        & (~df["office"].str.upper().str.contains("STATE", na=False))
        & (df["party_simplified"].isin({"DEMOCRAT", "REPUBLICAN"}))
        & (df[state_col].isin(STATE_PO_SET))
    )
    return df[mask].copy()


def _compute_total_votes(yr: "pd.DataFrame", year: int) -> "pd.Series":
    """Compute total votes per county, preferring totalvotes when available.

    MEDSL convention: when a totalvotes column is present it already counts
    all candidates (including third-party). We sum the D rows to handle
    multi-race county-years (e.g. GA 2020 regular + special). When absent
    we fall back to summing all candidatevotes directly.
    """
    if "totalvotes" in yr.columns:
        return (
            yr[yr["party_simplified"] == "DEMOCRAT"]
            .groupby("county_fips")["totalvotes"]
            .sum()
            .rename(f"senate_total_{year}")
        )
    return (
        yr.groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"senate_total_{year}")
    )


def aggregate_county_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate filtered MEDSL/VEST rows to one row per county for a given year.

    Public wrapper for MEDSL/VEST-style long-form input (party_simplified +
    candidatevotes). When totalvotes is present it is used as the denominator
    so third-party votes are included in the total; otherwise candidatevotes
    are summed. Multiple races per county-year (e.g. GA 2020) are summed.
    """
    yr = df[df["year"] == year].copy() if "year" in df.columns else df.copy()

    if yr.empty:
        return pd.DataFrame(columns=[
            "county_fips", "state_abbr",
            f"senate_dem_{year}", f"senate_rep_{year}",
            f"senate_total_{year}", f"senate_dem_share_{year}",
        ])

    yr["county_fips"] = _normalise_fips(yr["county_fips"])
    dem, rep, _ = _sum_party_votes(yr, year)
    total = _compute_total_votes(yr, year)

    result = pd.concat([dem, rep, total], axis=1).reset_index()
    result[f"senate_dem_share_{year}"] = (
        result[f"senate_dem_{year}"] / result[f"senate_total_{year}"]
    )
    result = _drop_uncontested(result, year)
    result["state_abbr"] = result["county_fips"].str[:2].map(STATES)
    dem_col, rep_col = f"senate_dem_{year}", f"senate_rep_{year}"
    return result[["county_fips", "state_abbr",
                   dem_col, rep_col,
                   f"senate_total_{year}", f"senate_dem_share_{year}"]]


# ── 2022 MEDSL county-level ────────────────────────────────────────────────────

def fetch_2022() -> pd.DataFrame:
    """Download and process MEDSL 2022 county-level Senate data."""
    path = _download_file(MEDSL_2022_FILE_ID, MEDSL_2022_CACHE)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log.info("2022 MEDSL raw rows: %d", len(df))

    # Filter to configured states, US Senate, D/R, general election, TOTAL mode
    mask = (
        (df["state_po"].isin(STATE_PO_SET))
        & (df["office"].str.upper().str.contains("SENATE"))
        & (~df["office"].str.upper().str.contains("STATE"))
        & (df["party_simplified"].isin({"DEMOCRAT", "REPUBLICAN"}))
        & (df["stage"].str.upper().str.strip().isin({"GEN", "GENERAL"}))
    )
    df = df[mask].copy()
    log.info("2022 after filter: %d rows", len(df))

    # Resolve voting modes: prefer TOTAL rows when present; otherwise sum modes
    mode_col = "mode"
    if mode_col in df.columns:
        has_total = (
            df[df[mode_col].str.upper().str.strip() == "TOTAL"]
            .groupby("county_fips").size().reset_index()[["county_fips"]]
            .assign(_has_total=True)
        )
        df = df.merge(has_total, on="county_fips", how="left")
        df["_has_total"] = df["_has_total"].fillna(False)
        total_rows = df[df["_has_total"] & (df[mode_col].str.upper().str.strip() == "TOTAL")]
        other_rows = df[~df["_has_total"]].groupby(
            ["county_fips", "party_simplified"], dropna=False
        ).agg(candidatevotes=("candidatevotes", "sum"),
              totalvotes=("totalvotes", "first")).reset_index()
        df = pd.concat([total_rows, other_rows], ignore_index=True)

    # Add year column expected by _aggregate_to_county
    df["year"] = 2022
    return _aggregate_to_county(df, 2022)


# ── 2024 MEDSL GitHub per-state precinct ZIPs ────────────────────────────────


def _download_2024_zip(state_abbr: str, filename: str) -> Path | None:
    """Download a per-state MEDSL 2024 ZIP from GitHub. Returns path or None on 404.

    Reuses the same cache directory as fetch_2024_president.py, so ZIPs that
    were already downloaded for presidential data won't be re-downloaded.
    """
    dest = MEDSL_2024_RAW_DIR / filename
    if dest.exists():
        log.info("  Cached: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MEDSL_2024_BASE_URL}/{filename}"
    log.info("  Downloading %s ...", url)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.HTTPError:
        log.warning("  %s: HTTP error downloading %s — skipping", state_abbr, url)
        return None

    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
    log.info("  Saved → %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def _load_2024_zip_csv(zip_path: Path) -> pd.DataFrame:
    """Read the precinct-level CSV from a MEDSL 2024 state ZIP.

    2024 repo stores files without .csv extension inside the ZIP (e.g. 'fl24'
    not 'fl24.csv').  Falls back to the first non-directory entry.
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            csv_files = [n for n in zf.namelist() if not zf.getinfo(n).is_dir()]
        if not csv_files:
            raise FileNotFoundError(f"No data file in {zip_path}")
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)
    # Coerce vote columns — some states store as strings
    for col in ("votes", "totalvotes", "candidatevotes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _extract_senate_from_2024_precinct(
    df: pd.DataFrame, state_abbr: str
) -> pd.DataFrame:
    """Filter 2024 precinct data to US Senate general election and aggregate to county.

    Uses the same office/stage/writein filter logic as fetch_2024_president.py
    but targets Senate instead of President.  Mode dedup (TOTAL vs per-mode)
    follows the standard MEDSL pattern.

    Returns a long-form DataFrame with columns:
      county_fips, party_simplified, candidatevotes
    ready for _aggregate_to_county.  Returns empty DataFrame if no Senate rows
    exist for this state (not every state has a Senate race in 2024).
    """
    # Required columns — if missing, state file has unexpected schema
    required = {"office", "county_fips"}
    if not required.issubset(df.columns):
        log.warning("  %s: missing required columns %s — skipping", state_abbr, required - set(df.columns))
        return pd.DataFrame()

    # Filter to US Senate general election rows (exclude state senate, specials).
    # stage may not exist in all state files — when absent, include all rows.
    mask = (
        (df["office"].str.upper().str.contains("SENATE", na=False))
        & (~df["office"].str.upper().str.contains("STATE", na=False))
    )
    if "stage" in df.columns:
        mask &= df["stage"].str.lower().str.strip().isin({"gen", "general"})
    if "writein" in df.columns:
        mask &= ~df["writein"].fillna(False).astype(str).str.upper().str.strip().isin({"TRUE", "1"})

    senate = df[mask].copy()
    if senate.empty:
        # No Senate race in this state for 2024 — perfectly normal (Class I/II/III cycle)
        return pd.DataFrame()

    log.info("  %s: %d Senate precinct rows before mode dedup", state_abbr, len(senate))

    # Mode dedup: prefer TOTAL rows if they exist; otherwise sum across all modes
    if "mode" in senate.columns:
        modes = senate["mode"].str.upper().unique()
        if "TOTAL" in modes:
            senate = senate[senate["mode"].str.upper() == "TOTAL"]
            log.info("  Using TOTAL mode rows only (%d rows)", len(senate))

    # Normalise party column — 2024 data uses 'party_simplified'
    if "party_simplified" not in senate.columns and "party" in senate.columns:
        senate["party_simplified"] = senate["party"].fillna("OTHER").str.upper().str.strip()
    elif "party_simplified" in senate.columns:
        senate["party_simplified"] = senate["party_simplified"].fillna("OTHER").str.upper().str.strip()
    else:
        log.warning("  %s: no party column found — skipping", state_abbr)
        return pd.DataFrame()

    # Normalise party names (DEMOCRATIC → DEMOCRAT, etc.)
    party_map = {"DEMOCRATIC": "DEMOCRAT", "DEM": "DEMOCRAT", "REP": "REPUBLICAN"}
    senate["party_simplified"] = senate["party_simplified"].replace(party_map)

    # Use 'votes' as candidatevotes if the column name differs
    vote_col = "candidatevotes" if "candidatevotes" in senate.columns else "votes"
    if vote_col not in senate.columns:
        log.warning("  %s: no vote column found — skipping", state_abbr)
        return pd.DataFrame()

    # Zero-pad county_fips
    senate["county_fips"] = (
        senate["county_fips"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # Aggregate to county × party (summing across precincts and any multi-race rows)
    agg = (
        senate.groupby(["county_fips", "party_simplified"])[vote_col]
        .sum()
        .reset_index()
        .rename(columns={vote_col: "candidatevotes"})
    )
    log.info(
        "  %s 2024 Senate: %d county-party rows (%d counties)",
        state_abbr, len(agg), agg["county_fips"].nunique(),
    )
    return agg


def fetch_2024() -> pd.DataFrame:
    """Download MEDSL 2024 per-state ZIPs and aggregate Senate results to county.

    Iterates over all states in config.  States without a Senate race in 2024
    are silently skipped (expected — only ~33 states have a Senate race each cycle).
    States whose ZIP file is unavailable (HTTP 404) are also skipped with a warning.

    Returns a single aggregated DataFrame with the standard county-level columns
    for 2024, or an empty DataFrame if no data could be loaded.
    """
    frames: list[pd.DataFrame] = []

    for state_abbr, (filename, _fips) in MEDSL_2024_STATES.items():
        zip_path = _download_2024_zip(state_abbr, filename)
        if zip_path is None:
            continue

        try:
            raw = _load_2024_zip_csv(zip_path)
        except Exception as exc:
            log.error("  %s: failed to read ZIP — %s", state_abbr, exc)
            continue

        county_party = _extract_senate_from_2024_precinct(raw, state_abbr)
        if county_party.empty:
            continue

        frames.append(county_party)

    if not frames:
        log.warning("2024: no Senate data loaded from any state")
        return pd.DataFrame(columns=[
            "county_fips", "state_abbr",
            "senate_dem_2024", "senate_rep_2024",
            "senate_total_2024", "senate_dem_share_2024",
        ])

    combined = pd.concat(frames, ignore_index=True)
    return _aggregate_to_county(combined, 2024)


# ── VEST precinct-level (2016/2018/2020) ──────────────────────────────────────

_VEST_PARTY_MAP: dict[str, str] = {
    "DEMOCRATIC": "DEMOCRAT",
    "DEM": "DEMOCRAT",
    "REP": "REPUBLICAN",
    "": "OTHER",
    "NAN": "OTHER",
}


def _normalise_vest_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename VEST columns to canonical names and build party_simplified.

    Column names differ by year:
      2016: state_postal, party (raw lowercase), votes
      2018/2020: state_po, party_simplified (mixed case), votes

    All years end up with state_po, candidatevotes, and party_simplified
    mapped to DEMOCRAT / REPUBLICAN / OTHER.
    """
    df = df.copy()
    if "state_postal" in df.columns and "state_po" not in df.columns:
        df = df.rename(columns={"state_postal": "state_po"})
    if "votes" in df.columns and "candidatevotes" not in df.columns:
        df = df.rename(columns={"votes": "candidatevotes"})

    if "party_simplified" not in df.columns:
        src = df["party"].fillna("OTHER") if "party" in df.columns else "OTHER"
        df["party_simplified"] = src if isinstance(src, str) else src.str.upper().str.strip()
    else:
        df["party_simplified"] = df["party_simplified"].fillna("OTHER").str.upper().str.strip()

    df["party_simplified"] = df["party_simplified"].replace(_VEST_PARTY_MAP)
    return df


def _build_vest_row_mask(df: pd.DataFrame) -> "pd.Series":
    """Build boolean mask selecting US Senate general-election rows.

    Excludes state-level senate races (contains "STATE"), filters to the
    state set, and optionally filters by stage and writein columns.
    """
    mask = (
        (df["state_po"].str.upper().str.strip().isin(STATE_PO_SET))
        & (df["office"].str.upper().str.contains("SENATE", na=False))
        & (~df["office"].str.upper().str.contains("STATE", na=False))
    )
    if "stage" in df.columns:
        mask &= df["stage"].str.lower().str.strip().isin({"gen", "general"})
    if "writein" in df.columns:
        # Exclude pure write-in rows for cleaner totals
        mask &= ~df["writein"].astype(str).str.upper().str.strip().isin({"TRUE", "1"})
    return mask


def _filter_vest_senate(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter VEST precinct data to US Senate all-candidate general rows.

    Returns one row per (county_fips, party_simplified) with summed candidatevotes.
    Keeps ALL party rows (not just D/R) so _aggregate_to_county uses total-vote
    denominator; non-D/R rows appear in total but not in dem/rep sums.
    """
    df = _normalise_vest_columns(df)
    df = df[_build_vest_row_mask(df)].copy()
    df["candidatevotes"] = pd.to_numeric(df["candidatevotes"], errors="coerce").fillna(0)

    agg = (
        df.groupby(["county_fips", "party_simplified"], dropna=False)["candidatevotes"]
        .sum()
        .reset_index()
    )
    log.info("VEST %d after filter+agg: %d county-party rows (%d counties)",
             year, len(agg), agg["county_fips"].nunique())
    return agg


def fetch_vest_year(year: int) -> pd.DataFrame | None:
    """Download VEST Senate precinct data for a given year and aggregate to county."""
    if year not in VEST_SOURCES:
        log.warning("No VEST source defined for year %d", year)
        return None

    file_id, cache_path, sep = VEST_SOURCES[year]
    try:
        path = _download_file(file_id, cache_path)
    except Exception as exc:
        log.error("Failed to download VEST %d: %s", year, exc)
        return None

    log.info("Loading VEST %d (%.1f MB)...", year, path.stat().st_size / 1e6)
    df = pd.read_csv(path, sep=sep, low_memory=False)
    log.info("VEST %d raw rows: %d", year, len(df))

    df = _filter_vest_senate(df, year)
    if df.empty:
        log.warning("VEST %d: no rows after filtering", year)
        return None

    return _aggregate_to_county(df, year)


# ── Algara & Amlani historical Senate (2002-2014) ─────────────────────────────

def _load_algara_senate() -> "pd.DataFrame":
    """Download and cache the Algara & Amlani Senate Rdata, return raw DataFrame.

    The returned DataFrame is the full senate_elections_release object covering
    Senate general elections 1908-2020 at county level.  Key columns:
        election_year     float64   e.g. 2002.0
        fips              object    5-char or shorter county FIPS (needs zfill)
        state             object    2-letter abbreviation e.g. 'FL'
        election_type     object    'G' = general, 'S' = special
        democratic_raw_votes  float64
        republican_raw_votes  float64
        raw_county_vote_totals float64  total across ALL candidates
    """
    try:
        import pyreadr  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyreadr is required to read Algara .Rdata files. "
            "Install via: uv add pyreadr"
        ) from exc

    path = _download_file(ALGARA_SENATE_FILE_ID, ALGARA_SENATE_CACHE)
    log.info("Loading Algara Senate Rdata (%s)...", path.name)
    result = pyreadr.read_r(str(path))
    df = result["senate_elections_release"].reset_index(drop=True)
    log.info("Algara Senate raw rows: %d", len(df))
    return df


def _algara_year_to_long(year_df: "pd.DataFrame") -> "pd.DataFrame":
    """Pivot Algara wide-format year slice to candidatevotes long format.

    Algara stores dem and rep votes as separate columns. We pivot to the
    party_simplified + candidatevotes long form expected by _aggregate_to_county.
    Other/third-party votes = raw_county_vote_totals - dem - rep, included so
    _aggregate_to_county uses the correct total-vote denominator.
    """
    dem_rows = (
        year_df[["county_fips", "democratic_raw_votes"]].copy()
        .rename(columns={"democratic_raw_votes": "candidatevotes"})
        .assign(party_simplified="DEMOCRAT")
    )
    rep_rows = (
        year_df[["county_fips", "republican_raw_votes"]].copy()
        .rename(columns={"republican_raw_votes": "candidatevotes"})
        .assign(party_simplified="REPUBLICAN")
    )
    other_votes = (
        year_df["raw_county_vote_totals"]
        - year_df["democratic_raw_votes"].fillna(0)
        - year_df["republican_raw_votes"].fillna(0)
    ).clip(lower=0)
    other_rows = pd.DataFrame({
        "county_fips": year_df["county_fips"],
        "candidatevotes": other_votes,
        "party_simplified": "OTHER",
    })
    cols = ["county_fips", "party_simplified", "candidatevotes"]
    long_df = pd.concat(
        [dem_rows[cols], rep_rows[cols], other_rows[cols]],
        ignore_index=True,
    )
    long_df["candidatevotes"] = pd.to_numeric(long_df["candidatevotes"], errors="coerce").fillna(0)
    return long_df


def fetch_algara_historical(years: list[int] | None = None) -> dict[int, "pd.DataFrame"]:
    """Download Algara Senate data and produce one aggregated DataFrame per year.

    Parameters
    ----------
    years:
        Election years to process. Defaults to 2002-2014 (the years not covered
        by VEST or MEDSL county-level data).  Years where no state had a race
        will return an empty DataFrame (not an error).

    Returns
    -------
    dict mapping year → aggregated county DataFrame (may be empty for years
    with no contested races in the configured states).
    """
    if years is None:
        years = [2002, 2004, 2006, 2008, 2010, 2012, 2014]

    raw = _load_algara_senate()

    raw_filtered = (
        raw[raw["state"].isin(STATE_PO_SET) & (raw["election_type"] == "G")]
        .copy().reset_index(drop=True)
    )
    log.info("Algara Senate after state + general filter: %d rows", len(raw_filtered))
    raw_filtered["county_fips"] = _normalise_fips(raw_filtered["fips"])

    results: dict[int, pd.DataFrame] = {}
    for year in years:
        year_df = raw_filtered[raw_filtered["election_year"] == float(year)].copy().reset_index(drop=True)

        if year_df.empty:
            log.info("Algara Senate %d: no general races found — skipping", year)
            results[year] = pd.DataFrame()
            continue

        log.info(
            "Algara Senate %d: %d county-race rows across states: %s",
            year, len(year_df), year_df["state"].value_counts().to_dict(),
        )
        long_df = _algara_year_to_long(year_df)
        results[year] = _aggregate_to_county(long_df, year)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def _save_parquet(agg: "pd.DataFrame | None", year: int) -> None:
    """Save an aggregated county DataFrame to the assembled parquet directory.

    Silently skips when agg is None or empty (expected for years where no
    race was on the ballot in the configured states).
    """
    if agg is None or agg.empty:
        return
    out = ASSEMBLED_DIR / f"medsl_county_senate_{year}.parquet"
    agg.to_parquet(out, index=False)
    log.info("%d → %s (%d counties)", year, out.name, len(agg))


def _log_summary() -> None:
    """Log a one-line status for every output parquet in ASSEMBLED_DIR."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    all_years = [2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    for year in all_years:
        p = ASSEMBLED_DIR / f"medsl_county_senate_{year}.parquet"
        if p.exists():
            n = pq.read_metadata(p).num_rows
            log.info("  %d: %d counties ✓", year, n)
        else:
            log.info("  %d: no parquet (no eligible race this cycle)", year)


def main() -> None:
    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Processing 2002-2014 (Algara & Amlani county-level Senate) ===")
    try:
        historical = fetch_algara_historical([2002, 2004, 2006, 2008, 2010, 2012, 2014])
        for year, agg in historical.items():
            _save_parquet(agg, year)
    except Exception as exc:
        log.error("Algara historical fetch failed: %s", exc)

    for year in [2016, 2018, 2020]:
        log.info("=== Processing %d (VEST precinct aggregation) ===", year)
        try:
            _save_parquet(fetch_vest_year(year), year)
        except Exception as exc:
            log.error("%d failed: %s", year, exc)

    log.info("=== Processing 2022 (MEDSL county-level) ===")
    try:
        _save_parquet(fetch_2022(), 2022)
    except Exception as exc:
        log.error("2022 failed: %s", exc)

    log.info("=== Processing 2024 (MEDSL GitHub per-state precinct ZIPs) ===")
    try:
        _save_parquet(fetch_2024(), 2024)
    except Exception as exc:
        log.error("2024 failed: %s", exc)

    log.info("=== Senate fetch complete ===")
    _log_summary()


if __name__ == "__main__":
    main()
