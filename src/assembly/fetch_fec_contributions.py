"""Fetch FEC individual contribution totals for ActBlue and WinRed by county.

Source: FEC Open Data API v1
  https://api.open.fec.gov/v1/

Strategy:
  Use the schedules/schedule_a/by_zip/ endpoint which returns contribution
  totals (count + total amount) aggregated by ZIP code and election cycle.
  Map ZIP codes to county FIPS via the Census 2020 ZCTA-to-county relationship
  file (largest-overlap assignment). Filter to FL, GA, AL counties.

For each election cycle (2020, 2022, 2024):
  - Download ActBlue (C00401224) ZIP totals for FL, GA, AL
  - Download WinRed (C00694323) ZIP totals for FL, GA, AL
  - Map ZIP → county_fips (largest land-area overlap)
  - Aggregate to county level
  - Compute fec_dem_ratio = actblue / (actblue + winred)

Output:
  data/assembled/fec_county_contributions.parquet
  Columns: county_fips, state_abbr,
           fec_actblue_2020, fec_winred_2020, fec_dem_ratio_2020,
           fec_actblue_2022, fec_winred_2022, fec_dem_ratio_2022,
           fec_actblue_2024, fec_winred_2024, fec_dem_ratio_2024

Cache:
  data/raw/fec/zip_totals_{committee_id}_{cycle}.parquet
  data/raw/fec/zcta_county_crosswalk.parquet
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fec"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

FEC_BASE = "https://api.open.fec.gov/v1"
FEC_API_KEY = "DEMO_KEY"  # rate-limited to 1000/hr; set env var FEC_API_KEY to use key

ACTBLUE_ID = "C00401224"
WINRED_ID = "C00694323"

# FEC election cycles to fetch (2-year periods ending in even year)
FEC_CYCLES = [2020, 2022, 2024]

# Maps fips_prefix → state_abbr  (e.g. "12" → "FL")
STATE_ABBR: dict[str, str] = _cfg.STATE_ABBR

# State abbreviations in scope
TARGET_STATES = set(STATE_ABBR.values())  # {"FL", "GA", "AL"}

# Census ZCTA-to-county relationship file (2020 vintage)
ZCTA_COUNTY_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_county20_natl.txt"
)
ZCTA_CROSSWALK_CACHE = RAW_DIR / "zcta_county_crosswalk.parquet"

# Polite delay between API pages (seconds)
_PAGE_SLEEP = 0.5


# ── ZIP-to-county crosswalk ───────────────────────────────────────────────────


def _load_zcta_crosswalk() -> pd.DataFrame:
    """Return DataFrame with columns: zip5, county_fips.

    Uses 2020 Census ZCTA-to-county relationship file. Assigns each ZIP to
    the county with the largest land-area overlap (unique mapping).
    Downloads once and caches to data/raw/fec/.
    """
    if ZCTA_CROSSWALK_CACHE.exists():
        log.info("Using cached ZCTA crosswalk: %s", ZCTA_CROSSWALK_CACHE)
        return pd.read_parquet(ZCTA_CROSSWALK_CACHE)

    log.info("Downloading Census ZCTA-county relationship file...")
    resp = requests.get(ZCTA_COUNTY_URL, timeout=120)
    resp.raise_for_status()

    # Pipe-delimited; BOM-stripped; relevant columns: GEOID_ZCTA5_20, GEOID_COUNTY_20,
    # AREALAND_PART (land area of ZIP within county)
    from io import StringIO
    text = resp.text.lstrip("\ufeff")  # strip BOM
    df = pd.read_csv(StringIO(text), sep="|", low_memory=False)

    # Some rows have blank GEOID_ZCTA5_20 (county-only rows without ZIP assignment)
    df = df[df["GEOID_ZCTA5_20"].notna()].copy()
    df["zip5"] = df["GEOID_ZCTA5_20"].astype(str).str.zfill(5)
    df["county_fips"] = df["GEOID_COUNTY_20"].astype(str).str.zfill(5)
    df["arealand"] = pd.to_numeric(df["AREALAND_PART"], errors="coerce").fillna(0)

    # Assign each ZIP to county with largest land-area overlap
    idx = df.groupby("zip5")["arealand"].idxmax()
    crosswalk = df.loc[idx, ["zip5", "county_fips"]].reset_index(drop=True)

    # Filter to FL, GA, AL counties only
    target_prefixes = set(STATE_ABBR.keys())  # {"12", "13", "01"}
    crosswalk = crosswalk[
        crosswalk["county_fips"].str[:2].isin(target_prefixes)
    ].copy()

    log.info("ZCTA crosswalk: %d ZIPs mapped to FL/GA/AL counties", len(crosswalk))

    ZCTA_CROSSWALK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(ZCTA_CROSSWALK_CACHE, index=False)
    log.info("Saved crosswalk → %s", ZCTA_CROSSWALK_CACHE)
    return crosswalk


# ── FEC API helpers ───────────────────────────────────────────────────────────


def _fetch_zip_totals_page(
    committee_id: str,
    cycle: int,
    state: str,
    page: int,
    per_page: int = 100,
) -> dict:
    """Fetch one page from the FEC schedules/schedule_a/by_zip/ endpoint."""
    import os
    api_key = os.environ.get("FEC_API_KEY", FEC_API_KEY)
    params = {
        "committee_id": committee_id,
        "two_year_transaction_period": cycle,
        "state": state,
        "per_page": per_page,
        "page": page,
        "sort": "zip",
        "api_key": api_key,
    }
    url = f"{FEC_BASE}/schedules/schedule_a/by_zip/"
    resp = requests.get(url, params=params, timeout=120)
    if resp.status_code == 429:
        log.warning("FEC rate limit hit — sleeping 65s")
        time.sleep(65)
        resp = requests.get(url, params=params, timeout=120)
    elif resp.status_code >= 500:
        log.warning("FEC server error %d — retrying in 30s", resp.status_code)
        time.sleep(30)
        resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _fetch_zip_totals(
    committee_id: str,
    cycle: int,
    state: str,
) -> pd.DataFrame:
    """Fetch all pages of ZIP-level contribution totals for one committee/cycle/state."""
    rows: list[dict] = []
    page = 1
    while True:
        data = _fetch_zip_totals_page(committee_id, cycle, state, page)
        results = data.get("results", [])
        rows.extend(results)
        pagination = data.get("pagination", {})
        total_pages = pagination.get("pages", 1)
        log.debug(
            "  %s cycle=%d state=%s page=%d/%d rows=%d",
            committee_id, cycle, state, page, total_pages, len(results),
        )
        if page >= total_pages or not results:
            break
        page += 1
        time.sleep(_PAGE_SLEEP)

    if not rows:
        return pd.DataFrame(columns=["zip", "state", "cycle", "total", "count"])

    df = pd.DataFrame(rows)
    df = df[["zip", "state", "cycle", "total", "count"]].copy()
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    return df


def fetch_committee_zip_totals(
    committee_id: str,
    cycle: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and cache ZIP totals for a committee + cycle across FL, GA, AL.

    Cache path: data/raw/fec/zip_totals_{committee_id}_{cycle}.parquet
    """
    cache_path = RAW_DIR / f"zip_totals_{committee_id}_{cycle}.parquet"
    if cache_path.exists() and not force_refresh:
        log.info("Using cached FEC data: %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info(
        "Fetching FEC ZIP totals: committee=%s cycle=%d states=%s",
        committee_id, cycle, sorted(TARGET_STATES),
    )
    frames: list[pd.DataFrame] = []
    for state in sorted(TARGET_STATES):
        log.info("  state=%s ...", state)
        df = _fetch_zip_totals(committee_id, cycle, state)
        frames.append(df)
        time.sleep(_PAGE_SLEEP)

    if not frames or all(f.empty for f in frames):
        combined = pd.DataFrame(columns=["zip", "state", "cycle", "total", "count"])
    else:
        combined = pd.concat(frames, ignore_index=True)

    combined["zip5"] = combined["zip"].astype(str).str[:5].str.zfill(5)
    log.info("  Total ZIP rows: %d", len(combined))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    log.info("  Saved → %s", cache_path)
    return combined


# ── Aggregation ───────────────────────────────────────────────────────────────


def zip_to_county(
    df: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Map ZIP-level totals to county level using the ZCTA crosswalk.

    Joins on zip5, groups to county_fips, sums total and count.
    Rows whose ZIP is not in the crosswalk (outside FL/GA/AL or unmapped) are dropped.
    """
    merged = df.merge(crosswalk[["zip5", "county_fips"]], on="zip5", how="inner")
    agg = (
        merged.groupby("county_fips", as_index=False)
        .agg(total=("total", "sum"), count=("count", "sum"))
    )
    return agg


def compute_dem_ratio(
    actblue: float | pd.Series,
    winred: float | pd.Series,
) -> float | pd.Series:
    """Compute Democratic ratio: actblue / (actblue + winred).

    Returns 0.5 when both are zero (neutral imputation).
    Returns 1.0 when only actblue > 0; 0.0 when only winred > 0.
    """
    total = actblue + winred
    if isinstance(total, pd.Series):
        ratio = pd.Series(0.5, index=total.index, dtype=float)
        nonzero = total > 0
        ratio[nonzero] = actblue[nonzero] / total[nonzero]
        return ratio
    # Scalar case
    if total == 0:
        return 0.5
    return actblue / total


# ── Spine of all FL/GA/AL counties ───────────────────────────────────────────


def _build_county_spine(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Build a complete list of FL/GA/AL county FIPS codes with state_abbr."""
    spine = crosswalk[["county_fips"]].drop_duplicates().copy()
    spine["state_abbr"] = spine["county_fips"].str[:2].map(STATE_ABBR)
    spine = spine[spine["state_abbr"].notna()].reset_index(drop=True)
    return spine


# ── Main assembly ─────────────────────────────────────────────────────────────


def build_fec_features(
    cycles: list[int] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build county-level FEC partisan ratio features for requested cycles.

    Returns a DataFrame with county_fips, state_abbr, and per-cycle columns:
      fec_actblue_{cycle}, fec_winred_{cycle}, fec_dem_ratio_{cycle}
    """
    if cycles is None:
        cycles = FEC_CYCLES

    crosswalk = _load_zcta_crosswalk()
    result = _build_county_spine(crosswalk)

    for cycle in cycles:
        log.info("Processing cycle %d ...", cycle)

        # ActBlue
        ab_zip = fetch_committee_zip_totals(ACTBLUE_ID, cycle, force_refresh)
        ab_county = zip_to_county(ab_zip, crosswalk).rename(
            columns={"total": f"fec_actblue_{cycle}", "count": f"fec_ab_count_{cycle}"}
        )

        # WinRed
        wr_zip = fetch_committee_zip_totals(WINRED_ID, cycle, force_refresh)
        wr_county = zip_to_county(wr_zip, crosswalk).rename(
            columns={"total": f"fec_winred_{cycle}", "count": f"fec_wr_count_{cycle}"}
        )

        result = result.merge(
            ab_county[["county_fips", f"fec_actblue_{cycle}"]],
            on="county_fips", how="left",
        )
        result = result.merge(
            wr_county[["county_fips", f"fec_winred_{cycle}"]],
            on="county_fips", how="left",
        )
        result[f"fec_actblue_{cycle}"] = result[f"fec_actblue_{cycle}"].fillna(0.0)
        result[f"fec_winred_{cycle}"] = result[f"fec_winred_{cycle}"].fillna(0.0)

        result[f"fec_dem_ratio_{cycle}"] = compute_dem_ratio(
            result[f"fec_actblue_{cycle}"],
            result[f"fec_winred_{cycle}"],
        )

        n_nonzero = (
            (result[f"fec_actblue_{cycle}"] > 0) | (result[f"fec_winred_{cycle}"] > 0)
        ).sum()
        log.info(
            "  Cycle %d: %d counties with data, %d zero (imputed to 0.5)",
            cycle, n_nonzero, len(result) - n_nonzero,
        )

    return result


def main() -> None:
    df = build_fec_features()

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSEMBLED_DIR / "fec_county_contributions.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved → %s (%d counties)", out, len(df))

    # Print summary
    for cycle in FEC_CYCLES:
        col = f"fec_dem_ratio_{cycle}"
        if col in df.columns:
            log.info(
                "  %d: mean dem_ratio=%.3f  min=%.3f  max=%.3f",
                cycle,
                df[col].mean(),
                df[col].min(),
                df[col].max(),
            )


if __name__ == "__main__":
    main()
