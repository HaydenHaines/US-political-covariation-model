"""Fetch Algara & Amlani county-level presidential returns 1948–2000.

Dataset: Algara & Amlani County Electoral Dataset (doi:10.7910/DVN/DGUMFI)
Harvard Dataverse. Same dataset as the governor/senate files already cached
in data/raw/algara_amlani/. The presidential file covers 1868–2020.

This module:
1. Downloads the presidential .Rdata file from Harvard Dataverse and caches it
   to data/raw/algara_amlani/ (no re-download if already present).
2. Filters to presidential general-election rows for years 1948–2000.
3. Excludes Alaska (FIPS prefix 02) for elections before 1972 — Alaska's FIPS
   codes were unstable before the 1970 census assignment.
4. Computes 2-party dem share per county per target year.
5. Writes one parquet per year to data/assembled/:
     algara_county_presidential_{year}.parquet

Output columns per year:
    county_fips           str    5-char zero-padded FIPS
    state_abbr            str    e.g. 'FL'
    pres_dem_{year}       float  raw Democratic votes
    pres_rep_{year}       float  raw Republican votes
    pres_total_{year}     float  total votes (raw_county_vote_totals)
    pres_dem_share_{year} float  dem / total_all_candidates
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRES_YEARS: list[int] = [
    1948, 1952, 1956, 1960, 1964,
    1968, 1972, 1976, 1980, 1984,
    1988, 1992, 1996, 2000,
]

_DATAVERSE_BASE = "https://dataverse.harvard.edu/api"
_DATASET_PID = "doi:10.7910/DVN/DGUMFI"
_FILE_NAME = "dataverse_shareable_presidential_county_returns_1868_2020.Rdata"
_RAW_DIR = Path("data/raw/algara_amlani")
_OUT_DIR = Path("data/assembled")

# Alaska FIPS prefix — excluded before 1972 due to FIPS instability
_ALASKA_PREFIX = "02"
_ALASKA_STABLE_FROM = 1972


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _get_file_id() -> int:
    """Look up the Dataverse file ID for the presidential .Rdata file."""
    import requests

    url = f"{_DATAVERSE_BASE}/datasets/:persistentId/versions/:latest/files"
    resp = requests.get(url, params={"persistentId": _DATASET_PID}, timeout=30)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["dataFile"]["filename"] == _FILE_NAME:
            return int(entry["dataFile"]["id"])
    raise RuntimeError(f"File '{_FILE_NAME}' not found in dataset {_DATASET_PID}")


def _download_raw() -> Path:
    """Download the raw .Rdata file if not already cached. Returns local path."""
    import requests

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RAW_DIR / _FILE_NAME
    if out_path.exists():
        print(f"[fetch_algara_presidential] Using cached file: {out_path}", flush=True)
        return out_path

    print("[fetch_algara_presidential] Looking up file ID on Dataverse...", flush=True)
    file_id = _get_file_id()
    url = f"{_DATAVERSE_BASE}/access/datafile/{file_id}"
    print(f"[fetch_algara_presidential] Downloading {_FILE_NAME} (file_id={file_id})...", flush=True)
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(65536):
                fh.write(chunk)
    print(
        f"[fetch_algara_presidential] Saved to {out_path} ({out_path.stat().st_size:,} bytes)",
        flush=True,
    )
    return out_path


def _load_raw() -> pd.DataFrame:
    """Load the raw .Rdata file. Returns the presidential election DataFrame."""
    import pyreadr

    path = _download_raw()
    result = pyreadr.read_r(str(path))
    # Key name follows the dataset convention: pres_elections_release
    # Fall back to the first non-.Random.seed key if the name differs.
    candidate_keys = [k for k in result.keys() if not k.startswith(".")]
    if not candidate_keys:
        raise RuntimeError(f"No usable keys in {path}. Keys: {list(result.keys())}")
    key = "pres_elections_release" if "pres_elections_release" in result else candidate_keys[0]
    print(f"[fetch_algara_presidential] Loading R object: '{key}'", flush=True)
    return result[key]


# ---------------------------------------------------------------------------
# Core filter / aggregation logic (testable without network)
# ---------------------------------------------------------------------------


def filter_presidential_rows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter to general-election presidential rows for a single year.

    Excludes Alaska (FIPS prefix '02') for years before 1972.

    Parameters
    ----------
    df:
        Full presidential elections DataFrame.
    year:
        Election year to select.

    Returns
    -------
    Filtered DataFrame.
    """
    mask = (df["election_year"] == float(year))
    filtered = df[mask].copy()

    if year < _ALASKA_STABLE_FROM:
        pre_alaska = filtered["fips"].astype(str).str.zfill(5).str.startswith(_ALASKA_PREFIX)
        n_dropped = pre_alaska.sum()
        if n_dropped:
            print(
                f"[fetch_algara_presidential] {year}: dropping {n_dropped} Alaska rows (pre-1972 FIPS instability)",
                flush=True,
            )
        filtered = filtered[~pre_alaska].copy()

    return filtered


def aggregate_county_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate a filtered DataFrame to one row per county for a given year.

    Parameters
    ----------
    df:
        DataFrame already filtered to presidential rows for the target year
        (output of filter_presidential_rows or a synthetic equivalent).
    year:
        Election year (e.g. 1976).

    Returns
    -------
    DataFrame with columns:
        county_fips, state_abbr,
        pres_dem_{year}, pres_rep_{year}, pres_total_{year}, pres_dem_share_{year}
    """
    out = pd.DataFrame()
    out["county_fips"] = df["fips"].astype(str).str.zfill(5).values
    out["state_abbr"] = df["state"].values

    dem_col = f"pres_dem_{year}"
    rep_col = f"pres_rep_{year}"
    total_col = f"pres_total_{year}"
    share_col = f"pres_dem_share_{year}"

    dem = df["democratic_raw_votes"].values.astype(float)
    rep = df["republican_raw_votes"].values.astype(float)
    total_all = df["raw_county_vote_totals"].values.astype(float)
    two_party_sum = dem + rep
    use_fallback = (total_all == 0) | np.isnan(total_all)
    total = np.where(use_fallback, two_party_sum, total_all)

    out[dem_col] = dem
    out[rep_col] = rep
    out[total_col] = total

    share = dem / total
    share = pd.array(share, dtype=float)
    share[total == 0] = float("nan")
    out[share_col] = share

    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(years: list[int] | None = None) -> None:
    """Download, filter, aggregate, and write parquets for each target year."""
    if years is None:
        years = PRES_YEARS

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[fetch_algara_presidential] Loading raw data...", flush=True)
    raw = _load_raw()
    print(f"[fetch_algara_presidential] Raw data: {len(raw)} rows", flush=True)

    for year in years:
        filtered = filter_presidential_rows(raw, year)
        if len(filtered) == 0:
            print(f"[fetch_algara_presidential] {year}: no rows after filtering — skipping", flush=True)
            continue
        out = aggregate_county_year(filtered, year)
        out_path = _OUT_DIR / f"algara_county_presidential_{year}.parquet"
        out.to_parquet(out_path, index=False)
        print(
            f"[fetch_algara_presidential] {year}: {len(out)} counties → {out_path}",
            flush=True,
        )

    print("[fetch_algara_presidential] Done.", flush=True)


if __name__ == "__main__":
    run()
