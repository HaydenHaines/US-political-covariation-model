"""Fetch Algara & Amlani county-level governor returns 2002–2018.

Dataset: Algara & Amlani County Electoral Dataset (doi:10.7910/DVN/DGUMFI)
Harvard Dataverse. Contains county-level returns for governor, president, and
U.S. Senate from 1865–2020.

This module:
1. Downloads the gubernatorial .Rdata file from Harvard Dataverse and caches it
   to data/raw/algara_amlani/ (no re-download if already present).
2. Filters to general-election governor races for FL, GA, AL.
3. Computes 2-party dem share per county per target year.
4. Writes one parquet per year to data/assembled/:
     algara_county_governor_{year}.parquet

Output columns per year:
    county_fips       str  5-char zero-padded FIPS
    state_abbr        str  'FL' / 'GA' / 'AL'
    gov_dem_{year}    float  raw Democratic votes
    gov_rep_{year}    float  raw Republican votes
    gov_total_{year}  float  two-party total
    gov_dem_share_{year}  float  dem / (dem + rep), 2-party

Note: AL 2018 governor was contested (Kay Ivey vs. Walter Maddox). The dataset
includes actual vote counts for both candidates. The shift builder downstream
handles structural zeros for truly uncontested cycles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATES: dict[str, str] = {"FL": "12", "GA": "13", "AL": "01"}
GOV_YEARS: list[int] = [2002, 2006, 2010, 2014, 2018]

_DATAVERSE_BASE = "https://dataverse.harvard.edu/api"
_DATASET_PID = "doi:10.7910/DVN/DGUMFI"
_FILE_NAME = "dataverse_shareable_gubernatorial_county_returns_1865_2020.Rdata"
_RAW_DIR = Path("data/raw/algara_amlani")
_OUT_DIR = Path("data/assembled")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _get_file_id() -> int:
    """Look up the Dataverse file ID for the gubernatorial .Rdata file."""
    import requests  # local import to avoid hard dep at module load

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
        print(f"[fetch_algara_amlani] Using cached file: {out_path}", flush=True)
        return out_path

    print(f"[fetch_algara_amlani] Looking up file ID on Dataverse...", flush=True)
    file_id = _get_file_id()
    url = f"{_DATAVERSE_BASE}/access/datafile/{file_id}"
    print(f"[fetch_algara_amlani] Downloading {_FILE_NAME} (file_id={file_id})...", flush=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(65536):
                fh.write(chunk)
    print(f"[fetch_algara_amlani] Saved to {out_path} ({out_path.stat().st_size:,} bytes)", flush=True)
    return out_path


def _load_raw() -> pd.DataFrame:
    """Load the raw .Rdata file, returning the gov_elections_release DataFrame."""
    import pyreadr

    path = _download_raw()
    result = pyreadr.read_r(str(path))
    return result["gov_elections_release"]


# ---------------------------------------------------------------------------
# Core filter / aggregation logic
# ---------------------------------------------------------------------------


def filter_governor_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to general-election governor rows for FL, GA, AL only.

    Parameters
    ----------
    df:
        Full gov_elections_release DataFrame (or a synthetic equivalent).

    Returns
    -------
    Filtered DataFrame with the same columns as input.
    """
    target_states = set(STATES.keys())
    mask = (
        (df["office"] == "GOV")
        & (df["state"].isin(target_states))
    )
    return df[mask].copy()


def aggregate_county_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate a pre-filtered DataFrame to one row per county for a given year.

    Parameters
    ----------
    df:
        DataFrame already filtered to governor rows for FL/GA/AL (output of
        filter_governor_rows or a synthetic equivalent).  May contain multiple
        years; this function selects the requested year.
    year:
        Election year to aggregate (e.g. 2006).

    Returns
    -------
    DataFrame with columns:
        county_fips, state_abbr,
        gov_dem_{year}, gov_rep_{year}, gov_total_{year}, gov_dem_share_{year}
    """
    year_df = df[df["election_year"] == float(year)].copy()

    # Build output
    out = pd.DataFrame()
    out["county_fips"] = year_df["fips"].astype(str).str.zfill(5).values
    out["state_abbr"] = year_df["state"].values

    dem_col = f"gov_dem_{year}"
    rep_col = f"gov_rep_{year}"
    total_col = f"gov_total_{year}"
    share_col = f"gov_dem_share_{year}"

    dem = year_df["democratic_raw_votes"].values
    rep = year_df["republican_raw_votes"].values
    total = year_df["gov_raw_county_vote_totals_two_party"].values

    out[dem_col] = dem
    out[rep_col] = rep
    out[total_col] = total

    # 2-party dem share; NaN where total is 0 (uncontested with no votes recorded)
    import numpy as np

    two_party_sum = dem + rep
    share = dem / two_party_sum
    share = pd.array(share, dtype=float)
    # Where two_party_sum is 0 or NaN, set share to NaN
    share[two_party_sum == 0] = float("nan")
    out[share_col] = share

    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(years: list[int] | None = None) -> None:
    """Download, filter, aggregate, and write parquets for each target year."""
    if years is None:
        years = GOV_YEARS

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[fetch_algara_amlani] Loading raw data...", flush=True)
    raw = _load_raw()

    filtered = filter_governor_rows(raw)
    print(
        f"[fetch_algara_amlani] After filtering to FL/GA/AL governor rows: {len(filtered)} rows",
        flush=True,
    )

    for year in years:
        out = aggregate_county_year(filtered, year)
        out_path = _OUT_DIR / f"algara_county_governor_{year}.parquet"
        out.to_parquet(out_path, index=False)
        n = len(out)
        by_state = out.groupby("state_abbr").size().to_dict()
        print(
            f"[fetch_algara_amlani] {year}: {n} counties → {out_path}  ({by_state})",
            flush=True,
        )

    print("[fetch_algara_amlani] Done.", flush=True)


if __name__ == "__main__":
    run()
