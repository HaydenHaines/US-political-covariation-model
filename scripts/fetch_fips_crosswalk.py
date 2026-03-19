# scripts/fetch_fips_crosswalk.py
"""Download Census 2023 Gazetteer county file and save FIPS→name crosswalk.

Output: data/raw/fips_county_crosswalk.csv
Columns: county_fips (5-char str), county_name (e.g. "Alachua County, FL")
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"

# Census 2023 Gazetteer: tab-separated, columns include USPS, GEOID, NAME
GAZ_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.txt"

# Fallback: Census 2020 national county codes (pipe-separated)
FALLBACK_URL = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"


def _load_gazetteer(text: str) -> pd.DataFrame:
    """Parse the Census Gazetteer tab-separated format."""
    df = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        dtype=str,
        usecols=["USPS", "GEOID", "NAME"],
    )
    df = df.rename(columns={"GEOID": "county_fips", "USPS": "state_abbr", "NAME": "name"})
    df["county_name"] = df["name"] + ", " + df["state_abbr"]
    df = df[["county_fips", "county_name"]].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)
    return df


def _load_fallback(text: str) -> pd.DataFrame:
    """Parse the Census 2020 national_county pipe-separated format.

    Columns: STATE|STATEFP|COUNTYFP|COUNTYNS|COUNTYNAME
    """
    df = pd.read_csv(
        io.StringIO(text),
        sep="|",
        dtype=str,
        usecols=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"],
    )
    df["county_fips"] = df["STATEFP"].str.zfill(2) + df["COUNTYFP"].str.zfill(3)
    df["county_name"] = df["COUNTYNAME"] + ", " + df["STATE"]
    df = df[["county_fips", "county_name"]].copy()
    return df


def main() -> None:
    df: pd.DataFrame | None = None

    # Try primary URL first
    print(f"Downloading Census Gazetteer from {GAZ_URL}")
    resp = requests.get(GAZ_URL, timeout=60)
    if resp.ok:
        df = _load_gazetteer(resp.text)
    else:
        print(f"Primary URL returned {resp.status_code}; trying fallback: {FALLBACK_URL}")
        resp2 = requests.get(FALLBACK_URL, timeout=60)
        resp2.raise_for_status()
        df = _load_fallback(resp2.text)

    assert df is not None

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    # Spot-check
    sample = df[df["county_fips"].str.startswith("12")].head(3)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
