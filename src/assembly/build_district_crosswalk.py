"""
Build county-to-congressional-district crosswalk from Census Block Assignment Files.

Strategy:
  1. Download Census BAF2020 _CD.txt files (block → district mapping)
  2. Join with DRA block-level election data (block → vote totals)
  3. Aggregate to county-district pairs weighted by 2024 presidential votes

The BAF2020 files map 2020 Census blocks to post-redistricting congressional
districts (118th/119th Congress, effective 2023-2027). The DRA block data uses
the same 2020 Census block geography, so GEOIDs join directly.

Output: data/districts/county_district_crosswalk.parquet
  - county_fips: str, 5-digit FIPS code
  - state_fips: str, 2-digit state FIPS
  - district_id: str, format "SS-DD" (e.g., "01-07" for AL-7)
  - overlap_votes: int, total 2024 presidential votes in the county-district intersection
  - overlap_fraction: float, fraction of county's votes in this district (sums to 1.0 per county)

At-large states (AK, DE, MT, ND, SD, VT, WY) have a single district "SS-00"
with overlap_fraction=1.0 for all counties.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
DRA_DIR = PROJECT_ROOT / "data" / "raw" / "dra-block-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "districts"

BAF_BASE_URL = "https://www2.census.gov/geo/docs/maps-data/data/baf2020"

# State FIPS codes (50 states + DC)
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

STATE_ABBR_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}

# At-large states: single congressional district (post-2020 redistricting)
# MT gained a 2nd district after 2020 Census — no longer at-large
AT_LARGE_STATES = {"AK", "DE", "ND", "SD", "VT", "WY"}

# DRA block data vote column (2024 presidential total as population weight)
VOTE_COL = "E_24_PRES_Total"
VOTE_COL_FALLBACK = "E_20_PRES_Total"


def download_baf_cd(state_fips: str, state_abbr: str) -> pd.DataFrame:
    """Download Census BAF2020 block-to-CD mapping for one state.

    Returns DataFrame with columns: block_geoid (str), district (str, 2-digit).
    """
    url = f"{BAF_BASE_URL}/BlockAssign_ST{state_fips}_{state_abbr}.zip"
    cd_filename = f"BlockAssign_ST{state_fips}_{state_abbr}_CD.txt"

    log.info("Downloading BAF for %s (%s)", state_abbr, url)
    response = urllib.request.urlopen(url, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(response.read()))

    with z.open(cd_filename) as f:
        df = pd.read_csv(
            f,
            sep="|",
            dtype={"BLOCKID": str, "DISTRICT": str},
        )

    df = df.rename(columns={"BLOCKID": "block_geoid", "DISTRICT": "district"})
    return df[["block_geoid", "district"]]


def load_dra_block_votes(state_abbr: str, dra_dir: Path | None = None) -> pd.DataFrame:
    """Load DRA block-level vote totals for one state.

    Returns DataFrame with columns: block_geoid (str), votes (int).
    Uses 2024 presidential vote total; falls back to 2020 if 2024 unavailable.
    """
    if dra_dir is None:
        dra_dir = DRA_DIR
    state_dir = dra_dir / state_abbr
    csv_pattern = f"election_data_block_{state_abbr}.v06.csv"

    # Find the CSV (may be in a version subdirectory)
    candidates = list(state_dir.rglob(csv_pattern))
    if not candidates:
        # Try case-insensitive
        candidates = [
            p for p in state_dir.rglob("*.csv")
            if p.name.lower() == csv_pattern.lower()
        ]
    if not candidates:
        raise FileNotFoundError(f"No DRA block data found for {state_abbr} in {state_dir}")

    csv_path = candidates[0]
    df = pd.read_csv(csv_path, dtype={"GEOID": str}, usecols=lambda c: c in {"GEOID", VOTE_COL, VOTE_COL_FALLBACK})

    if VOTE_COL in df.columns:
        df["votes"] = df[VOTE_COL].fillna(0).astype(int)
    elif VOTE_COL_FALLBACK in df.columns:
        log.warning("%s: using fallback vote column %s", state_abbr, VOTE_COL_FALLBACK)
        df["votes"] = df[VOTE_COL_FALLBACK].fillna(0).astype(int)
    else:
        raise ValueError(f"No vote column found in DRA data for {state_abbr}")

    df = df.rename(columns={"GEOID": "block_geoid"})
    return df[["block_geoid", "votes"]]


def build_state_crosswalk(
    state_fips: str,
    state_abbr: str,
    dra_dir: Path | None = None,
) -> pd.DataFrame:
    """Build county-district crosswalk for one state.

    Downloads BAF, joins with DRA vote data, aggregates to county-district level.
    """
    # Download block-to-district mapping
    baf = download_baf_cd(state_fips, state_abbr)

    # Load block-level votes
    votes = load_dra_block_votes(state_abbr, dra_dir=dra_dir)

    # Join on block GEOID
    merged = baf.merge(votes, on="block_geoid", how="left")
    merged["votes"] = merged["votes"].fillna(0).astype(int)

    # Extract county FIPS from block GEOID (first 5 digits)
    merged["county_fips"] = merged["block_geoid"].str[:5]

    # Aggregate to county-district level
    crosswalk = (
        merged
        .groupby(["county_fips", "district"], as_index=False)
        .agg(overlap_votes=("votes", "sum"))
    )

    # Compute overlap fraction per county (votes in this district / total county votes)
    county_totals = crosswalk.groupby("county_fips")["overlap_votes"].transform("sum")
    # Avoid division by zero for counties with no votes (use equal weight)
    county_n_districts = crosswalk.groupby("county_fips")["district"].transform("count")
    crosswalk["overlap_fraction"] = crosswalk["overlap_votes"] / county_totals.where(
        county_totals > 0, other=1
    )
    # For zero-vote counties, split equally across districts
    zero_mask = county_totals == 0
    crosswalk.loc[zero_mask, "overlap_fraction"] = 1.0 / county_n_districts[zero_mask]

    # Format district_id as "SS-DD"
    crosswalk["state_fips"] = state_fips
    crosswalk["district_id"] = state_fips + "-" + crosswalk["district"]

    return crosswalk[["county_fips", "state_fips", "district_id", "overlap_votes", "overlap_fraction"]]


def build_crosswalk(
    states: list[str] | None = None,
    dra_dir: Path | None = None,
) -> pd.DataFrame:
    """Build national county-to-district crosswalk.

    Args:
        states: List of state abbreviations to process. None = all 50 states + DC.
        dra_dir: Path to DRA block data root. Defaults to PROJECT_ROOT/data/raw/dra-block-data.

    Returns:
        DataFrame with county-district crosswalk for all requested states.
    """
    if states is None:
        states = sorted(STATE_FIPS.values())

    # DC has a non-voting delegate, not a congressional district -- skip
    states = [s for s in states if s != "DC"]

    all_crosswalks = []

    for state_abbr in states:
        state_fips = STATE_ABBR_TO_FIPS[state_abbr]
        try:
            cw = build_state_crosswalk(state_fips, state_abbr, dra_dir=dra_dir)
            log.info(
                "%s: %d county-district pairs, %d counties, %d districts",
                state_abbr, len(cw),
                cw["county_fips"].nunique(),
                cw["district_id"].nunique(),
            )
            all_crosswalks.append(cw)
        except Exception as e:
            log.error("Failed to build crosswalk for %s: %s", state_abbr, e)
            raise

    crosswalk = pd.concat(all_crosswalks, ignore_index=True)

    # Validate: overlap fractions sum to ~1.0 per county
    county_sums = crosswalk.groupby("county_fips")["overlap_fraction"].sum()
    bad_counties = county_sums[~county_sums.between(0.999, 1.001)]
    if len(bad_counties) > 0:
        log.warning(
            "%d counties have overlap fractions not summing to 1.0: %s",
            len(bad_counties),
            bad_counties.head(5).to_dict(),
        )

    log.info(
        "National crosswalk: %d county-district pairs, %d counties, %d districts",
        len(crosswalk),
        crosswalk["county_fips"].nunique(),
        crosswalk["district_id"].nunique(),
    )

    return crosswalk


def main() -> None:
    """Build and save the national county-district crosswalk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "county_district_crosswalk.parquet"

    crosswalk = build_crosswalk()

    crosswalk.to_parquet(output_path, index=False)
    log.info("Saved crosswalk to %s (%d rows)", output_path, len(crosswalk))


if __name__ == "__main__":
    main()
