"""
Fetch 2022 gubernatorial election results at county level for all 50 states + DC.

Source: MIT Election Data + Science Lab (MEDSL) 2022-elections-official GitHub repo.
  https://github.com/MEDSL/2022-elections-official/tree/main/individual_states

Not every state holds a governor race in even-numbered years (many states use
odd-year cycles, e.g. KY, LA, MS, NJ, VA). States without a ZIP file in MEDSL,
or whose file exists but contains no GOVERNOR rows, are skipped gracefully.

Data is precinct-level CSV with county_fips attached — we aggregate to county.
No spatial join required (county-level validation only; cannot extend Stan model
without tract-level spatial data).

Mode dedup: MEDSL sometimes includes both mode-specific rows (ABSENTEE, ELECTION DAY)
and a TOTAL row. We use TOTAL rows only if present; otherwise sum all modes.

Vote share: gov_dem_share_2022 = gov_dem_2022 / gov_total_2022, where
gov_total_2022 is the sum of votes for ALL candidates (not just D+R).

Output:
  data/assembled/medsl_county_2022_governor.parquet
  Columns: county_fips, state_abbr, gov_dem_2022, gov_rep_2022,
           gov_total_2022, gov_dem_share_2022
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medsl" / "2022"
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"

# MEDSL 2022-elections-official GitHub raw download URLs
MEDSL_BASE = (
    "https://github.com/MEDSL/2022-elections-official/raw/main/individual_states"
)

# Build STATES dynamically from config: abbr → (zip_filename, fips_prefix)
# Pattern: 2022-{abbr_lower}-local-precinct-general.zip  (e.g. 2022-fl-local-precinct-general.zip)
STATES: dict[str, tuple[str, str]] = {
    abbr: (f"2022-{abbr.lower()}-local-precinct-general.zip", fips)
    for abbr, fips in _cfg.STATES.items()
}


def download_file(url: str, dest: Path, desc: str) -> None:
    """Download with progress bar. No-ops if cached."""
    if dest.exists():
        log.info("  Cached: %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", desc)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))


def load_medsl_csv(zip_path: Path) -> pd.DataFrame:
    """Unzip and read the MEDSL precinct CSV from the zip file.

    2022 repo typically uses .csv files inside the ZIP.
    Fall back to first non-directory entry if no .csv file found (defensive,
    consistent with 2024 fetcher behaviour).
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            # Fallback: single entry with no extension (matches 2024 repo format)
            csv_files = [n for n in zf.namelist() if not zf.getinfo(n).is_dir()]
        if not csv_files:
            raise FileNotFoundError(f"No data file in {zip_path}")
        log.info("  Reading %s from %s", csv_files[0], zip_path.name)
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)
    return df


def extract_governor_county(df: pd.DataFrame, state_abbr: str) -> pd.DataFrame:
    """
    Filter to governor general election, dedup modes, aggregate to county level.

    gov_total_2022 = total votes for ALL candidates (not just D+R), captured
    from the totalvotes column which MEDSL populates for all rows in a county.
    gov_dem_share_2022 = gov_dem_2022 / gov_total_2022.

    Returns DataFrame with county_fips, state_abbr, and vote columns.
    """
    # Filter to governor general (include all parties for total votes)
    gov = df[
        (df["office"].str.upper().str.contains("GOVERNOR", na=False)) &
        (df["stage"].str.lower() == "gen") &
        (df["writein"].fillna(False).astype(str).str.upper() != "TRUE")
    ].copy()

    if len(gov) == 0:
        log.warning("  No GOVERNOR/gen rows found for %s", state_abbr)
        return pd.DataFrame()

    log.info("  %s: %d governor precinct rows before mode dedup", state_abbr, len(gov))

    # Mode dedup: prefer TOTAL rows if they exist
    if "mode" in gov.columns:
        modes = gov["mode"].str.upper().unique()
        log.info("  Modes present: %s", list(modes))
        if "TOTAL" in modes:
            gov = gov[gov["mode"].str.upper() == "TOTAL"]
            log.info("  Using TOTAL mode rows only (%d rows)", len(gov))
        else:
            log.info("  No TOTAL mode — summing all modes")

    # Ensure county_fips is zero-padded 5-digit string
    # MEDSL sometimes reads county_fips as float (e.g. 13001.0) — strip the .0 first
    gov["county_fips"] = (
        gov["county_fips"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # Capture total votes (all candidates) from totalvotes column.
    # MEDSL reports the county total in every row for that county — use the
    # first value per county (all rows carry the same totalvotes for that county).
    if "totalvotes" in gov.columns:
        county_total = (
            gov.groupby("county_fips")["totalvotes"]
            .first()
            .reset_index()
            .rename(columns={"totalvotes": "gov_total_2022"})
        )
    else:
        # Fallback: sum all candidate votes as total
        county_total = (
            gov.groupby("county_fips")["votes"]
            .sum()
            .reset_index()
            .rename(columns={"votes": "gov_total_2022"})
        )
        log.warning("  %s: totalvotes column missing — using sum of all votes as total", state_abbr)

    # Aggregate D and R votes by county
    county_party = (
        gov.groupby(["county_fips", "party_simplified"])["votes"]
        .sum()
        .reset_index()
    )

    dem = county_party[county_party["party_simplified"].str.upper() == "DEMOCRAT"]
    rep = county_party[county_party["party_simplified"].str.upper() == "REPUBLICAN"]

    dem = dem[["county_fips", "votes"]].rename(columns={"votes": "gov_dem_2022"})
    rep = rep[["county_fips", "votes"]].rename(columns={"votes": "gov_rep_2022"})

    result = dem.merge(rep, on="county_fips", how="outer").fillna(0)
    result = result.merge(county_total, on="county_fips", how="left")
    # Fill missing totals with D+R sum (safety net)
    result["gov_total_2022"] = result["gov_total_2022"].fillna(
        result["gov_dem_2022"] + result["gov_rep_2022"]
    )
    result["gov_dem_share_2022"] = result["gov_dem_2022"] / result["gov_total_2022"].replace(
        0, float("nan")
    )
    result["state_abbr"] = state_abbr

    log.info(
        "  %s 2022 governor: %d counties, total dem=%s rep=%s",
        state_abbr,
        len(result),
        f"{result['gov_dem_2022'].sum():,.0f}",
        f"{result['gov_rep_2022'].sum():,.0f}",
    )
    return result


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for state_abbr, (filename, _fips) in STATES.items():
        url = f"{MEDSL_BASE}/{filename}"
        dest = RAW_DIR / filename

        log.info("=== %s 2022 ===", state_abbr)
        try:
            download_file(url, dest, f"MEDSL {state_abbr} 2022")
        except requests.HTTPError as e:
            log.warning("  Could not download %s: %s — skipping", state_abbr, e)
            continue

        df = load_medsl_csv(dest)
        county_df = extract_governor_county(df, state_abbr)
        if len(county_df) > 0:
            frames.append(county_df)

    if not frames:
        log.error("No data loaded — check download errors above")
        return

    combined = pd.concat(frames, ignore_index=True)
    out_path = OUTPUT_DIR / "medsl_county_2022_governor.parquet"
    combined.to_parquet(out_path, index=False)

    total_dem = combined["gov_dem_2022"].sum()
    total_rep = combined["gov_rep_2022"].sum()
    overall_dem_share = total_dem / (total_dem + total_rep)

    log.info(
        "Saved → %s | %d counties | overall dem share: %.1f%%",
        out_path, len(combined), overall_dem_share * 100,
    )

    # Summary by state
    print("\n── 2022 Governor results by state ──────────────────────")
    print(f"  {'State':<6}  {'Counties':>8}  {'Dem votes':>12}  {'Rep votes':>12}  {'Dem share':>10}")
    print("  " + "-" * 55)
    for state_abbr in combined["state_abbr"].unique():
        s = combined[combined["state_abbr"] == state_abbr]
        d = s["gov_dem_2022"].sum()
        r = s["gov_rep_2022"].sum()
        print(f"  {state_abbr:<6}  {len(s):>8}  {d:>12,.0f}  {r:>12,.0f}  {d/(d+r):.1%}")


if __name__ == "__main__":
    main()
