"""Fetch BEA county-level economic data and produce county-level features.

Sources:
  CAINC1 — County Personal Income Summary
    https://apps.bea.gov/regional/zip/CAINC1.zip
    1969–2024, all US counties

  CAGDP1 — County GDP Summary
    https://apps.bea.gov/regional/zip/CAGDP1.zip
    2001–2024, all US counties

CAINC1 line codes used:
  LineCode 2: Population (persons)
  LineCode 3: Per capita personal income (dollars)

CAGDP1 line codes used:
  LineCode 3: Current-dollar GDP (thousands of current dollars)

Features produced (most recent year available, targeting 2023 or 2024):
  pci           — per capita personal income (dollars)
  pci_growth    — 3-year relative change in PCI: (pci_t - pci_{t-3}) / |pci_{t-3}|
  gdp_per_capita — current-dollar GDP per person (thousands of dollars)
  gdp_growth     — 3-year relative change in GDP: (gdp_t - gdp_{t-3}) / |gdp_{t-3}|

FIPS format:
  BEA encodes GeoFIPS as quoted 5-digit strings (e.g. ' "01001"').
  State-level aggregates use XX000 format and are excluded.
  Output county_fips is a plain unquoted 5-digit zero-padded string.

Cache:
  Raw ZIPs stored at data/raw/bea/CAINC1.zip and data/raw/bea/CAGDP1.zip.
  Downloads are skipped if cache files exist (use --force-refresh to re-download).

Output:
  data/assembled/county_bea_features.parquet
  Columns: county_fips, pci, pci_growth, gdp_per_capita, gdp_growth
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bea"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

CAINC1_URL = "https://apps.bea.gov/regional/zip/CAINC1.zip"
CAGDP1_URL = "https://apps.bea.gov/regional/zip/CAGDP1.zip"

CAINC1_ZIP_NAME = "CAINC1.zip"
CAGDP1_ZIP_NAME = "CAGDP1.zip"

# Most recent year to target; falls back one year at a time if data is absent
PRIMARY_YEAR = 2024
FALLBACK_YEARS = (2023, 2022)

# Number of years back for growth rate computation
GROWTH_LOOKBACK = 3

# CAINC1 line codes
CAINC1_LINE_POPULATION = 2       # Population (persons)
CAINC1_LINE_PCI = 3              # Per capita personal income (dollars)

# CAGDP1 line codes
CAGDP1_LINE_CURRENT_GDP = 3      # Current-dollar GDP (thousands of current dollars)

# BEA trailer lines that must be stripped before CSV parsing
_BEA_TRAILER_PREFIXES = ('"Note', '"CAINC', '"CAGDP', '"Last', '"U.S.')


# ── Download helpers ───────────────────────────────────────────────────────────


def download_zip(url: str, dest: Path, force_refresh: bool = False) -> Path:
    """Download a BEA bulk ZIP if not already cached.

    Args:
        url: URL of the ZIP file.
        dest: Local destination path (parent directory must exist).
        force_refresh: Re-download even if dest already exists.

    Returns:
        Path to the local ZIP file.
    """
    if dest.exists() and not force_refresh:
        log.info("Using cached ZIP: %s", dest)
        return dest

    log.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()

    with dest.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=65536):
            fh.write(chunk)

    log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# ── CSV parsing ────────────────────────────────────────────────────────────────


def _strip_bea_trailer(raw: str) -> str:
    """Remove BEA footer lines that appear after the data rows.

    BEA CSVs end with several quoted informational lines (Note, dataset name,
    last-updated date, attribution) that are not valid CSV rows. Strip them
    so pandas can parse cleanly.
    """
    clean_lines = [
        line for line in raw.splitlines()
        if not any(line.strip().startswith(prefix) for prefix in _BEA_TRAILER_PREFIXES)
        and line.strip()
    ]
    return "\n".join(clean_lines)


def _clean_fips(raw_fips: pd.Series) -> pd.Series:
    """Normalize BEA GeoFIPS column to plain 5-digit strings.

    BEA encodes FIPS as ' "01001"' (leading space, quoted). Strip whitespace
    and quotes, then zero-pad to 5 digits.
    """
    return raw_fips.astype(str).str.strip().str.strip('"').str.zfill(5)


def _parse_bea_csv(raw_bytes: bytes, low_memory: bool = False) -> pd.DataFrame:
    """Parse a single BEA wide-format CSV from raw bytes.

    The CSV uses wide format: fixed metadata columns followed by year columns.
    GeoFIPS values include both state aggregates (XX000) and counties (XXXXX
    where last 3 digits != 000). All values in year columns are numeric but
    may appear as strings with commas or suppression codes.

    Returns a DataFrame with:
      - county_fips: cleaned 5-digit FIPS string
      - GeoName: county name as reported by BEA
      - LineCode: integer line code
      - year columns: float values (NaN for suppressed/missing)
    """
    text = raw_bytes.decode("latin-1")
    text = _strip_bea_trailer(text)

    df = pd.read_csv(
        io.StringIO(text),
        dtype=str,        # read everything as string to avoid mixed-type warnings
        low_memory=low_memory,
    )

    df["county_fips"] = _clean_fips(df["GeoFIPS"])
    df["LineCode"] = pd.to_numeric(df["LineCode"], errors="coerce")

    # Exclude state-level aggregates (last 3 digits of FIPS = "000")
    df = df[df["county_fips"].str[2:] != "000"].copy()

    # Validate: keep only 5-digit numeric FIPS (exclude MSAs, metro areas, etc.)
    valid_fips = df["county_fips"].str.match(r"^\d{5}$")
    if (~valid_fips).any():
        log.debug(
            "Dropping %d rows with non-county FIPS codes", (~valid_fips).sum()
        )
    df = df[valid_fips].copy()

    # Convert year columns to float (comma-formatted numbers, suppression codes → NaN)
    year_cols = [c for c in df.columns if c.isdigit()]
    for col in year_cols:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        suppressed = cleaned.isin(["(D)", "(NA)", "(L)", "(S)", "(X)", "--", "", "nan"])
        df[col] = pd.to_numeric(cleaned.where(~suppressed, other=np.nan), errors="coerce")

    return df


def read_zip_state_files(zip_path: Path, prefix: str) -> pd.DataFrame:
    """Read and concatenate all per-state CSV files from a BEA ZIP.

    Each state has its own CSV file (e.g. CAINC1_AL_1969_2024.csv). The
    ALL_AREAS file is skipped because it contains suppressed values and mixed
    types that are harder to handle reliably.

    Args:
        zip_path: Path to the local BEA ZIP file.
        prefix: Dataset prefix used to identify per-state files (e.g. "CAINC1").

    Returns:
        Concatenated DataFrame of all state files.
    """
    frames: list[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as zf:
        state_files = [
            name for name in zf.namelist()
            if name.startswith(f"{prefix}_") and "__" not in name and name.endswith(".csv")
        ]
        log.info("Reading %d state files from %s", len(state_files), zip_path.name)

        for fname in sorted(state_files):
            with zf.open(fname) as fh:
                raw_bytes = fh.read()
            try:
                df = _parse_bea_csv(raw_bytes)
                frames.append(df)
            except Exception as exc:
                log.warning("Failed to parse %s: %s", fname, exc)

    if not frames:
        raise ValueError(f"No state CSV files parsed from {zip_path}")

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined %s: %d rows, %d counties", prefix, len(combined), combined["county_fips"].nunique())
    return combined


# ── Feature computation ────────────────────────────────────────────────────────


def _pick_year(df: pd.DataFrame, primary: int, fallbacks: tuple[int, ...]) -> int:
    """Choose the most recent year column present and non-null in df.

    Tries primary year first, then each fallback in order. Raises ValueError
    if no suitable year is found.
    """
    candidates = [primary, *fallbacks]
    year_cols = {int(c) for c in df.columns if c.isdigit()}

    for year in candidates:
        col = str(year)
        if year in year_cols and df[col].notna().any():
            return year

    raise ValueError(
        f"None of the candidate years {candidates} have data in the DataFrame. "
        f"Available year columns: {sorted(year_cols)}"
    )


def _compute_growth(
    series_t: pd.Series,
    series_t_minus_n: pd.Series,
) -> pd.Series:
    """Compute relative growth: (t - t_minus_n) / |t_minus_n|.

    Returns NaN where either value is missing or t_minus_n is zero.
    """
    result = pd.Series(np.nan, index=series_t.index)
    valid = series_t.notna() & series_t_minus_n.notna() & (series_t_minus_n != 0)
    result[valid] = (
        (series_t[valid] - series_t_minus_n[valid]) / series_t_minus_n[valid].abs()
    )
    return result


def build_pci_features(cainc1_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per capita income and 3-year growth from CAINC1.

    Uses:
      LineCode 3 — Per capita personal income (dollars)

    Returns DataFrame with columns: county_fips, pci, pci_growth
    """
    pci_df = cainc1_df[cainc1_df["LineCode"] == CAINC1_LINE_PCI].copy()

    year_t = _pick_year(pci_df, PRIMARY_YEAR, FALLBACK_YEARS)
    year_t_minus_n = year_t - GROWTH_LOOKBACK

    col_t = str(year_t)
    col_prior = str(year_t_minus_n)

    log.info("PCI features: year=%d, growth period=%d→%d", year_t, year_t_minus_n, year_t)

    if col_prior not in pci_df.columns:
        log.warning(
            "Growth base year %d not available in CAINC1 — pci_growth will be NaN",
            year_t_minus_n,
        )
        pci_df[col_prior] = np.nan

    result = pci_df[["county_fips", col_t, col_prior]].copy()
    result = result.rename(columns={col_t: "pci"})
    result["pci_growth"] = _compute_growth(result["pci"], result[col_prior])
    result = result.drop(columns=[col_prior])

    # Deduplicate (should not occur with per-state files, but guard defensively)
    result = result.drop_duplicates(subset=["county_fips"]).reset_index(drop=True)
    log.info("PCI features: %d counties", len(result))
    return result


def build_gdp_features(
    cagdp1_df: pd.DataFrame,
    cainc1_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute GDP per capita and 3-year GDP growth from CAGDP1 + CAINC1.

    GDP per capita = current-dollar GDP (thousands) / population (persons)
    → result is in thousands of dollars per person.

    Uses:
      CAGDP1 LineCode 3 — Current-dollar GDP (thousands of current dollars)
      CAINC1 LineCode 2 — Population (persons)

    Returns DataFrame with columns: county_fips, gdp_per_capita, gdp_growth
    """
    gdp_df = cagdp1_df[cagdp1_df["LineCode"] == CAGDP1_LINE_CURRENT_GDP].copy()
    pop_df = cainc1_df[cainc1_df["LineCode"] == CAINC1_LINE_POPULATION].copy()

    year_t = _pick_year(gdp_df, PRIMARY_YEAR, FALLBACK_YEARS)
    year_t_minus_n = year_t - GROWTH_LOOKBACK

    col_t = str(year_t)
    col_prior = str(year_t_minus_n)

    log.info("GDP features: year=%d, growth period=%d→%d", year_t, year_t_minus_n, year_t)

    if col_prior not in gdp_df.columns:
        log.warning(
            "Growth base year %d not available in CAGDP1 — gdp_growth will be NaN",
            year_t_minus_n,
        )
        gdp_df[col_prior] = np.nan

    # GDP in current dollars (thousands)
    gdp = gdp_df[["county_fips", col_t, col_prior]].copy()
    gdp = gdp.rename(columns={col_t: "gdp_current", col_prior: "gdp_prior"})

    # Population from CAINC1 for the same year
    pop_col = str(year_t)
    if pop_col not in pop_df.columns:
        log.warning("Population year %d not available — gdp_per_capita will be NaN", year_t)
        pop_df[pop_col] = np.nan

    pop = pop_df[["county_fips", pop_col]].copy()
    pop = pop.rename(columns={pop_col: "population"})

    # Merge GDP with population
    result = gdp.merge(pop, on="county_fips", how="left")

    # GDP per capita: current-dollar GDP (thousands) / population → thousands $/person
    valid_pop = result["population"].notna() & (result["population"] > 0)
    result["gdp_per_capita"] = np.nan
    result.loc[valid_pop, "gdp_per_capita"] = (
        result.loc[valid_pop, "gdp_current"] / result.loc[valid_pop, "population"]
    )

    # GDP growth
    result["gdp_growth"] = _compute_growth(result["gdp_current"], result["gdp_prior"])

    result = result[["county_fips", "gdp_per_capita", "gdp_growth"]]
    result = result.drop_duplicates(subset=["county_fips"]).reset_index(drop=True)
    log.info("GDP features: %d counties", len(result))
    return result


# ── Main assembly ──────────────────────────────────────────────────────────────


def fetch_bea_features(force_refresh: bool = False) -> pd.DataFrame:
    """Download BEA CAINC1 and CAGDP1 data and assemble county-level features.

    Downloads are cached locally. Pass force_refresh=True to re-download.

    Returns:
        DataFrame with columns:
          county_fips   — 5-digit string
          pci           — per capita personal income (dollars)
          pci_growth    — 3-year relative change in PCI (fraction)
          gdp_per_capita — current-dollar GDP per person (thousands of dollars)
          gdp_growth    — 3-year relative change in current-dollar GDP (fraction)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cainc1_zip = download_zip(CAINC1_URL, RAW_DIR / CAINC1_ZIP_NAME, force_refresh)
    cagdp1_zip = download_zip(CAGDP1_URL, RAW_DIR / CAGDP1_ZIP_NAME, force_refresh)

    log.info("Parsing CAINC1...")
    cainc1_df = read_zip_state_files(cainc1_zip, "CAINC1")

    log.info("Parsing CAGDP1...")
    cagdp1_df = read_zip_state_files(cagdp1_zip, "CAGDP1")

    pci_features = build_pci_features(cainc1_df)
    gdp_features = build_gdp_features(cagdp1_df, cainc1_df)

    # Outer join: some counties may appear in CAINC1 but not CAGDP1 (very small)
    result = pci_features.merge(gdp_features, on="county_fips", how="outer")
    result = result.sort_values("county_fips").reset_index(drop=True)

    log.info(
        "BEA features assembled: %d counties | "
        "pci valid=%d | pci_growth valid=%d | gdp_pc valid=%d | gdp_growth valid=%d",
        len(result),
        result["pci"].notna().sum(),
        result["pci_growth"].notna().sum(),
        result["gdp_per_capita"].notna().sum(),
        result["gdp_growth"].notna().sum(),
    )
    return result


def main() -> None:
    """Download BEA data and write county_bea_features.parquet."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch BEA county economic features")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download ZIPs even if cached",
    )
    args = parser.parse_args()

    df = fetch_bea_features(force_refresh=args.force_refresh)

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSEMBLED_DIR / "county_bea_features.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved → %s (%d counties)", out, len(df))

    for col in ["pci", "pci_growth", "gdp_per_capita", "gdp_growth"]:
        if col in df.columns:
            valid = df[col].dropna()
            log.info(
                "  %-20s  n=%d  mean=%.3g  min=%.3g  max=%.3g",
                col,
                len(valid),
                valid.mean(),
                valid.min(),
                valid.max(),
            )


if __name__ == "__main__":
    main()
