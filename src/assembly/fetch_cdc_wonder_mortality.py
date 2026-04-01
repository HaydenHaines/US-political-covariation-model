"""
Stage 1 data assembly: fetch CDC county-level mortality data.

Sources: data.cdc.gov Socrata (SODA) API — two datasets:
  - Drug overdose mortality: dataset gb4e-yj24 (VSRR Provisional County-Level
    Drug Overdose Death Counts) — 12-month rolling counts by county, 2020–present
  - COVID deaths by county: dataset kn79-hsxy (Provisional COVID-19 Death Counts
    by County and Race) — cumulative 2020–2023 deaths per county

Scope: ALL US counties (3,000+ counties across all 50 states + DC)

**Why mortality data?**
Deaths of despair (drug overdoses) and place-based mortality outcomes are among
the most powerful correlates of political realignment in rural America. Counties
where the mortality crisis is severe have moved toward Republicans strongly since
2016. COVID mortality captures both baseline health conditions and the political
response to the pandemic. Together these features characterize community health
stress, a critical dimension of electoral behavior.

**Dataset schemas (actual SODA column names as of 2026):**

Dataset gb4e-yj24 — VSRR Provisional County-Level Drug Overdose Death Counts:
  fips                   : 4-5 digit county FIPS (may need zero-padding)
  countyname             : County name
  st_abbrev              : State abbreviation
  year                   : 4-digit year string
  month                  : Month number
  provisional_drug_overdose : 12-month provisional drug overdose death count
  footnote               : Suppression flag (non-null = data suppressed or 1-9)

  Strategy: for each county, take the most recent 12-month window count as the
  drug overdose death proxy. Convert to a rate per 100K using population from
  the COVID dataset's total_death denominators (if available), or impute.

Dataset kn79-hsxy — Provisional COVID-19 Death Counts by County and Race (2020–2023):
  county_fips_code       : 4-5 digit county FIPS
  county_name            : County name
  state_name             : State name
  covid_death            : Cumulative COVID-19 deaths (may be absent if suppressed)
  total_death            : All-cause deaths over the pandemic period
  footnote               : Suppression flag

**Output**: data/raw/cdc_mortality.parquet
  One row per (county_fips, cause) where cause is:
    'drug_overdose'   : Drug overdose deaths (12-month count from VSRR)
    'covid'           : COVID-19 deaths (cumulative 2020–2023)
    'allcause_covid'  : All-cause deaths during COVID period
  Columns: county_fips, year, cause, deaths, population, death_rate, age_adjusted_rate

**Derived features** (computed in build_cdc_mortality_features.py):
  drug_overdose_rate     : Drug overdose deaths per 100K
  covid_death_rate       : COVID deaths per 100K (cumulative 2020–2023)
  excess_mortality_ratio : county rate / state median rate (relative distress)
  despair_death_rate     : drug_overdose_rate proxy
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "cdc_mortality.parquet"

# CDC Socrata (data.cdc.gov) SODA base URL
SODA_BASE = "https://data.cdc.gov/resource"

# Dataset IDs on data.cdc.gov
DATASET_DRUG_OVERDOSE = "gb4e-yj24"   # VSRR Provisional County-Level Drug Overdose Death Counts
DATASET_COVID_DEATHS = "kn79-hsxy"    # Provisional COVID-19 Deaths by County

# SODA page size (max 50,000; use 10,000 for politeness)
SODA_PAGE_SIZE = 10_000

# Polite delay between API requests (seconds)
REQUEST_DELAY = 1.0

# Drug OD years — use 2022-2023 to get stable recent counts
DRUG_OVERDOSE_YEARS = [2022, 2023]

# Cause codes in the output
CAUSE_DRUG_OVERDOSE = "drug_overdose"
CAUSE_COVID = "covid"
CAUSE_ALLCAUSE_COVID_PERIOD = "allcause_covid"


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def _build_soda_url(
    dataset_id: str,
    where: str,
    select: str,
    limit: int = SODA_PAGE_SIZE,
    offset: int = 0,
    order: str | None = None,
) -> str:
    """Build a CDC Socrata SODA API URL with query parameters.

    Args:
        dataset_id: CDC data.cdc.gov dataset identifier (e.g. 'gb4e-yj24').
        where: SODA $where clause (SQL-like filter expression).
        select: Comma-separated column names to retrieve.
        limit: Max rows per request.
        offset: Row offset for pagination.
        order: Optional $order clause for consistent pagination.

    Returns:
        Full URL string with properly encoded query parameters.
    """
    base = f"{SODA_BASE}/{dataset_id}.json"
    params: dict[str, str | int] = {
        "$where": where,
        "$select": select,
        "$limit": limit,
        "$offset": offset,
    }
    if order:
        params["$order"] = order

    query_parts = [f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()]
    return f"{base}?{'&'.join(query_parts)}"


def build_drug_overdose_url(offset: int = 0, limit: int = SODA_PAGE_SIZE) -> str:
    """Construct SODA URL for VSRR county drug overdose death counts (all US).

    Fetches from dataset gb4e-yj24 (VSRR Provisional County-Level Drug Overdose
    Death Counts). Filters to recent years to limit data volume. The dataset has
    monthly rows per county; we take the most recent available per county.

    Args:
        offset: Row offset for pagination.
        limit: Rows per page.

    Returns:
        Full SODA URL.
    """
    # Filter to recent years for efficiency; we only need the latest count per county
    year_list = ",".join(f"'{y}'" for y in DRUG_OVERDOSE_YEARS)
    where = f"year IN ({year_list})"
    select = "fips,countyname,st_abbrev,year,month,provisional_drug_overdose,footnote"
    return _build_soda_url(
        DATASET_DRUG_OVERDOSE,
        where=where,
        select=select,
        limit=limit,
        offset=offset,
        order="fips,year,month",
    )


def build_covid_deaths_url(offset: int = 0, limit: int = SODA_PAGE_SIZE) -> str:
    """Construct SODA URL for provisional COVID-19 deaths by county (all US).

    Fetches from dataset kn79-hsxy (Provisional COVID-19 Deaths by County).
    This dataset has one row per county (cumulative 2020–2023) so pagination
    is unlikely needed (~3,000 rows total).

    Args:
        offset: Row offset for pagination.
        limit: Rows per page.

    Returns:
        Full SODA URL.
    """
    # No state filter — national scope. county_fips_code IS NOT NULL equivalent:
    # filter to rows where county_fips_code has data (SODA: use length > 0)
    where = "county_fips_code > '0'"
    select = "county_fips_code,county_name,state_name,covid_death,total_death,footnote"
    return _build_soda_url(
        DATASET_COVID_DEATHS,
        where=where,
        select=select,
        limit=limit,
        offset=offset,
        order="county_fips_code",
    )


# ---------------------------------------------------------------------------
# Generic SODA paginator
# ---------------------------------------------------------------------------


def _fetch_soda_all_pages(
    url_builder: object,
    dataset_label: str,
) -> list[dict]:
    """Paginate through a SODA API endpoint until all rows are fetched.

    Keeps fetching pages until a response has fewer rows than SODA_PAGE_SIZE,
    indicating the last page. Applies REQUEST_DELAY between pages.

    Args:
        url_builder: Callable(offset, limit) → URL string.
        dataset_label: Human-readable label for logging.

    Returns:
        List of row dicts from the combined pages.
    """
    all_rows: list[dict] = []
    offset = 0

    while True:
        url = url_builder(offset=offset, limit=SODA_PAGE_SIZE)
        log.info("  [%s] Fetching rows %d–%d...", dataset_label, offset, offset + SODA_PAGE_SIZE - 1)

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            page: list[dict] = resp.json()
        except requests.RequestException as exc:
            log.warning("  [%s] Request failed at offset=%d: %s", dataset_label, offset, exc)
            break
        except Exception as exc:
            log.warning("  [%s] Parse error at offset=%d: %s", dataset_label, offset, exc)
            break

        all_rows.extend(page)
        log.info(
            "  [%s] Got %d rows (total so far: %d)",
            dataset_label, len(page), len(all_rows),
        )

        if len(page) < SODA_PAGE_SIZE:
            break

        offset += SODA_PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    return all_rows


# ---------------------------------------------------------------------------
# Drug overdose fetcher
# ---------------------------------------------------------------------------


# Output columns for all mortality cause DataFrames
_MORTALITY_OUTPUT_COLS = [
    "county_fips", "year", "cause", "deaths",
    "population", "death_rate", "age_adjusted_rate",
]

# Year assigned to cumulative COVID period (2020–2023 endpoint)
_COVID_PERIOD_YEAR = 2023


def _validate_and_filter_fips(df: pd.DataFrame, fips_col: str, dataset_label: str) -> pd.DataFrame:
    """Zero-pad FIPS, keep only valid 5-digit county codes, return filtered DataFrame.

    Excludes state aggregate rows where digits 3-5 are '000'.
    Returns an empty DataFrame if the FIPS column is absent.
    """
    if fips_col not in df.columns:
        log.warning("'%s' column missing from %s data.", fips_col, dataset_label)
        return pd.DataFrame()
    df[fips_col] = df[fips_col].astype(str).str.strip().str.zfill(5)
    fips_ok = df[fips_col].str.match(r"^\d{5}$")
    county_ok = df[fips_col].str[2:] != "000"
    result = df[fips_ok & county_ok].copy()
    log.info("After FIPS validation: %d rows", len(result))
    return result


def _drop_suppressed_rows(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """Remove rows where the footnote column is non-null (suppressed data).

    CDC datasets use a non-null footnote to indicate counts of 1-9 deaths
    (suppressed to protect privacy). These rows have no usable death count.
    """
    if "footnote" not in df.columns:
        return df
    n_before = len(df)
    suppressed = df["footnote"].notna() & (df["footnote"].astype(str).str.strip() != "")
    n_suppressed = suppressed.sum()
    if n_suppressed > 0:
        log.info("  Dropping %d suppressed %s rows (footnote non-null)", n_suppressed, dataset_label)
    result = df[~suppressed].copy()
    log.info("After suppression filter: %d → %d rows", n_before, len(result))
    return result


def _make_cause_rows(df: pd.DataFrame, deaths_col: str, cause: str) -> pd.DataFrame:
    """Build a single-cause output DataFrame with all required columns.

    Sets population, death_rate, and age_adjusted_rate to NaN — these are
    computed downstream in build_cdc_mortality_features.py.
    """
    out = df[["county_fips", deaths_col]].rename(columns={deaths_col: "deaths"}).copy()
    out["year"] = _COVID_PERIOD_YEAR
    out["cause"] = cause
    out["population"] = float("nan")
    out["death_rate"] = float("nan")
    out["age_adjusted_rate"] = float("nan")
    return out[_MORTALITY_OUTPUT_COLS]


# ---------------------------------------------------------------------------
# Drug overdose fetcher
# ---------------------------------------------------------------------------


def _select_latest_per_county(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent (year, month) row per county with a non-null count.

    VSRR data has monthly rows per county; we want the single most recent
    non-suppressed count as the overdose mortality proxy.
    """
    has_count = df["provisional_drug_overdose"].notna()
    df_valid = df[has_count].copy()
    if df_valid.empty:
        log.warning("No non-suppressed drug overdose counts found.")
        return pd.DataFrame()
    df_valid["year_month"] = df_valid["year"] * 100 + df_valid["month"]
    idx = df_valid.groupby("fips")["year_month"].idxmax()
    latest = df_valid.loc[idx].copy()
    log.info("After dedup to latest row per county: %d counties", len(latest))
    return latest


def fetch_drug_overdose() -> pd.DataFrame:
    """Fetch VSRR provisional county drug overdose death counts for all US counties.

    Uses the most recent non-suppressed 12-month count per county.
    Returns cause == 'drug_overdose'. Empty DataFrame on failure.
    """
    log.info("Fetching VSRR county drug overdose deaths (dataset %s)...", DATASET_DRUG_OVERDOSE)
    rows = _fetch_soda_all_pages(build_drug_overdose_url, "drug_overdose")
    if not rows:
        log.warning("No drug overdose data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Raw rows: %d, columns: %s", len(df), list(df.columns))

    df = _validate_and_filter_fips(df, "fips", "drug overdose")
    if df.empty:
        return pd.DataFrame()

    df["year"] = pd.to_numeric(df.get("year", pd.Series(dtype=str)), errors="coerce")
    df["month"] = pd.to_numeric(df.get("month", pd.Series(dtype=str)), errors="coerce")
    df = _drop_suppressed_rows(df, "drug OD")
    if df.empty:
        return pd.DataFrame()

    df["provisional_drug_overdose"] = pd.to_numeric(
        df.get("provisional_drug_overdose"), errors="coerce"
    )
    latest = _select_latest_per_county(df)
    if latest.empty:
        return pd.DataFrame()

    latest["county_fips"] = latest["fips"]
    latest["cause"] = CAUSE_DRUG_OVERDOSE
    latest["deaths"] = latest["provisional_drug_overdose"]
    latest["population"] = float("nan")
    latest["death_rate"] = float("nan")  # Rate computed in build_cdc_mortality_features.py
    latest["age_adjusted_rate"] = float("nan")
    return latest[_MORTALITY_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# COVID deaths fetcher
# ---------------------------------------------------------------------------


def fetch_covid_deaths() -> pd.DataFrame:
    """Fetch provisional COVID-19 deaths by county for all US counties.

    Dataset kn79-hsxy has one row per county (cumulative 2020–2023).
    Returns two rows per county: one for 'covid', one for 'allcause_covid'.
    Empty DataFrame on failure.
    """
    log.info("Fetching provisional COVID deaths (dataset %s)...", DATASET_COVID_DEATHS)
    rows = _fetch_soda_all_pages(build_covid_deaths_url, "covid_deaths")
    if not rows:
        log.warning("No COVID deaths data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Raw rows: %d, columns: %s", len(df), list(df.columns))

    df = _validate_and_filter_fips(df, "county_fips_code", "COVID deaths")
    if df.empty:
        return pd.DataFrame()

    df = _drop_suppressed_rows(df, "COVID")
    if df.empty:
        return pd.DataFrame()

    df["deaths_covid"] = pd.to_numeric(df.get("covid_death"), errors="coerce")
    df["deaths_all"] = pd.to_numeric(df.get("total_death"), errors="coerce")
    df["county_fips"] = df["county_fips_code"]

    combined = pd.concat(
        [_make_cause_rows(df, "deaths_covid", CAUSE_COVID),
         _make_cause_rows(df, "deaths_all", CAUSE_ALLCAUSE_COVID_PERIOD)],
        ignore_index=True,
    )
    log.info("COVID deaths: %d county-cause rows", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all numeric output columns to float, silencing bad values."""
    for col in ["year", "deaths", "population", "death_rate", "age_adjusted_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _log_mortality_summary(combined: pd.DataFrame) -> None:
    """Log per-cause row/county/NaN-death breakdown."""
    causes = sorted(combined["cause"].unique())
    log.info(
        "\nSummary: %d rows | %d counties | %d cause codes: %s",
        len(combined), combined["county_fips"].nunique(), len(causes), causes,
    )
    for cause in causes:
        sub = combined[combined["cause"] == cause]
        log.info(
            "  cause=%-25s  %d rows, %d counties, %d NaN deaths",
            cause, len(sub), sub["county_fips"].nunique(), sub["deaths"].isna().sum(),
        )


def _fetch_all_sources() -> list[pd.DataFrame]:
    """Fetch drug overdose and COVID mortality, returning non-empty DataFrames."""
    frames: list[pd.DataFrame] = []

    drug_df = fetch_drug_overdose()
    if not drug_df.empty:
        log.info("Drug overdose: %d rows (%d counties)", len(drug_df), drug_df["county_fips"].nunique())
        frames.append(drug_df)
    else:
        log.warning("No drug overdose data retrieved.")

    time.sleep(REQUEST_DELAY)

    covid_df = fetch_covid_deaths()
    if not covid_df.empty:
        log.info("COVID deaths: %d rows (%d counties)", len(covid_df), covid_df["county_fips"].nunique())
        frames.append(covid_df)
    else:
        log.warning("No COVID deaths data retrieved.")

    return frames


def main() -> None:
    """Fetch CDC mortality data for all US counties and save combined parquet."""
    log.info("=" * 60)
    log.info("Fetching CDC county-level mortality data (national — all US counties)")
    log.info("=" * 60)

    frames = _fetch_all_sources()
    if not frames:
        log.error("No mortality data retrieved from any source. Aborting.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Final FIPS guard — should be clean from individual fetchers
    fips_ok = combined["county_fips"].str.match(r"^\d{5}$", na=False)
    n_bad = (~fips_ok).sum()
    if n_bad > 0:
        log.warning("Dropping %d rows with non-5-digit FIPS", n_bad)
        combined = combined[fips_ok]

    combined = _coerce_numeric_columns(combined)
    _log_mortality_summary(combined)

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(RAW_OUTPUT_PATH, index=False)
    log.info("\nSaved → %s  (%d rows × %d cols)", RAW_OUTPUT_PATH, len(combined), len(combined.columns))


if __name__ == "__main__":
    main()
