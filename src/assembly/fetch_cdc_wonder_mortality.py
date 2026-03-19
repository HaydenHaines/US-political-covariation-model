"""
Stage 1 data assembly: fetch CDC county-level mortality data.

Sources: data.cdc.gov Socrata (SODA) API — multiple datasets:
  - Drug overdose mortality: dataset jx6g-fdh6 (NCHS Drug Poisoning Mortality by County)
  - All-cause age-adjusted mortality: dataset bi63-dtpu (NCHS Age-adjusted Death Rates)
  - COVID deaths by county: dataset kn79-hsxy (Provisional COVID-19 Deaths by County)

Scope: FL (FIPS 12), GA (FIPS 13), AL (FIPS 01) — 293 counties total
Temporal: 2018–2023 for drug overdose and all-cause; 2020–2023 for COVID

**Why mortality data?**
Deaths of despair (drug overdoses, alcohol, suicide) and place-based mortality
outcomes are among the most powerful correlates of political realignment in rural
America. Counties where the mortality crisis is severe have moved toward Republicans
strongly since 2016. All-cause and COVID mortality capture both baseline health
conditions and the political response to the pandemic. Together these features
characterize community health stress, a critical dimension of electoral behavior.

**CDC data.cdc.gov SODA API design:**
All three datasets are hosted on data.cdc.gov and accessible via the Socrata Open
Data API (SODA). No API key is required for public datasets but rate limits apply.

URL pattern:
  https://data.cdc.gov/resource/{dataset_id}.json?{params}

Key SODA parameters:
  $where  : SQL-like filter (e.g., "fips like '12%' OR fips like '13%'")
  $select : Comma-separated column names to return
  $limit  : Max rows per request (max 50,000; default 1,000)
  $offset : Row offset for pagination

**Dataset schemas:**

Dataset jx6g-fdh6 — NCHS Drug Poisoning Mortality by County:
  fips             : 5-digit county FIPS
  county           : County name
  state            : State name
  year             : 4-digit year string (2002–2021, updated annually)
  deaths           : Estimated deaths (may be suppressed for small counts)
  population       : County population
  model_based_death_rate : Modeled rate per 100K (used instead of raw deaths for stability)
  age_adjusted_rate: (if available) Age-adjusted rate

Dataset bi63-dtpu — NCHS Age-adjusted Death Rates (Leading Causes):
  year             : Data year
  state            : State name
  state_fips_code  : 2-digit state FIPS (no county-level in this dataset)
  cause_name       : Cause of death (e.g., 'All Causes', 'Heart Disease', 'Cancer')
  age_adjusted_death_rate : Rate per 100K age-adjusted

  NOTE: This dataset is state-level only. County-level age-adjusted all-cause
  mortality comes from jx6g-fdh6 (drug overdose dataset only) or from the
  CDC WONDER compressed mortality files. For county-level all-cause data we
  use the CDC PLACES dataset or derive from the drug overdose dataset's
  population column.

Dataset kn79-hsxy — Provisional COVID-19 Deaths by County (2020–present):
  fips_code        : 5-digit county FIPS
  county_name      : County name
  state            : State name
  start_date       : Period start date
  end_date         : Period end date
  deaths_covid19   : COVID-19 deaths in period
  deaths_all_cause : All-cause deaths in period (useful for excess mortality)
  footnote         : Suppression flag (non-null = data suppressed)

**Output**: data/raw/cdc_mortality.parquet
  One row per (county_fips, year, cause) where cause is one of:
    'drug_overdose'   : Drug poisoning deaths, model-based rate per 100K
    'covid'           : COVID-19 deaths per 100K
    'allcause_covid'  : All-cause deaths per 100K during COVID period (from COVID dataset)
  Columns: county_fips, year, cause, deaths, population, death_rate, age_adjusted_rate

**Derived features** (computed in build_cdc_mortality_features.py):
  drug_overdose_rate     : Drug poisoning model-based rate per 100K
  covid_death_rate       : COVID deaths per 100K (2020–2023 cumulative)
  allcause_age_adj_rate  : All-cause age-adjusted mortality rate per 100K
  excess_mortality_ratio : county rate / state median rate (relative distress)
  despair_death_rate     : drug_overdose_rate (combined deaths-of-despair proxy)
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

# Target states: abbreviation → FIPS prefix
STATES = {"AL": "01", "FL": "12", "GA": "13"}

# Set of 2-digit state FIPS prefixes (for filtering)
TARGET_STATE_FIPS = frozenset(STATES.values())

# CDC Socrata (data.cdc.gov) SODA base URL
SODA_BASE = "https://data.cdc.gov/resource"

# Dataset IDs on data.cdc.gov
DATASET_DRUG_OVERDOSE = "jx6g-fdh6"   # NCHS Drug Poisoning Mortality by County
DATASET_COVID_DEATHS = "kn79-hsxy"    # Provisional COVID-19 Deaths by County

# SODA page size (max 50,000; use 10,000 for politeness)
SODA_PAGE_SIZE = 10_000

# Polite delay between API requests (seconds)
REQUEST_DELAY = 1.0

# Years to fetch for drug overdose data (API covers 2002–2021 typically)
DRUG_OVERDOSE_YEARS = list(range(2018, 2022))  # 2018–2021 (2022 not yet available)

# COVID years (Provisional dataset covers 2020–present)
COVID_YEARS = [2020, 2021, 2022, 2023]

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
        dataset_id: CDC data.cdc.gov dataset identifier (e.g. 'jx6g-fdh6').
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
    """Construct SODA URL for drug poisoning mortality data (FL/GA/AL counties).

    Fetches from dataset jx6g-fdh6 (NCHS Drug Poisoning Mortality by County).
    Filters by state (FL/GA/AL) using FIPS prefixes.

    Args:
        offset: Row offset for pagination.
        limit: Rows per page.

    Returns:
        Full SODA URL.
    """
    where = "state IN ('Florida', 'Georgia', 'Alabama')"
    select = "fips,county,state,year,deaths,population,model_based_death_rate,age_adjusted_rate"
    return _build_soda_url(
        DATASET_DRUG_OVERDOSE,
        where=where,
        select=select,
        limit=limit,
        offset=offset,
        order="fips,year",
    )


def build_covid_deaths_url(offset: int = 0, limit: int = SODA_PAGE_SIZE) -> str:
    """Construct SODA URL for provisional COVID-19 deaths by county (FL/GA/AL).

    Fetches from dataset kn79-hsxy (Provisional COVID-19 Deaths by County).
    Filters to FL, GA, AL.

    Args:
        offset: Row offset for pagination.
        limit: Rows per page.

    Returns:
        Full SODA URL.
    """
    where = "state IN ('Florida', 'Georgia', 'Alabama')"
    select = "fips_code,county_name,state,start_date,end_date,deaths_covid19,deaths_all_cause,footnote"
    return _build_soda_url(
        DATASET_COVID_DEATHS,
        where=where,
        select=select,
        limit=limit,
        offset=offset,
        order="fips_code,start_date",
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


def fetch_drug_overdose() -> pd.DataFrame:
    """Fetch NCHS drug poisoning mortality data for FL/GA/AL counties.

    Downloads the NCHS Drug Poisoning Mortality dataset (jx6g-fdh6) from
    data.cdc.gov. The dataset contains model-based county-level estimates of
    drug overdose death rates. Model-based rates are used because raw county
    counts are suppressed for small populations — the modeled estimates
    provide stable estimates for all counties.

    Returns:
        DataFrame with columns:
            county_fips, year, cause, deaths, population,
            death_rate, age_adjusted_rate
        where cause == 'drug_overdose'. Empty DataFrame on failure.
    """
    log.info("Fetching drug overdose mortality (dataset %s)...", DATASET_DRUG_OVERDOSE)
    rows = _fetch_soda_all_pages(build_drug_overdose_url, "drug_overdose")

    if not rows:
        log.warning("No drug overdose data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Raw rows: %d, columns: %s", len(df), list(df.columns))

    # Normalize FIPS: must be 5-digit zero-padded string
    if "fips" not in df.columns:
        log.warning("'fips' column missing from drug overdose data.")
        return pd.DataFrame()

    df["fips"] = df["fips"].astype(str).str.strip().str.zfill(5)

    # Filter to target states
    state_ok = df["fips"].str[:2].isin(TARGET_STATE_FIPS) & df["fips"].str.match(r"^\d{5}$")
    county_ok = df["fips"].str[2:] != "000"
    df = df[state_ok & county_ok].copy()
    log.info("After state filter: %d rows", len(df))

    if df.empty:
        return pd.DataFrame()

    # Parse year
    df["year"] = pd.to_numeric(df.get("year", pd.Series(dtype=str)), errors="coerce")

    # Filter to target years
    df = df[df["year"].isin(DRUG_OVERDOSE_YEARS)].copy()
    log.info("After year filter (%s): %d rows", DRUG_OVERDOSE_YEARS, len(df))

    # Coerce numeric columns
    df["deaths"] = pd.to_numeric(df.get("deaths"), errors="coerce")
    df["population"] = pd.to_numeric(df.get("population"), errors="coerce")
    df["death_rate"] = pd.to_numeric(df.get("model_based_death_rate"), errors="coerce")
    df["age_adjusted_rate"] = pd.to_numeric(df.get("age_adjusted_rate"), errors="coerce")

    df["county_fips"] = df["fips"]
    df["cause"] = CAUSE_DRUG_OVERDOSE

    output_cols = ["county_fips", "year", "cause", "deaths", "population", "death_rate", "age_adjusted_rate"]
    return df[output_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# COVID deaths fetcher
# ---------------------------------------------------------------------------


def _extract_year_from_date(date_series: pd.Series) -> pd.Series:
    """Extract 4-digit year from a date string Series (ISO 8601 format).

    Args:
        date_series: Series of date strings (e.g. '2020-01-01T00:00:00.000').

    Returns:
        Series of integer years.
    """
    parsed = pd.to_datetime(date_series, errors="coerce")
    return parsed.dt.year


def fetch_covid_deaths() -> pd.DataFrame:
    """Fetch provisional COVID-19 deaths by county for FL/GA/AL.

    Downloads the CDC Provisional COVID-19 Deaths dataset (kn79-hsxy) from
    data.cdc.gov. This dataset provides cumulative COVID-19 and all-cause
    deaths by county for rolling time periods.

    Strategy: aggregate all records per county to get cumulative COVID deaths
    and all-cause deaths across the pandemic period. Then derive a per-100K
    rate using the maximum population available in the drug overdose dataset,
    or directly from deaths/population if population is provided.

    Returns:
        DataFrame with columns:
            county_fips, year, cause, deaths, population,
            death_rate, age_adjusted_rate
        Two rows per county: one for 'covid', one for 'allcause_covid'.
        Empty DataFrame on failure.
    """
    log.info("Fetching provisional COVID deaths (dataset %s)...", DATASET_COVID_DEATHS)
    rows = _fetch_soda_all_pages(build_covid_deaths_url, "covid_deaths")

    if not rows:
        log.warning("No COVID deaths data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Raw rows: %d, columns: %s", len(df), list(df.columns))

    # Normalize FIPS
    fips_col = "fips_code"
    if fips_col not in df.columns:
        log.warning("'fips_code' column missing from COVID deaths data.")
        return pd.DataFrame()

    df[fips_col] = df[fips_col].astype(str).str.strip().str.zfill(5)

    # Filter to target states (exclude state/US aggregate rows)
    state_ok = df[fips_col].str[:2].isin(TARGET_STATE_FIPS) & df[fips_col].str.match(r"^\d{5}$")
    county_ok = df[fips_col].str[2:] != "000"
    df = df[state_ok & county_ok].copy()
    log.info("After state filter: %d rows", len(df))

    if df.empty:
        return pd.DataFrame()

    # Drop suppressed rows (footnote is non-null when data is suppressed)
    if "footnote" in df.columns:
        n_before = len(df)
        suppressed = df["footnote"].notna() & (df["footnote"].astype(str).str.strip() != "")
        n_suppressed = suppressed.sum()
        if n_suppressed > 0:
            log.info("  Dropping %d suppressed COVID rows (footnote non-null)", n_suppressed)
            df = df[~suppressed].copy()
        log.info("After suppression filter: %d → %d rows", n_before, len(df))

    # Extract year from start_date
    df["year"] = _extract_year_from_date(df.get("start_date", pd.Series(dtype=str)))

    # Filter to COVID years
    df = df[df["year"].isin(COVID_YEARS)].copy()
    log.info("After year filter (%s): %d rows", COVID_YEARS, len(df))

    if df.empty:
        return pd.DataFrame()

    # Coerce death counts
    df["deaths_covid19"] = pd.to_numeric(df.get("deaths_covid19"), errors="coerce")
    df["deaths_all_cause"] = pd.to_numeric(df.get("deaths_all_cause"), errors="coerce")
    df["county_fips"] = df[fips_col]

    # Aggregate per county per year: sum deaths across all reporting periods
    # (the dataset has multiple overlapping periods per county/year)
    covid_agg = (
        df.groupby(["county_fips", "year"])
        .agg(
            deaths_covid=("deaths_covid19", "sum"),
            deaths_all=("deaths_all_cause", "sum"),
        )
        .reset_index()
    )

    # Produce two cause rows per county-year
    covid_rows = covid_agg.rename(columns={"deaths_covid": "deaths"}).copy()
    covid_rows["cause"] = CAUSE_COVID
    covid_rows["population"] = float("nan")
    covid_rows["death_rate"] = float("nan")  # Computed in build_cdc_mortality_features.py
    covid_rows["age_adjusted_rate"] = float("nan")

    allcause_rows = covid_agg.rename(columns={"deaths_all": "deaths"}).copy()
    allcause_rows["cause"] = CAUSE_ALLCAUSE_COVID_PERIOD
    allcause_rows["population"] = float("nan")
    allcause_rows["death_rate"] = float("nan")
    allcause_rows["age_adjusted_rate"] = float("nan")

    output_cols = ["county_fips", "year", "cause", "deaths", "population", "death_rate", "age_adjusted_rate"]

    combined = pd.concat(
        [covid_rows[output_cols], allcause_rows[output_cols]],
        ignore_index=True,
    )
    log.info("COVID deaths: %d county-year-cause rows", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch CDC mortality data for FL/GA/AL counties and save combined parquet.

    Fetches:
      1. Drug overdose mortality (NCHS, 2018–2021, model-based rates)
      2. COVID-19 deaths by county (Provisional, 2020–2023)

    Combines into a single edge-list DataFrame (county_fips × year × cause)
    and saves to data/raw/cdc_mortality.parquet.
    """
    log.info("=" * 60)
    log.info("Fetching CDC county-level mortality data (FL, GA, AL)")
    log.info("Target states: %s", list(STATES.keys()))
    log.info("=" * 60)

    frames: list[pd.DataFrame] = []

    # 1. Drug overdose mortality
    drug_df = fetch_drug_overdose()
    if not drug_df.empty:
        log.info("Drug overdose: %d rows", len(drug_df))
        frames.append(drug_df)
    else:
        log.warning("No drug overdose data retrieved.")

    time.sleep(REQUEST_DELAY)

    # 2. COVID deaths
    covid_df = fetch_covid_deaths()
    if not covid_df.empty:
        log.info("COVID deaths: %d rows", len(covid_df))
        frames.append(covid_df)
    else:
        log.warning("No COVID deaths data retrieved.")

    if not frames:
        log.error("No mortality data retrieved from any source. Aborting.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Validate FIPS format
    fips_ok = combined["county_fips"].str.match(r"^\d{5}$", na=False)
    n_bad = (~fips_ok).sum()
    if n_bad > 0:
        log.warning("Dropping %d rows with non-5-digit FIPS", n_bad)
        combined = combined[fips_ok]

    # Ensure types
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    combined["deaths"] = pd.to_numeric(combined["deaths"], errors="coerce")
    combined["population"] = pd.to_numeric(combined["population"], errors="coerce")
    combined["death_rate"] = pd.to_numeric(combined["death_rate"], errors="coerce")
    combined["age_adjusted_rate"] = pd.to_numeric(combined["age_adjusted_rate"], errors="coerce")

    # Summary
    n_rows = len(combined)
    n_counties = combined["county_fips"].nunique()
    n_causes = combined["cause"].nunique()
    causes = sorted(combined["cause"].unique())

    log.info(
        "\nSummary: %d rows | %d counties | %d cause codes: %s",
        n_rows, n_counties, n_causes, causes,
    )
    state_counts = combined["county_fips"].str[:2].value_counts().to_dict()
    fips_to_abbr = {"01": "AL", "12": "FL", "13": "GA"}
    for fips_pref, count in sorted(state_counts.items()):
        abbr = fips_to_abbr.get(fips_pref, fips_pref)
        log.info("  %s: %d rows", abbr, count)

    for cause in causes:
        sub = combined[combined["cause"] == cause]
        n_nan_rate = sub["death_rate"].isna().sum()
        log.info(
            "  cause=%-25s  %d rows, %d counties, %d NaN death_rate",
            cause, len(sub), sub["county_fips"].nunique(), n_nan_rate,
        )

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(RAW_OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        RAW_OUTPUT_PATH, len(combined), len(combined.columns),
    )


if __name__ == "__main__":
    main()
