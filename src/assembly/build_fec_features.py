"""Build county-level FEC donor density features from state-level aggregates.

Reads FEC individual contribution totals aggregated by state (from fetch_fec_donors.py)
and maps them to counties via state FIPS prefix, then combines with ACS population
counts to produce per-capita and per-1,000-population donor metrics.

Design choice — state-level mapping (not ZIP-to-county):
  The FEC by_zip endpoint returns one row per (committee, ZIP, cycle), producing
  ~2.3M rows for the 2024 cycle. Paginating that at 100/page would require ~23K
  API calls — infeasible at 1,000 req/hr without caching. The by_state endpoint
  returns ~1,200 pages for the same cycle and is within budget.

  State-to-county mapping is appropriate here because the dominant signal in FEC
  donor density is cross-state variation (high-donor coastal metros vs low-donor
  rural interior), not within-state county variation. This matches the approach
  used by build_bea_state_features.py for state-level economic context signals.

Features produced per county (all state-level values mapped via FIPS prefix):
  fec_donors_per_1k     — individual donor records per 1,000 population
  fec_total_per_capita  — total contribution dollars per capita
  fec_avg_contribution  — average contribution amount in dollars

Missing counties (no matching state in FEC data) are filled with national median.
National median fill avoids dropping any county from the feature matrix.

FIPS format: all county_fips values are 5-char zero-padded strings throughout.

Data sources:
  data/raw/fec/fec_by_state_{cycle}.parquet  (from fetch_fec_donors.py)
  data/assembled/county_acs_features.parquet  (for population denominator)

Outputs:
  data/assembled/county_fec_features.parquet
      county_fips + fec_donors_per_1k + fec_total_per_capita + fec_avg_contribution
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.assembly.fetch_fec_donors import RAW_CACHE_PATH as DEFAULT_RAW_CACHE_PATH
from src.core import config as _cfg

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_fec_features.parquet"

# Maps 2-digit FIPS prefix (e.g., "12") → state abbreviation (e.g., "FL").
# Derived from project config; covers all 50 states + DC.
_FIPS_TO_STATE_ABBR: dict[str, str] = _cfg.STATE_ABBR

# Inverse: state abbreviation → 2-digit FIPS prefix.
_STATE_ABBR_TO_FIPS: dict[str, str] = {v: k for k, v in _FIPS_TO_STATE_ABBR.items()}

# Output feature column names — exposed as module constants so callers can reference
# them without importing the full pipeline (matches pattern in build_bea_state_features.py).
COL_DONORS_PER_1K = "fec_donors_per_1k"
COL_TOTAL_PER_CAPITA = "fec_total_per_capita"
COL_AVG_CONTRIBUTION = "fec_avg_contribution"
FEATURE_COLS = [COL_DONORS_PER_1K, COL_TOTAL_PER_CAPITA, COL_AVG_CONTRIBUTION]

# Minimum population to avoid division-by-zero or nonsensical per-capita values.
# Counties below this threshold still get a value (via national median fill) but
# do not contribute to the national median calculation with extreme per-capita values.
_MIN_POPULATION = 100


def load_fec_state_totals(raw_cache_path: Path = DEFAULT_RAW_CACHE_PATH) -> pd.DataFrame:
    """Load the raw FEC by-state aggregated totals.

    Returns a DataFrame with columns:
        state          — 2-letter state abbreviation
        total_amount   — total contribution dollars (all committees summed)
        total_count    — total individual contribution records

    The file is produced by fetch_fec_donors.py. If it does not exist, a
    helpful error is raised rather than silently returning empty data.
    """
    if not raw_cache_path.exists():
        raise FileNotFoundError(
            f"FEC by-state cache not found: {raw_cache_path}. "
            "Run src/assembly/fetch_fec_donors.py to download it."
        )
    df = pd.read_parquet(raw_cache_path)
    required = {"state", "total_amount", "total_count"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"FEC by-state cache missing required columns. "
            f"Found: {list(df.columns)}, expected: {sorted(required)}"
        )
    return df


def compute_state_fec_metrics(
    fec_state: pd.DataFrame,
    acs: pd.DataFrame,
) -> pd.Series:
    """Compute per-capita and per-1k FEC donor metrics at the state level.

    Maps FEC state abbreviations to 2-digit FIPS prefixes, joins with
    state-level population aggregated from the ACS county spine, and
    computes three per-capita metrics.

    Parameters
    ----------
    fec_state:
        DataFrame with columns: state, total_amount, total_count.
        One row per 2-letter state abbreviation.
    acs:
        ACS county spine with columns: county_fips, pop_total.
        Used to aggregate state population for the denominator.

    Returns
    -------
    DataFrame with columns: fips_prefix, fec_donors_per_1k,
    fec_total_per_capita, fec_avg_contribution. One row per state
    that has both FEC data and ACS population.
    """
    # Aggregate ACS to state-level population using the first 2 chars of FIPS.
    acs_copy = acs[["county_fips", "pop_total"]].copy()
    acs_copy["fips_prefix"] = acs_copy["county_fips"].astype(str).str.zfill(5).str[:2]
    state_pop = (
        acs_copy.groupby("fips_prefix")["pop_total"]
        .sum()
        .reset_index()
        .rename(columns={"pop_total": "state_population"})
    )

    # Map FEC state abbreviations to FIPS prefixes.
    fec = fec_state.copy()
    fec["fips_prefix"] = fec["state"].map(_STATE_ABBR_TO_FIPS)
    missing_states = fec[fec["fips_prefix"].isna()]["state"].tolist()
    if missing_states:
        log.warning(
            "FEC states with no FIPS mapping (likely territories): %s", missing_states
        )
    fec = fec.dropna(subset=["fips_prefix"])

    # Join FEC totals with state-level population.
    merged = fec.merge(state_pop, on="fips_prefix", how="inner")

    # Guard against zero/missing population to avoid division by zero.
    merged = merged[merged["state_population"] >= _MIN_POPULATION].copy()

    # Compute per-capita metrics.
    # fec_donors_per_1k: donation records per 1,000 population.
    merged[COL_DONORS_PER_1K] = (
        merged["total_count"] / merged["state_population"] * 1_000
    )

    # fec_total_per_capita: total dollars donated per person.
    merged[COL_TOTAL_PER_CAPITA] = (
        merged["total_amount"] / merged["state_population"]
    )

    # fec_avg_contribution: average contribution amount in dollars.
    # Guard against zero count (divide-by-zero → NaN → handled downstream).
    count_safe = merged["total_count"].replace(0, pd.NA)
    merged[COL_AVG_CONTRIBUTION] = (
        merged["total_amount"] / count_safe
    )

    log.info(
        "FEC state metrics: %d states  donors_per_1k range [%.1f, %.1f]"
        "  total_per_capita range [$%.2f, $%.2f]",
        len(merged),
        merged[COL_DONORS_PER_1K].min(),
        merged[COL_DONORS_PER_1K].max(),
        merged[COL_TOTAL_PER_CAPITA].min(),
        merged[COL_TOTAL_PER_CAPITA].max(),
    )

    return merged[["fips_prefix"] + FEATURE_COLS].reset_index(drop=True)


def build_county_fec_features(
    county_fips: list[str],
    raw_cache_path: Path = DEFAULT_RAW_CACHE_PATH,
    acs_path: Path = ACS_PATH,
) -> pd.DataFrame:
    """Map state-level FEC donor density metrics to counties via FIPS prefix.

    Each county inherits its state's FEC donor activity values. Counties in
    states without FEC data are filled with the national median so the output
    covers all input counties without gaps.

    Parameters
    ----------
    county_fips:
        List of 5-char zero-padded county FIPS strings (e.g., "12001").
    raw_cache_path:
        Path to the FEC by-state raw cache. Defaults to the standard path
        produced by fetch_fec_donors.py.
    acs_path:
        Path to the ACS county spine (for state population aggregation).

    Returns
    -------
    DataFrame with columns: county_fips, fec_donors_per_1k,
    fec_total_per_capita, fec_avg_contribution. One row per county_fips.

    Notes
    -----
    - All county_fips values must be exactly 5 characters.
    - Duplicate FIPS in the input are preserved (not deduplicated).
    - National medians are used for any state not in the FEC cache.
    """
    if not acs_path.exists():
        raise FileNotFoundError(
            f"ACS county spine not found: {acs_path}. "
            "Run src/assembly/build_county_acs_features.py first."
        )

    fec_state = load_fec_state_totals(raw_cache_path)
    acs = pd.read_parquet(acs_path)

    state_metrics = compute_state_fec_metrics(fec_state, acs)
    # Indexed by fips_prefix for fast lookup.
    metrics_idx = state_metrics.set_index("fips_prefix")

    # Build output DataFrame from the input county_fips list.
    df = pd.DataFrame({"county_fips": county_fips})
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    if not df["county_fips"].str.len().eq(5).all():
        bad = df[df["county_fips"].str.len() != 5]["county_fips"].tolist()[:5]
        raise ValueError(f"county_fips must be 5-char strings. Bad examples: {bad}")

    state_prefix = df["county_fips"].str[:2]

    # Map state-level metrics to counties.
    for col in FEATURE_COLS:
        df[col] = state_prefix.map(metrics_idx[col])

    # Fill counties in states with no FEC data (territories, missing states)
    # with the national median computed from available states.
    for col in FEATURE_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            national_median = float(df[col].median())
            df[col] = df[col].fillna(national_median)
            log.info(
                "%d counties lack FEC data for %s — filled with national median %.3f",
                n_missing, col, national_median,
            )

    log.info(
        "Built FEC county features: %d counties  donors_per_1k range [%.1f, %.1f]",
        len(df),
        df[COL_DONORS_PER_1K].min(),
        df[COL_DONORS_PER_1K].max(),
    )

    return df[["county_fips"] + FEATURE_COLS].reset_index(drop=True)


def main() -> None:
    """Build and save the county FEC donor density features parquet.

    Loads all county FIPS from the ACS spine and maps state-level FEC
    donor activity to each county. Output written to
    data/assembled/county_fec_features.parquet.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not ACS_PATH.exists():
        raise FileNotFoundError(
            f"ACS spine not found: {ACS_PATH}. "
            "Run src/assembly/build_county_acs_features.py first."
        )

    log.info("Loading ACS county spine from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)
    county_fips = acs["county_fips"].astype(str).str.zfill(5).tolist()
    log.info("Found %d counties in ACS spine", len(county_fips))

    features = build_county_fec_features(county_fips)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %d county FEC features → %s", len(features), OUTPUT_PATH)

    for col in FEATURE_COLS:
        log.info(
            "%s — Q1=%.2f  median=%.2f  Q3=%.2f",
            col,
            *features[col].quantile([0.25, 0.5, 0.75]),
        )


if __name__ == "__main__":
    main()
