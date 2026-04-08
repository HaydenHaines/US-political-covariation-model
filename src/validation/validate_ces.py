"""
CES/CCES Survey Validation Pipeline

Downloads the Cooperative Election Study (CES) cumulative dataset, maps
respondents to WetherVane community types via county FIPS, and compares
survey-observed type-level D-share against model predictions.

This is the first external validation of the type model using a large
independent survey with validated voter records.

Usage:
    uv run python -m src.validation.validate_ces
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CES_DIR = PROJECT_ROOT / "data" / "raw" / "ces"
CES_FILE = CES_DIR / "cumulative_2006-2024.feather"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

COUNTY_TYPE_FILE = PROJECT_ROOT / "data" / "communities" / "county_type_assignments_full.parquet"
TYPE_PRIORS_FILE = PROJECT_ROOT / "data" / "communities" / "type_priors.parquet"

OUTPUT_JSON = VALIDATION_DIR / "ces_type_validation.json"
OUTPUT_CSV = VALIDATION_DIR / "ces_type_comparison.csv"

# Harvard Dataverse file ID for CES cumulative 2006-2024
CES_DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/12134962"

# CES column names (verified from actual file)
COL_YEAR = "year"
COL_COUNTY_FIPS = "county_fips"
COL_VV_TURNOUT = "vv_turnout_gvm"
COL_VOTED_PRES_PARTY = "voted_pres_party"
COL_VOTED_GOV_PARTY = "voted_gov_party"
COL_VOTED_SEN_PARTY = "voted_sen_party"
COL_WEIGHT = "weight_cumulative"  # Cumulative weight (stable across years)

# Validated voter identifier in vv_turnout_gvm
VV_VOTED = "Voted"

# Presidential election years in cumulative file
PRES_YEARS = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]

# Two-party presidential vote parties
PARTY_DEM = "Democratic"
PARTY_REP = "Republican"

# Minimum respondents per type-year to include in comparison
MIN_N_PER_TYPE = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class ValidationResults(NamedTuple):
    """Summary statistics from the CES type-level validation."""

    pearson_r: float
    rmse: float
    bias: float
    n_types: int
    n_respondents: int
    comparison_year: int | None
    ces_dem_share_mean: float
    model_dem_share_mean: float


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_ces(url: str = CES_DATAVERSE_URL, dest: Path = CES_FILE) -> Path:
    """
    Download the CES cumulative feather file from Harvard Dataverse if not cached.

    The file is ~135MB. No authentication required.

    Args:
        url: Dataverse API download URL for the feather file.
        dest: Local destination path.

    Returns:
        Path to the downloaded (or already-cached) file.
    """
    if dest.exists():
        log.info("CES file already cached at %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading CES cumulative file from %s ...", url)

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    bytes_downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1_048_576):  # 1MB chunks
            if chunk:
                f.write(chunk)
                bytes_downloaded += len(chunk)

    log.info("Downloaded %d MB to %s", bytes_downloaded // 1_048_576, dest)
    return dest


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------


def load_ces(path: Path = CES_FILE) -> pd.DataFrame:
    """
    Load the CES feather file and return the full dataframe.

    Verifies expected columns are present.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CES file not found at {path}. Run download_ces() first."
        )

    df = pd.read_feather(path)

    required_cols = [COL_YEAR, COL_COUNTY_FIPS, COL_VV_TURNOUT, COL_VOTED_PRES_PARTY, COL_WEIGHT]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CES file missing expected columns: {missing}")

    log.info("Loaded CES: %d rows × %d columns", *df.shape)
    return df


def filter_validated_presidential_voters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter CES to validated presidential voters only.

    Rules:
    1. vv_turnout_gvm == "Voted" (Catalist-validated turnout)
    2. voted_pres_party in {Democratic, Republican} (two-party vote only)
    3. county_fips not null (need geography for type assignment)

    Why these filters:
    - Validated turnout is the gold standard — self-reported is noisier
    - Limiting to D/R two-party mirrors how WetherVane models D-share
    - Null FIPS rows can't be matched to types
    """
    validated = df[
        (df[COL_VV_TURNOUT] == VV_VOTED)
        & (df[COL_VOTED_PRES_PARTY].isin([PARTY_DEM, PARTY_REP]))
        & (df[COL_COUNTY_FIPS].notna())
    ].copy()

    log.info(
        "After filter: %d validated presidential voters (from %d total)",
        len(validated),
        len(df),
    )
    return validated


# ---------------------------------------------------------------------------
# County FIPS join
# ---------------------------------------------------------------------------


def normalize_fips(series: pd.Series) -> pd.Series:
    """
    Normalize a county FIPS series to zero-padded 5-digit strings.

    Handles both numeric (int/float) and string inputs.
    CES stores FIPS as 5-digit strings already, but we defensively
    handle numeric types in case the format changes.
    """
    if pd.api.types.is_numeric_dtype(series):
        # Numeric FIPS (e.g., 1001 for Alabama, Autauga) → zero-pad to 5 digits
        return series.dropna().astype(int).astype(str).str.zfill(5)
    else:
        # Already strings — ensure 5-digit zero-padded
        return series.str.strip().str.zfill(5)


def join_county_types(
    ces_df: pd.DataFrame,
    type_file: Path = COUNTY_TYPE_FILE,
) -> tuple[pd.DataFrame, dict]:
    """
    Join CES respondents to county type assignments via county_fips.

    Args:
        ces_df: Filtered CES dataframe with county_fips column.
        type_file: Path to county_type_assignments_full.parquet.

    Returns:
        Tuple of (merged_df, match_stats) where match_stats contains
        match rate information.
    """
    types = pd.read_parquet(type_file, columns=["county_fips", "dominant_type"])

    # Normalize FIPS in both tables to ensure consistent format
    ces_df = ces_df.copy()
    ces_df[COL_COUNTY_FIPS] = normalize_fips(ces_df[COL_COUNTY_FIPS])
    types["county_fips"] = normalize_fips(types["county_fips"])

    n_before = len(ces_df)
    merged = ces_df.merge(types, left_on=COL_COUNTY_FIPS, right_on="county_fips", how="inner")
    n_after = len(merged)

    match_rate = n_after / n_before if n_before > 0 else 0.0
    n_ces_counties = ces_df[COL_COUNTY_FIPS].nunique()
    n_matched_counties = merged[COL_COUNTY_FIPS].nunique()

    match_stats = {
        "n_respondents_before": n_before,
        "n_respondents_after": n_after,
        "respondent_match_rate": round(match_rate, 4),
        "n_ces_counties": n_ces_counties,
        "n_matched_counties": n_matched_counties,
        "county_match_rate": round(n_matched_counties / n_ces_counties, 4) if n_ces_counties > 0 else 0.0,
    }

    log.info(
        "County join: %d/%d rows matched (%.1f%%), %d/%d counties matched",
        n_after,
        n_before,
        match_rate * 100,
        n_matched_counties,
        n_ces_counties,
    )

    return merged, match_stats


# ---------------------------------------------------------------------------
# Type-level aggregation
# ---------------------------------------------------------------------------


def aggregate_by_type_year(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CES-observed D-share per (type, year) among validated voters.

    Uses cumulative survey weights (weight_cumulative) for proper weighted
    aggregation. CES cumulative weights adjust for differential non-response
    and target the national voting-age population.

    Why weight? YouGov uses opt-in panel + MrP weighting. Raw respondent
    counts oversample highly-educated and high-interest voters. Weights
    correct for this. For type-level aggregation the effect is moderate
    (types are correlated with education/interest), but we use weights
    to be rigorous.

    Returns:
        DataFrame with columns: type_id, year, ces_dem_share, n_respondents,
        n_weighted (effective sample size).
    """
    merged = merged.copy()

    # Validated presidential vote: 1 = Democratic, 0 = Republican
    merged["is_dem"] = (merged[COL_VOTED_PRES_PARTY] == PARTY_DEM).astype(float)

    # Ensure weights are non-null and positive
    merged[COL_WEIGHT] = pd.to_numeric(merged[COL_WEIGHT], errors="coerce").fillna(1.0)
    merged[COL_WEIGHT] = merged[COL_WEIGHT].clip(lower=0.0)

    # Weighted D-share per type-year
    def weighted_dem_share(group: pd.DataFrame) -> pd.Series:
        w = group[COL_WEIGHT]
        dem = group["is_dem"]
        total_weight = w.sum()
        if total_weight == 0:
            return pd.Series({"ces_dem_share": np.nan, "n_respondents": len(group), "n_weighted": 0.0})
        weighted_share = (dem * w).sum() / total_weight
        return pd.Series(
            {
                "ces_dem_share": weighted_share,
                "n_respondents": len(group),
                "n_weighted": total_weight,
            }
        )

    result = (
        merged.groupby(["dominant_type", COL_YEAR])
        .apply(weighted_dem_share, include_groups=False)
        .reset_index()
        .rename(columns={"dominant_type": "type_id", COL_YEAR: "year"})
    )

    log.info("Aggregated to %d type-year cells", len(result))
    return result


def compute_type_means(type_year: pd.DataFrame) -> pd.DataFrame:
    """
    Compute population-weighted mean D-share per type across all years.

    Uses n_weighted (sum of survey weights) as the aggregation weight,
    so years and types with more respondents contribute proportionally more.

    This is the CES-observed type fingerprint — what each type actually votes.
    """
    result = (
        type_year.groupby("type_id")
        .apply(
            lambda g: pd.Series(
                {
                    "ces_dem_share_mean": np.average(
                        g["ces_dem_share"].dropna(),
                        weights=g.loc[g["ces_dem_share"].notna(), "n_weighted"],
                    )
                    if g["ces_dem_share"].notna().any()
                    else np.nan,
                    "total_respondents": g["n_respondents"].sum(),
                    "total_weighted": g["n_weighted"].sum(),
                    "n_years": g["ces_dem_share"].notna().sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return result


# ---------------------------------------------------------------------------
# Comparison with model
# ---------------------------------------------------------------------------


def load_type_priors(priors_file: Path = TYPE_PRIORS_FILE) -> pd.DataFrame:
    """
    Load model type-level D-share priors.

    type_priors.parquet has two columns: type_id and prior_dem_share.
    These are the model's historical type-level D-share estimates —
    computed from election returns weighted by type membership.

    We compare these against CES-observed D-share as external validation.
    """
    return pd.read_parquet(priors_file)


def compare_ces_to_model(
    type_means: pd.DataFrame,
    model_priors: pd.DataFrame,
    min_respondents: int = MIN_N_PER_TYPE,
) -> pd.DataFrame:
    """
    Merge CES-observed type means with model type priors and compute error metrics.

    Args:
        type_means: CES type means from compute_type_means().
        model_priors: Model type priors with type_id and prior_dem_share.
        min_respondents: Minimum total respondents to include a type in comparison.

    Returns:
        DataFrame with one row per type, columns for CES share, model share,
        and error metrics.
    """
    # Filter to types with sufficient sample
    type_means = type_means[type_means["total_respondents"] >= min_respondents].copy()

    merged = type_means.merge(model_priors, on="type_id", how="inner")

    # Per-type error
    merged["error"] = merged["ces_dem_share_mean"] - merged["prior_dem_share"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2

    return merged.sort_values("type_id").reset_index(drop=True)


def compute_validation_stats(comparison: pd.DataFrame) -> ValidationResults:
    """
    Compute summary validation statistics from the type comparison DataFrame.

    Pearson r: Correlation between CES-observed and model-predicted D-share.
    RMSE: Root mean squared error (equally-weighted, not respondent-weighted).
    Bias: Mean signed error (positive = CES sees more D than model predicts).
    """
    valid = comparison.dropna(subset=["ces_dem_share_mean", "prior_dem_share"])

    ces_vals = valid["ces_dem_share_mean"].values
    model_vals = valid["prior_dem_share"].values

    if len(valid) < 2:
        raise ValueError(f"Insufficient types for validation: {len(valid)} (need >=2)")

    pearson_r = float(np.corrcoef(ces_vals, model_vals)[0, 1])
    rmse = float(np.sqrt(np.mean((ces_vals - model_vals) ** 2)))
    bias = float(np.mean(ces_vals - model_vals))

    return ValidationResults(
        pearson_r=round(pearson_r, 4),
        rmse=round(rmse, 4),
        bias=round(bias, 4),
        n_types=len(valid),
        n_respondents=int(valid["total_respondents"].sum()),
        comparison_year=None,  # Across all years
        ces_dem_share_mean=round(float(np.mean(ces_vals)), 4),
        model_dem_share_mean=round(float(np.mean(model_vals)), 4),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(
    comparison: pd.DataFrame,
    results: ValidationResults,
    match_stats: dict,
    type_year: pd.DataFrame,
) -> None:
    """
    Save validation results to JSON summary and CSV per-type comparison.
    """
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary = {
        "pearson_r": results.pearson_r,
        "rmse": results.rmse,
        "bias": results.bias,
        "n_types": results.n_types,
        "n_respondents": results.n_respondents,
        "ces_dem_share_mean": results.ces_dem_share_mean,
        "model_dem_share_mean": results.model_dem_share_mean,
        "match_stats": match_stats,
        "years_included": sorted(type_year["year"].unique().tolist()),
        "min_respondents_threshold": MIN_N_PER_TYPE,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved validation summary to %s", OUTPUT_JSON)

    # CSV per-type comparison
    comparison.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved per-type comparison to %s", OUTPUT_CSV)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: ValidationResults, comparison: pd.DataFrame, match_stats: dict) -> None:
    """Print a clear human-readable summary of the validation results."""
    print("\n" + "=" * 70)
    print("CES/CCES Type Model External Validation")
    print("=" * 70)

    print("\n── Data Coverage ──────────────────────────────────────────────────────")
    print(f"  Respondents matched:  {match_stats['n_respondents_after']:,} / {match_stats['n_respondents_before']:,}")
    print(f"  Respondent match rate: {match_stats['respondent_match_rate']:.1%}")
    print(f"  Counties matched:     {match_stats['n_matched_counties']:,} / {match_stats['n_ces_counties']:,}")
    print(f"  County match rate:    {match_stats['county_match_rate']:.1%}")
    print(f"  Types with data:      {results.n_types} / 100")

    print("\n── Validation Results (CES D-share vs. Model Prior) ──────────────────")
    print(f"  Pearson r:            {results.pearson_r:+.4f}")
    print(f"  RMSE:                 {results.rmse:.4f}  ({results.rmse * 100:.2f}pp)")
    print(f"  Bias:                 {results.bias:+.4f}  ({results.bias * 100:+.2f}pp)")
    print(f"  CES mean D-share:     {results.ces_dem_share_mean:.4f}  ({results.ces_dem_share_mean * 100:.2f}%)")
    print(f"  Model mean D-share:   {results.model_dem_share_mean:.4f}  ({results.model_dem_share_mean * 100:.2f}%)")

    print("\n── Interpretation ─────────────────────────────────────────────────────")
    if results.pearson_r >= 0.9:
        print("  r >= 0.90: Excellent alignment between CES survey and model types.")
    elif results.pearson_r >= 0.8:
        print("  r >= 0.80: Strong alignment — types capture real partisan structure.")
    elif results.pearson_r >= 0.7:
        print("  r >= 0.70: Good alignment — types useful, some structural noise.")
    else:
        print(f"  r = {results.pearson_r:.2f}: Moderate alignment — investigate types with large errors.")

    # Bias = CES - model: positive means CES sees more D than model predicts
    if results.bias > 0:
        print(f"  Bias: CES sees +{results.bias * 100:.2f}pp more D than model predicts (model under-predicts D).")
    else:
        print(f"  Bias: CES sees {results.bias * 100:.2f}pp less D than model predicts (model over-predicts D).")

    print("\n── Top 5 Types by Absolute Error ─────────────────────────────────────")
    worst = comparison.nlargest(5, "abs_error")[
        ["type_id", "ces_dem_share_mean", "prior_dem_share", "error", "total_respondents"]
    ]
    print(f"  {'Type':<8}  {'CES D%':>8}  {'Model D%':>10}  {'Error':>8}  {'N':>8}")
    print("  " + "-" * 48)
    for _, row in worst.iterrows():
        print(
            f"  {int(row['type_id']):<8}  "
            f"{row['ces_dem_share_mean'] * 100:>7.1f}%  "
            f"{row['prior_dem_share'] * 100:>9.1f}%  "
            f"{row['error'] * 100:>+7.1f}pp  "
            f"{int(row['total_respondents']):>8,}"
        )

    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_validation(
    ces_path: Path = CES_FILE,
    county_type_path: Path = COUNTY_TYPE_FILE,
    priors_path: Path = TYPE_PRIORS_FILE,
) -> tuple[ValidationResults, pd.DataFrame]:
    """
    Run the full CES validation pipeline end-to-end.

    Steps:
    1. Download CES data if not cached
    2. Load and filter to validated presidential voters
    3. Join to county type assignments
    4. Aggregate D-share by type and year
    5. Compute type means across years
    6. Compare to model type priors
    7. Save outputs and print report

    Returns:
        (ValidationResults, per-type comparison DataFrame)
    """
    # Step 1: Ensure file is available
    download_ces(dest=ces_path)

    # Step 2: Load and filter
    ces = load_ces(ces_path)
    validated = filter_validated_presidential_voters(ces)

    # Step 3: Join county types
    merged, match_stats = join_county_types(validated, county_type_path)

    # Step 4: Aggregate by type-year
    type_year = aggregate_by_type_year(merged)

    # Step 5: Compute type means across all years
    type_means = compute_type_means(type_year)

    # Step 6: Load model priors and compare
    model_priors = load_type_priors(priors_path)
    comparison = compare_ces_to_model(type_means, model_priors)
    results = compute_validation_stats(comparison)

    # Step 7: Save and report
    save_outputs(comparison, results, match_stats, type_year)
    print_report(results, comparison, match_stats)

    return results, comparison


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_validation()


if __name__ == "__main__":
    main()
