"""
Tests for the CES/CCES survey validation pipeline.

Covers:
- County FIPS normalization and matching logic
- Type-level aggregation correctness (known input → known output)
- Edge cases: types with 0 respondents, missing county_fips, all-one-party types
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.validate_ces import (
    aggregate_by_type_year,
    compare_ces_to_model,
    compute_type_means,
    compute_validation_stats,
    filter_validated_presidential_voters,
    join_county_types,
    normalize_fips,
    validate_per_year,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ces_df() -> pd.DataFrame:
    """
    Minimal CES-like DataFrame with validated voters and county_fips.

    Two types:
    - Type 0: county 01001, 3 Dem + 2 Rep → D-share = 0.60
    - Type 1: county 02001, 1 Dem + 3 Rep → D-share = 0.25
    """
    return pd.DataFrame(
        {
            "year": [2020] * 9,
            "county_fips": ["01001"] * 5 + ["02001"] * 4,
            "vv_turnout_gvm": ["Voted"] * 9,
            "voted_pres_party": ["Democratic", "Democratic", "Democratic", "Republican", "Republican",
                                  "Democratic", "Republican", "Republican", "Republican"],
            "weight_cumulative": [1.0] * 9,
        }
    )


@pytest.fixture
def sample_type_assignments() -> pd.DataFrame:
    """County-to-type mapping matching sample_ces_df."""
    return pd.DataFrame(
        {
            "county_fips": ["01001", "02001"],
            "dominant_type": [0, 1],
        }
    )


@pytest.fixture
def sample_model_priors() -> pd.DataFrame:
    """Model-predicted D-share per type."""
    return pd.DataFrame(
        {
            "type_id": [0, 1],
            "prior_dem_share": [0.55, 0.30],
        }
    )


# ---------------------------------------------------------------------------
# normalize_fips tests
# ---------------------------------------------------------------------------


def test_normalize_fips_string_already_padded() -> None:
    series = pd.Series(["01001", "06037", "48453"])
    result = normalize_fips(series)
    assert result.tolist() == ["01001", "06037", "48453"]


def test_normalize_fips_short_string_gets_padded() -> None:
    """FIPS like '1001' (4 chars, Alabama) should become '01001'."""
    series = pd.Series(["1001", "6037"])
    result = normalize_fips(series)
    assert result.tolist() == ["01001", "06037"]


def test_normalize_fips_numeric_int() -> None:
    """Integer FIPS values should be zero-padded to 5 digits."""
    series = pd.Series([1001, 6037, 48453])
    result = normalize_fips(series)
    assert result.tolist() == ["01001", "06037", "48453"]


def test_normalize_fips_numeric_float() -> None:
    """Float FIPS (e.g. from CSV with NaN coercion) should be handled."""
    series = pd.Series([1001.0, 6037.0])
    result = normalize_fips(series)
    assert result.tolist() == ["01001", "06037"]


def test_normalize_fips_five_digit_stays_unchanged() -> None:
    """5-digit FIPS should not be changed."""
    series = pd.Series(["36061", "04013"])
    result = normalize_fips(series)
    assert result.tolist() == ["36061", "04013"]


# ---------------------------------------------------------------------------
# filter_validated_presidential_voters tests
# ---------------------------------------------------------------------------


def test_filter_keeps_voted_dem_rep(sample_ces_df: pd.DataFrame) -> None:
    result = filter_validated_presidential_voters(sample_ces_df)
    assert len(result) == 9
    assert set(result["voted_pres_party"].unique()) == {"Democratic", "Republican"}


def test_filter_removes_non_voter() -> None:
    df = pd.DataFrame(
        {
            "year": [2020, 2020],
            "county_fips": ["01001", "01001"],
            "vv_turnout_gvm": ["Voted", "No Record of Voting"],
            "voted_pres_party": ["Democratic", "Democratic"],
            "weight_cumulative": [1.0, 1.0],
        }
    )
    result = filter_validated_presidential_voters(df)
    assert len(result) == 1


def test_filter_removes_third_party() -> None:
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2020],
            "county_fips": ["01001", "01001", "01001"],
            "vv_turnout_gvm": ["Voted", "Voted", "Voted"],
            "voted_pres_party": ["Democratic", "Republican", "Other"],
            "weight_cumulative": [1.0, 1.0, 1.0],
        }
    )
    result = filter_validated_presidential_voters(df)
    assert len(result) == 2
    assert "Other" not in result["voted_pres_party"].values


def test_filter_removes_null_fips() -> None:
    df = pd.DataFrame(
        {
            "year": [2020, 2020],
            "county_fips": ["01001", None],
            "vv_turnout_gvm": ["Voted", "Voted"],
            "voted_pres_party": ["Democratic", "Democratic"],
            "weight_cumulative": [1.0, 1.0],
        }
    )
    result = filter_validated_presidential_voters(df)
    assert len(result) == 1
    assert result["county_fips"].notna().all()


# ---------------------------------------------------------------------------
# join_county_types tests
# ---------------------------------------------------------------------------


def test_join_county_types_basic(
    sample_ces_df: pd.DataFrame,
    sample_type_assignments: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Basic join should produce correct type assignments."""
    type_file = tmp_path / "county_types.parquet"
    sample_type_assignments.to_parquet(type_file, index=False)

    validated = filter_validated_presidential_voters(sample_ces_df)
    merged, stats = join_county_types(validated, type_file)

    assert "dominant_type" in merged.columns
    assert stats["respondent_match_rate"] == pytest.approx(1.0)
    # Types should be correctly assigned
    assert set(merged.loc[merged["county_fips"] == "01001", "dominant_type"].unique()) == {0}
    assert set(merged.loc[merged["county_fips"] == "02001", "dominant_type"].unique()) == {1}


def test_join_county_types_unmatched_fips(
    sample_ces_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """County FIPS in CES not in type assignments → those rows dropped."""
    # Only map county 01001, not 02001
    partial_types = pd.DataFrame(
        {"county_fips": ["01001"], "dominant_type": [0]}
    )
    type_file = tmp_path / "county_types.parquet"
    partial_types.to_parquet(type_file, index=False)

    validated = filter_validated_presidential_voters(sample_ces_df)
    merged, stats = join_county_types(validated, type_file)

    # Only county 01001 (5 rows) should survive
    assert len(merged) == 5
    assert stats["respondent_match_rate"] < 1.0
    assert stats["n_matched_counties"] == 1


def test_join_county_types_empty_ces(
    sample_type_assignments: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Empty CES DataFrame should produce empty result with 0% match rate."""
    type_file = tmp_path / "county_types.parquet"
    sample_type_assignments.to_parquet(type_file, index=False)

    empty_ces = pd.DataFrame(
        columns=["year", "county_fips", "vv_turnout_gvm", "voted_pres_party", "weight_cumulative"]
    )
    validated = filter_validated_presidential_voters(empty_ces)
    merged, stats = join_county_types(validated, type_file)

    assert len(merged) == 0
    assert stats["respondent_match_rate"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_by_type_year tests
# ---------------------------------------------------------------------------


def test_aggregate_basic_dem_share(
    sample_ces_df: pd.DataFrame,
    sample_type_assignments: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """
    Type 0 (county 01001): 3 Dem / 5 total → D-share = 0.60
    Type 1 (county 02001): 1 Dem / 4 total → D-share = 0.25
    """
    type_file = tmp_path / "county_types.parquet"
    sample_type_assignments.to_parquet(type_file, index=False)

    validated = filter_validated_presidential_voters(sample_ces_df)
    merged, _ = join_county_types(validated, type_file)
    agg = aggregate_by_type_year(merged)

    t0 = agg.loc[(agg["type_id"] == 0) & (agg["year"] == 2020), "ces_dem_share"].iloc[0]
    t1 = agg.loc[(agg["type_id"] == 1) & (agg["year"] == 2020), "ces_dem_share"].iloc[0]

    assert t0 == pytest.approx(0.60, abs=1e-6)
    assert t1 == pytest.approx(0.25, abs=1e-6)


def test_aggregate_uniform_weight_equals_simple_mean(
    sample_ces_df: pd.DataFrame,
    sample_type_assignments: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """With uniform weights, weighted mean == simple mean."""
    type_file = tmp_path / "county_types.parquet"
    sample_type_assignments.to_parquet(type_file, index=False)

    validated = filter_validated_presidential_voters(sample_ces_df)
    merged, _ = join_county_types(validated, type_file)
    agg = aggregate_by_type_year(merged)

    # Simple unweighted mean for type 0
    type_0 = merged[merged["dominant_type"] == 0]
    simple_mean = (type_0["voted_pres_party"] == "Democratic").mean()
    weighted_share = agg.loc[(agg["type_id"] == 0) & (agg["year"] == 2020), "ces_dem_share"].iloc[0]

    assert weighted_share == pytest.approx(simple_mean, abs=1e-6)


def test_aggregate_weighted_dem_share() -> None:
    """Verify that weights are actually used in aggregation (non-uniform weights)."""
    merged = pd.DataFrame(
        {
            "dominant_type": [0, 0, 0],
            "year": [2020, 2020, 2020],
            "county_fips": ["01001", "01001", "01001"],
            "voted_pres_party": ["Democratic", "Republican", "Republican"],
            "is_dem": [1.0, 0.0, 0.0],
            "weight_cumulative": [3.0, 1.0, 1.0],  # weight Dem 3x more
        }
    )
    # Manually add is_dem (normally created inside aggregate_by_type_year)
    # We call aggregate_by_type_year which will add is_dem itself
    merged_raw = pd.DataFrame(
        {
            "dominant_type": [0, 0, 0],
            "year": [2020, 2020, 2020],
            "county_fips": ["01001", "01001", "01001"],
            "voted_pres_party": ["Democratic", "Republican", "Republican"],
            "weight_cumulative": [3.0, 1.0, 1.0],
        }
    )
    agg = aggregate_by_type_year(merged_raw)

    # Weighted: (1*3 + 0*1 + 0*1) / (3+1+1) = 3/5 = 0.60
    # Unweighted would be: 1/3 ≈ 0.333
    result = agg.loc[(agg["type_id"] == 0) & (agg["year"] == 2020), "ces_dem_share"].iloc[0]
    assert result == pytest.approx(0.60, abs=1e-6)


def test_aggregate_multiple_years() -> None:
    """Aggregation should produce separate rows per year."""
    merged = pd.DataFrame(
        {
            "dominant_type": [0, 0, 0, 0],
            "year": [2020, 2020, 2016, 2016],
            "county_fips": ["01001"] * 4,
            "voted_pres_party": ["Democratic", "Republican", "Democratic", "Democratic"],
            "weight_cumulative": [1.0] * 4,
        }
    )
    agg = aggregate_by_type_year(merged)

    assert len(agg) == 2
    share_2020 = agg.loc[agg["year"] == 2020, "ces_dem_share"].iloc[0]
    share_2016 = agg.loc[agg["year"] == 2016, "ces_dem_share"].iloc[0]
    assert share_2020 == pytest.approx(0.50, abs=1e-6)
    assert share_2016 == pytest.approx(1.00, abs=1e-6)


def test_aggregate_type_with_no_respondents_excluded() -> None:
    """Types with no rows in merged (because county unmatched) should not appear."""
    merged = pd.DataFrame(
        {
            "dominant_type": [0, 0],
            "year": [2020, 2020],
            "county_fips": ["01001", "01001"],
            "voted_pres_party": ["Democratic", "Republican"],
            "weight_cumulative": [1.0, 1.0],
        }
    )
    agg = aggregate_by_type_year(merged)
    # Type 1 and 2 etc. should not be present since they had no rows
    assert set(agg["type_id"].unique()) == {0}


# ---------------------------------------------------------------------------
# compute_type_means tests
# ---------------------------------------------------------------------------


def test_compute_type_means_single_year() -> None:
    """With one year, means should match that year's values."""
    type_year = pd.DataFrame(
        {
            "type_id": [0, 1],
            "year": [2020, 2020],
            "ces_dem_share": [0.60, 0.25],
            "n_respondents": [100, 80],
            "n_weighted": [100.0, 80.0],
        }
    )
    means = compute_type_means(type_year)
    t0 = means.loc[means["type_id"] == 0, "ces_dem_share_mean"].iloc[0]
    t1 = means.loc[means["type_id"] == 1, "ces_dem_share_mean"].iloc[0]
    assert t0 == pytest.approx(0.60, abs=1e-6)
    assert t1 == pytest.approx(0.25, abs=1e-6)


def test_compute_type_means_weighted_by_n_weighted() -> None:
    """Multi-year means should be weighted by n_weighted (not equal-weight)."""
    type_year = pd.DataFrame(
        {
            "type_id": [0, 0],
            "year": [2020, 2016],
            "ces_dem_share": [0.60, 0.40],
            "n_respondents": [100, 200],
            "n_weighted": [100.0, 200.0],
        }
    )
    means = compute_type_means(type_year)
    mean_val = means.loc[means["type_id"] == 0, "ces_dem_share_mean"].iloc[0]
    # Weighted: (0.60*100 + 0.40*200) / 300 = (60+80)/300 = 140/300 ≈ 0.467
    expected = (0.60 * 100 + 0.40 * 200) / 300
    assert mean_val == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# compare_ces_to_model tests
# ---------------------------------------------------------------------------


def test_compare_perfect_correlation() -> None:
    """If CES matches model exactly, error should be zero."""
    type_means = pd.DataFrame(
        {
            "type_id": [0, 1, 2],
            "ces_dem_share_mean": [0.60, 0.40, 0.50],
            "total_respondents": [100, 100, 100],
            "total_weighted": [100.0, 100.0, 100.0],
            "n_years": [1, 1, 1],
        }
    )
    model_priors = pd.DataFrame(
        {
            "type_id": [0, 1, 2],
            "prior_dem_share": [0.60, 0.40, 0.50],
        }
    )
    comparison = compare_ces_to_model(type_means, model_priors, min_respondents=1)
    assert comparison["error"].abs().max() == pytest.approx(0.0, abs=1e-10)


def test_compare_min_respondents_filter() -> None:
    """Types below minimum respondent threshold should be excluded."""
    type_means = pd.DataFrame(
        {
            "type_id": [0, 1],
            "ces_dem_share_mean": [0.60, 0.40],
            "total_respondents": [100, 5],  # type 1 below threshold
            "total_weighted": [100.0, 5.0],
            "n_years": [1, 1],
        }
    )
    model_priors = pd.DataFrame(
        {"type_id": [0, 1], "prior_dem_share": [0.55, 0.35]}
    )
    comparison = compare_ces_to_model(type_means, model_priors, min_respondents=10)
    assert len(comparison) == 1
    assert comparison.iloc[0]["type_id"] == 0


def test_compare_error_sign_convention() -> None:
    """error = ces_dem_share - model_dem_share (positive = CES sees more D)."""
    type_means = pd.DataFrame(
        {
            "type_id": [0],
            "ces_dem_share_mean": [0.65],
            "total_respondents": [100],
            "total_weighted": [100.0],
            "n_years": [1],
        }
    )
    model_priors = pd.DataFrame({"type_id": [0], "prior_dem_share": [0.55]})
    comparison = compare_ces_to_model(type_means, model_priors, min_respondents=1)
    assert comparison.iloc[0]["error"] == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_validation_stats tests
# ---------------------------------------------------------------------------


def test_compute_validation_stats_perfect_r() -> None:
    """Perfect correlation should give r=1.0 and rmse=0."""
    comparison = pd.DataFrame(
        {
            "type_id": list(range(10)),
            "ces_dem_share_mean": np.linspace(0.2, 0.8, 10),
            "prior_dem_share": np.linspace(0.2, 0.8, 10),
            "total_respondents": [100] * 10,
            "error": [0.0] * 10,
        }
    )
    results = compute_validation_stats(comparison)
    assert results.pearson_r == pytest.approx(1.0, abs=1e-6)
    assert results.rmse == pytest.approx(0.0, abs=1e-6)
    assert results.bias == pytest.approx(0.0, abs=1e-6)


def test_compute_validation_stats_known_bias() -> None:
    """CES consistently +0.10 above model → bias = +0.10."""
    n = 10
    model_vals = np.linspace(0.3, 0.7, n)
    comparison = pd.DataFrame(
        {
            "type_id": list(range(n)),
            "ces_dem_share_mean": model_vals + 0.10,
            "prior_dem_share": model_vals,
            "total_respondents": [100] * n,
            "error": [0.10] * n,
        }
    )
    results = compute_validation_stats(comparison)
    assert results.bias == pytest.approx(0.10, abs=1e-6)
    assert results.pearson_r == pytest.approx(1.0, abs=1e-6)
    assert results.rmse == pytest.approx(0.10, abs=1e-6)


def test_compute_validation_stats_insufficient_types() -> None:
    """Fewer than 2 types should raise an error."""
    comparison = pd.DataFrame(
        {
            "type_id": [0],
            "ces_dem_share_mean": [0.50],
            "prior_dem_share": [0.45],
            "total_respondents": [100],
        }
    )
    with pytest.raises(ValueError, match="Insufficient types"):
        compute_validation_stats(comparison)


# ---------------------------------------------------------------------------
# validate_per_year tests
# ---------------------------------------------------------------------------


def test_validate_per_year_returns_only_pres_years() -> None:
    """Per-year validation should only include presidential years (divisible by 4)."""
    # Include both presidential (2016, 2020) and midterm (2018) years
    type_year = pd.DataFrame(
        {
            "type_id": list(range(10)) * 3,
            "year": [2016] * 10 + [2018] * 10 + [2020] * 10,
            "ces_dem_share": np.linspace(0.3, 0.7, 10).tolist() * 3,
            "n_respondents": [50] * 30,
            "n_weighted": [50.0] * 30,
        }
    )
    model_priors = pd.DataFrame(
        {"type_id": list(range(10)), "prior_dem_share": np.linspace(0.3, 0.7, 10)}
    )
    results = validate_per_year(type_year, model_priors, min_respondents=1)
    years = [r.comparison_year for r in results]
    assert 2016 in years
    assert 2020 in years
    assert 2018 not in years


def test_validate_per_year_perfect_correlation() -> None:
    """When CES exactly matches model, per-year r should be 1.0."""
    vals = np.linspace(0.2, 0.8, 10)
    type_year = pd.DataFrame(
        {
            "type_id": list(range(10)),
            "year": [2020] * 10,
            "ces_dem_share": vals,
            "n_respondents": [100] * 10,
            "n_weighted": [100.0] * 10,
        }
    )
    model_priors = pd.DataFrame(
        {"type_id": list(range(10)), "prior_dem_share": vals}
    )
    results = validate_per_year(type_year, model_priors, min_respondents=1)
    assert len(results) == 1
    assert results[0].pearson_r == pytest.approx(1.0, abs=1e-6)
    assert results[0].comparison_year == 2020


def test_validate_per_year_skips_insufficient_types() -> None:
    """Years with fewer than 5 types above min_respondents are skipped."""
    type_year = pd.DataFrame(
        {
            "type_id": [0, 1, 2, 3],
            "year": [2020, 2020, 2020, 2020],
            "ces_dem_share": [0.4, 0.5, 0.6, 0.7],
            "n_respondents": [100, 100, 100, 100],
            "n_weighted": [100.0, 100.0, 100.0, 100.0],
        }
    )
    model_priors = pd.DataFrame(
        {"type_id": [0, 1, 2, 3], "prior_dem_share": [0.4, 0.5, 0.6, 0.7]}
    )
    results = validate_per_year(type_year, model_priors, min_respondents=1)
    assert len(results) == 0  # Only 4 types, need >=5


def test_validate_per_year_respects_min_respondents() -> None:
    """Types below min_respondents threshold should be excluded per year."""
    type_year = pd.DataFrame(
        {
            "type_id": list(range(10)),
            "year": [2020] * 10,
            "ces_dem_share": np.linspace(0.3, 0.7, 10),
            "n_respondents": [100] * 5 + [3] * 5,  # 5 types below threshold
            "n_weighted": [100.0] * 5 + [3.0] * 5,
        }
    )
    model_priors = pd.DataFrame(
        {"type_id": list(range(10)), "prior_dem_share": np.linspace(0.3, 0.7, 10)}
    )
    results = validate_per_year(type_year, model_priors, min_respondents=10)
    assert len(results) == 1
    assert results[0].n_types == 5


def test_validate_per_year_bias_varies_by_year() -> None:
    """Different years should show different biases reflecting temporal shift."""
    n_types = 10
    vals = np.linspace(0.3, 0.7, n_types)
    type_year = pd.DataFrame(
        {
            "type_id": list(range(n_types)) * 2,
            "year": [2016] * n_types + [2020] * n_types,
            # 2016: CES matches model; 2020: CES is +0.05 higher
            "ces_dem_share": list(vals) + list(vals + 0.05),
            "n_respondents": [100] * (n_types * 2),
            "n_weighted": [100.0] * (n_types * 2),
        }
    )
    model_priors = pd.DataFrame(
        {"type_id": list(range(n_types)), "prior_dem_share": vals}
    )
    results = validate_per_year(type_year, model_priors, min_respondents=1)
    assert len(results) == 2
    r_2016 = next(r for r in results if r.comparison_year == 2016)
    r_2020 = next(r for r in results if r.comparison_year == 2020)
    assert r_2016.bias == pytest.approx(0.0, abs=1e-4)
    assert r_2020.bias == pytest.approx(0.05, abs=1e-4)


# ---------------------------------------------------------------------------
# Integration: end-to-end with actual data files (if available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("data/raw/ces/cumulative_2006-2024.feather").exists()
    or not Path("data/communities/county_type_assignments_full.parquet").exists(),
    reason="CES or type assignment data not available on disk",
)
def test_full_pipeline_produces_valid_correlation() -> None:
    """
    Integration test: full pipeline should produce Pearson r >= 0.70 against
    model type priors. A lower value would indicate a structural problem.
    """
    from src.validation.validate_ces import run_validation
    results, comparison = run_validation()

    # Sanity checks on outputs
    assert results.pearson_r >= 0.70, f"Expected r >= 0.70, got {results.pearson_r}"
    assert results.n_types >= 50, f"Expected >= 50 types with data, got {results.n_types}"
    assert results.n_respondents >= 100_000, f"Expected >= 100K respondents, got {results.n_respondents}"
    assert 0.3 <= results.ces_dem_share_mean <= 0.6, "CES mean D-share outside expected range"
    assert len(comparison) == results.n_types
    assert "ces_dem_share_mean" in comparison.columns
    assert "prior_dem_share" in comparison.columns

    # Output files should exist
    assert OUTPUT_JSON.exists()
    assert OUTPUT_CSV.exists()

    # JSON should be valid and contain key fields
    with open(OUTPUT_JSON) as f:
        summary = json.load(f)
    assert "pearson_r" in summary
    assert "rmse" in summary
    assert summary["pearson_r"] >= 0.70


# Import for integration test
from src.validation.validate_ces import OUTPUT_JSON, OUTPUT_CSV  # noqa: E402
