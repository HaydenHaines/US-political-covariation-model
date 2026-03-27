"""Tests for SCI-type validation module.

Verifies:
- SCI loading correctly deduplicates symmetric pairs
- Type similarity computation produces valid cosine similarities
- Same-type pairs are correctly identified
- Haversine distance computation is accurate
- Partial correlation residualization is correct
- Full validation pipeline produces coherent results on synthetic data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.validate_sci_types import (
    SCITypeValidationResult,
    add_geodesic_distance,
    compute_partial_correlation,
    compute_pairwise_type_similarity,
    format_results,
    haversine_km,
    load_sci_upper_triangle,
    run_validation,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data generators
# ---------------------------------------------------------------------------


def _make_type_assignments(
    n_counties: int = 20,
    n_types: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create synthetic type assignments.

    Counties are grouped: first n_counties/n_types share type 0, next share
    type 1, etc. Soft scores are concentrated on the dominant type with noise.
    """
    rng = np.random.default_rng(seed)
    fips_list = [f"{i:05d}" for i in range(1, n_counties + 1)]
    group_size = n_counties // n_types

    # Build soft score matrix: dominant type gets ~0.8, rest split ~0.2
    score_matrix = rng.uniform(0.01, 0.05, size=(n_counties, n_types))
    dominant_types = np.zeros(n_counties, dtype=int)
    for t in range(n_types):
        start = t * group_size
        end = start + group_size if t < n_types - 1 else n_counties
        score_matrix[start:end, t] = rng.uniform(0.7, 0.9, size=end - start)
        dominant_types[start:end] = t

    # Normalize rows to sum to 1
    row_sums = score_matrix.sum(axis=1, keepdims=True)
    score_matrix = score_matrix / row_sums

    score_cols = [f"type_{t}_score" for t in range(n_types)]
    df = pd.DataFrame(score_matrix, columns=score_cols)
    df["county_fips"] = fips_list
    df["dominant_type"] = dominant_types
    df["super_type"] = dominant_types % 3

    return df, score_matrix, dominant_types


def _make_sci_pairs(
    fips_list: list[str],
    dominant_types: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic SCI pairs where same-type pairs have higher SCI.

    This encodes the expected signal: socially connected counties share types.
    """
    rng = np.random.default_rng(seed)
    fips_to_type = dict(zip(fips_list, dominant_types))
    rows = []
    for i, u in enumerate(fips_list):
        for j, f in enumerate(fips_list):
            if i >= j:  # upper triangle only
                continue
            same_type = fips_to_type[u] == fips_to_type[f]
            # Same-type pairs get ~10x higher SCI on average
            base = 10000 if same_type else 1000
            sci = max(1, int(rng.normal(base, base * 0.3)))
            rows.append({
                "user_fips": u,
                "friend_fips": f,
                "scaled_sci": sci,
            })
    return pd.DataFrame(rows)


def _make_centroids(fips_list: list[str], seed: int = 42) -> pd.DataFrame:
    """Create synthetic county centroids spread across the US."""
    rng = np.random.default_rng(seed)
    n = len(fips_list)
    return pd.DataFrame({
        "county_fips": fips_list,
        "latitude": rng.uniform(25, 48, size=n),
        "longitude": rng.uniform(-125, -70, size=n),
    })


def _make_sci_csv_file(tmp_path, fips_list, dominant_types, seed=42):
    """Write a synthetic SCI CSV file in the original format (with duplicates)."""
    rng = np.random.default_rng(seed)
    fips_to_type = dict(zip(fips_list, dominant_types))
    rows = []
    for i, u in enumerate(fips_list):
        for j, f in enumerate(fips_list):
            if i == j:
                continue
            same_type = fips_to_type[u] == fips_to_type[f]
            base = 10000 if same_type else 1000
            sci = max(1, int(rng.normal(base, base * 0.3)))
            # Write both directions (symmetric)
            rows.append({
                "user_country": "US",
                "friend_country": "US",
                "user_region": int(u),
                "friend_region": int(f),
                "scaled_sci": sci,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "test_sci.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests: haversine_km
# ---------------------------------------------------------------------------


class TestHaversineKm:
    """Tests for the haversine distance function."""

    def test_same_point_is_zero(self):
        """Distance from a point to itself should be ~0."""
        d = haversine_km(
            np.array([40.0]), np.array([-74.0]),
            np.array([40.0]), np.array([-74.0]),
        )
        assert abs(d[0]) < 0.01

    def test_known_distance_nyc_la(self):
        """NYC (40.7, -74.0) to LA (34.05, -118.25) is ~3944 km."""
        d = haversine_km(
            np.array([40.7]), np.array([-74.0]),
            np.array([34.05]), np.array([-118.25]),
        )
        assert 3900 < d[0] < 4000

    def test_vectorized(self):
        """Should handle arrays of multiple points."""
        lats1 = np.array([40.7, 34.05])
        lons1 = np.array([-74.0, -118.25])
        lats2 = np.array([34.05, 40.7])
        lons2 = np.array([-118.25, -74.0])
        d = haversine_km(lats1, lons1, lats2, lons2)
        assert len(d) == 2
        # Both should be the same distance (symmetric)
        assert abs(d[0] - d[1]) < 1.0

    def test_antipodal_points(self):
        """Opposite sides of the Earth should be ~20,000 km."""
        d = haversine_km(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([180.0]),
        )
        assert 19_900 < d[0] < 20_100


# ---------------------------------------------------------------------------
# Tests: load_sci_upper_triangle
# ---------------------------------------------------------------------------


class TestLoadSciUpperTriangle:
    """Tests for SCI loading and deduplication."""

    def test_upper_triangle_deduplication(self, tmp_path):
        """Loading should keep only upper triangle (user < friend)."""
        assignments, _, dominant_types = _make_type_assignments(n_counties=6, n_types=2)
        fips_list = assignments["county_fips"].tolist()
        csv_path = _make_sci_csv_file(tmp_path, fips_list, dominant_types)

        result = load_sci_upper_triangle(csv_path)
        # Upper triangle of 6 counties = 6*5/2 = 15 pairs
        assert len(result) == 15
        # Verify all pairs have user < friend
        assert (result["user_fips"] < result["friend_fips"]).all()

    def test_fips_filter(self, tmp_path):
        """Should filter to only valid FIPS when provided."""
        assignments, _, dominant_types = _make_type_assignments(n_counties=6, n_types=2)
        fips_list = assignments["county_fips"].tolist()
        csv_path = _make_sci_csv_file(tmp_path, fips_list, dominant_types)

        # Only keep first 3 counties
        valid = set(fips_list[:3])
        result = load_sci_upper_triangle(csv_path, valid_fips=valid)
        # 3 counties = 3*2/2 = 3 pairs
        assert len(result) == 3

    def test_no_self_connections(self, tmp_path):
        """Self-connections should be excluded."""
        assignments, _, dominant_types = _make_type_assignments(n_counties=4, n_types=2)
        fips_list = assignments["county_fips"].tolist()
        csv_path = _make_sci_csv_file(tmp_path, fips_list, dominant_types)

        result = load_sci_upper_triangle(csv_path)
        assert (result["user_fips"] != result["friend_fips"]).all()

    def test_fips_zero_padded(self, tmp_path):
        """FIPS codes should be zero-padded to 5 characters."""
        assignments, _, dominant_types = _make_type_assignments(n_counties=4, n_types=2)
        fips_list = assignments["county_fips"].tolist()
        csv_path = _make_sci_csv_file(tmp_path, fips_list, dominant_types)

        result = load_sci_upper_triangle(csv_path)
        assert all(len(f) == 5 for f in result["user_fips"])
        assert all(len(f) == 5 for f in result["friend_fips"])


# ---------------------------------------------------------------------------
# Tests: compute_pairwise_type_similarity
# ---------------------------------------------------------------------------


class TestPairwiseTypeSimilarity:
    """Tests for type similarity computation."""

    def test_cosine_sim_range(self):
        """Cosine similarity should be in [0, 1] for non-negative score vectors."""
        assignments, score_matrix, dominant_types = _make_type_assignments()
        fips_list = assignments["county_fips"].tolist()
        fips_to_idx = {f: i for i, f in enumerate(fips_list)}
        pairs = _make_sci_pairs(fips_list, dominant_types)

        result = compute_pairwise_type_similarity(
            pairs, fips_to_idx, score_matrix, dominant_types
        )
        assert (result["cosine_sim"] >= -0.01).all()  # allow small float error
        assert (result["cosine_sim"] <= 1.01).all()

    def test_same_type_flag(self):
        """same_type should be True iff both counties share dominant type."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=10, n_types=2
        )
        fips_list = assignments["county_fips"].tolist()
        fips_to_idx = {f: i for i, f in enumerate(fips_list)}
        pairs = _make_sci_pairs(fips_list, dominant_types)

        result = compute_pairwise_type_similarity(
            pairs, fips_to_idx, score_matrix, dominant_types
        )

        # Verify same_type matches dominant type comparison
        for _, row in result.iterrows():
            u_type = dominant_types[fips_to_idx[row["user_fips"]]]
            f_type = dominant_types[fips_to_idx[row["friend_fips"]]]
            assert row["same_type"] == (u_type == f_type)

    def test_same_county_has_cosine_sim_one(self):
        """A county paired with itself should have cosine_sim == 1."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=4, n_types=2
        )
        fips_list = assignments["county_fips"].tolist()
        fips_to_idx = {f: i for i, f in enumerate(fips_list)}

        # Manually create a "self-pair" (this wouldn't happen in practice)
        self_pair = pd.DataFrame([{
            "user_fips": fips_list[0],
            "friend_fips": fips_list[0],
            "scaled_sci": 999999,
        }])
        result = compute_pairwise_type_similarity(
            self_pair, fips_to_idx, score_matrix, dominant_types
        )
        assert abs(result["cosine_sim"].iloc[0] - 1.0) < 1e-6

    def test_log_sci_computed(self):
        """log_sci should be log10 of scaled_sci."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=6, n_types=2
        )
        fips_list = assignments["county_fips"].tolist()
        fips_to_idx = {f: i for i, f in enumerate(fips_list)}
        pairs = _make_sci_pairs(fips_list, dominant_types)

        result = compute_pairwise_type_similarity(
            pairs, fips_to_idx, score_matrix, dominant_types
        )
        expected = np.log10(np.clip(result["scaled_sci"].values, 1, None))
        np.testing.assert_allclose(result["log_sci"].values, expected, rtol=1e-6)

    def test_same_state_flag(self):
        """same_state should be True when first 2 digits of FIPS match."""
        # Use FIPS from same state (01xxx)
        assignments = pd.DataFrame({
            "county_fips": ["01001", "01003", "12001", "12003"],
            "type_0_score": [0.8, 0.8, 0.2, 0.2],
            "type_1_score": [0.2, 0.2, 0.8, 0.8],
            "dominant_type": [0, 0, 1, 1],
            "super_type": [0, 0, 1, 1],
        })
        score_matrix = assignments[["type_0_score", "type_1_score"]].values
        dominant_types = assignments["dominant_type"].values
        fips_list = assignments["county_fips"].tolist()
        fips_to_idx = {f: i for i, f in enumerate(fips_list)}

        pairs = pd.DataFrame([
            {"user_fips": "01001", "friend_fips": "01003", "scaled_sci": 1000},
            {"user_fips": "01001", "friend_fips": "12001", "scaled_sci": 500},
        ])

        result = compute_pairwise_type_similarity(
            pairs, fips_to_idx, score_matrix, dominant_types
        )
        assert result.iloc[0]["same_state"] is True or result.iloc[0]["same_state"] == True
        assert result.iloc[1]["same_state"] is False or result.iloc[1]["same_state"] == False


# ---------------------------------------------------------------------------
# Tests: add_geodesic_distance
# ---------------------------------------------------------------------------


class TestAddGeodesicDistance:
    """Tests for distance enrichment."""

    def test_distance_added(self):
        """Pairs should have distance_km and log_distance columns after enrichment."""
        fips_list = ["01001", "01003", "12001"]
        centroids = pd.DataFrame({
            "county_fips": fips_list,
            "latitude": [32.5, 30.7, 29.7],
            "longitude": [-86.5, -87.7, -82.3],
        })
        pairs = pd.DataFrame([
            {"user_fips": "01001", "friend_fips": "01003", "scaled_sci": 1000},
            {"user_fips": "01001", "friend_fips": "12001", "scaled_sci": 500},
        ])

        result = add_geodesic_distance(pairs, centroids)
        assert "distance_km" in result.columns
        assert "log_distance" in result.columns
        assert len(result) == 2
        assert (result["distance_km"] > 0).all()

    def test_missing_centroids_dropped(self):
        """Pairs with missing centroids should be dropped."""
        centroids = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "latitude": [32.5, 30.7],
            "longitude": [-86.5, -87.7],
        })
        pairs = pd.DataFrame([
            {"user_fips": "01001", "friend_fips": "01003", "scaled_sci": 1000},
            {"user_fips": "01001", "friend_fips": "99999", "scaled_sci": 500},
        ])

        result = add_geodesic_distance(pairs, centroids)
        assert len(result) == 1  # second pair dropped


# ---------------------------------------------------------------------------
# Tests: compute_partial_correlation
# ---------------------------------------------------------------------------


class TestPartialCorrelation:
    """Tests for partial correlation computation."""

    def test_uncorrelated_control(self):
        """When z is uncorrelated with both x and y, partial r ≈ raw r."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n) * 0.5
        z = rng.normal(size=n)  # independent of both

        raw_r, _ = np.corrcoef(x, y)[0, 1], None
        partial_r, partial_p = compute_partial_correlation(x, y, z)

        # Should be close to raw correlation
        assert abs(partial_r - np.corrcoef(x, y)[0, 1]) < 0.05
        assert partial_p < 0.001

    def test_confounded_relationship(self):
        """When x-y correlation is entirely due to z, partial r should be ~0."""
        rng = np.random.default_rng(42)
        n = 2000
        z = rng.normal(size=n)
        x = z + rng.normal(size=n) * 0.3
        y = z + rng.normal(size=n) * 0.3

        # Raw correlation should be high
        raw_r = np.corrcoef(x, y)[0, 1]
        assert raw_r > 0.5

        # Partial correlation should be near zero
        partial_r, _ = compute_partial_correlation(x, y, z)
        assert abs(partial_r) < 0.1

    def test_partial_r_range(self):
        """Partial correlation should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        z = rng.normal(size=n)
        r, _ = compute_partial_correlation(x, y, z)
        assert -1.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Tests: run_validation (integration)
# ---------------------------------------------------------------------------


class TestRunValidation:
    """Integration tests for the full validation pipeline."""

    def test_basic_pipeline(self):
        """Pipeline should complete and return populated result object."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=20, n_types=4
        )
        fips_list = assignments["county_fips"].tolist()
        pairs = _make_sci_pairs(fips_list, dominant_types)

        result = run_validation(
            sci_pairs=pairs,
            type_assignments=assignments,
            score_matrix=score_matrix,
            dominant_types=dominant_types,
        )

        assert result.n_counties == 20
        assert result.n_types == 4
        assert result.n_pairs == 20 * 19 // 2  # 190

    def test_same_type_higher_sci(self):
        """With synthetic data where same-type SCI > diff-type, ratio should be > 1."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=20, n_types=4
        )
        fips_list = assignments["county_fips"].tolist()
        pairs = _make_sci_pairs(fips_list, dominant_types)

        result = run_validation(
            sci_pairs=pairs,
            type_assignments=assignments,
            score_matrix=score_matrix,
            dominant_types=dominant_types,
        )

        # Our synthetic data encodes 10x SCI for same-type pairs
        assert result.sci_ratio_same_over_diff > 2.0

    def test_positive_correlation(self):
        """With signal-bearing synthetic data, correlations should be positive."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=30, n_types=5
        )
        fips_list = assignments["county_fips"].tolist()
        pairs = _make_sci_pairs(fips_list, dominant_types)

        result = run_validation(
            sci_pairs=pairs,
            type_assignments=assignments,
            score_matrix=score_matrix,
            dominant_types=dominant_types,
        )

        assert result.pearson_r_log_sci_vs_cosine > 0.1
        assert result.spearman_r_log_sci_vs_cosine > 0.1

    def test_with_centroids(self):
        """Pipeline should work with distance controls."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=20, n_types=4
        )
        fips_list = assignments["county_fips"].tolist()
        pairs = _make_sci_pairs(fips_list, dominant_types)
        centroids = _make_centroids(fips_list)

        result = run_validation(
            sci_pairs=pairs,
            type_assignments=assignments,
            score_matrix=score_matrix,
            dominant_types=dominant_types,
            centroids=centroids,
        )

        assert result.n_pairs > 0
        # Partial correlation should be computed
        # (may not be strongly positive with random centroids, but should run)
        assert result.partial_r_sci_cosine_given_distance != 0.0

    def test_sample_size_respected(self):
        """When sample_size is set, correlations should still be computed."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=20, n_types=4
        )
        fips_list = assignments["county_fips"].tolist()
        pairs = _make_sci_pairs(fips_list, dominant_types)

        result = run_validation(
            sci_pairs=pairs,
            type_assignments=assignments,
            score_matrix=score_matrix,
            dominant_types=dominant_types,
            sample_size=50,
        )

        # Should still produce valid results
        assert result.pearson_r_log_sci_vs_cosine != 0.0

    def test_distance_bin_results(self):
        """Distance binning should produce results when centroids are provided."""
        assignments, score_matrix, dominant_types = _make_type_assignments(
            n_counties=30, n_types=5
        )
        fips_list = assignments["county_fips"].tolist()
        pairs = _make_sci_pairs(fips_list, dominant_types)
        centroids = _make_centroids(fips_list)

        result = run_validation(
            sci_pairs=pairs,
            type_assignments=assignments,
            score_matrix=score_matrix,
            dominant_types=dominant_types,
            centroids=centroids,
        )

        # Should have at least one distance bin with data
        assert len(result.distance_bin_results) > 0
        for bin_result in result.distance_bin_results:
            assert bin_result["n_pairs"] > 0
            assert "sci_ratio" in bin_result


# ---------------------------------------------------------------------------
# Tests: format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    """Tests for report formatting."""

    def test_produces_markdown(self):
        """format_results should produce valid markdown with key sections."""
        result = SCITypeValidationResult(
            n_counties=3154,
            n_pairs=4_500_000,
            n_types=100,
            mean_sci_same_type=5000,
            mean_sci_diff_type=2000,
            sci_ratio_same_over_diff=2.5,
            mean_log_sci_same_type=3.7,
            mean_log_sci_diff_type=3.3,
            pearson_r_log_sci_vs_cosine=0.15,
            pearson_p_log_sci_vs_cosine=1e-30,
            spearman_r_log_sci_vs_cosine=0.18,
            spearman_p_log_sci_vs_cosine=1e-35,
            pct_same_type_same_state=0.05,
            pct_same_type_diff_state=0.02,
            mean_sci_same_type_diff_state=3000,
            mean_sci_diff_type_diff_state=1500,
            sci_ratio_same_over_diff_across_states=2.0,
            partial_r_sci_cosine_given_distance=0.12,
            partial_p_sci_cosine_given_distance=1e-20,
        )

        report = format_results(result)
        assert "# SCI-Type Validation Results" in report
        assert "Research Question" in report
        assert "3,154" in report
        assert "Partial r" in report
        assert "0.12" in report

    def test_no_distance_section_without_data(self):
        """When partial_r is 0, the distance section should be omitted."""
        result = SCITypeValidationResult(
            n_counties=100,
            n_pairs=1000,
            n_types=10,
            mean_sci_same_type=5000,
            mean_sci_diff_type=2000,
            sci_ratio_same_over_diff=2.5,
        )

        report = format_results(result)
        assert "Finding 4" not in report
        assert "Finding 5" not in report
