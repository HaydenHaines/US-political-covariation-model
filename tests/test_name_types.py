"""Tests for the type naming pipeline (src/description/name_types.py).

Covers:
  - compute_zscores(): correct weighted z-scores on synthetic data
  - name_types(): proper DataFrame structure, uniqueness, word count
  - Determinism: same input always produces same output
  - Edge cases: single type, no county_assignments
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.description.name_types import compute_zscores, name_types, name_super_types


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_profiles(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Build a minimal type_profiles DataFrame with n types."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "type_id": list(range(n)),
            "n_counties": rng.integers(2, 20, size=n).tolist(),
            "pop_total": (rng.uniform(1e4, 1e6, size=n)).tolist(),
            "pct_white_nh": rng.uniform(0.2, 0.9, size=n).tolist(),
            "pct_black": rng.uniform(0.02, 0.6, size=n).tolist(),
            "pct_hispanic": rng.uniform(0.02, 0.5, size=n).tolist(),
            "pct_asian": rng.uniform(0.005, 0.15, size=n).tolist(),
            "median_age": rng.uniform(34, 52, size=n).tolist(),
            "median_hh_income": rng.uniform(28000, 110000, size=n).tolist(),
            "pct_bachelors_plus": rng.uniform(0.08, 0.55, size=n).tolist(),
            "pct_graduate": rng.uniform(0.04, 0.30, size=n).tolist(),
            "pct_management": rng.uniform(0.20, 0.55, size=n).tolist(),
            "evangelical_share": rng.uniform(0.25, 0.85, size=n).tolist(),
            "religious_adherence_rate": rng.uniform(300, 700, size=n).tolist(),
            "log_pop_density": rng.uniform(0.8, 3.2, size=n).tolist(),
            "pct_wfh": rng.uniform(0.01, 0.18, size=n).tolist(),
            "pct_car": rng.uniform(0.60, 0.92, size=n).tolist(),
            "pct_owner_occupied": rng.uniform(0.45, 0.78, size=n).tolist(),
            "net_migration_rate": rng.uniform(-0.15, 0.85, size=n).tolist(),
            "inflow_outflow_ratio": rng.uniform(0.45, 0.85, size=n).tolist(),
        }
    )


def _make_county_assignments(type_ids: list[int], super_ids: list[int]) -> pd.DataFrame:
    """Build a county_assignments stub with dominant_type and super_type."""
    assert len(type_ids) == len(super_ids)
    return pd.DataFrame(
        {
            "county_fips": [f"1200{i}" for i in range(len(type_ids))],
            "dominant_type": type_ids,
            "super_type": super_ids,
        }
    )


# ---------------------------------------------------------------------------
# Tests: compute_zscores
# ---------------------------------------------------------------------------


class TestComputeZscores:
    def test_output_has_type_id_column(self):
        profiles = _make_profiles(5)
        z = compute_zscores(profiles, ["pct_white_nh"])
        assert "type_id" in z.columns

    def test_output_has_feature_column(self):
        profiles = _make_profiles(5)
        z = compute_zscores(profiles, ["pct_white_nh", "median_age"])
        assert "pct_white_nh" in z.columns
        assert "median_age" in z.columns

    def test_row_count_matches_input(self):
        profiles = _make_profiles(8)
        z = compute_zscores(profiles, ["pct_white_nh"])
        assert len(z) == 8

    def test_missing_feature_skipped(self):
        """Features not in profiles are silently skipped."""
        profiles = _make_profiles(4)
        z = compute_zscores(profiles, ["pct_white_nh", "nonexistent_col"])
        assert "pct_white_nh" in z.columns
        assert "nonexistent_col" not in z.columns

    def test_weighted_mean_is_zero(self):
        """Pop-weighted mean of z-scores must be zero (by construction)."""
        profiles = _make_profiles(10, seed=7)
        z = compute_zscores(profiles, ["pct_white_nh", "median_age"])
        weights = profiles["pop_total"].values
        for feat in ["pct_white_nh", "median_age"]:
            wmean = float(np.dot(weights, z[feat].values) / weights.sum())
            assert abs(wmean) < 1e-9, f"Weighted mean non-zero for {feat}: {wmean}"

    def test_weighted_std_is_one(self):
        """Pop-weighted variance of z-scores must be 1.0."""
        profiles = _make_profiles(10, seed=13)
        z = compute_zscores(profiles, ["pct_white_nh"])
        weights = profiles["pop_total"].values
        vals = z["pct_white_nh"].values
        wmean = float(np.dot(weights, vals) / weights.sum())
        wvar = float(np.dot(weights, (vals - wmean) ** 2) / weights.sum())
        assert abs(wvar - 1.0) < 1e-9, f"Weighted variance is {wvar}, expected 1.0"

    def test_known_values_two_types(self):
        """Hand-verified z-score for a 2-type system."""
        profiles = pd.DataFrame(
            {
                "type_id": [0, 1],
                "pop_total": [100.0, 100.0],  # equal weights
                "pct_white_nh": [0.2, 0.8],
            }
        )
        z = compute_zscores(profiles, ["pct_white_nh"])
        # With equal weights: mean = 0.5, var = 0.09, std = 0.3
        # z[0] = (0.2-0.5)/0.3 = -1.0, z[1] = (0.8-0.5)/0.3 = +1.0
        assert abs(z.loc[z["type_id"] == 0, "pct_white_nh"].iloc[0] - (-1.0)) < 1e-9
        assert abs(z.loc[z["type_id"] == 1, "pct_white_nh"].iloc[0] - 1.0) < 1e-9

    def test_known_values_population_weighted(self):
        """Heavier-weight type pulls mean closer to it."""
        profiles = pd.DataFrame(
            {
                "type_id": [0, 1],
                "pop_total": [900.0, 100.0],  # type 0 has 9x weight
                "pct_white_nh": [0.5, 0.9],
            }
        )
        z = compute_zscores(profiles, ["pct_white_nh"])
        weights = np.array([900.0, 100.0])
        total = weights.sum()
        vals = np.array([0.5, 0.9])
        wmean = float(np.dot(weights, vals) / total)   # = 0.54
        wvar = float(np.dot(weights, (vals - wmean) ** 2) / total)
        wstd = float(np.sqrt(wvar))
        expected_z0 = (0.5 - wmean) / wstd
        expected_z1 = (0.9 - wmean) / wstd
        actual_z0 = float(z.loc[z["type_id"] == 0, "pct_white_nh"].iloc[0])
        actual_z1 = float(z.loc[z["type_id"] == 1, "pct_white_nh"].iloc[0])
        assert abs(actual_z0 - expected_z0) < 1e-9
        assert abs(actual_z1 - expected_z1) < 1e-9

    def test_all_same_values_returns_zero(self):
        """When all types have the same value, z-score should be 0 (not NaN)."""
        profiles = pd.DataFrame(
            {
                "type_id": [0, 1, 2],
                "pop_total": [100.0, 200.0, 300.0],
                "pct_white_nh": [0.5, 0.5, 0.5],
            }
        )
        z = compute_zscores(profiles, ["pct_white_nh"])
        assert (z["pct_white_nh"] == 0.0).all()


# ---------------------------------------------------------------------------
# Tests: name_types return structure
# ---------------------------------------------------------------------------


class TestNameTypesStructure:
    def test_returns_dataframe(self):
        profiles = _make_profiles(5)
        result = name_types(profiles=profiles)
        assert isinstance(result, pd.DataFrame)

    def test_has_type_id_column(self):
        profiles = _make_profiles(5)
        result = name_types(profiles=profiles)
        assert "type_id" in result.columns

    def test_has_display_name_column(self):
        profiles = _make_profiles(5)
        result = name_types(profiles=profiles)
        assert "display_name" in result.columns

    def test_row_count_matches_types(self):
        """One row per type in the profiles."""
        for n in [1, 5, 10, 43, 55]:
            profiles = _make_profiles(n)
            result = name_types(profiles=profiles)
            assert len(result) == n, f"Expected {n} rows, got {len(result)}"

    def test_all_type_ids_present(self):
        """Every type_id 0..n-1 must appear in the output."""
        n = 12
        profiles = _make_profiles(n)
        result = name_types(profiles=profiles)
        assert set(result["type_id"].tolist()) == set(range(n))

    def test_sorted_by_type_id(self):
        """Output should be sorted ascending by type_id."""
        profiles = _make_profiles(8)
        result = name_types(profiles=profiles)
        assert result["type_id"].tolist() == sorted(result["type_id"].tolist())

    def test_no_null_names(self):
        profiles = _make_profiles(10)
        result = name_types(profiles=profiles)
        assert result["display_name"].notna().all()

    def test_no_empty_names(self):
        profiles = _make_profiles(10)
        result = name_types(profiles=profiles)
        assert (result["display_name"].str.strip() != "").all()


# ---------------------------------------------------------------------------
# Tests: uniqueness
# ---------------------------------------------------------------------------


class TestNameTypesUniqueness:
    def test_all_names_unique_small(self):
        """5 types should all get unique names."""
        profiles = _make_profiles(5, seed=0)
        result = name_types(profiles=profiles)
        assert result["display_name"].nunique() == len(result)

    def test_all_names_unique_medium(self):
        """20 types should all get unique names."""
        profiles = _make_profiles(20, seed=1)
        result = name_types(profiles=profiles)
        assert result["display_name"].nunique() == len(result), (
            f"Duplicates: {result[result['display_name'].duplicated(keep=False)][['type_id','display_name']].to_string()}"
        )

    def test_all_names_unique_full_43(self):
        """Full 43-type synthetic dataset — all names unique."""
        profiles = _make_profiles(43, seed=99)
        result = name_types(profiles=profiles)
        assert result["display_name"].nunique() == 43, (
            f"Duplicates found: {result[result['display_name'].duplicated(keep=False)][['type_id','display_name']].to_string()}"
        )

    def test_all_names_unique_full_55(self):
        """Full 55-type synthetic dataset (national model) — all names unique."""
        profiles = _make_profiles(55, seed=42)
        result = name_types(profiles=profiles)
        assert result["display_name"].nunique() == 55, (
            f"Duplicates found: {result[result['display_name'].duplicated(keep=False)][['type_id','display_name']].to_string()}"
        )


# ---------------------------------------------------------------------------
# Tests: word count (2–4 words)
# ---------------------------------------------------------------------------


class TestNameTypesWordCount:
    def _word_count(self, name: str) -> int:
        return len(name.split())

    def test_names_have_at_least_2_words(self):
        profiles = _make_profiles(15, seed=5)
        result = name_types(profiles=profiles)
        for _, row in result.iterrows():
            wc = self._word_count(row["display_name"])
            assert wc >= 2, f"Type {row.type_id} name too short: '{row.display_name}'"

    def test_names_have_at_most_4_words(self):
        profiles = _make_profiles(55, seed=7)
        result = name_types(profiles=profiles)
        for _, row in result.iterrows():
            wc = self._word_count(row["display_name"])
            assert wc <= 4, f"Type {row.type_id} name too long: '{row.display_name}'"


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


class TestNameTypesDeterminism:
    def test_same_input_same_output(self):
        """Calling name_types twice on identical input must give identical results."""
        profiles = _make_profiles(20, seed=42)
        result1 = name_types(profiles=profiles.copy())
        result2 = name_types(profiles=profiles.copy())
        assert list(result1["display_name"]) == list(result2["display_name"])

    def test_deterministic_across_different_seeds(self):
        """Two different-seeded datasets each produce stable (deterministic) output."""
        for seed in [0, 13, 77]:
            p = _make_profiles(10, seed=seed)
            r1 = name_types(profiles=p.copy())
            r2 = name_types(profiles=p.copy())
            assert list(r1["display_name"]) == list(r2["display_name"])


# ---------------------------------------------------------------------------
# Tests: super_type context used when provided
# ---------------------------------------------------------------------------


class TestNameTypesSuperContext:
    def test_super_type_context_accepted(self):
        """When county_assignments provides super_type, function runs without error."""
        profiles = _make_profiles(5)
        ca = _make_county_assignments(
            type_ids=[0, 1, 2, 3, 4],
            super_ids=[0, 1, 2, 3, 4],
        )
        result = name_types(profiles=profiles, county_assignments=ca)
        assert len(result) == 5

    def test_no_county_assignments_still_works(self):
        """Works when county_assignments is None (falls back to super_id=0)."""
        profiles = _make_profiles(8)
        result = name_types(profiles=profiles, county_assignments=None)
        assert len(result) == 8
        assert result["display_name"].nunique() == 8


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestNameTypesEdgeCases:
    def test_single_type(self):
        """A DataFrame with a single type must produce exactly one name."""
        profiles = _make_profiles(1)
        result = name_types(profiles=profiles)
        assert len(result) == 1
        assert isinstance(result["display_name"].iloc[0], str)
        assert len(result["display_name"].iloc[0]) > 0

    def test_two_types(self):
        profiles = _make_profiles(2, seed=3)
        result = name_types(profiles=profiles)
        assert len(result) == 2
        assert result["display_name"].nunique() == 2

    def test_profiles_with_minimal_columns(self):
        """Works when only type_id and pop_total are present."""
        profiles = pd.DataFrame(
            {
                "type_id": [0, 1, 2],
                "pop_total": [10000.0, 20000.0, 30000.0],
            }
        )
        result = name_types(profiles=profiles)
        assert len(result) == 3
        assert result["display_name"].nunique() == 3

    def test_no_disk_write_when_profiles_provided(self, tmp_path, monkeypatch):
        """When profiles DataFrame is passed directly, no file is written."""
        import src.description.name_types as mod

        written: list[str] = []

        original_to_parquet = pd.DataFrame.to_parquet

        def mock_to_parquet(self, path, *args, **kwargs):
            written.append(str(path))
            original_to_parquet(self, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)

        profiles = _make_profiles(5)
        name_types(profiles=profiles)  # profiles passed directly → no disk write
        assert len(written) == 0, f"Unexpected disk writes: {written}"


# ---------------------------------------------------------------------------
# Tests: name_super_types
# ---------------------------------------------------------------------------


def _make_super_assignments(n_types: int, n_supers: int = 5) -> pd.DataFrame:
    """Build a county_assignments stub with super_type mapping."""
    rng = np.random.default_rng(0)
    n_counties = n_types * 3
    dominant = [i % n_types for i in range(n_counties)]
    super_t = [i % n_supers for i in range(n_counties)]
    return pd.DataFrame(
        {
            "county_fips": [f"{12000 + i:05d}" for i in range(n_counties)],
            "dominant_type": dominant,
            "super_type": super_t,
        }
    )


class TestNameSuperTypes:
    def test_returns_dataframe(self):
        profiles = _make_profiles(10)
        ca = _make_super_assignments(10, n_supers=5)
        result = name_super_types(profiles=profiles, county_assignments=ca)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        profiles = _make_profiles(10)
        ca = _make_super_assignments(10, n_supers=5)
        result = name_super_types(profiles=profiles, county_assignments=ca)
        assert "super_type_id" in result.columns
        assert "display_name" in result.columns

    def test_row_count_matches_n_supers(self):
        profiles = _make_profiles(10)
        ca = _make_super_assignments(10, n_supers=5)
        result = name_super_types(profiles=profiles, county_assignments=ca)
        assert len(result) == 5

    def test_all_names_unique(self):
        profiles = _make_profiles(15)
        ca = _make_super_assignments(15, n_supers=5)
        result = name_super_types(profiles=profiles, county_assignments=ca)
        assert result["display_name"].nunique() == len(result), (
            f"Duplicate super-type names: {result.to_string()}"
        )

    def test_no_null_or_empty_names(self):
        profiles = _make_profiles(10)
        ca = _make_super_assignments(10, n_supers=5)
        result = name_super_types(profiles=profiles, county_assignments=ca)
        assert result["display_name"].notna().all()
        assert (result["display_name"].str.strip() != "").all()

    def test_returns_empty_on_missing_data(self):
        """Returns empty DataFrame when county_assignments is None."""
        profiles = _make_profiles(5)
        result = name_super_types(profiles=profiles, county_assignments=None)
        assert isinstance(result, pd.DataFrame)
        assert "super_type_id" in result.columns
