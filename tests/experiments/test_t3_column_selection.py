"""Tests for T3 column selection experiment (Phase T.3 tract-level type discovery).

Tests focus on the core computation functions, not the full KMeans run
(which is expensive and non-deterministic within the test suite).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.t3_column_selection import (
    PRES_TRAIN_COLS,
    PRES_HOLDOUT_COL,
    PRESIDENTIAL_WEIGHT,
    GOV_COLS,
    SEN_COLS,
    compute_aggregated_offcycle,
    compute_holdout_r,
    prepare_matrix,
    type_size_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic shift dataframe with pres + off-cycle columns."""
    rng = np.random.default_rng(seed)
    data = {
        "tract_geoid": [f"tract_{i:04d}" for i in range(n)],
        "pres_shift_08_12": rng.normal(0, 0.05, n),
        "pres_shift_12_16": rng.normal(0, 0.05, n),
        "pres_shift_16_20": rng.normal(0, 0.05, n),
        "pres_shift_20_24": rng.normal(0, 0.05, n),
        "gov_shift_18_22_centered": rng.normal(0, 0.05, n),
        "sen_shift_16_18_centered": rng.normal(0, 0.05, n),
        "mean_gov_shift": rng.normal(0, 0.05, n),
        "mean_sen_shift": rng.normal(0, 0.05, n),
    }
    return pd.DataFrame(data)


def _make_df_with_nans(n: int = 30, seed: int = 1) -> pd.DataFrame:
    """Dataframe with realistic NaN patterns."""
    rng = np.random.default_rng(seed)
    df = _make_df(n, seed)

    # Make ~20% of tracts missing all training pres data (should be dropped)
    drop_idx = rng.choice(n, size=max(1, n // 5), replace=False)
    for col in PRES_TRAIN_COLS:
        df.loc[drop_idx, col] = np.nan

    # Make ~50% of off-cycle columns NaN (sparse, should be filled with 0)
    for col in ["gov_shift_18_22_centered", "sen_shift_16_18_centered"]:
        if col in df.columns:
            sparse_idx = rng.choice(n, size=n // 2, replace=False)
            df.loc[sparse_idx, col] = np.nan

    # Make ~30% of holdout NaN
    holdout_nan_idx = rng.choice(n, size=n // 3, replace=False)
    df.loc[holdout_nan_idx, PRES_HOLDOUT_COL] = np.nan

    return df


# ── Tests: prepare_matrix ────────────────────────────────────────────────────

class TestPrepareMatrix:
    def test_drops_all_pres_nan_tracts(self) -> None:
        """Tracts where all pres training cols are NaN should be dropped."""
        df = _make_df(20)
        # Force first 5 tracts to have all NaN in training pres columns
        for col in PRES_TRAIN_COLS:
            df.loc[:4, col] = np.nan

        X, _, _ = prepare_matrix(df, PRES_TRAIN_COLS)
        assert X.shape[0] == 15, f"Expected 15 tracts after dropping 5 all-NaN, got {X.shape[0]}"

    def test_fills_nan_with_zero(self) -> None:
        """NaN in feature columns should be filled with 0 (not propagate as NaN)."""
        df = _make_df(10)
        df.loc[3, "gov_shift_18_22_centered"] = np.nan
        feature_cols = PRES_TRAIN_COLS + ["gov_shift_18_22_centered"]

        X, _, _ = prepare_matrix(df, feature_cols)
        assert not np.isnan(X).any(), "Output matrix should have no NaN"

    def test_presidential_weight_applied(self) -> None:
        """After scaling, pres columns should have larger magnitude than unweighted."""
        # Build a df where all shifts are identical standard normal
        rng = np.random.default_rng(99)
        n = 50
        val = rng.normal(0, 1, n)
        data = {
            "tract_geoid": [f"t{i}" for i in range(n)],
            "pres_shift_08_12": val.copy(),
            "pres_shift_12_16": val.copy(),
            "pres_shift_16_20": val.copy(),
            "pres_shift_20_24": val.copy(),
            "gov_shift_18_22_centered": val.copy(),
        }
        df = pd.DataFrame(data)
        feature_cols = PRES_TRAIN_COLS + ["gov_shift_18_22_centered"]

        X, pres_mask, _ = prepare_matrix(df, feature_cols)

        pres_std = X[:, pres_mask].std()
        gov_std = X[:, ~pres_mask].std()

        # After scaling to unit var then weighting by 8.0, pres should be ~8x gov
        ratio = pres_std / gov_std
        assert 6.0 < ratio < 10.0, (
            f"Pres columns should be ~{PRESIDENTIAL_WEIGHT}x gov after weighting, got ratio={ratio:.2f}"
        )

    def test_output_shape_matches_feature_count(self) -> None:
        """Output matrix columns should match len(feature_cols)."""
        df = _make_df(20)
        feature_cols = PRES_TRAIN_COLS
        X, _, _ = prepare_matrix(df, feature_cols)
        assert X.shape[1] == len(feature_cols)

    def test_holdout_vals_same_length_as_matrix(self) -> None:
        """Holdout values should correspond 1-to-1 with retained tracts."""
        df = _make_df_with_nans(30)
        feature_cols = PRES_TRAIN_COLS
        X, _, holdout_vals = prepare_matrix(df, feature_cols)
        assert len(holdout_vals) == X.shape[0]


# ── Tests: compute_holdout_r ─────────────────────────────────────────────────

class TestComputeHoldoutR:
    def test_perfect_prediction_gives_r_one(self) -> None:
        """If type means exactly recover actual shifts, r should be 1.0."""
        n, j = 100, 5
        rng = np.random.default_rng(7)

        # Hard assignment: each tract belongs fully to one type
        labels = rng.integers(0, j, size=n)
        soft_scores = np.zeros((n, j))
        soft_scores[np.arange(n), labels] = 1.0

        # Actual holdout = type index * 0.1 (perfect type structure)
        type_vals = np.arange(j) * 0.1
        actual = type_vals[labels]

        r = compute_holdout_r(soft_scores, actual)
        assert abs(r - 1.0) < 1e-6, f"Expected r≈1.0, got {r}"

    def test_ignores_nan_holdout_tracts(self) -> None:
        """Tracts with NaN holdout should be excluded from r computation."""
        n, j = 50, 3
        rng = np.random.default_rng(5)
        # Hard assignment so type means are well-defined (not degenerate)
        labels = rng.integers(0, j, size=n)
        soft_scores = np.zeros((n, j))
        soft_scores[np.arange(n), labels] = 1.0

        # Actual holdout: type-based signal so r is non-degenerate
        type_vals = np.array([0.1, 0.2, 0.3])
        actual = type_vals[labels] + rng.normal(0, 0.01, n)
        actual[:10] = np.nan  # first 10 are NaN

        # Should not raise; should only use 40 tracts, r should be finite
        r = compute_holdout_r(soft_scores, actual)
        assert np.isfinite(r), "r should be finite when there are non-NaN holdout tracts"

    def test_returns_float(self) -> None:
        """Output should be a Python float."""
        n, j = 20, 4
        rng = np.random.default_rng(3)
        soft_scores = rng.dirichlet(np.ones(j), size=n)
        actual = rng.normal(0, 0.05, n)
        r = compute_holdout_r(soft_scores, actual)
        assert isinstance(r, float)

    def test_range_is_valid_correlation(self) -> None:
        """r should always be in [-1, 1]."""
        rng = np.random.default_rng(42)
        n, j = 200, 10
        soft_scores = rng.dirichlet(np.ones(j), size=n)
        actual = rng.normal(0, 0.05, n)
        r = compute_holdout_r(soft_scores, actual)
        assert -1.0 <= r <= 1.0, f"r={r} out of range"


# ── Tests: compute_aggregated_offcycle ───────────────────────────────────────

class TestComputeAggregatedOffcycle:
    def test_adds_mean_columns(self) -> None:
        """Should add mean_gov_shift and mean_sen_shift columns."""
        df = pd.DataFrame({
            "tract_geoid": ["a", "b"],
            "pres_shift_08_12": [0.1, 0.2],
            "pres_shift_12_16": [0.1, 0.2],
            "pres_shift_16_20": [0.1, 0.2],
            "pres_shift_20_24": [0.1, 0.2],
            "gov_shift_18_22_centered": [0.3, np.nan],
            "gov_shift_16_20_centered": [0.1, 0.2],
            "sen_shift_16_18_centered": [0.5, 0.6],
        })
        result = compute_aggregated_offcycle(df)
        assert "mean_gov_shift" in result.columns
        assert "mean_sen_shift" in result.columns

    def test_ignores_nan_in_mean(self) -> None:
        """NaN values in individual off-cycle cols should be skipped in the mean."""
        df = pd.DataFrame({
            "tract_geoid": ["a"],
            "gov_shift_18_22_centered": [0.4],
            "gov_shift_16_20_centered": [np.nan],  # should be ignored
        })
        # Add all other gov/sen cols as NaN so they don't affect the result
        for col in GOV_COLS:
            if col not in df.columns:
                df[col] = np.nan
        for col in SEN_COLS:
            if col not in df.columns:
                df[col] = np.nan

        result = compute_aggregated_offcycle(df)
        # Only non-NaN gov col is 0.4; mean should be 0.4
        assert abs(result.loc[0, "mean_gov_shift"] - 0.4) < 1e-9

    def test_all_nan_gov_stays_nan(self) -> None:
        """Tract with all NaN gov cols should get NaN mean_gov_shift (later filled to 0)."""
        df = pd.DataFrame({
            "tract_geoid": ["x"],
        })
        for col in GOV_COLS:
            df[col] = np.nan
        for col in SEN_COLS:
            df[col] = np.nan

        result = compute_aggregated_offcycle(df)
        assert np.isnan(result.loc[0, "mean_gov_shift"])

    def test_does_not_modify_original(self) -> None:
        """Function should return a copy; original df should be unchanged."""
        df = pd.DataFrame({
            "tract_geoid": ["a"],
            "gov_shift_18_22_centered": [0.1],
            "sen_shift_16_18_centered": [0.2],
        })
        original_cols = set(df.columns)
        _ = compute_aggregated_offcycle(df)
        assert set(df.columns) == original_cols, "Original df columns should not be modified"


# ── Tests: type_size_summary ─────────────────────────────────────────────────

class TestTypeSizeSummary:
    def test_output_keys(self) -> None:
        """Should return dict with expected keys."""
        labels = np.array([0, 0, 1, 1, 2, 2, 2])
        summary = type_size_summary(labels)
        assert set(summary.keys()) == {"min", "max", "median", "top5", "bottom5"}

    def test_min_max_correct(self) -> None:
        """Min and max should match smallest and largest type."""
        labels = np.array([0, 0, 0, 1, 1, 2])  # sizes: [3, 2, 1]
        summary = type_size_summary(labels)
        assert summary["min"] == 1
        assert summary["max"] == 3

    def test_top5_descending(self) -> None:
        """Top5 should be in descending order."""
        labels = np.repeat(np.arange(10), np.arange(1, 11))
        summary = type_size_summary(labels)
        assert summary["top5"] == sorted(summary["top5"], reverse=True)
