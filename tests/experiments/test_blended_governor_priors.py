"""Tests for the blended governor priors experiment.

Covers:
  - blend_weights at w=0.0 and w=1.0 equal the pure presidential and pure
    governor predictions respectively
  - metrics computed correctly on synthetic data
  - run_blend_sweep returns the expected 11 entries
  - find_optimal_weight picks the composite-score winner
  - _aggregate_to_states handles missing data and edge cases
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.blended_governor_priors import (
    BLEND_WEIGHTS,
    _aggregate_to_states,
    compute_metrics,
    find_optimal_weight,
    run_blend_sweep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_county_arrays(n: int, seed: int = 42) -> tuple:
    """Return (gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips)."""
    rng = np.random.default_rng(seed)
    gov_pred = rng.uniform(0.30, 0.65, n)
    pres_share = gov_pred + rng.uniform(-0.05, 0.05, n)
    actual_2022 = gov_pred + rng.uniform(-0.08, 0.08, n)
    vote_weights = rng.uniform(5000, 50000, n)
    # Assign counties to two fake states (TS and TR) evenly
    state_abbr = ["TS" if i < n // 2 else "TR" for i in range(n)]
    fips = np.array([str(i).zfill(5) for i in range(10001, 10001 + n)])
    return gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips


# ---------------------------------------------------------------------------
# Test: w=0.0 matches pure presidential
# ---------------------------------------------------------------------------

class TestBlendsAtBoundaries:
    def test_weight_zero_equals_presidential(self):
        """gov_weight=0.0 should produce blends identical to pres_share."""
        n = 20
        gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips = _make_county_arrays(n)

        results = run_blend_sweep(
            gov_pred=gov_pred,
            pres_share=pres_share,
            actual_2022=actual_2022,
            vote_weights=vote_weights,
            state_abbr=state_abbr,
            matched_fips=fips,
            blend_weights=[0.0],
        )

        # Build the pure-presidential state aggregation manually
        pure_pres_state = _aggregate_to_states(fips, state_abbr, pres_share, actual_2022, vote_weights)
        pure_pres_metrics = compute_metrics(pure_pres_state)

        assert results[0]["r"] == pytest.approx(pure_pres_metrics["r"], abs=1e-6), (
            "w=0.0 blend should equal pure presidential correlation"
        )
        assert results[0]["bias_pp"] == pytest.approx(pure_pres_metrics["bias_pp"], abs=1e-4)

    def test_weight_one_equals_governor(self):
        """gov_weight=1.0 should produce blends identical to gov_pred."""
        n = 20
        gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips = _make_county_arrays(n)

        results = run_blend_sweep(
            gov_pred=gov_pred,
            pres_share=pres_share,
            actual_2022=actual_2022,
            vote_weights=vote_weights,
            state_abbr=state_abbr,
            matched_fips=fips,
            blend_weights=[1.0],
        )

        pure_gov_state = _aggregate_to_states(fips, state_abbr, gov_pred, actual_2022, vote_weights)
        pure_gov_metrics = compute_metrics(pure_gov_state)

        assert results[0]["r"] == pytest.approx(pure_gov_metrics["r"], abs=1e-6), (
            "w=1.0 blend should equal pure governor correlation"
        )
        assert results[0]["bias_pp"] == pytest.approx(pure_gov_metrics["bias_pp"], abs=1e-4)

    def test_blend_at_half_is_intermediate(self):
        """w=0.5 blended pred per county should be the mean of gov and pres values.

        We verify this by checking the state-level aggregation directly —
        using two states so compute_metrics can produce non-NaN values.
        """
        n = 10
        gov_pred = np.full(n, 0.60)
        pres_share = np.full(n, 0.40)
        actual_2022 = np.full(n, 0.50)
        vote_weights = np.ones(n) * 10000
        # Two states so state-level metrics are not degenerate (need n>=2)
        state_abbr = ["TS"] * (n // 2) + ["TR"] * (n // 2)
        fips = np.array([str(i).zfill(5) for i in range(10001, 10001 + n)])

        results = run_blend_sweep(
            gov_pred=gov_pred,
            pres_share=pres_share,
            actual_2022=actual_2022,
            vote_weights=vote_weights,
            state_abbr=state_abbr,
            matched_fips=fips,
            blend_weights=[0.5],
        )

        # State-level blended pred = 0.5*0.60 + 0.5*0.40 = 0.50 = actual
        # → bias_pp should be 0
        assert abs(results[0]["bias_pp"]) < 0.01, (
            f"0.5*0.60 + 0.5*0.40 = 0.50, equal to actual → bias should be ~0, "
            f"got bias_pp={results[0]['bias_pp']}"
        )


# ---------------------------------------------------------------------------
# Test: sweep returns all 11 entries
# ---------------------------------------------------------------------------

class TestSweepStructure:
    def test_returns_eleven_entries(self):
        """Sweep over default BLEND_WEIGHTS should return exactly 11 results."""
        n = 20
        gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips = _make_county_arrays(n)

        results = run_blend_sweep(
            gov_pred=gov_pred,
            pres_share=pres_share,
            actual_2022=actual_2022,
            vote_weights=vote_weights,
            state_abbr=state_abbr,
            matched_fips=fips,
        )

        assert len(results) == 11, f"Expected 11 blend weights, got {len(results)}"

    def test_gov_weights_span_zero_to_one(self):
        """gov_weight should span [0.0, 0.1, ..., 1.0]."""
        n = 20
        gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips = _make_county_arrays(n)

        results = run_blend_sweep(
            gov_pred=gov_pred,
            pres_share=pres_share,
            actual_2022=actual_2022,
            vote_weights=vote_weights,
            state_abbr=state_abbr,
            matched_fips=fips,
        )

        gov_weights = [r["gov_weight"] for r in results]
        assert gov_weights == BLEND_WEIGHTS

    def test_pres_weight_is_complement(self):
        """pres_weight should equal 1 - gov_weight for all entries."""
        n = 20
        gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips = _make_county_arrays(n)

        results = run_blend_sweep(
            gov_pred=gov_pred,
            pres_share=pres_share,
            actual_2022=actual_2022,
            vote_weights=vote_weights,
            state_abbr=state_abbr,
            matched_fips=fips,
        )

        for row in results:
            assert abs(row["gov_weight"] + row["pres_weight"] - 1.0) < 1e-9, (
                f"gov_weight + pres_weight != 1.0 at w={row['gov_weight']}"
            )

    def test_required_metric_keys_present(self):
        """Every result dict should have the required metric keys."""
        n = 20
        gov_pred, pres_share, actual_2022, vote_weights, state_abbr, fips = _make_county_arrays(n)

        results = run_blend_sweep(
            gov_pred=gov_pred,
            pres_share=pres_share,
            actual_2022=actual_2022,
            vote_weights=vote_weights,
            state_abbr=state_abbr,
            matched_fips=fips,
        )

        required_keys = {"gov_weight", "pres_weight", "r", "rmse_pp", "bias_pp",
                         "direction_accuracy", "n_states"}
        for row in results:
            assert required_keys.issubset(row.keys()), (
                f"Missing keys in result row: {required_keys - row.keys()}"
            )


# ---------------------------------------------------------------------------
# Test: compute_metrics on synthetic data
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_correlation_gives_r_one(self):
        """Identical pred and actual should give r=1.0 and bias/rmse=0."""
        n = 10
        share = np.linspace(0.35, 0.65, n)
        state_df = pd.DataFrame({
            "pred_dem_share": share,
            "actual_dem_share": share,
        })
        m = compute_metrics(state_df)
        assert m["r"] == pytest.approx(1.0, abs=1e-6)
        assert m["rmse_pp"] == pytest.approx(0.0, abs=1e-6)
        assert m["bias_pp"] == pytest.approx(0.0, abs=1e-6)
        assert m["direction_accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_constant_over_prediction_gives_positive_bias(self):
        """Uniformly +0.05 over-prediction should produce +5pp bias."""
        n = 8
        actual = np.linspace(0.35, 0.60, n)
        pred = actual + 0.05
        state_df = pd.DataFrame({"pred_dem_share": pred, "actual_dem_share": actual})
        m = compute_metrics(state_df)
        assert m["bias_pp"] == pytest.approx(5.0, abs=0.01)

    def test_direction_accuracy_all_wrong(self):
        """Predictions all on wrong side of 50% should give direction_accuracy=0."""
        # All actuals > 0.5 (Dem win), all preds < 0.5 (Rep predicted)
        n = 6
        actual = np.linspace(0.55, 0.70, n)
        pred = np.linspace(0.30, 0.45, n)
        state_df = pd.DataFrame({"pred_dem_share": pred, "actual_dem_share": actual})
        m = compute_metrics(state_df)
        assert m["direction_accuracy"] == pytest.approx(0.0, abs=1e-6)

    def test_rmse_calculated_correctly(self):
        """RMSE should match manual calculation."""
        pred = np.array([0.50, 0.55, 0.60])
        actual = np.array([0.45, 0.50, 0.55])
        expected_rmse_pp = float(np.sqrt(np.mean((pred - actual) ** 2)) * 100)
        state_df = pd.DataFrame({"pred_dem_share": pred, "actual_dem_share": actual})
        m = compute_metrics(state_df)
        assert m["rmse_pp"] == pytest.approx(expected_rmse_pp, abs=0.01)

    def test_returns_nan_metrics_for_single_state(self):
        """Single-state data can't compute Pearson r — should return nan."""
        state_df = pd.DataFrame({
            "pred_dem_share": [0.52],
            "actual_dem_share": [0.48],
        })
        m = compute_metrics(state_df)
        assert np.isnan(m["r"])


# ---------------------------------------------------------------------------
# Test: find_optimal_weight picks the composite-score winner
# ---------------------------------------------------------------------------

class TestFindOptimalWeight:
    def test_picks_highest_composite_score(self):
        """find_optimal_weight should select the entry with max r - |bias|/10."""
        results = [
            {"gov_weight": 0.0, "pres_weight": 1.0, "r": 0.80, "rmse_pp": 9.8, "bias_pp": 4.6,
             "direction_accuracy": 0.84, "n_states": 32},
            {"gov_weight": 0.7, "pres_weight": 0.3, "r": 0.82, "rmse_pp": 9.7, "bias_pp": 3.0,
             "direction_accuracy": 0.88, "n_states": 32},  # best composite: 0.82 - 0.30 = 0.52
            {"gov_weight": 1.0, "pres_weight": 0.0, "r": 0.70, "rmse_pp": 10.7, "bias_pp": 2.2,
             "direction_accuracy": 0.88, "n_states": 32},
        ]
        optimal = find_optimal_weight(results)
        assert optimal["gov_weight"] == 0.7, (
            "Weight 0.7 has composite 0.82 - 0.30 = 0.52, highest of the three"
        )

    def test_handles_tie_by_returning_one(self):
        """Should return a single result even when composite scores are tied."""
        results = [
            {"gov_weight": 0.5, "pres_weight": 0.5, "r": 0.75, "rmse_pp": 9.5, "bias_pp": 0.0,
             "direction_accuracy": 0.80, "n_states": 30},
            {"gov_weight": 0.6, "pres_weight": 0.4, "r": 0.75, "rmse_pp": 9.5, "bias_pp": 0.0,
             "direction_accuracy": 0.80, "n_states": 30},
        ]
        optimal = find_optimal_weight(results)
        assert "gov_weight" in optimal


# ---------------------------------------------------------------------------
# Test: _aggregate_to_states handles edge cases
# ---------------------------------------------------------------------------

class TestAggregateToStates:
    def test_excludes_nan_actuals(self):
        """Counties with NaN actuals should not contribute to state aggregates."""
        fips = np.array(["10001", "10002", "10003"])
        state_abbr = ["TS", "TS", "TS"]
        pred = np.array([0.55, 0.60, 0.65])
        actual = np.array([0.50, float("nan"), 0.60])  # 10002 has no data
        weights = np.array([10000.0, 10000.0, 10000.0])

        result = _aggregate_to_states(fips, state_abbr, pred, actual, weights)
        assert len(result) == 1
        # Only counties 10001 and 10003 contribute (equal weights)
        expected_pred = (0.55 * 10000 + 0.65 * 10000) / (10000 + 10000)
        assert result.iloc[0]["pred_dem_share"] == pytest.approx(expected_pred, abs=1e-6)

    def test_excludes_none_state(self):
        """Counties with None state should be excluded from aggregation."""
        fips = np.array(["10001", "10002"])
        state_abbr = [None, "TS"]
        pred = np.array([0.55, 0.60])
        actual = np.array([0.50, 0.55])
        weights = np.array([10000.0, 10000.0])

        result = _aggregate_to_states(fips, state_abbr, pred, actual, weights)
        # Only the TS county should appear
        assert len(result) == 1
        assert result.iloc[0]["state"] == "TS"

    def test_vote_weighted_average_correct(self):
        """State prediction should be the vote-weighted mean of county predictions."""
        fips = np.array(["10001", "10002"])
        state_abbr = ["TS", "TS"]
        pred = np.array([0.40, 0.60])
        actual = np.array([0.45, 0.55])
        # County 10002 has 3x the votes → should pull result toward 0.60
        weights = np.array([1000.0, 3000.0])

        result = _aggregate_to_states(fips, state_abbr, pred, actual, weights)
        expected_pred = (0.40 * 1000 + 0.60 * 3000) / 4000  # = 0.55
        assert result.iloc[0]["pred_dem_share"] == pytest.approx(expected_pred, abs=1e-6)

    def test_empty_input_returns_empty_dataframe(self):
        """Empty input should return an empty DataFrame with the correct columns."""
        result = _aggregate_to_states(
            np.array([]), [], np.array([]), np.array([]), np.array([])
        )
        assert list(result.columns) == ["state", "pred_dem_share", "actual_dem_share", "n_counties"]
        assert len(result) == 0

    def test_multiple_states(self):
        """Aggregation should produce one row per state."""
        fips = np.array(["10001", "10002", "10003", "10004"])
        state_abbr = ["TS", "TS", "TR", "TR"]
        pred = np.array([0.50, 0.55, 0.40, 0.45])
        actual = np.array([0.52, 0.57, 0.42, 0.47])
        weights = np.ones(4) * 10000.0

        result = _aggregate_to_states(fips, state_abbr, pred, actual, weights)
        assert len(result) == 2
        states = set(result["state"].tolist())
        assert states == {"TS", "TR"}
