"""Model regression guard — fail CI if committed metrics degrade below thresholds.

Reads data/model/accuracy_metrics.json (a committed artifact) and asserts
minimum quality thresholds. Any model retrain that pushes metrics below these
floors MUST be investigated before merging.

Thresholds are conservative — they allow some headroom from the current best
observed values, but will catch regressions that suggest something has broken
in the pipeline or modeling approach.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ============================================================================
# Threshold Constants — Update when baselines improve
# ============================================================================

# LOO r on county model holdout split. Current best is 0.732 (Ridge+HGB ensemble).
# Floor allows ~1.2% headroom, catches significant regressions.
# Context: S306 achieved 0.731 (Ridge+features); ensemble added 0.001
LOO_R_THRESHOLD = 0.72

# Standard holdout r (legacy metric for backward compatibility with non-LOO).
# Current: 0.698. Floor at 0.68 allows 1% headroom. This is the weakest metric
# (inflates due to type self-prediction) but still a useful baseline.
HOLDOUT_R_THRESHOLD = 0.68

# Type coherence via Silhouette score. Current: 0.783. Communities should be
# tight; floor at 0.75 flags type drift. S238 achieved 0.783 with J=100.
COHERENCE_THRESHOLD = 0.75

# Root Mean Squared Error on vote share predictions. Current: 0.073 (7.3pp).
# Floor at 0.08 (8pp) allows ~1pp headroom. Errors above 10pp are unacceptable.
RMSE_THRESHOLD = 0.08

# Ledoit-Wolf regularized covariance validation r via correlation with observed.
# Current: 0.936. High value signals types truly covary in observed data.
# Floor at 0.90 flags covariance structure degradation. S306 improved from
# 0.915 to 0.936 via PCA whitening + feature pruning.
COVARIANCE_VAL_R_THRESHOLD = 0.90

# Cross-election mean LOO r (holdout each election pair in turn).
# Current: mean across 4 cycles = (0.45 + 0.64 + 0.42 + 0.40) / 4 = 0.4775.
# Floor at 0.40 is generous — captures catastrophic degradation. S197 baseline
# was 0.476±0.10. Values below 0.40 suggest types have lost predictive power.
CROSS_ELECTION_MEAN_LOO_THRESHOLD = 0.40

# Data pipeline integrity: number of counties in the model.
# Must be exactly 3154 (all 50 states + DC, no missing counties).
EXPECTED_N_COUNTIES = 3154

# Model config integrity: number of types (K in KMeans).
# Must be exactly 100 (J selection was done at this value; changing it is a
# conscious decision and must be reflected in all downstream artifacts).
EXPECTED_N_TYPES = 100


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def metrics() -> dict:
    """Load accuracy_metrics.json once per test session."""
    metrics_path = Path(__file__).parent.parent / "data" / "model" / "accuracy_metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


# ============================================================================
# Tests
# ============================================================================


def test_accuracy_metrics_file_exists() -> None:
    """Assert data/model/accuracy_metrics.json exists and is readable."""
    metrics_path = Path(__file__).parent.parent / "data" / "model" / "accuracy_metrics.json"
    assert metrics_path.exists(), f"accuracy_metrics.json not found at {metrics_path}"
    assert metrics_path.is_file(), f"accuracy_metrics.json is not a file: {metrics_path}"


def test_loo_r_above_threshold(metrics: dict) -> None:
    """Test overall.loo_r >= 0.72 (current best 0.732, floor allows 0.012 headroom)."""
    loo_r = metrics["overall"]["loo_r"]
    assert (
        loo_r >= LOO_R_THRESHOLD
    ), f"LOO r degraded to {loo_r}; minimum is {LOO_R_THRESHOLD}"


def test_holdout_r_above_threshold(metrics: dict) -> None:
    """Test overall.holdout_r >= 0.68 (current 0.698, weaker metric than LOO)."""
    holdout_r = metrics["overall"]["holdout_r"]
    assert (
        holdout_r >= HOLDOUT_R_THRESHOLD
    ), f"Holdout r degraded to {holdout_r}; minimum is {HOLDOUT_R_THRESHOLD}"


def test_coherence_above_threshold(metrics: dict) -> None:
    """Test overall.coherence >= 0.75 (current 0.783, flags type drift)."""
    coherence = metrics["overall"]["coherence"]
    assert (
        coherence >= COHERENCE_THRESHOLD
    ), f"Coherence degraded to {coherence}; minimum is {COHERENCE_THRESHOLD}"


def test_rmse_below_threshold(metrics: dict) -> None:
    """Test overall.rmse <= 0.08 (current 0.073, allows ~1pp headroom)."""
    rmse = metrics["overall"]["rmse"]
    assert (
        rmse <= RMSE_THRESHOLD
    ), f"RMSE degraded to {rmse}; maximum is {RMSE_THRESHOLD}"


def test_covariance_val_r_above_threshold(metrics: dict) -> None:
    """Test overall.covariance_val_r >= 0.90 (current 0.936, flags structure degradation)."""
    cov_val_r = metrics["overall"]["covariance_val_r"]
    assert (
        cov_val_r >= COVARIANCE_VAL_R_THRESHOLD
    ), f"Covariance validation r degraded to {cov_val_r}; minimum is {COVARIANCE_VAL_R_THRESHOLD}"


def test_cross_election_mean_loo_above_threshold(metrics: dict) -> None:
    """Test mean of cross_election loo_r values >= 0.40 (current mean ~0.48)."""
    cross_election = metrics["cross_election"]
    assert (
        len(cross_election) > 0
    ), "cross_election array is empty; cannot compute mean"
    loo_values = [entry["loo_r"] for entry in cross_election]
    mean_loo = sum(loo_values) / len(loo_values)
    assert mean_loo >= CROSS_ELECTION_MEAN_LOO_THRESHOLD, (
        f"Cross-election mean LOO degraded to {mean_loo:.4f}; "
        f"minimum is {CROSS_ELECTION_MEAN_LOO_THRESHOLD}"
    )


def test_ensemble_is_best_method(metrics: dict) -> None:
    """Test the last entry in method_comparison has the highest loo_r.

    The ensemble should dominate all baseline approaches. If a simpler
    method outperforms it, something is wrong (e.g., ensemble overfit,
    feature engineering regressed, or coefficients not refit).
    """
    methods = metrics["method_comparison"]
    assert len(methods) > 0, "method_comparison is empty"
    loo_values = [m["loo_r"] for m in methods]
    last_loo = loo_values[-1]
    max_loo = max(loo_values)
    assert (
        last_loo == max_loo
    ), f"Ensemble (last method, loo_r={last_loo}) is not the best. "
    f"Best method has loo_r={max_loo}"


def test_county_count_stable(metrics: dict) -> None:
    """Test n_counties == 3154 (data pipeline integrity).

    If this fails, the county ingestion pipeline has changed: new counties
    added, old ones removed, or FIPS mapping is broken. Investigate before
    merging.
    """
    n_counties = metrics["overall"]["n_counties"]
    assert (
        n_counties == EXPECTED_N_COUNTIES
    ), f"County count changed to {n_counties}; expected {EXPECTED_N_COUNTIES}"


def test_type_count_stable(metrics: dict) -> None:
    """Test n_types == 100 (model config integrity).

    If this fails, the J (number of types) parameter was changed. This
    requires careful analysis of whether new J improves holdout CV, and
    synchronization with all downstream: type assignments, super-types,
    frontend artifact generation, etc. Not a change to make lightly.
    """
    n_types = metrics["overall"]["n_types"]
    assert (
        n_types == EXPECTED_N_TYPES
    ), f"Type count changed to {n_types}; expected {EXPECTED_N_TYPES}"
