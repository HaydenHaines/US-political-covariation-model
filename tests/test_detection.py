"""Tests for community type detection (NMF).

These tests exercise the NMF functions directly using synthetic data so they
run without the real ACS dataset, which is large and requires a network fetch.
Tests verify mathematical properties of the NMF outputs:
  - normalize_memberships produces rows that sum to 1
  - fit_nmf produces the expected output shapes
  - K=7 components are extracted
  - No political data columns are present in FEATURE_COLS
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

# Import the functions/constants we want to test directly
from src.detection.nmf_fit import (
    FEATURE_COLS,
    K,
    fit_nmf,
    normalize_memberships,
)


# ---------------------------------------------------------------------------
# Constants / canonical values
# ---------------------------------------------------------------------------

# Political keywords that must NOT appear in the non-political feature set
POLITICAL_KEYWORDS = {
    "vote", "dem", "rep", "republican", "democrat", "election", "party",
    "trump", "biden", "clinton", "pres", "gov", "senate", "house",
    "partisan", "ballot", "poll", "candidate",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_X() -> np.ndarray:
    """Small synthetic non-negative feature matrix for NMF testing (50 tracts × 12 features)."""
    rng = np.random.default_rng(42)
    X_raw = rng.uniform(0.0, 1.0, size=(50, len(FEATURE_COLS)))
    # MinMaxScaler ensures values in [0, 1], matching the real pipeline
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_raw)


@pytest.fixture(scope="module")
def nmf_outputs(synthetic_X) -> tuple[np.ndarray, np.ndarray]:
    """Run NMF on synthetic data and return (W_raw, H)."""
    return fit_nmf(synthetic_X, K)


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------


def test_feature_cols_count():
    """NMF must use exactly 12 ACS feature columns."""
    assert len(FEATURE_COLS) == 12, f"Expected 12 feature columns, got {len(FEATURE_COLS)}"


def test_no_political_data_in_feature_cols():
    """
    Two-stage separation: FEATURE_COLS must contain only non-political ACS features.
    No election results, vote counts, party affiliations, or poll data.
    This is the core falsifiability mechanism of the model.
    """
    for col in FEATURE_COLS:
        col_lower = col.lower()
        for keyword in POLITICAL_KEYWORDS:
            assert keyword not in col_lower, (
                f"Political keyword '{keyword}' found in feature column '{col}'. "
                "Community detection must use ONLY non-political data."
            )


def test_k_is_seven():
    """Canonical K=7 community types (separates Asian from Knowledge Worker)."""
    assert K == 7, f"Expected K=7 (canonical), got K={K}"


# ---------------------------------------------------------------------------
# fit_nmf output shape tests
# ---------------------------------------------------------------------------


def test_fit_nmf_w_shape(synthetic_X, nmf_outputs):
    """W matrix must be (n_tracts, K) shaped."""
    W, H = nmf_outputs
    n_tracts, n_features = synthetic_X.shape
    assert W.shape == (n_tracts, K), f"Expected W shape ({n_tracts}, {K}), got {W.shape}"


def test_fit_nmf_h_shape(synthetic_X, nmf_outputs):
    """H matrix must be (K, n_features) shaped."""
    W, H = nmf_outputs
    n_features = len(FEATURE_COLS)
    assert H.shape == (K, n_features), f"Expected H shape ({K}, {n_features}), got {H.shape}"


def test_fit_nmf_w_non_negative(nmf_outputs):
    """NMF W matrix must be non-negative by construction."""
    W, H = nmf_outputs
    assert (W >= 0).all(), "W matrix has negative values (violates NMF non-negativity)"


def test_fit_nmf_h_non_negative(nmf_outputs):
    """NMF H matrix must be non-negative by construction."""
    W, H = nmf_outputs
    assert (H >= 0).all(), "H matrix has negative values (violates NMF non-negativity)"


# ---------------------------------------------------------------------------
# normalize_memberships tests
# ---------------------------------------------------------------------------


def test_normalize_memberships_sums_to_one():
    """After normalization, each tract's membership vector must sum to 1.0."""
    rng = np.random.default_rng(42)
    W_raw = rng.uniform(0.1, 1.0, size=(100, K))  # all positive, no zero rows
    W_norm = normalize_memberships(W_raw)
    row_sums = W_norm.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), (
        f"Normalized memberships don't sum to 1. Max deviation: {abs(row_sums - 1.0).max():.2e}"
    )


def test_normalize_memberships_output_shape():
    """normalize_memberships must preserve the input shape."""
    rng = np.random.default_rng(0)
    W_raw = rng.uniform(0.0, 1.0, size=(200, K))
    W_norm = normalize_memberships(W_raw)
    assert W_norm.shape == W_raw.shape


def test_normalize_memberships_non_negative():
    """Normalized memberships must remain non-negative."""
    rng = np.random.default_rng(1)
    W_raw = rng.uniform(0.0, 1.0, size=(50, K))
    W_norm = normalize_memberships(W_raw)
    assert (W_norm >= 0).all(), "Normalized memberships contain negative values"


def test_normalize_memberships_handles_zero_row():
    """An all-zero row (uninhabited tract proxy) must not produce NaN or inf."""
    W_raw = np.ones((3, K))
    W_raw[1, :] = 0.0  # simulate a zero row
    W_norm = normalize_memberships(W_raw)
    assert not np.isnan(W_norm).any(), "normalize_memberships produced NaN for zero row"
    assert not np.isinf(W_norm).any(), "normalize_memberships produced Inf for zero row"


def test_normalize_memberships_relative_order_preserved():
    """Within a row, the ranking of community weights must be preserved after normalization."""
    rng = np.random.default_rng(7)
    W_raw = rng.uniform(0.01, 1.0, size=(20, K))
    W_norm = normalize_memberships(W_raw)
    for i in range(len(W_raw)):
        assert np.argsort(W_raw[i]).tolist() == np.argsort(W_norm[i]).tolist(), (
            f"Row {i}: normalization changed the community weight ranking"
        )


# ---------------------------------------------------------------------------
# End-to-end NMF pipeline test (synthetic data)
# ---------------------------------------------------------------------------


def test_nmf_pipeline_soft_assignments_sum_to_one(synthetic_X):
    """Full pipeline: fit → normalize → each tract's soft assignments sum to 1."""
    W_raw, H = fit_nmf(synthetic_X, K)
    W_norm = normalize_memberships(W_raw)
    row_sums = W_norm.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_nmf_reconstruction_quality(synthetic_X):
    """NMF reconstruction should explain a reasonable fraction of input variance."""
    W_raw, H = fit_nmf(synthetic_X, K)
    X_reconstructed = W_raw @ H
    # Frobenius norm ratio: reconstruction error / total norm
    error_ratio = np.linalg.norm(synthetic_X - X_reconstructed) / np.linalg.norm(synthetic_X)
    # With K=7 on 12 features, error should be reasonably small (< 80%)
    assert error_ratio < 0.80, (
        f"NMF reconstruction error ratio {error_ratio:.2f} is too high; "
        "K=7 should explain the synthetic data structure"
    )


def test_each_community_has_nonzero_loading(nmf_outputs):
    """Every community component must have at least one non-zero feature loading."""
    W, H = nmf_outputs
    for k in range(K):
        assert H[k].max() > 0, f"Community c{k+1} has all-zero feature loadings in H"
