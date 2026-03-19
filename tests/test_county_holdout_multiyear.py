"""Tests for multi-year county holdout validation."""
from __future__ import annotations
import numpy as np
import pytest
from src.validation.validate_county_holdout_multiyear import (
    split_training_holdout,
    community_correlation,
)

N_TRAINING = 30
N_HOLDOUT = 3


def test_split_training_shape():
    shifts = np.random.rand(293, N_TRAINING + N_HOLDOUT)
    train, holdout = split_training_holdout(shifts, N_TRAINING)
    assert train.shape == (293, N_TRAINING)
    assert holdout.shape == (293, N_HOLDOUT)


def test_split_holdout_is_last_cols():
    shifts = np.arange(293 * 33).reshape(293, 33).astype(float)
    train, holdout = split_training_holdout(shifts, 30)
    np.testing.assert_array_equal(holdout, shifts[:, 30:])


def test_community_correlation_perfect():
    # 3 communities, each with 2 members
    labels = np.array([0, 0, 1, 1, 2, 2])
    train_members = np.array([[0.1, 0.1, 0.5, 0.5, -0.1, -0.1]]).T  # shape (6,1)
    holdout_members = np.array([[0.2, 0.2, 0.6, 0.6, 0.0, 0.0]]).T
    comm_train = np.array([train_members[labels == k].mean(axis=0) for k in [0, 1, 2]])
    comm_holdout = np.array([holdout_members[labels == k].mean(axis=0) for k in [0, 1, 2]])
    r, mae = community_correlation(comm_train, comm_holdout)
    assert r > 0.99


def test_community_correlation_range():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=10)
    train = rng.random((10, 30))
    holdout = rng.random((10, 3))
    comm_train = np.array([train[labels == k].mean(axis=0) for k in np.unique(labels)])
    comm_holdout = np.array([holdout[labels == k].mean(axis=0) for k in np.unique(labels)])
    r, mae = community_correlation(comm_train, comm_holdout)
    assert -1.0 <= r <= 1.0
    assert mae >= 0
