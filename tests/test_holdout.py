"""Tests for temporal holdout validation."""
from __future__ import annotations

import numpy as np
import pytest

from src.validation.validate_holdout import (
    split_training_holdout,
    community_level_prediction_accuracy,
)
from src.description.compare_to_nmf import within_community_variance


class TestSplitTrainingHoldout:
    def test_training_has_6_dims(self):
        """Training = pres 16->20 (cols 0-2) + midterm 18->22 (cols 6-8) = 6 dims."""
        shifts_9d = np.random.default_rng(42).normal(size=(100, 9))
        train, holdout = split_training_holdout(shifts_9d)
        assert train.shape == (100, 6)

    def test_holdout_has_3_dims(self):
        """Holdout = pres 20->24 (cols 3-5) = 3 dims."""
        shifts_9d = np.random.default_rng(42).normal(size=(100, 9))
        train, holdout = split_training_holdout(shifts_9d)
        assert holdout.shape == (100, 3)

    def test_holdout_is_cols_3_to_5(self):
        """Verify holdout contains exactly the 20->24 presidential shift."""
        shifts_9d = np.arange(900).reshape(100, 9).astype(float)
        _, holdout = split_training_holdout(shifts_9d)
        np.testing.assert_array_equal(holdout, shifts_9d[:, 3:6])


class TestCommunityLevelPrediction:
    def test_perfect_prediction_gives_correlation_1(self):
        training_means = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        holdout_means = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        corr, mae = community_level_prediction_accuracy(training_means, holdout_means)
        assert abs(corr - 1.0) < 1e-6

    def test_returns_positive_mae(self):
        rng = np.random.default_rng(42)
        training_means = rng.normal(size=(20, 3))
        holdout_means = rng.normal(size=(20, 3))
        corr, mae = community_level_prediction_accuracy(training_means, holdout_means)
        assert mae > 0.0
