"""Tests for the public API of src/prediction/predict_2026_types.py.

Covers:
  - ForecastParams dataclass construction and defaults
  - load_forecast_params() loading from prediction_params.json
  - Public helper functions: load_type_data, load_county_metadata,
    load_county_votes, load_polls
  - run_forecast_pipeline() signature accepts parameters
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.prediction.predict_2026_types import (
    ForecastParams,
    load_county_metadata,
    load_county_votes,
    load_forecast_params,
    load_polls,
    load_type_data,
    run_forecast_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# ForecastParams dataclass
# ---------------------------------------------------------------------------


class TestForecastParams:
    """Tests for the ForecastParams dataclass."""

    def test_defaults(self):
        """Default ForecastParams should have sensible values."""
        p = ForecastParams()
        assert p.lam == 1.0
        assert p.mu == 1.0
        assert p.w_vector_mode == "core"
        assert p.poll_blend_scale == 5.0
        assert p.half_life_days == 30.0
        assert p.pre_primary_discount == 0.5
        assert p.accuracy_path is None
        assert p.methodology_weights == {}
        assert p.fundamentals_enabled is False
        assert p.fundamentals_weight == 0.3

    def test_custom_values(self):
        """ForecastParams should accept custom values."""
        p = ForecastParams(
            lam=10.0,
            mu=2.0,
            w_vector_mode="full",
            poll_blend_scale=15.0,
            half_life_days=60.0,
            pre_primary_discount=0.3,
            fundamentals_enabled=True,
            fundamentals_weight=0.5,
        )
        assert p.lam == 10.0
        assert p.mu == 2.0
        assert p.w_vector_mode == "full"
        assert p.poll_blend_scale == 15.0
        assert p.half_life_days == 60.0
        assert p.pre_primary_discount == 0.3
        assert p.fundamentals_enabled is True
        assert p.fundamentals_weight == 0.5

    def test_methodology_weights_mutable_default(self):
        """Each ForecastParams instance should get its own methodology_weights dict."""
        p1 = ForecastParams()
        p2 = ForecastParams()
        p1.methodology_weights["phone"] = 1.15
        # p2 should not be affected by mutation of p1's dict.
        assert "phone" not in p2.methodology_weights


# ---------------------------------------------------------------------------
# load_forecast_params
# ---------------------------------------------------------------------------


class TestLoadForecastParams:
    """Tests for loading params from prediction_params.json."""

    def test_loads_from_default_path(self):
        """load_forecast_params() should load successfully from the default config."""
        params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
        if not params_path.exists():
            pytest.skip("prediction_params.json not present on disk")

        p = load_forecast_params()
        # Values from the JSON should override defaults.
        assert isinstance(p, ForecastParams)
        assert isinstance(p.lam, float)
        assert isinstance(p.mu, float)
        assert p.w_vector_mode in ("core", "full")
        assert p.poll_blend_scale > 0
        assert p.half_life_days > 0

    def test_loads_from_explicit_path(self):
        """load_forecast_params() should accept an explicit path."""
        params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
        if not params_path.exists():
            pytest.skip("prediction_params.json not present on disk")

        p = load_forecast_params(params_path=params_path)
        assert isinstance(p, ForecastParams)
        # Sanity: lam from the file is 10.0 per the current config.
        assert p.lam == 10.0

    def test_methodology_weights_populated(self):
        """Methodology weights should be loaded from the JSON when present."""
        params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
        if not params_path.exists():
            pytest.skip("prediction_params.json not present on disk")

        p = load_forecast_params()
        # The current config has methodology_weights with at least "phone".
        assert len(p.methodology_weights) > 0
        assert "phone" in p.methodology_weights


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


class TestLoadTypeData:
    """Tests for load_type_data()."""

    def test_returns_correct_shapes(self):
        """load_type_data should return county_fips, type_scores, covariance, priors."""
        ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
        if not ta_path.exists():
            pytest.skip("type_assignments.parquet not present on disk")

        county_fips, type_scores, type_covariance, type_priors = load_type_data()

        N = len(county_fips)
        J = type_scores.shape[1]

        assert N > 0
        assert type_scores.shape == (N, J)
        assert type_covariance.shape == (J, J)
        assert type_priors.shape == (J,)

    def test_fips_are_strings(self):
        """County FIPS should be zero-padded 5-digit strings."""
        ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
        if not ta_path.exists():
            pytest.skip("type_assignments.parquet not present on disk")

        county_fips, _, _, _ = load_type_data()
        assert all(isinstance(f, str) for f in county_fips[:10])
        assert all(len(f) == 5 for f in county_fips[:10])


class TestLoadCountyMetadata:
    """Tests for load_county_metadata()."""

    def test_returns_states_and_names(self):
        """Should return state abbreviations and county names."""
        fips = ["12001", "13001", "01001"]
        states, names = load_county_metadata(fips)

        assert len(states) == 3
        assert len(names) == 3
        assert states[0] == "FL"
        assert states[1] == "GA"
        assert states[2] == "AL"


class TestLoadCountyVotes:
    """Tests for load_county_votes()."""

    def test_returns_array_matching_fips_length(self):
        """Should return an array with one entry per county."""
        fips = ["12001", "13001", "01001"]
        votes = load_county_votes(fips)

        assert isinstance(votes, np.ndarray)
        assert len(votes) == 3
        # All values should be positive (at least fallback 1.0).
        assert (votes > 0).all()


class TestLoadPolls:
    """Tests for load_polls()."""

    def test_loads_default_polls(self):
        """Should load polls from the default 2026 path."""
        polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
        if not polls_path.exists():
            pytest.skip("polls_2026.csv not present on disk")

        fips = ["12001"]  # Minimal FIPS list (not used for filtering).
        polls_by_race, poll_lookup = load_polls(fips)

        assert isinstance(polls_by_race, dict)
        assert isinstance(poll_lookup, dict)

    def test_accepts_custom_path(self):
        """Should accept a custom polls_path parameter."""
        polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
        if not polls_path.exists():
            pytest.skip("polls_2026.csv not present on disk")

        fips = ["12001"]
        polls_by_race, _ = load_polls(fips, polls_path=polls_path)
        assert isinstance(polls_by_race, dict)


# ---------------------------------------------------------------------------
# run_forecast_pipeline signature
# ---------------------------------------------------------------------------


class TestRunForecastPipelineSignature:
    """Tests that run_forecast_pipeline accepts the documented parameters.

    These are lightweight signature tests -- full integration tests require
    all data files on disk and are expensive.
    """

    def test_accepts_forecast_params(self):
        """run_forecast_pipeline should accept a ForecastParams object."""
        # Just verify the function signature accepts the parameter.
        # We don't call it here because it requires full data on disk.
        import inspect

        sig = inspect.signature(run_forecast_pipeline)
        param_names = list(sig.parameters.keys())
        assert "params" in param_names
        assert "year" in param_names
        assert "polls_path" in param_names
        assert "output_path" in param_names
        assert "reference_date" in param_names
        assert "include_baseline" in param_names

    def test_all_params_keyword_only(self):
        """All parameters should be keyword-only (enforced by *)."""
        import inspect

        sig = inspect.signature(run_forecast_pipeline)
        for name, param in sig.parameters.items():
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"Parameter '{name}' should be keyword-only"
            )
