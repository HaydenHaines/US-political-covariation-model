"""Tests for src/models/contracts.py — data contract definitions.

Covers: Layer1Output, Layer2Output dataclasses, path generators, column constants.
"""
from __future__ import annotations

import pytest

from src.models.contracts import (
    LAYER1_REQUIRED_COLS,
    LAYER1_STANDARD_COLS,
    LAYER2_REQUIRED_COLS,
    Layer1Output,
    Layer2Output,
    layer1_output_path,
    layer2_output_path,
    type_profiles_path,
)


# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

class TestColumnContracts:
    def test_layer1_required_has_county_fips(self):
        assert "county_fips" in LAYER1_REQUIRED_COLS

    def test_layer1_required_has_community_id(self):
        assert "community_id" in LAYER1_REQUIRED_COLS

    def test_layer2_required_has_community_id(self):
        assert "community_id" in LAYER2_REQUIRED_COLS

    def test_layer2_required_has_dominant_type_id(self):
        assert "dominant_type_id" in LAYER2_REQUIRED_COLS

    def test_layer1_standard_is_superset_of_required(self):
        for col in LAYER1_REQUIRED_COLS:
            assert col in LAYER1_STANDARD_COLS


# ---------------------------------------------------------------------------
# Layer1Output dataclass
# ---------------------------------------------------------------------------

class TestLayer1Output:
    def test_create_minimal(self):
        out = Layer1Output(
            model_id="test-v1",
            k=20,
            shift_type="logodds",
            vote_share_type="twoparty",
            training_dims=33,
            holdout_r=0.698,
            assignment_file="data/test.parquet",
        )
        assert out.model_id == "test-v1"
        assert out.k == 20
        assert out.holdout_r == pytest.approx(0.698)

    def test_shift_type_values(self):
        """Contract: shift_type should be logodds or raw."""
        out = Layer1Output("m", 10, "logodds", "total", 10, 0.5, "f")
        assert out.shift_type in ("logodds", "raw")

    def test_vote_share_type_values(self):
        out = Layer1Output("m", 10, "raw", "twoparty", 10, 0.5, "f")
        assert out.vote_share_type in ("total", "twoparty")


# ---------------------------------------------------------------------------
# Layer2Output dataclass
# ---------------------------------------------------------------------------

class TestLayer2Output:
    def test_create_minimal(self):
        out = Layer2Output(
            model_id="test-v1",
            j=6,
            communities_file="data/types.parquet",
            type_profiles_file="data/profiles.parquet",
        )
        assert out.j == 6
        assert out.model_id == "test-v1"


# ---------------------------------------------------------------------------
# Path generators
# ---------------------------------------------------------------------------

class TestPathGenerators:
    def test_layer1_output_path_default_base(self):
        path = layer1_output_path("v1")
        assert path == "data/models/versions/v1/layer1_assignments.parquet"

    def test_layer1_output_path_custom_base(self):
        path = layer1_output_path("v2", base_dir="output")
        assert path == "output/versions/v2/layer1_assignments.parquet"

    def test_layer2_output_path_default(self):
        path = layer2_output_path("v1")
        assert path == "data/models/versions/v1/layer2_type_assignments.parquet"

    def test_type_profiles_path_default(self):
        path = type_profiles_path("v1")
        assert path == "data/models/versions/v1/type_profiles.parquet"

    def test_path_generators_use_model_id(self):
        """Changing model_id changes the path."""
        p1 = layer1_output_path("alpha")
        p2 = layer1_output_path("beta")
        assert p1 != p2
        assert "alpha" in p1
        assert "beta" in p2

    def test_all_paths_end_in_parquet(self):
        assert layer1_output_path("x").endswith(".parquet")
        assert layer2_output_path("x").endswith(".parquet")
        assert type_profiles_path("x").endswith(".parquet")
