"""Tests for voter behavior layer: turnout ratio (τ) and choice shift (δ).

Covers:
  - Original core functions (unchanged behavior)
  - Column mapping from T.1 format (race_type/total_votes) to internal format
  - Integration with T.1 tract_elections and T.3 tract_type_assignments formats
  - τ clipping to [0.1, 1.5]
  - δ computation correctness
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.behavior.voter_behavior import (
    apply_behavior_adjustment,
    compute_choice_shifts,
    compute_turnout_ratios,
    train_and_save,
    _map_race_type_to_race,
    PRESIDENTIAL_RACE_TYPES,
    OFFCYCLE_RACE_TYPES,
    TAU_CLIP_MIN,
    TAU_CLIP_MAX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tract_data():
    """Minimal tract data: 4 tracts, 2 types, presidential + off-cycle.

    Uses the internal column format (race, votes_total) that compute_* functions expect.
    """
    tract_votes = pd.DataFrame({
        "tract_geoid": (["T1", "T2", "T3", "T4"] * 4),
        "year": ([2020] * 4 + [2024] * 4 + [2018] * 4 + [2022] * 4),
        "race": (["president"] * 4 + ["president"] * 4 +
                 ["governor"] * 4 + ["governor"] * 4),
        "votes_total": ([1000, 800, 600, 400] + [1000, 800, 600, 400] +
                        [700, 500, 500, 350] + [720, 520, 480, 340]),
        "votes_dem": ([500, 300, 400, 100] + [500, 300, 400, 100] +
                      [380, 200, 320, 80] + [400, 220, 300, 75]),
        "dem_share": None,
        "state": ["AL", "AL", "GA", "GA"] * 4,
    })
    tract_votes["dem_share"] = np.where(
        tract_votes["votes_total"] > 0,
        tract_votes["votes_dem"] / tract_votes["votes_total"],
        np.nan,
    )

    type_scores = pd.DataFrame({
        "GEOID": ["T1", "T2", "T3", "T4"],
        "type_0_score": [0.8, 0.7, 0.2, 0.1],
        "type_1_score": [0.2, 0.3, 0.8, 0.9],
    }).set_index("GEOID")

    return tract_votes, type_scores


@pytest.fixture
def t1_format_elections():
    """Minimal tract elections in T.1 pipeline format (race_type, total_votes).

    This mimics what tract_elections.parquet looks like after T.1 assembly.
    Includes relevant races (PRES, GOV, SEN, SEN_SPEC) and irrelevant ones
    (CONG, AG) that should be filtered out.
    """
    return pd.DataFrame({
        "tract_geoid": ["T1", "T2", "T3", "T4",   # PRES 2020
                        "T1", "T2", "T3", "T4",   # GOV 2022
                        "T1", "T2", "T3", "T4",   # SEN 2022
                        "T1", "T2", "T3", "T4",   # SEN_SPEC 2021
                        "T1", "T2", "T3", "T4",   # CONG — should be filtered
                        "T1", "T2", "T3", "T4",   # AG — should be filtered
                        ],
        "year": ([2020] * 4 + [2022] * 4 + [2022] * 4 + [2021] * 4
                 + [2022] * 4 + [2022] * 4),
        "race_type": (["PRES"] * 4 + ["GOV"] * 4 + ["SEN"] * 4
                      + ["SEN_SPEC"] * 4 + ["CONG"] * 4 + ["AG"] * 4),
        "total_votes": ([1000, 800, 600, 400] + [700, 500, 500, 350]
                        + [750, 520, 480, 360] + [200, 150, 130, 100]
                        + [900, 700, 500, 300] + [500, 400, 300, 200]),
        "dem_votes":   ([500, 300, 400, 100] + [380, 200, 320, 80]
                        + [400, 250, 290, 90] + [110, 80, 70, 60]
                        + [450, 350, 240, 120] + [250, 200, 150, 100]),
        "rep_votes":   ([480, 480, 180, 290] + [300, 290, 170, 260]
                        + [330, 260, 180, 260] + [80, 60, 55, 35]
                        + [430, 330, 250, 170] + [240, 190, 140, 95]),
        "dem_share":   None,
    })


@pytest.fixture
def t3_format_assignments():
    """Minimal tract type assignments in T.3 pipeline format.

    Matches tract_type_assignments.parquet: tract_geoid column + type_j_score columns.
    """
    return pd.DataFrame({
        "tract_geoid": ["T1", "T2", "T3", "T4"],
        "type_0_score": [0.8, 0.7, 0.2, 0.1],
        "type_1_score": [0.2, 0.3, 0.8, 0.9],
        "dominant_type": [0, 0, 1, 1],
    })


# ---------------------------------------------------------------------------
# Original core function tests (unchanged behavior)
# ---------------------------------------------------------------------------

def test_turnout_ratios_shape(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert tau.shape == (2,)


def test_turnout_ratios_less_than_one(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert (tau < 1.0).all(), f"Expected τ < 1, got {tau}"
    assert (tau > 0.0).all()


def test_turnout_ratios_vary_by_type(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    assert tau[0] != tau[1]


def test_choice_shift_shape(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    delta = compute_choice_shifts(votes, scores, tau, n_types=2)
    assert delta.shape == (2,)


def test_choice_shift_bounded(mock_tract_data):
    votes, scores = mock_tract_data
    tau = compute_turnout_ratios(votes, scores, n_types=2)
    delta = compute_choice_shifts(votes, scores, tau, n_types=2)
    assert (np.abs(delta) < 0.2).all(), f"δ too large: {delta}"


def test_behavior_adjustment_noop_for_presidential():
    priors = np.array([0.45, 0.55, 0.50])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    tau = np.array([0.65, 0.85])
    delta = np.array([0.02, -0.01])
    result = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=False)
    np.testing.assert_array_equal(result, priors)


def test_behavior_adjustment_changes_for_offcycle():
    priors = np.array([0.45, 0.55, 0.50])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    tau = np.array([0.65, 0.85])
    delta = np.array([0.02, -0.01])
    result = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    assert not np.allclose(result, priors)


def test_behavior_adjustment_bounded():
    priors = np.array([0.02, 0.98, 0.50])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    tau = np.array([0.5, 0.9])
    delta = np.array([0.05, -0.05])
    result = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    assert (result >= 0.0).all() and (result <= 1.0).all()


# ---------------------------------------------------------------------------
# T.1 column mapping tests
# ---------------------------------------------------------------------------

def test_map_race_type_drops_irrelevant_races(t1_format_elections):
    """Non-PRES/GOV/SEN race_types (CONG, AG, etc.) must be filtered out."""
    t1_format_elections["dem_share"] = (
        t1_format_elections["dem_votes"] / t1_format_elections["total_votes"]
    )
    mapped = _map_race_type_to_race(t1_format_elections)
    assert "CONG" not in mapped.get("race_type", pd.Series()).values
    assert "AG" not in mapped.get("race_type", pd.Series()).values
    assert set(mapped["race"].unique()).issubset({"president", "governor", "senate"})


def test_map_race_type_keeps_all_relevant_races(t1_format_elections):
    """PRES, GOV, SEN, and SEN_SPEC must all survive the mapping."""
    t1_format_elections["dem_share"] = (
        t1_format_elections["dem_votes"] / t1_format_elections["total_votes"]
    )
    mapped = _map_race_type_to_race(t1_format_elections)
    races = set(mapped["race"].unique())
    assert "president" in races
    assert "governor" in races
    assert "senate" in races


def test_map_race_type_renames_votes_column(t1_format_elections):
    """total_votes must be renamed to votes_total for downstream functions."""
    t1_format_elections["dem_share"] = (
        t1_format_elections["dem_votes"] / t1_format_elections["total_votes"]
    )
    mapped = _map_race_type_to_race(t1_format_elections)
    assert "votes_total" in mapped.columns
    assert "total_votes" not in mapped.columns


def test_map_race_type_senate_variants_all_map_to_senate():
    """SEN, SEN_SPEC, SEN_ROFF, SEN_SPECROFF must all map to 'senate'."""
    df = pd.DataFrame({
        "tract_geoid": ["T1", "T1", "T1", "T1"],
        "year": [2022, 2021, 2016, 2021],
        "race_type": ["SEN", "SEN_SPEC", "SEN_ROFF", "SEN_SPECROFF"],
        "total_votes": [1000, 900, 800, 700],
        "dem_share": [0.5, 0.5, 0.5, 0.5],
    })
    mapped = _map_race_type_to_race(df)
    assert (mapped["race"] == "senate").all()
    assert len(mapped) == 4


def test_map_race_type_pres_maps_to_president():
    """PRES must map to 'president'."""
    df = pd.DataFrame({
        "tract_geoid": ["T1", "T1"],
        "year": [2020, 2024],
        "race_type": ["PRES", "PRES"],
        "total_votes": [1000, 1000],
        "dem_share": [0.52, 0.51],
    })
    mapped = _map_race_type_to_race(df)
    assert (mapped["race"] == "president").all()


# ---------------------------------------------------------------------------
# τ clipping test
# ---------------------------------------------------------------------------

def test_turnout_ratios_clipped_to_bounds():
    """τ must always stay within [TAU_CLIP_MIN, TAU_CLIP_MAX] regardless of raw ratios."""
    # Create a pathological case: off-cycle votes exceed presidential (τ > 1.5 raw).
    tract_votes = pd.DataFrame({
        "tract_geoid": ["T1", "T1"],
        "year": [2020, 2022],
        "race": ["president", "governor"],
        "votes_total": [100, 1000],  # raw ratio = 10.0, should clip to 1.5
        "dem_share": [0.5, 0.5],
    })
    type_scores = pd.DataFrame(
        {"type_0_score": [1.0]}, index=pd.Index(["T1"], name="GEOID")
    )
    tau = compute_turnout_ratios(tract_votes, type_scores, n_types=1)
    assert tau[0] <= TAU_CLIP_MAX, f"τ={tau[0]} exceeds TAU_CLIP_MAX={TAU_CLIP_MAX}"
    assert tau[0] >= TAU_CLIP_MIN, f"τ={tau[0]} below TAU_CLIP_MIN={TAU_CLIP_MIN}"


# ---------------------------------------------------------------------------
# δ computation correctness test
# ---------------------------------------------------------------------------

def test_choice_shift_direction():
    """δ sign must match the direction of off-cycle Dem share shift."""
    # Type 0 tracts (T1, T2) go more Dem in off-cycle; type 1 tracts (T3, T4) go less Dem.
    tract_votes = pd.DataFrame({
        "tract_geoid": ["T1", "T2", "T3", "T4",   # PRES
                        "T1", "T2", "T3", "T4"],   # GOV
        "year": [2020] * 4 + [2022] * 4,
        "race": ["president"] * 4 + ["governor"] * 4,
        "votes_total": [1000] * 8,
        "dem_share": [0.45, 0.45, 0.55, 0.55,   # presidential
                      0.50, 0.50, 0.50, 0.50],   # off-cycle (T1/T2 +0.05, T3/T4 -0.05)
    })
    type_scores = pd.DataFrame({
        "type_0_score": [0.9, 0.9, 0.1, 0.1],
        "type_1_score": [0.1, 0.1, 0.9, 0.9],
    }, index=pd.Index(["T1", "T2", "T3", "T4"], name="GEOID"))

    tau = compute_turnout_ratios(tract_votes, type_scores, n_types=2)
    delta = compute_choice_shifts(tract_votes, type_scores, tau, n_types=2)

    # Type 0 gained Dem share → positive δ
    assert delta[0] > 0, f"Expected δ[0] > 0, got {delta[0]}"
    # Type 1 lost Dem share → negative δ
    assert delta[1] < 0, f"Expected δ[1] < 0, got {delta[1]}"


# ---------------------------------------------------------------------------
# T.1/T.3 integration test: train_and_save with pipeline format
# ---------------------------------------------------------------------------

def test_train_and_save_with_t1_t3_format(t1_format_elections, t3_format_assignments):
    """End-to-end integration: train_and_save must work with T.1/T.3 data formats."""
    # Compute dem_share (normally done by T.1 pipeline).
    t1_format_elections["dem_share"] = (
        t1_format_elections["dem_votes"] / t1_format_elections["total_votes"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        votes_path = tmpdir / "tract_elections.parquet"
        assignments_path = tmpdir / "tract_type_assignments.parquet"
        output_dir = tmpdir / "behavior"

        t1_format_elections.to_parquet(votes_path, index=False)
        t3_format_assignments.to_parquet(assignments_path, index=False)

        summary = train_and_save(votes_path, assignments_path, output_dir)

        # Outputs must exist.
        assert (output_dir / "tau.npy").exists()
        assert (output_dir / "delta.npy").exists()
        assert (output_dir / "summary.json").exists()

        tau = np.load(output_dir / "tau.npy")
        delta = np.load(output_dir / "delta.npy")

        assert tau.shape == (2,), f"Expected τ shape (2,), got {tau.shape}"
        assert delta.shape == (2,), f"Expected δ shape (2,), got {delta.shape}"

        # τ must be within the clipping range.
        assert (tau >= TAU_CLIP_MIN).all()
        assert (tau <= TAU_CLIP_MAX).all()

        # Summary dict must contain key fields.
        assert summary["n_types"] == 2
        assert "n_tracts" in summary
        assert "tau_mean" in summary
        assert "delta_mean" in summary

        # summary.json must be valid JSON with the same content.
        with open(output_dir / "summary.json") as f:
            saved_summary = json.load(f)
        assert saved_summary["n_types"] == 2


def test_train_and_save_filters_irrelevant_races(t1_format_elections, t3_format_assignments):
    """CONG and AG rows in tract_elections must not inflate τ or distort δ."""
    t1_format_elections["dem_share"] = (
        t1_format_elections["dem_votes"] / t1_format_elections["total_votes"]
    )

    # Control: drop CONG and AG rows manually.
    relevant_only = t1_format_elections[
        t1_format_elections["race_type"].isin({"PRES", "GOV", "SEN", "SEN_SPEC"})
    ].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        assignments_path = tmpdir / "assignments.parquet"
        t3_format_assignments.to_parquet(assignments_path, index=False)

        full_path = tmpdir / "full_elections.parquet"
        filtered_path = tmpdir / "filtered_elections.parquet"
        t1_format_elections.to_parquet(full_path, index=False)
        relevant_only.to_parquet(filtered_path, index=False)

        summary_full = train_and_save(full_path, assignments_path, tmpdir / "out_full")
        summary_filtered = train_and_save(filtered_path, assignments_path, tmpdir / "out_filtered")

    # Results must be identical — CONG/AG rows should have zero impact.
    assert abs(summary_full["tau_mean"] - summary_filtered["tau_mean"]) < 1e-10
    assert abs(summary_full["delta_mean"] - summary_filtered["delta_mean"]) < 1e-10
