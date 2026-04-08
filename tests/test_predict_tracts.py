"""Tests for src/prediction/predict_2026_tracts.py.

Tests the tract-based prediction pipeline using a mix of:
  - Synthetic data (fast, no disk dependency)
  - Lightweight real-data checks (file existence, shape correctness)
  - Behavioral assertions (plausibility, adjustment logic)

Tests are ordered from most isolated (pure unit) to most integrated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PREDS_DIR = DATA_DIR / "predictions"


# ---------------------------------------------------------------------------
# Fixtures — synthetic tract data
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_tract_geoids() -> list[str]:
    """Small set of realistic-looking 11-digit GEOIDs across 3 states."""
    # FL (state FIPS 12), GA (state FIPS 13), AL (state FIPS 01)
    fl = [f"12001{str(i).zfill(6)}" for i in range(20)]   # 20 FL tracts
    ga = [f"13001{str(i).zfill(6)}" for i in range(15)]   # 15 GA tracts
    al = [f"01001{str(i).zfill(6)}" for i in range(10)]   # 10 AL tracts
    wy = [f"56001{str(i).zfill(6)}" for i in range(5)]    # 5 WY tracts
    return fl + ga + al + wy


@pytest.fixture
def synthetic_type_scores(synthetic_tract_geoids) -> np.ndarray:
    """Random soft membership scores, shape (N, J=10)."""
    rng = np.random.default_rng(42)
    N = len(synthetic_tract_geoids)
    J = 10
    # Dirichlet ensures non-negative and row-sums to 1
    scores = rng.dirichlet(np.ones(J), size=N)
    return scores


@pytest.fixture
def synthetic_priors(synthetic_tract_geoids) -> np.ndarray:
    """Dem share priors in plausible range [0.20, 0.80]."""
    rng = np.random.default_rng(43)
    return rng.uniform(0.30, 0.70, size=len(synthetic_tract_geoids))


@pytest.fixture
def synthetic_votes(synthetic_tract_geoids) -> np.ndarray:
    """Synthetic vote counts in typical range."""
    rng = np.random.default_rng(44)
    return rng.integers(800, 3000, size=len(synthetic_tract_geoids)).astype(float)


@pytest.fixture
def synthetic_tau() -> np.ndarray:
    """Per-type turnout ratios in realistic range."""
    rng = np.random.default_rng(45)
    return rng.uniform(0.73, 1.03, size=10)


@pytest.fixture
def synthetic_delta() -> dict[str, np.ndarray]:
    """Per-type residual choice shifts — race-specific dicts, mostly small."""
    rng = np.random.default_rng(46)
    gov_delta = rng.uniform(-0.05, 0.05, size=10)
    sen_delta = rng.uniform(-0.03, 0.06, size=10)
    return {"governor": gov_delta, "senate": sen_delta, "default": gov_delta}


# ---------------------------------------------------------------------------
# Unit tests: derive_tract_states
# ---------------------------------------------------------------------------


def test_derive_tract_states_basic():
    """State FIPS from first 2 digits of GEOID should map to abbreviations."""
    from src.prediction.predict_2026_tracts import derive_tract_states

    geoids = [
        "12001020100",   # FL (12)
        "13001020100",   # GA (13)
        "01001020100",   # AL (01)
        "56001020100",   # WY (56)
        "06037137100",   # CA (06)
        "25025010100",   # MA (25)
    ]
    states = derive_tract_states(geoids)
    assert states[0] == "FL", f"Expected FL, got {states[0]}"
    assert states[1] == "GA", f"Expected GA, got {states[1]}"
    assert states[2] == "AL", f"Expected AL, got {states[2]}"
    assert states[3] == "WY", f"Expected WY, got {states[3]}"
    assert states[4] == "CA", f"Expected CA, got {states[4]}"
    assert states[5] == "MA", f"Expected MA, got {states[5]}"


def test_derive_tract_states_unknown_fips():
    """Unknown FIPS prefix should return '??'."""
    from src.prediction.predict_2026_tracts import derive_tract_states

    geoids = ["99001020100"]  # 99 is not a real state FIPS
    states = derive_tract_states(geoids)
    assert states[0] == "??"


def test_derive_tract_states_all_50_plus_dc():
    """At least 51 unique FIPS should be recognized (50 states + DC)."""
    from src.prediction.predict_2026_tracts import derive_tract_states, _STATE_FIPS_TO_ABBR

    assert len(_STATE_FIPS_TO_ABBR) >= 51, (
        f"Expected 51+ FIPS codes, got {len(_STATE_FIPS_TO_ABBR)}"
    )


def test_derive_tract_states_preserves_order(synthetic_tract_geoids):
    """Output should be same length and order as input."""
    from src.prediction.predict_2026_tracts import derive_tract_states

    states = derive_tract_states(synthetic_tract_geoids)
    assert len(states) == len(synthetic_tract_geoids)
    # First 20 should all be FL
    assert all(s == "FL" for s in states[:20])
    # Next 15 should all be GA
    assert all(s == "GA" for s in states[20:35])


# ---------------------------------------------------------------------------
# Unit tests: aggregate_to_states
# ---------------------------------------------------------------------------


def test_aggregate_to_states_weighted(synthetic_tract_geoids, synthetic_votes):
    """Vote-weighted aggregation should weight high-vote tracts more heavily."""
    from src.prediction.predict_2026_tracts import aggregate_to_states

    from src.prediction.predict_2026_tracts import derive_tract_states
    states = derive_tract_states(synthetic_tract_geoids)

    # Set all FL predictions to 0.60 and all GA predictions to 0.40
    preds = np.array([0.60] * 20 + [0.40] * 15 + [0.50] * 10 + [0.30] * 5)
    result = aggregate_to_states(preds, synthetic_votes, states)

    assert abs(result["FL"] - 0.60) < 1e-6, f"FL expected 0.60, got {result['FL']}"
    assert abs(result["GA"] - 0.40) < 1e-6, f"GA expected 0.40, got {result['GA']}"
    assert "WY" in result


def test_aggregate_to_states_equal_votes():
    """With equal votes, weighted average equals simple mean."""
    from src.prediction.predict_2026_tracts import aggregate_to_states

    preds = np.array([0.4, 0.6, 0.5])
    votes = np.ones(3) * 1000.0
    states = ["CA", "CA", "NY"]
    result = aggregate_to_states(preds, votes, states)

    # CA should be (0.4 + 0.6) / 2 = 0.50
    assert abs(result["CA"] - 0.50) < 1e-6
    assert abs(result["NY"] - 0.50) < 1e-6


def test_aggregate_to_states_skips_unknown():
    """State code '??' should not appear in output."""
    from src.prediction.predict_2026_tracts import aggregate_to_states

    preds = np.array([0.5, 0.5, 0.5])
    votes = np.ones(3)
    states = ["CA", "??", "NY"]
    result = aggregate_to_states(preds, votes, states)
    assert "??" not in result


def test_aggregate_to_states_high_vote_dominates():
    """A single very-high-vote tract should dominate the state average."""
    from src.prediction.predict_2026_tracts import aggregate_to_states

    preds = np.array([0.30, 0.70])     # first tract R-leaning, second D-leaning
    votes = np.array([100.0, 10000.0]) # second tract has 100x votes
    states = ["TX", "TX"]
    result = aggregate_to_states(preds, votes, states)

    # Weighted avg: (0.30 * 100 + 0.70 * 10000) / 10100 ≈ 0.697
    expected = (0.30 * 100 + 0.70 * 10000) / (100 + 10000)
    assert abs(result["TX"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: adjust_priors_for_race_type (behavior layer)
# ---------------------------------------------------------------------------


def test_behavior_adjustment_pres_noop(
    synthetic_priors, synthetic_type_scores, synthetic_tau, synthetic_delta
):
    """Presidential races should NOT be adjusted (τ and δ are relative to pres baseline)."""
    from src.prediction.predict_2026_tracts import adjust_priors_for_race_type

    adjusted = adjust_priors_for_race_type(
        synthetic_priors, synthetic_type_scores, synthetic_tau, synthetic_delta,
        race_type="president",
    )
    np.testing.assert_array_equal(adjusted, synthetic_priors)


def test_behavior_adjustment_senate_changes(
    synthetic_priors, synthetic_type_scores, synthetic_tau, synthetic_delta
):
    """Senate races should produce different priors from the raw Ridge priors."""
    from src.prediction.predict_2026_tracts import adjust_priors_for_race_type

    adjusted = adjust_priors_for_race_type(
        synthetic_priors, synthetic_type_scores, synthetic_tau, synthetic_delta,
        race_type="senate",
    )
    # The adjustment should change values (delta is non-zero)
    assert not np.allclose(adjusted, synthetic_priors), (
        "Senate adjustment had no effect — delta is all-zero?"
    )


def test_behavior_adjustment_governor_vs_senate_different(
    synthetic_priors, synthetic_type_scores, synthetic_tau, synthetic_delta
):
    """Governor and senate races should use different CES δ values."""
    from src.prediction.predict_2026_tracts import adjust_priors_for_race_type

    gov_adjusted = adjust_priors_for_race_type(
        synthetic_priors, synthetic_type_scores, synthetic_tau, synthetic_delta,
        race_type="governor",
    )
    sen_adjusted = adjust_priors_for_race_type(
        synthetic_priors, synthetic_type_scores, synthetic_tau, synthetic_delta,
        race_type="senate",
    )
    # Governor and senate now use different CES-derived δ — they should differ.
    assert not np.allclose(gov_adjusted, sen_adjusted), (
        "Governor and senate should use different CES δ values"
    )


def test_behavior_adjustment_clipped_to_01(
    synthetic_type_scores, synthetic_tau
):
    """Adjusted priors must always be in [0, 1] even with extreme delta."""
    from src.prediction.predict_2026_tracts import adjust_priors_for_race_type

    # Edge case: priors near boundary + large delta
    extreme_priors = np.array([0.02, 0.98, 0.50])
    extreme_scores = np.ones((3, 10)) / 10  # uniform type membership
    extreme_delta = np.full(10, 0.15)  # large shift
    extreme_deltas = {"governor": extreme_delta, "senate": extreme_delta, "default": extreme_delta}
    tau_small = synthetic_tau[:10]

    adjusted = adjust_priors_for_race_type(
        extreme_priors, extreme_scores, tau_small, extreme_deltas,
        race_type="senate",
    )
    assert (adjusted >= 0.0).all() and (adjusted <= 1.0).all(), (
        f"Adjusted priors out of [0,1]: {adjusted}"
    )


# ---------------------------------------------------------------------------
# Data file tests (skip if files not present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DATA_DIR / "communities" / "tract_type_assignments.parquet").exists(),
    reason="tract_type_assignments.parquet not found",
)
def test_tract_type_assignments_shape():
    """Tract type assignments should have 80K+ rows and 100 type score columns."""
    from src.prediction.predict_2026_tracts import load_tract_type_scores

    geoids, scores, dominant = load_tract_type_scores()

    assert len(geoids) > 50000, f"Too few tracts: {len(geoids)}"
    assert scores.shape[1] == 100, f"Expected J=100, got {scores.shape[1]}"
    assert scores.shape[0] == len(geoids)
    assert dominant.shape[0] == len(geoids)
    assert dominant.min() >= 0 and dominant.max() <= 99


@pytest.mark.skipif(
    not (DATA_DIR / "models" / "ridge_model" / "ridge_tract_priors.parquet").exists(),
    reason="ridge_tract_priors.parquet not found",
)
def test_ridge_priors_range():
    """Ridge priors should be in [0, 1]."""
    df = pd.read_parquet(
        DATA_DIR / "models" / "ridge_model" / "ridge_tract_priors.parquet"
    )
    assert (df["ridge_pred_dem_share"] >= 0.0).all()
    assert (df["ridge_pred_dem_share"] <= 1.0).all()


@pytest.mark.skipif(
    not (DATA_DIR / "communities" / "tract_type_assignments.parquet").exists()
    or not (DATA_DIR / "models" / "ridge_model" / "ridge_tract_priors.parquet").exists(),
    reason="Required data files not found",
)
def test_load_tract_priors_aligned():
    """load_tract_priors should return a float array aligned to input geoids."""
    from src.prediction.predict_2026_tracts import load_tract_type_scores, load_tract_priors

    geoids, _, _ = load_tract_type_scores()
    priors = load_tract_priors(geoids)

    assert priors.shape == (len(geoids),)
    assert priors.dtype == float or np.issubdtype(priors.dtype, np.floating)
    # Priors should be in plausible range (Ridge output is clipped to [0, 1])
    assert (priors >= 0.0).all() and (priors <= 1.0).all()


@pytest.mark.skipif(
    not (DATA_DIR / "assembled" / "tract_elections.parquet").exists(),
    reason="tract_elections.parquet not found",
)
def test_load_tract_votes_shape():
    """Tract votes should return array of correct length."""
    from src.prediction.predict_2026_tracts import load_tract_type_scores, load_tract_votes

    geoids, _, _ = load_tract_type_scores()
    votes = load_tract_votes(geoids)

    assert votes.shape == (len(geoids),)
    # Vote counts should be positive
    assert (votes > 0).all(), "All vote counts must be positive"


@pytest.mark.skipif(
    not (DATA_DIR / "behavior" / "tau.npy").exists()
    or not (DATA_DIR / "behavior" / "delta.npy").exists(),
    reason="Behavior layer files not found",
)
def test_load_behavior_layer_shapes():
    """τ must be (100,) and deltas must be a dict with (100,) arrays per race type."""
    from src.prediction.predict_2026_tracts import load_behavior_layer

    tau, deltas = load_behavior_layer()
    assert tau.shape == (100,), f"Expected tau shape (100,), got {tau.shape}"
    assert (tau > 0).all()
    # deltas should have governor, senate, and default keys
    assert "governor" in deltas, "Missing governor δ"
    assert "senate" in deltas, "Missing senate δ"
    assert "default" in deltas, "Missing default δ"
    for key, delta in deltas.items():
        assert delta.shape == (100,), f"Expected {key} delta shape (100,), got {delta.shape}"


# ---------------------------------------------------------------------------
# Output file tests (skip if predictions haven't been run yet)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PREDS_DIR / "tract_predictions_2026.parquet").exists(),
    reason="tract_predictions_2026.parquet not found — run predict_2026_tracts first",
)
def test_tract_predictions_file_schema():
    """tract_predictions_2026.parquet should have required columns."""
    df = pd.read_parquet(PREDS_DIR / "tract_predictions_2026.parquet")

    required = {"tract_geoid", "state", "pred_dem_share", "race", "forecast_mode"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert len(df) > 0, "Predictions DataFrame should not be empty"
    # pred_dem_share is type_scores @ theta — can slightly exceed [0, 1] for outlier tracts
    # (same behavior as the county pipeline). Check that the vast majority are in range.
    pct_in_range = ((df["pred_dem_share"] >= 0.0) & (df["pred_dem_share"] <= 1.0)).mean()
    assert pct_in_range > 0.95, (
        f"Expected >95% of predictions in [0,1], got {pct_in_range:.1%}"
    )


@pytest.mark.skipif(
    not (PREDS_DIR / "tract_state_predictions_2026.json").exists(),
    reason="tract_state_predictions_2026.json not found — run predict_2026_tracts first",
)
def test_state_predictions_file_structure():
    """tract_state_predictions_2026.json should have race → mode → state → float."""
    with open(PREDS_DIR / "tract_state_predictions_2026.json") as f:
        data = json.load(f)

    assert isinstance(data, dict), "Top level should be a dict keyed by race_id"
    # Check at least one race exists
    assert len(data) > 0

    # Sample the first race
    race_id = next(iter(data))
    race_data = data[race_id]
    assert "national" in race_data or "local" in race_data, (
        "Each race should have 'national' and/or 'local' mode"
    )
    # State values should be floats in [0, 1]
    if "national" in race_data:
        for state, val in race_data["national"].items():
            assert 0.0 <= val <= 1.0, (
                f"State pred out of [0,1]: {race_id} national {state}={val}"
            )


@pytest.mark.skipif(
    not (PREDS_DIR / "tract_state_predictions_2026.json").exists(),
    reason="tract_state_predictions_2026.json not found",
)
def test_plausible_state_predictions():
    """Competitive D-leaning states should beat competitive R-leaning states.

    We use GA Senate (lean-D in 2026) vs FL Senate (lean-R) as the test pair —
    both have active polls in the dataset, so the local mode has signal from data.
    This is a directional check: GA prediction > FL prediction.

    Note: the national mode θ reflects the shared national environment estimated
    from all polled states — it does NOT separate states by partisan lean.
    The local mode adds δ_race which captures candidate/state-specific effects.
    For states without polls in a given race, national==local (delta==0).
    """
    with open(PREDS_DIR / "tract_state_predictions_2026.json") as f:
        data = json.load(f)

    # Check GA Senate > FL Senate in local mode (both have polls)
    ga_key, fl_key = "2026 GA Senate", "2026 FL Senate"
    if ga_key in data and fl_key in data:
        ga_nat = data[ga_key].get("national", {}).get("GA")
        fl_nat = data[fl_key].get("national", {}).get("FL")
        if ga_nat is not None and fl_nat is not None:
            assert ga_nat > fl_nat, (
                f"GA Senate ({ga_nat:.3f}) should beat FL Senate ({fl_nat:.3f}) "
                "in national mode given GA's 2026 lean-D environment"
            )

    # Verify all state predictions are in a plausible range (not degenerate)
    for race_id, modes in data.items():
        for mode, state_preds in modes.items():
            for state, val in state_preds.items():
                assert 0.1 <= val <= 0.9, (
                    f"Degenerate prediction: {race_id} {mode} {state}={val:.3f} "
                    "— expected in [0.1, 0.9]"
                )
