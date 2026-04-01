"""Test that the prediction pipeline produces forecasts for all registered races."""
import pytest
import pandas as pd
from pathlib import Path

from src.assembly.define_races import load_races


PREDICTIONS_PATH = Path("data/predictions/county_predictions_2026_types.parquet")


@pytest.fixture
def predictions():
    if not PREDICTIONS_PATH.exists():
        pytest.skip("Predictions not generated yet")
    return pd.read_parquet(PREDICTIONS_PATH)


def test_all_registered_races_have_predictions(predictions):
    """Every race in the registry must appear in predictions output."""
    pred_races = set(predictions["race"].unique())
    registry_races = {r.race_id for r in load_races(2026)}
    missing = registry_races - pred_races
    assert len(missing) == 0, f"Races in registry but not predicted: {missing}"


def test_unpolled_race_gets_baseline_prediction(predictions):
    """A race with no polls should still get a prediction (baseline)."""
    polls_path = Path("data/polls/polls_2026.csv")
    if not polls_path.exists():
        pytest.skip("No polls CSV")
    polls = pd.read_csv(polls_path)
    polled_races = set(polls["race"].unique())
    registry_races = {r.race_id for r in load_races(2026)}
    unpolled = registry_races - polled_races
    if not unpolled:
        pytest.skip("All races have polls")
    sample_race = next(iter(unpolled))
    race_preds = predictions[predictions["race"] == sample_race]
    assert len(race_preds) > 0, f"Unpolled race '{sample_race}' has no predictions"
    assert race_preds["pred_dem_share"].notna().all(), "Predictions should not be NaN"


def test_prediction_count_matches_counties_times_races(predictions):
    """Total prediction rows = n_counties * (n_real_races * 2 modes + 1 baseline)."""
    n_real_races = len(load_races(2026))
    n_counties = predictions["county_fips"].nunique()
    # Real races have both national + local modes; baseline has national only
    expected = n_counties * (n_real_races * 2 + 1)
    actual = len(predictions)
    assert actual == expected, (
        f"Expected {expected} rows ({n_counties} counties x ({n_real_races} races × 2 modes + 1 baseline)), got {actual}"
    )
