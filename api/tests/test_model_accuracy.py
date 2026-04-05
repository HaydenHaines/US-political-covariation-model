# api/tests/test_model_accuracy.py
"""Tests for GET /api/v1/model/accuracy endpoint."""
import pytest

# These values mirror data/model/accuracy_metrics.json (the seed file).
# Update this file after any model retrain that changes these metrics.
EXPECTED_LOO_R = pytest.approx(0.732, abs=0.05)
EXPECTED_HOLDOUT_R = pytest.approx(0.698, abs=0.05)
EXPECTED_COHERENCE = pytest.approx(0.783, abs=0.05)
EXPECTED_RMSE = pytest.approx(0.073, abs=0.005)
EXPECTED_COVARIANCE_VAL_R = pytest.approx(0.936, abs=0.05)
EXPECTED_N_COUNTIES = 3154
EXPECTED_N_TYPES = 100
EXPECTED_N_SUPER_TYPES = 5
# Ensemble LOO r (best method); must equal EXPECTED_LOO_R by definition.
EXPECTED_ENSEMBLE_LOO_R = pytest.approx(0.732, abs=0.05)


def test_accuracy_returns_200(client):
    resp = client.get("/api/v1/model/accuracy")
    assert resp.status_code == 200


def test_accuracy_overall_shape(client):
    data = client.get("/api/v1/model/accuracy").json()
    overall = data["overall"]
    assert "loo_r" in overall
    assert "holdout_r" in overall
    assert "coherence" in overall
    assert "rmse" in overall
    assert "covariance_val_r" in overall
    assert "n_counties" in overall
    assert "n_types" in overall
    assert "n_super_types" in overall


def test_accuracy_overall_values(client):
    overall = client.get("/api/v1/model/accuracy").json()["overall"]
    assert overall["loo_r"] == EXPECTED_LOO_R
    assert overall["holdout_r"] == EXPECTED_HOLDOUT_R
    assert overall["coherence"] == EXPECTED_COHERENCE
    assert overall["rmse"] == EXPECTED_RMSE
    assert overall["covariance_val_r"] == EXPECTED_COVARIANCE_VAL_R
    assert overall["n_counties"] == EXPECTED_N_COUNTIES
    assert overall["n_types"] == EXPECTED_N_TYPES
    assert overall["n_super_types"] == EXPECTED_N_SUPER_TYPES


def test_accuracy_cross_election(client):
    data = client.get("/api/v1/model/accuracy").json()
    cycles = data["cross_election"]
    assert len(cycles) == 4
    # All entries have required fields
    for entry in cycles:
        assert "cycle" in entry
        assert "loo_r" in entry
        assert "label" in entry
    # Values are within plausible range
    for entry in cycles:
        assert 0.0 <= entry["loo_r"] <= 1.0


def test_accuracy_cross_election_best_and_worst(client):
    data = client.get("/api/v1/model/accuracy").json()
    cycles = data["cross_election"]
    cycle_map = {e["cycle"]: e["loo_r"] for e in cycles}
    # 2012→2016 was most predictable, 2020→2024 was hardest
    assert cycle_map["2012→2016"] > cycle_map["2020→2024"]


def test_accuracy_method_comparison(client):
    data = client.get("/api/v1/model/accuracy").json()
    methods = data["method_comparison"]
    assert len(methods) == 4
    # All entries have required fields
    for entry in methods:
        assert "method" in entry
        assert "loo_r" in entry
    # Methods should be in ascending order of performance
    loo_values = [m["loo_r"] for m in methods]
    assert loo_values == sorted(loo_values), (
        "method_comparison should be ordered by ascending loo_r"
    )


def test_accuracy_ensemble_is_best(client):
    data = client.get("/api/v1/model/accuracy").json()
    methods = data["method_comparison"]
    loo_values = [m["loo_r"] for m in methods]
    # The ensemble should achieve the highest LOO r
    assert max(loo_values) == EXPECTED_ENSEMBLE_LOO_R
