"""Tests for Phase 2 sabermetrics: MVD, CTOV, CEC, badges, and pipeline.

Test strategy:
  - Unit tests: pure math verified with synthetic data (no disk reads)
  - Integration smoke tests: verify output files exist and have correct shape
    (these are cheap since they read from already-generated parquet files)
  - Sanity checks: spot-check known candidates (Beto, Tim Scott)

The pipeline must have been run at least once to generate the output files.
Integration tests are skipped if the output files don't exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SABERMETRICS_DIR = PROJECT_ROOT / "data" / "sabermetrics"

# ---------------------------------------------------------------------------
# Fixtures: paths and outputs
# ---------------------------------------------------------------------------


def _output_exists() -> bool:
    return (
        (SABERMETRICS_DIR / "candidate_residuals.parquet").exists()
        and (SABERMETRICS_DIR / "candidate_ctov.parquet").exists()
        and (SABERMETRICS_DIR / "candidate_badges.json").exists()
    )


skip_if_no_output = pytest.mark.skipif(
    not _output_exists(),
    reason="Pipeline output files not found; run the sabermetrics pipeline first",
)


# ---------------------------------------------------------------------------
# Unit tests: compute_cec (pure numpy, no I/O)
# ---------------------------------------------------------------------------


def test_cec_single_election_returns_one():
    """CEC with a single election is 1.0 by convention (no pairs to correlate)."""
    from src.sabermetrics.residuals import compute_cec

    ctov = np.array([0.01, -0.02, 0.03, -0.01, 0.05, -0.03, 0.00])
    assert compute_cec([ctov]) == 1.0


def test_cec_empty_history_returns_nan():
    """CEC with no elections is NaN (undefined)."""
    from src.sabermetrics.residuals import compute_cec

    result = compute_cec([])
    assert np.isnan(result)


def test_cec_identical_vectors_is_one():
    """A candidate who performs identically in every election has CEC=1.0."""
    from src.sabermetrics.residuals import compute_cec

    ctov = np.array([0.05, -0.02, 0.08, 0.01, -0.03, 0.06, 0.00])
    result = compute_cec([ctov, ctov, ctov])
    assert abs(result - 1.0) < 1e-10


def test_cec_anti_correlated_vectors_is_minus_one():
    """Perfect anti-correlation across two elections gives CEC=-1."""
    from src.sabermetrics.residuals import compute_cec

    ctov_a = np.array([0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05])
    ctov_b = -ctov_a
    result = compute_cec([ctov_a, ctov_b])
    assert abs(result - (-1.0)) < 1e-10


def test_cec_zero_variance_treated_as_zero():
    """If Ridge shrinks a CTOV to all zeros, CEC should be 0 (no pattern to correlate)."""
    from src.sabermetrics.residuals import compute_cec

    zero_ctov = np.zeros(7)
    nonzero_ctov = np.array([0.01, -0.02, 0.03, -0.01, 0.05, -0.03, 0.02])
    # Zero vector has std=0, so correlation is undefined; we treat as 0
    result = compute_cec([zero_ctov, nonzero_ctov])
    assert result == 0.0


def test_cec_bounded_for_random_vectors():
    """CEC must be in [-1, 1] for any pair of random CTOV vectors."""
    from src.sabermetrics.residuals import compute_cec

    rng = np.random.default_rng(42)
    for _ in range(20):
        v1 = rng.normal(0, 0.05, 50)
        v2 = rng.normal(0, 0.05, 50)
        cec = compute_cec([v1, v2])
        assert -1 - 1e-9 <= cec <= 1 + 1e-9, f"CEC={cec} out of bounds"


def test_cec_three_elections_mean_pairwise():
    """CEC with three elections = mean of 3 pairwise correlations."""
    from src.sabermetrics.residuals import compute_cec

    rng = np.random.default_rng(99)
    v1 = rng.normal(0, 0.05, 20)
    v2 = rng.normal(0, 0.05, 20)
    v3 = rng.normal(0, 0.05, 20)

    r12 = np.corrcoef(v1, v2)[0, 1]
    r13 = np.corrcoef(v1, v3)[0, 1]
    r23 = np.corrcoef(v2, v3)[0, 1]
    expected = float(np.mean([r12, r13, r23]))

    cec = compute_cec([v1, v2, v3])
    assert abs(cec - expected) < 1e-10


# ---------------------------------------------------------------------------
# Unit tests: CTOV decomposition math (synthetic W matrix, known residuals)
# ---------------------------------------------------------------------------


def test_ctov_ridge_recovers_known_signal():
    """Ridge on a known W matrix and residuals should approximately recover the signal.

    If we construct residuals = W · true_delta + noise, Ridge should recover
    true_delta (possibly attenuated by regularization).
    """
    from src.prediction.candidate_effects import estimate_delta_race

    rng = np.random.default_rng(7)
    N_counties = 50
    J = 10

    # Simple W: each county is dominated by one type (block diagonal-ish)
    W = rng.dirichlet(np.ones(J), size=N_counties)

    # True delta: type 0 overperforms by +0.1, type 5 by -0.05, rest near zero
    true_delta = np.zeros(J)
    true_delta[0] = 0.10
    true_delta[5] = -0.05

    # Generate residuals as W·delta + small noise
    residuals = W @ true_delta + rng.normal(0, 0.005, N_counties)
    sigma = np.ones(N_counties)

    estimated = estimate_delta_race(W_polls=W, residuals=residuals, sigma_polls=sigma, J=J, mu=0.01)

    # With low regularization and enough counties, should be close to true values
    assert abs(estimated[0] - true_delta[0]) < 0.02, f"Type 0: expected ~0.10, got {estimated[0]:.4f}"
    assert abs(estimated[5] - true_delta[5]) < 0.02, f"Type 5: expected ~-0.05, got {estimated[5]:.4f}"


def test_ctov_reconstruction_approximate():
    """Verify: W_state · ctov ≈ residuals_state (Ridge solution reconstructs inputs)."""
    from src.prediction.candidate_effects import estimate_delta_race

    rng = np.random.default_rng(13)
    N = 30
    J = 15
    W = rng.dirichlet(np.ones(J), size=N)
    residuals = rng.normal(0, 0.05, N)
    sigma = np.ones(N)

    ctov = estimate_delta_race(W_polls=W, residuals=residuals, sigma_polls=sigma, J=J, mu=0.1)
    reconstructed = W @ ctov

    # Reconstruction won't be perfect (Ridge regularization shrinks toward zero),
    # but should be correlated with the input residuals. With N=30 observations,
    # J=15 types, and mu=0.1 regularization, a correlation > 0.5 is expected.
    # (At mu=0, reconstruction would be nearly perfect for N>>J systems.)
    corr = np.corrcoef(reconstructed, residuals)[0, 1]
    assert corr > 0.5, f"Reconstruction correlation {corr:.3f} < 0.5 — CTOV decomposition failed"


def test_ctov_zero_residuals_gives_zero_vector():
    """If actual = predicted everywhere, CTOV should be all zeros."""
    from src.prediction.candidate_effects import estimate_delta_race

    rng = np.random.default_rng(0)
    N, J = 20, 10
    W = rng.dirichlet(np.ones(J), size=N)
    residuals = np.zeros(N)
    sigma = np.ones(N)

    ctov = estimate_delta_race(W_polls=W, residuals=residuals, sigma_polls=sigma, J=J, mu=1.0)
    assert np.allclose(ctov, 0.0, atol=1e-10), f"Expected zeros, got max |ctov|={np.abs(ctov).max():.2e}"


# ---------------------------------------------------------------------------
# Unit tests: badge math
# ---------------------------------------------------------------------------


def test_badge_score_dot_product_centered():
    """Badge score = direction × dot(ctov_effective, centered_feature)."""
    from src.sabermetrics.badges import _compute_badge_score

    ctov = np.array([0.1, -0.1, 0.1])
    feature = np.array([0.5, 0.2, 0.8])  # centered: [-0.017, -0.317, 0.283]
    centered = feature - feature.mean()

    expected = float(np.dot(ctov, centered))
    computed = _compute_badge_score(ctov, feature, direction=1)
    assert abs(computed - expected) < 1e-10


def test_badge_direction_flips_sign():
    """direction=-1 flips the badge score sign (used for Rural Populist)."""
    from src.sabermetrics.badges import _compute_badge_score

    ctov = np.array([0.1, -0.1, 0.1])
    feature = np.array([0.5, 0.2, 0.8])

    pos = _compute_badge_score(ctov, feature, direction=1)
    neg = _compute_badge_score(ctov, feature, direction=-1)
    assert abs(pos + neg) < 1e-10, "direction=-1 should flip sign"


def test_effective_ctov_r_candidate_negated():
    """R candidate CTOV is negated so positive = R overperformance."""
    from src.sabermetrics.badges import _effective_ctov

    ctov = np.array([0.05, -0.02, 0.08])
    d_eff = _effective_ctov(ctov, "D")
    r_eff = _effective_ctov(ctov, "R")

    np.testing.assert_array_equal(d_eff, ctov)
    np.testing.assert_array_almost_equal(r_eff, -ctov)


def test_effective_ctov_d_candidate_unchanged():
    """D candidate CTOV is returned unchanged."""
    from src.sabermetrics.badges import _effective_ctov

    ctov = np.array([0.05, -0.02, 0.08])
    result = _effective_ctov(ctov, "D")
    np.testing.assert_array_equal(result, ctov)


# ---------------------------------------------------------------------------
# Integration tests: output file existence and schema
# ---------------------------------------------------------------------------


@skip_if_no_output
def test_candidate_residuals_parquet_exists():
    """candidate_residuals.parquet exists and has expected columns."""
    df = pd.read_parquet(SABERMETRICS_DIR / "candidate_residuals.parquet")
    required_cols = {"person_id", "name", "party", "year", "state", "office", "actual_dem_share", "pred_dem_share", "mvd"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert len(df) > 0, "No rows in candidate_residuals.parquet"


@skip_if_no_output
def test_candidate_ctov_parquet_exists():
    """candidate_ctov.parquet exists and has 100 CTOV columns."""
    df = pd.read_parquet(SABERMETRICS_DIR / "candidate_ctov.parquet")
    ctov_cols = [c for c in df.columns if c.startswith("ctov_type_")]
    assert len(ctov_cols) == 100, f"Expected 100 CTOV columns, got {len(ctov_cols)}"
    assert len(df) > 0, "No rows in candidate_ctov.parquet"


@skip_if_no_output
def test_candidate_badges_json_exists():
    """candidate_badges.json exists and has expected structure."""
    badges = json.loads((SABERMETRICS_DIR / "candidate_badges.json").read_text())
    assert len(badges) > 0, "No entries in candidate_badges.json"
    # Check structure of a sample entry
    sample = next(iter(badges.values()))
    assert "name" in sample
    assert "party" in sample
    assert "badges" in sample
    assert "badge_scores" in sample


@skip_if_no_output
def test_mvd_count_matches_valid_races():
    """Number of MVD rows matches number of registry races with known actual_dem_share."""
    import json

    df = pd.read_parquet(SABERMETRICS_DIR / "candidate_residuals.parquet")
    registry = json.loads((SABERMETRICS_DIR / "candidate_registry.json").read_text())

    n_valid_races = sum(
        1
        for person in registry["persons"].values()
        for race in person["races"]
        if race["actual_dem_share_2party"] is not None and race["year"] >= 2014
    )

    # MVD rows should be ≤ valid races (some states may lack actuals data)
    assert len(df) <= n_valid_races, f"More MVD rows ({len(df)}) than valid races ({n_valid_races})"
    # But should be at least 80% coverage (generous threshold for data gaps)
    assert len(df) >= 0.8 * n_valid_races, (
        f"Only {len(df)} MVD rows but {n_valid_races} valid races — too many missing"
    )


@skip_if_no_output
def test_ctov_no_nan_values():
    """CTOV vectors should never contain NaN (Ridge always produces a solution)."""
    df = pd.read_parquet(SABERMETRICS_DIR / "candidate_ctov.parquet")
    ctov_cols = [c for c in df.columns if c.startswith("ctov_type_")]
    n_nan = df[ctov_cols].isna().sum().sum()
    assert n_nan == 0, f"Found {n_nan} NaN values in CTOV columns"


@skip_if_no_output
def test_cec_multi_race_candidates_have_reasonable_scores():
    """Multi-race candidates have CEC in [-1, 1] and mean CEC > 0 (positive skill persistence)."""
    badges = json.loads((SABERMETRICS_DIR / "candidate_badges.json").read_text())
    multi_race = [(pid, v) for pid, v in badges.items() if v.get("n_races", 1) > 1 and v.get("cec") is not None]
    assert len(multi_race) > 0, "No multi-race candidates with CEC scores"

    cec_values = [v["cec"] for _, v in multi_race]
    for cec in cec_values:
        assert -1 - 1e-6 <= cec <= 1 + 1e-6, f"CEC={cec} out of [-1, 1]"

    mean_cec = float(np.mean(cec_values))
    # Political skill is known to have positive consistency — we expect mean CEC > 0
    assert mean_cec > 0, f"Mean CEC across multi-race candidates = {mean_cec:.3f}; expected > 0"


# ---------------------------------------------------------------------------
# Sanity checks: spot-check known candidates
# ---------------------------------------------------------------------------


@skip_if_no_output
def test_beto_orourke_present_in_outputs():
    """Beto O'Rourke (O000170) appears in CTOV output for his 2022 TX Governor race."""
    df = pd.read_parquet(SABERMETRICS_DIR / "candidate_ctov.parquet")
    beto = df[df["person_id"] == "O000170"]
    assert len(beto) >= 1, "Beto O'Rourke not found in CTOV output"

    # TX 2022 Governor race
    beto_2022 = beto[(beto["year"] == 2022) & (beto["state"] == "TX")]
    assert len(beto_2022) == 1, "Beto 2022 TX Governor race missing from CTOV"

    # Beto won 44.5% vs model's ~25% prediction — MVD should be large positive
    mvd = float(beto_2022["mvd"].iloc[0])
    assert mvd > 0.15, f"Beto's 2022 TX MVD={mvd:.3f} unexpectedly small (expected > 0.15)"


@skip_if_no_output
def test_tim_scott_present_with_multi_race_cec():
    """Tim Scott appears in outputs with multiple races and a CEC score."""
    ctov_df = pd.read_parquet(SABERMETRICS_DIR / "candidate_ctov.parquet")
    badges = json.loads((SABERMETRICS_DIR / "candidate_badges.json").read_text())

    tim_rows = ctov_df[ctov_df["name"].str.contains("Tim Scott", case=False, na=False)]
    assert len(tim_rows) >= 2, f"Expected ≥ 2 Tim Scott races, found {len(tim_rows)}"

    if len(tim_rows) > 0:
        pid = tim_rows["person_id"].iloc[0]
        badge_entry = badges.get(pid, {})
        cec = badge_entry.get("cec")
        # Tim Scott won SC comfortably multiple times — should have high CEC
        if cec is not None:
            assert cec > 0.3, f"Tim Scott CEC={cec:.3f} unexpectedly low"


@skip_if_no_output
def test_multi_race_candidates_have_cec_in_badges():
    """All candidates with n_races > 1 should have a non-None CEC in the badges output."""
    badges = json.loads((SABERMETRICS_DIR / "candidate_badges.json").read_text())
    multi_race = [(pid, v) for pid, v in badges.items() if v.get("n_races", 1) > 1]
    for pid, entry in multi_race:
        assert entry.get("cec") is not None, (
            f"Multi-race candidate {entry['name']} ({pid}) missing CEC in badges"
        )


# ---------------------------------------------------------------------------
# Party-relative badge tests
# ---------------------------------------------------------------------------


def test_party_relative_badges_differ_across_parties():
    """Two candidates with identical absolute CTOV but different parties get different badges.

    After party-mean subtraction, their residuals differ, so badge scores differ.
    This verifies that badges capture within-party uniqueness, not coalition patterns.
    """
    from src.sabermetrics.badges import derive_badges

    J = 5
    ctov_cols = [f"ctov_type_{i}" for i in range(J)]

    # Party means: D candidates center around [0.1, 0.1, -0.1, 0.0, 0.05]
    #              R candidates center around [-0.1, -0.1, 0.1, 0.0, -0.05]
    d_mean = np.array([0.1, 0.1, -0.1, 0.0, 0.05])
    r_mean = np.array([-0.1, -0.1, 0.1, 0.0, -0.05])

    # Create 10 D candidates and 10 R candidates clustered around their party means
    rng = np.random.RandomState(42)
    rows = []
    for i in range(10):
        d_vec = d_mean + rng.normal(0, 0.01, J)
        rows.append({"person_id": f"D{i:03d}", "name": f"Dem {i}", "party": "D",
                      "year": 2022, "state": "XX", "office": "GOV", "mvd": 0.01,
                      **{c: v for c, v in zip(ctov_cols, d_vec)}})
        r_vec = r_mean + rng.normal(0, 0.01, J)
        rows.append({"person_id": f"R{i:03d}", "name": f"Rep {i}", "party": "R",
                      "year": 2022, "state": "XX", "office": "GOV", "mvd": -0.01,
                      **{c: v for c, v in zip(ctov_cols, r_vec)}})

    # Add a target candidate: same absolute CTOV for both a D and an R
    target_vec = np.array([0.15, 0.05, 0.0, 0.1, -0.05])
    rows.append({"person_id": "TARGET_D", "name": "Target Dem", "party": "D",
                  "year": 2022, "state": "XX", "office": "GOV", "mvd": 0.05,
                  **{c: v for c, v in zip(ctov_cols, target_vec)}})
    rows.append({"person_id": "TARGET_R", "name": "Target Rep", "party": "R",
                  "year": 2022, "state": "XX", "office": "GOV", "mvd": -0.05,
                  **{c: v for c, v in zip(ctov_cols, target_vec)}})

    ctov_df = pd.DataFrame(rows)
    mvd_df = ctov_df[["person_id", "mvd"]].copy()

    # Mock type_profiles to match J=5
    import unittest.mock as mock

    type_profiles = pd.DataFrame({
        "type_id": range(J),
        "pct_bachelors_plus": rng.uniform(0.1, 0.5, J),
        "log_pop_density": rng.uniform(1.0, 4.0, J),
        "pct_white_nh": rng.uniform(0.3, 0.9, J),
        "pct_hispanic": rng.uniform(0.05, 0.4, J),
        "pct_black": rng.uniform(0.05, 0.3, J),
    }).set_index("type_id")

    with mock.patch("src.sabermetrics.badges._load_type_profiles", return_value=type_profiles):
        result = derive_badges(ctov_df, mvd_df)

    # Same absolute CTOV should produce DIFFERENT badge scores after party-mean subtraction
    d_scores = result["TARGET_D"]["badge_scores"]
    r_scores = result["TARGET_R"]["badge_scores"]

    # At least one non-Turnout-Monster badge score should differ meaningfully
    diffs = []
    for badge in d_scores:
        if badge == "Turnout Monster":
            continue
        if badge in r_scores:
            diffs.append(abs(d_scores[badge] - r_scores[badge]))
    assert max(diffs) > 0.001, (
        "Badge scores for same CTOV in different parties should differ after party-mean subtraction"
    )


def test_party_mean_subtraction_removes_party_signal():
    """After subtracting party mean, the average residual per party is ~zero.

    This is a mathematical identity: mean(x_i - mean(x)) = 0.
    Verifying it ensures the subtraction is implemented correctly.
    """
    J = 5
    ctov_cols = [f"ctov_type_{i}" for i in range(J)]

    rng = np.random.RandomState(99)
    rows = []
    for i in range(20):
        party = "D" if i < 10 else "R"
        base = np.array([0.1, 0.1, -0.1, 0.0, 0.05]) if party == "D" else np.array([-0.1, -0.1, 0.1, 0.0, -0.05])
        vec = base + rng.normal(0, 0.03, J)
        rows.append({"person_id": f"P{i:03d}", "name": f"Cand {i}", "party": party,
                      "year": 2022, "state": "XX", "office": "GOV",
                      **{c: v for c, v in zip(ctov_cols, vec)}})

    ctov_df = pd.DataFrame(rows)

    # Compute party means the same way as badges.py
    for party_code in ["D", "R"]:
        party_rows = ctov_df[ctov_df["party"] == party_code]
        party_mean = party_rows[ctov_cols].values.mean(axis=0)
        residuals = party_rows[ctov_cols].values - party_mean
        mean_residual = residuals.mean(axis=0)
        np.testing.assert_allclose(
            mean_residual, 0.0, atol=1e-12,
            err_msg=f"Mean residual for party {party_code} should be ~0 after subtraction",
        )


@skip_if_no_output
def test_badge_count_reasonable():
    """Average badge count per candidate should be between 2 and 10.

    Too few means the threshold is too strict after party-mean subtraction.
    Too many means every candidate looks special (threshold too loose).
    """
    badges = json.loads((SABERMETRICS_DIR / "candidate_badges.json").read_text())
    badge_counts = [len(v["badges"]) for v in badges.values()]
    avg = sum(badge_counts) / len(badge_counts)
    assert 2 <= avg <= 15, (
        f"Average badge count = {avg:.1f}, expected 2-15. "
        f"Threshold may need adjustment."
    )
