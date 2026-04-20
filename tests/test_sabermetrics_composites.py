"""Tests for src/sabermetrics/composites.py — fit scoring and career CTOV aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sabermetrics.composites import (
    compute_career_ctovs,
    compute_fit_score,
    rank_candidates_for_district,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

J = 10  # number of types (small for testing)


def _make_ctov_df(n_candidates: int = 4, races_per: int = 2) -> pd.DataFrame:
    """Synthetic CTOV DataFrame with n_candidates × races_per rows."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_candidates):
        pid = f"CAND{i:03d}"
        for r in range(races_per):
            ctov = rng.normal(0, 0.02, J)
            row = {
                "person_id": pid,
                "name": f"Candidate {i}",
                "party": "D" if i % 2 == 0 else "R",
                "year": 2018 + r * 4,
                "state": "GA",
                "office": "Senate",
                "actual_dem_share": 0.5,
                "pred_dem_share": 0.48,
                "mvd": 0.02,
            }
            for j in range(J):
                row[f"ctov_type_{j}"] = ctov[j]
            rows.append(row)
    return pd.DataFrame(rows)


def _make_career_ctov(n_candidates: int = 4, races_per: int = 2) -> pd.DataFrame:
    return compute_career_ctovs(_make_ctov_df(n_candidates, races_per))


# ---------------------------------------------------------------------------
# compute_career_ctovs
# ---------------------------------------------------------------------------


class TestComputeCareerCtovs:
    def test_output_shape(self):
        ctov = _make_ctov_df(n_candidates=4, races_per=2)
        career = compute_career_ctovs(ctov)
        # One row per (person_id, party, name)
        assert len(career) == 4

    def test_n_races_counts(self):
        ctov = _make_ctov_df(n_candidates=3, races_per=3)
        career = compute_career_ctovs(ctov)
        assert (career["n_races"] == 3).all()

    def test_single_race_career(self):
        ctov = _make_ctov_df(n_candidates=2, races_per=1)
        career = compute_career_ctovs(ctov)
        assert len(career) == 2
        assert (career["n_races"] == 1).all()

    def test_ctov_is_mean(self):
        """Career CTOV should be the mean across races, not the raw values."""
        ctov = _make_ctov_df(n_candidates=1, races_per=2)
        pid = ctov["person_id"].iloc[0]
        rows = ctov[ctov["person_id"] == pid]
        expected_type0 = rows["ctov_type_0"].mean()

        career = compute_career_ctovs(ctov)
        cand_row = career[career["person_id"] == pid].iloc[0]
        assert abs(cand_row["ctov_type_0"] - expected_type0) < 1e-10

    def test_returns_required_columns(self):
        career = _make_career_ctov()
        required = {"person_id", "name", "party", "n_races"}
        assert required.issubset(set(career.columns))

    def test_has_ctov_columns(self):
        career = _make_career_ctov()
        ctov_cols = [c for c in career.columns if c.startswith("ctov_type_")]
        assert len(ctov_cols) == J

    def test_no_ctov_columns_raises(self):
        bad_df = pd.DataFrame({"person_id": ["A"], "name": ["Bob"], "party": ["D"], "year": [2022]})
        with pytest.raises(ValueError, match="ctov_type"):
            compute_career_ctovs(bad_df)

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        """When ctov_df=None and the default parquet doesn't exist, should raise FileNotFoundError."""
        import src.sabermetrics.composites as composites

        monkeypatch.setattr(composites, "_CTOV_PATH", tmp_path / "nonexistent.parquet")
        with pytest.raises(FileNotFoundError):
            compute_career_ctovs(None)


# ---------------------------------------------------------------------------
# compute_fit_score
# ---------------------------------------------------------------------------


class TestComputeFitScore:
    def test_basic_dot_product(self):
        ctov = np.array([0.1, -0.05, 0.2, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        W = np.array([0.5, 0.3, 0.1, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
        expected = float(np.dot(ctov, W))
        assert abs(compute_fit_score(ctov, W) - expected) < 1e-10

    def test_zero_ctov(self):
        ctov = np.zeros(J)
        W = np.ones(J) / J
        assert compute_fit_score(ctov, W) == 0.0

    def test_perfect_alignment(self):
        """Candidate overperforms in exact types the district weights most → high score."""
        W = np.zeros(J)
        W[0] = 1.0  # District is entirely type 0
        ctov = np.zeros(J)
        ctov[0] = 0.05  # Candidate overperforms in type 0
        assert compute_fit_score(ctov, W) == pytest.approx(0.05)

    def test_anti_alignment(self):
        """Candidate underperforms where district is strong → negative score."""
        W = np.zeros(J)
        W[0] = 1.0
        ctov = np.zeros(J)
        ctov[0] = -0.05
        assert compute_fit_score(ctov, W) == pytest.approx(-0.05)

    def test_returns_float(self):
        assert isinstance(compute_fit_score(np.zeros(J), np.ones(J) / J), float)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_fit_score(np.zeros(5), np.zeros(J))


# ---------------------------------------------------------------------------
# rank_candidates_for_district
# ---------------------------------------------------------------------------


class TestRankCandidatesForDistrict:
    def _uniform_W(self) -> np.ndarray:
        return np.ones(J) / J

    def test_output_columns(self):
        career = _make_career_ctov(4, 2)
        W = self._uniform_W()
        result = rank_candidates_for_district(career, W)
        required = {"rank", "person_id", "name", "party", "n_races", "fit_score", "top_types"}
        assert required.issubset(set(result.columns))

    def test_rank_starts_at_1(self):
        career = _make_career_ctov(4, 2)
        result = rank_candidates_for_district(career, self._uniform_W())
        assert result["rank"].iloc[0] == 1

    def test_sorted_descending(self):
        career = _make_career_ctov(4, 2)
        result = rank_candidates_for_district(career, self._uniform_W())
        scores = result["fit_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_party_filter(self):
        career = _make_career_ctov(4, 2)
        result_d = rank_candidates_for_district(career, self._uniform_W(), party_filter="D")
        assert (result_d["party"] == "D").all()

        result_r = rank_candidates_for_district(career, self._uniform_W(), party_filter="R")
        assert (result_r["party"] == "R").all()

    def test_min_races_filter(self):
        """Candidates with fewer than min_races should be excluded."""
        # Create some candidates with 1 race and some with 2
        ctov_1 = _make_ctov_df(n_candidates=2, races_per=1)
        ctov_2 = _make_ctov_df(n_candidates=2, races_per=2)
        # Adjust person_ids to avoid collision
        ctov_2["person_id"] = ctov_2["person_id"] + "_v2"
        ctov_2["name"] = ctov_2["name"] + " B"
        combined = pd.concat([ctov_1, ctov_2], ignore_index=True)
        career = compute_career_ctovs(combined)

        # min_races=2 should exclude the 1-race candidates
        result = rank_candidates_for_district(career, self._uniform_W(), min_races=2)
        assert (result["n_races"] >= 2).all()

    def test_empty_pool_returns_empty(self):
        career = _make_career_ctov(4, 2)
        result = rank_candidates_for_district(career, self._uniform_W(), party_filter="L")
        assert len(result) == 0
        assert "rank" in result.columns

    def test_top_types_length(self):
        career = _make_career_ctov(3, 2)
        result = rank_candidates_for_district(career, self._uniform_W())
        for top_types in result["top_types"]:
            assert len(top_types) <= 5  # _N_TOP_TYPES

    def test_top_types_are_valid_indices(self):
        career = _make_career_ctov(3, 2)
        result = rank_candidates_for_district(career, self._uniform_W())
        for top_types in result["top_types"]:
            assert all(0 <= t < J for t in top_types)

    def test_W_dimension_mismatch_raises(self):
        career = _make_career_ctov(2, 2)
        wrong_W = np.ones(J + 3) / (J + 3)
        with pytest.raises(ValueError, match="W length"):
            rank_candidates_for_district(career, wrong_W)

    def test_high_fit_candidate_ranked_first(self):
        """Candidate that strongly overperforms in the district's dominant types scores highest."""
        # Construct a synthetic scenario: district is 100% type 0.
        # Candidate A overperforms in type 0; Candidate B is flat.
        W = np.zeros(J)
        W[0] = 1.0

        ctov_df = _make_ctov_df(2, 2)
        # Force candidate 0 to have strong positive CTOV in type 0.
        ctov_df.loc[ctov_df["person_id"] == "CAND000", "ctov_type_0"] = 0.10
        # Force candidate 1 to have negative CTOV in type 0.
        ctov_df.loc[ctov_df["person_id"] == "CAND001", "ctov_type_0"] = -0.10

        career = compute_career_ctovs(ctov_df)
        result = rank_candidates_for_district(career, W)

        top_candidate = result.iloc[0]["person_id"]
        assert top_candidate == "CAND000"
