"""Tests for make_xt_impact_report() and its core helper _xt_delta_from_polls.

Tests exercise _xt_delta_from_polls directly with synthetic data so they run
without any gitignored data files on disk.  The integration test for the public
make_xt_impact_report() is skipped when production data is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import skip_if_missing
from src.prediction.forecast_engine import _xt_delta_from_polls, make_xt_impact_report


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

def _type_scores(n_counties: int = 6, J: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    raw = rng.rand(n_counties, J)
    return raw / raw.sum(axis=1, keepdims=True)


def _type_profiles(J: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "pct_bachelors_plus": np.linspace(0.15, 0.65, J),
        "pct_white_nh":       np.linspace(0.90, 0.40, J),
        "pct_black":          np.linspace(0.02, 0.35, J),
        "pct_hispanic":       np.linspace(0.03, 0.30, J),
        "pct_asian":          np.linspace(0.01, 0.12, J),
        "median_age":         np.linspace(30.0, 55.0, J),
        "log_pop_density":    np.linspace(1.5, 5.0, J),
        "evangelical_share":  np.linspace(0.50, 0.05, J),
    })


def _base_kwargs(J: int = 4, n_counties: int = 6, with_type_profiles: bool = True) -> dict:
    """Return the structural kwargs needed by _xt_delta_from_polls (minus polls + races)."""
    return dict(
        type_scores=_type_scores(n_counties, J),
        county_priors=np.full(n_counties, 0.5),
        states=["AZ"] * 3 + ["NV"] * 3,
        county_votes=np.ones(n_counties) * 1000.0,
        type_profiles=_type_profiles(J) if with_type_profiles else None,
        lam=0.1,
        mu=0.1,
    )


# ---------------------------------------------------------------------------
# a. Dict has required keys
# ---------------------------------------------------------------------------

class TestRequiredKeys:
    REQUIRED = {"enriched_deltas", "mean_delta", "max_delta", "races_with_xt", "report_date"}

    def test_result_has_all_required_keys(self):
        kwargs = _base_kwargs()
        polls = {"2026 AZ Governor": [{"dem_share": 0.52, "n_sample": 600, "state": "AZ"}]}
        result = _xt_delta_from_polls(
            polls_by_race=polls,
            all_race_ids=["2026 AZ Governor"],
            **kwargs,
        )
        assert self.REQUIRED <= result.keys(), (
            f"Missing keys: {self.REQUIRED - result.keys()}"
        )

    def test_enriched_deltas_is_dict(self):
        kwargs = _base_kwargs()
        polls = {"2026 AZ Governor": [{"dem_share": 0.52, "n_sample": 600, "state": "AZ"}]}
        result = _xt_delta_from_polls(
            polls_by_race=polls,
            all_race_ids=["2026 AZ Governor"],
            **kwargs,
        )
        assert isinstance(result["enriched_deltas"], dict)

    def test_scalar_fields_are_numeric(self):
        kwargs = _base_kwargs()
        polls = {"2026 AZ Governor": [{"dem_share": 0.52, "n_sample": 600, "state": "AZ"}]}
        result = _xt_delta_from_polls(
            polls_by_race=polls,
            all_race_ids=["2026 AZ Governor"],
            **kwargs,
        )
        assert isinstance(result["mean_delta"], float)
        assert isinstance(result["max_delta"], float)
        assert isinstance(result["races_with_xt"], int)
        assert isinstance(result["report_date"], str)


# ---------------------------------------------------------------------------
# b. Zero xt_ polls produce zero delta
# ---------------------------------------------------------------------------

class TestZeroXtDelta:
    def test_no_xt_polls_zero_mean_delta(self):
        """Polls without xt_ fields produce enriched == stripped → mean_delta == 0."""
        kwargs = _base_kwargs()
        polls = {
            "2026 AZ Governor": [
                {"dem_share": 0.52, "n_sample": 600, "state": "AZ"},
                {"dem_share": 0.50, "n_sample": 800, "state": "AZ"},
            ],
            "2026 NV Senate": [
                {"dem_share": 0.49, "n_sample": 500, "state": "NV"},
            ],
        }
        result = _xt_delta_from_polls(
            polls_by_race=polls,
            all_race_ids=["2026 AZ Governor", "2026 NV Senate"],
            **kwargs,
        )
        assert result["mean_delta"] == pytest.approx(0.0)
        assert result["max_delta"] == pytest.approx(0.0)
        assert result["races_with_xt"] == 0

    def test_no_xt_polls_zero_per_race_delta(self):
        """Each race's delta should be exactly 0 when no xt_ fields are present."""
        kwargs = _base_kwargs()
        polls = {"2026 AZ Governor": [{"dem_share": 0.55, "n_sample": 700, "state": "AZ"}]}
        result = _xt_delta_from_polls(
            polls_by_race=polls,
            all_race_ids=["2026 AZ Governor"],
            **kwargs,
        )
        for race, delta in result["enriched_deltas"].items():
            assert delta == pytest.approx(0.0), f"Expected 0 delta for {race}, got {delta}"


# ---------------------------------------------------------------------------
# c. Enriched differs from stripped when xt_ data is present
# ---------------------------------------------------------------------------

class TestEnrichedVsStripped:
    def _xt_poll_with_vote_shares(self) -> dict[str, list[dict]]:
        """Poll with per-group vote shares (xt_vote_*) that differ from topline.

        Without per-group vote shares, all Tier 2 observations share the same y
        (topline dem_share), so W-vector shifts mostly cancel at state level.
        Per-group shares inject genuinely different y values → clear delta signal.
        """
        return {
            "2026 AZ Governor": [{
                "dem_share": 0.46,
                "n_sample": 850,
                "state": "AZ",
                "xt_race_white": 0.80,
                "xt_vote_race_white": 0.40,   # 40% dem among white voters
                "xt_race_black": 0.15,
                "xt_vote_race_black": 0.85,   # 85% dem among Black voters
            }],
        }

    def test_xt_poll_produces_nonzero_delta(self):
        """A poll with xt_vote_* fields must produce a non-trivial enriched delta."""
        kwargs = _base_kwargs(with_type_profiles=True)
        result = _xt_delta_from_polls(
            polls_by_race=self._xt_poll_with_vote_shares(),
            all_race_ids=["2026 AZ Governor"],
            **kwargs,
        )
        deltas = list(result["enriched_deltas"].values())
        assert any(abs(d) > 0.1 for d in deltas), (
            "Expected >0.1pp delta when per-group vote shares are present, got all near-zero"
        )

    def test_races_with_xt_count_is_correct(self):
        """races_with_xt should count only races that have at least one xt_ poll."""
        kwargs = _base_kwargs(with_type_profiles=True)
        polls = {
            **self._xt_poll_with_vote_shares(),
            "2026 NV Senate": [
                {"dem_share": 0.49, "n_sample": 600, "state": "NV"},
            ],
        }
        result = _xt_delta_from_polls(
            polls_by_race=polls,
            all_race_ids=["2026 AZ Governor", "2026 NV Senate"],
            **kwargs,
        )
        assert result["races_with_xt"] == 1

    def test_xt_race_delta_is_nontrivial(self):
        """The race with per-group vote shares must produce a delta well above noise."""
        kwargs = _base_kwargs(with_type_profiles=True)
        result = _xt_delta_from_polls(
            polls_by_race=self._xt_poll_with_vote_shares(),
            all_race_ids=["2026 AZ Governor"],
            **kwargs,
        )
        az_delta = abs(result["enriched_deltas"].get("2026 AZ Governor", 0.0))
        assert az_delta > 0.1  # >0.1pp — far above floating-point noise


# ---------------------------------------------------------------------------
# Integration test: make_xt_impact_report with real data
# ---------------------------------------------------------------------------

@skip_if_missing(
    "data/communities/type_assignments.parquet",
    "data/polls/polls_2026.csv",
    "data/communities/type_profiles.parquet",
)
class TestMakeXtImpactReportIntegration:
    REQUIRED = {"enriched_deltas", "mean_delta", "max_delta", "races_with_xt", "report_date"}

    def test_returns_required_keys(self):
        result = make_xt_impact_report()
        assert self.REQUIRED <= result.keys()

    def test_enriched_deltas_contains_race_ids(self):
        result = make_xt_impact_report()
        assert len(result["enriched_deltas"]) > 0
        for race_id in result["enriched_deltas"]:
            assert isinstance(race_id, str)
            assert isinstance(result["enriched_deltas"][race_id], float)

    def test_races_with_xt_nonzero(self):
        """polls_2026.csv is known to have ≥1 race with xt_ crosstab data."""
        result = make_xt_impact_report()
        assert result["races_with_xt"] >= 1
