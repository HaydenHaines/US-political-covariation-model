"""TDD tests for party-relative badge improvements.

The party-mean CTOV subtraction was shipped in commit ba41c0a.
This test suite covers the remaining three changes:

1. Within-party thresholding + small-pool fallback (n < 20 → global fallback)
2. Provisional marker (badge_details with provisional=True for n_races < 2)
3. Signature badges — derive_signature_badges() auto-discovery
4. Party-balance regression (symmetric input → balanced output)
"""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ── Shared constants & helpers ────────────────────────────────────────────────

J = 100  # must match production type count

_BADGE_COLS = [
    "pct_hispanic", "pct_black", "pct_white_nh", "pct_asian",
    "median_age", "pct_bachelors_plus", "pct_graduate", "pct_management",
    "log_pop_density", "pct_transit", "pct_owner_occupied",
    "log_median_hh_income", "pct_wfh", "evangelical_share",
    "mainline_share", "catholic_share", "black_protestant_share",
    "religious_adherence_rate", "earnings_share", "transfers_share",
    "investment_share", "net_migration_rate", "avg_inflow_income",
]


@pytest.fixture
def type_profiles() -> pd.DataFrame:
    """Synthetic type_profiles — no filesystem required."""
    rng = np.random.default_rng(42)
    data: dict = {"type_id": list(range(J))}
    for col in _BADGE_COLS:
        data[col] = rng.uniform(0, 1, J).tolist()
    return pd.DataFrame(data)


def _make_ctov_df(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row = {
            "person_id": c["person_id"],
            "name": c["name"],
            "party": c["party"],
            "year": c.get("year", 2022),
            "state": c.get("state", "TX"),
            "office": c.get("office", "Senate"),
            "mvd": float(c.get("mvd", 0.0)),
        }
        ctov = np.asarray(c.get("ctov", np.zeros(J)), dtype=float)
        for j in range(J):
            row[f"ctov_type_{j}"] = float(ctov[j])
        rows.append(row)
    return pd.DataFrame(rows)


def _make_mvd_df(ctov_df: pd.DataFrame) -> pd.DataFrame:
    return ctov_df[["person_id", "name", "party", "year", "state", "office", "mvd"]].copy()


def _run_derive_badges(candidates: list[dict], type_profiles_df: pd.DataFrame) -> dict:
    ctov_df = _make_ctov_df(candidates)
    mvd_df = _make_mvd_df(ctov_df)
    from src.sabermetrics.badges import derive_badges
    with patch("src.sabermetrics.badges._load_type_profiles", return_value=type_profiles_df):
        return derive_badges(ctov_df, mvd_df)


def _diverse_pool(party: str, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {
            "person_id": f"{party}_{i}",
            "name": f"{party} Candidate {i}",
            "party": party,
            "ctov": rng.normal(0, 1, J),
            "mvd": float(rng.normal(0, 0.03)),
        }
        for i in range(n)
    ]


# ── Provisional marker ────────────────────────────────────────────────────────


class TestProvisionalMarker:
    """derive_badges must return badge_details with provisional field."""

    def test_badge_details_key_exists_in_output(self, type_profiles):
        """derive_badges output includes 'badge_details' key for candidates with badges."""
        candidates = _diverse_pool("D", 25, seed=6) + _diverse_pool("R", 25, seed=7)
        result = _run_derive_badges(candidates, type_profiles)

        pids_with_badges = [pid for pid, e in result.items() if e.get("badges")]
        assert pids_with_badges, "Fixture must produce at least one candidate with badges"

        for pid in pids_with_badges:
            assert "badge_details" in result[pid], (
                f"derive_badges output for {pid} (has badges) must include 'badge_details' key"
            )

    def test_provisional_true_for_single_race(self, type_profiles):
        """badge_details entries have provisional=True when n_races == 1."""
        candidates = _diverse_pool("D", 25, seed=6) + _diverse_pool("R", 25, seed=7)
        result = _run_derive_badges(candidates, type_profiles)

        pids_with_badge_details = [
            pid for pid, e in result.items() if e.get("badge_details")
        ]
        assert pids_with_badge_details, "Fixture must produce at least one badge_detail entry"

        for pid in pids_with_badge_details:
            entry = result[pid]
            assert entry["n_races"] == 1
            for detail in entry["badge_details"]:
                assert detail.get("provisional") is True, (
                    f"{pid}: badge '{detail.get('name')}' should have provisional=True (1 race)"
                )

    def test_provisional_false_for_multi_race(self, type_profiles):
        """badge_details entries have provisional=False when n_races >= 2."""
        rng = np.random.default_rng(8)
        # Two rows for the same person_id = multi-race
        multi_race = [
            {
                "person_id": "multi_D",
                "name": "Multi D",
                "party": "D",
                "year": yr,
                "ctov": rng.normal(2, 0.5, J),  # above party mean → earns badges
                "mvd": 0.05,
            }
            for yr in [2018, 2022]
        ]
        pool = _diverse_pool("D", 24, seed=9) + _diverse_pool("R", 25, seed=10)
        result = _run_derive_badges(multi_race + pool, type_profiles)

        entry = result["multi_D"]
        assert entry["n_races"] == 2
        assert "badge_details" in entry, "'badge_details' key must exist"
        assert len(entry["badge_details"]) > 0, (
            "Multi D with elevated CTOV should earn at least one badge"
        )
        for detail in entry["badge_details"]:
            assert detail.get("provisional") is False, (
                f"Multi-race badge '{detail.get('name')}' should have provisional=False"
            )

    def test_badge_details_name_matches_badges_list(self, type_profiles):
        """badge_details names must match the existing badges list (backward compat)."""
        candidates = _diverse_pool("D", 25, seed=11) + _diverse_pool("R", 25, seed=12)
        result = _run_derive_badges(candidates, type_profiles)

        for pid, entry in result.items():
            badges_list = set(entry.get("badges", []))
            details_names = {d["name"] for d in entry.get("badge_details", [])}
            assert badges_list == details_names, (
                f"{pid}: badges={badges_list} != badge_details names={details_names}"
            )


# ── Small-pool fallback ───────────────────────────────────────────────────────


class TestSmallPoolFallback:
    """Pools < 20 → fallback to global threshold + fallback_reason='small_pool'."""

    def test_fallback_reason_set_for_small_party_pool(self, type_profiles):
        """Badges for a party with < 20 candidates carry fallback_reason='small_pool'."""
        # 5 independents — well below the 20-candidate floor
        rng = np.random.default_rng(100)
        independents = [
            {
                "person_id": f"I_{i}",
                "name": f"Ind {i}",
                "party": "I",
                "ctov": rng.normal(0, 1, J),
                "mvd": float(rng.normal(0.1, 0.05)),
            }
            for i in range(5)
        ]
        pool = _diverse_pool("D", 25, seed=101) + _diverse_pool("R", 25, seed=102)
        result = _run_derive_badges(independents + pool, type_profiles)

        ind_with_details = [
            f"I_{i}" for i in range(5)
            if result.get(f"I_{i}", {}).get("badge_details")
        ]
        if not ind_with_details:
            pytest.skip("No independent earned a badge; adjust fixture if needed")

        for pid in ind_with_details:
            for detail in result[pid]["badge_details"]:
                # Turnout Monster is intentionally global regardless of party pool size
                if detail.get("name") == "Turnout Monster":
                    continue
                assert detail.get("fallback_reason") == "small_pool", (
                    f"{pid}: badge '{detail.get('name')}' should have fallback_reason='small_pool', "
                    f"got {detail.get('fallback_reason')!r}"
                )

    def test_no_fallback_reason_for_normal_pools(self, type_profiles):
        """D and R candidates (>= 20 each) should not have fallback_reason."""
        candidates = _diverse_pool("D", 25, seed=13) + _diverse_pool("R", 25, seed=14)
        result = _run_derive_badges(candidates, type_profiles)

        pids_with_details = [pid for pid, e in result.items() if e.get("badge_details")]
        assert pids_with_details, "Fixture must produce at least one badge_detail entry"

        for pid in pids_with_details:
            for detail in result[pid]["badge_details"]:
                assert detail.get("fallback_reason") is None, (
                    f"{pid}: badge '{detail.get('name')}' should not have fallback_reason, "
                    f"got {detail.get('fallback_reason')!r}"
                )


# ── Within-party thresholding ─────────────────────────────────────────────────


class TestWithinPartyThreshold:
    """Badges should be awarded relative to within-party distribution."""

    def test_within_party_outlier_earns_badge_even_if_below_global_mean(
        self, type_profiles
    ):
        """A D candidate above D-party mean but below global mean still earns the badge.

        Scenario: 25 Rs all have very high scores on a dimension; 25 Ds have
        moderate scores with one D outlier above D-party mean but below global mean.
        With within-party thresholding: D outlier earns the badge.
        With global-only thresholding: D outlier might not (depends on Rs dominating).
        """
        rng = np.random.default_rng(200)
        # Build structured type_profiles: only type 0 matters for pct_hispanic
        tp = pd.DataFrame({col: np.full(J, 0.05) for col in ["type_id"] + _BADGE_COLS})
        tp["type_id"] = list(range(J))
        tp["pct_hispanic"] = np.where(np.arange(J) == 0, 0.9, 0.05)

        # 25 Rs with very high effective CTOV on dim 0 (i.e., raw = -5, effective = +5)
        reps = [
            {
                "person_id": f"R_{i}",
                "name": f"R {i}",
                "party": "R",
                "ctov": np.where(np.arange(J) == 0, -5.0 + rng.normal(0, 0.1), 0.0),
                "mvd": -0.05,
            }
            for i in range(25)
        ]
        # 25 Ds with moderate CTOV (mean ≈ 0)
        dems = [
            {
                "person_id": f"D_{i}",
                "name": f"D {i}",
                "party": "D",
                "ctov": np.where(np.arange(J) == 0, float(rng.normal(0, 0.5)), 0.0),
                "mvd": 0.0,
            }
            for i in range(25)
        ]
        # One D clearly above D-party mean on dim 0
        d_outlier_ctov = np.zeros(J)
        d_outlier_ctov[0] = 5.0  # well above D mean ≈ 0

        d_outlier = {
            "person_id": "d_outlier",
            "name": "D Outlier",
            "party": "D",
            "ctov": d_outlier_ctov,
            "mvd": 0.05,
        }

        ctov_df = _make_ctov_df(dems + reps + [d_outlier])
        mvd_df = _make_mvd_df(ctov_df)
        from src.sabermetrics.badges import derive_badges
        with patch("src.sabermetrics.badges._load_type_profiles", return_value=tp):
            result = derive_badges(ctov_df, mvd_df)

        # D outlier is above D-party mean on Hispanic types → should earn badge
        # With within-party thresholding this is guaranteed; with global it might not be
        assert "Hispanic Appeal" in result["d_outlier"]["badges"], (
            "D candidate above D-party mean should earn 'Hispanic Appeal' "
            "via within-party thresholding"
        )


# ── Signature badges ──────────────────────────────────────────────────────────


class TestSignatureBadges:
    """Auto-discovered signature badges from derive_signature_badges()."""

    def _run_signatures(
        self,
        candidates: list[dict],
        type_profiles_df: pd.DataFrame,
        super_type_names: dict[int, str],
    ) -> dict:
        ctov_df = _make_ctov_df(candidates)
        from src.sabermetrics.badges import derive_signature_badges
        with patch("src.sabermetrics.badges._load_type_profiles", return_value=type_profiles_df):
            return derive_signature_badges(ctov_df, super_type_names=super_type_names)

    @pytest.fixture
    def super_type_names(self) -> dict[int, str]:
        return {j: f"Super Type {j}" for j in range(J)}

    def test_signature_cap_at_three(self, type_profiles, super_type_names):
        """No candidate ever receives more than 3 signature badges."""
        rng = np.random.default_rng(13)
        extreme = {
            "person_id": "extreme_D",
            "name": "Extreme D",
            "party": "D",
            "ctov": rng.normal(6, 0.5, J),  # far above party mean on all dims
            "mvd": 0.2,
        }
        pool = _diverse_pool("D", 25, seed=14) + _diverse_pool("R", 25, seed=15)
        result = self._run_signatures([extreme] + pool, type_profiles, super_type_names)

        n_sig = len(result.get("extreme_D", {}).get("signature_badges", []))
        assert n_sig <= 3, f"Expected <= 3 signature badges; got {n_sig}"

    def test_signature_cosine_dedup(self, type_profiles, super_type_names):
        """Two correlated types (cos > 0.6) should not both get a signature badge."""
        rng = np.random.default_rng(16)
        n_pool = 25

        # Types 0 and 1 are near-identical across the candidate pool → high cosine
        base_signal = rng.normal(0, 1, n_pool)
        pool_ctovs = [rng.normal(0, 0.3, J) for _ in range(n_pool)]
        for i in range(n_pool):
            pool_ctovs[i][0] = base_signal[i]
            pool_ctovs[i][1] = base_signal[i] + rng.normal(0, 0.05)  # type 1 ≈ type 0

        dems = [
            {"person_id": f"D_{i}", "name": f"D {i}", "party": "D",
             "ctov": pool_ctovs[i], "mvd": 0.0}
            for i in range(n_pool)
        ]
        reps = _diverse_pool("R", 25, seed=17)

        # Candidate spikes equally on both correlated types
        spike = np.zeros(J)
        spike[0] = 8.0
        spike[1] = 8.0
        extreme = {
            "person_id": "extreme_D",
            "name": "Extreme D",
            "party": "D",
            "ctov": spike,
            "mvd": 0.1,
        }

        result = self._run_signatures(dems + reps + [extreme], type_profiles, super_type_names)

        sig_badges = result.get("extreme_D", {}).get("signature_badges", [])
        type_0 = any(b.get("type_id") == 0 for b in sig_badges)
        type_1 = any(b.get("type_id") == 1 for b in sig_badges)
        assert not (type_0 and type_1), (
            "Correlated types 0 and 1 (cosine > 0.6) should not both produce signature badges"
        )

    def test_signature_badge_names_from_super_type_names(self, type_profiles, super_type_names):
        """Signature badge names derive from the provided super_type_names dict."""
        rng = np.random.default_rng(18)
        spike = np.zeros(J)
        spike[5] = 10.0  # strong spike on type 5

        pool = _diverse_pool("D", 24, seed=19) + _diverse_pool("R", 25, seed=20)
        extreme = {
            "person_id": "spike_D", "name": "Spike D", "party": "D",
            "ctov": spike, "mvd": 0.0,
        }
        result = self._run_signatures([extreme] + pool, type_profiles, super_type_names)

        sig_badges = result.get("spike_D", {}).get("signature_badges", [])
        assert sig_badges, "Candidate spiking on type 5 should earn a signature badge"
        assert any("Super Type" in b["name"] for b in sig_badges), (
            f"Signature badge name should reference super type name; got {[b['name'] for b in sig_badges]}"
        )

    def test_signature_badges_have_type_id(self, type_profiles, super_type_names):
        """Each signature badge must include type_id for frontend rendering."""
        spike = np.zeros(J)
        spike[7] = 10.0

        pool = _diverse_pool("D", 24, seed=22) + _diverse_pool("R", 25, seed=23)
        extreme = {
            "person_id": "spike_D", "name": "Spike D", "party": "D",
            "ctov": spike, "mvd": 0.0,
        }
        result = self._run_signatures([extreme] + pool, type_profiles, super_type_names)

        for badge in result.get("spike_D", {}).get("signature_badges", []):
            assert "type_id" in badge, "Signature badge must include type_id"
            assert isinstance(badge["type_id"], int)


# ── Party balance regression ──────────────────────────────────────────────────


class TestPartyBalance:
    def test_symmetric_input_produces_balanced_badges(self, type_profiles):
        """Symmetric D and R distributions → mean |D_share - 0.5| <= 0.15."""
        rng = np.random.default_rng(30)
        n = 50
        dems = [
            {"person_id": f"D_{i}", "name": f"D {i}", "party": "D",
             "ctov": rng.normal(0, 1, J), "mvd": float(rng.normal(0, 0.05))}
            for i in range(n)
        ]
        reps = [
            {"person_id": f"R_{i}", "name": f"R {i}", "party": "R",
             "ctov": rng.normal(0, 1, J), "mvd": float(rng.normal(0, 0.05))}
            for i in range(n)
        ]
        result = _run_derive_badges(dems + reps, type_profiles)

        badge_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"D": 0, "R": 0})
        for entry in result.values():
            p = entry["party"]
            for b in entry.get("badges", []):
                if "Signature" not in b:
                    badge_counts[b][p] += 1

        imbalances = [
            abs(counts["D"] / (counts["D"] + counts["R"]) - 0.5)
            for counts in badge_counts.values()
            if counts["D"] + counts["R"] >= 10
        ]
        if not imbalances:
            pytest.skip("No badge with >=10 recipients; skip balance check")

        mean_imbalance = sum(imbalances) / len(imbalances)
        assert mean_imbalance <= 0.15, (
            f"Mean party imbalance {mean_imbalance:.3f} > 0.15 with symmetric input"
        )
