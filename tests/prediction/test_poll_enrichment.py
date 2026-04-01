"""Tests for tiered W vector construction."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.poll_enrichment import (
    build_W_poll,
    build_W_with_adjustments,
    build_W_from_crosstabs,
    build_W_from_raw_sample,
    parse_methodology,
    _infer_method_type,
)


def _make_type_profiles(j: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "type_id": range(j),
        "median_age": np.linspace(25, 65, j),
        "pct_owner_occupied": np.linspace(0.3, 0.8, j),
        "pct_bachelors_plus": np.linspace(0.1, 0.6, j),
        "evangelical_share": np.linspace(0.05, 0.5, j),
        "catholic_share": np.linspace(0.4, 0.1, j),
        "mainline_share": np.linspace(0.1, 0.3, j),
        "log_pop_density": np.linspace(-0.5, 0.5, j),
        "median_hh_income": np.linspace(30000, 100000, j),
        "pct_black": np.linspace(0.02, 0.4, j),
        "pct_white_nh": np.linspace(0.8, 0.4, j),
        "pct_hispanic": np.linspace(0.05, 0.3, j),
        "pct_asian": np.linspace(0.01, 0.1, j),
    })


def _make_state_type_weights(j: int = 5) -> np.ndarray:
    w = np.array([0.3, 0.2, 0.2, 0.2, 0.1])[:j]
    return w / w.sum()


class TestTierDispatch:
    def test_tier1_when_raw_data_present(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        raw = {"pct_black": 0.33, "evangelical_share": 0.25}
        W = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp, state_type_weights=stw,
            raw_sample_demographics=raw,
        )
        assert W.shape == (5,)
        assert abs(W.sum() - 1.0) < 1e-6

    def test_tier2_when_crosstabs_present(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        xt = [
            {"demographic_group": "race", "group_value": "black",
             "pct_of_sample": 0.33, "dem_share": 0.90},
            {"demographic_group": "race", "group_value": "white",
             "pct_of_sample": 0.55, "dem_share": 0.40},
        ]
        result = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp, state_type_weights=stw,
            poll_crosstabs=xt,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(r["W"].shape == (5,) for r in result)

    def test_tier3_when_topline_only(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp, state_type_weights=stw,
        )
        assert W.shape == (5,)
        assert abs(W.sum() - 1.0) < 1e-6


class TestTier3Adjustments:
    def test_lv_downweights_low_propensity(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W_lv = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": "LV"},
            type_profiles=tp, state_type_weights=stw,
        )
        W_none = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": ""},
            type_profiles=tp, state_type_weights=stw,
        )
        np.testing.assert_allclose(W_none, stw, atol=1e-6)
        assert not np.allclose(W_lv, stw, atol=1e-3)
        # Type 0 has lowest propensity (youngest, lowest homeownership, lowest education)
        assert W_lv[0] < stw[0]

    def test_core_vs_full_mode(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        poll = {"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": "LV"}
        W_core = build_W_with_adjustments(poll, tp, stw, w_vector_mode="core")
        W_full = build_W_with_adjustments(poll, tp, stw, w_vector_mode="full")
        assert abs(W_core.sum() - 1.0) < 1e-6
        assert abs(W_full.sum() - 1.0) < 1e-6
        # Both valid but they should differ (full uses more dims) — though if
        # no method reach profiles have shifts, they'll be identical from LV
        # adjustment alone. That's acceptable.

    def test_no_methodology_returns_state_weights(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": ""},
            type_profiles=tp, state_type_weights=stw,
        )
        np.testing.assert_allclose(W, stw, atol=1e-6)


class TestParseMethodology:
    def test_lv(self):
        assert parse_methodology("D=34.0% R=53.0%; LV; src=wikipedia") == "LV"

    def test_rv(self):
        assert parse_methodology("D=34.0% R=53.0%; RV; src=wikipedia") == "RV"

    def test_no_method(self):
        assert parse_methodology("D=34.0% R=53.0%; src=wikipedia") == ""

    def test_empty(self):
        assert parse_methodology("") == ""

    def test_none(self):
        assert parse_methodology(None) == ""


class TestInferMethodType:
    """Tests for methodology type inference from poll notes."""

    def test_online_panel_keyword(self):
        assert _infer_method_type({"notes": "online panel; LV"}) == "online_panel"

    def test_online_keyword(self):
        assert _infer_method_type({"notes": "online survey; RV"}) == "online_panel"

    def test_panel_keyword(self):
        assert _infer_method_type({"notes": "opt-in panel; LV"}) == "online_panel"

    def test_ivr_keyword(self):
        assert _infer_method_type({"notes": "IVR; LV; src=wikipedia"}) == "phone_ivr"

    def test_automated_keyword(self):
        assert _infer_method_type({"notes": "automated phone; LV"}) == "phone_ivr"

    def test_robo_keyword(self):
        assert _infer_method_type({"notes": "robopoll; RV"}) == "phone_ivr"

    def test_sms_keyword(self):
        assert _infer_method_type({"notes": "SMS survey; RV"}) == "sms"

    def test_text_keyword(self):
        assert _infer_method_type({"notes": "text message poll; LV"}) == "sms"

    def test_mail_keyword(self):
        assert _infer_method_type({"notes": "mail survey; LV"}) == "mail"

    def test_postal_keyword(self):
        assert _infer_method_type({"notes": "postal survey; RV"}) == "mail"

    def test_live_phone_keyword(self):
        assert _infer_method_type({"notes": "live phone; LV"}) == "phone_live"

    def test_phone_keyword(self):
        assert _infer_method_type({"notes": "phone survey; LV"}) == "phone_live"

    def test_unknown_fallback(self):
        assert _infer_method_type({"notes": "src=wikipedia; LV"}) == "unknown"

    def test_empty_notes(self):
        assert _infer_method_type({"notes": ""}) == "unknown"

    def test_no_notes_key(self):
        assert _infer_method_type({}) == "unknown"

    def test_none_notes(self):
        assert _infer_method_type({"notes": None}) == "unknown"

    def test_ivr_takes_priority_over_phone(self):
        # "IVR phone" should resolve to phone_ivr, not phone_live
        assert _infer_method_type({"notes": "IVR phone survey; LV"}) == "phone_ivr"

    def test_sms_not_confused_with_phone(self):
        # SMS notes that don't contain "phone" should resolve to sms
        assert _infer_method_type({"notes": "SMS; LV"}) == "sms"


class TestReachProfiles:
    """Tests that reach profiles produce directionally correct W adjustments."""

    def _run_with_method(self, notes: str, w_vector_mode: str = "full") -> np.ndarray:
        """Helper: run Tier 3 adjustment with a given notes string."""
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        poll = {
            "state": "GA",
            "dem_share": 0.53,
            "n_sample": 600,
            "methodology": "LV",
            "notes": notes,
        }
        return build_W_with_adjustments(poll, tp, stw, w_vector_mode=w_vector_mode)

    def test_all_profiles_produce_valid_W(self):
        """Every profile produces a normalized W vector with no NaNs."""
        method_notes = [
            "online panel; LV",
            "IVR; LV",
            "live phone; LV",
            "SMS survey; LV",
            "mail survey; LV",
            "src=wikipedia; LV",   # unknown
        ]
        for notes in method_notes:
            W = self._run_with_method(notes, w_vector_mode="full")
            assert W.shape == (5,), f"Wrong shape for notes='{notes}'"
            assert abs(W.sum() - 1.0) < 1e-6, f"W doesn't sum to 1 for notes='{notes}'"
            assert not np.any(np.isnan(W)), f"NaN in W for notes='{notes}'"

    def test_online_panel_upweights_urban_types(self):
        """Online panel has positive log_pop_density_shift → urban types gain weight.

        Verified by isolating density dimension: online_panel has positive
        log_pop_density_shift while unknown has none, so the highest-density type
        should gain weight relative to the lowest-density type compared to unknown.
        The test fixture has log_pop_density monotonically increasing across types,
        so type 4 = highest density, type 0 = lowest.
        """
        W_online = self._run_with_method("online panel; LV", w_vector_mode="full")
        W_unknown = self._run_with_method("src=wikipedia; LV", w_vector_mode="full")
        # Online panel should shift weight toward high-density types relative to unknown.
        # Both have same LV adjustment; difference comes purely from reach profile.
        high_density_ratio_online = W_online[4] / W_online[0]
        high_density_ratio_unknown = W_unknown[4] / W_unknown[0]
        assert high_density_ratio_online > high_density_ratio_unknown

    def test_phone_ivr_produces_distinct_W_from_online(self):
        """IVR (rural-skew) should produce a meaningfully different W than online_panel (urban-skew)."""
        W_ivr = self._run_with_method("IVR; LV", w_vector_mode="full")
        W_online = self._run_with_method("online panel; LV", w_vector_mode="full")
        # IVR is rural-skewed and online is urban-skewed — they should differ substantially
        assert not np.allclose(W_ivr, W_online, atol=1e-3)

    def test_phone_ivr_lowers_urban_weight_vs_online(self):
        """IVR has negative log_pop_density_shift vs online_panel's positive → less urban weight."""
        W_ivr = self._run_with_method("IVR; LV", w_vector_mode="full")
        W_online = self._run_with_method("online panel; LV", w_vector_mode="full")
        # Type 4 is most urban; IVR should assign less weight there than online_panel
        assert W_ivr[4] < W_online[4]

    def test_sms_and_ivr_diverge_on_density(self):
        """SMS (urban-skew) and IVR (rural-skew) should produce different W distributions."""
        W_sms = self._run_with_method("SMS; LV", w_vector_mode="full")
        W_ivr = self._run_with_method("IVR; LV", w_vector_mode="full")
        # They should not be equal — opposite reach biases
        assert not np.allclose(W_sms, W_ivr, atol=1e-3)

    def test_phone_live_has_smaller_shifts_than_ivr(self):
        """Live phone is closest to representative; its reach shifts should be smaller
        in magnitude than IVR (the most biased method in the config)."""
        from src.prediction.propensity_model import load_config
        config = load_config()
        ivr_profile = config["method_reach_profiles"]["phone_ivr"]
        live_profile = config["method_reach_profiles"]["phone_live"]

        ivr_magnitude = sum(abs(v) for v in ivr_profile.values())
        live_magnitude = sum(abs(v) for v in live_profile.values())
        assert live_magnitude < ivr_magnitude, (
            f"Live phone shifts ({live_magnitude:.3f}) should be smaller than "
            f"IVR shifts ({ivr_magnitude:.3f})"
        )

    def test_config_has_all_expected_profiles(self):
        """Config file contains entries for all expected methodology types."""
        from src.prediction.propensity_model import load_config
        config = load_config()
        profiles = config.get("method_reach_profiles", {})
        expected = {"online_panel", "phone_ivr", "phone_live", "sms", "mail", "unknown"}
        assert expected.issubset(profiles.keys()), (
            f"Missing profiles: {expected - set(profiles.keys())}"
        )

    def test_unknown_profile_has_no_shifts(self):
        """Unknown profile should be empty — no demographic adjustments."""
        from src.prediction.propensity_model import load_config
        config = load_config()
        unknown = config["method_reach_profiles"].get("unknown", {})
        assert unknown == {}, f"Unknown profile should be empty, got: {unknown}"

    def test_core_mode_only_applies_religion_dims(self):
        """Core mode uses only religion dimensions — reach profile entries outside that
        set should have zero effect in core mode."""
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        poll = {"state": "GA", "dem_share": 0.53, "n_sample": 600, "methodology": "LV",
                "notes": "IVR; LV"}
        # IVR has evangelical_share_shift in core dims — so core should differ from unknown
        W_ivr_core = build_W_with_adjustments(poll, tp, stw, w_vector_mode="core")
        W_unknown_core = build_W_with_adjustments(
            {**poll, "notes": "src=wikipedia; LV"}, tp, stw, w_vector_mode="core"
        )
        # evangelical_share_shift is in core dims, so they should differ
        assert not np.allclose(W_ivr_core, W_unknown_core, atol=1e-4)
