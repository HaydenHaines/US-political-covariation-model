"""Tests for the type narrative generation module.

Covers:
- All 43 types get a narrative
- Narratives are plain strings of reasonable length (1-4 sentences)
- No raw z-score numbers exposed
- Distinctive features are mentioned for high-signal types
- Near-zero profiles produce a generic fallback
"""
from __future__ import annotations

import re
import string

import pandas as pd
import pytest

from src.description.generate_narratives import (
    generate_all_narratives,
    generate_type_narrative,
    _FEATURE_PHRASES,
    _adverb,
    _income_label,
    _urbanicity_label,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def count_sentences(text: str) -> int:
    """Rough sentence counter: splits on . / ! / ? followed by whitespace or end."""
    return len(re.findall(r"[.!?](?:\s|$)", text))


def has_raw_number(text: str) -> bool:
    """Return True if the text contains an isolated decimal number (e.g. z-scores)."""
    # Allow integers (county counts like "8 counties") but flag decimals like "1.95"
    return bool(re.search(r"\b\d+\.\d+\b", text))


# ── Unit tests for helpers ────────────────────────────────────────────────────

class TestAdverb:
    def test_very_high_z(self):
        assert _adverb(2.5) == "dramatically"

    def test_high_z(self):
        assert _adverb(1.6) == "significantly"

    def test_moderate_z(self):
        assert _adverb(1.1) == "notably"

    def test_low_z(self):
        assert _adverb(0.7) == "somewhat"

    def test_negative_uses_magnitude(self):
        assert _adverb(-2.5) == "dramatically"
        assert _adverb(-1.1) == "notably"


class TestUrbanicityLabel:
    def test_urban(self):
        assert _urbanicity_label(2.0) == "urban"

    def test_suburban(self):
        assert _urbanicity_label(0.8) == "suburban"

    def test_mixed(self):
        assert _urbanicity_label(0.0) == "mixed urban-rural"

    def test_rural(self):
        assert _urbanicity_label(-1.0) == "rural"

    def test_deep_rural(self):
        assert _urbanicity_label(-2.0) == "deep-rural"


class TestIncomeLabel:
    def test_high(self):
        assert _income_label(2.0) == "high-income"

    def test_upper_middle(self):
        assert _income_label(0.8) == "upper-middle-income"

    def test_middle(self):
        assert _income_label(0.0) == "middle-income"

    def test_lower_middle(self):
        assert _income_label(-1.0) == "lower-middle-income"

    def test_lower(self):
        assert _income_label(-2.0) == "lower-income"


# ── Unit tests for generate_type_narrative ────────────────────────────────────

class TestGenerateTypeNarrative:
    def _profile_all_zero(self) -> dict[str, float]:
        return {k: 0.0 for k in _FEATURE_PHRASES}

    def _profile_high_education_income(self) -> dict[str, float]:
        p = self._profile_all_zero()
        p["pct_bachelors_plus"] = 2.2
        p["pct_graduate"] = 1.8
        p["median_hh_income"] = 1.9
        p["log_pop_density"] = 1.0
        return p

    def _profile_black_belt_rural(self) -> dict[str, float]:
        p = self._profile_all_zero()
        p["pct_black"] = 2.0
        p["black_protestant_share"] = 1.7
        p["median_hh_income"] = -1.5
        p["log_pop_density"] = -2.0
        return p

    def _profile_strong_migration(self) -> dict[str, float]:
        p = self._profile_all_zero()
        p["net_migration_rate"] = 1.8
        p["avg_inflow_income"] = 1.2
        p["log_pop_density"] = 0.3
        p["median_hh_income"] = 0.5
        return p

    def test_returns_string(self):
        result = generate_type_narrative(self._profile_all_zero(), "Test Community")
        assert isinstance(result, str)

    def test_not_empty(self):
        result = generate_type_narrative(self._profile_all_zero(), "Test Community")
        assert len(result) > 10

    def test_display_name_in_output(self):
        result = generate_type_narrative(self._profile_all_zero(), "Suburban Professionals")
        assert "Suburban Professionals" in result

    def test_no_raw_z_scores_exposed(self):
        profile = self._profile_high_education_income()
        result = generate_type_narrative(profile, "College Town")
        assert not has_raw_number(result), f"Raw decimal found in: {result!r}"

    def test_sentence_count_all_zero(self):
        result = generate_type_narrative(self._profile_all_zero(), "Average Place")
        n = count_sentences(result)
        assert 1 <= n <= 4, f"Expected 1-4 sentences, got {n}: {result!r}"

    def test_sentence_count_high_signal(self):
        result = generate_type_narrative(self._profile_high_education_income(), "College Metro")
        n = count_sentences(result)
        assert 1 <= n <= 4

    def test_sentence_count_migration_type(self):
        result = generate_type_narrative(self._profile_strong_migration(), "Sunbelt Growth")
        n = count_sentences(result)
        assert 1 <= n <= 4

    def test_high_education_mentioned(self):
        result = generate_type_narrative(self._profile_high_education_income(), "University Town")
        assert "education" in result.lower() or "degree" in result.lower() or "graduate" in result.lower()

    def test_black_population_mentioned(self):
        result = generate_type_narrative(self._profile_black_belt_rural(), "Black Belt")
        assert "black" in result.lower()

    def test_rural_label_for_low_density(self):
        p = self._profile_all_zero()
        p["log_pop_density"] = -2.0
        result = generate_type_narrative(p, "Deep Rural")
        assert "rural" in result.lower()

    def test_urban_label_for_high_density(self):
        p = self._profile_all_zero()
        p["log_pop_density"] = 2.0
        result = generate_type_narrative(p, "Big City")
        assert "urban" in result.lower()

    def test_migration_growth_sentence(self):
        result = generate_type_narrative(self._profile_strong_migration(), "Sunbelt Growth")
        assert "migration" in result.lower() or "growth" in result.lower() or "in-migration" in result.lower()

    def test_out_migration_mentioned(self):
        p = self._profile_all_zero()
        p["net_migration_rate"] = -1.5
        p["log_pop_density"] = -1.0
        p["median_hh_income"] = -0.8
        result = generate_type_narrative(p, "Rust Belt")
        assert "loss" in result.lower() or "departure" in result.lower() or "out-migration" in result.lower()

    def test_generic_fallback_for_flat_profile(self):
        result = generate_type_narrative(self._profile_all_zero(), "Average County")
        # Should mention being near average, not assert a distinctive feature
        assert "average" in result.lower() or "characterised" in result.lower()

    def test_ends_with_period(self):
        result = generate_type_narrative(self._profile_all_zero(), "Test")
        assert result.strip().endswith(".")

    def test_county_count_included_when_provided(self):
        p = self._profile_all_zero()
        p["n_counties"] = 12.0
        result = generate_type_narrative(p, "Sample Type")
        assert "12" in result


# ── Integration tests against real data ──────────────────────────────────────

class TestGenerateAllNarratives:
    @pytest.fixture(scope="class")
    def narratives(self):
        return generate_all_narratives()

    def test_returns_dict(self, narratives):
        assert isinstance(narratives, dict)

    def test_all_43_types_covered(self, narratives):
        assert len(narratives) == 43, f"Expected 43 types, got {len(narratives)}"

    def test_all_type_ids_are_ints(self, narratives):
        for tid in narratives:
            assert isinstance(tid, int), f"type_id {tid!r} is not an int"

    def test_all_narratives_are_strings(self, narratives):
        for tid, text in narratives.items():
            assert isinstance(text, str), f"Type {tid} narrative is not a string"

    def test_no_empty_narratives(self, narratives):
        for tid, text in narratives.items():
            assert len(text.strip()) > 20, f"Type {tid} narrative is too short: {text!r}"

    def test_sentence_count_range(self, narratives):
        for tid, text in narratives.items():
            n = count_sentences(text)
            assert 1 <= n <= 4, f"Type {tid} has {n} sentences: {text!r}"

    def test_no_raw_z_scores_in_any_narrative(self, narratives):
        for tid, text in narratives.items():
            assert not has_raw_number(text), (
                f"Type {tid} exposes a raw decimal number: {text!r}"
            )

    def test_display_name_in_first_sentence(self, narratives):
        """Each narrative should start with the type's display name."""
        profiles = pd.read_parquet(
            "data/communities/type_profiles.parquet"
        )
        id_to_name = dict(zip(profiles["type_id"], profiles["display_name"]))
        for tid, text in narratives.items():
            name = id_to_name[tid]
            assert text.startswith(name), (
                f"Type {tid} narrative doesn't start with display_name.\n"
                f"  Name: {name!r}\n  Text: {text[:80]!r}"
            )

    def test_type_ids_match_profiles(self, narratives):
        profiles = pd.read_parquet(
            "data/communities/type_profiles.parquet"
        )
        expected_ids = set(profiles["type_id"].astype(int))
        assert set(narratives.keys()) == expected_ids

    def test_high_signal_types_mention_distinctive_features(self, narratives):
        """Types with very distinctive demographics should reference those demographics."""
        profiles = pd.read_parquet(
            "data/communities/type_profiles.parquet"
        )
        # Find a type with large Black population share
        feat_df = profiles[["type_id", "pct_black"]].copy()
        top_black = feat_df.nlargest(3, "pct_black")["type_id"].tolist()
        for tid in top_black:
            text = narratives[tid]
            assert "black" in text.lower(), (
                f"Type {tid} has high pct_black but 'black' not in narrative: {text!r}"
            )
