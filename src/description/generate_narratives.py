"""Auto-generate 2-3 sentence human-readable descriptions for each type.

Uses z-score profiles computed from type_profiles.parquet to produce
template-based narratives without LLM calls.  Output style: 538-audience,
accessible language, no raw z-score numbers exposed.

Usage
-----
    from src.description.generate_narratives import generate_all_narratives
    narratives = generate_all_narratives()   # {type_id: narrative_str}
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROFILES_PATH = _ROOT / "data" / "communities" / "type_profiles.parquet"

# ── Feature configuration ─────────────────────────────────────────────────────

# Columns that are raw population counts / totals — not suitable for z-scoring
_SKIP_COLS = {
    "type_id",
    "n_counties",
    "pop_total",
    "pop_white_nh",
    "pop_black",
    "pop_asian",
    "pop_hispanic",
    "housing_total",
    "housing_owner",
    "educ_total",
    "educ_bachelors_plus",
    "commute_total",
    "commute_car",
    "commute_transit",
    "commute_wfh",
    "display_name",
    "log_median_hh_income",   # redundant with median_hh_income
    "land_area_sq_mi",        # size, not character
    "pop_per_sq_mi",          # captured by log_pop_density
}

# Human-readable labels and polarity descriptions for narrative phrases
# Tuple: (positive_high_phrase, positive_low_phrase)
# e.g. if z > 0: "significantly higher education levels"
#      if z < 0: "notably lower education levels"
_FEATURE_PHRASES: dict[str, tuple[str, str]] = {
    "pct_bachelors_plus":      ("high educational attainment", "low educational attainment"),
    "pct_graduate":            ("high graduate degree rates", "low graduate degree rates"),
    "median_hh_income":        ("high household incomes", "low household incomes"),
    "pct_white_nh":            ("predominantly white non-Hispanic populations", "relatively few white non-Hispanic residents"),
    "pct_black":               ("large Black populations", "small Black populations"),
    "pct_hispanic":            ("large Hispanic populations", "small Hispanic populations"),
    "pct_asian":               ("large Asian populations", "small Asian populations"),
    "black_protestant_share":  ("strong Black Protestant tradition", "minimal Black Protestant presence"),
    "evangelical_share":       ("strong evangelical tradition", "low evangelical affiliation"),
    "mainline_share":          ("strong mainline Protestant tradition", "low mainline Protestant affiliation"),
    "catholic_share":          ("significant Catholic presence", "minimal Catholic presence"),
    "religious_adherence_rate":("high overall religious adherence", "low overall religious adherence"),
    "congregations_per_1000":  ("dense congregational networks", "sparse congregational networks"),
    "median_age":              ("older-than-average populations", "younger-than-average populations"),
    "log_pop_density":         ("high population density", "low population density"),
    "pct_owner_occupied":      ("high homeownership rates", "low homeownership rates"),
    "pct_wfh":                 ("high work-from-home rates", "low work-from-home rates"),
    "pct_transit":             ("significant transit commuting", "minimal transit use"),
    "pct_car":                 ("near-universal car commuting", "lower-than-average car commuting"),
    "pct_management":          ("high rates of management and professional occupations", "low rates of management occupations"),
    "net_migration_rate":      ("strong population in-migration", "net population out-migration"),
    "avg_inflow_income":       ("high-income in-migrants", "lower-income in-migrants"),
    "migration_diversity":     ("diverse in-migration from many origins", "in-migration from a narrow set of origins"),
    "inflow_outflow_ratio":    ("more arrivals than departures", "more departures than arrivals"),
}

# z-score magnitude thresholds for adverb selection
def _adverb(z: float) -> str:
    a = abs(z)
    if a >= 2.0:
        return "dramatically"
    if a >= 1.5:
        return "significantly"
    if a >= 1.0:
        return "notably"
    return "somewhat"


def _phrase_for(feature: str, z: float) -> str | None:
    """Return a human-readable phrase for a feature at a given z-score.

    Returns None if the feature is not in the phrase dictionary or |z| < 0.5.
    """
    if abs(z) < 0.5:
        return None
    entry = _FEATURE_PHRASES.get(feature)
    if entry is None:
        return None
    high_phrase, low_phrase = entry
    adv = _adverb(z)
    if z > 0:
        return f"{adv} {high_phrase}"
    else:
        return f"{adv} {low_phrase}"


# ── Urbanicity categorisation ─────────────────────────────────────────────────

def _urbanicity_label(z_density: float) -> str:
    if z_density >= 1.5:
        return "urban"
    if z_density >= 0.5:
        return "suburban"
    if z_density >= -0.5:
        return "mixed urban-rural"
    if z_density >= -1.5:
        return "rural"
    return "deep-rural"


# ── Income tier ───────────────────────────────────────────────────────────────

def _income_label(z_income: float) -> str:
    if z_income >= 1.5:
        return "high-income"
    if z_income >= 0.5:
        return "upper-middle-income"
    if z_income >= -0.5:
        return "middle-income"
    if z_income >= -1.5:
        return "lower-middle-income"
    return "lower-income"


# ── Core generator ────────────────────────────────────────────────────────────

def generate_type_narrative(profile: dict[str, float], display_name: str) -> str:
    """Generate a 2-3 sentence description from demographic z-scores.

    Parameters
    ----------
    profile:
        Dict mapping feature name to z-score (standardised across all types).
    display_name:
        The type's display name, used as the subject of the first sentence.

    Returns
    -------
    A plain-text narrative string (2-3 sentences).  No raw z-score numbers
    are exposed.
    """
    # ── Sentence 1: character overview ───────────────────────────────────────
    # Pick urbanicity and income framing
    density_z = profile.get("log_pop_density", 0.0)
    income_z = profile.get("median_hh_income", 0.0)
    urban_label = _urbanicity_label(density_z)
    income_label = _income_label(income_z)

    # Lead sentence construction
    n_counties = profile.get("n_counties")
    if n_counties is not None and not np.isnan(n_counties):
        n_str = f" across {int(n_counties)} counties"
    else:
        n_str = ""

    sentence1 = (
        f"{display_name} communities are {urban_label}, {income_label} areas"
        f"{n_str}."
    )

    # ── Sentence 2: most distinctive features ────────────────────────────────
    # Rank features by |z|, excluding urbanicity/income (already in sentence 1)
    priority_exclude = {"log_pop_density", "median_hh_income", "n_counties"}
    ranked = sorted(
        [(k, v) for k, v in profile.items() if k not in priority_exclude and k in _FEATURE_PHRASES],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )

    # Collect up to 3 distinct high-signal phrases (|z| > 0.75 for sentence 2)
    phrases = []
    for feat, z in ranked:
        if abs(z) < 0.75:
            break
        p = _phrase_for(feat, z)
        if p and p not in phrases:
            phrases.append(p)
        if len(phrases) >= 3:
            break

    if phrases:
        if len(phrases) == 1:
            feature_clause = phrases[0]
        elif len(phrases) == 2:
            feature_clause = f"{phrases[0]} and {phrases[1]}"
        else:
            feature_clause = f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"
        sentence2 = f"These areas are characterised by {feature_clause}."
    else:
        # Fallback for types near the demographic average
        sentence2 = "Demographically, these areas are close to the regional average across most dimensions."

    # ── Sentence 3: migration / growth note (if distinctive) ─────────────────
    migration_z = profile.get("net_migration_rate", 0.0)
    inflow_income_z = profile.get("avg_inflow_income", 0.0)
    diversity_z = profile.get("migration_diversity", 0.0)

    sentence3 = ""
    if abs(migration_z) >= 1.0:
        direction = "strong population growth driven by in-migration" if migration_z > 0 else "persistent population loss"
        if migration_z > 0 and inflow_income_z >= 0.75:
            sentence3 = f"These communities are experiencing {direction}, with arriving residents tending to have above-average incomes."
        elif migration_z > 0 and diversity_z >= 0.75:
            sentence3 = f"These communities are experiencing {direction} from a wide variety of origin locations."
        else:
            sentence3 = f"These communities are experiencing {direction}."

    if sentence3:
        return f"{sentence1} {sentence2} {sentence3}"
    return f"{sentence1} {sentence2}"


# ── Batch generator ───────────────────────────────────────────────────────────

def generate_all_narratives(profiles_path: str | None = None) -> dict[int, str]:
    """Generate narratives for all types.

    Parameters
    ----------
    profiles_path:
        Path to type_profiles.parquet.  Defaults to the canonical pipeline path.

    Returns
    -------
    Dict mapping type_id (int) to narrative string.
    """
    path = Path(profiles_path) if profiles_path else _DEFAULT_PROFILES_PATH
    df = pd.read_parquet(path)

    # Identify numeric feature columns to z-score
    feat_cols = [
        c for c in df.columns
        if c not in _SKIP_COLS and df[c].dtype != object
    ]

    # Compute z-scores across all types (column-wise)
    feat_df = df[feat_cols].copy()
    means = feat_df.mean()
    stds = feat_df.std().replace(0, 1)   # avoid division by zero for constant cols
    z_df = (feat_df - means) / stds

    # Also pass n_counties raw (not z-scored) so sentence 1 can report it
    narratives: dict[int, str] = {}
    for i, row in df.iterrows():
        type_id = int(row["type_id"])
        display_name = str(row["display_name"])
        z_row = z_df.iloc[i].to_dict()
        # Inject raw n_counties so narrative can use it
        z_row["n_counties"] = float(row["n_counties"])
        narrative = generate_type_narrative(z_row, display_name)
        narratives[type_id] = narrative
        log.debug("Type %d: %s", type_id, narrative[:80])

    log.info("Generated narratives for %d types", len(narratives))
    return narratives


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    narratives = generate_all_narratives()
    for tid, text in sorted(narratives.items()):
        print(f"\n[Type {tid}]")
        print(text)


if __name__ == "__main__":
    main()
