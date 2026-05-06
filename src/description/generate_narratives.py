"""Auto-generate 2-3 sentence human-readable descriptions for each type.

Uses z-score profiles computed from type_profiles.parquet and mean prediction
data to produce template-based narratives without LLM calls.  Output style:
538-audience, accessible language, no raw z-score numbers exposed.

Narrative structure (3 sentences):
  1. Geography/setting overview (urbanicity, income, county count)
  2. Most distinctive demographic features (top 3 by |z-score|)
  3. Political lean + trend signal (from mean predicted Dem share + shifts)

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
    "pct_white_nh":            (
        "predominantly white non-Hispanic populations", "relatively few white non-Hispanic residents"
    ),
    "pct_black":               ("large Black populations", "small Black populations"),
    "pct_hispanic":            ("large Hispanic populations", "small Hispanic populations"),
    "pct_asian":               ("large Asian populations", "small Asian populations"),
    "black_protestant_share":  ("strong Black Protestant tradition", "minimal Black Protestant presence"),
    "evangelical_share":       ("strong evangelical tradition", "low evangelical affiliation"),
    "mainline_share":          ("strong mainline Protestant tradition", "low mainline Protestant affiliation"),
    "catholic_share":          ("large Catholic populations", "minimal Catholic presence"),
    "religious_adherence_rate":("high overall religious adherence", "low overall religious adherence"),
    "congregations_per_1000":  ("dense congregational networks", "sparse congregational networks"),
    "median_age":              ("older-than-average populations", "younger-than-average populations"),
    "log_pop_density":         ("high population density", "low population density"),
    "pct_owner_occupied":      ("high homeownership rates", "low homeownership rates"),
    "pct_wfh":                 ("high work-from-home rates", "low work-from-home rates"),
    "pct_transit":             ("high transit ridership", "minimal transit use"),
    # pct_car excluded — not insightful in narratives (near-universal everywhere)
    "pct_management":          (
        "high rates of management and professional occupations", "low rates of management occupations"
    ),
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


# ── Urbanicity categorization ─────────────────────────────────────────────────

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

def _lean_label(dem_share: float) -> str:
    """Convert a predicted two-party Democratic share to a lean label.

    Examples: 0.58 → "D+8", 0.46 → "R+4", 0.502 → "D+0.4"
    """
    margin_pp = (dem_share - 0.5) * 100
    party = "D" if margin_pp >= 0 else "R"
    magnitude = abs(margin_pp)
    # Round to one decimal for close races, integer for blowouts
    if magnitude < 5:
        return f"{party}+{magnitude:.1f}"
    return f"{party}+{round(magnitude)}"


def _trend_sentence(
    shift_12_16: float | None,
    shift_16_20: float | None,
    shift_20_24: float | None,
) -> str | None:
    """Return a trend observation if recent shifts show a consistent direction.

    A "consistent" trend requires at least two of the three recent shifts to
    point in the same direction.  Only called when we have at least two shifts.

    The shift values are Democratic-share changes (positive = Dem gain).
    Threshold of ±0.03 (3pp) filters out noise from near-zero cycles.

    Returns None when there is no clear trend to narrate.
    """
    shifts: list[float] = [
        s for s in (shift_12_16, shift_16_20, shift_20_24) if s is not None
    ]
    if len(shifts) < 2:
        return None

    # Most recent and overall direction
    recent_shift = shifts[-1]
    dem_gains = sum(1 for s in shifts if s > 0.03)
    dem_losses = sum(1 for s in shifts if s < -0.03)

    # Consistent swing toward Republicans (each recent cycle)
    if dem_losses >= 2 and recent_shift < -0.03:
        return "These communities have shifted toward Republicans in recent elections."
    # Consistent swing toward Democrats
    if dem_gains >= 2 and recent_shift > 0.03:
        return "These communities have trended toward Democrats in recent elections."
    # Recent sharp reversal (most recent cycle large and opposite earlier trend)
    if recent_shift < -0.08:
        return "These communities moved sharply toward Republicans in 2020–2024."
    if recent_shift > 0.08:
        return "These communities moved sharply toward Democrats in 2020–2024."

    return None


def generate_type_narrative(
    profile: dict[str, float],
    display_name: str,
    mean_pred_dem_share: float | None = None,
    shift_12_16: float | None = None,
    shift_16_20: float | None = None,
    shift_20_24: float | None = None,
) -> str:
    """Generate a 2-3 sentence description from demographic z-scores and predictions.

    Parameters
    ----------
    profile:
        Dict mapping feature name to z-score (standardised across all types).
    display_name:
        The type's display name, used as the subject of the first sentence.
    mean_pred_dem_share:
        Mean predicted Democratic two-party share across member counties (0–1).
        When provided, the third sentence describes political lean and trend.
    shift_12_16:
        Mean Democratic share shift from 2012→2016 across member counties.
    shift_16_20:
        Mean Democratic share shift from 2016→2020 across member counties.
    shift_20_24:
        Mean Democratic share shift from 2020→2024 across member counties.

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
        sentence2 = f"These areas are characterized by {feature_clause}."
    else:
        # Fallback for types near the demographic average
        sentence2 = "Demographically, these areas are close to the regional average across most dimensions."

    # ── Sentence 3: political lean + trend ───────────────────────────────────
    # When prediction data is available, describe the political character.
    # This replaces the migration-only third sentence — political lean is
    # more useful context for a forecast site's audience.
    sentence3 = ""
    if mean_pred_dem_share is not None:
        lean = _lean_label(mean_pred_dem_share)
        trend = _trend_sentence(shift_12_16, shift_16_20, shift_20_24)
        if trend:
            sentence3 = f"The model forecasts an average lean of {lean}. {trend}"
        else:
            sentence3 = f"The model forecasts an average lean of {lean}."
    else:
        # Fallback: migration / growth note when no prediction data
        migration_z = profile.get("net_migration_rate", 0.0)
        inflow_income_z = profile.get("avg_inflow_income", 0.0)
        diversity_z = profile.get("migration_diversity", 0.0)

        if abs(migration_z) >= 1.0:
            direction = (
                "strong population growth driven by in-migration"
                if migration_z > 0
                else "persistent population loss"
            )
            if migration_z > 0 and inflow_income_z >= 0.75:
                sentence3 = (
                    f"These communities are experiencing {direction}, "
                    "with arriving residents tending to have above-average incomes."
                )
            elif migration_z > 0 and diversity_z >= 0.75:
                sentence3 = f"These communities are experiencing {direction} from a wide variety of origin locations."
            else:
                sentence3 = f"These communities are experiencing {direction}."

    if sentence3:
        return f"{sentence1} {sentence2} {sentence3}"
    return f"{sentence1} {sentence2}"


# ── Batch generator ───────────────────────────────────────────────────────────

def generate_all_narratives(
    profiles_path: str | None = None,
    predictions: dict[int, float] | None = None,
    shifts: dict[int, dict[str, float]] | None = None,
) -> dict[int, str]:
    """Generate narratives for all types.

    Parameters
    ----------
    profiles_path:
        Path to type_profiles.parquet.  Defaults to the canonical pipeline path.
    predictions:
        Dict mapping type_id to mean predicted Democratic share (0–1).
        When provided, each narrative includes political lean in sentence 3.
    shifts:
        Dict mapping type_id to a dict with keys:
        ``shift_12_16``, ``shift_16_20``, ``shift_20_24``
        (mean Democratic share shifts across member counties).
        Only used when ``predictions`` is also provided.

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

    has_display_name = "display_name" in df.columns
    has_n_counties = "n_counties" in df.columns

    # Also pass n_counties raw (not z-scored) so sentence 1 can report it
    narratives: dict[int, str] = {}
    for i, row in df.iterrows():
        type_id = int(row["type_id"])
        display_name = str(row["display_name"]) if has_display_name else f"Type {type_id}"
        z_row = z_df.iloc[i].to_dict()
        if has_n_counties:
            z_row["n_counties"] = float(row["n_counties"])

        # Extract prediction and shift data if available
        pred = predictions.get(type_id) if predictions else None
        type_shifts = shifts.get(type_id, {}) if shifts else {}
        narrative = generate_type_narrative(
            z_row,
            display_name,
            mean_pred_dem_share=pred,
            shift_12_16=type_shifts.get("shift_12_16"),
            shift_16_20=type_shifts.get("shift_16_20"),
            shift_20_24=type_shifts.get("shift_20_24"),
        )
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
