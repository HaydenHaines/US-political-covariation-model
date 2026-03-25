"""Generate descriptive names for the 55 fine electoral types (national model).

Names are derived deterministically from demographic z-scores relative to the
national population-weighted mean across all 55 types.  No LLM calls.

Algorithm overview
------------------
1. Compute national population-weighted z-scores for all demographic features.
   (All 55 types, national reference — NOT within-state.)
2. For each type, scan an ordered vocabulary and collect 2-3 descriptive tokens.
3. Assemble a 2-word base name: "Token1 Token2".
4. Disambiguate duplicates by promoting a 3rd or 4th token from extended vocab.
5. Any remaining duplicates try a political-lean suffix before ordinal fallback.

Super-type naming
-----------------
Super-type names are also computed data-driven from the aggregate demographic
profile of the types they contain.  See ``name_super_types()``.

Usage (CLI)::

    python -m src.description.name_types
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
TYPE_PROFILES_PATH = _ROOT / "data" / "communities" / "type_profiles.parquet"
COUNTY_TYPE_ASSIGNMENTS_PATH = (
    _ROOT / "data" / "communities" / "county_type_assignments_full.parquet"
)
SUPER_TYPES_PATH = _ROOT / "data" / "communities" / "super_types.parquet"

# Complete FIPS → state abbreviation mapping (all 50 states + DC).
# Source: US Census Bureau FIPS state codes.
_FIPS_TO_STATE: dict[int, str] = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

# Path to shift data for political lean disambiguation
_SHIFTS_PATH = _ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
_TYPE_ASSIGNMENTS_PATH = _ROOT / "data" / "communities" / "type_assignments.parquet"

# ---------------------------------------------------------------------------
# Z-score thresholds
# ---------------------------------------------------------------------------
Z_HIGH = 1.2
Z_MOD = 0.6
Z_LOW = 0.3

# Minimum absolute values for race/ethnicity labels (avoid noise amplification)
_MIN_ABS: dict[str, float] = {
    "pct_black": 0.15,
    "pct_hispanic": 0.10,
    "pct_asian": 0.05,
}

# ---------------------------------------------------------------------------
# Primary vocabulary — ordered by specificity (rare demographics first).
# Each entry: (feature, threshold, positive_label, negative_label)
# Labels are hyphenated single tokens so word counting works.
# Uses NATIONAL population-weighted z-scores to reflect genuine distinctiveness
# across the full 55-type national population.
#
# Ordering rationale:
# 1. Rare racial/ethnic composition distinguishes Black Belt, Hispanic, Asian types
# 2. Urbanicity — Urban vs Deep-Rural is the biggest structural split
# 3. Religion — evangelical/Catholic/secular is highly differentiating
# 4. Age — distinguishes retirement communities from young rural
# 5. Income — differentiates remaining types before education
# 6. Migration — growth vs decline
# 7. Education / professional class — last because many rural types share low-edu
# ---------------------------------------------------------------------------
_VOCAB: list[tuple[str, float, str, str]] = [
    # Rare racial/ethnic composition (with absolute minimums enforced).
    # Three-level pct_black system:
    #   z >= 3.0 (~>45% Black)  → "Majority-Black"  (extreme Black Belt core)
    #   z >= 1.2 (~>30% Black)  → "Black-Belt"       (traditional Black Belt)
    #   z >= 0.6 (~>20% Black)  → "Black-Belt"       (elevated minority share)
    # Using 3.0 keeps "Majority-Black" to ~3 types (the most extreme),
    # leaving "Black-Belt" for the moderate-to-high tier.
    ("pct_black",               3.0,    "Majority-Black", ""),
    ("pct_hispanic",            Z_HIGH, "Hispanic",       ""),
    ("pct_asian",               Z_HIGH, "Asian",          ""),
    ("pct_black",               Z_MOD,  "Black-Belt",     ""),
    ("pct_hispanic",            Z_MOD,  "Hispanic",       ""),

    # Urbanicity / density
    ("log_pop_density",         Z_HIGH, "Urban",        "Deep-Rural"),
    ("log_pop_density",         Z_MOD,  "Suburban",     "Rural"),

    # Religion — ordered to maximise differentiation:
    # 1. Evangelical Z_HIGH: identifies the dominant rural Protestant bloc
    # 2. Catholic Z_HIGH: coastal/Hispanic types
    # 3. Mainline Z_HIGH: identifies specific mainline-dominant communities
    # 4. religious_adherence_rate: "Devout" (very high adherence) fires early;
    #    negative = "Low-Church" (NOT "Secular" — avoids contradiction with
    #    "Evangelical" since a community can be ethnically evangelical but
    #    have lower formal church participation).
    # 5. religious_adherence_rate Z_MOD: Churched/Unchurched
    # 6. black_protestant_share: placed AFTER adherence so that within-the-
    #    evangelical-blob types get "Low-Church"/"Unchurched" first, leaving
    #    "Black-Church" as a 4th token for further disambiguation.
    ("evangelical_share",       Z_HIGH, "Evangelical",  ""),
    ("catholic_share",          Z_HIGH, "Catholic",     ""),
    ("mainline_share",          Z_HIGH, "Mainline",     ""),
    ("religious_adherence_rate", Z_HIGH, "Devout",      ""),
    ("evangelical_share",       Z_MOD,  "Evangelical",  ""),
    ("catholic_share",          Z_MOD,  "Catholic",     ""),
    ("religious_adherence_rate", Z_MOD,  "Churched",    "Unchurched"),
    ("black_protestant_share",  Z_HIGH, "Black-Church", ""),

    # Age — retirement communities and young growth areas.
    # Before income so that rural types like "Deep-Rural Evangelical Retiree"
    # vs "Deep-Rural Evangelical Younger" are distinguished early.
    ("median_age",              Z_HIGH, "Retiree",      "Young"),
    ("median_age",              Z_MOD,  "Older",        "Younger"),

    # Income — after religion+age so the remaining types haven't already diverged
    ("median_hh_income",        Z_HIGH, "Affluent",     "Low-Income"),
    ("avg_inflow_income",       Z_HIGH, "Wealth-Magnet", ""),
    ("median_hh_income",        Z_MOD,  "Mid-Income",   "Blue-Collar"),

    # Migration / growth
    ("net_migration_rate",      Z_HIGH, "High-Growth",  "Declining"),
    ("inflow_outflow_ratio",    Z_HIGH, "Inflow",       "Outflow"),
    ("migration_diversity",     Z_HIGH, "Cosmopolitan", ""),
    ("net_migration_rate",      Z_MOD,  "Growing",      "Shrinking"),

    # Education / professional class — last to avoid Working-Class domination
    ("pct_bachelors_plus",      Z_HIGH, "College",      ""),
    ("pct_graduate",            Z_HIGH, "Grad-Degree",  ""),
    ("pct_management",          Z_HIGH, "Professional", ""),
    ("pct_bachelors_plus",      Z_MOD,  "Educated",     "Working-Class"),

    # Work / commute
    ("pct_wfh",                 Z_HIGH, "Remote-Work",  ""),
    ("pct_car",                 Z_MOD,  "Auto-Commute", ""),

    # Homeownership
    ("pct_owner_occupied",      Z_HIGH, "Homeowner",    "Renter"),

    # Land area (sprawling counties)
    ("land_area_sq_mi",         Z_HIGH, "Sprawling",    "Compact"),
]

# Feature families — only one token per family per type
_FAMILY: dict[str, str] = {
    "pct_graduate":             "education",
    "pct_bachelors_plus":       "education",
    "pct_management":           "professional",
    "log_pop_density":          "density",
    "pop_per_sq_mi":            "density",
    "pct_wfh":                  "work",
    "pct_car":                  "commute",
    "evangelical_share":        "religion_ev",
    "catholic_share":           "religion_cat",
    "mainline_share":           "religion_ml",
    "black_protestant_share":   "religion_bp",
    "religious_adherence_rate":  "religion_adh",
    "congregations_per_1000":   "religion_cong",
    "pct_black":                "race_black",
    "pct_hispanic":             "race_hisp",
    "pct_asian":                "race_asian",
    "pct_white_nh":             "race_white",
    "median_hh_income":         "income",
    "avg_inflow_income":        "income_in",
    "median_age":               "age",
    "net_migration_rate":       "migration",
    "inflow_outflow_ratio":     "migration_io",
    "migration_diversity":      "migration_div",
    "pct_owner_occupied":       "ownership",
    "pct_transit":              "transit",
    "land_area_sq_mi":          "area",
    "pop_total":                "population",
    "n_counties":               "breadth",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _get_label(feat: str, z: float, threshold: float, pos: str, neg: str) -> str | None:
    """Return label if |z| >= threshold, else None."""
    if abs(z) < threshold:
        return None
    return (pos if z > 0 else neg) or None


def _top_tokens(
    z_row: pd.Series,
    vocab: list[tuple[str, float, str, str]],
    n: int = 2,
    exclude: set[str] | None = None,
    raw_row: pd.Series | None = None,
) -> list[str]:
    """Extract up to n non-redundant descriptive tokens from vocab.

    Parameters
    ----------
    raw_row:
        Raw (un-z-scored) values for the type. Used to enforce minimum
        absolute thresholds from ``_MIN_ABS`` on race/ethnicity features.
    """
    tokens: list[str] = []
    seen_families: set[str] = set()
    exclude = exclude or set()

    for feat, thresh, pos, neg in vocab:
        if feat not in z_row.index:
            continue
        fam = _FAMILY.get(feat, feat)
        if fam in seen_families:
            continue
        z = float(z_row[feat])
        label = _get_label(feat, z, thresh, pos, neg)
        if not label or label in tokens or label in exclude:
            continue
        # Enforce minimum absolute value for race features
        if feat in _MIN_ABS and raw_row is not None and feat in raw_row.index:
            if float(raw_row[feat]) < _MIN_ABS[feat]:
                continue
        tokens.append(label)
        seen_families.add(fam)
        if len(tokens) >= n:
            break

    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_zscores(
    profiles: pd.DataFrame,
    features: list[str],
    weight_col: str = "pop_total",
) -> pd.DataFrame:
    """Compute population-weighted z-scores for each feature column.

    Z-scores are computed relative to the population-weighted mean and
    standard deviation across ALL rows in ``profiles``.  For the national
    model this means the reference is the national population-weighted mean,
    not a within-state mean.

    Parameters
    ----------
    profiles:
        DataFrame with one row per type.  Must contain ``type_id`` and
        the columns listed in ``features`` (missing columns are skipped).
    features:
        Column names to z-score.
    weight_col:
        Column used as population weights when computing the weighted mean
        and variance across types.

    Returns
    -------
    DataFrame with ``type_id`` plus one z-score column per available feature.
    """
    available = [f for f in features if f in profiles.columns]
    weights = profiles[weight_col].fillna(1.0).values.astype(float)
    total_w = weights.sum()
    if total_w <= 0:
        total_w = 1.0

    z_data: dict[str, list[float]] = {"type_id": profiles["type_id"].tolist()}

    for feat in available:
        vals = profiles[feat].fillna(0.0).values.astype(float)
        wmean = float(np.dot(weights, vals) / total_w)
        wvar = float(np.dot(weights, (vals - wmean) ** 2) / total_w)
        wstd = float(np.sqrt(wvar)) if wvar > 0 else 1.0
        z_data[feat] = list((vals - wmean) / wstd)

    return pd.DataFrame(z_data)


def name_types(
    profiles: pd.DataFrame | None = None,
    county_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate descriptive display names for all fine electoral types.

    Names are 2–4 token strings derived from national population-weighted
    z-scores.  No state prefix is included — types now span the whole country
    so a state label would be misleading.

    Parameters
    ----------
    profiles:
        type_profiles DataFrame (55 rows × demographic columns).
        If None, loaded from ``data/communities/type_profiles.parquet``.
    county_assignments:
        county_type_assignments_full DataFrame; used to map each fine type
        to its super-type.  If None, loaded from disk.

    Returns
    -------
    DataFrame with columns ``[type_id, display_name]``, one row per type,
    sorted by ``type_id``.  All display names are unique.

    Side effects
    ------------
    When *profiles* is None (loading from disk), the ``display_name`` column
    is persisted back into ``type_profiles.parquet``.
    """
    save_to_disk = profiles is None

    if profiles is None:
        profiles = pd.read_parquet(TYPE_PROFILES_PATH)
    if county_assignments is None and COUNTY_TYPE_ASSIGNMENTS_PATH.exists():
        county_assignments = pd.read_parquet(COUNTY_TYPE_ASSIGNMENTS_PATH)

    # ---- Compute national z-scores ------------------------------------------
    # Use ALL types together as the reference population (national model).
    # This replaces the old within-state z-score approach.
    unique_features: list[str] = []
    seen_f: set[str] = set()
    for feat, *_ in _VOCAB:
        if feat not in seen_f:
            unique_features.append(feat)
            seen_f.add(feat)

    z_df = compute_zscores(profiles, unique_features, weight_col="pop_total")
    feat_cols = [c for c in z_df.columns if c != "type_id"]

    # ---- Determine each type's super-type -----------------------------------
    type_to_super: dict[int, int] = {}
    if county_assignments is not None and "super_type" in county_assignments.columns:
        mapping = (
            county_assignments.groupby("dominant_type")["super_type"]
            .agg(lambda x: int(x.mode().iloc[0]))
            .reset_index()
        )
        type_to_super = {
            int(r["dominant_type"]): int(r["super_type"])
            for _, r in mapping.iterrows()
        }

    # ---- Collect base token lists for every type ----------------------------
    # Collect up to 3 tokens for initial naming (token-only, no state prefix).
    _all_tokens: dict[int, list[str]] = {}
    raw_names: dict[int, str] = {}

    for _, row in profiles.iterrows():
        tid = int(row["type_id"])
        z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
        if z_sub.empty:
            raw_names[tid] = f"Type {tid}"
            _all_tokens[tid] = []
            continue

        z_row = z_sub.iloc[0]
        # Collect up to 3 tokens from national z-scores
        tokens = _top_tokens(z_row, _VOCAB, n=3, raw_row=row)
        _all_tokens[tid] = tokens

        # If fewer than 2 tokens, use super-type as fallback context
        super_id = type_to_super.get(tid, -1)

        if len(tokens) >= 2:
            raw_names[tid] = f"{tokens[0]} {tokens[1]}"
        elif len(tokens) == 1:
            # One demographic token + super-type fallback word
            fallback = _super_fallback_word(super_id)
            raw_names[tid] = f"{tokens[0]} {fallback}"
        else:
            # No tokens at all — use super-type description
            fallback = _super_fallback_word(super_id)
            raw_names[tid] = f"Mixed {fallback}"

    # ---- Pre-promote additional tokens to resolve duplicate names -----------
    # Strategy: iteratively promote vocab tokens into names until each name
    # is unique or we hit the 4-word budget.
    #
    # For each duplicate group, scan the extended token list to find the first
    # position where types diverge.  Sub-groups that remain tied after a split
    # are recursively split at subsequent token positions.

    # Collect extended tokens (up to 8) for all types
    _ext_tokens: dict[int, list[str]] = {}
    for _, row in profiles.iterrows():
        tid = int(row["type_id"])
        z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
        if z_sub.empty:
            _ext_tokens[tid] = _all_tokens.get(tid, [])
            continue
        z_row = z_sub.iloc[0]
        _ext_tokens[tid] = _top_tokens(z_row, _VOCAB, n=8, raw_row=row)

    def _try_promote_group(tids_in: list[int], current_names: dict[int, str]) -> None:
        """Recursively promote tokens to resolve a single duplicate group.

        Strategy:
        1. Prefer a position that makes ALL types in the group unique (full split).
        2. If no full split exists within the 4-word budget, use the first
           partial split and recurse on remaining sub-groups.
        3. For sub-groups already at 4 words, try REPLACING the last token
           with a more differentiating token that creates full uniqueness.
        """
        # Group the types by their current name
        by_name: dict[str, list[int]] = defaultdict(list)
        for tid in tids_in:
            by_name[current_names[tid]].append(tid)

        for name, group_tids in by_name.items():
            if len(group_tids) < 2:
                continue
            n_words = len(name.split())
            name_parts = name.split()
            start_pos = len(name_parts)    # next token index to try

            max_pos = max(len(_ext_tokens.get(t, [])) for t in group_tids)

            if n_words >= 4:
                # Can't append — try replacing the last token instead.
                name_prefix = " ".join(name.split()[:-1])  # drop last word
                replace_start = start_pos - 1  # re-try last position
                for pos in range(replace_start, max_pos):
                    candidate = {
                        tid: (_ext_tokens[tid][pos] if len(_ext_tokens[tid]) > pos else "")
                        for tid in group_tids
                    }
                    non_empty = [v for v in candidate.values() if v]
                    if len(set(non_empty)) == len(group_tids):
                        # Replace last token with a differentiating one
                        for tid in group_tids:
                            tok = candidate[tid]
                            if tok:
                                current_names[tid] = f"{name_prefix} {tok}"
                        return
                continue  # no replacement found; leave for DISAMBIG

            # Budget available: scan for best position
            # Prefer fully-unique over partially-unique splits
            first_partial_pos: int | None = None
            for pos in range(start_pos, max_pos):
                candidate = {
                    tid: (_ext_tokens[tid][pos] if len(_ext_tokens[tid]) > pos else "")
                    for tid in group_tids
                }
                non_empty = [v for v in candidate.values() if v]
                n_unique = len(set(non_empty))
                if n_unique < 2:
                    continue
                if n_unique == len(group_tids):
                    # Fully unique — use this position directly
                    for tid in group_tids:
                        tok = candidate[tid]
                        if tok:
                            current_names[tid] = f"{name} {tok}"
                    return  # all resolved, no recursion needed
                # Partially differentiating — remember the first such pos
                if first_partial_pos is None:
                    first_partial_pos = pos

            # No fully-unique position found — use the first partial split
            # and recurse on each sub-group
            if first_partial_pos is not None:
                pos = first_partial_pos
                candidate = {
                    tid: (_ext_tokens[tid][pos] if len(_ext_tokens[tid]) > pos else "")
                    for tid in group_tids
                }
                for tid in group_tids:
                    tok = candidate[tid]
                    if tok:
                        current_names[tid] = f"{name} {tok}"
                # Recurse to split any remaining sub-group duplicates
                _try_promote_group(group_tids, current_names)

    # Run up to 3 passes of pre-promotion (each pass may create new sub-groups)
    for _pre_pass in range(3):
        dupe_counts: Counter[str] = Counter(raw_names.values())
        dup_names = [n for n, c in dupe_counts.items() if c >= 2]
        if not dup_names:
            break
        for name in dup_names:
            tids_with_name = [t for t, n in raw_names.items() if n == name]
            _try_promote_group(tids_with_name, raw_names)

    # ---- Disambiguation -----------------------------------------------------
    display_names = _disambiguate(raw_names, profiles, z_df, feat_cols)

    # ---- Build result -------------------------------------------------------
    result = pd.DataFrame(
        [{"type_id": tid, "display_name": name}
         for tid, name in sorted(display_names.items())]
    )

    # ---- Persist ------------------------------------------------------------
    if save_to_disk and TYPE_PROFILES_PATH.exists():
        profiles_on_disk = pd.read_parquet(TYPE_PROFILES_PATH)
        name_map = dict(zip(result["type_id"], result["display_name"]))
        profiles_on_disk["display_name"] = profiles_on_disk["type_id"].map(name_map)
        profiles_on_disk.to_parquet(TYPE_PROFILES_PATH, index=False)
        log.info("Saved display_name to %s", TYPE_PROFILES_PATH)

    return result


def _super_fallback_word(super_id: int) -> str:
    """Return a short fallback word for a super-type when vocab tokens are scarce."""
    # These are broad characterizations used only when vocab is exhausted.
    # They are NOT the canonical super-type names (those come from name_super_types()).
    _FALLBACKS = {
        0: "Heartland",
        1: "Diverse",
        2: "Mainstream",
        3: "Border",
        4: "Sunbelt",
    }
    return _FALLBACKS.get(super_id, "Mixed")


# ---------------------------------------------------------------------------
# Super-type naming  (data-driven, deterministic)
# ---------------------------------------------------------------------------

# Super-type vocabulary: same structure as _VOCAB but tuned for aggregate profiles.
# Since super-types aggregate many fine types, z-scores are compressed — use lower
# thresholds so more features fire.
_SUPER_VOCAB: list[tuple[str, float, str, str]] = [
    # Race / ethnicity
    ("pct_black",               0.8,    "Black-Belt",    ""),
    ("pct_hispanic",            0.8,    "Hispanic",      ""),
    ("pct_asian",               0.8,    "Asian-Pacific", ""),
    # Urbanicity
    ("log_pop_density",         0.8,    "Urban",         "Rural"),
    ("log_pop_density",         0.4,    "Suburban",      "Exurban"),
    # Religion
    ("evangelical_share",       0.8,    "Evangelical",   ""),
    ("catholic_share",          0.8,    "Catholic",      ""),
    # Age
    ("median_age",              0.8,    "Retiree",       "Young"),
    # Income / education
    ("median_hh_income",        0.8,    "Affluent",      "Low-Income"),
    ("pct_bachelors_plus",      0.8,    "College",       ""),
    # Migration
    ("net_migration_rate",      0.8,    "High-Growth",   "Declining"),
    # Professional / WFH
    ("pct_wfh",                 0.8,    "Remote-Work",   ""),
    ("pct_management",          0.8,    "Professional",  ""),
]


def name_super_types(
    profiles: pd.DataFrame | None = None,
    county_assignments: pd.DataFrame | None = None,
    fine_type_names: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate descriptive display names for the 5 super-types.

    Builds an aggregate demographic profile for each super-type by averaging
    the type_profiles rows weighted by county count, then applies the same
    vocabulary approach used for fine types.

    Parameters
    ----------
    profiles:
        type_profiles DataFrame (55 rows).  If None, loaded from disk.
    county_assignments:
        county_type_assignments_full DataFrame.  If None, loaded from disk.
    fine_type_names:
        Output of name_types() — used to build a summary of constituent types
        for logging.  If None, fine type names are not included in log output.

    Returns
    -------
    DataFrame with columns ``[super_type_id, display_name]``, one row per
    super-type.  Names are unique and descriptive.
    """
    save_to_disk = profiles is None

    if profiles is None and TYPE_PROFILES_PATH.exists():
        profiles = pd.read_parquet(TYPE_PROFILES_PATH)
    if county_assignments is None and COUNTY_TYPE_ASSIGNMENTS_PATH.exists():
        county_assignments = pd.read_parquet(COUNTY_TYPE_ASSIGNMENTS_PATH)

    if profiles is None or county_assignments is None:
        log.warning("Cannot name super types: missing profiles or county_assignments")
        return pd.DataFrame(columns=["super_type_id", "display_name"])

    if "super_type" not in county_assignments.columns:
        log.warning("county_assignments missing super_type column")
        return pd.DataFrame(columns=["super_type_id", "display_name"])

    super_ids = sorted(county_assignments["super_type"].unique())

    # Build aggregate profile per super-type
    # Weight each fine type by the number of counties assigned to it within the super
    numeric_cols = [
        c for c in profiles.columns
        if c not in ("type_id", "display_name") and profiles[c].dtype.kind in "fi"
    ]
    agg_rows: list[dict] = []
    type_id_lists: dict[int, list[int]] = {}

    for sid in super_ids:
        st_counties = county_assignments[county_assignments["super_type"] == sid]
        type_counts = st_counties["dominant_type"].value_counts()
        type_ids_in = [t for t in type_counts.index if t in profiles["type_id"].values]
        type_id_lists[sid] = sorted(type_ids_in)

        subset = profiles[profiles["type_id"].isin(type_ids_in)].copy()
        # Weight = county_count × pop_total (county-population weighted average)
        county_w = subset["type_id"].map(type_counts).fillna(1.0)
        pop_w = subset["pop_total"].fillna(1.0)
        weights = (county_w * pop_w).values.astype(float)
        total_w = weights.sum()
        if total_w <= 0:
            total_w = 1.0

        row: dict = {"type_id": sid, "pop_total": float(weights.sum())}
        for col in numeric_cols:
            if col in subset.columns:
                vals = subset[col].fillna(0.0).values.astype(float)
                row[col] = float(np.dot(weights, vals) / total_w)
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    # Compute z-scores across super-types (national reference = all super-types)
    super_features = [e[0] for e in _SUPER_VOCAB]
    unique_sf = list(dict.fromkeys(super_features))
    z_super = compute_zscores(agg_df, unique_sf, weight_col="pop_total")
    sfeat_cols = [c for c in z_super.columns if c != "type_id"]

    # Build raw names from vocab
    raw_super: dict[int, str] = {}
    for sid in super_ids:
        z_sub = z_super.loc[z_super["type_id"] == sid, sfeat_cols]
        raw_row = agg_df[agg_df["type_id"] == sid].iloc[0]
        if z_sub.empty:
            raw_super[sid] = f"Type-{sid}"
            continue
        z_row = z_sub.iloc[0]
        tokens = _top_tokens(z_row, _SUPER_VOCAB, n=2, raw_row=raw_row)
        if len(tokens) >= 2:
            raw_super[sid] = f"{tokens[0]} {tokens[1]}"
        elif len(tokens) == 1:
            raw_super[sid] = f"{tokens[0]} Coalition"
        else:
            raw_super[sid] = f"Mixed Coalition"

    # Ensure uniqueness with disambiguation
    final_super = _disambiguate_super(raw_super, agg_df, z_super, sfeat_cols)

    result = pd.DataFrame(
        [{"super_type_id": sid, "display_name": name}
         for sid, name in sorted(final_super.items())]
    )

    if save_to_disk and SUPER_TYPES_PATH.exists():
        result.to_parquet(SUPER_TYPES_PATH, index=False)
        log.info("Saved super-type display names to %s", SUPER_TYPES_PATH)

    # Log summary
    for sid, name in sorted(final_super.items()):
        n_types = len(type_id_lists.get(sid, []))
        log.info(
            "Super-type %d → '%s'  (%d fine types: %s)",
            sid, name, n_types, type_id_lists.get(sid, [])
        )

    return result


def _disambiguate_super(
    raw: dict[int, str],
    agg_df: pd.DataFrame,
    z_df: pd.DataFrame,
    feat_cols: list[str],
) -> dict[int, str]:
    """Ensure super-type names are unique."""
    final = dict(raw)
    _ORDINALS = ["", " II", " III", " IV", " V"]

    def _get_dupes(d: dict[int, str]) -> dict[str, list[int]]:
        counts = Counter(d.values())
        return {
            name: sorted(t for t, n in d.items() if n == name)
            for name, cnt in counts.items()
            if cnt > 1
        }

    # One pass of variance-based disambiguation
    for name, sids in _get_dupes(final).items():
        n_words = len(name.split())
        if n_words >= 4:
            continue
        best_var = -1.0
        best_labels: dict[int, str] = {}
        existing_tokens = set(name.split())
        for feat, _thresh, pos, neg in _DISAMBIG_VOCAB:
            z_vals: dict[int, float] = {}
            for sid in sids:
                z_sub = z_df.loc[z_df["type_id"] == sid, feat_cols]
                if z_sub.empty or feat not in z_sub.columns:
                    continue
                z_vals[sid] = float(z_sub.iloc[0][feat])
            if len(z_vals) < len(sids):
                continue
            vals_arr = np.array(list(z_vals.values()))
            var = float(np.var(vals_arr))
            if var <= best_var:
                continue
            median_z = float(np.median(vals_arr))
            labels: dict[int, str] = {}
            for sid in sids:
                z = z_vals[sid]
                if z > median_z and pos and pos not in existing_tokens:
                    labels[sid] = pos
                elif z < median_z and neg and neg not in existing_tokens:
                    labels[sid] = neg
                elif pos and pos not in existing_tokens:
                    labels[sid] = pos
                elif neg and neg not in existing_tokens:
                    labels[sid] = neg
                else:
                    labels[sid] = ""
            non_empty = [lbl for lbl in labels.values() if lbl]
            if len(set(non_empty)) >= 2:
                best_var = var
                best_labels = labels
        if best_labels:
            for sid in sids:
                lbl = best_labels.get(sid, "")
                if lbl:
                    final[sid] = f"{name} {lbl}"

    # Ordinal fallback
    for name, sids in _get_dupes(final).items():
        sorted_sids = sorted(sids)
        for i, sid in enumerate(sorted_sids):
            if i == 0:
                continue
            suffix = _ORDINALS[min(i, len(_ORDINALS) - 1)]
            final[sid] = f"{name}{suffix}"

    return final


# ---------------------------------------------------------------------------
# Disambiguation
# ---------------------------------------------------------------------------

# Extended vocab for 4th-token disambiguation — different features from primary.
# Ordered to try the most differentiating features first.
# Public-facing labels — no "Blacker"/"Whiter" or relative ethnicity framing.
_DISAMBIG_VOCAB: list[tuple[str, float, str, str]] = [
    # Age is the single best differentiator within Rural/Deep-Rural groups
    ("median_age",              Z_LOW,  "Older",        "Younger"),
    # Migration: growth vs decline cleaves stagnant from boom rural types
    ("net_migration_rate",      Z_LOW,  "Growing",      "Shrinking"),
    ("inflow_outflow_ratio",    Z_LOW,  "Inflow",       "Outflow"),
    # Religion: evangelical intensity splits Deep-Rural types cleanly
    ("evangelical_share",       Z_LOW,  "Evangelical",  ""),
    ("catholic_share",          Z_LOW,  "Catholic",     ""),
    ("mainline_share",          Z_LOW,  "Mainline",     ""),
    ("black_protestant_share",  Z_LOW,  "Black-Church", ""),
    ("religious_adherence_rate", Z_LOW, "Devout",       "Unchurched"),
    # Income inflow: wealth magnet vs not
    ("avg_inflow_income",       Z_LOW,  "Wealth-Magnet", ""),
    ("median_hh_income",        Z_LOW,  "Higher-Inc",   "Lower-Inc"),
    # Education / professional
    ("pct_management",          Z_LOW,  "White-Collar", "Blue-Collar"),
    ("pct_bachelors_plus",      Z_LOW,  "Educated",     ""),
    # Minority composition — public-friendly framing
    ("pct_black",               Z_LOW,  "Minority-Mix", "Anglo-Majority"),
    ("pct_hispanic",            Z_LOW,  "Latino-Mix",   ""),
    # Housing / ownership
    ("pct_owner_occupied",      Z_LOW,  "Homeowner",    "Renter"),
    # Migration diversity
    ("migration_diversity",     Z_LOW,  "Cosmopolitan", "Insular"),
    # Spatial / density
    ("land_area_sq_mi",         Z_LOW,  "Sprawling",    "Compact"),
    ("log_pop_density",         Z_LOW,  "Denser",       "Sparser"),
    # Commute patterns
    ("pct_car",                 Z_LOW,  "Auto-Commute", ""),
    ("pct_wfh",                 Z_LOW,  "Remote-Work",  ""),
    ("pct_transit",             Z_LOW,  "Transit",      ""),
    # Type breadth
    ("congregations_per_1000",  Z_LOW,  "Many-Churches", ""),
    ("pop_total",               Z_LOW,  "Large",        "Small"),
    ("n_counties",              Z_LOW,  "Broad",        "Concentrated"),
]


def _compute_type_lean() -> dict[int, float]:
    """Compute mean presidential Dem shift per type from shift data.

    Returns a dict of type_id → mean_dem_shift (positive = bluer over time).
    Used as a last-resort disambiguator when demographics can't distinguish types.
    Returns empty dict if data isn't available.
    """
    if not _SHIFTS_PATH.exists() or not _TYPE_ASSIGNMENTS_PATH.exists():
        return {}
    try:
        shifts = pd.read_parquet(_SHIFTS_PATH)
        ta = pd.read_parquet(_TYPE_ASSIGNMENTS_PATH)

        # Get presidential D-share shift columns
        pres_cols = [c for c in shifts.columns if c.startswith("pres_d_shift_")]
        if not pres_cols:
            return {}

        # Score columns: type_X_score pattern
        score_cols = sorted([c for c in ta.columns if c.endswith("_score")])
        if not score_cols:
            return {}

        scores = ta[score_cols].values  # (N, J)
        shift_vals = shifts[pres_cols].mean(axis=1).values  # (N,) mean across elections

        weights = np.abs(scores)
        w_sum = weights.sum(axis=0) + 1e-12
        type_lean = (weights * shift_vals[:, None]).sum(axis=0) / w_sum

        return {i: float(type_lean[i]) for i in range(len(type_lean))}
    except Exception:
        return {}


def _lean_label(lean: float) -> str:
    """Convert a mean Dem shift to a human-readable lean label."""
    if lean > 0.01:
        return "Blue-Shift"
    elif lean < -0.01:
        return "Red-Shift"
    else:
        return "Swing"


def _disambiguate(
    raw_names: dict[int, str],
    profiles: pd.DataFrame,
    z_df: pd.DataFrame,
    feat_cols: list[str],
) -> dict[int, str]:
    """Ensure every type gets a unique display name (max 4 words).

    Strategy:
    1. For each duplicate group, find the feature that best splits the group
       (maximizes variance of z-scores within the group).
    2. Assign per-type labels using the splitting feature's direction.
    3. When a group is larger than 2, iterate sub-group splits across multiple
       passes until all duplicates are resolved or word budget is exhausted.
    4. Any remaining duplicates get ordinal suffixes.

    All z-scores used are NATIONAL population-weighted z-scores (z_df),
    so features show full spread across the national 55-type population.
    """
    _ORDINALS = ["", " II", " III", " IV", " V", " VI", " VII", " VIII"]

    def _get_dupes(d: dict[int, str]) -> dict[str, list[int]]:
        counts = Counter(d.values())
        return {
            name: sorted(t for t, n in d.items() if n == name)
            for name, cnt in counts.items()
            if cnt > 1
        }

    def _get_z(tid: int, feat: str) -> float | None:
        """Return national z-score for a type/feature pair."""
        z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
        if z_sub.empty or feat not in z_sub.columns:
            return None
        return float(z_sub.iloc[0][feat])

    def _splitting_token(tids: list[int], existing_name: str) -> dict[int, str]:
        """Find the feature that best splits a group of types and return
        per-type labels.

        Uses NATIONAL z-scores for maximum separation signal.
        Chooses the feature with highest within-group variance that produces
        at least 2 distinct label outcomes.
        """
        existing_tokens = set(existing_name.split())
        best_var = -1.0
        best_labels: dict[int, str] = {}

        for feat, _thresh, pos, neg in _DISAMBIG_VOCAB:
            if not pos and not neg:
                continue
            # Get national z-values for all types in the group
            z_vals: dict[int, float] = {}
            for tid in tids:
                z = _get_z(tid, feat)
                if z is not None:
                    z_vals[tid] = z
            if len(z_vals) < len(tids):
                continue

            vals_arr = np.array(list(z_vals.values()))
            var = float(np.var(vals_arr))
            if var <= best_var:
                continue

            # Generate labels: above median gets positive, below gets negative
            median_z = float(np.median(vals_arr))
            labels: dict[int, str] = {}
            for tid in tids:
                z = z_vals[tid]
                if z > median_z and pos and pos not in existing_tokens:
                    labels[tid] = pos
                elif z < median_z and neg and neg not in existing_tokens:
                    labels[tid] = neg
                elif z > median_z and pos and pos not in existing_tokens:
                    labels[tid] = pos
                elif pos and pos not in existing_tokens:
                    labels[tid] = pos
                elif neg and neg not in existing_tokens:
                    labels[tid] = neg
                else:
                    labels[tid] = ""

            # Only accept if it actually differentiates (at least 2 distinct labels)
            non_empty = [lbl for lbl in labels.values() if lbl]
            if len(set(non_empty)) >= 2:
                best_var = var
                best_labels = labels

        return best_labels

    def _pop_order(tids: list[int]) -> list[int]:
        return (
            profiles[profiles["type_id"].isin(tids)]
            .sort_values(["n_counties", "type_id"], ascending=[False, True])["type_id"]
            .tolist()
        )

    final: dict[int, str] = dict(raw_names)

    # Pass 1a — iteratively split duplicate groups by APPENDING tokens.
    # Run up to 6 passes so that splits of large groups cascade into sub-group
    # splits in subsequent iterations (e.g., a 9-way tie may need 4 passes).
    for _pass in range(6):
        dupes = _get_dupes(final)
        if not dupes:
            break
        changed = False
        for name, tids in dupes.items():
            n_words = len(name.split())
            if n_words >= 4:
                continue  # word budget exhausted for this name
            labels = _splitting_token(tids, name)
            for tid in tids:
                label = labels.get(tid, "")
                if label:
                    new_name = f"{name} {label}"
                    if new_name != final[tid]:
                        final[tid] = new_name
                        changed = True
        if not changed:
            break  # no progress — stop early

    # Cap at 4 words
    for tid in final:
        parts = final[tid].split()
        if len(parts) > 4:
            final[tid] = " ".join(parts[:4])

    # Pass 1b — for 4-word duplicate groups, try REPLACING the last token
    # with a more differentiating one.  Run in a loop so that the replacements
    # from one group don't create new collisions with other groups.
    for _pass1b in range(4):
        dupes_4 = {n: t for n, t in _get_dupes(final).items() if len(n.split()) == 4}
        if not dupes_4:
            break
        any_changed = False
        for name, tids in dupes_4.items():
            name_prefix = " ".join(name.split()[:3])  # 3-word prefix
            existing_tokens = set(name_prefix.split())

            best_labels: dict[int, str] = {}
            best_var = -1.0

            for feat, _thresh, pos, neg in _DISAMBIG_VOCAB:
                if not pos and not neg:
                    continue
                z_vals: dict[int, float] = {}
                for tid in tids:
                    z = _get_z(tid, feat)
                    if z is not None:
                        z_vals[tid] = z
                if len(z_vals) < len(tids):
                    continue

                vals_arr = np.array(list(z_vals.values()))
                var = float(np.var(vals_arr))
                if var <= best_var:
                    continue

                median_z = float(np.median(vals_arr))
                labels: dict[int, str] = {}
                for tid in tids:
                    z = z_vals[tid]
                    if z > median_z and pos and pos not in existing_tokens:
                        labels[tid] = pos
                    elif z < median_z and neg and neg not in existing_tokens:
                        labels[tid] = neg
                    elif pos and pos not in existing_tokens:
                        labels[tid] = pos
                    elif neg and neg not in existing_tokens:
                        labels[tid] = neg
                    else:
                        labels[tid] = ""

                non_empty = [lbl for lbl in labels.values() if lbl]
                if len(set(non_empty)) >= 2:
                    best_var = var
                    best_labels = labels

            if best_labels:
                # Check that proposed replacements don't collide with existing names.
                all_current_names = set(final.values()) - set(final[t] for t in tids)

                def _try_collision_free() -> bool:
                    """Try every DISAMBIG feature in variance order; return True if applied."""
                    # Collect all candidate splits sorted by variance (desc)
                    candidates: list[tuple[float, dict[int, str]]] = []
                    for feat2, _t2, pos2, neg2 in _DISAMBIG_VOCAB:
                        if not pos2 and not neg2:
                            continue
                        z_vals2: dict[int, float] = {}
                        for tid in tids:
                            z = _get_z(tid, feat2)
                            if z is not None:
                                z_vals2[tid] = z
                        if len(z_vals2) < len(tids):
                            continue
                        vals2 = np.array(list(z_vals2.values()))
                        var2 = float(np.var(vals2))
                        med2 = float(np.median(vals2))
                        lbls: dict[int, str] = {}
                        for tid in tids:
                            z = z_vals2[tid]
                            if z > med2 and pos2 and pos2 not in existing_tokens:
                                lbls[tid] = pos2
                            elif z < med2 and neg2 and neg2 not in existing_tokens:
                                lbls[tid] = neg2
                            elif pos2 and pos2 not in existing_tokens:
                                lbls[tid] = pos2
                            elif neg2 and neg2 not in existing_tokens:
                                lbls[tid] = neg2
                            else:
                                lbls[tid] = ""
                        non_emp = [l for l in lbls.values() if l]
                        if len(set(non_emp)) >= 2:
                            candidates.append((var2, lbls))
                    # Try highest-variance first; skip any that cause collisions
                    for _, lbls in sorted(candidates, key=lambda x: -x[0]):
                        proposed2: dict[int, str] = {}
                        ok = True
                        proposed_names: set[str] = set()
                        for tid in tids:
                            lbl = lbls.get(tid, "")
                            if lbl:
                                nn = f"{name_prefix} {lbl}"
                                if nn in all_current_names or nn in proposed_names:
                                    ok = False
                                    break
                                proposed2[tid] = nn
                                proposed_names.add(nn)
                        if ok and proposed2:
                            for tid, nn in proposed2.items():
                                if nn != final[tid]:
                                    final[tid] = nn
                                    nonlocal any_changed  # type: ignore[misc]
                                    any_changed = True
                            return True
                    return False

                _try_collision_free()
        if not any_changed:
            break

    # Pass 2 — political lean disambiguation (last-resort before ordinals)
    # Types defined by electoral behavior can legitimately be distinguished
    # by their mean partisan shift direction.
    type_lean = _compute_type_lean()
    if type_lean:
        for name, tids in _get_dupes(final).items():
            n_words = len(name.split())
            if n_words >= 4:
                continue
            lean_labels = {tid: _lean_label(type_lean.get(tid, 0.0)) for tid in tids}
            if len(set(lean_labels.values())) >= 2:
                for tid in tids:
                    final[tid] = f"{name} {lean_labels[tid]}"

    # Pass 3 — ordinal suffixes for any remaining duplicates
    for name, tids in _get_dupes(final).items():
        for i, tid in enumerate(_pop_order(tids)):
            if i == 0:
                continue  # largest group keeps the name
            suffix = _ORDINALS[min(i, len(_ORDINALS) - 1)]
            final[tid] = f"{name}{suffix}"

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Step 1: name fine types
    result = name_types()
    print(f"\nGenerated {len(result)} type names:\n")
    for _, row in result.iterrows():
        print(f"  Type {int(row.type_id):2d}: {row.display_name}")
    n_unique = result["display_name"].nunique()
    print(f"\n{n_unique}/{len(result)} unique names.")
    if n_unique < len(result):
        dupes = result[result["display_name"].duplicated(keep=False)]
        print("\nDUPLICATES:")
        print(dupes.to_string())

    # Step 2: name super-types
    print("\n--- Super-type names ---")
    super_result = name_super_types(fine_type_names=result)
    for _, row in super_result.iterrows():
        print(f"  Super {int(row.super_type_id)}: {row.display_name}")


if __name__ == "__main__":
    main()
