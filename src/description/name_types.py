"""Generate descriptive names for the 43 fine electoral types.

Names are derived deterministically from demographic z-scores relative to
population-weighted means.  No LLM calls.

Algorithm overview
------------------
1. Determine each type's dominant state from county assignments.
2. Compute population-weighted z-scores (within-state) for demographic features.
3. For each type, scan an ordered vocabulary and collect 2 descriptive tokens.
4. Assemble a 3-word name: "STATE Descriptor1 Descriptor2".
5. Disambiguate duplicates by adding a 4th token from extended vocab.
6. Any remaining duplicates get ordinal suffixes.

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

_FIPS_TO_STATE = {1: "AL", 12: "FL", 13: "GA"}

# Super-type short labels for context
_SUPER_LABELS: dict[int, str] = {
    0: "Conservative",
    1: "Mixed",
    2: "Professional",
    3: "Diverse",
    4: "Coastal",
}

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
# Uses GLOBAL z-scores to reflect genuine distinctiveness.
# ---------------------------------------------------------------------------
_VOCAB: list[tuple[str, float, str, str]] = [
    # Rare racial/ethnic composition (with absolute minimums enforced)
    ("pct_black",               Z_HIGH, "Black-Belt",   ""),
    ("pct_hispanic",            Z_HIGH, "Hispanic",     ""),
    ("pct_asian",               Z_HIGH, "Asian",        ""),
    ("pct_black",               Z_MOD,  "Black-Belt",   ""),
    ("pct_hispanic",            Z_MOD,  "Hispanic",     ""),

    # Urbanicity / density
    ("log_pop_density",         Z_HIGH, "Urban",        "Deep-Rural"),
    ("log_pop_density",         Z_MOD,  "Suburban",     "Rural"),

    # Education / professional class
    ("pct_bachelors_plus",      Z_HIGH, "College",      ""),
    ("pct_graduate",            Z_HIGH, "Grad-Degree",  ""),
    ("pct_management",          Z_HIGH, "Professional", ""),
    ("pct_bachelors_plus",      Z_MOD,  "Educated",     "Working-Class"),

    # Income
    ("median_hh_income",        Z_HIGH, "Affluent",     "Low-Income"),
    ("avg_inflow_income",       Z_HIGH, "Wealth-Magnet", ""),
    ("median_hh_income",        Z_MOD,  "Mid-Income",   "Modest-Income"),

    # Religion — differentiate evangelical, Catholic, mainline, secular
    ("evangelical_share",       Z_HIGH, "Evangelical",  ""),
    ("catholic_share",          Z_HIGH, "Catholic",     ""),
    ("mainline_share",          Z_HIGH, "Mainline",     ""),
    ("black_protestant_share",  Z_HIGH, "Bk-Protestant", ""),
    ("religious_adherence_rate", Z_HIGH, "Devout",      "Unchurched"),
    ("evangelical_share",       Z_MOD,  "Evangelical",  "Secular"),
    ("catholic_share",          Z_MOD,  "Catholic",     ""),
    ("religious_adherence_rate", Z_MOD,  "Churched",    "Unchurched"),

    # Age
    ("median_age",              Z_HIGH, "Retiree",      "Young"),
    ("median_age",              Z_MOD,  "Older",        "Younger"),

    # Migration / growth
    ("net_migration_rate",      Z_HIGH, "High-Growth",  "Declining"),
    ("inflow_outflow_ratio",    Z_HIGH, "Inflow",       ""),
    ("migration_diversity",     Z_HIGH, "Cosmopolitan", ""),
    ("net_migration_rate",      Z_MOD,  "Growing",      "Shrinking"),

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


def _get_type_state(
    type_id: int,
    county_assignments: pd.DataFrame | None,
) -> str:
    """Determine the dominant state for a type from county FIPS codes."""
    if county_assignments is None:
        return ""
    subset = county_assignments[county_assignments["dominant_type"] == type_id]
    if subset.empty:
        return ""
    # Extract state FIPS from county FIPS (first 1-2 digits)
    state_counts: Counter[str] = Counter()
    for fips in subset["county_fips"]:
        fips_str = str(int(fips)).zfill(5)
        state_fips = int(fips_str[:2])
        state_name = _FIPS_TO_STATE.get(state_fips, "")
        if state_name:
            state_counts[state_name] += 1
    if not state_counts:
        return ""
    return state_counts.most_common(1)[0][0]


def name_types(
    profiles: pd.DataFrame | None = None,
    county_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate descriptive display names for all fine electoral types.

    Parameters
    ----------
    profiles:
        type_profiles DataFrame (43 rows × demographic columns).
        If None, loaded from ``data/communities/type_profiles.parquet``.
    county_assignments:
        county_type_assignments_full DataFrame; used to map each fine type
        to its super-type and dominant state.  If None, loaded from disk.

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

    # ---- Compute global z-scores --------------------------------------------
    unique_features: list[str] = []
    seen_f: set[str] = set()
    for feat, *_ in _VOCAB:
        if feat not in seen_f:
            unique_features.append(feat)
            seen_f.add(feat)

    z_df = compute_zscores(profiles, unique_features, weight_col="pop_total")
    feat_cols = [c for c in z_df.columns if c != "type_id"]

    # ---- Determine each type's state and super-type -------------------------
    type_states: dict[int, str] = {}
    type_to_super: dict[int, int] = {}
    for _, row in profiles.iterrows():
        tid = int(row["type_id"])
        type_states[tid] = _get_type_state(tid, county_assignments)

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

    # ---- Compute within-state z-scores (for disambiguation only) ------------
    state_z_dfs: dict[str, pd.DataFrame] = {}
    for state in ["FL", "GA", "AL"]:
        state_tids = [t for t, s in type_states.items() if s == state]
        if not state_tids:
            continue
        state_profiles = profiles[profiles["type_id"].isin(state_tids)].copy()
        if len(state_profiles) > 1:
            state_z_dfs[state] = compute_zscores(
                state_profiles, unique_features, weight_col="pop_total"
            )

    # ---- First-pass names: STATE + 2 demographic tokens --------------------
    # Uses within-state z-scores for primary tokens (with min absolute
    # thresholds to prevent noise amplification on race features).
    raw_names: dict[int, str] = {}
    for _, row in profiles.iterrows():
        tid = int(row["type_id"])
        state = type_states.get(tid, "")

        # Prefer within-state z-scores for better differentiation
        z_source = state_z_dfs.get(state, z_df)
        z_sub = z_source.loc[z_source["type_id"] == tid, feat_cols]
        if z_sub.empty:
            z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
        if z_sub.empty:
            raw_names[tid] = f"Type {tid}"
            continue

        z_row = z_sub.iloc[0]
        raw_row = row  # for absolute threshold checks
        tokens = _top_tokens(z_row, _VOCAB, n=2, raw_row=raw_row)

        # If fewer than 2 tokens, use super-type context
        super_id = type_to_super.get(tid, -1)
        super_label = _SUPER_LABELS.get(super_id, "")

        if state:
            if len(tokens) >= 2:
                raw_names[tid] = f"{state} {tokens[0]} {tokens[1]}"
            elif len(tokens) == 1:
                ctx = super_label or "Mixed"
                raw_names[tid] = f"{state} {tokens[0]} {ctx}"
            else:
                raw_names[tid] = f"{state} {super_label}" if super_label else f"{state} Mixed"
        else:
            raw_names[tid] = " ".join(tokens) if tokens else f"Type {tid}"

    # ---- Disambiguation -----------------------------------------------------
    display_names = _disambiguate(
        raw_names, profiles, z_df, state_z_dfs, type_states, feat_cols
    )

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


# ---------------------------------------------------------------------------
# Disambiguation
# ---------------------------------------------------------------------------

# Extended vocab for 4th-token disambiguation — different features from primary
_DISAMBIG_VOCAB: list[tuple[str, float, str, str]] = [
    ("median_age",              Z_LOW,  "Older",        "Younger"),
    ("evangelical_share",       Z_LOW,  "Evangelical",  "Secular"),
    ("catholic_share",          Z_LOW,  "Catholic",     ""),
    ("mainline_share",          Z_LOW,  "Mainline",     ""),
    ("black_protestant_share",  Z_LOW,  "Bk-Protestant", ""),
    ("religious_adherence_rate", Z_LOW, "Devout",       "Unchurched"),
    ("net_migration_rate",      Z_LOW,  "Growing",      "Shrinking"),
    ("migration_diversity",     Z_LOW,  "Cosmopolitan", "Insular"),
    ("avg_inflow_income",       Z_LOW,  "Wealth-Magnet", ""),
    ("pct_owner_occupied",      Z_LOW,  "Homeowner",    "Renter"),
    ("pct_white_nh",            Z_LOW,  "White",        ""),
    ("pct_car",                 Z_LOW,  "Auto-Commute", ""),
    ("pct_wfh",                 Z_LOW,  "Remote-Work",  ""),
    ("land_area_sq_mi",         Z_LOW,  "Sprawling",    "Compact"),
    ("congregations_per_1000",  Z_LOW,  "Many-Churches", ""),
    ("pct_management",          Z_LOW,  "White-Collar", "Blue-Collar"),
    ("pct_transit",             Z_LOW,  "Transit",      ""),
    ("inflow_outflow_ratio",    Z_LOW,  "Inflow",       "Outflow"),
    ("median_hh_income",        Z_LOW,  "Higher-Inc",   "Lower-Inc"),
    ("pct_bachelors_plus",      Z_LOW,  "More-Edu",     "Less-Edu"),
    ("log_pop_density",         Z_LOW,  "Denser",       "Sparser"),
    ("pop_total",               Z_LOW,  "Large",        "Small"),
    ("n_counties",              Z_LOW,  "Broad",        "Concentrated"),
    ("pct_black",               Z_LOW,  "Blacker",      "Whiter"),
    ("pct_hispanic",            Z_LOW,  "More-Hisp",    "Less-Hisp"),
]


def _disambiguate(
    raw_names: dict[int, str],
    profiles: pd.DataFrame,
    z_df: pd.DataFrame,
    state_z_dfs: dict[str, pd.DataFrame],
    type_states: dict[int, str],
    feat_cols: list[str],
) -> dict[int, str]:
    """Ensure every type gets a unique display name (max 4 words).

    Strategy:
    1. For each duplicate group, find the feature that best splits the
       group (maximizes variance of z-scores within the group).
    2. Assign above/below-median labels from that feature.
    3. Any remaining duplicates get ordinal suffixes.
    """
    _ORDINALS = ["", " II", " III", " IV", " V", " VI", " VII", " VIII"]

    def _get_dupes(d: dict[int, str]) -> dict[str, list[int]]:
        counts = Counter(d.values())
        return {
            name: sorted(t for t, n in d.items() if n == name)
            for name, cnt in counts.items()
            if cnt > 1
        }

    def _splitting_token(tids: list[int], existing_name: str) -> dict[int, str]:
        """Find the feature that best splits a group of types and return
        per-type labels.
        """
        existing_tokens = set(existing_name.split())
        best_var = -1.0
        best_labels: dict[int, str] = {}

        for feat, thresh, pos, neg in _DISAMBIG_VOCAB:
            if not pos and not neg:
                continue
            # Get z-values for all types in the group
            z_vals: dict[int, float] = {}
            for tid in tids:
                state = type_states.get(tid, "")
                z_source = state_z_dfs.get(state, z_df)
                z_sub = z_source.loc[z_source["type_id"] == tid, feat_cols]
                if z_sub.empty:
                    z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
                if not z_sub.empty and feat in z_sub.columns:
                    z_vals[tid] = float(z_sub.iloc[0][feat])
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
                if z >= median_z and pos and pos not in existing_tokens:
                    labels[tid] = pos
                elif z < median_z and neg and neg not in existing_tokens:
                    labels[tid] = neg
                elif pos and pos not in existing_tokens:
                    labels[tid] = pos
                elif neg and neg not in existing_tokens:
                    labels[tid] = neg
                else:
                    labels[tid] = ""

            # Only accept if it actually differentiates (at least 2 distinct labels)
            non_empty = [l for l in labels.values() if l]
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

    # Pass 1 — split duplicate groups using the most differentiating feature
    for _pass in range(3):  # iterate to resolve cascading duplicates
        dupes = _get_dupes(final)
        if not dupes:
            break
        for name, tids in dupes.items():
            n_words = len(name.split())
            if n_words >= 4:
                continue  # can't add more words
            labels = _splitting_token(tids, name)
            for tid in tids:
                label = labels.get(tid, "")
                if label:
                    final[tid] = f"{name} {label}"

    # Cap at 4 words
    for tid in final:
        parts = final[tid].split()
        if len(parts) > 4:
            final[tid] = " ".join(parts[:4])

    # Pass 2 — ordinal suffixes for any remaining duplicates
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


if __name__ == "__main__":
    main()
