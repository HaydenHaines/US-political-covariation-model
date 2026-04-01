"""Super-type naming logic.

Generates human-readable names for grouped super-types.  Super-types aggregate
many fine types; their names are derived from the county-population-weighted
average demographic profile of their constituent fine types.

See name_fine_types.py for fine type naming.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.description.naming_scoring import _top_tokens, compute_zscores
from src.description.naming_vocab import _DISAMBIG_VOCAB, _SUPER_VOCAB

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
TYPE_PROFILES_PATH = _ROOT / "data" / "communities" / "type_profiles.parquet"
COUNTY_TYPE_ASSIGNMENTS_PATH = (
    _ROOT / "data" / "communities" / "county_type_assignments_full.parquet"
)
SUPER_TYPES_PATH = _ROOT / "data" / "communities" / "super_types.parquet"


def _disambiguate_super(
    raw: dict[int, str],
    agg_df: pd.DataFrame,
    z_df: pd.DataFrame,
    feat_cols: list[str],
) -> dict[int, str]:
    """Ensure super-type names are unique.

    One pass of variance-based disambiguation followed by ordinal fallback.
    Uses the same variance-maximization approach as fine-type disambiguation,
    but applied to the compressed z-scores of the super-type aggregate profiles.
    """
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


def _build_super_aggregate_profiles(
    profiles: pd.DataFrame,
    county_assignments: pd.DataFrame,
    super_ids: list[int],
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    """Build one aggregate demographic row per super-type.

    Each fine type within a super-type is weighted by county_count × pop_total,
    producing a county-population weighted average over the fine types.

    Returns (agg_df, type_id_lists) where type_id_lists maps super_id → sorted
    list of constituent fine type IDs.
    """
    agg_rows: list[dict] = []
    type_id_lists: dict[int, list[int]] = {}

    for sid in super_ids:
        st_counties = county_assignments[county_assignments["super_type"] == sid]
        type_counts = st_counties["dominant_type"].value_counts()
        type_ids_in = [t for t in type_counts.index if t in profiles["type_id"].values]
        type_id_lists[sid] = sorted(type_ids_in)

        subset = profiles[profiles["type_id"].isin(type_ids_in)].copy()
        county_w = subset["type_id"].map(type_counts).fillna(1.0)
        pop_w = subset["pop_total"].fillna(1.0)
        weights = (county_w * pop_w).values.astype(float)
        total_w = max(weights.sum(), 1.0)  # guard against zero-weight edge case

        row: dict = {"type_id": sid, "pop_total": float(weights.sum())}
        for col in numeric_cols:
            if col in subset.columns:
                vals = subset[col].fillna(0.0).values.astype(float)
                row[col] = float(np.dot(weights, vals) / total_w)
        agg_rows.append(row)

    return pd.DataFrame(agg_rows), type_id_lists


def _build_raw_super_names(
    agg_df: pd.DataFrame,
    z_super: pd.DataFrame,
    sfeat_cols: list[str],
    super_ids: list[int],
) -> dict[int, str]:
    """Generate a raw (possibly non-unique) name for each super-type from the vocab."""
    raw_super: dict[int, str] = {}
    for sid in super_ids:
        z_sub = z_super.loc[z_super["type_id"] == sid, sfeat_cols]
        raw_row = agg_df[agg_df["type_id"] == sid].iloc[0]
        if z_sub.empty:
            raw_super[sid] = f"Type-{sid}"
            continue
        tokens = _top_tokens(z_sub.iloc[0], _SUPER_VOCAB, n=2, raw_row=raw_row)
        if len(tokens) >= 2:
            raw_super[sid] = f"{tokens[0]} {tokens[1]}"
        elif len(tokens) == 1:
            raw_super[sid] = f"{tokens[0]} Coalition"
        else:
            raw_super[sid] = "Mixed Coalition"
    return raw_super


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
    numeric_cols = [
        c for c in profiles.columns
        if c not in ("type_id", "display_name") and profiles[c].dtype.kind in "fi"
    ]

    agg_df, type_id_lists = _build_super_aggregate_profiles(
        profiles, county_assignments, super_ids, numeric_cols
    )

    # Compute z-scores across super-types (national reference = all super-types)
    unique_sf = list(dict.fromkeys(e[0] for e in _SUPER_VOCAB))
    z_super = compute_zscores(agg_df, unique_sf, weight_col="pop_total")
    sfeat_cols = [c for c in z_super.columns if c != "type_id"]

    raw_super = _build_raw_super_names(agg_df, z_super, sfeat_cols, super_ids)
    final_super = _disambiguate_super(raw_super, agg_df, z_super, sfeat_cols)

    result = pd.DataFrame(
        [{"super_type_id": sid, "display_name": name}
         for sid, name in sorted(final_super.items())]
    )

    if save_to_disk and SUPER_TYPES_PATH.exists():
        result.to_parquet(SUPER_TYPES_PATH, index=False)
        log.info("Saved super-type display names to %s", SUPER_TYPES_PATH)

    for sid, name in sorted(final_super.items()):
        n_types = len(type_id_lists.get(sid, []))
        log.info(
            "Super-type %d → '%s'  (%d fine types: %s)",
            sid, name, n_types, type_id_lists.get(sid, [])
        )

    return result
