"""Z-score computation, threshold logic, and name disambiguation for type naming.

Provides shared utilities consumed by both name_fine_types and name_super_types:

  - compute_zscores()               — national population-weighted z-scores
  - _get_label()                    — resolve a vocab entry to a label given a z-score
  - _top_tokens()                   — extract the top N non-redundant tokens for a type
  - _best_split_labels()            — find the highest-variance feature that splits a group
  - _apply_collision_free_replace() — attempt collision-safe last-token replacement
  - _disambiguate()                 — deduplicate type names via variance-maximizing splits

Note: electoral lean helpers (_compute_type_lean, _lean_label) live in
name_fine_types.py because they read shift data and are fine-type-specific.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.description.naming_vocab import (
    _DISAMBIG_VOCAB,
    _FAMILY,
    _MIN_ABS,
    _MIN_ABS_PER_LABEL,
)

# Ordinals used in several disambiguation passes
_DISAMBIG_ORDINALS = [
    "", " II", " III", " IV", " V", " VI", " VII", " VIII",
    " IX", " X", " XI", " XII", " XIII", " XIV", " XV", " XVI",
    " XVII", " XVIII", " XIX", " XX",
]


# ---------------------------------------------------------------------------
# Z-score computation
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


# ---------------------------------------------------------------------------
# Token extraction
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
    z_row:
        Series of z-scores indexed by feature name.
    vocab:
        Ordered list of (feature, threshold, positive_label, negative_label).
        Entries are scanned in order; the first entry in each feature family
        that fires wins (subsequent same-family entries are skipped).
    n:
        Maximum number of tokens to return.
    exclude:
        Labels to skip even if they would otherwise fire.
    raw_row:
        Raw (un-z-scored) values for the type.  Used to enforce minimum
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
        # Enforce minimum absolute value for race/ethnicity features
        if feat in _MIN_ABS and raw_row is not None and feat in raw_row.index:
            if float(raw_row[feat]) < _MIN_ABS[feat]:
                continue
        # Enforce per-label absolute minimums (e.g., "Majority-Black" requires >50% Black)
        if raw_row is not None and feat in raw_row.index:
            per_label_min = _MIN_ABS_PER_LABEL.get((feat, label))
            if per_label_min is not None and float(raw_row[feat]) < per_label_min:
                continue
        tokens.append(label)
        seen_families.add(fam)
        if len(tokens) >= n:
            break

    return tokens


# ---------------------------------------------------------------------------
# Electoral lean helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Disambiguation helpers
# ---------------------------------------------------------------------------


def _lean_label_token(lean: float) -> str:
    """Convert a mean Dem shift to a human-readable lean label.

    Used in Pass 2 disambiguation when demographics can't distinguish types.
    The thresholds (±0.01) are chosen to be noise-tolerant — shifts within
    ±1 percentage point are treated as swing.
    """
    if lean > 0.01:
        return "Blue-Shift"
    elif lean < -0.01:
        return "Red-Shift"
    else:
        return "Swing"


def _get_z_from_df(
    tid: int,
    feat: str,
    z_df: pd.DataFrame,
    feat_cols: list[str],
) -> float | None:
    """Return national z-score for a type/feature pair (None if unavailable)."""
    z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
    if z_sub.empty or feat not in z_sub.columns:
        return None
    return float(z_sub.iloc[0][feat])


def _assign_direction_labels(
    tids: list[int],
    z_vals: dict[int, float],
    pos: str,
    neg: str,
    existing_tokens: set[str],
) -> dict[int, str]:
    """Assign direction labels (pos/neg) to each type based on position vs median z.

    Types above the group median get the positive label; below get the negative.
    Returns an empty string for types when neither label is available.
    """
    median_z = float(np.median(list(z_vals.values())))
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
    return labels


def _best_split_labels(
    tids: list[int],
    existing_name: str,
    z_df: pd.DataFrame,
    feat_cols: list[str],
) -> dict[int, str]:
    """Find the highest-variance feature that splits a duplicate group.

    Scans _DISAMBIG_VOCAB and returns per-type labels for the feature that
    maximizes within-group z-score variance while producing at least 2 distinct
    label outcomes.  Returns an empty dict if no suitable feature is found.
    """
    existing_tokens = set(existing_name.split())
    best_var = -1.0
    best_labels: dict[int, str] = {}

    for feat, _thresh, pos, neg in _DISAMBIG_VOCAB:
        if not pos and not neg:
            continue
        z_vals: dict[int, float] = {}
        for tid in tids:
            z = _get_z_from_df(tid, feat, z_df, feat_cols)
            if z is not None:
                z_vals[tid] = z
        if len(z_vals) < len(tids):
            continue

        var = float(np.var(list(z_vals.values())))
        if var <= best_var:
            continue

        labels = _assign_direction_labels(tids, z_vals, pos, neg, existing_tokens)
        non_empty = [lbl for lbl in labels.values() if lbl]
        if len(set(non_empty)) >= 2:
            best_var = var
            best_labels = labels

    return best_labels


def _apply_collision_free_replace(
    tids: list[int],
    name_prefix: str,
    existing_tokens: set[str],
    final: dict[int, str],
    z_df: pd.DataFrame,
    feat_cols: list[str],
) -> bool:
    """Try replacing the last word of a 4-word name with a collision-free disambiguator.

    Iterates all _DISAMBIG_VOCAB features in variance-descending order, proposing
    ``name_prefix + label`` for each type.  Applies the first candidate set that
    produces no collisions with existing names.

    Returns True if any replacement was applied, False otherwise.
    """
    all_current_names = set(final.values()) - {final[t] for t in tids}
    candidates: list[tuple[float, dict[int, str]]] = []

    for feat, _t2, pos2, neg2 in _DISAMBIG_VOCAB:
        if not pos2 and not neg2:
            continue
        z_vals: dict[int, float] = {}
        for tid in tids:
            z = _get_z_from_df(tid, feat, z_df, feat_cols)
            if z is not None:
                z_vals[tid] = z
        if len(z_vals) < len(tids):
            continue

        var2 = float(np.var(list(z_vals.values())))
        lbls = _assign_direction_labels(tids, z_vals, pos2, neg2, existing_tokens)
        non_emp = [l for l in lbls.values() if l]
        if len(set(non_emp)) >= 2:
            candidates.append((var2, lbls))

    # Try highest-variance first; skip any that cause collisions with other names
    for _, lbls in sorted(candidates, key=lambda x: -x[0]):
        proposed: dict[int, str] = {}
        ok = True
        proposed_names: set[str] = set()
        for tid in tids:
            lbl = lbls.get(tid, "")
            if lbl:
                nn = f"{name_prefix} {lbl}"
                if nn in all_current_names or nn in proposed_names:
                    ok = False
                    break
                proposed[tid] = nn
                proposed_names.add(nn)
        if ok and proposed:
            for tid, nn in proposed.items():
                if nn != final[tid]:
                    final[tid] = nn
            return True
    return False


# ---------------------------------------------------------------------------
# Main disambiguation function
# ---------------------------------------------------------------------------


def _disambiguate(
    raw_names: dict[int, str],
    profiles: pd.DataFrame,
    z_df: pd.DataFrame,
    feat_cols: list[str],
    type_lean: dict[int, float] | None = None,
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

    Parameters
    ----------
    type_lean:
        Optional dict of type_id → mean Dem shift, used as Pass 2 lean
        disambiguation.  If None, Pass 2 is skipped.  Callers should provide
        this from _compute_type_lean() (defined in name_fine_types.py).
    """

    def _get_dupes(d: dict[int, str]) -> dict[str, list[int]]:
        counts = Counter(d.values())
        return {
            name: sorted(t for t, n in d.items() if n == name)
            for name, cnt in counts.items()
            if cnt > 1
        }

    def _pop_order(tids: list[int]) -> list[int]:
        """Return type IDs sorted by county count descending, then type_id ascending."""
        return (
            profiles[profiles["type_id"].isin(tids)]
            .sort_values(["n_counties", "type_id"], ascending=[False, True])["type_id"]
            .tolist()
        )

    def _is_frontier(tid: int) -> bool:
        """Return True if type has near-zero demographics (unincorporated frontier)."""
        row = profiles.loc[profiles["type_id"] == tid]
        if row.empty:
            return False
        r = row.iloc[0]
        n = int(r.get("n_counties", 99))
        income = float(r.get("median_hh_income", 99999))
        age = float(r.get("median_age", 99))
        return n <= 3 and (income < 5_000 or age < 5)

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
            if len(name.split()) >= 4:
                continue  # word budget exhausted for this name
            labels = _best_split_labels(tids, name, z_df, feat_cols)
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
            changed = _apply_collision_free_replace(
                tids, name_prefix, existing_tokens, final, z_df, feat_cols
            )
            if changed:
                any_changed = True
        if not any_changed:
            break

    # Pass 2 — political lean disambiguation (last-resort before ordinals)
    # Types defined by electoral behavior can legitimately be distinguished
    # by their mean partisan shift direction.  Caller passes type_lean dict.
    if type_lean:
        for name, tids in _get_dupes(final).items():
            if len(name.split()) >= 4:
                continue
            lean_labels = {tid: _lean_label_token(type_lean.get(tid, 0.0)) for tid in tids}
            if len(set(lean_labels.values())) >= 2:
                for tid in tids:
                    final[tid] = f"{name} {lean_labels[tid]}"

    # Pass 2.5 — geographic frontier detection for near-empty types.
    # Some types consist entirely of Alaska Census Areas or similar sparsely-populated
    # geographic entities with near-zero demographic data.  These types cannot be
    # distinguished by vocabulary z-scores, so they fall through to ordinals unless
    # we catch them first and assign geographic labels.
    #
    # Detection criteria: n_counties <= 3 AND (median_hh_income < 5_000 OR median_age < 5)
    # These thresholds safely identify types whose demographic averages are driven by
    # unincorporated land areas with essentially no measured population.
    frontier_counter = [0]  # use list for closure mutability

    for name, tids in list(_get_dupes(final).items()):
        frontier_tids = [t for t in tids if _is_frontier(t)]
        if not frontier_tids:
            continue
        # Rename frontier types that still share a name with others
        for tid in [t for t in frontier_tids if final[t] == name]:
            frontier_counter[0] += 1
            ordinal = _DISAMBIG_ORDINALS[frontier_counter[0]].strip() or "I"
            final[tid] = f"Alaska Census Area {ordinal}"
        # If ALL siblings are frontier, rename the first one to "Alaska Frontier"
        if not [t for t in tids if t not in frontier_tids]:
            first = _pop_order(tids)[0]
            if _is_frontier(first):
                frontier_counter[0] += 1
                final[first] = "Alaska Frontier"

    # Pass 3 — ordinal suffixes for any remaining duplicates
    for name, tids in _get_dupes(final).items():
        for i, tid in enumerate(_pop_order(tids)):
            if i == 0:
                continue  # largest group keeps the name
            suffix = _DISAMBIG_ORDINALS[min(i, len(_DISAMBIG_ORDINALS) - 1)]
            final[tid] = f"{name}{suffix}"

    return final
