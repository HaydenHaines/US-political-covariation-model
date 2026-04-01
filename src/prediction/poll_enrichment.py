"""Tiered W vector construction for poll-specific type composition.

Three tiers, each with a defined off-ramp:
  Tier 1: Raw unweighted sample demographics -> direct type mapping
  Tier 2: Weighted topline + crosstabs -> per-group type mapping
  Tier 3: Weighted topline only -> LV/RV screen + non-weighted dimensions

See docs/superpowers/specs/2026-03-29-rich-poll-ingestion-design.md
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.prediction.propensity_model import compute_propensity_scores, load_config


_LV_PATTERN = re.compile(r"\bLV\b", re.IGNORECASE)
_RV_PATTERN = re.compile(r"\bRV\b", re.IGNORECASE)

# Controls how steeply similarity decays with normalized demographic distance.
# Higher values make the W vector more peaked (more weight on the closest type);
# lower values spread weight more evenly. 5.0 was chosen empirically: at distance=1.0
# (one normalized unit apart), similarity drops to 1/6 ≈ 0.17.
_SIMILARITY_SHARPNESS = 5.0


def parse_methodology(notes: str | None) -> str:
    """Extract LV/RV methodology from poll notes string. Returns "LV", "RV", or ""."""
    if not notes:
        return ""
    if _LV_PATTERN.search(notes):
        return "LV"
    if _RV_PATTERN.search(notes):
        return "RV"
    return ""


def build_W_with_adjustments(
    poll: dict,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
    w_vector_mode: str = "core",
    config: dict | None = None,
) -> np.ndarray:
    """Tier 3: Adjust state-level W for LV/RV screen and non-weighted dimensions.
    With no methodology info, returns state_type_weights unchanged."""
    if config is None:
        config = load_config()

    J = len(state_type_weights)
    W = state_type_weights.copy().astype(float)

    methodology = poll.get("methodology", "")
    if not methodology:
        methodology = parse_methodology(poll.get("notes", ""))

    # Adjustment 1: LV/RV type screen
    if methodology in ("LV", "RV"):
        propensity = compute_propensity_scores(type_profiles, config)
        if methodology == "LV":
            factor = config.get("lv_downweight_factor", 0.5)
        else:
            factor = config.get("rv_downweight_factor", 0.8)
        adjustment = factor + (1.0 - factor) * propensity
        W = W * adjustment

    # Adjustment 2: Non-weighted dimensions
    if methodology:
        dims = config.get("w_vector_dimensions", {}).get(w_vector_mode, [])
        if dims:
            method_key = _infer_method_type(poll)
            reach = config.get("method_reach_profiles", {}).get(method_key, {})
            for dim in dims:
                shift = reach.get(f"{dim}_shift", 0.0)
                if shift != 0.0 and dim in type_profiles.columns:
                    col = type_profiles[dim].values.astype(float)
                    col_norm = col - col.mean()
                    col_range = col.max() - col.min()
                    if col_range > 0:
                        col_norm = col_norm / col_range
                    W = W * (1.0 + shift * col_norm)

    W_sum = W.sum()
    if W_sum > 0:
        W = W / W_sum
    else:
        W = np.ones(J) / J
    return W


def _infer_method_type(poll: dict) -> str:
    """Infer polling methodology type from poll notes for reach profile lookup.

    Returns one of: online_panel, phone_ivr, phone_live, sms, mail, unknown.
    Order matters — check most specific signals first (e.g. IVR before generic phone).
    """
    notes = (poll.get("notes", "") or "").lower()
    if "online" in notes or "panel" in notes:
        return "online_panel"
    if "ivr" in notes or "automated" in notes or "robo" in notes:
        return "phone_ivr"
    # Check SMS/text before generic "phone" to avoid misclassifying text polls
    if "sms" in notes or "text" in notes:
        return "sms"
    if "mail" in notes or "postal" in notes:
        return "mail"
    if "live" in notes or "phone" in notes:
        return "phone_live"
    return "unknown"


def build_W_from_crosstabs(
    poll: dict,
    crosstabs: list[dict],
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
) -> list[dict]:
    """Tier 2: Map crosstab groups to types, return multiple observations.
    Returns list of {"W": np.ndarray, "y": float, "sigma": float} dicts."""
    J = len(state_type_weights)
    observations = []

    for xt in crosstabs:
        dem_share = xt.get("dem_share")
        pct_of_sample = xt.get("pct_of_sample", 0.0)
        n_sample = poll.get("n_sample", 600)

        if dem_share is None or pct_of_sample <= 0:
            continue

        sub_n = max(int(n_sample * pct_of_sample), 1)
        sigma = np.sqrt(dem_share * (1 - dem_share) / sub_n)

        group = xt.get("demographic_group", "")
        value = xt.get("group_value", "")
        W = _map_demographic_to_types(group, value, type_profiles, state_type_weights)

        observations.append({"W": W, "y": dem_share, "sigma": max(sigma, 1e-6)})

    if not observations:
        dem_share = poll["dem_share"]
        n_sample = poll.get("n_sample", 600)
        sigma = np.sqrt(dem_share * (1 - dem_share) / max(n_sample, 1))
        return [{"W": state_type_weights.copy(), "y": dem_share, "sigma": max(sigma, 1e-6)}]

    return observations


def _map_demographic_to_types(
    group: str, value: str,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
) -> np.ndarray:
    J = len(state_type_weights)
    demo_col_map = {
        ("race", "black"): "pct_black",
        ("race", "white"): "pct_white_nh",
        ("race", "hispanic"): "pct_hispanic",
        ("race", "asian"): "pct_asian",
        ("education", "college"): "pct_bachelors_plus",
        ("education", "college_plus"): "pct_bachelors_plus",
        ("religion", "evangelical"): "evangelical_share",
    }
    col = demo_col_map.get((group.lower(), value.lower()))
    if col is None or col not in type_profiles.columns:
        return state_type_weights.copy()

    type_vals = type_profiles[col].values.astype(float)
    W = state_type_weights * type_vals
    W_sum = W.sum()
    return W / W_sum if W_sum > 0 else np.ones(J) / J


def build_W_from_raw_sample(
    poll: dict,
    raw_demographics: dict[str, float],
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
) -> np.ndarray:
    """Tier 1: Map raw unweighted sample demographics directly to type weights."""
    J = len(state_type_weights)
    dims = [k for k in raw_demographics if k in type_profiles.columns]
    if not dims:
        return state_type_weights.copy()

    diffs_sq = np.zeros(J)
    for dim in dims:
        poll_val = raw_demographics[dim]
        type_vals = type_profiles[dim].values.astype(float)
        col_range = type_vals.max() - type_vals.min()
        if col_range > 0:
            diffs_sq += ((poll_val - type_vals) / col_range) ** 2

    distance = np.sqrt(diffs_sq / len(dims))
    similarity = 1.0 / (1.0 + distance * _SIMILARITY_SHARPNESS)

    W = similarity * state_type_weights
    W_sum = W.sum()
    return W / W_sum if W_sum > 0 else np.ones(J) / J


def build_W_poll(
    poll: dict,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
    poll_crosstabs: list[dict] | None = None,
    raw_sample_demographics: dict[str, float] | None = None,
    w_vector_mode: str = "core",
) -> np.ndarray | list[dict]:
    """Construct poll-specific W vector at the best available tier.

    Returns:
      - Tier 1 & 3: np.ndarray of shape (J,)
      - Tier 2: list of {"W": ndarray, "y": float, "sigma": float} dicts
    """
    if raw_sample_demographics is not None:
        return build_W_from_raw_sample(
            poll, raw_sample_demographics, type_profiles, state_type_weights,
        )
    if poll_crosstabs is not None:
        return build_W_from_crosstabs(
            poll, poll_crosstabs, type_profiles, state_type_weights,
        )
    return build_W_with_adjustments(
        poll, type_profiles, state_type_weights, w_vector_mode=w_vector_mode,
    )
