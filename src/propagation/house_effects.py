"""House effect (partisan bias) correction for polls.

Corrects each poll's dem_share for systematic over/under-estimation
of Democrats by each pollster.  Uses a priority cascade:
  1. Silver Bulletin "House Effect" column (mean-reverted, percentage points)
  2. 538 ``bias_ppm`` column (same sign convention)
  3. 0.0 (no correction) for unknown pollsters

Positive house_effect means the pollster overestimates Democrats,
so subtracting it moves the share toward the true value.
"""

from __future__ import annotations

import logging
from copy import copy

from src.propagation.propagate_polls import PollObservation

log = logging.getLogger(__name__)

# Clamp corrected dem_share to this range so it stays a valid proportion.
_HE_DEM_SHARE_MIN: float = 0.01
_HE_DEM_SHARE_MAX: float = 0.99

# Cached house effect lookups: None = not yet loaded.
_SB_HOUSE_EFFECTS: dict[str, float] | None = None
_538_BIAS: dict[str, float] | None = None


def _ensure_house_effect_caches() -> None:
    """Populate module-level house effect caches on first call."""
    global _SB_HOUSE_EFFECTS, _538_BIAS

    if _SB_HOUSE_EFFECTS is None:
        try:
            from src.assembly.silver_bulletin_ratings import load_pollster_house_effects
            _SB_HOUSE_EFFECTS = load_pollster_house_effects()
            log.debug("Loaded %d Silver Bulletin house effects", len(_SB_HOUSE_EFFECTS))
        except FileNotFoundError:
            log.debug("Silver Bulletin XLSX not found; SB house effects unavailable")
            _SB_HOUSE_EFFECTS = {}
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load Silver Bulletin house effects: %s", exc)
            _SB_HOUSE_EFFECTS = {}

    if _538_BIAS is None:
        try:
            from src.assembly.silver_bulletin_ratings import load_538_bias
            _538_BIAS = load_538_bias()
            log.debug("Loaded %d 538 bias estimates", len(_538_BIAS))
        except FileNotFoundError:
            log.debug("538 pollster-ratings CSV not found; 538 bias unavailable")
            _538_BIAS = {}
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load 538 bias estimates: %s", exc)
            _538_BIAS = {}


def _lookup_house_effect(
    pollster_name: str,
    sb_house_effects: dict[str, float],
    bias_538: dict[str, float],
) -> tuple[float, str]:
    """Return (house_effect_pp, source) for a pollster using priority order.

    Priority: Silver Bulletin exact -> Silver Bulletin fuzzy -> 538 exact -> 0.0.
    Returns the source string for logging ('sb_exact', 'sb_fuzzy', '538', 'none').
    """
    # 1. Silver Bulletin — delegate fuzzy matching to the silver_bulletin module
    if sb_house_effects:
        try:
            from src.assembly.silver_bulletin_ratings import _normalize, _name_similarity
        except ImportError:  # pragma: no cover
            pass
        else:
            # Exact match
            if pollster_name in sb_house_effects:
                return sb_house_effects[pollster_name], "sb_exact"

            # Normalised / fuzzy match
            norm_query = _normalize(pollster_name)
            norm_sb = {_normalize(k): v for k, v in sb_house_effects.items()}
            if norm_query in norm_sb:
                return norm_sb[norm_query], "sb_exact"

            # Best Jaccard fuzzy match (threshold 0.4 matches get_pollster_quality)
            _FUZZY_THRESHOLD = 0.4
            best_val: float | None = None
            best_sim = 0.0
            for norm_key, val in norm_sb.items():
                sim = _name_similarity(norm_query, norm_key)
                if sim > best_sim:
                    best_sim = sim
                    best_val = val
            if best_sim >= _FUZZY_THRESHOLD and best_val is not None:
                return best_val, "sb_fuzzy"

    # 2. 538 bias — exact name match only (no fuzzy; 538 names differ more)
    if bias_538 and pollster_name in bias_538:
        return bias_538[pollster_name], "538"

    return 0.0, "none"


def apply_house_effect_correction(
    polls: list[PollObservation],
    sb_house_effects: dict[str, float] | None = None,
    bias_538: dict[str, float] | None = None,
) -> list[PollObservation]:
    """Correct each poll's dem_share for pollster partisan bias (house effects).

    For each poll, looks up the pollster's house effect using priority order:
      1. Silver Bulletin "House Effect" (mean-reverted, pp)
      2. 538 ``bias_ppm``
      3. 0.0 (no correction)

    Adjustment: corrected_dem_share = dem_share - house_effect / 100
      Positive house_effect means pollster overestimates Democrats, so subtracting
      moves the share toward the true value.

    Result is clamped to [_HE_DEM_SHARE_MIN, _HE_DEM_SHARE_MAX] to keep it
    a valid proportion.  The original ``dem_share`` is preserved on the returned
    PollObservation; ``dem_share`` is updated to the corrected value.

    Parameters
    ----------
    polls:
        List of PollObservation objects to correct.
    sb_house_effects:
        Optional pre-loaded Silver Bulletin house effects dict.  If None,
        loads from the module-level cache (populated from XLSX on first call).
    bias_538:
        Optional pre-loaded 538 bias dict.  If None, loads from the module-level
        cache (populated from CSV on first call).

    Returns
    -------
    list[PollObservation]
        New PollObservation copies with dem_share corrected for house effects.
        The uncorrected value is accessible via poll.raw_dem_share if needed
        (the field is not currently on PollObservation; original logged here).
    """
    _ensure_house_effect_caches()
    he_dict = sb_house_effects if sb_house_effects is not None else (_SB_HOUSE_EFFECTS or {})
    b538_dict = bias_538 if bias_538 is not None else (_538_BIAS or {})

    result: list[PollObservation] = []
    for poll in polls:
        new_poll = copy(poll)
        if not poll.pollster:
            result.append(new_poll)
            continue

        house_effect_pp, source = _lookup_house_effect(poll.pollster, he_dict, b538_dict)

        if source == "none":
            log.debug("House effect: pollster %r not found, no correction applied", poll.pollster)
        else:
            log.debug(
                "House effect: pollster %r -> %.3f pp (%s); dem_share %.4f -> %.4f",
                poll.pollster,
                house_effect_pp,
                source,
                poll.dem_share,
                poll.dem_share - house_effect_pp / 100.0,
            )

        correction = house_effect_pp / 100.0
        corrected = poll.dem_share - correction
        corrected = max(_HE_DEM_SHARE_MIN, min(_HE_DEM_SHARE_MAX, corrected))
        new_poll.dem_share = corrected
        result.append(new_poll)

    return result


def reset_house_effect_cache() -> None:
    """Reset house effect caches (useful in tests that inject custom data)."""
    global _SB_HOUSE_EFFECTS, _538_BIAS
    _SB_HOUSE_EFFECTS = None
    _538_BIAS = None
