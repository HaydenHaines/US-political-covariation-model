"""Shared rating classification for all forecast endpoints.

Both the forecast and senate routers need to classify predictions into
rating labels (tossup, lean_d, likely_d, safe_d, etc.).  This module is
the single source of truth for the thresholds and conversion logic.

Two entry points cover the two calling conventions used across the codebase:

- ``margin_to_rating(margin)`` — margin = dem_share - 0.5 (signed float)
- ``dem_share_to_rating(dem_share)`` — raw Dem two-party share in [0, 1]
"""
from __future__ import annotations

# Rating classification thresholds (absolute margin from 0.5)
TOSSUP_MAX = 0.03    # |margin| < 3pp → tossup
LEAN_MAX = 0.08      # 3pp ≤ |margin| < 8pp → lean
LIKELY_MAX = 0.15    # 8pp ≤ |margin| < 15pp → likely
                     # |margin| ≥ 15pp → safe


def margin_to_rating(margin: float) -> str:
    """Convert signed Dem margin to a rating label.

    margin = state_pred - 0.5 (positive = Dem-favored, negative = GOP-favored).
    """
    abs_m = abs(margin)
    if abs_m < TOSSUP_MAX:
        return "tossup"
    direction = "_d" if margin > 0 else "_r"
    if abs_m < LEAN_MAX:
        return f"lean{direction}"
    if abs_m < LIKELY_MAX:
        return f"likely{direction}"
    return f"safe{direction}"


def dem_share_to_rating(dem_share: float) -> str:
    """Convert raw Dem two-party share to a rating label.

    Convenience wrapper: subtracts 0.5 and delegates to ``margin_to_rating``.
    This is the function formerly known as ``marginToRating`` in the forecast
    helpers — renamed to follow snake_case convention.
    """
    return margin_to_rating(dem_share - 0.5)
