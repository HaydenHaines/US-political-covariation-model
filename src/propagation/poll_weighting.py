"""Poll weighting: time decay, pollster quality, house effect correction, and aggregation.

This module is a backward-compatible facade that re-exports all public
symbols from the focused sub-modules introduced in #110:

  - ``poll_quality``  -- Silver Bulletin ratings and 538 grade fallback
  - ``house_effects`` -- house effect (partisan bias) correction
  - ``poll_decay``    -- time decay and pre-primary discount
  - ``poll_pipeline`` -- orchestration, aggregation, and CSV loading

All existing imports from ``src.propagation.poll_weighting`` continue to work.

Usage:
  from src.propagation.poll_weighting import apply_all_weights, aggregate_polls

  weighted = apply_all_weights(polls, notes, reference_date="2020-11-03")
  combined_share, combined_n = aggregate_polls(weighted)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export from poll_quality
# ---------------------------------------------------------------------------
from src.propagation.poll_quality import (  # noqa: F401
    _DEFAULT_GRADE_MULTIPLIERS,
    _NO_GRADE_MULTIPLIER,
    _SB_MAX_MULTIPLIER,
    _SB_MIN_MULTIPLIER,
    _get_sb_quality,
    _numeric_grade_to_letter,
    _sb_score_to_multiplier,
    apply_pollster_quality,
    extract_grade_from_notes,
    grade_to_multiplier,
    reset_sb_cache,
)

# ---------------------------------------------------------------------------
# Re-export from house_effects
# ---------------------------------------------------------------------------
from src.propagation.house_effects import (  # noqa: F401
    _HE_DEM_SHARE_MAX,
    _HE_DEM_SHARE_MIN,
    apply_house_effect_correction,
    reset_house_effect_cache,
)

# ---------------------------------------------------------------------------
# Re-export from poll_decay
# ---------------------------------------------------------------------------
from src.propagation.poll_decay import (  # noqa: F401
    _PRE_PRIMARY_DISCOUNT,
    apply_primary_discount,
    apply_time_decay,
    election_day_for_cycle,
)

# ---------------------------------------------------------------------------
# Re-export from poll_pipeline
# ---------------------------------------------------------------------------
from src.propagation.poll_pipeline import (  # noqa: F401
    aggregate_polls,
    apply_all_weights,
    load_poll_notes,
    load_polls_with_notes,
)
