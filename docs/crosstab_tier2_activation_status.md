# Tier 2 Crosstab Activation Status

**Date verified:** 2026-05-07
**Verified by:** agentic-box research-assistant (task queued by curator)
**Commit under review:** `d3c10c1` — "Preserve poll enrichment metadata through weighting"

## Conclusion: Tier 2 IS LIVE

### Delta result

Script run: `uv run python scripts/experiments/compare_xt_impact_v2.py` (2026-05-07)

| Comparison | n (races w/ xt_) | mean \|Δ\| | max \|Δ\| | median \|Δ\| |
|---|---|---|---|---|
| Live (enriched) vs stripped | 19 | **1.769 pp** | 4.818 pp | 1.466 pp |
| Tier 2 bypass vs stripped | 19 | 1.461 pp | 3.197 pp | 1.384 pp |

Pre-fix baseline (2026-04-24, before d3c10c1): mean |Δ| = 0.020 pp (noise floor).

Criterion: >1.0 pp matching bypass-run magnitude = Tier 2 live. **Both columns well above threshold.**

### Code path verification

**`forecast_engine.py:prepare_polls()` lines 62-90:**
- `core_keys` excludes all `xt_*` and `methodology` keys.
- `metadata={k: v for k, v in p.items() if k not in core_keys}` captures every non-core key — including all `xt_*` columns — into `PollObservation.metadata`.

**`forecast_engine.py` lines 114-128:**
- `d = dict(obs.metadata)` restores the full enrichment dict including all `xt_*` keys.
- Topline fields (dem_share, n_sample, state, date, pollster, notes, geo_level) are updated on top.
- xt_* keys survive into downstream W-vector construction.

**`poll_enrichment.py:build_W_poll()` lines 289-295:**
- `if poll_crosstabs is not None: return build_W_from_crosstabs(...)` — Tier 2 takes priority.
- xt_* keys from poll dicts are converted upstream (in the predict pipeline) to `poll_crosstabs` list before reaching this function.

### Prior open question (now closed)

From `knowledge/projects/wethervane.md` (2026-04-24 entry):
> "Tier 2 crosstabs are silently disabled in the live forecast."

**This is resolved.** Commit `d3c10c1` fixed the metadata-stripping bug in `prepare_polls()`.
The live forecast now applies Tier 2 crosstab W-vectors for all 19 races that have xt_* data.
