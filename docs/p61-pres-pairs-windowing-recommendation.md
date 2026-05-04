# P6.1 — Pre-2000 PRES_PAIRS Windowing: Architect Recommendation

**Date:** 2026-05-04
**Author:** Software Architect (autonomous review)
**Companion doc:** `docs/p61-pre2000-county-returns-research.md`
**Code touchpoints:** `config/model.yaml`, `src/core/config.py`, `src/assembly/build_county_shifts_multiyear.py`, `src/validation/validate_county_holdout_multiyear.py`

---

## TL;DR

1. **Initial PRES_PAIRS scope:** **5 new pairs (post-1980, 1980→1984 through 1996→2000).** Adopt the post-Reagan-realignment window as the default. Doubles historical training signal (5→10 pres pairs) without introducing pre-realignment regime shift.
2. **Alaska pre-1972:** **No action required.** Algara pre-2000 data contains zero Alaska rows (empirically verified for 1948–2000). The research doc's concern is moot at the data layer.
3. **Validation split:** **Do not repurpose any pre-2000 pair as a validation split.** Their generative process differs from the holdout (2020→2024) and would yield misleading hyperparameter signal. Keep 2020→2024 as the sole holdout; if a true validation slot is later needed, carve it from 2016→2020 (within the modern-polarization regime).
4. **Implementation policy:** Have the developer fetch and persist all 13 pre-2000 parquets (sunk cost — files already exist in `data/assembled/`), but **edit `config/model.yaml: presidential_pairs:`** to enumerate only the 5 active pairs. Add the others back via a controlled experimental sweep, not the production config.
5. **Companion config change:** When the active pres-pair count grows beyond 5, **drop `types.presidential_weight` from 8.0 toward 4.0–5.0** to preserve the dim-balance the current weight was tuned for. Treat this as a hyperparameter to retune, not a constant to keep.

---

## Question framing

The task asks three questions. Resolve them in order:

### Q1 — How many of the 13 unlocked pairs to include?

The dominant constraint is **regime stationarity between training pairs and the holdout (2020→2024).** A pair (A→B) contributes useful signal to the model iff the *generative process* relating county-level demographics to D/R/turnout shifts in (A→B) resembles the process active in 2020→2024. Pairs from a structurally different era contribute noise (or worse: contradicts modern signal at high `presidential_weight`).

**Three eras of US presidential coalitions:**

| Era | Years | Coalition character |
|---|---|---|
| New Deal | 1932–1964 | Solid Democratic South, GOP urban North, ethnic-religious cleavages dominant |
| Realignment transition | 1968–1988 | Southern Realignment (D→R for white Southerners), Reagan Democrats, suburban realignment underway |
| Modern polarization | 1992–present | Education-polarization era; Sun Belt / Rust Belt patterns hardening; current diagonals stable |

The 2020→2024 holdout is firmly in the third era. Pre-1972 pairs (the New Deal coalition) contain a one-time structural break that runs **opposite** to modern dynamics in the South. Pairs spanning 1968–1988 contain the realignment transition itself — a non-stationary process by definition.

**Existing project evidence** (DECISIONS_LOG.md, 2026-03-19) already shows that extending the training window degrades short-run holdout r:

> "Multi-year holdout r=0.87 vs 3-cycle baseline r=0.98 (delta ~-0.11). Multi-year model is slightly weaker on short-run holdout: 30 dims average over 20 years of history, pulling communities toward older patterns."

Adding 13 pairs back to 1948 will amplify this drag. The project's stated value for adopting more pairs is "temporal depth and structural robustness" — but at some point, depth becomes counterproductive when the training pairs no longer share a generative process with the prediction target.

**Three approaches considered:**

**A. All 13 pairs (1948→2000).** Simple. Maximizes data quantity. **Reject.** Adds 39 dims (×2.6 inflation in pres dim count), pre-realignment signal contradicts modern dynamics, and `presidential_weight=8.0` would amplify the contradiction. High blast radius on holdout r without a pre-validated payoff.

**B. Two pairs (1992→1996, 1996→2000).** Most conservative. **Pass over as default but keep available.** Adds 6 dims; both pairs are within the modern-polarization era. Low risk but small data gain. Useful as a validation rung in a sweep.

**C. Five pairs (post-1980).** **Recommended default.** Adds 15 dims. The Reagan realignment is largely complete by 1984 in the South; 1980-onward pairs share the modern coalition structure (with some lingering Reagan-Democrat dynamics in the Midwest that resemble the 2016 Trump shift). Doubles historical training signal without crossing the realignment break. This is the highest-evidence window for "stationary with the holdout."

**D. Differential weighting (per-pair decay).** Conceptually attractive (downweight older pairs by `exp(-(2024-year)/τ)`). **Reject for now.** Introduces a new hyperparameter (`τ`), requires modifying the StandardScaler+presidential_weight pipeline, and adds tuning surface area without first establishing that any pre-2000 pair helps at all. Revisit if Approach C plus a sweep shows clear benefit beyond 5 pairs.

**E. Phased empirical sweep.** **Adopt as the operational plan around Approach C.** Start with 5 pairs (post-1980) in production. Run an experimental sweep (5 → 7 → 10 → 13 pres pairs) measuring holdout r on 2020→2024 at production K and J. Adopt the largest window that does not degrade holdout r below the current baseline (`holdout_r_minimum: 0.85`).

**Recommended default:** Approach C — **5 new pairs (post-1980)**. Companion experiment under Approach E to determine whether the window can be widened.

### Q2 — Alaska pre-1972 handling (exclude or remap)?

Empirically resolved. Spot-checked five Algara presidential parquets in `data/assembled/` (1948, 1968, 1972, 1996, 2000); every one returns `AK count = 0` for `county_fips.str.startswith("02")`. Algara/Amlani's pre-2000 presidential file does not include Alaska boroughs/census areas at all. There is nothing to exclude or remap.

**Recommendation:** No special-case code. Document this in the `fetch_algara_presidential.py` module docstring so the next maintainer doesn't re-investigate. The 2000+ MEDSL spine continues to carry Alaska; any inner join with pre-2000 shifts naturally produces NaN that the existing zero-fill path in `build_county_shifts_multiyear.py` handles.

### Q3 — Should any pre-2000 pair serve as a validation split?

**Recommendation: No.**

A validation split exists to tune hyperparameters without leaking holdout-pair signal. Useful only if it shares the holdout's generative process — otherwise tuning against it actively misguides the model.

- A pre-2000 pair (any era) has a **different** generative process than the 2020→2024 holdout. Tuning J, K, `presidential_weight`, and `pca_components` against it would optimize for fit on a regime that no longer applies.
- The existing pipeline tunes against 2020→2024 directly. This is methodologically imperfect (single holdout used both for hyperparameter selection and reported error) but it's an order-of-magnitude better than tuning against 1996→2000.
- If a validation slot is later required (e.g., to formalize hyperparameter selection without holdout leakage), carve it from **2016→2020** — same regime as the holdout, only one cycle removed. This costs 3 training dims but yields an in-regime validation signal.

**Net:** Keep `holdout_pairs.presidential: [[2020, 2024]]` as the only holdout. Do not promote any pre-2000 pair to validation status.

---

## Concrete config changes

### `config/model.yaml`

Replace the current `presidential_pairs:` block (lines 161–197) with:

```yaml
  presidential_pairs:
  # Post-realignment (Reagan era forward) — added P6.1
  - - 1980
    - 1984
  - - 1984
    - 1988
  - - 1988
    - 1992
  - - 1992
    - 1996
  - - 1996
    - 2000
  # Modern era (existing)
  - - 2000
    - 2004
  - - 2004
    - 2008
  - - 2008
    - 2012
  - - 2012
    - 2016
  - - 2016
    - 2020
```

Total: 10 training pres pairs (30 dims). Holdout unchanged.

Pre-1980 pairs (1948→1952 through 1976→1980) **stay defined** in `data.presidential_files:` (lines 302–315) so the parquets remain referenced and reachable for experiments, but are **not** enumerated in `presidential_pairs:` until the empirical sweep validates extending the window.

### `types.presidential_weight`

Currently `8.0`. With 5 pres pairs that yields a weighted-pres footprint of 5 × 3 × 8 = **120 effective dim units**, vs ~21 gov + ~24 senate dims. Adding 5 more pairs at the same weight would push pres-weighted dims to **240** — pres dominance roughly doubles.

The 8.0 value was tuned in the 5-pair regime (DECISIONS_LOG 2026-03-20: "presidential×2.5 weighting" originally; later raised to 8.0 in `model.yaml`). When the developer commits the 10-pair config, **retune `presidential_weight` over `[2.5, 4.0, 5.0, 6.0, 8.0]`** against 2020→2024 holdout r. Initial expectation: optimum lands near `4.0–5.0` to preserve the original pres-vs-gov-vs-senate balance.

### `src/core/config.py`

No code change required. `PRES_PAIRS` is built dynamically from `model.yaml`. The `(str(a)[-2:].zfill(2), str(b)[-2:].zfill(2))` two-digit truncation **silently aliases** centuries (e.g., 2048 and 1948 both become `"48"`), but for the 1948–2024 range the keys `"48"`–`"24"` are unique. **Flag for the developer:** if any future pair uses a 21st-century year that collides with a 20th-century year already in the config (e.g., 2048→2052 colliding with 1948→1952), this scheme breaks. Not an immediate concern, but worth a comment in `config.py` so the assumption is visible.

### `src/assembly/build_county_shifts_multiyear.py`

The script's docstring (line 4–7) already says "Pre-2000 Algara pairs: 1948→1952, 1952→1956, ..., 1996→2000". **Update the docstring** to reflect the actual active window (1980→1984 through 1996→2000). Stale docstrings on this file mislead future architects/devs into thinking the full 13-pair config is in use.

### Test fixtures

`tests/test_core_config.py` (referenced by Grep results) likely has expectations about `PRES_PAIRS` length. **Update it** to expect 10 pairs (not 5). The developer should grep for any other length-based assertion.

---

## Trade-offs explicitly acknowledged

| Trade-off | Decision | Why this side |
|---|---|---|
| Data quantity vs regime stationarity | Stationarity wins | Project's own DECISIONS_LOG (2026-03-19) shows historical depth costs short-run holdout r |
| Simplicity vs experimental rigor | Simplicity (single config) | Differential weighting and multiple validation splits add hyperparameter surface that is not yet justified by evidence |
| 5 vs 13 pairs | 5 (post-1980) | Pre-realignment signal (1968 break) is a structural opposite of modern shifts in the South; high contamination risk |
| In-regime validation vs single holdout | Single holdout | Currently the project tunes informally against 2020→2024; carving a validation pair is a separate (later) decision |
| Reduce `presidential_weight` now vs after sweep | After sweep | The current 8.0 is paired with 5 pres pairs; the new 10-pair config requires retuning, not preemptive scaling |

## What this enables for the developer

The developer's task (`fetch_algara_presidential.py` per the research doc) can proceed as follows:

1. **Fetch all 13 pairs of pre-2000 parquets** (already partially complete — files exist in `data/assembled/`). The fetcher should still be implemented for reproducibility and to mirror `fetch_algara_amlani.py`'s pattern; it just won't add new files on first run.
2. **Update `config/model.yaml`** with the 5-pair extension as specified above.
3. **Run `build_county_shifts_multiyear.py`** — verify all 30 training dims (10 pres + 7 gov + 8 senate × 3) build cleanly with no all-zero columns.
4. **Run `validate_county_holdout_multiyear`** — record holdout r at K=5,7,10,15,20 and compare to the current 5-pair baseline. **Acceptance criterion:** new holdout r ≥ baseline holdout r (i.e., adding 5 historical pairs does not degrade the model). If it degrades, narrow to Approach B (2 pairs, post-1992) and re-test.
5. **Retune `presidential_weight`** as a separate sub-task once the 10-pair config is committed.
6. **Sweep the additional pairs** (Approach E) as a post-merge experiment, not a precondition for shipping the 10-pair default.

## Update to DECISIONS_LOG when adopted

Append to `docs/DECISIONS_LOG.md` after merge:

> | 2026-05-04 | PRES_PAIRS extended to 10 pairs (post-1980) via Algara pre-2000 returns. | Adds 1980→1984 through 1996→2000 (5 new pairs). Pre-1980 pairs cross the Southern Realignment regime break and were excluded as non-stationary with the 2020→2024 holdout. Alaska pre-2000 not present in Algara, so no exclusion code required. `presidential_weight` retuned alongside this change. | -- |

---

## Open items for follow-up tasks

- **EXP-1 (sweep):** Holdout r vs window depth (5 → 7 → 10 → 13 pres pairs). One run per width. Adopt the deepest config that does not degrade holdout r.
- **EXP-2 (`presidential_weight`):** Sweep `[2.5, 4.0, 5.0, 6.0, 8.0]` at the chosen window. Lock in the optimum.
- **EXP-3 (decay weighting):** *Only if* EXP-1 shows non-monotonic holdout r in window depth, try Approach D (per-pair decay weight). Otherwise skip.
- **DOC-1:** Update `build_county_shifts_multiyear.py` docstring (currently claims all 13 pairs are active).
- **DOC-1b:** Comment in `src/core/config.py` flagging the 2-digit year aliasing assumption.
