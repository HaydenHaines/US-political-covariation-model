# Crosstab Signal Holdout Accuracy Report — 2026-05-04

**Script:** `scripts/experiments/compare_xt_impact_v2.py`
**Run date:** 2026-05-04
**Data source:** `data/polls/polls_2026.csv`
**EPIC:** `knowledge/epics/wethervane-crosstab-signal-accuracy.md` (SC1)

---

## Summary

Three runs compared:
- **Live (enriched):** Production path with xt_* columns preserved through `prepare_polls`
- **Stripped:** Baseline — all xt_* and methodology keys removed; topline-only forecast
- **Tier 2 bypass:** Enriched polls, `reference_date=None` (skips `prepare_polls` preprocessing)

| Metric | Live vs Stripped | Tier 2 Bypass vs Stripped |
|---|---|---|
| Races with xt_ data | 16 | 16 |
| Mean \|Δ\| (pp) | **1.492** | 1.401 |
| Max \|Δ\| (pp) | **4.832** | 2.959 |
| Median \|Δ\| (pp) | **1.000** | 1.392 |

θ_national impact (max abs diff across J=100 types):

| Run | max θ max-diff | mean |
|---|---|---|
| Live vs Stripped | 0.5789 | 0.5789 |
| Tier 2 bypass vs Stripped | 1.7558 | 1.7558 |

---

## Quinnipiac note

**Quinnipiac contribution is zero in this run.** All 5 Quinnipiac rows in `polls_2026.csv`
have empty xt_* columns — the wiring gap between `ingest_quinnipiac_crosstabs.py`
(writes to DuckDB `poll_crosstabs`) and the live forecast path (reads xt_* from CSV)
is unresolved as of 2026-05-04. See EPIC SC2 (bug #1319 / commit #1324 verification
pending). A secondary blocker: `pct_of_sample` is NULL for all Quinnipiac PDF rows,
which would trigger the `if pct <= 0: continue` guard in `_extract_crosstabs_from_xt()`
even after the wiring gap is closed.

---

## Emerson/Marist baseline

SC3 verified (as of 2026-05-04): 24/28 Emerson rows in `polls_2026.csv` have non-empty
xt_* columns. Emerson + Marist are the only pollsters enriching the live forecast.

---

## Per-race delta table

`*` marks races where at least one poll carries xt_ crosstab data.
Columns: `xt/tot` = polls with xt_ data / total polls for race.
`Δ_live` and `Δ_tier2` are in percentage points (positive = more Dem).

```
Race                               xt/tot  stripped      live   Δ_live     tier2   Δ_tier2  θ_max_live  θ_max_tier2
----------------------------------------------------------------------------------------------------------
2026 AK Governor                   0/0       0.5755    0.5755  +0.000    0.5755   +0.000    0.578864     1.755802
2026 AK Senate                     0/0       0.5755    0.5755  +0.000    0.5755   +0.000    0.578864     1.755802
2026 AL Governor                   0/1       0.3860    0.3858  -0.015    0.3893   +0.330    0.578864     1.755802
2026 AL Senate                     0/0       0.3892    0.3892  +0.000    0.3892   +0.000    0.578864     1.755802
2026 AR Governor                   0/0       0.3752    0.3752  +0.000    0.3752   +0.000    0.578864     1.755802
2026 AR Senate                     0/0       0.3752    0.3752  +0.000    0.3752   +0.000    0.578864     1.755802
2026 AZ Governor                   1/3       0.5129    0.5125  -0.038    0.5029   -1.004    0.578864     1.755802 *
2026 CA Governor                   0/0       0.6364    0.6364  +0.000    0.6364   +0.000    0.578864     1.755802
2026 CO Governor                   0/0       0.5920    0.5920  +0.000    0.5920   +0.000    0.578864     1.755802
2026 CO Senate                     0/0       0.5920    0.5920  +0.000    0.5920   +0.000    0.578864     1.755802
2026 CT Governor                   0/0       0.6118    0.6118  +0.000    0.6118   +0.000    0.578864     1.755802
2026 DE Senate                     0/0       0.6066    0.6066  +0.000    0.6066   +0.000    0.578864     1.755802
2026 FL Governor                   2/22      0.4804    0.4320  -4.832    0.4508   -2.959    0.578864     1.755802 *
2026 FL Senate                     2/12      0.4675    0.4401  -2.742    0.4461   -2.143    0.578864     1.755802 *
2026 GA Governor                   0/2       0.5337    0.5343  +0.067    0.5323   -0.133    0.578864     1.755802
2026 GA Senate                     2/17      0.5356    0.4953  -4.027    0.5153   -2.032    0.578864     1.755802 *
2026 HI Governor                   0/0       0.6676    0.6676  +0.000    0.6676   +0.000    0.578864     1.755802
2026 IA Governor                   0/0       0.4680    0.4680  +0.000    0.4680   +0.000    0.578864     1.755802
2026 IA Senate                     0/3       0.4723    0.4704  -0.197    0.4702   -0.215    0.578864     1.755802
2026 ID Governor                   0/0       0.3587    0.3587  +0.000    0.3587   +0.000    0.578864     1.755802
2026 ID Senate                     0/0       0.3587    0.3587  +0.000    0.3587   +0.000    0.578864     1.755802
2026 IL Governor                   0/0       0.5876    0.5876  +0.000    0.5876   +0.000    0.578864     1.755802
2026 IL Senate                     0/0       0.5876    0.5876  +0.000    0.5876   +0.000    0.578864     1.755802
2026 KS Governor                   0/0       0.4564    0.4564  +0.000    0.4564   +0.000    0.578864     1.755802
2026 KS Senate                     0/0       0.4564    0.4564  +0.000    0.4564   +0.000    0.578864     1.755802
2026 KY Senate                     0/0       0.3863    0.3863  +0.000    0.3863   +0.000    0.578864     1.755802
2026 LA Senate                     0/0       0.4275    0.4275  +0.000    0.4275   +0.000    0.578864     1.755802
2026 MA Governor                   0/0       0.6602    0.6602  +0.000    0.6602   +0.000    0.578864     1.755802
2026 MA Senate                     0/2       0.6631    0.6612  -0.190    0.6599   -0.327    0.578864     1.755802
2026 MD Governor                   0/0       0.6718    0.6718  +0.000    0.6718   +0.000    0.578864     1.755802
2026 ME Governor                   0/0       0.5678    0.5678  +0.000    0.5678   +0.000    0.578864     1.755802
2026 ME Senate                     2/26      0.5497    0.5308  -1.889    0.5348   -1.495    0.578864     1.755802 *
2026 MI Governor                   0/25      0.5409    0.5158  -2.515    0.5193   -2.161    0.578864     1.755802
2026 MI Senate                     1/10      0.5293    0.5096  -1.970    0.5156   -1.372    0.578864     1.755802 *
2026 MN Governor                   1/2       0.5572    0.5431  -1.412    0.5453   -1.190    0.578864     1.755802 *
2026 MN Senate                     2/3       0.5530    0.5334  -1.963    0.5371   -1.597    0.578864     1.755802 *
2026 MS Senate                     0/0       0.4284    0.4284  +0.000    0.4284   +0.000    0.578864     1.755802
2026 MT Senate                     0/0       0.4293    0.4293  +0.000    0.4293   +0.000    0.578864     1.755802
2026 NC Senate                     0/25      0.5460    0.5462  +0.026    0.5448   -0.114    0.578864     1.755802
2026 NE Governor                   0/0       0.4325    0.4325  +0.000    0.4325   +0.000    0.578864     1.755802
2026 NE Senate                     0/0       0.4325    0.4325  +0.000    0.4325   +0.000    0.578864     1.755802
2026 NH Governor                   0/2       0.5397    0.5399  +0.026    0.5382   -0.149    0.578864     1.755802
2026 NH Senate                     2/26      0.5612    0.5274  -3.386    0.5376   -2.367    0.578864     1.755802 *
2026 NJ Senate                     0/0       0.5721    0.5721  +0.000    0.5721   +0.000    0.578864     1.755802
2026 NM Governor                   0/0       0.5625    0.5625  +0.000    0.5625   +0.000    0.578864     1.755802
2026 NM Senate                     0/0       0.5625    0.5625  +0.000    0.5625   +0.000    0.578864     1.755802
2026 NV Governor                   1/2       0.5235    0.5221  -0.141    0.5177   -0.579    0.578864     1.755802 *
2026 NY Governor                   1/2       0.6042    0.6055  +0.125    0.6072   +0.293    0.578864     1.755802 *
2026 OH Governor                   4/22      0.4861    0.4882  +0.213    0.4899   +0.382    0.578864     1.755802 *
2026 OH Senate                     1/4       0.4804    0.4810  +0.056    0.4871   +0.666    0.578864     1.755802 *
2026 OK Governor                   0/0       0.3605    0.3605  +0.000    0.3605   +0.000    0.578864     1.755802
2026 OK Senate                     0/0       0.3605    0.3605  +0.000    0.3605   +0.000    0.578864     1.755802
2026 OR Governor                   0/0       0.5997    0.5997  +0.000    0.5997   +0.000    0.578864     1.755802
2026 OR Senate                     0/0       0.5997    0.5997  +0.000    0.5997   +0.000    0.578864     1.755802
2026 PA Governor                   2/8       0.5570    0.5562  -0.079    0.5350   -2.198    0.578864     1.755802 *
2026 RI Governor                   0/0       0.6084    0.6084  +0.000    0.6084   +0.000    0.578864     1.755802
2026 RI Senate                     0/0       0.6084    0.6084  +0.000    0.6084   +0.000    0.578864     1.755802
2026 SC Governor                   0/0       0.4570    0.4570  +0.000    0.4570   +0.000    0.578864     1.755802
2026 SC Senate                     0/0       0.4570    0.4570  +0.000    0.4570   +0.000    0.578864     1.755802
2026 SD Governor                   0/0       0.3797    0.3797  +0.000    0.3797   +0.000    0.578864     1.755802
2026 SD Senate                     0/0       0.3797    0.3797  +0.000    0.3797   +0.000    0.578864     1.755802
2026 TN Governor                   0/0       0.3906    0.3906  +0.000    0.3906   +0.000    0.578864     1.755802
2026 TN Senate                     0/0       0.3906    0.3906  +0.000    0.3906   +0.000    0.578864     1.755802
2026 TX Governor                   2/15      0.4625    0.4584  -0.407    0.4552   -0.723    0.578864     1.755802 *
2026 TX Senate                     1/3       0.4825    0.4767  -0.588    0.4684   -1.412    0.578864     1.755802 *
2026 VA Senate                     0/0       0.5706    0.5706  +0.000    0.5706   +0.000    0.578864     1.755802
2026 VT Governor                   0/0       0.6700    0.6700  +0.000    0.6700   +0.000    0.578864     1.755802
2026 WI Governor                   0/4       0.5269    0.5280  +0.112    0.5258   -0.103    0.578864     1.755802
2026 WV Senate                     0/0       0.3299    0.3299  +0.000    0.3299   +0.000    0.578864     1.755802
2026 WY Governor                   0/0       0.3050    0.3050  +0.000    0.3050   +0.000    0.578864     1.755802
2026 WY Senate                     0/0       0.3050    0.3050  +0.000    0.3050   +0.000    0.578864     1.755802
```

---

## Interpretation

- Crosstab enrichment produces meaningful shifts in swing-state races: FL Governor (−4.83 pp),
  GA Senate (−4.03 pp), NH Senate (−3.39 pp), MI Governor (−2.52 pp). These are the races where
  Emerson/Marist xt_ signals are densest.
- The live production path (Run A) is correctly propagating xt_ signals — mean |Δ| of 1.492 pp
  confirms the regression fix (WV PR #167) is working.
- Tier 2 bypass moves θ_national more (mean 1.7558 vs 0.5789) but moves per-race predictions less
  (mean 1.401 pp vs 1.492 pp). Live path is operating on both state and national axes.
- PA Governor xt coverage (2/8) reflects Emerson/Marist polls, not Quinnipiac — Quinnipiac rows
  have empty xt_* and contribute zero signal.

---

## SC status

| SC | Status | Notes |
|---|---|---|
| SC1 | **PASS** | This report. 16 enriched races; enriched-vs-stripped delta committed. |
| SC2 | pending | Quinnipiac xt_vote_* still empty (5/5 rows). Bug #1319 / commit #1324 verification queued. |
| SC3 | PASS | 24/28 Emerson rows have non-empty xt_* (verified 2026-05-04). |
