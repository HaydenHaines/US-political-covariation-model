---
epic: wethervane-frontend-features-v23
project: wethervane
status: closed
created: 2026-05-17
closed: 2026-05-17
merge_sha: 923c368925c0fc0ca3c42de1ced1bca2718bac32
branch: feat/wethervane-v23-scatterplot-axis-url-sync
head_sha: c8962b4b7877c795f7788054e21b6648e2f5ee7f
t1_task: "#2949"
t2_regate_task: "#2950"
---

# EPIC: WetherVane Frontend Features v23

**Last updated:** 2026-05-17
**Status:** closed

## Scope

Closed the ScatterPlot axis URL-sync cycle for `/explore/types`.
The shipped work syncs desktop ScatterPlot axis selections through
`?x=` and `?y=`, validates URL values against the allowed axis fields,
omits default axis values from the URL, and extends
`web/e2e/explore-types.spec.ts` with coverage for the new behavior.

## Closeout Evidence

- QA re-gate task `#2950` completed with verdict PASS: "all 6 SCs pass."
- T1 redo branch `feat/wethervane-v23-scatterplot-axis-url-sync` was verified at `c8962b4b7877c795f7788054e21b6648e2f5ee7f`.
- Local `main` merge commit: `923c368925c0fc0ca3c42de1ced1bca2718bac32`.
- QA #2950 evidence included `pnpm run lint` exit 0 and `13 passed (20.3s)` for `PORT=3101 pnpm exec playwright test e2e/explore-types.spec.ts`.

## Follow-Ups

No outstanding T2 follow-up items remain. The first T2 gate failed and queued redo task `#2949`; task `#2950` re-gated the redo branch and passed.

