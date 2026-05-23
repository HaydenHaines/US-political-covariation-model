# WetherVane v24 Frontend Direction

Date: 2026-05-17
Status: Accepted

## Question

What should the next WetherVane frontend feature be after v23 shipped ScatterPlot axis URL-sync on `/explore/types` and extended the `/explore/types` Playwright coverage?

## Current State

Recent frontend slices have moved through the explore surface by making interactive state shareable and then locking it with Playwright coverage:

- v19 mounted `TypeNarrative` and `TypeCountyList` on `/type/[id]`.
- v20 shipped Pollster search URL-sync.
- v21 shipped `/explore/shifts` Playwright E2E coverage.
- v22 shipped `TypeGrid` search URL-sync via `?q=`.
- v23 shipped `ScatterPlot` axis URL-sync via `?x=` and `?y=`, plus `/explore/types` E2E tests.

The audited frontend files exist:

- `web/app/explore/types/page.tsx`
- `web/app/explore/shifts/page.tsx`
- `web/app/explore/map/page.tsx`
- `web/app/explore/map/MapPageClient.tsx`
- `web/components/explore/TypeGrid.tsx`
- `web/components/explore/ScatterPlot.tsx`
- `web/components/explore/ComparisonTable.tsx`
- `web/components/explore/XtImpactPanel.tsx`
- `web/components/explore/MapOverlayToggle.tsx`
- `web/components/nav/SiteHeader.tsx`
- `web/e2e/explore-types.spec.ts`
- `web/e2e/shifts.spec.ts`
- `web/e2e/navigation.spec.ts`
- `web/e2e/forecast.spec.ts`

Coverage gap observed:

- `web/e2e/explore-types.spec.ts` now covers the page shell, `TypeGrid` search URL-sync, and `ScatterPlot` axis URL-sync.
- `web/components/explore/ComparisonTable.tsx` already documents and implements `?types=1,5,23,67` URL persistence, but there is no dedicated Playwright coverage for hydrating, mutating, clearing, invalid/duplicate handling, or preserving unrelated query parameters on `/explore/types`.
- `web/components/explore/XtImpactPanel.tsx` is indirectly mocked in `/explore/types` tests as a data dependency, but its table behavior is not the best next URL-state slice.
- `/explore/map` has smoke navigation coverage in `web/e2e/navigation.spec.ts`; `MapOverlayToggle` has no URL contract today.

## Options Considered

### Option A: Harden `ComparisonTable` URL-sync and add `/explore/types` E2E coverage

Add explicit tests around the existing `?types=` contract in `web/components/explore/ComparisonTable.tsx`, and make small implementation fixes if the tests expose gaps.

Expected scope:

- Extend `web/e2e/explore-types.spec.ts` with mocked `/api/v1/types` and `/api/v1/types/:id` responses.
- Verify `/explore/types?types=101,202` hydrates selected columns.
- Verify selecting a type from the combobox writes `types=`.
- Verify removing one selected type updates `types=`.
- Verify "Clear all" removes `types=` while preserving unrelated params such as `q=`, `x=`, and `y=`.
- Verify duplicates, invalid IDs, and more than four IDs do not break rendering.
- Add `data-testid` hooks only where role/name selectors are not stable enough.

Trade-offs:

- Pros: Follows the v20-v23 URL-state momentum, exercises a user-visible shareable workflow, and closes the largest uncovered `/explore/types` component after v23.
- Pros: Small blast radius: mostly E2E plus narrowly scoped component hardening.
- Pros: Uses deterministic Playwright route mocks and does not require deck.gl/canvas assertions.
- Cons: It is mostly reliability/product polish rather than a new visual capability.
- Cons: `ComparisonTable` fetches up to four details via `useTypeDetail`, so tests need more mock fixtures than the v23 axis tests.

### Option B: Add URL-sync for `/explore/map` overlay mode

Persist `MapOverlayToggle` state in `/explore/map` with a query parameter such as `?overlay=types|forecast`, then add Playwright coverage.

Trade-offs:

- Pros: Gives `/explore/map` a real state contract beyond smoke coverage.
- Pros: Aligns with the shareable-state pattern established in v20-v23.
- Cons: The map page is backed by `MapShell`, deck.gl, and map context state. Reliable E2E assertions are heavier and more fragile than comparison-table assertions.
- Cons: Forecast overlay behavior depends on broader map/race selection state; overlay URL-sync alone is a thinner user feature.

### Option C: Add E2E coverage for the header tipping-point bar

Exercise `HeaderTippingPointBar` in `web/components/nav/SiteHeader.tsx`: data load, segment labels, hover tooltip, keyboard activation, and navigation to `/forecast/[slug]`.

Trade-offs:

- Pros: Header bar is high-visibility and currently lacks direct E2E coverage.
- Pros: Interaction coverage would protect an important navigation affordance.
- Cons: The component depends on `useSenateOverview()` data and renders conditionally when races are present, so tests need API mocking or stable backend data.
- Cons: The candidate called out "Homepage TippingPointBar"; the audited implementation is a site-header bar, not a homepage-only feature. That makes it a less clean v24 seed than a verified page-local explore feature.

### Option D: Governor URL-sync test hardening

Search for unfinished v17 governor URL-sync work and harden it.

Trade-offs:

- Pros: Governor surfaces are important and have only lightweight overview coverage in `web/e2e/forecast.spec.ts`.
- Cons: The audit did not find a concrete v17 URL-sync residue in the current frontend files comparable to the explicit `ComparisonTable` `?types=` contract.
- Cons: Without a specific broken contract, this is a discovery task rather than a focused v24 implementation seed.

## Decision

Choose Option A: v24 should harden and test `ComparisonTable` shareable type selection on `/explore/types`.

The feature name is:

**v24: Explore Types comparison URL-state hardening**

## Rationale

`ComparisonTable` is the most natural next step after v22 and v23 because it sits on the same `/explore/types` page and already promises shareable URL behavior in both code comments and page copy. The current Playwright suite covers `TypeGrid` `?q=` and `ScatterPlot` `?x=`/`?y=`, but not `ComparisonTable` `?types=`, leaving the page's third URL-state contract unprotected.

This is the best v24 seed because it is:

- Incremental: one known page, one existing component, one existing URL parameter.
- Testable: route mocks can fully control `/api/v1/types` and `/api/v1/types/:id`.
- Product-relevant: comparison links are meant to be shared and revisited.
- Low risk: no deck.gl/canvas behavior, no backend schema change, and no dependency on live Senate or governor data.

Rejected candidates are still valid future work:

- `/explore/map` overlay URL-sync should follow once map E2E has stronger deterministic harnessing.
- Header tipping-point coverage should be scheduled when the suite has a stable mocked Senate overview fixture.
- Governor URL hardening should wait for a verified open contract or bug, not be seeded from a vague historical reference.

## T1 Developer Task Queued

Implement v24 T1 from `knowledge/epics/wethervane-frontend-features-v24.md`.

Verified write targets:

- `web/e2e/explore-types.spec.ts`
- `web/components/explore/ComparisonTable.tsx`

Verified read/context targets:

- `web/app/explore/types/page.tsx`
- `web/lib/hooks/use-types.ts`
- `web/lib/hooks/use-type-detail.ts`
- `web/lib/api.ts`
- `web/lib/types.ts`
