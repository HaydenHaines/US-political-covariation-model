# EPIC: WetherVane Frontend Features v24

## Goal

Finish the `/explore/types` shareable-state trilogy by hardening `ComparisonTable` URL persistence for selected electoral types.

v22 covered `TypeGrid` search state with `?q=`.
v23 covered `ScatterPlot` axis state with `?x=` and `?y=`.
v24 should cover `ComparisonTable` selected type state with `?types=`.

## Chosen Feature

**Explore Types comparison URL-state hardening**

The user-facing contract is that selected comparison types on `/explore/types` are encoded into a shareable URL:

```text
/explore/types?types=101,202
```

That contract exists in `web/components/explore/ComparisonTable.tsx`, and the page copy in `web/app/explore/types/page.tsx` says "The comparison URL is shareable." v24 should make that promise tested and robust.

## Acceptance Criteria

- `/explore/types?types=101,202` hydrates the comparison table with both selected type columns.
- Selecting a type from the comparison combobox writes `types=<id>` to the current URL.
- Adding a second type appends it to `types=` in selection order.
- Removing a selected type updates `types=` without clearing unrelated query parameters.
- Clicking "Clear all" removes `types=` without clearing unrelated query parameters such as `q=`, `x=`, or `y=`.
- Invalid type IDs in `types=` do not crash the page.
- Duplicate IDs are deduped or otherwise rendered only once.
- More than four IDs are capped at four, matching `MAX_TYPES = 4`.
- Existing `/explore/types` tests for `?q=`, `?x=`, and `?y=` still pass.
- Tests use deterministic Playwright route mocks for `/api/v1/types`, `/api/v1/types/scatter-data`, `/api/v1/super-types`, `/api/forecast/xt-impact?limit=10`, and `/api/v1/types/:id`.

## T1 Developer Task

### Title

v24 T1: Add Playwright coverage and harden `ComparisonTable` `?types=` URL state

### Scope

Read:

- `web/app/explore/types/page.tsx`
- `web/components/explore/ComparisonTable.tsx`
- `web/e2e/explore-types.spec.ts`
- `web/lib/hooks/use-types.ts`
- `web/lib/hooks/use-type-detail.ts`
- `web/lib/api.ts`
- `web/lib/types.ts`

Write:

- `web/e2e/explore-types.spec.ts`
- `web/components/explore/ComparisonTable.tsx`

Do not change backend APIs or data artifacts for this task.

### Implementation Notes

- Extend the existing `mockExploreTypesData(page)` helper in `web/e2e/explore-types.spec.ts` to fulfill `/api/v1/types/:id` for at least IDs `101`, `202`, `303`, `404`, and `505`.
- Keep detail fixtures small but include enough `demographics` fields for table rows to render, such as `pct_white_nh`, `pct_bachelors_plus`, `median_hh_income`, and `mean_pred_dem_share`.
- Prefer role/name selectors where stable. Add targeted `data-testid` attributes in `ComparisonTable.tsx` only if Playwright cannot reliably select the combobox, clear button, remove buttons, or rendered comparison table.
- Preserve existing query parameters when writing `types=`. The component currently uses `new URLSearchParams(searchParams.toString())`; keep that behavior covered by tests.
- Consider normalizing parsed `types=` values in `ComparisonTable.tsx` so invalid IDs, duplicates, and more than four IDs produce a stable selected ID list.
- If normalization changes the selected IDs after initial hydration, update the URL to the canonical value.

### Suggested Playwright Cases

Add a new `test.describe("ComparisonTable URL sync (/explore/types v24)", ...)` block in `web/e2e/explore-types.spec.ts`:

- `hydrates selected type columns from the types URL parameter`
- `writes selected type changes to the URL`
- `adds a second selected type without dropping q/x/y URL state`
- `removes one selected type and preserves unrelated URL state`
- `clears all selected types and removes only the types URL parameter`
- `ignores invalid and duplicate type IDs from the URL`
- `caps hydrated type selections at four IDs`

### Verification

Run the focused Playwright file:

```bash
cd web
npm run test:e2e -- explore-types.spec.ts
```

If that script name is unavailable, use the repository's existing Playwright command from `web/package.json`.

Also run the frontend type/lint check if available:

```bash
cd web
npm run lint
```

## Out Of Scope

- `/explore/map` overlay URL-sync.
- Header tipping-point bar E2E.
- Governor forecast URL-state hardening.
- Backend API changes.
- Visual redesign of the comparison table.

## Follow-Up Candidates

- v25: `/explore/map` overlay URL-sync once map E2E is deterministic enough for overlay assertions.
- v26: Header tipping-point bar E2E with mocked Senate overview data.
- Future: Governor URL hardening after a concrete current contract or bug is verified.
