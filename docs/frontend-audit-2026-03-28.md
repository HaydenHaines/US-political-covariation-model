# WetherVane Frontend Audit Report
**Date:** 2026-03-28
**Auditor:** Claude Opus 4.6 (automated)
**URL:** http://localhost:3001
**Framework:** Next.js App Router

---

## Executive Summary

The WetherVane frontend is a functional MVP with strong content quality (especially the methodology page) and a solid map+panel layout. However, it has several critical data display bugs, significant data consistency issues, and numerous UX problems that undermine credibility. The most severe issue is that **every Senate race on the forecast page shows "EVEN" despite the API returning real margins** -- the single most important piece of information on the entire site is missing.

---

## Page-by-Page Audit

### 1. `/` (Root) → Redirects to `/forecast`

**Behavior:** 301 redirect to `/forecast`. Correct.

---

### 2. `/forecast` (Main Forecast Page)

**Visual Design:** Good overall. The map+panel split layout works well. Muted earth-tone color palette is distinctive. The "stained glass" map metaphor is visually appealing and differentiated from competitor sites.

**CRITICAL BUGS:**

- **All 32 Senate races display "EVEN" instead of their actual margins.** The API (`/api/v1/senate/overview`) returns real margins (e.g., ME: R+2.8%, MI: D+0.5%, GA: D+7.7%) but the frontend displays "EVEN" for every single race. This is the most important information on the page and it is completely wrong.
  - Root cause: The `SenateControlBar.tsx` has a local `formatMargin()` that checks `if (abs < 0.5) return "EVEN"` -- but the `margin` field from the API is already in decimal form (e.g., 0.028 for 2.8%), not percentage points. So `abs(0.028) < 0.5` is always true, and everything shows "EVEN". The threshold should be `0.005` (0.5pp), not `0.5`.

- **Race cards show no actual lean/rating colors.** All "Tossup" badges are shown with the same muted brown color. There is no visual differentiation between "Tossup", "Lean D", "Lean R", "Likely D", "Likely R", etc. The race card colors should match the map legend colors.

- **"Key Races" vs "Other Races" grouping appears arbitrary.** The 6 "Key Races" (ME, MI, MN, NH, NM, VA) all show as "Tossup EVEN" and the 26 "Other Races" also all show "EVEN". Without actual margins, there is no way to distinguish them.

**MAJOR ISSUES:**

- **Map legend shows Senate forecast colors but says "Safe D / Likely D / etc."** which contradicts the panel showing everything as "EVEN". The map paints some states (like NY) as "Safe D" and others (like ND) as "Safe R", but the panel does not reflect this.

- **Clicking a race card only changes the map view (zooms to state)** but does not navigate to the race detail page (`/forecast/[slug]`). The race cards are buttons, not links. Users have no obvious way to get to the detailed race page from the main forecast view.

- **Senate balance bar shows 47D–53R** but has no visual indication that tossup states could change this. The bar states (ME, MI, MN, NH, NM, VA) are clickable but their behavior is unclear.

- **No Governor races visible on the forecast page.** Only Senate races are shown. The API has 36 governor race slugs. The forecast page title says "2026 United States Senate" but there is no toggle or tab to view governor races.

- **No Presidential race.** The page says "2026" but there is no presidential race (correct -- there is no 2026 presidential race). However, users might expect to see the term "midterm" mentioned.

**MINOR ISSUES:**

- **favicon.ico returns 404** (2 console errors on every page load).
- **Loading state shows bare "Loading..." text** instead of a skeleton or spinner.
- **No scroll indicator** that content exists below the fold.
- **Map takes up exactly half the viewport** but the bottom half (panel) cannot be scrolled independently on some viewport sizes.

**Dark Mode:**
- Map colors remain unchanged (acceptable -- cartographic colors should not invert).
- Panel background and text invert properly.
- **Race cards retain white/light backgrounds in dark mode** -- contrast issue. They should have dark card backgrounds.
- Tab bar separator line may be hard to see.

**Mobile (375px):**
- Layout adapts well: map stacks on top, content below.
- Article/Dashboard toggle hidden (correct -- Dashboard mode is desktop-only).
- Race cards stack vertically (good).
- Senate balance bar compresses nicely.

---

### 3. `/forecast` — Dashboard Mode

**Visual Design:** Full-screen map with a right-side overlay panel.

**MAJOR ISSUES:**

- **Overlay panel says "No polls available for this state"** when viewing the national map (no state selected). Should say something like "Click a state to see polls" or not show the polls section at all.
- **"Recalculate with Polls" button is disabled** with no explanation of what it does or when it becomes enabled.
- **Navigation tabs (Forecast, Explore, About) disappear** in Dashboard mode. The only way back to Article mode is the radio toggle in the top-right corner. This could trap users.
- **No state selection mechanism is visible.** The user has to click on states on the map, but there is no visual affordance (hover state, cursor change) suggesting states are clickable.

---

### 4. `/explore` (Shift Explorer)

**Visual Design:** Scatter plot with X/Y axis dropdowns. Clean and functional.

**MAJOR ISSUES:**

- **Map legend still shows Senate forecast colors** (Safe D, Likely D, etc.) even though the Explore page has nothing to do with the Senate forecast. The map should either show no legend or show the explore-relevant coloring.
- **Super-type legend at bottom uses generic labels**: "Super-type 1", "Super-type 2", etc. instead of their actual names ("Hispanic & Immigrant Gateway", "Black & Minority Urban", etc.).
- **Says "130 types shown"** but the About page and methodology say "100 fine types". Data inconsistency.

**MINOR ISSUES:**

- **Y-axis raw variable names** (e.g., "Pres d shift 20 24") are technical/internal. Should be human-readable (e.g., "Presidential Dem Shift 2020-2024").
- **X-axis label overlaps** the axis tick labels at the bottom right when the chart is wide.
- **No tooltip on hover** for the scatter dots -- you have to click to see which type a dot represents.
- Chart **does not resize** smoothly when the browser window changes size.

---

### 5. `/about` (About Page)

**Visual Design:** Clean, well-structured. Good use of the metric cards (Counties: 3,154, Fine types: 100, etc.).

**MAJOR ISSUES:**

- **Says "5 super-types" and "100 fine types"** but the /types page shows 6 super-types and 130 fine types. The API also returns type IDs up to at least 130. The About page is stale.

**MINOR ISSUES:**

- **"Read the full methodology" link** is a large block link that takes up the full width. Good UX.
- **"Built by Hayden Haines"** attribution is at the bottom but could be more prominent.
- No link to the types listing page.

---

### 6. `/county/[fips]` (County Detail Pages)

**Tested:** `/county/13121` (Fulton County, GA) and `/county/06037` (Los Angeles County, CA)

**Visual Design:** Clean, well-organized. Demographics presented in a clear two-column layout. Good use of breadcrumbs.

**CRITICAL DATA ISSUE:**

- **Type names contradict demographics.** Los Angeles County (48.7% Hispanic, 25.2% White, D+15.0) is classified as "White Red" in the "White Working & Rural" super-type. This is deeply misleading. The type names appear to be based on demographic characteristics of the type centroid, but when applied to individual counties, they create confusing mismatches.
  - Similarly, Fulton County (43% Black, D+22.3) is classified as "White College-Educated Red". The "Red" in the type name contradicts the D+22 margin shown.
  - **Root cause:** Type names contain a party lean suffix ("Red" or "Blue") that refers to the type centroid's overall lean, not the individual county's lean. This is confusing for users.

**MAJOR ISSUES:**

- **"Similar Counties" section** lists counties from the same type, but the type name ("White Red") is confusing when the county itself is majority-minority and strongly Democratic.
- **No election history shown.** County pages show demographics and type classification but no historical vote data. Adding a simple margin-by-year chart would be high value.
- **"View on Map" link** goes to `/forecast` (the national forecast page) instead of zooming to the county on the map.

**MINOR ISSUES:**

- **Breadcrumbs show "Home / Map / GA / Fulton County"** but "Home" and "Map" both link to `/` (which redirects to `/forecast`). These should be distinguishable.
- No link from the county page to the type detail page for that county's type.
- **Tags ("White College-Educated Red", "Middle-Class Suburban", "D+22.3")** are shown as pill badges but are not clickable. The type name should link to the type detail page.

---

### 7. `/type/[id]` (Type Detail Pages)

**Tested:** `/type/1` (Black Blue)

**Visual Design:** Same clean layout as county pages. Good breadcrumb navigation.

**CRITICAL DATA ISSUE:**

- **Type 1 "Black Blue" shows R+23.0.** A type named "Black Blue" in the "Black & Minority Urban" super-type having a strong Republican lean is deeply confusing. The "Blue" suffix appears to indicate historical lean, not current predicted lean. This naming scheme is broken.

**MAJOR ISSUES:**

- **Member counties list** is organized by state (good) but can be very long. Type 1 has 70 counties across 15 states. No truncation, collapsing, or pagination.
- **Religion section** is shown on type pages but not on county pages -- inconsistency.
- **No shift history** or trend visualization. Type pages should show how this type has moved over time.

**MINOR ISSUES:**

- "View on Map" link goes to `/forecast` instead of highlighting the type's counties on the map.

---

### 8. `/types` (Types Listing Page)

**Visual Design:** Well-organized with type cards grouped by super-type. Each card shows type ID, name, lean, county count, income, and college rate.

**MAJOR ISSUES:**

- **Shows 130 fine types and 6 super-types** but About/Methodology pages say 100/5. The J=130 model appears to be deployed to the API but the static content pages have not been updated.
- **Super-type "Rural Institutional Outlier" has only 2 types and 2 counties** -- seems like a data artifact rather than a meaningful category.
- **Very long page** (130 type cards). No search, filter, or sort functionality. Should at minimum have jump links to super-type sections.
- **Type names with "Blue" / "Red" suffixes** are systematically confusing (see county and type detail page issues above).

**MINOR ISSUES:**

- Summary stats at top ("130 fine types, 6 super-types, 3,154 counties") are accurate for the current data but inconsistent with methodology claims.

---

### 9. `/methodology` (Methodology Page)

**Visual Design:** Excellent. This is the best page on the entire site. Clean typography, collapsible sections with TOC, numbered step-by-step explanation, well-designed metric cards. The writing quality is strong -- it reads like a professional methodology article.

**MAJOR ISSUES:**

- **States "100 fine-grained electoral types and 5 super-types"** in the body text, but the actual model has 130/6. Needs updating.
- **All sections are expanded by default.** With 8 sections, this makes for a very long scroll. Consider collapsing all except the first section by default.

**MINOR ISSUES:**

- TOC floats on the left side -- works on desktop but may not be visible on mobile.
- Section heading links show "#" text that could be distracting.
- The "Backtesting results" card at the bottom links to `/methodology/accuracy` -- good cross-linking.

---

### 10. `/methodology/accuracy` (Accuracy Subpage)

**Visual Design:** Clean, well-structured. Good use of progressive bar charts for model improvement ladder and cross-election validation.

**ISSUES:**

- **Contradictory statement in "What This Means" section:** "The standard holdout r (0.698) is slightly higher than LOO (0.711) is lower because..." -- this sentence contradicts itself. The holdout r (0.698) is lower than the LOO r (0.711), but the text says the opposite. This appears to be a copy/paste editing error.
- No visual charts (scatter plots, residual plots) -- all information is text-based. Adding a predicted-vs-actual scatter plot would greatly strengthen this page.

---

### 11. `/forecast/[slug]` (Race Detail Pages)

**Tested:** `/forecast/2026-ga-senate`, `/forecast/2026-ga-governor`

**Visual Design:** Clean layout with breadcrumbs, model prediction callout, forecast toggle, polls table, and electoral types breakdown.

**MAJOR ISSUES:**

- **Slug format is non-obvious.** The URL format is `2026-ga-senate`, not `senate-ga-2026` or `president-2026`. Invalid slugs show "Race Not Found" with no suggestions. The forecast page race cards do not link to these pages at all.
- **Forecast toggle (National Environment / Local Polling)** works but:
  - When "National Environment" is selected, the margin value disappears. It should still show the national-environment-only prediction.
  - When "Local Polling" is selected for a race with no polls (like GA Governor), the radio is disabled -- good behavior.
- **GA Senate shows D+10.4** but the forecast page shows "EVEN" for the same race. This proves the margin data exists but is not being rendered on the main page.
- **Electoral Types section** shows the top 5 types in the state but their type names contain confusing "Blue" suffixes even when they lean Republican (e.g., "Hispanic Working-Class Blue" with R+12.7).
- **No link from the main forecast page to race detail pages.** This is a critical navigation gap.

**MINOR ISSUES:**

- Poll table is clean and readable.
- "Back to Forecast" link at bottom is good.
- No pollster ratings or quality indicators on polls.

---

### 12. `/embed/[slug]` (Embed Widget)

**Tested:** `/embed/2026-ga-senate`

**Visual Design:** Compact widget with margin, rating label, D/R bar chart, and attribution. Functional.

**MAJOR ISSUES:**

- **6 React hydration errors** (Error #418 x5, Error #423 x1) in the console. These are server/client mismatch errors suggesting the embed page renders differently on server vs client.
- **Widget renders in the top-right corner** of the viewport instead of centering or filling the available space. As an embed widget, it should be self-contained and left-aligned or centered.

**MINOR ISSUES:**

- "Powered by WetherVane" link points to the production domain (wethervane.hhaines.duckdns.org) -- correct for production, but means it links away from localhost in dev.
- No iframe wrapper or copy-to-clipboard embed code provided.

---

### 13. `/compare` (Compare Tab)

**Behavior:** Redirects to `/explore`. The Compare tab listed in the routes does not exist as a separate page. This is fine if intentional, but the route should probably 404 or the Compare functionality should be noted as "coming soon."

---

### 14. 404 Page

**Visual Design:** Default Next.js 404 page. Plain white page with "404 | This page could not be found."

**ISSUES:**

- No WetherVane branding, no navigation links, no helpful suggestions. Should have navigation back to the site and the same visual style.

---

## Cross-Cutting Issues

### Console Errors
- **favicon.ico 404** on every page load (2 errors per page).
- **React hydration errors** on the embed page (6 errors).
- No other JavaScript errors observed on main pages.

### Data Consistency
| Metric | About Page | Methodology | Types Page | API Reality |
|--------|-----------|-------------|------------|-------------|
| Fine types | 100 | 100 | 130 | 130 |
| Super-types | 5 | 5 | 6 | 6 |

The About and Methodology pages are stale -- they describe the J=100 model, but the deployed model uses J=130. This is a credibility issue for a methodology-focused site.

### Navigation Gaps
- No way to navigate from the main forecast page to individual race detail pages.
- No way to navigate from the main forecast page to governor races.
- County pages have no link to the type detail page.
- Type detail pages have no link to highlight those counties on the map.
- No global footer with links to methodology, about, types, etc.
- No site logo or clear "home" link (the brand name "WetherVane" is not clickable on map pages).

### Accessibility
- Skip-to-main-content link is present (good).
- Alert region exists for dynamic content announcements (good).
- Map is in a `region` with "Electoral map" label (good).
- However: the SVG-based map has no keyboard navigation or text alternatives for counties/states.
- Race cards are buttons with reasonable labels.
- Color is the primary differentiator on the map -- some users may need pattern or label alternatives.

---

## Summary by Severity

### CRITICAL (Ship-blocking)

1. **All Senate race margins display "EVEN"** instead of actual values. The `SenateControlBar.tsx` `formatMargin()` threshold is `0.5` instead of `0.005`. This makes the entire forecast page uninformative.

2. **Type names contradict observable data.** "Black Blue" is R+23. "White Red" is applied to LA County (48.7% Hispanic, D+15). Users will immediately lose trust. The naming scheme (type centroid's historical lean) does not survive contact with individual counties.

3. **Data consistency: 100 types vs 130 types.** The methodology and about pages describe a different model than what is actually deployed. This directly undermines the transparency claim.

### MAJOR (Should fix before public launch)

4. **No navigation from forecast to race detail pages.** Race cards are buttons that change the map but do not link to `/forecast/[slug]`. Users cannot access the detailed race data.

5. **No governor races on the forecast page.** Only Senate races are shown despite 36 governor slugs being available.

6. **Race detail page: margin disappears when toggling to "National Environment."**

7. **Dashboard mode overlay shows "No polls available for this state"** when no state is selected.

8. **Embed widget has 6 React hydration errors.**

9. **Dark mode: race cards retain white backgrounds** -- unreadable in dark mode.

10. **404 page has no branding or navigation.**

### MINOR (Polish items)

11. favicon.ico returns 404.
12. "Loading..." text instead of skeleton/spinner.
13. Explore page: generic "Super-type 1/2/3" labels instead of names.
14. Explore page: technical variable names in dropdowns ("Pres d shift 20 24").
15. Explore page: map legend shows Senate colors, not relevant explore visualization.
16. County pages: tags not clickable/linked.
17. County pages: "View on Map" goes to national instead of zooming to county.
18. Types listing: no search/filter for 130 items.
19. Methodology: all sections expanded by default.
20. Accuracy page: contradictory sentence about holdout r vs LOO r.
21. No global footer.
22. Breadcrumbs: "Home" and "Map" are synonymous.
23. Type pages: no shift history visualization.
24. County pages: no election history.
25. No embed code copy functionality on embed page.

### COSMETIC

26. Section heading anchor links show "#" text.
27. Explore scatter dot tooltips missing.
28. Types page very long -- jump links would help.
29. X-axis label overlap on wide scatter charts.

---

## Strengths

1. **Methodology page is outstanding.** Well-written, well-structured, excellent use of collapsible sections and metric cards. This is publication-quality content.
2. **Map visualization is distinctive.** The "stained glass" map with earth-tone colors is visually differentiated from 538/Economist/Cook-style maps.
3. **Mobile responsiveness is solid.** The map-over-panel stacking works well at 375px.
4. **Article/Dashboard layout modes** are a good concept -- different users want different views.
5. **Dark mode support** exists and mostly works.
6. **Accessibility basics** are in place (skip links, ARIA regions, button labels).
7. **County and type detail pages** provide genuinely useful information with good data density.
8. **Breadcrumb navigation** is consistently implemented across detail pages.
9. **Theme toggle cycles through system/light/dark** -- good UX pattern.
10. **Poll table on race detail pages** is clean and informative.

---

## Top 5 Recommendations (Priority Order)

1. **Fix the EVEN bug immediately.** Change `SenateControlBar.tsx` threshold from `0.5` to `0.005`. This is a one-line fix that makes the entire forecast page functional.

2. **Fix type naming.** Remove "Blue"/"Red" suffixes from type names -- they create confusion when applied to individual counties and races. Use descriptive names only (e.g., "Hispanic Working-Class" not "Hispanic Working-Class Blue").

3. **Update stale numbers.** Change About and Methodology pages to reflect J=130, 6 super-types. Or (better) pull these numbers dynamically from the API.

4. **Add race detail navigation.** Make forecast page race cards link to `/forecast/[slug]` race detail pages. Add governor races to the forecast page.

5. **Fix embed hydration errors.** The embed widget is meant for third-party embedding -- hydration errors will break in iframes and SSR contexts.
